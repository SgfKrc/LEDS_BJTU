"""Read-only model directory health sweep."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from .gguf import verify_gguf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is not None:
        try:
            if checker():
                return True
        except OSError:
            pass
    return path.is_symlink()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path != root else "."


def _iter_files(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    junctions: list[str] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            candidate = current_path / name
            if _is_junction(candidate):
                junctions.append(_relative(candidate, root))
            else:
                kept.append(name)
        directories[:] = kept
        files.extend(current_path / name for name in names)
    return files, junctions


def _discover_diffusion_manifests(root: Path) -> list[tuple[Path, str | None]]:
    result: list[tuple[Path, str | None]] = []
    for current, directories, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if not _is_junction(current_path / name)]
        manifest_path = current_path / ".qlh-sd-asset.json"
        if not manifest_path.is_file():
            continue
        asset_id: str | None = None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            asset_id = str((payload.get("asset") or {}).get("asset_id") or payload.get("asset_id") or "") or None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        result.append((current_path, asset_id))
    return result


def _sweep_pytorch_dir(path: Path, root: Path, full_hash: bool) -> dict[str, Any]:
    weight_files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    )
    records: list[dict[str, Any]] = []
    for item in weight_files:
        row: dict[str, Any] = {"path": _relative(item, root), "size_bytes": item.stat().st_size}
        if full_hash:
            from .gguf import _sha256
            row["sha256"] = _sha256(item)
        records.append(row)
    return {
        "path": _relative(path, root),
        "weight_file_count": len(records),
        "has_config": (path / "config.json").is_file(),
        "files": records,
        "valid": bool(records) and (path / "config.json").is_file(),
    }


def _discover_pytorch_dirs(files: Iterable[Path], root: Path) -> list[Path]:
    candidates: set[Path] = set()
    manifest_roots: list[Path] = []
    for current, directories, _ in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if not _is_junction(current_path / name)]
        if (current_path / ".qlh-sd-asset.json").is_file():
            manifest_roots.append(current_path)
    for item in files:
        if item.suffix.lower() not in {".safetensors", ".bin", ".pt", ".pth"}:
            continue
        current = item.parent
        while current != root and current not in candidates:
            if any(current == manifest_root or manifest_root in current.parents for manifest_root in manifest_roots):
                break
            if (current / "config.json").is_file():
                candidates.add(current)
                break
            current = current.parent
    return sorted(candidates)


def sweep_models(root: str | Path, *, full_hash: bool = False) -> dict[str, Any]:
    """Inspect model files and manifests without changing the model tree."""
    target = Path(root).expanduser()
    if not target.exists():
        return {"schema_version": 1, "root": str(target), "valid": False, "errors": ["model root does not exist"]}
    if not target.is_dir():
        return {"schema_version": 1, "root": str(target), "valid": False, "errors": ["model root is not a directory"]}
    files, junctions = _iter_files(target)
    gguf_reports = [verify_gguf(item, full_hash=full_hash) for item in files if item.suffix.lower() == ".gguf"]
    manifests: list[dict[str, Any]] = []
    for directory, asset_id in _discover_diffusion_manifests(target):
        report: dict[str, Any] = {"path": _relative(directory, target), "asset_id": asset_id, "valid": False}
        if asset_id:
            try:
                from diffusion.assets import verify_asset_directory
                report.update(verify_asset_directory(directory, asset_id, full_hash=full_hash))
            except Exception as exc:  # A sweep must report one bad asset and continue.
                report["errors"] = [str(exc)]
        else:
            report["errors"] = ["manifest has no asset_id"]
        manifests.append(report)
    pytorch_dirs = [_sweep_pytorch_dir(directory, target, full_hash) for directory in _discover_pytorch_dirs(files, target)]
    associated_sidecars = {
        candidate
        for item in files if item.suffix.lower() == ".gguf"
        for candidate in (item.with_name(item.name + ".sha256"), item.with_suffix(".sha256"))
    }
    orphan_files = [
        _relative(item, target)
        for item in files
        if item.name.endswith((".part", ".tmp"))
        or item.name == ".cache"
        or (item.suffix.lower() == ".sha256" and item not in associated_sidecars and item.name != "model.sha256")
    ]
    invalid_reports = [item for item in gguf_reports + manifests + pytorch_dirs if not item.get("valid", False)]
    warnings = [f"junction not traversed: {item}" for item in junctions]
    warnings.extend(f"orphan candidate: {item}" for item in orphan_files)
    return {
        "schema_version": 1,
        "root": str(target.resolve()),
        "root_is_junction": _is_junction(target),
        "valid": not invalid_reports,
        "gguf": gguf_reports,
        "diffusion_assets": manifests,
        "pytorch_directories": pytorch_dirs,
        "junctions": junctions,
        "orphan_files": orphan_files,
        "warnings": warnings,
        "read_only": True,
    }
