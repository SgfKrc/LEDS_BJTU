"""Junction-aware model storage accounting and conservative cleanup."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MODEL_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
CLEANUP_DIR_NAMES = {".cache", "models_old_backup"}
CONFIRMATION_WORD = "CLEAN"


def _is_junction(path: Path) -> bool:
    """Treat junctions and symlinks as reparse points that must not be followed."""
    checker = getattr(path, "is_junction", None)
    if checker is not None:
        try:
            if checker():
                return True
        except OSError:
            pass
    try:
        return path.is_symlink()
    except OSError:
        return False


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path != root else "."


def _resolved(path: Path) -> Path:
    return path.expanduser().absolute().resolve(strict=False)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(candidate)]) == str(root)
    except ValueError:
        return False


def _allocated_size(path: Path, stat_result: os.stat_result) -> int:
    """Return allocated bytes where the platform exposes them, with a safe fallback."""
    blocks = getattr(stat_result, "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return blocks * 512
    if sys.platform == "win32":
        try:
            get_size = ctypes.windll.kernel32.GetCompressedFileSizeW
            get_size.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
            get_size.restype = ctypes.c_ulong
            high = ctypes.c_ulong(0)
            low = get_size(str(path), ctypes.byref(high))
            if low != 0xFFFFFFFF or ctypes.GetLastError() == 0:
                return (int(high.value) << 32) | int(low)
        except (AttributeError, OSError):
            pass
    return int(stat_result.st_size)


def _iter_files(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    skipped: list[str] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            candidate = current_path / name
            if _is_junction(candidate):
                skipped.append(_relative(candidate, root))
            else:
                kept.append(name)
        directories[:] = kept
        for name in names:
            candidate = current_path / name
            if _is_junction(candidate):
                skipped.append(_relative(candidate, root))
            elif candidate.is_file():
                files.append(candidate)
    return files, skipped


def _root_report(root: Path) -> dict[str, Any]:
    target = root.expanduser().absolute()
    if not target.exists():
        return {"schema_version": 1, "root": str(target), "valid": False, "errors": ["model root does not exist"]}
    if not target.is_dir():
        return {"schema_version": 1, "root": str(target), "valid": False, "errors": ["model root is not a directory"]}
    resolved = _resolved(target)
    return {
        "schema_version": 1,
        "root": str(resolved),
        "root_input": str(target),
        "root_is_junction": _is_junction(target),
        "volume": resolved.anchor,
        "read_only": True,
    }


def model_disk_usage(root: str | Path) -> dict[str, Any]:
    """Report logical and allocated storage without following nested junctions."""
    target = Path(root).expanduser().absolute()
    base = _root_report(target)
    if not base.get("valid", True):
        return base
    files, skipped = _iter_files(target)
    entries: dict[str, dict[str, Any]] = {}
    global_ids: set[tuple[int, int]] = set()
    total_logical = total_allocated = total_unique_allocated = 0
    errors: list[str] = []
    for item in files:
        relative = _relative(item, target)
        top = relative.split("/", 1)[0]
        row = entries.setdefault(top, {
            "path": top,
            "file_count": 0,
            "logical_size_bytes": 0,
            "allocated_size_bytes": 0,
            "unique_allocated_size_bytes": 0,
            "extensions": Counter(),
            "_ids": set(),
        })
        try:
            stat_result = item.stat()
        except OSError as exc:
            errors.append(f"stat failed for {relative}: {exc}")
            continue
        logical = int(stat_result.st_size)
        allocated = _allocated_size(item, stat_result)
        identity = (int(getattr(stat_result, "st_dev", 0)), int(getattr(stat_result, "st_ino", 0)))
        row["file_count"] += 1
        row["logical_size_bytes"] += logical
        row["allocated_size_bytes"] += allocated
        row["extensions"][item.suffix.lower() or "<none>"] += 1
        if identity not in row["_ids"]:
            row["_ids"].add(identity)
            row["unique_allocated_size_bytes"] += allocated
        total_logical += logical
        total_allocated += allocated
        if identity not in global_ids:
            global_ids.add(identity)
            total_unique_allocated += allocated
    for row in entries.values():
        row["extensions"] = dict(sorted(row["extensions"].items()))
        row.pop("_ids", None)
    base.update({
        "valid": not errors,
        "entries": sorted(entries.values(), key=lambda value: value["path"]),
        "totals": {
            "file_count": len(files),
            "logical_size_bytes": total_logical,
            "allocated_size_bytes": total_allocated,
            "unique_allocated_size_bytes": total_unique_allocated,
        },
        "junctions_skipped": sorted(skipped),
        "warnings": [f"junction/reparse point not traversed: {item}" for item in sorted(skipped)],
        "errors": errors,
    })
    return base


def _tree_size(path: Path) -> int:
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    total = 0
    for current, directories, names in os.walk(path, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if not _is_junction(current_path / name)]
        for name in names:
            candidate = current_path / name
            if _is_junction(candidate):
                continue
            try:
                total += int(candidate.stat().st_size)
            except OSError:
                pass
    return total


def _tree_mtime(path: Path) -> float:
    try:
        latest = path.stat().st_mtime
    except OSError:
        return 0.0
    if path.is_dir() and not _is_junction(path):
        for current, directories, names in os.walk(path, followlinks=False):
            current_path = Path(current)
            directories[:] = [name for name in directories if not _is_junction(current_path / name)]
            for name in names:
                candidate = current_path / name
                if _is_junction(candidate):
                    continue
                try:
                    latest = max(latest, candidate.stat().st_mtime)
                except OSError:
                    pass
    return latest


def _candidate(path: Path, root: Path, kind: str, *, safe: bool, reason: str, now: float) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "kind": kind,
        "size_bytes": _tree_size(path),
        "age_hours": round(max(0.0, (now - _tree_mtime(path)) / 3600.0), 3),
        "safe_to_delete": safe,
        "reason": reason,
        "requires_confirmation": CONFIRMATION_WORD,
    }


def _is_protected_tree(path: Path, cleanup_roots: set[Path]) -> bool:
    return any(path == candidate or candidate in path.parents for candidate in cleanup_roots)


def _find_cleanup_candidates(target: Path, *, min_age_hours: float, include_duplicates: bool) -> tuple[list[dict[str, Any]], list[str]]:
    files, skipped = _iter_files(target)
    now = time.time()
    age_seconds = max(0.0, min_age_hours * 3600.0)
    candidates: list[dict[str, Any]] = []
    cleanup_roots: set[Path] = set()
    for current, directories, names in os.walk(target, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            candidate = current_path / name
            if _is_junction(candidate):
                skipped.append(_relative(candidate, target))
                continue
            if name in CLEANUP_DIR_NAMES:
                cleanup_roots.add(candidate)
                if now - _tree_mtime(candidate) >= age_seconds:
                    kind = "cache" if name == ".cache" else "old_backup"
                    candidates.append(_candidate(candidate, target, kind, safe=False, reason=f"stale {name} tree; may contain reusable asset metadata", now=now))
                continue
            kept.append(name)
        directories[:] = kept
        for name in names:
            item = current_path / name
            if _is_junction(item):
                skipped.append(_relative(item, target))
                continue
            if _is_protected_tree(item, cleanup_roots):
                continue
            try:
                age_ok = now - item.stat().st_mtime >= age_seconds
            except OSError:
                age_ok = False
            if age_ok and item.name.endswith((".part", ".tmp")):
                candidates.append(_candidate(item, target, "partial", safe=True, reason="stale partial/temp file", now=now))

    sidecar_targets = {item for item in files if item.suffix.lower() != ".sha256"}
    for item in files:
        if (
            item.suffix.lower() != ".sha256"
            or item.name == "model.sha256"
            or _is_protected_tree(item, cleanup_roots)
        ):
            continue
        base_name = item.name[:-7]
        possibilities = {item.with_name(base_name), item.with_suffix("")}
        if not any(candidate in sidecar_targets for candidate in possibilities) and now - item.stat().st_mtime >= age_seconds:
            candidates.append(_candidate(item, target, "orphan_sidecar", safe=True, reason="sha256 sidecar has no local target", now=now))

    if include_duplicates:
        groups: dict[int, list[Path]] = defaultdict(list)
        for item in files:
            if item.suffix.lower() in MODEL_SUFFIXES and not _is_protected_tree(item, cleanup_roots):
                try:
                    groups[item.stat().st_size].append(item)
                except OSError:
                    pass
        for same_size in groups.values():
            if len(same_size) < 2:
                continue
            digests: dict[str, list[Path]] = defaultdict(list)
            for item in same_size:
                digest = hashlib.sha256()
                with item.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                digests[digest.hexdigest()].append(item)
            for duplicate_paths in digests.values():
                if len(duplicate_paths) < 2:
                    continue
                for duplicate in sorted(duplicate_paths)[1:]:
                    candidates.append(_candidate(duplicate, target, "duplicate_model", safe=False, reason="same-size identical model content; path may be referenced", now=now))
    return candidates, sorted(set(skipped))


def _contains_reparse(path: Path) -> bool:
    if _is_junction(path):
        return True
    if path.is_dir():
        for current, directories, names in os.walk(path, followlinks=False):
            current_path = Path(current)
            if any(_is_junction(current_path / name) for name in directories + names):
                return True
    return False


def clean_models(
    root: str | Path,
    *,
    apply: bool = False,
    confirmation: str | None = None,
    min_age_hours: float = 24.0,
    include_duplicates: bool = False,
    include_caches: bool = False,
    include_old_backups: bool = False,
) -> dict[str, Any]:
    """List or remove stale unowned files; never follows or removes reparse points."""
    target = Path(root).expanduser().absolute()
    base = _root_report(target)
    if not base.get("valid", True):
        return base
    if min_age_hours < 0:
        return {**base, "valid": False, "errors": ["min_age_hours must be non-negative"]}
    if apply and confirmation != CONFIRMATION_WORD:
        return {**base, "valid": False, "errors": [f"apply requires --confirm {CONFIRMATION_WORD}"]}
    candidates, skipped = _find_cleanup_candidates(target, min_age_hours=min_age_hours, include_duplicates=include_duplicates)
    for row in candidates:
        if row["kind"] == "cache" and include_caches:
            row["safe_to_delete"] = True
        elif row["kind"] == "old_backup" and include_old_backups:
            row["safe_to_delete"] = True
    resolved_root = _resolved(target)
    deleted: list[str] = []
    errors: list[str] = []
    if apply:
        for row in candidates:
            if not row["safe_to_delete"]:
                continue
            candidate = target / Path(row["path"])
            resolved_candidate = _resolved(candidate)
            if candidate == target or not _inside(resolved_root, resolved_candidate) or _is_junction(candidate) or _contains_reparse(candidate):
                errors.append(f"refused unsafe cleanup target: {row['path']}")
                continue
            try:
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
                deleted.append(row["path"])
            except OSError as exc:
                errors.append(f"delete failed for {row['path']}: {exc}")
    return {
        **base,
        "valid": not errors,
        "read_only": not apply,
        "applied": apply,
        "min_age_hours": min_age_hours,
        "include_duplicates": include_duplicates,
        "include_caches": include_caches,
        "include_old_backups": include_old_backups,
        "candidates": candidates,
        "deleted": deleted,
        "junctions_skipped": skipped,
        "warnings": [f"junction/reparse point not traversed: {item}" for item in skipped],
        "errors": errors,
    }


__all__ = ["CONFIRMATION_WORD", "clean_models", "model_disk_usage"]
