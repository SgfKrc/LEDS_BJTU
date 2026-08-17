"""Read-only discovery of local LLM asset packages under ``models/``.

The classic model registry describes runnable legacy engines.  Newer model
packages (notably Qwen3 sidecars) are intentionally not injected into that
registry because doing so would advertise an unsafe ``/models/switch`` path.
This module exposes only present, local assets for inventory UIs and routing
preflight.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


MANIFEST_NAME = ".qlh-model-asset.json"
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_LLM_MODEL_PREFIXES = (
    "qwen", "deepseek", "gemma", "llama", "mistral", "mixtral", "phi",
    "glm", "internlm", "baichuan", "yi", "falcon", "starcoder",
)


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _display_path(path: Path, app_root: Path) -> str:
    try:
        # Keep the user-facing app-relative path even when ``models`` is a
        # junction to another drive.  Resolving first would leak a host path.
        return path.relative_to(app_root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(app_root.resolve()).as_posix()
        except (OSError, ValueError):
            return path.name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_asset_id(value: object, fallback: str) -> str:
    candidate = str(value or fallback).strip().lower()
    candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate).strip("-._")
    return candidate or fallback


def _canonical_id(asset_id: str) -> str:
    return re.sub(r"(?:[-_.]gguf)$", "", asset_id, flags=re.IGNORECASE)


def _canonical_repo_id(repo_id: object) -> str:
    value = str(repo_id or "").strip()
    return re.sub(r"-gguf$", "", value, flags=re.IGNORECASE)


def _context_length(config: dict[str, Any]) -> int:
    candidates: Iterable[object] = (
        config.get("max_position_embeddings"),
        config.get("max_sequence_length"),
        (config.get("text_config") or {}).get("max_position_embeddings")
        if isinstance(config.get("text_config"), dict) else None,
    )
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if 512 <= parsed <= 10_000_000:
            return parsed
    return 4096


def _is_llm_config(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type.startswith(_LLM_MODEL_PREFIXES):
        return True
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        return False
    joined = " ".join(str(item).lower() for item in architectures)
    return any(prefix in joined for prefix in _LLM_MODEL_PREFIXES)


def _weight_files(directory: Path) -> list[Path]:
    try:
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}
            and path.stat().st_size > 0
        )
    except OSError:
        return []


def _gguf_files(directory: Path) -> list[Path]:
    try:
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".gguf" and path.stat().st_size > 0
        )
    except OSError:
        return []


def _primary_gguf(files: list[Path]) -> Path | None:
    if not files:
        return None
    llm_files = [path for path in files if "mmproj" not in path.name.lower()]
    return max(llm_files or files, key=lambda path: path.stat().st_size)


def _runtime_profile(model_id: str, config: dict[str, Any]) -> tuple[str, str]:
    model_type = str(config.get("model_type") or "").lower()
    marker = f"{model_id} {model_type}".lower()
    if "qwen3" in marker:
        if "vl" in marker:
            return (
                "qwen3_multimodal_sidecar",
                "本地资产已发现；当前仅安全登记，尚无可执行的 Qwen3-VL Sidecar 加载控制面，不使用旧单机加载器。",
            )
        return (
            "qwen3_sidecar",
            "本地资产已发现；当前仅安全登记，尚无可执行的 Qwen3 Sidecar 加载控制面，不使用旧单机加载器。",
        )
    if "gemma4" in marker or "gemma-4" in marker:
        return (
            "gemma4_pipeline",
            "本地资产已发现；当前仅安全登记，尚无可执行的 Gemma 4 Sidecar 加载控制面，不使用旧单机加载器。",
        )
    return (
        "manual_runtime_selection",
        "本地资产已发现；请先选择兼容运行时或登记到模型控制面。",
    )


def _manifest_asset(directory: Path, root: Path) -> dict[str, Any] | None:
    manifest_path = directory / MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if not manifest:
        return None

    asset = manifest.get("asset")
    files = manifest.get("files")
    if not isinstance(asset, dict) or not isinstance(files, list):
        return None

    config = _read_json(directory / "config.json")
    artifact_kind = str(manifest.get("artifact_kind") or "")
    if artifact_kind not in {"transformers_safetensors", "llama.cpp_gguf"}:
        return None
    listed = []
    complete = True
    for entry in files:
        if not isinstance(entry, dict):
            complete = False
            continue
        relative = str(entry.get("path") or "").replace("\\", "/")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            complete = False
            continue
        candidate = directory / relative
        if not _is_within(candidate, directory) or not candidate.is_file():
            complete = False
            continue
        try:
            if candidate.stat().st_size > 0:
                listed.append(candidate)
            else:
                complete = False
        except OSError:
            complete = False
    if not complete:
        return None

    safetensors = [path for path in listed if path.suffix.lower() in {".safetensors", ".bin"}]
    ggufs = [path for path in listed if path.suffix.lower() == ".gguf"]
    has_safetensors = bool((directory / "config.json").is_file() and safetensors)
    has_gguf = bool(ggufs)
    if not has_safetensors and not has_gguf:
        return None
    if artifact_kind == "transformers_safetensors" and not has_safetensors:
        return None
    if artifact_kind == "llama.cpp_gguf" and not has_gguf:
        return None

    return {
        "asset_id": _safe_asset_id(asset.get("asset_id"), directory.name),
        "repo_id": str(asset.get("repo_id") or "").strip(),
        "config": config,
        "safetensors_dir": directory if has_safetensors else None,
        "gguf_files": ggufs,
        "bytes": sum(path.stat().st_size for path in listed),
        "manifest_path": manifest_path,
        "integrity": "manifest_verified",
        "architectures": asset.get("architectures") if isinstance(asset.get("architectures"), list) else [],
        "root": root,
    }


def _filesystem_asset(directory: Path, root: Path) -> dict[str, Any] | None:
    config = _read_json(directory / "config.json")
    safetensors = _weight_files(directory)
    ggufs = _gguf_files(directory)
    has_safetensors = bool(config and _is_llm_config(config) and safetensors)
    has_gguf = bool(ggufs)
    if not has_safetensors and not has_gguf:
        return None

    all_files = list(safetensors) + list(ggufs)
    return {
        "asset_id": _safe_asset_id(directory.name, directory.name),
        "repo_id": "",
        "config": config,
        "safetensors_dir": directory if has_safetensors else None,
        "gguf_files": ggufs,
        "bytes": sum(path.stat().st_size for path in all_files),
        "manifest_path": None,
        "integrity": "filesystem_discovered",
        "architectures": config.get("architectures") if isinstance(config.get("architectures"), list) else [],
        "root": root,
    }


def _merge_assets(parts: list[dict[str, Any]], app_root: Path) -> dict[str, Any]:
    first = parts[0]
    model_id = _canonical_id(first["asset_id"])
    repo_id = next((part["repo_id"] for part in parts if part["repo_id"]), "")
    canonical_repo = _canonical_repo_id(repo_id)
    config = next((part["config"] for part in parts if part["config"]), {})
    safetensors_dir = next((part["safetensors_dir"] for part in parts if part["safetensors_dir"]), None)
    gguf_files = [path for part in parts for path in part["gguf_files"]]
    primary_gguf = _primary_gguf(gguf_files)
    formats = []
    if safetensors_dir is not None:
        formats.append("safetensors")
    if primary_gguf is not None:
        formats.append("gguf")
    model_type = "both" if len(formats) == 2 else formats[0]
    runtime_profile, runtime_hint = _runtime_profile(model_id, config)
    architectures = []
    for part in parts:
        for architecture in part["architectures"]:
            text = str(architecture).strip()
            if text and text not in architectures:
                architectures.append(text)
    manifest_paths = [
        _display_path(part["manifest_path"], app_root)
        for part in parts if part["manifest_path"] is not None
    ]
    source_paths = []
    for part in parts:
        path = part["safetensors_dir"] or _primary_gguf(part["gguf_files"])
        if path is not None:
            display = _display_path(path, app_root)
            if display not in source_paths:
                source_paths.append(display)

    return {
        "model_id": model_id,
        "name": canonical_repo or model_id,
        "huggingface_id": canonical_repo,
        "model_type": model_type,
        "available_formats": formats,
        "model_path": _display_path(safetensors_dir, app_root) if safetensors_dir else "",
        "gguf_path": _display_path(primary_gguf, app_root) if primary_gguf else "",
        "max_context": _context_length(config),
        "architectures": architectures,
        "total_bytes": sum(part["bytes"] for part in parts),
        "asset_ids": sorted({part["asset_id"] for part in parts}),
        "source_paths": source_paths,
        "manifest_paths": manifest_paths,
        "integrity": "manifest_verified" if manifest_paths else "filesystem_discovered",
        "runtime_profile": runtime_profile,
        "runtime_hint": runtime_hint,
        "runtime_status": "inventory_only",
        "runtime_action": "qwen3_preflight" if runtime_profile == "qwen3_sidecar" else None,
    }


def discover_local_model_assets(models_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return a stable, read-only inventory of local LLM assets.

    Only immediate directories below ``models/`` are considered.  This avoids
    treating cache or arbitrary nested files as installable model packages.
    A malformed manifest, a missing listed weight, or a non-LLM config is
    ignored rather than being exposed as a selectable model.
    """
    app_root = _app_root()
    root = Path(models_root).resolve() if models_root else app_root / "models"
    if not root.is_dir():
        return {"assets": [], "summary": {"total": 0, "total_bytes": 0}}

    parts: list[dict[str, Any]] = []
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    except OSError:
        directories = []
    for directory in directories:
        if (directory / MANIFEST_NAME).is_file():
            parts.append(_manifest_asset(directory, root))
        else:
            parts.append(_filesystem_asset(directory, root))
    parts = [part for part in parts if part is not None]

    groups: dict[str, list[dict[str, Any]]] = {}
    for part in parts:
        groups.setdefault(_canonical_id(part["asset_id"]), []).append(part)
    assets = [_merge_assets(group, app_root) for _, group in sorted(groups.items())]
    assets.sort(key=lambda asset: (asset["name"].lower(), asset["model_id"]))
    return {
        "assets": assets,
        "summary": {
            "total": len(assets),
            "total_bytes": sum(asset["total_bytes"] for asset in assets),
        },
    }


def resolve_local_model_asset_metadata(model_id: str) -> dict[str, Any] | None:
    """Resolve one discovered Safetensors asset for server-side planning.

    This helper is intentionally separate from the inventory response.  It
    returns an absolute path only to trusted in-process callers; API callers
    receive the path-free contract projection instead.
    """
    requested_id = str(model_id or "").strip()
    if not requested_id:
        return None
    asset = next(
        (entry for entry in discover_local_model_assets()["assets"]
         if entry.get("model_id") == requested_id),
        None,
    )
    if asset is None or not asset.get("model_path"):
        return None
    app_root = _app_root()
    models_root = (app_root / "models").resolve()
    model_path = (app_root / Path(str(asset["model_path"]))).resolve()
    if not _is_within(model_path, models_root) or not model_path.is_dir():
        return None
    manifest = _read_json(model_path / MANIFEST_NAME)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list) or not files:
        return None
    identity = []
    for entry in files:
        if not isinstance(entry, dict):
            return None
        relative = str(entry.get("path") or "").replace("\\", "/")
        digest = str(entry.get("sha256") or "").lower()
        try:
            size = int(entry.get("size", 0) or 0)
        except (TypeError, ValueError):
            return None
        if (
            not relative or not digest or len(digest) != 64 or size <= 0
            or relative.startswith("/") or ".." in Path(relative).parts
        ):
            return None
        candidate = model_path / relative
        if not _is_within(candidate, model_path) or not candidate.is_file():
            return None
        try:
            if candidate.stat().st_size != size:
                return None
        except OSError:
            return None
        identity.append({"path": relative, "size": size, "sha256": digest})
    digest = hashlib.sha256()
    for entry in sorted(identity, key=lambda item: item["path"]):
        digest.update(
            f"{entry['path']}\0{entry['size']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return {
        "model_id": requested_id,
        "model_path": model_path,
        "config": asset.get("_config") if isinstance(asset.get("_config"), dict) else _read_json(model_path / "config.json"),
        "runtime_profile": asset.get("runtime_profile"),
        "integrity": asset.get("integrity"),
        "model_sha256": digest.hexdigest(),
    }


def preflight_local_model_asset(model_id: str) -> dict[str, Any]:
    """Run the supported read-only runtime preflight for one discovered asset.

    The caller supplies an inventory ID rather than a filesystem path.  This
    keeps the endpoint scoped to ``models/`` and prevents it from becoming a
    generic subprocess launcher.  It intentionally does not start a serving
    sidecar or materialize model weights.
    """
    requested_id = str(model_id or "").strip()
    asset = next(
        (entry for entry in discover_local_model_assets()["assets"]
         if entry.get("model_id") == requested_id),
        None,
    )
    base = {
        "schema_version": 1,
        "operation": "local_asset_preflight",
        "model_id": requested_id,
        "runtime_profile": asset.get("runtime_profile") if asset else None,
        "read_only": True,
        "starts_sidecar": False,
        "gate_passed": False,
    }
    if asset is None:
        return {
            **base,
            "status": "asset_not_found",
            "errors": [{"code": "asset_not_found", "message": "local model asset was not found"}],
        }
    if asset.get("runtime_profile") != "qwen3_sidecar" or not asset.get("model_path"):
        return {
            **base,
            "status": "preflight_not_supported",
            "errors": [{
                "code": "preflight_not_supported",
                "message": "this asset has no supported read-only sidecar preflight",
            }],
        }

    model_path = _app_root() / Path(str(asset["model_path"]))
    models_root = _app_root() / "models"
    if not _is_within(model_path, models_root):
        return {
            **base,
            "status": "asset_rejected",
            "errors": [{"code": "asset_path_invalid", "message": "local asset path escapes models root"}],
        }

    from scripts.model_tools.qwen3_sidecar_probe import run_qwen3_sidecar_probe

    report = run_qwen3_sidecar_probe(model=model_path)
    return {
        **base,
        "status": report.get("status", "worker_failed"),
        "gate_passed": bool(report.get("gate_passed")),
        "preflight": report,
        "errors": report.get("errors", []),
    }
