"""Read-only resolver for the pinned local Ollama Gemma 4 native candidate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from .gguf import inspect_gguf


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = Path(__file__).with_name("gemma4_native.lock.json")
TOOL = "gemma4_native_assets"
SCHEMA_VERSION = 1
_MANIFEST_RELATIVE_PATH = Path("manifests") / "registry.ollama.ai" / "library" / "gemma4" / "12b"


def load_lock(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION or raw.get("tool") != TOOL:
        raise ValueError("unsupported Gemma 4 native asset lock")
    for section in ("source", "artifacts"):
        if not isinstance(raw.get(section), dict):
            raise ValueError("Gemma 4 native asset lock is incomplete")
    if not isinstance(raw["artifacts"].get("model"), dict) or not isinstance(raw["artifacts"].get("mmproj"), dict):
        raise ValueError("Gemma 4 native asset lock is incomplete")
    return raw


def default_ollama_models_root() -> Path:
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        return Path(configured).expanduser().absolute().resolve(strict=False)
    return (Path.home() / ".ollama" / "models").absolute().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _blob_path(models_root: Path, digest: str) -> Path:
    return models_root / "blobs" / f"sha256-{digest}"


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _artifact_report(
    *,
    path: Path,
    expected: dict[str, Any],
    full_hash: bool,
) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {
        "digest": expected["sha256"],
        "expected_size_bytes": int(expected["size_bytes"]),
        "exists": path.is_file(),
        "size_matches": False,
        "content_sha256_checked": bool(full_hash),
        "content_sha256_matches": False,
    }
    if not path.is_file():
        return report, False
    report["size_matches"] = path.stat().st_size == int(expected["size_bytes"])
    if full_hash and report["size_matches"]:
        report["content_sha256_matches"] = _sha256(path) == str(expected["sha256"])
    return report, bool(report["size_matches"] and (report["content_sha256_matches"] if full_hash else True))


def _metadata_summary(
    model_path: Path,
    mmproj_path: Path,
    lock: dict[str, Any],
    inspect_fn: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    model = inspect_fn(model_path).get("metadata", {})
    mmproj = inspect_fn(mmproj_path).get("metadata", {})
    expected_model = lock["artifacts"]["model"]
    expected_mmproj = lock["artifacts"]["mmproj"]
    summary = {
        "model_architecture": model.get("general.architecture"),
        "model_base_model": model.get("general.base_model.0.name"),
        "model_context_length": model.get("gemma4.context_length"),
        "mmproj_architecture": mmproj.get("general.architecture"),
        "mmproj_type": mmproj.get("general.type"),
        "mmproj_base_model": mmproj.get("general.base_model.0.name"),
        "mmproj_has_vision_encoder": mmproj.get("clip.has_vision_encoder"),
        "mmproj_has_audio_encoder": mmproj.get("clip.has_audio_encoder"),
        "vision_projector_type": mmproj.get("clip.vision.projector_type"),
    }
    matched = (
        summary["model_architecture"] == expected_model["architecture"]
        and summary["model_base_model"] == expected_model["base_model"]
        and summary["model_context_length"] == expected_model["context_length"]
        and summary["mmproj_architecture"] == expected_mmproj["architecture"]
        and summary["mmproj_type"] == "mmproj"
        and summary["mmproj_base_model"] == expected_mmproj["base_model"]
        and summary["mmproj_has_vision_encoder"] is True
        and summary["vision_projector_type"] == expected_mmproj["vision_projector_type"]
    )
    return summary, matched


def resolve_ollama_gemma4_12b(
    *,
    models_root: Path | None = None,
    full_hash: bool = False,
    lock_path: Path = LOCK_PATH,
    inspect_fn: Callable[[Path], dict[str, Any]] = inspect_gguf,
) -> tuple[dict[str, Any], dict[str, Path] | None]:
    """Resolve the exact local candidate without emitting user filesystem paths."""
    lock = load_lock(lock_path)
    root = (models_root or default_ollama_models_root()).expanduser().absolute().resolve(strict=False)
    source = lock["source"]
    errors: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "resolve_ollama_gemma4_12b",
        "candidate_id": lock["candidate_id"],
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "source": {
            "registry": source["registry"],
            "tag": source["tag"],
            "model_id": source["model_id"],
            "manifest_sha256": source["manifest_sha256"],
            "license": source["license"],
            "manifest_exists": False,
            "manifest_sha256_matches": False,
        },
        "artifacts": {},
        "metadata": {},
        "identity_verified": False,
        "content_verified": False,
        "gate_passed": False,
        "errors": errors,
    }
    manifest_path = root / _MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        errors.append(_error("manifest_missing", "pinned local Ollama manifest is unavailable"))
        return report, None
    report["source"]["manifest_exists"] = True
    report["source"]["manifest_sha256_matches"] = _sha256(manifest_path) == source["manifest_sha256"]
    if not report["source"]["manifest_sha256_matches"]:
        errors.append(_error("manifest_digest_mismatch", "local Ollama manifest does not match the pinned candidate"))
        return report, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append(_error("manifest_invalid", "local Ollama manifest is not valid JSON"))
        return report, None

    layers = {item.get("mediaType"): item for item in manifest.get("layers", []) if isinstance(item, dict)}
    expected_model = lock["artifacts"]["model"]
    expected_mmproj = lock["artifacts"]["mmproj"]
    for name, expected in (("model", expected_model), ("mmproj", expected_mmproj)):
        layer = layers.get(expected["media_type"])
        if not isinstance(layer, dict) or layer.get("digest") != f"sha256:{expected['sha256']}" or int(layer.get("size", -1)) != int(expected["size_bytes"]):
            errors.append(_error(f"{name}_layer_mismatch", "pinned manifest layer does not match the native candidate"))
            return report, None
        if name == "mmproj" and layer.get("from") != expected["from"]:
            errors.append(_error("mmproj_pairing_mismatch", "pinned projector relationship does not match the native candidate"))
            return report, None

    model_path = _blob_path(root, expected_model["sha256"])
    mmproj_path = _blob_path(root, expected_mmproj["sha256"])
    model_report, model_ok = _artifact_report(path=model_path, expected=expected_model, full_hash=full_hash)
    mmproj_report, mmproj_ok = _artifact_report(path=mmproj_path, expected=expected_mmproj, full_hash=full_hash)
    report["artifacts"] = {"model": model_report, "mmproj": mmproj_report}
    if not model_ok or not mmproj_ok:
        errors.append(_error("artifact_verification_failed", "pinned local artifact is missing, size-mismatched or digest-mismatched"))
        return report, None
    report["identity_verified"] = True
    report["content_verified"] = bool(full_hash)
    try:
        metadata, metadata_ok = _metadata_summary(model_path, mmproj_path, lock, inspect_fn)
    except Exception:
        errors.append(_error("metadata_inspection_failed", "candidate GGUF metadata could not be inspected"))
        return report, None
    report["metadata"] = metadata
    if not metadata_ok:
        errors.append(_error("metadata_pairing_mismatch", "candidate GGUF metadata does not describe the pinned Gemma 4 pair"))
        return report, None
    report["gate_passed"] = bool(full_hash)
    return report, {"model": model_path, "mmproj": mmproj_path}


def run_ollama_gemma4_12b_preflight(
    *,
    models_root: Path | None = None,
    n_ctx: int = 128,
    require_audio: bool = False,
    timeout_seconds: float = 180.0,
    asset_resolver: Callable[..., tuple[dict[str, Any], dict[str, Path] | None]] = resolve_ollama_gemma4_12b,
    probe_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify the pinned pair, then run its isolated native preflight.

    Content hashing is intentional here: the command is the single supported
    bridge from a mutable Ollama tag to a native runtime experiment.
    """
    assets, pair = asset_resolver(models_root=models_root, full_hash=True)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "gemma4_ollama_native_preflight",
        "operation": "verify_then_native_preflight",
        "valid": bool(assets.get("valid", False)),
        "read_only": True,
        "network_access": "disabled",
        "status": "artifact_rejected",
        "gate_passed": False,
        "assets": assets,
        "native": None,
        "errors": list(assets.get("errors", [])),
    }
    if not assets.get("gate_passed") or pair is None:
        return result
    if probe_runner is None:
        from .gemma4_native_probe import run_native_probe

        probe_runner = run_native_probe
    native = probe_runner(
        model=pair["model"],
        mmproj=pair["mmproj"],
        model_sha256=assets["artifacts"]["model"]["digest"],
        mmproj_sha256=assets["artifacts"]["mmproj"]["digest"],
        n_ctx=n_ctx,
        require_audio=require_audio,
        timeout_seconds=timeout_seconds,
    )
    result["native"] = native
    result["status"] = native.get("status", "worker_failed")
    result["gate_passed"] = bool(native.get("gate_passed"))
    result["errors"].extend(native.get("errors", []))
    return result


__all__ = [
    "default_ollama_models_root",
    "load_lock",
    "resolve_ollama_gemma4_12b",
    "run_ollama_gemma4_12b_preflight",
]
