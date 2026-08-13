"""Offline Qwen3 sidecar preflight worker.

This worker is intentionally small and read-only. It imports the isolated
Transformers runtime, reads a managed local Qwen3 artifact, and exercises the
tokenizer chat template without loading model weights.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


TOOL = "qwen3_sidecar_probe"
SCHEMA_VERSION = 1
MIN_TRANSFORMERS = (4, 51, 0)
MAX_INPUT_BYTES = 256 * 1024


def _base_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_sidecar_preflight",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "runtime_unavailable",
        "runtime": {
            "transformers_version": None,
            "minimum_transformers": ".".join(map(str, MIN_TRANSFORMERS)),
            "isolated": True,
        },
        "artifact": {
            "provided": bool(request.get("model_path")),
            "exists": False,
            "model_type": None,
            "architectures": [],
        },
        "tokenizer": {
            "loaded": False,
            "chat_template_available": False,
            "enable_thinking_supported": False,
            "rendered_without_thinking": False,
        },
        "errors": [],
    }


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(value).split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def _safe_artifact_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser().absolute().resolve(strict=False)
    return path if path.is_dir() else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_managed_metadata(model_path: Path) -> dict[str, Any]:
    manifest = _load_json(model_path / ".qlh-model-asset.json")
    asset = manifest.get("asset") if isinstance(manifest.get("asset"), dict) else {}
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    if manifest.get("artifact_kind") != "transformers_safetensors":
        raise ValueError("managed artifact kind is not Transformers Safetensors")
    if asset.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in asset.get("architectures", []):
        raise ValueError("managed artifact identity is not Qwen3ForCausalLM")
    entries = {item.get("path"): item for item in files if isinstance(item, dict)}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        item = entries.get(name)
        target = model_path / name
        if not isinstance(item, dict) or target.stat().st_size != int(item.get("size", -1)):
            raise ValueError(f"managed metadata size mismatch: {name}")
        if _sha256(target) != str(item.get("sha256", "")).lower():
            raise ValueError(f"managed metadata digest mismatch: {name}")
    return asset


def _has_nonempty_thinking(rendered: str) -> bool:
    """Treat the official empty ``<think></think>`` scaffold as disabled."""
    lower = rendered.lower()
    start = lower.find("<think>")
    end = lower.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start >= 0 and end >= 0:
        if rendered[start + len("<think>"):end].strip():
            return True
        lower = lower[:start] + lower[end + len("</think>"):]
    return "<|think|>" in lower or "</think>" in lower or "<think>" in lower


def execute_request(
    request: dict[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    result = _base_result(request)
    model_path = _safe_artifact_path(str(request.get("model_path", "")))
    if model_path is None:
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "model_path_invalid", "message": "managed Qwen3 model directory is missing"})
        return result

    config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    tokenizer_path = model_path / "tokenizer.json"
    manifest_path = model_path / ".qlh-model-asset.json"
    if not all(path.is_file() for path in (manifest_path, config_path, tokenizer_config_path, tokenizer_path)):
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "required_file_missing", "message": "Qwen3 config/tokenizer files are incomplete"})
        return result

    try:
        managed_asset = _verify_managed_metadata(model_path)
        config = _load_json(config_path)
        tokenizer_config = _load_json(tokenizer_config_path)
    except Exception as exc:
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "metadata_invalid", "message": exc.__class__.__name__})
        return result

    artifact = result["artifact"]
    artifact.update({
        "exists": True,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures") if isinstance(config.get("architectures"), list) else [],
        "config_transformers_version": config.get("transformers_version"),
        "tokenizer_chat_template_declared": bool(tokenizer_config.get("chat_template")),
        "managed_asset_id": managed_asset.get("asset_id"),
        "managed_revision": managed_asset.get("revision"),
    })
    if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in artifact["architectures"]:
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "not_qwen3", "message": "artifact metadata is not Qwen3ForCausalLM"})
        return result
    config_version = config.get("transformers_version")
    if config_version and _version_tuple(str(config_version)) < MIN_TRANSFORMERS:
        result["status"] = "artifact_rejected"
        result["errors"].append({"code": "config_transformers_too_old", "message": "Qwen3 config requires transformers >= 4.51.0"})
        return result

    try:
        transformers = module_loader("transformers")
        version = str(getattr(transformers, "__version__", "0.0.0"))
        result["runtime"]["transformers_version"] = version
        sidecar_python = Path(sys.executable).absolute().resolve(strict=False)
        controller_python = Path(str(request.get("controller_python", ""))).absolute().resolve(strict=False)
        result["runtime"]["isolated"] = sidecar_python != controller_python
        if not result["runtime"]["isolated"]:
            result["status"] = "runtime_rejected"
            result["errors"].append({"code": "runtime_not_isolated", "message": "Qwen3 worker must use a dedicated Python environment"})
            return result
        if _version_tuple(version) < MIN_TRANSFORMERS:
            result["status"] = "runtime_rejected"
            result["errors"].append({"code": "transformers_too_old", "message": "isolated sidecar requires transformers >= 4.51.0"})
            return result
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        auto_config = getattr(transformers, "AutoConfig", None)
        if auto_tokenizer is None or auto_config is None:
            raise RuntimeError("Transformers auto classes are unavailable")
        loaded_config = auto_config.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        if getattr(loaded_config, "model_type", None) != "qwen3":
            raise RuntimeError("isolated Transformers runtime did not register qwen3")
        result["runtime"]["qwen3_config_registered"] = True
        tokenizer = auto_tokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        tokenizer_report = result["tokenizer"]
        tokenizer_report["loaded"] = True
        template = getattr(tokenizer, "chat_template", None)
        tokenizer_report["chat_template_available"] = bool(template)
        if not template:
            raise RuntimeError("Qwen3 tokenizer chat template is missing")
        messages = [{"role": "user", "content": "Reply with the word OK."}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            raise RuntimeError("tokenizer chat template does not accept enable_thinking") from exc
        tokenizer_report["enable_thinking_supported"] = True
        tokenizer_report["rendered_without_thinking"] = not _has_nonempty_thinking(str(rendered))
        if not tokenizer_report["rendered_without_thinking"]:
            raise RuntimeError("enable_thinking=false still rendered thinking markers")
        tokenizer_report["rendered_length"] = len(str(rendered))
        result["gate_passed"] = True
        result["status"] = "ready_for_qwen3_smoke"
    except Exception as exc:
        result["status"] = "preflight_failed"
        result["errors"].append({"code": "tokenizer_preflight_failed", "message": exc.__class__.__name__})
    finally:
        gc.collect()
    return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("qwen3 sidecar request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    if request.get("schema_version") != SCHEMA_VERSION or request.get("operation") != "qwen3_sidecar_preflight":
        raise ValueError("unsupported Qwen3 sidecar protocol")
    if request.get("network_access") != "disabled" or request.get("read_only") is not True:
        raise ValueError("Qwen3 sidecar must be read-only and network-disabled")
    print(json.dumps(execute_request(request), ensure_ascii=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fallback = _base_result({})
        fallback["valid"] = False
        fallback["status"] = "invalid_request"
        fallback["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
        print(json.dumps(fallback, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
