"""Isolated Qwen3 multimodal AutoProcessor construction worker."""

from __future__ import annotations

import gc
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_preflight import (  # noqa: E402
    Qwen3MultimodalPreflightError,
    build_mm1_processor_smoke_response,
    inspect_mm1_processor_assets,
    validate_mm1_visual_worker_request,
)


TOOL = "qwen3_multimodal_processor_probe"
SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 256 * 1024


def _base_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_worker_processor_smoke",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "runtime_unavailable",
        "errors": [],
    }


def _safe_model_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().absolute().resolve(strict=False)
    return path if path.is_dir() else None


def _version_tuple(value: Any) -> tuple[int, ...]:
    parts: list[int] = []
    for part in str(value).split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def execute_request(
    request: Mapping[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    result = _base_result()
    if (
        request.get("schema_version") != SCHEMA_VERSION
        or request.get("operation") != "qwen3_visual_worker_processor_smoke"
        or request.get("tool") != TOOL
        or request.get("read_only") is not True
        or request.get("network_access") != "disabled"
    ):
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "protocol_invalid", "message": "processor smoke protocol is invalid"}]
        return result
    model_path = _safe_model_path(request.get("model_path"))
    manifest = request.get("manifest")
    visual_request = request.get("visual_request")
    if model_path is None or not isinstance(manifest, dict) or not isinstance(visual_request, dict):
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "request_incomplete", "message": "processor smoke request is incomplete"}]
        return result
    try:
        inspection = inspect_mm1_processor_assets(model_path, manifest)
        safe_request = validate_mm1_visual_worker_request(
            visual_request, manifest=manifest, inspection=inspection,
        )
    except Exception as exc:
        result["status"] = "artifact_rejected"
        result["errors"] = [{"code": "mm1_preflight_rejected", "message": exc.__class__.__name__}]
        return result
    try:
        transformers = module_loader("transformers")
        version = str(getattr(transformers, "__version__", "0.0.0"))
        if _version_tuple(version) < (4, 51, 0):
            result["status"] = "runtime_rejected"
            result["errors"] = [{"code": "transformers_too_old", "message": "isolated processor worker requires transformers >= 4.51.0"}]
            return result
        sidecar_python = Path(sys.executable).absolute().resolve(strict=False)
        controller_python = Path(str(request.get("controller_python", ""))).absolute().resolve(strict=False)
        isolated = sidecar_python != controller_python
        if not isolated:
            result["status"] = "runtime_rejected"
            result["errors"] = [{"code": "runtime_not_isolated", "message": "processor worker must use a dedicated Python environment"}]
            return result
        auto_processor = getattr(transformers, "AutoProcessor", None)
        if auto_processor is None:
            raise RuntimeError("Transformers AutoProcessor is unavailable")
        processor = auto_processor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_processor is None or video_processor is None or tokenizer is None:
            raise RuntimeError("AutoProcessor did not construct all multimodal components")
        runtime = {
            "transformers_version": version,
            "isolated": isolated,
            "local_files_only": True,
            "trust_remote_code": False,
            "processor_class": type(processor).__name__,
            "image_processor_class": type(image_processor).__name__,
            "video_processor_class": type(video_processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "declared_tokenizer_class": safe_request["processor"]["tokenizer_class"],
            "image_token_id": getattr(processor, "image_token_id", None),
            "video_token_id": getattr(processor, "video_token_id", None),
            "patch_size": getattr(image_processor, "patch_size", None),
            "temporal_patch_size": getattr(image_processor, "temporal_patch_size", None),
            "merge_size": getattr(image_processor, "merge_size", None),
        }
        response = build_mm1_processor_smoke_response(
            safe_request, manifest=manifest, inspection=inspection, runtime=runtime,
        )
        del processor, image_processor, video_processor, tokenizer
        gc.collect()
        result.update({
            "gate_passed": True,
            "status": "ready_for_offline_start",
            "response": response,
        })
        return result
    except Qwen3MultimodalPreflightError as exc:
        result["status"] = "processor_contract_rejected"
        result["errors"] = [{"code": "processor_contract_rejected", "message": exc.__class__.__name__}]
        return result
    except Exception as exc:
        result["status"] = "processor_smoke_failed"
        result["errors"] = [{"code": "processor_construction_failed", "message": exc.__class__.__name__}]
        return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("processor smoke request exceeds protocol limit")
    request = json.loads(raw.decode("utf-8"))
    result = execute_request(request)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result.get("valid") is not False else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        result = _base_result()
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
        print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
        raise SystemExit(2)
