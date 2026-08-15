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
    build_mm1_media_tensor_reference,
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


def _run_media_preprocess_smoke(
    processor: Any,
    image_processor: Any,
    video_processor: Any,
    media: Mapping[str, Any],
) -> dict[str, Any]:
    """MM1.7：合成媒体预处理与张量摘要（受限尺寸/帧数，只投影摘要）。

    控制面契约限制：图像/视频边长 ≤ 1024、帧数 1..32；pixel_values、
    原始媒体、路径与 prompt 一律不进入响应。
    """
    import numpy as np

    image_size = media.get("image_size") or (32, 32)
    video_size = media.get("video_size") or (32, 32)
    video_frames = int(media.get("video_frames", 2))  # 不用 or：0 帧是非法值必须拒绝
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("media image_size must be [height, width]")
    if not isinstance(video_size, (list, tuple)) or len(video_size) != 2:
        raise ValueError("media video_size must be [height, width]")
    image_h, image_w = (int(v) for v in image_size)
    video_h, video_w = (int(v) for v in video_size)
    if (
        not 8 <= image_h <= 1024 or not 8 <= image_w <= 1024
        or not 8 <= video_h <= 1024 or not 8 <= video_w <= 1024
        or not 1 <= video_frames <= 32
    ):
        raise ValueError("media dimensions are outside the MM1.7 contract limits")

    # 合成媒体（内存 numpy，避免 PIL 依赖）
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(image_h, image_w, 3), dtype=np.uint8)
    frames = [
        rng.integers(0, 256, size=(video_h, video_w, 3), dtype=np.uint8)
        for _ in range(video_frames)
    ]

    image_inputs = image_processor(image, return_tensors="pt")
    video_inputs = video_processor(frames, return_tensors="pt")

    def _tensor_shape(value: Any) -> list[int]:
        tensor = (
            value.get("pixel_values")
            if isinstance(value, dict)
            else getattr(value, "pixel_values", value)
        )
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0] if tensor else None
        shape = getattr(tensor, "shape", None)
        return [int(v) for v in shape] if shape is not None else []

    image_pixels = _tensor_shape(image_inputs)
    video_pixels = _tensor_shape(video_inputs)

    # token 数摘要（geometry：patch 网格；真实 processor 的 pixel_values
    # 可能是 [H,W] 网格或 [B,C,H,W]——统一按最后两维 ÷ patch² 估算）
    def _tokens_from_shape(shape: list[int], patch: int | None) -> int | None:
        if len(shape) < 2 or not patch:
            return None
        h, w = shape[-2], shape[-1]
        return max(1, (int(h) // patch) * (int(w) // patch))

    image_tokens = _tokens_from_shape(image_pixels, getattr(image_processor, "patch_size", None))
    video_tokens = _tokens_from_shape(video_pixels, getattr(video_processor, "patch_size", None))
    image_dtype = str(getattr(image_inputs.get("pixel_values"), "dtype", "")) if isinstance(image_inputs, dict) else ""
    video_dtype = str(getattr(video_inputs.get("pixel_values"), "dtype", "")) if isinstance(video_inputs, dict) else ""
    if not image_dtype:
        pixels = getattr(image_inputs, "pixel_values", None)
        if pixels is not None:
            image_dtype = str(getattr(pixels, "dtype", ""))
    if not video_dtype:
        pixels = getattr(video_inputs, "pixel_values", None)
        if pixels is not None:
            video_dtype = str(getattr(pixels, "dtype", ""))

    output_bytes = 0
    for value in (image_pixels, video_pixels):
        if value:
            item = 1
            for dim in value:
                item *= max(1, int(dim))
            output_bytes += item * 2  # fp16 估算（dtype 不定时保守）
    summary = {
        "image": {
            "requested_size": [image_h, image_w],
            "pixel_values_shape": image_pixels,
            "dtype": image_dtype,
            "token_count_estimate": image_tokens,
        },
        "video": {
            "requested_frames": video_frames,
            "requested_size": [video_h, video_w],
            "pixel_values_shape": video_pixels,
            "dtype": video_dtype,
            "token_count_estimate": video_tokens,
        },
        "output_bytes_estimate": output_bytes,
        "weight_materialized": False,
        "full_model_materialized": False,
    }
    del image_inputs, video_inputs
    gc.collect()
    return summary


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
        # MM1.7：CPU 合成媒体预处理与张量摘要合同——用受限合成图像/帧
        # 调用 processor 预处理，只投影 shape/dtype/token 摘要；不加载
        # 视觉塔或文本权重（processor 预处理纯计算）。
        media = request.get("media_smoke") or {}
        try:
            media_summary = _run_media_preprocess_smoke(
                processor, image_processor, video_processor, media,
            )
        except Exception as exc:
            # MM1.7：媒体超限/异常一律契约拒绝（fail-closed）
            raise Qwen3MultimodalPreflightError(
                f"media preprocess failed: {exc.__class__.__name__}",
            ) from exc
        # MM1.8：投影为 path-free 媒体张量参考（视觉组件占位/容量预算）
        media_tensor_reference = build_mm1_media_tensor_reference(
            media_summary,
            model_id=safe_request["model_id"],
            component_ids=safe_request["component_ids"],
        )
        response = build_mm1_processor_smoke_response(
            safe_request, manifest=manifest, inspection=inspection, runtime=runtime,
            media_summary=media_summary,
            media_tensor_reference=media_tensor_reference,
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
