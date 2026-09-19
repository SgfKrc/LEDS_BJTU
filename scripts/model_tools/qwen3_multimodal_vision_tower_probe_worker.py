"""Isolated Qwen3-VL vision tower weight-loading probe (MM1.15).

Loads only the vision tower (Qwen3VLVisionModel) from the real safetensors
shards (``visual.`` prefix), runs a synthetic image through the processor
and vision tower forward, and projects the real feature shape/dtype for
comparison against the MM1.14 synthetic placeholder.  Text weights are
never loaded; everything stays weight-scoped and path-free in responses.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_preflight import (  # noqa: E402
    Qwen3MultimodalPreflightError,
    build_mm1_media_tensor_reference,
)


TOOL = "qwen3_multimodal_vision_tower_probe"
SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 256 * 1024
MIN_RAM_GATE = 4 * 2**30
MAX_INDEX_BYTES = 2 * 1024 * 1024


def _base_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_tower_weight_smoke",
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


def execute_request(
    request: Mapping[str, Any],
    *,
    module_loader: Any = None,
) -> dict[str, Any]:
    result = _base_result()
    if (
        request.get("schema_version") != SCHEMA_VERSION
        or request.get("operation") != "qwen3_visual_tower_weight_smoke"
        or request.get("tool") != TOOL
        or request.get("read_only") is not True
        or request.get("network_access") != "disabled"
    ):
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "protocol_invalid", "message": "vision tower probe protocol is invalid"}]
        return result
    model_path = _safe_model_path(request.get("model_path"))
    if model_path is None:
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "request_incomplete", "message": "vision tower probe request is incomplete"}]
        return result
    try:
        import psutil
    except Exception as exc:
        result["status"] = "runtime_rejected"
        result["errors"] = [{"code": "memory_probe_unavailable", "message": exc.__class__.__name__}]
        return result
    if psutil.virtual_memory().available < MIN_RAM_GATE:
        result["status"] = "resource_rejected"
        result["errors"] = [{"code": "insufficient_ram", "message": "MM1.15 requires >= 4 GiB available RAM"}]
        return result
    try:
        from safetensors import safe_open
        import torch
        from transformers import AutoProcessor, AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    except Exception as exc:
        result["status"] = "runtime_rejected"
        result["errors"] = [{"code": "vision_runtime_unavailable", "message": exc.__class__.__name__}]
        return result

    try:
        config = AutoConfig.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise Qwen3MultimodalPreflightError("model has no vision_config")
        if str(getattr(config, "model_type", "")) != "qwen3_vl":
            raise Qwen3MultimodalPreflightError("model is not a Qwen3-VL checkpoint")

        # 只构造视觉塔（不构造文本/embedding）
        vision_model = Qwen3VLVisionModel(vision_config)

        # 从 safetensors 分片 filter 加载 visual. 前缀权重
        index_path = model_path / "model.safetensors.index.json"
        visual_map: dict[str, Path] = {}
        if index_path.is_file():
            try:
                if index_path.stat().st_size <= 0 or index_path.stat().st_size > MAX_INDEX_BYTES:
                    raise Qwen3MultimodalPreflightError("safetensors index exceeds size limit")
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Qwen3MultimodalPreflightError("safetensors index is invalid") from exc
            if not isinstance(weight_map, dict):
                raise Qwen3MultimodalPreflightError("safetensors index has no weight_map")
            for key, shard_name in weight_map.items():
                if not isinstance(key, str) or not key.startswith("model.visual."):
                    continue
                if not isinstance(shard_name, str):
                    raise Qwen3MultimodalPreflightError("safetensors shard name is invalid")
                shard = (model_path / shard_name).resolve(strict=False)
                if shard.parent != model_path or not shard.is_file():
                    raise Qwen3MultimodalPreflightError("safetensors shard escapes model directory")
                if key in visual_map and visual_map[key] != shard:
                    raise Qwen3MultimodalPreflightError("visual weight is mapped to multiple shards")
                visual_map[key] = shard
        else:
            for shard in sorted(model_path.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key.startswith("model.visual."):
                            visual_map[key] = shard
        if not visual_map:
            raise Qwen3MultimodalPreflightError("no visual weights found in safetensors index")

        expected = set(vision_model.state_dict().keys())
        parameters = dict(vision_model.named_parameters())
        buffers = dict(vision_model.named_buffers())
        loaded_keys: set[str] = set()
        for full_key, shard in sorted(visual_map.items()):
            key = full_key[len("model.visual."):]
            target = parameters.get(key)
            if target is None:
                target = buffers.get(key)
            if target is None:
                raise Qwen3MultimodalPreflightError(f"unexpected vision tower key: {key}")
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(full_key)
            if tuple(tensor.shape) != tuple(target.shape):
                raise Qwen3MultimodalPreflightError(f"vision tower shape mismatch: {key}")
            with torch.no_grad():
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))
            loaded_keys.add(key)
            del tensor
        missing = expected - loaded_keys
        if missing:
            raise Qwen3MultimodalPreflightError(
                f"vision tower missing keys: {sorted(missing)[:5]}",
            )
        vision_model.eval()

        # 合成图像 → processor 预处理（真实像素管线）
        import numpy as np
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
        )
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise Qwen3MultimodalPreflightError("processor has no image_processor")
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        inputs = image_processor(image, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or grid_thw is None:
            raise Qwen3MultimodalPreflightError("processor produced no pixel values/grid")

        with torch.no_grad():
            vision_out = vision_model(pixel_values, grid_thw)
        # ⚠️ 返回形状与语义**跨版本都不同**（2026-09-19 实测）：
        #   * transformers 4.x：`(merged_hidden_states, deepstack_features)` 的 **2 元 tuple**；
        #   * transformers 5.x：`BaseModelOutputWithDeepstackFeatures`，其中
        #       `last_hidden_state`  = merger **之前**（hidden_size=1024）
        #       `pooler_output`      = merger **之后**（投影到文本维度）★ 才是「图像嵌入」
        #       `deepstack_features` = 各 deepstack 层特征
        #     且 ModelOutput 的 `__iter__` 会迭代**所有非 None 字段** ⇒ 直接解包会
        #     `ValueError: too many values to unpack (expected 2)`。
        # 故：**按属性取、回退按位置取**，并确保取的是 **merger 之后** 的那个。
        image_embeds = getattr(vision_out, "pooler_output", None)
        deepstack_features = getattr(vision_out, "deepstack_features", None)
        if not hasattr(vision_out, "last_hidden_state"):
            # 4.x：tuple 形态，位置 0 即 merger 之后
            image_embeds = vision_out[0]
            if len(vision_out) > 1:
                deepstack_features = vision_out[1]
        elif image_embeds is None:
            image_embeds = vision_out.last_hidden_state  # 理论上不会发生
        _deepstack = deepstack_features  # 保留原名以最小化改动
        seq_len = int(image_embeds.shape[0])
        hidden_dim = int(image_embeds.shape[-1])

        # 合成占位对照（MM1.14 投影口径）
        synthetic_summary = {
            "image": {
                "pixel_values_shape": [int(value) for value in pixel_values.shape],
                "dtype": str(pixel_values.dtype),
                "token_count_estimate": seq_len,
            },
            "video": {"pixel_values_shape": [], "dtype": "", "token_count_estimate": 0},
            "output_bytes_estimate": seq_len * hidden_dim * int(image_embeds.element_size()),
            "weight_materialized": False,
            "full_model_materialized": False,
        }
        synthetic_reference = build_mm1_media_tensor_reference(
            synthetic_summary,
            model_id=str(config.model_type),
            component_ids=["vision_tower"],
        )

        result.update({
            "gate_passed": True,
            "status": "vision_tower_weights_loaded",
            "response": {
                "schema_version": SCHEMA_VERSION,
                "response_kind": "qwen3_visual_tower_weight_smoke",
                "model_id": str(config.model_type),
                "vision_tower": {
                    "class_name": type(vision_model).__name__,
                    "depth": int(vision_config.depth),
                    "hidden_size": int(vision_config.hidden_size),
                    "loaded_weights": len(loaded_keys),
                    "shards_loaded": len(set(visual_map.values())),
                },
                "real_feature": {
                    "shape": [1, seq_len, hidden_dim],
                    "dtype": str(image_embeds.dtype),
                },
                "synthetic_reference_sha256": synthetic_reference["reference_sha256"],
                "consistency": {
                    "tokens_match": bool(
                        seq_len == int(
                            synthetic_reference["capacity"]["total_media_tokens"],
                        ),
                    ),
                    # 视觉特征经 merger 投影到文本段 hidden（visual_to_text 边界）
                    "hidden_matches_text_config": bool(
                        hidden_dim == int(config.text_config.hidden_size),
                    ),
                    "hidden_matches_vision_config": bool(
                        hidden_dim == int(vision_config.hidden_size),
                    ),
                },
                "weight_materialized": True,   # 视觉塔权重已加载（如实登记）
                "full_model_materialized": False,
                "text_weights_loaded": False,
            },
        })
        return result
    except Qwen3MultimodalPreflightError as exc:
        result["status"] = "vision_tower_contract_rejected"
        result["errors"] = [{"code": "vision_tower_contract_rejected",
                             "message": _debug_message(exc)}]
        return result
    except Exception as exc:
        result["status"] = "vision_tower_load_failed"
        result["errors"] = [{"code": "vision_tower_load_failed",
                             "message": _debug_message(exc)}]
        return result
    finally:
        # Drop references on both success and failure; the worker is isolated,
        # but explicit cleanup keeps repeated probes from retaining CPU pages.
        vision_model = processor = image_embeds = pixel_values = inputs = None
        gc.collect()


def _debug_message(exc: BaseException) -> str:
    """错误信息：默认只给**类名**（fail-closed，不把内部路径/细节带进控制面响应）。

    设 `QLH_MM_DEBUG=1` 时附带异常文本，便于本地定位（主运行时升级后出现过
    「只给类名无从下手」的情况）。
    """
    if os.environ.get("QLH_MM_DEBUG", "").strip() in {"1", "true", "yes"}:
        return f"{exc.__class__.__name__}: {exc}"
    return exc.__class__.__name__


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("vision tower probe request exceeds protocol limit")
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
