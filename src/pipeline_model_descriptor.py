"""Metadata-only inspection for distributed PyTorch pipeline models.

The inspector reads ``config.json``, the Safetensors index, and tensor headers.
It never calls ``from_pretrained`` or ``safe_open.get_tensor`` and therefore
does not materialize model weights in host RAM or VRAM.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DESCRIPTOR_SCHEMA_VERSION = 1
# These model types have an in-process loader and forward executor.  Architectures
# may still have a metadata/assignment layout below while remaining fail-closed
# here when they require an isolated runtime.
PIPELINE_RUNTIME_MODEL_TYPES = frozenset({"qwen", "qwen2"})
_MAX_JSON_BYTES = 64 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class PipelineModelDescriptorError(ValueError):
    """The local artifact cannot produce a safe pipeline descriptor."""


_ARCHITECTURE_LAYOUTS = {
    "qwen": {
        "layer_pattern": re.compile(r"^transformer\.h\.(\d+)\."),
        "layer_prefix": "transformer.h.",
        "embedding_prefixes": ("transformer.wte.",),
        "final_norm_prefixes": ("transformer.ln_f.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen2": {
        "layer_pattern": re.compile(r"^model\.layers\.(\d+)\."),
        "layer_prefix": "model.layers.",
        "embedding_prefixes": ("model.embed_tokens.",),
        "final_norm_prefixes": ("model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen3": {
        "layer_pattern": re.compile(r"^model\.layers\.(\d+)\."),
        "layer_prefix": "model.layers.",
        "embedding_prefixes": ("model.embed_tokens.",),
        "final_norm_prefixes": ("model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
    },
    "qwen3_vl": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": ("model.visual.", "visual."),
        "mtp_prefixes": (),
    },
    "qwen3_5": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": ("model.visual.", "visual."),
        "mtp_prefixes": ("mtp.",),
        "multimodal_prefixes": (),
    },
    "gemma4_unified": {
        "layer_pattern": re.compile(
            r"^model\.language_model\.layers\.(\d+)\."
        ),
        "layer_prefix": "model.language_model.layers.",
        "embedding_prefixes": ("model.language_model.embed_tokens.",),
        "final_norm_prefixes": ("model.language_model.norm.",),
        "lm_head_prefixes": ("lm_head.",),
        "visual_prefixes": (),
        "mtp_prefixes": (),
        "multimodal_prefixes": (
            "model.embed_vision.",
            "model.embed_audio.",
        ),
    },
}

for _layout in _ARCHITECTURE_LAYOUTS.values():
    _layout.setdefault("multimodal_prefixes", ())


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PipelineModelDescriptorError(f"模型元数据不可读: {path.name}") from exc
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise PipelineModelDescriptorError(
            f"模型元数据大小异常: {path.name} ({size} bytes)"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineModelDescriptorError(f"模型元数据不是有效 JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise PipelineModelDescriptorError(f"模型元数据必须是对象: {path.name}")
    return payload


def _decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _safe_shard_path(root: Path, filename: str) -> Path:
    if not filename or Path(filename).is_absolute():
        raise PipelineModelDescriptorError("Safetensors 索引包含无效分片路径")
    shard = (root / filename).resolve()
    try:
        shard.relative_to(root.resolve())
    except ValueError as exc:
        raise PipelineModelDescriptorError("Safetensors 分片路径越出模型目录") from exc
    if not shard.is_file() or shard.suffix.lower() != ".safetensors":
        raise PipelineModelDescriptorError(f"Safetensors 分片不存在: {filename}")
    return shard


def _weight_map(root: Path) -> tuple[dict[str, str], int]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_object(index_path)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise PipelineModelDescriptorError("Safetensors 索引缺少 weight_map")
        weight_map: dict[str, str] = {}
        for key, filename in raw_map.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise PipelineModelDescriptorError("Safetensors weight_map 字段类型无效")
            weight_map[key] = filename
        metadata = index.get("metadata")
        declared_size = 0
        if isinstance(metadata, dict):
            try:
                declared_size = int(metadata.get("total_size", 0) or 0)
            except (TypeError, ValueError):
                declared_size = 0
        return weight_map, declared_size

    from safetensors import safe_open

    weight_map = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in weight_map:
                    raise PipelineModelDescriptorError(
                        f"Safetensors tensor 重名: {key}"
                    )
                weight_map[key] = shard.name
    if not weight_map:
        raise PipelineModelDescriptorError("模型目录中没有 Safetensors 权重")
    return weight_map, 0


def _tensor_nbytes(tensor_slice: Any) -> int:
    dtype = str(tensor_slice.get_dtype()).upper()
    element_size = _DTYPE_BYTES.get(dtype)
    if element_size is None:
        raise PipelineModelDescriptorError(f"不支持的 Safetensors dtype: {dtype}")
    shape = tensor_slice.get_shape()
    try:
        elements = math.prod(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise PipelineModelDescriptorError("Safetensors tensor shape 无效") from exc
    if elements < 0:
        raise PipelineModelDescriptorError("Safetensors tensor shape 不能为负数")
    return elements * element_size


def _component_for_key(key: str, layout: dict[str, Any]) -> tuple[str, int | None]:
    layer_match = layout["layer_pattern"].match(key)
    if layer_match:
        return "layers", int(layer_match.group(1))
    for component, field in (
        ("embedding", "embedding_prefixes"),
        ("final_norm", "final_norm_prefixes"),
        ("lm_head", "lm_head_prefixes"),
        ("visual", "visual_prefixes"),
        ("mtp", "mtp_prefixes"),
        ("multimodal", "multimodal_prefixes"),
    ):
        if any(key.startswith(prefix) for prefix in layout[field]):
            return component, None
    return "other", None


def inspect_pipeline_model(
    model_path: str | Path,
    *,
    model_id: str = "",
    layer_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build an exact, weight-free pipeline descriptor for a local model."""

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise PipelineModelDescriptorError("模型目录不存在")
    config = _read_json_object(root / "config.json")
    model_type = str(config.get("model_type", "") or "").strip().lower()
    layout = _ARCHITECTURE_LAYOUTS.get(model_type)
    if layout is None:
        raise PipelineModelDescriptorError(
            f"尚未登记流水线元数据布局: {model_type or 'unknown'}"
        )

    decoder_config = _decoder_config(config)
    try:
        total_layers = int(decoder_config.get("num_hidden_layers", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise PipelineModelDescriptorError("config 的 num_hidden_layers 无效") from exc
    if total_layers <= 0:
        raise PipelineModelDescriptorError("config 缺少有效 num_hidden_layers")

    weight_map, declared_weight_bytes = _weight_map(root)
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_shard[filename].append(key)

    from safetensors import safe_open

    component_bytes = {
        "layers": 0,
        "embedding": 0,
        "final_norm": 0,
        "lm_head": 0,
        "visual": 0,
        "mtp": 0,
        "multimodal": 0,
        "other": 0,
    }
    layer_bytes = [0] * total_layers
    observed_weight_bytes = 0
    for filename in sorted(keys_by_shard):
        shard = _safe_shard_path(root, filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available_keys = set(handle.keys())
            for key in keys_by_shard[filename]:
                if key not in available_keys:
                    raise PipelineModelDescriptorError(
                        f"Safetensors 索引引用不存在的 tensor: {key}"
                    )
                nbytes = _tensor_nbytes(handle.get_slice(key))
                component, layer_index = _component_for_key(key, layout)
                if layer_index is not None:
                    if layer_index >= total_layers:
                        raise PipelineModelDescriptorError(
                            f"权重层号越界: {layer_index} >= {total_layers}"
                        )
                    layer_bytes[layer_index] += nbytes
                component_bytes[component] += nbytes
                observed_weight_bytes += nbytes

    partial_range = None
    if layer_range is not None:
        try:
            partial_range = (int(layer_range[0]), int(layer_range[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise PipelineModelDescriptorError("layer_range is invalid") from exc
        if (
            partial_range[0] < 0
            or partial_range[1] <= partial_range[0]
            or partial_range[1] > total_layers
        ):
            raise PipelineModelDescriptorError("layer_range is outside model bounds")
    missing_layers = [
        index for index, value in enumerate(layer_bytes)
        if value <= 0 and (
            partial_range is None
            or partial_range[0] <= index < partial_range[1]
        )
    ]
    if missing_layers:
        shown = ", ".join(str(value) for value in missing_layers[:8])
        raise PipelineModelDescriptorError(f"Safetensors 缺少声明层: {shown}")
    if declared_weight_bytes and not partial_range and declared_weight_bytes != observed_weight_bytes:
        raise PipelineModelDescriptorError(
            "Safetensors 索引总字节与文件头不一致: "
            f"declared={declared_weight_bytes}, observed={observed_weight_bytes}"
        )

    runtime_supported = model_type in PIPELINE_RUNTIME_MODEL_TYPES
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []
    descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "inspection_mode": "safetensors_headers_only",
        "model_id": str(model_id or ""),
        "model_type": model_type,
        "architectures": [str(value) for value in architectures],
        "total_layers": total_layers,
        "layer_prefix": layout["layer_prefix"],
        "weight_bytes": observed_weight_bytes,
        "indexed_tensor_count": len(weight_map),
        "weight_file_count": len(keys_by_shard),
        "layer_weight_bytes": layer_bytes,
        "component_weight_bytes": component_bytes,
        "tie_word_embeddings": bool(decoder_config.get("tie_word_embeddings", False)),
        "pipeline_runtime_supported": runtime_supported,
        "runtime_block_reason": (
            "" if runtime_supported
            else (
                "gemma4_unified 需要隔离 Transformers sidecar；"
                "主运行时禁止复用 Qwen2 执行器"
                if model_type == "gemma4_unified"
                else f"{model_type} 尚未实现按层执行 adapter"
            )
        ),
        "partial_assignment": bool(partial_range),
    }
    if partial_range:
        descriptor["assignment_layer_range"] = list(partial_range)
    return descriptor


def public_pipeline_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Return the non-sensitive projection used by APIs and diagnostics."""

    allowed = {
        "schema_version",
        "inspection_mode",
        "model_id",
        "model_type",
        "architectures",
        "total_layers",
        "layer_prefix",
        "weight_bytes",
        "indexed_tensor_count",
        "weight_file_count",
        "layer_weight_bytes",
        "component_weight_bytes",
        "tie_word_embeddings",
        "pipeline_runtime_supported",
        "runtime_block_reason",
        "model_sha256",
    }
    return {key: descriptor[key] for key in allowed if key in descriptor}
