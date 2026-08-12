"""Bounded, read-only inspection of Stable Diffusion LoRA Safetensors files."""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TOOL = "sd15_lora_inspect"
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_TENSOR_COUNT = 200_000
MAX_METADATA_VALUE_BYTES = 1024 * 1024
_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0FNU": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}
_NUMERIC_METADATA_KEYS = frozenset({
    "ss_epoch",
    "ss_learning_rate",
    "ss_max_train_steps",
    "ss_network_alpha",
    "ss_network_dim",
    "ss_num_reg_images",
    "ss_num_train_images",
    "ss_text_encoder_lr",
    "ss_unet_lr",
})
_NUMERIC_VALUE_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$",
)
_LORA_DOWN_MARKERS = ("lora_down.weight", ".down.weight")
_LORA_UP_MARKERS = ("lora_up.weight", ".up.weight")


class LoRAInspectError(ValueError):
    """Raised internally when a Safetensors header is unsafe or malformed."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def inspect_lora(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Inspect only a bounded Safetensors header and return a safe summary.

    No Safetensors library, CUDA context, tensor allocation, or weight data read
    is used.  ``root`` is optional; when supplied, symlink-resolved inputs must
    remain inside it.
    """

    target = Path(path).expanduser()
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "input_kind": "safetensors_file",
        "read_only": True,
        "weights_loaded": False,
        "cuda_used": False,
        "valid": False,
        "errors": [],
    }
    try:
        resolved = _resolve_input(target, root)
        if resolved.suffix.lower() != ".safetensors":
            raise LoRAInspectError(
                "unsupported_extension", "input must have a .safetensors extension",
            )
        file_size = int(resolved.stat().st_size)
        header, header_bytes_read = _read_header(resolved, file_size)
        metadata, tensor_names = _validate_header(header, file_size, header_bytes_read)
        report.update({
            "file_size_bytes": file_size,
            "header_bytes_read": header_bytes_read,
            "tensor_summary": _tensor_summary(tensor_names),
            "metadata": _metadata_summary(metadata),
            "valid": True,
            "errors": [],
        })
    except (OSError, LoRAInspectError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        code = exc.code if isinstance(exc, LoRAInspectError) else "unreadable_or_invalid"
        report["errors"] = [{"code": code, "message": _safe_error_message(exc)}]
    return report


def _resolve_input(path: Path, root: str | Path | None) -> Path:
    if not path.is_file():
        raise LoRAInspectError("file_missing", "input file does not exist")
    resolved = path.absolute().resolve(strict=True)
    if root is None:
        return resolved
    base = Path(root).expanduser()
    if not base.is_dir():
        raise LoRAInspectError("root_missing", "inspection root does not exist or is not a directory")
    try:
        resolved.relative_to(base.absolute().resolve(strict=True))
    except ValueError as exc:
        raise LoRAInspectError("path_outside_root", "input is outside the inspection root") from exc
    return resolved


def _read_header(path: Path, file_size: int) -> tuple[dict[str, Any], int]:
    if file_size < 8:
        raise LoRAInspectError("truncated_prefix", "Safetensors file is shorter than its header prefix")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise LoRAInspectError("truncated_prefix", "Safetensors header prefix is truncated")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length > MAX_HEADER_BYTES:
            raise LoRAInspectError("header_too_large", "Safetensors header exceeds the inspection limit")
        if header_length > file_size - 8:
            raise LoRAInspectError("truncated_header", "Safetensors header exceeds the file size")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise LoRAInspectError("truncated_header", "Safetensors header is truncated")
    try:
        decoded = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoRAInspectError("invalid_header_utf8", "Safetensors header is not UTF-8") from exc
    try:
        header = json.loads(decoded, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise LoRAInspectError("invalid_header_json", "Safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise LoRAInspectError("invalid_header_shape", "Safetensors header must be an object")
    return header, 8 + int(header_length)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LoRAInspectError("duplicate_header_key", "Safetensors header has duplicate keys")
        result[key] = value
    return result


def _validate_header(
    header: dict[str, Any], file_size: int, header_bytes_read: int,
) -> tuple[dict[str, str], list[str]]:
    metadata_value = header.get("__metadata__", {})
    if not isinstance(metadata_value, dict):
        raise LoRAInspectError("invalid_metadata", "Safetensors metadata must be an object")
    metadata: dict[str, str] = {}
    for key, value in metadata_value.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LoRAInspectError("invalid_metadata", "Safetensors metadata entries must be strings")
        metadata[key] = value

    tensors = [(key, value) for key, value in header.items() if key != "__metadata__"]
    if len(tensors) > MAX_TENSOR_COUNT:
        raise LoRAInspectError("too_many_tensors", "Safetensors header exceeds the tensor limit")
    data_size = file_size - header_bytes_read
    ranges: list[tuple[int, int]] = []
    names: list[str] = []
    for name, descriptor in tensors:
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise LoRAInspectError("invalid_tensor_descriptor", "Safetensors tensor descriptor is invalid")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if dtype not in _DTYPE_SIZES or not isinstance(shape, list):
            raise LoRAInspectError("invalid_tensor_descriptor", "Safetensors tensor dtype or shape is invalid")
        if (
            len(shape) > 64
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or item > 2**63 - 1
                for item in shape
            )
        ):
            raise LoRAInspectError("invalid_tensor_shape", "Safetensors tensor shape is invalid")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise LoRAInspectError("invalid_tensor_offsets", "Safetensors tensor offsets are invalid")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise LoRAInspectError("tensor_out_of_range", "Safetensors tensor range exceeds file data")
        expected_bytes = _tensor_byte_size(shape, _DTYPE_SIZES[dtype])
        if end - start != expected_bytes:
            raise LoRAInspectError("tensor_size_mismatch", "Safetensors tensor range does not match dtype and shape")
        ranges.append((start, end))
        names.append(name)
    for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] > current[0]:
            raise LoRAInspectError("overlapping_tensors", "Safetensors tensor ranges overlap")
    return metadata, names


def _tensor_byte_size(shape: list[int], element_size: int) -> int:
    elements = 1
    for dimension in shape:
        if dimension and elements > MAX_TENSOR_COUNT * MAX_HEADER_BYTES // dimension:
            raise LoRAInspectError("tensor_size_unreasonable", "Safetensors tensor shape is unreasonably large")
        elements *= dimension
    if elements > MAX_TENSOR_COUNT * MAX_HEADER_BYTES // element_size:
        raise LoRAInspectError("tensor_size_unreasonable", "Safetensors tensor shape is unreasonably large")
    return elements * element_size


def _tensor_summary(names: list[str]) -> dict[str, Any]:
    lowered = [name.lower() for name in names]
    down_count = sum(any(marker in name for marker in _LORA_DOWN_MARKERS) for name in lowered)
    up_count = sum(any(marker in name for marker in _LORA_UP_MARKERS) for name in lowered)
    alpha_count = sum(".alpha" in name or name.endswith("_alpha") for name in lowered)
    components: set[str] = set()
    for name in lowered:
        if "unet" in name:
            components.add("unet")
        elif "text_encoder_2" in name or "lora_te2" in name:
            components.add("text_encoder_2")
        elif "text_encoder" in name or "lora_te" in name:
            components.add("text_encoder")
    return {
        "tensor_count": len(names),
        "lora_down_tensor_count": down_count,
        "lora_up_tensor_count": up_count,
        "alpha_tensor_count": alpha_count,
        "lora_detected": down_count > 0 and up_count > 0,
        "components": sorted(components),
    }


def _metadata_summary(metadata: dict[str, str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    ss_items = sorted((key, value) for key, value in metadata.items() if key.startswith("ss_"))
    unknown_index = 0
    for key, value in ss_items:
        if (
            key in _NUMERIC_METADATA_KEYS
            and len(value) <= 128
            and _NUMERIC_VALUE_RE.fullmatch(value.strip())
        ):
            fields[key] = {"kind": "numeric", "value": value.strip()}
        elif key == "ss_dataset_dirs":
            fields[key] = _structured_summary(value, "dataset")
        elif key == "ss_tag_frequency":
            fields[key] = _structured_summary(value, "trigger")
        else:
            unknown_index += 1
            fields[f"ss_unknown_{unknown_index}"] = _redacted_text_summary(value)
    return {
        "ss_field_count": len(ss_items),
        "fields": fields,
    }


def _structured_summary(value: str, category: str) -> dict[str, Any]:
    summary = _redacted_text_summary(value)
    summary["kind"] = f"{category}_summary"
    if len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
        summary["parse_status"] = "not_parsed_too_large"
        return summary
    try:
        payload = json.loads(value, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, LoRAInspectError, RecursionError):
        summary["parse_status"] = "not_json"
        return summary
    if not isinstance(payload, dict):
        summary["parse_status"] = "not_object"
        return summary
    leaf_count, numeric_total = _count_numeric_leaves(payload)
    summary.update({
        "parse_status": "object",
        "group_count": len(payload),
        "numeric_entry_count": leaf_count,
        "numeric_value_total": numeric_total,
    })
    return summary


def _count_numeric_leaves(value: Any, depth: int = 0) -> tuple[int, int]:
    if depth > 8:
        return 0, 0
    if isinstance(value, dict):
        counts = [_count_numeric_leaves(item, depth + 1) for item in value.values()]
        return sum(item[0] for item in counts), sum(item[1] for item in counts)
    if isinstance(value, list):
        counts = [_count_numeric_leaves(item, depth + 1) for item in value]
        return sum(item[0] for item in counts), sum(item[1] for item in counts)
    if isinstance(value, int) and not isinstance(value, bool):
        return 1, value
    return 0, 0


def _redacted_text_summary(value: str) -> dict[str, Any]:
    return {
        "kind": "redacted_text",
        "value_length": len(value),
    }


def _safe_error_message(error: BaseException) -> str:
    if isinstance(error, LoRAInspectError):
        return str(error)
    if isinstance(error, OSError):
        return f"file read failed (errno={error.errno})"
    return "Safetensors inspection failed"


__all__ = [
    "LoRAInspectError",
    "MAX_HEADER_BYTES",
    "inspect_lora",
]
