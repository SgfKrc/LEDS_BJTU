"""Canonical control contract for Gemma 4 Unified text sidecars."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable


SCHEMA_VERSION = 1
CONTRACT_KIND = "gemma4_pipeline_sidecar"
TRANSFORMERS_VERSION = "5.10.1"
MAX_CONTRACT_BYTES = 64 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_TYPES = {"full_attention", "sliding_attention"}
_BANNED_KEYS = {
    "model_path", "prompt", "messages", "input_ids", "hidden_states",
    "past_key_values", "shared_kv_states", "logits", "weights", "tensor",
    "tensors", "artifact_root", "input_ref", "output_ref", "kv_ref",
}


class Gemma4PipelineContractError(ValueError):
    """A Gemma 4 pipeline contract cannot be proven canonical and safe."""


def _canonical_bytes(value: Any, *, label: str = "contract") -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Gemma4PipelineContractError(f"{label} is not JSON serializable") from exc
    if len(encoded) > MAX_CONTRACT_BYTES:
        raise Gemma4PipelineContractError(f"{label} exceeds 64 KiB")
    return encoded


def _digest(value: Any, *, label: str = "contract") -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise Gemma4PipelineContractError(f"{label} must be a lowercase SHA-256")
    return normalized


def _reject_sensitive(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _BANNED_KEYS:
                raise Gemma4PipelineContractError(
                    f"Gemma 4 control contract cannot contain {key}"
                )
            _reject_sensitive(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive(item)


def _shared_kv_sources(
    layer_types: list[str], num_kv_shared_layers: int,
) -> tuple[int, dict[str, int]]:
    total_layers = len(layer_types)
    first_shared = total_layers - num_kv_shared_layers
    if num_kv_shared_layers == 0:
        return total_layers, {}
    shared_types = set(layer_types[first_shared:])
    sources: dict[str, int] = {}
    for layer_type in sorted(shared_types):
        candidates = [
            index for index in range(first_shared)
            if layer_types[index] == layer_type
        ]
        if not candidates:
            raise Gemma4PipelineContractError(
                f"shared-KV type has no source layer: {layer_type}"
            )
        sources[layer_type] = candidates[-1]
    return first_shared, sources


def build_gemma4_pipeline_contract(
    *,
    config_id: str,
    plan_id: str,
    generation: int,
    model_id: str,
    model_sha256: str,
    total_layers: int,
    hidden_size: int,
    layer_types: Iterable[str],
    num_kv_shared_layers: int,
    segments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build a two/three-segment, text-only Gemma 4 sidecar contract."""
    config_id = str(config_id or "")
    plan_id = str(plan_id or "")
    model_id = str(model_id or "")
    if not config_id or not plan_id or not model_id:
        raise Gemma4PipelineContractError("Gemma 4 pipeline identity is incomplete")
    try:
        generation = int(generation)
        total_layers = int(total_layers)
        hidden_size = int(hidden_size)
        num_kv_shared_layers = int(num_kv_shared_layers)
    except (TypeError, ValueError) as exc:
        raise Gemma4PipelineContractError("Gemma 4 dimensions are invalid") from exc
    if (
        generation <= 0
        or total_layers <= 0
        or hidden_size <= 0
        or num_kv_shared_layers < 0
        or num_kv_shared_layers >= total_layers
    ):
        raise Gemma4PipelineContractError("Gemma 4 dimensions are invalid")
    normalized_layer_types = [str(value) for value in layer_types]
    if (
        len(normalized_layer_types) != total_layers
        or any(value not in _LAYER_TYPES for value in normalized_layer_types)
    ):
        raise Gemma4PipelineContractError("Gemma 4 layer_types are invalid")
    first_shared, sources = _shared_kv_sources(
        normalized_layer_types, num_kv_shared_layers,
    )
    raw_segments = list(segments)
    if len(raw_segments) not in {2, 3}:
        raise Gemma4PipelineContractError("Gemma 4 pipeline requires two or three segments")
    normalized: list[dict[str, Any]] = []
    expected_start = 0
    nodes: set[str] = set()
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise Gemma4PipelineContractError("Gemma 4 segment must be an object")
        _reject_sensitive(raw)
        node_id = str(raw.get("node_id", "") or "")
        layer_range = raw.get("layer_range")
        if not node_id or node_id in nodes:
            raise Gemma4PipelineContractError("Gemma 4 node IDs must be unique")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            raise Gemma4PipelineContractError("Gemma 4 layer_range is invalid")
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
            required_bytes = int(raw.get("required_bytes", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise Gemma4PipelineContractError("Gemma 4 segment dimensions are invalid") from exc
        if start != expected_start or end <= start or end > total_layers:
            raise Gemma4PipelineContractError(
                "Gemma 4 segments must be contiguous and in bounds"
            )
        if required_bytes <= 0:
            raise Gemma4PipelineContractError("Gemma 4 required_bytes is invalid")
        has_embedding = bool(raw.get("has_embedding", False))
        has_lm_head = bool(raw.get("has_lm_head", False))
        if has_embedding != (index == 0):
            raise Gemma4PipelineContractError("only the first segment owns embedding")
        if has_lm_head != (index == len(raw_segments) - 1):
            raise Gemma4PipelineContractError("only the last segment owns LM Head")
        device = str(raw.get("execution_device", "cpu") or "cpu")
        dtype = str(raw.get("dtype", "float32") or "float32")
        if device not in {"cpu", "cuda"}:
            raise Gemma4PipelineContractError("Gemma 4 execution_device is invalid")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise Gemma4PipelineContractError("Gemma 4 dtype is invalid")
        produces = sorted(
            layer_type for layer_type, source in sources.items()
            if start <= source < end
        )
        requires = sorted(
            layer_type for layer_type, source in sources.items()
            if source < start
            and any(
                normalized_layer_types[layer] == layer_type
                for layer in range(max(start, first_shared), end)
            )
        )
        propagates = sorted(
            layer_type for layer_type, source in sources.items()
            if source < end
            and any(
                normalized_layer_types[layer] == layer_type
                for layer in range(max(end, first_shared), total_layers)
            )
        )
        segment = {
            "segment_index": index,
            "node_id": node_id,
            "layer_range": [start, end],
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "assignment_manifest_sha256": _sha256(
                raw.get("assignment_manifest_sha256"),
                "assignment_manifest_sha256",
            ),
            "required_bytes": required_bytes,
            "execution_device": device,
            "dtype": dtype,
            "produces_shared_kv_types": produces,
            "requires_shared_kv_types": requires,
            "propagates_shared_kv_types": propagates,
        }
        segment["segment_sha256"] = _digest(segment, label="segment")
        normalized.append(segment)
        nodes.add(node_id)
        expected_start = end
    if expected_start != total_layers:
        raise Gemma4PipelineContractError("Gemma 4 segments do not cover all layers")
    handoffs: list[dict[str, Any]] = []
    for left, right in zip(normalized, normalized[1:]):
        shared_types = sorted(set(right["requires_shared_kv_types"]) | set(left["propagates_shared_kv_types"]))
        handoff = {
            "from_segment": left["segment_index"],
            "to_segment": right["segment_index"],
            "from_node_id": left["node_id"],
            "to_node_id": right["node_id"],
            "rank": 3,
            "hidden_size": hidden_size,
            "transport_device": "cpu",
            "shared_kv_types": shared_types,
            "shared_kv_sequence_axis": -2,
            "generation_required": True,
        }
        handoff["handoff_sha256"] = _digest(handoff, label="handoff")
        handoffs.append(handoff)
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_kind": CONTRACT_KIND,
        "config_id": config_id,
        "plan_id": plan_id,
        "generation": generation,
        "model_id": model_id,
        "model_sha256": _sha256(model_sha256, "model_sha256"),
        "model_type": "gemma4_unified",
        "total_layers": total_layers,
        "hidden_size": hidden_size,
        "layer_types": normalized_layer_types,
        "num_kv_shared_layers": num_kv_shared_layers,
        "first_shared_kv_layer": first_shared,
        "shared_kv_source_layers": sources,
        "segments": normalized,
        "handoffs": handoffs,
        "execution_mode": "node_local_sidecar",
        "runtime_environment": ".venv-gemma4-pipeline",
        "transformers_version": TRANSFORMERS_VERSION,
        "full_model_fallback": False,
        "multimodal_materialized": False,
        "production_admitted": False,
    }
    contract["contract_sha256"] = _digest(contract)
    _canonical_bytes(contract)
    return contract


def validate_gemma4_pipeline_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise Gemma4PipelineContractError("Gemma 4 contract must be an object")
    _reject_sensitive(contract)
    expected = _sha256(contract.get("contract_sha256"), "contract_sha256")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if _digest(payload) != expected:
        raise Gemma4PipelineContractError("Gemma 4 contract digest mismatch")
    rebuilt = build_gemma4_pipeline_contract(
        config_id=contract.get("config_id", ""),
        plan_id=contract.get("plan_id", ""),
        generation=contract.get("generation", 0),
        model_id=contract.get("model_id", ""),
        model_sha256=contract.get("model_sha256", ""),
        total_layers=contract.get("total_layers", 0),
        hidden_size=contract.get("hidden_size", 0),
        layer_types=contract.get("layer_types", []),
        num_kv_shared_layers=contract.get("num_kv_shared_layers", -1),
        segments=contract.get("segments", []),
    )
    if rebuilt != contract:
        raise Gemma4PipelineContractError("Gemma 4 contract is not canonical")
    return rebuilt


__all__ = [
    "CONTRACT_KIND",
    "Gemma4PipelineContractError",
    "SCHEMA_VERSION",
    "TRANSFORMERS_VERSION",
    "build_gemma4_pipeline_contract",
    "validate_gemma4_pipeline_contract",
]
