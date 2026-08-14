"""Runtime-safe Qwen3 segment and KV metadata contracts.

This module intentionally has no dependency on ``scripts.model_tools``.  The
Scheduler and packaged worker processes may import it when the repository root
is not on ``sys.path`` (for example, an installed helper process).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


QWEN3_HANDOFF_SCHEMA_VERSION = 1
QWEN3_KV_CONTRACT_VERSION = 1


class Qwen3PipelineContractError(ValueError):
    """A segment topology or metadata contract is not admissible."""


def validate_segment_plan(
    segments: Iterable[dict[str, Any]], *, total_layers: int,
) -> list[dict[str, Any]]:
    values = list(segments)
    if len(values) not in {2, 3}:
        raise Qwen3PipelineContractError(
            "Qwen3 chain requires exactly two or three segments",
        )
    total_layers = int(total_layers)
    if total_layers <= 0:
        raise Qwen3PipelineContractError("Qwen3 chain total layer count is invalid")
    normalized: list[dict[str, Any]] = []
    expected_start = 0
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise Qwen3PipelineContractError("Qwen3 chain segment must be an object")
        layer_range = raw.get("layer_range")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            raise Qwen3PipelineContractError("Qwen3 chain segment layer_range is invalid")
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
        except (TypeError, ValueError) as exc:
            raise Qwen3PipelineContractError(
                "Qwen3 chain segment layer_range is invalid",
            ) from exc
        if start != expected_start or start < 0 or end <= start or end > total_layers:
            raise Qwen3PipelineContractError(
                "Qwen3 chain segments must be contiguous and in bounds",
            )
        has_embedding = bool(raw.get("has_embedding", False))
        has_lm_head = bool(raw.get("has_lm_head", False))
        if has_embedding != (index == 0):
            raise Qwen3PipelineContractError(
                "only the first Qwen3 segment may own embedding",
            )
        if has_lm_head != (index == len(values) - 1):
            raise Qwen3PipelineContractError(
                "only the last Qwen3 segment may own LM Head",
            )
        normalized.append({
            "segment_index": index,
            "layer_range": [start, end],
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "device": str(raw.get("device", "auto")),
            "dtype": raw.get("dtype"),
        })
        expected_start = end
    if expected_start != total_layers:
        raise Qwen3PipelineContractError(
            "Qwen3 chain does not cover all model layers",
        )
    return normalized


def build_kv_contract(
    *, chain_id: str, segment_index: int, layer_range: Sequence[int],
    sequence_length: int, batch_size: int, dtype: str, device: str,
    phase: str, generation: int,
) -> dict[str, Any]:
    if phase not in {"prefill", "decode"}:
        raise Qwen3PipelineContractError("KV contract phase is invalid")
    if len(layer_range) != 2 or int(layer_range[1]) <= int(layer_range[0]):
        raise Qwen3PipelineContractError("KV contract layer range is invalid")
    if int(sequence_length) <= 0 or int(batch_size) <= 0 or int(generation) < 0:
        raise Qwen3PipelineContractError("KV contract dimensions are invalid")
    return {
        "schema_version": QWEN3_KV_CONTRACT_VERSION,
        "chain_id": str(chain_id),
        "segment_index": int(segment_index),
        "layer_range": [int(layer_range[0]), int(layer_range[1])],
        "sequence_length": int(sequence_length),
        "batch_size": int(batch_size),
        "dtype": str(dtype),
        "device": str(device),
        "phase": phase,
        "generation": int(generation),
    }


__all__ = [
    "QWEN3_HANDOFF_SCHEMA_VERSION",
    "QWEN3_KV_CONTRACT_VERSION",
    "Qwen3PipelineContractError",
    "build_kv_contract",
    "validate_segment_plan",
]
