"""Qwen3 pipeline segment and handoff contracts.

This module deliberately stays free of torch/transformers imports.  It is
used by the isolated sidecar to validate a segment topology and by tests to
exercise multi-segment execution with small in-memory adapters.  Transport
layers may serialize the returned handoff/KV metadata without exposing model
weights or prompt content.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

try:
    from .qwen3_pipeline_adapter import Qwen3AdapterError
except ImportError:  # direct sidecar script execution
    from qwen3_pipeline_adapter import Qwen3AdapterError  # type: ignore


QWEN3_CHAIN_SCHEMA_VERSION = 1
QWEN3_HANDOFF_SCHEMA_VERSION = 1
QWEN3_KV_CONTRACT_VERSION = 1


def _shape(value: Any) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise Qwen3AdapterError("Qwen3 hidden handoff has no tensor shape")
    try:
        result = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise Qwen3AdapterError("Qwen3 hidden handoff shape is invalid") from exc
    if len(result) != 3 or any(item <= 0 for item in result):
        raise Qwen3AdapterError("Qwen3 hidden handoff must be [batch, sequence, hidden]")
    return result


def _dtype(value: Any) -> str:
    result = str(getattr(value, "dtype", ""))
    if not result or result == "None":
        raise Qwen3AdapterError("Qwen3 hidden handoff dtype is missing")
    return result


def _device(value: Any) -> str:
    result = str(getattr(value, "device", ""))
    if not result or result == "None":
        raise Qwen3AdapterError("Qwen3 hidden handoff device is missing")
    return result


def validate_segment_plan(
    segments: Iterable[dict[str, Any]],
    *,
    total_layers: int,
) -> list[dict[str, Any]]:
    """Validate a contiguous 2/3 segment topology before loading weights."""
    values = list(segments)
    if len(values) not in {2, 3}:
        raise Qwen3AdapterError("Qwen3 chain requires exactly two or three segments")
    total_layers = int(total_layers)
    if total_layers <= 0:
        raise Qwen3AdapterError("Qwen3 chain total layer count is invalid")
    normalized: list[dict[str, Any]] = []
    expected_start = 0
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise Qwen3AdapterError("Qwen3 chain segment must be an object")
        layer_range = raw.get("layer_range")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            raise Qwen3AdapterError("Qwen3 chain segment layer_range is invalid")
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
        except (TypeError, ValueError) as exc:
            raise Qwen3AdapterError("Qwen3 chain segment layer_range is invalid") from exc
        if start != expected_start or start < 0 or end <= start or end > total_layers:
            raise Qwen3AdapterError("Qwen3 chain segments must be contiguous and in bounds")
        has_embedding = bool(raw.get("has_embedding", False))
        has_lm_head = bool(raw.get("has_lm_head", False))
        if has_embedding != (index == 0):
            raise Qwen3AdapterError("only the first Qwen3 segment may own embedding")
        if has_lm_head != (index == len(values) - 1):
            raise Qwen3AdapterError("only the last Qwen3 segment may own LM Head")
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
        raise Qwen3AdapterError("Qwen3 chain does not cover all model layers")
    return normalized


def build_hidden_handoff(
    hidden_states: Any,
    *,
    chain_id: str,
    from_segment: int,
    to_segment: int,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Return metadata for a hidden-state boundary without serializing data."""
    shape = _shape(hidden_states)
    sequence = int(sequence_length if sequence_length is not None else shape[1])
    if sequence != shape[1]:
        raise Qwen3AdapterError("hidden handoff sequence length does not match tensor")
    if not str(chain_id).strip() or int(to_segment) != int(from_segment) + 1:
        raise Qwen3AdapterError("hidden handoff segment transition is invalid")
    return {
        "schema_version": QWEN3_HANDOFF_SCHEMA_VERSION,
        "chain_id": str(chain_id),
        "from_segment": int(from_segment),
        "to_segment": int(to_segment),
        "shape": list(shape),
        "batch_size": shape[0],
        "sequence_length": sequence,
        "hidden_size": shape[2],
        "dtype": _dtype(hidden_states),
        "device": _device(hidden_states),
    }


def validate_hidden_handoff(
    hidden_states: Any,
    handoff: dict[str, Any],
    *,
    chain_id: str,
    expected_from: int,
    expected_to: int,
) -> None:
    """Fail closed when a remote/local hidden boundary changes shape or type."""
    if not isinstance(handoff, dict) or handoff.get("schema_version") != QWEN3_HANDOFF_SCHEMA_VERSION:
        raise Qwen3AdapterError("hidden handoff schema is unsupported")
    expected = build_hidden_handoff(
        hidden_states,
        chain_id=chain_id,
        from_segment=expected_from,
        to_segment=expected_to,
    )
    for key in ("chain_id", "from_segment", "to_segment", "shape", "dtype", "device", "sequence_length"):
        if handoff.get(key) != expected.get(key):
            raise Qwen3AdapterError(f"hidden handoff {key} does not match")


def build_kv_contract(
    *,
    chain_id: str,
    segment_index: int,
    layer_range: Sequence[int],
    sequence_length: int,
    batch_size: int,
    dtype: str,
    device: str,
    phase: str,
    generation: int,
) -> dict[str, Any]:
    """Build the metadata contract for a segment-owned KV cache."""
    if phase not in {"prefill", "decode"}:
        raise Qwen3AdapterError("KV contract phase is invalid")
    if len(layer_range) != 2 or int(layer_range[1]) <= int(layer_range[0]):
        raise Qwen3AdapterError("KV contract layer range is invalid")
    if int(sequence_length) <= 0 or int(batch_size) <= 0 or int(generation) < 0:
        raise Qwen3AdapterError("KV contract dimensions are invalid")
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


def validate_kv_contract(
    contract: dict[str, Any],
    *,
    chain_id: str,
    segment_index: int,
    layer_range: Sequence[int],
    sequence_length: int,
    batch_size: int,
    dtype: str,
    device: str,
    phase: str,
    generation: int,
) -> None:
    expected = build_kv_contract(
        chain_id=chain_id,
        segment_index=segment_index,
        layer_range=layer_range,
        sequence_length=sequence_length,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        phase=phase,
        generation=generation,
    )
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected.items()):
        raise Qwen3AdapterError("Qwen3 KV cache contract mismatch")


def _sequence_length(cache: Any, expected: int) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError, IndexError):
            pass
    candidates: list[int] = []

    def visit(value: Any) -> None:
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                candidates.extend(int(item) for item in shape)
            except (TypeError, ValueError):
                return
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(cache)
    if expected in candidates:
        return int(expected)
    raise Qwen3AdapterError("Qwen3 KV cache sequence length is unavailable")


def execute_segment_chain(
    adapters: Sequence[Any],
    *,
    input_ids: Any,
    segments: Sequence[dict[str, Any]],
    chain_id: str = "local-smoke",
    decode_input_ids: Any | None = None,
) -> dict[str, Any]:
    """Run prefill and optional one-token decode through loaded adapters.

    Each segment owns its own KV cache.  Only hidden states cross a segment
    boundary; passing a cache from another segment is intentionally rejected
    by the contract shape, layer range and segment index checks.
    """
    if len(adapters) != len(segments):
        raise Qwen3AdapterError("Qwen3 chain adapter/segment count mismatch")
    if not adapters:
        raise Qwen3AdapterError("Qwen3 chain has no adapters")
    hidden = None
    prefill_kv: list[Any] = []
    handoffs: list[dict[str, Any]] = []
    prefill_contracts: list[dict[str, Any]] = []
    for index, (adapter, segment) in enumerate(zip(adapters, segments)):
        if index == 0:
            result = adapter.forward(input_ids=input_ids, use_cache=True)
        else:
            if hidden is None:
                raise Qwen3AdapterError("Qwen3 hidden handoff is missing")
            target_device = getattr(adapter, "_device_dtype", lambda _model: (None, None))(getattr(adapter, "model", None))[0]
            if target_device is not None and getattr(hidden, "device", None) != target_device:
                raise Qwen3AdapterError("Qwen3 hidden handoff device does not match next segment")
            result = adapter.forward(hidden_states=hidden, use_cache=True)
        hidden = result.get("hidden_states")
        if index < len(adapters) - 1:
            if hidden is None:
                raise Qwen3AdapterError("non-final Qwen3 segment did not return hidden states")
            handoff = build_hidden_handoff(
                hidden, chain_id=chain_id, from_segment=index, to_segment=index + 1,
            )
            handoffs.append(handoff)
        cache = result.get("past_key_values")
        if cache is None:
            raise Qwen3AdapterError("Qwen3 segment did not return KV cache")
        output_tensor = hidden if hidden is not None else result.get("logits")
        prefill_kv.append(cache)
        prefill_contracts.append(build_kv_contract(
            chain_id=chain_id,
            segment_index=index,
            layer_range=segment["layer_range"],
            sequence_length=int(input_ids.shape[1]),
            batch_size=int(input_ids.shape[0]),
            dtype=str(getattr(output_tensor, "dtype", "unknown")),
            device=str(getattr(output_tensor, "device", "unknown")),
            phase="prefill",
            generation=0,
        ))
    final = result
    decode = None
    decode_contracts: list[dict[str, Any]] = []
    if decode_input_ids is not None:
        hidden = None
        for index, (adapter, segment, cache) in enumerate(zip(adapters, segments, prefill_kv)):
            if index == 0:
                decode = adapter.forward(
                    input_ids=decode_input_ids, past_key_values=cache, use_cache=True,
                )
            else:
                if hidden is None:
                    raise Qwen3AdapterError("Qwen3 decode hidden handoff is missing")
                target_device = getattr(adapter, "_device_dtype", lambda _model: (None, None))(getattr(adapter, "model", None))[0]
                if target_device is not None and getattr(hidden, "device", None) != target_device:
                    raise Qwen3AdapterError("Qwen3 decode handoff device does not match next segment")
                decode = adapter.forward(
                    hidden_states=hidden, past_key_values=cache, use_cache=True,
                )
            hidden = decode.get("hidden_states")
            next_cache = decode.get("past_key_values")
            if next_cache is None:
                raise Qwen3AdapterError("Qwen3 decode segment did not return KV cache")
            output_tensor = hidden if hidden is not None else decode.get("logits")
            decode_contracts.append(build_kv_contract(
                chain_id=chain_id,
                segment_index=index,
                layer_range=segment["layer_range"],
                sequence_length=int(input_ids.shape[1]) + int(decode_input_ids.shape[1]),
                batch_size=int(decode_input_ids.shape[0]),
                dtype=str(getattr(output_tensor, "dtype", "unknown")),
                device=str(getattr(output_tensor, "device", "unknown")),
                phase="decode",
                generation=1,
            ))
    return {
        "prefill": final,
        "decode": decode,
        "hidden_handoffs": handoffs,
        "kv_contracts": {"prefill": prefill_contracts, "decode": decode_contracts},
        "full_model_materialized": False,
    }


__all__ = [
    "QWEN3_CHAIN_SCHEMA_VERSION",
    "QWEN3_HANDOFF_SCHEMA_VERSION",
    "QWEN3_KV_CONTRACT_VERSION",
    "build_hidden_handoff",
    "build_kv_contract",
    "execute_segment_chain",
    "validate_hidden_handoff",
    "validate_kv_contract",
    "validate_segment_plan",
]
