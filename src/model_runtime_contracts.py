"""Path-free, canonical contracts for the experimental model runtimes.

The contract is deliberately narrower than a model artifact manifest.  It
binds an already verified artifact identity to a capacity plan and a bounded
set of layer assignments; paths and weight contents stay on the owning node.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from gemma4_pipeline_contract import (
    Gemma4PipelineContractError,
    build_gemma4_pipeline_contract,
    validate_gemma4_pipeline_contract,
)
from qwen3_pipeline_transaction import (
    Qwen3PipelineProtocolError,
    build_qwen3_dry_run_contract,
    validate_qwen3_dry_run_contract,
)


CONTRACT_STORE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_PROFILES = {"qwen3_sidecar", "gemma4_pipeline"}


class ModelRuntimeContractError(ValueError):
    """A model runtime contract cannot be bound safely."""


def _canonical(value: Any, *, label: str = "contract") -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeContractError(f"{label} is not JSON serializable") from exc


def _digest(value: Any, *, label: str = "contract") -> str:
    return hashlib.sha256(_canonical(value, label=label)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise ModelRuntimeContractError(f"{label} must be a lowercase SHA-256")
    return normalized


def assignment_manifest_sha256(
    assignment: dict[str, Any], *, model_sha256: str, plan_id: str,
) -> str:
    """Derive a stable assignment identity without retaining a model path."""
    safe = {
        "model_sha256": _sha256(model_sha256, "model_sha256"),
        "plan_id": str(plan_id or ""),
        "node_id": str(assignment.get("node_id", "") or ""),
        "start_layer": int(assignment.get("start_layer", 0) or 0),
        "end_layer": int(assignment.get("end_layer", 0) or 0),
        "has_embedding": bool(assignment.get("has_embedding", False)),
        "has_lm_head": bool(assignment.get("has_lm_head", False)),
        "required_bytes": int(assignment.get("required_bytes", 0) or 0),
        "capacity_bytes": int(assignment.get("capacity_bytes", 0) or 0),
        "execution_device": str(assignment.get("execution_device", "cpu") or "cpu"),
    }
    if not safe["plan_id"] or not safe["node_id"]:
        raise ModelRuntimeContractError("assignment identity is incomplete")
    return _digest(safe, label="assignment manifest")


def _normalized_dimensions(descriptor: dict[str, Any]) -> tuple[int, int]:
    config = descriptor.get("config") if isinstance(descriptor.get("config"), dict) else {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config
    try:
        total_layers = int(descriptor.get("total_layers", config.get("num_hidden_layers", 0)) or 0)
        hidden_size = int(descriptor.get("hidden_size", config.get("hidden_size", 0)) or 0)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeContractError("model dimensions are invalid") from exc
    if total_layers <= 0 or hidden_size <= 0:
        raise ModelRuntimeContractError("model dimensions are unavailable")
    return total_layers, hidden_size


def _segment_inputs(
    assignments: Iterable[dict[str, Any]], *,
    model_sha256: str, plan_id: str, generation: int,
) -> list[dict[str, Any]]:
    values = list(assignments)
    if len(values) not in {2, 3}:
        raise ModelRuntimeContractError(
            "the current Sidecar contract supports exactly two or three segments"
        )
    normalized = []
    for assignment in values:
        if not isinstance(assignment, dict):
            raise ModelRuntimeContractError("capacity assignment must be an object")
        start = int(assignment.get("start_layer", 0) or 0)
        end = int(assignment.get("end_layer", 0) or 0)
        required = int(assignment.get("required_bytes", 0) or 0)
        if start < 0 or end <= start or required <= 0:
            raise ModelRuntimeContractError("capacity assignment dimensions are invalid")
        device = str(assignment.get("execution_device", "cpu") or "cpu")
        dtype = "float16" if device == "cuda" else "float32"
        normalized.append({
            "node_id": str(assignment.get("node_id", "") or ""),
            "layer_range": [start, end],
            "has_embedding": bool(assignment.get("has_embedding", False)),
            "has_lm_head": bool(assignment.get("has_lm_head", False)),
            "assignment_manifest_sha256": assignment_manifest_sha256(
                assignment, model_sha256=model_sha256, plan_id=plan_id,
            ),
            "required_bytes": required,
            "execution_device": device,
            "dtype": dtype,
        })
    return normalized


def build_model_runtime_contract(
    profile: str,
    *,
    config_id: str,
    plan_id: str,
    generation: int,
    model_id: str,
    model_sha256: str,
    descriptor: dict[str, Any],
    assignments: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build a canonical Qwen3 or Gemma4 contract from a capacity plan."""
    profile = str(profile or "")
    if profile not in _SUPPORTED_PROFILES:
        raise ModelRuntimeContractError("model runtime profile is unsupported")
    model_sha256 = _sha256(model_sha256, "model_sha256")
    total_layers, hidden_size = _normalized_dimensions(descriptor)
    segments = _segment_inputs(
        assignments, model_sha256=model_sha256, plan_id=plan_id,
        generation=generation,
    )
    if profile == "qwen3_sidecar":
        return build_qwen3_dry_run_contract(
            config_id=config_id,
            plan_id=plan_id,
            generation=generation,
            model_id=model_id,
            model_sha256=model_sha256,
            total_layers=total_layers,
            hidden_size=hidden_size,
            segments=segments,
            execution_mode="node_local_sidecar",
        )

    config = descriptor.get("config") if isinstance(descriptor.get("config"), dict) else {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        config = text_config
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list):
        raise ModelRuntimeContractError("Gemma 4 layer_types are unavailable")
    try:
        shared_layers = int(config.get("num_kv_shared_layers", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ModelRuntimeContractError("Gemma 4 shared-KV dimensions are invalid") from exc
    return build_gemma4_pipeline_contract(
        config_id=config_id,
        plan_id=plan_id,
        generation=generation,
        model_id=model_id,
        model_sha256=model_sha256,
        total_layers=total_layers,
        hidden_size=hidden_size,
        layer_types=layer_types,
        num_kv_shared_layers=shared_layers,
        segments=segments,
    )


def validate_model_runtime_contract(profile: str, contract: dict[str, Any]) -> dict[str, Any]:
    profile = str(profile or "")
    if profile == "qwen3_sidecar":
        return validate_qwen3_dry_run_contract(contract)
    if profile == "gemma4_pipeline":
        return validate_gemma4_pipeline_contract(contract)
    raise ModelRuntimeContractError("model runtime profile is unsupported")


def contract_summary(profile: str, contract: dict[str, Any], *, status: str = "bound") -> dict[str, Any]:
    """Return a UI-safe projection; the full path-free contract stays server-owned."""
    validated = validate_model_runtime_contract(profile, contract)
    return {
        "contract_id": validated["contract_sha256"],
        "profile": profile,
        "status": status,
        "schema_version": CONTRACT_STORE_SCHEMA_VERSION,
        "contract_sha256": validated["contract_sha256"],
        "config_id": validated.get("config_id", ""),
        "plan_id": validated.get("plan_id", ""),
        "generation": int(validated.get("generation", 0) or 0),
        "model_id": validated.get("model_id", ""),
        "model_sha256": validated.get("model_sha256", ""),
        "segment_count": len(validated.get("segments", [])),
        "total_layers": int(validated.get("total_layers", 0) or 0),
        "production_admitted": False,
        "execution_mode": validated.get("execution_mode", "node_local_sidecar"),
    }


__all__ = [
    "CONTRACT_STORE_SCHEMA_VERSION",
    "ModelRuntimeContractError",
    "assignment_manifest_sha256",
    "build_model_runtime_contract",
    "contract_summary",
    "validate_model_runtime_contract",
]
