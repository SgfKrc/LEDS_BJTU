"""User-owned SQLite projection for the explicit local Qwen3 chain."""

from __future__ import annotations

from typing import Any


STATE_KEY = "qwen3_local_chain_v1"
NETWORK_LEDGER_KEY = "qwen3_network_ledger_v1"
SCHEMA_VERSION = 1
NETWORK_LEDGER_SCHEMA_VERSION = 1
MAX_NETWORK_LEDGER_RECORDS = 256
_ALLOWED_PHASES = {
    "idle", "starting", "prepared", "committed", "prefilled", "decoded",
    "parity_passed", "released", "aborted", "recovered_aborted",
}


def _sanitize_comparison(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    shape = value.get("shape")
    return {
        "passed": bool(value.get("passed", False)),
        "shape": [int(item) for item in shape] if isinstance(shape, list) else [],
        "max_abs_error": float(value.get("max_abs_error", 0.0) or 0.0),
        "max_relative_error": float(value.get("max_relative_error", 0.0) or 0.0),
        "rtol": float(value.get("rtol", 0.0) or 0.0),
        "atol": float(value.get("atol", 0.0) or 0.0),
    }


def _sanitize_execution(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        item = value.get(phase)
        if not isinstance(item, dict):
            continue
        result[phase] = {
            "phase": str(item.get("phase", phase)),
            "segment_count": int(item.get("segment_count", 0) or 0),
            "generation": int(item.get("generation", 0) or 0),
            "artifact_count": int(item.get("artifact_count", 0) or 0),
        }
    return result


def _sanitize_parity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    errors = value.get("errors")
    safe_errors = []
    if isinstance(errors, list):
        for item in errors[:16]:
            if isinstance(item, dict):
                safe_errors.append({
                    "code": str(item.get("code", ""))[:128],
                    "message": str(item.get("message", ""))[:2048],
                })
    return {
        "schema_version": int(value.get("schema_version", 1) or 1),
        "gate": str(value.get("gate", ""))[:128],
        "status": str(value.get("status", ""))[:64],
        "gate_passed": bool(value.get("gate_passed", False)),
        "full_model_fallback": bool(value.get("full_model_fallback", False)),
        "full_model_materialized": bool(value.get("full_model_materialized", False)),
        "prefill": _sanitize_comparison(value.get("prefill")),
        "decode": _sanitize_comparison(value.get("decode")),
        "execution": _sanitize_execution(value.get("execution")),
        "errors": safe_errors,
    }


def _store():
    import local_store

    return local_store


def _sanitize(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Qwen3 local chain state must be an object")
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": str(value.get("contract_sha256", "")),
        "config_id": str(value.get("config_id", "")),
        "plan_id": str(value.get("plan_id", "")),
        "generation": int(value.get("generation", 0) or 0),
        "phase": str(value.get("phase", "idle")),
        "segment_count": int(value.get("segment_count", 0) or 0),
        "cleanup_complete": bool(value.get("cleanup_complete", False)),
        "parity": _sanitize_parity(value.get("parity")),
        "updated_at": str(value.get("updated_at", "")),
    }
    if result["phase"] not in _ALLOWED_PHASES:
        raise ValueError("Qwen3 local chain state phase is invalid")
    if result["generation"] < 0 or result["segment_count"] < 0:
        raise ValueError("Qwen3 local chain state dimensions are invalid")
    if result["contract_sha256"] and len(result["contract_sha256"]) != 64:
        raise ValueError("Qwen3 local chain state contract digest is invalid")
    return result


def load_qwen3_local_chain_state() -> dict[str, Any]:
    value = _store().get_local_setting(STATE_KEY, {})
    if not value:
        return {"schema_version": SCHEMA_VERSION, "phase": "idle", "cleanup_complete": True}
    try:
        return _sanitize(value)
    except (TypeError, ValueError):
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "recovered_aborted",
            "cleanup_complete": False,
            "error": "persisted Qwen3 local chain state is invalid",
        }


def save_qwen3_local_chain_state(value: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime, timezone

    sanitized = _sanitize({
        **value,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    _store().set_local_setting(STATE_KEY, sanitized)
    return sanitized


def _bounded_text(value: Any, maximum: int = 128) -> str:
    return str(value or "")[:maximum]


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _network_node_id(value: str) -> bool:
    return bool(value) and all(
        character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
        for character in value
    )


def _network_artifact_id(value: str, *, output_only: bool = False) -> bool:
    prefixes = ("qout_",) if output_only else ("qtx_", "qout_")
    return any(value.startswith(prefix) and _is_hex(value[len(prefix):], 32) for prefix in prefixes)


def _network_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    required = {
        "schema_version", "mode", "artifact_id", "source_node_id",
        "target_node_id", "chain_id", "generation", "phase",
        "from_segment", "to_segment", "size_bytes", "sha256", "status",
        "full_model_materialized",
    }
    if set(value) != required:
        raise ValueError("Qwen3 network ledger reference fields are invalid")
    result = {
        "schema_version": int(value.get("schema_version", 0) or 0),
        "mode": _bounded_text(value.get("mode"), 16),
        "artifact_id": _bounded_text(value.get("artifact_id"), 128),
        "source_node_id": _bounded_text(value.get("source_node_id"), 128),
        "target_node_id": _bounded_text(value.get("target_node_id"), 128),
        "chain_id": _bounded_text(value.get("chain_id"), 64),
        "generation": int(value.get("generation", 0) or 0),
        "phase": _bounded_text(value.get("phase"), 16),
        "from_segment": int(value.get("from_segment", -1)),
        "to_segment": int(value.get("to_segment", -1)),
        "size_bytes": int(value.get("size_bytes", 0) or 0),
        "sha256": _bounded_text(value.get("sha256"), 64),
        "status": _bounded_text(value.get("status"), 16),
        "full_model_materialized": bool(value.get("full_model_materialized", True)),
    }
    if (
        result["schema_version"] != 1
        or result["mode"] != "network"
        or not _network_artifact_id(result["artifact_id"])
        or not _network_node_id(result["source_node_id"])
        or not _network_node_id(result["target_node_id"])
        or not _is_hex(result["chain_id"], 64)
        or result["generation"] < 0
        or result["phase"] not in {"prefill", "decode"}
        or result["from_segment"] not in {0, 1}
        or result["to_segment"] != result["from_segment"] + 1
        or result["size_bytes"] <= 0
        or not _is_hex(result["sha256"], 64)
        or result["status"] != "committed"
        or result["full_model_materialized"] is not False
    ):
        raise ValueError("Qwen3 network ledger reference is invalid")
    return result


def _network_kv(value: Any, *, generation: int, phase: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}
    shape = value.get("shape", [])
    if (
        not isinstance(shape, list)
        or len(shape) > 16
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError("Qwen3 network ledger KV shape is invalid")
    return {
        "present": bool(value.get("present", False)),
        "shape": [int(item) for item in shape],
        "generation": int(generation),
        "phase": str(phase),
    }


def _network_transfer_record(key: Any, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Qwen3 network ledger transfer is invalid")
    transfer_id = _bounded_text(key, 64)
    generation = int(value.get("generation", 0) or 0)
    phase = _bounded_text(value.get("phase"), 16)
    status = _bounded_text(value.get("status"), 24)
    result = {
        "transfer_id": transfer_id,
        "source_node_id": _bounded_text(value.get("source_node_id"), 128),
        "target_node_id": _bounded_text(value.get("target_node_id"), 128),
        "chain_id": _bounded_text(value.get("chain_id"), 64),
        "generation": generation,
        "phase": phase,
        "from_segment": int(value.get("from_segment", -1)),
        "to_segment": int(value.get("to_segment", -1)),
        "size_bytes": int(value.get("size_bytes", 0) or 0),
        "sha256": _bounded_text(value.get("sha256"), 64),
        "status": status,
        "received_bytes": int(value.get("received_bytes", 0) or 0),
        "input_reference": _network_reference(value.get("input_reference")) if value.get("input_reference") else {},
        "output_reference_id": _bounded_text(value.get("output_reference_id"), 128),
        "kv_contract": _network_kv(value.get("kv_contract"), generation=generation, phase=phase),
        "updated_at": _bounded_text(value.get("updated_at"), 40),
    }
    if (
        not _network_artifact_id(transfer_id)
        or not transfer_id.startswith("qtx_")
        or not _network_node_id(result["source_node_id"])
        or not _network_node_id(result["target_node_id"])
        or not _is_hex(result["chain_id"], 64)
        or generation < 0
        or phase not in {"prefill", "decode"}
        or result["from_segment"] not in {0, 1}
        or result["to_segment"] != result["from_segment"] + 1
        or result["size_bytes"] <= 0
        or not _is_hex(result["sha256"], 64)
        or status not in {
            "receiving", "committed", "consuming", "consumed", "cancelled",
            "released", "invalidated",
        }
        or not 0 <= result["received_bytes"] <= result["size_bytes"]
    ):
        raise ValueError("Qwen3 network ledger transfer contract is invalid")
    return result


def _network_output_record(key: Any, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Qwen3 network ledger output is invalid")
    artifact_id = _bounded_text(key, 128)
    reference = _network_reference(value.get("reference"))
    status = _bounded_text(value.get("status"), 24)
    result = {
        "artifact_id": artifact_id,
        "reference": reference,
        "parent_transfer_id": _bounded_text(value.get("parent_transfer_id"), 64),
        "status": status,
        "lease_state": _bounded_text(value.get("lease_state"), 24),
        "next_transfer_id": _bounded_text(value.get("next_transfer_id"), 64),
        "confirmed_offset": int(value.get("confirmed_offset", 0) or 0),
        "updated_at": _bounded_text(value.get("updated_at"), 40),
    }
    if (
        artifact_id != reference.get("artifact_id")
        or not _network_artifact_id(artifact_id, output_only=True)
        or not result["parent_transfer_id"].startswith("qtx_")
        or status not in {"registered", "transferring", "committed", "released", "invalidated"}
        or result["lease_state"] not in {"available", "leased", "released", "invalidated"}
        or not 0 <= result["confirmed_offset"] <= reference["size_bytes"]
        or result["next_transfer_id"] and (
            not _network_artifact_id(result["next_transfer_id"])
            or not result["next_transfer_id"].startswith("qtx_")
        )
    ):
        raise ValueError("Qwen3 network ledger output contract is invalid")
    return result


def _sanitize_network_ledger(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Qwen3 network ledger must be an object")
    active = value.get("active_contract", {})
    if active and not isinstance(active, dict):
        raise ValueError("Qwen3 network ledger active contract is invalid")
    active_result = {}
    if active:
        active_result = {
            "contract_sha256": _bounded_text(active.get("contract_sha256"), 64),
            "generation": int(active.get("generation", 0) or 0),
            "phase": _bounded_text(active.get("phase"), 24),
            "segment_count": int(active.get("segment_count", 0) or 0),
            "restart_epoch": int(active.get("restart_epoch", 0) or 0),
        }
        if (
            not _is_hex(active_result["contract_sha256"], 64)
            or active_result["generation"] < 0
            or active_result["phase"] not in {
                "prepared", "prefill", "prefilled", "decode", "decoded",
                "released", "recovered",
            }
            or active_result["segment_count"] not in {2, 3}
            or active_result["restart_epoch"] < 0
        ):
            raise ValueError("Qwen3 network ledger active contract is invalid")
    transfers = value.get("transfers", {})
    outputs = value.get("outputs", {})
    if not isinstance(transfers, dict) or not isinstance(outputs, dict):
        raise ValueError("Qwen3 network ledger record sets are invalid")
    safe_transfers = {
        str(key): _network_transfer_record(key, item)
        for key, item in list(transfers.items())[-MAX_NETWORK_LEDGER_RECORDS:]
    }
    safe_outputs = {
        str(key): _network_output_record(key, item)
        for key, item in list(outputs.items())[-MAX_NETWORK_LEDGER_RECORDS:]
    }
    return {
        "schema_version": NETWORK_LEDGER_SCHEMA_VERSION,
        "local_node_id": _bounded_text(value.get("local_node_id"), 128),
        "last_generation": int(value.get("last_generation", -1)),
        "active_contract": active_result,
        "transfers": safe_transfers,
        "outputs": safe_outputs,
        "updated_at": _bounded_text(value.get("updated_at"), 40),
    }


def _network_ledger_key(local_node_id: str | None) -> str:
    node_id = str(local_node_id or "")
    if not node_id:
        return NETWORK_LEDGER_KEY
    if len(node_id) > 128 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
        for character in node_id
    ):
        raise ValueError("Qwen3 network ledger node identity is invalid")
    return f"{NETWORK_LEDGER_KEY}:{node_id}"


def load_qwen3_network_ledger(local_node_id: str | None = None) -> dict[str, Any]:
    value = _store().get_local_setting(_network_ledger_key(local_node_id), {})
    if not value:
        return {
            "schema_version": NETWORK_LEDGER_SCHEMA_VERSION,
            "local_node_id": "",
            "last_generation": -1,
            "active_contract": {},
            "transfers": {},
            "outputs": {},
            "updated_at": "",
        }
    return _sanitize_network_ledger(value)


def save_qwen3_network_ledger(
    value: dict[str, Any], local_node_id: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    sanitized = _sanitize_network_ledger({
        **value,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    node_id = str(local_node_id or sanitized.get("local_node_id", "") or "")
    _store().set_local_setting(_network_ledger_key(node_id), sanitized)
    return sanitized


__all__ = [
    "NETWORK_LEDGER_KEY",
    "STATE_KEY",
    "load_qwen3_local_chain_state",
    "load_qwen3_network_ledger",
    "save_qwen3_local_chain_state",
    "save_qwen3_network_ledger",
]
