"""User-owned SQLite projection for the explicit local Qwen3 chain."""

from __future__ import annotations

from typing import Any


STATE_KEY = "qwen3_local_chain_v1"
SCHEMA_VERSION = 1
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


__all__ = [
    "STATE_KEY",
    "load_qwen3_local_chain_state",
    "save_qwen3_local_chain_state",
]
