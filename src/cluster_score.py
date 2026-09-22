"""Read-only, versioned management-capability scoring for P4.5.

The score is an explainable observation of management capacity.  It is not an
inference-placement weight, a quorum vote, or a source of write authority.
All calculations are pure for a supplied observation time so snapshots can be
replayed in tests and audit tooling.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cluster_device_qualification import qualify_device


MANAGEMENT_SCORE_SCHEMA_VERSION = "qlh.ha.management-score.v1"
MANAGEMENT_SCORE_ALGORITHM_VERSION = "management-v1"
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 30.0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return min(upper, max(lower, value))


def _node_mapping(node: Any) -> Mapping[str, Any]:
    if isinstance(node, Mapping):
        return node
    to_dict = getattr(node, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    return {
        "node_id": getattr(node, "node_id", ""),
        "role": getattr(node, "role", ""),
        "node_type": getattr(node, "node_type", ""),
        "state": getattr(getattr(node, "state", ""), "value", getattr(node, "state", "")),
        "device_info": getattr(node, "device_info", {}),
        "last_heartbeat": getattr(node, "last_heartbeat", 0.0),
        "avg_rtt_ms": getattr(node, "avg_rtt_ms", 0.0),
        "error_count": getattr(node, "error_count", 0),
    }


def _device_info(node: Mapping[str, Any]) -> Mapping[str, Any]:
    value = node.get("device_info", {})
    return value if isinstance(value, Mapping) else {}


def _score_node(
    node: Any,
    *,
    now: float,
    heartbeat_timeout_seconds: float,
) -> dict[str, Any]:
    data = _node_mapping(node)
    node_id = str(data.get("node_id", "")).strip() or "unknown"
    info = _device_info(data)
    cpu = info.get("cpu", {}) if isinstance(info.get("cpu"), Mapping) else {}
    ram = info.get("ram", {}) if isinstance(info.get("ram"), Mapping) else {}
    state = str(data.get("state", "unknown") or "unknown").lower()

    physical_cores = _as_float(cpu.get("physical_cores"))
    max_frequency = _as_float(cpu.get("freq_max_mhz"))
    capacity = _clamp(
        ((_clamp(physical_cores / 16.0, 0.0, 1.0) * 50.0)
         + (_clamp(max_frequency / 4000.0, 0.0, 1.0) * 50.0))
    )

    total_memory = _as_float(ram.get("total_gb"))
    available_memory = _as_float(ram.get("available_gb"))
    memory = _clamp(available_memory / total_memory * 100.0) if total_memory > 0 else 0.0

    usage = _as_float(cpu.get("usage_percent"), default=-1.0)
    load = _clamp(100.0 - usage) if 0.0 <= usage <= 100.0 else 0.0

    heartbeat = _as_float(data.get("last_heartbeat"), default=0.0)
    heartbeat_age_ms = None
    if heartbeat > 0:
        heartbeat_age_ms = max(0, int((now - heartbeat) * 1000))
    if heartbeat_age_ms is None:
        stability = 0.0
    else:
        stability = _clamp(
            100.0 - (heartbeat_age_ms / (heartbeat_timeout_seconds * 1000.0) * 100.0)
        )
        if state not in {"online", "busy"}:
            stability = 0.0

    rtt = _as_float(data.get("avg_rtt_ms"), default=-1.0)
    if rtt < 0:
        network = 0.0
    elif rtt == 0:
        network = 100.0
    else:
        network = _clamp(100.0 - (rtt / 250.0 * 100.0))

    qualification = qualify_device(
        data,
        now=now,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )
    minimum_qualified = (
        qualification.heartbeat_fresh
        and qualification.control_capability
        and qualification.state in {"online", "busy"}
    )
    reasons = list(qualification.reasons)
    if not cpu:
        reasons.append("cpu_profile_missing")
    if not ram:
        reasons.append("ram_profile_missing")
    if usage < 0:
        reasons.append("cpu_load_missing")
    if rtt < 0:
        reasons.append("network_rtt_missing")

    breakdown = {
        "capacity": round(capacity, 3),
        "memory_headroom": round(memory, 3),
        "load_headroom": round(load, 3),
        "stability": round(stability, 3),
        "network": round(network, 3),
    }
    score = (
        capacity * 0.35
        + memory * 0.20
        + load * 0.20
        + stability * 0.15
        + network * 0.10
    )
    return {
        "node_id": node_id,
        "role": str(data.get("role", "") or ""),
        "state": state,
        "score": round(score, 3),
        "score_breakdown": breakdown,
        "minimum_qualified": minimum_qualified,
        "eligible_voter": qualification.eligible_voter,
        "heartbeat_age_ms": heartbeat_age_ms,
        "error_count": max(0, int(_as_float(data.get("error_count")))),
        "reasons": sorted(set(reasons)),
    }


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_management_score_snapshot(
    nodes: Sequence[Any],
    *,
    now: float | None = None,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    signing_secret: str | bytes = "",
) -> dict[str, Any]:
    """Build a deterministic, explainable score snapshot.

    ``signing_secret`` produces an HMAC integrity signature only.  It never
    grants quorum authority; control writes still require a control certificate.
    """

    if heartbeat_timeout_seconds <= 0:
        raise ValueError("heartbeat timeout must be positive")
    observed_at = time.time() if now is None else float(now)
    if observed_at < 0:
        raise ValueError("now must be non-negative")
    scored = [
        _score_node(
            node,
            now=observed_at,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )
        for node in nodes
    ]
    scored.sort(key=lambda item: (-item["score"], item["node_id"]))
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    observed_at_ms = int(observed_at * 1000)
    body: dict[str, Any] = {
        "schema_version": MANAGEMENT_SCORE_SCHEMA_VERSION,
        "algorithm_version": MANAGEMENT_SCORE_ALGORITHM_VERSION,
        "observed_at_ms": observed_at_ms,
        "observed_at": datetime.fromtimestamp(
            observed_at, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "tie_break": "node_id_ascending",
        "nodes": scored,
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    secret = signing_secret.encode("utf-8") if isinstance(signing_secret, str) else bytes(signing_secret)
    signature = hmac.new(secret, _canonical(body), hashlib.sha256).hexdigest() if secret else ""
    return {
        **body,
        "snapshot_digest": digest,
        "signature_algorithm": "hmac-sha256" if signature else None,
        "signature": signature or None,
        "signature_status": "signed" if signature else "unsigned_no_cluster_secret",
    }


__all__ = [
    "DEFAULT_HEARTBEAT_TIMEOUT_SECONDS",
    "MANAGEMENT_SCORE_ALGORITHM_VERSION",
    "MANAGEMENT_SCORE_SCHEMA_VERSION",
    "build_management_score_snapshot",
]
