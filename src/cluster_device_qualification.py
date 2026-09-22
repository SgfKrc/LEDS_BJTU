"""Platform-neutral device composition and voter qualification for P4.5.

This module produces an observation snapshot.  It does not elect a leader,
change a node role, or add a node to the quorum voter set.  Platform names are
reported for composition analysis only; eligibility is based on an explicit
policy flag, a control capability, and fresh liveness evidence.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from koakuma_engine import Capability, backend_capabilities


DEVICE_QUALIFICATION_SCHEMA_VERSION = "qlh.cluster.device-qualification.v1"
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 30.0
_CONTROL_CAPABILITIES = frozenset({"cluster_control", "control_plane", "ha_voter"})


class DeviceQualificationError(ValueError):
    """Raised when qualification policy input is malformed."""


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
        "node_type": getattr(node, "node_type", ""),
        "state": getattr(getattr(node, "state", ""), "value", getattr(node, "state", "")),
        "device_info": getattr(node, "device_info", {}),
        "last_heartbeat": getattr(node, "last_heartbeat", 0.0),
    }


def _device_info(node: Mapping[str, Any]) -> Mapping[str, Any]:
    value = node.get("device_info", {})
    return value if isinstance(value, Mapping) else {}


def _values(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value.strip().lower()}) if value.strip() else frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip().lower() for item in value if str(item).strip())
    return frozenset()


def _forward_layers_supported(info: Mapping[str, Any]) -> bool:
    capabilities = _values(info.get("capabilities"))
    if Capability.FORWARD_LAYERS in capabilities:
        return True
    backend = info.get("backend_id") or info.get("engine")
    if isinstance(backend, str) and backend.strip():
        try:
            return backend_capabilities(backend).supports(Capability.FORWARD_LAYERS)
        except KeyError:
            return False
    return bool(info.get("forward_layers") is True)


def _control_capability_present(info: Mapping[str, Any]) -> bool:
    direct = _values(info.get("control_capabilities"))
    declared = _values(info.get("capabilities"))
    return bool(_CONTROL_CAPABILITIES.intersection(direct | declared))


def _explicit_voter(info: Mapping[str, Any]) -> bool:
    value = info.get("cluster_voter")
    return value is True


def _heartbeat_age_ms(last_heartbeat: Any, now: float) -> int | None:
    try:
        timestamp = float(last_heartbeat)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    return max(0, int((now - timestamp) * 1000))


@dataclass(frozen=True)
class DeviceQualification:
    node_id: str
    platform: str
    state: str
    heartbeat_age_ms: int | None
    heartbeat_fresh: bool
    explicit_voter: bool
    control_capability: bool
    forward_layers: bool
    eligible_voter: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "platform": self.platform,
            "state": self.state,
            "heartbeat_age_ms": self.heartbeat_age_ms,
            "heartbeat_fresh": self.heartbeat_fresh,
            "explicit_voter": self.explicit_voter,
            "control_capability": self.control_capability,
            "forward_layers": self.forward_layers,
            "eligible_voter": self.eligible_voter,
            "reasons": list(self.reasons),
        }


def qualify_device(
    node: Any,
    *,
    now: float | None = None,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
) -> DeviceQualification:
    """Return a fail-closed, platform-neutral qualification snapshot."""

    if heartbeat_timeout_seconds <= 0:
        raise DeviceQualificationError("heartbeat timeout must be positive")
    data = _node_mapping(node)
    info = _device_info(data)
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        raise DeviceQualificationError("node_id is required")
    platform = str(data.get("node_type") or info.get("platform") or "unknown").strip().lower()
    state = str(data.get("state") or "unknown").strip().lower()
    current = time.time() if now is None else float(now)
    age_ms = _heartbeat_age_ms(data.get("last_heartbeat"), current)
    heartbeat_fresh = age_ms is not None and age_ms <= int(heartbeat_timeout_seconds * 1000)
    explicit_voter = _explicit_voter(info)
    control_capability = _control_capability_present(info)
    forward_layers = _forward_layers_supported(info)

    reasons: list[str] = []
    if not explicit_voter:
        reasons.append("voter_not_explicit")
    if not control_capability:
        reasons.append("control_capability_missing")
    if age_ms is None:
        reasons.append("heartbeat_missing")
    elif not heartbeat_fresh:
        reasons.append("heartbeat_expired")
    if state not in {"online", "busy"}:
        reasons.append(f"state_{state or 'unknown'}")
    return DeviceQualification(
        node_id=node_id,
        platform=platform or "unknown",
        state=state or "unknown",
        heartbeat_age_ms=age_ms,
        heartbeat_fresh=heartbeat_fresh,
        explicit_voter=explicit_voter,
        control_capability=control_capability,
        forward_layers=forward_layers,
        eligible_voter=not reasons,
        reasons=tuple(reasons),
    )


def summarize_device_composition(
    nodes: Sequence[Any],
    *,
    now: float | None = None,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Build a deterministic composition snapshot without changing cluster state."""

    qualifications = tuple(
        qualify_device(
            node,
            now=now,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )
        for node in nodes
    )
    node_ids = [item.node_id for item in qualifications]
    duplicate_ids = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    platform_counts = Counter(item.platform for item in qualifications)
    return {
        "schema_version": DEVICE_QUALIFICATION_SCHEMA_VERSION,
        "node_count": len(qualifications),
        "platform_counts": dict(sorted(platform_counts.items())),
        "platform_affects_eligibility": False,
        "eligible_voter_count": sum(item.eligible_voter for item in qualifications),
        "eligible_voter_ids": [item.node_id for item in qualifications if item.eligible_voter],
        "forward_layer_node_ids": [item.node_id for item in qualifications if item.forward_layers],
        "sleeping_or_unavailable_ids": [
            item.node_id for item in qualifications if item.state not in {"online", "busy"}
        ],
        "duplicate_node_ids": duplicate_ids,
        "nodes": [item.to_dict() for item in qualifications],
    }


__all__ = [
    "DEFAULT_HEARTBEAT_TIMEOUT_SECONDS",
    "DEVICE_QUALIFICATION_SCHEMA_VERSION",
    "DeviceQualification",
    "DeviceQualificationError",
    "qualify_device",
    "summarize_device_composition",
]
