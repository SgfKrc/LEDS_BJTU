"""Conservative cut-layer planning for llama.cpp -> llama.cpp relay pipelines.

Relay is an architecture-compatibility track that is **off by default** and must never be
admitted on a guess, so this planner mirrors the RPC planner's stance: it only proposes a
cut when the request is explicit and valid, or when the downstream side has usable
telemetry. Rejections always name the rule, and the caller falls back to the
single-process llama.cpp path (see :mod:`src.relay_contract`).

The module is stdlib-only and imports nothing from the control plane.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.relay_contract import (
    CUT_LAYER_MIN,
    RELAY_ENGINE,
    RelayHiddenSpec,
)

#: Lowest number of layers the downstream side must keep for a handoff to make sense.
MIN_DOWNSTREAM_LAYERS = 1


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class RelayDownstreamProfile:
    """Minimal downstream capability view (shape-compatible with ``DeviceProfiler.to_dict()``)."""

    node_id: str
    execution_device: str = "CPU"
    profile_available: bool = False
    ram_available_gb: float = 0.0
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0

    @classmethod
    def from_device_info(
        cls,
        info: Mapping[str, Any] | None,
        *,
        node_id: str = "relay-downstream",
        execution_device: str = "CPU",
        rtt_ms: float = 0.0,
        bandwidth_mbps: float = 0.0,
    ) -> "RelayDownstreamProfile":
        data = info if isinstance(info, Mapping) else {}
        ram = _mapping(data.get("ram"))
        cpu = _mapping(data.get("cpu"))
        available = _number(ram.get("available_gb", data.get("ram_available_gb")), 0.0)
        profile_available = bool(
            any(key in ram for key in ("available_gb", "total_gb"))
            or any(key in data for key in ("ram_available_gb", "ram_total_gb"))
            or any(key in cpu for key in ("physical_cores", "usage_percent"))
        )
        return cls(
            node_id=str(node_id),
            execution_device=str(execution_device or "CPU").upper(),
            profile_available=profile_available,
            # Total RAM is not usable headroom. Without an available-memory
            # measurement, refuse automatic admission rather than over-admit.
            ram_available_gb=round(max(0.0, available), 2),
            rtt_ms=round(max(0.0, _number(rtt_ms)), 2),
            bandwidth_mbps=round(max(0.0, _number(bandwidth_mbps)), 2),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayCutDecision:
    """Result of planning the cut point of an L -> L relay."""

    admitted: bool
    reason: str
    total_layers: int
    cut_layer: int
    upstream_layers: int
    downstream_layers: int
    hidden_bytes_per_token: int
    profile: RelayDownstreamProfile

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile"] = self.profile.to_dict()
        return result


def plan_relay_cut(
    total_layers: int,
    *,
    n_embd: int,
    requested_cut: int | None = None,
    downstream_profile: Mapping[str, Any] | RelayDownstreamProfile | None = None,
    hidden_dtype: str = "float32",
    device: str = RELAY_ENGINE,
) -> RelayCutDecision:
    """Return a conservative cut layer for an L -> L relay, or a named rejection.

    ``cut_layer = K`` means the upstream side computes layers ``0..K-1`` and the hidden of
    ``layer K-1`` enters the trimmed downstream artifact at its ``layer 0``. ``K`` is capped
    at ``n_layer - 1`` so the handed-off layer stays ``<= n_layer - 2`` (the engine-specific
    last layer never crosses the boundary).
    """
    try:
        total = max(0, int(total_layers))
    except (TypeError, ValueError):
        total = 0
    try:
        embedding_width = int(n_embd)
    except (TypeError, ValueError):
        embedding_width = 0
    hidden = RelayHiddenSpec(n_embd=embedding_width, dtype=hidden_dtype)
    if isinstance(downstream_profile, RelayDownstreamProfile):
        profile = downstream_profile
    else:
        profile = RelayDownstreamProfile.from_device_info(downstream_profile)

    def reject(reason: str, cut: int = 0) -> RelayCutDecision:
        return RelayCutDecision(
            admitted=False,
            reason=reason,
            total_layers=total,
            cut_layer=max(0, cut),
            upstream_layers=max(0, cut),
            downstream_layers=max(0, total - cut),
            hidden_bytes_per_token=hidden.bytes_per_token,
            profile=profile,
        )

    if total < 2 + MIN_DOWNSTREAM_LAYERS - 1:
        return reject("model_has_no_relay_range")
    if not hidden.supported:
        return reject("unsupported_hidden_format")

    max_cut = total - 1  # keeps upstream_last_layer <= total - 2
    if max_cut < CUT_LAYER_MIN:
        return reject("model_has_no_relay_range")

    if requested_cut is not None:
        try:
            cut = int(requested_cut)
        except (TypeError, ValueError):
            return reject("cut_layer_invalid")
        if cut < CUT_LAYER_MIN or cut > max_cut:
            return reject("cut_layer_out_of_range", cut)
        return _admit(cut, total, hidden.bytes_per_token, profile)

    if not profile.profile_available:
        return reject("downstream_profile_missing")
    if profile.ram_available_gb <= 0:
        return reject("downstream_capacity_insufficient")

    # Conservative auto split: keep one layer downstream, never more than the memory the
    # profile reports as available (hidden bytes per token are tiny, layers are not).
    estimated_layer_mib = 32.0
    usable_mib = max(0.0, profile.ram_available_gb * 1024.0 * 0.5)
    downstream_cap = int(usable_mib / estimated_layer_mib)
    downstream_layers = max(MIN_DOWNSTREAM_LAYERS, min(downstream_cap, total - 1))
    cut = total - downstream_layers
    cut = max(CUT_LAYER_MIN, min(cut, max_cut))
    if cut >= total:
        return reject("downstream_capacity_insufficient")
    return _admit(cut, total, hidden.bytes_per_token, profile, reason="auto_capacity_split")


def _admit(
    cut: int,
    total: int,
    hidden_bytes_per_token: int,
    profile: RelayDownstreamProfile,
    *,
    reason: str = "requested_cut_admitted",
) -> RelayCutDecision:
    return RelayCutDecision(
        admitted=True,
        reason=reason,
        total_layers=total,
        cut_layer=cut,
        upstream_layers=cut,
        downstream_layers=max(0, total - cut),
        hidden_bytes_per_token=hidden_bytes_per_token,
        profile=profile,
    )
