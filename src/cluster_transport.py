"""Path selection and fencing primitives for the cluster transport layer.

This module deliberately contains no socket or WebSocket implementation.  It
freezes the control-plane contract needed before a second physical transport
is introduced, while keeping the existing TCP route as the default.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping


LEGACY_TCP = "legacy_tcp"
WSS_443 = "wss_443"
_TRANSPORTS = frozenset({LEGACY_TCP, WSS_443})


class TransportContractError(ValueError):
    """Raised when a transport contract or fencing transition is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TransportCandidate:
    """A sanitized observation of one possible physical path.

    ``endpoint`` is retained for local connection setup but is intentionally
    omitted from :meth:`public_view`; callers must not expose addresses in
    diagnostics unless they explicitly choose to do so.
    """

    transport: str
    endpoint: str
    tcp_available: bool
    tls_available: bool = False
    authenticated: bool = False
    expires_at: float | None = None
    priority: int = 0
    path_kind: str = ""

    def __post_init__(self) -> None:
        if self.transport not in _TRANSPORTS:
            raise TransportContractError("transport_unknown", "unsupported transport")
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise TransportContractError("endpoint_missing", "transport endpoint is required")
        if not isinstance(self.tcp_available, bool) or not isinstance(self.tls_available, bool):
            raise TransportContractError("probe_invalid", "probe availability must be boolean")
        if not isinstance(self.authenticated, bool):
            raise TransportContractError("probe_invalid", "authentication state must be boolean")
        if self.expires_at is not None and not isinstance(self.expires_at, (int, float)):
            raise TransportContractError("probe_invalid", "expiry must be numeric")

    @classmethod
    def from_probe(
        cls,
        transport: str,
        endpoint: str,
        report: Mapping[str, Any],
        *,
        priority: int = 0,
        authenticated: bool = False,
        expires_at: float | None = None,
    ) -> "TransportCandidate":
        """Build a candidate from ``cluster_node_preflight``-style data."""
        tcp = report.get("tcp")
        tcp_available = bool(
            tcp.get("available") if isinstance(tcp, Mapping) else report.get("available", False)
        )
        tls = report.get("tls_probe")
        tls_available = bool(
            isinstance(tls, Mapping) and tls.get("status") == "available"
        )
        return cls(
            transport=transport,
            endpoint=endpoint,
            tcp_available=tcp_available,
            tls_available=tls_available,
            authenticated=authenticated,
            expires_at=expires_at,
            priority=priority,
            path_kind=str(report.get("path_kind", "") or ""),
        )

    def status(self, *, require_authentication: bool, now: float) -> str:
        if not self.tcp_available:
            return "tcp_unavailable"
        if self.transport == WSS_443:
            if not self.tls_available:
                return "tls_unavailable"
            if require_authentication and not self.authenticated:
                return "authentication_pending"
            if self.expires_at is not None and float(self.expires_at) <= now:
                return "probe_expired"
        return "available"

    def public_view(self, *, require_authentication: bool, now: float) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "tcp_available": self.tcp_available,
            "tls_available": self.tls_available,
            "authenticated": self.authenticated,
            "path_kind": self.path_kind,
            "status": self.status(require_authentication=require_authentication, now=now),
        }


@dataclass(frozen=True)
class TransportPolicy:
    """Selection policy; legacy TCP remains the safe default."""

    prefer_wss: bool = False
    require_wss_authentication: bool = True
    allow_legacy_tcp: bool = True


@dataclass(frozen=True)
class TransportDecision:
    selected_transport: str
    fallback_reason: str
    considered: tuple[dict[str, Any], ...]
    contract_sha256: str

    def public_view(self) -> dict[str, Any]:
        return {
            "selected_transport": self.selected_transport,
            "fallback_reason": self.fallback_reason,
            "considered": [dict(item) for item in self.considered],
            "contract_sha256": self.contract_sha256,
        }


def select_transport(
    candidates: tuple[TransportCandidate, ...] | list[TransportCandidate],
    *,
    policy: TransportPolicy | None = None,
    now: float | None = None,
) -> TransportDecision:
    """Select one route without opening a connection.

    A normal TCP observation on port 8888 can never make ``wss_443`` usable;
    both TCP and trusted TLS observations are required.  Authentication is
    required by default because a TLS certificate alone is not a cluster ACL.
    """
    resolved_policy = policy or TransportPolicy()
    observed_at = time.time() if now is None else float(now)
    by_transport: dict[str, TransportCandidate] = {}
    for candidate in candidates:
        if candidate.transport in by_transport:
            raise TransportContractError("transport_duplicate", "duplicate transport candidate")
        by_transport[candidate.transport] = candidate

    considered: list[dict[str, Any]] = []
    usable: dict[str, TransportCandidate] = {}
    for transport in (WSS_443, LEGACY_TCP):
        candidate = by_transport.get(transport)
        if candidate is None:
            considered.append({"transport": transport, "status": "not_observed"})
            continue
        status = candidate.status(
            require_authentication=resolved_policy.require_wss_authentication,
            now=observed_at,
        )
        considered.append(candidate.public_view(
            require_authentication=resolved_policy.require_wss_authentication,
            now=observed_at,
        ))
        if status == "available" and (transport != LEGACY_TCP or resolved_policy.allow_legacy_tcp):
            usable[transport] = candidate

    order = (WSS_443, LEGACY_TCP) if resolved_policy.prefer_wss else (LEGACY_TCP, WSS_443)
    selected = next((kind for kind in order if kind in usable), None)
    if selected is None:
        raise TransportContractError("no_transport_available", "no usable cluster transport")
    fallback_reason = ""
    if selected == LEGACY_TCP and resolved_policy.prefer_wss:
        wss_status = next(
            (item["status"] for item in considered if item.get("transport") == WSS_443),
            "not_observed",
        )
        fallback_reason = f"wss_unavailable:{wss_status}"
    digest_payload = {
        "selected_transport": selected,
        "fallback_reason": fallback_reason,
        "considered": considered,
        "policy": {
            "prefer_wss": resolved_policy.prefer_wss,
            "require_wss_authentication": resolved_policy.require_wss_authentication,
            "allow_legacy_tcp": resolved_policy.allow_legacy_tcp,
        },
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TransportDecision(selected, fallback_reason, tuple(considered), digest)


@dataclass(frozen=True)
class TransportLease:
    attempt_id: str
    generation: int
    transport: str
    lease_token: str
    fallback_reason: str = ""


@dataclass
class _AttemptRecord:
    generation: int
    transport: str
    lease_token: str
    state: str = "active"
    fallback_used: bool = False
    fallback_reason: str = ""


class TransportSession:
    """Fence one node's transport attempts across reconnect generations."""

    def __init__(self, node_id: str) -> None:
        if not str(node_id).strip():
            raise TransportContractError("node_id_missing", "node id is required")
        self.node_id = str(node_id)
        self._generation = -1
        self._attempts: dict[str, _AttemptRecord] = {}

    @property
    def generation(self) -> int:
        return self._generation

    def begin(self, *, generation: int, attempt_id: str, transport: str) -> TransportLease:
        self._advance_generation(generation)
        self._validate_transport(transport)
        attempt = str(attempt_id).strip()
        if not attempt:
            raise TransportContractError("attempt_id_missing", "attempt id is required")
        if attempt in self._attempts:
            raise TransportContractError("attempt_duplicate", "attempt is already fenced or active")
        token = uuid.uuid4().hex
        self._attempts[attempt] = _AttemptRecord(generation, transport, token)
        return TransportLease(attempt, generation, transport, token)

    def fallback(self, *, generation: int, attempt_id: str, reason: str) -> TransportLease:
        record = self._require_active(generation, attempt_id)
        if record.transport != WSS_443:
            raise TransportContractError("fallback_invalid", "only WSS attempts may fall back")
        if record.fallback_used:
            raise TransportContractError("fallback_duplicate", "fallback already consumed")
        record.transport = LEGACY_TCP
        record.fallback_used = True
        record.fallback_reason = str(reason).strip() or "wss_failed"
        return TransportLease(
            str(attempt_id), generation, LEGACY_TCP, record.lease_token, record.fallback_reason,
        )

    def complete(self, *, generation: int, attempt_id: str, transport: str) -> None:
        record = self._require_active(generation, attempt_id)
        if record.transport != transport:
            raise TransportContractError("transport_fenced", "result belongs to an old transport")
        record.state = "completed"

    def abort(self, *, generation: int, attempt_id: str, reason: str = "") -> None:
        record = self._require_active(generation, attempt_id)
        record.state = "failed"
        record.fallback_reason = str(reason).strip()

    def snapshot(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "generation": self._generation,
            "attempts": {
                attempt_id: {
                    "generation": record.generation,
                    "transport": record.transport,
                    "state": record.state,
                    "fallback_used": record.fallback_used,
                    "fallback_reason": record.fallback_reason,
                }
                for attempt_id, record in self._attempts.items()
            },
        }

    def _advance_generation(self, generation: int) -> None:
        try:
            resolved = int(generation)
        except (TypeError, ValueError) as exc:
            raise TransportContractError("generation_invalid", "generation must be an integer") from exc
        if resolved < self._generation:
            raise TransportContractError("generation_stale", "transport generation is stale")
        if resolved == self._generation:
            return
        for record in self._attempts.values():
            if record.state == "active":
                record.state = "fenced"
        self._generation = resolved

    @staticmethod
    def _validate_transport(transport: str) -> None:
        if transport not in _TRANSPORTS:
            raise TransportContractError("transport_unknown", "unsupported transport")

    def _require_active(self, generation: int, attempt_id: str) -> _AttemptRecord:
        if int(generation) != self._generation:
            raise TransportContractError("generation_stale", "transport generation is stale")
        record = self._attempts.get(str(attempt_id))
        if record is None or record.state != "active":
            raise TransportContractError("attempt_fenced", "transport attempt is not active")
        if record.generation != self._generation:
            raise TransportContractError("attempt_fenced", "transport attempt belongs to an old generation")
        return record

