"""Path selection and fencing primitives for the cluster transport layer.

This module deliberately contains no socket or WebSocket implementation.  It
freezes the control-plane contract needed before a second physical transport
is introduced, while keeping the existing TCP route as the default.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping


LEGACY_TCP = "legacy_tcp"
WSS_443 = "wss_443"
_TRANSPORTS = frozenset({LEGACY_TCP, WSS_443})

TRANSPORT_V2_SCHEMA = "qlh.cluster_transport.v2"
TRANSPORT_V2_VERSION = 1
CONTROL_CHANNEL = "control"
STREAM_CHANNEL = "stream"
BLOB_CHANNEL = "blob"
_CHANNELS = frozenset({CONTROL_CHANNEL, STREAM_CHANNEL, BLOB_CHANNEL})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# These codes are deliberately protocol-facing.  Callers can attach local
# diagnostics, but retries and UI decisions must use this stable table.
TRANSPORT_FAILURE_MATRIX: dict[str, dict[str, Any]] = {
    "connection_timeout": {"retryable": True, "next_action": "retry_backoff"},
    "tls_auth_failed": {"retryable": False, "next_action": "fallback_or_revoke"},
    "connection_reset": {"retryable": True, "next_action": "reconnect"},
    "sequence_duplicate": {"retryable": False, "next_action": "ack_or_abort"},
    "sequence_out_of_order": {"retryable": False, "next_action": "request_resync"},
    "ack_size_mismatch": {"retryable": False, "next_action": "abort_chunk"},
    "ack_context_mismatch": {"retryable": False, "next_action": "abort_chunk"},
    "payload_mismatch": {"retryable": False, "next_action": "abort_chunk"},
    "window_exhausted": {"retryable": True, "next_action": "wait_for_ack"},
    "deadline_exceeded": {"retryable": False, "next_action": "cancel_attempt"},
    "generation_stale": {"retryable": False, "next_action": "fence_attempt"},
    "attempt_fenced": {"retryable": False, "next_action": "fence_attempt"},
    "circuit_open": {"retryable": True, "next_action": "cooldown_probe"},
}


class TransportContractError(ValueError):
    """Raised when a transport contract or fencing transition is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TransportFailure:
    """A stable, serializable failure classification for Transport v2."""

    code: str
    retryable: bool
    next_action: str

    @classmethod
    def from_code(cls, code: str) -> "TransportFailure":
        entry = TRANSPORT_FAILURE_MATRIX.get(str(code))
        if entry is None:
            raise TransportContractError("failure_unknown", "unknown transport failure code")
        return cls(str(code), bool(entry["retryable"]), str(entry["next_action"]))

    def public_view(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "retryable": self.retryable,
            "next_action": self.next_action,
        }


def _required_text(value: Any, field: str, *, max_length: int = 128) -> str:
    resolved = str(value or "").strip()
    if not resolved or len(resolved) > max_length:
        raise TransportContractError(f"{field}_invalid", f"{field} is invalid")
    return resolved


@dataclass(frozen=True)
class TransportEnvelope:
    """Transport v2 application envelope, independent of the physical socket."""

    request_id: str
    connection_generation: int
    attempt_id: str
    channel: str
    sequence: int
    deadline_ms: int
    payload_digest: str
    payload_size: int
    schema: str = TRANSPORT_V2_SCHEMA
    version: int = TRANSPORT_V2_VERSION

    def __post_init__(self) -> None:
        if self.schema != TRANSPORT_V2_SCHEMA or self.version != TRANSPORT_V2_VERSION:
            raise TransportContractError("schema_unsupported", "unsupported transport envelope schema")
        _required_text(self.request_id, "request_id")
        _required_text(self.attempt_id, "attempt_id")
        if not isinstance(self.connection_generation, int) or self.connection_generation < 0:
            raise TransportContractError("generation_invalid", "connection generation must be non-negative")
        if self.channel not in _CHANNELS:
            raise TransportContractError("channel_invalid", "unsupported transport channel")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise TransportContractError("sequence_invalid", "sequence must be non-negative")
        if not isinstance(self.deadline_ms, int) or self.deadline_ms <= 0:
            raise TransportContractError("deadline_invalid", "deadline must be a positive epoch millisecond")
        if not isinstance(self.payload_size, int) or self.payload_size < 0:
            raise TransportContractError("payload_size_invalid", "payload size must be non-negative")
        if not isinstance(self.payload_digest, str) or not _DIGEST_RE.fullmatch(self.payload_digest):
            raise TransportContractError("payload_digest_invalid", "payload digest must be lowercase SHA-256")

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        *,
        request_id: str,
        connection_generation: int,
        attempt_id: str,
        channel: str,
        sequence: int,
        deadline_ms: int,
    ) -> "TransportEnvelope":
        body = bytes(payload)
        return cls(
            request_id=_required_text(request_id, "request_id"),
            connection_generation=connection_generation,
            attempt_id=_required_text(attempt_id, "attempt_id"),
            channel=channel,
            sequence=sequence,
            deadline_ms=deadline_ms,
            payload_digest=hashlib.sha256(body).hexdigest(),
            payload_size=len(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "request_id": self.request_id,
            "connection_generation": self.connection_generation,
            "attempt_id": self.attempt_id,
            "channel": self.channel,
            "sequence": self.sequence,
            "deadline_ms": self.deadline_ms,
            "payload_digest": self.payload_digest,
            "payload_size": self.payload_size,
        }

    def encode(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def decode(cls, encoded: bytes | bytearray | str) -> "TransportEnvelope":
        try:
            raw = encoded.decode("utf-8") if isinstance(encoded, (bytes, bytearray)) else str(encoded)
            value = json.loads(raw)
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TransportContractError("envelope_invalid", "transport envelope is not valid JSON") from exc
        if not isinstance(value, dict):
            raise TransportContractError("envelope_invalid", "transport envelope must be an object")
        expected = {
            "schema", "version", "request_id", "connection_generation", "attempt_id",
            "channel", "sequence", "deadline_ms", "payload_digest", "payload_size",
        }
        if set(value) != expected:
            raise TransportContractError("envelope_fields_invalid", "transport envelope fields are not canonical")
        return cls(**value)

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        current = int(time.time() * 1000) if now_ms is None else int(now_ms)
        return current >= self.deadline_ms


@dataclass(frozen=True)
class TransportChunkAck:
    request_id: str
    connection_generation: int
    attempt_id: str
    sequence: int
    payload_size: int

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.attempt_id, "attempt_id")
        if not isinstance(self.connection_generation, int) or self.connection_generation < 0:
            raise TransportContractError("generation_invalid", "connection generation must be non-negative")
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise TransportContractError("sequence_invalid", "sequence must be non-negative")
        if not isinstance(self.payload_size, int) or self.payload_size < 0:
            raise TransportContractError("payload_size_invalid", "payload size must be non-negative")


class TransportWindow:
    """Bounded in-flight byte window for stream/blob channels."""

    def __init__(self, capacity_bytes: int) -> None:
        if not isinstance(capacity_bytes, int) or capacity_bytes <= 0:
            raise TransportContractError("window_invalid", "window capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self._inflight: dict[tuple[str, int, str, int], int] = {}
        self._acked: set[tuple[str, int, str, int]] = set()

    @property
    def inflight_bytes(self) -> int:
        return sum(self._inflight.values())

    def reserve(
        self,
        *,
        request_id: str,
        connection_generation: int,
        attempt_id: str,
        sequence: int,
        payload_size: int,
    ) -> None:
        request = _required_text(request_id, "request_id")
        attempt = _required_text(attempt_id, "attempt_id")
        if not isinstance(connection_generation, int) or connection_generation < 0:
            raise TransportContractError("generation_invalid", "connection generation must be non-negative")
        if not isinstance(sequence, int) or sequence < 0 or not isinstance(payload_size, int) or payload_size <= 0:
            raise TransportContractError("chunk_invalid", "chunk sequence and size are invalid")
        key = (request, connection_generation, attempt, sequence)
        if key in self._inflight or key in self._acked:
            raise TransportContractError("sequence_duplicate", "chunk sequence was already reserved")
        if self.inflight_bytes + payload_size > self.capacity_bytes:
            raise TransportContractError("window_exhausted", "transport send window is exhausted")
        self._inflight[key] = payload_size

    def acknowledge(self, ack: TransportChunkAck) -> None:
        key = (ack.request_id, ack.connection_generation, ack.attempt_id, ack.sequence)
        expected = self._inflight.get(key)
        if expected is None:
            if key in self._acked:
                raise TransportContractError("sequence_duplicate", "chunk acknowledgement was already received")
            if any(pending[3] == ack.sequence for pending in self._inflight):
                raise TransportContractError("ack_context_mismatch", "chunk acknowledgement belongs to another request")
            raise TransportContractError("sequence_out_of_order", "chunk acknowledgement is not pending")
        if expected != ack.payload_size:
            raise TransportContractError("ack_size_mismatch", "chunk acknowledgement size does not match")
        del self._inflight[key]
        self._acked.add(key)

    def snapshot(self) -> dict[str, int]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "inflight_bytes": self.inflight_bytes,
            "pending_chunks": len(self._inflight),
            "acked_chunks": len(self._acked),
        }


class TransportCircuitBreaker:
    """Small deterministic breaker for repeated retryable transport failures."""

    def __init__(self, *, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> None:
        if failure_threshold <= 0 or cooldown_seconds <= 0:
            raise TransportContractError("circuit_invalid", "circuit breaker parameters must be positive")
        self.failure_threshold = int(failure_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def allow(self, *, now: float | None = None) -> bool:
        if self._opened_at is None:
            return True
        current = time.monotonic() if now is None else float(now)
        return current - self._opened_at >= self.cooldown_seconds

    def record(self, code: str, *, now: float | None = None) -> TransportFailure:
        failure = TransportFailure.from_code(code)
        if failure.retryable:
            self._failures += 1
            if self._failures >= self.failure_threshold and self._opened_at is None:
                self._opened_at = time.monotonic() if now is None else float(now)
        return failure

    def success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def snapshot(self) -> dict[str, Any]:
        return {"state": "open" if self.is_open else "closed", "failures": self._failures}


class DeterministicClock:
    """Test-only clock; advancing it never sleeps or touches wall-clock time."""

    def __init__(self, *, epoch_ms: int = 0) -> None:
        if not isinstance(epoch_ms, int) or epoch_ms < 0:
            raise TransportContractError("clock_invalid", "clock epoch must be non-negative")
        self._epoch_ms = epoch_ms

    @property
    def now_ms(self) -> int:
        return self._epoch_ms

    @property
    def now(self) -> float:
        return self._epoch_ms / 1000.0

    def advance(self, milliseconds: int) -> int:
        if not isinstance(milliseconds, int) or milliseconds < 0:
            raise TransportContractError("clock_invalid", "clock advance must be non-negative")
        self._epoch_ms += milliseconds
        return self._epoch_ms


class EventBarrier:
    """Non-blocking named barrier for deterministic ordering assertions."""

    def __init__(self, names: Mapping[str, bool] | list[str] | tuple[str, ...] = ()) -> None:
        if isinstance(names, Mapping):
            initial = {str(name): bool(value) for name, value in names.items()}
        else:
            initial = {str(name): False for name in names}
        if any(not name.strip() for name in initial):
            raise TransportContractError("barrier_invalid", "barrier names must be non-empty")
        self._released = initial

    def release(self, name: str) -> None:
        key = _required_text(name, "barrier_name")
        self._released[key] = True

    def is_released(self, name: str) -> bool:
        key = _required_text(name, "barrier_name")
        return bool(self._released.get(key, False))

    def require_released(self, name: str) -> None:
        if not self.is_released(name):
            raise TransportContractError("barrier_blocked", "required event barrier is not released")

    def snapshot(self) -> dict[str, bool]:
        return dict(sorted(self._released.items()))


@dataclass(frozen=True)
class TransportFrame:
    """In-memory test frame; never exposed through a production status API."""

    envelope: TransportEnvelope
    payload: bytes


class FakeTransportLink:
    """Deterministic in-memory link used by NW4.1/T-RACE tests only.

    Frames are queued until ``deliver_next`` is called, making drop, duplicate,
    reorder and close races explicit without threads or real sockets.
    """

    def __init__(self) -> None:
        self._outbound: dict[str, Deque[TransportFrame]] = {"left": deque(), "right": deque()}
        self._inbound: dict[str, Deque[TransportFrame]] = {"left": deque(), "right": deque()}
        self._closed: set[str] = set()

    @classmethod
    def pair(cls) -> tuple["FakeTransportEndpoint", "FakeTransportEndpoint"]:
        link = cls()
        return (
            FakeTransportEndpoint(link, "left"),
            FakeTransportEndpoint(link, "right"),
        )

    def enqueue(self, side: str, frame: TransportFrame) -> None:
        self._outbound[side].append(frame)

    def deliver_next(self, side: str) -> TransportFrame:
        queue = self._outbound[side]
        if not queue:
            raise TransportContractError("transport_empty", "no queued frame is available")
        frame = queue.popleft()
        peer = "right" if side == "left" else "left"
        self._inbound[peer].append(frame)
        return frame

    def drop_next(self, side: str) -> TransportFrame:
        queue = self._outbound[side]
        if not queue:
            raise TransportContractError("transport_empty", "no queued frame is available")
        return queue.popleft()

    def duplicate_next(self, side: str) -> None:
        queue = self._outbound[side]
        if not queue:
            raise TransportContractError("transport_empty", "no queued frame is available")
        queue.appendleft(queue[0])

    def tamper_next(self, side: str, payload: bytes) -> None:
        queue = self._outbound[side]
        if not queue:
            raise TransportContractError("transport_empty", "no queued frame is available")
        frame = queue[0]
        queue[0] = TransportFrame(frame.envelope, bytes(payload))

    def reorder(self, side: str, first: int, second: int) -> None:
        queue = self._outbound[side]
        if min(first, second) < 0 or max(first, second) >= len(queue):
            raise TransportContractError("transport_index_invalid", "reorder index is out of range")
        values = list(queue)
        values[first], values[second] = values[second], values[first]
        queue.clear()
        queue.extend(values)

    def receive(self, side: str) -> TransportFrame:
        queue = self._inbound[side]
        if not queue:
            raise TransportContractError("transport_empty", "no delivered frame is available")
        return queue.popleft()

    def close(self, side: str) -> None:
        self._closed.add(side)

    def is_closed(self, side: str) -> bool:
        return side in self._closed


class FakeTransportEndpoint:
    """One endpoint of :class:`FakeTransportLink`, for deterministic tests."""

    def __init__(self, link: FakeTransportLink, side: str) -> None:
        self._link = link
        self.side = side
        self._generation = 0
        self._attempt_id = ""
        self._window = TransportWindow(1 << 20)
        self._received: dict[str, int] = {}

    def open(self, *, generation: int, attempt_id: str, window_bytes: int = 1 << 20) -> None:
        if not isinstance(generation, int) or generation < 0:
            raise TransportContractError("generation_invalid", "generation must be non-negative")
        self._generation = generation
        self._attempt_id = _required_text(attempt_id, "attempt_id")
        self._window = TransportWindow(window_bytes)
        self._received.clear()
        self._link._closed.discard(self.side)

    def send(self, envelope: TransportEnvelope, payload: bytes, *, now_ms: int | None = None) -> None:
        if self._link.is_closed(self.side) or self._link.is_closed("right" if self.side == "left" else "left"):
            raise TransportContractError("connection_reset", "fake transport endpoint is closed")
        if envelope.connection_generation != self._generation:
            raise TransportContractError("generation_stale", "envelope generation does not match endpoint")
        if envelope.attempt_id != self._attempt_id:
            raise TransportContractError("attempt_fenced", "envelope attempt does not match endpoint")
        body = bytes(payload)
        if envelope.is_expired(now_ms=now_ms):
            raise TransportContractError("deadline_exceeded", "transport envelope deadline has expired")
        if envelope.payload_size != len(body) or envelope.payload_digest != hashlib.sha256(body).hexdigest():
            raise TransportContractError("payload_mismatch", "transport payload does not match envelope")
        if envelope.channel in {STREAM_CHANNEL, BLOB_CHANNEL}:
            self._window.reserve(
                request_id=envelope.request_id,
                connection_generation=envelope.connection_generation,
                attempt_id=envelope.attempt_id,
                sequence=envelope.sequence,
                payload_size=len(body),
            )
        self._link.enqueue(self.side, TransportFrame(envelope, body))

    def acknowledge(self, ack: TransportChunkAck) -> None:
        if ack.connection_generation != self._generation or ack.attempt_id != self._attempt_id:
            raise TransportContractError("attempt_fenced", "acknowledgement belongs to an old endpoint")
        self._window.acknowledge(ack)

    def deliver_next(self) -> TransportFrame:
        return self._link.deliver_next(self.side)

    def drop_next(self) -> TransportFrame:
        return self._link.drop_next(self.side)

    def duplicate_next(self) -> None:
        self._link.duplicate_next(self.side)

    def tamper_next(self, payload: bytes) -> None:
        self._link.tamper_next(self.side, payload)

    def reorder(self, first: int, second: int) -> None:
        self._link.reorder(self.side, first, second)

    def receive(self, *, now_ms: int | None = None) -> TransportFrame:
        frame = self._link.receive(self.side)
        if frame.envelope.connection_generation != self._generation:
            raise TransportContractError("generation_stale", "received frame belongs to an old generation")
        if frame.envelope.attempt_id != self._attempt_id:
            raise TransportContractError("attempt_fenced", "received frame belongs to an old attempt")
        if frame.envelope.is_expired(now_ms=now_ms):
            raise TransportContractError("deadline_exceeded", "received frame deadline has expired")
        if (
            frame.envelope.payload_size != len(frame.payload)
            or frame.envelope.payload_digest != hashlib.sha256(frame.payload).hexdigest()
        ):
            raise TransportContractError("payload_mismatch", "received payload does not match envelope")
        channel = frame.envelope.channel
        previous = self._received.get(channel, -1)
        if frame.envelope.sequence <= previous:
            raise TransportContractError("sequence_duplicate", "received sequence was already delivered")
        if frame.envelope.sequence != previous + 1:
            raise TransportContractError("sequence_out_of_order", "received sequence is not contiguous")
        self._received[channel] = frame.envelope.sequence
        return frame

    def close(self) -> None:
        self._link.close(self.side)

    def snapshot(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "generation": self._generation,
            "attempt_id": self._attempt_id,
            "window": self._window.snapshot(),
            "received": dict(sorted(self._received.items())),
        }


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
