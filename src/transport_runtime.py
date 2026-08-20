"""Runtime seam for Transport v2 without changing the Legacy TCP wire format.

The current cluster protocol still uses the established length-prefixed TCP
frames.  :class:`TransportRuntimeBridge` records those frames as an envelope
observation and, when explicitly injected, can drive a v2 endpoint such as the
deterministic fake link used by tests.  This keeps production behaviour stable
while giving TCPClient/Scheduler a single place to adopt a real v2 transport.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

try:
    from .cluster_transport import (
        BLOB_CHANNEL,
        CONTROL_CHANNEL,
        LEGACY_TCP,
        STREAM_CHANNEL,
        TransportCircuitBreaker,
        TransportChunkAck,
        TransportContractError,
        TransportEnvelope,
        TransportWindow,
    )
except ImportError:  # PYTHONPATH=src compatibility
    from cluster_transport import (
        BLOB_CHANNEL,
        CONTROL_CHANNEL,
        LEGACY_TCP,
        STREAM_CHANNEL,
        TransportCircuitBreaker,
        TransportChunkAck,
        TransportContractError,
        TransportEnvelope,
        TransportWindow,
    )


def _failure_code(error: BaseException | str) -> str:
    """Map local exceptions to the stable Transport v2 failure matrix."""

    if isinstance(error, TransportContractError):
        return error.code
    text = str(error).lower()
    if "timeout" in text:
        return "connection_timeout"
    if "reset" in text or "closed" in text or "socket" in text:
        return "connection_reset"
    return "connection_reset"


class TransportRuntimeBridge:
    """Small, thread-safe runtime adapter shared by TCPClient and Scheduler.

    ``mode=legacy_tcp`` is observational: it allocates local envelope metadata
    but never changes bytes written to the existing TCP socket.  ``send_v2``
    and ``receive_v2`` are intentionally explicit and are used by fake/WSS
    endpoints during migration and tests.
    """

    def __init__(
        self,
        *,
        mode: str = LEGACY_TCP,
        window_bytes: int = 1 << 20,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        client_id: str = "runtime-client",
    ) -> None:
        self.mode = str(mode or LEGACY_TCP)
        if self.mode not in {LEGACY_TCP, "v2"}:
            raise TransportContractError("transport_unknown", "unsupported runtime transport mode")
        self.client_id = str(client_id or "runtime-client")
        self._window_bytes = int(window_bytes)
        self._failure_threshold = int(failure_threshold)
        self._cooldown_seconds = float(cooldown_seconds)
        self._lock = threading.RLock()
        self._generation = 0
        self._attempt_id = ""
        self._sequences: dict[str, int] = {}
        self._window = TransportWindow(self._window_bytes)
        self._breaker = TransportCircuitBreaker(
            failure_threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown_seconds,
        )
        self._counters = {
            "legacy_outbound": 0,
            "legacy_inbound": 0,
            "v2_outbound": 0,
            "v2_inbound": 0,
            "failures": 0,
        }
        self._last_failure: dict[str, Any] | None = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def attempt_id(self) -> str:
        with self._lock:
            return self._attempt_id

    def connected(self, generation: int, *, attempt_id: str | None = None) -> str:
        """Fence old work and start a fresh transport attempt."""

        if not isinstance(generation, int) or generation < 0:
            raise TransportContractError("generation_invalid", "runtime generation must be non-negative")
        with self._lock:
            self._generation = generation
            self._attempt_id = str(attempt_id or uuid.uuid4().hex)
            self._sequences.clear()
            self._window = TransportWindow(self._window_bytes)
            self._breaker.success()
            return self._attempt_id

    def _ensure_attempt(self, generation: int | None = None, attempt_id: str | None = None) -> tuple[int, str]:
        with self._lock:
            if generation is not None and generation != self._generation:
                raise TransportContractError("generation_stale", "runtime generation is stale")
            if not self._attempt_id:
                self._attempt_id = str(attempt_id or uuid.uuid4().hex)
            elif attempt_id is not None and attempt_id != self._attempt_id:
                raise TransportContractError("attempt_fenced", "runtime attempt is fenced")
            return self._generation, self._attempt_id

    def _next_sequence(self, channel: str) -> int:
        with self._lock:
            sequence = self._sequences.get(channel, 0)
            self._sequences[channel] = sequence + 1
            return sequence

    def envelope_for(
        self,
        payload: bytes,
        *,
        request_id: str,
        channel: str = CONTROL_CHANNEL,
        deadline_ms: int | None = None,
        connection_generation: int | None = None,
        attempt_id: str | None = None,
    ) -> TransportEnvelope:
        generation, attempt = self._ensure_attempt(connection_generation, attempt_id)
        return TransportEnvelope.from_payload(
            bytes(payload),
            request_id=str(request_id or self.client_id),
            connection_generation=generation,
            attempt_id=attempt,
            channel=channel,
            sequence=self._next_sequence(channel),
            deadline_ms=int(deadline_ms or (time.time() * 1000) + 30_000),
        )

    def _allow_v2(self) -> None:
        with self._lock:
            if not self._breaker.allow():
                raise TransportContractError("circuit_open", "transport circuit is open")

    def send_v2(
        self,
        endpoint: Any,
        payload: bytes,
        *,
        request_id: str,
        channel: str = STREAM_CHANNEL,
        deadline_ms: int | None = None,
    ) -> TransportEnvelope:
        """Send one envelope through an explicit v2 endpoint."""

        self._allow_v2()
        envelope = self.envelope_for(
            payload,
            request_id=request_id,
            channel=channel,
            deadline_ms=deadline_ms,
        )
        try:
            endpoint.send(envelope, bytes(payload))
        except Exception as exc:
            self.record_failure(exc)
            raise
        with self._lock:
            if channel in {STREAM_CHANNEL, BLOB_CHANNEL}:
                self._window.reserve(
                    request_id=envelope.request_id,
                    connection_generation=envelope.connection_generation,
                    attempt_id=envelope.attempt_id,
                    sequence=envelope.sequence,
                    payload_size=envelope.payload_size,
                )
            self._counters["v2_outbound"] += 1
            self._breaker.success()
        return envelope

    def acknowledge_v2(self, ack: TransportChunkAck) -> None:
        """Release one locally tracked stream/blob chunk after its ACK."""

        with self._lock:
            if ack.connection_generation != self._generation:
                raise TransportContractError("generation_stale", "acknowledgement generation is stale")
            if ack.attempt_id != self._attempt_id:
                raise TransportContractError("attempt_fenced", "acknowledgement attempt is fenced")
            self._window.acknowledge(ack)

    def receive_v2(self, endpoint: Any) -> Any:
        """Receive and validate one frame from an explicit v2 endpoint."""

        self._allow_v2()
        try:
            frame = endpoint.receive()
        except Exception as exc:
            self.record_failure(exc)
            raise
        with self._lock:
            self._counters["v2_inbound"] += 1
            self._breaker.success()
        return frame

    def observe_legacy_send(
        self,
        payload: bytes,
        *,
        request_id: str | None = None,
        channel: str = CONTROL_CHANNEL,
        connection_generation: int | None = None,
    ) -> TransportEnvelope:
        """Record a Legacy TCP frame without changing the wire bytes."""

        envelope = self.envelope_for(
            payload,
            request_id=request_id or self.client_id,
            channel=channel,
            connection_generation=connection_generation,
        )
        with self._lock:
            self._counters["legacy_outbound"] += 1
        return envelope

    def observe_legacy_receive(self, payload: Any, *, connection_generation: int | None = None) -> None:
        """Record an inbound Legacy TCP message for runtime diagnostics."""

        if connection_generation is not None:
            self._ensure_attempt(connection_generation)
        with self._lock:
            self._counters["legacy_inbound"] += 1

    def record_failure(self, error: BaseException | str) -> dict[str, Any]:
        code = _failure_code(error)
        with self._lock:
            view = self._breaker.record(code).public_view()
            self._counters["failures"] += 1
            self._last_failure = view
        return view

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "generation": self._generation,
                "attempt_id": self._attempt_id,
                "sequences": dict(sorted(self._sequences.items())),
                "window": self._window.snapshot(),
                "breaker": self._breaker.snapshot(),
                "counters": dict(self._counters),
                "last_failure": dict(self._last_failure) if self._last_failure else None,
            }


__all__ = ["TransportRuntimeBridge"]
