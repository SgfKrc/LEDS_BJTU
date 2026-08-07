"""Dedicated protocol-v3 control and worker adapter for diffusion stages.

This module deliberately does not register a TaskGraph provider.  It owns the
image-worker half of the wire protocol so v2 text workers remain untouched
until a RemoteDiffusionProvider has passed its own admission tests.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from task_provider import (
    DEPENDENCY_FAILURES_KEY,
    ProviderBusy,
    ProviderCapabilities,
    ProviderExecutionError,
    ProviderReservationError,
    ProviderUnavailable,
    Reservation,
    StageAttempt,
    StageRequest,
    StageResult,
)
from task_worker_protocol import (
    WorkerMessage,
    WorkerProtocolError,
    build_message,
    canonical_message_bytes,
    decode_message,
    negotiate_protocol_version,
    stage_input_sha256,
    stage_output_sha256,
)


IMAGE_PROTOCOL_VERSION = 3
_MESSAGE_CACHE_LIMIT = 1024


def _message_id(prefix: str) -> str:
    return f"msg_{prefix}{uuid.uuid4().hex}"


def _message_digest(message: WorkerMessage) -> str:
    return hashlib.sha256(canonical_message_bytes(message)).hexdigest()


def remote_diffusion_provider_id(node_id: str) -> str:
    """Return the Provider ID reserved for one v3 diffusion Worker."""
    raw = str(node_id or "")
    safe = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "_.-")
        else "_"
        for character in raw
    ).strip("._") or "worker"
    base = f"remote_diffusion_{safe}"
    if len(base) <= 64 and safe == raw:
        return base
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"remote_diffusion_{safe[:36]}_{digest}"[:64]


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: payload[name]
        for name in (
            "workflow_id",
            "stage_id",
            "attempt_id",
            "lease_id",
            "lease_epoch",
            "provider_id",
        )
    }


def _attempt_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: payload[name]
        for name in (
            "workflow_id",
            "stage_id",
            "attempt_id",
            "lease_id",
            "lease_epoch",
        )
    }


@dataclass(frozen=True)
class DiffusionExecutionResult:
    output: dict[str, Any]
    metadata: dict[str, Any]
    transfer_plan: dict[str, Any]


DiffusionExecutor = Callable[[Mapping[str, Any], threading.Event], DiffusionExecutionResult]
SendMessage = Callable[[WorkerMessage], None]
DiffusionResultIngestor = Callable[[StageAttempt, Mapping[str, Any], Mapping[str, Any]], dict[str, Any]]
DiffusionInputTransferPlanBuilder = Callable[[StageAttempt, StageRequest], dict[str, Any]]


@dataclass
class _ActiveStage:
    offer: WorkerMessage
    lease_expires_at_ms: int
    lease_deadline_monotonic: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    accepted: WorkerMessage | None = None
    terminal: WorkerMessage | None = None
    lease_expired: bool = False


@dataclass
class _PendingDiffusionAttempt:
    attempt: StageAttempt
    lease_expires_at: float
    accepted: bool = False
    accept_event: threading.Event = field(default_factory=threading.Event)
    result_event: threading.Event = field(default_factory=threading.Event)
    cancel_ack_event: threading.Event = field(default_factory=threading.Event)
    output: dict[str, Any] | None = None
    output_transfer_plan: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    error: BaseException | None = None
    cancel_requested: bool = False
    cancel_acknowledged: bool = False
    released: bool = False
    released_at: float = 0.0


class DiffusionCoordinatorControlPlane:
    """V3-only hello negotiation and capability snapshots for image workers."""

    def __init__(self, *, health_timeout_seconds: float = 120.0, clock=time.time):
        self._health_timeout_seconds = max(1.0, float(health_timeout_seconds))
        self._clock = clock
        self._workers: dict[str, dict[str, Any]] = {}
        self._seen: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._seen_order: collections.deque[tuple[str, str]] = collections.deque()
        self._rejected = 0
        self._lock = threading.RLock()

    @staticmethod
    def _reject(code: str, message: str) -> WorkerProtocolError:
        return WorkerProtocolError(message, code=code, field="message_type")

    def _remember(self, peer_id: str, request: WorkerMessage, response: WorkerMessage) -> None:
        key = (peer_id, request.message_id)
        self._seen[key] = (_message_digest(request), response.snapshot())
        self._seen_order.append(key)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen.pop(expired, None)

    def _duplicate(self, peer_id: str, message: WorkerMessage) -> WorkerMessage | None:
        cached = self._seen.get((peer_id, message.message_id))
        if cached is None:
            return None
        if cached[0] != _message_digest(message):
            self._rejected += 1
            raise self._reject(
                "message_id_conflict",
                "message_id was reused with different content",
            )
        return decode_message(cached[1])

    def receive_hello(
        self,
        peer_id: str,
        raw: bytes | str | Mapping[str, Any],
        *,
        coordinator_node_id: str,
        sent_at_ms: int | None = None,
    ) -> WorkerMessage:
        message = decode_message(raw)
        with self._lock:
            duplicate = self._duplicate(peer_id, message)
            if duplicate is not None:
                return duplicate
            if message.message_type != "hello":
                self._rejected += 1
                raise self._reject(
                    "control_plane_only",
                    "the diffusion coordinator accepts hello messages only",
                )
            payload = message.payload
            accepted = message.version == IMAGE_PROTOCOL_VERSION and payload["node_id"] == peer_id
            selected_version = 0
            reason_code = ""
            if accepted:
                try:
                    selected_version = negotiate_protocol_version(
                        payload["min_version"],
                        payload["max_version"],
                        local_min_version=IMAGE_PROTOCOL_VERSION,
                        local_max_version=IMAGE_PROTOCOL_VERSION,
                    )
                except WorkerProtocolError:
                    accepted = False
                    reason_code = "protocol_v3_required"
            elif message.version != IMAGE_PROTOCOL_VERSION:
                reason_code = "protocol_v3_required"
            else:
                reason_code = "node_identity_mismatch"
            now_ms = int(self._clock() * 1000) if sent_at_ms is None else int(sent_at_ms)
            ack = build_message(
                "hello_ack",
                {
                    "coordinator_node_id": coordinator_node_id,
                    "accepted": accepted,
                    "selected_version": selected_version,
                    "reason_code": reason_code,
                },
                message_id=_message_id("diffhelloack_"),
                sent_at_ms=now_ms,
                version=IMAGE_PROTOCOL_VERSION if accepted else message.version,
            )
            now = self._clock()
            self._workers[peer_id] = {
                "node_id": peer_id,
                "connected": True,
                "accepted": accepted,
                "selected_version": selected_version,
                "capabilities": payload["capabilities"],
                "last_transport_heartbeat": now,
                "reason_code": reason_code,
            }
            if not accepted:
                self._rejected += 1
            self._remember(peer_id, message, ack)
            return ack

    def mark_heartbeat(self, peer_id: str) -> None:
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is not None:
                worker["last_transport_heartbeat"] = self._clock()

    def disconnect_worker(self, peer_id: str) -> None:
        """Mark a transport peer unavailable without discarding its snapshot."""
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is not None:
                worker["connected"] = False

    def record_rejection(self) -> None:
        """Record a Scheduler-level rejection before control-plane parsing."""
        with self._lock:
            self._rejected += 1

    def worker_snapshot(self, peer_id: str) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is None:
                return {}
            snapshot = dict(worker)
            last_seen = float(snapshot.get("last_transport_heartbeat", 0.0) or 0.0)
            snapshot["healthy"] = bool(
                snapshot.get("connected")
                and snapshot.get("accepted")
                and self._clock() - last_seen <= self._health_timeout_seconds
            )
            return snapshot

    def status(self) -> dict[str, Any]:
        with self._lock:
            workers = [self.worker_snapshot(node_id) for node_id in sorted(self._workers)]
            return {
                "protocol_version": IMAGE_PROTOCOL_VERSION,
                "control_plane_ready": True,
                "control_plane_connected": any(item.get("healthy") for item in workers),
                "adapter_connected": False,
                "task_dispatch_enabled": False,
                "workers": workers,
                "rejected_message_count": self._rejected,
            }


class DiffusionWorkerAdapter:
    """V3 image worker that accepts at most one active offer at a time."""

    def __init__(
        self,
        *,
        node_id: str,
        capabilities: Mapping[str, Any],
        executor: DiffusionExecutor,
        send_message: SendMessage,
        clock=time.time,
        monotonic=time.monotonic,
    ) -> None:
        self.node_id = str(node_id)
        self._capabilities = dict(capabilities)
        self._executor = executor
        self._send_message = send_message
        self._clock = clock
        self._monotonic = monotonic
        self._hello_pending = False
        self._coordinator: dict[str, Any] = {}
        self._active: _ActiveStage | None = None
        self._seen: dict[str, tuple[str, WorkerMessage | None]] = {}
        self._seen_order: collections.deque[str] = collections.deque()
        self._cancel_seen: dict[str, tuple[str, WorkerMessage]] = {}
        self._cancel_seen_order: collections.deque[str] = collections.deque()
        self._renew_seen: dict[str, str] = {}
        self._renew_seen_order: collections.deque[str] = collections.deque()
        self._rejected = 0
        self._lock = threading.RLock()

    @staticmethod
    def _reject(code: str, message: str) -> WorkerProtocolError:
        return WorkerProtocolError(message, code=code, field="message_type")

    def _remember(self, message: WorkerMessage, terminal: WorkerMessage | None = None) -> None:
        self._seen[message.message_id] = (_message_digest(message), terminal)
        self._seen_order.append(message.message_id)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen.pop(expired, None)

    def _remember_cancel(self, message: WorkerMessage, response: WorkerMessage) -> None:
        self._cancel_seen[message.message_id] = (_message_digest(message), response)
        self._cancel_seen_order.append(message.message_id)
        while len(self._cancel_seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._cancel_seen_order.popleft()
            self._cancel_seen.pop(expired, None)

    def _remember_renew(self, message: WorkerMessage) -> None:
        self._renew_seen[message.message_id] = _message_digest(message)
        self._renew_seen_order.append(message.message_id)
        while len(self._renew_seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._renew_seen_order.popleft()
            self._renew_seen.pop(expired, None)

    def begin_hello(self, *, sent_at_ms: int | None = None) -> WorkerMessage | None:
        with self._lock:
            if self._hello_pending:
                return None
            now_ms = int(self._clock() * 1000) if sent_at_ms is None else int(sent_at_ms)
            hello = build_message(
                "hello",
                {
                    "node_id": self.node_id,
                    "worker_kind": "pc_diffusion_worker",
                    "min_version": IMAGE_PROTOCOL_VERSION,
                    "max_version": IMAGE_PROTOCOL_VERSION,
                    "capabilities": self._capabilities,
                },
                message_id=_message_id("diffhello_"),
                sent_at_ms=now_ms,
                version=IMAGE_PROTOCOL_VERSION,
            )
            self._hello_pending = True
            return hello

    def receive_hello_ack(self, raw: bytes | str | Mapping[str, Any]) -> WorkerMessage:
        message = decode_message(raw)
        with self._lock:
            if message.message_type != "hello_ack":
                self._rejected += 1
                raise self._reject("control_plane_only", "expected a v3 hello acknowledgement")
            if not self._hello_pending:
                self._rejected += 1
                raise self._reject("unexpected_hello_ack", "hello_ack arrived without a pending hello")
            if message.version != IMAGE_PROTOCOL_VERSION or (
                message.payload["accepted"]
                and message.payload["selected_version"] != IMAGE_PROTOCOL_VERSION
            ):
                self._rejected += 1
                raise self._reject("protocol_v3_required", "diffusion worker requires protocol v3")
            self._hello_pending = False
            self._coordinator = {
                "node_id": message.payload["coordinator_node_id"],
                "connected": True,
                "accepted": bool(message.payload["accepted"]),
                "reason_code": message.payload["reason_code"],
                "selected_version": message.payload["selected_version"],
            }
            return message

    def disconnect_coordinator(self) -> None:
        """Fence local work after the authenticated coordinator transport drops."""
        with self._lock:
            self._hello_pending = False
            self._coordinator = {
                **self._coordinator,
                "connected": False,
                "accepted": False,
            }
            if self._active is not None:
                self._active.cancel_event.set()

    def _send(self, message: WorkerMessage) -> None:
        try:
            self._send_message(message)
        except Exception:
            with self._lock:
                self._rejected += 1

    def _stage_accept(self, offer: WorkerMessage, *, accepted: bool, reason_code: str) -> WorkerMessage:
        return build_message(
            "stage_accept",
            {
                **_identity(offer.payload),
                "accepted": accepted,
                "reason_code": reason_code,
                "retryable": not accepted,
            },
            message_id=_message_id("diffaccept_"),
            sent_at_ms=int(self._clock() * 1000),
            version=IMAGE_PROTOCOL_VERSION,
        )

    def receive_offer(self, raw: bytes | str | Mapping[str, Any]) -> WorkerMessage:
        offer = decode_message(raw)
        with self._lock:
            if offer.message_type != "stage_offer":
                self._rejected += 1
                raise self._reject("invalid_message_direction", "expected a stage_offer")
            if not self._coordinator.get("accepted"):
                self._rejected += 1
                raise self._reject("control_plane_not_ready", "diffusion hello was not accepted")
            if offer.payload["provider_id"] != remote_diffusion_provider_id(self.node_id):
                rejected = self._stage_accept(
                    offer, accepted=False, reason_code="provider_identity_mismatch",
                )
                self._remember(offer, rejected)
                self._send(rejected)
                return rejected
            cached = self._seen.get(offer.message_id)
            if cached is not None:
                if cached[0] != _message_digest(offer):
                    self._rejected += 1
                    raise self._reject("message_id_conflict", "message_id was reused with different content")
                if cached[1] is not None:
                    self._send(cached[1])
                    return cached[1]
                if self._active is not None and self._active.offer.message_id == offer.message_id:
                    accepted = self._active.accepted
                    if accepted is not None:
                        self._send(accepted)
                        return accepted
            if self._active is not None:
                rejected = self._stage_accept(offer, accepted=False, reason_code="worker_busy")
                self._remember(offer, rejected)
                self._send(rejected)
                return rejected
            if offer.payload["lease_expires_at_ms"] <= offer.sent_at_ms:
                rejected = self._stage_accept(offer, accepted=False, reason_code="lease_expired")
                self._remember(offer, rejected)
                self._send(rejected)
                return rejected
            accepted = self._stage_accept(offer, accepted=True, reason_code="")
            active = _ActiveStage(
                offer=offer,
                lease_expires_at_ms=int(offer.payload["lease_expires_at_ms"]),
                lease_deadline_monotonic=self._monotonic() + max(
                    0.0,
                    (int(offer.payload["lease_expires_at_ms"]) - offer.sent_at_ms) / 1000.0,
                ),
                accepted=accepted,
            )
            self._active = active
            self._remember(offer)
            self._send(accepted)
            threading.Thread(
                target=self._watch_lease,
                args=(active,),
                name=f"diffusion-worker-lease-{self.node_id}",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._execute_stage,
                args=(active,),
                name=f"diffusion-worker-{self.node_id}",
                daemon=True,
            ).start()
            return accepted

    def _watch_lease(self, active: _ActiveStage) -> None:
        while not active.finished.is_set():
            with self._lock:
                if active.terminal is not None:
                    return
                remaining_seconds = active.lease_deadline_monotonic - self._monotonic()
                if remaining_seconds <= 0:
                    active.lease_expired = True
                    active.cancel_event.set()
                    return
            active.finished.wait(min(0.05, remaining_seconds))

    def _execute_stage(self, active: _ActiveStage) -> None:
        offer = active.offer
        try:
            execution = self._executor(offer.payload, active.cancel_event)
            if not isinstance(execution, DiffusionExecutionResult):
                raise TypeError("diffusion executor returned an invalid result")
            with self._lock:
                lease_expired = active.lease_expired or (
                    active.lease_deadline_monotonic <= self._monotonic()
                )
            if lease_expired:
                terminal = build_message(
                    "stage_error",
                    {
                        **_identity(offer.payload),
                        "error_code": "lease_expired",
                        "retryable": True,
                    },
                    message_id=_message_id("differror_"),
                    sent_at_ms=int(self._clock() * 1000),
                    version=IMAGE_PROTOCOL_VERSION,
                )
            elif active.cancel_event.is_set():
                terminal = self._cancelled(offer, "cancelled")
            else:
                terminal = build_message(
                    "stage_result",
                    {
                        **_identity(offer.payload),
                        "output": execution.output,
                        "output_sha256": stage_output_sha256(
                            execution.output,
                            execution.transfer_plan,
                        ),
                        "metadata": execution.metadata,
                        "transfer_plan": execution.transfer_plan,
                    },
                    message_id=_message_id("diffresult_"),
                    sent_at_ms=int(self._clock() * 1000),
                    version=IMAGE_PROTOCOL_VERSION,
                )
        except Exception:
            terminal = build_message(
                "stage_error",
                {
                    **_identity(offer.payload),
                    "error_code": "diffusion_execution_failed",
                    "retryable": True,
                },
                message_id=_message_id("differror_"),
                sent_at_ms=int(self._clock() * 1000),
                version=IMAGE_PROTOCOL_VERSION,
            )
        with self._lock:
            # A cancel acknowledgement may have fenced this execution while
            # the executor was still unwinding.  Its terminal message wins.
            if active.terminal is None:
                active.terminal = terminal
                active.finished.set()
                self._seen[offer.message_id] = (_message_digest(offer), terminal)
                if self._active is active:
                    self._active = None
                outbound = terminal
            else:
                active.finished.set()
                if self._active is active:
                    self._active = None
                outbound = None
        if outbound is not None:
            self._send(outbound)

    def _cancelled(self, offer: WorkerMessage, reason_code: str) -> WorkerMessage:
        return build_message(
            "stage_cancelled",
            {**_identity(offer.payload), "reason_code": reason_code},
            message_id=_message_id("diffcancelled_"),
            sent_at_ms=int(self._clock() * 1000),
            version=IMAGE_PROTOCOL_VERSION,
        )

    def receive_lease_renew(self, raw: bytes | str | Mapping[str, Any]) -> None:
        message = decode_message(raw)
        with self._lock:
            if message.message_type != "lease_renew":
                self._rejected += 1
                raise self._reject("invalid_message_direction", "expected a lease_renew")
            digest = self._renew_seen.get(message.message_id)
            if digest is not None:
                if digest != _message_digest(message):
                    self._rejected += 1
                    raise self._reject(
                        "message_id_conflict",
                        "message_id was reused with different content",
                    )
                return
            active = self._active
            if active is None or (
                _attempt_identity(active.offer.payload)
                != _attempt_identity(message.payload)
            ):
                self._rejected += 1
                raise self._reject("unknown_attempt", "lease_renew has no active diffusion attempt")
            if active.terminal is not None or active.lease_expired:
                self._rejected += 1
                raise self._reject("lease_expired", "a terminal diffusion lease cannot be renewed")
            deadline = int(message.payload["lease_expires_at_ms"])
            if deadline <= active.lease_expires_at_ms:
                self._rejected += 1
                raise self._reject("stale_lease", "lease_renew must extend the active deadline")
            if deadline <= message.sent_at_ms:
                self._rejected += 1
                raise self._reject("lease_expired", "renewed diffusion lease is already expired")
            active.lease_expires_at_ms = deadline
            active.lease_deadline_monotonic = self._monotonic() + (
                deadline - message.sent_at_ms
            ) / 1000.0
            self._remember_renew(message)

    def receive_cancel(self, raw: bytes | str | Mapping[str, Any]) -> WorkerMessage:
        message = decode_message(raw)
        with self._lock:
            if message.message_type != "stage_cancel":
                self._rejected += 1
                raise self._reject("invalid_message_direction", "expected a stage_cancel")
            cached = self._cancel_seen.get(message.message_id)
            if cached is not None:
                if cached[0] != _message_digest(message):
                    self._rejected += 1
                    raise self._reject(
                        "message_id_conflict",
                        "message_id was reused with different content",
                    )
                self._send(cached[1])
                return cached[1]
            active = self._active
            if active is None or (
                _attempt_identity(active.offer.payload)
                != _attempt_identity(message.payload)
            ):
                self._rejected += 1
                raise self._reject("unknown_attempt", "stage_cancel has no active diffusion attempt")
            if active.terminal is not None:
                # The first cancellation fences the attempt.  A later,
                # differently identified cancellation is still idempotent,
                # but cannot replace its durable terminal reason.
                self._remember_cancel(message, active.terminal)
                self._send(active.terminal)
                return active.terminal
            active.cancel_event.set()
            cancelled = self._cancelled(active.offer, message.payload["reason_code"])
            active.terminal = cancelled
            self._seen[active.offer.message_id] = (_message_digest(active.offer), cancelled)
            self._remember_cancel(message, cancelled)
        self._send(cancelled)
        return cancelled

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "protocol_version": IMAGE_PROTOCOL_VERSION,
                "hello_pending": self._hello_pending,
                "control_plane_connected": bool(
                    self._coordinator.get("connected")
                    and self._coordinator.get("accepted")
                ),
                "adapter_connected": False,
                "active_stage": self._active.offer.payload["attempt_id"] if self._active else "",
                "rejected_message_count": self._rejected,
            }


class RemoteDiffusionProvider:
    """V3 image Provider that never persists output transfer grants.

    The result ingestor consumes the worker-owned output transfer plan before a
    generic ``StageResult`` is returned.  This keeps short-lived Bearer grants
    out of TaskGraph metadata and its journal.
    """

    provider_kind = "remote_diffusion_worker"

    def __init__(
        self,
        *,
        node_id: str,
        peer_snapshot: Callable[[], Mapping[str, Any]],
        send_message: SendMessage,
        result_ingestor: DiffusionResultIngestor | None,
        input_transfer_plan_builder: DiffusionInputTransferPlanBuilder | None = None,
        dispatch_enabled: bool = False,
        accept_timeout_seconds: float = 10.0,
    ) -> None:
        self.node_id = str(node_id)
        self.provider_id = remote_diffusion_provider_id(self.node_id)
        self._peer_snapshot = peer_snapshot
        self._send_message = send_message
        self._result_ingestor = result_ingestor
        self._input_transfer_plan_builder = input_transfer_plan_builder
        self._dispatch_enabled = bool(dispatch_enabled)
        self._accept_timeout_seconds = max(
            0.1, min(float(accept_timeout_seconds), 60.0),
        )
        self._reservations: dict[str, tuple[Reservation, StageRequest]] = {}
        self._executed_reservations: set[str] = set()
        self._reservation_attempts: dict[str, str] = {}
        self._pending: dict[str, _PendingDiffusionAttempt] = {}
        self._seen_messages: dict[str, str] = {}
        self._seen_order: collections.deque[str] = collections.deque()
        self._closed = False
        self._lock = threading.RLock()

    def _snapshot(self) -> dict[str, Any]:
        try:
            return dict(self._peer_snapshot() or {})
        except Exception:
            return {}

    @staticmethod
    def _image_capabilities(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        capabilities = snapshot.get("capabilities", {})
        return dict(capabilities) if isinstance(capabilities, Mapping) else {}

    @staticmethod
    def _manifest_supported(manifest: Any, capabilities: Mapping[str, Any]) -> bool:
        if not isinstance(manifest, Mapping):
            return False
        image = capabilities.get("image", {})
        if not isinstance(image, Mapping):
            return False
        manifests = image.get("artifact_manifests", [])
        if not isinstance(manifests, list):
            return False
        digest = manifest.get("sha256")
        return isinstance(digest, str) and any(
            isinstance(candidate, Mapping) and candidate.get("sha256") == digest
            for candidate in manifests
        )

    def _healthy(self, snapshot: Mapping[str, Any]) -> bool:
        return bool(
            not self._closed
            and self._dispatch_enabled
            and self._result_ingestor is not None
            and snapshot.get("healthy")
            and snapshot.get("selected_version") == IMAGE_PROTOCOL_VERSION
        )

    def _prune_pending_locked(self) -> None:
        cutoff = time.time() - 5.0
        expired = [
            attempt_id
            for attempt_id, pending in self._pending.items()
            if pending.released and pending.released_at <= cutoff
        ]
        for attempt_id in expired:
            self._pending.pop(attempt_id, None)

    def _is_duplicate_message_locked(self, message: WorkerMessage) -> bool:
        digest = _message_digest(message)
        previous = self._seen_messages.get(message.message_id)
        if previous is not None:
            if previous != digest:
                raise WorkerProtocolError(
                    "message_id was reused with different Stage content",
                    code="message_id_conflict",
                field="message_id",
            )
            return True

        return False

    def _remember_message_locked(self, message: WorkerMessage) -> None:
        digest = _message_digest(message)
        self._seen_messages[message.message_id] = digest
        self._seen_order.append(message.message_id)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen_messages.pop(expired, None)

    def inspect(self) -> ProviderCapabilities:
        snapshot = self._snapshot()
        capabilities = self._image_capabilities(snapshot)
        stage_types = capabilities.get("stage_types", [])
        if not isinstance(stage_types, list):
            stage_types = []
        with self._lock:
            self._prune_pending_locked()
            active = len(self._reservations)
            healthy = self._healthy(snapshot)
        supported = tuple(
            item
            for item in stage_types
            if item in {"image_generate", "image_edit", "image_grid"}
        )
        return ProviderCapabilities(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            supported_stage_types=supported,
            max_concurrency=1,
            active_reservations=active,
            healthy=healthy,
            available=healthy and active < 1,
            node_id=self.node_id,
        )

    def _request_manifest(self, request: StageRequest) -> Any:
        return request.runtime_context.get("diffusion_artifact_manifest")

    def artifact_manifests(self) -> tuple[dict[str, Any], ...]:
        """Return immutable copies of the exact artifacts advertised by the peer."""
        capabilities = self._image_capabilities(self._snapshot())
        image = capabilities.get("image", {})
        manifests = image.get("artifact_manifests", []) if isinstance(image, Mapping) else []
        if not isinstance(manifests, list):
            return ()
        return tuple(
            copy.deepcopy(dict(manifest))
            for manifest in manifests
            if isinstance(manifest, Mapping)
        )

    def reserve(self, request: StageRequest) -> Reservation:
        status = self.inspect()
        capabilities = self._image_capabilities(self._snapshot())
        if request.provider_id != self.provider_id:
            raise ProviderReservationError(
                "stage request targets a different remote diffusion provider",
                code="provider_request_mismatch",
                provider_id=self.provider_id,
            )
        if DEPENDENCY_FAILURES_KEY in request.dependencies or request.dependencies:
            raise ProviderUnavailable(
                "v3 image workers require Blob descriptors in root_input",
                code="image_dependencies_not_supported",
                provider_id=self.provider_id,
                retryable=True,
            )
        if request.stage_type not in status.supported_stage_types:
            raise ProviderUnavailable(
                "remote worker does not support the requested image stage",
                code="unsupported_stage_type",
                provider_id=self.provider_id,
            )
        manifest = self._request_manifest(request)
        if request.stage_type == "image_grid":
            if manifest is not None:
                raise ProviderUnavailable(
                    "image_grid must not carry a diffusion artifact manifest",
                    code="invalid_artifact_manifest",
                    provider_id=self.provider_id,
                )
        elif not self._manifest_supported(manifest, capabilities):
            raise ProviderUnavailable(
                "remote worker does not advertise the requested diffusion artifact",
                code="artifact_manifest_mismatch",
                provider_id=self.provider_id,
                retryable=True,
            )
        with self._lock:
            self._prune_pending_locked()
            if self._closed or not status.healthy:
                raise ProviderUnavailable(
                    "remote diffusion worker is not dispatchable",
                    code="remote_diffusion_unavailable",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            if len(self._reservations) >= 1:
                raise ProviderBusy(
                    "remote diffusion worker has no free Stage slot",
                    code="remote_diffusion_busy",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            reservation = Reservation(
                reservation_id=f"res_{uuid.uuid4().hex}",
                provider_id=self.provider_id,
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                created_at=time.time(),
                selection_reason="explicit_remote_diffusion_provider",
                provider_kind=self.provider_kind,
                provider_node_id=self.node_id,
            )
            self._reservations[reservation.reservation_id] = (reservation, request)
            return reservation

    @staticmethod
    def _identity_matches(payload: Mapping[str, Any], attempt: StageAttempt) -> bool:
        return all((
            payload.get("workflow_id") == attempt.request.workflow_id,
            payload.get("stage_id") == attempt.request.stage_id,
            payload.get("attempt_id") == attempt.attempt_id,
            payload.get("lease_id") == attempt.lease_id,
            payload.get("lease_epoch") == attempt.lease_epoch,
            payload.get("provider_id") == attempt.provider_id,
        ))

    @staticmethod
    def _wait(
        event: threading.Event,
        pending: _PendingDiffusionAttempt,
        cancel_event: threading.Event,
        deadline: float | Callable[[], float],
        *,
        timeout_code: str,
        provider_id: str,
    ) -> None:
        while not event.wait(0.05):
            if cancel_event.is_set():
                raise ProviderExecutionError(
                    "remote diffusion Stage wait was cancelled locally",
                    code="provider_cancelled",
                    provider_id=provider_id,
                )
            current_deadline = deadline() if callable(deadline) else deadline
            if time.time() >= current_deadline:
                raise ProviderExecutionError(
                    "remote diffusion Stage response timed out",
                    code=timeout_code,
                    provider_id=provider_id,
                    retryable=True,
                )
        if pending.error is not None:
            raise pending.error

    def _input_transfer_plan(
        self,
        attempt: StageAttempt,
        request: StageRequest,
    ) -> dict[str, Any]:
        if self._input_transfer_plan_builder is not None:
            return dict(self._input_transfer_plan_builder(attempt, request))
        return {"base_url": None, "downloads": []}

    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult:
        with self._lock:
            owned = self._reservations.get(reservation.reservation_id)
            if owned is None or owned[0] != reservation:
                raise ProviderReservationError(
                    "remote diffusion reservation is unknown",
                    code="invalid_reservation",
                    provider_id=self.provider_id,
                )
            if (
                owned[1] != attempt.request
                or attempt.provider_id != self.provider_id
                or reservation.provider_id != self.provider_id
            ):
                raise ProviderReservationError(
                    "remote diffusion attempt does not match its reservation",
                    code="attempt_reservation_mismatch",
                    provider_id=self.provider_id,
                )
            if reservation.reservation_id in self._executed_reservations:
                raise ProviderReservationError(
                    "remote diffusion reservation has already been executed",
                    code="reservation_already_executed",
                    provider_id=self.provider_id,
                )
            if self._result_ingestor is None:
                raise ProviderExecutionError(
                    "remote diffusion output ingestion is not configured",
                    code="result_transfer_unavailable",
                    provider_id=self.provider_id,
                )
            pending = _PendingDiffusionAttempt(
                attempt=attempt,
                lease_expires_at=attempt.lease_expires_at,
            )
            self._pending[attempt.attempt_id] = pending
            self._reservation_attempts[reservation.reservation_id] = attempt.attempt_id
            self._executed_reservations.add(reservation.reservation_id)
        try:
            transfer_plan = self._input_transfer_plan(attempt, attempt.request)
            offer = build_message(
                "stage_offer",
                {
                    "workflow_id": attempt.request.workflow_id,
                    "request_id": attempt.request.request_id,
                    "stage_id": attempt.request.stage_id,
                    "stage_type": attempt.request.stage_type,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at_ms": int(attempt.lease_expires_at * 1000),
                    "provider_id": self.provider_id,
                    "root_input": attempt.request.root_input,
                    "dependencies": attempt.request.dependencies,
                    "input_sha256": stage_input_sha256(
                        attempt.request.root_input,
                        attempt.request.dependencies,
                        transfer_plan,
                    ),
                    "artifact_manifest": self._request_manifest(attempt.request),
                    "transfer_plan": transfer_plan,
                },
                message_id=_message_id("diffoffer_"),
                sent_at_ms=int(time.time() * 1000),
                version=IMAGE_PROTOCOL_VERSION,
            )
        except (TypeError, ValueError, WorkerProtocolError) as exc:
            with self._lock:
                pending.error = ProviderExecutionError(
                    "remote diffusion Stage input is not transportable",
                    code="invalid_diffusion_stage_input",
                    provider_id=self.provider_id,
                )
                pending.accept_event.set()
                pending.result_event.set()
            raise pending.error from exc
        try:
            self._send_message(offer)
        except Exception as exc:
            error = ProviderExecutionError(
                "failed to send Stage offer to the remote diffusion worker",
                code="remote_diffusion_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
            with self._lock:
                pending.error = error
                pending.accept_event.set()
                pending.result_event.set()
            raise error from exc
        accept_deadline = min(
            attempt.lease_expires_at,
            time.time() + min(
                self._accept_timeout_seconds,
                max(0.001, float(attempt.accept_timeout_seconds)),
            ),
        )
        try:
            self._wait(
                pending.accept_event,
                pending,
                cancel_event,
                accept_deadline,
                timeout_code="remote_accept_timeout",
                provider_id=self.provider_id,
            )
        except ProviderExecutionError as exc:
            if exc.code in {"remote_accept_timeout", "provider_cancelled"}:
                self.cancel(attempt.attempt_id)
            raise
        if not pending.accepted:
            raise ProviderExecutionError(
                "remote diffusion worker did not accept the Stage",
                code="remote_stage_not_accepted",
                provider_id=self.provider_id,
            )
        try:
            self._wait(
                pending.result_event,
                pending,
                cancel_event,
                lambda: pending.lease_expires_at,
                timeout_code="lease_expired",
                provider_id=self.provider_id,
            )
        except ProviderExecutionError as exc:
            if exc.code in {"lease_expired", "provider_cancelled"}:
                self.cancel(attempt.attempt_id)
            raise
        if (
            pending.output is None
            or pending.output_transfer_plan is None
            or pending.metadata is None
        ):
            raise ProviderExecutionError(
                "remote diffusion worker returned no Stage result",
                code="invalid_provider_result",
                provider_id=self.provider_id,
            )
        try:
            safe_output = self._result_ingestor(
                attempt, pending.output, pending.output_transfer_plan,
            )
        except Exception as exc:
            raise ProviderExecutionError(
                "remote diffusion output transfer could not be verified",
                code="result_transfer_failed",
                provider_id=self.provider_id,
                retryable=True,
            ) from exc
        if cancel_event.is_set():
            self.cancel(attempt.attempt_id)
            raise ProviderExecutionError(
                "remote diffusion output arrived after local cancellation",
                code="provider_cancelled",
                provider_id=self.provider_id,
            )
        return StageResult(
            output=dict(safe_output),
            provider_id=self.provider_id,
            metadata=dict(pending.metadata),
            attempt_id=attempt.attempt_id,
            lease_epoch=attempt.lease_epoch,
        )

    def handle_message(self, raw: bytes | str | Mapping[str, Any]) -> WorkerMessage:
        message = decode_message(raw)
        if message.version != IMAGE_PROTOCOL_VERSION or message.message_type not in {
            "stage_accept", "stage_result", "stage_error", "stage_cancelled",
        }:
            raise WorkerProtocolError(
                "message is not a v3 coordinator-side image Stage response",
                code="invalid_message_direction",
                field="message_type",
            )
        payload = message.payload
        attempt_id = str(payload.get("attempt_id", ""))
        with self._lock:
            self._prune_pending_locked()
            if self._is_duplicate_message_locked(message):
                return message
            pending = self._pending.get(attempt_id)
            if pending is None:
                raise WorkerProtocolError(
                    "image Stage response has no pending attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._identity_matches(payload, pending.attempt):
                raise WorkerProtocolError(
                    "image Stage response identity does not match the pending attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            if message.message_type == "stage_accept":
                if pending.accept_event.is_set():
                    raise WorkerProtocolError(
                        "image Stage acceptance was already recorded",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                if payload["accepted"]:
                    pending.accepted = True
                else:
                    pending.error = ProviderReservationError(
                        "remote diffusion worker rejected the Stage offer",
                        code=payload["reason_code"],
                        provider_id=self.provider_id,
                        retryable=bool(payload["retryable"]),
                    )
                pending.accept_event.set()
            elif message.message_type == "stage_result":
                if not pending.accepted:
                    raise WorkerProtocolError(
                        "image Stage result arrived before acceptance",
                        code="result_before_accept",
                        field="message_type",
                    )
                if pending.result_event.is_set():
                    raise WorkerProtocolError(
                        "image Stage already has a terminal response",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                pending.output = dict(payload["output"])
                pending.output_transfer_plan = dict(payload["transfer_plan"])
                pending.metadata = dict(payload["metadata"])
                pending.result_event.set()
            elif message.message_type == "stage_error":
                if pending.result_event.is_set():
                    raise WorkerProtocolError(
                        "image Stage already has a terminal response",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                pending.error = ProviderExecutionError(
                    "remote diffusion worker reported a Stage error",
                    code=payload["error_code"],
                    provider_id=self.provider_id,
                    retryable=bool(payload["retryable"]),
                )
                pending.accept_event.set()
                pending.result_event.set()
            else:
                if not pending.cancel_requested:
                    raise WorkerProtocolError(
                        "image Stage cancellation acknowledgement was not requested",
                        code="unexpected_stage_cancelled",
                        field="message_type",
                    )
                pending.cancel_acknowledged = True
                pending.cancel_ack_event.set()
                if pending.released:
                    self._pending.pop(attempt_id, None)
            self._remember_message_locked(message)
        return message

    def renew_lease(
        self,
        attempt_id: str,
        lease_id: str,
        lease_epoch: int,
        lease_expires_at: float,
    ) -> bool:
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is None:
                raise ProviderExecutionError(
                    "remote diffusion lease renewal has no pending attempt",
                    code="unknown_attempt",
                    provider_id=self.provider_id,
                )
            attempt = pending.attempt
            deadline = float(lease_expires_at)
            if (
                attempt.lease_id != lease_id
                or attempt.lease_epoch != int(lease_epoch)
                or deadline <= pending.lease_expires_at
            ):
                raise ProviderExecutionError(
                    "remote diffusion lease renewal identity is stale",
                    code="stale_lease",
                    provider_id=self.provider_id,
                )
            message = build_message(
                "lease_renew",
                {
                    "workflow_id": attempt.request.workflow_id,
                    "stage_id": attempt.request.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at_ms": int(deadline * 1000),
                },
                message_id=_message_id("diffrenew_"),
                sent_at_ms=int(time.time() * 1000),
                version=IMAGE_PROTOCOL_VERSION,
            )
            pending.lease_expires_at = deadline
        try:
            self._send_message(message)
        except Exception as exc:
            error = ProviderExecutionError(
                "failed to renew the remote diffusion Stage lease",
                code="remote_diffusion_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.error = error
                    current.accept_event.set()
                    current.result_event.set()
            raise error from exc
        return True

    def cancel(self, attempt_id: str) -> None:
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is None or pending.cancel_requested:
                return
            pending.cancel_requested = True
            pending.error = ProviderExecutionError(
                "remote diffusion Stage was cancelled locally",
                code="provider_cancelled",
                provider_id=self.provider_id,
            )
            pending.accept_event.set()
            pending.result_event.set()
            attempt = pending.attempt
            message = build_message(
                "stage_cancel",
                {
                    "workflow_id": attempt.request.workflow_id,
                    "stage_id": attempt.request.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "reason_code": "coordinator_cancelled",
                },
                message_id=_message_id("diffcancel_"),
                sent_at_ms=int(time.time() * 1000),
                version=IMAGE_PROTOCOL_VERSION,
            )
        try:
            self._send_message(message)
        except Exception:
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.cancel_acknowledged = True
                    current.cancel_ack_event.set()
                    if current.released:
                        self._pending.pop(attempt_id, None)

    def notify_disconnect(self) -> None:
        with self._lock:
            for attempt_id, pending in list(self._pending.items()):
                pending.error = ProviderExecutionError(
                    "remote diffusion worker disconnected",
                    code="remote_diffusion_disconnected",
                    provider_id=self.provider_id,
                    retryable=True,
                )
                pending.accept_event.set()
                pending.result_event.set()
                pending.cancel_ack_event.set()
                if pending.released:
                    self._pending.pop(attempt_id, None)

    def release(self, reservation_id: str) -> None:
        with self._lock:
            self._reservations.pop(reservation_id, None)
            self._executed_reservations.discard(reservation_id)
            attempt_id = self._reservation_attempts.pop(reservation_id, "")
            pending = self._pending.get(attempt_id)
            if pending is not None:
                pending.released = True
                pending.released_at = time.time()
                if not pending.cancel_requested or pending.cancel_acknowledged:
                    self._pending.pop(attempt_id, None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.notify_disconnect()
        with self._lock:
            self._reservations.clear()
            self._executed_reservations.clear()
            self._reservation_attempts.clear()
            self._pending.clear()


__all__ = [
    "DiffusionCoordinatorControlPlane",
    "DiffusionExecutionResult",
    "DiffusionInputTransferPlanBuilder",
    "DiffusionResultIngestor",
    "DiffusionWorkerAdapter",
    "IMAGE_PROTOCOL_VERSION",
    "RemoteDiffusionProvider",
    "remote_diffusion_provider_id",
]
