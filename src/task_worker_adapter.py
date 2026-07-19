"""TC-N2.4 task-worker control plane with physical admission pending."""

from __future__ import annotations

import collections
import hashlib
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from task_provider import (
    ModelIdentity,
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
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    WorkerMessage,
    WorkerProtocolError,
    build_message,
    canonical_message_bytes,
    decode_message,
    negotiate_protocol_version,
    stage_input_sha256,
)


_MESSAGE_CACHE_LIMIT = 1024


def remote_provider_id(node_id: str) -> str:
    """Return a stable Provider ID for one authenticated worker node."""
    raw = str(node_id or "")
    safe = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "_.-")
        else "_"
        for character in raw
    ).strip("._") or "worker"
    base = f"remote_{safe}"
    if len(base) <= 64 and safe == raw:
        return base
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"remote_{safe[:46]}_{digest}"[:64]


def _message_id(prefix: str) -> str:
    return f"msg_{prefix}{uuid.uuid4().hex}"


def _message_digest(message: WorkerMessage) -> str:
    return hashlib.sha256(canonical_message_bytes(message)).hexdigest()


class TaskWorkerControlPlane:
    """Own hello negotiation and health state without accepting Stage work."""

    def __init__(self, *, health_timeout_seconds: float = 120.0):
        self._health_timeout_seconds = max(1.0, float(health_timeout_seconds))
        self._workers: dict[str, dict[str, Any]] = {}
        self._coordinator: dict[str, Any] = {}
        self._worker_hello_pending = False
        self._seen: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self._seen_order: collections.deque[tuple[str, str]] = collections.deque()
        self._rejected_message_count = 0
        self._lock = threading.RLock()

    @staticmethod
    def _reject(code: str, message: str) -> WorkerProtocolError:
        return WorkerProtocolError(message, code=code, field="message_type")

    def _remember(
        self,
        peer_id: str,
        message: WorkerMessage,
        response: Optional[WorkerMessage] = None,
    ) -> None:
        key = (peer_id, message.message_id)
        response_snapshot = response.snapshot() if response is not None else {}
        self._seen[key] = (_message_digest(message), response_snapshot)
        self._seen_order.append(key)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen.pop(expired, None)

    def _duplicate_response(
        self, peer_id: str, message: WorkerMessage,
    ) -> Optional[WorkerMessage]:
        cached = self._seen.get((peer_id, message.message_id))
        if cached is None:
            return None
        digest, response = cached
        if digest != _message_digest(message):
            self._rejected_message_count += 1
            raise self._reject(
                "message_id_conflict",
                "message_id was reused with different content",
            )
        if not response:
            return message
        return decode_message(response)

    def begin_worker_hello(
        self,
        *,
        node_id: str,
        capabilities: Mapping[str, Any],
        sent_at_ms: Optional[int] = None,
    ) -> Optional[WorkerMessage]:
        """Build one v2 hello; overlapping negotiations are deliberately fenced."""
        now_ms = int(time.time() * 1000) if sent_at_ms is None else int(sent_at_ms)
        with self._lock:
            if self._worker_hello_pending:
                return None
            message = build_message(
                "hello",
                {
                    "node_id": node_id,
                    "worker_kind": "pc_full_worker",
                    "min_version": PROTOCOL_VERSION,
                    "max_version": PROTOCOL_VERSION,
                    "capabilities": dict(capabilities),
                },
                message_id=_message_id("hello_"),
                sent_at_ms=now_ms,
                version=PROTOCOL_VERSION,
            )
            self._worker_hello_pending = True
            self._coordinator = {
                "node_id": "",
                "connected": True,
                "healthy": False,
                "accepted": False,
                "selected_version": 0,
                "hello_sent_at": now_ms / 1000.0,
                "last_transport_heartbeat": time.time(),
                "reason_code": "negotiation_pending",
            }
            return message

    def receive_on_coordinator(
        self,
        peer_id: str,
        raw: bytes | str | Mapping[str, Any],
        *,
        coordinator_node_id: str,
        sent_at_ms: Optional[int] = None,
    ) -> WorkerMessage:
        """Validate a worker hello and return a deterministic hello_ack."""
        message = decode_message(raw)
        with self._lock:
            duplicate = self._duplicate_response(peer_id, message)
            if duplicate is not None:
                return duplicate
            if message.message_type != "hello":
                self._rejected_message_count += 1
                raise self._reject(
                    "control_plane_only",
                    "the coordinator hello handler accepts hello messages only",
                )

            payload = message.payload
            accepted = True
            selected_version = 0
            reason_code = ""
            if payload["node_id"] != peer_id:
                accepted = False
                reason_code = "node_identity_mismatch"
            else:
                try:
                    selected_version = negotiate_protocol_version(
                        payload["min_version"],
                        payload["max_version"],
                        local_min_version=PROTOCOL_VERSION,
                        local_max_version=PROTOCOL_VERSION,
                    )
                except WorkerProtocolError:
                    accepted = False
                    selected_version = 0
                    reason_code = "protocol_v2_required"

            ack_version = selected_version if accepted else message.version
            now_ms = (
                int(time.time() * 1000)
                if sent_at_ms is None else int(sent_at_ms)
            )
            ack = build_message(
                "hello_ack",
                {
                    "coordinator_node_id": coordinator_node_id,
                    "accepted": accepted,
                    "selected_version": selected_version,
                    "reason_code": reason_code,
                },
                message_id=_message_id("helloack_"),
                sent_at_ms=now_ms,
                version=ack_version,
            )
            now = time.time()
            self._workers[peer_id] = {
                "node_id": peer_id,
                "worker_kind": payload["worker_kind"],
                "connected": True,
                "accepted": accepted,
                "selected_version": selected_version,
                "capabilities": payload["capabilities"],
                "hello_received_at": now,
                "last_transport_heartbeat": now,
                "reason_code": reason_code,
            }
            if not accepted:
                self._rejected_message_count += 1
            self._remember(peer_id, message, ack)
            return ack

    def receive_on_worker(
        self, raw: bytes | str | Mapping[str, Any],
    ) -> WorkerMessage:
        """Accept the coordinator's v2 hello acknowledgement."""
        message = decode_message(raw)
        with self._lock:
            duplicate = self._duplicate_response("coordinator", message)
            if duplicate is not None:
                return duplicate
            if message.message_type != "hello_ack":
                self._rejected_message_count += 1
                raise self._reject(
                    "control_plane_only",
                    "the worker hello handler accepts hello_ack messages only",
                )
            payload = message.payload
            if not self._worker_hello_pending:
                self._rejected_message_count += 1
                raise self._reject(
                    "unexpected_hello_ack",
                    "hello_ack arrived without a pending hello",
                )
            accepted = bool(payload["accepted"])
            if accepted and payload["selected_version"] != PROTOCOL_VERSION:
                self._rejected_message_count += 1
                raise self._reject(
                    "protocol_v2_required",
                    "the PC Full Worker adapter requires protocol v2",
                )
            now = time.time()
            self._coordinator = {
                "node_id": payload["coordinator_node_id"],
                "connected": True,
                "healthy": accepted,
                "accepted": accepted,
                "selected_version": payload["selected_version"],
                "hello_ack_received_at": now,
                "last_transport_heartbeat": now,
                "reason_code": payload["reason_code"],
            }
            self._worker_hello_pending = False
            if not accepted:
                self._rejected_message_count += 1
            self._remember("coordinator", message)
            return message

    def mark_worker_heartbeat(self, peer_id: str) -> None:
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is not None and worker.get("connected"):
                worker["last_transport_heartbeat"] = time.time()

    def record_rejection(self) -> None:
        """Account for outer transport or node-admission rejection."""
        with self._lock:
            self._rejected_message_count += 1

    def mark_coordinator_heartbeat(self) -> None:
        with self._lock:
            if self._coordinator.get("connected"):
                self._coordinator["last_transport_heartbeat"] = time.time()

    def disconnect_worker(self, peer_id: str) -> None:
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is not None:
                worker["connected"] = False
                worker["disconnected_at"] = time.time()

    def disconnect_coordinator(self) -> None:
        with self._lock:
            if self._coordinator:
                self._coordinator["connected"] = False
                self._coordinator["healthy"] = False
                self._coordinator["disconnected_at"] = time.time()
            self._worker_hello_pending = False

    def worker_snapshot(self, peer_id: str) -> dict[str, Any]:
        """Return one public worker snapshot for Provider inspection."""
        now = time.time()
        with self._lock:
            worker = self._workers.get(peer_id)
            if worker is None:
                return {}
            snapshot = self._healthy_snapshot(worker, now)
            snapshot["provider_id"] = remote_provider_id(peer_id)
            return snapshot

    def coordinator_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if not self._coordinator:
                return {}
            return self._healthy_snapshot(self._coordinator, now)

    def _healthy_snapshot(self, peer: Mapping[str, Any], now: float) -> dict:
        snapshot = dict(peer)
        capabilities = snapshot.get("capabilities")
        if isinstance(capabilities, dict):
            snapshot["capabilities"] = {
                key: (
                    [dict(item) if isinstance(item, dict) else item for item in value]
                    if isinstance(value, list) else value
                )
                for key, value in capabilities.items()
            }
        last_seen = float(snapshot.get("last_transport_heartbeat", 0.0) or 0.0)
        snapshot["healthy"] = bool(
            snapshot.get("connected")
            and snapshot.get("accepted")
            and last_seen > 0
            and now - last_seen <= self._health_timeout_seconds
        )
        snapshot["task_dispatch_enabled"] = False
        snapshot["manual_stage_dispatch_enabled"] = bool(
            snapshot["healthy"] and snapshot.get("selected_version") == 2
        )
        return snapshot

    def status(self, *, role: str) -> dict[str, Any]:
        """Return public N2.4 control state before deployment gate overlay."""
        now = time.time()
        with self._lock:
            workers = []
            for node_id in sorted(self._workers):
                worker = self._healthy_snapshot(self._workers[node_id], now)
                worker["provider_id"] = remote_provider_id(node_id)
                workers.append(worker)
            coordinator = (
                self._healthy_snapshot(self._coordinator, now)
                if self._coordinator else {}
            )
            if role == "master":
                connected = any(item["healthy"] for item in workers)
            else:
                connected = bool(coordinator.get("healthy", False))
            return {
                "transport": "existing_tcp_length_prefixed",
                "transport_max_message_bytes": MAX_MESSAGE_BYTES,
                "phase": "TC-N2.4",
                "control_plane_ready": True,
                "control_plane_connected": connected,
                "adapter_connected": False,
                "task_dispatch_enabled": False,
                "manual_stage_dispatch_enabled": connected,
                "lease_renew_enabled": connected,
                "stage_cancel_enabled": connected,
                "stage_message_replay_enabled": True,
                "auto_provider_selection_enabled": connected,
                "admission_state": "n2_4_physical_validation_pending",
                "connected_worker_count": sum(
                    bool(item["healthy"]) for item in workers
                ),
                "workers": workers,
                "coordinator": coordinator,
                "rejected_message_count": self._rejected_message_count,
            }


@dataclass
class _PendingRemoteAttempt:
    attempt: StageAttempt
    lease_expires_at: float
    accepted: bool = False
    accept_event: threading.Event = field(default_factory=threading.Event)
    result_event: threading.Event = field(default_factory=threading.Event)
    cancel_ack_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[StageResult] = None
    error: Optional[BaseException] = None
    cancel_requested: bool = False
    cancel_acknowledged: bool = False
    released: bool = False
    released_at: float = 0.0


class RemoteFullWorkerProvider:
    """Remote Provider whose results always enter TaskGraph fencing."""

    provider_kind = "remote_full_worker"

    def __init__(
        self,
        *,
        node_id: str,
        peer_snapshot: Callable[[], Mapping[str, Any]],
        send_message: Callable[[WorkerMessage], None],
        accept_timeout_seconds: float = 10.0,
    ):
        self.node_id = str(node_id)
        self.provider_id = remote_provider_id(self.node_id)
        self._peer_snapshot = peer_snapshot
        self._send_message = send_message
        self._accept_timeout_seconds = max(
            0.1, min(float(accept_timeout_seconds), 60.0)
        )
        self._reservations: dict[str, tuple[Reservation, StageRequest]] = {}
        self._executed_reservations: set[str] = set()
        self._pending: dict[str, _PendingRemoteAttempt] = {}
        self._reservation_attempts: dict[str, str] = {}
        self._seen_messages: dict[str, str] = {}
        self._seen_order: collections.deque[str] = collections.deque()
        self._closed = False
        self._lock = threading.RLock()
        self._outbound_queue: queue.Queue[
            tuple[WorkerMessage, Callable[[Exception], None]]
        ] = queue.Queue(maxsize=64)
        self._outbound_stop = threading.Event()
        self._outbound_start_lock = threading.Lock()
        self._outbound_thread: Optional[threading.Thread] = None

    def _send_outbound_messages(self) -> None:
        while True:
            try:
                message, on_error = self._outbound_queue.get(timeout=0.2)
            except queue.Empty:
                with self._outbound_start_lock:
                    if self._outbound_queue.empty():
                        self._outbound_thread = None
                        return
                continue
            try:
                self._send_message(message)
            except Exception as exc:
                try:
                    on_error(exc)
                except Exception:
                    pass
            finally:
                self._outbound_queue.task_done()

    def _queue_outbound_message(
        self,
        message: WorkerMessage,
        on_error: Callable[[Exception], None],
    ) -> bool:
        with self._outbound_start_lock:
            if self._outbound_stop.is_set():
                on_error(ConnectionError("remote worker Provider is closed"))
                return False
            try:
                self._outbound_queue.put_nowait((message, on_error))
            except queue.Full as exc:
                on_error(exc)
                return False
            if self._outbound_thread is None:
                self._outbound_thread = threading.Thread(
                    target=self._send_outbound_messages,
                    name=f"task-worker-outbound-{self.provider_id}",
                    daemon=True,
                )
                self._outbound_thread.start()
            return True

    def _check_duplicate_locked(self, message: WorkerMessage) -> bool:
        digest = self._seen_messages.get(message.message_id)
        if digest is None:
            return False
        if digest != _message_digest(message):
            raise WorkerProtocolError(
                "message_id was reused with different Stage content",
                code="message_id_conflict",
                field="message_id",
            )
        return True

    def _remember_message_locked(self, message: WorkerMessage) -> None:
        self._seen_messages[message.message_id] = _message_digest(message)
        self._seen_order.append(message.message_id)
        while len(self._seen_order) > _MESSAGE_CACHE_LIMIT:
            expired = self._seen_order.popleft()
            self._seen_messages.pop(expired, None)

    def _prune_pending_locked(self) -> None:
        now = time.time()
        expired = [
            attempt_id
            for attempt_id, pending in self._pending.items()
            if pending.released
            and pending.released_at > 0
            and now - pending.released_at >= 5.0
        ]
        for attempt_id in expired:
            self._pending.pop(attempt_id, None)

    def _snapshot(self) -> dict[str, Any]:
        try:
            return dict(self._peer_snapshot() or {})
        except Exception:
            return {}

    @staticmethod
    def _model_matches(
        requested: Optional[ModelIdentity], models: Any,
    ) -> bool:
        if requested is None or not isinstance(models, list):
            return False
        expected = requested.snapshot()
        return any(
            isinstance(model, dict) and model == expected
            for model in models
        )

    def inspect(self) -> ProviderCapabilities:
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        stage_types = capabilities.get("stage_types", [])
        if not isinstance(stage_types, list):
            stage_types = []
        # N2.3 still exposes one controlled slot even if hello advertises
        # future multi-Stage capacity.
        max_concurrency = 1
        with self._lock:
            self._prune_pending_locked()
            active = len(self._reservations)
            closed = self._closed
        healthy = bool(
            not closed
            and snapshot.get("healthy")
            and snapshot.get("selected_version") == 2
            and snapshot.get("manual_stage_dispatch_enabled")
        )
        return ProviderCapabilities(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            supported_stage_types=tuple(
                value for value in stage_types
                if value in {"full_inference", "aggregate"}
            ),
            max_concurrency=max_concurrency,
            active_reservations=active,
            healthy=healthy,
            available=healthy and active < max_concurrency,
            node_id=self.node_id,
        )

    def supports_model_identity(
        self, model_identity: ModelIdentity, stage_type: str,
    ) -> bool:
        status = self.inspect()
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            return False
        return bool(
            status.healthy
            and stage_type in status.supported_stage_types
            and self._model_matches(
                model_identity, capabilities.get("models", []),
            )
        )

    def reserve(self, request: StageRequest) -> Reservation:
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        status = self.inspect()
        if request.provider_id != self.provider_id:
            raise ProviderReservationError(
                "stage request targets a different remote provider",
                code="provider_request_mismatch",
                provider_id=self.provider_id,
            )
        if request.stage_type not in status.supported_stage_types:
            raise ProviderUnavailable(
                "remote worker does not support the requested stage type",
                code="unsupported_stage_type",
                provider_id=self.provider_id,
            )
        if request.model_identity is None:
            raise ProviderUnavailable(
                "remote execution requires an exact model identity",
                code="model_identity_required",
                provider_id=self.provider_id,
            )
        if not self._model_matches(
            request.model_identity, capabilities.get("models", []),
        ):
            raise ProviderUnavailable(
                "remote worker does not have the exact requested model",
                code="model_identity_mismatch",
                provider_id=self.provider_id,
                retryable=True,
            )
        with self._lock:
            self._prune_pending_locked()
            if self._closed or not status.healthy:
                raise ProviderUnavailable(
                    "remote worker is not healthy",
                    code="remote_worker_unavailable",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            if len(self._reservations) >= status.max_concurrency:
                raise ProviderBusy(
                    "remote worker has no free manual Stage slot",
                    code="remote_worker_busy",
                    provider_id=self.provider_id,
                    retryable=True,
                )
            reservation = Reservation(
                reservation_id=f"res_{uuid.uuid4().hex}",
                provider_id=self.provider_id,
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                created_at=time.time(),
                selection_reason=(
                    "auto_remote_provider"
                    if request.runtime_context.get(
                        "task_graph_remote_policy"
                    ) == "auto"
                    else "explicit_remote_provider"
                ),
                provider_kind=self.provider_kind,
                provider_node_id=self.node_id,
            )
            self._reservations[reservation.reservation_id] = (
                reservation, request,
            )
            return reservation

    @staticmethod
    def _identity_matches(
        payload: Mapping[str, Any], attempt: StageAttempt,
    ) -> bool:
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
        pending: _PendingRemoteAttempt,
        cancel_event: threading.Event,
        deadline: float | Callable[[], float],
        *,
        timeout_code: str,
        provider_id: str,
    ) -> None:
        while not event.wait(0.05):
            if cancel_event.is_set():
                raise ProviderExecutionError(
                    "remote Stage wait was cancelled locally",
                    code="provider_cancelled",
                    provider_id=provider_id,
                )
            current_deadline = deadline() if callable(deadline) else deadline
            if time.time() >= current_deadline:
                raise ProviderExecutionError(
                    "remote Stage response timed out",
                    code=timeout_code,
                    provider_id=provider_id,
                    retryable=True,
                )
        if pending.error is not None:
            raise pending.error

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
                    "remote reservation is unknown",
                    code="invalid_reservation",
                    provider_id=self.provider_id,
                )
            if (
                owned[1] != attempt.request
                or attempt.provider_id != self.provider_id
                or reservation.provider_id != self.provider_id
            ):
                raise ProviderReservationError(
                    "remote attempt does not match its reservation",
                    code="attempt_reservation_mismatch",
                    provider_id=self.provider_id,
                )
            if reservation.reservation_id in self._executed_reservations:
                raise ProviderReservationError(
                    "remote reservation has already been executed",
                    code="reservation_already_executed",
                    provider_id=self.provider_id,
                )
            if attempt.request.model_identity is None:
                raise ProviderExecutionError(
                    "remote attempt has no model identity",
                    code="model_identity_required",
                    provider_id=self.provider_id,
                )
            pending = _PendingRemoteAttempt(
                attempt=attempt,
                lease_expires_at=attempt.lease_expires_at,
            )
            self._pending[attempt.attempt_id] = pending
            self._reservation_attempts[reservation.reservation_id] = (
                attempt.attempt_id
            )
            self._executed_reservations.add(reservation.reservation_id)

        sent_at_ms = int(time.time() * 1000)
        lease_expires_at_ms = int(attempt.lease_expires_at * 1000)
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
                    "lease_expires_at_ms": lease_expires_at_ms,
                    "provider_id": self.provider_id,
                    "root_input": attempt.request.root_input,
                    "dependencies": attempt.request.dependencies,
                    "input_sha256": stage_input_sha256(
                        attempt.request.root_input,
                        attempt.request.dependencies,
                    ),
                    "model_identity": attempt.request.model_identity.snapshot(),
                },
                message_id=_message_id("offer_"),
                sent_at_ms=sent_at_ms,
                version=PROTOCOL_VERSION,
        )
        try:
            self._send_message(offer)
        except Exception as exc:
            raise ProviderExecutionError(
                "failed to send Stage offer to the remote worker",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            ) from exc
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
            if exc.code == "remote_accept_timeout":
                self.cancel(attempt.attempt_id)
            raise
        if not pending.accepted:
            raise ProviderExecutionError(
                "remote worker did not accept the Stage",
                code="remote_stage_not_accepted",
                provider_id=self.provider_id,
            )
        self._wait(
            pending.result_event,
            pending,
            cancel_event,
            lambda: pending.lease_expires_at,
            timeout_code="lease_expired",
            provider_id=self.provider_id,
        )
        if pending.result is None:
            raise ProviderExecutionError(
                "remote worker returned no Stage result",
                code="invalid_provider_result",
                provider_id=self.provider_id,
            )
        return pending.result

    def handle_message(
        self, raw: bytes | str | Mapping[str, Any],
    ) -> WorkerMessage:
        message = decode_message(raw)
        if message.message_type not in {
            "stage_accept", "stage_result", "stage_error", "stage_cancelled",
        }:
            raise WorkerProtocolError(
                "message is not a coordinator-side Stage response",
                code="invalid_message_direction",
                field="message_type",
            )
        payload = message.payload
        attempt_id = str(payload.get("attempt_id", ""))
        with self._lock:
            self._prune_pending_locked()
            if self._check_duplicate_locked(message):
                return message
            pending = self._pending.get(attempt_id)
            if pending is None:
                raise WorkerProtocolError(
                    "Stage response has no pending attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._identity_matches(payload, pending.attempt):
                raise WorkerProtocolError(
                    "Stage response identity does not match the pending attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            if message.message_type == "stage_accept":
                if pending.accept_event.is_set():
                    raise WorkerProtocolError(
                        "Stage acceptance was already recorded",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                if payload["accepted"]:
                    pending.accepted = True
                else:
                    pending.error = ProviderReservationError(
                        "remote worker rejected the Stage offer",
                        code=payload["reason_code"],
                        provider_id=self.provider_id,
                        retryable=bool(payload["retryable"]),
                    )
                pending.accept_event.set()
            elif message.message_type == "stage_result":
                if not pending.accepted:
                    raise WorkerProtocolError(
                        "Stage result arrived before acceptance",
                        code="result_before_accept",
                        field="message_type",
                    )
                if pending.result_event.is_set():
                    raise WorkerProtocolError(
                        "Stage attempt already has a terminal response",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                pending.result = StageResult(
                    output=payload["output"],
                    provider_id=payload["provider_id"],
                    metadata=payload["metadata"],
                    attempt_id=payload["attempt_id"],
                    lease_epoch=payload["lease_epoch"],
                )
                pending.result_event.set()
            elif message.message_type == "stage_error":
                if pending.result_event.is_set():
                    raise WorkerProtocolError(
                        "Stage attempt already has a terminal response",
                        code="duplicate_stage_response",
                        field="message_type",
                    )
                pending.error = ProviderExecutionError(
                    "remote worker reported a Stage error",
                    code=payload["error_code"],
                    provider_id=self.provider_id,
                    retryable=bool(payload["retryable"]),
                )
                pending.accept_event.set()
                pending.result_event.set()
            else:
                if not pending.cancel_requested:
                    raise WorkerProtocolError(
                        "Stage cancellation acknowledgement was not requested",
                        code="unexpected_stage_cancelled",
                        field="message_type",
                    )
                if pending.cancel_acknowledged:
                    self._remember_message_locked(message)
                    return message
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
                    "remote lease renewal has no pending attempt",
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
                    "remote lease renewal identity is stale",
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
                message_id=_message_id("renew_"),
                sent_at_ms=int(time.time() * 1000),
                version=PROTOCOL_VERSION,
            )
            pending.lease_expires_at = deadline

        def on_send_error(_exc: Exception) -> None:
            error = ProviderExecutionError(
                "failed to renew the remote Stage lease",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.error = error
                    current.accept_event.set()
                    current.result_event.set()

        if not self._queue_outbound_message(message, on_send_error):
            raise ProviderExecutionError(
                "remote Stage lease renewal queue is full",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            )
        return True

    def cancel(self, attempt_id: str) -> None:
        with self._lock:
            pending = self._pending.get(attempt_id)
            if pending is None or pending.cancel_requested:
                return
            pending.cancel_requested = True
            pending.error = ProviderExecutionError(
                "remote Stage was cancelled locally",
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
                message_id=_message_id("cancel_"),
                sent_at_ms=int(time.time() * 1000),
                version=PROTOCOL_VERSION,
            )

        def on_send_error(_exc: Exception) -> None:
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.cancel_acknowledged = True
                    current.cancel_ack_event.set()
                    if current.released:
                        self._pending.pop(attempt_id, None)

        self._queue_outbound_message(message, on_send_error)

    def notify_disconnect(self) -> None:
        with self._lock:
            released = []
            for attempt_id, pending in self._pending.items():
                pending.error = ProviderExecutionError(
                    "remote worker disconnected",
                    code="remote_worker_disconnected",
                    provider_id=self.provider_id,
                    retryable=True,
                )
                pending.accept_event.set()
                pending.result_event.set()
                pending.cancel_ack_event.set()
                if pending.released:
                    released.append(attempt_id)
            for attempt_id in released:
                self._pending.pop(attempt_id, None)

    def release(self, reservation_id: str) -> None:
        with self._lock:
            self._reservations.pop(reservation_id, None)
            self._executed_reservations.discard(reservation_id)
            attempt_id = self._reservation_attempts.pop(
                reservation_id, "",
            )
            pending = self._pending.get(attempt_id)
            if pending is not None:
                pending.released = True
                pending.released_at = time.time()
                if (
                    not pending.cancel_requested
                    or pending.cancel_acknowledged
                ):
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
        self._outbound_stop.set()
