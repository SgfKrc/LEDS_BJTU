"""TC-N2.4 task-worker control plane with physical admission pending."""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

from task_provider import (
    DEPENDENCY_FAILURES_KEY,
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


def _canonical_capabilities(value: Any) -> str:
    """把 capabilities 折成可比字符串（dict 顺序无关）。

    用于「幂等 hello」：判断 worker 这次上报的能力与上次是否**内容相同**。
    不可 JSON 序列化的值退化为 `repr` —— 宁可判成"变了"（多重推一次层配置），
    也不要漏判（漏了就永远不推）。
    """
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(value)


class TaskWorkerControlPlane:
    """Own hello negotiation and health state without accepting Stage work."""

    def __init__(self, *, health_timeout_seconds: float = 120.0):
        self._health_timeout_seconds = max(1.0, float(health_timeout_seconds))
        self._workers: dict[str, dict[str, Any]] = {}
        self._coordinator: dict[str, Any] = {}
        self._worker_hello_pending = False
        # REGISTER ACK may precede the peer's task-worker hello. Fence that
        # connection from legacy layer assignment during this handshake.
        self._coordinator_pending_workers: set[str] = set()
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
        worker_kind: str = "pc_full_worker",
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
                    "worker_kind": worker_kind,
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
                    # ⚠️ 该 code 是稳定契约值（**不改**，避免破坏既有消费者）；
                    # 但语义是「双方版本区间无交集」，不是「必须 v2」——见 v3 层段引入后。
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
            previous = self._workers.get(peer_id)
            # ★ 幂等 hello（#28）：capabilities 的**内容**是否变化。协调方据它决定要不要
            #   重推层配置 —— 此前每次 hello 后都无条件
            #   `push_layer_config_to_clients()`（无内容比较），一旦层段路径也补
            #   `refresh_task_worker_capabilities()`，就形成
            #   `hello → push → load_layer_range → refresh → hello` 自激环：每次 push 都取
            #   新 generation，worker 端永远判成"新配置"。
            #   首次 hello（`previous is None`）等效"变化"，保证仍会推一次。
            connection_rebound = peer_id in self._coordinator_pending_workers
            capabilities_changed = connection_rebound or _canonical_capabilities(
                payload.get("capabilities")
            ) != _canonical_capabilities(
                previous.get("capabilities") if isinstance(previous, dict) else None
            )
            self._workers[peer_id] = {
                "node_id": peer_id,
                "worker_kind": payload["worker_kind"],
                "connected": True,
                "accepted": accepted,
                "selected_version": selected_version,
                "capabilities": payload["capabilities"],
                "capabilities_changed": capabilities_changed,
                "connection_rebound": connection_rebound,
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

    def mark_worker_connection_pending(self, peer_id: str) -> None:
        with self._lock:
            self._coordinator_pending_workers.add(str(peer_id))

    def resolve_worker_connection_pending(self, peer_id: str) -> None:
        with self._lock:
            self._coordinator_pending_workers.discard(str(peer_id))

    def pending_worker_ids(self) -> set[str]:
        with self._lock:
            return set(self._coordinator_pending_workers)

    def connected_layer_stage_workers(
        self, peer_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return accepted layer-capable workers without the health overlay.

        Recovery must publish a fresh assignment before readiness is checked.
        The normal status projection is heartbeat-gated, so using it as the
        candidate source can deadlock recovery at ``not_configured``.  This
        still requires an accepted hello and a live task-worker connection;
        execution keeps the normal provider health gate.
        """
        allowed = {str(value) for value in peer_ids} if peer_ids is not None else None
        with self._lock:
            workers = []
            for node_id, worker in self._workers.items():
                if allowed is not None and node_id not in allowed:
                    continue
                if not worker.get("connected") or not worker.get("accepted"):
                    continue
                capabilities = worker.get("capabilities")
                if not isinstance(capabilities, dict):
                    continue
                stage_types = capabilities.get("stage_types", [])
                ranges = capabilities.get("layer_ranges")
                if (
                    worker.get("selected_version", 0) < 2
                    or "layer_forward" not in stage_types
                    or not ranges
                ):
                    continue
                workers.append({
                    "node_id": node_id,
                    "worker_kind": worker.get("worker_kind", ""),
                    "selected_version": worker.get("selected_version", 0),
                    "capabilities": dict(capabilities),
                    "connected": True,
                    "accepted": True,
                })
            return workers

    def mark_coordinator_heartbeat(self) -> None:
        with self._lock:
            if self._coordinator.get("connected"):
                self._coordinator["last_transport_heartbeat"] = time.time()

    def disconnect_worker(self, peer_id: str) -> None:
        with self._lock:
            self._coordinator_pending_workers.discard(str(peer_id))
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
            # ★ 2026-09-20：原为硬编码 `selected_version == 2`，协议升到 v3 后
            #   **永远为 False**（实测：能力正常但 provider 一直 unhealthy）。
            #   语义是「v2 及以上的数据面通道可用」，故与当前版本号解耦。
            snapshot["healthy"] and int(snapshot.get("selected_version") or 0) >= 2
        )
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        stage_types = capabilities.get("stage_types", [])
        snapshot["layer_stage_dispatch_enabled"] = bool(
            snapshot["manual_stage_dispatch_enabled"]
            and "layer_forward" in stage_types
            and capabilities.get("layer_ranges")
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
                "layer_stage_dispatch_enabled": any(
                    item.get("layer_stage_dispatch_enabled") for item in workers
                ) if role == "master" else connected,
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
    offer_sent: bool = False
    cancel_requested: bool = False
    cancel_enqueued: bool = False
    cancel_acknowledged: bool = False
    #: ★ 2026-10-07（DIST-NEXT-1）：「已收到取消 ACK」与「执行确实已停止」是两个
    #: 事实。前者由 `cancel_acknowledged` 表示；后者只有对端显式回报
    #: `execution_state == "execution_stopped"` 时才成立（缺省/旧对端 = 未证明）。
    cancel_ack_execution_state: str = ""
    cancel_execution_stopped: bool = False
    #: 对端**主动**取消（本地用户取消 / Service 回收），本端并未请求。
    cancel_remote_initiated: bool = False
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
        #: ★ 2026-10-07（DIST-NEXT-1）：取消合同的分类计数。把「请求已发出」
        #: 「ACK 已收到」「执行确证已停止」「对端主动取消」「迟到结果被吸收」
        #: 分开统计 —— 否则「请求已取消」会被当成「执行已停止」。
        self._cancel_ack_states: collections.Counter[str] = collections.Counter()
        self._cancel_execution_stopped = 0
        self._cancel_remote_initiated = 0
        self._late_stage_responses = 0
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
                if message.message_type == "stage_cancel":
                    logger.info(
                        "event=task_worker_stage_cancel_sent node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s",
                        self.node_id,
                        message.payload.get("workflow_id", ""),
                        message.payload.get("stage_id", ""),
                        message.payload.get("attempt_id", ""),
                    )
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

    def _take_cancel_message_locked(
        self, pending: _PendingRemoteAttempt,
    ) -> Optional[WorkerMessage]:
        """Build one cancellation only after its Stage offer is on the wire."""
        if (
            not pending.offer_sent
            or not pending.cancel_requested
            or pending.cancel_enqueued
        ):
            return None
        pending.cancel_enqueued = True
        attempt = pending.attempt
        return build_message(
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

    def _queue_cancel_message(
        self, attempt_id: str, message: WorkerMessage,
    ) -> None:
        def on_send_error(_exc: Exception) -> None:
            with self._lock:
                current = self._pending.get(attempt_id)
                if current is not None:
                    current.cancel_acknowledged = True
                    current.cancel_ack_event.set()
                    if current.released:
                        self._pending.pop(attempt_id, None)

        self._queue_outbound_message(message, on_send_error)

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

    @staticmethod
    def _layer_model_matches(
        requested: Optional[ModelIdentity], models: Any,
    ) -> bool:
        """Match the physical artifact identity used by Route A layer workers.

        The coordinator's logical model id and revision may differ from the
        deterministic layer alias, but the engine, format, and source digest
        must remain exact so a crop from another model cannot be selected.
        """
        if requested is None or not isinstance(models, list):
            return False
        expected = requested.snapshot()
        return any(
            isinstance(model, dict)
            and all(model.get(key) == expected.get(key)
                    for key in ("engine", "format", "sha256"))
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
        worker_kind = snapshot.get("worker_kind", "pc_full_worker")
        resource_gate = capabilities.get("resource_gate")
        resource_admitted = True
        if worker_kind == "android_full_worker":
            resource_admitted = bool(
                isinstance(resource_gate, dict)
                and resource_gate.get("admitted") is True
                and not resource_gate.get("reason_code")
            )
        # N2.3 still exposes one controlled slot even if hello advertises
        # future multi-Stage capacity.
        max_concurrency = 1
        with self._lock:
            self._prune_pending_locked()
            active = len(self._reservations)
            closed = self._closed
        manual_dispatch_enabled = bool(
            snapshot.get("manual_stage_dispatch_enabled")
        )
        layer_dispatch_enabled = bool(
            snapshot.get("layer_stage_dispatch_enabled")
        )
        healthy = bool(
            not closed
            and snapshot.get("healthy")
            # ★ 2026-09-20：原为硬编码 `== 2`（第二处），协议升 v3 后 provider 恒 unhealthy。
            #   语义是「v2 及以上的数据面通道可用」，与具体版本号解耦。
            and int(snapshot.get("selected_version") or 0) >= 2
            # Provider health is generic.  The reservation path applies the
            # stage-specific gate so a layer-only capability cannot disable
            # ordinary full_inference dispatch on the same worker.
            and (manual_dispatch_enabled or layer_dispatch_enabled)
            and resource_admitted
        )
        return ProviderCapabilities(
            provider_id=self.provider_id,
            provider_kind=self.provider_kind,
            supported_stage_types=tuple(
                value for value in stage_types
                # ★ 2026-09-20：`layer_forward`（v3 层段）也必须能透传，否则
                #   安卓 worker 即使声明了层段能力，也会在这里被静默抹掉。
                if value in {"full_inference", "aggregate", "layer_forward"}
            ),
            max_concurrency=max_concurrency,
            active_reservations=active,
            healthy=healthy,
            available=healthy and active < max_concurrency,
            node_id=self.node_id,
        )

    def cancel_diagnostics(self) -> dict[str, Any]:
        """取消合同的分类计数（DIST-NEXT-1）。

        把「收到 ACK」「执行确证已停止」「对端主动取消」「迟到结果被吸收」
        分开统计。调用方（health / 测试 / 后续 metrics）据此区分
        「请求已取消」与「执行已停止」，不再由单一布尔冒充。
        """
        with self._lock:
            ack_states = dict(self._cancel_ack_states)
            pending_with_cancel = sum(
                1 for pending in self._pending.values()
                if pending.cancel_requested
            )
            still_in_flight = sum(
                1 for pending in self._pending.values()
                if pending.cancel_requested
                and pending.cancel_acknowledged
                and not pending.cancel_execution_stopped
            )
            return {
                "cancel_ack_execution_states": ack_states,
                "cancel_execution_stopped_confirmed": self._cancel_execution_stopped,
                "cancel_remote_initiated": self._cancel_remote_initiated,
                "late_stage_responses_absorbed": self._late_stage_responses,
                "pending_cancel_requests": pending_with_cancel,
                "pending_cancel_ack_in_flight": still_in_flight,
            }

    def supports_model_identity(
        self, model_identity: ModelIdentity, stage_type: str,
    ) -> bool:
        status = self.inspect()
        snapshot = self._snapshot()
        capabilities = snapshot.get("capabilities", {})
        if not isinstance(capabilities, dict):
            return False
        matcher = (
            self._layer_model_matches
            if stage_type == "layer_forward"
            else self._model_matches
        )
        return bool(
            status.healthy
            and stage_type in status.supported_stage_types
            and matcher(
                model_identity, capabilities.get("models", []),
            )
        )

    def model_identities(self) -> tuple[ModelIdentity, ...]:
        """Return validated immutable identities advertised by this v2 Worker."""
        capabilities = self._snapshot().get("capabilities", {})
        models = capabilities.get("models", []) if isinstance(capabilities, dict) else []
        identities = []
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict):
                continue
            try:
                identities.append(ModelIdentity(**model))
            except (TypeError, ValueError):
                continue
        return tuple(identities)

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
        if DEPENDENCY_FAILURES_KEY in request.dependencies:
            raise ProviderUnavailable(
                "protocol v2 remote workers do not support partial dependency input",
                code="partial_dependencies_not_supported",
                provider_id=self.provider_id,
                retryable=True,
            )
        if request.stage_type not in status.supported_stage_types:
            raise ProviderUnavailable(
                "remote worker does not support the requested stage type",
                code="unsupported_stage_type",
                provider_id=self.provider_id,
            )
        stage_dispatch_enabled = (
            bool(snapshot.get("layer_stage_dispatch_enabled"))
            if request.stage_type == "layer_forward"
            else bool(snapshot.get("manual_stage_dispatch_enabled"))
        )
        if not stage_dispatch_enabled:
            raise ProviderUnavailable(
                "remote worker is not admitted for the requested Stage type",
                code="stage_dispatch_not_admitted",
                provider_id=self.provider_id,
                retryable=True,
            )
        if request.model_identity is None:
            raise ProviderUnavailable(
                "remote execution requires an exact model identity",
                code="model_identity_required",
                provider_id=self.provider_id,
            )
        matcher = (
            self._layer_model_matches
            if request.stage_type == "layer_forward"
            else self._model_matches
        )
        if not matcher(request.model_identity, capabilities.get("models", [])):
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
        offer_payload = {
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
                }
        if attempt.request.stage_type == "layer_forward":
            offer_payload.update(attempt.request.stage_fields)
        elif attempt.request.stage_fields:
            raise ProviderExecutionError(
                "non-layer stage contains specialized stage fields",
                code="invalid_stage_fields",
                provider_id=self.provider_id,
            )
        offer = build_message(
                "stage_offer",
                offer_payload,
                message_id=_message_id("offer_"),
                sent_at_ms=sent_at_ms,
                version=PROTOCOL_VERSION,
        )
        try:
            self._send_message(offer)
        except Exception as exc:
            # A cancellation that raced with a failed offer send has no
            # remote work left to acknowledge. Mark it terminal so release()
            # can reclaim the pending entry immediately.
            with self._lock:
                current = self._pending.get(attempt.attempt_id)
                if current is pending and pending.cancel_requested:
                    pending.cancel_acknowledged = True
                    pending.cancel_ack_event.set()
            raise ProviderExecutionError(
                "failed to send Stage offer to the remote worker",
                code="remote_worker_disconnected",
                provider_id=self.provider_id,
                retryable=True,
            ) from exc
        cancel_message = None
        with self._lock:
            current = self._pending.get(attempt.attempt_id)
            if current is pending:
                pending.offer_sent = True
                cancel_message = self._take_cancel_message_locked(pending)
        if cancel_message is not None:
            self._queue_cancel_message(attempt.attempt_id, cancel_message)
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
            # ★ 2026-10-05（DIST-3「取消」场景，已知问题 #47）：本地取消必须
            #   **向对端传播** `stage_cancel`。本类 `cancel()` 里早就有构造与发送
            #   逻辑，但此前只在 `remote_accept_timeout` 这一条路径调用过。
            if exc.code in ("remote_accept_timeout", "provider_cancelled"):
                self.cancel(attempt.attempt_id)
            raise
        if not pending.accepted:
            raise ProviderExecutionError(
                "remote worker did not accept the Stage",
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
            # ★ 2026-10-05（同上）：等结果阶段被取消（`provider_cancelled`）或租约
            #   过期时，同样要让对端停手 —— 否则 worker 白跑完整个 Stage，回传结果
            #   时本端已无对应 pending（`reason=unknown_attempt`）。
            if exc.code in ("provider_cancelled", "lease_expired"):
                self.cancel(attempt.attempt_id)
            raise
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
                # ★ 2026-10-03：这类不匹配此前只报码、看不出是哪个字段对不上，
                #   真机排 Route A 时只能反复试。把两侧值一并打出来。
                logger.warning(
                    "Stage response identity mismatch: reply=%s expected=%s",
                    {
                        key: payload.get(key)
                        for key in (
                            "workflow_id", "stage_id", "attempt_id",
                            "lease_id", "lease_epoch", "provider_id",
                        )
                    },
                    {
                        "workflow_id": pending.attempt.request.workflow_id,
                        "stage_id": pending.attempt.request.stage_id,
                        "attempt_id": pending.attempt.attempt_id,
                        "lease_id": pending.attempt.lease_id,
                        "lease_epoch": pending.attempt.lease_epoch,
                        "provider_id": pending.attempt.provider_id,
                    },
                )
                raise WorkerProtocolError(
                    "Stage response identity does not match the pending attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            if (
                pending.cancel_requested
                and message.message_type in {
                    "stage_accept", "stage_result", "stage_error",
                }
            ):
                # A response already in flight can legally cross the local
                # cancellation. Keep provider_cancelled authoritative and
                # absorb the stale response idempotently instead of treating
                # the peer as a protocol violator.
                self._remember_message_locked(message)
                self._late_stage_responses += 1
                logger.info(
                    "event=task_worker_late_stage_response_ignored node_id=%s "
                    "message_type=%s attempt_id=%s",
                    self.node_id, message.message_type, attempt_id,
                )
                return message
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
                    # ★ 2026-10-03：把 worker 给的 `reason_code` 带进消息。此前只塞进
                    #   `code` 字段，异常在别处被转述成一层笼统的
                    #   `route_a_stage_execution_failed: remote worker rejected the Stage offer`
                    #   ⇒ 跨机层段失败时完全看不出是「身份不符」「租约过期」还是
                    #   「不支持该 stage」，只能靠逐层加日志去猜。
                    pending.error = ProviderReservationError(
                        "remote worker rejected the Stage offer"
                        f" (reason_code={payload['reason_code']}"
                        f", retryable={bool(payload['retryable'])})",
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
                # ★ 2026-10-07（DIST-NEXT-1）：取消合同。
                #   `execution_state` 可选：旧对端不带 ⇒ `unknown`（不冒充已停止）。
                execution_state = str(payload.get("execution_state") or "unknown")
                if not pending.cancel_requested:
                    # 对端**主动**取消（Android 本地用户取消 / Service 回收）。
                    # 此前一律按 `unexpected_stage_cancelled` 拒绝 ⇒ master 只能等到
                    # 租约/步骤超时，且在 coordinator 侧看不出是谁取消的。现在收敛成
                    # 单一 reason：该 attempt 由对端终止，本端以可重试的远端取消结束等待。
                    pending.cancel_remote_initiated = True
                    pending.cancel_acknowledged = True
                    pending.cancel_ack_execution_state = execution_state
                    pending.cancel_execution_stopped = (
                        execution_state == "execution_stopped"
                    )
                    if pending.cancel_execution_stopped:
                        self._cancel_execution_stopped += 1
                    pending.error = ProviderExecutionError(
                        "remote worker cancelled the Stage",
                        code=str(
                            payload.get("reason_code") or "remote_worker_cancelled"
                        ),
                        provider_id=self.provider_id,
                        retryable=True,
                    )
                    pending.accept_event.set()
                    pending.result_event.set()
                    pending.cancel_ack_event.set()
                    self._cancel_remote_initiated += 1
                    self._cancel_ack_states[execution_state] += 1
                    logger.info(
                        "event=task_worker_stage_cancel_remote_initiated node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s reason_code=%s "
                        "execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        payload.get("reason_code", ""),
                        execution_state,
                    )
                    self._remember_message_locked(message)
                    return message
                if pending.cancel_acknowledged and not pending.cancel_execution_stopped:
                    # 第二条 ACK 可以把「仍在执行」升级为「已停止」——这是取消合同
                    # 允许的唯一迟到的正向更新（其余迟到响应走上面的吸收分支）。
                    pass
                elif pending.cancel_acknowledged:
                    self._remember_message_locked(message)
                    return message
                # 一条 ACK 只证明「取消请求已送达」；只有对端显式回报
                # `execution_stopped` 才证明「执行已停止」。
                first_ack = not pending.cancel_acknowledged
                previous_state = pending.cancel_ack_execution_state
                pending.cancel_acknowledged = True
                if execution_state != "unknown":
                    pending.cancel_ack_execution_state = execution_state
                if (
                    execution_state == "execution_stopped"
                    and not pending.cancel_execution_stopped
                ):
                    pending.cancel_execution_stopped = True
                    self._cancel_execution_stopped += 1
                if first_ack or execution_state != previous_state:
                    self._cancel_ack_states[execution_state] += 1
                pending.cancel_ack_event.set()
                if first_ack:
                    logger.info(
                        "event=task_worker_stage_cancel_acknowledged node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        execution_state,
                    )
                elif execution_state != previous_state:
                    logger.info(
                        "event=task_worker_stage_cancel_execution_stopped node_id=%s "
                        "workflow_id=%s stage_id=%s attempt_id=%s execution_state=%s",
                        self.node_id,
                        payload.get("workflow_id", ""),
                        payload.get("stage_id", ""),
                        attempt_id,
                        execution_state,
                    )
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
        message = None
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
            message = self._take_cancel_message_locked(pending)

        if message is not None:
            self._queue_cancel_message(attempt_id, message)

    def notify_disconnect(self) -> None:
        """对端断连：唤醒所有 pending，并**主动回收本 provider 的全部 reservation**。

        ★ 2026-10-05（DIST-2 要求 2：「节点掉线…必须释放 lease」）：此前这里只把
        pending 置 error 并 set 三个 event，**不释放 reservation** —— 释放完全依赖
        上层 `task_graph._run_stage` 的 `finally`。一旦上层没走到那里（异常路径、
        外层取消、进程卡住），条目就会留在 `_reservations` /
        `_executed_reservations` / `_reservation_attempts` 里，而该 provider 在对端
        重连前不会再有活动 ⇒ 「已预留未执行」的槽位被永久占用。

        `release()` 是**纯本地 dict 操作**（不向 worker 发任何消息），在断连路径上
        调用没有副作用；先唤醒 pending 再回收，顺序保证等待方先拿到
        `remote_worker_disconnected` 错误。
        """
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
            reservation_ids = list(self._reservations.keys())
        # `release()` 内部自己取 `self._lock` ⇒ 必须在锁外调用（这里是普通 Lock，
        # 锁内再取会自锁）。
        for reservation_id in reservation_ids:
            self.release(reservation_id)

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
