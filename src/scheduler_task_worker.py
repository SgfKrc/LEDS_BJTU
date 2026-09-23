"""Task-worker protocol methods mixed into the Scheduler facade."""

from __future__ import annotations

import hashlib
import importlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from scheduler_types import NodeRole
from task_provider import (
    ModelIdentity as TaskModelIdentity,
    StageRequest as TaskProviderStageRequest,
    sanitize_result_metadata as sanitize_task_result_metadata,
)
from task_worker_adapter import RemoteFullWorkerProvider, remote_provider_id
from task_worker_protocol import (
    PROTOCOL_VERSION as TASK_WORKER_PROTOCOL_VERSION,
    WorkerMessage,
    WorkerProtocolError,
    build_message as build_task_worker_message,
    canonical_message_bytes as task_worker_message_bytes,
    canonical_sha256 as task_worker_sha256,
    decode_message as decode_task_worker_message,
    worker_protocol_status,
)
from torch_runtime import torch_available

logger = logging.getLogger("scheduler")


@dataclass
class _TaskWorkerActiveAttempt:
    workflow_id: str
    stage_id: str
    attempt_id: str
    lease_id: str
    lease_epoch: int
    provider_id: str
    lease_expires_at_ms: int
    lease_deadline_monotonic: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    lease_expired: bool = False
    cancel_reason: str = ""


class SchedulerTaskWorkerMixin:
    def _task_worker_capabilities(self) -> dict:
        """Build an honest PC Full Worker snapshot without loading a model."""
        engines = []
        if torch_available():
            engines.append("pytorch")
        try:
            if importlib.util.find_spec("llama_cpp") is not None:
                engines.append("llama_cpp")
        except (ImportError, ValueError):
            pass
        # TP 孤岛网关：以 island 引擎承担整请求推理（配置启用即上报能力）
        try:
            import config as _island_cfg
            if (getattr(_island_cfg, "ISLAND_ENABLED", False)
                    and getattr(_island_cfg, "ISLAND_BASE_URL", "")):
                engines.append("island")
        except Exception:
            pass
        # 外部推理服务（路线 B）：以 external_api 引擎承担整请求推理。
        # 仅声明能力，实际外发仍受数据作用域门控约束（默认 opt_in 不出集群）。
        try:
            import config as _external_cfg
            if (getattr(_external_cfg, "EXTERNAL_ENABLED", False)
                    and getattr(_external_cfg, "EXTERNAL_BASE_URL", "")):
                engines.append("external_api")
        except Exception:
            pass
        if not engines:
            # The PC application currently requires the PyTorch runtime. Keeping
            # the schema valid also makes a broken installation visible in hello.
            engines.append("pytorch")

        models = []
        try:
            # 阶段 0.2：完整模型判定直接走 host 代理（不依赖内部 manager 容器
            # 的 _instance 结构，阶段 1 替换远程 host 适配器后依然成立）
            has_loaded_model = getattr(self._host, "has_loaded_model", None)
            if callable(has_loaded_model):
                # ModelHost owns the authoritative loaded-state check and
                # correctly accounts for its lazy manager proxy.  Requiring
                # the legacy ``is_loaded`` attribute here made a real full
                # Safetensors worker advertise an empty model list.
                full_model_loaded = bool(has_loaded_model())
            else:
                full_model_loaded = bool(
                    getattr(self._host, "model_loaded", False)
                    and getattr(self._host, "is_loaded", False)
                )
            full_model_loaded = bool(
                full_model_loaded and getattr(self._host, "layer_range", None) is None
            )
            if full_model_loaded:
                identity = self._require_callbacks().active_task_graph_model_identity()
                if identity is not None and identity.engine in engines:
                    models.append({
                        "model_id": identity.model_id,
                        "engine": identity.engine,
                        "format": identity.format,
                        "revision": identity.revision,
                        "sha256": identity.sha256,
                    })
        except Exception:
            logger.warning(
                "构建 PC Full Worker 模型能力快照失败，暂不上报模型",
                exc_info=True,
            )
        return {
            "stage_types": ["full_inference", "aggregate"],
            "engines": engines,
            "models": models,
            "max_concurrency": 1,
        }


    def _send_task_worker_hello(
        self, client=None, refresh_generation: int = 0,
    ) -> bool:
        """Send the TC-N2.0 hello after authenticated TCP registration."""
        if (
            self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
        ):
            return False
        target = client or getattr(self, "_tcp_client", None)
        if not target or not getattr(target, "is_registered", False):
            return False
        try:
            from transport_port import MessageType

            message = self._task_worker_control.begin_worker_hello(
                node_id=self.get_effective_node_id(),
                capabilities=self._task_worker_capabilities(),
            )
            if message is None:
                return False
            with self._task_worker_refresh_lock:
                self._task_worker_refresh_requested = bool(
                    self._task_worker_refresh_generation > refresh_generation
                )
            target.send_data(message.snapshot(), MessageType.TASK_WORKER)
            logger.info(
                "event=task_worker_hello_sent node_id=%s version=%s",
                self.get_effective_node_id(), message.version,
            )
            return True
        except Exception as exc:
            self._task_worker_control.disconnect_coordinator()
            logger.warning(
                "event=task_worker_hello_failed node_id=%s error=%s",
                self.get_effective_node_id(),
                getattr(exc, "code", type(exc).__name__),
                exc_info=True,
            )
            return False


    def refresh_task_worker_capabilities(self) -> bool:
        """Refresh hello asynchronously after a local full-model change."""
        client = getattr(self, "_tcp_client", None)
        if (
            self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
            or not client
            or not getattr(client, "is_registered", False)
        ):
            return False
        with self._task_worker_refresh_lock:
            self._task_worker_refresh_requested = True
            self._task_worker_refresh_generation += 1
            refresh_generation = self._task_worker_refresh_generation
        threading.Thread(
            target=self._send_task_worker_hello,
            args=(client, refresh_generation),
            name="task-worker-capability-refresh",
            daemon=True,
        ).start()
        return True


    def _send_task_worker_to_node(
        self, node_id: str, message: WorkerMessage,
    ) -> None:
        server = self._tcp_server
        if server is None:
            raise ConnectionError("task worker TCP server is unavailable")
        from transport_port import MessageType

        server.send_to_client(
            node_id, message.snapshot(), MessageType.TASK_WORKER,
        )


    def _send_task_worker_to_master(self, message: WorkerMessage) -> None:
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "is_registered", False):
            raise ConnectionError("task worker is not connected to its master")
        from transport_port import MessageType

        client.send_data(message.snapshot(), MessageType.TASK_WORKER)


    def _ensure_remote_task_worker_provider(
        self, node_id: str,
    ) -> RemoteFullWorkerProvider:
        with self._task_worker_stage_lock:
            provider = self._remote_task_worker_providers.get(node_id)
            if provider is None:
                provider = RemoteFullWorkerProvider(
                    node_id=node_id,
                    peer_snapshot=lambda bound_node=node_id: (
                        self._task_worker_control.worker_snapshot(bound_node)
                    ),
                    send_message=lambda message, bound_node=node_id: (
                        self._send_task_worker_to_node(bound_node, message)
                    ),
                )
                self._remote_task_worker_providers[node_id] = provider
            return provider


    def remote_task_worker_providers(self) -> list[RemoteFullWorkerProvider]:
        """Return stable Provider objects for explicit TaskGraph registration."""
        with self._task_worker_stage_lock:
            return [
                self._remote_task_worker_providers[node_id]
                for node_id in sorted(self._remote_task_worker_providers)
            ]


    @staticmethod
    def _task_worker_message_digest(message: WorkerMessage) -> str:
        return hashlib.sha256(task_worker_message_bytes(message)).hexdigest()


    def _prepare_task_worker_request(
        self, message: WorkerMessage,
    ) -> tuple[bool, list[dict]]:
        digest = self._task_worker_message_digest(message)
        with self._task_worker_stage_lock:
            cached = self._task_worker_seen_messages.get(message.message_id)
            if cached is not None:
                if cached[0] != digest:
                    raise WorkerProtocolError(
                        "message_id was reused with different Stage content",
                        code="message_id_conflict",
                        field="message_id",
                    )
                return True, [dict(response) for response in cached[1]]
            self._task_worker_seen_messages[message.message_id] = (digest, [])
            self._task_worker_seen_order.append(message.message_id)
            while len(self._task_worker_seen_order) > 1024:
                expired = self._task_worker_seen_order.popleft()
                self._task_worker_seen_messages.pop(expired, None)
            return False, []


    def _cache_task_worker_response(
        self, request_message_id: str, response: WorkerMessage,
    ) -> None:
        with self._task_worker_stage_lock:
            cached = self._task_worker_seen_messages.get(request_message_id)
            if cached is not None:
                cached[1].append(response.snapshot())


    def _forget_task_worker_request(self, request_message_id: str) -> None:
        with self._task_worker_stage_lock:
            self._task_worker_seen_messages.pop(request_message_id, None)
            try:
                self._task_worker_seen_order.remove(request_message_id)
            except ValueError:
                pass


    def _send_task_worker_response(
        self, request_message_id: str, response: WorkerMessage,
    ) -> None:
        self._cache_task_worker_response(request_message_id, response)
        self._send_task_worker_to_master(response)


    def _replay_task_worker_responses(self, responses: list[dict]) -> None:
        for response in responses:
            self._send_task_worker_to_master(
                decode_task_worker_message(response)
            )


    @staticmethod
    def _task_worker_active_identity_matches(
        payload: dict, active: _TaskWorkerActiveAttempt,
    ) -> bool:
        return all((
            payload.get("workflow_id") == active.workflow_id,
            payload.get("stage_id") == active.stage_id,
            payload.get("attempt_id") == active.attempt_id,
            payload.get("lease_id") == active.lease_id,
            payload.get("lease_epoch") == active.lease_epoch,
        ))


    def _watch_task_worker_lease(self, attempt_id: str) -> None:
        while True:
            with self._task_worker_stage_lock:
                active = self._task_worker_active_attempts.get(attempt_id)
                if active is None:
                    return
                remaining = (
                    active.lease_deadline_monotonic - time.monotonic()
                )
                if remaining <= 0:
                    active.lease_expired = True
                    active.cancel_event.set()
                    return
                done_event = active.done_event
            if done_event.wait(min(0.05, remaining)):
                return


    def _handle_task_worker_lease_renew(
        self, message: WorkerMessage,
    ) -> None:
        payload = message.payload
        attempt_id = str(payload["attempt_id"])
        with self._task_worker_stage_lock:
            active = self._task_worker_active_attempts.get(attempt_id)
            if active is None:
                raise WorkerProtocolError(
                    "lease renewal has no active attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._task_worker_active_identity_matches(payload, active):
                raise WorkerProtocolError(
                    "lease renewal identity does not match the active attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            if (
                active.lease_expired
                or active.lease_deadline_monotonic <= time.monotonic()
            ):
                active.lease_expired = True
                active.cancel_event.set()
                raise WorkerProtocolError(
                    "an expired Stage lease cannot be renewed",
                    code="lease_expired",
                    field="payload.lease_expires_at_ms",
                )
            deadline = int(payload["lease_expires_at_ms"])
            if deadline <= active.lease_expires_at_ms:
                raise WorkerProtocolError(
                    "lease renewal must extend the active deadline",
                    code="stale_lease",
                    field="payload.lease_expires_at_ms",
                )
            active.lease_expires_at_ms = deadline
            active.lease_deadline_monotonic = time.monotonic() + (
                deadline - message.sent_at_ms
            ) / 1000.0


    def _handle_task_worker_stage_cancel(
        self, message: WorkerMessage,
    ) -> None:
        payload = message.payload
        attempt_id = str(payload["attempt_id"])
        with self._task_worker_stage_lock:
            active = self._task_worker_active_attempts.get(attempt_id)
            if active is None:
                raise WorkerProtocolError(
                    "Stage cancellation has no active attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._task_worker_active_identity_matches(payload, active):
                raise WorkerProtocolError(
                    "Stage cancellation identity does not match the active attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            active.cancel_reason = str(payload["reason_code"])
            active.cancel_event.set()
            provider_id = active.provider_id
        response_payload = self._task_worker_attempt_payload(
            payload, provider_id=provider_id,
        )
        response_payload["reason_code"] = str(payload["reason_code"])
        response = build_task_worker_message(
            "stage_cancelled",
            response_payload,
            message_id=f"msg_cancelled_{uuid.uuid4().hex}",
            sent_at_ms=int(time.time() * 1000),
            version=TASK_WORKER_PROTOCOL_VERSION,
        )
        self._send_task_worker_response(message.message_id, response)


    @staticmethod
    def _task_worker_attempt_payload(
        offer_payload: dict,
        *,
        provider_id: str,
    ) -> dict:
        return {
            "workflow_id": offer_payload["workflow_id"],
            "stage_id": offer_payload["stage_id"],
            "attempt_id": offer_payload["attempt_id"],
            "lease_id": offer_payload["lease_id"],
            "lease_epoch": offer_payload["lease_epoch"],
            "provider_id": provider_id,
        }


    def _send_task_worker_stage_accept(
        self,
        request_message_id: str,
        offer_payload: dict,
        *,
        accepted: bool,
        reason_code: str = "",
        retryable: bool = False,
    ) -> None:
        payload = self._task_worker_attempt_payload(
            offer_payload,
            provider_id=str(offer_payload.get("provider_id", "")),
        )
        payload.update({
            "accepted": bool(accepted),
            "reason_code": "" if accepted else reason_code,
            "retryable": False if accepted else bool(retryable),
        })
        response = build_task_worker_message(
            "stage_accept",
            payload,
            message_id=f"msg_accept_{uuid.uuid4().hex}",
            sent_at_ms=int(time.time() * 1000),
            version=TASK_WORKER_PROTOCOL_VERSION,
        )
        self._send_task_worker_response(request_message_id, response)


    def _handle_task_worker_stage_offer(
        self, message: WorkerMessage,
    ) -> None:
        """Validate and execute one explicit remote Stage on a PC worker."""
        offer = message.payload
        attempt_id = str(offer["attempt_id"])
        expected_provider = remote_provider_id(self.get_effective_node_id())
        reject_reason = ""
        reject_retryable = False
        active: Optional[_TaskWorkerActiveAttempt] = None

        coordinator = self._task_worker_control.coordinator_snapshot()
        if not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'):
            reject_reason = "worker_experiment_disabled"
            reject_retryable = True
        elif not coordinator.get("manual_stage_dispatch_enabled"):
            reject_reason = "worker_not_admitted"
            reject_retryable = True
        elif offer["provider_id"] != expected_provider:
            reject_reason = "provider_identity_mismatch"
        else:
            capabilities = self._task_worker_capabilities()
            if offer["stage_type"] not in capabilities["stage_types"]:
                reject_reason = "unsupported_stage_type"
                reject_retryable = True
            elif offer["model_identity"] not in capabilities["models"]:
                reject_reason = "model_identity_mismatch"
                reject_retryable = True

        with self._task_worker_stage_lock:
            if not reject_reason and (
                attempt_id in self._task_worker_active_attempts
                or self._task_worker_active_attempts
            ):
                reject_reason = "remote_worker_busy"
                reject_retryable = True
            if not reject_reason:
                active = _TaskWorkerActiveAttempt(
                    workflow_id=str(offer["workflow_id"]),
                    stage_id=str(offer["stage_id"]),
                    attempt_id=attempt_id,
                    lease_id=str(offer["lease_id"]),
                    lease_epoch=int(offer["lease_epoch"]),
                    provider_id=expected_provider,
                    lease_expires_at_ms=int(offer["lease_expires_at_ms"]),
                    lease_deadline_monotonic=time.monotonic() + (
                        int(offer["lease_expires_at_ms"])
                        - message.sent_at_ms
                    ) / 1000.0,
                )
                self._task_worker_active_attempts[attempt_id] = active

        if reject_reason:
            try:
                self._send_task_worker_stage_accept(
                    message.message_id,
                    offer,
                    accepted=False,
                    reason_code=reject_reason,
                    retryable=reject_retryable,
                )
            except Exception:
                self._forget_task_worker_request(message.message_id)
                logger.warning(
                    "event=task_worker_stage_reject_send_failed attempt_id=%s",
                    attempt_id, exc_info=True,
                )
            return
        if active is None:
            raise RuntimeError("accepted task-worker Stage has no active record")

        try:
            self._send_task_worker_stage_accept(
                message.message_id, offer, accepted=True,
            )
        except Exception:
            with self._task_worker_stage_lock:
                removed = self._task_worker_active_attempts.pop(
                    attempt_id, None,
                )
                if removed is not None:
                    removed.done_event.set()
            self._forget_task_worker_request(message.message_id)
            logger.warning(
                "event=task_worker_stage_accept_send_failed attempt_id=%s",
                attempt_id, exc_info=True,
            )
            return

        threading.Thread(
            target=self._watch_task_worker_lease,
            args=(attempt_id,),
            name=f"task-worker-lease-{attempt_id}",
            daemon=True,
        ).start()

        try:
            model_identity = TaskModelIdentity(**offer["model_identity"])
            request = TaskProviderStageRequest(
                workflow_id=offer["workflow_id"],
                request_id=offer["request_id"],
                stage_id=offer["stage_id"],
                stage_type=offer["stage_type"],
                provider_id=offer["provider_id"],
                dependencies=offer["dependencies"],
                root_input=offer["root_input"],
                model_identity=model_identity,
            )
            with self._host.full_chat_execution_lock:
                with self._inference_lock:
                    output = self._require_callbacks().execute_task_worker_stage(
                        request, active.cancel_event,
                    )
            if not isinstance(output, dict):
                raise RuntimeError("remote Stage executor returned non-object output")
            with self._task_worker_stage_lock:
                current = self._task_worker_active_attempts.get(attempt_id)
                if current is None:
                    raise RuntimeError("remote Stage attempt is no longer active")
                if (
                    current.lease_expired
                    or current.lease_deadline_monotonic <= time.monotonic()
                ):
                    current.lease_expired = True
                    current.cancel_event.set()
                    raise RuntimeError("remote Stage lease expired")
                if current.cancel_reason:
                    raise RuntimeError("remote Stage was cancelled")
            result_payload = self._task_worker_attempt_payload(
                offer, provider_id=expected_provider,
            )
            result_payload.update({
                "output": output,
                "output_sha256": task_worker_sha256(output),
                "metadata": sanitize_task_result_metadata(output),
            })
            response = build_task_worker_message(
                "stage_result",
                result_payload,
                message_id=f"msg_result_{uuid.uuid4().hex}",
                sent_at_ms=int(time.time() * 1000),
                version=TASK_WORKER_PROTOCOL_VERSION,
            )
            try:
                self._send_task_worker_response(message.message_id, response)
            except Exception:
                logger.warning(
                    "event=task_worker_stage_result_send_failed attempt_id=%s",
                    attempt_id, exc_info=True,
                )
                return
            logger.info(
                "event=task_worker_stage_result_sent workflow_id=%s stage_id=%s attempt_id=%s",
                offer["workflow_id"], offer["stage_id"], attempt_id,
            )
        except Exception as exc:
            with self._task_worker_stage_lock:
                current = self._task_worker_active_attempts.get(attempt_id)
                lease_expired = bool(
                    current is not None and current.lease_expired
                )
                cancelled_by_coordinator = bool(
                    current is not None and current.cancel_reason
                )
            if cancelled_by_coordinator and not lease_expired:
                logger.info(
                    "event=task_worker_stage_cancelled workflow_id=%s stage_id=%s attempt_id=%s",
                    offer["workflow_id"], offer["stage_id"], attempt_id,
                )
                return
            error_code = (
                "lease_expired"
                if lease_expired
                else "provider_cancelled"
                if active.cancel_event.is_set()
                else "remote_stage_execution_failed"
            )
            error_payload = self._task_worker_attempt_payload(
                offer, provider_id=expected_provider,
            )
            error_payload.update({
                "error_code": error_code,
                "retryable": error_code == "lease_expired",
            })
            try:
                response = build_task_worker_message(
                    "stage_error",
                    error_payload,
                    message_id=f"msg_error_{uuid.uuid4().hex}",
                    sent_at_ms=int(time.time() * 1000),
                    version=TASK_WORKER_PROTOCOL_VERSION,
                )
                self._send_task_worker_response(
                    message.message_id, response,
                )
            except Exception:
                logger.warning(
                    "event=task_worker_stage_error_send_failed attempt_id=%s",
                    attempt_id, exc_info=True,
                )
            logger.warning(
                "event=task_worker_stage_execution_failed workflow_id=%s stage_id=%s attempt_id=%s reason=%s",
                offer["workflow_id"], offer["stage_id"], attempt_id,
                type(exc).__name__, exc_info=True,
            )
        finally:
            with self._task_worker_stage_lock:
                removed = self._task_worker_active_attempts.pop(
                    attempt_id, None,
                )
                if removed is not None:
                    removed.done_event.set()


    def _handle_task_worker_message(self, client_id: str, msg: dict) -> None:
        """Route N2.2 hello, Stage, renewal, and cancellation messages."""
        raw = msg.get("data")
        if not isinstance(raw, dict):
            self._task_worker_control.record_rejection()
            logger.warning(
                "event=task_worker_message_rejected peer=%s reason=invalid_outer_payload",
                client_id,
            )
            return
        try:
            message = decode_task_worker_message(raw)
            if self._effective_role() == "master":
                if client_id == "master":
                    raise WorkerProtocolError(
                        "master cannot register as its own task worker",
                        code="invalid_message_direction",
                        field="message_type",
                    )
                worker_kind = (
                    message.payload.get("worker_kind")
                    if message.message_type == "hello"
                    else ""
                )
                with self._nodes_lock:
                    registered_node = self.nodes.get(client_id)
                    registered_role = getattr(
                        getattr(registered_node, "role", ""),
                        "value",
                        getattr(registered_node, "role", ""),
                    )
                    admitted_worker_node = bool(
                        registered_node is not None
                        and registered_node.node_type in {"pc", "android"}
                        and registered_role == NodeRole.CLIENT.value
                    )
                if not admitted_worker_node:
                    self._task_worker_control.record_rejection()
                    raise WorkerProtocolError(
                        "only a registered PC or Android client may negotiate a full worker",
                        code="unsupported_worker_node",
                        field="payload.worker_kind",
                    )
                if message.message_type == "hello":
                    expected_kind = (
                        "android_full_worker"
                        if registered_node.node_type == "android"
                        else "pc_full_worker"
                    )
                    if worker_kind != expected_kind:
                        self._task_worker_control.record_rejection()
                        raise WorkerProtocolError(
                            "worker kind does not match the registered node type",
                            code="worker_kind_node_type_mismatch",
                            field="payload.worker_kind",
                        )
                if message.message_type == "hello":
                    ack = self._task_worker_control.receive_on_coordinator(
                        client_id,
                        raw,
                        coordinator_node_id=self.get_effective_node_id(),
                    )
                    self._send_task_worker_to_node(client_id, ack)
                    if ack.payload["accepted"]:
                        self._ensure_remote_task_worker_provider(client_id)
                        # A node that has just advertised a complete model is
                        # a Full Worker, not a layer partition.  Registration
                        # may have pushed a layer config before the hello
                        # arrived; release that reservation so the two roles
                        # cannot race and make the Worker reject its Stage.
                        advertised_models = (
                            message.payload.get("capabilities", {}).get("models", [])
                            if isinstance(message.payload, dict)
                            else []
                        )
                        if advertised_models:
                            self._handle_layer_worker_opt_out(
                                client_id,
                                {"data": {"node_id": client_id}},
                            )
                    logger.info(
                        "event=task_worker_hello_acked node_id=%s accepted=%s version=%s",
                        client_id,
                        ack.payload["accepted"],
                        ack.payload["selected_version"],
                    )
                elif message.message_type in {
                    "stage_accept", "stage_result", "stage_error",
                    "stage_cancelled",
                }:
                    provider = self._remote_task_worker_providers.get(client_id)
                    if provider is None:
                        raise WorkerProtocolError(
                            "Stage response arrived before an accepted worker hello",
                            code="worker_not_admitted",
                            field="message_type",
                        )
                    provider.handle_message(raw)
                else:
                    raise WorkerProtocolError(
                        "message is not valid in the N2.2 coordinator direction",
                        code="invalid_message_direction",
                        field="message_type",
                    )
            else:
                if client_id != "master":
                    raise WorkerProtocolError(
                        "worker accepts task-worker control messages from master only",
                        code="invalid_message_direction",
                        field="message_type",
                    )
                if message.message_type == "hello_ack":
                    accepted = self._task_worker_control.receive_on_worker(raw)
                    logger.info(
                        "event=task_worker_hello_ack_received coordinator=%s accepted=%s version=%s",
                        accepted.payload["coordinator_node_id"],
                        accepted.payload["accepted"],
                        accepted.payload["selected_version"],
                    )
                    with self._task_worker_refresh_lock:
                        refresh_requested = self._task_worker_refresh_requested
                    if refresh_requested:
                        self.refresh_task_worker_capabilities()
                elif message.message_type == "stage_offer":
                    duplicate, responses = self._prepare_task_worker_request(
                        message
                    )
                    if duplicate:
                        self._replay_task_worker_responses(responses)
                        return
                    threading.Thread(
                        target=self._handle_task_worker_stage_offer,
                        args=(message,),
                        name=f"task-worker-stage-{message.payload['attempt_id']}",
                        daemon=True,
                    ).start()
                elif message.message_type in {"lease_renew", "stage_cancel"}:
                    duplicate, responses = self._prepare_task_worker_request(
                        message
                    )
                    if duplicate:
                        self._replay_task_worker_responses(responses)
                        return
                    if message.message_type == "lease_renew":
                        self._handle_task_worker_lease_renew(message)
                    else:
                        self._handle_task_worker_stage_cancel(message)
                else:
                    raise WorkerProtocolError(
                        "message is not valid in the N2.2 worker direction",
                        code="invalid_message_direction",
                        field="message_type",
                    )
        except (WorkerProtocolError, ConnectionError) as exc:
            logger.warning(
                "event=task_worker_message_rejected peer=%s reason=%s",
                client_id,
                getattr(exc, "code", type(exc).__name__),
                exc_info=True,
            )
        except Exception as exc:
            # A malformed control-plane message must not collapse the shared TCP
            # receive loop or any existing inference path.
            logger.warning(
                "event=task_worker_message_failed peer=%s reason=%s",
                client_id, type(exc).__name__, exc_info=True,
            )


    def get_task_worker_protocol_status(self) -> dict:
        role = self._effective_role()
        runtime = self._task_worker_control.status(role=role)
        connected = bool(runtime.get("control_plane_connected", False))
        if role == "master":
            healthy_workers = [
                worker for worker in runtime.get("workers", [])
                if isinstance(worker, dict) and worker.get("healthy")
            ]
            full_model_worker_ids = sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if isinstance(worker.get("capabilities"), dict)
                and bool(worker["capabilities"].get("models"))
                and (
                    worker.get("worker_kind") != "android_full_worker"
                    or (
                        isinstance(worker["capabilities"].get("resource_gate"), dict)
                        and worker["capabilities"]["resource_gate"].get("admitted") is True
                        and not worker["capabilities"]["resource_gate"].get("reason_code")
                    )
                )
            )
            workers_missing_full_model = sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if str(worker.get("node_id", "")) not in full_model_worker_ids
            )
        else:
            local_models = self._task_worker_capabilities().get("models", [])
            full_model_worker_ids = (
                [self.get_effective_node_id()] if connected and local_models else []
            )
            workers_missing_full_model = (
                [self.get_effective_node_id()] if connected and not local_models else []
            )
        full_model_ready = bool(full_model_worker_ids)
        if not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'):
            readiness_reason = "task_worker_experiment_disabled"
        elif not connected:
            readiness_reason = "task_worker_control_plane_not_connected"
        elif not full_model_ready:
            readiness_reason = "task_worker_full_model_not_advertised"
        else:
            readiness_reason = "task_worker_manual_dispatch_ready"
        runtime.update({
            "phase": "TC-N2.4",
            "experiment_enabled": self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'),
            "experimental_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "auto_provider_selection_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "manual_stage_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "full_model_worker_count": len(full_model_worker_ids),
            "full_model_worker_ids": full_model_worker_ids,
            "workers_missing_full_model": workers_missing_full_model,
            "worker_readiness_reason": readiness_reason,
            "admission_state": (
                "n2_4_experimental_physical_validation_pending"
                if self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected
                else "n2_4_experiment_enabled_not_connected"
                if self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
                else "n2_4_experiment_disabled"
            ),
        })
        return worker_protocol_status(runtime)
