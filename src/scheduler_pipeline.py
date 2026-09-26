"""Distributed inference and pipeline methods mixed into Scheduler."""

from __future__ import annotations

import logging
import threading
import time
import uuid
import base64

from koakuma_engine import Capability, backend_id_for, runtime_supports
from config import (PIPELINE_MODEL_SYNC_TIMEOUT, PIPELINE_RELAY_ENABLED,
                    PIPELINE_RELAY_SEGMENTS)
from relay_segment_client import RelaySegmentClient, RelaySegmentError
from relay_transport import is_loopback_host
from scheduler_types import PreemptState
from torch_runtime import require_torch

logger = logging.getLogger("scheduler")


RELAY_HIDDEN_WIRE_FORMAT = "qlh.relay_hidden.f32.v1"


def _encode_relay_hidden(tensor) -> tuple[str, list[int]]:
    """Encode relay input as the explicit raw-f32 wire contract."""
    torch = require_torch()
    cpu = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return base64.b64encode(cpu.numpy().tobytes()).decode("ascii"), [
        int(size) for size in cpu.shape
    ]


def _decode_relay_hidden(raw: bytes, shape: object):
    """Decode and validate a relay raw-f32 payload on the Torch side."""
    if not isinstance(shape, list) or not shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in shape
    ):
        raise ValueError("relay hidden_shape must be a non-empty positive integer list")
    expected_items = 1
    for size in shape:
        expected_items *= size
    if len(raw) != expected_items * 4:
        raise ValueError(
            f"relay raw f32 length mismatch: bytes={len(raw)} expected={expected_items * 4}"
        )
    torch = require_torch()
    return torch.frombuffer(memoryview(raw), dtype=torch.float32).reshape(shape).clone()


class SchedulerPipelineMixin:
    def request_authoritative_layer_sync(
        self, *, require_distributed: bool = False,
    ) -> bool:
        """让在线 PC 从节点服从主节点当前模型和分层配置。

        从节点显式执行本地模型操作后会暂时退出分层 worker。主节点模型
        加载完成或收到新的分布式请求时，通过这个一次性标记重新取得
        配置权威；普通拓扑刷新仍尊重从节点的临时退出状态。
        """
        if self._effective_role() != "master":
            return False
        with self._layer_config_lock:
            self._authoritative_layer_sync_requests += 1
        try:
            # Explicit distributed requests bypass a single-node capacity plan;
            # ordinary authoritative refreshes preserve the existing policy.
            with self._layer_config_push_lock:
                self._push_layer_config_to_clients_locked(
                    require_distributed=require_distributed,
                )
        finally:
            with self._layer_config_lock:
                self._authoritative_layer_sync_requests = max(
                    0, self._authoritative_layer_sync_requests - 1,
                )
        return True


    def _push_layer_config_to_clients_locked(
        self, *, require_distributed: bool = False,
    ) -> None:
        """
        向所有 TCP 连接的从节点推送其分层配置。

        assignment 携带当前 PyTorch 模型身份和摘要。从节点缺少或模型不一致时
        先从主节点同步模型，校验成功并加载层范围后再返回 ready ACK。
        """
        if not self._tcp_server or not self._tcp_server._running:
            return
        get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
        connected_ids = (
            get_client_ids()
            if callable(get_client_ids)
            else list(getattr(self._tcp_server, "clients", {}).keys())
        )
        if not connected_ids:
            return

        with self._nodes_lock:
            # 这是旧版 PyTorch LAYER_CONFIG 线路。Android 的 llama.cpp
            # worker 只能走 v3 task_worker stage offer，不能收到这份配置。
            releasable_legacy_ids = {
                node_id for node_id, node in self.nodes.items()
                if node_id in connected_ids
                and node_id != self.get_effective_node_id()
                and getattr(node, "node_type", "pc") == "pc"
            }

        # A healthy full-model Task Worker has priority over legacy automatic
        # layer assignment. Keep its local model intact and release any stale
        # layer reservation instead of reassigning a subset of layers.
        full_worker_release_ids = (
            releasable_legacy_ids & self._task_worker_full_model_ids()
        )
        layer_releasable_worker_ids = releasable_legacy_ids - full_worker_release_ids

        with self._layer_config_lock:
            if full_worker_release_ids:
                self._pipeline_worker_opt_out.update(full_worker_release_ids)
            authoritative_sync = bool(
                self._authoritative_layer_sync_requests
            )
            reenabled_nodes = (
                self._pipeline_worker_opt_out & layer_releasable_worker_ids
                if authoritative_sync else set()
            )
            if reenabled_nodes:
                self._pipeline_worker_opt_out.difference_update(reenabled_nodes)
        if reenabled_nodes:
            logger.info(
                "主节点权威模型同步重新启用分层 worker: %s",
                sorted(reenabled_nodes),
            )

        with self._layer_config_lock:
            self._layer_config_generation = max(
                self._layer_config_generation + 1,
                time.time_ns(),
            )
            generation = self._layer_config_generation
        config_id = uuid.uuid4().hex

        model_info = self._get_active_pipeline_model_info()
        master_sha256 = model_info.get("model_sha256", "")
        model_id = model_info.get("model_id", "")
        model_type = model_info.get("model_type", "")
        if not master_sha256 or not model_id or model_type not in {"qwen", "qwen2"}:
            releases = {
                node_id: {
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": generation,
                    "release": True,
                }
                for node_id in releasable_legacy_ids
            }
            self._publish_layer_configs(releases)
            logger.warning("主节点尚未加载可校验的 PyTorch 模型，暂不推送层配置")
            return

        distributed_only = bool(
            require_distributed
            or getattr(self._host, "is_pipeline_prepared", False)
        )
        capacity_plan = None
        if distributed_only:
            manual_override = bool(self._runtime_layer_override)
            if manual_override and not require_distributed:
                layer_info = self.get_layer_assignments()
                capacity_plan = self._build_manual_pipeline_capacity_plan(
                    layer_info.get("assignments", [])
                )
            else:
                if manual_override:
                    logger.info(
                        "分布式请求忽略单机手动分层覆盖，改用多节点容量求解"
                    )
                eligible_node_ids = set(layer_releasable_worker_ids)
                eligible_node_ids.update({"master", self.get_effective_node_id()})
                capacity_plan = self.get_pipeline_capacity_plan(
                    eligible_node_ids,
                    require_distributed=require_distributed,
                )
            if not capacity_plan.get("admitted"):
                releases = {
                    node_id: {
                        "node_id": node_id,
                        "config_id": config_id,
                        "generation": generation,
                        "release": True,
                        "abort": True,
                        "reason_code": capacity_plan.get(
                            "reason_code", "pipeline_capacity_rejected"
                        ),
                    }
                for node_id in releasable_legacy_ids
                }
                with self._layer_config_lock:
                    self._pipeline_load_transaction = {
                        "config_id": config_id,
                        "generation": generation,
                        "phase": "rejected",
                        "plan": dict(capacity_plan),
                        "prepared_nodes": set(),
                        "reason_code": capacity_plan.get("reason_code", ""),
                    }
                    self._active_pipeline_capacity_plan = None
                self._publish_layer_configs(releases)
                logger.warning(
                    "集群容量准入拒绝流水线加载: reason=%s",
                    capacity_plan.get("reason_code", "unknown"),
                )
                return
            layer_info = {
                "assignments": capacity_plan.get("assignments", []),
            }
        else:
            layer_info = self.get_layer_assignments()
        assignments = {}
        from config import API_PORT

        for a in layer_info["assignments"]:
            nid = a["node_id"]
            if (
                nid in {"master", self.get_effective_node_id()}
                or nid not in layer_releasable_worker_ids
            ):
                continue

            # 新一轮配置开始后，旧 ACK 立即失效。
            self._clear_layer_config_state(nid)

            assignments[nid] = {
                "node_id": nid,
                "config_id": config_id,
                "generation": generation,
                "start_layer": a["start_layer"],
                "end_layer": a["end_layer"],
                "has_embedding": a.get("has_embedding", False),
                "has_lm_head": a.get("has_lm_head", False),
                "model_id": model_id,
                "model_sha256": master_sha256,
                "model_type": model_type,
                "total_layers": int(model_info["total_layers"]),
                "master_quant_type": model_info.get("quant_type", ""),
                "engine": (
                    "relay_middle"
                    if self._relay_segment_for_worker(nid) is not None
                    else "pytorch"
                ),
                "sync_policy": (
                    "master_authoritative" if authoritative_sync else "normal"
                ),
                "authoritative_sync": authoritative_sync,
                "master_api_port": API_PORT,
            }
            relay_segment = self._relay_segment_for_worker(nid)
            if relay_segment is not None:
                assignments[nid]["relay_segment"] = relay_segment
            if capacity_plan is not None:
                assignments[nid].update({
                    "phase": "prepare",
                    "assignment_manifest": True,
                    "plan_id": capacity_plan.get("plan_id", ""),
                    "required_bytes": int(a.get("required_bytes", 0) or 0),
                    "capacity_bytes": int(a.get("capacity_bytes", 0) or 0),
                    "capacity_source": a.get("capacity_source", ""),
                    "safety_margin": capacity_plan.get("safety_margin", 1.0),
                })

        releases = {
            node_id: {
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": generation,
                    "release": True,
            }
            for node_id in (layer_releasable_worker_ids | full_worker_release_ids)
            if node_id not in assignments
        }
        configs = {**assignments, **releases}
        if capacity_plan is not None:
            if not assignments:
                capacity_plan = dict(capacity_plan)
                capacity_plan.update({
                    "status": "rejected",
                    "admitted": False,
                    "reason_code": "pipeline_capacity_workers_unavailable",
                    "assignments": [],
                    "transaction_phase": "rejected",
                })
            with self._layer_config_lock:
                self._pipeline_load_transaction = {
                    "config_id": config_id,
                    "generation": generation,
                    "phase": "preparing" if assignments else "rejected",
                    "plan": dict(capacity_plan),
                    "worker_ids": set(assignments),
                    "prepared_nodes": set(),
                }
                self._active_pipeline_capacity_plan = None
        self._publish_layer_configs(configs)
        if assignments:
            logger.info(
                f"分层配置已推送到 {len(assignments)} 个从节点，"
                f"等待加载 ACK (config_id={config_id}, generation={generation})"
            )
        else:
            logger.warning("没有可用的从节点接收分层配置")


    def _publish_layer_configs(self, configs: dict[str, dict]) -> None:
        """Register every assignment/release before sending so both are retried."""
        if not configs:
            return
        with self._layer_config_lock:
            for node_id, config in configs.items():
                self._layer_config_pushed.discard(node_id)
                self._layer_config_acks.pop(node_id, None)
                self._layer_config_expected[node_id] = dict(config)
                self._layer_config_retry_state[node_id] = {
                    "attempts": 1,
                    "next_retry": time.monotonic() + 5.0,
                }
        self._start_layer_config_retry_monitor()
        for node_id, config in configs.items():
            try:
                self._tcp_server.send_layer_config(node_id, config)
            except Exception:
                logger.warning(
                    "分层配置首次发送失败，将由退避线程重试: node=%s",
                    node_id,
                    exc_info=True,
                )


    def _clear_layer_config_state(self, node_id: str) -> None:
        """清除节点的层配置期望、ACK 和 ready 状态。"""
        with self._layer_config_lock:
            self._layer_config_pushed.discard(node_id)
            self._layer_config_expected.pop(node_id, None)
            self._layer_config_acks.pop(node_id, None)
            self._layer_config_retry_state.pop(node_id, None)


    def _abort_pipeline_load_transaction(
        self, config_id: str, reason_code: str, reason: str = "",
    ) -> None:
        """Abort one capacity transaction and release every worker atomically."""
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if not transaction or transaction.get("config_id") != config_id:
                return
            worker_ids = set(transaction.get("worker_ids", set()))
            self._layer_config_generation = max(
                self._layer_config_generation + 1,
                time.time_ns(),
            )
            generation = self._layer_config_generation
            transaction["phase"] = "aborted"
            transaction["reason_code"] = reason_code
            transaction["reason"] = reason
            self._active_pipeline_capacity_plan = None
            model_id = str(transaction.get("plan", {}).get("model_id", "") or "")
        abort_materialization = getattr(
            self._host, "abort_pipeline_materialization", None
        )
        if callable(abort_materialization):
            try:
                abort_materialization()
                self._host.model_loaded = False
            except Exception:
                logger.warning("主节点回滚流水线层段失败", exc_info=True)
        abort_id = uuid.uuid4().hex
        configs = {
            node_id: {
                "node_id": node_id,
                "config_id": abort_id,
                "generation": generation,
                "release": True,
                "abort": True,
                "aborted_config_id": config_id,
                "model_id": model_id,
                "reason_code": reason_code,
            }
            for node_id in worker_ids
        }
        self._publish_layer_configs(configs)
        logger.error(
            "流水线加载事务已中止: config=%s reason_code=%s reason=%s",
            config_id, reason_code, reason,
        )


    def _commit_pipeline_load_transaction(self, config_id: str) -> None:
        """Materialize the local segment, then publish commit to all workers."""
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                not transaction
                or transaction.get("config_id") != config_id
                or transaction.get("phase") != "preparing"
            ):
                return
            plan = dict(transaction.get("plan", {}))
            worker_ids = set(transaction.get("worker_ids", set()))
            expected = {
                node_id: dict(self._layer_config_expected.get(node_id, {}))
                for node_id in worker_ids
            }
            transaction["phase"] = "committing_local"

        master_ids = {"master", self.get_effective_node_id()}
        local_assignment = next((
            item for item in plan.get("assignments", [])
            if item.get("node_id") in master_ids
        ), None)
        try:
            prepare_tokenizer = getattr(self._host, "prepare_pipeline_tokenizer", None)
            if callable(prepare_tokenizer):
                prepare_tokenizer()
            if local_assignment is not None:
                self._host.load_layer_range(
                    int(local_assignment["start_layer"]),
                    int(local_assignment["end_layer"]),
                    has_embedding=bool(local_assignment.get("has_embedding")),
                    has_lm_head=bool(local_assignment.get("has_lm_head")),
                    model_path=getattr(self._host, "_full_model_path", None),
                    quant_type=getattr(self._host, "quant_type", None),
                    total_layers=int(plan.get("total_layers", 0) or 0),
                    model_id=str(plan.get("model_id", "") or ""),
                )
        except Exception as exc:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_local_commit_failed", str(exc)
            )
            return

        commit_configs = {}
        for node_id, item in expected.items():
            if not item or item.get("release"):
                continue
            item["phase"] = "commit"
            commit_configs[node_id] = item
        if not commit_configs:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_commit_workers_missing",
                "prepared worker set disappeared before commit",
            )
            return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if transaction and transaction.get("config_id") == config_id:
                transaction["phase"] = "committing"
        self._publish_layer_configs(commit_configs)
        logger.info(
            "流水线 prepare 全部通过，已下发 commit: config=%s workers=%s",
            config_id, sorted(commit_configs),
        )


    def _invalidate_worker_layer_ready(
        self, node_id: str, config_id: str, reason: str,
    ) -> bool:
        """Revoke one worker's ready ACK and immediately resend its generation."""
        with self._layer_config_lock:
            expected = self._layer_config_expected.get(node_id)
            if not expected or expected.get("config_id") != config_id:
                return False
            assignment = dict(expected)
            self._layer_config_pushed.discard(node_id)
            self._layer_config_acks[node_id] = {
                "node_id": node_id,
                "config_id": config_id,
                "status": "error",
                "error": reason,
            }
            state = self._layer_config_retry_state.setdefault(
                node_id, {"attempts": 0, "next_retry": 0.0},
            )
            state["attempts"] = int(state.get("attempts", 0)) + 1
            state["next_retry"] = time.monotonic() + 5.0

        try:
            if self._tcp_server and self._tcp_server._running:
                self._tcp_server.send_layer_config(node_id, assignment)
                logger.warning(
                    "worker 层配置已失效，立即重发: node=%s config=%s reason=%s",
                    node_id, config_id, reason,
                )
        except Exception:
            logger.warning(
                "worker 层配置立即重发失败，将由退避线程重试: node=%s",
                node_id, exc_info=True,
            )
        return True


    def _start_layer_config_retry_monitor(self) -> None:
        if (self._layer_config_retry_thread is not None
                and self._layer_config_retry_thread.is_alive()):
            return
        self._layer_config_retry_thread = threading.Thread(
            target=self._layer_config_retry_loop,
            name="layer-config-retry",
            daemon=True,
        )
        self._layer_config_retry_thread.start()


    def _layer_config_retry_loop(self) -> None:
        """重发未确认配置；节点错误或 ACK 丢失不能永久禁用流水线。"""
        while self._running and self._effective_role() == "master":
            self._retry_pending_layer_configs()
            time.sleep(1.0)


    def _retry_pending_layer_configs(self, now: float = None) -> int:
        """执行一次层配置重发扫描，返回成功发出的配置数量。"""
        now = time.monotonic() if now is None else now
        pending = []
        with self._layer_config_lock:
            for node_id, expected in self._layer_config_expected.items():
                if node_id in self._layer_config_pushed:
                    continue
                state = self._layer_config_retry_state.setdefault(
                    node_id, {"attempts": 0, "next_retry": now}
                )
                if now < state.get("next_retry", 0):
                    continue
                state["attempts"] = int(state.get("attempts", 0)) + 1
                delay = min(60.0, 5.0 * (2 ** min(state["attempts"] - 1, 4)))
                state["next_retry"] = now + delay
                pending.append((node_id, dict(expected), state["attempts"]))

        connected = set()
        if self._tcp_server and self._tcp_server._running:
            try:
                connected = set(self._tcp_server.get_client_ids())
            except Exception:
                logger.debug("读取层配置重试节点失败", exc_info=True)
        sent = 0
        for node_id, assignment, attempt in pending:
            if node_id not in connected:
                continue
            try:
                self._tcp_server.send_layer_config(node_id, assignment)
                sent += 1
                logger.info(
                    "重发分层配置: node=%s config=%s attempt=%d",
                    node_id, assignment.get("config_id", ""), attempt,
                )
            except Exception:
                logger.warning(
                    "重发分层配置失败: node=%s attempt=%d",
                    node_id, attempt, exc_info=True,
                )
        return sent


    def _get_master_model_sha256(self) -> str:
        """
        获取主节点当前加载模型的 SHA256。

        对当前已加载或已显式准备的 PyTorch Safetensors/BIN 模型计算摘要。
        llama.cpp/GGUF 不支持层拆分，不得作为流水线模型基准。
        """
        from model_sync import compute_model_sha256

        mgr = self._host
        if not mgr or not runtime_supports(mgr, Capability.FORWARD_LAYERS):
            return ""

        get_descriptor = getattr(mgr, "get_pipeline_descriptor", None)
        if callable(get_descriptor):
            try:
                cached = str((get_descriptor() or {}).get("model_sha256", ""))
                if cached:
                    return cached
            except Exception:
                logger.debug("读取流水线描述器摘要失败", exc_info=True)

        model_path = (
            getattr(mgr, '_full_model_path', '')
            or getattr(mgr, '_model_path', '')
            or ''
        )
        if not model_path or not os.path.isdir(model_path):
            return ""

        try:
            return compute_model_sha256(model_path)
        except Exception:
            logger.warning("计算主节点 PyTorch 模型摘要失败", exc_info=True)
            return ""


    def _clear_pipeline_runtime_state(self, task_id: str) -> None:
        """清理主节点侧单个流水线任务的等待结果与链路 ACK 状态。"""
        if not task_id:
            return
        self._close_relay_segment_client(task_id)
        prefix = f"{task_id}:"
        with self._pipeline_lock:
            self._pipeline_active_tasks.discard(task_id)
            for key in list(self._pipeline_results):
                if key.startswith(prefix):
                    self._pipeline_results.pop(key, None)
            for key in list(self._pipeline_events):
                if key.startswith(prefix):
                    self._pipeline_events.pop(key, None)
            self._chain_ack_state.pop(task_id, None)
            self._pipeline_task_contracts.pop(task_id, None)


    def has_pipeline_worker_reservation(self) -> bool:
        """Return whether this PC is reserved for a master's layer pipeline."""
        with self._layer_config_lock:
            return bool(self._pipeline_worker_reserved)


    def release_pipeline_worker_for_local_model(self) -> bool:
        """Opt this client out before an explicit local model operation."""
        with self._layer_config_lock:
            self._pipeline_worker_reserved = False
            self._pipeline_worker_opted_out = True
            self._active_layer_config = None
            self._last_layer_config_ack_payload = None
            self._local_pipeline_steps.clear()
        if self._effective_role() != "client":
            return True
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "_running", False):
            logger.warning("本地模型切换时主节点未连接，已仅清理本地分层预留")
            return False
        try:
            from transport_port import MessageType

            client.send_data(
                {
                    "node_id": self.get_effective_node_id(),
                    "reason": "explicit_local_model_change",
                },
                MessageType.LAYER_WORKER_OPT_OUT,
            )
            return True
        except Exception:
            logger.warning("通知主节点退出分层 worker 失败", exc_info=True)
            return False


    def _begin_local_pipeline_task(self, task_id: str) -> None:
        """Track local work so layer reconfiguration cannot replace an active model."""
        if not task_id:
            return
        with self._layer_config_lock:
            self._active_pipeline_task_ids.add(task_id)


    def _finish_local_pipeline_task(self, task_id: str) -> None:
        """Release local work state and apply the newest deferred layer config."""
        pending = None
        with self._layer_config_lock:
            self._active_pipeline_task_ids.discard(task_id)
            self._local_pipeline_steps.pop(task_id, None)
            if not self._active_pipeline_task_ids and self._pending_layer_config is not None:
                pending_config_id = str(
                    self._pending_layer_config[1].get("config_id", "")
                )
                if not pending_config_id or pending_config_id not in self._layer_config_inflight:
                    pending = self._pending_layer_config
                    self._pending_layer_config = None
        self._close_relay_segment_client(task_id)
        if pending is not None:
            client_id, data = pending
            logger.info("当前流水线任务已结束，开始应用延后的分层配置")
            self._schedule_layer_config(client_id, data)


    def _relay_segment_client_for_task(
        self, task_id: str, spec: dict[str, object], *, n_embd: int,
    ) -> RelaySegmentClient:
        """Reuse one relay TCP session for all steps in a pipeline task."""
        cache = getattr(self, "_relay_segment_clients", None)
        if cache is None:
            cache = {}
            self._relay_segment_clients = cache
        key = (
            str(spec["host"]), int(spec["port"]), int(n_embd),
            float(spec["timeout"]),
        )
        current = cache.get(task_id)
        if current is not None and current[0] == key:
            return current[1]
        if current is not None:
            try:
                current[1].close()
            except Exception:
                logger.debug("close stale relay session failed", exc_info=True)
        client = RelaySegmentClient(
            str(spec["host"]), int(spec["port"]), n_embd=int(n_embd),
            role="middle", timeout=float(spec["timeout"]),
        )
        cache[task_id] = (key, client)
        return client


    def _close_relay_segment_client(self, task_id: str) -> None:
        cache = getattr(self, "_relay_segment_clients", None)
        if not cache:
            return
        current = cache.pop(task_id, None)
        if current is not None:
            try:
                current[1].close()
            except Exception:
                logger.debug("close relay session failed: task=%s", task_id, exc_info=True)


    def _close_all_relay_segment_clients(self) -> None:
        cache = getattr(self, "_relay_segment_clients", None)
        if not cache:
            return
        for task_id in list(cache):
            self._close_relay_segment_client(task_id)


    def _mark_local_pipeline_cancelled(self, task_id: str) -> None:
        if not task_id:
            return
        with self._layer_config_lock:
            if task_id not in self._local_pipeline_cancelled:
                self._local_pipeline_cancelled.add(task_id)
                self._local_pipeline_cancelled_order.append(task_id)
            while len(self._local_pipeline_cancelled_order) > 4096:
                expired = self._local_pipeline_cancelled_order.popleft()
                self._local_pipeline_cancelled.discard(expired)


    def _fail_pending_pipeline_results_for_node(self, node_id: str,
                                                reason: str) -> None:
        """节点断连/不可用时，立即失败所有正在等待该节点的流水线步骤。"""
        if not node_id:
            return
        failed = []
        with self._pipeline_lock:
            for key, event in list(self._pipeline_events.items()):
                try:
                    task_id, waiting_node_id = key.split(":", 1)
                except ValueError:
                    continue
                if waiting_node_id != node_id:
                    continue
                self._pipeline_results[key] = {
                    "task_id": task_id,
                    "node_id": node_id,
                    "error": reason,
                    "step": -1,
                }
                event.set()
                failed.append(task_id)
        if failed:
            logger.warning(
                "节点 %s 不可用，已唤醒 %d 个流水线等待任务: %s",
                node_id, len(failed), ", ".join(failed),
            )


    def _handle_chain_forward_ack(self, client_id: str, msg: dict) -> None:
        """主节点：记录链式转发每跳 ACK/错误，并在错误时立即唤醒流水线。"""
        data = msg.get("data", {})
        task_id = str(data.get("task_id", "") or "")
        try:
            step = int(data.get("step", -1))
        except (TypeError, ValueError):
            logger.warning("丢弃 step 无效的链式 ACK: task=%s", task_id or "-")
            return
        status = data.get("status", "received")
        error = data.get("error", "")
        config_id = str(data.get("config_id", ""))
        reporter_node_id = str(data.get("node_id", client_id))
        target_node_id = data.get("target_node_id", "")
        node_id = target_node_id if status in ("sent", "error") and target_node_id else reporter_node_id

        if not task_id or not node_id:
            return
        if reporter_node_id != client_id:
            logger.warning(
                "丢弃来源不一致的链式 ACK: connection=%s payload=%s",
                client_id, reporter_node_id,
            )
            return
        if status not in {"sent", "received", "error"}:
            logger.warning("丢弃未知链式 ACK 状态: %s", status)
            return

        now = time.time()
        with self._pipeline_lock:
            if task_id not in self._pipeline_active_tasks:
                return
            contract = self._pipeline_task_contracts.get(task_id, {})
            worker_ids = list(contract.get("worker_ids", []))
            expected_nodes = set(worker_ids)
            if (step != contract.get("current_step")
                    or config_id != contract.get("config_id")
                    or reporter_node_id not in expected_nodes
                    or node_id not in expected_nodes):
                logger.warning(
                    "丢弃不符合执行契约的链式 ACK: task=%s step=%s "
                    "node=%s config=%s",
                    task_id, step, node_id, config_id,
                )
                return
            if status == "sent" or (status == "error" and target_node_id):
                reporter_index = worker_ids.index(reporter_node_id)
                expected_target = (
                    worker_ids[reporter_index + 1]
                    if reporter_index + 1 < len(worker_ids) else ""
                )
                if not target_node_id or target_node_id != expected_target:
                    logger.warning(
                        "丢弃非相邻链路 %s ACK: task=%s reporter=%s "
                        "target=%s expected=%s",
                        status, task_id, reporter_node_id,
                        target_node_id, expected_target,
                    )
                    return
            elif status == "received":
                receiver_index = worker_ids.index(reporter_node_id)
                expected_source = (
                    worker_ids[receiver_index - 1] if receiver_index > 0 else ""
                )
                if str(data.get("from_node_id", "")) != expected_source:
                    logger.warning(
                        "丢弃非相邻链路 received ACK: task=%s receiver=%s "
                        "source=%s expected=%s",
                        task_id, reporter_node_id,
                        data.get("from_node_id", ""), expected_source,
                    )
                    return
            task_state = self._chain_ack_state.setdefault(task_id, {})
            step_state = task_state.setdefault(step, {})
            existing = step_state.get(node_id, {})
            new_state = {
                "status": status,
                "error": error,
                "from_node_id": data.get("from_node_id", reporter_node_id),
                "target_node_id": target_node_id or node_id,
                "reporter_node_id": reporter_node_id,
                "updated_at": now,
            }
            if status == "sent":
                new_state["sent_at"] = now
                if existing.get("status") == "received":
                    # 下游 ACK 可能比上游 sent 回报更早到达；不要把
                    # received 状态倒退为 sent。
                    new_state["status"] = "received"
                    new_state["acked_at"] = existing.get("acked_at", existing.get("updated_at", now))
                    new_state["error"] = existing.get("error", "")
            elif status == "received":
                new_state["acked_at"] = now
                if existing.get("sent_at"):
                    new_state["sent_at"] = existing["sent_at"]
            step_state[node_id] = new_state

        if status == "error" or error:
            message = error or f"链式转发节点 {node_id} 返回错误 ACK"
            logger.error(
                "链式转发 ACK 错误: task=%s step=%s node=%s error=%s",
                task_id, step, node_id, message,
            )
            self._set_pipeline_result_error(task_id, node_id, message, step)


    def _get_chain_ack_failure(self, task_id: str, step: int,
                               expected_node_ids: list,
                               ack_timeout: float) -> Optional[dict]:
        """检测已发送但迟迟未被下游确认接收的链式转发。"""
        if not task_id or not expected_node_ids:
            return None
        now = time.time()
        with self._pipeline_lock:
            step_state = self._chain_ack_state.get(task_id, {}).get(step, {})
            for node_id in expected_node_ids:
                state = step_state.get(node_id)
                if not state:
                    continue
                if state.get("status") == "error" or state.get("error"):
                    return {
                        "task_id": task_id,
                        "node_id": node_id,
                        "error": state.get("error") or f"链式转发到 {node_id} 失败",
                        "step": step,
                    }
                if state.get("status") == "sent":
                    sent_at = state.get("sent_at", state.get("updated_at", now))
                    if now - sent_at >= ack_timeout:
                        return {
                            "task_id": task_id,
                            "node_id": node_id,
                            "error": (
                                f"链式转发到 {node_id} 未收到接收 ACK "
                                f"({ack_timeout:.1f}s)"
                            ),
                            "step": step,
                        }
        return None


    def handle_infer_forward(self, client_id: str, msg: dict) -> None:
        """
        处理从节点转发的推理请求（统一流水线调度）。

        主节点收到 INFER_FORWARD 后:
          1. 创建推理任务
          2. 通过 run_pipeline_safe() 统一调度:
             - 流水线节点就绪 → 分布式流水线推理
             - 流水线节点未就绪 → 自动回退到主节点全模型推理
          3. 将结果通过 INFER_RESULT 回传给请求方

        路径 A 和路径 B 已统一 — 无论请求来自 HTTP /api/chat 还是
        TCP INFER_FORWARD，都走同一套 run_pipeline_safe() 调度。
        """
        data = msg.get("data", {})
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        routing_preference = str(data.get("routing_preference", "auto") or "auto")
        show_thinking = data.get("show_thinking", False)
        # ★ 2026-09-19：主节点收到转发请求后同样透传深度思考**开关**。
        enable_thinking = data.get("enable_thinking")
        session_id = data.get("session_id")
        messages = data.get("messages")
        request_id = data.get("request_id")   # L5: 链路追踪
        forward_request_id = str(data.get("forward_request_id", ""))
        cancel_key = forward_request_id or f"legacy_{uuid.uuid4().hex}"

        import threading as _thr

        if not self._forward_infer_slots.acquire(blocking=False):
            self._send_infer_result(
                client_id, "", "", {},
                forward_request_id=forward_request_id,
                status="error",
                error="主节点转发请求已达并发上限，请稍后重试",
            )
            return

        cancel_event = threading.Event()
        with self._forward_cancel_lock:
            self._forward_cancel_events[(client_id, cancel_key)] = cancel_event

        def _run_inference():
            task_id = ""
            try:
                task_id = self.start_infer_task(prompt, request_id=request_id)
                logger.info(
                    "event=infer_forward_recv task_id=%s request_id=%s "
                    "client_id=%s prompt_len=%d max_tokens=%d",
                    task_id, request_id or "-", client_id, len(prompt), max_new_tokens,
                )

                # ★ 统一流水线调度（替代原来的 mgr.chat() 全模型直调）
                pipeline_result = self.run_pipeline_safe(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    session_id=session_id,
                    messages=messages,
                    show_thinking=show_thinking,
                    enable_thinking=enable_thinking,
                    _require_distributed=(routing_preference == "distributed_required"),
                    _force_distributed_assignment=(routing_preference != "local_only"),
                    _cancel_event=cancel_event,
                )

                content = pipeline_result.get("response", "")
                thinking_content = pipeline_result.get("thinking")
                error = pipeline_result.get("error")
                metrics = pipeline_result.get("metrics", {})

                if error:
                    self.fail_infer_task(task_id, error)
                    logger.warning(
                        f"⚠️ 流水线推理失败 → {client_id}: task={task_id}, "
                        f"error={error}"
                    )
                    self._send_infer_result(
                        client_id, task_id, content,
                        {**metrics, "error": error},
                        thinking_content=thinking_content,
                        forward_request_id=forward_request_id,
                        status="error",
                        error=error,
                    )
                    return

                # 保存到对话历史（主节点侧）
                if content:
                    try:
                        import local_store
                        local_store.save_local_conversation_turn(
                            session_id=session_id or "default",
                            user_message=prompt,
                            assistant_message=content,
                            metrics=metrics,
                            operation_id=f"pipeline:{task_id}",
                        )
                    except Exception:
                        pass

                if not metrics.get("distributed_used"):
                    try:
                        self.record_task_complete(success=True)
                    except Exception:
                        pass

                self.complete_infer_task(task_id, content, metrics)
                self._send_infer_result(
                    client_id, task_id, content, metrics,
                    thinking_content=thinking_content,
                    forward_request_id=forward_request_id,
                )
                logger.info(
                    f"✅ 推理完成 → {client_id}: task={task_id}, "
                    f"len={len(content)}, engine={metrics.get('engine', '?')}"
                )

            except Exception as e:
                if task_id:
                    self.fail_infer_task(task_id, str(e))
                logger.error(f"转发推理执行失败: {e}", exc_info=True)
                self._send_infer_result(
                    client_id, task_id, "",
                    {"error": str(e)},
                    forward_request_id=forward_request_id,
                    status="error",
                    error=str(e),
                )
            finally:
                with self._forward_cancel_lock:
                    self._forward_cancel_events.pop((client_id, cancel_key), None)
                self._forward_infer_slots.release()

        _thr.Thread(
            target=_run_inference,
            name=f"infer-{client_id}-{cancel_key[-8:]}",
            daemon=True,
        ).start()


    def _send_infer_result(self, client_id: str, task_id: str,
                           content: str, metrics: dict = None,
                           thinking_content: str = None,
                           followups: list = None,
                           forward_request_id: str = "",
                           status: str = "ok",
                           error: str = "") -> None:
        """向从节点回传推理结果"""
        if self._tcp_server and self._tcp_server._running:
            try:
                from transport_port import MessageType
                result_data = {
                    "task_id": task_id,
                    "forward_request_id": forward_request_id,
                    "status": status,
                    "content": content,
                    "metrics": metrics or {},
                }
                if error:
                    result_data["error"] = error
                if thinking_content:
                    result_data["thinking_content"] = thinking_content
                if followups:
                    result_data["followups"] = followups
                self._tcp_server.send_to_client(
                    client_id,
                    result_data,
                    msg_type=MessageType.INFER_RESULT,
                )
            except Exception as e:
                logger.error(f"回传推理结果失败 ({client_id}): {e}")


    def _schedule_layer_config(self, client_id: str, data: dict) -> None:
        config_id = str(data.get("config_id", "")) if isinstance(data, dict) else ""
        incoming_phase = str(
            data.get("phase", "commit") or "commit"
        ) if isinstance(data, dict) else "commit"
        authoritative_sync = bool(
            isinstance(data, dict) and data.get("authoritative_sync")
        )
        resend_opt_out = False
        authoritative_opt_in = False
        with self._layer_config_lock:
            if (
                authoritative_sync
                and self._pipeline_worker_opted_out
                and isinstance(data, dict)
                and not data.get("release")
            ):
                self._pipeline_worker_opted_out = False
                authoritative_opt_in = True
            if (self._pipeline_worker_opted_out
                    and isinstance(data, dict)
                    and not data.get("release")):
                resend_opt_out = True
            if resend_opt_out:
                payload = None
                receive_sequence = 0
                generation = 0
            else:
                cached = self._last_layer_config_ack_payload
                if (config_id and cached
                        and cached.get("config_id") == config_id
                        and str(cached.get("phase", "commit") or "commit")
                        == incoming_phase
                        and config_id not in self._layer_config_inflight):
                    payload = dict(cached)
                else:
                    payload = None
            if resend_opt_out:
                pass
            elif config_id and config_id in self._layer_config_inflight:
                return
            if resend_opt_out:
                pass
            elif payload is not None:
                receive_sequence = 0
                generation = 0
            else:
                self._layer_config_receive_sequence += 1
                receive_sequence = self._layer_config_receive_sequence
                try:
                    generation = int(data.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    generation = 0
            if resend_opt_out:
                pass
            elif payload is not None:
                pass
            elif (self._latest_layer_config_generation
                  and generation < self._latest_layer_config_generation):
                logger.info(
                    "忽略过期分层配置: config=%s generation=%s latest=%s",
                    config_id,
                    generation,
                    self._latest_layer_config_generation,
                )
                return
            else:
                self._latest_layer_config_receive_sequence = receive_sequence
                self._latest_layer_config_generation = max(
                    self._latest_layer_config_generation,
                    generation,
                )
                if isinstance(data, dict) and not data.get("release"):
                    self._pipeline_worker_reserved = True
                if config_id:
                    self._layer_config_inflight.add(config_id)

        if authoritative_opt_in:
            logger.info(
                "收到主节点权威模型配置，自动重新加入分层 worker: config=%s",
                config_id,
            )
        if resend_opt_out:
            logger.info(
                "本设备已选择本地模型，拒绝分层配置并重发退出请求: config=%s",
                config_id,
            )
            self.release_pipeline_worker_for_local_model()
            return
        if payload is not None:
            self._send_layer_config_ack(payload)
            return

        def _load() -> None:
            try:
                self._handle_layer_config(
                    client_id,
                    data,
                    receive_sequence=receive_sequence,
                    generation=generation,
                )
            finally:
                if config_id:
                    with self._layer_config_lock:
                        self._layer_config_inflight.discard(config_id)
                        pending_matches = bool(
                            self._pending_layer_config
                            and str(self._pending_layer_config[1].get(
                                "config_id", ""
                            )) == config_id
                        )
                    if pending_matches:
                        with self._layer_config_lock:
                            can_apply = not self._active_pipeline_task_ids
                            pending = (
                                self._pending_layer_config
                                if can_apply else None
                            )
                            if pending is not None:
                                self._pending_layer_config = None
                        if pending is not None:
                            pending_client_id, pending_data = pending
                            self._schedule_layer_config(
                                pending_client_id, pending_data
                            )

        threading.Thread(
            target=_load,
            name=f"layer-config-{config_id[-8:] or 'legacy'}",
            daemon=True,
        ).start()


    def _handle_layer_config(
        self,
        client_id: str,
        data: dict,
        *,
        receive_sequence: int = None,
        generation: int = None,
    ) -> None:
        with self._layer_execution_lock:
            if receive_sequence is not None:
                with self._layer_config_lock:
                    if (receive_sequence
                            != self._latest_layer_config_receive_sequence):
                        logger.info(
                            "跳过已被新消息取代的分层配置: config=%s sequence=%s latest=%s",
                            data.get("config_id", ""),
                            receive_sequence,
                            self._latest_layer_config_receive_sequence,
                        )
                        return
                    if (self._latest_layer_config_generation
                            and generation
                            < self._latest_layer_config_generation):
                        return
            self._handle_layer_config_locked(
                client_id,
                data,
                receive_sequence=receive_sequence,
                generation=generation,
            )


    def _handle_layer_config_locked(
        self,
        client_id: str,
        data: dict,
        *,
        receive_sequence: int = None,
        generation: int = None,
    ) -> None:
        """
        从节点：收到主节点推送的分层配置 → 加载指定层范围。

        新版主节点只发送本节点的 assignment；同时兼容旧版
        ``{node_id: assignment}`` 外层映射。加载结束后必须发送 ACK，
        主节点收到当前 config_id 的成功 ACK 才会将本节点视为 ready。
        """
        with self._layer_config_lock:
            if self._active_pipeline_task_ids:
                self._pending_layer_config = (client_id, dict(data))
                logger.warning(
                    "本节点仍有流水线任务执行中，分层配置已延后: active=%s",
                    sorted(self._active_pipeline_task_ids),
                )
                return

        node_id = self.get_effective_node_id()
        ack_generation = (
            data.get("generation", 0) if isinstance(data, dict) else 0
        )
        if isinstance(data, dict) and data.get("release"):
            target_node_id = str(data.get("node_id", node_id))
            if target_node_id != node_id:
                logger.warning(
                    "忽略目标不匹配的分层释放: target=%s local=%s",
                    target_node_id, node_id,
                )
                return
            with self._layer_config_lock:
                self._pipeline_worker_reserved = False
                self._active_layer_config = None
                self._last_layer_config_ack_payload = None
                self._local_pipeline_steps.clear()
                aborted_config_id = str(data.get("aborted_config_id", "") or "")
                if aborted_config_id:
                    self._prepared_layer_configs.pop(aborted_config_id, None)
            if data.get("abort"):
                abort_materialization = getattr(
                    self._host, "abort_pipeline_materialization", None
                )
                if callable(abort_materialization):
                    abort_materialization()
                self._host.model_loaded = False
                try:
                    from model_sync import remove_pipeline_assignment_cache

                    model_id = str(data.get("model_id", "") or "")
                    if model_id and aborted_config_id:
                        remove_pipeline_assignment_cache(
                            model_id, aborted_config_id, node_id,
                        )
                except Exception:
                    logger.warning("清理已中止的 assignment 缓存失败", exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": str(data.get("config_id", "")),
                "generation": ack_generation,
                "status": "released",
                "release": True,
                "timestamp": time.time(),
            })
            logger.info("主节点已释放本设备的分层 worker 预留")
            return
        # TP 孤岛网关节点不参与 PyTorch 层拆分：直接拒绝分层配置并退出
        # 分层 worker 池（与 llama_cpp 全模型节点的语义一致，防止孤岛引擎
        # 被 load_layer_range 覆盖为 PyTorch 层段）。
        try:
            import config as _island_cfg
            island_gateway = bool(getattr(_island_cfg, "ISLAND_ENABLED", False))
        except Exception:
            island_gateway = False
        if island_gateway:
            error = "本设备为 TP 孤岛网关节点，不参与 PyTorch 层拆分"
            logger.info(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": str(data.get("config_id", "")) if isinstance(data, dict) else "",
                "generation": ack_generation,
                "status": "error",
                "error": error,
            })
            self.release_pipeline_worker_for_local_model()
            return

        if node_id in data and isinstance(data.get(node_id), dict):
            cfg = dict(data[node_id])
        elif isinstance(data, dict) and "start_layer" in data and "end_layer" in data:
            cfg = dict(data)
        else:
            error = f"分层配置中未找到本节点 {node_id} 的有效 assignment"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": data.get("config_id", "") if isinstance(data, dict) else "",
                "generation": ack_generation,
                "status": "error",
                "error": error,
            })
            return

        config_id = str(cfg.get("config_id", ""))
        ack_generation = cfg.get("generation", ack_generation)
        target_node_id = str(cfg.get("node_id", node_id))
        start = cfg.get("start_layer", 0)
        end = cfg.get("end_layer", 24)
        has_embed = cfg.get("has_embedding", False)
        has_lm = cfg.get("has_lm_head", False)
        model_id = str(cfg.get("model_id", ""))
        expected_sha256 = str(cfg.get("model_sha256", ""))
        expected_model_type = str(cfg.get("model_type", "")).lower()
        expected_engine = str(cfg.get("engine", "pytorch") or "pytorch").lower()
        master_quant_type = str(cfg.get("master_quant_type", "") or "")
        phase = str(cfg.get("phase", "commit") or "commit").lower()
        plan_id = str(cfg.get("plan_id", "") or "")
        try:
            start = int(start)
            end = int(end)
            total_layers = int(cfg.get("total_layers", 0) or 0)
            master_api_port = int(cfg.get("master_api_port", 8000) or 8000)
            required_bytes = int(cfg.get("required_bytes", 0) or 0)
        except (TypeError, ValueError) as exc:
            error = f"分层配置数字字段无效: {exc}"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": config_id,
                "generation": ack_generation,
                "status": "error",
                "error": error,
            })
            return
        configuration_invalidated = False

        logger.info(
            f"🔧 收到分层配置: 节点={node_id}, "
            f"Layer {start}-{end}, embed={has_embed}, lm_head={has_lm}, "
            f"config_id={config_id or 'legacy'}"
        )

        try:
            if target_node_id != node_id:
                raise ValueError(f"层配置目标节点 {target_node_id} 与本节点 {node_id} 不一致")
            if expected_model_type not in {"qwen", "qwen2"}:
                raise ValueError(f"不支持的流水线模型架构: {expected_model_type or 'unknown'}")
            if expected_engine not in {"pytorch", "relay_middle"}:
                raise ValueError(
                    f"分层配置引擎必须为 pytorch 或 relay_middle，实际为 {expected_engine}"
                )
            missing_contract = [
                name for name, value in (
                    ("config_id", config_id),
                    ("model_id", model_id),
                    ("model_sha256", expected_sha256),
                    ("total_layers", total_layers),
                )
                if not value
            ]
            if missing_contract:
                raise ValueError(
                    "分层配置执行契约不完整: " + ", ".join(missing_contract)
                )
            if phase not in {"prepare", "commit"}:
                raise ValueError(f"不支持的分层加载阶段: {phase}")
            if phase == "prepare" and not plan_id:
                raise ValueError("prepare 阶段缺少 plan_id")
            if phase == "prepare" and expected_engine != "relay_middle" and required_bytes <= 0:
                raise ValueError("prepare 阶段缺少 required_bytes")
            if expected_engine == "relay_middle":
                relay_spec = self._normalize_relay_segment(cfg.get("relay_segment"))
                if not PIPELINE_RELAY_ENABLED or relay_spec is None:
                    raise ValueError("relay_middle requires an enabled valid relay_segment")
                # relay_middle is endpoint-backed.  The independently
                # supervised relay_mid_service owns its segment artifact;
                # this scheduler worker must not load a full model or a
                # PyTorch layer range just to accept the logical assignment.
                active_config = {
                    "node_id": node_id, "config_id": config_id,
                    "model_id": model_id, "model_sha256": expected_sha256,
                    "model_type": expected_model_type, "layer_range": [start, end],
                    "engine": expected_engine, "relay_segment": relay_spec,
                }
                if phase == "prepare":
                    with self._layer_config_lock:
                        self._prepared_layer_configs[config_id] = {
                            **active_config, "plan_id": plan_id,
                        }
                    self._send_layer_config_ack({
                        "node_id": node_id, "config_id": config_id,
                        "generation": ack_generation, "status": "prepared", "phase": phase,
                        "plan_id": plan_id, "layer_range": [start, end],
                        "model_sha256": expected_sha256, "model_type": expected_model_type,
                        "engine": expected_engine, "relay_segment": relay_spec,
                        "timestamp": time.time(),
                    })
                    return
                if plan_id:
                    with self._layer_config_lock:
                        prepared = dict(self._prepared_layer_configs.get(config_id, {}))
                    if (
                        prepared.get("plan_id") != plan_id
                        or prepared.get("layer_range") != [start, end]
                        or prepared.get("model_sha256") != expected_sha256
                    ):
                        raise RuntimeError("commit 未命中同代际 relay prepared 记录")
                active_config = {
                    "node_id": node_id, "config_id": config_id,
                    "model_id": model_id, "model_sha256": expected_sha256,
                    "model_type": expected_model_type, "layer_range": [start, end],
                    "engine": expected_engine, "relay_segment": relay_spec,
                }
                with self._layer_config_lock:
                    self._pipeline_worker_reserved = True
                    self._active_layer_config = dict(active_config)
                    self._local_pipeline_steps.clear()
                    self._prepared_layer_configs.pop(config_id, None)
                self._send_layer_config_ack({
                    "node_id": node_id, "config_id": config_id,
                    "generation": ack_generation, "status": "ready", "phase": phase,
                    "plan_id": plan_id, "layer_range": [start, end],
                    "has_embedding": has_embed, "has_lm_head": has_lm,
                    "model_sha256": expected_sha256, "model_type": expected_model_type,
                    "engine": expected_engine, "relay_segment": relay_spec,
                    "timestamp": time.time(),
                })
                return
            prepared = {}
            if phase == "commit" and plan_id:
                with self._layer_config_lock:
                    prepared = dict(
                        self._prepared_layer_configs.get(config_id, {})
                    )
                if (
                    prepared.get("plan_id") != plan_id
                    or prepared.get("layer_range") != [start, end]
                    or prepared.get("model_sha256") != expected_sha256
                ):
                    raise RuntimeError("commit 未命中同代际 prepared 记录")

            # A new generation supersedes the old segment immediately. If model
            # synchronization or selective loading then fails, neither the API
            # nor a repeated ACK may advertise the stale generation as ready.
            with self._layer_config_lock:
                self._pipeline_worker_reserved = True
                self._active_layer_config = None
                self._last_layer_config_ack_payload = None
                self._local_pipeline_steps.clear()
            self._host.model_loaded = False
            configuration_invalidated = True

            local_sha256 = ""
            local_model_path = None
            if phase == "commit" and plan_id:
                local_sha256 = str(prepared.get("model_sha256", "") or "")
                local_model_path = prepared.get("model_path")
            elif phase == "prepare" and plan_id and cfg.get("assignment_manifest"):
                from model_sync import (
                    ensure_pipeline_assignment_available,
                    resolve_worker_model_path,
                )
                from transport_port import compute_local_model_sha256

                tcp_client = getattr(self, "_tcp_client", None)
                master_host = getattr(tcp_client, "server_host", "")
                if not master_host:
                    raise RuntimeError("无法确定主节点模型下载地址")
                # Prefer an already provisioned full model. The assignment
                # manifest is only a cold-start fallback for workers that do
                # not own the same revision locally.
                local_model_path = resolve_worker_model_path(model_id)
                local_sha256 = compute_local_model_sha256(
                    model_path=local_model_path,
                    model_id=model_id,
                )
                if local_sha256 == expected_sha256:
                    logger.info(
                        "worker local model revision matches; skip assignment weight transfer: model=%s",
                        model_id,
                    )
                else:
                    local_model_path, assignment_manifest = ensure_pipeline_assignment_available(
                        master_host,
                        master_api_port,
                        {
                            **cfg,
                            "model_id": model_id,
                            "model_sha256": expected_sha256,
                        },
                    )
                    local_sha256 = expected_sha256
            elif expected_sha256:
                from transport_port import compute_local_model_sha256
                if model_id:
                    from model_sync import (
                        ensure_model_available,
                        resolve_worker_model_path,
                    )

                    local_model_path = resolve_worker_model_path(model_id)
                    local_sha256 = compute_local_model_sha256(
                        model_path=local_model_path,
                        model_id=model_id,
                    )
                    if local_sha256 != expected_sha256:
                        tcp_client = getattr(self, "_tcp_client", None)
                        master_host = getattr(tcp_client, "server_host", "")
                        if not master_host:
                            raise RuntimeError("无法确定主节点模型下载地址")
                        logger.info("从主节点同步流水线模型: %s", model_id)
                        local_model_path = ensure_model_available(
                            master_host,
                            master_api_port,
                            model_id,
                            expected_sha256,
                        )
                        local_sha256 = compute_local_model_sha256(
                            model_path=local_model_path,
                            model_id=model_id,
                        )
                else:
                    local_sha256 = compute_local_model_sha256()
                if not local_sha256:
                    raise FileNotFoundError("本节点未找到可校验的 PyTorch 模型权重")
                if local_sha256 != expected_sha256:
                    raise ValueError(
                        f"模型 SHA256 不一致: local={local_sha256[:16]}... "
                        f"master={expected_sha256[:16]}..."
                    )

            if phase == "prepare":
                from pipeline_model_descriptor import inspect_pipeline_model

                try:
                    descriptor = inspect_pipeline_model(
                        local_model_path,
                        model_id=model_id,
                        layer_range=(start, end),
                    )
                except TypeError as exc:
                    # Keep compatibility with older test/sidecar adapters
                    # that still expose the C1 two-argument inspector.
                    if "layer_range" not in str(exc):
                        raise
                    descriptor = inspect_pipeline_model(
                        local_model_path, model_id=model_id,
                    )
                if (
                    descriptor.get("model_type") != expected_model_type
                    or int(descriptor.get("total_layers", 0) or 0) != total_layers
                    or start < 0 or end <= start or end > total_layers
                ):
                    raise ValueError("worker 工件描述器与 prepare 契约不一致")
                profile = dict(self._local_device_profile or {})
                gpu = self._select_scoring_gpu(profile)
                cuda_discrete = bool(
                    isinstance(gpu, dict)
                    and gpu.get("cuda_available", False)
                    and not self._gpu_is_integrated(gpu)
                )
                if cuda_discrete:
                    free_gb = float(gpu.get("vram_free_gb", 0) or 0)
                    capacity_source = "gpu.vram_free_gb"
                else:
                    ram = profile.get("ram", {})
                    free_gb = float(
                        ram.get("available_gb", 0) or 0
                    ) if isinstance(ram, dict) else 0.0
                    capacity_source = "ram.available_gb"
                available_bytes = max(0, int(free_gb * 1024 ** 3))
                if available_bytes < required_bytes:
                    raise RuntimeError(
                        "worker 实时容量不足: "
                        f"required={required_bytes}, available={available_bytes}"
                    )
                prepare_manager = getattr(
                    self._host, "prepare_pipeline_model", None
                )
                if callable(prepare_manager):
                    prepare_manager(
                        model_id=model_id,
                        model_path=local_model_path,
                        quant_type=master_quant_type or None,
                        layer_range=(start, end),
                        model_sha256=local_sha256 or expected_sha256,
                    )
                prepared_record = {
                    "config_id": config_id,
                    "plan_id": plan_id,
                    "model_id": model_id,
                    "model_sha256": local_sha256,
                    "model_type": expected_model_type,
                    "model_path": local_model_path,
                    "layer_range": [start, end],
                    "required_bytes": required_bytes,
                    "available_bytes": available_bytes,
                    "capacity_source": capacity_source,
                }
                with self._layer_config_lock:
                    self._prepared_layer_configs[config_id] = prepared_record
                self._send_layer_config_ack({
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": ack_generation,
                    "status": "prepared",
                    "phase": "prepare",
                    "plan_id": plan_id,
                    "layer_range": [start, end],
                    "model_sha256": local_sha256,
                    "model_type": expected_model_type,
                    "engine": "pytorch",
                    "required_bytes": required_bytes,
                    "available_bytes": available_bytes,
                    "capacity_source": capacity_source,
                    "timestamp": time.time(),
                })
                return

            mgr = self._host
            if mgr and mgr.is_loaded:
                # 如果已加载完整模型，重新加载指定层范围
                logger.info(f"🔄 重新加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                    model_path=local_model_path,
                    quant_type=master_quant_type or None,
                    total_layers=total_layers or None,
                    model_id=model_id or None,
                )
            elif mgr:
                # 模型尚未加载，先加载层范围
                logger.info(f"📥 首次加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                    model_path=local_model_path,
                    quant_type=master_quant_type or None,
                    total_layers=total_layers or None,
                    model_id=model_id or None,
                )
            else:
                raise RuntimeError("model_manager 不可用，无法加载层范围")

            actual_range = getattr(mgr, 'layer_range', None)
            if actual_range is not None and tuple(actual_range) != (start, end):
                raise RuntimeError(
                    f"模型层范围加载结果不一致: actual={actual_range}, expected=({start}, {end})"
                )
            engine = backend_id_for(mgr, default='pytorch') or 'pytorch'
            if engine != 'pytorch':
                raise RuntimeError(f"层拆分要求 PyTorch 引擎，实际为 {engine}")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(
                getattr(loaded_config, "model_type", "") or ""
            ).lower()
            if actual_model_type != expected_model_type:
                raise RuntimeError(
                    f"模型架构不一致: actual={actual_model_type}, "
                    f"expected={expected_model_type}"
                )

            # Model loading can take minutes. A newer assignment/release or a
            # master disconnect may arrive while this thread owns the execution
            # lock; never publish the obsolete load as ready afterwards.
            if receive_sequence is not None:
                with self._layer_config_lock:
                    if (receive_sequence
                            != self._latest_layer_config_receive_sequence
                            or (self._latest_layer_config_generation
                                and generation
                                < self._latest_layer_config_generation)):
                        raise RuntimeError(
                            "分层配置在模型加载期间已被更新代际取代"
                        )

            active_config = {
                "node_id": node_id,
                "config_id": config_id,
                "model_id": model_id,
                "model_sha256": local_sha256 or expected_sha256,
                "model_type": actual_model_type,
                "layer_range": [start, end],
                "engine": engine,
                "master_quant_type": master_quant_type,
                "runtime_quant_type": getattr(mgr, "quant_type", "") or "",
            }
            with self._layer_config_lock:
                self._pipeline_worker_reserved = True
                self._active_layer_config = dict(active_config)
                self._local_pipeline_steps.clear()
                self._prepared_layer_configs.pop(config_id, None)
            # Layer config may be the first model load on a clean worker. Keep the
            # API's compatibility globals aligned so the first forwarded chat does
            # not auto-load a full/GGUF model over this segment.
            self._host.model_loaded = True
            self._host.current_quant = getattr(mgr, "quant_type", None) or "fp16"

            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": config_id,
                "generation": ack_generation,
                "status": "ready",
                "phase": phase,
                "plan_id": plan_id,
                "layer_range": [start, end],
                "has_embedding": has_embed,
                "has_lm_head": has_lm,
                "model_sha256": local_sha256 or expected_sha256,
                "model_type": actual_model_type,
                "engine": engine,
                "master_quant_type": master_quant_type,
                "runtime_quant_type": getattr(mgr, "quant_type", "") or "",
                "timestamp": time.time(),
            })
            logger.info(
                f"✅ 模型层加载完成并已确认: node={node_id}, "
                f"Layer {start}-{end}, config_id={config_id or 'legacy'}"
            )
        except Exception as e:
            if configuration_invalidated:
                with self._layer_config_lock:
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
                    self._local_pipeline_steps.clear()
                self._host.model_loaded = False
            logger.error(f"加载层范围失败: {e}", exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id,
                "config_id": config_id,
                "generation": ack_generation,
                "status": "error",
                "layer_range": [start, end],
                "model_sha256": "",
                "model_type": expected_model_type,
                "engine": expected_engine,
                "error": str(e),
                "timestamp": time.time(),
            })


    def _send_layer_config_ack(self, payload: dict) -> bool:
        """从节点向主节点回传层配置加载结果。"""
        from transport_port import MessageType

        if (
            payload.get("config_id")
            and payload.get("status") in {"prepared", "ready"}
        ):
            with self._layer_config_lock:
                self._last_layer_config_ack_payload = dict(payload)

        client = getattr(self, '_tcp_client', None)
        if client is None:
            logger.warning("TCP 客户端未连接，无法发送层配置 ACK")
            return False
        try:
            client.send_data(payload, MessageType.LAYER_CONFIG_ACK)
            return True
        except Exception as e:
            logger.error(f"发送层配置 ACK 失败: {e}", exc_info=True)
            return False


    def _handle_layer_config_ack(self, client_id: str, msg: dict) -> None:
        """Validate legacy ready ACKs and capacity prepare/commit phases."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        config_id = str(data.get("config_id", ""))
        if node_id != client_id:
            logger.warning(
                "忽略节点标识不一致的层配置 ACK: connection=%s payload=%s",
                client_id, node_id,
            )
            return

        commit_config_id = ""
        abort_details = None
        activated_plan = None
        with self._layer_config_lock:
            expected = self._layer_config_expected.get(client_id)
            if not expected:
                logger.warning("忽略未请求的层配置 ACK: node=%s", client_id)
                return
            if config_id != expected.get("config_id"):
                logger.warning(
                    "忽略过期层配置 ACK: node=%s config_id=%s expected=%s",
                    client_id, config_id, expected.get("config_id"),
                )
                return

            # A versioned assignment is fenced by both config_id and
            # generation. This rejects a delayed prepare/ready ACK after a
            # reconnect or replacement assignment, while preserving the
            # legacy path for expectations that never carried generation.
            if not expected.get("release") and "generation" in expected:
                try:
                    expected_generation = int(expected.get("generation"))
                    ack_generation = int(data.get("generation"))
                except (TypeError, ValueError):
                    logger.warning(
                        "忽略缺少或无效 generation 的层配置 ACK: node=%s config=%s",
                        client_id, config_id,
                    )
                    return
                if ack_generation != expected_generation:
                    logger.warning(
                        "忽略过期层配置 ACK generation: node=%s config=%s ack=%s expected=%s",
                        client_id, config_id, ack_generation, expected_generation,
                    )
                    return

            if expected.get("release"):
                try:
                    ack_generation = int(data.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    ack_generation = -1
                released = (
                    data.get("status") == "released"
                    and data.get("release") is True
                    and ack_generation == int(expected.get("generation", 0) or 0)
                )
                self._layer_config_acks[client_id] = dict(data)
                if released:
                    self._layer_config_expected.pop(client_id, None)
                    self._layer_config_pushed.discard(client_id)
                    self._layer_config_retry_state.pop(client_id, None)
                else:
                    state = self._layer_config_retry_state.setdefault(
                        client_id, {"attempts": 0, "next_retry": 0.0}
                    )
                    state["next_retry"] = time.monotonic() + 5.0
                release_ack = True
                ready = False
                prepared = False
                prepared_late = False
                expected_range = []
            else:
                release_ack = False
                expected_range = [expected["start_layer"], expected["end_layer"]]
                expected_phase = str(expected.get("phase", "commit") or "commit")
                prepared = (
                    expected_phase == "prepare"
                    and data.get("status") == "prepared"
                    and data.get("phase") == "prepare"
                    and data.get("plan_id") == expected.get("plan_id")
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                    and int(data.get("available_bytes", 0) or 0)
                    >= int(expected.get("required_bytes", 0) or 0)
                )
                # A fast worker may deliver prepare after the coordinator has
                # already published commit. This is a valid same-generation
                # state update, not a worker failure.
                prepared_late = (
                    expected_phase == "commit"
                    and data.get("status") == "prepared"
                    and data.get("phase") == "prepare"
                    and data.get("plan_id") == expected.get("plan_id")
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                )
                ready = (
                    expected_phase == "commit"
                    and data.get("status") == "ready"
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                    and (
                        "has_embedding" not in data
                        or bool(data.get("has_embedding"))
                        == bool(expected.get("has_embedding"))
                    )
                    and (
                        "has_lm_head" not in data
                        or bool(data.get("has_lm_head"))
                        == bool(expected.get("has_lm_head"))
                    )
                    and (
                        not expected.get("plan_id")
                        or data.get("plan_id") == expected.get("plan_id")
                    )
                )
                self._layer_config_acks[client_id] = dict(data)
                if ready:
                    self._layer_config_pushed.add(client_id)
                    self._layer_config_retry_state.pop(client_id, None)
                    transaction = self._pipeline_load_transaction
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and transaction.get("phase") == "committing"
                    ):
                        ready_nodes = set(transaction.get("ready_nodes", set()))
                        ready_nodes.add(client_id)
                        transaction["ready_nodes"] = ready_nodes
                        if ready_nodes == set(transaction.get("worker_ids", set())):
                            transaction["phase"] = "ready"
                            active_plan = dict(transaction.get("plan", {}))
                            active_plan["computed_at"] = time.time()
                            active_plan["transaction_phase"] = "ready"
                            self._active_pipeline_capacity_plan = active_plan
                            activated_plan = dict(active_plan)
                elif prepared:
                    self._layer_config_pushed.discard(client_id)
                    self._layer_config_retry_state.pop(client_id, None)
                    transaction = self._pipeline_load_transaction
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and transaction.get("phase") == "preparing"
                    ):
                        prepared_nodes = set(transaction.get("prepared_nodes", set()))
                        prepared_nodes.add(client_id)
                        transaction["prepared_nodes"] = prepared_nodes
                        if prepared_nodes == set(transaction.get("worker_ids", set())):
                            commit_config_id = config_id
                elif prepared_late:
                    self._layer_config_retry_state.pop(client_id, None)
                else:
                    self._layer_config_pushed.discard(client_id)
                    state = self._layer_config_retry_state.setdefault(
                        client_id, {"attempts": 0, "next_retry": 0.0}
                    )
                    state["next_retry"] = time.monotonic() + 5.0
                    transaction = self._pipeline_load_transaction
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and data.get("status") == "error"
                    ):
                        abort_details = (
                            config_id,
                            f"pipeline_{expected_phase}_failed",
                            str(data.get("error", "") or "worker rejected phase"),
                        )

        if release_ack:
            if released:
                logger.info(
                    "从节点已确认退出分层 worker: node=%s config_id=%s",
                    client_id, config_id,
                )
            else:
                logger.error("从节点分层释放 ACK 未通过: node=%s", client_id)
            return
        if abort_details is not None:
            self._abort_pipeline_load_transaction(*abort_details)
            return
        if activated_plan is not None:
            reshard_committed = self._commit_ready_pipeline_reshard(activated_plan)
            if reshard_committed is None:
                self._activate_pipeline_reshard_coordinator(activated_plan)
            elif not reshard_committed:
                with self._layer_config_lock:
                    self._active_pipeline_capacity_plan = None
        if commit_config_id:
            self._commit_pipeline_load_transaction(commit_config_id)
            return
        if prepared:
            logger.info(
                "流水线 worker prepare 通过: node=%s config=%s",
                client_id, config_id,
            )
        elif ready:
            logger.info(
                "从节点层配置已就绪: node=%s range=%s config=%s",
                client_id, expected_range, config_id,
            )
        elif prepared_late:
            logger.info(
                "忽略已进入 commit 的迟到 prepare ACK: node=%s config=%s",
                client_id, config_id,
            )
        else:
            logger.error(
                "从节点层配置阶段未通过: node=%s status=%s error=%s",
                client_id, data.get("status"), data.get("error", ""),
            )


    def _handle_layer_worker_opt_out(self, client_id: str, msg: dict) -> None:
        """Remove a connected client from future layer assignments."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        if node_id != client_id:
            logger.warning(
                "忽略来源不一致的分层退出请求: connection=%s payload=%s",
                client_id,
                node_id,
            )
            return
        with self._layer_config_lock:
            self._pipeline_worker_opt_out.add(client_id)
        self._clear_layer_config_state(client_id)
        logger.info("从节点已退出 PyTorch 分层计算: node=%s", client_id)
        self.push_layer_config_to_clients()


    def _handle_layer_worker_opt_in(self, client_id: str, msg: dict) -> None:
        """Allow a connected client to receive layer assignments again."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        if node_id != client_id:
            logger.warning(
                "忽略来源不一致的分层加入请求: connection=%s payload=%s",
                client_id,
                node_id,
            )
            return
        with self._layer_config_lock:
            self._pipeline_worker_opt_out.discard(client_id)
        logger.info("从节点已重新加入 PyTorch 分层计算: node=%s", client_id)
        # A worker may still carry a local opt-out from its previous model
        # operation.  When the master has a prepared distributed artifact,
        # use the authoritative path so the assignment clears that stale
        # local state instead of immediately eliciting another opt-out.
        if getattr(self._host, "is_pipeline_prepared", False):
            self.request_authoritative_layer_sync()
        else:
            self.push_layer_config_to_clients()


    def _run_master_lm_head(self, hidden_states):
        """在主节点对 worker 返回的尾层 hidden states 执行 Norm + LM Head。"""
        mgr = self._host
        if (
            not mgr
            or not mgr.is_loaded
            or not runtime_supports(mgr, Capability.FORWARD_LAYERS)
        ):
            raise RuntimeError("主节点 PyTorch 模型未加载，无法执行 LM Head")
        project = getattr(mgr, "forward_lm_head", None)
        if not callable(project):
            raise RuntimeError("当前模型管理器不支持架构感知 LM Head")
        return project(hidden_states)


    def _handle_layer_forward(self, client_id: str, msg: dict) -> None:
        with self._layer_execution_lock:
            self._handle_layer_forward_locked(client_id, msg)


    def _handle_layer_forward_locked(self, client_id: str, msg: dict) -> None:
        """
        从节点：收到主节点的 LAYER_FORWARD → 执行本节点层前向 → 返回 LAYER_RESULT。

        消息格式:
            LAYER_FORWARD: { task_id, step, use_kv_cache,
                             input_ids?, hidden_states?,
                             attention_mask?, position_ids?,
                             temperature, top_p }

        **KV Cache 支持 (Phase 3)**:
         - use_kv_cache=True: 从本地 _kv_cache[task_id] 读取缓存的 KV，
           仅处理新 token（增量解码），计算后将新 KV 存回。
         - use_kv_cache=False: Prefill 模式，处理完整序列，构建新 KV cache。

        处理流程:
            1. 反序列化输入（input_ids 或 hidden_states）
            2. 根据 use_kv_cache 读取/写入本地 KV cache
            3. 调用 model_manager.forward_layers()
            4. 序列化输出（hidden_states 或 logits，不含 KV cache）
            5. 发送 LAYER_RESULT 回主节点
        """
        from transport_port import MessageType

        data = msg.get("data", {})
        task_id = str(data.get("task_id", "unknown") or "unknown")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = -1
        use_kv_cache = data.get("use_kv_cache", False)
        config_id = str(data.get("config_id", ""))
        model_sha256 = str(data.get("model_sha256", ""))
        model_type = str(data.get("model_type", "")).lower()
        layer_config_invalid = False
        received_chain_path = data.get("chain_path", [])
        if not isinstance(received_chain_path, list):
            received_chain_path = []
        logical_predecessor = (
            str(data.get("_chain_predecessor", "") or client_id)
            if client_id == "master" else client_id
        )

        logger.info(
            f"🔬 收到层前向指令: task={task_id}, step={step}, from={client_id}, "
            f"kv_cache={'on' if use_kv_cache else 'off'}"
        )

        try:
            if received_chain_path:
                normalized_path = [str(item) for item in received_chain_path]
                if (normalized_path[-1] != logical_predecessor
                        or self.get_effective_node_id() in normalized_path
                        or len(normalized_path) != len(set(normalized_path))):
                    raise RuntimeError(
                        f"链式转发路径与 TCP 前驱不一致: "
                        f"path={normalized_path}, predecessor={logical_predecessor}"
                    )
                received_chain_path = normalized_path
            with self._layer_config_lock:
                if task_id in self._local_pipeline_cancelled:
                    logger.info("忽略已取消任务的迟到层前向: task=%s", task_id)
                    return
                active_config = dict(self._active_layer_config or {})
                last_step = self._local_pipeline_steps.get(task_id)
                task_active = task_id in self._active_pipeline_task_ids
            if not active_config:
                layer_config_invalid = True
                raise RuntimeError("本节点没有已确认的活动层配置")
            for field, actual in (
                ("config_id", config_id),
                ("model_sha256", model_sha256),
                ("model_type", model_type),
            ):
                if not actual or actual != str(active_config.get(field, "")):
                    raise RuntimeError(
                        f"流水线执行契约不一致: {field}={actual or '-'}, "
                        f"expected={active_config.get(field, '-') }"
                    )
            if step < 0:
                raise RuntimeError(f"无效流水线 step: {step}")
            if step == 0:
                if use_kv_cache:
                    raise RuntimeError("prefill step 0 不得声明使用既有 KV cache")
                if task_active or last_step is not None:
                    raise RuntimeError(f"重复 prefill: task={task_id}")
            else:
                if not use_kv_cache:
                    raise RuntimeError(f"decode step {step} 必须使用 KV cache")
                if not task_active or last_step != step - 1:
                    raise RuntimeError(
                        f"流水线 step 越序: task={task_id}, step={step}, "
                        f"last_step={last_step}"
                    )
            if str(active_config.get("engine", "pytorch") or "pytorch").lower() == "relay_middle":
                # relay_middle is endpoint-backed. The separately supervised
                # relay_mid_service owns the segment artifact; this scheduler
                # host does not need a local ModelHost/model loaded.
                relay_spec = self._normalize_relay_segment(data.get("relay_segment"))
                if not PIPELINE_RELAY_ENABLED or relay_spec is None:
                    layer_config_invalid = True
                    raise RuntimeError("relay_middle requires an enabled valid relay_segment")
                if relay_spec != active_config.get("relay_segment"):
                    layer_config_invalid = True
                    raise RuntimeError("relay segment does not match active layer config")
                return self._handle_layer_forward_via_relay(
                    relay_spec, data=data, task_id=task_id, step=step,
                    config_id=config_id, model_sha256=model_sha256,
                    model_type=model_type, received_chain_path=received_chain_path,
                )
            mgr = self._host
            if not mgr or not mgr.is_loaded:
                layer_config_invalid = True
                raise RuntimeError("模型未加载")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(
                getattr(loaded_config, "model_type", "") or ""
            ).lower()
            if backend_id_for(mgr) != "pytorch":
                # ★ A1 / X 档（2026-09-24）：本节点不跑 pytorch 层段时的**保守** Relay 委托。
                #   仅当 ① 全局开关打开 且 ② 本步携带合法的 `relay_segment` 规格（middle 角色）
                #   才把本段交给远端 relay 段；否则**保持原行为** —— 直接拒绝，绝不静默降级。
                relay_spec = self._normalize_relay_segment(data.get("relay_segment"))
                if not PIPELINE_RELAY_ENABLED or relay_spec is None:
                    layer_config_invalid = True
                    raise RuntimeError(f"worker 引擎已变化: {backend_id_for(mgr)}")
                if relay_spec != active_config.get("relay_segment"):
                    layer_config_invalid = True
                    raise RuntimeError("relay segment does not match active layer config")
                return self._handle_layer_forward_via_relay(
                    relay_spec, data=data, task_id=task_id, step=step,
                    config_id=config_id, model_sha256=model_sha256,
                    model_type=model_type, received_chain_path=received_chain_path,
                )
            require_torch()
            from transport_port import deserialize_tensor, serialize_tensor
            if actual_model_type != model_type:
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 模型架构已变化: actual={actual_model_type}, expected={model_type}"
                )
            if str(getattr(mgr, "active_model_id", "") or "") != str(
                active_config.get("model_id", "")
            ):
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 模型 ID 已变化: actual={getattr(mgr, 'active_model_id', '')}, "
                    f"expected={active_config.get('model_id', '')}"
                )
            if list(getattr(mgr, "layer_range", ()) or ()) != active_config.get("layer_range"):
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 层范围已变化: actual={getattr(mgr, 'layer_range', None)}, "
                    f"expected={active_config.get('layer_range')}"
                )

            # ---- 反序列化输入 ----
            input_ids = None
            hidden_states = None
            attention_mask = None
            position_ids = None

            if "input_ids" in data and data["input_ids"] is not None:
                input_ids = self._scheduler_facade_global('torch').tensor(data["input_ids"], dtype=self._scheduler_facade_global('torch').long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # (seq_len,) → (1, seq_len)

            if "hidden_states" in data and data["hidden_states"] is not None:
                hs_bytes = data["hidden_states"]
                if isinstance(hs_bytes, str):
                    import base64
                    hs_bytes = base64.b64decode(hs_bytes)
                elif isinstance(hs_bytes, list):
                    hs_bytes = bytes(hs_bytes)
                hidden_states = deserialize_tensor(hs_bytes)

            if "attention_mask" in data and data["attention_mask"] is not None:
                attention_mask = self._scheduler_facade_global('torch').tensor(data["attention_mask"], dtype=self._scheduler_facade_global('torch').long)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            if "position_ids" in data and data["position_ids"] is not None:
                position_ids = self._scheduler_facade_global('torch').tensor(data["position_ids"], dtype=self._scheduler_facade_global('torch').long)
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)

            # ---- KV Cache: 读取缓存的 past_key_values ----
            past_kv = None
            if use_kv_cache:
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        past_kv = self._kv_cache[task_id]
                if past_kv is None:
                    raise RuntimeError(
                        f"decode step {step} 缺少本地 KV cache: task={task_id}"
                    )
                if past_kv:
                    cached_shape = past_kv[0][0].shape
                    cached_seq_len = (
                        cached_shape[1] if model_type == "qwen" else cached_shape[2]
                    )
                    logger.debug(
                        f"📦 KV cache 命中: task={task_id}, "
                        f"layers={len(past_kv)}, "
                        f"seq_len={cached_seq_len}"
                    )

            # ---- 执行前向传播 ----
            self._begin_local_pipeline_task(task_id)
            t_start = time.time()
            result = mgr.forward_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,  # 始终缓存 KV（prefill 构建，decode 更新）
            )
            with self._layer_config_lock:
                task_cancelled = task_id in self._local_pipeline_cancelled
            if task_cancelled:
                with self._kv_cache_lock:
                    self._kv_cache.pop(task_id, None)
                self._finish_local_pipeline_task(task_id)
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                logger.info("丢弃已取消任务的迟到计算结果: task=%s", task_id)
                return
            elapsed_ms = (time.time() - t_start) * 1000
            # ---- KV Cache: 存储更新后的 past_key_values ----
            if result.get("past_key_values"):
                with self._kv_cache_lock:
                    self._kv_cache[task_id] = result["past_key_values"]
                kv_shape = result["past_key_values"][0][0].shape
                kv_seq_len = kv_shape[1] if model_type == "qwen" else kv_shape[2]
                logger.debug(
                    f"💾 KV cache 已更新: task={task_id}, "
                    f"seq_len={kv_seq_len}"
                )
            else:
                raise RuntimeError("分层前向未返回 KV cache")
            with self._layer_config_lock:
                self._local_pipeline_steps[task_id] = step

            # ---- 序列化输出 ----
            response = {
                "task_id": task_id,
                "node_id": self.get_effective_node_id(),
                "step": step,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
                "chain_path": [
                    *[str(item) for item in received_chain_path],
                    self.get_effective_node_id(),
                ],
                "metrics": {
                    "time_ms": round(elapsed_ms, 1),
                    "kv_cache": use_kv_cache,  # 标记是否使用了 KV cache
                    "kv_seq_len": (
                        kv_seq_len
                        if result.get("past_key_values") else 0
                    ),
                    "memory_allocated_gb": (
                        round(self._scheduler_facade_global('torch').cuda.memory_allocated() / (1024**3), 2)
                        if self._scheduler_facade_global('torch').cuda.is_available() else 0
                    ),
                },
            }

            if "hidden_states" in result:
                # 中间节点：返回隐藏状态
                hs_cpu = result["hidden_states"].detach().cpu()
                response["hidden_states"] = serialize_tensor(hs_cpu)
                response["hidden_shape"] = list(hs_cpu.shape)
                logger.info(
                    f"✅ 层前向完成: task={task_id}, step={step}, "
                    f"output=hidden_states {list(hs_cpu.shape)}, "
                    f"kv={'on' if use_kv_cache else 'prefill'}, "
                    f"time={elapsed_ms:.0f}ms"
                )

            if "logits" in result:
                # 末节点：返回 logits
                logits_cpu = result["logits"].detach().cpu()
                response["logits"] = serialize_tensor(logits_cpu)
                response["logits_shape"] = list(logits_cpu.shape)
                logger.info(
                    f"✅ 层前向完成: task={task_id}, step={step}, "
                    f"output=logits {list(logits_cpu.shape)}, "
                    f"kv={'on' if use_kv_cache else 'prefill'}, "
                    f"time={elapsed_ms:.0f}ms"
                )

            # ---- 链式直连：转发给下一个从节点（P2 优化 + 主节点中转回退）----
            chain_next = data.get("chain_next")
            chain_remaining = data.get("chain_remaining", [])

            if chain_next and isinstance(chain_next, dict) and chain_next.get("node_id"):
                # 非末节点：通过 TCP 直连转发 hidden_states 给下一个节点
                # ★ hidden_states 为 bytes → base64 编码（JSON 兼容，接收端自动解码）
                import base64 as _b64
                _hs = response.get("hidden_states")
                chain_data = {
                    "task_id": task_id,
                    "step": step,
                    "config_id": config_id,
                    "model_sha256": model_sha256,
                    "model_type": model_type,
                    "chain_path": response["chain_path"],
                    "hidden_states": _b64.b64encode(_hs).decode("ascii") if _hs else None,
                    "hidden_shape": response.get("hidden_shape"),
                    "chain_next": chain_remaining[0] if chain_remaining else None,
                    "chain_remaining": chain_remaining[1:] if len(chain_remaining) > 1 else [],
                    "use_kv_cache": use_kv_cache,
                    "temperature": data.get("temperature", 0.7),
                    "top_p": data.get("top_p", 0.9),
                }

                # L1: 直连下一个从节点
                ok = self._send_chain_forward(chain_next["node_id"], chain_data)
                if ok:
                    logger.debug(f"🔗 L1 直连成功: → {chain_next['node_id']}")
                    self._send_chain_forward_ack(
                        task_id=task_id,
                        step=step,
                        config_id=config_id,
                        from_node_id=self.get_effective_node_id(),
                        target_node_id=chain_next["node_id"],
                        status="sent",
                    )
                else:
                    # L2: 主节点中转（从节点 → 主节点 → 目标从节点）
                    logger.warning(
                        f"⚠️ L1 直连 {chain_next['node_id']} 失败，"
                        f"尝试 L2 主节点中转"
                    )
                    chain_data["_relay_to"] = chain_next["node_id"]
                    sent = self._send_layer_result("master", task_id, result_data=chain_data)
                    if sent:
                        logger.info(
                            f"🔄 L2 中转请求已发送至主节点: "
                            f"{self._scheduler_facade_global('NODE_ID')} → master → {chain_next['node_id']}"
                        )
                    else:
                        error_msg = (
                            f"链式转发到 {chain_next['node_id']} 失败 "
                            f"(L1直连失败，L2中转请求发送失败)"
                        )
                        logger.error(
                            f"❌ {error_msg}，回退到全模型推理"
                        )
                        self._send_chain_forward_ack(
                            task_id=task_id,
                            step=step,
                            config_id=config_id,
                            from_node_id=self.get_effective_node_id(),
                            target_node_id=chain_next["node_id"],
                            status="error",
                            error=error_msg,
                        )
            else:
                # 末节点（或无链配置）：发送 LAYER_RESULT 回主节点
                self._send_layer_result("master", task_id, result_data=response)

        except Exception as e:
            with self._kv_cache_lock:
                self._kv_cache.pop(task_id, None)
            with self._layer_config_lock:
                self._local_pipeline_steps.pop(task_id, None)
                if layer_config_invalid:
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            if layer_config_invalid:
                try:
                    self._host.model_loaded = False
                except Exception:
                    logger.debug("worker 层配置失效后更新 API 状态失败", exc_info=True)
            logger.error(f"层前向传播失败: task={task_id}, error={e}", exc_info=True)
            error_result = {
                "node_id": self.get_effective_node_id(),
                "step": step,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
            }
            if layer_config_invalid:
                error_result["layer_config_invalid"] = True
            self._send_layer_result(
                "master",
                task_id,
                result_data=error_result,
                error=str(e),
            )
            self._send_chain_forward_ack(
                task_id=task_id,
                step=step,
                config_id=config_id,
                from_node_id=client_id,
                status="error",
                error=str(e),
            )


    @staticmethod
    def _normalize_relay_segment(raw: object) -> dict[str, object] | None:
        """★ A1 / X 档：校验并规范化 `LAYER_FORWARD` 里**可选**的 `relay_segment` 规格。

        返回 `None` 表示"不适用"（缺失 / 非法 / 超出 X 档范围）—— 调用方据此走**既有**拒绝路径。
        严格到"多一个未知键就整条不认"，避免"半懂"的规格被误用。

        **X 档只支持 `middle`**（hidden → hidden，与本节点"中间节点返回 hidden_states"的既有契约
        完全对齐）；`head`（吃 token 列表）与 `tail`（吐 token）需要主节点侧接受 token 语义，
        属 Y 档 ⇒ 这里**显式不认**（返回 `None` ⇒ 走原有拒绝，不会静默降级）。
        """
        if not isinstance(raw, dict):
            return None
        if set(raw) - {"role", "host", "port", "n_embd", "timeout"}:
            return None
        if str(raw.get("role", "")).strip().lower() != "middle":
            return None
        host = str(raw.get("host", "")).strip()
        if not is_loopback_host(host):
            return None     # 跨机必须走本地 SSH 隧道端点（Relay 传输层自身也强制 loopback）
        try:
            port = int(raw.get("port", 0))
            n_embd = int(raw.get("n_embd", 0))
            timeout = float(raw.get("timeout", 60.0))
        except (TypeError, ValueError):
            return None
        if not (0 < port <= 65535) or n_embd < 1 or not (0.0 < timeout <= 3600.0):
            return None
        return {"role": "middle", "host": host, "port": port, "n_embd": n_embd,
                "timeout": timeout}

    def _handle_layer_forward_via_relay(self, spec: dict[str, object], *, data: dict,
                                        task_id: str, step: int, config_id: str,
                                        model_sha256: str, model_type: str,
                                        received_chain_path: list) -> None:
        """★ A1 / X 档：把本步委托给远端 **middle** relay 段（hidden → hidden），再回传主节点。

        与中间节点语义对齐（吃 hidden、吐 hidden）；KV 由远端段自管，本节点**不碰**本地
        `_kv_cache`（所以本分支在 KV 检查之前就 return，见 `_handle_layer_forward_locked`）。

        范围：**只做 2 段拓扑** —— 请求里带 `chain_next` 时显式拒绝（>2 段属 Y 档）。
        失败：抛 :class:`RelaySegmentError` ⇒ 被外层 `except` 捕获 ⇒ 经既有
        `_send_layer_result(..., error=str(e))` 回传**具名**错误（形如
        `relay_segment_failed:runner_failed#middle@127.0.0.1:50183`），**绝不**静默产出空 hidden
        （那会退化成"模型算错"，无从区分）。
        """
        if data.get("chain_next"):
            raise RuntimeError("relay 段委托不支持链式转发（>2 段拓扑属 Y 档）")

        width = int(spec["n_embd"])
        raw_hidden = data.get("hidden_states")
        if isinstance(raw_hidden, str):
            import base64
            hidden_bytes = base64.b64decode(raw_hidden)
        elif isinstance(raw_hidden, (bytes, bytearray)):
            hidden_bytes = bytes(raw_hidden)
        else:
            raise RuntimeError("relay 段委托需要 hidden_states（token 输入不在 X 档范围）")
        if not hidden_bytes or len(hidden_bytes) % (width * 4):
            raise RuntimeError("relay 段委托的 hidden 长度与 n_embd 不匹配（需 f32 且整除）")
        n_tokens = len(hidden_bytes) // (width * 4)
        hidden_shape = data.get("hidden_shape")
        if isinstance(hidden_shape, list):
            if not hidden_shape or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in hidden_shape
            ) or hidden_shape[-1] != width:
                raise RuntimeError("relay hidden_shape must end in n_embd and contain positive integers")
            shape_items = 1
            for size in hidden_shape:
                shape_items *= size
            if shape_items != n_tokens * width:
                raise RuntimeError("relay hidden_shape does not match raw f32 payload")
        else:
            hidden_shape = [n_tokens, width]

        # ★ P3：显式给了 seq_ids / positions 就走 `HIDDEN_SEQ`（多序列必须逐 token 绑定）。
        seq_ids = data.get("seq_ids")
        positions = data.get("positions")
        seq_meta = None
        if seq_ids is not None or positions is not None:
            if not (isinstance(seq_ids, list) and isinstance(positions, list)
                    and len(seq_ids) == n_tokens and len(positions) == n_tokens):
                raise RuntimeError("relay 段委托的 seq_ids/positions 必须与 token 数等长")
            n_seq_id = data.get("n_seq_id")
            seq_meta = {
                "n_seq_id": [int(v) for v in (n_seq_id or [1] * n_tokens)],
                "seq_ids": [int(v) for v in seq_ids],
                "positions": [int(v) for v in positions],
            }

        self._begin_local_pipeline_task(task_id)   # 与既有执行路径对齐（保证 begin/finish 平衡）
        started = time.time()
        client = self._relay_segment_client_for_task(task_id, spec, n_embd=width)
        try:
            outcome = client.forward_hidden(hidden_bytes, n_tokens=n_tokens, seq_meta=seq_meta)
        except Exception:
            self._close_relay_segment_client(task_id)
            raise
        elapsed_ms = (time.time() - started) * 1000

        if not outcome.ok:
            # ★ 这一层的语义是「**调度层的段委托**失败」⇒ 消息要能让主节点直接落进
            #   `_fallback_reason`（形如 `pipeline_error_result: ... relay_segment_failed:runner_failed#...`）。
            #   `detail` 只进可读消息；`RelaySegmentError.code` 仍是白名单码（可供线上/日志使用）。
            _code = outcome.error or "relay_internal_error"
            raise RelaySegmentError(_code, role="middle", endpoint=outcome.endpoint,
                                    detail=f"relay_segment_failed:{_code}")

        layer_lock = getattr(self, "_layer_config_lock", None)
        steps = getattr(self, "_local_pipeline_steps", None)
        if steps is None:
            steps = {}
            self._local_pipeline_steps = steps
        if layer_lock is None:
            steps[task_id] = step
        else:
            with layer_lock:
                steps[task_id] = step

        response = {
            "task_id": task_id,
            "node_id": self.get_effective_node_id(),
            "step": step,
            "config_id": config_id,
            "model_sha256": model_sha256,
            "model_type": model_type,
            "chain_path": [*[str(item) for item in received_chain_path],
                           self.get_effective_node_id()],
            "hidden_states": bytes(outcome.hidden),
            "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
            "hidden_shape": hidden_shape,
            "metrics": {
                "time_ms": round(elapsed_ms, 1),
                "kv_cache": False,       # KV 在远端段，本节点没有本地 KV
                "kv_seq_len": 0,
                "relay_executed": True,
                # relay_segment / relay_frames / relay_tokens / relay_payload_bytes / relay_error
                **outcome.to_metrics(),
            },
        }
        logger.info(
            f"🔁 relay 段委托完成: task={task_id}, step={step}, "
            f"段={outcome.role}@{spec['host']}:{spec['port']}, tokens={n_tokens}, "
            f"time={elapsed_ms:.0f}ms"
        )
        self._send_layer_result("master", task_id, result_data=response)

    def _handle_chain_forward(self, client_id: str, msg: dict) -> None:
        """
        从节点：收到另一从节点的 CHAIN_FORWARD → 执行本节点层前向 → 继续转发或回传。

        CHAIN_FORWARD 的消息结构与 LAYER_FORWARD 一致（均为 hidden_states + chain 信息），
        直接委托 _handle_layer_forward 处理（其内部根据 chain_next 决定下一步动作）。
        """
        data = msg.get("data", {})
        task_id = data.get("task_id", "")
        step = data.get("step", -1)
        logger.info(f"🔗 收到链式转发: from={client_id}, task={task_id or '?'}")
        self._send_chain_forward_ack(
            task_id=task_id,
            step=step,
            config_id=str(data.get("config_id", "")),
            from_node_id=(
                str(data.get("_chain_predecessor", "") or client_id)
                if client_id == "master" else client_id
            ),
            status="received",
        )
        self._handle_layer_forward(client_id, msg)


    def _send_layer_result(self, client_id: str, task_id: str,
                           result_data: dict = None, error: str = None) -> bool:
        """从节点 → 主节点：发送层前向传播结果"""
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            return False

        from transport_port import MessageType
        import base64

        payload = result_data or {}
        payload["task_id"] = task_id
        if error:
            payload["error"] = error

        # 将 bytes 字段转为 base64 字符串（JSON 兼容）
        safe_payload = {}
        for k, v in payload.items():
            if isinstance(v, bytes):
                safe_payload[k] = base64.b64encode(v).decode("ascii")
            else:
                safe_payload[k] = v

        try:
            self._tcp_client.send_data(safe_payload, MessageType.LAYER_RESULT)
            return True
        except Exception as e:
            logger.error(f"发送层前向结果失败: {e}")
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass
            return False


    def _send_chain_forward_ack(self, task_id: str, step: int,
                                config_id: str = "",
                                from_node_id: str = "",
                                target_node_id: str = "",
                                status: str = "received",
                                error: str = "") -> bool:
        """从节点 → 主节点：发送链式转发接收/错误 ACK。"""
        if not task_id:
            return False
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送链式转发 ACK")
            return False

        from transport_port import MessageType

        payload = {
            "task_id": task_id,
            "step": step,
            "config_id": config_id,
            "node_id": self.get_effective_node_id(),
            "from_node_id": from_node_id,
            "status": status,
        }
        if target_node_id:
            payload["target_node_id"] = target_node_id
        if error:
            payload["error"] = error
        try:
            self._tcp_client.send_data(payload, MessageType.CHAIN_FORWARD_ACK)
            return True
        except Exception as e:
            logger.error(f"发送链式转发 ACK 失败: {e}")
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass
            return False


    def _handle_layer_result(self, client_id: str, msg: dict) -> None:
        """
        主节点：收到从节点的 LAYER_RESULT → 存储到流水线结果字典，
        唤醒正在等待的 run_pipeline() 主循环。

        特殊处理: 如果 data 中包含 _relay_to 字段，说明从节点请求
        主节点中转 hidden_states 到目标节点（L2 链式回退），此时
        主节点转发后直接返回，不存储结果也不唤醒 run_pipeline()。
        """
        data = msg.get("data", {})
        task_id = data.get("task_id", "")
        node_id = str(data.get("node_id", client_id))
        try:
            step = int(data.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        config_id = str(data.get("config_id", ""))

        if node_id != client_id:
            logger.warning(
                "丢弃来源不一致的层结果: connection=%s payload=%s task=%s",
                client_id, node_id, task_id or "-",
            )
            return

        with self._pipeline_lock:
            is_active = task_id in self._pipeline_active_tasks
            contract = dict(self._pipeline_task_contracts.get(task_id, {}))
        if not is_active:
            logger.warning(
                "丢弃非活跃流水线任务结果: task=%s node=%s",
                task_id or "-", node_id,
            )
            return
        worker_ids = list(contract.get("worker_ids", []))
        expected_nodes = set(worker_ids)
        if (node_id not in expected_nodes
                or step != contract.get("current_step")
                or config_id != contract.get("config_id")
                or data.get("model_sha256") != contract.get("model_sha256")
                or data.get("model_type") != contract.get("model_type")):
            logger.warning(
                "丢弃不符合执行契约的层结果: task=%s node=%s step=%s config=%s",
                task_id, node_id, step, config_id,
            )
            return

        # ★ 中转请求：从节点直连失败 → 请主节点转发到目标节点
        relay_target = data.get("_relay_to")
        if relay_target:
            source_index = worker_ids.index(node_id)
            expected_target = (
                worker_ids[source_index + 1]
                if source_index + 1 < len(worker_ids) else ""
            )
            if relay_target != expected_target:
                error = (
                    f"非相邻中转目标: source={node_id}, "
                    f"target={relay_target}, expected={expected_target or '-'}"
                )
                logger.error(
                    "终止非法链路中转: task=%s %s",
                    task_id, error,
                )
                self._set_pipeline_result_error(
                    task_id, node_id, error, step
                )
                return
            logger.info(
                f"🔄 主节点中转: {node_id} → {relay_target} "
                f"(task={task_id}, step={data.get('step', '?')})"
            )
            try:
                # 构建转发 payload（去掉 _relay_to 内部标记）
                relay_data = {
                    k: v for k, v in data.items()
                    if k != "_relay_to"
                }
                relay_data["_chain_predecessor"] = node_id
                from transport_port import MessageType
                self._send_to_worker(relay_target, relay_data,
                                     MessageType.CHAIN_FORWARD)
                self._handle_chain_forward_ack(
                    node_id,
                    {
                        "data": {
                            "task_id": task_id,
                            "step": step,
                            "config_id": config_id,
                            "node_id": node_id,
                            "target_node_id": relay_target,
                            "status": "sent",
                        }
                    },
                )
                logger.info(f"✅ 中转成功: master → {relay_target}")
                return  # 不存储结果，不唤醒 run_pipeline，链继续
            except Exception as e:
                logger.error(
                    f"❌ 主节点中转失败 → {relay_target}: {e}，"
                    f"触发全模型回退"
                )
                # 中转失败 → 存储错误，唤醒 run_pipeline
                self._set_pipeline_result_error(
                    task_id,
                    relay_target,
                    f"主节点中转到 {relay_target} 失败: {e}",
                    data.get("step", -1),
                )
                return

        if data.get("layer_config_invalid"):
            self._invalidate_worker_layer_ready(
                node_id, config_id, str(data.get("error", "") or "worker 层配置失效")
            )

        if not data.get("error") and node_id != contract.get("last_node_id"):
            logger.warning(
                "丢弃非末节点成功结果: task=%s node=%s expected=%s",
                task_id, node_id, contract.get("last_node_id"),
            )
            return
        if (not data.get("error")
                and data.get("chain_path") != worker_ids):
            error = (
                f"链路路径不完整: path={data.get('chain_path')}, "
                f"expected={worker_ids}"
            )
            logger.error(
                "终止链路路径不完整的任务: task=%s path=%s expected=%s",
                task_id, data.get("chain_path"), worker_ids,
            )
            self._set_pipeline_result_error(
                task_id, node_id, error, step
            )
            return

        logger.info(
            f"📥 收到层前向结果: task={task_id}, node={node_id}, "
            f"step={data.get('step', '?')}, "
            f"error={data.get('error', 'none')}"
        )

        # 解码 base64 bytes 字段
        import base64
        decoded = {}
        for k, v in data.items():
            if isinstance(v, str) and k in ("hidden_states", "logits"):
                try:
                    decoded[k] = base64.b64decode(v)
                except Exception:
                    decoded[k] = v  # 保持原样
            else:
                decoded[k] = v

        key = f"{task_id}:{node_id}"
        with self._pipeline_lock:
            self._pipeline_results[key] = decoded
            if key in self._pipeline_events:
                self._pipeline_events[key].set()


    def _get_pipeline_readiness(self) -> dict:
        """返回流水线 worker 的真实就绪状态和首个阻塞原因。"""
        if not self._tcp_server or not self._tcp_server._running:
            return {
                "ready": False,
                "reason_code": "tcp_server_not_running",
                "reason": "主节点 TCP 服务未运行",
                "workers": [],
            }

        assignments = self.get_layer_assignments()
        master_ids = {"master", self.get_effective_node_id()}
        pipeline_nodes = [
            a for a in assignments.get("assignments", [])
            if a.get("node_id") not in master_ids
            and a.get("layers_count", 1) > 0
        ]
        if not pipeline_nodes:
            return {
                "ready": False,
                "reason_code": "no_pipeline_workers",
                "reason": "未分配任何 PC 从节点参与模型层计算",
                "workers": [],
            }

        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)
        with self._layer_config_lock:
            ready_nodes = set(self._layer_config_pushed)
            expected_configs = dict(self._layer_config_expected)
            ack_snapshot = dict(self._layer_config_acks)
        get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
        connected = set(
            get_client_ids()
            if callable(get_client_ids)
            else getattr(self._tcp_server, "clients", {}).keys()
        )

        first_failure = None
        worker_status = []
        now = time.time()
        for assignment in pipeline_nodes:
            node_id = assignment["node_id"]
            node_info = nodes_snapshot.get(node_id)
            online = bool(node_info and node_info.is_available())
            tcp_connected = node_id in connected
            heartbeat_age = (
                max(0.0, now - node_info.last_heartbeat)
                if node_info and node_info.last_heartbeat else None
            )
            expected = expected_configs.get(node_id, {})
            ack = ack_snapshot.get(node_id, {})
            expected_range = [
                assignment.get("start_layer", 0),
                assignment.get("end_layer", 0),
            ]
            layer_ready = (
                node_id in ready_nodes
                and ack.get("config_id") == expected.get("config_id")
                and ack.get("layer_range") == expected_range
                and ack.get("model_sha256") == expected.get("model_sha256")
                and ack.get("model_type") == expected.get("model_type")
                and ack.get("engine") == expected.get("engine", "pytorch")
            )
            layer_status = "ready" if layer_ready else (
                "error" if ack.get("status") == "error" else
                "loading" if expected else "not_configured"
            )
            error = str(ack.get("error", ""))

            failure = None
            if node_info is None:
                failure = ("worker_not_registered", f"从节点 {node_id} 未注册")
            elif not online:
                failure = ("worker_offline", f"从节点 {node_id} 已离线")
            elif not tcp_connected:
                failure = ("worker_tcp_disconnected", f"从节点 {node_id} TCP 已断开")
            elif heartbeat_age is None or heartbeat_age > 10:
                age_text = "未知" if heartbeat_age is None else f"{heartbeat_age:.1f}s"
                failure = (
                    "worker_heartbeat_stale",
                    f"从节点 {node_id} 心跳已过期 ({age_text})",
                )
            elif not layer_ready and error:
                failure = (
                    "worker_layer_load_failed",
                    f"从节点 {node_id} 模型同步或层加载失败: {error}",
                )
            elif not layer_ready and expected:
                failure = (
                    "worker_layer_loading",
                    f"从节点 {node_id} 正在同步同款 PyTorch 模型或加载分配层",
                )
            elif not layer_ready:
                failure = (
                    "worker_layer_not_configured",
                    f"从节点 {node_id} 尚未收到模型分层配置",
                )

            if first_failure is None and failure is not None:
                first_failure = failure
            worker_status.append({
                "node_id": node_id,
                "online": online,
                "tcp_connected": tcp_connected,
                "heartbeat_age_seconds": (
                    round(heartbeat_age, 1) if heartbeat_age is not None else None
                ),
                "layer_ready": layer_ready,
                "layer_status": layer_status,
                "layer_error": error,
                "config_id": expected.get("config_id", ""),
                "model_id": expected.get("model_id", ""),
                "layer_range": expected_range,
            })

        if first_failure is None:
            return {
                "ready": True,
                "reason_code": "ready",
                "reason": "所有 PC 从节点已确认同款 PyTorch 模型和分配层",
                "workers": worker_status,
            }
        return {
            "ready": False,
            "reason_code": first_failure[0],
            "reason": first_failure[1],
            "workers": worker_status,
        }


    def _all_pipeline_nodes_ready(self) -> bool:
        """检查所有流水线节点是否在线并已确认模型层加载完成。"""
        readiness = self._get_pipeline_readiness()
        if readiness["ready"]:
            logger.info(
                "✅ 所有流水线节点就绪: %s",
                [worker["node_id"] for worker in readiness["workers"]],
            )
            return True
        logger.warning("流水线未就绪: %s", readiness["reason"])
        return False


    def _connected_pc_worker_ids(self) -> list[str]:
        """Return online PC clients that can receive an authoritative config."""
        server = self._tcp_server
        if not server or not getattr(server, "_running", False):
            return []
        get_client_ids = getattr(server, "get_client_ids", None)
        connected = set(
            get_client_ids()
            if callable(get_client_ids)
            else getattr(server, "clients", {}).keys()
        )
        local_node_id = self.get_effective_node_id()
        with self._nodes_lock:
            return sorted(
                node_id for node_id, node in self.nodes.items()
                if node_id in connected
                and node_id != local_node_id
                and getattr(node, "node_type", "pc") == "pc"
                and node.is_available()
                and not self._node_is_island_gateway(node.device_info)
            )


    def _has_active_distributed_pipeline_plan(self) -> bool:
        """Return whether the current generation actually uses two nodes."""
        with self._layer_config_lock:
            plan = self._active_pipeline_capacity_plan
            transaction = self._pipeline_load_transaction
            if not plan and transaction:
                plan = transaction.get("plan")
            assignments = plan.get("assignments", []) if isinstance(plan, dict) else []
            return len(assignments) >= 2


    def _synchronize_pipeline_workers_for_request(
        self,
        timeout: float = PIPELINE_MODEL_SYNC_TIMEOUT,
        *,
        force_distributed_assignment: bool = False,
    ) -> dict:
        """Synchronize worker model segments before falling back to the master."""
        readiness = self._get_pipeline_readiness()
        if readiness.get("ready") and (
            not force_distributed_assignment
            or self._has_active_distributed_pipeline_plan()
        ):
            return readiness

        worker_ids = self._connected_pc_worker_ids()
        if not worker_ids:
            if force_distributed_assignment:
                return {
                    **readiness,
                    "ready": False,
                    "reason_code": "pipeline_distributed_workers_unavailable",
                    "reason": "没有在线 PC 从节点可参与强制分布式分层",
                }
            return readiness

        recoverable = {
            "no_pipeline_workers",
            "worker_layer_not_configured",
            "worker_layer_loading",
            "worker_layer_load_failed",
        }
        if (
            not force_distributed_assignment
            and readiness.get("reason_code") not in recoverable
        ):
            return readiness

        # An existing loading generation should finish without being superseded.
        # Other recoverable states need a fresh authoritative generation.
        if (
            force_distributed_assignment
            or readiness.get("reason_code") != "worker_layer_loading"
        ):
            logger.info(
                "分布式请求触发主节点权威模型同步: workers=%s reason=%s",
                worker_ids,
                readiness.get("reason", ""),
            )
            if force_distributed_assignment:
                self.request_authoritative_layer_sync(require_distributed=True)
            else:
                self.request_authoritative_layer_sync()

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            readiness = self._get_pipeline_readiness()
            if force_distributed_assignment:
                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction or {}
                    phase = str(transaction.get("phase", "") or "")
                    plan = transaction.get("plan") or {}
                if phase in {"rejected", "aborted"}:
                    return {
                        **readiness,
                        "ready": False,
                        "reason_code": (
                            transaction.get("reason_code")
                            or plan.get("reason_code")
                            or "pipeline_distributed_capacity_rejected"
                        ),
                        "reason": (
                            transaction.get("reason")
                            or plan.get("reason")
                            or "强制分布式容量计划未获准入"
                        ),
                    }
            if readiness.get("ready") and (
                not force_distributed_assignment
                or self._has_active_distributed_pipeline_plan()
            ):
                logger.info(
                    "主从模型配置同步完成，流水线已就绪: workers=%s",
                    worker_ids,
                )
                return readiness
            # 旧安装包不认识 authoritative_sync，会再次发送 opt-out。
            # 这是明确的不可恢复信号；不应让本次推理无谓等待完整同步
            # 超时，直接按既有安全路径回退到主节点。
            with self._layer_config_lock:
                opted_out = sorted(
                    set(worker_ids) & self._pipeline_worker_opt_out
                )
            if opted_out:
                logger.warning(
                    "从节点拒绝主节点权威模型同步，立即回退: workers=%s",
                    opted_out,
                )
                return readiness
            if (
                not force_distributed_assignment
                and readiness.get("reason_code") not in recoverable
            ):
                return readiness
            if readiness.get("ready") and force_distributed_assignment:
                readiness = {
                    **readiness,
                    "ready": False,
                    "reason_code": "pipeline_single_node_plan_active",
                    "reason": "当前计划仍是单节点，等待多节点容量计划生效",
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "等待主从模型配置同步超时: timeout=%.1fs reason=%s",
                    float(timeout),
                    readiness.get("reason", ""),
                )
                return readiness
            time.sleep(min(0.1, remaining))


    def _verify_pipeline_readiness(self, pipeline_nodes: list
                                   ) -> tuple:
        """
        二次就绪检查（出队后 / 立即执行前调用）。

        与 _all_pipeline_nodes_ready 的区别：
        - _all_pipeline_nodes_ready: 入队前的快速筛选（Pre-queue gate）
        - _verify_pipeline_readiness: tokenize 前的最终确认（Post-queue gate）

        入队等待期间节点可能离线 / 心跳超时 / TCP 断开，
        此检查在即将开始推理前做最后验证，避免浪费 prefill 计算。

        Returns:
            (ok: bool, reason: str)
        """
        if not self._tcp_server or not self._tcp_server._running:
            return False, "TCP 服务端未运行"

        # Phase 2.1+: 快照避免循环中并发修改
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)

        for node in pipeline_nodes:
            node_id = node["node_id"]
            node_info = nodes_snapshot.get(node_id)
            if not node_info:
                return False, f"节点 {node_id} 已消失（可能被注销）"
            if not node_info.is_available():
                return False, f"节点 {node_id} 已离线 (state={node_info.state.value})"
            get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
            connected_ids = (
                get_client_ids()
                if callable(get_client_ids)
                else getattr(self._tcp_server, "clients", {}).keys()
            )
            if node_id not in connected_ids:
                return False, f"节点 {node_id} TCP 连接已断开"

            # 心跳新鲜度
            heartbeat_age = time.time() - node_info.last_heartbeat
            if heartbeat_age > 10:
                return False, (
                    f"节点 {node_id} 心跳过期 "
                    f"({heartbeat_age:.1f}s > 10s)"
                )

            with self._layer_config_lock:
                expected = self._layer_config_expected.get(node_id, {})
                ack = self._layer_config_acks.get(node_id, {})
                expected_range = [node.get("start_layer"), node.get("end_layer")]
                layer_ready = (
                    node_id in self._layer_config_pushed
                    and ack.get("config_id") == expected.get("config_id")
                    and ack.get("layer_range") == expected_range
                    and ack.get("model_sha256") == expected.get("model_sha256")
                    and ack.get("model_type") == expected.get("model_type")
                    and ack.get("engine") == expected.get("engine", "pytorch")
                )
            if not layer_ready:
                return False, f"节点 {node_id} 尚未确认层配置加载成功"

        logger.info(
            f"✅ 二次就绪检查通过: "
            f"{' → '.join(n['node_id'] for n in pipeline_nodes)}"
        )
        return True, "ok"


    def _broadcast_pipeline_abort(self, pipeline_nodes: list, task_id: str,
                                   reason: str, count_error: bool = True) -> None:
        """向所有流水线节点广播 PIPELINE_ABORT（清理各节点 + master 本地 KV cache）。"""
        from transport_port import MessageType
        failed_nodes = []
        for n in pipeline_nodes:
            node_id = n.get("node_id")
            if not node_id:
                continue
            try:
                self._send_to_worker(
                    node_id,
                    {
                        "task_id": task_id,
                        "reason": reason,
                        "count_error": count_error,
                    },
                    MessageType.PIPELINE_ABORT,
                )
            except Exception as e:
                failed_nodes.append(f"{node_id}: {e}")
                logger.warning(
                    "PIPELINE_ABORT 发送失败: node=%s task=%s error=%s",
                    node_id, task_id, e,
                    exc_info=True,
                )
        if failed_nodes:
            logger.warning(
                "PIPELINE_ABORT 部分节点清理失败: task=%s failed=%s",
                task_id, "; ".join(failed_nodes),
            )
        # ★ 同时清理 master 自身 KV cache（master_participates 路径会产生本地缓存）
        if task_id:
            with self._kv_cache_lock:
                if task_id in self._kv_cache:
                    del self._kv_cache[task_id]
            with self._pipeline_lock:
                self._chain_ack_state.pop(task_id, None)


    def _get_node_address(self, node_id: str) -> Optional[dict]:
        """
        获取节点的 (host, port) 地址信息。

        返回 {"host": str, "port": int} 或 None（节点未知/离线）。
        """
        with self._nodes_lock:
            node = self.nodes.get(node_id)
        if not node or not node.address:
            return None
        # address 格式: "host:port"
        addr = node.address
        if ":" in addr:
            host, port_str = addr.rsplit(":", 1)
            try:
                return {"host": host, "port": int(port_str)}
            except ValueError:
                logger.warning(f"节点 {node_id} 地址格式无效 (端口非数字): {addr}")
                return None
        logger.warning(f"节点 {node_id} 地址缺失或格式错误: {addr or '(空)'}")
        return None


    def _send_chain_forward(self, target_node_id: str, data: dict) -> bool:
        """
        从节点 → 下一个从节点：链式直连转发 hidden_states。

        通过目标节点已有的 TCP 服务端建立短连接，发送 CHAIN_FORWARD
        后立即关闭（fire-and-forget）。

        Returns:
            True 发送成功，False 连接失败
        """
        from transport_port import create_client, MessageType

        addr = self._get_node_address(target_node_id)
        if not addr:
            logger.error(f"无法获取节点 {target_node_id} 的地址")
            return False

        try:
            t0 = time.time()
            target = (addr["host"], addr["port"])
            with self._chain_clients_lock:
                cached = self._chain_clients.get(target_node_id)
                cached_target = (
                    getattr(cached, "server_host", ""),
                    getattr(cached, "server_port", 0),
                ) if cached else None
                cached_ready = bool(
                    cached
                    and cached_target == target
                    and getattr(cached, "_running", False)
                    and getattr(cached, "is_registered", False)
                    and getattr(cached, "sock", None) is not None
                )
                if cached_ready:
                    client = cached
                else:
                    if cached is not None:
                        try:
                            cached.disconnect()
                        except Exception:
                            pass
                    client = create_client(
                        server_host=addr["host"],
                        server_port=addr["port"],
                        client_id=self.get_effective_node_id(),
                        role="client",
                        node_type="pipeline_peer",
                        **self._transport_runtime_kwargs(target_node_id),
                    )
                    if self._control_fence is not None:
                        client.set_control_fence(self._control_fence)
                    if not client.connect():
                        logger.error(
                            "链式转发: 连接 %s (%s:%s) 失败",
                            target_node_id, addr["host"], addr["port"],
                        )
                        return False
                    self._chain_clients[target_node_id] = client

            client.send_data(data, MessageType.CHAIN_FORWARD)
            elapsed_ms = (time.time() - t0) * 1000
            hs_shape = data.get("hidden_shape", "?")
            logger.debug(
                f"🔗 链式转发: {self._scheduler_facade_global('NODE_ID')} → {target_node_id} "
                f"hidden_states={hs_shape}, time={elapsed_ms:.0f}ms"
            )
            return True
        except Exception as e:
            logger.error(f"链式转发到 {target_node_id} 失败: {e}")
            with self._chain_clients_lock:
                failed_client = self._chain_clients.pop(target_node_id, None)
            if failed_client is not None:
                try:
                    failed_client.disconnect()
                except Exception:
                    pass
            return False


    def _send_to_worker(self, worker_id: str, data: dict,
                        msg_type=None) -> None:
        """主节点 → 从节点：发送消息"""
        from transport_port import MessageType
        if msg_type is None:
            msg_type = MessageType.LAYER_FORWARD
        if not self._tcp_server or not self._tcp_server._running:
            raise ConnectionError("TCP 服务端未运行")
        self._tcp_server.send_to_client(worker_id, data, msg_type)

    @staticmethod
    def _parse_relay_segment_map(raw: str) -> dict[str, dict[str, object]]:
        """★ A1 / X 档：解析 `QLH_RELAY_SEGMENTS`（`node=role@host:port#n_embd`，`;`/`,` 分隔）。

        只接受 **X 档范围内**的规格（`middle`、loopback、合法端口/宽度）—— 校验**复用**
        `_normalize_relay_segment`（单一真源，避免两套判据漂移）。任何不合法的条目**整条丢弃**
        （宁可不下发，也不下发"半懂"的规格）；空配置 ⇒ `{}`（行为与接线前一致）。
        """
        result: dict[str, dict[str, object]] = {}
        for chunk in str(raw or "").replace(",", ";").split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk or "@" not in chunk:
                continue
            name, _, value = chunk.partition("=")
            body, _, n_embd_text = value.partition("#")
            role, _, host_port = body.partition("@")
            host, _, port_text = host_port.rpartition(":")
            try:
                spec = SchedulerPipelineMixin._normalize_relay_segment({
                    "role": role.strip(),
                    "host": host.strip(),
                    "port": int(port_text),
                    "n_embd": int(n_embd_text),
                })
            except ValueError:
                continue
            if name.strip() and spec is not None:
                result[name.strip()] = spec
        return result

    def _relay_segment_for_worker(self, worker_id: str) -> Optional[dict]:
        """★ A1 / X 档：该 worker 是否由远端 relay 段代跑本段（主节点侧配置，解析一次后缓存）。

        开关关闭 ⇒ 直接 `None`（对既有路径零影响）。缓存用 `getattr` 惰性挂在实例上，
        **不**改 `__init__`（本方法是 mixin 方法，实例可能来自多种构造路径）。
        """
        if not PIPELINE_RELAY_ENABLED:
            return None
        cache = getattr(self, "_relay_segment_map_cache", None)
        if cache is None:
            cache = self._parse_relay_segment_map(PIPELINE_RELAY_SEGMENTS)
            self._relay_segment_map_cache = cache
        return cache.get(str(worker_id))


    def _wait_for_layer_result(self, task_id: str, node_ids,
                               timeout: float = 30.0,
                               ack_node_ids: list = None,
                               ack_step: int = None,
                               ack_timeout: float = None,
                               cancel_event: threading.Event = None) -> Optional[dict]:
        """
        主节点：等待指定节点的 LAYER_RESULT。

        node_ids 可以是单个 str 或 list[str]。当传入 list 时，
        等待其中任一节点返回结果（链式拓扑中错误可能来自任意节点）。

        使用 threading.Event 实现同步等待，由 _handle_layer_result 唤醒。
        """
        import base64

        if isinstance(node_ids, str):
            node_ids = [node_ids]

        keys = [f"{task_id}:{nid}" for nid in node_ids]

        # 先消费已经到达的结果，避免 worker 极快返回时发生
        # "结果先写入、event 后创建" 的竞态。
        events = []
        result = None
        signaled_key = None
        with self._pipeline_lock:
            for key in keys:
                data = self._pipeline_results.pop(key, None)
                if data is not None:
                    result = data
                    signaled_key = key
                    break
            if result is None:
                for key in keys:
                    event = threading.Event()
                    self._pipeline_events[key] = event
                    events.append((key, event))

        # 等待任一 event 触发
        deadline = time.time() + timeout
        while result is None and time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                result = {
                    "task_id": task_id,
                    "error": "流水线任务已取消",
                    "cancelled": True,
                    "step": ack_step if ack_step is not None else -1,
                }
                break
            if ack_node_ids and ack_step is not None and ack_timeout is not None:
                ack_failure = self._get_chain_ack_failure(
                    task_id, ack_step, ack_node_ids, ack_timeout,
                )
                if ack_failure is not None:
                    result = ack_failure
                    signaled_key = f"{task_id}:{ack_failure.get('node_id')}"
                    break
            for key, event in events:
                if event.is_set():
                    signaled_key = key
                    break
            if signaled_key:
                break
            time.sleep(0.05)

        # 清理所有 events，收集结果
        with self._pipeline_lock:
            for key, _ in events:
                self._pipeline_events.pop(key, None)

            if result is None:
                # 查找第一个有结果或超时的 key。即使 event 轮询刚好错过
                # 最后一瞬间，也以实际结果为准。
                for key in keys:
                    data = self._pipeline_results.pop(key, None)
                    if data is not None:
                        result = data
                        signaled_key = key
                        break

        if result is None:
            logger.error(f"⏰ 等待流水线结果超时 ({timeout}s), task={task_id}")
            return None

        # 解码 base64 → bytes（供调用方反序列化张量）
        decoded = {}
        for k, v in result.items():
            if isinstance(v, str) and k in ("hidden_states", "logits"):
                try:
                    decoded[k] = base64.b64decode(v)
                except Exception:
                    decoded[k] = v
            else:
                decoded[k] = v
        return decoded


    def _wait_for_layer_result_with_ack(self, task_id: str, node_ids,
                                        timeout: float,
                                        ack_node_ids: list,
                                        ack_step: int,
                                        ack_timeout: float,
                                        cancel_event: threading.Event = None) -> Optional[dict]:
        """等待流水线结果；兼容测试或旧扩展中替换掉的三参数等待函数。"""
        try:
            return self._wait_for_layer_result(
                task_id,
                node_ids,
                timeout=timeout,
                ack_node_ids=ack_node_ids,
                ack_step=ack_step,
                ack_timeout=ack_timeout,
                cancel_event=cancel_event,
            )
        except TypeError as e:
            if ("ack_node_ids" not in str(e)
                    and "cancel_event" not in str(e)):
                raise
            logger.debug(
                "_wait_for_layer_result 不支持 ACK 参数，退回旧签名调用",
                exc_info=True,
            )
            return self._wait_for_layer_result(task_id, node_ids, timeout)


    def _check_preempt_conditions(self, current_step: int) -> bool:
        """
        检查是否满足抢占条件（防抖动 + 最小 token 阈值）。

        条件:
        1. PIPELINE_PREEMPT_ENABLED=True
        2. 未被禁用（_preempt_disabled=False）
        3. 当前未在执行抢占（防嵌套）
        4. 已生成 >= MIN_TOKENS 个 token
        5. 距上次抢占 >= MIN_INTERVAL 秒
        """
        if not self._scheduler_facade_global('PIPELINE_PREEMPT_ENABLED') or self._preempt_disabled:
            return False
        if self._preempting:  # ★ 防嵌套：Q0 内部不触发二次抢占
            return False
        if current_step < self._scheduler_facade_global('PIPELINE_PREEMPT_MIN_TOKENS'):
            return False
        if self._preempt_last_time > 0:
            if time.time() - self._preempt_last_time < self._scheduler_facade_global('PIPELINE_PREEMPT_MIN_INTERVAL'):
                return False
        return True


    def _save_preempt_state(self, *, task_id: str, generated_ids: list,
                            full_input_ids, current_step: int,
                            max_new_tokens: int, temperature: float,
                            top_p: float, prompt: str,
                            pipeline_nodes: list, first_node_id: str,
                            _stream_callback=None) -> PreemptState:
        """
        保存当前 decode 循环的所有局部状态到 PreemptState。

        generated_ids 做 shallow copy（list 在恢复后独立 append）。
        full_input_ids 仅保存 tensor 引用（只读，不会被 Q0 修改）。
        """
        state = PreemptState(
            task_id=task_id,
            generated_ids=generated_ids,
            full_input_ids=full_input_ids,
            current_step=current_step,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt=prompt,
            pipeline_nodes=pipeline_nodes,
            first_node_id=first_node_id,
            _stream_callback=_stream_callback,
        )
        self._preempted_task = state
        return state


    def _execute_q0_inline(self, q0_task: QueueTask,
                           preempt_state: PreemptState) -> None:
        """
        内联执行 Q0 抢占任务。

        调用方已释放 _inference_lock 并设置 _preempting=True，本方法负责:
        1. 标记 Q0 为 current_task（try/finally 保护恢复）
        2. 检查节点就绪 → 获取推理锁 → 执行 Q0 → 释放推理锁
        3. 存储 Q0 结果 + 唤醒等待的 API 线程
        4. 恢复被抢占任务为 current_task
        5. 重新获取推理锁（为被抢占任务继续执行）

        Q0 自身异常不影响被抢占任务——错误结果照常存储并唤醒调用方。
        """
        q0_id = q0_task.task_id
        preempted_id = preempt_state.task_id
        q0_result = None
        q0_error = None
        lock_reacquired = False  # ★ BUG1 fix: track if lock was re-acquired in step 5

        try:
            # 1. 标记 Q0 为当前执行任务（在 try 内，异常时由外层的 except 恢复）
            with self.pipeline_queue._lock:
                self.pipeline_queue._current_task_id = q0_id
                if q0_id not in self.pipeline_queue._results:
                    self.pipeline_queue._results[q0_id] = {"status": "pending", "created_at": time.time()}
                self.pipeline_queue._results[q0_id]["status"] = "running"
                self.pipeline_queue._results[q0_id]["started_at"] = time.time()

            t_q0_start = time.time()

            # 2. 获取推理锁 → 执行 Q0（★ 含节点就绪检查与回退）
            self._inference_lock.acquire()
            try:
                if not self._all_pipeline_nodes_ready():
                    logger.warning("Q0 抢占: 流水线节点不可用，回退到全模型推理")
                    q0_result = self._run_full_model_inference(
                        prompt=q0_task.prompt,
                        max_new_tokens=q0_task.max_new_tokens,
                        temperature=q0_task.temperature,
                        top_p=q0_task.top_p,
                        session_id=q0_task.session_id,
                    )
                else:
                    # 透传 QueueTask 中保存的额外参数（如 _stream_callback）
                    extra = q0_task._extra_kwargs if q0_task._extra_kwargs else {}
                    q0_result = self.run_pipeline(
                        prompt=q0_task.prompt,
                        max_new_tokens=q0_task.max_new_tokens,
                        temperature=q0_task.temperature,
                        top_p=q0_task.top_p,
                        session_id=q0_task.session_id,
                        _cancel_event=q0_task.cancel_event,
                        **extra,
                    )
            except Exception as e:
                q0_error = str(e)
                logger.error(f"❌ Q0 抢占任务执行失败: {q0_id} — {e}")
            finally:
                self._inference_lock.release()

            q0_elapsed = time.time() - t_q0_start

            # 3. 存储 Q0 结果 + 唤醒 API 线程 + 恢复 current_task
            with self.pipeline_queue._lock:
                if q0_error:
                    self.pipeline_queue._results[q0_id] = {
                        "status": "error", "error": q0_error,
                        "created_at": self.pipeline_queue._results.get(q0_id, {}).get("created_at", 0),
                        "completed_at": time.time(),
                        "elapsed_s": round(q0_elapsed, 2),
                    }
                else:
                    self.pipeline_queue._results[q0_id] = {
                        "status": "done", "result": q0_result,
                        "created_at": self.pipeline_queue._results.get(q0_id, {}).get("created_at", 0),
                        "started_at": self.pipeline_queue._results.get(q0_id, {}).get("started_at", 0),
                        "completed_at": time.time(),
                        "elapsed_s": round(q0_elapsed, 2),
                    }
                event = self.pipeline_queue._events.get(q0_id)
                if event:
                    event.set()
                # 4. 恢复被抢占任务为 current_task
                self.pipeline_queue._current_task_id = preempted_id

            # 5. 重新获取推理锁（为被抢占任务继续）
            self._inference_lock.acquire()
            lock_reacquired = True  # ★ 标记：在此点之后异常需释放锁

            logger.info(
                f"✅ Q0 抢占完成: {q0_id} ({q0_elapsed:.1f}s) "
                f"→ 恢复 {preempted_id}"
            )
        except Exception:
            # ★ C2 修复: _current_task_id 损坏保护
            with self.pipeline_queue._lock:
                if self.pipeline_queue._current_task_id == q0_id:
                    self.pipeline_queue._current_task_id = preempted_id
            # ★ BUG1 修复: 若锁已被重新获取，释放它以防死锁
            if lock_reacquired:
                try:
                    self._inference_lock.release()
                except RuntimeError:
                    pass
            raise


    def _update_preempt_stats(self, overhead_ms: float) -> None:
        """
        更新抢占统计。

        若单次抢占开销超过 PIPELINE_PREEMPT_MAX_OVERHEAD_MS，
        自动禁用后续抢占（防止 thrashing）。
        统计同步到 PipelineQueue 以支持 get_queue_detail()。
        """
        self._preempt_count += 1
        self._preempt_total_overhead_ms += overhead_ms
        self._preempt_last_time = time.time()

        if overhead_ms > self._scheduler_facade_global('PIPELINE_PREEMPT_MAX_OVERHEAD_MS'):
            self._preempt_disabled = True
            logger.warning(
                f"⚠️ 抢占开销 {overhead_ms:.1f}ms 超过阈值 "
                f"({self._scheduler_facade_global('PIPELINE_PREEMPT_MAX_OVERHEAD_MS')}ms)，已禁用后续抢占"
            )

        # 同步到 PipelineQueue（get_queue_detail 读取此处）
        with self.pipeline_queue._lock:
            self.pipeline_queue._preempt_count = self._preempt_count
            self.pipeline_queue._preempt_total_overhead_ms = self._preempt_total_overhead_ms
            self.pipeline_queue._last_preempt_time = self._preempt_last_time


    def run_pipeline(self, *args, **kwargs) -> dict:
        """Run one pipeline task and abort every task context added by this call on exceptions."""
        stack = getattr(self._pipeline_context, "stack", None)
        if stack is None:
            stack = []
            self._pipeline_context.stack = stack
        initial_depth = len(stack)
        try:
            return self._run_pipeline(*args, **kwargs)
        except Exception as exc:
            for context in reversed(stack[initial_depth:]):
                self._broadcast_pipeline_abort(
                    context["pipeline_nodes"], context["task_id"], str(exc)
                )
                self._clear_pipeline_runtime_state(context["task_id"])
            raise
        finally:
            del stack[initial_depth:]


    def _run_pipeline(self, prompt: str, max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.9,
                     session_id: str = None,
                     messages: list = None,
                     show_thinking: bool = False,
                     _stream_callback=None,
                     _cancel_event: threading.Event = None) -> dict:
        """
        主节点：协调多节点流水线推理。

        **KV Cache 支持 (Phase 3)**:
         - Prefill (step 0): use_kv_cache=False，发送完整 prompt input_ids，
           各节点构建 KV cache 并本地存储。
         - Decode (step 1+): use_kv_cache=True，仅发送最后 1 个 token
           (shape 1×1)，各节点基于本地 KV cache 增量计算。
         - 通信量: hidden_states 从 O(seq_len×2048) FP16 降至 O(1×2048) FP16
         - 计算量: 每 step 从 O(seq_len) 降至 O(1)

        流程:
            1. 获取当前分层配置
            2. 确定流水线节点顺序（按 start_layer 排序）
            3. Tokenize prompt → input_ids
            4. 自回归生成循环:
               a. Prefill (step 0): 发送完整 input_ids + chain_info 给首节点
               b. Decode (step 1+): 发送新 token + chain_info 给首节点
               c. 首节点处理 → 直连转发 hidden_states 给下一个节点（CHAIN_FORWARD）
               d. 中间节点处理 → 继续链式转发
               e. 末节点处理 → 直接返回 logits 给主节点（LAYER_RESULT）
               f. 主节点从 logits 采样下一个 token
               g. 判断 EOS / max_tokens → 继续或结束
            5. 广播 PIPELINE_DONE，各节点清理 KV cache

        **链式拓扑 (P2)**:
            - 主节点仅与首、末节点通信（O(1) 网络开销/step）
            - 中间节点间 TCP 直连转发 hidden_states
            - 每个 step 网络传输: N+1 次（vs 旧方案 2N 次）
            6. 解码完整序列 → 返回 response text

        Returns:
            {"response": str, "thinking": str, "metrics": dict, ...}
        """
        require_torch()
        import uuid
        from transport_port import MessageType, deserialize_tensor, serialize_tensor

        mgr = self._host
        if not mgr:
            return {"response": "", "error": "模型运行时不可用"}

        # ---- Step 1: 获取分层配置 ----
        layer_info = self.get_layer_assignments()
        assignments = [
            a for a in layer_info.get("assignments", [])
            if a.get("layers_count", 0) > 0
        ]
        assignments.sort(key=lambda a: a.get("start_layer", 0))

        master_ids = {"master", self.get_effective_node_id()}
        master_assignment = next(
            (a for a in assignments if a.get("node_id") in master_ids),
            None,
        )
        master_participates = bool(
            master_assignment and master_assignment.get("layers_count", 0) > 0
        )
        pipeline_nodes = [
            a for a in assignments
            if a.get("node_id") not in master_ids
        ]
        # 按 start_layer 排序，确保 worker 流水线顺序正确
        pipeline_nodes.sort(key=lambda a: a.get("start_layer", 0))

        if not pipeline_nodes:
            return {"response": "", "error": "没有可用的流水线从节点"}

        # master 参与时，在本地保留 Embedding + 首段 Transformer + LM Head。
        # 后续 step 由 master.forward_layers(input_ids) 生成 hidden_states，
        # 再交给第一个 worker；避免 RTX 独显主节点只做调度而不计算。
        if master_participates:
            try:
                ensure_layer_range = getattr(mgr, "ensure_layer_range", None)
                if callable(ensure_layer_range):
                    ensure_layer_range(
                        master_assignment["start_layer"],
                        master_assignment["end_layer"],
                        has_embedding=master_assignment.get("has_embedding", True),
                        has_lm_head=master_assignment.get("has_lm_head", True),
                    )
                else:
                    mgr.load_layer_range(
                        master_assignment["start_layer"],
                        master_assignment["end_layer"],
                        has_embedding=master_assignment.get("has_embedding", True),
                        has_lm_head=master_assignment.get("has_lm_head", True),
                    )
            except Exception as e:
                logger.error(f"❌ 主节点本地层范围加载失败: {e}", exc_info=True)
                return {"response": "", "error": f"主节点本地层范围加载失败: {e}"}

        if not mgr.tokenizer:
            return {"response": "", "error": "流水线 tokenizer 未加载"}
        tokenizer = mgr.tokenizer
        device = mgr.get_device()

        # ★ 二次就绪检查（出队后 / 立即执行前）
        #   入队等待期间节点可能离线，tokenize 前最后确认。
        ok, err_msg = self._verify_pipeline_readiness(pipeline_nodes)
        if not ok:
            logger.error(f"❌ 流水线就绪检查失败: {err_msg}")
            # Phase 5 review C3: 恢复完整模型，避免残留裁剪状态导致后续推理失败
            if master_participates:
                try:
                    ensure_full = getattr(mgr, 'ensure_full_model', None)
                    if callable(ensure_full):
                        ensure_full()
                except Exception as restore_err:
                    logger.warning(f"模型恢复失败（将继续）: {restore_err}")
            return {"response": "", "error": err_msg}

        # Freeze the worker configuration generation for the complete task.
        # A topology/model refresh after this point must not be mixed into an
        # already running token sequence.
        worker_ids = [node["node_id"] for node in pipeline_nodes]
        with self._layer_config_lock:
            worker_contracts = [
                dict(self._layer_config_expected.get(node_id, {}))
                for node_id in worker_ids
            ]
        config_ids = {item.get("config_id") for item in worker_contracts if item}
        model_hashes = {item.get("model_sha256") for item in worker_contracts if item}
        model_types = {item.get("model_type") for item in worker_contracts if item}
        if (len(worker_contracts) != len(worker_ids)
                or any(not item for item in worker_contracts)
                or len(config_ids) != 1
                or len(model_hashes) != 1
                or len(model_types) != 1):
            return {"response": "", "error": "worker 层配置代际不一致，请等待重新就绪"}
        pipeline_config_id = str(next(iter(config_ids)))
        pipeline_model_sha256 = str(next(iter(model_hashes)))
        pipeline_model_type = str(next(iter(model_types)))
        master_model_type = str(
            getattr(getattr(getattr(mgr, "model", None), "config", None), "model_type", "")
            or ""
        ).lower()
        if (master_model_type != pipeline_model_type
                or self._get_master_model_sha256() != pipeline_model_sha256):
            return {"response": "", "error": "主节点模型已变化，请等待层配置重新同步"}

        full_chain = ([master_assignment] if master_participates else []) + pipeline_nodes
        logger.info(
            f"🚀 启动流水线推理: prompt_len={len(prompt)}, "
            f"max_tokens={max_new_tokens}, worker数={len(pipeline_nodes)}, "
            f"顺序: {' → '.join(n['node_id'] for n in full_chain)}, "
            f"master_local={'✅' if master_participates else '❌'}, KV Cache: ✅"
        )

        # ---- Step 2: Tokenize ----
        chat_messages = messages or [{"role": "user", "content": prompt}]
        callbacks = self._require_callbacks()
        thinking_prompt = callbacks.thinking_system_prompt if show_thinking else None
        thinking_prefill = "【思考】\n" if show_thinking else None
        model_prompt = callbacks.build_model_chat_prompt(
            tokenizer,
            chat_messages,
            system_prompt=thinking_prompt,
            assistant_prefill=thinking_prefill,
        )
        inputs = tokenizer(model_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # (1, prompt_len)
        attention_mask = inputs.get("attention_mask")
        prompt_len = input_ids.shape[1]

        # ---- Step 3: 自回归生成 ----
        task_id = uuid.uuid4().hex[:12]
        with self._pipeline_lock:
            self._pipeline_active_tasks.add(task_id)
            self._pipeline_task_contracts[task_id] = {
                "config_id": pipeline_config_id,
                "model_sha256": pipeline_model_sha256,
                "model_type": pipeline_model_type,
                "worker_ids": worker_ids,
                "last_node_id": pipeline_nodes[-1]["node_id"],
                "current_step": -1,
            }
        self._pipeline_context.stack.append({
            "task_id": task_id,
            "pipeline_nodes": pipeline_nodes,
        })
        generated_ids = []
        merge_stops = getattr(mgr, "_merge_stop_sequences", None)
        stop_sequences = merge_stops(None) if callable(merge_stops) else []
        get_eos = getattr(mgr, "_get_generation_eos_token_ids", None)
        eos_token_ids = get_eos(stop_sequences) if callable(get_eos) else tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_ids = {tokenizer.eos_token_id}
        elif isinstance(eos_token_ids, int):
            eos_ids = {eos_token_ids}
        else:
            eos_ids = set(eos_token_ids)
        native_thinking_prompt = bool(
            not show_thinking and "<think" in model_prompt[-128:].lower()
        )
        suppress_native_thinking = native_thinking_prompt
        stream_buffer = ""
        workers_used = [n["node_id"] for n in pipeline_nodes]
        pipeline_metrics = {
            "steps": [],
            "total_time_ms": 0,
            "kv_cache": True,
            "chain_topology": True,
            "engine": "distributed_pipeline",
            "execution_mode": "distributed_pipeline",
            "distributed_requested": True,
            "distributed_used": True,
            "fallback": False,
            "fallback_reason": "",
            "route": "master_pipeline",
            "task_id": task_id,
            "serving_node_id": self.get_effective_node_id(),
            "workers_used": workers_used,
            "layer_assignments": pipeline_nodes,
        }
        t_pipeline_start = time.time()

        # 仅用于最终解码，不再用于发送
        full_input_ids = input_ids

        for step in range(max_new_tokens):
            if _cancel_event is not None and _cancel_event.is_set():
                step_error = "流水线任务已取消"
                self._broadcast_pipeline_abort(
                    pipeline_nodes, task_id, step_error, count_error=False
                )
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error, "cancelled": True}

            # ---- Phase 2: 协同抢占检查 ----
            # 在每个 decode 步边界检测 Q0 任务，若存在则执行内联抢占。
            # Prefill (step=0) 不抢占——此时尚未生成任何 token。
            if (step > 0
                    and self._scheduler_facade_global('PIPELINE_PREEMPT_ENABLED')
                    and not self._preempt_disabled
                    and self._check_preempt_conditions(step)):

                # ★ 原子检查 + 弹出（消除 TOCTOU 窗口）
                q0_task = None
                with self.pipeline_queue._lock:
                    if self.pipeline_queue._q0:
                        q0_task = self.pipeline_queue._q0.popleft()

                if q0_task is not None:
                    t_preempt = time.time()

                    # 保存被抢占任务的执行状态
                    preempt_state = self._save_preempt_state(
                        task_id=task_id,
                        generated_ids=generated_ids,
                        full_input_ids=full_input_ids,
                        current_step=step,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        prompt=prompt,
                        pipeline_nodes=pipeline_nodes,
                        first_node_id=pipeline_nodes[0]["node_id"],
                        _stream_callback=_stream_callback,
                    )

                    logger.info(
                        f"⚡ 抢占触发: step={step}, {task_id} "
                        f"→ Q0={q0_task.task_id} "
                        f"(已生成 {len(generated_ids)} tokens)"
                    )

                    # 释放推理锁，内联执行 Q0
                    self._inference_lock.release()

                    self._preempting = True  # ★ 防嵌套抢占
                    try:
                        self._execute_q0_inline(q0_task, preempt_state)
                        ensure_layer_range = getattr(mgr, "ensure_layer_range", None)
                        if callable(ensure_layer_range):
                            ensure_layer_range(
                                master_assignment["start_layer"],
                                master_assignment["end_layer"],
                                has_embedding=master_assignment.get("has_embedding", True),
                                has_lm_head=master_assignment.get("has_lm_head", True),
                            )
                    except Exception as e:
                        logger.error(
                            f"❌ Q0 抢占异常: {e}，中止 {task_id}"
                        )
                        # 尝试恢复锁平衡
                        try:
                            self._inference_lock.acquire()
                        except RuntimeError:
                            pass
                        self._broadcast_pipeline_abort(
                            pipeline_nodes, task_id, f"抢占失败: {e}"
                        )
                        self._preempted_task = None
                        self._preempting = False
                        self._clear_pipeline_runtime_state(task_id)
                        return {"response": "", "error": f"抢占失败: {e}"}
                    finally:
                        self._preempting = False

                    # 恢复被抢占任务状态
                    generated_ids = preempt_state.generated_ids
                    full_input_ids = preempt_state.full_input_ids
                    temperature = preempt_state.temperature
                    top_p = preempt_state.top_p
                    prompt = preempt_state.prompt
                    _stream_callback = preempt_state._stream_callback
                    self._preempted_task = None  # ★ M2: 清除泄漏

                    overhead_ms = (time.time() - t_preempt) * 1000
                    self._update_preempt_stats(overhead_ms)

                    logger.info(
                        f"🔄 抢占恢复: {task_id} step {step} "
                        f"(剩余 {max_new_tokens - step} tokens)"
                    )
                    # ★ 循环继续，step 不变——被推迟的这一步现在执行

            step_start = time.time()
            logits = None
            step_error = None

            with self._pipeline_lock:
                contract = self._pipeline_task_contracts.get(task_id)
                if contract is None:
                    return {"response": "", "error": "流水线任务执行契约已失效"}
                contract["current_step"] = step
                prefix = f"{task_id}:"
                for stale_key in list(self._pipeline_results):
                    if stale_key.startswith(prefix):
                        self._pipeline_results.pop(stale_key, None)

            # 判断 Prefill vs Decode
            is_prefill = (step == 0)

            # ---- 链式拓扑：构建节点链信息（P2 优化）----
            # 每个从节点收到 chain_next（下一个节点地址），处理完后直接
            # TCP 转发 hidden_states 给下一个节点。主节点仅与首尾节点通信。
            chain_info = []
            for i, node in enumerate(pipeline_nodes):
                nid = node["node_id"]
                addr = self._get_node_address(nid)
                chain_info.append({
                    "node_id": nid,
                    "host": addr["host"] if addr else "",
                    "port": addr["port"] if addr else 0,
                })

            first_node_id = pipeline_nodes[0]["node_id"]
            last_node_id = pipeline_nodes[-1]["node_id"]
            has_chain = len(pipeline_nodes) >= 2

            # ---- 构建 LAYER_FORWARD 消息（发给首个 worker）----
            forward_data = {
                "task_id": task_id,
                "step": step,
                "config_id": pipeline_config_id,
                "model_sha256": pipeline_model_sha256,
                "model_type": pipeline_model_type,
                "chain_path": [],
                "temperature": temperature,
                "top_p": top_p,
                "use_kv_cache": not is_prefill,  # ★ Prefill=False, Decode=True
            }

            if master_participates:
                # master 本地首段：input_ids → Embedding + master layers → hidden_states。
                # worker 不再需要 Embedding，因此收到的一定是 hidden_states。
                try:
                    past_kv = None
                    if not is_prefill:
                        with self._kv_cache_lock:
                            past_kv = self._kv_cache.get(task_id)
                        if past_kv is None:
                            raise RuntimeError(
                                f"主节点 decode step {step} 缺少 KV cache"
                            )
                    local_input_ids = input_ids if is_prefill else self._scheduler_facade_global('torch').tensor(
                        [[new_token_id]], dtype=self._scheduler_facade_global('torch').long
                    )
                    local_attention_mask = attention_mask if is_prefill else None

                    t_master = time.time()
                    local_result = mgr.forward_layers(
                        input_ids=local_input_ids,
                        attention_mask=local_attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        apply_lm_head=False,
                    )
                    master_elapsed_ms = (time.time() - t_master) * 1000
                    if local_result.get("past_key_values"):
                        with self._kv_cache_lock:
                            self._kv_cache[task_id] = local_result["past_key_values"]
                    if "hidden_states" not in local_result:
                        raise RuntimeError("主节点首段未返回 hidden_states")
                    hs_cpu = local_result["hidden_states"].detach().cpu()
                    import base64 as _b64
                    relay_segment = (
                        self._relay_segment_for_worker(first_node_id)
                        if master_participates else None
                    )
                    if relay_segment is not None:
                        forward_data["hidden_states"], forward_data["hidden_shape"] = (
                            _encode_relay_hidden(hs_cpu)
                        )
                        forward_data["hidden_wire_format"] = RELAY_HIDDEN_WIRE_FORMAT
                    else:
                        forward_data["hidden_states"] = _b64.b64encode(
                            serialize_tensor(hs_cpu)
                        ).decode("ascii")
                        forward_data["hidden_shape"] = list(hs_cpu.shape)
                    logger.debug(
                        f"🏠 Master 本地 Step {step}: Layer "
                        f"{master_assignment['start_layer']}-{master_assignment['end_layer']} "
                        f"hidden_states={list(hs_cpu.shape)}, time={master_elapsed_ms:.0f}ms"
                    )
                except Exception as e:
                    step_error = f"主节点本地首段 forward 失败: {e}"
                    logger.error(step_error, exc_info=True)
            else:
                if is_prefill:
                    # 兼容旧配置：首 worker 含 Embedding，发送完整 prompt input_ids
                    forward_data["input_ids"] = input_ids.cpu().tolist()
                    if attention_mask is not None:
                        forward_data["attention_mask"] = attention_mask.cpu().tolist()
                else:
                    # Decode: 仅发送最后 1 个 token
                    forward_data["input_ids"] = [[new_token_id]]

            if has_chain:
                # 链式拓扑：附加上下一个节点的地址信息
                forward_data["chain_next"] = chain_info[1] if len(chain_info) > 1 else None
                forward_data["chain_remaining"] = chain_info[2:] if len(chain_info) > 2 else []
                logger.debug(
                    f"🔗 Step {step} 链式路由: "
                    f"{'master → ' if master_participates else ''}"
                    f"{' → '.join(c['node_id'] for c in chain_info)}"
                )
            else:
                forward_data["chain_next"] = None
                forward_data["chain_remaining"] = []

            # ★ A1 / X 档（2026-09-24）：若该节点被配置为「由远端 relay 段代跑本段」，
            #   随 LAYER_FORWARD 下发规格；worker 侧仅在开关打开时才会认它（默认关 ⇒ 零影响）。
            relay_segment = (
                relay_segment if "relay_segment" in locals()
                else self._relay_segment_for_worker(first_node_id)
            )
            if relay_segment is not None:
                forward_data["relay_segment"] = relay_segment

            # ---- 发送给首个 worker ----
            try:
                if not step_error:
                    self._send_to_worker(first_node_id, forward_data, MessageType.LAYER_FORWARD)
            except Exception as e:
                step_error = f"发送到首节点 {first_node_id} 失败: {e}"
                logger.error(step_error)

            if step_error:
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # ---- 等待链上任一节点返回结果（末节点=成功，其他=错误）----
            result = self._wait_for_layer_result_with_ack(
                task_id,
                [n["node_id"] for n in pipeline_nodes],  # 任一节点都可能报错
                timeout=self._scheduler_facade_global('PIPELINE_STEP_TIMEOUT'),
                ack_node_ids=[n["node_id"] for n in pipeline_nodes[1:]] if has_chain else [],
                ack_step=step,
                ack_timeout=min(5.0, max(1.0, self._scheduler_facade_global('PIPELINE_STEP_TIMEOUT') / 6)),
                cancel_event=_cancel_event,
            )
            if _cancel_event is not None and _cancel_event.is_set():
                step_error = "流水线任务已取消"
                self._broadcast_pipeline_abort(
                    pipeline_nodes, task_id, step_error, count_error=False
                )
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error, "cancelled": True}
            if result is None:
                step_error = f"末节点 {last_node_id} 响应超时"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            if result.get("error"):
                step_error = f"流水线错误: {result['error']}"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # 提取末端输出。推荐拓扑由 worker 返回 hidden_states，主节点在
            # CUDA 上执行 Norm + LM Head；兼容旧配置直接返回 logits。
            if "logits" in result and result["logits"] is not None:
                logits_data = result["logits"]
                if isinstance(logits_data, bytes):
                    logits = deserialize_tensor(logits_data).to(device=device)
                elif self._scheduler_facade_global('torch') is not None and isinstance(logits_data, self._scheduler_facade_global('torch').Tensor):
                    logits = logits_data.to(device=device)
                else:
                    step_error = f"未知 logits 类型: {type(logits_data).__name__}"
                    logger.error(step_error)
            elif "hidden_states" in result and result["hidden_states"] is not None:
                hidden_data = result["hidden_states"]
                if isinstance(hidden_data, bytes):
                    if result.get("hidden_wire_format") == RELAY_HIDDEN_WIRE_FORMAT:
                        try:
                            final_hidden = _decode_relay_hidden(
                                hidden_data, result.get("hidden_shape")
                            )
                        except Exception as exc:
                            step_error = f"relay hidden 解码失败: {exc}"
                            logger.error(step_error)
                            final_hidden = None
                    else:
                        final_hidden = deserialize_tensor(hidden_data)
                elif self._scheduler_facade_global('torch') is not None and isinstance(hidden_data, self._scheduler_facade_global('torch').Tensor):
                    final_hidden = hidden_data
                else:
                    step_error = (
                        f"未知 hidden_states 类型: {type(hidden_data).__name__}"
                    )
                    logger.error(step_error)
                    final_hidden = None
                if final_hidden is not None:
                    try:
                        logits = self._run_master_lm_head(final_hidden)
                    except Exception as e:
                        step_error = f"主节点 LM Head 执行失败: {e}"
                        logger.error(step_error, exc_info=True)
            else:
                step_error = "末节点未返回 logits"
                logger.error(step_error)

            if step_error:
                # ★ 统一中止路径：广播 ABORT → 清理各节点 KV cache → 返回错误
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # ---- Step 4: 从 logits 选择下一个 token ----
            # temperature=0 与单机路径一致采用贪心解码；正温度才执行
            # FP32 top-p 采样并在进入 CUDA multinomial 前校验概率。
            new_token_id = self._scheduler_facade_global('_sample_pipeline_token_id')(
                logits, temperature=temperature, top_p=top_p,
            )

            # 检查 EOS
            if new_token_id in eos_ids:
                logger.info(f"🏁 EOS token 生成于 step {step}")
                break

            generated_ids.append(new_token_id)

            # ★ 流式回调：每生成一个 token 立即推送
            if _stream_callback:
                new_token_text = tokenizer.decode([new_token_id])
                if suppress_native_thinking:
                    stream_buffer += new_token_text
                    marker = stream_buffer.lower().find("</think>")
                    if marker >= 0:
                        visible = stream_buffer[marker + len("</think>"):]
                        suppress_native_thinking = False
                        stream_buffer = ""
                        if visible:
                            _stream_callback({"token": visible})
                else:
                    _stream_callback({"token": new_token_text})

            # 更新完整序列仅用于最终解码（不再发送给首节点）
            new_token_tensor = self._scheduler_facade_global('torch').tensor([[new_token_id]], dtype=self._scheduler_facade_global('torch').long)
            full_input_ids = self._scheduler_facade_global('torch').cat([full_input_ids, new_token_tensor], dim=1)

            step_ms = (time.time() - step_start) * 1000
            pipeline_metrics["steps"].append({
                "step": step,
                "token": new_token_id,
                "time_ms": round(step_ms, 1),
                "mode": "prefill" if is_prefill else "decode",
            })
            logger.info(
                f"🪜 Step {step}: token={new_token_id}, "
                f"seq_len={full_input_ids.shape[1]}, "
                f"mode={'prefill' if is_prefill else 'decode'}, "
                f"time={step_ms:.0f}ms"
            )

        # ---- Step 5: 广播 PIPELINE_DONE（各节点清理 KV cache） ----
        for n in pipeline_nodes:
            try:
                self._send_to_worker(
                    n["node_id"],
                    {"task_id": task_id},
                    MessageType.PIPELINE_DONE,
                )
            except Exception as e:
                logger.warning(
                    "发送 PIPELINE_DONE 失败: node=%s task=%s error=%s",
                    n.get("node_id"), task_id, e,
                )

        # ★ 清理 master 自身 KV cache（master_participates 路径会产生本地缓存）
        with self._kv_cache_lock:
            if task_id in self._kv_cache:
                del self._kv_cache[task_id]

        # ---- Step 6: 解码结果 ----
        if generated_ids:
            full_ids = self._scheduler_facade_global('torch').cat([
                input_ids.squeeze(0),
                self._scheduler_facade_global('torch').tensor(generated_ids, dtype=self._scheduler_facade_global('torch').long)
            ], dim=0)
            response_text = tokenizer.decode(full_ids, skip_special_tokens=True)
            raw_new_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            response_text = tokenizer.decode(
                input_ids.squeeze(0), skip_special_tokens=True
            )
            raw_new_text = ""

        new_text, thinking_content = self._require_callbacks().format_model_response(
            raw_new_text,
            show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )

        pipeline_metrics["total_time_ms"] = round(
            (time.time() - t_pipeline_start) * 1000, 1
        )
        pipeline_metrics["tokens_generated"] = len(generated_ids)
        pipeline_metrics["generated_tokens"] = len(generated_ids)
        pipeline_metrics["nodes_used"] = len(pipeline_nodes)
        pipeline_metrics["elapsed_seconds"] = round(pipeline_metrics["total_time_ms"] / 1000, 3)

        tokens_per_sec = (
            len(generated_ids) / (pipeline_metrics["total_time_ms"] / 1000)
            if pipeline_metrics["total_time_ms"] > 0 and generated_ids
            else 0
        )
        pipeline_metrics["tokens_per_second"] = round(tokens_per_sec, 1)

        accounting = self._record_pipeline_task_accounting(
            task_id=task_id,
            pipeline_nodes=pipeline_nodes,
            success=True,
        )
        pipeline_metrics["node_task_accounting"] = accounting
        pipeline_metrics["workers_counted"] = accounting.get("workers_counted", [])
        pipeline_metrics["counted_nodes"] = accounting.get("counted_nodes", [])

        logger.info(
            f"✅ 流水线推理完成: {len(generated_ids)} tokens, "
            f"{pipeline_metrics['total_time_ms']:.0f}ms, "
            f"{tokens_per_sec:.1f} tok/s (KV Cache: ✅)"
        )

        result = {
            "response": new_text,
            "full_text": response_text,
            "thinking": thinking_content,
            "metrics": pipeline_metrics,
        }

        # ★ 流式完成通知
        if _stream_callback:
            _stream_callback({"done": True, **result})

        self._clear_pipeline_runtime_state(task_id)
        return result


    def run_pipeline_stream(self, prompt: str, **kwargs):
        """
        流式版本：逐 token yield 事件字典，用于 SSE 推送。

        内部通过线程+队列包装 run_pipeline() 的 _stream_callback，
        将 callback 调用转为 generator yield。

        Yields:
            {"token": str}       — 新生成的 token 文本
            {"done": True, "response": str, "metrics": dict, ...}
                                  — 完成信号（含完整响应和指标）
            {"done": True, "error": str}
                                  — 错误信号
        """
        import queue
        import threading as _thr

        q = queue.Queue()
        callback_called = _thr.Event()
        cancel_event = kwargs.pop("_cancel_event", None) or _thr.Event()

        def on_token(event):
            if "done" in event:
                callback_called.set()
            q.put(event)

        def _run():
            try:
                result = self.run_pipeline_safe(
                    prompt,
                    _stream_callback=on_token,
                    _cancel_event=cancel_event,
                    **kwargs,
                )
                # 错误路径：run_pipeline 直接返回了 error（未走 callback）
                if not callback_called.is_set():
                    q.put({
                        "done": True,
                        "error": result.get("error", "unknown"),
                        "response": result.get("response", ""),
                        "metrics": result.get("metrics", {}),
                    })
            except Exception as e:
                logger.error(f"流式推理异常: {e}", exc_info=True)
                q.put({"done": True, "error": str(e)})

        _thr.Thread(target=_run, name="pipeline-stream", daemon=True).start()

        try:
            while True:
                event = q.get()
                yield event
                if "done" in event:
                    break
        finally:
            if not callback_called.is_set():
                cancel_event.set()


    @staticmethod
    def _stream_output_started(kwargs: dict) -> bool:
        """返回流式调用是否已向客户端发送过正文 token。"""
        callback = kwargs.get("_stream_callback")
        return bool(getattr(callback, "_qlh_tokens_emitted", False))


    @staticmethod
    def _track_stream_output(kwargs: dict) -> None:
        """包装流式回调，供失败回退判断是否会造成回答重放。"""
        callback = kwargs.get("_stream_callback")
        if not callable(callback) or getattr(callback, "_qlh_stream_tracker", False):
            return

        def tracked_callback(event):
            if isinstance(event, dict) and event.get("token"):
                tracked_callback._qlh_tokens_emitted = True
            callback(event)

        tracked_callback._qlh_stream_tracker = True
        tracked_callback._qlh_tokens_emitted = False
        kwargs["_stream_callback"] = tracked_callback


    def _process_queued_pipeline_task(self, prompt: str, **kwargs) -> dict:
        """
        队列工作线程的回调：执行流水线推理并返回结果。

        ★ 直接调用 run_pipeline（绕过 run_pipeline_safe 的排队检查），
           避免死锁：队列 worker 已设置 _current_task_id，若走 run_pipeline_safe
           会再次检测 is_busy=True → enqueue → 永久等待自己完成。

        ★ 手动管理 _inference_lock：正常路径在 finally 中释放；
           抢占路径中 run_pipeline 内部会 release/re-acquire，
           返回时锁仍被持有，由 finally 统一释放。
        """
        self._inference_lock.acquire()
        lock_held = True
        try:
            # 检查节点是否就绪
            require_distributed = bool(kwargs.pop("_require_distributed", False))
            force_distributed = bool(
                kwargs.pop("_force_distributed_assignment", require_distributed)
            )
            sync_timeout = kwargs.pop(
                "_pipeline_model_sync_timeout", self._scheduler_facade_global('PIPELINE_MODEL_SYNC_TIMEOUT'),
            )
            pipeline_ready = self._all_pipeline_nodes_ready()
            if force_distributed and not self._has_active_distributed_pipeline_plan():
                pipeline_ready = False
            if not pipeline_ready and force_distributed:
                readiness = self._synchronize_pipeline_workers_for_request(
                    timeout=sync_timeout,
                    force_distributed_assignment=True,
                )
                pipeline_ready = bool(readiness.get("ready"))
            if not pipeline_ready:
                if require_distributed:
                    return {
                        "response": "",
                        "error": (
                            "distributed_required: queued pipeline workers not ready"
                        ),
                        "metrics": {"distributed_used": False, "fallback": False},
                    }
                logger.warning("流水线节点不可用，队列任务回退到全模型推理")
                # ★ H1 修复: 保持 lock_held=True，回退推理在锁保护下执行（防止 GPU 并发）
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason="queue_pipeline_nodes_not_ready",
                    **kwargs,
                )
            result = self.run_pipeline(prompt, **kwargs)
            if result.get("error"):
                if self._stream_output_started(kwargs):
                    logger.warning(
                        "流水线已输出部分内容，跳过全模型回退以避免重复回答: %s",
                        result.get("error"),
                    )
                    return result
                logger.warning(
                    "队列任务流水线单步失败，回退到全模型推理: %s",
                    result.get("error"),
                )
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=f"queue_pipeline_error_result: {result.get('error')}",
                    **kwargs,
                )
            return result
        except Exception as e:
            if self._stream_output_started(kwargs):
                logger.error(
                    "流水线已输出部分内容后异常，跳过全模型回退: %s",
                    e,
                    exc_info=True,
                )
                return {"response": "", "error": str(e)}
            logger.error(f"队列任务流水线推理失败: {e}，回退到全模型推理", exc_info=True)
            # 锁可能在抢占异常路径中已被释放
            try:
                self._inference_lock.release()
                lock_held = False
            except RuntimeError:
                lock_held = False  # 抢占路径中锁已被 release
            # Phase 5 review H2: 回退推理需持有推理锁
            self._inference_lock.acquire()
            lock_held = True
            return self._run_full_model_inference(
                prompt,
                _fallback_reason=f"queue_pipeline_error: {e}",
                **kwargs,
            )
        finally:
            if lock_held:
                self._inference_lock.release()


    def run_pipeline_safe(self, prompt: str, **kwargs) -> dict:
        """
        带自动回退的流水线推理（支持排队）。

        规则:
        - 流水线节点不可用 → 回退到全模型推理
        - 队列中有任务执行中 → 新请求自动入队等待
        - 队列空闲 → 立即执行

        ★ 立即执行路径与 is_busy 检查在同一锁内完成，消除 TOCTOU 竞态：
          多个调用方线程不可能同时看到 is_busy=False 并绕过队列。
        """
        # ---- 引擎检查：流水线仅支持 PyTorch 引擎 ----
        # llama.cpp(GGUF) 不支持层拆分，直接走全模型推理。
        # 已显式准备的 distributed-only 模型尚未物化任何权重，也允许进入；
        # 主节点首段会在 worker 就绪后由 run_pipeline 按分配范围首次加载。
        queue_timeout = kwargs.pop('_queue_timeout', self._scheduler_facade_global('PIPELINE_TIMEOUT'))
        require_distributed = bool(kwargs.pop("_require_distributed", False))
        force_distributed_assignment = bool(
            kwargs.pop("_force_distributed_assignment", require_distributed)
        )
        model_sync_timeout = kwargs.pop(
            '_pipeline_model_sync_timeout', self._scheduler_facade_global('PIPELINE_MODEL_SYNC_TIMEOUT'),
        )
        self._track_stream_output(kwargs)
        mgr = self._host
        pipeline_prepared = bool(
            mgr and getattr(mgr, "is_pipeline_prepared", False)
        )
        if not mgr or (not mgr.is_loaded and not pipeline_prepared):
            logger.warning("模型未加载，无法执行流水线推理")
            if require_distributed:
                return {
                    "response": "",
                    "error": "distributed_required: pipeline model is not prepared",
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            return self._run_full_model_inference(
                prompt,
                _fallback_reason="model_not_loaded_for_pipeline",
                **kwargs,
            )
        engine_type = backend_id_for(mgr)
        if engine_type and not runtime_supports(mgr, Capability.FORWARD_LAYERS):
            logger.info(
                f"引擎类型为 {engine_type}，不支持流水线层拆分，"
                f"使用全模型推理"
            )
            if require_distributed:
                return {
                    "response": "",
                    "error": (
                        "distributed_required: pipeline execution requires "
                        f"pytorch, got {engine_type}"
                    ),
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            return self._run_full_model_inference(
                prompt,
                _fallback_reason=f"engine {engine_type} does not support layer-split pipeline",
                **kwargs,
            )

        # ---- 自动回退：节点不可用 → 全模型推理 ----
        readiness = None
        try:
            pipeline_ready = (
                False
                if force_distributed_assignment
                else self._all_pipeline_nodes_ready()
            )
            if not pipeline_ready:
                if force_distributed_assignment:
                    readiness = self._synchronize_pipeline_workers_for_request(
                        timeout=model_sync_timeout,
                        force_distributed_assignment=True,
                    )
                else:
                    readiness = self._synchronize_pipeline_workers_for_request(
                        timeout=model_sync_timeout,
                    )
                pipeline_ready = bool(readiness.get("ready"))
        except Exception:
            logger.warning(
                "请求前主从模型配置同步失败，将按未就绪处理",
                exc_info=True,
            )
            pipeline_ready = False

        if not pipeline_ready:
            try:
                readiness = readiness or self._get_pipeline_readiness()
                readiness_reason = readiness.get("reason") or "未知原因"
            except Exception:
                readiness_reason = "就绪状态检查失败"
            logger.warning(
                "部分流水线节点未就绪，回退到全层主节点模式: %s",
                readiness_reason,
            )
            if require_distributed:
                return {
                    "response": "",
                    "error": (
                        "distributed_required: pipeline workers not ready: "
                        f"{readiness_reason}"
                    ),
                    "metrics": {
                        "distributed_used": False,
                        "fallback": False,
                        "pipeline_readiness": readiness or {},
                    },
                }
            # 回退仍会执行完整模型推理，必须与其他 GPU 推理共享同一把锁。
            # 这里阻塞等待，避免锁被占用时直接绕过互斥保护。
            self._inference_lock.acquire()
            try:
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=(
                        f"pipeline_nodes_not_ready: {readiness_reason}"
                    ),
                    **kwargs,
                )
            finally:
                self._inference_lock.release()

        # ---- 排队逻辑（锁内原子判断 + 入队/执行）----
        # ★ 同时检查 is_busy 和 queue_size，消除竞态缺口：
        #   T1 刚完成（_current_task_id=None）但队列还残留 T2 的请求，
        #   此时 T3 若仅检查 is_busy 会绕过队列直接执行 → T2 被插队。
        with self.pipeline_queue._lock:
            if self.pipeline_queue.is_busy or self.pipeline_queue.queue_size > 0:
                # 有任务执行中 或 队列非空 → 入队（保证 FIFO 顺序）
                queued_kwargs = dict(kwargs)
                # Preserve request-scoped routing semantics across the queue.
                # These flags are consumed by _process_queued_pipeline_task
                # before it calls run_pipeline, so they never reach the model
                # engine as unexpected keyword arguments.
                queued_kwargs.update({
                    "_require_distributed": require_distributed,
                    "_force_distributed_assignment": force_distributed_assignment,
                    "_pipeline_model_sync_timeout": model_sync_timeout,
                })
                task_id = self.pipeline_queue.enqueue(
                    prompt=prompt, **queued_kwargs,
                )
            else:
                # 空闲且队列空 → 标记为"即将执行"（阻止其他线程绕过队列）
                task_id = None
                self.pipeline_queue._current_task_id = "__reserved__"

        if task_id is not None:
            # 入队路径：阻塞等待结果
            logger.info(
                f"⏳ 流水线正忙，请求已排队: task={task_id}, "
                f"queue_depth={self.pipeline_queue.queue_size}"
            )
            result = self.pipeline_queue.wait_for_result(
                task_id,
                timeout=queue_timeout,
                cancel_event=kwargs.get("_cancel_event"),
            )
            if result.get("status") == "done":
                payload = result.get("result", {})
                if isinstance(payload, dict) and payload.get("error"):
                    if self._stream_output_started(kwargs):
                        return payload
                    if require_distributed:
                        return payload
                    self._inference_lock.acquire()
                    try:
                        logger.warning(
                            "排队流水线任务返回错误，回退到全模型推理: %s",
                            payload.get("error"),
                        )
                        return self._run_full_model_inference(
                            prompt,
                            _fallback_reason=f"queued_pipeline_error_result: {payload.get('error')}",
                            **kwargs,
                        )
                    finally:
                        self._inference_lock.release()
                return payload
            elif result.get("status") == "timeout":
                self.pipeline_queue.cancel_task(task_id)
                return {"response": "", "error": f"排队超时 ({queue_timeout}s)"}
            else:
                return {"response": "", "error": result.get("error", "排队请求失败")}

        # ---- 立即执行（已通过原子检查）----
        # ★ 非阻塞获取推理锁：防止与 _process_loop 残留任务并发
        if not self._inference_lock.acquire(blocking=False):
            logger.warning("推理引擎正忙（锁竞争），返回繁忙错误")
            self.pipeline_queue._current_task_id = None
            return {"response": "", "error": "推理引擎正忙，请稍后重试"}
        try:
            try:
                result = self.run_pipeline(prompt, **kwargs)
                if result.get("error"):
                    if self._stream_output_started(kwargs):
                        return result
                    if require_distributed:
                        return result
                    logger.warning(
                        "流水线推理返回错误，回退到全层主节点模式: %s",
                        result.get("error"),
                    )
                    return self._run_full_model_inference(
                        prompt,
                        _fallback_reason=f"pipeline_error_result: {result.get('error')}",
                        **kwargs,
                    )
                return result
            except Exception as e:
                if self._stream_output_started(kwargs):
                    return {"response": "", "error": str(e)}
                if require_distributed:
                    return {
                        "response": "",
                        "error": f"distributed_required: {e}",
                        "metrics": {"distributed_used": False, "fallback": False},
                    }
                logger.error(f"流水线推理失败: {e}，回退到全层主节点模式", exc_info=True)
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=f"pipeline_error: {e}",
                    **kwargs,
                )
        finally:
            self._inference_lock.release()
            # ★ 释放预留标记（无论成功/失败/回退）
            if task_id is None:
                self.pipeline_queue._current_task_id = None


    def _run_full_model_inference(self, prompt: str,
                                   max_new_tokens: int = 512,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9,
                                   session_id: str = None,
                                   **kwargs) -> dict:
        """
        回退模式：在主节点本地执行完整模型推理。

        当流水线节点不可用时，使用 model_manager.chat() 直接推理。
        若调用方传入 _stream_callback，则使用 chat_stream() 逐 token 推送。
        """
        mgr = self._host
        if mgr and getattr(mgr, "is_pipeline_prepared", False):
            return {
                "response": "",
                "error": (
                    "当前模型仅以分布式流水线模式准备，禁止整模回退；"
                    "请等待从节点就绪或显式执行普通模型加载"
                ),
            }
        if not mgr or not mgr.is_loaded:
            return {"response": "", "error": "模型未加载"}

        _stream_callback = kwargs.pop('_stream_callback', None)
        fallback_reason = kwargs.pop('_fallback_reason', '') or 'pipeline_fallback_full_model'
        show_thinking = bool(kwargs.pop("show_thinking", False))
        # ★ 2026-09-19：深度思考**开关**（与 show_thinking 的「展示」语义区分开）。
        #   None ⇒ 不干预，沿用模型模板默认（Qwen3 模板默认会思考，故会出现超长 <think>）。
        #   显式 False ⇒ 引擎 `_set_thinking_mode(False)` 会经 chat template 的
        #   `enable_thinking=False` **真正阻止**模型生成思考内容（省算力），
        #   而不是靠事后剥离（后者依赖模板含 `<think>` 且能找到 `</think>`，任一不成立即失效）。
        enable_thinking = kwargs.pop("enable_thinking", None)
        cancel_event = kwargs.pop("_cancel_event", None)

        # ★ 若 master 刚执行过流水线裁剪（layer_range != None），
        #   需要先重新加载完整模型，否则 chat()/chat_stream() 会因
        #   缺 Embedding/LM Head 而报错（如 RuntimeError: 缺少 lm_head）。
        try:
            ensure_full = getattr(mgr, 'ensure_full_model', None)
            if callable(ensure_full):
                ensure_full()
        except Exception as e:
            logger.error(f"完整模型重载失败: {e}")
            return {"response": "", "error": f"完整模型恢复失败: {e}"}

        try:
            messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
            callbacks = self._require_callbacks()
            if show_thinking and not any(item.get("role") == "system" for item in messages):
                messages = [
                    {"role": "system", "content": callbacks.thinking_system_prompt},
                    *messages,
                ]
            try:
                fallback_prompt = callbacks.build_model_chat_prompt(mgr.tokenizer, messages)
                native_thinking_prompt = "<think>" in fallback_prompt[-128:].lower()
            except Exception:
                native_thinking_prompt = False

            if _stream_callback:
                # 流式路径：逐 token 推送
                full_text_parts = []
                visible_buffer = ""
                suppress_thinking = bool(native_thinking_prompt and not show_thinking)
                t0 = time.time()
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                ):
                    if chunk:
                        full_text_parts.append(chunk)
                        if suppress_thinking:
                            visible_buffer += chunk
                            marker = visible_buffer.lower().find("</think>")
                            if marker >= 0:
                                visible = visible_buffer[marker + len("</think>"):]
                                suppress_thinking = False
                                visible_buffer = ""
                                if visible:
                                    _stream_callback({"token": visible})
                        else:
                            _stream_callback({"token": chunk})
                raw_response_text = "".join(full_text_parts)
                response_text, thinking_content = callbacks.format_model_response(
                    raw_response_text,
                    show_thinking,
                    native_thinking_prompt=native_thinking_prompt,
                )
                elapsed = time.time() - t0
                metrics = {
                    "engine": backend_id_for(mgr, default='unknown') or 'unknown',
                    "mode": "fallback_full_model_streaming",
                    "execution_mode": "fallback_full_model_streaming",
                    "distributed_requested": True,
                    "distributed_used": False,
                    "fallback": True,
                    "fallback_reason": fallback_reason,
                    "route": "master_pipeline_fallback_full_model_streaming",
                    "serving_node_id": self.get_effective_node_id(),
                    "workers_used": [],
                    "layer_assignments": [],
                    "tokens_per_second": len(full_text_parts) / elapsed if elapsed > 0 else 0,
                    "chunks": len(full_text_parts),
                    "elapsed_seconds": round(elapsed, 3),
                }
                # ★ 发送完成信号（与 run_pipeline 一致）
                _stream_callback({
                    "done": True,
                    "response": response_text,
                    "thinking": thinking_content,
                    "metrics": metrics,
                })
            else:
                result = mgr.chat(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                )
                raw_response_text = result.get("content", "")
                response_text, thinking_content = callbacks.format_model_response(
                    raw_response_text,
                    show_thinking,
                    native_thinking_prompt=native_thinking_prompt,
                )
                usage = result.get("usage", {}) or {}
                completion_tokens = usage.get("completion_tokens", 0)
                metrics = {
                    "engine": backend_id_for(mgr, default='unknown') or 'unknown',
                    "mode": "fallback_full_model",
                    "execution_mode": "fallback_full_model",
                    "distributed_requested": True,
                    "distributed_used": False,
                    "fallback": True,
                    "fallback_reason": fallback_reason,
                    "route": "master_pipeline_fallback_full_model",
                    "serving_node_id": self.get_effective_node_id(),
                    "workers_used": [],
                    "layer_assignments": [],
                    "tokens_per_second": result.get("tokens_per_second", 0),
                    "generated_tokens": completion_tokens,
                    "completion_tokens": completion_tokens,
                    "usage": usage,
                }

            return {
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"全模型回退推理失败: {e}")
            return {"response": "", "error": str(e)}


    def _run_full_model_inference_stream(self, prompt: str, **kwargs):
        """
        单机 PyTorch 流式推理 — 逐 token yield 事件字典，用于 SSE 推送。

        通过线程+队列包装 model_manager.chat_stream()，
        将文本 chunk 转为 {"token": text} 事件。

        Yields:
            {"token": str}       — 增量文本 chunk
            {"done": True, "response": str, "metrics": dict}
                                  — 完成信号
            {"done": True, "error": str}
                                  — 错误信号
        """
        import queue
        import threading as _thr

        mgr = self._host
        if mgr and getattr(mgr, "is_pipeline_prepared", False):
            yield {
                "done": True,
                "error": (
                    "当前模型仅以分布式流水线模式准备，禁止整模回退；"
                    "请等待从节点就绪或显式执行普通模型加载"
                ),
            }
            return
        if not mgr or not mgr.is_loaded:
            yield {"done": True, "error": "模型未加载"}
            return

        try:
            callbacks = self._require_callbacks()
        except RuntimeError as e:
            yield {"done": True, "error": str(e)}
            return

        self._inference_lock.acquire()
        try:
            ensure_full = getattr(mgr, "ensure_full_model", None)
            if callable(ensure_full):
                ensure_full()
        except Exception as e:
            self._inference_lock.release()
            yield {"done": True, "error": f"完整模型恢复失败: {e}"}
            return

        max_new_tokens = kwargs.pop('max_new_tokens', 512)
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)
        show_thinking = bool(kwargs.pop('show_thinking', False))
        # ★ 2026-09-19：深度思考**开关**（同另一处路径；None ⇒ 不干预，沿用模板默认）。
        enable_thinking = kwargs.pop('enable_thinking', None)
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        engine_name = backend_id_for(mgr, default="pytorch") or "pytorch"
        try:
            model_prompt = callbacks.build_model_chat_prompt(mgr.tokenizer, messages)
            native_thinking_prompt = "<think>" in model_prompt[-128:].lower()
        except Exception:
            native_thinking_prompt = bool(
                engine_name == "llama_cpp"
                and getattr(mgr, "_chat_template", "") == "qwen3_chat_v1"
                and not getattr(mgr, "_thinking_controlled", False)
            )

        q = queue.Queue()
        full_text_parts = []
        error_info = [None]
        metrics_info = [{}]
        cancel_event = kwargs.pop("_cancel_event", None) or _thr.Event()

        def _run():
            try:
                t0 = time.time()
                token_count = 0
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    show_thinking=show_thinking,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                ):
                    if chunk:
                        full_text_parts.append(chunk)
                        token_count += 1
                        q.put({"token": chunk})
                elapsed = time.time() - t0
                metrics_info[0] = {
                    "mode": "single_streaming",
                    "engine": engine_name,
                    "chunks": token_count,
                    "elapsed_seconds": round(elapsed, 3),
                }
            except Exception as e:
                logger.error(f"单机流式推理异常: {e}", exc_info=True)
                error_info[0] = str(e)
            finally:
                q.put(None)  # sentinel

        worker = _thr.Thread(target=_run, name="full-model-stream", daemon=True)
        suppress_thinking = bool(native_thinking_prompt and not show_thinking)
        visible_buffer = ""
        try:
            worker.start()
            while True:
                event = q.get()
                if event is None:
                    break
                chunk = event.get("token", "")
                if suppress_thinking:
                    visible_buffer += chunk
                    marker = visible_buffer.lower().find("</think>")
                    if marker >= 0:
                        visible = visible_buffer[marker + len("</think>"):]
                        suppress_thinking = False
                        visible_buffer = ""
                        if visible:
                            yield {"token": visible}
                else:
                    yield event
        finally:
            cancel_event.set()
            if worker.is_alive():
                worker.join()
            self._inference_lock.release()

        raw_response_text = "".join(full_text_parts)
        response_text, thinking_content = callbacks.format_model_response(
            raw_response_text,
            show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )
        if error_info[0]:
            yield {
                "done": True,
                "error": error_info[0],
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics_info[0],
            }
        else:
            yield {
                "done": True,
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics_info[0],
            }


    def _get_pipeline_status(self) -> dict:
        """
        获取流水线模式状态（供前端展示）。

        Returns:
            {
                "available": bool,       # 条件是否满足（PyTorch + 分布式 + 有从节点）
                "active": bool,          # 当前是否可用（所有节点在线）
                "degraded": bool,        # 降级模式（部分从节点离线）
                "worker_count": int,     # 流水线从节点总数
                "online_worker_count": int,  # 在线从节点数
                "engine_compatible": bool,   # 引擎是否兼容（PyTorch）
                "workers": [             # 各从节点详情
                    {node_id, online, layer_range, has_embedding, has_lm_head}
                ],
            }
        """
        # 检查引擎兼容性
        mgr = self._host
        engine_ok = (
            mgr is not None
            and (
                getattr(mgr, 'is_loaded', False)
                or getattr(mgr, 'is_pipeline_prepared', False)
            )
            and runtime_supports(mgr, Capability.FORWARD_LAYERS)
        )

        # 获取分层配置
        layer_info = self.get_layer_assignments()
        workers = [
            a for a in layer_info.get("assignments", [])
            if a.get("node_id") != "master"
        ]
        workers.sort(key=lambda a: a.get("start_layer", 0))

        readiness = self._get_pipeline_readiness()
        readiness_by_node = {
            item["node_id"]: item for item in readiness.get("workers", [])
        }
        worker_status = []
        online_count = 0
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)
        for w in workers:
            nid = w["node_id"]
            node = nodes_snapshot.get(nid)
            is_online = node.is_available() if node else False
            if is_online:
                online_count += 1
            detail = readiness_by_node.get(nid, {})
            worker_status.append({
                "node_id": nid,
                "online": is_online,
                "tcp_connected": detail.get("tcp_connected", False),
                "heartbeat_age_seconds": detail.get("heartbeat_age_seconds"),
                "layer_ready": detail.get("layer_ready", False),
                "layer_status": detail.get("layer_status", "not_configured"),
                "layer_error": detail.get("layer_error", ""),
                "model_id": detail.get("model_id", ""),
                "layer_range": [w.get("start_layer", 0), w.get("end_layer", 24)],
                "has_embedding": w.get("has_embedding", False),
                "has_lm_head": w.get("has_lm_head", False),
            })

        distributed_enabled = self.get_distributed_inference_enabled()
        available = (
            engine_ok
            and self._scheduler_facade_global('RUN_MODE') == "distributed"
            and self._effective_role() == "master"
            and len(workers) > 0
            and distributed_enabled
        )
        active = available and readiness.get("ready", False)
        degraded = available and not active and online_count > 0

        if not distributed_enabled:
            reason_code = "distributed_disabled"
            reason = "分布式推理开关已关闭"
        elif self._scheduler_facade_global('RUN_MODE') != "distributed":
            reason_code = "not_distributed_mode"
            reason = "当前不是 distributed 运行模式"
        elif self._effective_role() != "master":
            reason_code = "not_master"
            reason = "当前节点不是主节点"
        elif not engine_ok:
            reason_code = "engine_not_pytorch"
            reason = "主节点必须加载 PyTorch 引擎模型才能进行模型层拆分"
        else:
            reason_code = readiness.get("reason_code", "unknown")
            reason = readiness.get("reason", "流水线状态未知")

        return {
            "available": available,
            "active": active,
            "degraded": degraded,
            "worker_count": len(workers),
            "online_worker_count": online_count,
            "engine_compatible": engine_ok,
            "distributed_enabled": distributed_enabled,
            "readiness_reason_code": reason_code,
            "readiness_reason": reason,
            "workers": worker_status,
        }
