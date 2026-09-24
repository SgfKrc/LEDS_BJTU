"""
调度控制模块 — 节点状态管理、推理任务分发、流水线调度
========================================================
功能职责:
1. 节点状态管理（空闲/忙碌/离线）
2. 推理任务分发、流程启停
3. 异常捕获、错误上报
4. 流水线数据流调度控制
5. TCP 服务端集成 — 接收从节点注册、维护连接

依赖: threading, logging, socket
"""

import collections
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from enum import Enum
from typing import Any, Mapping, Optional, Callable, Sequence, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from model_host import InferenceHost, SchedulerCallbacks

from model_host import get_model_host
from koakuma_engine import (
    Capability,
    backend_capabilities,
    backend_id_for,
    runtime_supports,
)
from network_path import build_client_network_path_view
from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
from pipeline_node_contract import (
    PipelineNodeContractError,
    build_aggregate_resource_view,
    pipeline_layout_from_capacity_plan,
)
from pipeline_reshard import PipelineArtifactAvailability, PipelineReshardCoordinator
import scheduler_layer_plan as _layer_plan
from scheduler_types import InferenceTask, NodeInfo, NodeRole, NodeState, PreemptState, QueueTask
from scheduler_sidecars import SchedulerSidecarMixin
from llama_rpc_contract import RpcShardLeaseBook
from qwen3_pipeline_transaction import (
    Qwen3PipelineDryRunTransaction,
    Qwen3PipelineProtocolError,
)
from qwen3_pipeline_loopback import (
    Qwen3LoopbackError,
    Qwen3PipelineLoopbackWorker,
    sign_loopback_message,
    validate_loopback_base_url,
    verify_loopback_message,
)
from qwen3_pipeline_sidecar import Qwen3PipelineSidecarSession, Qwen3SidecarError
from qwen3_pipeline_multisidecar import (
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
    cleanup_qwen3_local_artifacts,
)
from gemma4_pipeline_multisidecar import (
    Gemma4MultiSidecarError,
    Gemma4PipelineMultiSidecar,
)
from gemma4_pipeline_sidecar import (
    Gemma4PipelineSidecarSession,
    Gemma4SidecarError,
)
from cluster_fence import ControlFence
from cluster_auto_role import AutoRoleController
from cluster_handoff import HandoffCoordinator

from task_provider import (
    ModelIdentity as TaskModelIdentity,
    StageRequest as TaskProviderStageRequest,
    sanitize_result_metadata as sanitize_task_result_metadata,
)
from task_worker_adapter import (
    RemoteFullWorkerProvider,
    TaskWorkerControlPlane,
    remote_provider_id,
)
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

# PyTorch is optional. Keep it out of the llama.cpp control-plane import path.
from torch_runtime import LazyTorch, require_torch, torch_available

torch = LazyTorch()

from config import (
    RUN_MODE, HEARTBEAT_INTERVAL,
    SERVER_IP, SERVER_PORT,
    NODE_ROLE, NODE_ID, MAX_NODES,
    PIPELINE_TIMEOUT, PIPELINE_MODEL_SYNC_TIMEOUT, PIPELINE_STEP_TIMEOUT,
    PIPELINE_QUEUE_POLL_INTERVAL,
    PIPELINE_QUEUE_MAX_SIZE, PIPELINE_QUEUE_RESULT_TTL,
    PIPELINE_SCHEDULING_STRATEGY,
    PIPELINE_Q0_MAX_TOKENS, PIPELINE_Q1_MAX_TOKENS,
    PIPELINE_AGING_Q1_TO_Q0_SECONDS, PIPELINE_AGING_Q2_TO_Q1_SECONDS,
    PIPELINE_AGING_MAX_WAIT_SECONDS,
    PIPELINE_PREEMPT_ENABLED,
    PIPELINE_PREEMPT_MIN_INTERVAL,       # 两次抢占最小间隔（防抖动）
    PIPELINE_PREEMPT_MIN_TOKENS,         # 至少生成 N token 后才接受抢占
    PIPELINE_PREEMPT_MAX_OVERHEAD_MS,    # checkpoint 超限自动禁用
    TASK_WORKER_EXPERIMENTAL_ENABLED,
)

logger = logging.getLogger(__name__)

ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS = 45
ANDROID_HTTP_CLIENT_LEASE_SECONDS = 120
ANDROID_HTTP_CLIENT_TIMEOUT_SECONDS = ANDROID_HTTP_CLIENT_LEASE_SECONDS
_LAYER_ASSIGNMENT_CACHE_VERSION = 3

from scheduler_task_worker import SchedulerTaskWorkerMixin, _TaskWorkerActiveAttempt
from scheduler_cluster import SchedulerClusterMixin
from scheduler_pipeline import SchedulerPipelineMixin

# Keep the current scheduler import surface explicit while the implementation
# is split into smaller modules. Private helpers listed here are compatibility
# hooks used by bootstrap and Android capability gates.
__all__ = [
    "Scheduler",
    "PipelineQueue",
    "NodeInfo",
    "NodeState",
    "NodeRole",
    "_node_supports_forward_layers",
    "_bootstrap_api_port",
]

def _sample_pipeline_token_id(logits, temperature: float, top_p: float) -> int:
    """Sample one token with the same zero-temperature semantics as local inference."""
    torch_module = require_torch()
    if logits is None or logits.ndim != 3 or logits.shape[0] != 1:
        shape = getattr(logits, "shape", None)
        raise ValueError(f"流水线 logits 形状无效: {shape}")

    # Local ModelManager uses do_sample=False when temperature <= 0. Keeping
    # that exact contract also avoids dividing fp16/bf16 logits by 1e-8,
    # which can create infinities and poison the CUDA context in multinomial.
    next_logits = logits[:, -1, :].float()
    if float(temperature) <= 0:
        return int(torch_module.argmax(next_logits, dim=-1).item())

    scaled_logits = next_logits / max(float(temperature), 1e-5)
    if not bool(torch_module.isfinite(scaled_logits).all().item()):
        raise RuntimeError("流水线 logits 包含 NaN/Inf，拒绝执行采样")
    probs = torch_module.softmax(scaled_logits, dim=-1)
    if not bool(torch_module.isfinite(probs).all().item()):
        raise RuntimeError("流水线采样概率包含 NaN/Inf")

    sorted_probs, sorted_indices = torch_module.sort(probs, descending=True, dim=-1)
    nucleus = min(1.0, max(0.0, float(top_p)))
    cumsum = torch_module.cumsum(sorted_probs, dim=-1)
    cutoff = cumsum > nucleus
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    filtered_probs = sorted_probs.masked_fill(cutoff, 0.0)
    probability_sum = filtered_probs.sum(dim=-1, keepdim=True)
    if (not bool(torch_module.isfinite(probability_sum).all().item())
            or bool((probability_sum <= 0).any().item())):
        raise RuntimeError("流水线采样概率无有效候选 token")
    filtered_probs = filtered_probs / probability_sum
    sampled_rank = torch_module.multinomial(filtered_probs, 1)
    return int(sorted_indices.gather(-1, sampled_rank)[0, 0].item())

def _bootstrap_api_port(default: int = 8000) -> int:
    """Return the master API port used for first-connect bootstrap."""
    for name in ("QLH_BOOTSTRAP_API_PORT", "QLH_MASTER_API_PORT"):
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            logger.warning("%s=%r 不是有效端口，回退到 %s", name, raw, default)
            continue
        if 1 <= port <= 65535:
            return port
        logger.warning("%s=%r 超出端口范围，回退到 %s", name, raw, default)
    return default

def _is_auth_register_failure(reason: str) -> bool:
    text = reason or ""
    return any(marker in text for marker in (
        "认证失败", "HMAC", "签名不匹配", "集群密钥"
    ))

def _configured_node_id() -> str:
    try:
        from node_runtime import node_runtime
        return node_runtime.get_node_id()
    except Exception:
        return NODE_ID

def _sync_runtime_node_config(node_id: str = None, node_role: str = None) -> None:
    """Keep runtime node identity aligned with runtime config mutations.

    阶段 0.3：运行时状态由 node_runtime 单例持有（config.py 不再被写回）；
    模块级 NODE_ID/NODE_ROLE 同步更新以保持本模块内既有读取行为不变。
    """
    global NODE_ID, NODE_ROLE
    try:
        from node_runtime import node_runtime
    except Exception:
        node_runtime = None

    if node_id:
        NODE_ID = str(node_id)
        if node_runtime is not None:
            node_runtime.set_node_id(NODE_ID)
    if node_role:
        NODE_ROLE = str(node_role)
        if node_runtime is not None:
            node_runtime.set_node_role(NODE_ROLE)

def _node_supports_forward_layers(node: "NodeInfo") -> bool:
    """节点能否承担**层前向传播**（即作为层流水线的一段）。

    ★ 2026-09-19：判据从「**平台**」改为「**能力**」。

    原实现（Phase 4.1，位于 `validate_layer_override`）按 `node_type == "android"` 一律拒绝，
    理由写作「Android 无 PyTorch 推理能力」。该判据**过宽**：**Android 不能跑 PyTorch，
    但能跑 llama.cpp/GGUF 引擎**，而 llama.cpp 现在**也能做层前向**
    （`LlamaCppEngine.forward_layers_from_hidden()` 当下游、`forward_layers_to_hidden()` 当上游，
    且 `BackendId.LLAMA_CPP` 已声明 `Capability.FORWARD_LAYERS`）⇒ 有 GGUF 引擎的 Android
    可以参与层流水线。

    判定顺序（**保守、向后兼容**）：

    1. `device_info["capabilities"]` 明确包含 `Capability.FORWARD_LAYERS` ⇒ **允许**
       （节点自报能力，最权威）；
    2. 否则若 `device_info["backend_id"]` 给出已知 backend ⇒ 按 `_BACKEND_CAPABILITIES` 判定；
    3. **两者都未提供** ⇒ 退回旧行为：**仅 `node_type == "pc"` 允许** ⇒
       **未声明能力的 Android 仍被拒绝**（不会因为本改动被无意放行）。
    """
    if isinstance(node, dict):
        node_type = node.get("node_type", "pc")
        info = node.get("device_info")
    else:
        node_type = getattr(node, "node_type", "pc")
        info = getattr(node, "device_info", None)
    if not isinstance(info, dict):
        info = {}

    reported = info.get("capabilities")
    if isinstance(reported, (list, tuple, set, frozenset)):
        if Capability.FORWARD_LAYERS in set(reported):
            return True
    elif isinstance(reported, str):
        if reported == Capability.FORWARD_LAYERS:
            return True

    backend_id = info.get("backend_id") or info.get("engine")
    if isinstance(backend_id, str) and backend_id:
        # 用公开 API，不触碰私有表。
        if backend_capabilities(backend_id).supports(Capability.FORWARD_LAYERS):
            return True

    # 兜底：保持旧语义（pc 可、android 不可），避免未上报能力的节点被无意放行。
    return node_type == "pc"

class PipelineQueue:
    """
    流水线请求队列 — MLFQ 三级反馈队列 + FIFO 兼容模式。

    调度策略:
    - "mlfq": Q0(交互,≤128tk) → Q1(普通,≤512tk) → Q2(批量,>512tk)
               同级 SJF 排序，老化提升防饥饿
    - "fifo": 级内先进先出（Q0→Q1→Q2 优先级，同级按入队顺序）。
              注意: 非严格全局 FIFO——短请求(Q0)始终优先于长请求(Q2)。

    特性:
    - 仅 1 个流水线任务执行中，后续请求自动排队
    - 调用方通过 task_id 轮询或阻塞等待结果
    - 已完成结果保留 TTL 秒后自动清理
    - 线程安全（RLock）

    用法:
        queue = PipelineQueue()
        queue.start(process_fn=scheduler.run_pipeline)
        task_id = queue.enqueue(prompt="hello", max_new_tokens=256)
        result = queue.wait_for_result(task_id, timeout=120)
    """

    def __init__(self, max_size: int = 100, result_ttl: float = 300.0,
                 strategy: str = "mlfq",
                 q0_max_tokens: int = 128, q1_max_tokens: int = 512,
                 aging_q1_to_q0: float = 60.0, aging_q2_to_q1: float = 120.0,
                 aging_max_wait: float = 300.0):

        # 三级队列（MLFQ）
        self._q0: collections.deque = collections.deque()  # Q0 交互级
        self._q1: collections.deque = collections.deque()  # Q1 普通级
        self._q2: collections.deque = collections.deque()  # Q2 批量级
        self._results: dict = {}           # task_id → {status, result, created_at, ...}
        self._events: dict = {}            # task_id → threading.Event
        self._cancel_events: dict = {}     # task_id → cooperative cancellation event
        self._lock = threading.RLock()  # 可重入锁：run_pipeline_safe 在持有锁时调用 enqueue
        self._current_task_id: Optional[str] = None
        self._running = False
        self._max_size = max_size
        self._result_ttl = result_ttl
        self._worker_thread: Optional[threading.Thread] = None
        self._process_fn: Optional[Callable] = None

        # ---- MLFQ 调度配置 ----
        self._strategy: str = strategy
        self._paused: bool = False         # 暂停接受新请求
        # 分级阈值
        self._q0_max_tokens: int = q0_max_tokens
        self._q1_max_tokens: int = q1_max_tokens
        # 老化参数
        self._aging_q1_to_q0: float = aging_q1_to_q0
        self._aging_q2_to_q1: float = aging_q2_to_q1
        self._aging_max_wait: float = aging_max_wait

        # ---- 抢占统计（二期实施，参数预留） ----
        self._preempt_count: int = 0
        self._preempt_total_overhead_ms: float = 0.0
        self._last_preempt_time: float = 0.0

        # ---- _queue 兼容属性（FIFO 回退时合并视图） ----
        # 保留为 property，不再作为独立存储

    # ---- 队列兼容属性（FIFO 回退时合并三级队列视图） ----

    @property
    def _queue(self):
        """
        兼容属性：合并三级队列为单一 deque 只读视图。

        ⚠️ 警告:
        - 每次访问创建新 deque 副本，返回后对副本的修改不会反映到内部队列。
        - 未加锁，并发修改期间读取可能快照不一致。
        - 仅用于向后兼容的只读遍历。请使用 enqueue() / _get_next_task() 进行修改。
        """
        result = collections.deque()
        result.extend(self._q0)
        result.extend(self._q1)
        result.extend(self._q2)
        return result

    def start(self, process_fn: Callable) -> None:
        """启动后台工作线程，开始处理队列中的任务。"""
        if self._running:
            logger.debug("流水线请求队列已在运行，忽略重复 start")
            return
        self._process_fn = process_fn
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_loop, name="pipeline-queue", daemon=True
        )
        self._worker_thread.start()
        logger.info(
            "流水线请求队列已启动 (strategy=%s, max_size=%d, worker=%s)",
            self._strategy.upper(), self._max_size, self._worker_thread.name
        )

    def stop(self) -> None:
        """停止工作线程，清理等待中的任务。"""
        self._running = False
        cancelled = 0
        # 唤醒所有等待者
        with self._lock:
            queue_depth = len(self._q0) + len(self._q1) + len(self._q2)
            current_task = self._current_task_id
            for task_id, event in self._events.items():
                if not event.is_set():
                    self._results[task_id] = {
                        "status": "cancelled", "error": "队列已停止"
                    }
                    event.set()
                    cancelled += 1
        logger.info(
            f"流水线请求队列已停止，唤醒等待任务 {cancelled} 个 "
            f"(queue_depth={queue_depth}, current_task={current_task})"
        )

    def enqueue(self, task_id: str = None, **task_data) -> str:
        """
        将推理请求加入队列（MLFQ 自动分级）。

        Args:
            task_id: 任务标识（None 则自动生成）
            **task_data: 传递给 process_fn 的关键字参数
                        必须包含 prompt, max_new_tokens

        Returns:
            task_id 字符串

        Raises:
            RuntimeError: 队列已满或已暂停
        """
        import uuid

        if task_id is None:
            task_id = f"q_{uuid.uuid4().hex[:12]}"

        with self._lock:
            # 暂停检查
            if self._paused:
                raise RuntimeError("请求队列已暂停，暂不接受新请求")

            # 容量检查
            total_size = len(self._q0) + len(self._q1) + len(self._q2)
            if total_size >= self._max_size:
                logger.warning(
                    f"⚠️ 请求队列已满 ({self._max_size}/{self._max_size})，"
                    f"拒绝新请求"
                )
                raise RuntimeError(
                    f"请求队列已满 ({self._max_size} 上限)，请稍后重试"
                )

            # 构建 QueueTask
            max_tokens = task_data.pop("max_new_tokens", 512)
            request_id = task_data.pop("request_id", None)   # L5: API request_id 链路追踪
            # 仅供调用方等待使用，不能透传给 run_pipeline()。
            task_data.pop("_queue_timeout", None)
            priority_level = self._classify(max_tokens)
            cancel_event = task_data.pop("_cancel_event", None)
            if cancel_event is None:
                cancel_event = threading.Event()
            task = QueueTask(
                task_id=task_id,
                prompt=task_data.pop("prompt", ""),
                max_new_tokens=max_tokens,
                temperature=task_data.pop("temperature", 0.7),
                top_p=task_data.pop("top_p", 0.9),
                session_id=task_data.pop("session_id", None),
                request_id=request_id,
                priority_level=priority_level,
                original_level=priority_level,
                _extra_kwargs=task_data,  # 保留其余 kwargs（如 _stream_callback）
                cancel_event=cancel_event,
            )

            # 按级别入队
            self._get_queue(priority_level).append(task)
            self._events[task_id] = threading.Event()
            self._cancel_events[task_id] = task.cancel_event
            self._results[task_id] = {
                "status": "queued",
                "created_at": task.created_at,
            }

        logger.info(
            "event=task_enqueue task_id=%s request_id=%s priority_level=Q%d "
            "max_tokens=%d total_depth=%d",
            task_id, request_id or "-", priority_level, max_tokens, total_size + 1,
        )
        return task_id

    def wait_for_result(self, task_id: str, timeout: float = 120.0,
                        cancel_event: threading.Event = None) -> dict:
        """
        阻塞等待任务完成。

        Args:
            task_id: 任务标识
            timeout: 超时秒数

        Returns:
            {status: "done"|"error"|"timeout"|"cancelled"|"unknown",
             result?: dict, error?: str}
        """
        event = self._events.get(task_id)
        if event is None:
            return {"status": "unknown", "error": f"未知任务: {task_id}"}

        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return {"status": "timeout", "error": f"任务 {task_id} 超时 ({timeout}s)"}
            if event.wait(timeout=min(0.1, remaining)):
                with self._lock:
                    result = self._results.get(task_id, {"status": "unknown"})
                    return dict(result)
            if cancel_event is not None and cancel_event.is_set():
                self.cancel_task(task_id)
                with self._lock:
                    return dict(self._results.get(task_id, {
                        "status": "cancelled",
                    }))

    # ---- 内部分级与队列选择 ----

    def _classify(self, max_new_tokens: int) -> int:
        """
        根据 max_new_tokens 确定优先级级别。

        MLFQ 模式: ≤Q0_MAX→0, ≤Q1_MAX→1, >Q1_MAX→2
        FIFO 模式: 统一返回 1（所有任务进入同一队列，保持入队顺序）
        """
        if self._strategy == "fifo":
            return 1
        if max_new_tokens <= self._q0_max_tokens:
            return 0
        elif max_new_tokens <= self._q1_max_tokens:
            return 1
        else:
            return 2

    def _get_queue(self, level: int) -> collections.deque:
        """返回对应级别的队列。"""
        if level == 0:
            return self._q0
        elif level == 1:
            return self._q1
        else:
            return self._q2

    # ---- MLFQ 调度核心 ----

    def _get_next_task(self) -> Optional[QueueTask]:
        """
        从队列中选择下一个任务。

        - FIFO 模式：从 Q0 → Q1 → Q2 按入队顺序弹出
        - MLFQ 模式：调用 schedule_next()（含 aging + SJF）
        """
        if self._strategy == "fifo":
            # 统一 FIFO：按 Q0 → Q1 → Q2 顺序，每级内先进先出
            for q in [self._q0, self._q1, self._q2]:
                if q:
                    return q.popleft()
            return None
        else:
            return self._schedule_next()

    def _schedule_next(self) -> Optional[QueueTask]:
        """
        MLFQ 调度：从三级队列中选择下一个要执行的任务。

        ★ 必须在持有 self._lock 时调用（线程安全）。

        规则:
        1. 老化提升（饥饿保护）
        2. 每级队列内部 SJF 排序
        3. 严格优先级：Q0 → Q1 → Q2
        """
        self._apply_aging()
        self._apply_sjf_sorting()

        for q in [self._q0, self._q1, self._q2]:
            if q:
                return q.popleft()
        return None

    def _apply_aging(self, now: float = None) -> None:
        """
        老化提升：等待过久的请求逐级上浮（常规路径每次调用仅提升一级）。

        - Q2 → Q1: 等待超过 aging_q2_to_q1 秒
        - Q1 → Q0: 等待超过 aging_q1_to_q0 秒（不含刚从 Q2 提升的）
        - 绝对上限: 等待超过 aging_max_wait 秒 → 直接置顶 Q0
          （可越级提升，步骤 1 刚升入 Q1 的任务若同时超绝对上限也会被置顶 Q0）
        """
        now = now or time.time()
        just_promoted: set = set()  # 本轮已提升的 task_id，避免重复提升

        # Q2 → Q1
        aged_up = [t for t in list(self._q2)
                   if now - t.created_at > self._aging_q2_to_q1]
        for t in aged_up:
            self._q2.remove(t)
            t.priority_level = 1
            self._q1.append(t)
            just_promoted.add(t.task_id)
            logger.info(
                f"⬆️ 老化提升 Q2→Q1: {t.task_id} "
                f"(等待 {now - t.created_at:.0f}s)"
            )

        # Q1 → Q0（排除刚从 Q2 提升上来的，每次仅提升一级）
        aged_up = [t for t in list(self._q1)
                   if now - t.created_at > self._aging_q1_to_q0
                   and t.task_id not in just_promoted]
        for t in aged_up:
            self._q1.remove(t)
            t.priority_level = 0
            # 超过绝对上限 → 置顶 Q0；否则追加到队尾
            if now - t.created_at > self._aging_max_wait:
                self._q0.appendleft(t)
                logger.warning(
                    f"🔴 绝对上限老化 Q1→Q0(置顶): {t.task_id} "
                    f"(等待 {now - t.created_at:.0f}s > {self._aging_max_wait}s)"
                )
            else:
                self._q0.append(t)
                logger.info(
                    f"⬆️ 老化提升 Q1→Q0: {t.task_id} "
                    f"(等待 {now - t.created_at:.0f}s)"
                )

        # 绝对上限 → 强制置顶 Q0（对所有队列中等待超过上限的任务）
        for q in [self._q1, self._q2]:
            aged_max = [t for t in list(q)
                        if now - t.created_at > self._aging_max_wait]
            for t in aged_max:
                q.remove(t)
                t.priority_level = 0
                self._q0.appendleft(t)  # 放到 Q0 队首
                logger.warning(
                    f"🔴 绝对上限老化: {t.task_id} 强制置顶 Q0 "
                    f"(等待 {now - t.created_at:.0f}s > {self._aging_max_wait}s)"
                )

    def _apply_sjf_sorting(self) -> None:
        """每级队列内部按预估剩余时间升序排列（SJF）。"""
        for q in [self._q0, self._q1, self._q2]:
            if len(q) <= 1:
                continue
            q_sorted = sorted(q, key=lambda t: t.estimated_duration_seconds())
            q.clear()
            q.extend(q_sorted)

    # ---- 策略控制 ----

    def set_strategy(self, strategy: str) -> None:
        """切换调度策略: "fifo" | "mlfq"。"""
        if strategy not in ("fifo", "mlfq"):
            raise ValueError(f"无效调度策略: {strategy}，仅支持 fifo/mlfq")
        with self._lock:
            self._strategy = strategy
        logger.info(f"调度策略已切换: {strategy.upper()}")

    def pause(self) -> None:
        """暂停接受新请求（已在队列中的任务继续执行）。"""
        with self._lock:
            was_paused = self._paused
            self._paused = True
            queue_depth = len(self._q0) + len(self._q1) + len(self._q2)
            current_task = self._current_task_id
        logger.info(
            f"⏸️ 请求队列已暂停 (was_paused={was_paused}, "
            f"queue_depth={queue_depth}, current_task={current_task})"
        )

    def resume(self) -> None:
        """恢复接受新请求。"""
        with self._lock:
            was_paused = self._paused
            self._paused = False
            queue_depth = len(self._q0) + len(self._q1) + len(self._q2)
            current_task = self._current_task_id
        logger.info(
            f"▶️ 请求队列已恢复 (was_paused={was_paused}, "
            f"queue_depth={queue_depth}, current_task={current_task})"
        )

    def cancel_task(self, task_id: str) -> bool:
        """
        取消指定任务。

        排队中的任务: 从队列移除，标记 cancelled。
        执行中的任务: 设置协作取消信号，流水线在当前 step 完成后广播 ABORT。
        已完成的任务: 返回 False。

        Returns: True 表示已取消，False 表示无法取消。

        Complexity: O(n²) 线性扫描 + deque.remove，Q_MAX_SIZE=100 时可接受。
        """
        with self._lock:
            # 搜索三级队列
            for q in (self._q0, self._q1, self._q2):
                for task in q:
                    if task.task_id == task_id:
                        q.remove(task)
                        self._results[task_id] = {
                            "status": "cancelled",
                            "created_at": self._results.get(task_id, {}).get("created_at", 0),
                            "completed_at": time.time(),
                        }
                        event = self._events.get(task_id)
                        if event:
                            event.set()
                        logger.info(f"🚫 排队任务已取消: {task_id}")
                        return True
            # 执行中的任务通过协作信号在当前 step 完成后中止。
            if self._current_task_id == task_id:
                cancel_event = self._cancel_events.get(task_id)
                if cancel_event is None:
                    return False
                cancel_event.set()
                self._results[task_id] = {
                    "status": "cancelled",
                    "created_at": self._results.get(task_id, {}).get("created_at", 0),
                    "completed_at": time.time(),
                }
                event = self._events.get(task_id)
                if event:
                    event.set()
                logger.info(f"🚫 执行中任务已请求取消: {task_id}")
                return True
        return False

    def clear(self) -> int:
        """
        清空所有排队任务。

        执行中的任务不受影响。
        Returns: 已取消的任务数量。
        """
        count = 0
        with self._lock:
            for q in (self._q0, self._q1, self._q2):
                while q:
                    task = q.popleft()
                    self._results[task.task_id] = {
                        "status": "cancelled",
                        "created_at": self._results.get(task.task_id, {}).get("created_at", 0),
                        "completed_at": time.time(),
                    }
                    event = self._events.get(task.task_id)
                    if event:
                        event.set()
                    count += 1
        logger.info(f"🧹 已清空 {count} 个排队任务")
        return count

    def get_queue_detail(self) -> dict:
        """返回三级队列详情（供 API 和前端使用）。"""
        with self._lock:
            return {
                "running": self._running,
                "strategy": self._strategy,
                "paused": self._paused,
                "current_task": self._current_task_id,
                "queue_size": len(self._q0) + len(self._q1) + len(self._q2),
                "q0_depth": len(self._q0),
                "q1_depth": len(self._q1),
                "q2_depth": len(self._q2),
                "q0": [t.to_dict() for t in list(self._q0)],
                "q1": [t.to_dict() for t in list(self._q1)],
                "q2": [t.to_dict() for t in list(self._q2)],
                "aging_params": {
                    "q0_max_tokens": self._q0_max_tokens,
                    "q1_max_tokens": self._q1_max_tokens,
                    "q1_to_q0_s": self._aging_q1_to_q0,
                    "q2_to_q1_s": self._aging_q2_to_q1,
                    "max_wait_s": self._aging_max_wait,
                },
                "preempt_stats": {
                    "count": self._preempt_count,
                    "last_time": self._last_preempt_time,
                    "total_overhead_ms": round(self._preempt_total_overhead_ms, 1),
                },
                "completed_count": sum(
                    1 for r in self._results.values()
                    if r.get("status") in ("done", "error", "cancelled")
                ),
                "max_size": self._max_size,
            }

    # ---- 内部方法 ----

    def _process_loop(self) -> None:
        """
        后台工作循环：从队列取任务 → 调用 process_fn → 存储结果。

        ★ 并发安全：pop 前检查 is_busy，防止与 run_pipeline_safe
          的立即执行路径并发调用 run_pipeline（GPU OOM）。

        ★ MLFQ: 使用 _get_next_task() 代替直接 pop，支持多级调度。
        """
        logger.info("流水线队列工作线程已启动")
        while self._running:
            task = None
            with self._lock:
                if not self.is_busy:
                    task = self._get_next_task()
                    if task is not None:
                        # ★ 原子化：pop + 标记 busy 在同一锁内完成，
                        # 消除与 run_pipeline_safe 立即执行路径的 TOCTOU 竞态窗口
                        self._current_task_id = task.task_id

            if task is None:
                time.sleep(0.1)
                continue

            # task 是 QueueTask 对象
            task_id = task.task_id
            task_data = task.to_task_data()
            task_data["_cancel_event"] = task.cancel_event

            with self._lock:
                self._results[task_id]["status"] = "running"
                self._results[task_id]["started_at"] = time.time()

            logger.info(
                "event=task_dispatch task_id=%s request_id=%s "
                "Q%d orig=Q%d wait=%.0fs",
                task_id, task.request_id or "-",
                task.priority_level, task.original_level, task.wait_seconds(),
            )
            t_start = time.time()

            try:
                if task.cancel_event.is_set():
                    raise RuntimeError("任务已取消")
                result = self._process_fn(**task_data)
                elapsed = time.time() - t_start
                with self._lock:
                    if not task.cancel_event.is_set():
                        self._results[task_id] = {
                            "status": "done",
                            "result": result,
                            "created_at": self._results.get(task_id, {}).get("created_at", 0),
                            "started_at": self._results.get(task_id, {}).get("started_at", 0),
                            "completed_at": time.time(),
                            "elapsed_s": round(elapsed, 2),
                        }
                logger.info(
                    "event=task_complete task_id=%s request_id=%s elapsed=%.1fs",
                    task_id, task.request_id or "-", elapsed,
                )
            except Exception as e:
                elapsed = time.time() - t_start
                with self._lock:
                    if not task.cancel_event.is_set():
                        self._results[task_id] = {
                            "status": "error",
                            "error": str(e),
                            "created_at": self._results.get(task_id, {}).get("created_at", 0),
                            "completed_at": time.time(),
                            "elapsed_s": round(elapsed, 2),
                        }
                logger.error(
                    "event=task_failed task_id=%s request_id=%s error=%s",
                    task_id, task.request_id or "-", str(e)[:200],
                    exc_info=True,
                )
            finally:
                event = self._events.get(task_id)
                if event:
                    event.set()
                with self._lock:
                    self._current_task_id = None
                    # 清理过期结果
                    self._cleanup_expired()

        logger.info("流水线队列工作线程已退出")

    def _cleanup_expired(self) -> None:
        """清理超过 TTL 的已完成结果。"""
        now = time.time()
        expired = [
            tid for tid, r in self._results.items()
            if r.get("status") in ("done", "error", "cancelled")
            and now - r.get("completed_at", 0) > self._result_ttl
        ]
        for tid in expired:
            del self._results[tid]
            self._events.pop(tid, None)
            self._cancel_events.pop(tid, None)
        if expired:
            logger.debug(f"清理 {len(expired)} 个过期结果")

    # ---- 状态查询 ----

    @property
    def is_busy(self) -> bool:
        """当前是否有任务在执行中（线程安全）。"""
        with self._lock:
            return self._current_task_id is not None

    @property
    def queue_size(self) -> int:
        """当前队列总长度（不含正在执行的任务）。"""
        with self._lock:
            return len(self._q0) + len(self._q1) + len(self._q2)

    def get_status(self) -> dict:
        """获取队列整体状态。"""
        with self._lock:
            return {
                "running": self._running,
                "strategy": self._strategy,
                "current_task": self._current_task_id,
                "queue_size": len(self._q0) + len(self._q1) + len(self._q2),
                "q0_depth": len(self._q0),
                "q1_depth": len(self._q1),
                "q2_depth": len(self._q2),
                "completed_count": sum(
                    1 for r in self._results.values()
                    if r.get("status") in ("done", "error", "cancelled")
                ),
                "max_size": self._max_size,
            }

class Scheduler(
    SchedulerSidecarMixin, SchedulerTaskWorkerMixin,
    SchedulerClusterMixin, SchedulerPipelineMixin,
):
    """
    主节点调度器

    负责:
    - 管理所有从节点状态
    - 接收前端推理请求，分发给流水线
    - 监控任务执行，处理异常
    - 控制流水线启停
    - 集成 TCP 服务端，接收从节点注册
    - 流水线请求队列（PipelineQueue）
    """

    def _scheduler_facade_global(self, name: str):
        return globals()[name]

    def _qwen3_multisidecar_factory(self):
        return Qwen3PipelineMultiSidecar

    def __init__(
        self,
        host: Optional["InferenceHost"] = None,
        callbacks: Optional["SchedulerCallbacks"] = None,
    ):
        # 推理宿主默认使用全局单例；回调由 API composition root 以具名
        # Protocol bundle 注入，避免 scheduler 反向依赖 api_server。
        self._host = host if host is not None else get_model_host()
        self._control_fence: ControlFence | None = None
        self._auto_role_controller: AutoRoleController | None = None
        self._handoff_coordinator: HandoffCoordinator | None = None
        self._callbacks = callbacks if callbacks is not None else getattr(
            self._host, "scheduler_callbacks", None,
        )
        self.nodes: dict[str, NodeInfo] = {}
        self._current_task: Optional[InferenceTask] = None
        self._infer_tasks: dict[str, InferenceTask] = {}
        self._task_lock = threading.Lock()
        self._running = False
        # 启动期后台发现线程必须能被 stop() 立即唤醒，避免停止后仍发起连接。
        self._startup_cancel_event = threading.Event()
        self._network_identity_thread: Optional[threading.Thread] = None
        self._network_identity_lock = threading.Lock()
        self.on_task_complete: Optional[Callable] = None

        # TCP 服务端（分布式模式下启动）。回调线程通过
        # _tcp_callback_context 绑定触发事件的具体 server；普通控制面
        # 线程则读取 _tcp_server_default。
        self._tcp_server_default = None  # 延迟导入，避免循环依赖
        self._tcp_callback_context = threading.local()
        # TCP 客户端（从节点连接主节点后创建）。必须在此默认初始化，
        # 否则从未连接过主节点的实例在错误路径直接访问
        # self._tcp_client 会抛 AttributeError 而不是优雅返回 False
        self._tcp_client = None

        # 从节点：等待主节点推理结果
        self._client_pending_results: dict = {}
        self._client_pending_events: dict[str, threading.Event] = {}
        self._client_pending_lock = threading.Lock()

        # 主节点：转发请求取消和并发准入。队列仍负责实际推理串行化，
        # 信号量只限制等待队列结果的包装线程数量。
        self._forward_cancel_events: dict[tuple[str, str], threading.Event] = {}
        self._forward_cancel_lock = threading.Lock()
        self._forward_infer_slots = threading.BoundedSemaphore(
            max(1, min(PIPELINE_QUEUE_MAX_SIZE + 1, 32))
        )

        # 流水线推理状态（主节点侧）
        self._pipeline_results: dict = {}       # key → result data
        self._pipeline_events: dict = {}        # key → threading.Event
        self._pipeline_active_tasks: set[str] = set()
        self._pipeline_task_contracts: dict[str, dict] = {}
        self._chain_ack_state: dict = {}        # task_id → step → node_id → ack/error
        self._pipeline_lock = threading.Lock()
        self._nodes_lock = threading.RLock()    # Phase 2.1: 保护 self.nodes 并发读写（可重入）
        self._kv_cache_lock = threading.Lock()   # Phase 2.2: 保护 _kv_cache 并发读写
        self._kv_cache: dict = {}               # task_id → past_key_values（本节点层范围的 KV cache）
        # 节点只有完成模型层加载并返回当前 config_id 的 ACK 后才进入该集合。
        # 保留旧字段名，避免状态接口和测试夹具发生无关改动。
        self._layer_config_pushed: set = set()
        self._layer_config_expected: dict[str, dict] = {}
        self._layer_config_acks: dict[str, dict] = {}
        self._layer_config_retry_state: dict[str, dict] = {}
        self._layer_config_lock = threading.Lock()
        self._layer_config_push_lock = threading.Lock()
        self._layer_execution_lock = threading.RLock()
        self._active_pipeline_task_ids: set[str] = set()
        self._pending_layer_config: Optional[tuple[str, dict]] = None
        self._layer_config_inflight: set[str] = set()
        self._last_layer_config_ack_payload: Optional[dict] = None
        self._active_layer_config: Optional[dict] = None
        self._layer_config_generation = 0
        self._layer_config_receive_sequence = 0
        self._latest_layer_config_receive_sequence = 0
        self._latest_layer_config_generation = 0
        self._pipeline_worker_reserved = False
        self._pipeline_worker_opted_out = False
        self._pipeline_worker_opt_out: set[str] = set()
        self._pipeline_load_transaction: Optional[dict] = None
        # Qwen3 remains outside production runtime admission.  This isolated
        # state machine exercises the C2 lifecycle without network dispatch,
        # weight materialization, or full-model fallback.
        self._qwen3_pipeline_dry_run: Optional[Qwen3PipelineDryRunTransaction] = None
        self._qwen3_loopback_base_url = ""
        self._qwen3_loopback_workers: dict[str, Qwen3PipelineLoopbackWorker] = {}
        self._qwen3_loopback_ack_nonces: dict[str, str] = {}
        self._qwen3_local_chain: Optional[Qwen3PipelineMultiSidecar] = None
        self._qwen3_local_contract: Optional[dict] = None
        self._qwen3_local_parity: dict = {}
        self._qwen3_local_chain_lock = threading.RLock()
        self._qwen3_local_artifact_root_override: Optional[str] = None
        self._qwen3_network_handoff_transport = None
        self._qwen3_network_transfer_coordinator = None
        self._qwen3_artifact_transfer_runtime = None
        self._qwen3_peer_request_verifier = None
        # Gemma 4 Unified uses a distinct Transformers 5.17.0 sidecar.  This
        # local chain is an explicit development route and never changes the
        # production pipeline runtime allow-list.
        self._gemma4_local_chain: Optional[Gemma4PipelineMultiSidecar] = None
        self._gemma4_local_contract: Optional[dict] = None
        self._gemma4_local_chain_lock = threading.RLock()
        self._gemma4_local_artifact_root_override: Optional[str] = None
        self._gemma4_assignment_paths: dict[str, str] = {}
        self._gemma4_sidecar_python_override: Optional[str] = None
        self._model_runtime_contract_lock = threading.RLock()
        self._active_pipeline_capacity_plan: Optional[dict] = None
        # Recovery is control-plane state. A candidate is never published to
        # an executor before its capacity, artifact, and epoch gates pass.
        self._pipeline_reshard_coordinator: Optional[PipelineReshardCoordinator] = None
        self._pipeline_reshard_last_decision: Optional[dict] = None
        self._prepared_layer_configs: dict[str, dict] = {}
        # 同时到达的分布式请求都可能要求权威同步，必须用计数而非
        # 布尔值，避免前一个请求结束时把后一个请求降级为普通推送。
        self._authoritative_layer_sync_requests = 0
        self._local_pipeline_cancelled: set[str] = set()
        self._local_pipeline_cancelled_order: collections.deque = collections.deque()
        self._local_pipeline_steps: dict[str, int] = {}
        self._chain_clients: dict[str, object] = {}
        self._chain_clients_lock = threading.Lock()
        # Optional Transport v2 runtime factory.  It is unset by default so
        # all existing TCP clients keep the Legacy wire path unchanged.
        self._transport_runtime_factory: Optional[Callable[[str], object]] = None
        self._pipeline_accounted_tasks: set = set()  # 主节点侧：已完成记账的流水线任务
        self._pipeline_accounted_order: collections.deque = collections.deque()
        self._local_pipeline_counted_tasks: set = set()  # 从节点侧：已本地计数的流水线任务
        self._local_pipeline_error_tasks: set = set()    # 从节点侧：已本地计错的流水线任务
        self._local_pipeline_accounted_order: collections.deque = collections.deque()
        self._inference_lock = threading.Lock()  # GPU 推理互斥锁（防止并发执行）

        # ---- L5: 多节点日志聚合状态 ----
        self._pending_log_responses: dict = {}    # node_id → response data
        self._pending_log_events: dict = {}       # node_id → threading.Event
        self._pending_log_lock = threading.Lock()

        # ---- 协同抢占状态 (Phase 2) ----
        self._preempted_task: Optional[PreemptState] = None
        self._preempt_count: int = 0
        self._preempt_total_overhead_ms: float = 0.0
        self._preempt_last_time: float = 0.0
        self._preempt_disabled: bool = False   # 超过 MAX_OVERHEAD_MS 后自动禁用
        self._preempting: bool = False          # 正在执行抢占（防嵌套）
        self._pipeline_context = threading.local()

        # 最大节点数（可动态调整）
        self._max_nodes: int = MAX_NODES
        self._master_connect_lock = threading.Lock()
        self._role_transition_lock = threading.Lock()
        self._client_health_thread: Optional[threading.Thread] = None
        self._client_health_start_lock = threading.Lock()
        self._layer_config_retry_thread: Optional[threading.Thread] = None
        self._distributed_inference_enabled: Optional[bool] = None
        self._local_device_profile: Optional[dict] = {}
        self._runtime_layer_override: Optional[list] = None
        self._ha_state_lock = threading.RLock()
        self._ha_state_loaded = False
        self._ha_state: dict = {
            "spare_master": None,
            "spare_master_active": False,
            "pending_new_master_id": "",
            "transfer_logs": [],
            "spare_master_logs": [],
        }
        self._task_worker_control = TaskWorkerControlPlane(
            health_timeout_seconds=max(30.0, HEARTBEAT_INTERVAL * 4.0),
        )
        self._task_worker_refresh_lock = threading.Lock()
        self._task_worker_refresh_requested = False
        self._task_worker_refresh_generation = 0
        self._remote_task_worker_providers: dict[
            str, RemoteFullWorkerProvider
        ] = {}
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_active_attempts: dict[
            str, _TaskWorkerActiveAttempt
        ] = {}
        self._task_worker_seen_messages: dict[
            str, tuple[str, list[dict]]
        ] = {}
        self._task_worker_seen_order: collections.deque[str] = (
            collections.deque()
        )

        # 流水线请求队列（MLFQ 三级反馈队列，兼容 FIFO）
        self.pipeline_queue = PipelineQueue(
            max_size=PIPELINE_QUEUE_MAX_SIZE,
            result_ttl=PIPELINE_QUEUE_RESULT_TTL,
            strategy=PIPELINE_SCHEDULING_STRATEGY,
            q0_max_tokens=PIPELINE_Q0_MAX_TOKENS,
            q1_max_tokens=PIPELINE_Q1_MAX_TOKENS,
            aging_q1_to_q0=PIPELINE_AGING_Q1_TO_Q0_SECONDS,
            aging_q2_to_q1=PIPELINE_AGING_Q2_TO_Q1_SECONDS,
            aging_max_wait=PIPELINE_AGING_MAX_WAIT_SECONDS,
        )

    def set_transport_runtime_factory(self, factory: Optional[Callable[[str], object]]) -> None:
        """Inject a per-peer Transport v2 bridge for local/production rollout."""

        if factory is not None and not callable(factory):
            raise TypeError("transport runtime factory must be callable or None")
        self._transport_runtime_factory = factory

    def _new_transport_runtime(self, node_id: str) -> object | None:
        factory = self._transport_runtime_factory
        if not callable(factory):
            return None
        try:
            return factory(str(node_id))
        except Exception:
            logger.warning("transport runtime factory failed for %s", node_id, exc_info=True)
            return None

    def _transport_runtime_kwargs(self, node_id: str) -> dict[str, object]:
        """Return an opt-in constructor kwarg without breaking test doubles."""

        runtime = self._new_transport_runtime(node_id)
        return {"transport_runtime": runtime} if runtime is not None else {}

    # ================================================================
    # 启动 / 停止
    # ================================================================

    def start(self, host: str = None, port: int = None) -> None:
        """
        启动调度器。

        初始化节点状态；若为分布式模式，启动 TCP 服务端监听。

        主节点启动后自动:
        - 检测当前可广告地址（已登录时优先 Tailscale，否则可达 LAN）
        - 将物理 MAC 身份保存到用户自持的主节点 SQLite
        - 通过本机 bootstrap 配置和 Tailnet 提供发现，不依赖远端数据库

        Args:
            host: TCP 监听地址（默认 0.0.0.0，接受所有接口连接）
            port: TCP 监听端口（默认 config.SERVER_PORT）
        """
        self._startup_cancel_event.clear()
        self.init_nodes()
        self._running = True

        # Reconcile only the active model's assignment cache.  This is a
        # local, bounded cleanup and never touches the user's full model tree.
        try:
            active_model_id = str(
                getattr(self._host, "active_model_id", "")
                or getattr(self._host, "_active_model_id", "")
                or ""
            )
            if active_model_id:
                from model_sync import reconcile_pipeline_assignment_cache

                reconcile_pipeline_assignment_cache(active_model_id)
        except Exception:
            logger.warning("pipeline assignment cache reconcile failed", exc_info=True)

        # 存储检测到的局域网 IP 和 MAC 地址
        self._lan_ip: str = ""
        self._mac_addresses: list[str] = []
        self._master_identity_verified: bool = False
        self._master_identity_reason: str = ""

        if RUN_MODE == "distributed":
            from transport_port import create_server

            # 绑定到 0.0.0.0 接受所有接口连接（而非占位符 192.168.x.x）
            bind_host = host or "0.0.0.0"
            actual_port = SERVER_PORT if port is None else port

            try:
                server = create_server(bind_host, actual_port)
                self._tcp_server = server
                if self._control_fence is not None:
                    server.set_control_fence(self._control_fence)
                server.start(
                    on_message=self._bind_tcp_server_callback(
                        server, self._on_tcp_message,
                    ),
                    on_disconnect=self._bind_tcp_server_callback(
                        server, self._on_tcp_disconnect,
                    ),
                    on_registration_confirmed=self._bind_tcp_server_callback(
                        server, self._on_tcp_registration_confirmed,
                    ),
                )
            except Exception as e:
                self._tcp_server = None
                logger.error(
                    "分布式 TCP 监听启动失败 (%s:%s): %s；"
                    "继续提供本地主节点全模型推理",
                    bind_host, actual_port, e, exc_info=True,
                )

            # 检测实际局域网 IP 和 MAC 地址
            if NODE_ROLE == "master":
                # Network address and MAC identity are populated by the
                # post-startup worker below.
                logger.info(f"调度器已启动（分布式模式），监听 {bind_host}:{actual_port}，局域网 IP: {self._lan_ip}，MAC: {self._mac_addresses}")

                # 验证主节点身份（MAC 匹配）
                self._master_identity_reason = "startup_deferred"

                # ★ MAC 不匹配时的处理策略
                if self._master_identity_reason == "mac_mismatch":
                    # 尝试在本机 bootstrap 配置或 Tailnet 中发现已确认的主节点。
                    discovery = self.discover_master()
                    if discovery.get("found"):
                        # bootstrap/Tailnet 发现已确认主节点 → 自动切换为从节点
                        stale_note = "（心跳过期，IP 可能已变更）" if discovery.get("stale") else ""
                        logger.warning(
                            f"⛔ 主节点身份验证失败！本机 MAC 与本地 SQLite 记录不匹配。\n"
                            f"   已发现主节点 ({discovery['master_host']}:{discovery['master_port']}){stale_note}，\n"
                            f"   自动切换为从节点模式并尝试连接..."
                        )
                        # 启动后台线程处理切换（避免阻塞 start()）
                        threading.Thread(
                            target=self._auto_switch_to_client,
                            args=(discovery["master_host"], discovery["master_port"]),
                            name="auto-switch-client",
                            daemon=True,
                        ).start()
                    else:
                        # 未发现其他主节点：保留本机服务，但不覆盖已有身份记录。
                        logger.error(
                            f"⛔ 主节点身份验证失败 — 拒绝覆盖本机 SQLite 身份记录！\n"
                            f"   本机 MAC 与已记录身份不匹配，且未发现其他主节点。\n"
                            f"   如需更换主节点机器，请先在原主节点的后台管理中"
                            f"使用「重置主节点身份」功能。\n"
                            f"   当前将以单机模式运行（不覆盖本机 SQLite 身份）。"
                        )
                else:
                    if self._tcp_server and self._tcp_server._running:
                        port_text = self._tcp_server.port
                        if "master" in self.nodes:
                            self.nodes["master"].address = f"{self._lan_ip}:{port_text}"
                        logger.info(
                            "主节点本地身份已就绪，广告地址 %s:%s；"
                            "从节点通过 bootstrap 配置和 Tailnet 发现",
                            self._lan_ip,
                            port_text,
                        )
                    else:
                        logger.warning(
                            "主节点 TCP 未监听，跳过主节点地址广告；"
                            "本地推理仍可用"
                        )
                # 检查是否需要向备用主节点发送接管通知（转让后新主节点启动）
                threading.Thread(
                    target=self.deactivate_spare_master_on_startup,
                    name="spare-deactivate",
                    daemon=True,
                ).start()
            else:
                logger.info(f"调度器已启动（分布式模式，从节点），监听 {bind_host}:{actual_port}")
                # 从节点启动后台线程：监控主节点健康状态 + 自动重连
                self._start_client_health_monitor()
                # 启动后尝试自动发现并连接主节点
                threading.Thread(
                    target=self._auto_connect_on_startup,
                    name="auto-connect-startup",
                    daemon=True,
                ).start()
        else:
            logger.info("调度器已启动（单机模式）")

        if RUN_MODE == "distributed" and NODE_ROLE == "master":
            self._start_deferred_network_identity()

        # 启动流水线请求队列（仅主节点，FIFO 串行）
        if self._effective_role() == "master":
            self.pipeline_queue.start(process_fn=self._process_queued_pipeline_task)
            logger.info("流水线请求队列已就绪")

        # 已经是 client 的节点由上面的 auto-connect 唯一路径处理；这里只处理
        # 尚未确认身份、可能需要从 provisional master 切换的节点。
        if (self._effective_role() == "master"
                and self._master_identity_reason != "startup_deferred"
                and self.can_join_existing_master()):
            threading.Thread(
                target=self._auto_join_tailnet_master_on_startup,
                name="tailnet-master-discovery",
                daemon=True,
            ).start()

    def _start_deferred_network_identity(self) -> None:
        """Schedule network address, identity, and discovery work after start."""
        with self._network_identity_lock:
            thread = self._network_identity_thread
            if thread is not None and thread.is_alive():
                return
            self._network_identity_thread = threading.Thread(
                target=self._initialize_network_identity,
                name="network-identity-startup",
                daemon=True,
            )
            self._network_identity_thread.start()

    def _initialize_network_identity(self) -> None:
        """Complete distributed identity without delaying local service start."""
        if self._startup_cancel_event.is_set() or not self._running:
            return
        try:
            from transport_port import detect_lan_ip, get_mac_addresses

            self._lan_ip = detect_lan_ip()
            self._mac_addresses = get_mac_addresses()
            if self._startup_cancel_event.is_set() or not self._running:
                return

            if self._effective_role() != "master":
                return
            self._verify_master_identity()
            if self._master_identity_reason == "mac_mismatch":
                discovery = self.discover_master()
                if discovery.get("found"):
                    logger.warning(
                        "master identity mismatch; switching to discovered master "
                        "%s:%s",
                        discovery["master_host"], discovery["master_port"],
                    )
                    threading.Thread(
                        target=self._auto_switch_to_client,
                        args=(discovery["master_host"], discovery["master_port"]),
                        name="auto-switch-client",
                        daemon=True,
                    ).start()
                else:
                    logger.error(
                        "master identity mismatch; no confirmed master discovered",
                    )
            elif self._tcp_server and self._tcp_server._running:
                if "master" in self.nodes:
                    self.nodes["master"].address = (
                        f"{self._lan_ip}:{self._tcp_server.port}"
                    )

            if (self._master_identity_reason != "mac_mismatch"
                    and self._effective_role() == "master"
                    and self.can_join_existing_master()):
                threading.Thread(
                    target=self._auto_join_tailnet_master_on_startup,
                    name="tailnet-master-discovery",
                    daemon=True,
                ).start()
        except Exception as exc:
            self._master_identity_reason = "network_probe_failed"
            logger.warning("deferred network identity initialization failed: %s", exc)

    def stop(self) -> None:
        """停止调度器"""
        self._startup_cancel_event.set()
        self._running = False
        self.pipeline_queue.stop()
        tcp_client = getattr(self, "_tcp_client", None)
        if tcp_client is not None:
            tcp_client.on_disconnect = None
            try:
                tcp_client.disconnect()
            except Exception:
                logger.debug("停止主节点连接失败", exc_info=True)
        with self._chain_clients_lock:
            chain_clients = list(self._chain_clients.values())
            self._chain_clients.clear()
        for chain_client in chain_clients:
            try:
                chain_client.disconnect()
            except Exception:
                logger.debug("停止链式连接失败", exc_info=True)
        if self._tcp_server:
            self._tcp_server.stop()
        logger.info("调度器已停止")

    @property
    def _tcp_server(self):
        """Return the server bound to the current TCP callback thread.

        The default is used by ordinary scheduler work.  Receive callbacks
        use a thread-local binding supplied when each server is registered,
        so a second server cannot redirect an existing scheduler instance.
        """
        callback_server = getattr(self._tcp_callback_context, "server", None)
        return callback_server or self._tcp_server_default

    @_tcp_server.setter
    def _tcp_server(self, value):
        self._tcp_server_default = value

    def _bind_tcp_server_callback(self, server, callback):
        """Bind one control-plane event to its explicit TCPServer source."""
        def bound_callback(*args, **kwargs):
            previous = getattr(self._tcp_callback_context, "server", None)
            self._tcp_callback_context.server = server
            try:
                return callback(*args, **kwargs)
            finally:
                if previous is None:
                    try:
                        del self._tcp_callback_context.server
                    except AttributeError:
                        pass
                else:
                    self._tcp_callback_context.server = previous

        return bound_callback

    @property
    def inference_host(self) -> "InferenceHost":
        """Return the host selected for this scheduler instance."""
        return self._host

    @property
    def inference_callbacks(self) -> Optional["SchedulerCallbacks"]:
        """Return the explicit callback bundle selected for this scheduler."""
        return self._callbacks

    def configure_callbacks(self, callbacks: "SchedulerCallbacks") -> None:
        """Set one complete callback bundle; individual callback ordering is irrelevant."""
        if callbacks is None:
            raise TypeError("scheduler callbacks are required")
        self._callbacks = callbacks

    def _require_callbacks(self) -> "SchedulerCallbacks":
        callbacks = self._callbacks
        if callbacks is None:
            callbacks = getattr(self._host, "scheduler_callbacks", None)
            self._callbacks = callbacks
        if callbacks is None:
            raise RuntimeError("scheduler callbacks are not configured")
        return callbacks

    # ================================================================
    # 节点管理
    # ================================================================

    def _effective_role(self) -> str:
        """
        返回当前节点的有效角色。

        正常情况返回 scheduler 模块持有的运行时 NODE_ROLE；若 MAC 不匹配
        时自动切换到 client 模式，则返回 "client"（通过 _role_override 覆盖）。
        node_config/bootstrap 会同步这个模块级值，不能回读启动时已经过期的
        config.NODE_ROLE。
        """
        configured_role = NODE_ROLE
        override = getattr(self, '_role_override', None)
        if override:
            return override
        controller = getattr(self, "_auto_role_controller", None)
        if controller is not None:
            return controller.runtime_role
        return configured_role

    # ================================================================
    # 动态模型分层
    # ================================================================

    @staticmethod
    def _gpu_is_integrated(gpu: dict) -> bool:
        """Compatibility facade for GPU classification."""
        return _layer_plan.gpu_is_integrated(gpu)

    @classmethod
    def _select_scoring_gpu(cls, device_info: dict) -> dict:
        """Compatibility facade for scoring GPU selection."""
        return _layer_plan.select_scoring_gpu(
            device_info, gpu_is_integrated_fn=cls._gpu_is_integrated,
        )

    @staticmethod
    def _node_is_island_gateway(device_info: dict) -> bool:
        """Compatibility facade for island-gateway profile detection."""
        return _layer_plan.node_is_island_gateway(device_info)

    def _compute_node_weight(self, device_info: dict) -> float:
        """Compatibility facade for device-profile scoring."""
        return _layer_plan.compute_node_weight(
            device_info,
            select_gpu=self._select_scoring_gpu,
            gpu_is_integrated_fn=self._gpu_is_integrated,
            debug=logger.debug,
        )

    def _sync_node_rtt(self, node_id: str, tcp_client) -> None:
        """
        从 TCP 客户端同步 RTT 到 NodeInfo。

        每次心跳后调用，将 TCP 层测量的 RTT 同步到节点信息中，
        供分层算法和前端展示使用。
        """
        with self._nodes_lock:
            node = self.nodes.get(node_id)
            if node is None:
                return
            node.last_heartbeat = time.time()
            avg_rtt = getattr(tcp_client, 'avg_rtt_ms', 0.0)
            if avg_rtt > 0:
                node.last_rtt_ms = avg_rtt  # 当前 EWMA 值
                node.avg_rtt_ms = avg_rtt

    def _get_node_vram_mb(self, node_id: str) -> float:
        """
        从节点设备画像中提取可用显存 (MB)。

        优先级: GPU 专用显存 > 系统可用 RAM（CPU/集显模式）
        """
        with self._nodes_lock:
            node = self.nodes.get(node_id)
        if not node or not node.device_info:
            return 0.0
        gpu = self._select_scoring_gpu(node.device_info)
        ram = node.device_info.get("ram", {})
        uses_dedicated_vram = bool(
            isinstance(gpu, dict)
            and gpu.get("cuda_available", False)
            and not self._gpu_is_integrated(gpu)
            and gpu.get("vram_total_gb", 0) > 0
        )
        if uses_dedicated_vram:
            return gpu["vram_total_gb"] * 1024  # GB → MB
        if isinstance(ram, dict):
            return ram.get("available_gb", 0) * 1024
        return 0.0

    def _check_vram_constraint(self, node_id: str, layers_count: int,
                                has_embedding: bool = False,
                                has_lm_head: bool = False) -> tuple:
        """
        检查节点是否有足够显存承载分配的层范围。

        Args:
            node_id: 节点 ID
            layers_count: 分配的 Transformer 层数
            has_embedding: 是否包含 Embedding 层
            has_lm_head: 是否包含 LM Head

        Returns:
            (ok: bool, needed_mb: float, available_mb: float)
        """
        vram_available = self._get_node_vram_mb(node_id)
        if vram_available <= 0:
            return (True, 0, 0)  # 无法判断 → 放行

        # 根据当前模型结构和分层加载的真实精度估算内存。CUDA 分层使用
        # FP16，CPU/集显分层使用 FP32，都不能套用全模型 int4/int8 缩放系数。
        from config import (
            MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB,
            SAFE_VRAM_MARGIN, LAYER_VRAM_FACTOR, QUANT_TYPE,
        )
        layer_mb, embedding_mb, lm_head_mb = self._get_layer_memory_estimate_mb(
            node_id=node_id,
            fallback=(MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB),
            quant_factors=LAYER_VRAM_FACTOR,
            configured_quant=QUANT_TYPE,
        )
        vram_needed = layers_count * layer_mb
        if has_embedding:
            vram_needed += embedding_mb
        if has_lm_head:
            vram_needed += lm_head_mb
        vram_needed *= SAFE_VRAM_MARGIN

        ok = vram_available >= vram_needed
        return (ok, round(vram_needed, 1), round(vram_available, 1))

    def _get_layer_memory_estimate_mb(self, node_id: str,
                                      fallback: tuple,
                                      quant_factors: dict,
                                      configured_quant: str) -> tuple:
        """Return per-layer, embedding and LM-head memory in MiB."""
        manager = self._host
        model_config = getattr(getattr(manager, "model", None), "config", None)

        with self._nodes_lock:
            node = self.nodes.get(node_id)
        gpu = self._select_scoring_gpu(node.device_info if node else {})
        uses_cuda = bool(
            gpu.get("cuda_available", False)
            and not self._gpu_is_integrated(gpu)
        ) if isinstance(gpu, dict) else False
        bytes_per_parameter = 2 if uses_cuda else 4

        descriptor = {}
        get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
        if callable(get_descriptor):
            try:
                descriptor = get_descriptor() or {}
            except Exception:
                logger.debug("读取流水线模型描述器失败，继续使用结构估算", exc_info=True)
        layer_bytes = descriptor.get("layer_weight_bytes") or []
        component_bytes = descriptor.get("component_weight_bytes") or {}
        if layer_bytes:
            # The artifact byte counts are exact for its source dtype. CPU layer
            # loading widens FP16/BF16 to FP32; CUDA preserves the source width.
            source_bits_factor = 1.0
            if not uses_cuda:
                source_bits_factor = 2.0
            mib = 1024.0 * 1024.0
            average_layer_mb = (
                sum(int(value) for value in layer_bytes) / len(layer_bytes)
                * source_bits_factor / mib
            )
            embedding_mb = (
                int(component_bytes.get("embedding", 0) or 0)
                * source_bits_factor / mib
            )
            lm_head_bytes = int(component_bytes.get("lm_head", 0) or 0)
            if not lm_head_bytes and descriptor.get("tie_word_embeddings", False):
                lm_head_bytes = int(component_bytes.get("embedding", 0) or 0)
            lm_head_mb = lm_head_bytes * source_bits_factor / mib
            if average_layer_mb > 0:
                return average_layer_mb, embedding_mb, lm_head_mb

        if model_config is None and manager is not None:
            model_path = getattr(manager, "_model_path", "") or ""
            config_path = os.path.join(model_path, "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        model_config = type("ConfigView", (), json.load(handle))()
                except (OSError, TypeError, ValueError):
                    model_config = None

        if getattr(model_config, "model_type", "") == "qwen2":
            try:
                hidden = int(getattr(model_config, "hidden_size"))
                intermediate = int(getattr(model_config, "intermediate_size"))
                attention_heads = int(getattr(model_config, "num_attention_heads"))
                kv_heads = int(getattr(model_config, "num_key_value_heads", attention_heads))
                vocab = int(getattr(model_config, "vocab_size"))
                head_dim = hidden // attention_heads
                attention_params = (
                    hidden * hidden * 2
                    + hidden * kv_heads * head_dim * 2
                )
                mlp_params = hidden * intermediate * 3
                norm_params = hidden * 2
                mib = 1024.0 * 1024.0
                layer_mb = (
                    attention_params + mlp_params + norm_params
                ) * bytes_per_parameter / mib
                io_mb = vocab * hidden * bytes_per_parameter / mib
                return layer_mb, io_mb, io_mb
            except (TypeError, ValueError, ZeroDivisionError, AttributeError):
                logger.debug("读取 Qwen2 模型结构内存参数失败，使用回退估算", exc_info=True)

        quant = getattr(manager, "quant_type", None) or configured_quant
        # CPU workers materialize FP32 weights; fallback constants use FP16 as baseline.
        factor = quant_factors.get(quant, 1.0) if uses_cuda else 2.0
        return tuple(float(value) * factor for value in fallback)

    def _layer_assignment_cache_key(self, total_layers: int) -> str:
        """Bind dynamic cache to executable nodes, profiles, model and algorithm."""
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        with self._nodes_lock:
            nodes = []
            for node_id, info in self.nodes.items():
                if not _node_supports_forward_layers(info):
                    continue
                if node_id in opted_out:
                    continue
                # 孤岛网关不参与层拆分，缓存键同步排除保持一致
                if self._node_is_island_gateway(info.device_info):
                    continue
                if info.role != NodeRole.MASTER and not info.is_available():
                    continue
                nodes.append({
                    "node_id": node_id,
                    "role": str(info.role),
                    "device_info": info.device_info or {},
                })
        manager = self._host
        model_path = os.path.abspath(
            getattr(manager, "_model_path", "") or ""
        ) if manager else ""
        weight_fingerprint = []
        if model_path and os.path.isdir(model_path):
            try:
                for root, _dirs, files in os.walk(model_path):
                    for name in sorted(files):
                        if not name.lower().endswith((".safetensors", ".bin")):
                            continue
                        path = os.path.join(root, name)
                        stat = os.stat(path)
                        weight_fingerprint.append((
                            os.path.relpath(path, model_path).replace(os.sep, "/"),
                            stat.st_size,
                            stat.st_mtime_ns,
                        ))
            except OSError:
                weight_fingerprint = []
        payload = {
            "version": _LAYER_ASSIGNMENT_CACHE_VERSION,
            "total_layers": int(total_layers),
            "model_id": getattr(manager, "active_model_id", "") if manager else "",
            "model_path": model_path,
            "quant_type": getattr(manager, "quant_type", "") if manager else "",
            "weight_fingerprint": weight_fingerprint,
            "nodes": sorted(nodes, key=lambda item: item["node_id"]),
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _get_total_model_layers(self) -> int:
        """Return the active PyTorch model's real decoder-layer count."""
        from config import TOTAL_MODEL_LAYERS

        manager = self._host
        if manager is not None:
            get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
            if callable(get_descriptor):
                try:
                    descriptor_layers = int(
                        (get_descriptor() or {}).get("total_layers", 0) or 0
                    )
                    if descriptor_layers > 0:
                        return descriptor_layers
                except (TypeError, ValueError):
                    pass
                except Exception:
                    logger.debug("读取流水线描述器层数失败", exc_info=True)
            for value in (
                getattr(manager, "_total_model_layers", 0),
                getattr(getattr(getattr(manager, "model", None), "config", None),
                        "num_hidden_layers", 0),
            ):
                try:
                    if int(value) > 0:
                        return int(value)
                except (TypeError, ValueError):
                    pass

            model_path = getattr(manager, "_model_path", "") or ""
            config_path = os.path.join(model_path, "config.json")
            if os.path.isfile(config_path):
                try:
                    import json

                    with open(config_path, "r", encoding="utf-8") as handle:
                        value = int(json.load(handle).get("num_hidden_layers", 0))
                    if value > 0:
                        return value
                except (OSError, TypeError, ValueError):
                    logger.debug("读取当前模型层数失败: %s", config_path, exc_info=True)
        return TOTAL_MODEL_LAYERS

    def _get_active_pipeline_model_info(self) -> dict:
        """Describe a PyTorch artifact without requiring a full model load."""
        manager = self._host
        if not manager or not runtime_supports(manager, Capability.FORWARD_LAYERS):
            return {}
        get_descriptor = getattr(manager, "get_pipeline_descriptor", None)
        if not callable(get_descriptor):
            return {}
        try:
            descriptor = get_descriptor() or {}
        except Exception:
            logger.warning("读取主节点流水线模型描述器失败", exc_info=True)
            return {}
        if not descriptor.get("pipeline_runtime_supported", False):
            return {}
        model_path = os.path.abspath(
            getattr(manager, "_full_model_path", "")
            or getattr(manager, "_model_path", "")
            or ""
        )
        if not model_path or not os.path.isdir(model_path):
            return {}
        model_type = str(descriptor.get("model_type", "") or "").lower()
        if model_type not in {"qwen", "qwen2"}:
            return {}
        return {
            "model_id": descriptor.get("model_id")
            or getattr(manager, "active_model_id", "") or "",
            "model_path": model_path,
            "model_sha256": descriptor.get("model_sha256")
            or self._get_master_model_sha256(),
            "model_type": model_type,
            "total_layers": int(descriptor["total_layers"]),
            "quant_type": getattr(manager, "quant_type", "") or "",
            "inspection_mode": descriptor.get("inspection_mode", ""),
            "weight_bytes": int(descriptor.get("weight_bytes", 0) or 0),
        }

    def compute_layer_assignment(self, nodes: list = None) -> list:
        """
        根据节点硬件配置动态计算模型分层方案。

        算法选择（两级策略）:
          - 节点数 > GRAPH_ORCHESTRATOR_THRESHOLD (默认5):
            ★ 图算法智能编排 — 最大带宽生成树 + DFS 路径搜索
            → 输出带宽感知的最优链式拓扑
          - 节点数 ≤ 阈值:
            简单算力权重比例分配（master 优先排序）

        Args:
            nodes: 可选，指定节点列表；若为 None 则使用 self.nodes 中所有节点

        Returns:
            [{node_id, role, start_layer, end_layer, layers_count,
              has_embedding, has_lm_head, score}]
        """
        from config import GRAPH_ORCHESTRATOR_THRESHOLD
        from config import (
            QUANT_TYPE, MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB,
            LM_HEAD_VRAM_MB, LAYER_VRAM_FACTOR, SAFE_VRAM_MARGIN,
        )

        total_layers = self._get_total_model_layers()

        # 收集明确声明 forward_layers 能力的节点；未声明能力的 Android
        # 继续按旧行为排除，避免把 HTTP 客户端误当成层工作器。
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        if nodes is None:
            # Phase 2.1: 快照 self.nodes 后解锁迭代，防止 TCP 回调并发修改 dict
            with self._nodes_lock:
                nodes_snapshot = list(self.nodes.items())
            node_list = [
                {"node_id": nid, "role": info.role,
                 "node_type": info.node_type,
                 "device_info": info.device_info}
                for nid, info in nodes_snapshot
                if _node_supports_forward_layers(info)
                and nid not in opted_out
                and not self._node_is_island_gateway(info.device_info)
                and (
                    info.role == NodeRole.MASTER
                    or not hasattr(info, "is_available")
                    or info.is_available()
                )
            ]
        else:
            node_list = [
                n for n in nodes
                if _node_supports_forward_layers(n)
                and n.get("node_id") not in opted_out
                and not self._node_is_island_gateway(n.get("device_info", {}))
            ]

        if not node_list:
            logger.warning("没有可用的层前向节点参与流水线层拆分")
            return []

        # 单节点：全部层给该节点。多节点时 master 也参与首段层计算，
        # run_pipeline() 会先在 master 本地执行其层范围，再把 hidden_states
        # 交给第一个 worker，避免浪费主节点 CUDA 独显算力。
        if len(node_list) == 1:
            n = node_list[0]
            return [{
                "node_id": n["node_id"],
                "role": n["role"],
                "start_layer": 0,
                "end_layer": total_layers,
                "layers_count": total_layers,
                "has_embedding": True,
                "has_lm_head": True,
                "score": 50.0,
            }]

        # ============================================================
        # ★ 图算法智能编排（节点数 > 阈值）
        #   用最大带宽生成树 + DFS 替代纯算力权重，生成带宽感知
        #   最优链式拓扑，指导节点间直连排序。
        # ============================================================
        if len(node_list) > GRAPH_ORCHESTRATOR_THRESHOLD:
            try:
                from graph_orchestrator import GraphOrchestrator

                # 图编排使用真实分层加载内存；按节点运行后端估计，
                # 后续仍会逐节点执行精确显存约束。
                layer_mb, embedding_mb, lm_head_mb = self._get_layer_memory_estimate_mb(
                    node_id=node_list[0]["node_id"],
                    fallback=(MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB),
                    quant_factors=LAYER_VRAM_FACTOR,
                    configured_quant=QUANT_TYPE,
                )
                model_memory_mb = (
                    total_layers * layer_mb + embedding_mb + lm_head_mb
                ) * SAFE_VRAM_MARGIN

                # 构建 nodes dict（GraphOrchestrator 需要的格式）
                # Phase 2.1+: 锁保护，防止 in/get 之间的并发删除
                with self._nodes_lock:
                    orch_nodes = {}
                    for n in node_list:
                        nid = n["node_id"]
                        if nid in self.nodes:
                            orch_nodes[nid] = self.nodes[nid]

                if len(orch_nodes) > GRAPH_ORCHESTRATOR_THRESHOLD:
                    orchestrator = GraphOrchestrator(
                        nodes=orch_nodes,
                        model_memory_mb=model_memory_mb,
                        total_layers=total_layers,
                        quant_factor=1.0,
                    )
                    assignments = orchestrator.orchestrate()

                    # 先确定 master 锚点和 I/O 头归属，再按真实内存负担校验。
                    assignments = self._normalize_master_anchor(
                        assignments, node_list, total_layers
                    )
                    assignments = self._apply_vram_constraints(assignments)
                    assignments = self._normalize_master_anchor(
                        assignments, node_list, total_layers
                    )

                    logger.info(
                        f"🧠 图算法智能编排完成: {len(assignments)} 节点, "
                        f"总 {total_layers} 层, 策略=graph_orchestrator"
                    )
                    for a in assignments:
                        logger.info(
                            f"  {a['node_id']}: Layer {a['start_layer']}-{a['end_layer']} "
                            f"({a['layers_count']}层) embed={a['has_embedding']} "
                            f"lm_head={a['has_lm_head']} score={a['score']}"
                        )
                    return assignments
            except Exception as e:
                logger.warning(
                    f"图算法智能编排失败: {e}，回退到简单权重分配",
                    exc_info=True,
                )

        # ============================================================
        # 回退：简单权重比例分配（节点数 ≤ 阈值 或 图编排器异常）
        # ============================================================
        # 双重回退链：
        #   1. GraphOrchestrator.orchestrate()
        #        → _dfs_path_search 无可行路径
        #        → 内部回退 _fallback_weight_assignment（VRAM 比例分）
        #   2. GraphOrchestrator 顶层抛异常
        #        → 外部回退到此 _simple_weight_assignment（算力权重分）
        # 两层回退确保即使图算法崩溃，系统仍能降级到可用状态。
        return self._simple_weight_assignment(node_list, total_layers)

    def _simple_weight_assignment(self, node_list: list,
                                   total_layers: int) -> list:
        """
        简单权重比例分配（节点数 ≤ GRAPH_ORCHESTRATOR_THRESHOLD 时使用）。

        算法:
          1. 计算各节点算力权重
          2. 按权重比例分配 Transformer 层
          3. 按 master 优先排序
          4. 首节点含 Embedding，末节点含 LM Head
          5. 显存约束校验
        """
        # Step 1: 计算权重
        for n in node_list:
            n["score"] = self._compute_node_weight(n.get("device_info", {}))

        total_weight = sum(n["score"] for n in node_list)

        # 权重全为 0 → 均分
        if total_weight <= 0:
            base = total_layers // len(node_list)
            remainder = total_layers % len(node_list)
            node_list.sort(key=lambda n: (n["role"] != "master", -n.get("score", 0)))
            assignments = []
            cursor = 0
            for i, n in enumerate(node_list):
                count = base + (1 if i < remainder else 0)
                assignments.append({
                    "node_id": n["node_id"],
                    "role": n["role"],
                    "start_layer": cursor,
                    "end_layer": cursor + count,
                    "layers_count": count,
                    "has_embedding": (i == 0),
                    "has_lm_head": (i == len(node_list) - 1),
                    "score": 0.0,
                })
                cursor += count
            assignments = self._normalize_master_anchor(
                assignments, node_list, total_layers
            )
            assignments = self._apply_vram_constraints(assignments)
            return self._normalize_master_anchor(assignments, node_list, total_layers)

        # Step 2: 按比例分配全部 Transformer 层
        distributable = total_layers
        raw_layers = []
        for n in node_list:
            proportion = n["score"] / total_weight
            raw = max(1, round(proportion * distributable))
            raw_layers.append(raw)

        # 修正 rounding 误差
        diff = distributable - sum(raw_layers)
        if diff > 0:
            for i in range(diff):
                idx = i % len(raw_layers)
                raw_layers[idx] += 1
        elif diff < 0:
            # 从低分节点优先削减，保留高分节点层数
            sorted_indices = sorted(range(len(raw_layers)), key=lambda i: node_list[i]["score"])
            for _ in range(-diff):
                reduced = False
                for idx in sorted_indices:
                    if raw_layers[idx] > 1:
                        raw_layers[idx] -= 1
                        reduced = True
                        break
                # 所有剩余节点都已降至 1 层 → 削去最低分节点（将被过滤移除）
                if not reduced:
                    for idx in sorted_indices:
                        if raw_layers[idx] >= 1:
                            raw_layers[idx] -= 1
                            break

        # ★ 清理分配层数 ≤ 0 的节点（极端情况：节点数远超层数）
        valid_pairs = [(idx, layers) for idx, layers in enumerate(raw_layers) if layers > 0]
        if len(valid_pairs) < len(raw_layers):
            logger.warning(
                f"节点数 ({len(raw_layers)}) 超过可分配层数 ({distributable})，"
                f"{len(raw_layers) - len(valid_pairs)} 个低分节点将被排除"
            )

        # Step 3: 排序（master 优先，同角色按权重降序），跳过已移除节点
        sorted_pairs = sorted(
            enumerate(node_list),
            key=lambda x: (x[1]["role"] != "master", -x[1]["score"])
        )
        sorted_indices = [i for i, _ in sorted_pairs if raw_layers[i] > 0]

        # Step 4: 构建区间
        assignments = []
        cursor = 0
        for order, idx in enumerate(sorted_indices):
            n = node_list[idx]
            count = raw_layers[idx]
            assignments.append({
                "node_id": n["node_id"],
                "role": n["role"],
                "start_layer": cursor,
                "end_layer": cursor + count,
                "layers_count": count,
                "has_embedding": (order == 0),
                "has_lm_head": (order == len(sorted_indices) - 1),
                "score": round(n["score"], 1),
            })
            cursor += count

        # Step 5: 先决定 I/O 头归属，再按真实内存负担校验。
        assignments = self._normalize_master_anchor(
            assignments, node_list, total_layers
        )
        assignments = self._apply_vram_constraints(assignments)
        assignments = self._normalize_master_anchor(
            assignments, node_list, total_layers
        )

        logger.info(
            f"动态分层计算完成: {len(assignments)} 节点, "
            f"总 {total_layers} 层, 策略=simple_weight"
        )
        for a in assignments:
            logger.info(
                f"  {a['node_id']}: Layer {a['start_layer']}-{a['end_layer']} "
                f"({a['layers_count']}层) embed={a['has_embedding']} "
                f"lm_head={a['has_lm_head']} score={a['score']}"
            )

        return assignments

    def _normalize_master_anchor(self, assignments: list, node_list: list,
                                 total_layers: int) -> list:
        """Compatibility facade for master-anchored assignment normalization."""
        return _layer_plan.normalize_master_anchor(
            assignments,
            node_list,
            total_layers,
            compute_weight=self._compute_node_weight,
            resequence=self._resequence_assignments,
            warning=logger.warning,
        )

    @staticmethod
    def _resequence_assignments(assignments: list) -> list:
        """Compatibility facade for contiguous assignment ranges."""
        return _layer_plan.resequence_assignments(assignments)

    def _apply_vram_constraints(self, assignments: list) -> list:
        """
        显存约束校验：检查每节点是否有足够显存承载分配的层。

        若不足，将超额层转移给 VRAM 最充裕的节点；层数为 0 的节点从列表中移除。
        转移后重新计算所有节点的 layer 区间以保持连续性。

        ★ 保护规则（方案 A）:
          - 首节点（has_embedding）至少保留 1 层 Transformer，确保 Embedding 权重有
            同节点层可锚定，避免出现「纯 Embedding 节点」。
          - 末节点（has_lm_head）同样至少保留 1 层。
          - 若连 1 层都保不住，打 ERROR 日志并跳过转移（降级为本地推理兜底）。
        """
        pending = []
        for a in assignments:
            ok, needed, available = self._check_vram_constraint(
                a["node_id"], a["layers_count"],
                a["has_embedding"], a["has_lm_head"],
            )
            if not ok and available > 0:
                max_fit = 0
                for count in range(int(a["layers_count"]), -1, -1):
                    fits, _needed, _available = self._check_vram_constraint(
                        a["node_id"], count, a["has_embedding"], a["has_lm_head"]
                    )
                    if fits:
                        max_fit = count
                        break
                minimum = 1 if (a["has_embedding"] or a["has_lm_head"]) else 0
                if max_fit < minimum:
                    logger.error(
                        f"❌ 节点 {a['node_id']} 连 I/O 头和 1 层都无法容纳"
                        f"（需 {needed}MB / 可用 {available}MB），保留原分配并由就绪检查回退。"
                    )
                    continue
                overflow = max(0, int(a["layers_count"]) - max_fit)
                if overflow:
                    a["layers_count"] = max_fit
                    pending.append((a["node_id"], overflow))

        for source_id, overflow in pending:
            while overflow > 0:
                candidates = []
                for other in assignments:
                    if other["node_id"] == source_id:
                        continue
                    fits, _, _ = self._check_vram_constraint(
                        other["node_id"], other["layers_count"] + 1,
                        other["has_embedding"], other["has_lm_head"],
                    )
                    if fits:
                        candidates.append(other)
                if not candidates:
                    logger.error("❌ %s 的 %s 个超额层没有节点具备剩余容量", source_id, overflow)
                    # 保持完整覆盖，让 worker 加载失败 ACK 触发全模型回退，而不是静默丢层。
                    source = next(a for a in assignments if a["node_id"] == source_id)
                    source["layers_count"] += overflow
                    break
                target = max(
                    candidates,
                    key=lambda item: (
                        self._get_node_vram_mb(item["node_id"]),
                        float(item.get("score", 0) or 0),
                    ),
                )
                target["layers_count"] += 1
                overflow -= 1

        # 清除层数为 0 的节点
        assignments[:] = [a for a in assignments if a["layers_count"] > 0]

        # ★ 重新计算所有节点的 layer 区间，确保 start_layer/end_layer 连续。
        # LM Head 是输出投影，不属于 Transformer 层区间；主节点可在 worker
        # 返回尾层 hidden_states 后执行它，避免弱 worker 计算/回传全词表 logits。
        master_index = next((
            i for i, item in enumerate(assignments)
            if item.get("node_id") == "master" or item.get("role") == "master"
        ), None)
        lm_head_index = len(assignments) - 1
        if master_index is not None and assignments:
            try:
                master_score = float(assignments[master_index].get("score", 0) or 0)
                tail_score = float(assignments[-1].get("score", 0) or 0)
                if master_score >= tail_score:
                    lm_head_index = master_index
            except (TypeError, ValueError):
                pass
        cursor = 0
        for i, a in enumerate(assignments):
            a["start_layer"] = cursor
            a["end_layer"] = cursor + a["layers_count"]
            a["has_embedding"] = (i == 0)
            a["has_lm_head"] = (i == lm_head_index)
            cursor += a["layers_count"]

        return assignments

    def get_layer_assignments(self) -> dict:
        """
        获取当前分层配置。

        优先返回当前进程的手动覆盖，否则根据实时节点画像动态计算。

        Returns:
            {
                "total": 24,
                "strategy": "dynamic" | "graph_orchestrator" | "manual",
                "assignments": [...],
                "computed_at": timestamp | null,
            }
        """
        from config import GRAPH_ORCHESTRATOR_THRESHOLD

        total_layers = self._get_total_model_layers()
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
            capacity_plan = (
                dict(self._active_pipeline_capacity_plan)
                if self._active_pipeline_capacity_plan else None
            )
            if capacity_plan is None and self._pipeline_load_transaction:
                candidate = self._pipeline_load_transaction.get("plan")
                if isinstance(candidate, dict) and candidate.get("admitted"):
                    capacity_plan = dict(candidate)
                    capacity_plan["transaction_phase"] = (
                        self._pipeline_load_transaction.get("phase", "")
                    )

        if capacity_plan and capacity_plan.get("admitted"):
            return {
                "total": int(capacity_plan.get("total_layers", total_layers)),
                "strategy": "capacity",
                "assignments": [
                    dict(item) for item in capacity_plan.get("assignments", [])
                ],
                "computed_at": capacity_plan.get("computed_at"),
                "plan_id": capacity_plan.get("plan_id", ""),
            }

        if self._runtime_layer_override:
            overrides = self._normalize_manual_assignments(
                [
                    item for item in self._runtime_layer_override
                    if item.get("node_id") not in opted_out
                ]
            )
            if (
                overrides
                and self._manual_assignments_are_executable(overrides)
                and max(int(item.get("end_layer", 0)) for item in overrides)
                == total_layers
            ):
                return {
                    "total": total_layers,
                    "strategy": "manual",
                    "assignments": overrides,
                    "computed_at": None,
                }

        cache_key = self._layer_assignment_cache_key(total_layers)

        # 动态计算
        assignments = self.compute_layer_assignment()

        # 判断实际使用的策略：参与层前向的 worker 才计入图编排阈值；
        # Android llama.cpp worker 与 PC worker 在这里具有同等语义。
        worker_nodes_count = sum(
            1 for info in self.nodes.values()
            if _node_supports_forward_layers(info)
            and info.node_id != "master" and info.role != "master"
            and info.is_available()
        )
        actual_strategy = (
            "graph_orchestrator"
            if worker_nodes_count > GRAPH_ORCHESTRATOR_THRESHOLD
            else "dynamic"
        )

        result = {
            "total": total_layers,
            "strategy": actual_strategy,
            "assignments": assignments,
            "computed_at": time.time(),
            "cache_key": cache_key,
        }

        return result

    def _get_pipeline_capacity_nodes(
        self, eligible_node_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Project live layer-worker profiles into explicit free-memory budgets."""
        from config import PIPELINE_CAPACITY_RESERVE_MB

        reserve_bytes = int(PIPELINE_CAPACITY_RESERVE_MB * 1024 * 1024)
        with self._layer_config_lock:
            opted_out = set(self._pipeline_worker_opt_out)
        with self._nodes_lock:
            snapshot = list(self.nodes.items())

        records = []
        effective_id = self.get_effective_node_id()
        for node_id, node in snapshot:
            if (
                not _node_supports_forward_layers(node)
                or (eligible_node_ids is not None and node_id not in eligible_node_ids)
                or node_id in opted_out
                or self._node_is_island_gateway(node.device_info)
                or (
                    node.role != NodeRole.MASTER
                    and node_id != effective_id
                    and not node.is_available()
                )
            ):
                continue
            device_info = dict(node.device_info or {})
            gpu = self._select_scoring_gpu(device_info)
            cuda_discrete = bool(
                isinstance(gpu, dict)
                and gpu.get("cuda_available", False)
                and not self._gpu_is_integrated(gpu)
            )
            capacity_gb = 0.0
            capacity_source = ""
            runtime_multiplier = 2.0
            execution_device = "cpu"
            if cuda_discrete:
                try:
                    capacity_gb = float(gpu.get("vram_free_gb", 0) or 0)
                except (TypeError, ValueError):
                    capacity_gb = 0.0
                capacity_source = "gpu.vram_free_gb" if capacity_gb > 0 else ""
                runtime_multiplier = 1.0
                execution_device = "cuda"
            else:
                ram = device_info.get("ram", {})
                if isinstance(ram, dict):
                    try:
                        capacity_gb = float(ram.get("available_gb", 0) or 0)
                    except (TypeError, ValueError):
                        capacity_gb = 0.0
                capacity_source = "ram.available_gb" if capacity_gb > 0 else ""
            records.append({
                "node_id": node_id,
                "role": node.role,
                "capacity_bytes": max(0, int(capacity_gb * 1024 ** 3)),
                "reserve_bytes": reserve_bytes,
                "runtime_multiplier": runtime_multiplier,
                "execution_device": execution_device,
                "capacity_source": capacity_source,
                "score": self._compute_node_weight(device_info),
            })
        return records

    def _pipeline_node_metadata(self) -> dict[str, dict]:
        """Project live scheduler nodes into opaque layout-location metadata."""
        local_node_id = (
            "master" if self._effective_role() == "master"
            else self.get_effective_node_id()
        )
        with self._nodes_lock:
            snapshot = list(self.nodes.items())
        return {
            node_id: {
                "is_local": node_id == local_node_id,
                # Public plans distinguish local/remote without exporting an
                # address that belongs to the transport control plane.
                "location": "local" if node_id == local_node_id else f"node:{node_id}",
                "kind": "local" if node_id == local_node_id else "remote_pipeline",
                "federated": node_id != local_node_id,
                "engine": (
                    "relay_middle"
                    if callable(getattr(self, "_relay_segment_for_worker", None))
                    and self._relay_segment_for_worker(node_id) is not None
                    else "pytorch"
                ),
            }
            for node_id, _node in snapshot
        }

    def _attach_pipeline_node_contract(self, plan: dict) -> dict:
        """Attach the canonical node layout or reject a malformed admission."""
        if not isinstance(plan, dict) or plan.get("admitted") is not True:
            return plan
        result = dict(plan)
        try:
            layout = pipeline_layout_from_capacity_plan(
                result, node_metadata=self._pipeline_node_metadata(),
            )
        except PipelineNodeContractError as exc:
            result.update({
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_node_contract_invalid",
                "reason": str(exc),
                "assignments": [],
                "pipeline_layout": None,
            })
            return result
        result["pipeline_layout"] = layout.to_dict()
        return result

    def _activate_pipeline_reshard_coordinator(self, plan: Mapping[str, Any]) -> None:
        """Bind a ready capacity plan to an epoch-fenced recovery topology."""
        try:
            layout = pipeline_layout_from_capacity_plan(
                plan, node_metadata=self._pipeline_node_metadata(),
            )
            coordinator = PipelineReshardCoordinator(
                layout,
                lease_book=RpcShardLeaseBook(control_fence=self._control_fence),
            )
        except (PipelineNodeContractError, ValueError) as exc:
            logger.warning("未启用自动重分片合同: %s", exc)
            with self._layer_config_lock:
                self._pipeline_reshard_coordinator = None
                self._pipeline_reshard_last_decision = {
                    "status": "unavailable",
                    "reason_code": "pipeline_reshard_layout_invalid",
                    "reason": str(exc),
                }
            return
        with self._layer_config_lock:
            self._pipeline_reshard_coordinator = coordinator
            self._pipeline_reshard_last_decision = {
                "status": "active",
                "reason_code": "",
                "epoch": coordinator.epoch,
            }

    def _stage_pipeline_reshard_after_disconnect(self, node_id: str) -> Optional[dict]:
        """Re-solve an active topology after a worker loss without writeback."""
        if self._effective_role() != "master":
            return None
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
        if coordinator is None or node_id not in {
            item.node_id for item in coordinator.layout.nodes
        }:
            return None
        get_descriptor = getattr(self._host, "get_pipeline_descriptor", None)
        descriptor = get_descriptor() if callable(get_descriptor) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            report = {
                "status": "rejected",
                "accepted": False,
                "reason_code": "pipeline_reshard_descriptor_unavailable",
            }
        else:
            from config import PIPELINE_CAPACITY_SAFETY_MARGIN

            server = self._tcp_server
            get_client_ids = getattr(server, "get_client_ids", None)
            connected_ids = (
                set(get_client_ids()) if callable(get_client_ids)
                else set(getattr(server, "clients", {}).keys()) if server else set()
            )
            connected_ids.update({"master", self.get_effective_node_id()})
            connected_ids.difference_update(self._task_worker_full_model_ids())

            failed_node_ids = {node_id}
            snapshot = getattr(coordinator, "snapshot", None)
            if callable(snapshot):
                for staged in snapshot().get("staged", []):
                    failed_node_ids.update(
                        str(item) for item in staged.get("failed_node_ids", [])
                        if str(item)
                    )

            decision = coordinator.stage_failure(
                failed_node_ids,
                descriptor=descriptor,
                capacity_nodes=self._get_pipeline_capacity_nodes(connected_ids),
                node_metadata=self._pipeline_node_metadata(),
                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
            )
            report = decision.to_dict()
        with self._layer_config_lock:
            self._pipeline_reshard_last_decision = report
        logger.warning(
            "节点断线自动重分片计划: node=%s status=%s reason=%s",
            node_id, report.get("status", ""), report.get("reason_code", ""),
        )
        return report

    def _commit_ready_pipeline_reshard(self, plan: Mapping[str, Any]) -> Optional[bool]:
        """Commit the staged epoch after the matching load transaction is ready."""
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
        if coordinator is None:
            return None
        staged = tuple(coordinator.snapshot().get("staged", []))
        if not staged:
            return None
        plan_id = str(plan.get("plan_id", "") or "")
        matching = next((
            item for item in staged if item.get("capacity_plan_id") == plan_id
        ), None)
        if matching is None:
            with self._layer_config_lock:
                self._pipeline_reshard_last_decision = {
                    "status": "rejected",
                    "reason_code": "pipeline_reshard_plan_mismatch",
                }
            logger.error(
                "重分片事务计划不匹配，拒绝发布: active=%s staged=%s",
                plan_id, [item.get("capacity_plan_id", "") for item in staged],
            )
            return False
        staged_plan = coordinator.staged_plan(str(matching.get("plan_id", "")))
        if staged_plan is None:
            return False
        try:
            ready_layout = pipeline_layout_from_capacity_plan(
                plan, node_metadata=self._pipeline_node_metadata(),
            )
        except PipelineNodeContractError:
            return False
        if ready_layout.contract_sha256 != staged_plan.candidate_layout.contract_sha256:
            return False
        # Reaching ready proves the master materialized its segment and every
        # worker ACKed its own model identity, range, and boundary duties.
        for requirement in staged_plan.requirements:
            coordinator.record_artifact(PipelineArtifactAvailability(
                node_id=requirement.node_id,
                model_sha256=requirement.model_sha256,
                artifact_kind=requirement.artifact_kind,
                layer_range=requirement.layer_range,
                artifact_sha256=requirement.artifact_sha256,
                verified=True,
                has_embedding=requirement.has_embedding,
                has_lm_head=requirement.has_lm_head,
            ))
        committed = coordinator.commit(
            staged_plan.plan_id, expected_epoch=staged_plan.base_epoch,
        )
        with self._layer_config_lock:
            self._pipeline_reshard_last_decision = committed.to_dict()
        if not committed.accepted:
            logger.error("重分片 epoch 提交被拒绝: %s", committed.reason_code)
        return committed.accepted

    def get_pipeline_reshard_status(self) -> dict:
        """Return an address-free recovery projection for the API and TUI."""
        with self._layer_config_lock:
            coordinator = self._pipeline_reshard_coordinator
            last = (
                dict(self._pipeline_reshard_last_decision)
                if self._pipeline_reshard_last_decision else None
            )
        if coordinator is None:
            return {"status": "inactive", "epoch": 0, "last_decision": last}
        snapshot = coordinator.snapshot()
        return {
            "status": snapshot["status"],
            "epoch": snapshot["epoch"],
            "active_contract_sha256": snapshot["layout"]["contract_sha256"],
            "staged": snapshot["staged"],
            "last_decision": last,
        }

    def get_pipeline_capacity_plan(
        self, eligible_node_ids: Optional[set[str]] = None,
        *, descriptor: Optional[dict] = None,
        require_distributed: bool = False,
    ) -> dict:
        """Compute an all-or-nothing metadata-only cluster capacity plan.

        ``require_distributed`` is request-scoped. It bypasses a cached
        single-node plan and asks the solver for at least two participating
        nodes, so an explicit distributed request cannot be satisfied by a
        master-only placement.
        """
        if (
            not require_distributed
            and eligible_node_ids is None
            and descriptor is None
        ):
            with self._layer_config_lock:
                active = (
                    dict(self._active_pipeline_capacity_plan)
                    if self._active_pipeline_capacity_plan else None
                )
                transaction = self._pipeline_load_transaction
                if transaction:
                    transaction_snapshot = {
                        "config_id": transaction.get("config_id", ""),
                        "generation": transaction.get("generation", 0),
                        "transaction_phase": transaction.get("phase", ""),
                        "prepared_node_count": len(
                            transaction.get("prepared_nodes", set())
                        ),
                        "ready_node_count": len(
                            transaction.get("ready_nodes", set())
                        ),
                        "worker_count": len(transaction.get("worker_ids", set())),
                        "reason_code": transaction.get("reason_code", ""),
                        "reason": transaction.get("reason", ""),
                    }
                    transaction_plan = dict(transaction.get("plan", {}))
                else:
                    transaction_snapshot = None
                    transaction_plan = None
            if active:
                return self._attach_pipeline_node_contract(active)
            if transaction_snapshot and transaction_plan:
                transaction_plan.update(transaction_snapshot)
                return self._attach_pipeline_node_contract(transaction_plan)

        if descriptor is None:
            get_descriptor = getattr(self._host, "get_pipeline_descriptor", None)
            descriptor = get_descriptor() if callable(get_descriptor) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            return {
                "status": "unavailable",
                "admitted": False,
                "reason_code": "pipeline_descriptor_unavailable",
                "assignments": [],
            }
        if not descriptor.get("pipeline_runtime_supported", False):
            return {
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_runtime_unsupported",
                "model_id": str(descriptor.get("model_id", "") or ""),
                "model_type": str(descriptor.get("model_type", "") or ""),
                "assignments": [],
            }
        from config import PIPELINE_CAPACITY_SAFETY_MARGIN

        try:
            result = solve_pipeline_capacity(
                descriptor,
                self._get_pipeline_capacity_nodes(eligible_node_ids),
                safety_margin=PIPELINE_CAPACITY_SAFETY_MARGIN,
                require_distributed=require_distributed,
            )
        except PipelineCapacityError as exc:
            return {
                "status": "rejected",
                "admitted": False,
                "reason_code": "pipeline_capacity_descriptor_invalid",
                "reason": str(exc),
                "assignments": [],
            }
        result["computed_at"] = time.time()
        result["transaction_phase"] = "planned" if result.get("admitted") else "rejected"
        result.setdefault("require_distributed", bool(require_distributed))
        return self._attach_pipeline_node_contract(result)

    def _build_manual_pipeline_capacity_plan(
        self, assignments: list[dict], *, descriptor: Optional[dict] = None,
    ) -> dict:
        """Validate a user-forced split without replacing it with full-model fit.

        The normal solver intentionally minimizes participating nodes.  That is
        correct for automatic admission, but it would make a deliberate 21/24
        split impossible whenever the master can still hold the whole model.
        Manual plans retain the same byte and reserve checks while preserving
        the requested contiguous ranges.
        """
        from config import PIPELINE_CAPACITY_SAFETY_MARGIN
        from pipeline_capacity import _descriptor_costs, _required_bytes

        if descriptor is None:
            getter = getattr(self._host, "get_pipeline_descriptor", None)
            descriptor = getter() if callable(getter) else {}
        if not isinstance(descriptor, dict) or not descriptor:
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "pipeline_descriptor_unavailable",
                "assignments": [],
            }

        try:
            layer_bytes, embedding_bytes, per_node_bytes, output_bytes = (
                _descriptor_costs(descriptor)
            )
        except PipelineCapacityError as exc:
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "pipeline_capacity_descriptor_invalid",
                "reason": str(exc), "assignments": [],
            }

        total_layers = len(layer_bytes)
        node_ids = {str(item.get("node_id", "")) for item in assignments}
        records = {
            item["node_id"]: item
            for item in self._get_pipeline_capacity_nodes(node_ids)
        }
        planned = []
        for item in assignments:
            node_id = str(item.get("node_id", ""))
            start = int(item.get("start_layer", 0) or 0)
            end = int(item.get("end_layer", 0) or 0)
            record = records.get(node_id)
            if record is None:
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_capacity_manual_node_unavailable",
                    "reason": f"node {node_id} has no usable capacity",
                    "assignments": [],
                }
            raw_bytes = sum(layer_bytes[start:end]) + per_node_bytes
            if bool(item.get("has_embedding")):
                raw_bytes += embedding_bytes
            if bool(item.get("has_lm_head")):
                raw_bytes += output_bytes
            required = _required_bytes(
                raw_bytes, record, PIPELINE_CAPACITY_SAFETY_MARGIN,
            )
            if required > int(record["capacity_bytes"]):
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_capacity_manual_insufficient",
                    "reason": (
                        f"node {node_id} requires {required} bytes, "
                        f"has {record['capacity_bytes']} bytes"
                    ),
                    "assignments": [],
                }
            planned.append({
                **dict(item),
                "raw_weight_bytes": raw_bytes,
                "required_bytes": required,
                "capacity_bytes": record["capacity_bytes"],
                "headroom_bytes": record["capacity_bytes"] - required,
                "reserve_bytes": record["reserve_bytes"],
                "runtime_multiplier": record["runtime_multiplier"],
                "execution_device": record["execution_device"],
                "capacity_source": record["capacity_source"],
            })

        plan_identity = {
            "model_id": descriptor.get("model_id", ""),
            "model_sha256": descriptor.get("model_sha256", ""),
            "assignments": [
                {
                    key: item.get(key)
                    for key in (
                        "node_id", "start_layer", "end_layer",
                        "required_bytes", "capacity_bytes",
                    )
                }
                for item in planned
            ],
        }
        return self._attach_pipeline_node_contract({
            "schema_version": 1,
            "status": "admitted",
            "admitted": True,
            "reason_code": "manual_override",
            "plan_id": hashlib.sha256(
                json.dumps(plan_identity, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "model_id": str(descriptor.get("model_id", "") or ""),
            "model_type": str(descriptor.get("model_type", "") or ""),
            "model_sha256": str(descriptor.get("model_sha256", "") or ""),
            "total_layers": total_layers,
            "raw_model_bytes": sum(layer_bytes) + embedding_bytes
            + per_node_bytes + output_bytes,
            "safety_margin": PIPELINE_CAPACITY_SAFETY_MARGIN,
            "candidate_node_count": len(records),
            "excluded_nodes": [],
            "assignments": planned,
            "control_only_nodes": [
                node_id for node_id in records if node_id not in node_ids
            ],
            "participating_node_count": len(planned),
            "single_node_full_model_candidates": [],
            "aggregate_only": len(planned) > 1,
            "computed_at": time.time(),
        })

    def _normalize_manual_assignments(self, assignments: list) -> list:
        """补齐手动区间的运行字段，并按节点能力放置 Embedding/LM Head。"""
        normalized = []
        for raw in sorted(assignments, key=lambda item: int(item.get("start_layer", 0))):
            item = dict(raw)
            start = int(item.get("start_layer", 0))
            end = int(item.get("end_layer", 0))
            node = self.nodes.get(item.get("node_id", ""))
            item["start_layer"] = start
            item["end_layer"] = end
            item["layers_count"] = end - start
            item["role"] = item.get("role") or (node.role if node else "client")
            item["score"] = self._compute_node_weight(
                node.device_info if node else {}
            )
            normalized.append(item)
        return self._resequence_assignments(normalized)

    @staticmethod
    def _manual_assignments_are_executable(assignments: list) -> bool:
        """The current runner always executes the master Embedding segment first."""
        masters = [
            item for item in assignments
            if item.get("node_id") == "master" or item.get("role") == "master"
        ]
        return bool(
            len(masters) == 1
            and int(masters[0].get("start_layer", -1)) == 0
            and masters[0].get("has_embedding", False)
        )

    def reset_layer_assignments(self) -> dict:
        """
        清除手动分层覆盖，恢复自动（dynamic）策略。

        仅主节点可调用。

        Returns:
            {"status": "ok", "strategy": "dynamic"}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可重置分层配置"}

        self._runtime_layer_override = None

        # 强制重新计算
        assignments = self.compute_layer_assignment()
        result = {
            "total": self._get_total_model_layers(),
            "strategy": "dynamic",
            "assignments": assignments,
            "computed_at": time.time(),
        }

        logger.info("分层配置已重置为自动策略")
        self.push_layer_config_to_clients()
        return {"status": "ok", "strategy": "dynamic", "assignments": result["assignments"]}

    def override_layer_assignments(self, assignments: list) -> dict:
        """
        手动覆盖分层配置（仅主节点）。

        验证:
          - 所有区间必须连续且完整覆盖 0-24
          - node_id 必须是已注册节点
          - 区间不能重叠

        Args:
            assignments: [{node_id, start_layer, end_layer}]

        Returns:
            {status, message, current_assignments}
        """
        self._require_control_write("cluster.layers.override")
        total_layers = self._get_total_model_layers()

        def invalid(reason_code: str, reason: str) -> dict:
            return {
                "status": "invalid",
                "reason_code": reason_code,
                "reason": reason,
            }

        if self._effective_role() != "master":
            return {
                "status": "denied",
                "reason_code": "layer_override_master_required",
                "reason": "仅主节点可覆盖分层配置",
            }

        # 基本验证
        if not assignments or not isinstance(assignments, list):
            return invalid("layer_assignments_empty", "分层配置不能为空")

        # 收集所有区间，排序验证连续性
        intervals = []
        for a in assignments:
            node_id = a.get("node_id", "")
            start = a.get("start_layer", 0)
            end = a.get("end_layer", 0)

            if node_id not in self.nodes:
                return invalid("layer_assignment_node_unknown", f"未知节点: {node_id}")
            # ★ 2026-09-19：判据由「平台」改为「**能力**」。原实现（Phase 4.1）按
            #   `node_type == "android"` 一律拒绝，理由写「Android 无 PyTorch 推理能力」——
            #   但 **Android 可以跑 llama.cpp/GGUF 引擎，而 llama.cpp 现在也能做层前向**
            #   （`forward_layers_from_hidden` / `forward_layers_to_hidden`）⇒ 原判据**过宽**，
            #   把「有 GGUF 引擎的 Android」也一并挡掉了。详见 `_node_supports_forward_layers`。
            node = self.nodes[node_id]
            if not _node_supports_forward_layers(node):
                return invalid(
                    "layer_assignment_node_unsupported",
                    f"节点 {node_id}（{node.node_type}）不支持层前向传播",
                )
            if start < 0 or end > total_layers or start >= end:
                return invalid(
                    "layer_assignment_range_invalid",
                    f"节点 {node_id} 区间 [{start}, {end}) 无效（范围 0-{total_layers}）",
                )
            intervals.append((start, end, node_id))

        # 排序后验证连续性
        intervals.sort(key=lambda x: x[0])

        covered = 0
        for start, end, node_id in intervals:
            if start != covered:
                return invalid(
                    "layer_assignment_range_discontinuous",
                    f"节点 {node_id} 区间 [{start}, {end}) 不连续（期望从 {covered} 开始）",
                )
            covered = end

        if covered != total_layers:
            return invalid(
                "layer_assignment_coverage_incomplete",
                f"总覆盖范围 {covered} ≠ {total_layers}，分层未完整覆盖",
            )

        master_intervals = [item for item in intervals if item[2] == "master"]
        if len(master_intervals) != 1 or master_intervals[0][0] != 0:
            return invalid(
                "layer_assignment_master_must_be_first",
                "主节点必须且只能承担从 Layer 0 开始的首段",
            )

        normalized_assignments = self._normalize_manual_assignments([
            {"node_id": node_id, "start_layer": start, "end_layer": end}
            for start, end, node_id in intervals
        ])
        self._runtime_layer_override = [dict(item) for item in normalized_assignments]

        # 推送到已连接从节点
        self.push_layer_config_to_clients()

        logger.info(f"分层配置已手动覆盖: {len(assignments)} 个节点")
        return {
            "status": "ok",
            "message": "分层配置已更新（手动模式），已推送至从节点",
            "current_assignments": {
                "total": total_layers,
                "strategy": "manual",
                "assignments": normalized_assignments,
                "computed_at": None,
            },
        }

    def push_layer_config_to_clients(self) -> None:
        with self._layer_config_push_lock:
            self._push_layer_config_to_clients_locked()

    def _task_worker_full_model_ids(self) -> set[str]:
        """Return healthy Task Worker peers that advertise a full model."""
        if (
            self._effective_role() != "master"
            or not TASK_WORKER_EXPERIMENTAL_ENABLED
        ):
            return set()
        try:
            status = self._task_worker_control.status(role="master")
        except Exception:
            return set()
        worker_ids: set[str] = set()
        for worker in status.get("workers", []):
            if not isinstance(worker, dict):
                continue
            if not (
                worker.get("healthy")
                and worker.get("accepted")
                and worker.get("manual_stage_dispatch_enabled")
            ):
                continue
            capabilities = worker.get("capabilities") or {}
            if not isinstance(capabilities, dict):
                continue
            # A worker that advertises an exact layer range is eligible for the
            # layer pipeline even when it also exposes a full-model identity.
            # Only full-model-only workers keep the legacy opt-out priority.
            stage_types = capabilities.get("stage_types", [])
            has_layer_stage = (
                isinstance(stage_types, list)
                and "layer_forward" in stage_types
                and bool(capabilities.get("layer_ranges"))
            )
            if capabilities.get("models") and not has_layer_stage:
                node_id = str(worker.get("node_id", "") or "")
                if node_id:
                    worker_ids.add(node_id)
        return worker_ids

    # ================================================================
    # TCP 消息处理
    # ================================================================

    def _on_tcp_registration_confirmed(self, client_id: str) -> None:
        """Push scheduling state only after the REGISTER ACK is on the wire."""
        if self._effective_role() != "master":
            return
        with self._nodes_lock:
            node = self.nodes.get(client_id)
            if node is None or node.role == NodeRole.MASTER:
                return

        qwen3_release = []
        with self._layer_config_lock:
            qwen3_transaction = self._qwen3_pipeline_dry_run
            if (
                qwen3_transaction is not None
                and qwen3_transaction.network_dispatch
                and qwen3_transaction.phase in {"aborted", "releasing"}
                and client_id in qwen3_transaction.worker_ids
            ):
                qwen3_release = [
                    item for item in qwen3_transaction.release_messages()
                    if item.get("node_id") == client_id
                ]
        if qwen3_release:
            self._dispatch_qwen3_loopback_messages(
                qwen3_release, best_effort=True,
            )

        self.push_layer_config_to_clients()
        self._push_node_list_to_client(client_id)
        self._push_node_update_to_all_clients(
            client_id, "add", self.nodes.get(client_id)
        )

    def _on_tcp_message(self, client_id: str, msg: dict) -> None:
        """
        TCP 消息回调（由 TCPServer 调用）。

        根据消息类型更新节点状态、心跳等。
        """
        msg_type = msg.get("type", "")

        if msg_type == "register":
            data = msg.get("data", {})
            if data.get("node_type") == "pipeline_peer":
                if self._tcp_server:
                    self._tcp_server.confirm_registration(client_id)
                logger.debug("节点间流水线传输连接已认证: %s", client_id)
                return
            # get_client_info 对未知 client_id 返回 None（如从节点收到主节点的
            # register 拒绝回执、或注册与断开竞态），必须兜底为空 dict，
            # 否则回调崩溃会杀死接收循环、触发无限重连
            client_info = (
                self._tcp_server.get_client_info(client_id)
                if self._tcp_server else {}
            ) or {}
            advertised_addr = (
                client_info.get("advertised_addr")
                or data.get("advertised_address")
                or client_info.get("addr", "")
            )
            device_info = dict(data.get("device_info", {}) or {})
            peer_addr = client_info.get("peer_addr", "")
            if peer_addr:
                device_info["tcp_peer_addr"] = peer_addr
            device_info["tcp_advertised_addr"] = advertised_addr
            registered = self.register_node(
                node_id=client_id,
                role=data.get("role", ""),
                address=advertised_addr,
                hostname=data.get("hostname", ""),
                device_info=device_info,
                network_type=data.get("network_type", client_info.get("network_type", "unknown")),
                node_type=data.get("node_type", "pc"),
                model_sha256=data.get("model_sha256", ""),
            )
            if not registered:
                reason = "节点注册被调度器拒绝：容量已满或角色无效"
                logger.warning("event=tcp_register_rejected client_id=%s reason=%s", client_id, reason)
                if self._tcp_server:
                    self._tcp_server.reject_client(client_id, reason)
                return

            # TCPServer sends REGISTER ACK first, then invokes
            # _on_tcp_registration_confirmed.  Keeping all business pushes in
            # that post-ACK callback prevents layer_config from racing the
            # client's registration handshake.

        elif msg_type == "heartbeat":
            if self._effective_role() == "master":
                self._task_worker_control.mark_worker_heartbeat(client_id)
            client_info = (
                self._tcp_server.get_client_info(client_id)
                if self._tcp_server else None
            ) or {}
            with self._nodes_lock:
                if client_id in self.nodes:
                    node = self.nodes[client_id]
                    node.last_heartbeat = time.time()
                    if client_info.get("avg_rtt_ms"):
                        node.avg_rtt_ms = client_info["avg_rtt_ms"]
                        node.last_rtt_ms = client_info.get(
                            "last_rtt_ms", node.last_rtt_ms,
                        )

        elif msg_type == "task_worker":
            self._handle_task_worker_message(client_id, msg)

        elif msg_type == "status_res":
            # 从节点状态上报
            data = msg.get("data", {})
            profile_changed = False
            with self._nodes_lock:
                if client_id in self.nodes:
                    if "state" in data:
                        try:
                            self.nodes[client_id].state = NodeState(data["state"])
                        except ValueError:
                            pass
                    device_info = data.get("device_info")
                    if isinstance(device_info, dict) and device_info:
                        updated_info = dict(device_info)
                        previous_info = dict(self.nodes[client_id].device_info or {})
                        for key in ("tcp_peer_addr", "tcp_advertised_addr"):
                            if key in previous_info and key not in updated_info:
                                updated_info[key] = previous_info[key]
                        profile_changed = previous_info != updated_info
                        self.nodes[client_id].device_info = updated_info
            if profile_changed and self._effective_role() == "master":
                logger.info("从节点设备画像已更新，重新计算分层: node=%s", client_id)
                try:
                    self.push_layer_config_to_clients()
                except Exception as e:
                    logger.warning("设备画像更新后重新下发分层失败: %s", e, exc_info=True)

        elif msg_type == "error":
            data = msg.get("data", {})
            with self._nodes_lock:
                if client_id in self.nodes:
                    self.nodes[client_id].error_count += 1
            logger.error(f"节点 {client_id} 上报错误: {data.get('message', 'unknown')}")

        elif msg_type == "infer_forward":
            # 从节点转发推理请求给主节点
            self.handle_infer_forward(client_id, msg)

        elif msg_type == "infer_cancel":
            data = msg.get("data", {})
            forward_request_id = str(data.get("forward_request_id", ""))
            with self._forward_cancel_lock:
                cancel_event = self._forward_cancel_events.get(
                    (client_id, forward_request_id)
                )
            if cancel_event is not None:
                cancel_event.set()
                logger.info(
                    "收到转发推理取消: client=%s request=%s",
                    client_id, forward_request_id,
                )

        elif msg_type == "infer_result":
            # 主节点返回推理结果给从节点
            data = msg.get("data", {})
            forward_request_id = str(data.get("forward_request_id", ""))
            result_entry = {
                "task_id": data.get("task_id", ""),
                "forward_request_id": forward_request_id,
                "status": data.get("status", "ok"),
                "content": data.get("content", ""),
                "metrics": data.get("metrics", {}),
                "error": data.get("error", ""),
                "thinking_content": data.get("thinking_content"),
                "followups": data.get("followups", []),
            }
            with self._client_pending_lock:
                if not forward_request_id and len(self._client_pending_events) == 1:
                    # 兼容尚未升级的主节点；并发时绝不猜测结果归属。
                    forward_request_id = next(iter(self._client_pending_events))
                    result_entry["forward_request_id"] = forward_request_id
                event = self._client_pending_events.get(forward_request_id)
                if event is not None:
                    self._client_pending_results[forward_request_id] = result_entry
                    event.set()
            if event is None:
                logger.warning(
                    "丢弃无等待者的迟到推理结果: task=%s request=%s",
                    data.get("task_id", ""), forward_request_id or "-",
                )
            else:
                logger.info(
                    "收到推理结果: task=%s request=%s len=%d",
                    data.get("task_id", ""), forward_request_id,
                    len(data.get("content", "")),
                )

        elif msg_type == "role_transfer":
            # 从节点收到主节点转让通知
            self._handle_role_transfer(client_id, msg)

        elif msg_type == "role_transfer_ack":
            # 主节点收到从节点的转让确认
            self._handle_role_transfer_ack(client_id, msg)

        elif msg_type == "spare_master_designate":
            # 从节点收到备用主节点指定通知
            self._handle_spare_master_designate(client_id, msg)

        elif msg_type == "spare_master_designate_ack":
            # 主节点收到从节点的备用指定确认
            self._handle_spare_master_designate_ack(client_id, msg)

        elif msg_type == "spare_master_activate":
            # 备用主节点收到激活（暂代主节点职责）通知
            self._handle_spare_master_activate(client_id, msg)

        elif msg_type == "spare_master_activate_ack":
            # 主节点收到备用主节点的激活确认
            self._handle_spare_master_activate_ack(client_id, msg)

        elif msg_type == "spare_master_deactivate":
            # 备用主节点收到新主节点的接管通知
            self._handle_spare_master_deactivate(client_id, msg)

        elif msg_type == "node_list_sync":
            # ---- 从节点收到主节点推送的全量节点列表 ----
            data = msg.get("data", {})
            if data.get("request") == "node_list":
                # 从节点请求节点列表 → 主节点响应
                self._push_node_list_to_client(client_id)
            else:
                # 从节点接收全量节点列表
                nodes_data = data.get("nodes", [])
                if nodes_data:
                    self._apply_node_list_sync(nodes_data)
                    logger.info(
                        f"📋 收到主节点推送的全量节点列表: {len(nodes_data)} 个节点"
                    )

        elif msg_type == "node_update":
            # ---- 从节点收到单节点变更通知 ----
            data = msg.get("data", {})
            action = data.get("action", "")
            node_data = data.get("node", {})
            if action and node_data:
                self._apply_node_update(action, node_data)
                logger.info(
                    f"📋 节点变更: {action} {node_data.get('node_id', '?')}"
                )

        elif msg_type == "layer_forward":
            # ---- 从节点：收到主节点的层前向传播指令 ----
            threading.Thread(
                target=self._handle_layer_forward,
                args=(client_id, msg),
                name=f"layer-forward-{msg.get('data', {}).get('task_id', 'unknown')}",
                daemon=True,
            ).start()

        elif msg_type == "layer_result":
            # ---- 主节点：收到从节点的层前向传播结果 ----
            self._handle_layer_result(client_id, msg)

        elif msg_type == "chain_forward":
            # ---- 从节点：收到另一从节点的链式直连转发（P2 优化）----
            threading.Thread(
                target=self._handle_chain_forward,
                args=(client_id, msg),
                name=f"chain-forward-{msg.get('data', {}).get('task_id', 'unknown')}",
                daemon=True,
            ).start()

        elif msg_type == "chain_forward_ack":
            # ---- 主节点：收到链式转发每跳 ACK / 错误回报 ----
            self._handle_chain_forward_ack(client_id, msg)

        elif msg_type == "pipeline_done":
            # ---- 从节点：流水线任务完成，清理 KV 缓存 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                    self._local_pipeline_steps.pop(task_id, None)
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
                        logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")
                self._record_local_pipeline_participation(task_id, success=True)
                self._finish_local_pipeline_task(task_id)

        elif msg_type == "pipeline_abort":
            # ---- 从节点/主节点：流水线任务取消 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._layer_config_lock:
                    participated = task_id in self._active_pipeline_task_ids
                    self._local_pipeline_steps.pop(task_id, None)
                self._mark_local_pipeline_cancelled(task_id)
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
                if participated and data.get("count_error", True):
                    self._record_local_pipeline_participation(task_id, success=False)
                self._finish_local_pipeline_task(task_id)
            logger.warning(f"⚠️ 流水线任务 {task_id} 已取消")

        elif msg_type == "pipeline_pause":
            # ---- 从节点/主节点：流水线暂停（二期协同抢占，协议预留）----
            logger.info(f"⏸️ 收到 PIPELINE_PAUSE: client={client_id}")

        elif msg_type == "pipeline_resume":
            # ---- 从节点/主节点：流水线恢复（二期协同抢占，协议预留）----
            logger.info(f"▶️ 收到 PIPELINE_RESUME: client={client_id}")

        elif msg_type == "layer_config":
            # ---- 从节点：收到主节点推送的分层配置 ----
            data = msg.get("data", {})
            self._schedule_layer_config(client_id, data)

        elif msg_type == "layer_config_ack":
            # ---- 主节点：从节点完成模型层加载后的确认 ----
            self._handle_layer_config_ack(client_id, msg)

        elif msg_type == "qwen3_pipeline_dry_run":
            self._handle_qwen3_loopback_request(client_id, msg)

        elif msg_type == "qwen3_pipeline_dry_run_ack":
            self._handle_qwen3_loopback_ack(client_id, msg)

        elif msg_type == "layer_worker_opt_out":
            # ---- 主节点：从节点明确切换为本地推理，不再分配模型层 ----
            if self._effective_role() == "master":
                self._handle_layer_worker_opt_out(client_id, msg)
            else:
                logger.warning("非主节点忽略分层 worker 退出请求: %s", client_id)

        elif msg_type == "layer_worker_opt_in":
            # ---- 主节点：从节点重新允许接收 PyTorch 分层配置 ----
            if self._effective_role() == "master":
                self._handle_layer_worker_opt_in(client_id, msg)
            else:
                logger.warning("非主节点忽略分层 worker 加入请求: %s", client_id)

        elif msg_type == "log_request":
            # ---- L5: 主节点拉取从节点最近日志 ----
            data = msg.get("data", {})
            limit = data.get("limit", 100)
            try:
                # Keep this branch self-contained: log aggregation can be
                # invoked independently of the inference message handlers.
                from transport_port import MessageType

                callbacks = self._require_callbacks()
                entries, _ = callbacks.snapshot_recent_logs()
                filtered = callbacks.filter_recent_logs(
                    entries,
                    level=data.get("level", ""),
                    name=data.get("name", ""),
                    node_id=data.get("node_id", ""),
                    request_id=data.get("request_id", ""),
                )
                result_entries = filtered[-limit:]
                response = {
                    "node_id": self.get_effective_node_id(),
                    "logs": result_entries,
                    "count": len(result_entries),
                    "matched": len(filtered),
                    "buffer_size": len(entries),
                }
                # 从节点使用 _tcp_client.send_data 回复主节点
                tcp_client = getattr(self, '_tcp_client', None)
                if tcp_client:
                    tcp_client.send_data(response, MessageType.LOG_RESPONSE)
                    logger.debug(
                        "event=log_aggregation_sent node_id=%s count=%d requester=%s",
                        self.get_effective_node_id(), len(result_entries), client_id,
                    )
                else:
                    logger.warning(
                        "event=log_aggregation_send_failed node_id=%s reason=tcp_client_none",
                        self.get_effective_node_id(),
                    )
            except Exception as e:
                logger.warning(
                    "event=log_aggregation_error node_id=%s error=%s",
                    self.get_effective_node_id(), str(e)[:200],
                )

        elif msg_type == "log_response":
            # ---- L5: 主节点收到从节点日志响应 ----
            data = msg.get("data", {})
            worker_node_id = data.get("node_id", client_id)
            with self._pending_log_lock:
                self._pending_log_responses[worker_node_id] = data
                _evt = self._pending_log_events.get(worker_node_id)
                if _evt is not None:
                    _evt.set()

        else:
            logger.debug(f"未知消息类型: {msg_type}, client={client_id}")

    def _set_pipeline_result_error(self, task_id: str, node_id: str,
                                   error: str, step: int = -1) -> None:
        """主节点侧：写入节点错误并唤醒等待该节点结果的流水线线程。"""
        if not task_id or not node_id:
            return
        key = f"{task_id}:{node_id}"
        with self._pipeline_lock:
            if task_id not in self._pipeline_active_tasks:
                return
            self._pipeline_results[key] = {
                "task_id": task_id,
                "node_id": node_id,
                "error": error,
                "step": step,
            }
            event = self._pipeline_events.get(key)
            if event is not None:
                event.set()

    def _on_tcp_disconnect(self, client_id: str) -> None:
        """TCP 断连回调（由 TCPServer 调用）"""
        abort_details = None
        qwen3_disconnect = None
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                transaction
                and transaction.get("phase") in {
                    "preparing", "committing_local", "committing",
                }
                and client_id in set(transaction.get("worker_ids", set()))
            ):
                abort_details = (
                    str(transaction.get("config_id", "")),
                    "pipeline_worker_disconnected",
                    f"worker {client_id} disconnected during transaction",
                )
            qwen3_transaction = self._qwen3_pipeline_dry_run
            if qwen3_transaction is not None:
                qwen3_disconnect = qwen3_transaction.disconnect(client_id)
                if qwen3_disconnect is not None and qwen3_transaction.network_dispatch:
                    worker = self._qwen3_loopback_workers.get(
                        qwen3_transaction.contract["contract_sha256"], None,
                    )
                    if worker is not None:
                        worker.abort_sidecar_sessions()
                    self._qwen3_loopback_workers.pop(
                        qwen3_transaction.contract["contract_sha256"], None,
                    )
        if abort_details is not None:
            self._abort_pipeline_load_transaction(*abort_details)
        if qwen3_disconnect is not None:
            logger.warning(
                "Qwen3 dry-run 因节点断线中止: node=%s contract=%s",
                client_id,
                self.get_qwen3_pipeline_dry_run_status().get(
                    "contract_sha256", ""
                ),
            )
            outbound = qwen3_disconnect.get("outbound", [])
            if outbound:
                self._dispatch_qwen3_loopback_messages(
                    outbound, best_effort=True,
                )
        self._task_worker_control.disconnect_worker(client_id)
        with self._task_worker_stage_lock:
            remote_provider = self._remote_task_worker_providers.get(client_id)
        if remote_provider is not None:
            remote_provider.notify_disconnect()
        with self._forward_cancel_lock:
            client_cancellations = [
                event for (owner_id, _), event
                in self._forward_cancel_events.items()
                if owner_id == client_id
            ]
        for event in client_cancellations:
            event.set()
        self._fail_pending_pipeline_results_for_node(
            client_id, f"节点 {client_id} TCP 连接已断开"
        )
        with self._nodes_lock:
            old_state = None
            if client_id in self.nodes:
                old_state = self.nodes[client_id].state.value
                self.nodes[client_id].state = NodeState.OFFLINE
                logger.info(
                    f"🔌 节点 {client_id} TCP 断开，已标记 offline "
                    f"(old_state={old_state}, role={self._effective_role()})"
                )
            else:
                logger.debug(f"未知节点 {client_id} TCP 断开，跳过状态标记")
            # 快照数据后释放锁，避免持锁进行 TCP 发送
            need_push = (self._effective_role() == "master")
            node_info = self.nodes.get(client_id)

        # L5: 清理断连节点的待处理日志聚合状态（避免残留 Event 导致超时等待）
        with self._pending_log_lock:
            _evt = self._pending_log_events.pop(client_id, None)
            if _evt is not None:
                _evt.set()  # 唤醒等待线程（将收到空结果）
            self._pending_log_responses.pop(client_id, None)

        if need_push:
            self._push_node_update_to_all_clients(
                client_id, "update", node_info
            )
        self.deregister_node(client_id)
        reshard = self._stage_pipeline_reshard_after_disconnect(client_id)
        if need_push:
            if reshard and reshard.get("accepted"):
                self.request_authoritative_layer_sync(require_distributed=True)
            else:
                self.push_layer_config_to_clients()

    def _on_master_connection_lost(self, source_client=None) -> None:
        """Worker-side cleanup when PIPELINE_DONE/ABORT can no longer arrive."""
        if (source_client is not None
                and getattr(self, "_tcp_client", None) is not source_client):
            logger.debug("忽略旧主节点连接的迟到断连回调")
            return

        self._task_worker_control.disconnect_coordinator()
        with self._task_worker_stage_lock:
            for active in self._task_worker_active_attempts.values():
                active.cancel_reason = "coordinator_disconnected"
                active.cancel_event.set()

        with self._client_pending_lock:
            for request_id, event in self._client_pending_events.items():
                self._client_pending_results[request_id] = {
                    "forward_request_id": request_id,
                    "status": "error",
                    "content": "",
                    "metrics": {},
                    "error": "与主节点的连接已断开",
                }
                event.set()
        with self._chain_clients_lock:
            chain_clients = list(self._chain_clients.values())
            self._chain_clients.clear()
        for chain_client in chain_clients:
            try:
                chain_client.disconnect()
            except Exception:
                logger.debug("主节点断线时关闭链式连接失败", exc_info=True)
        with self._kv_cache_lock:
            active_tasks = list(self._kv_cache)
            self._kv_cache.clear()
        with self._layer_config_lock:
            active_tasks.extend(self._active_pipeline_task_ids)
            active_tasks = list(dict.fromkeys(active_tasks))
            self._active_pipeline_task_ids.clear()
            self._local_pipeline_steps.clear()
            self._pending_layer_config = None
            self._active_layer_config = None
            self._last_layer_config_ack_payload = None
            self._pipeline_worker_reserved = False
            self._layer_config_inflight.clear()
            self._layer_config_receive_sequence += 1
            self._latest_layer_config_receive_sequence = (
                self._layer_config_receive_sequence
            )
            self._latest_layer_config_generation = 0
        for task_id in active_tasks:
            self._record_local_pipeline_participation(task_id, success=False)
        if active_tasks:
            logger.warning(
                "主节点连接中断，已清理 %d 个本地流水线任务",
                len(active_tasks),
            )

    # ================================================================
    # 节点列表同步（主 → 从）
    # ================================================================

    # ================================================================
    # 角色转让 — 主节点身份转移
    # ================================================================

    # ================================================================
    # 备用主节点管理
    # ================================================================

    # ---- 备用主节点：激活（暂代） / 接管（退出暂代） ----

    # ---- 新主节点启动：向备用主节点发送接管通知 ----

    # ================================================================
    # 任务调度
    # ================================================================

    def start_infer_task(self, prompt: str, request_id: str = None) -> str:
        """
        启动一轮完整推理任务。

        Args:
            prompt: 用户输入文本
            request_id: API 请求 ID（L5: 链路追踪）

        Returns:
            task_id: 任务唯一标识

        """
        # 节点就绪与降级由 run_pipeline_safe() 统一决定。这里不能先把
        # worker 标成 BUSY，否则 readiness 会把 BUSY 误判为离线并强制回退。
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = InferenceTask(task_id=task_id, prompt=prompt)
        task.state = "running"
        task.start_time = time.time()

        with self._task_lock:
            self._current_task = task
            self._infer_tasks[task_id] = task

        logger.info(
            "event=infer_task_start task_id=%s request_id=%s prompt_len=%d",
            task_id, request_id or "-", len(prompt),
        )

        # TODO: 触发流水线推理流程
        # 1. 主节点 Prefill → 中间特征 → TCP 发送至 client1
        # 2. client1 计算 → 转发特征至 client2
        # 3. client2 计算 → Decode 循环 → 回传结果

        return task_id

    def complete_infer_task(self, task_id: str, result: str,
                            metrics: dict = None) -> None:
        """完成指定的转发任务；并发任务不能清除彼此的状态。"""
        completed = None
        with self._task_lock:
            current = self._infer_tasks.pop(task_id, None)
            if current is None:
                return
            current.state = "done"
            current.end_time = time.time()
            current.result = result
            current.metrics = metrics or {}
            completed = current
            if self._current_task is current:
                self._current_task = None

        logger.info(
            "推理任务完成: %s, 耗时=%.2fs, 结果长度=%d",
            task_id, completed.end_time - completed.start_time, len(result),
        )
        if self.on_task_complete:
            self.on_task_complete(completed)

    def fail_infer_task(self, task_id: str, error_msg: str) -> None:
        """失败指定的转发任务，不影响稍后启动的其他任务。"""
        with self._task_lock:
            current = self._infer_tasks.pop(task_id, None)
            if current is None:
                return
            current.state = "error"
            current.end_time = time.time()
            current.error_msg = error_msg
            if self._current_task is current:
                self._current_task = None
        logger.error("推理任务异常: %s - %s", task_id, error_msg)

    def stop_infer_task(self) -> None:
        """强制停止推理、重置流水线"""
        with self._task_lock:
            if self._current_task:
                self._current_task.state = "done"
                self._current_task.end_time = time.time()
                logger.info(f"推理任务已停止: {self._current_task.task_id}")
                self._current_task = None

        # 恢复所有节点为空闲
        with self._nodes_lock:
            for nid in self.nodes:
                if self.nodes[nid].state == NodeState.BUSY:
                    self.update_node_state(nid, NodeState.ONLINE)

        # TODO: 发送 TASK_STOP 指令给所有从节点
        # TODO: 清空所有节点 KV 缓存

    def on_task_finished(self, result: str, metrics: dict = None) -> None:
        """
        任务完成回调。

        NOTE: 此方法当前无调用方，为死代码（dead code），保留供未来流水线任务完成通知使用。
        """
        with self._task_lock:
            task_id = self._current_task.task_id if self._current_task else ""
        if task_id:
            self.complete_infer_task(task_id, result, metrics)

    def on_task_error(self, error_msg: str) -> None:
        """
        任务异常回调。

        Args:
            error_msg: 错误描述
        """
        with self._task_lock:
            task_id = self._current_task.task_id if self._current_task else ""
        if task_id:
            self.fail_infer_task(task_id, error_msg)

    # ================================================================
    # 状态查询
    # ================================================================

    # L5: 多节点日志聚合

            # 保持本机服务可用，但绝不覆盖已有身份记录。

    # ================================================================
    # 分布式推理开关
    # ================================================================

    # ================================================================
    # 任务转发（从节点 → 主节点）
    # ================================================================

    # ================================================================
    # 分布式流水线推理（阶段 3：流水线调度引擎）
    # ================================================================

    # ================================================================
    # 协同抢占辅助方法 (Phase 2)
    # ================================================================

    # ================================================================
    # 流水线请求队列集成（Phase 4 — 多请求排队）
    # ================================================================

    # ================================================================
    # 从节点：主节点健康监控 + 自动重连
    # ================================================================
