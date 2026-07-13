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
import logging
import os
import threading
import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass, field

# PyTorch 可用性检查（分布式推理按需导入，避免 llama.cpp 模式下硬依赖）
try:
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:
    torch = None  # type: ignore[assignment]

from config import (
    RUN_MODE, HEARTBEAT_INTERVAL,
    SERVER_IP, SERVER_PORT,
    NODE_ROLE, NODE_ID, MAX_NODES,
    MASTER_DOWN_EMAIL_TIMEOUT,
    PIPELINE_TIMEOUT, PIPELINE_STEP_TIMEOUT, PIPELINE_QUEUE_POLL_INTERVAL,
    PIPELINE_QUEUE_MAX_SIZE, PIPELINE_QUEUE_RESULT_TTL,
    PIPELINE_SCHEDULING_STRATEGY,
    PIPELINE_Q0_MAX_TOKENS, PIPELINE_Q1_MAX_TOKENS,
    PIPELINE_AGING_Q1_TO_Q0_SECONDS, PIPELINE_AGING_Q2_TO_Q1_SECONDS,
    PIPELINE_AGING_MAX_WAIT_SECONDS,
    PIPELINE_PREEMPT_ENABLED,
    PIPELINE_PREEMPT_MIN_INTERVAL,       # 两次抢占最小间隔（防抖动）
    PIPELINE_PREEMPT_MIN_TOKENS,         # 至少生成 N token 后才接受抢占
    PIPELINE_PREEMPT_MAX_OVERHEAD_MS,    # checkpoint 超限自动禁用
)

logger = logging.getLogger(__name__)

ANDROID_HTTP_CLIENT_TIMEOUT_SECONDS = 120


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
        import config as cfg
        return getattr(cfg, "NODE_ID", NODE_ID)
    except Exception:
        return NODE_ID


def _sync_runtime_node_config(node_id: str = None, node_role: str = None) -> None:
    """Keep imported scheduler constants aligned with runtime config mutations."""
    global NODE_ID, NODE_ROLE
    try:
        import config as cfg
    except Exception:
        cfg = None

    if node_id:
        NODE_ID = str(node_id)
        if cfg is not None:
            cfg.NODE_ID = NODE_ID
    if node_role:
        NODE_ROLE = str(node_role)
        if cfg is not None:
            cfg.NODE_ROLE = NODE_ROLE

# 数据库模块（延迟导入，避免 psycopg2 未安装时直接崩溃）
_db = None
_db_available = False


def _get_db():
    """获取数据库模块，首次失败后缓存不可用状态"""
    global _db, _db_available
    if _db is not None:
        return _db
    if _db_available is False and _db is None:
        try:
            from db import get_pool
            get_pool()  # 预热连接池
            import db as _db_mod
            _db = _db_mod
            _db_available = True
            logger.info("数据库已连接，节点管理将持久化到 PostgreSQL")
        except Exception as e:
            _db = None
            _db_available = False
            logger.warning(f"数据库不可用，使用内存模式: {e}")
    return _db


class NodeState(str, Enum):
    """节点状态枚举"""
    ONLINE = "online"       # 在线空闲
    BUSY = "busy"           # 推理中
    OFFLINE = "offline"     # 离线/断连
    ERROR = "error"         # 异常


class NodeRole(str, Enum):
    """节点角色（字符串枚举，便于比较）"""
    MASTER = "master"
    CLIENT = "client"

@dataclass
class NodeInfo:
    """节点信息（分布式模式下通过 TCP 注册填充）"""
    node_id: str
    role: str                      # 节点角色: "master" | "client"
    node_type: str = "pc"          # 设备平台: "pc" | "android"
    state: NodeState = NodeState.OFFLINE
    address: str = ""              # "ip:port" 字符串
    hostname: str = ""             # 客户端主机名
    device_info: dict = field(default_factory=dict)  # 客户端设备信息
    network_type: str = "unknown"  # 网络连接类型: wifi | ethernet | unknown
    connected_at: float = 0.0      # 连接/注册时间
    last_heartbeat: float = 0.0    # 上次心跳时间
    avg_rtt_ms: float = 0.0        # 滑动平均 RTT（指数加权，仅从节点有效）
    last_rtt_ms: float = 0.0       # 最近一次 RTT
    task_count: int = 0            # 已完成任务数
    error_count: int = 0           # 错误计数
    model_sha256: str = ""         # 模型 SHA256 校验值（阶段 7）

    def is_available(self) -> bool:
        return self.state == NodeState.ONLINE

    def to_dict(self) -> dict:
        """转为可序列化的字典"""
        return {
            "node_id": self.node_id,
            "role": self.role,
            "node_type": self.node_type,
            "state": self.state.value,
            "address": self.address,
            "hostname": self.hostname,
            "device_info": self.device_info,
            "network_type": self.network_type,
            "connected_at": self.connected_at,
            "last_heartbeat": self.last_heartbeat,
            "avg_rtt_ms": round(self.avg_rtt_ms, 1),
            "last_rtt_ms": round(self.last_rtt_ms, 1),
            "task_count": self.task_count,
            "error_count": self.error_count,
            "model_sha256": self.model_sha256,
            "is_available": self.is_available(),
        }


@dataclass
class InferenceTask:
    """单个推理任务"""
    task_id: str
    prompt: str
    state: str = "pending"         # pending | running | done | error
    start_time: float = 0.0
    end_time: float = 0.0
    result: Optional[str] = None
    error_msg: Optional[str] = None
    metrics: dict = field(default_factory=dict)  # 性能指标


@dataclass
class QueueTask:
    """
    流水线队列中的单个任务 — MLFQ 调度单元。

    字段映射:
    - priority_level: 0=Q0(交互,≤128tk), 1=Q1(普通,≤512tk), 2=Q2(批量,>512tk)
    - original_level: 入队时的初始级别（用于展示老化状态）
    - created_at: 入队时间戳（用于老化提升计算）
    """

    task_id: str
    prompt: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    session_id: Optional[str] = None
    request_id: Optional[str] = None   # L5: API 请求 ID，用于链路追踪
    priority_level: int = 1       # 0=Q0(交互), 1=Q1(普通), 2=Q2(批量)
    created_at: float = field(default_factory=time.time)
    original_level: int = 1       # 入队时的初始级别
    # 保留原始 kwargs 以便透传给 process_fn（如 _stream_callback, _queue_timeout）
    _extra_kwargs: dict = field(default_factory=dict)

    def estimated_duration_seconds(self) -> float:
        """预估剩余推理时间（用于 SJF 排序）。假设 ~50ms/token。"""
        return self.max_new_tokens * 0.05

    def wait_seconds(self) -> float:
        """已等待秒数。"""
        return time.time() - self.created_at

    def to_task_data(self) -> dict:
        """重建 **task_data 字典以调用 process_fn（向后兼容）。"""
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "session_id": self.session_id,
            **self._extra_kwargs,
        }

    def to_dict(self) -> dict:
        """序列化为前端展示。"""
        return {
            "task_id": self.task_id,
            "priority_level": self.priority_level,
            "original_level": self.original_level,
            "max_new_tokens": self.max_new_tokens,
            "wait_seconds": round(self.wait_seconds(), 1),
            "estimated_duration_s": round(self.estimated_duration_seconds(), 1),
            "is_aged": self.priority_level != self.original_level,
            "created_at": self.created_at,
        }


class PreemptState:
    """
    被抢占任务的执行状态快照。

    在 decode 步边界保存 Q1/Q2 任务的所有局部状态，供 Q0 完成后恢复。
    KV cache 按 task_id 保留在各节点，不需 GPU↔CPU checkpoint。
    """
    __slots__ = (
        "task_id", "generated_ids", "full_input_ids", "current_step",
        "max_new_tokens", "temperature", "top_p", "prompt",
        "pipeline_nodes", "first_node_id", "_stream_callback",
    )

    def __init__(self, task_id: str, generated_ids: list,
                 full_input_ids, current_step: int,
                 max_new_tokens: int, temperature: float, top_p: float,
                 prompt: str, pipeline_nodes: list, first_node_id: str,
                 _stream_callback=None):
        self.task_id = task_id
        self.generated_ids = list(generated_ids)       # shallow copy
        self.full_input_ids = full_input_ids            # Tensor 引用（只读）
        self.current_step = current_step
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt = prompt
        self.pipeline_nodes = pipeline_nodes
        self.first_node_id = first_node_id
        self._stream_callback = _stream_callback


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
            max_tokens = task_data.get("max_new_tokens", 512)
            request_id = task_data.get("request_id")   # L5: API request_id 链路追踪
            priority_level = self._classify(max_tokens)
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
            )

            # 按级别入队
            self._get_queue(priority_level).append(task)
            self._events[task_id] = threading.Event()
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

    def wait_for_result(self, task_id: str, timeout: float = 120.0) -> dict:
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

        if event.wait(timeout=timeout):
            with self._lock:
                result = self._results.get(task_id, {"status": "unknown"})
                return dict(result)
        else:
            return {"status": "timeout", "error": f"任务 {task_id} 超时 ({timeout}s)"}

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
        执行中的任务: 不支持中途取消（需通过 abort 协议），返回 False。
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
            # 如果是当前执行中的任务，无法取消
            if self._current_task_id == task_id:
                logger.warning(f"无法取消执行中的任务: {task_id}")
                return False
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
                result = self._process_fn(**task_data)
                elapsed = time.time() - t_start
                with self._lock:
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


class Scheduler:
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

    def __init__(self):
        self.nodes: dict[str, NodeInfo] = {}
        self._current_task: Optional[InferenceTask] = None
        self._task_lock = threading.Lock()
        self._running = False
        self.on_task_complete: Optional[Callable] = None

        # TCP 服务端（分布式模式下启动）
        self._tcp_server = None  # 延迟导入，避免循环依赖

        # 从节点：等待主节点推理结果
        self._client_pending_results: dict = {}

        # 流水线推理状态（主节点侧）
        self._pipeline_results: dict = {}       # key → result data
        self._pipeline_events: dict = {}        # key → threading.Event
        self._chain_ack_state: dict = {}        # task_id → step → node_id → ack/error
        self._pipeline_lock = threading.Lock()
        self._nodes_lock = threading.RLock()    # Phase 2.1: 保护 self.nodes 并发读写（可重入）
        self._kv_cache_lock = threading.Lock()   # Phase 2.2: 保护 _kv_cache 并发读写
        self._kv_cache: dict = {}               # task_id → past_key_values（本节点层范围的 KV cache）
        self._layer_config_pushed: set = set()  # 已成功推送层配置的从节点 ID（就绪检查用）
        self._pipeline_accounted_tasks: set = set()  # 主节点侧：已完成记账的流水线任务
        self._local_pipeline_counted_tasks: set = set()  # 从节点侧：已本地计数的流水线任务
        self._local_pipeline_error_tasks: set = set()    # 从节点侧：已本地计错的流水线任务
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

        # 最大节点数（可动态调整）
        self._max_nodes: int = MAX_NODES

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

    # ================================================================
    # 启动 / 停止
    # ================================================================

    def start(self, host: str = None, port: int = None) -> None:
        """
        启动调度器。

        初始化节点状态；若为分布式模式，启动 TCP 服务端监听。

        主节点启动后自动:
        - 检测局域网 IP（非 192.168.x.x 占位符）
        - 将 IP:Port 写入数据库 cluster_config 表
        - 周期性刷新数据库心跳（30s），供从节点自动发现

        Args:
            host: TCP 监听地址（默认 0.0.0.0，接受所有接口连接）
            port: TCP 监听端口（默认 config.SERVER_PORT）
        """
        self.init_nodes()
        self._running = True

        # 存储检测到的局域网 IP 和 MAC 地址
        self._lan_ip: str = ""
        self._mac_addresses: list[str] = []
        self._master_identity_verified: bool = False
        self._master_identity_reason: str = ""

        if RUN_MODE == "distributed":
            from tcp_comm import TCPServer, detect_lan_ip, get_mac_addresses

            # 绑定到 0.0.0.0 接受所有接口连接（而非占位符 192.168.x.x）
            bind_host = host or "0.0.0.0"
            actual_port = port or SERVER_PORT

            self._tcp_server = TCPServer(bind_host, actual_port)
            self._tcp_server.start(
                on_message=self._on_tcp_message,
                on_disconnect=self._on_tcp_disconnect,
            )

            # 检测实际局域网 IP 和 MAC 地址
            if NODE_ROLE == "master":
                self._lan_ip = detect_lan_ip()
                self._mac_addresses = get_mac_addresses()
                logger.info(f"调度器已启动（分布式模式），监听 {bind_host}:{actual_port}，局域网 IP: {self._lan_ip}，MAC: {self._mac_addresses}")

                # 验证主节点身份（MAC 匹配）
                self._verify_master_identity()

                # ★ MAC 不匹配时的处理策略
                if self._master_identity_reason == "mac_mismatch":
                    # 尝试在数据库中发现真正的主节点
                    discovery = self.discover_master()
                    if discovery.get("found"):
                        # 数据库中存在真正的主节点记录 → 自动切换为从节点
                        stale_note = "（心跳过期，IP 可能已变更）" if discovery.get("stale") else ""
                        logger.warning(
                            f"⛔ 主节点身份验证失败！本机 MAC 与数据库中记录不匹配。\n"
                            f"   数据库中存在真正的主节点 ({discovery['master_host']}:{discovery['master_port']}){stale_note}，\n"
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
                        # 数据库中没有主节点记录 → 可能是首次配置错误
                        logger.error(
                            f"⛔ 主节点身份验证失败 — 拒绝注册到数据库！\n"
                            f"   本机 MAC 与数据库中记录不匹配，且未发现其他主节点。\n"
                            f"   如需更换主节点机器，请先在原主节点的后台管理中"
                            f"使用「重置主节点身份」功能，或手动清除数据库中的 MAC 记录。\n"
                            f"   当前将以单机模式运行（不写入 DB 注册信息）。"
                        )
                else:
                    # 自动注册到数据库，供从节点发现
                    self._register_master_in_db()
                    # 启动数据库心跳刷新线程
                    self._start_master_db_heartbeat()
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

        # 启动流水线请求队列（仅主节点，FIFO 串行）
        if self._effective_role() == "master":
            self.pipeline_queue.start(process_fn=self._process_queued_pipeline_task)
            logger.info("流水线请求队列已就绪")

    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        self.pipeline_queue.stop()
        if self._tcp_server:
            self._tcp_server.stop()
        logger.info("调度器已停止")

    # ================================================================
    # 节点管理
    # ================================================================

    def _effective_role(self) -> str:
        """
        返回当前节点的有效角色。

        正常情况返回 config.NODE_ROLE；若 MAC 不匹配时自动切换到
        client 模式，则返回 "client"（通过 _role_override 覆盖）。
        """
        try:
            import config as cfg
            configured_role = getattr(cfg, "NODE_ROLE", NODE_ROLE)
        except Exception:
            configured_role = NODE_ROLE
        return getattr(self, '_role_override', None) or configured_role

    def init_nodes(self) -> None:
        """
        初始化节点状态。
        - 主节点：仅创建 master 自身，从节点通过 TCP 注册动态加入（不再预创建空槽位）
        - 从节点：创建自身记录，等待用户操作连接主节点
        - 优先从数据库恢复已注册节点
        """
        effective_role = self._effective_role()
        db = _get_db()
        db_nodes = {}
        if db and _db_available:
            try:
                db_nodes = {n["node_id"]: n for n in db.get_all_nodes()}
                logger.info(f"从数据库恢复 {len(db_nodes)} 个节点记录")
            except Exception as e:
                logger.warning(f"数据库读取失败，使用默认初始化: {e}")

        # ---- 主节点模式：仅创建 master，不预创建 client 空位 ----
        if effective_role == "master":
            now_ts = time.time()
            if "master" in db_nodes:
                self.nodes["master"] = self._node_from_db(db_nodes["master"])
                # 确保主节点上线时时间戳正确
                self.nodes["master"].state = NodeState.ONLINE
                self.nodes["master"].connected_at = self.nodes["master"].connected_at or now_ts
                self.nodes["master"].last_heartbeat = now_ts
            else:
                self.nodes["master"] = NodeInfo(
                    node_id="master", role=NodeRole.MASTER,
                    state=NodeState.ONLINE,
                    hostname="localhost",
                    network_type="localhost",
                    connected_at=now_ts,
                    last_heartbeat=now_ts,
                )

            # 从 DB 恢复所有之前注册过的从节点
            # 跳过幽灵节点：从未连接过、无地址、无主机名（旧代码预创建的空槽位）
            for nid, row in db_nodes.items():
                if nid == "master":
                    continue
                node = self._node_from_db(row)
                if (not node.address and not node.hostname
                        and not node.connected_at and not node.last_heartbeat):
                    logger.info(f"  跳过幽灵节点: {nid}（无连接记录），从数据库清理")
                    try:
                        db.delete_node(nid)
                    except Exception:
                        pass
                    continue
                if RUN_MODE == "distributed":
                    # 分布式模式：PC 从节点尚未 TCP 重连，标记为 offline。
                    # Android HTTP 薄客户端不依赖 TCP，保持 DB 中记录的状态。
                    if node.node_type != "android":
                        node.state = NodeState.OFFLINE
                self.nodes[nid] = node
                logger.info(f"  恢复从节点: {nid} (state={node.state.value})")

        # ---- 从节点模式：仅创建自身 ----
        else:
            # ★ 安全：从节点绝不能使用 "master" 作为 node_id
            # 否则会从数据库加载主节点记录，导致后台面板显示主节点数据
            configured_node_id = _configured_node_id()
            if not configured_node_id or configured_node_id == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
                if configured_node_id == "master":
                    logger.warning(
                        f"⚠️ 从节点 NODE_ID 配置错误（仍为 \"master\"），"
                        f"已自动生成: {node_id}"
                    )
            else:
                node_id = configured_node_id
            if node_id in db_nodes:
                self.nodes[node_id] = self._node_from_db(db_nodes[node_id])
            else:
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id, role=NodeRole.CLIENT,
                    state=NodeState.ONLINE,
                    hostname="localhost",
                )

        # 从 DB 同步 MAX_NODES（仅主节点）
        if db and _db_available and effective_role == "master":
            try:
                stored_max = db.get_config("max_nodes", "")
                if stored_max and stored_max.isdigit():
                    new_max = int(stored_max)
                    if new_max != self._max_nodes:
                        old = self._max_nodes
                        self._max_nodes = new_max
                        import config as cfg
                        cfg.MAX_NODES = new_max
                        logger.info(f"从数据库恢复 max_nodes: {old} → {new_max}")
            except Exception:
                pass

            # 仅主节点持久化自身到 DB（从节点不自动写入数据库）
            try:
                master_node = self.nodes.get("master")
                if master_node:
                    db.upsert_node(
                        node_id="master", role="master", state=master_node.state.value,
                        address=master_node.address, hostname=master_node.hostname,
                        device_info=master_node.device_info, network_type=master_node.network_type,
                        connected_at=master_node.connected_at, last_heartbeat=master_node.last_heartbeat,
                        task_count=master_node.task_count, error_count=master_node.error_count,
                    )
            except Exception as e:
                logger.warning(f"主节点数据库同步失败: {e}")

        logger.info(
            f"节点初始化完成: {len(self.nodes)} 个节点 "
            f"(mode={RUN_MODE}, max_nodes={MAX_NODES}, my_role={effective_role})"
        )
        for nid, info in self.nodes.items():
            logger.info(f"  {nid}: role={info.role}, state={info.state.value}")

    def _node_from_db(self, db_row: dict) -> NodeInfo:
        """将数据库行转换为 NodeInfo"""
        return NodeInfo(
            node_id=db_row["node_id"],
            role=db_row.get("role", "client"),
            node_type=db_row.get("node_type", "pc"),
            state=NodeState(db_row.get("state", "offline")),
            address=db_row.get("address", ""),
            hostname=db_row.get("hostname", ""),
            device_info=db_row.get("device_info", {}),
            network_type=db_row.get("network_type", "unknown"),
            connected_at=db_row.get("connected_at", 0.0),
            last_heartbeat=db_row.get("last_heartbeat", 0.0),
            task_count=db_row.get("task_count", 0),
            error_count=db_row.get("error_count", 0),
            model_sha256=db_row.get("model_sha256", ""),
        )

    def register_node(self, node_id: str, role: str, address: str = "",
                      hostname: str = "", device_info: dict = None,
                      network_type: str = "unknown",
                      node_type: str = "pc",
                      model_sha256: str = "") -> bool:
        """
        注册一个从节点。

        调用时机：TCP 服务端收到 REGISTER 消息时。
        支持动态节点数量，允许 MAX_NODES 范围内的任意 client ID 注册。

        Args:
            node_id: 节点标识（"client1" / "client2" / ...）
            role: 节点角色 ("master" | "client")
            address: 客户端地址 "ip:port"
            hostname: 客户端主机名
            device_info: 客户端设备信息
            network_type: 网络连接类型 "wifi" | "ethernet" | "unknown"
            node_type: 设备平台 "pc" | "android"（默认 "pc"）
            model_sha256: 模型 SHA256 校验值（阶段 7，用于跨节点模型一致性验证）

        Returns:
            注册是否成功
        """
        # Android 节点只能作为 client
        if node_type == "android" and role == "master":
            logger.error(f"注册失败: Android 节点不能担任 master 角色")
            return False

        # 动态添加新节点（需检查容量限制）
        # Phase 2.1+: 所有 self.nodes 读写均在 _nodes_lock 保护下
        # Phase 5 review H4: 属性写入也纳入锁内，防止与 deregister_node TOCTOU
        with self._nodes_lock:
            if node_id not in self.nodes:
                if role == NodeRole.MASTER:
                    logger.warning(f"注册失败: 不能动态注册 master 节点")
                    return False
                # 容量检查：只统计在线节点（离线/幽灵不占位）
                online_non_master = [
                    n for n in self.nodes.values()
                    if n.role != "master" and n.is_available()
                ]
                if len(online_non_master) >= self._max_nodes - 1:
                    logger.warning(
                        f"注册失败: 已达到最大在线从节点数量 "
                        f"({len(online_non_master)}/{self._max_nodes - 1})"
                    )
                    return False
                logger.info(f"动态添加节点: {node_id} (type={node_type})")
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id, role=NodeRole.CLIENT,
                    node_type=node_type,
                    state=NodeState.OFFLINE,
                )

            node = self.nodes[node_id]

            if node.role == NodeRole.MASTER:
                logger.warning(f"注册失败: {node_id} 角色为 master，不可被注册覆盖")
                return False

            # NodeInfo 字段更新
            node.state = NodeState.ONLINE
            node.node_type = node_type
            node.model_sha256 = model_sha256
            node.address = address
            node.hostname = hostname
            node.device_info = device_info or {}
            node.network_type = network_type
            node.connected_at = time.time()
            node.last_heartbeat = time.time()

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.upsert_node(
                    node_id=node_id,
                    role=node.role,
                    node_type=node.node_type,
                    state=node.state.value,
                    address=node.address,
                    hostname=node.hostname,
                    device_info=node.device_info,
                    network_type=node.network_type,
                    connected_at=node.connected_at,
                    last_heartbeat=node.last_heartbeat,
                    task_count=node.task_count,
                    error_count=node.error_count,
                    model_sha256=node.model_sha256,
                )
            except Exception as e:
                logger.warning(f"节点注册 DB 持久化失败: {e}")

        # ★ 清除缓存的层分配方案，强制下次查询重新计算（含新节点）
        if db and _db_available and self._effective_role() == "master":
            try:
                db.set_layer_assignments({})
            except Exception:
                pass

        logger.info(
            f"✅ 节点注册: {node_id} role={role} type={node_type} "
            f"hostname={hostname} addr={address} net={network_type}"
        )
        return True

    def deregister_node(self, node_id: str) -> bool:
        """
        注销一个从节点（断连或主动离线）。

        Args:
            node_id: 节点标识

        Returns:
            注销是否成功
        """
        with self._nodes_lock:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]
            if node.role == NodeRole.MASTER:
                return False  # master 不可注销

            old_state = node.state
            node.state = NodeState.OFFLINE
            node.address = ""
            node.connected_at = 0.0

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.update_node_state(node_id=node_id, state="offline")
            except Exception as e:
                logger.warning(f"节点注销 DB 持久化失败: {e}")

        # 主节点：推送节点离线更新给所有已连接从节点
        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(
                node_id, "update", node
            )
            # ★ 清除缓存的层分配方案，强制下次查询重新计算（剔除已注销节点）
            if db and _db_available:
                try:
                    db.set_layer_assignments({})
                except Exception:
                    pass
            # ★ 清除层配置推送记录（节点离线后需重新推送）
            self._layer_config_pushed.discard(node_id)

        logger.info(f"节点注销: {node_id} ({old_state.value} → offline)")
        return True

    def update_node_state(self, node_id: str, state: NodeState) -> None:
        """更新节点状态"""
        with self._nodes_lock:
            if node_id in self.nodes:
                old_state = self.nodes[node_id].state
                self.nodes[node_id].state = state
                self.nodes[node_id].last_heartbeat = time.time()
                logger.info(f"节点 {node_id} 状态变更: {old_state.value} -> {state.value}")
            else:
                logger.warning(f"未知节点: {node_id}")

    def record_task_complete(self, node_id: str = None, success: bool = True) -> bool:
        """
        记录一次推理任务完成。

        Args:
            node_id: 执行推理的节点（默认本节点）
            success: 是否成功

        Returns:
            True = 成功更新了已知节点；False = 节点未知或更新失败。
        """
        nid = node_id or self.get_effective_node_id()
        with self._nodes_lock:
            if nid not in self.nodes:
                logger.warning(
                    f"任务计数跳过：未知节点 {nid}，"
                    f"当前已知节点={list(self.nodes.keys())}"
                )
                return False

            if success:
                self.nodes[nid].task_count += 1
            else:
                self.nodes[nid].error_count += 1
            self.nodes[nid].last_heartbeat = time.time()
            node_snapshot = self.nodes[nid]  # 锁内快照，供外部 I/O 使用

        db = _get_db()
        if db and _db_available:
            try:
                db.update_node_state(
                    node_id=nid,
                    state=node_snapshot.state.value,
                    last_heartbeat=node_snapshot.last_heartbeat,
                    task_count=node_snapshot.task_count,
                    error_count=node_snapshot.error_count,
                )
            except Exception as e:
                logger.debug(f"任务计数 DB 更新失败: {nid}: {e}", exc_info=True)

        if self._effective_role() == "master":
            try:
                self._push_node_update_to_all_clients(nid, "update", node_snapshot)
            except Exception:
                pass
        return True

    def record_task_error(self, node_id: str = None) -> None:
        """记录一次推理任务失败（便捷方法）"""
        self.record_task_complete(node_id, success=False)

    def _record_local_pipeline_participation(self, task_id: str,
                                             success: bool = True) -> bool:
        """从节点本地记录一次流水线参与，同一 task 只记一次。"""
        if not task_id:
            return False
        task_set = (self._local_pipeline_counted_tasks if success
                    else self._local_pipeline_error_tasks)
        if task_id in task_set:
            return False
        task_set.add(task_id)
        return self.record_task_complete(success=success)

    def _record_pipeline_task_accounting(self, task_id: str,
                                         pipeline_nodes: list,
                                         success: bool = True) -> dict:
        """
        主节点侧记录一次分布式流水线任务参与情况。

        语义：每个完成的用户请求，master 计 1 次服务请求，每个实际参与的
        PC worker 计 1 次参与请求。按 task_id 幂等，避免 streaming / retry 重复计数。
        """
        if not task_id:
            task_id = f"anonymous_{time.time()}"
        if task_id in self._pipeline_accounted_tasks:
            return {
                "task_id": task_id,
                "deduplicated": True,
                "counted_nodes": [],
                "skipped_unknown_nodes": [],
                "accounting_errors": [],
            }
        self._pipeline_accounted_tasks.add(task_id)

        ordered_nodes = [self.get_effective_node_id()]
        for node in pipeline_nodes or []:
            nid = node.get("node_id") if isinstance(node, dict) else str(node)
            if nid and nid not in ordered_nodes:
                ordered_nodes.append(nid)

        counted = []
        skipped = []
        errors = []
        for nid in ordered_nodes:
            try:
                if self.record_task_complete(nid, success=success):
                    counted.append(nid)
                else:
                    skipped.append(nid)
            except Exception as e:
                errors.append({"node_id": nid, "error": str(e)})

        return {
            "task_id": task_id,
            "deduplicated": False,
            "success": success,
            "counted_nodes": counted,
            "workers_counted": [nid for nid in counted if nid != self.get_effective_node_id()],
            "skipped_unknown_nodes": skipped,
            "accounting_errors": errors,
        }

    def register_android_client(self, node_id: str, hostname: str = "",
                                address: str = "", network_type: str = "unknown",
                                device_info: dict = None,
                                client_mode: str = "thin",
                                app_variant: str = "full",
                                app_version: str = "",
                                http_peer: str = "") -> dict:
        """登记 Android HTTP 薄客户端在线状态（不是 TCP worker 注册）。"""
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可登记 Android 客户端"}
        if not node_id or node_id == "master":
            return {"status": "invalid", "reason": "Android node_id 无效"}

        now = time.time()
        info = dict(device_info or {})
        info.update({
            "connection_type": "http_thin",
            "pipeline_worker": False,
            "client_mode": client_mode or "thin",
            "app_variant": app_variant or "full",
            "app_version": app_version or "",
        })
        if http_peer:
            info["http_peer"] = http_peer

        with self._nodes_lock:
            existing = self.nodes.get(node_id)
            is_new = existing is None
            if is_new:
                existing = NodeInfo(
                    node_id=node_id,
                    role=NodeRole.CLIENT,
                    node_type="android",
                    state=NodeState.ONLINE,
                    connected_at=now,
                )
                self.nodes[node_id] = existing
            elif existing.role == NodeRole.MASTER:
                return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可覆盖"}

            existing.node_type = "android"
            existing.role = NodeRole.CLIENT
            existing.state = NodeState.ONLINE
            existing.hostname = hostname or existing.hostname or node_id
            existing.address = address or existing.address or ""
            existing.network_type = network_type or existing.network_type or "unknown"
            existing.device_info = info
            if not existing.connected_at:
                existing.connected_at = now
            existing.last_heartbeat = now

        db = _get_db()
        if db and _db_available:
            try:
                db.upsert_node(
                    node_id=node_id,
                    role="client",
                    node_type="android",
                    state="online",
                    address=existing.address,
                    hostname=existing.hostname,
                    device_info=existing.device_info,
                    network_type=existing.network_type,
                    connected_at=existing.connected_at,
                    last_heartbeat=existing.last_heartbeat,
                    task_count=existing.task_count,
                    error_count=existing.error_count,
                    model_sha256=existing.model_sha256,
                )
            except Exception as e:
                logger.warning(f"Android 客户端登记 DB 持久化失败: {e}")

        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(
                node_id, "add" if is_new else "update", existing
            )

        logger.info(
            f"📱 Android HTTP 客户端在线: {node_id} host={existing.hostname} "
            f"net={existing.network_type} peer={http_peer}"
        )
        return {
            "status": "registered" if is_new else "updated",
            "node_id": node_id,
            "state": existing.state.value,
            "message": "Android HTTP thin client online",
        }

    def _refresh_http_client_states(self, now: float = None) -> None:
        """按 last_heartbeat 将过期的 Android HTTP 薄客户端标记为 offline。"""
        now = now or time.time()
        changed = []
        with self._nodes_lock:
            for node in self.nodes.values():
                if node.node_type != "android":
                    continue
                info = node.device_info or {}
                if info.get("connection_type") != "http_thin":
                    continue
                if node.state == NodeState.ONLINE and node.last_heartbeat:
                    if now - node.last_heartbeat > ANDROID_HTTP_CLIENT_TIMEOUT_SECONDS:
                        node.state = NodeState.OFFLINE
                        changed.append(node)

        if not changed:
            return

        db = _get_db()
        for node in changed:
            if db and _db_available:
                try:
                    db.update_node_state(node.node_id, "offline")
                except Exception as e:
                    logger.warning(f"Android 客户端过期状态 DB 更新失败: {e}")
            if self._effective_role() == "master":
                self._push_node_update_to_all_clients(node.node_id, "update", node)
            logger.info(f"📱 Android HTTP 客户端心跳过期: {node.node_id} → offline")

    def get_available_nodes(self) -> list:
        """获取所有可用节点（含状态字典）"""
        with self._nodes_lock:
            return [n.to_dict() for n in self.nodes.values() if n.is_available()]

    def check_nodes_ready(self) -> bool:
        """检查 PC worker 是否就绪（Android HTTP 薄客户端不参与分布式计算）。"""
        if RUN_MODE == "single":
            return True
        self._refresh_http_client_states()
        with self._nodes_lock:
            clients = [
                n for n in self.nodes.values()
                if n.role != NodeRole.MASTER and n.node_type == "pc"
            ]
            if not clients:
                return True  # 没有 PC 从节点也算就绪（单节点集群）
            for n in clients:
                if not n.is_available():
                    return False
        return True

    # ================================================================
    # 动态模型分层
    # ================================================================

    @staticmethod
    def _gpu_is_integrated(gpu: dict) -> bool:
        """判断 GPU 是否为集显；兼容旧画像缺少 is_integrated 字段的情况。"""
        if not isinstance(gpu, dict):
            return True
        if "is_integrated" in gpu:
            return bool(gpu.get("is_integrated"))
        gpu_type = str(gpu.get("gpu_type", "")).lower()
        if gpu_type == "discrete":
            return False
        if gpu_type == "integrated":
            return True
        name = str(gpu.get("name", "")).lower()
        if gpu.get("cuda_available") and any(k in name for k in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")):
            return False
        integrated_markers = (
            "intel", "iris", "uhd", "xe", "adreno", "mali",
            "powervr", "video core", "videocore", "radeon graphics",
        )
        if any(k in name for k in integrated_markers):
            return True
        # 无已知标记 → 保守判为集显（如设备画像缺少字段）
        return True

    @classmethod
    def _select_scoring_gpu(cls, device_info: dict) -> dict:
        """
        选择用于评分/显存约束的 GPU。

        多 GPU 机器上 device_info["gpu"] 可能是当前前端选中的集显，
        真正参与分布式推理的 CUDA 独显保存在 gpus 列表中。评分必须优先
        选择 CUDA 独显，避免 RTX 4060 + 集显的主节点被误算成 8 分左右。
        """
        if not device_info:
            return {}
        candidates = []
        selected = device_info.get("gpu", {})
        if isinstance(selected, dict) and selected:
            candidates.append(selected)
        for g in device_info.get("gpus", []) or []:
            if isinstance(g, dict) and g:
                # 去重：同名同显存的 selected_gpu 不重复加入。
                key = (g.get("name"), g.get("vram_total_gb"), g.get("cuda_available"))
                if not any((c.get("name"), c.get("vram_total_gb"), c.get("cuda_available")) == key for c in candidates):
                    candidates.append(g)
        if not candidates:
            return {}

        def _vram(g: dict) -> float:
            try:
                return float(g.get("vram_total_gb", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        # 1) CUDA 独显优先；2) 任意 CUDA；3) 非集显；4) 最大显存；5) selected_gpu。
        cuda_discrete = [g for g in candidates if g.get("cuda_available") and not cls._gpu_is_integrated(g)]
        if cuda_discrete:
            return max(cuda_discrete, key=_vram)
        cuda_any = [g for g in candidates if g.get("cuda_available")]
        if cuda_any:
            return max(cuda_any, key=_vram)
        discrete_any = [g for g in candidates if not cls._gpu_is_integrated(g)]
        if discrete_any:
            return max(discrete_any, key=_vram)
        return max(candidates, key=_vram)

    def _compute_node_weight(self, device_info: dict) -> float:
        """
        根据完整设备画像计算节点的计算能力权重（满分 100 + 独显奖励）。

        权重分配:
          - GPU 显存: 50%
          - 系统内存: 30%
          - CPU 核心+频率: 20%
          - 独显奖励: +15（CUDA 可用且非集显）

        Args:
            device_info: DeviceProfiler.to_dict() 完整输出

        Returns:
            权重分数（0 ~ 115）
        """
        gpu = self._select_scoring_gpu(device_info)
        ram = device_info.get("ram", {}) if device_info else {}
        cpu = device_info.get("cpu", {}) if device_info else {}

        # VRAM 得分（归一化到 0-50，上限 24GB）
        vram_gb = gpu.get("vram_total_gb", 0) if isinstance(gpu, dict) else 0
        vram_score = min(vram_gb / 24.0, 1.0) * 50.0 if vram_gb > 0 else 0

        # RAM 得分（归一化到 0-30，上限 64GB）
        ram_gb = ram.get("total_gb", 4) if isinstance(ram, dict) else 4
        ram_score = min(ram_gb / 64.0, 1.0) * 30.0

        # CPU 得分（核心数 0-10 + 频率 0-10，共 0-20）
        cpu_cores = cpu.get("physical_cores", 2) if isinstance(cpu, dict) else 2
        cpu_freq = cpu.get("freq_max_mhz", 2000) if isinstance(cpu, dict) else 2000
        core_score = min(cpu_cores / 16.0, 1.0) * 10.0
        freq_score = min(cpu_freq / 4000.0, 1.0) * 10.0
        cpu_score = core_score + freq_score

        # 独显奖励：CUDA 可用 且 非集成显卡 → +15
        cuda_avail = gpu.get("cuda_available", False) if isinstance(gpu, dict) else False
        discrete_bonus = 15.0 if (cuda_avail and not self._gpu_is_integrated(gpu)) else 0.0

        weight = vram_score + ram_score + cpu_score + discrete_bonus
        logger.debug(
            f"节点权重: GPU={gpu.get('name', 'unknown') if isinstance(gpu, dict) else 'unknown'} "
            f"VRAM={vram_score:.1f} RAM={ram_score:.1f} "
            f"CPU={cpu_score:.1f} Bonus={discrete_bonus:.1f} → {weight:.1f}"
        )
        return weight

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
        if isinstance(gpu, dict) and gpu.get("vram_total_gb", 0) > 0:
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

        # 根据量化精度计算实际单层显存
        from config import (
            MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB, LM_HEAD_VRAM_MB,
            SAFE_VRAM_MARGIN, LAYER_VRAM_FACTOR, QUANT_TYPE,
        )
        factor = LAYER_VRAM_FACTOR.get(QUANT_TYPE, 1.0)
        layer_mb = MIN_VRAM_PER_LAYER_MB * factor
        vram_needed = layers_count * layer_mb
        if has_embedding:
            vram_needed += EMBEDDING_VRAM_MB * factor
        if has_lm_head:
            vram_needed += LM_HEAD_VRAM_MB * factor
        vram_needed *= SAFE_VRAM_MARGIN

        ok = vram_available >= vram_needed
        return (ok, round(vram_needed, 1), round(vram_available, 1))

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
        from config import TOTAL_MODEL_LAYERS, GRAPH_ORCHESTRATOR_THRESHOLD
        from config import (
            QUANT_TYPE, MIN_VRAM_PER_LAYER_MB, EMBEDDING_VRAM_MB,
            LM_HEAD_VRAM_MB, LAYER_VRAM_FACTOR, SAFE_VRAM_MARGIN,
        )

        total_layers = TOTAL_MODEL_LAYERS

        # 收集节点数据（仅 PC 节点参与层拆分，Android 节点跳过）
        if nodes is None:
            # Phase 2.1: 快照 self.nodes 后解锁迭代，防止 TCP 回调并发修改 dict
            with self._nodes_lock:
                nodes_snapshot = list(self.nodes.items())
            node_list = [
                {"node_id": nid, "role": info.role,
                 "node_type": info.node_type,
                 "device_info": info.device_info}
                for nid, info in nodes_snapshot
                if info.node_type == "pc"
            ]
        else:
            node_list = [
                n for n in nodes
                if n.get("node_type", "pc") == "pc"
            ]

        if not node_list:
            logger.warning("没有可用的 PC 节点参与流水线层拆分")
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

                # 计算模型总显存需求
                factor = LAYER_VRAM_FACTOR.get(QUANT_TYPE, 1.0)
                model_memory_mb = (
                    TOTAL_MODEL_LAYERS * MIN_VRAM_PER_LAYER_MB * factor
                    + EMBEDDING_VRAM_MB * factor
                    + LM_HEAD_VRAM_MB * factor
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
                        quant_factor=factor,
                    )
                    assignments = orchestrator.orchestrate()

                    # 应用显存约束校验，并强制 master 作为首段锚点参与计算
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

        # Step 5: 显存约束校验
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
        """
        统一分层不变量：master 作为首段 Embedding 锚点且至少执行 1 层。

        master-first 是流水线语义约束，不表示主节点一定拿最多层；真实层数
        仍由评分/显存决定。若 rounding 或显存后处理把 master 挤出，则从
        最低优先级 worker 回收 1 层，保证主节点 CUDA 独显可参与首段计算。
        """
        if not assignments:
            return []

        def _is_master(item: dict) -> bool:
            return item.get("node_id") == "master" or item.get("role") == "master"

        def _score(item: dict) -> float:
            try:
                return float(item.get("score", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        master = None
        rest = []
        for a in assignments:
            item = dict(a)
            if _is_master(item) and master is None:
                master = item
            elif item.get("layers_count", 0) > 0:
                rest.append(item)

        if master is None:
            master_src = next((n for n in node_list if _is_master(n)), None)
            if master_src is None:
                return self._resequence_assignments(
                    [dict(a) for a in assignments if a.get("layers_count", 0) > 0]
                )
            master = {
                "node_id": master_src.get("node_id", "master"),
                "role": master_src.get("role", "master"),
                "layers_count": 1,
                "score": round(
                    master_src.get("score", self._compute_node_weight(master_src.get("device_info", {}))),
                    1,
                ),
            }
        else:
            master["layers_count"] = max(1, int(master.get("layers_count", 0) or 0))

        ordered = [master] + rest

        def _total() -> int:
            return sum(int(a.get("layers_count", 0) or 0) for a in ordered)

        # 若 master 被重新插入导致总层数超出，从低分 worker 优先回收。
        while _total() > total_layers and len(ordered) > 1:
            reducible = [a for a in ordered[1:] if a.get("layers_count", 0) > 1]
            if reducible:
                victim = min(reducible, key=_score)
                victim["layers_count"] -= 1
                continue
            # 节点数多于层数时，移除最低分 worker（master 保留）。
            victim = min(ordered[1:], key=_score)
            logger.warning(
                f"节点 {victim.get('node_id')} 层数降至 0 将被移除，"
                f"以保留 master 首段计算锚点"
            )
            ordered.remove(victim)

        # 若显存转移/rounding 后总层数不足，把缺口给最高分 worker；
        # 无 worker 时给 master。
        while _total() < total_layers and ordered:
            receivers = ordered[1:] or ordered
            target = max(receivers, key=_score)
            target["layers_count"] = int(target.get("layers_count", 0) or 0) + 1

        # 极端情况下仍超出（例如 total_layers=0 不会发生），最后从 master 尾部削减。
        while _total() > total_layers and master.get("layers_count", 0) > 1:
            master["layers_count"] -= 1

        return self._resequence_assignments(ordered)

    @staticmethod
    def _resequence_assignments(assignments: list) -> list:
        """按当前顺序重新计算连续层区间、Embedding 和 LM Head 标记。"""
        cleaned = [a for a in assignments if a.get("layers_count", 0) > 0]
        cursor = 0
        for i, a in enumerate(cleaned):
            count = int(a.get("layers_count", 0) or 0)
            a["layers_count"] = count
            a["start_layer"] = cursor
            a["end_layer"] = cursor + count
            a["has_embedding"] = (i == 0)
            a["has_lm_head"] = (i == len(cleaned) - 1)
            cursor += count
        return cleaned

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
        for a in assignments:
            ok, needed, available = self._check_vram_constraint(
                a["node_id"], a["layers_count"],
                a["has_embedding"], a["has_lm_head"],
            )
            if not ok and available > 0:
                # ★ 保护：首/末节点必须保留至少 1 层与 Embedding/LM Head 共存
                if a["has_embedding"] and a["layers_count"] <= 1:
                    logger.error(
                        f"❌ 首节点 {a['node_id']} 显存不足（需 {needed}MB / 可用 {available}MB），"
                        f"但 Embedding 层不可脱离 Transformer 层独立存在。"
                        f"建议增加该节点显存或降级为本地推理。"
                    )
                    continue
                if a["has_lm_head"] and a["layers_count"] <= 1:
                    logger.error(
                        f"❌ 末节点 {a['node_id']} 显存不足（需 {needed}MB / 可用 {available}MB），"
                        f"但 LM Head 不可脱离 Transformer 层独立存在。"
                        f"建议增加该节点显存或降级为本地推理。"
                    )
                    continue

                best_node = max(
                    (other for other in assignments if other["node_id"] != a["node_id"]),
                    key=lambda x: self._get_node_vram_mb(x["node_id"]),
                    default=None,
                )
                if best_node:
                    # ★ 只转移 Transformer 层，保留至少 1 层给 embedding/lm_head
                    keep = 1 if (a["has_embedding"] or a["has_lm_head"]) else 0
                    overflow = max(0, a["layers_count"] - keep)
                    if overflow > 0:
                        logger.warning(
                            f"⚠️ 显存不足: {a['node_id']} 需要 {needed}MB, "
                            f"可用 {available}MB — {overflow} 层转给 {best_node['node_id']}"
                            + (f"（保留 {keep} 层锚定 Embedding/LM Head）" if keep > 0 else "")
                        )
                        best_node["layers_count"] += overflow
                        a["layers_count"] = keep
                    else:
                        logger.warning(
                            f"⚠️ 显存不足但无可转移层: {a['node_id']} "
                            f"需要 {needed}MB, 可用 {available}MB"
                        )
                else:
                    logger.error(
                        f"❌ 显存不足且无其他节点可接手: {a['node_id']} "
                        f"需要 {needed}MB, 可用 {available}MB"
                    )

        # 清除层数为 0 的节点
        assignments[:] = [a for a in assignments if a["layers_count"] > 0]

        # ★ 重新计算所有节点的 layer 区间，确保 start_layer/end_layer 连续
        cursor = 0
        for i, a in enumerate(assignments):
            a["start_layer"] = cursor
            a["end_layer"] = cursor + a["layers_count"]
            a["has_embedding"] = (i == 0)
            a["has_lm_head"] = (i == len(assignments) - 1)
            cursor += a["layers_count"]

        return assignments

    def get_layer_assignments(self) -> dict:
        """
        获取当前分层配置。

        优先返回 DB 中的手动覆盖，否则动态计算并缓存到 DB。

        Returns:
            {
                "total": 24,
                "strategy": "dynamic" | "graph_orchestrator" | "manual",
                "assignments": [...],
                "computed_at": timestamp | null,
            }
        """
        from config import TOTAL_MODEL_LAYERS, GRAPH_ORCHESTRATOR_THRESHOLD

        db = _get_db()
        if db and _db_available:
            try:
                strategy = db.get_layer_strategy()
            except Exception:
                strategy = "dynamic"

            if strategy == "manual":
                try:
                    overrides = db.get_layer_override()
                    if overrides:
                        return {
                            "total": TOTAL_MODEL_LAYERS,
                            "strategy": "manual",
                            "assignments": overrides,
                            "computed_at": None,
                        }
                except Exception:
                    pass

            # 尝试从 DB 读取缓存的计算结果
            try:
                cached = db.get_layer_assignments()
                if cached and cached.get("assignments"):
                    return cached
            except Exception:
                pass

        # 动态计算
        assignments = self.compute_layer_assignment()

        # 判断实际使用的策略：当前运行时由 PC worker 覆盖 Transformer 层，
        # master/Android 不计入 graph_orchestrator 触发条件。
        worker_nodes_count = sum(
            1 for info in self.nodes.values()
            if info.node_type == "pc" and info.node_id != "master" and info.role != "master"
        )
        actual_strategy = (
            "graph_orchestrator"
            if worker_nodes_count > GRAPH_ORCHESTRATOR_THRESHOLD
            else "dynamic"
        )

        result = {
            "total": TOTAL_MODEL_LAYERS,
            "strategy": actual_strategy,
            "assignments": assignments,
            "computed_at": time.time(),
        }

        # 缓存到 DB
        if db and _db_available:
            try:
                db.set_layer_assignments(result)
            except Exception as e:
                logger.debug(f"分层缓存失败: {e}")

        return result

    def reset_layer_assignments(self) -> dict:
        """
        清除手动分层覆盖，恢复自动（dynamic）策略。

        仅主节点可调用。

        Returns:
            {"status": "ok", "strategy": "dynamic"}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可重置分层配置"}

        db = _get_db()
        if db and _db_available:
            try:
                db.clear_layer_override()
            except Exception as e:
                logger.warning(f"清除层覆盖失败: {e}")

        # 强制重新计算
        assignments = self.compute_layer_assignment()
        result = {
            "total": assignments[-1]["end_layer"] if assignments else 24,
            "strategy": "dynamic",
            "assignments": assignments,
            "computed_at": time.time(),
        }

        if db and _db_available:
            try:
                db.set_layer_assignments(result)
            except Exception as e:
                logger.debug(f"分层缓存失败: {e}")

        logger.info("分层配置已重置为自动策略")
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
        from config import TOTAL_MODEL_LAYERS

        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可覆盖分层配置"}

        # 基本验证
        if not assignments or not isinstance(assignments, list):
            return {"status": "invalid", "reason": "分层配置不能为空"}

        # 收集所有区间，排序验证连续性
        intervals = []
        for a in assignments:
            node_id = a.get("node_id", "")
            start = a.get("start_layer", 0)
            end = a.get("end_layer", 0)

            if node_id not in self.nodes:
                return {"status": "invalid", "reason": f"未知节点: {node_id}"}
            # Phase 4.1: 阻止 Android 节点被分配层（Android 无 PyTorch 推理能力）
            node_type = self.nodes[node_id].node_type
            if node_type == "android":
                return {"status": "invalid", "reason": f"Android 节点 {node_id} 不支持层前向传播"}
            if start < 0 or end > TOTAL_MODEL_LAYERS or start >= end:
                return {
                    "status": "invalid",
                    "reason": f"节点 {node_id} 区间 [{start}, {end}) 无效（范围 0-{TOTAL_MODEL_LAYERS}）",
                }
            intervals.append((start, end, node_id))

        # 排序后验证连续性
        intervals.sort(key=lambda x: x[0])

        covered = 0
        for start, end, node_id in intervals:
            if start != covered:
                return {
                    "status": "invalid",
                    "reason": f"节点 {node_id} 区间 [{start}, {end}) 不连续（期望从 {covered} 开始）",
                }
            covered = end

        if covered != TOTAL_MODEL_LAYERS:
            return {
                "status": "invalid",
                "reason": f"总覆盖范围 {covered} ≠ {TOTAL_MODEL_LAYERS}，分层未完整覆盖",
            }

        # 存储到 DB + 切换策略为 manual
        db = _get_db()
        if db and _db_available:
            try:
                db.set_layer_strategy("manual")
                db.set_layer_override(assignments)
            except Exception as e:
                return {"status": "error", "reason": f"DB 存储失败: {e}"}

        # 推送到已连接从节点
        self.push_layer_config_to_clients()

        logger.info(f"分层配置已手动覆盖: {len(assignments)} 个节点")
        return {
            "status": "ok",
            "message": "分层配置已更新（手动模式），已推送至从节点",
            "current_assignments": {
                "total": TOTAL_MODEL_LAYERS,
                "strategy": "manual",
                "assignments": assignments,
                "computed_at": None,
            },
        }

    def push_layer_config_to_clients(self) -> None:
        """
        向所有 TCP 连接的从节点推送其分层配置。

        ★ 阶段 7：推送前校验从节点模型 SHA256，
          不匹配的节点将排除出层分配，避免因模型不一致产生推理错误。
        """
        if not self._tcp_server or not self._tcp_server._running:
            return

        # ★ 获取主节点的模型 SHA256 作为基准
        master_sha256 = self._get_master_model_sha256()

        layer_info = self.get_layer_assignments()
        assignments = {}
        skipped_nodes = []

        # Phase 2.1+: 快照 self.nodes，避免在循环中反复读取并被并发修改
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)

        for a in layer_info["assignments"]:
            nid = a["node_id"]
            if nid == "master":
                continue

            node_info = nodes_snapshot.get(nid)
            if not node_info:
                continue

            # ★ 模型校验：从节点模型 SHA256 必须与主节点一致
            slave_sha256 = node_info.model_sha256
            if master_sha256 and slave_sha256:
                if slave_sha256 != master_sha256:
                    logger.warning(
                        f"⚠️ 模型不一致: {nid} SHA256={slave_sha256[:16]}..."
                        f" ≠ master SHA256={master_sha256[:16]}...，"
                        f"跳过该节点的层分配"
                    )
                    skipped_nodes.append((nid, slave_sha256))
                    continue  # 不分配层给模型不一致的节点
            elif master_sha256 and not slave_sha256:
                # Phase 4.2: 无 SHA256 的节点安全风险更高，提升为 WARNING 级别
                logger.warning(
                    f"⚠️ 节点 {nid} 未上报模型 SHA256，降级放行（可能存在模型不一致风险）"
                )

            assignments[nid] = {
                "start_layer": a["start_layer"],
                "end_layer": a["end_layer"],
                "has_embedding": a.get("has_embedding", False),
                "has_lm_head": a.get("has_lm_head", False),
            }

        if skipped_nodes:
            logger.warning(
                f"模型校验: {len(skipped_nodes)} 个节点因模型不一致被排除: "
                + ", ".join(f"{nid}({sha[:12]}...)" for nid, sha in skipped_nodes)
            )

        try:
            if assignments:
                self._tcp_server.broadcast_layer_config(assignments)
                for nid in assignments:
                    self._layer_config_pushed.add(nid)
                logger.info(f"分层配置已推送到 {len(assignments)} 个从节点")
            else:
                logger.warning("没有可用的从节点接收分层配置")
        except Exception as e:
            logger.warning(f"分层配置推送失败: {e}")

    def _get_master_model_sha256(self) -> str:
        """
        获取主节点当前加载模型的 SHA256。

        优先级:
        1. 从 models/ 目录读取 .sha256 缓存文件
        2. 计算实际模型文件的 SHA256
        3. 无法获取时返回空字符串（降级放行）
        """
        import hashlib as _hl
        import api_server as _api

        mgr = getattr(_api, 'model_manager', None)

        # 尝试从 model_manager 获取模型路径
        if mgr and mgr.is_loaded:
            try:
                model_path = getattr(mgr, '_model_path', '') or ''
                model_path = model_path or getattr(cfg, 'MODEL_PATH', '')
                if model_path and os.path.exists(model_path):
                    if os.path.isdir(model_path):
                        # PyTorch/safetensors 模型: 对目录下所有 .safetensors 文件联合哈希
                        import glob as _glob
                        safetensors_files = sorted(
                            _glob.glob(os.path.join(model_path, "*.safetensors"))
                        )
                        if safetensors_files:
                            sha256_file = os.path.join(model_path, "model.sha256")
                            if os.path.isfile(sha256_file):
                                try:
                                    with open(sha256_file, "r") as f:
                                        return f.read().strip().split()[0]
                                except Exception:
                                    pass
                            try:
                                h = _hl.sha256()
                                for sf in safetensors_files:
                                    with open(sf, "rb") as f:
                                        for chunk in iter(lambda: f.read(8192), b""):
                                            h.update(chunk)
                                result = h.hexdigest()
                                try:
                                    with open(sha256_file, "w") as f:
                                        f.write(f"{result}  {os.path.basename(model_path)}\n")
                                except Exception:
                                    pass
                                return result
                            except Exception:
                                pass
                    elif os.path.isfile(model_path):
                        # 单文件模型 (GGUF)
                        sha256_file = model_path + ".sha256"
                        if os.path.isfile(sha256_file):
                            try:
                                with open(sha256_file, "r") as f:
                                    return f.read().strip().split()[0]
                            except Exception:
                                pass
                        # 计算并缓存
                        try:
                            h = _hl.sha256()
                            with open(model_path, "rb") as f:
                                for chunk in iter(lambda: f.read(8192), b""):
                                    h.update(chunk)
                            result = h.hexdigest()
                            with open(sha256_file, "w") as f:
                                f.write(f"{result}  {os.path.basename(model_path)}\n")
                            return result
                        except Exception:
                            pass
            except Exception:
                pass

        # 回退：检查 models/ 目录下的 GGUF 文件
        import config as cfg
        gguf_path = getattr(cfg, 'GGUF_MODEL_PATH', '')
        if gguf_path and os.path.isfile(gguf_path):
            sha256_file = gguf_path + ".sha256"
            if os.path.isfile(sha256_file):
                try:
                    with open(sha256_file, "r") as f:
                        return f.read().strip().split()[0]
                except Exception:
                    pass

        return ""  # 无法获取 → 降级放行

    # ================================================================
    # TCP 消息处理
    # ================================================================

    def _on_tcp_message(self, client_id: str, msg: dict) -> None:
        """
        TCP 消息回调（由 TCPServer 调用）。

        根据消息类型更新节点状态、心跳等。
        """
        msg_type = msg.get("type", "")

        if msg_type == "register":
            data = msg.get("data", {})
            client_info = self._tcp_server.get_client_info(client_id) if self._tcp_server else {}
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

            if self._tcp_server and not self._tcp_server.confirm_registration(client_id):
                logger.warning("event=tcp_register_confirm_failed client_id=%s", client_id)
                self._tcp_server.reject_client(client_id, "注册确认发送失败")
                self.deregister_node(client_id)
                return

            # 新节点注册后重新计算分层并推送
            if registered and self._effective_role() == "master":
                self.push_layer_config_to_clients()
                # 向新注册的从节点推送全量节点列表（同步管理面板）
                self._push_node_list_to_client(client_id)
                # 向其他从节点推送新节点加入更新
                self._push_node_update_to_all_clients(
                    client_id, "add", self.nodes.get(client_id)
                )

        elif msg_type == "heartbeat":
            with self._nodes_lock:
                if client_id in self.nodes:
                    self.nodes[client_id].last_heartbeat = time.time()

        elif msg_type == "status_res":
            # 从节点状态上报
            data = msg.get("data", {})
            with self._nodes_lock:
                if client_id in self.nodes:
                    if "state" in data:
                        try:
                            self.nodes[client_id].state = NodeState(data["state"])
                        except ValueError:
                            pass

        elif msg_type == "error":
            data = msg.get("data", {})
            with self._nodes_lock:
                if client_id in self.nodes:
                    self.nodes[client_id].error_count += 1
            logger.error(f"节点 {client_id} 上报错误: {data.get('message', 'unknown')}")

        elif msg_type == "infer_forward":
            # 从节点转发推理请求给主节点
            self.handle_infer_forward(client_id, msg)

        elif msg_type == "infer_result":
            # 主节点返回推理结果给从节点
            data = msg.get("data", {})
            result_entry = {
                "task_id": data.get("task_id", ""),
                "content": data.get("content", ""),
                "metrics": data.get("metrics", {}),
            }
            if not hasattr(self, "_client_pending_results"):
                self._client_pending_results = {}
            self._client_pending_results[data.get("task_id", "")] = result_entry
            logger.info(f"收到推理结果: task={data.get('task_id', '')}, len={len(data.get('content', ''))}")

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
            self._handle_layer_forward(client_id, msg)

        elif msg_type == "layer_result":
            # ---- 主节点：收到从节点的层前向传播结果 ----
            self._handle_layer_result(client_id, msg)

        elif msg_type == "chain_forward":
            # ---- 从节点：收到另一从节点的链式直连转发（P2 优化）----
            self._handle_chain_forward(client_id, msg)

        elif msg_type == "chain_forward_ack":
            # ---- 主节点：收到链式转发每跳 ACK / 错误回报 ----
            self._handle_chain_forward_ack(client_id, msg)

        elif msg_type == "pipeline_done":
            # ---- 从节点：流水线任务完成，清理 KV 缓存 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
                        logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")

        elif msg_type == "pipeline_abort":
            # ---- 从节点/主节点：流水线任务取消 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id:
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        del self._kv_cache[task_id]
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
            self._handle_layer_config(client_id, data)

        elif msg_type == "log_request":
            # ---- L5: 主节点拉取从节点最近日志 ----
            data = msg.get("data", {})
            limit = data.get("limit", 100)
            try:
                from api_server import _snapshot_recent_logs, _filter_recent_logs
                entries, _ = _snapshot_recent_logs()
                filtered = _filter_recent_logs(
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
            self._pipeline_results[key] = {
                "task_id": task_id,
                "node_id": node_id,
                "error": error,
                "step": step,
            }
            event = self._pipeline_events.get(key)
            if event is not None:
                event.set()

    def _clear_pipeline_runtime_state(self, task_id: str) -> None:
        """清理主节点侧单个流水线任务的等待结果与链路 ACK 状态。"""
        if not task_id:
            return
        prefix = f"{task_id}:"
        with self._pipeline_lock:
            for key in list(self._pipeline_results):
                if key.startswith(prefix):
                    self._pipeline_results.pop(key, None)
            for key in list(self._pipeline_events):
                if key.startswith(prefix):
                    self._pipeline_events.pop(key, None)
            self._chain_ack_state.pop(task_id, None)

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
        task_id = data.get("task_id", "")
        step = data.get("step", -1)
        status = data.get("status", "received")
        error = data.get("error", "")
        reporter_node_id = data.get("node_id", client_id)
        target_node_id = data.get("target_node_id", "")
        node_id = target_node_id if status in ("sent", "error") and target_node_id else reporter_node_id

        if not task_id or not node_id:
            return

        now = time.time()
        with self._pipeline_lock:
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

    def _on_tcp_disconnect(self, client_id: str) -> None:
        """TCP 断连回调（由 TCPServer 调用）"""
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

    # ================================================================
    # 节点列表同步（主 → 从）
    # ================================================================

    def _push_node_list_to_client(self, client_id: str) -> None:
        """
        向指定从节点推送全量节点列表。

        调用时机:
        - 从节点注册成功后
        - 从节点主动请求 (node_list_sync with request="node_list")
        """
        if not self._tcp_server or not self._tcp_server._running:
            return
        try:
            from tcp_comm import MessageType
            # Phase 2.1+: 快照后解锁，避免持锁进行 TCP 发送（防止锁排序问题）
            with self._nodes_lock:
                nodes_data = [info.to_dict() for info in self.nodes.values()]
            self._tcp_server.send_to_client(
                client_id,
                {"nodes": nodes_data},
                MessageType.NODE_LIST_SYNC,
            )
            logger.debug(f"已向 {client_id} 推送全量节点列表 ({len(nodes_data)} 个)")
        except Exception as e:
            logger.warning(f"推送节点列表到 {client_id} 失败: {e}")

    def _push_node_update_to_all_clients(self, changed_id: str,
                                         action: str, node_info) -> None:
        """
        向所有已连接从节点推送单节点变更通知。

        Args:
            changed_id: 变更的节点 ID
            action: "add" | "update" | "remove"
            node_info: NodeInfo 对象或 None (remove 时)
        """
        if not self._tcp_server or not self._tcp_server._running:
            return
        try:
            from tcp_comm import MessageType
            node_data = node_info.to_dict() if node_info else {"node_id": changed_id}
            payload = {
                "action": action,
                "node": node_data,
            }
            for cid in self._tcp_server.get_client_ids():
                if cid == changed_id:
                    continue  # 不推送给变更节点自身
                try:
                    self._tcp_server.send_to_client(
                        cid, payload, MessageType.NODE_UPDATE
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"推送节点更新失败: {e}")

    def _apply_node_list_sync(self, nodes_data: list) -> None:
        """
        从节点：应用主节点推送的全量节点列表。

        保留本地节点信息，用主节点的数据补充/覆盖其他节点。
        """
        # Phase 2.1+: 全量同步需要原子修改 self.nodes，防止与心跳/消息回调并发
        with self._nodes_lock:
            local_id = self.get_effective_node_id()
            now_ts = time.time()
            for nd in nodes_data:
                nid = nd.get("node_id", "")
                if nid == local_id:
                    # ★ 更新自身节点信息（主节点视角更准确: network_type, last_heartbeat 等）
                    local = self.nodes.get(local_id)
                    if local:
                        local.network_type = nd.get("network_type", local.network_type)
                        local.last_heartbeat = nd.get("last_heartbeat", local.last_heartbeat)
                        local.connected_at = nd.get("connected_at", local.connected_at)
                        local.avg_rtt_ms = nd.get("avg_rtt_ms", local.avg_rtt_ms)
                        local.last_rtt_ms = nd.get("last_rtt_ms", local.last_rtt_ms)
                        local.address = nd.get("address", local.address)
                        local.hostname = nd.get("hostname", local.hostname)
                        local.device_info = nd.get("device_info", local.device_info)
                        local.task_count = nd.get("task_count", local.task_count)
                        local.error_count = nd.get("error_count", local.error_count)
                        try:
                            local.state = NodeState(nd.get("state", local.state.value))
                        except ValueError:
                            pass
                    continue
                if nid in self.nodes:
                    # 更新已有节点
                    existing = self.nodes[nid]
                    existing.role = nd.get("role", existing.role)
                    existing.node_type = nd.get("node_type", existing.node_type)
                    existing.hostname = nd.get("hostname", existing.hostname)
                    existing.address = nd.get("address", existing.address)
                    existing.device_info = nd.get("device_info", existing.device_info)
                    existing.network_type = nd.get("network_type", existing.network_type)
                    existing.avg_rtt_ms = nd.get("avg_rtt_ms", existing.avg_rtt_ms)
                    existing.last_rtt_ms = nd.get("last_rtt_ms", existing.last_rtt_ms)
                    existing.task_count = nd.get("task_count", existing.task_count)
                    existing.error_count = nd.get("error_count", existing.error_count)
                    try:
                        existing.state = NodeState(nd.get("state", "offline"))
                    except ValueError:
                        pass
                else:
                    # 新增节点
                    try:
                        state = NodeState(nd.get("state", "offline"))
                    except ValueError:
                        state = NodeState.OFFLINE
                    self.nodes[nid] = NodeInfo(
                        node_id=nid,
                        role=nd.get("role", "client"),
                        node_type=nd.get("node_type", "pc"),
                        state=state,
                        address=nd.get("address", ""),
                        hostname=nd.get("hostname", ""),
                        device_info=nd.get("device_info", {}),
                        network_type=nd.get("network_type", "unknown"),
                        connected_at=nd.get("connected_at", now_ts),
                        last_heartbeat=nd.get("last_heartbeat", now_ts),
                        avg_rtt_ms=nd.get("avg_rtt_ms", 0.0),
                        last_rtt_ms=nd.get("last_rtt_ms", 0.0),
                        task_count=nd.get("task_count", 0),
                        error_count=nd.get("error_count", 0),
                    )

    def _apply_node_update(self, action: str, node_data: dict) -> None:
        """
        从节点：应用单节点变更通知。
        """
        nid = node_data.get("node_id", "")
        if not nid:
            return
        local_id = self.get_effective_node_id()

        # Phase 2.1+: 原子修改 self.nodes，防止与心跳/消息回调并发
        with self._nodes_lock:
            if nid == local_id:
                # ★ 更新自身节点信息（来自主节点的状态更新）
                if action in ("add", "update"):
                    local = self.nodes.get(local_id)
                    if local:
                        local.last_heartbeat = node_data.get("last_heartbeat", local.last_heartbeat)
                        local.network_type = node_data.get("network_type", local.network_type)
                        local.connected_at = node_data.get("connected_at", local.connected_at)
                        local.avg_rtt_ms = node_data.get("avg_rtt_ms", local.avg_rtt_ms)
                        local.last_rtt_ms = node_data.get("last_rtt_ms", local.last_rtt_ms)
                        local.task_count = node_data.get("task_count", local.task_count)
                        local.error_count = node_data.get("error_count", local.error_count)
                        try:
                            local.state = NodeState(node_data.get("state", local.state.value))
                        except ValueError:
                            pass
                return
            if action == "remove":
                self.nodes.pop(nid, None)
            elif action in ("add", "update"):
                now_ts = time.time()
                try:
                    state = NodeState(node_data.get("state", "offline"))
                except ValueError:
                    state = NodeState.OFFLINE
                if nid in self.nodes:
                    existing = self.nodes[nid]
                    existing.state = state
                    existing.role = node_data.get("role", existing.role)
                    existing.node_type = node_data.get("node_type", existing.node_type)
                    existing.hostname = node_data.get("hostname", existing.hostname)
                    existing.address = node_data.get("address", existing.address)
                    existing.device_info = node_data.get("device_info", existing.device_info)
                    existing.network_type = node_data.get("network_type", existing.network_type)
                    existing.avg_rtt_ms = node_data.get("avg_rtt_ms", existing.avg_rtt_ms)
                    existing.last_rtt_ms = node_data.get("last_rtt_ms", existing.last_rtt_ms)
                    existing.task_count = node_data.get("task_count", existing.task_count)
                    existing.error_count = node_data.get("error_count", existing.error_count)
                else:
                    self.nodes[nid] = NodeInfo(
                        node_id=nid,
                        role=node_data.get("role", "client"),
                    node_type=node_data.get("node_type", "pc"),
                    state=state,
                    address=node_data.get("address", ""),
                    hostname=node_data.get("hostname", ""),
                    device_info=node_data.get("device_info", {}),
                    network_type=node_data.get("network_type", "unknown"),
                    connected_at=node_data.get("connected_at", now_ts),
                    last_heartbeat=node_data.get("last_heartbeat", now_ts),
                    avg_rtt_ms=node_data.get("avg_rtt_ms", 0.0),
                    last_rtt_ms=node_data.get("last_rtt_ms", 0.0),
                    task_count=node_data.get("task_count", 0),
                    error_count=node_data.get("error_count", 0),
                )

    # ================================================================
    # 角色转让 — 主节点身份转移
    # ================================================================

    def transfer_master_role(self, target_node_id: str) -> dict:
        """
        将主节点身份转让给指定从节点（仅主节点可调用）。

        流程:
          1. 验证目标节点在线且为 client
          2. 通过 TCP 向目标发送 ROLE_TRANSFER 消息
          3. 等待 ROLE_TRANSFER_ACK 确认（超时 15s）
          4. 主节点保存降级日志 → 更新 DB master 信息 → 通知其他从节点
          5. 返回操作结果（建议重启以应用新角色）

        Args:
            target_node_id: 目标从节点 ID

        Returns:
            {status, message, transfer_id, ...}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可发起角色转让"}

        # ---- 备用主节点前置检查 ----
        # 备用主节点用于填补转让空窗期，不是转让目标
        spare = self.get_spare_master()
        if not spare or not spare.get("node_id"):
            return {
                "status": "invalid",
                "reason": (
                    "未指定备用主节点，无法转让。"
                    "备用主节点在转让空窗期暂代主节点职责，请先在「备用主节点」中指定。"
                ),
            }

        spare_id = spare.get("node_id")

        # 转让目标不能是备用主节点本身（备用主节点负责监政，不兼任新主节点）
        if target_node_id == spare_id:
            return {
                "status": "invalid",
                "reason": (
                    f"备用主节点 '{spare_id}' 负责在空窗期暂代监政，不能同时成为转让目标。"
                    "请选择其他在线从节点作为新主节点。"
                ),
            }

        # ---- 备用主节点检查通过 ----

        if target_node_id not in self.nodes:
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不存在"}

        target = self.nodes[target_node_id]
        if target.role not in ("client", NodeRole.CLIENT):
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不是从节点"}

        if not target.is_available():
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不在线，无法转让"}

        if not self._tcp_server or not self._tcp_server._running:
            return {"status": "error", "reason": "TCP 服务未运行，无法发送转让通知"}

        # ---- P3: 审查门控 ----
        # 主节点转让需要先通过审查投票（>= +2）
        try:
            from review import ReviewManager
            review_mgr = ReviewManager()
            approved = review_mgr.find_approved_ticket(target_node_id)
            if not approved:
                return {
                    "status": "needs_review",
                    "reason": (
                        f"主节点转让给 '{target_node_id}' 需要审查投票通过。\n"
                        f"请先通过管理面板创建审查工单（POST /api/cluster/review/create），\n"
                        f"获得 >= +2 票后重试转让操作。\n"
                        f"当前仅 PC 独显版节点可参与审查投票。"
                    ),
                }
            logger.info(
                f"审查门控通过: ticket={approved.ticket_id} "
                f"score={approved.score} target={target_node_id}"
            )
        except ImportError:
            logger.warning("审查模块不可用，跳过审查门控")
        # ---- 审查门控结束 ----

        transfer_id = f"transfer_{int(time.time() * 1000)}"

        # 收集当前集群信息，一并发送给新主节点
        cluster_info = {
            "transfer_id": transfer_id,
            "old_master_id": NODE_ID,
            "new_master_id": target_node_id,
            "server_ip": getattr(self, '_lan_ip', '') or SERVER_IP,
            "server_port": SERVER_PORT,
            "timestamp": time.time(),
            "layer_assignments": self.get_layer_assignments(),
            "registered_nodes": {
                nid: {"role": info.role, "state": info.state.value}
                for nid, info in self.nodes.items()
            },
        }

        # 步骤 1: 发送 ROLE_TRANSFER 给目标从节点（新主节点）
        try:
            from tcp_comm import MessageType
            self._tcp_server.send_to_client(
                target_node_id, cluster_info, MessageType.ROLE_TRANSFER
            )
            logger.info(
                f"角色转让请求已发送: {NODE_ID} → {target_node_id} "
                f"(transfer_id={transfer_id})"
            )
        except Exception as e:
            return {"status": "error", "reason": f"TCP 发送失败 (目标节点): {e}"}

        # 步骤 2: 发送 SPARE_MASTER_ACTIVATE 给备用主节点（暂代监政）
        activate_data = {
            "activate_id": f"activate_{transfer_id}",
            "transfer_id": transfer_id,
            "old_master_id": NODE_ID,
            "new_master_id": target_node_id,
            "server_ip": getattr(self, '_lan_ip', '') or SERVER_IP,
            "server_port": SERVER_PORT,
            "timestamp": time.time(),
            "message": (
                f"主节点身份即将从 {NODE_ID} 转让给 {target_node_id}。"
                "请暂代主节点职责，直到新主节点上线接管。"
            ),
        }
        try:
            self._tcp_server.send_to_client(
                spare_id, activate_data, MessageType.SPARE_MASTER_ACTIVATE
            )
            logger.info(
                f"备用主节点激活请求已发送: {spare_id} "
                f"(等待新主节点 {target_node_id} 上线)"
            )
        except Exception as e:
            return {"status": "error", "reason": f"TCP 发送失败 (备用主节点): {e}"}

        # 步骤 3: 等待两个 ACK（ROLE_TRANSFER_ACK + SPARE_MASTER_ACTIVATE_ACK）
        if not hasattr(self, "_transfer_acks"):
            self._transfer_acks = {}
        if not hasattr(self, "_spare_activate_acks"):
            self._spare_activate_acks = {}
        self._transfer_acks[transfer_id] = None
        self._spare_activate_acks[activate_data["activate_id"]] = None

        deadline = time.time() + 15
        target_ack = None
        spare_ack = None
        while time.time() < deadline:
            if target_ack is None:
                target_ack = self._transfer_acks.get(transfer_id)
            if spare_ack is None:
                spare_ack = self._spare_activate_acks.get(activate_data["activate_id"])
            if target_ack is not None and spare_ack is not None:
                break
            time.sleep(0.3)

        # 检查目标节点 ACK
        if target_ack is None:
            self._transfer_acks.pop(transfer_id, None)
            self._spare_activate_acks.pop(activate_data["activate_id"], None)
            return {
                "status": "timeout",
                "reason": f"目标节点 '{target_node_id}' 未在 15s 内确认转让",
                "transfer_id": transfer_id,
            }

        # 检查备用主节点 ACK
        if spare_ack is None:
            self._transfer_acks.pop(transfer_id, None)
            self._spare_activate_acks.pop(activate_data["activate_id"], None)
            return {
                "status": "timeout",
                "reason": f"备用主节点 '{spare_id}' 未在 15s 内确认激活",
                "transfer_id": transfer_id,
            }

        self._transfer_acks.pop(transfer_id, None)
        self._spare_activate_acks.pop(activate_data["activate_id"], None)

        # 步骤 4: 主节点保存降级日志 + 备用激活日志
        db = _get_db()
        try:
            if db and _db_available:
                db.append_transfer_log(
                    direction="demotion",
                    from_role="master",
                    to_role="client",
                    related_node=target_node_id,
                    details={
                        "transfer_id": transfer_id,
                        "target_ack": target_ack,
                        "spare_activated": spare_id,
                        "spare_ack": spare_ack,
                        "node_count": len(self.nodes),
                    },
                )
                db.append_spare_master_log(
                    direction="activated",
                    details={
                        "transfer_id": transfer_id,
                        "old_master_id": NODE_ID,
                        "new_master_id": target_node_id,
                        "spare_node_id": spare_id,
                        "ack": spare_ack,
                    },
                )
        except Exception as e:
            logger.warning(f"日志写入失败: {e}")

        # 步骤 5: 更新数据库 — 新主节点信息
        try:
            if db and _db_available:
                db.set_config("master_node_id", target_node_id)
                db.set_config("new_master_node_id", target_node_id)
                db.set_config("master_role_transferred", "true")
                db.set_config("spare_master_active", "true")  # 标记备用主节点已激活
                db.set_config("last_transfer_id", transfer_id)
                db.set_config("last_transfer_time", str(time.time()))
                logger.info(f"数据库已更新: 新主节点 = {target_node_id}, 备用 = {spare_id} (已激活)")
        except Exception as e:
            logger.warning(f"数据库更新失败: {e}")

        logger.info(
            f"✅ 角色转让完成: {NODE_ID} → {target_node_id} "
            f"备用主节点 {spare_id} 已激活暂代 (transfer_id={transfer_id})"
        )

        return {
            "status": "ok",
            "message": (
                f"主节点身份已转让给 '{target_node_id}'。"
                f"备用主节点 '{spare_id}' 已激活，将暂代主节点职责直到新主节点上线。"
                f"建议双方重启服务：目标节点以主节点模式运行，本节点以从节点模式运行。"
            ),
            "transfer_id": transfer_id,
            "from_node": NODE_ID,
            "to_node": target_node_id,
            "spare_activated": spare_id,
            "target_ack": target_ack,
            "spare_ack": spare_ack,
        }

    def _handle_role_transfer(self, client_id: str, msg: dict) -> None:
        """
        从节点收到 ROLE_TRANSFER 消息（被选为新主节点）。

        操作:
          1. 保存升级日志到数据库
          2. 更新本地节点角色标记
          3. 发送 ROLE_TRANSFER_ACK 确认
          4. 提示用户重启以应用新角色
        """
        data = msg.get("data", {})
        transfer_id = data.get("transfer_id", "")
        old_master_id = data.get("old_master_id", "")
        new_master_id = data.get("new_master_id", "")

        logger.info(
            f"🔔 收到角色转让通知: {old_master_id} → {new_master_id} "
            f"(transfer_id={transfer_id})"
        )

        # 保存升级日志
        db = _get_db()
        try:
            if db and _db_available:
                db.append_transfer_log(
                    direction="promotion",
                    from_role="client",
                    to_role="master",
                    related_node=old_master_id,
                    details={
                        "transfer_id": transfer_id,
                        "old_master_id": old_master_id,
                        "cluster_info": data,
                    },
                )
                # 更新数据库中的节点角色
                db.set_config("node_role_override", "master")
                db.set_config("last_transfer_id", transfer_id)
                logger.info(f"升级日志已保存: client → master (transfer_id={transfer_id})")
        except Exception as e:
            logger.warning(f"升级日志写入失败: {e}")

        # 发送 ACK（从节点通过 TCP 客户端连接回传给主节点）
        ack_payload = {
            "transfer_id": transfer_id,
            "ack": {
                "transfer_id": transfer_id,
                "accepted": True,
                "node_id": NODE_ID,
                "timestamp": time.time(),
            },
        }

        tcp_client = getattr(self, '_tcp_client', None)
        if tcp_client and tcp_client.sock:
            try:
                from tcp_comm import build_message, MessageType
                packet = build_message(MessageType.ROLE_TRANSFER_ACK, ack_payload)
                tcp_client.sock.sendall(packet)
                logger.info(f"已发送角色转让确认: transfer_id={transfer_id}")
            except Exception as e:
                logger.warning(f"发送 ACK 失败: {e}")
        else:
            logger.warning("TCP 客户端未连接，无法发送 ACK")

        logger.info(
            f"✅ 角色升级已确认: client → master。"
            f"请重启本节点以主节点模式运行。"
        )

    def _handle_role_transfer_ack(self, client_id: str, msg: dict) -> None:
        """
        主节点收到从节点的 ROLE_TRANSFER_ACK。

        将 ACK 结果存入 _transfer_acks 供 transfer_master_role() 读取。
        """
        data = msg.get("data", {})
        transfer_id = data.get("transfer_id", "")
        ack = data.get("ack", {})

        logger.info(f"收到角色转让确认: from={client_id}, transfer_id={transfer_id}")

        if not hasattr(self, "_transfer_acks"):
            self._transfer_acks = {}
        self._transfer_acks[transfer_id] = {
            "client_id": client_id,
            "ack": ack,
            "received_at": time.time(),
        }

    def can_node_vote(self, node_id: str) -> tuple[bool, str]:
        """
        检查节点是否有审查投票资格（P3: 主节点转让审查）。

        仅 node_type="pc" 且具备 NVIDIA CUDA 独显的节点可投票。

        Args:
            node_id: 节点 ID

        Returns:
            (can_vote: bool, reason: str)
        """
        effective_id = self.get_effective_node_id()
        is_local_master_query = (
            self._effective_role() == "master"
            and node_id in {effective_id, "master"}
        )

        with self._nodes_lock:
            node = self.nodes.get(node_id)
            # 主节点可能使用自定义 NODE_ID，但节点表中仍以 "master" 保存自身。
            if node is None and is_local_master_query:
                node = self.nodes.get("master")
            node_type = node.node_type if node else None
            device_info = dict(node.device_info or {}) if node else {}

        if node is None and not is_local_master_query:
            return False, f"节点 '{node_id}' 未注册"

        if node_type is None and is_local_master_query:
            node_type = "pc"

        if node_type != "pc":
            return False, "仅 PC 节点可参与审查投票"

        local_profile_cache = None

        def load_local_profile() -> dict:
            nonlocal local_profile_cache
            if local_profile_cache is not None:
                return local_profile_cache
            local_profile_cache = {}
            try:
                from device_profiler import get_profile
                profile = get_profile()
                if profile:
                    local_profile_cache = profile.to_dict()
            except Exception as e:
                logger.debug(f"读取本机设备画像失败，无法用于投票资格兜底: {e}")
            return local_profile_cache

        def has_cuda_discrete(info: dict) -> bool:
            gpu = self._select_scoring_gpu(info or {})
            return bool(
                isinstance(gpu, dict)
                and gpu.get("cuda_available", False)
                and not self._gpu_is_integrated(gpu)
            )

        if not device_info and is_local_master_query:
            device_info = load_local_profile()

        if has_cuda_discrete(device_info):
            return True, "ok"

        # 本地主节点的 DB/节点表可能保存了旧画像（例如只记录了集显）。
        # 仅对当前主节点再读取实时画像兜底，避免误把本机硬件套用到远端节点。
        if is_local_master_query:
            local_device_info = load_local_profile()
            if local_device_info and local_device_info != device_info:
                if has_cuda_discrete(local_device_info):
                    return True, "ok"

        if not device_info:
            return False, "节点缺少设备画像，无法确认 CUDA 独显"

        if not has_cuda_discrete(device_info):
            return False, "仅 NVIDIA CUDA 独显节点可参与审查投票"

        return True, "ok"

    def get_transfer_logs(self) -> list:
        """获取所有角色转让日志"""
        db = _get_db()
        if db and _db_available:
            try:
                return db.get_transfer_logs()
            except Exception:
                pass
        return []

    # ================================================================
    # 备用主节点管理
    # ================================================================

    def designate_spare_master(self, target_node_id: str) -> dict:
        """
        指定一个在线从节点为备用主节点（仅主节点可调用）。

        规则:
          - 集群节点数 ≥ 2（master + 至少 1 个 client）
          - 目标节点必须在线且为 client
          - 不能重复指定同一个节点

        流程:
          1. 验证条件和目标节点
          2. 通过 TCP 向目标发送 SPARE_MASTER_DESIGNATE 消息
          3. 等待 ACK 确认（超时 15s）
          4. 保存备用主节点信息到数据库 + 日志

        Args:
            target_node_id: 目标从节点 ID

        Returns:
            {status, message, spare_master, ...}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可指定备用主节点"}

        # 检查集群节点数 ≥ 2
        online_clients = [
            nid for nid, info in self.nodes.items()
            if info.role in ("client", NodeRole.CLIENT) and info.is_available()
        ]
        if len(self.nodes) < 2 or len(online_clients) < 1:
            return {
                "status": "invalid",
                "reason": "集群节点数不足（需要 ≥2 个节点，且至少有 1 个在线从节点）",
            }

        if target_node_id not in self.nodes:
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不存在"}

        target = self.nodes[target_node_id]
        if target.role not in ("client", NodeRole.CLIENT):
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不是从节点"}

        if not target.is_available():
            return {"status": "invalid", "reason": f"节点 '{target_node_id}' 不在线，无法指定为备用主节点"}

        # 检查是否已经是备用主节点
        db = _get_db()
        try:
            if db and _db_available:
                existing = db.get_spare_master()
                if existing and existing.get("node_id") == target_node_id:
                    return {
                        "status": "duplicate",
                        "reason": f"节点 '{target_node_id}' 已经是备用主节点",
                        "spare_master": existing,
                    }
        except Exception:
            pass

        if not self._tcp_server or not self._tcp_server._running:
            return {"status": "error", "reason": "TCP 服务未运行，无法发送通知"}

        if not hasattr(self, "_spare_acks"):
            self._spare_acks = {}

        designate_id = f"spare_{int(time.time() * 1000)}"

        # 收集集群信息一并发送
        designate_data = {
            "designate_id": designate_id,
            "master_id": NODE_ID,
            "target_node_id": target_node_id,
            "server_ip": getattr(self, '_lan_ip', '') or SERVER_IP,
            "server_port": SERVER_PORT,
            "timestamp": time.time(),
            "role": "spare_master",
        }

        # 步骤 1: 发送 SPARE_MASTER_DESIGNATE
        try:
            from tcp_comm import MessageType
            self._tcp_server.send_to_client(
                target_node_id, designate_data, MessageType.SPARE_MASTER_DESIGNATE
            )
            logger.info(
                f"备用主节点指定请求已发送: {target_node_id} "
                f"(designate_id={designate_id})"
            )
        except Exception as e:
            return {"status": "error", "reason": f"TCP 发送失败: {e}"}

        # 步骤 2: 等待 ACK
        self._spare_acks[designate_id] = None
        deadline = time.time() + 15
        ack_received = False
        while time.time() < deadline:
            if self._spare_acks.get(designate_id) is not None:
                ack_received = True
                break
            time.sleep(0.3)

        if not ack_received:
            self._spare_acks.pop(designate_id, None)
            return {
                "status": "timeout",
                "reason": f"节点 '{target_node_id}' 未在 15s 内确认备用主节点指定",
                "designate_id": designate_id,
            }

        ack_data = self._spare_acks.pop(designate_id)

        # 步骤 3: 持久化到数据库
        try:
            if db and _db_available:
                db.set_spare_master(
                    node_id=target_node_id,
                    hostname=target.hostname or "",
                    address=target.address or "",
                )
                db.append_spare_master_log(
                    direction="designated",
                    details={
                        "designate_id": designate_id,
                        "master_id": NODE_ID,
                        "target_node_id": target_node_id,
                        "ack": ack_data,
                    },
                )
                logger.info(f"备用主节点已持久化: {target_node_id}")
        except Exception as e:
            logger.warning(f"备用主节点持久化失败: {e}")

        return {
            "status": "ok",
            "message": (
                f"已将 '{target_node_id}' 指定为备用主节点。"
                f"当主节点需要转让身份时，可转让给该备用主节点。"
            ),
            "designate_id": designate_id,
            "spare_master": {
                "node_id": target_node_id,
                "hostname": target.hostname or "",
                "address": target.address or "",
                "designated_at": time.time(),
            },
        }

    def _handle_spare_master_designate(self, client_id: str, msg: dict) -> None:
        """
        从节点收到 SPARE_MASTER_DESIGNATE 消息（被指定为备用主节点）。

        操作:
          1. 保存备用主节点指定日志到数据库
          2. 发送 ACK 确认
        """
        data = msg.get("data", {})
        designate_id = data.get("designate_id", "")
        master_id = data.get("master_id", "")

        logger.info(
            f"🔔 收到备用主节点指定: master={master_id}, "
            f"designate_id={designate_id}"
        )

        # 保存日志
        db = _get_db()
        try:
            if db and _db_available:
                db.append_spare_master_log(
                    direction="designated",
                    details={
                        "designate_id": designate_id,
                        "master_id": master_id,
                        "role": "spare_master",
                    },
                )
                # 标记本节点为备用主节点
                db.set_config("node_is_spare_master", "true")
                logger.info(f"备用主节点日志已保存 (designate_id={designate_id})")
        except Exception as e:
            logger.warning(f"备用主节点日志写入失败: {e}")

        # 发送 ACK
        ack_payload = {
            "designate_id": designate_id,
            "ack": {
                "designate_id": designate_id,
                "accepted": True,
                "node_id": NODE_ID,
                "timestamp": time.time(),
            },
        }

        tcp_client = getattr(self, '_tcp_client', None)
        if tcp_client and tcp_client.sock:
            try:
                from tcp_comm import build_message, MessageType
                packet = build_message(MessageType.SPARE_MASTER_DESIGNATE_ACK, ack_payload)
                tcp_client.sock.sendall(packet)
                logger.info(f"已发送备用主节点指定确认: designate_id={designate_id}")
            except Exception as e:
                logger.warning(f"发送备用 ACK 失败: {e}")
        else:
            logger.warning("TCP 客户端未连接，无法发送备用 ACK")

    def _handle_spare_master_designate_ack(self, client_id: str, msg: dict) -> None:
        """
        主节点收到从节点的 SPARE_MASTER_DESIGNATE_ACK。

        将 ACK 结果存入 _spare_acks 供 designate_spare_master() 读取。
        """
        data = msg.get("data", {})
        designate_id = data.get("designate_id", "")
        ack = data.get("ack", {})

        logger.info(f"收到备用主节点指定确认: from={client_id}, designate_id={designate_id}")

        if not hasattr(self, "_spare_acks"):
            self._spare_acks = {}
        self._spare_acks[designate_id] = {
            "client_id": client_id,
            "ack": ack,
            "received_at": time.time(),
        }

    # ---- 备用主节点：激活（暂代） / 接管（退出暂代） ----

    def _handle_spare_master_activate(self, client_id: str, msg: dict) -> None:
        """
        备用主节点收到 SPARE_MASTER_ACTIVATE 消息（被要求暂代主节点职责）。

        操作:
          1. 记录激活日志
          2. 进入「暂代主节点」模式
          3. 发送 ACK 确认
        """
        data = msg.get("data", {})
        activate_id = data.get("activate_id", "")
        transfer_id = data.get("transfer_id", "")
        old_master_id = data.get("old_master_id", "")
        new_master_id = data.get("new_master_id", "")

        logger.info(
            f"🔔 收到备用主节点激活通知: master={old_master_id} → "
            f"new_master={new_master_id} (activate_id={activate_id})"
        )

        # 记录激活日志
        db = _get_db()
        try:
            if db and _db_available:
                db.append_spare_master_log(
                    direction="activated",
                    details={
                        "activate_id": activate_id,
                        "transfer_id": transfer_id,
                        "old_master_id": old_master_id,
                        "new_master_id": new_master_id,
                    },
                )
                db.set_config("spare_master_active", "true")
                db.set_config("pending_new_master_id", new_master_id)
                logger.info(f"备用主节点激活日志已保存，等待新主节点 {new_master_id} 上线")
        except Exception as e:
            logger.warning(f"备用主节点激活日志写入失败: {e}")

        # 发送 ACK
        ack_payload = {
            "activate_id": activate_id,
            "ack": {
                "activate_id": activate_id,
                "accepted": True,
                "node_id": NODE_ID,
                "timestamp": time.time(),
            },
        }

        tcp_client = getattr(self, '_tcp_client', None)
        if tcp_client and tcp_client.sock:
            try:
                from tcp_comm import build_message, MessageType
                packet = build_message(MessageType.SPARE_MASTER_ACTIVATE_ACK, ack_payload)
                tcp_client.sock.sendall(packet)
                logger.info(f"已发送备用主节点激活确认: activate_id={activate_id}")
            except Exception as e:
                logger.warning(f"发送激活 ACK 失败: {e}")
        else:
            logger.warning("TCP 客户端未连接，无法发送激活 ACK")

    def _handle_spare_master_activate_ack(self, client_id: str, msg: dict) -> None:
        """
        主节点收到备用主节点的 SPARE_MASTER_ACTIVATE_ACK。

        将 ACK 结果存入 _spare_activate_acks 供 transfer_master_role() 读取。
        """
        data = msg.get("data", {})
        activate_id = data.get("activate_id", "")
        ack = data.get("ack", {})

        logger.info(f"收到备用主节点激活确认: from={client_id}, activate_id={activate_id}")

        if not hasattr(self, "_spare_activate_acks"):
            self._spare_activate_acks = {}
        self._spare_activate_acks[activate_id] = {
            "client_id": client_id,
            "ack": ack,
            "received_at": time.time(),
        }

    def _handle_spare_master_deactivate(self, client_id: str, msg: dict) -> None:
        """
        备用主节点收到 SPARE_MASTER_DEACTIVATE 消息（新主节点已上线，退出暂代）。

        操作:
          1. 记录接管完成日志
          2. 退出「暂代主节点」模式
          3. 更新 DB 状态
        """
        data = msg.get("data", {})
        new_master_id = data.get("new_master_id", "")
        deactivate_id = data.get("deactivate_id", "")

        logger.info(
            f"🔔 收到备用主节点接管通知: new_master={new_master_id} 已上线, "
            f"退出暂代模式 (deactivate_id={deactivate_id})"
        )

        # 记录接管日志
        db = _get_db()
        try:
            if db and _db_available:
                db.append_spare_master_log(
                    direction="deactivated",
                    details={
                        "deactivate_id": deactivate_id,
                        "new_master_id": new_master_id,
                    },
                )
                db.set_config("spare_master_active", "false")
                db.set_config("pending_new_master_id", "")
                logger.info(f"备用主节点已退出暂代模式，新主节点 {new_master_id} 已接管")
        except Exception as e:
            logger.warning(f"备用主节点接管日志写入失败: {e}")

    # ---- 新主节点启动：向备用主节点发送接管通知 ----

    def deactivate_spare_master_on_startup(self) -> None:
        """
        新主节点启动时调用：检查是否有激活中的备用主节点，若有则发送接管通知。

        通过 TCP 服务器向备用主节点发送 SPARE_MASTER_DEACTIVATE，
        通知其退出暂代模式。
        """
        db = _get_db()
        if not db or not _db_available:
            return

        try:
            spare = db.get_spare_master()
            spare_active = db.get_config("spare_master_active", "false")
            pending_new_master = db.get_config("pending_new_master_id", "")

            # 仅当自己是 pending 的新主节点，且备用主节点处于激活状态时发送
            if (spare and spare.get("node_id")
                    and spare_active == "true"
                    and pending_new_master == NODE_ID):
                logger.info(
                    f"检测到本节点为转让目标，备用主节点 {spare['node_id']} 处于激活状态，"
                    "准备发送接管通知..."
                )

                # 等待 TCP 服务启动后发送（最多等 10s）
                waited = 0
                while (not self._tcp_server or not self._tcp_server._running) and waited < 10:
                    time.sleep(0.5)
                    waited += 0.5

                if not self._tcp_server or not self._tcp_server._running:
                    logger.warning("TCP 服务未在 10s 内就绪，跳过备用主节点接管通知")
                    return

                # 等备用主节点连接
                waited_conn = 0
                while spare['node_id'] not in (self._tcp_server.get_client_ids() if self._tcp_server else []) and waited_conn < 30:
                    time.sleep(1)
                    waited_conn += 1

                if spare['node_id'] not in (self._tcp_server.get_client_ids() if self._tcp_server else []):
                    logger.warning(f"备用主节点 {spare['node_id']} 未在 30s 内连接，跳过接管通知")
                    return

                deactivate_id = f"deactivate_{int(time.time() * 1000)}"
                deactivate_data = {
                    "deactivate_id": deactivate_id,
                    "new_master_id": NODE_ID,
                    "timestamp": time.time(),
                    "message": "新主节点已上线，请退出暂代模式。",
                }

                from tcp_comm import MessageType
                self._tcp_server.send_to_client(
                    spare['node_id'], deactivate_data, MessageType.SPARE_MASTER_DEACTIVATE
                )
                logger.info(
                    f"已向备用主节点 {spare['node_id']} 发送接管通知 "
                    f"(deactivate_id={deactivate_id})"
                )

                # 更新 DB 状态
                db.set_config("spare_master_active", "false")
                db.set_config("master_role_transferred", "false")
                db.set_config("pending_new_master_id", "")
                db.append_spare_master_log(
                    direction="deactivated",
                    details={
                        "deactivate_id": deactivate_id,
                        "new_master_id": NODE_ID,
                    },
                )

        except Exception as e:
            logger.warning(f"备用主节点接管通知失败: {e}")

    def get_spare_master(self) -> Optional[dict]:
        """获取当前备用主节点信息"""
        db = _get_db()
        if db and _db_available:
            try:
                spare = db.get_spare_master()
                if spare and spare.get("node_id"):
                    # 附加在线状态和激活状态
                    node_info = self.nodes.get(spare["node_id"])
                    spare["is_online"] = node_info.is_available() if node_info else False
                    spare["state"] = node_info.state.value if node_info else "unknown"
                    # 附加激活状态
                    try:
                        spare["is_active"] = db.get_config("spare_master_active", "false") == "true"
                    except Exception:
                        spare["is_active"] = False
                    return spare
            except Exception:
                pass
        return None

    def clear_spare_master(self) -> dict:
        """
        清除备用主节点指定（仅主节点可调用）。

        Returns:
            {status, message}
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可清除备用主节点"}

        db = _get_db()
        try:
            if db and _db_available:
                existing = db.get_spare_master()
                if existing:
                    db.clear_spare_master()
                    db.append_spare_master_log(
                        direction="undesignated",
                        details={
                            "master_id": NODE_ID,
                            "previous_spare": existing.get("node_id"),
                            "timestamp": time.time(),
                        },
                    )
                    logger.info(f"备用主节点已清除: {existing.get('node_id')}")
                    return {
                        "status": "ok",
                        "message": f"已取消 '{existing.get('node_id')}' 的备用主节点身份",
                    }
        except Exception as e:
            logger.warning(f"清除备用主节点失败: {e}")
        return {"status": "ok", "message": "备用主节点已清除（或无现有记录）"}

    def get_spare_master_logs(self) -> list:
        """获取备用主节点操作日志"""
        db = _get_db()
        if db and _db_available:
            try:
                return db.get_spare_master_logs()
            except Exception:
                pass
        return []

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

        Raises:
            RuntimeError: 节点未就绪
        """
        if not self.check_nodes_ready():
            raise RuntimeError("从节点未全部就绪，无法启动推理任务")

        task_id = f"task_{int(time.time() * 1000)}"
        task = InferenceTask(task_id=task_id, prompt=prompt)
        task.state = "running"
        task.start_time = time.time()

        with self._task_lock:
            self._current_task = task

        # 标记所有节点为忙碌
        with self._nodes_lock:
            for nid in self.nodes:
                if self.nodes[nid].role != NodeRole.MASTER:
                    self.update_node_state(nid, NodeState.BUSY)

        logger.info(
            "event=infer_task_start task_id=%s request_id=%s prompt_len=%d",
            task_id, request_id or "-", len(prompt),
        )

        # TODO: 触发流水线推理流程
        # 1. 主节点 Prefill → 中间特征 → TCP 发送至 client1
        # 2. client1 计算 → 转发特征至 client2
        # 3. client2 计算 → Decode 循环 → 回传结果

        return task_id

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
            if self._current_task:
                self._current_task.state = "done"
                self._current_task.end_time = time.time()
                self._current_task.result = result
                self._current_task.metrics = metrics or {}
                elapsed = self._current_task.end_time - self._current_task.start_time
                logger.info(
                    f"任务完成: {self._current_task.task_id}, "
                    f"耗时={elapsed:.2f}s, 结果长度={len(result)}"
                )

        # 恢复节点状态
        with self._nodes_lock:
            busy_nids = [
                nid for nid, info in self.nodes.items()
                if info.state == NodeState.BUSY
            ]
        for nid in busy_nids:
            self.update_node_state(nid, NodeState.ONLINE)

        if self.on_task_complete:
            self.on_task_complete(self._current_task)

    def on_task_error(self, error_msg: str) -> None:
        """
        任务异常回调。

        Args:
            error_msg: 错误描述
        """
        with self._task_lock:
            if self._current_task:
                self._current_task.state = "error"
                self._current_task.end_time = time.time()
                self._current_task.error_msg = error_msg
                logger.error(f"任务异常: {self._current_task.task_id} — {error_msg}")

        # 重置流水线
        self.stop_infer_task()

    # ================================================================
    # 状态查询
    # ================================================================

    def get_status(self) -> dict:
        """获取系统整体状态（含节点详情和 TCP 连接信息）"""
        self._refresh_http_client_states()
        node_status = {}
        with self._nodes_lock:
            for nid, info in self.nodes.items():
                node_status[nid] = info.to_dict()

        current_task = None
        if self._current_task:
            current_task = {
                "task_id": self._current_task.task_id,
                "state": self._current_task.state,
                "elapsed": time.time() - self._current_task.start_time,
            }

        # TCP 服务端状态
        tcp_info = None
        if self._tcp_server:
            tcp_info = {
                "host": self._tcp_server.host,
                "port": self._tcp_server.port,
                "connected_clients": self._tcp_server.get_client_ids(),
                "client_details": {
                    cid: self._tcp_server.get_client_info(cid)
                    for cid in self._tcp_server.get_client_ids()
                },
            }

        # 流水线状态
        pipeline_info = self._get_pipeline_status()

        # 请求队列状态
        queue_info = self.pipeline_queue.get_status()

        # TCP 客户端状态（从节点视角：到主节点的连接状态）
        tcp_client_info = None
        if self._effective_role() == "client":
            tcp_client = getattr(self, '_tcp_client', None)
            if tcp_client:
                tcp_client_info = {
                    "connected": getattr(tcp_client, 'is_registered', False),
                    "running": getattr(tcp_client, '_running', False),
                    "server_host": getattr(tcp_client, 'server_host', ''),
                    "server_port": getattr(tcp_client, 'server_port', 0),
                    "avg_rtt_ms": round(getattr(tcp_client, 'avg_rtt_ms', 0.0), 1),
                }

        return {
            "run_mode": RUN_MODE,
            "nodes": node_status,
            "current_task": current_task,
            "tcp_server": tcp_info,
            "tcp_client": tcp_client_info,
            "nodes_ready": self.check_nodes_ready(),
            "pipeline": pipeline_info,
            "pipeline_queue": queue_info,
        }

    def get_nodes(self) -> list:
        """获取所有节点详情列表"""
        self._refresh_http_client_states()
        with self._nodes_lock:
            return [info.to_dict() for info in self.nodes.values()]

    # L5: 多节点日志聚合
    def request_node_logs(self, node_id: str, limit: int = 100,
                          level: str = "", name: str = "",
                          timeout: float = 5.0) -> dict | None:
        """
        通过 TCP 向指定从节点拉取最近日志。

        Args:
            node_id: 目标节点 ID
            limit: 返回条数上限
            level: 日志级别过滤 (ERROR/WARNING/INFO/DEBUG)
            name: logger 名称过滤
            timeout: 等待超时秒数

        Returns:
            {node_id, logs, count, matched, buffer_size} 或 None（超时/错误）
        """
        import threading as _thr

        from tcp_comm import MessageType

        if not self._tcp_server:
            return None

        # 检查节点是否存在且在线
        with self._nodes_lock:
            if node_id not in self.nodes:
                return None
            if self.nodes[node_id].state != NodeState.ONLINE:
                return None

        # 准备信号（加锁保护，防止并发请求覆盖 Event）
        event = _thr.Event()
        with self._pending_log_lock:
            if node_id in self._pending_log_events:
                # 已有等待中的请求，避免 Event 被覆盖导致前一个请求永远超时
                logger.debug(
                    "event=log_aggregation_busy node_id=%s reason=pending_request",
                    node_id,
                )
                return None
            self._pending_log_events[node_id] = event
            self._pending_log_responses.pop(node_id, None)

        try:
            # 发送 LOG_REQUEST
            request_data = {
                "limit": limit,
                "level": level,
                "name": name,
                "node_id": node_id,
            }
            self._tcp_server.send_to_client(node_id, request_data, MessageType.LOG_REQUEST)

            # 等待响应
            signaled = event.wait(timeout)
            if signaled:
                with self._pending_log_lock:
                    result = self._pending_log_responses.pop(node_id, {})
                logger.info(
                    "event=log_aggregation_recv node_id=%s count=%d",
                    node_id, result.get("count", 0),
                )
                return result if result else None

            logger.warning(
                "event=log_aggregation_timeout node_id=%s timeout=%.1fs",
                node_id, timeout,
            )
            return None
        except Exception as e:
            logger.warning(
                "event=log_aggregation_failed node_id=%s error=%s",
                node_id, str(e)[:200],
            )
            return None
        finally:
            with self._pending_log_lock:
                self._pending_log_events.pop(node_id, None)
                # H1: 超时/异常时清理可能已到达的残留响应数据
                self._pending_log_responses.pop(node_id, None)

    def get_config(self) -> dict:
        """获取分布式配置信息（含当前节点角色、动态分层和实际局域网 IP）"""
        from config import (
            SERVER_IP, SERVER_PORT, HEARTBEAT_INTERVAL,
            TOTAL_MODEL_LAYERS,
            QUANT_TYPE, PAGE_SIZE, MAX_PAGE_NUM, MAX_SEQ_LEN,
        )
        # 优先使用运行时检测到的局域网 IP，回退到配置值
        server_ip = getattr(self, '_lan_ip', '') or SERVER_IP

        # 动态分层配置
        layers_info = self.get_layer_assignments()

        return {
            "run_mode": RUN_MODE,
            "node_role": self._effective_role(),
            "node_id": self.get_effective_node_id(),
            "max_nodes": self._max_nodes,
            "network": {
                "server_ip": server_ip,
                "server_port": SERVER_PORT,
                "heartbeat_interval_s": HEARTBEAT_INTERVAL,
            },
            "layers": layers_info,
            "distributed_inference": {
                "enabled": self.get_distributed_inference_enabled(),
            },
            "model": {
                "quant_type": QUANT_TYPE,
                "page_size": PAGE_SIZE,
                "max_page_num": MAX_PAGE_NUM,
                "max_seq_len": MAX_SEQ_LEN,
            },
            "task_stats": {
                node_id: {
                    "task_count": info.task_count,
                    "error_count": info.error_count,
                }
                for node_id, info in self.nodes.items()
            },
        }

    def connect_to_master(self, master_host: str, master_port: int) -> dict:
        """
        从节点主动连接主节点（由前端「连接主节点」按钮触发）。

        仅在 NODE_ROLE="client" 且有 TCP 客户端模块时可用。

        Args:
            master_host: 主节点 IP
            master_port: 主节点端口

        Returns:
            { status, node_id, master, message }
        """
        effective_role = self._effective_role()
        if effective_role != "client":
            return {"status": "denied", "reason": "仅从节点可以连接主节点"}

        try:
            import config as cfg

            # ★ 安全：从节点绝不能使用 "master" 作为 client_id
            configured_node_id = _configured_node_id()
            if not configured_node_id or configured_node_id == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
            else:
                node_id = configured_node_id

            def _run_first_connect_bootstrap(reason: str) -> None:
                nonlocal node_id, master_host, master_port
                from bootstrap import first_connect

                api_port = _bootstrap_api_port()
                logger.info(
                    "开始首次连接自动部署: reason=%s master_api=%s:%s",
                    reason, master_host, api_port,
                )
                bootstrap_result = first_connect(
                    master_api_host=master_host,
                    master_api_port=api_port,
                    node_id=node_id,
                    node_type=os.environ.get("QLH_NODE_TYPE", "pc"),
                )
                cluster = bootstrap_result.get("cluster", {})
                node = bootstrap_result.get("node", {})
                node_id = node.get("node_id") or node_id
                master_host = cluster.get("master_tcp_host") or master_host
                master_port = int(cluster.get("master_tcp_port") or master_port)
                logger.info(
                    "首次连接自动部署完成: node_id=%s master=%s:%s",
                    node_id, master_host, master_port,
                )

            if not getattr(cfg, "CLUSTER_SECRET", ""):
                try:
                    _run_first_connect_bootstrap("missing_secret")
                except Exception as e:
                    logger.error("首次连接自动部署失败: %s", e, exc_info=True)
                    return {
                        "status": "bootstrap_failed",
                        "reason": f"首次连接自动部署失败: {e}",
                    }

            from tcp_comm import TCPClient

            advertise_port = self._tcp_server.port if self._tcp_server else SERVER_PORT
            client = TCPClient(
                server_host=master_host,
                server_port=master_port,
                client_id=node_id,
                role="client",
                advertise_port=advertise_port,
            )
            # ★ 心跳回调：更新自身节点的心跳时间 + 同步 RTT 测量值
            # _sync_node_rtt 内部已有 _nodes_lock 保护
            client.on_heartbeat = lambda: self._sync_node_rtt(node_id, client)

            def _mark_local_node_online() -> None:
                # NodeInfo.address 表示本节点可被其他节点连接的服务端点，
                # 不能写成主节点地址；主节点地址由 tcp_client.server_host/server_port 表示。
                with self._nodes_lock:
                    if node_id in self.nodes:
                        self.nodes[node_id].state = NodeState.ONLINE
                        self.nodes[node_id].last_heartbeat = time.time()

            ok = client.connect(
                on_message=lambda msg: self._on_tcp_message("master", msg)
            )
            if ok:
                # 存储客户端引用，供分布式推理转发使用
                self._tcp_client = client
                _mark_local_node_online()

                # 更新运行时 node_id，避免 scheduler 模块导入常量滞后。
                _sync_runtime_node_config(node_id=node_id)
                # 若通过 activate_client_mode 切换而来，同步更新角色
                if getattr(self, '_role_override', None) == "client":
                    _sync_runtime_node_config(node_role="client")

                # 不在此处持久化到数据库 — 主节点收到 TCP 注册消息后
                # 会通过 _on_tcp_message → register_node() → db.upsert_node()
                # 统一写入，保证节点管理数据的一致性。
                # 从节点不应直接写入 master_host/master_port（数据库共享表）。

                # 主节点注册成功后，请求全量节点列表以同步管理面板
                try:
                    from tcp_comm import MessageType
                    client.send_data({"request": "node_list"}, MessageType.NODE_LIST_SYNC)
                except Exception:
                    pass

                logger.info(f"✅ 从节点 {node_id} 已连接到主节点 {master_host}:{master_port}")
                return {
                    "status": "connected",
                    "node_id": node_id,
                    "master": f"{master_host}:{master_port}",
                    "message": f"已成功注册到主节点 {master_host}:{master_port}",
                }
            reason = getattr(client, "last_register_error", "") or (
                f"TCP 连接失败 ({master_host}:{master_port})，请检查主节点地址和端口是否正确"
            )
            if _is_auth_register_failure(reason):
                try:
                    logger.info("TCP 注册认证失败，尝试刷新首次连接配置后重试: %s", reason)
                    _run_first_connect_bootstrap("auth_failed")
                    client = TCPClient(
                        server_host=master_host,
                        server_port=master_port,
                        client_id=node_id,
                        role="client",
                        advertise_port=advertise_port,
                    )
                    client.on_heartbeat = lambda: self._sync_node_rtt(node_id, client)
                    ok = client.connect(
                        on_message=lambda msg: self._on_tcp_message("master", msg)
                    )
                    if ok:
                        self._tcp_client = client
                        _mark_local_node_online()
                        _sync_runtime_node_config(node_id=node_id)
                        if getattr(self, '_role_override', None) == "client":
                            _sync_runtime_node_config(node_role="client")
                        try:
                            from tcp_comm import MessageType
                            client.send_data({"request": "node_list"}, MessageType.NODE_LIST_SYNC)
                        except Exception:
                            pass
                        logger.info(
                            "✅ 从节点 %s 刷新配置后已连接到主节点 %s:%s",
                            node_id, master_host, master_port,
                        )
                        return {
                            "status": "connected",
                            "node_id": node_id,
                            "master": f"{master_host}:{master_port}",
                            "message": f"已刷新自动部署配置并注册到主节点 {master_host}:{master_port}",
                        }
                    reason = getattr(client, "last_register_error", "") or reason
                except Exception as e:
                    logger.error("刷新首次连接配置后重试失败: %s", e, exc_info=True)
                    reason = f"{reason}; 刷新自动部署配置失败: {e}"

            return {
                "status": "failed",
                "reason": reason,
            }
        except Exception as e:
            logger.error(f"连接主节点 {master_host}:{master_port} 失败: {e}")
            return {"status": "error", "reason": f"{master_host}:{master_port} - {e}"}

    def forward_inference_to_master(self, message: str,
                                     max_new_tokens: int = 512,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9,
                                     show_thinking: bool = False,
                                     session_id: str = None,
                                     request_id: str = None,   # L5: 链路追踪
                                     timeout: float = 120.0) -> dict:
        """
        从节点将推理请求转发给主节点，并等待结果。

        仅从节点可调用，需要已通过 connect_to_master() 建立 TCP 连接。

        Args:
            message: 用户输入
            max_new_tokens: 最大新 token 数
            temperature: 温度
            top_p: top_p
            show_thinking: 是否启用深度思考展示
            session_id: 会话 ID（多会话支持）
            timeout: 等待结果超时秒数

        Returns:
            {status, content, metrics, error}
        """
        if self._effective_role() != "client":
            return {"status": "denied", "error": "仅从节点可转发推理请求"}

        tcp_client = getattr(self, '_tcp_client', None)
        if not tcp_client or not tcp_client._running:
            return {"status": "disconnected", "error": "未连接到主节点，请先建立连接"}

        try:
            from tcp_comm import MessageType

            # 清空之前的结果
            self._client_pending_results = {}

            # 发送推理请求
            infer_data = {
                "prompt": message,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "show_thinking": show_thinking,
                "session_id": session_id,
                "request_id": request_id,   # L5: 链路追踪
            }
            tcp_client.send_data(infer_data, MessageType.INFER_FORWARD)
            logger.info(
                "event=infer_forward task_id=n/a request_id=%s prompt_len=%d",
                request_id or "-", len(message),
            )

            # 等待结果（轮询 + 超时）
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._client_pending_results:
                    result = self._client_pending_results.popitem()[1]
                    logger.info(
                        f"📥 收到主节点推理结果: "
                        f"task={result.get('task_id', '')}, "
                        f"len={len(result.get('content', ''))}"
                    )
                    return {
                        "status": "ok",
                        "content": result.get("content", ""),
                        "metrics": result.get("metrics", {}),
                        "followups": result.get("followups", []),
                    }
                time.sleep(0.5)

            # 超时
            logger.warning(f"推理请求超时 ({timeout}s)")
            return {"status": "timeout", "error": f"等待主节点响应超时 ({timeout}s)"}

        except Exception as e:
            logger.error(f"转发推理请求失败: {e}")
            return {"status": "error", "error": str(e)}

    def get_invite_info(self) -> dict:
        """
        主节点获取邀请信息（供从节点连接使用）。

        优先使用运行时检测到的局域网 IP，回退到 SERVER_IP。
        主节点的连接信息同时持久化在数据库 cluster_config 中，
        从节点可通过 GET /api/cluster/discover 自动发现。

        Returns:
            { master_host, master_port, node_count, connected_clients, db_registered,
              mac_addresses, identity_verified, identity_reason }
        """
        # 使用运行时检测的 LAN IP 或已有的配置值
        lan_ip = getattr(self, '_lan_ip', '') or SERVER_IP
        port = self._tcp_server.port if self._tcp_server else SERVER_PORT
        macs = getattr(self, '_mac_addresses', [])

        # 检查是否已注册到数据库
        db_registered = False
        db = _get_db()
        if db and _db_available:
            try:
                db_host = db.get_config("master_host", "")
                db_registered = bool(db_host)
            except Exception:
                pass

        # Phase 2.1+: 锁保护迭代计数，防止并发修改
        with self._nodes_lock:
            online_count = sum(1 for n in self.nodes.values() if n.is_available())
            active_non_master = [
                n for n in self.nodes.values()
                if n.role != "master" and (n.is_available() or bool(n.address))
            ]
            capacity_used = 1 + len(active_non_master)
            total_records = len(self.nodes)

        return {
            "master_host": lan_ip,
            "master_port": port,
            "node_count": capacity_used,
            "total_node_records": total_records,
            "online_count": online_count,
            "max_nodes": self._max_nodes,
            "has_capacity": capacity_used < self._max_nodes,
            "connected_clients": (
                self._tcp_server.get_client_ids() if self._tcp_server else []
            ),
            "db_registered": db_registered,
            "mac_addresses": macs,
            "identity_verified": getattr(self, '_master_identity_verified', False),
            "identity_reason": getattr(self, '_master_identity_reason', ''),
        }

    def activate_client_mode(self, master_host: str = None, master_port: int = None) -> dict:
        """
        从失败的主节点模式自动切换到从节点模式。

        调用时机：MAC 地址不匹配时，若数据库中存在真正的主节点记录，
        表明本机并非主节点，应自动切换为从节点并尝试连接真正的主节点。

        Args:
            master_host: 主节点 IP（从 discover_master() 获取）
            master_port: 主节点端口

        Returns:
            { status, node_id, message }
        """
        import config as cfg

        logger.info("🔄 正在从失败的主节点模式切换到从节点模式...")

        # 1. 设置角色覆盖
        self._role_override = "client"
        _sync_runtime_node_config(node_role="client")

        # 2. 重新初始化为从节点
        with self._nodes_lock:
            self.nodes.clear()
        self.init_nodes()
        effective_id = self.get_effective_node_id()
        _sync_runtime_node_config(node_id=effective_id)

        # 3. 启动从节点健康监控
        self._start_client_health_monitor()

        logger.info(
            f"✅ 已切换到从节点模式: node_id={effective_id}, "
            f"role=client"
        )

        result = {
            "status": "switched",
            "node_id": effective_id,
            "message": f"已自动切换为从节点模式 (ID: {effective_id})",
        }

        # 4. 如果提供了主节点地址，尝试自动连接
        if master_host and master_port:
            logger.info(f"🔗 尝试自动连接主节点 {master_host}:{master_port}...")
            conn_result = self.connect_to_master(master_host, master_port)
            result["connect_result"] = conn_result
            if conn_result.get("status") == "connected":
                result["message"] += f"，已连接到主节点 {master_host}:{master_port}"
                logger.info(f"✅ 自动连接主节点成功")
            else:
                result["message"] += f"，自动连接主节点失败: {conn_result.get('reason', conn_result.get('status'))}"
                logger.warning(f"⚠️ 自动连接主节点失败: {conn_result}")
        else:
            result["message"] += "，未提供主节点地址，请手动连接"

        return result

    def _auto_switch_to_client(self, master_host: str, master_port: int) -> None:
        """
        后台线程：MAC 不匹配时自动切换到从节点模式并连接主节点。

        延迟 2 秒执行，确保 TCP 服务端完全就绪。
        """
        time.sleep(2)
        try:
            result = self.activate_client_mode(master_host, master_port)
            # 更新 api_server 中的 active_node_id
            try:
                from db import set_active_node_id
                set_active_node_id(self.get_effective_node_id())
            except Exception:
                pass
            logger.info(f"自动切换完成: {result.get('message', '')}")
        except Exception as e:
            logger.error(f"自动切换到从节点模式失败: {e}")

    def _auto_connect_on_startup(self) -> None:
        """
        后台线程：从节点启动后自动发现并连接主节点。

        启动后延迟 5 秒（给主节点 DB 心跳足够时间写入），
        然后尝试发现主节点并连接。
        """
        time.sleep(5)
        # 如果已经连接（例如通过 activate_client_mode），跳过
        tcp_client = getattr(self, '_tcp_client', None)
        if tcp_client and tcp_client._running:
            logger.info("已有活跃 TCP 连接，跳过启动自动连接")
            return

        try:
            discovery = self.discover_master()
            if discovery.get("found"):
                stale_note = "（心跳过期）" if discovery.get("stale") else ""
                host = discovery["master_host"]
                port = discovery["master_port"]
                logger.info(f"🔍 启动自动发现: 主节点 {host}:{port}{stale_note}，尝试连接...")
                result = self.connect_to_master(host, port)
                if result.get("status") == "connected":
                    logger.info(f"✅ 启动自动连接成功: {host}:{port}")
                else:
                    logger.info(f"启动自动连接失败: {result.get('reason', result.get('status'))}")
            else:
                logger.info("启动自动发现: 未找到可用主节点，稍后可通过前端手动连接")
        except Exception as e:
            logger.warning(f"启动自动连接异常: {e}")

    def get_effective_node_id(self) -> str:
        """
        返回当前节点的有效 ID。

        主节点 → "master"
        从节点 → 使用配置的 NODE_ID，若为 "master"（默认值）则自动生成
        """
        effective_role = self._effective_role()
        node_id = _configured_node_id()
        if effective_role == "client" and (not node_id or node_id == "master"):
            return f"client_{__import__('socket').gethostname()}"
        return node_id

    def get_my_role(self) -> dict:
        """
        获取当前节点的角色信息。

        用于前端判断是否显示后台管理 Tab：
        - master 节点：完全开放
        - client 节点：需开启"分布式推理优化"后才显示自己的后台

        从节点会自动查询数据库，尝试发现主节点连接信息。

        Returns:
            { node_role, node_id, is_master, max_nodes, master_discovery, ... }
        """
        effective_role = self._effective_role()
        _effective_id = self.get_effective_node_id()
        # Phase 2.1+: 锁保护读取
        with self._nodes_lock:
            my_info = self.nodes.get(_effective_id)
        result = {
            "node_role": effective_role,
            "node_id": _effective_id,
            "is_master": effective_role == "master",
            "is_client": effective_role == "client",
            "max_nodes": self._max_nodes,
            "run_mode": RUN_MODE,
            "my_node": my_info.to_dict() if my_info else None,
            "tcp_server_running": self._tcp_server is not None and self._tcp_server._running,
        }

        # 主节点：附加 MAC 身份验证状态
        if effective_role == "master":
            result["mac_addresses"] = getattr(self, '_mac_addresses', [])
            result["identity_verified"] = getattr(self, '_master_identity_verified', False)
            result["identity_reason"] = getattr(self, '_master_identity_reason', '')

        # 从节点：查询数据库以自动发现主节点
        if effective_role == "client":
            try:
                discovery = self.discover_master()
                result["master_discovery"] = discovery
            except Exception:
                result["master_discovery"] = {"found": False}

        return result

    def update_max_nodes(self, new_max: int) -> dict:
        """
        动态调整最大节点数量（仅 master 可调用）。

        仅修改容量上限，不预创建空槽位。从节点通过 TCP 注册动态加入。

        Args:
            new_max: 新的最大节点数 (>= 1, 包含 master)

        Returns:
            { status, max_nodes, nodes_added, nodes_removed, ... }
        """
        import config as cfg

        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可修改最大节点数"}

        if new_max < 1:
            return {"status": "invalid", "reason": "max_nodes 至少为 1 (仅 master)"}

        old_max = self._max_nodes
        if new_max == old_max:
            return {"status": "unchanged", "max_nodes": old_max}

        cfg.MAX_NODES = new_max
        self._max_nodes = new_max

        # 清理残留幽灵节点（旧代码扩容时预创建的空槽位，从未连接过）
        # Phase 2.1+: 锁保护迭代+删除操作
        with self._nodes_lock:
            phantoms = []
            for nid, node in list(self.nodes.items()):
                if (nid != "master" and node.role != "master"
                        and not node.address and not node.hostname
                        and not node.connected_at and not node.last_heartbeat
                        and node.state == NodeState.OFFLINE):
                    phantoms.append(nid)
            for pid in phantoms:
                del self.nodes[pid]
                logger.info(f"  清理幽灵节点: {pid}")
        if phantoms:
            self._layer_config_pushed.clear()

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.set_config("max_nodes", str(new_max))
                for pid in phantoms:
                    try:
                        db.delete_node(pid)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"max_nodes DB 持久化失败: {e}")

        logger.info(f"最大节点数已更新: {old_max} → {new_max}"
                    + (f" (清理幽灵: {phantoms})" if phantoms else ""))

        return {
            "status": "ok",
            "max_nodes": new_max,
            "old_max": old_max,
            "nodes_added": [],
            "nodes_removed": [],
            "total_nodes": len(self.nodes),
        }

    @property
    def tcp_server(self):
        """获取 TCP 服务端实例"""
        return self._tcp_server

    # ================================================================
    # 主节点数据库服务注册（从节点自动发现）
    # ================================================================

    def _register_master_in_db(self) -> None:
        """
        将主节点的连接信息写入数据库 cluster_config 表。

        写入内容:
        - 局域网 IP + 端口（可变动，从节点连接用）
        - 物理网卡 MAC 地址集合（不可变，身份验证用）
        - 心跳时间戳

        从节点启动时通过 discover_master() 查询该信息，
        即可在前端自动发现主节点，无需手动输入 IP。
        """
        db = _get_db()
        if not db or not _db_available:
            logger.warning("数据库不可用，跳过主节点注册（从节点将无法自动发现）")
            return

        lan_ip = getattr(self, '_lan_ip', '')
        if not lan_ip or lan_ip.startswith("127."):
            logger.warning(f"未检测到有效的局域网 IP ({lan_ip})，跳过数据库注册")
            return

        port = self._tcp_server.port if self._tcp_server else SERVER_PORT
        macs = getattr(self, '_mac_addresses', [])

        # 更新 master 自身节点的地址信息
        if "master" in self.nodes:
            self.nodes["master"].address = f"{lan_ip}:{port}"

        # 写入数据库 cluster_config（含 MAC 地址）
        try:
            db.register_master(
                host=lan_ip,
                port=port,
                node_id=NODE_ID,
                mac_addresses=macs if macs else None,
            )
            # 同步更新 nodes 表中的 master 记录
            try:
                db.upsert_node(
                    node_id="master", role="master", state="online",
                    address=f"{lan_ip}:{port}",
                    hostname=self.nodes.get("master", NodeInfo(node_id="master", role="master")).hostname,
                    network_type="localhost",
                )
            except Exception:
                pass
            mac_info = f"，MAC: {macs}" if macs else ""
            logger.info(f"✅ 主节点已注册到数据库: {lan_ip}:{port}{mac_info}（从节点可自动发现）")
        except Exception as e:
            logger.error(f"主节点数据库注册失败: {e}")

    def _verify_master_identity(self) -> None:
        """
        验证本机 MAC 地址是否与数据库中记录的主节点 MAC 匹配。

        验证逻辑:
        - DB 中尚无 MAC 记录 → 首次启动，身份为 "first_run"，后续 _register_master_in_db() 写入
        - 本机 MAC 与 DB 记录有交集 → 身份验证通过，"match"
        - 本机 MAC 与 DB 记录无交集 → 身份验证失败，"mac_mismatch"（可能是不同机器）

        验证结果存储在 self._master_identity_verified 和 self._master_identity_reason 中，
        前端可通过 get_my_role() 查询验证状态。
        """
        db = _get_db()
        if not db or not _db_available:
            self._master_identity_verified = True  # DB 不可用时不阻断
            self._master_identity_reason = "db_unavailable"
            logger.warning("数据库不可用，跳过主节点身份验证")
            return

        local_macs = self._mac_addresses

        try:
            result = db.verify_master_identity(local_macs)
        except Exception as e:
            logger.error(f"主节点身份验证异常: {e}")
            self._master_identity_verified = True  # 异常时不阻断
            self._master_identity_reason = "verify_error"
            return

        self._master_identity_verified = result["verified"]
        self._master_identity_reason = result["reason"]

        if result["reason"] == "first_run":
            logger.info("🔓 首次启动，数据库中尚无 MAC 记录，将自动写入本机 MAC 作为主节点身份标识")
        elif result["reason"] == "match":
            logger.info(f"🔒 主节点身份验证通过: MAC 匹配 {result['matched']}")
        elif result["reason"] == "mac_mismatch":
            logger.warning(
                f"⛔ 主节点身份验证失败！本机 MAC {result['local_macs']} "
                f"与数据库中记录 {result['db_macs']} 不匹配！"
                f"这可能意味着另一台机器正在尝试冒充主节点。"
                f"如需更换主节点机器，请在设置中使用「重置主节点身份」功能。"
            )
            # 注意：我们仍然允许启动（不阻断），但前端会显示警告
            # 因为也有可能是本机更换了网卡或禁用了某个网络接口
            # 如果是恶意冒充，数据库中的 MAC 记录不会被覆盖（除非调用 reset_master_identity）

    def _start_master_db_heartbeat(self) -> None:
        """
        启动主节点数据库心跳刷新线程。

        每 30 秒更新 cluster_config.master_last_seen，
        从节点据此判断主节点是否在线。
        """
        t = threading.Thread(target=self._master_db_heartbeat_loop, daemon=True)
        t.start()
        logger.info("主节点数据库心跳线程已启动（间隔 30s）")

    def _master_db_heartbeat_loop(self) -> None:
        """主节点数据库心跳循环（后台 daemon 线程）"""
        while self._running:
            try:
                db = _get_db()
                if db and _db_available:
                    db.update_master_heartbeat()
            except Exception as e:
                logger.debug(f"数据库心跳刷新失败: {e}")
            # 同时更新主节点自身的心跳时间戳（前端在线时长/心跳列显示）
            with self._nodes_lock:
                master_node = self.nodes.get("master")
                if master_node:
                    master_node.last_heartbeat = time.time()
            time.sleep(30)

    def discover_master(self) -> dict:
        """
        从数据库查询主节点的连接信息（供从节点自动发现）。

        从节点启动时和前端「自动发现」按钮调用此方法。
        如果 master_last_seen 超过 120 秒未更新，标记为 stale。

        Returns:
            {
                "found": bool,
                "master_host": str,
                "master_port": int,
                "stale": bool,
                "source": "database" | "config",
            }
        """
        db = _get_db()
        if db and _db_available:
            try:
                info = db.get_master_info()
                if info.get("found"):
                    info["source"] = "database"
                    return info
            except Exception as e:
                logger.warning(f"数据库查询主节点信息失败: {e}")

        # 数据库不可用时回退到 config 值
        import config as cfg
        if cfg.CLIENT_MASTER_HOST and cfg.CLIENT_MASTER_HOST != "192.168.x.x":
            return {
                "found": True,
                "master_host": cfg.CLIENT_MASTER_HOST,
                "master_port": cfg.CLIENT_MASTER_PORT,
                "stale": True,
                "source": "config",
            }

        return {"found": False, "source": "none"}

    # ================================================================
    # 分布式推理开关
    # ================================================================

    def get_distributed_inference_enabled(self) -> bool:
        """
        获取分布式推理开关状态。

        优先级: DB 记录 > 运行时变量 > config.py 默认值。
        """
        db = _get_db()
        if db and _db_available:
            try:
                return db.get_distributed_inference_enabled()
            except Exception:
                pass
        from config import DISTRIBUTED_INFERENCE_ENABLED
        return DISTRIBUTED_INFERENCE_ENABLED

    def set_distributed_inference_enabled(self, enabled: bool) -> dict:
        """
        设置分布式推理开关。

        - 持久化到 DB
        - 关闭时不影响已连接节点（不会主动断开），仅阻止新的分布式推理请求

        Returns:
            {status, enabled, message}
        """
        db = _get_db()
        if db and _db_available:
            try:
                db.set_distributed_inference_enabled(enabled)
            except Exception as e:
                return {"status": "error", "reason": f"DB 持久化失败: {e}"}

        logger.info(f"分布式推理已{'启用' if enabled else '禁用'}")
        return {
            "status": "ok",
            "enabled": enabled,
            "message": f"分布式推理已{'启用' if enabled else '禁用'}",
        }

    # ================================================================
    # 任务转发（从节点 → 主节点）
    # ================================================================

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
        show_thinking = data.get("show_thinking", False)
        session_id = data.get("session_id")
        request_id = data.get("request_id")   # L5: 链路追踪

        import threading as _thr

        def _run_inference():
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
                )

                content = pipeline_result.get("response", "")
                error = pipeline_result.get("error")
                metrics = pipeline_result.get("metrics", {})

                if error:
                    logger.warning(
                        f"⚠️ 流水线推理失败 → {client_id}: task={task_id}, "
                        f"error={error}"
                    )
                    self._send_infer_result(
                        client_id, task_id, content,
                        {**metrics, "error": error},
                    )
                    return

                # 保存到对话历史（主节点侧）
                if content:
                    try:
                        from db import save_message
                        save_message(
                            session_id=session_id or "default",
                            role="user",
                            content=prompt,
                        )
                        save_message(
                            session_id=session_id or "default",
                            role="assistant",
                            content=content,
                            metrics=metrics,
                        )
                    except Exception:
                        pass

                if not metrics.get("distributed_used"):
                    try:
                        self.record_task_complete(success=True)
                    except Exception:
                        pass

                self.complete_infer_task(task_id, content)
                self._send_infer_result(
                    client_id, task_id, content, metrics,
                )
                logger.info(
                    f"✅ 推理完成 → {client_id}: task={task_id}, "
                    f"len={len(content)}, engine={metrics.get('engine', '?')}"
                )

            except Exception as e:
                logger.error(f"转发推理执行失败: {e}", exc_info=True)
                self._send_infer_result(
                    client_id, "", "",
                    {"error": str(e)}
                )

        _thr.Thread(target=_run_inference, name=f"infer-{client_id}", daemon=True).start()

    def _send_infer_result(self, client_id: str, task_id: str,
                           content: str, metrics: dict = None,
                           thinking_content: str = None,
                           followups: list = None) -> None:
        """向从节点回传推理结果"""
        if self._tcp_server and self._tcp_server._running:
            try:
                from tcp_comm import MessageType
                result_data = {
                    "task_id": task_id,
                    "content": content,
                    "metrics": metrics or {},
                }
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

    # ================================================================
    # 分布式流水线推理（阶段 3：流水线调度引擎）
    # ================================================================

    def _handle_layer_config(self, client_id: str, data: dict) -> None:
        """
        从节点：收到主节点推送的分层配置 → 加载指定层范围。

        主节点广播的 assignments 格式:
            { "client1": {"start_layer":0, "end_layer":8, "has_embedding":true, "has_lm_head":false},
              "client2": {"start_layer":8, "end_layer":16, ...}, ... }

        从节点根据自身 node_id 取出对应配置，调用 load_layer_range() 加载层范围。
        """
        node_id = NODE_ID  # 本节点的 ID（从节点在注册时设置）
        if node_id not in data:
            logger.warning(f"分层配置中未找到本节点 {node_id}，跳过")
            return

        cfg = data[node_id]
        start = cfg.get("start_layer", 0)
        end = cfg.get("end_layer", 24)
        has_embed = cfg.get("has_embedding", False)
        has_lm = cfg.get("has_lm_head", False)

        logger.info(
            f"🔧 收到分层配置: 节点={node_id}, "
            f"Layer {start}-{end}, embed={has_embed}, lm_head={has_lm}"
        )

        try:
            import api_server as _api
            mgr = getattr(_api, 'model_manager', None)
            if mgr and mgr.is_loaded:
                # 如果已加载完整模型，重新加载指定层范围
                logger.info(f"🔄 重新加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                )
            elif mgr:
                # 模型尚未加载，先加载层范围
                logger.info(f"📥 首次加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                )
            else:
                logger.error("model_manager 不可用，无法加载层范围")
        except Exception as e:
            logger.error(f"加载层范围失败: {e}")

    def _handle_layer_forward(self, client_id: str, msg: dict) -> None:
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
        from tcp_comm import MessageType, serialize_tensor_fast

        data = msg.get("data", {})
        task_id = data.get("task_id", "unknown")
        step = data.get("step", 0)
        use_kv_cache = data.get("use_kv_cache", False)

        logger.info(
            f"🔬 收到层前向指令: task={task_id}, step={step}, from={client_id}, "
            f"kv_cache={'on' if use_kv_cache else 'off'}"
        )

        try:
            import api_server as _api
            from tcp_comm import deserialize_tensor_fast

            mgr = getattr(_api, 'model_manager', None)
            if not mgr or not mgr.is_loaded:
                self._send_layer_result(client_id, task_id, error="模型未加载")
                return

            # ---- 反序列化输入 ----
            input_ids = None
            hidden_states = None
            attention_mask = None
            position_ids = None

            if "input_ids" in data and data["input_ids"] is not None:
                input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # (seq_len,) → (1, seq_len)

            if "hidden_states" in data and data["hidden_states"] is not None:
                hs_bytes = data["hidden_states"]
                if isinstance(hs_bytes, str):
                    import base64
                    hs_bytes = base64.b64decode(hs_bytes)
                elif isinstance(hs_bytes, list):
                    hs_bytes = bytes(hs_bytes)
                hidden_states = deserialize_tensor_fast(hs_bytes)

            if "attention_mask" in data and data["attention_mask"] is not None:
                attention_mask = torch.tensor(data["attention_mask"], dtype=torch.long)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            if "position_ids" in data and data["position_ids"] is not None:
                position_ids = torch.tensor(data["position_ids"], dtype=torch.long)
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)

            # ---- KV Cache: 读取缓存的 past_key_values ----
            past_kv = None
            if use_kv_cache:
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        past_kv = self._kv_cache[task_id]
                if past_kv:
                    logger.debug(
                        f"📦 KV cache 命中: task={task_id}, "
                        f"layers={len(past_kv)}, "
                        f"seq_len={past_kv[0][0].shape[2] if past_kv else 0}"
                    )

            # ---- 执行前向传播 ----
            t_start = time.time()
            result = mgr.forward_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,  # 始终缓存 KV（prefill 构建，decode 更新）
            )
            elapsed_ms = (time.time() - t_start) * 1000
            self._record_local_pipeline_participation(task_id, success=True)

            # ---- KV Cache: 存储更新后的 past_key_values ----
            if result.get("past_key_values"):
                with self._kv_cache_lock:
                    self._kv_cache[task_id] = result["past_key_values"]
                kv_seq_len = result["past_key_values"][0][0].shape[2]
                logger.debug(
                    f"💾 KV cache 已更新: task={task_id}, "
                    f"seq_len={kv_seq_len}"
                )

            # ---- 序列化输出 ----
            response = {
                "task_id": task_id,
                "node_id": NODE_ID,
                "step": step,
                "metrics": {
                    "time_ms": round(elapsed_ms, 1),
                    "kv_cache": use_kv_cache,  # 标记是否使用了 KV cache
                    "kv_seq_len": (
                        result["past_key_values"][0][0].shape[2]
                        if result.get("past_key_values") else 0
                    ),
                    "memory_allocated_gb": (
                        round(torch.cuda.memory_allocated() / (1024**3), 2)
                        if torch.cuda.is_available() else 0
                    ),
                },
            }

            if "hidden_states" in result:
                # 中间节点：返回隐藏状态
                hs_cpu = result["hidden_states"].detach().cpu()
                response["hidden_states"] = serialize_tensor_fast(hs_cpu)
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
                response["logits"] = serialize_tensor_fast(logits_cpu)
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
                        from_node_id=NODE_ID,
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
                            f"{NODE_ID} → master → {chain_next['node_id']}"
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
                            from_node_id=NODE_ID,
                            target_node_id=chain_next["node_id"],
                            status="error",
                            error=error_msg,
                        )
            else:
                # 末节点（或无链配置）：发送 LAYER_RESULT 回主节点
                self._send_layer_result("master", task_id, result_data=response)

        except Exception as e:
            self._record_local_pipeline_participation(task_id, success=False)
            logger.error(f"层前向传播失败: task={task_id}, error={e}", exc_info=True)
            self._send_layer_result("master", task_id, error=str(e))
            self._send_chain_forward_ack(
                task_id=task_id,
                step=step,
                from_node_id=client_id,
                status="error",
                error=str(e),
            )

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
            from_node_id=client_id,
            status="received",
        )
        self._handle_layer_forward(client_id, msg)

    def _send_layer_result(self, client_id: str, task_id: str,
                           result_data: dict = None, error: str = None) -> bool:
        """从节点 → 主节点：发送层前向传播结果"""
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            if error:
                self._record_local_pipeline_participation(task_id, success=False)
            return False

        from tcp_comm import MessageType
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
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass
            return False

    def _send_chain_forward_ack(self, task_id: str, step: int,
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

        from tcp_comm import MessageType

        payload = {
            "task_id": task_id,
            "step": step,
            "node_id": NODE_ID,
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
        node_id = data.get("node_id", client_id)

        # ★ 中转请求：从节点直连失败 → 请主节点转发到目标节点
        relay_target = data.get("_relay_to")
        if relay_target:
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
                from tcp_comm import MessageType
                self._send_to_worker(relay_target, relay_data,
                                     MessageType.CHAIN_FORWARD)
                self._handle_chain_forward_ack(
                    relay_target,
                    {
                        "data": {
                            "task_id": task_id,
                            "step": data.get("step", -1),
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

    def _all_pipeline_nodes_ready(self) -> bool:
        """
        检查所有流水线节点是否就绪（预入队快速检查）：
        1. 节点已注册
        2. 节点在线（state == ONLINE）
        3. TCP 连接正常
        4. 心跳新鲜（< 10s，防止"僵尸"节点）
        5. 层配置已成功推送
        """
        if not self._tcp_server or not self._tcp_server._running:
            return False

        assignments = self.get_layer_assignments()
        pipeline_nodes = [
            a for a in assignments.get("assignments", [])
            if a.get("node_id") != "master"
        ]

        if not pipeline_nodes:
            logger.info("没有流水线从节点（单节点模式），跳过流水线检查")
            return False

        # Phase 2.1+: 快照避免循环中并发修改
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)

        for node in pipeline_nodes:
            node_id = node["node_id"]
            node_info = nodes_snapshot.get(node_id)
            if not node_info:
                logger.warning(f"流水线节点 {node_id} 未注册")
                return False
            if not node_info.is_available():
                logger.warning(f"流水线节点 {node_id} 不在线")
                return False
            if node_id not in (self._tcp_server.clients if self._tcp_server else {}):
                logger.warning(f"流水线节点 {node_id} TCP 未连接")
                return False
            # ★ 心跳新鲜度：超过 10 秒无心跳视为失联
            heartbeat_age = time.time() - node_info.last_heartbeat
            if heartbeat_age > 10:
                logger.warning(
                    f"流水线节点 {node_id} 心跳过期 "
                    f"({heartbeat_age:.1f}s > 10s)，视为失联"
                )
                return False
            # ★ 层配置：必须已成功推送
            if node_id not in self._layer_config_pushed:
                logger.warning(f"流水线节点 {node_id} 尚未收到层配置")
                return False

        logger.info(f"✅ 所有流水线节点就绪: {[n['node_id'] for n in pipeline_nodes]}")
        return True

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
            if node_id not in (self._tcp_server.clients if self._tcp_server else {}):
                return False, f"节点 {node_id} TCP 连接已断开"

            # 心跳新鲜度
            heartbeat_age = time.time() - node_info.last_heartbeat
            if heartbeat_age > 10:
                return False, (
                    f"节点 {node_id} 心跳过期 "
                    f"({heartbeat_age:.1f}s > 10s)"
                )

            # 层配置已推送
            if node_id not in self._layer_config_pushed:
                return False, f"节点 {node_id} 尚未收到层配置推送"

        logger.info(
            f"✅ 二次就绪检查通过: "
            f"{' → '.join(n['node_id'] for n in pipeline_nodes)}"
        )
        return True, "ok"

    def _broadcast_pipeline_abort(self, pipeline_nodes: list, task_id: str,
                                   reason: str) -> None:
        """向所有流水线节点广播 PIPELINE_ABORT（清理各节点 + master 本地 KV cache）。"""
        from tcp_comm import MessageType
        failed_nodes = []
        for n in pipeline_nodes:
            node_id = n.get("node_id")
            if not node_id:
                continue
            try:
                self._send_to_worker(
                    node_id,
                    {"task_id": task_id, "reason": reason},
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
        from tcp_comm import TCPClient, MessageType

        addr = self._get_node_address(target_node_id)
        if not addr:
            logger.error(f"无法获取节点 {target_node_id} 的地址")
            return False

        try:
            t0 = time.time()
            client = TCPClient(
                server_host=addr["host"],
                server_port=addr["port"],
                client_id=NODE_ID,
                role="client",
            )
            ok = client.connect()
            if not ok:
                logger.error(f"链式转发: 连接 {target_node_id} ({addr['host']}:{addr['port']}) 失败")
                return False

            client.send_data(data, MessageType.CHAIN_FORWARD)
            client.disconnect()
            elapsed_ms = (time.time() - t0) * 1000
            hs_shape = data.get("hidden_shape", "?")
            logger.debug(
                f"🔗 链式转发: {NODE_ID} → {target_node_id} "
                f"hidden_states={hs_shape}, time={elapsed_ms:.0f}ms"
            )
            return True
        except Exception as e:
            logger.error(f"链式转发到 {target_node_id} 失败: {e}")
            return False

    def _send_to_worker(self, worker_id: str, data: dict,
                        msg_type=None) -> None:
        """主节点 → 从节点：发送消息"""
        from tcp_comm import MessageType
        if msg_type is None:
            msg_type = MessageType.LAYER_FORWARD
        if not self._tcp_server or not self._tcp_server._running:
            raise ConnectionError("TCP 服务端未运行")
        self._tcp_server.send_to_client(worker_id, data, msg_type)

    def _wait_for_layer_result(self, task_id: str, node_ids,
                               timeout: float = 30.0,
                               ack_node_ids: list = None,
                               ack_step: int = None,
                               ack_timeout: float = None) -> Optional[dict]:
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
                                        ack_timeout: float) -> Optional[dict]:
        """等待流水线结果；兼容测试或旧扩展中替换掉的三参数等待函数。"""
        try:
            return self._wait_for_layer_result(
                task_id,
                node_ids,
                timeout=timeout,
                ack_node_ids=ack_node_ids,
                ack_step=ack_step,
                ack_timeout=ack_timeout,
            )
        except TypeError as e:
            if "ack_node_ids" not in str(e):
                raise
            logger.debug(
                "_wait_for_layer_result 不支持 ACK 参数，退回旧签名调用",
                exc_info=True,
            )
            return self._wait_for_layer_result(task_id, node_ids, timeout)

    # ================================================================
    # 协同抢占辅助方法 (Phase 2)
    # ================================================================

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
        if not PIPELINE_PREEMPT_ENABLED or self._preempt_disabled:
            return False
        if self._preempting:  # ★ 防嵌套：Q0 内部不触发二次抢占
            return False
        if current_step < PIPELINE_PREEMPT_MIN_TOKENS:
            return False
        if self._preempt_last_time > 0:
            if time.time() - self._preempt_last_time < PIPELINE_PREEMPT_MIN_INTERVAL:
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

        if overhead_ms > PIPELINE_PREEMPT_MAX_OVERHEAD_MS:
            self._preempt_disabled = True
            logger.warning(
                f"⚠️ 抢占开销 {overhead_ms:.1f}ms 超过阈值 "
                f"({PIPELINE_PREEMPT_MAX_OVERHEAD_MS}ms)，已禁用后续抢占"
            )

        # 同步到 PipelineQueue（get_queue_detail 读取此处）
        with self.pipeline_queue._lock:
            self.pipeline_queue._preempt_count = self._preempt_count
            self.pipeline_queue._preempt_total_overhead_ms = self._preempt_total_overhead_ms
            self.pipeline_queue._last_preempt_time = self._preempt_last_time

    def run_pipeline(self, prompt: str, max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.9,
                     session_id: str = None,
                     _stream_callback=None) -> dict:
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
        import uuid
        import api_server as _api
        from tcp_comm import MessageType, deserialize_tensor_fast, serialize_tensor_fast

        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.tokenizer:
            return {"response": "", "error": "模型未加载"}

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

        # master 参与时，先在本地加载 Embedding + 首段 Transformer 层。
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
                        has_lm_head=False,
                    )
                else:
                    mgr.load_layer_range(
                        master_assignment["start_layer"],
                        master_assignment["end_layer"],
                        has_embedding=master_assignment.get("has_embedding", True),
                        has_lm_head=False,
                    )
            except Exception as e:
                logger.error(f"❌ 主节点本地层范围加载失败: {e}", exc_info=True)
                return {"response": "", "error": f"主节点本地层范围加载失败: {e}"}

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

        full_chain = ([master_assignment] if master_participates else []) + pipeline_nodes
        logger.info(
            f"🚀 启动流水线推理: prompt_len={len(prompt)}, "
            f"max_tokens={max_new_tokens}, worker数={len(pipeline_nodes)}, "
            f"顺序: {' → '.join(n['node_id'] for n in full_chain)}, "
            f"master_local={'✅' if master_participates else '❌'}, KV Cache: ✅"
        )

        # ---- Step 2: Tokenize ----
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # (1, prompt_len)
        attention_mask = inputs.get("attention_mask")
        prompt_len = input_ids.shape[1]

        # ---- Step 3: 自回归生成 ----
        task_id = uuid.uuid4().hex[:12]
        generated_ids = []
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
            # ---- Phase 2: 协同抢占检查 ----
            # 在每个 decode 步边界检测 Q0 任务，若存在则执行内联抢占。
            # Prefill (step=0) 不抢占——此时尚未生成任何 token。
            if (step > 0
                    and PIPELINE_PREEMPT_ENABLED
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
                    local_input_ids = input_ids if is_prefill else torch.tensor(
                        [[new_token_id]], dtype=torch.long
                    )
                    local_attention_mask = attention_mask if is_prefill else None

                    t_master = time.time()
                    local_result = mgr.forward_layers(
                        input_ids=local_input_ids,
                        attention_mask=local_attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    master_elapsed_ms = (time.time() - t_master) * 1000
                    if local_result.get("past_key_values"):
                        with self._kv_cache_lock:
                            self._kv_cache[task_id] = local_result["past_key_values"]
                    if "hidden_states" not in local_result:
                        raise RuntimeError("主节点首段未返回 hidden_states")
                    hs_cpu = local_result["hidden_states"].detach().cpu()
                    import base64 as _b64
                    forward_data["hidden_states"] = _b64.b64encode(
                        serialize_tensor_fast(hs_cpu)
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
                timeout=PIPELINE_STEP_TIMEOUT,
                ack_node_ids=[n["node_id"] for n in pipeline_nodes[1:]] if has_chain else [],
                ack_step=step,
                ack_timeout=min(5.0, max(1.0, PIPELINE_STEP_TIMEOUT / 6)),
            )
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

            # 提取 logits（链式模式下仅末节点返回 logits）
            if "logits" in result and result["logits"] is not None:
                logits_data = result["logits"]
                if isinstance(logits_data, bytes):
                    logits = deserialize_tensor_fast(logits_data).to(device=device)
                elif torch is not None and isinstance(logits_data, torch.Tensor):
                    logits = logits_data.to(device=device)
                else:
                    step_error = f"未知 logits 类型: {type(logits_data).__name__}"
                    logger.error(step_error)
            else:
                step_error = "末节点未返回 logits"
                logger.error(step_error)

            if step_error:
                # ★ 统一中止路径：广播 ABORT → 清理各节点 KV cache → 返回错误
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # ---- Step 4: 从 logits 采样下一个 token ----
            # logits shape: prefill=(1, prompt_len, vocab), decode=(1, 1, vocab)
            # Phase 5 review H1: clamp temperature 防止除零导致 NaN
            safe_temperature = max(temperature, 1e-8)
            next_logits = logits[:, -1, :] / safe_temperature
            probs = torch.softmax(next_logits, dim=-1)

            # top-p (nucleus) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p).float()
            cutoff[..., 1:] = cutoff[..., :-1].clone()  # 右移一位
            cutoff[..., 0] = 0  # 始终保留概率最高的 token
            filtered_probs = sorted_probs * (1 - cutoff)
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            new_token_id = sorted_indices[0, torch.multinomial(filtered_probs, 1)].item()

            # 检查 EOS
            if new_token_id == tokenizer.eos_token_id:
                logger.info(f"🏁 EOS token 生成于 step {step}")
                break

            generated_ids.append(new_token_id)

            # ★ 流式回调：每生成一个 token 立即推送
            if _stream_callback:
                new_token_text = tokenizer.decode([new_token_id])
                _stream_callback({"token": new_token_text})

            # 更新完整序列仅用于最终解码（不再发送给首节点）
            new_token_tensor = torch.tensor([[new_token_id]], dtype=torch.long)
            full_input_ids = torch.cat([full_input_ids, new_token_tensor], dim=1)

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
        try:
            for n in pipeline_nodes:
                self._send_to_worker(
                    n["node_id"],
                    {"task_id": task_id},
                    MessageType.PIPELINE_DONE,
                )
        except Exception as e:
            logger.warning(f"广播 PIPELINE_DONE 失败: {e}")

        # ★ 清理 master 自身 KV cache（master_participates 路径会产生本地缓存）
        with self._kv_cache_lock:
            if task_id in self._kv_cache:
                del self._kv_cache[task_id]

        # ---- Step 6: 解码结果 ----
        if generated_ids:
            full_ids = torch.cat([
                input_ids.squeeze(0),
                torch.tensor(generated_ids, dtype=torch.long)
            ], dim=0)
            response_text = tokenizer.decode(full_ids, skip_special_tokens=True)
            new_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            response_text = tokenizer.decode(
                input_ids.squeeze(0), skip_special_tokens=True
            )
            new_text = ""

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

        def on_token(event):
            if "done" in event:
                callback_called.set()
            q.put(event)

        def _run():
            try:
                result = self.run_pipeline(
                    prompt, _stream_callback=on_token, **kwargs
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

        while True:
            event = q.get()
            yield event
            if "done" in event:
                break

    # ================================================================
    # 流水线请求队列集成（Phase 4 — 多请求排队）
    # ================================================================

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
            if not self._all_pipeline_nodes_ready():
                logger.warning("流水线节点不可用，队列任务回退到全模型推理")
                # ★ H1 修复: 保持 lock_held=True，回退推理在锁保护下执行（防止 GPU 并发）
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason="queue_pipeline_nodes_not_ready",
                    **kwargs,
                )
            result = self.run_pipeline(prompt, **kwargs)
            if result.get("error"):
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
        # 同时检查模型是否已加载，未加载时走回退路径（给出明确错误）。
        import api_server as _api
        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.is_loaded:
            logger.warning("模型未加载，无法执行流水线推理")
            return self._run_full_model_inference(
                prompt,
                _fallback_reason="model_not_loaded_for_pipeline",
                **kwargs,
            )
        engine_type = getattr(mgr, '_engine_type', '')
        if engine_type and engine_type != 'pytorch':
            logger.info(
                f"引擎类型为 {engine_type}，不支持流水线层拆分，"
                f"使用全模型推理"
            )
            return self._run_full_model_inference(
                prompt,
                _fallback_reason=f"engine {engine_type} does not support layer-split pipeline",
                **kwargs,
            )

        # ---- 自动回退：节点不可用 → 全模型推理 ----
        try:
            pipeline_ready = self._all_pipeline_nodes_ready()
        except Exception:
            pipeline_ready = False

        if not pipeline_ready:
            logger.warning("部分流水线节点未就绪，回退到全层主节点模式")
            # 回退仍会执行完整模型推理，必须与其他 GPU 推理共享同一把锁。
            # 这里阻塞等待，避免锁被占用时直接绕过互斥保护。
            self._inference_lock.acquire()
            try:
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason="pipeline_nodes_not_ready",
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
                task_id = self.pipeline_queue.enqueue(prompt=prompt, **kwargs)
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
                task_id, timeout=kwargs.get('_queue_timeout', PIPELINE_TIMEOUT)
            )
            if result.get("status") == "done":
                payload = result.get("result", {})
                if isinstance(payload, dict) and payload.get("error"):
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
                return {"response": "", "error": f"排队超时 ({PIPELINE_TIMEOUT}s)"}
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
        import api_server as _api

        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.is_loaded:
            return {"response": "", "error": "模型未加载"}

        _stream_callback = kwargs.pop('_stream_callback', None)
        fallback_reason = kwargs.pop('_fallback_reason', '') or 'pipeline_fallback_full_model'

        # ★ 若 master 刚执行过流水线裁剪（layer_range != None），
        #   需要先重新加载完整模型，否则 chat()/chat_stream() 会因
        #   缺 Embedding/LM Head 而报错（如 RuntimeError: 缺少 lm_head）。
        try:
            ensure_full = getattr(mgr, 'ensure_full_model', None)
            if callable(ensure_full):
                ensure_full()
        except Exception as e:
            logger.warning(f"完整模型重载失败（将继续尝试推理）: {e}")

        try:
            messages = [{"role": "user", "content": prompt}]

            if _stream_callback:
                # 流式路径：逐 token 推送
                full_text_parts = []
                t0 = time.time()
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    if chunk:
                        full_text_parts.append(chunk)
                        _stream_callback({"token": chunk})
                response_text = "".join(full_text_parts)
                elapsed = time.time() - t0
                metrics = {
                    "engine": getattr(mgr, '_engine_type', 'unknown') or 'unknown',
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
                _stream_callback({"done": True, "response": response_text, "metrics": metrics})
            else:
                result = mgr.chat(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                response_text = result.get("content", "")
                usage = result.get("usage", {}) or {}
                completion_tokens = usage.get("completion_tokens", 0)
                metrics = {
                    "engine": getattr(mgr, '_engine_type', 'unknown') or 'unknown',
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
                "thinking": "" if _stream_callback else result.get("thinking_content", ""),
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
        import api_server as _api

        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.is_loaded:
            yield {"done": True, "error": "模型未加载"}
            return

        max_new_tokens = kwargs.pop('max_new_tokens', 512)
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)

        q = queue.Queue()
        full_text_parts = []
        error_info = [None]
        metrics_info = [{}]

        def _run():
            try:
                messages = [{"role": "user", "content": prompt}]
                t0 = time.time()
                token_count = 0
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    if chunk:
                        full_text_parts.append(chunk)
                        token_count += 1
                        q.put({"token": chunk})
                elapsed = time.time() - t0
                metrics_info[0] = {
                    "mode": "single_streaming",
                    "engine": "pytorch",
                    "chunks": token_count,
                    "elapsed_seconds": round(elapsed, 3),
                }
            except Exception as e:
                logger.error(f"单机流式推理异常: {e}", exc_info=True)
                error_info[0] = str(e)
            finally:
                q.put(None)  # sentinel

        _thr.Thread(target=_run, name="full-model-stream", daemon=True).start()

        while True:
            event = q.get()
            if event is None:
                break
            yield event

        response_text = "".join(full_text_parts)
        if error_info[0]:
            yield {
                "done": True,
                "error": error_info[0],
                "response": response_text,
                "metrics": metrics_info[0],
            }
        else:
            yield {
                "done": True,
                "response": response_text,
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
        import api_server as _api

        # 检查引擎兼容性
        mgr = getattr(_api, 'model_manager', None)
        engine_ok = mgr is not None and getattr(mgr, '_engine_type', '') == 'pytorch'

        # 获取分层配置
        layer_info = self.get_layer_assignments()
        workers = [
            a for a in layer_info.get("assignments", [])
            if a.get("node_id") != "master"
        ]
        workers.sort(key=lambda a: a.get("start_layer", 0))

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
            worker_status.append({
                "node_id": nid,
                "online": is_online,
                "layer_range": [w.get("start_layer", 0), w.get("end_layer", 24)],
                "has_embedding": w.get("has_embedding", False),
                "has_lm_head": w.get("has_lm_head", False),
            })

        available = (
            engine_ok
            and RUN_MODE == "distributed"
            and self._effective_role() == "master"
            and len(workers) > 0
        )
        active = available and online_count == len(workers) and online_count > 0
        degraded = available and online_count > 0 and online_count < len(workers)

        return {
            "available": available,
            "active": active,
            "degraded": degraded,
            "worker_count": len(workers),
            "online_worker_count": online_count,
            "engine_compatible": engine_ok,
            "workers": worker_status,
        }

    def reset_master_identity(self) -> dict:
        """
        重置主节点身份标识（仅主节点可调用）。

        用于以下场景：
        - 更换主节点机器（新机器的 MAC 与 DB 中记录不匹配）
        - 主节点更换了网卡
        - 需要清除旧的 MAC 记录重新绑定

        调用后下一次启动时将自动记录新的 MAC 地址。
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可重置身份标识"}

        db = _get_db()
        if not db or not _db_available:
            return {"status": "error", "reason": "数据库不可用"}

        try:
            db.reset_master_identity(new_macs=None)  # 清除旧 MAC，下次启动重新记录
            logger.warning("⚠️ 主节点身份标识已重置，下次启动将重新记录 MAC")
            return {
                "status": "ok",
                "message": "主节点身份已重置。请重启后端服务以重新绑定 MAC 地址。",
            }
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def manual_register_node(self, node_id: str, hostname: str = "",
                             address: str = "", network_type: str = "unknown",
                             node_type: str = "pc") -> dict:
        """
        主节点手动注册一个从节点（无需 TCP 连接）。

        用于以下场景：
        - 管理员提前在后台录入从节点信息
        - 从节点尚未来得及通过 TCP 连接
        - 保留节点槽位供后续 TCP 激活

        手动注册的节点初始状态为 offline，待从节点 TCP 连接后自动变为 online。

        Args:
            node_id: 节点标识（如 "jetson-nano-01"）
            hostname: 主机名
            address: 预留地址（可选）
            network_type: 网络类型

        Returns:
            { status, node_id, message }
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可手动注册从节点"}

        if node_id == "master":
            return {"status": "invalid", "reason": "不能注册名为 'master' 的节点"}

        with self._nodes_lock:
            if node_id in self.nodes:
                existing = self.nodes[node_id]
                if existing.role == "master":
                    return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可覆盖"}
                if (existing.is_available() or existing.connected_at or existing.last_heartbeat
                        or existing.device_info or existing.model_sha256):
                    return {
                        "status": "conflict",
                        "node_id": node_id,
                        "reason": "节点已通过自动注册建立真实连接记录，请先注销/删除后再手动重建",
                        "state": existing.state.value,
                    }
                existing.hostname = hostname or existing.hostname or node_id
                existing.address = address
                existing.network_type = network_type
                existing.node_type = node_type
                state_value = existing.state.value
                hostname_snapshot = existing.hostname
            else:
                existing = None

            if existing is not None:
                pass  # 更新路径：锁内修改完成，退出锁后写 DB
            else:
                # 检查容量：只统计在线/已注册节点（离线/幽灵不占位）
                online_non_master = [
                    n for n in self.nodes.values()
                    if n.role != "master" and (n.is_available() or n.address)
                ]
                if len(online_non_master) >= self._max_nodes - 1:
                    return {"status": "full", "reason": f"已达到最大在册从节点数量 ({self._max_nodes - 1})"}

                # 创建节点（初始 offline）
                node = NodeInfo(
                    node_id=node_id,
                    role=NodeRole.CLIENT,
                    node_type=node_type,
                    state=NodeState.OFFLINE,
                    hostname=hostname or node_id,
                    address=address,
                    network_type=network_type,
                )
                self.nodes[node_id] = node

        if existing is not None:
            # 更新路径：锁外写 DB
            db = _get_db()
            if db and _db_available:
                try:
                    db.upsert_node(
                        node_id=node_id, role="client", node_type=node_type,
                        state=state_value,
                        address=address, hostname=hostname_snapshot,
                        network_type=network_type,
                    )
                except Exception as e:
                    logger.warning(f"手动更新节点 DB 持久化失败: {e}")
            logger.info(
                f"📝 手动注册节点已更新: {node_id} type={node_type} "
                f"(hostname={hostname_snapshot}, addr={address}, state={state_value})"
            )
            return {"status": "updated", "node_id": node_id,
                    "message": f"节点 '{node_id}' 已更新 (state={state_value})",
                    "state": state_value}

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.upsert_node(
                    node_id=node_id, role="client", node_type=node_type,
                    state="offline",
                    address=address, hostname=node.hostname,
                    network_type=network_type,
                )
            except Exception as e:
                logger.warning(f"手动注册节点 DB 持久化失败: {e}")

        logger.info(f"📝 主节点手动注册从节点: {node_id} type={node_type} (hostname={hostname}, addr={address})")
        return {
            "status": "registered",
            "node_id": node_id,
            "message": f"节点 '{node_id}' 已手动注册，等待 TCP 连接激活",
            "state": "offline",
        }

    def delete_node(self, node_id: str) -> dict:
        """
        删除离线节点记录（不同于 deregister：deregister 仅标记 offline）。

        仅允许删除非 master 且当前不在线的节点，常用于移除手动注册的
        Android/离线占位节点。
        """
        if self._effective_role() != "master":
            return {"status": "denied", "reason": "仅主节点可删除节点"}
        if node_id == "master":
            return {"status": "invalid", "reason": "不能删除主节点"}
        # Phase 2.1+: 原子化 get+检查+pop，防止并发修改
        with self._nodes_lock:
            node = self.nodes.get(node_id)
            if node is None:
                return {"status": "not_found", "reason": f"节点 '{node_id}' 不存在"}
            if node.is_available():
                return {"status": "online", "reason": "节点在线，请先注销后删除"}

            old_node = self.nodes.pop(node_id)

        self._layer_config_pushed.discard(node_id)

        db = _get_db()
        if db and _db_available:
            try:
                db.delete_node(node_id)
                db.set_layer_assignments({})
            except Exception as e:
                logger.warning(f"删除节点 DB 持久化失败: {e}")

        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(node_id, "remove", old_node)

        logger.info(
            f"🗑️ 节点已删除: {node_id} type={old_node.node_type} "
            f"hostname={old_node.hostname}"
        )
        return {"status": "deleted", "node_id": node_id,
                "message": f"节点 '{node_id}' 已删除"}

    def check_master_health(self) -> dict:
        """
        检查主节点是否在线。

        两级检测（从快到慢）:
          1. 本地 TCP 连接状态 — 秒级（~3s 心跳超时即可感知）
          2. 数据库心跳时间戳 — 120s 超时（兜底，防止 TCP 假连接）

        Returns:
            {
                "master_online": bool,
                "last_seen_seconds_ago": float | None,
                "stale": bool,
                "master_host": str,
                "master_port": int,
                "source": str,       # "tcp" | "database" | "self" | "db_unavailable"
                "tcp_connected": bool | None,  # 本地 TCP 是否连通
            }
        """
        # ★ 第 1 级：本地 TCP 连接状态（秒级感知断连）
        tcp_client = getattr(self, '_tcp_client', None)
        tcp_connected = (
            tcp_client is not None
            and getattr(tcp_client, '_running', False)
            and getattr(tcp_client, 'is_registered', False)
            and getattr(tcp_client, 'sock', None) is not None
        )

        db = _get_db()
        if not db or not _db_available:
            return {
                "master_online": False,
                "last_seen_seconds_ago": None,
                "stale": True,
                "master_host": "",
                "master_port": 0,
                "source": "db_unavailable",
                "tcp_connected": tcp_connected,
            }

        try:
            info = db.get_master_info()
            if not info.get("found"):
                return {
                    "master_online": False,
                    "last_seen_seconds_ago": None,
                    "stale": True,
                    "master_host": "",
                    "master_port": 0,
                    "source": "not_found",
                    "tcp_connected": tcp_connected,
                }

            now = time.time()
            last_seen = info.get("last_seen", 0)
            ago = now - last_seen if last_seen > 0 else None
            stale = info.get("stale", True)

            # ★ 如果本地 TCP 已断开，即使 DB 心跳未过期也立即报告离线
            if not tcp_connected and self._effective_role() == "client":
                return {
                    "master_online": False,
                    "last_seen_seconds_ago": round(ago, 1) if ago else None,
                    "stale": stale,
                    "master_host": info["master_host"],
                    "master_port": info["master_port"],
                    "source": "tcp_disconnected",
                    "tcp_connected": False,
                }

            return {
                "master_online": not stale,
                "last_seen_seconds_ago": round(ago, 1) if ago else None,
                "stale": stale,
                "master_host": info["master_host"],
                "master_port": info["master_port"],
                "source": "database",
                "tcp_connected": tcp_connected,
            }
        except Exception as e:
            logger.warning(f"主节点健康检查失败: {e}")
            return {
                "master_online": False,
                "last_seen_seconds_ago": None,
                "stale": True,
                "master_host": "",
                "master_port": 0,
                "source": "error",
                "tcp_connected": tcp_connected,
            }

    # ================================================================
    # 从节点：主节点健康监控 + 自动重连
    # ================================================================

    def _start_client_health_monitor(self) -> None:
        """
        启动从节点后台线程：监控主节点是否在线。

        每 15 秒检查一次数据库中的主节点心跳时间戳。
        当检测到主节点从在线变为离线时，记录告警日志。
        当宕机超过 MASTER_DOWN_EMAIL_TIMEOUT 秒时，发送邮件告警。
        当主节点恢复在线时，发送恢复通知 + 自动重连（如已配置）。
        """
        # ★ 从数据库读取主节点当前在线状态作为初始值，
        #    避免启动时 was_online=False + DB 显示在线 → 误触发 "恢复重连"
        try:
            initial_health = self.check_master_health()
            initial_online = initial_health.get("master_online", False)
        except Exception:
            initial_online = False
        self._client_master_was_online = initial_online
        self._client_master_online = initial_online
        self._client_reconnect_enabled = True

        # 邮件告警状态
        self._client_master_down_since = 0.0         # 主节点首次检测到宕机的时间戳
        self._client_master_down_email_sent = False  # 本轮宕机是否已发送告警邮件

        # 周期性重连：当主节点在线但本地 TCP 未连接时，每隔一定时间重试
        self._client_last_reconnect_attempt = 0.0    # 上次重连尝试的时间戳

        t = threading.Thread(target=self._client_health_monitor_loop, daemon=True)
        t.start()
        threshold_info = f"，宕机邮件告警阈值: {MASTER_DOWN_EMAIL_TIMEOUT}s" if MASTER_DOWN_EMAIL_TIMEOUT > 0 else "（邮件告警已禁用）"
        logger.info(f"从节点主节点健康监控已启动（间隔 15s，重连间隔 60s）{threshold_info}")

    def _client_health_monitor_loop(self) -> None:
        """从节点健康监控循环（后台 daemon 线程）"""
        while self._running:
            try:
                health = self.check_master_health()
                was_online = self._client_master_was_online
                is_online = health.get("master_online", False)

                self._client_master_online = is_online

                # ---- 检测主节点宕机 ----
                if was_online and not is_online:
                    self._client_master_down_since = time.time()
                    self._client_master_down_email_sent = False
                    logger.warning(
                        f"⚠️ 检测到主节点宕机！上次心跳: "
                        f"{health.get('last_seen_seconds_ago', '?')}s 前"
                    )

                # ---- 主节点持续宕机：检查是否需要发送邮件告警 ----
                if (not is_online
                        and self._client_master_down_since > 0
                        and MASTER_DOWN_EMAIL_TIMEOUT > 0
                        and not self._client_master_down_email_sent):
                    downtime = time.time() - self._client_master_down_since
                    if downtime >= MASTER_DOWN_EMAIL_TIMEOUT:
                        try:
                            from email_notifier import send_master_down_alert
                            host = health.get("master_host", "")
                            port = health.get("master_port", 0)
                            last_seen = health.get("last_seen_seconds_ago")
                            client_id = self.get_effective_node_id()
                            ok = send_master_down_alert(
                                host, port, downtime,
                                last_seen_seconds_ago=last_seen,
                                client_node_id=client_id,
                            )
                            if ok:
                                self._client_master_down_email_sent = True
                                logger.info(
                                    f"📧 主节点宕机告警邮件已发送 "
                                    f"（宕机 {downtime:.0f}s，阈值 {MASTER_DOWN_EMAIL_TIMEOUT}s）"
                                )
                        except Exception as e:
                            logger.error(f"发送宕机告警邮件失败: {e}")

                # ---- 检测主节点恢复 ----
                if not was_online and is_online:
                    total_downtime = time.time() - self._client_master_down_since if self._client_master_down_since > 0 else 0

                    # 发送恢复通知邮件（仅在本轮曾发送过宕机告警时）
                    if self._client_master_down_email_sent:
                        try:
                            from email_notifier import send_master_recovery_alert
                            host = health.get("master_host", "")
                            port = health.get("master_port", 0)
                            client_id = self.get_effective_node_id()
                            send_master_recovery_alert(
                                host, port, total_downtime,
                                client_node_id=client_id,
                            )
                            logger.info(f"📧 主节点恢复通知邮件已发送（总宕机 {total_downtime:.0f}s）")
                        except Exception as e:
                            logger.error(f"发送恢复通知邮件失败: {e}")

                    # 重置宕机追踪状态
                    self._client_master_down_since = 0.0
                    self._client_master_down_email_sent = False

                    logger.info(
                        f"✅ 主节点已恢复在线 "
                        f"({health.get('master_host')}:{health.get('master_port')})"
                    )
                    # 如果已有连接配置，尝试自动重连
                    if self._client_reconnect_enabled:
                        host = health.get("master_host", "")
                        port = health.get("master_port", 0)
                        if host and port:
                            result = self.connect_to_master(host, port)
                            if result.get("status") == "connected":
                                logger.info(f"🔄 已自动重连到主节点 {host}:{port}")

                # ---- 周期性重连：主节点在线但本地 TCP 未连接时定期重试 ----
                if (is_online
                        and self._client_reconnect_enabled
                        and self._effective_role() == "client"):
                    tcp_client = getattr(self, '_tcp_client', None)
                    tcp_connected = (tcp_client is not None
                                     and getattr(tcp_client, '_running', False)
                                     and getattr(tcp_client, 'sock', None) is not None)
                    if not tcp_connected:
                        now = time.time()
                        last_attempt = getattr(self, '_client_last_reconnect_attempt', 0.0)
                        if now - last_attempt >= 60:  # 每 60 秒重试一次
                            self._client_last_reconnect_attempt = now
                            host = health.get("master_host", "")
                            port = health.get("master_port", 0)
                            if host and port:
                                logger.info(
                                    f"🔄 周期性重连尝试: {host}:{port}"
                                )
                                result = self.connect_to_master(host, port)
                                if result.get("status") == "connected":
                                    logger.info(f"✅ 周期性重连成功: {host}:{port}")
                                    self._client_last_reconnect_attempt = 0.0  # 成功后重置

                self._client_master_was_online = is_online
            except Exception as e:
                logger.debug(f"健康监控循环异常: {e}")

            time.sleep(15)

    def get_client_master_status(self) -> dict:
        """
        获取从节点视角下的主节点在线状态。

        Returns:
            { master_online, last_seen_ago, health }
        """
        health = self.check_master_health()
        return {
            "master_online": health.get("master_online", False),
            "last_seen_seconds_ago": health.get("last_seen_seconds_ago"),
            "master_host": health.get("master_host", ""),
            "master_port": health.get("master_port", 0),
            "stale": health.get("stale", True),
            "health": health,
        }
