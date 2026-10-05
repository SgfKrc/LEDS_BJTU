"""Stable scheduler value objects shared by its implementation mixins."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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


#: 从节点心跳的**容忍上限**（秒）。取 45s 的 2 倍余量：App 侧
#: `AndroidPresenceStateMachine.heartbeatIntervalMs = 45_000`（可配 5–120s，见
#: `heartbeatIntervalSeconds.coerceIn(5, 120)`），而这里原先写死 10s ⇒ 45s 的间隔
#: 必然被判过期（实测 `11.4s > 10s`）。Route A 要求 Android 参与 readiness 后才暴露。
#:
#: ★ 2026-10-05（DIST-2「唯一超时来源」）：本常量由 `scheduler_pipeline` 上移到此处，
#: 因为**容量规划**（`scheduler._get_pipeline_capacity_nodes`）与 **readiness**
#: （`scheduler_pipeline._get_pipeline_readiness`）现在共用同一个阈值 —— 此前容量
#: 规划只看 `NodeInfo.is_available()`（`state == ONLINE`），完全不看心跳新鲜度，而
#: TCP 半开要等巡检约 129s 才置 OFFLINE ⇒ 静默节点在这段时间里一直占容量。
WORKER_HEARTBEAT_MAX_AGE = 120.0


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
    presence_generation: int = 0
    presence_lease_id: str = ""
    presence_expires_at: float = 0.0

    def is_available(self) -> bool:
        return self.state == NodeState.ONLINE

    def heartbeat_age(self, now: Optional[float] = None) -> Optional[float]:
        """距上次心跳的秒数；从未有过任何时间基准时返回 `None`。

        基准优先取 `last_heartbeat`，未收到过心跳时回退到 `connected_at`。
        """
        base = self.last_heartbeat or self.connected_at
        if not base:
            return None
        return max(0.0, (time.time() if now is None else now) - base)

    def is_heartbeat_fresh(self, now: float, max_age: float) -> bool:
        """心跳是否仍在容忍窗口内。

        ★ 2026-10-05（DIST-2）：把「节点可用」从单一的 `NodeState` 扩到
        「状态 + 心跳新鲜度」。此前容量规划只看 `is_available()`
        （`state == ONLINE`），而 TCP 半开（对端进程已死、不发 FIN）要等巡检约
        129s 才置 `OFFLINE` ⇒ 静默节点在这段时间里**一直占容量**，会分配出错的
        层区间（实测：Y700 被带走后 master 仍把它算进规划，给出既非声明区间、
        层数也不对的结果）。

        `max_age` 由调用方注入（复用既有常量，不另造一套数字）。

        **时间基准完全未知时视为新鲜**：那是「未知」而不是「已过期」。真实路径
        不会出现这种情况（`register_node` 与 Android 注册都会立即写入
        `connected_at` 与 `last_heartbeat`），但测试与嵌入场景构造的节点可能两个
        字段都没填，不应因此被误判掉线。
        """
        age = self.heartbeat_age(now)
        return age is None or age <= max_age

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
            "presence_generation": self.presence_generation,
            "presence_expires_at": self.presence_expires_at,
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
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

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
