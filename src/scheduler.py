"""
调度控制模块 — 节点状态管理、推理任务分发、流水线调度
========================================================
功能职责:
1. 节点状态管理（空闲/忙碌/离线）
2. 推理任务分发、流程启停
3. 异常捕获、错误上报
4. 流水线数据流调度控制

依赖: threading, logging
"""

import logging
import threading
import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass, field

from config import RUN_MODE, HEARTBEAT_INTERVAL

logger = logging.getLogger(__name__)


class NodeState(str, Enum):
    """节点状态枚举"""
    ONLINE = "online"       # 在线空闲
    BUSY = "busy"           # 推理中
    OFFLINE = "offline"     # 离线/断连
    ERROR = "error"         # 异常


class NodeRole(str, Enum):
    """节点角色"""
    MASTER = "master"
    CLIENT1 = "client1"
    CLIENT2 = "client2"


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    role: NodeRole
    state: NodeState = NodeState.OFFLINE
    address: tuple = None          # (ip, port)
    last_heartbeat: float = 0.0
    task_count: int = 0            # 已完成任务数
    error_count: int = 0           # 错误计数

    def is_available(self) -> bool:
        return self.state == NodeState.ONLINE


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


class Scheduler:
    """
    主节点调度器

    负责:
    - 管理所有从节点状态
    - 接收前端推理请求，分发给流水线
    - 监控任务执行，处理异常
    - 控制流水线启停
    """

    def __init__(self):
        self.nodes: dict[str, NodeInfo] = {}
        self._current_task: Optional[InferenceTask] = None
        self._task_lock = threading.Lock()
        self._running = False
        self.on_task_complete: Optional[Callable] = None

    # ================================================================
    # 节点管理
    # ================================================================

    def init_nodes(self) -> None:
        """初始化所有从节点状态"""
        self.nodes = {
            "master": NodeInfo(node_id="master", role=NodeRole.MASTER, state=NodeState.ONLINE),
            "client1": NodeInfo(node_id="client1", role=NodeRole.CLIENT1),
            "client2": NodeInfo(node_id="client2", role=NodeRole.CLIENT2),
        }
        logger.info(f"节点初始化完成: {len(self.nodes)} 个节点")
        for nid, info in self.nodes.items():
            logger.info(f"  {nid}: role={info.role.value}, state={info.state.value}")

    def update_node_state(self, node_id: str, state: NodeState) -> None:
        """更新节点状态"""
        if node_id in self.nodes:
            old_state = self.nodes[node_id].state
            self.nodes[node_id].state = state
            self.nodes[node_id].last_heartbeat = time.time()
            logger.info(f"节点 {node_id} 状态变更: {old_state.value} -> {state.value}")
        else:
            logger.warning(f"未知节点: {node_id}")

    def get_available_nodes(self) -> list[NodeInfo]:
        """获取所有可用节点"""
        return [n for n in self.nodes.values() if n.is_available()]

    def check_nodes_ready(self) -> bool:
        """检查所有从节点是否就绪（用于分布式模式）"""
        if RUN_MODE == "single":
            return True
        required = ["client1", "client2"]
        for nid in required:
            if nid not in self.nodes or not self.nodes[nid].is_available():
                return False
        return True

    # ================================================================
    # 任务调度
    # ================================================================

    def start_infer_task(self, prompt: str) -> str:
        """
        启动一轮完整推理任务。

        Args:
            prompt: 用户输入文本

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
        for nid in self.nodes:
            if self.nodes[nid].role != NodeRole.MASTER:
                self.update_node_state(nid, NodeState.BUSY)

        logger.info(f"推理任务启动: {task_id}, prompt_len={len(prompt)}")

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
        for nid in self.nodes:
            if self.nodes[nid].state == NodeState.BUSY:
                self.update_node_state(nid, NodeState.ONLINE)

        # TODO: 发送 TASK_STOP 指令给所有从节点
        # TODO: 清空所有节点 KV 缓存

    def on_task_finished(self, result: str, metrics: dict = None) -> None:
        """
        任务完成回调。

        Args:
            result: 推理结果文本
            metrics: 性能指标
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
        for nid in self.nodes:
            if self.nodes[nid].state == NodeState.BUSY:
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
        """获取系统整体状态"""
        node_status = {}
        for nid, info in self.nodes.items():
            node_status[nid] = {
                "role": info.role.value,
                "state": info.state.value,
                "task_count": info.task_count,
                "error_count": info.error_count,
                "last_heartbeat": info.last_heartbeat,
            }

        current_task = None
        if self._current_task:
            current_task = {
                "task_id": self._current_task.task_id,
                "state": self._current_task.state,
                "elapsed": time.time() - self._current_task.start_time,
            }

        return {
            "run_mode": RUN_MODE,
            "nodes": node_status,
            "current_task": current_task,
        }
