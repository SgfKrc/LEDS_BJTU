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

import logging
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
    PIPELINE_TIMEOUT, PIPELINE_QUEUE_POLL_INTERVAL,
)

logger = logging.getLogger(__name__)

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

    @staticmethod
    def client_ids(max_nodes: int) -> list:
        """根据最大节点数生成从节点 ID 列表 (不含 master)"""
        return [f"client{i}" for i in range(1, max_nodes)]


@dataclass
class NodeInfo:
    """节点信息（分布式模式下通过 TCP 注册填充）"""
    node_id: str
    role: str                      # 节点角色: "master" | "client"
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

    def is_available(self) -> bool:
        return self.state == NodeState.ONLINE

    def to_dict(self) -> dict:
        """转为可序列化的字典"""
        return {
            "node_id": self.node_id,
            "role": self.role,
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


class PipelineQueue:
    """
    流水线请求队列 — FIFO 串行执行，单任务并发。

    特性:
    - 仅 1 个流水线任务执行中，后续请求自动排队
    - 调用方通过 task_id 轮询或阻塞等待结果
    - 已完成结果保留 TTL 秒后自动清理
    - 线程安全

    用法:
        queue = PipelineQueue()
        queue.start(process_fn=scheduler.run_pipeline)
        task_id = queue.enqueue(prompt="hello", max_new_tokens=256)
        result = queue.wait_for_result(task_id, timeout=120)
    """

    def __init__(self, max_size: int = 100, result_ttl: float = 300.0):
        import collections

        self._queue: collections.deque = collections.deque()
        self._results: dict = {}           # task_id → {status, result, created_at, ...}
        self._events: dict = {}            # task_id → threading.Event
        self._lock = threading.RLock()  # 可重入锁：run_pipeline_safe 在持有锁时调用 enqueue
        self._current_task_id: Optional[str] = None
        self._running = False
        self._max_size = max_size
        self._result_ttl = result_ttl
        self._worker_thread: Optional[threading.Thread] = None
        self._process_fn: Optional[Callable] = None

    def start(self, process_fn: Callable) -> None:
        """启动后台工作线程，开始处理队列中的任务。"""
        if self._running:
            return
        self._process_fn = process_fn
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_loop, name="pipeline-queue", daemon=True
        )
        self._worker_thread.start()
        logger.info("流水线请求队列已启动 (FIFO, max_size=%d)", self._max_size)

    def stop(self) -> None:
        """停止工作线程，清理等待中的任务。"""
        self._running = False
        # 唤醒所有等待者
        with self._lock:
            for task_id, event in self._events.items():
                if not event.is_set():
                    self._results[task_id] = {
                        "status": "cancelled", "error": "队列已停止"
                    }
                    event.set()
        logger.info("流水线请求队列已停止")

    def enqueue(self, task_id: str = None, **task_data) -> str:
        """
        将推理请求加入队列末尾。

        Args:
            task_id: 任务标识（None 则自动生成）
            **task_data: 传递给 process_fn 的关键字参数

        Returns:
            task_id 字符串

        Raises:
            RuntimeError: 队列已满
        """
        import uuid

        if task_id is None:
            task_id = f"q_{uuid.uuid4().hex[:12]}"

        with self._lock:
            if len(self._queue) >= self._max_size:
                logger.warning(
                    f"⚠️ 请求队列已满 ({self._max_size}/{self._max_size})，"
                    f"拒绝新请求"
                )
                raise RuntimeError(
                    f"请求队列已满 ({self._max_size} 上限)，请稍后重试"
                )
            self._queue.append((task_id, task_data))
            self._events[task_id] = threading.Event()
            self._results[task_id] = {
                "status": "queued",
                "created_at": time.time(),
            }

        logger.info(
            f"📥 请求已入队: task={task_id}, "
            f"queue_depth={self.queue_size}"
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

    # ---- 内部方法 ----

    def _process_loop(self) -> None:
        """后台工作循环：从队列取任务 → 调用 process_fn → 存储结果。"""
        while self._running:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()

            if task is None:
                time.sleep(0.1)
                continue

            task_id, task_data = task
            if task_data is None:  # sentinel
                continue

            with self._lock:
                self._current_task_id = task_id
                self._results[task_id]["status"] = "running"
                self._results[task_id]["started_at"] = time.time()

            logger.info(f"🚀 开始处理排队任务: {task_id}")
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
                logger.info(f"✅ 排队任务完成: {task_id} ({elapsed:.1f}s)")
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
                logger.error(f"❌ 排队任务失败: {task_id} — {e}")
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
        """当前是否有任务在执行中。"""
        return self._current_task_id is not None

    @property
    def queue_size(self) -> int:
        """当前队列长度（不含正在执行的任务）。"""
        with self._lock:
            return len(self._queue)

    def get_status(self) -> dict:
        """获取队列整体状态。"""
        with self._lock:
            return {
                "running": self._running,
                "current_task": self._current_task_id,
                "queue_size": len(self._queue),
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
        self._pipeline_lock = threading.Lock()
        self._kv_cache: dict = {}               # task_id → past_key_values（本节点层范围的 KV cache）

        # 流水线请求队列（FIFO 串行执行）
        from config import PIPELINE_QUEUE_MAX_SIZE, PIPELINE_QUEUE_RESULT_TTL
        self.pipeline_queue = PipelineQueue(
            max_size=PIPELINE_QUEUE_MAX_SIZE,
            result_ttl=PIPELINE_QUEUE_RESULT_TTL,
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
        return getattr(self, '_role_override', None) or NODE_ROLE

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
            for nid, row in db_nodes.items():
                if nid == "master":
                    continue
                node = self._node_from_db(row)
                if RUN_MODE == "distributed":
                    # 分布式模式：从节点尚未重连，标记为 offline
                    node.state = NodeState.OFFLINE
                self.nodes[nid] = node
                logger.info(f"  恢复从节点: {nid} (state={node.state.value})")

        # ---- 从节点模式：仅创建自身 ----
        else:
            # ★ 安全：从节点绝不能使用 "master" 作为 node_id
            # 否则会从数据库加载主节点记录，导致后台面板显示主节点数据
            if not NODE_ID or NODE_ID == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
                if NODE_ID == "master":
                    logger.warning(
                        f"⚠️ 从节点 NODE_ID 配置错误（仍为 \"master\"），"
                        f"已自动生成: {node_id}"
                    )
            else:
                node_id = NODE_ID
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
                    if new_max != MAX_NODES:
                        import config as cfg
                        cfg.MAX_NODES = new_max
                        logger.info(f"从数据库恢复 max_nodes: {MAX_NODES} → {new_max}")
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
            state=NodeState(db_row.get("state", "offline")),
            address=db_row.get("address", ""),
            hostname=db_row.get("hostname", ""),
            device_info=db_row.get("device_info", {}),
            network_type=db_row.get("network_type", "unknown"),
            connected_at=db_row.get("connected_at", 0.0),
            last_heartbeat=db_row.get("last_heartbeat", 0.0),
            task_count=db_row.get("task_count", 0),
            error_count=db_row.get("error_count", 0),
        )

    def register_node(self, node_id: str, role: str, address: str = "",
                      hostname: str = "", device_info: dict = None,
                      network_type: str = "unknown") -> bool:
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

        Returns:
            注册是否成功
        """
        # 如果节点不在预定义列表中但仍在 MAX_NODES 范围内，动态添加
        if node_id not in self.nodes:
            if role == NodeRole.MASTER:
                logger.warning(f"注册失败: 不能动态注册 master 节点")
                return False
            # 检查是否在 MAX_NODES 范围内
            expected_client_ids = NodeRole.client_ids(MAX_NODES)
            # 也允许用户在 MAX_NODES 之后添加的自定义节点
            logger.info(f"动态添加节点: {node_id}")
            self.nodes[node_id] = NodeInfo(
                node_id=node_id, role=NodeRole.CLIENT,
                state=NodeState.OFFLINE,
            )

        node = self.nodes[node_id]
        if node.role == NodeRole.MASTER:
            logger.warning(f"注册失败: {node_id} 角色为 master，不可被注册覆盖")
            return False

        node.state = NodeState.ONLINE
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
                    state=node.state.value,
                    address=node.address,
                    hostname=node.hostname,
                    device_info=node.device_info,
                    network_type=node.network_type,
                    connected_at=node.connected_at,
                    last_heartbeat=node.last_heartbeat,
                    task_count=node.task_count,
                    error_count=node.error_count,
                )
            except Exception as e:
                logger.warning(f"节点注册 DB 持久化失败: {e}")

        logger.info(
            f"✅ 节点注册: {node_id} role={role} "
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

        logger.info(f"节点注销: {node_id} ({old_state.value} → offline)")
        return True

    def update_node_state(self, node_id: str, state: NodeState) -> None:
        """更新节点状态"""
        if node_id in self.nodes:
            old_state = self.nodes[node_id].state
            self.nodes[node_id].state = state
            self.nodes[node_id].last_heartbeat = time.time()
            logger.info(f"节点 {node_id} 状态变更: {old_state.value} -> {state.value}")
        else:
            logger.warning(f"未知节点: {node_id}")

    def record_task_complete(self, node_id: str = None, success: bool = True) -> None:
        """
        记录一次推理任务完成。

        Args:
            node_id: 执行推理的节点（默认本节点）
            success: 是否成功
        """
        nid = node_id or self.get_effective_node_id()
        if nid in self.nodes:
            if success:
                self.nodes[nid].task_count += 1
            else:
                self.nodes[nid].error_count += 1
            self.nodes[nid].last_heartbeat = time.time()

    def record_task_error(self, node_id: str = None) -> None:
        """记录一次推理任务失败（便捷方法）"""
        self.record_task_complete(node_id, success=False)

    def get_available_nodes(self) -> list:
        """获取所有可用节点（含状态字典）"""
        return [n.to_dict() for n in self.nodes.values() if n.is_available()]

    def check_nodes_ready(self) -> bool:
        """检查所有已注册从节点是否就绪（用于分布式模式）"""
        if RUN_MODE == "single":
            return True
        # 获取所有非 master 节点
        clients = [n for n in self.nodes.values() if n.role != NodeRole.MASTER]
        if not clients:
            return True  # 没有从节点也算就绪（单节点集群）
        for n in clients:
            if not n.is_available():
                return False
        return True

    # ================================================================
    # 动态模型分层
    # ================================================================

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
        gpu = device_info.get("gpu", {}) if device_info else {}
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
        cuda_avail = gpu.get("cuda_available", False)
        is_integrated = gpu.get("is_integrated", True)
        discrete_bonus = 15.0 if (cuda_avail and not is_integrated) else 0.0

        weight = vram_score + ram_score + cpu_score + discrete_bonus
        logger.debug(
            f"节点权重: VRAM={vram_score:.1f} RAM={ram_score:.1f} "
            f"CPU={cpu_score:.1f} Bonus={discrete_bonus:.1f} → {weight:.1f}"
        )
        return weight

    def _sync_node_rtt(self, node_id: str, tcp_client) -> None:
        """
        从 TCP 客户端同步 RTT 到 NodeInfo。

        每次心跳后调用，将 TCP 层测量的 RTT 同步到节点信息中，
        供分层算法和前端展示使用。
        """
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
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
        node = self.nodes.get(node_id)
        if not node or not node.device_info:
            return 0.0
        gpu = node.device_info.get("gpu", {})
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

        # 收集节点数据
        if nodes is None:
            node_list = [
                {"node_id": nid, "role": info.role,
                 "device_info": info.device_info}
                for nid, info in self.nodes.items()
            ]
        else:
            node_list = nodes

        if not node_list:
            return []

        # 单节点：全部层给该节点
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

                    # 应用显存约束校验
                    assignments = self._apply_vram_constraints(assignments)

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
                    f"图算法智能编排失败: {e}，回退到简单权重分配"
                )
                import traceback
                traceback.print_exc()

        # ============================================================
        # 回退：简单权重比例分配（节点数 ≤ 阈值）
        # ============================================================
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
            return assignments

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
            sorted_indices = sorted(range(len(raw_layers)), key=lambda i: node_list[i]["score"])
            for _ in range(-diff):
                for idx in reversed(sorted_indices):
                    if raw_layers[idx] > 1:
                        raw_layers[idx] -= 1
                        break

        # Step 3: 排序（master 优先，同角色按权重降序）
        sorted_pairs = sorted(
            enumerate(node_list),
            key=lambda x: (x[1]["role"] != "master", -x[1]["score"])
        )
        sorted_indices = [i for i, _ in sorted_pairs]

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

    def _apply_vram_constraints(self, assignments: list) -> list:
        """
        显存约束校验：检查每节点是否有足够显存承载分配的层。

        若不足，将超额层转移给 VRAM 最充裕的节点；层数为 0 的节点从列表中移除。
        """
        for a in assignments:
            ok, needed, available = self._check_vram_constraint(
                a["node_id"], a["layers_count"],
                a["has_embedding"], a["has_lm_head"],
            )
            if not ok and available > 0:
                best_node = max(
                    (other for other in assignments if other["node_id"] != a["node_id"]),
                    key=lambda x: self._get_node_vram_mb(x["node_id"]),
                    default=None,
                )
                if best_node:
                    overflow = a["layers_count"]
                    logger.warning(
                        f"⚠️ 显存不足: {a['node_id']} 需要 {needed}MB, "
                        f"可用 {available}MB — {overflow} 层转给 {best_node['node_id']}"
                    )
                    best_node["layers_count"] += overflow
                    best_node["end_layer"] = best_node["start_layer"] + best_node["layers_count"]
                    a["layers_count"] = 0
                    a["end_layer"] = a["start_layer"]
                else:
                    logger.error(
                        f"❌ 显存不足且无其他节点可接手: {a['node_id']} "
                        f"需要 {needed}MB, 可用 {available}MB"
                    )

        # 清除层数为 0 的节点
        assignments[:] = [a for a in assignments if a["layers_count"] > 0]
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

        # 判断实际使用的策略：节点数 > 阈值 → 图算法智能编排
        total_nodes = len(self.nodes)
        actual_strategy = (
            "graph_orchestrator"
            if total_nodes > GRAPH_ORCHESTRATOR_THRESHOLD
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
        """
        if not self._tcp_server or not self._tcp_server._running:
            return

        layer_info = self.get_layer_assignments()
        assignments = {
            a["node_id"]: {
                "start_layer": a["start_layer"],
                "end_layer": a["end_layer"],
                "has_embedding": a.get("has_embedding", False),
                "has_lm_head": a.get("has_lm_head", False),
            }
            for a in layer_info["assignments"]
        }

        try:
            self._tcp_server.broadcast_layer_config(assignments)
            logger.info(f"分层配置已推送到 {len(assignments)} 个从节点")
        except Exception as e:
            logger.warning(f"分层配置推送失败: {e}")

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
            registered = self.register_node(
                node_id=client_id,
                role=data.get("role", ""),
                address=client_info.get("addr", ""),
                hostname=data.get("hostname", ""),
                device_info=data.get("device_info", {}),
                network_type=data.get("network_type", client_info.get("network_type", "unknown")),
            )
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
            if client_id in self.nodes:
                self.nodes[client_id].last_heartbeat = time.time()

        elif msg_type == "status_res":
            # 从节点状态上报
            data = msg.get("data", {})
            if client_id in self.nodes:
                if "state" in data:
                    try:
                        self.nodes[client_id].state = NodeState(data["state"])
                    except ValueError:
                        pass

        elif msg_type == "error":
            data = msg.get("data", {})
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

        elif msg_type == "pipeline_done":
            # ---- 从节点：流水线任务完成，清理 KV 缓存 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id and task_id in self._kv_cache:
                del self._kv_cache[task_id]
                logger.info(f"🧹 流水线任务 {task_id} KV 缓存已清理")

        elif msg_type == "pipeline_abort":
            # ---- 从节点/主节点：流水线任务取消 ----
            data = msg.get("data", {})
            task_id = data.get("task_id", "")
            if task_id and task_id in self._kv_cache:
                del self._kv_cache[task_id]
            logger.warning(f"⚠️ 流水线任务 {task_id} 已取消")

        elif msg_type == "layer_config":
            # ---- 从节点：收到主节点推送的分层配置 ----
            data = msg.get("data", {})
            self._handle_layer_config(client_id, data)

    def _on_tcp_disconnect(self, client_id: str) -> None:
        """TCP 断连回调（由 TCPServer 调用）"""
        if client_id in self.nodes:
            self.nodes[client_id].state = NodeState.OFFLINE
        # 推送节点离线更新给所有连接的从节点
        if self._effective_role() == "master":
            self._push_node_update_to_all_clients(
                client_id, "update", self.nodes.get(client_id)
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

        # 检查是否有至少一个其他在线从节点（非备用、非转让目标）
        other_clients = [
            nid for nid, info in self.nodes.items()
            if info.role in ("client", NodeRole.CLIENT)
            and info.is_available()
            and nid not in (spare_id, target_node_id)
        ]
        if not other_clients and not (target_node_id != spare_id):
            # 如果转让目标就是唯一的另一个从节点，这是允许的最后一次转让
            pass  # 至少有一个其他从节点即可
        # 放宽限制：只要有备用主节点在线即可

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
        """获取系统整体状态（含节点详情和 TCP 连接信息）"""
        node_status = {}
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

        return {
            "run_mode": RUN_MODE,
            "nodes": node_status,
            "current_task": current_task,
            "tcp_server": tcp_info,
            "nodes_ready": self.check_nodes_ready(),
            "pipeline": pipeline_info,
            "pipeline_queue": queue_info,
        }

    def get_nodes(self) -> list:
        """获取所有节点详情列表"""
        return [info.to_dict() for info in self.nodes.values()]

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
            "max_nodes": MAX_NODES,
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
            from tcp_comm import TCPClient

            # ★ 安全：从节点绝不能使用 "master" 作为 client_id
            if not NODE_ID or NODE_ID == "master":
                node_id = f"client_{__import__('socket').gethostname()}"
            else:
                node_id = NODE_ID
            client = TCPClient(
                server_host=master_host,
                server_port=master_port,
                client_id=node_id,
                role="client",
            )
            # ★ 心跳回调：更新自身节点的心跳时间 + 同步 RTT 测量值
            client.on_heartbeat = lambda: (
                self._sync_node_rtt(node_id, client)
                if node_id in self.nodes else None
            )

            # 更新自身节点信息
            if node_id in self.nodes:
                self.nodes[node_id].state = NodeState.ONLINE
                self.nodes[node_id].address = f"{master_host}:{master_port}"

            ok = client.connect(
                on_message=lambda msg: self._on_tcp_message("master", msg)
            )
            if ok:
                # 存储客户端引用，供分布式推理转发使用
                self._tcp_client = client

                # 更新全局 node_id
                import config as cfg
                cfg.NODE_ID = node_id
                # 若通过 activate_client_mode 切换而来，同步更新角色
                if getattr(self, '_role_override', None) == "client":
                    cfg.NODE_ROLE = "client"

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
            else:
                return {
                    "status": "failed",
                    "reason": f"TCP 连接失败 ({master_host}:{master_port})，请检查主节点地址和端口是否正确",
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
            }
            tcp_client.send_data(infer_data, MessageType.INFER_FORWARD)
            logger.info(f"📤 推理请求已转发至主节点 (prompt_len={len(message)})")

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

        return {
            "master_host": lan_ip,
            "master_port": port,
            "node_count": len(self.nodes),
            "online_count": sum(1 for n in self.nodes.values() if n.is_available()),
            "max_nodes": MAX_NODES,
            "has_capacity": len(self.nodes) < MAX_NODES,
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
        cfg.NODE_ROLE = "client"

        # 2. 重新初始化为从节点
        self.nodes.clear()
        self.init_nodes()
        effective_id = self.get_effective_node_id()
        cfg.NODE_ID = effective_id

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
        if effective_role == "client" and (not NODE_ID or NODE_ID == "master"):
            return f"client_{__import__('socket').gethostname()}"
        return NODE_ID

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
        my_info = self.nodes.get(_effective_id)
        result = {
            "node_role": effective_role,
            "node_id": _effective_id,
            "is_master": effective_role == "master",
            "is_client": effective_role == "client",
            "max_nodes": MAX_NODES,
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

        增加时自动创建新的 NodeInfo；减少时保留已注册节点（仅移除未注册的空位）。

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

        old_max = cfg.MAX_NODES
        if new_max == old_max:
            return {"status": "unchanged", "max_nodes": old_max}

        added = []
        removed = []

        if new_max > old_max:
            # 扩大：创建新的 client 节点
            new_client_ids = [f"client{i}" for i in range(old_max, new_max)]
            for cid in new_client_ids:
                if cid not in self.nodes:
                    self.nodes[cid] = NodeInfo(
                        node_id=cid, role=NodeRole.CLIENT,
                        state=NodeState.OFFLINE,
                    )
                    added.append(cid)
        else:
            # 缩小：移除离线且未注册的节点
            client_ids_to_remove = [f"client{i}" for i in range(new_max, old_max)]
            for cid in client_ids_to_remove:
                if cid in self.nodes:
                    node = self.nodes[cid]
                    if node.state == NodeState.OFFLINE and not node.address:
                        del self.nodes[cid]
                        removed.append(cid)
                    else:
                        # 节点在线或已注册，不强制删除
                        logger.warning(
                            f"无法移除节点 {cid}: state={node.state.value}, "
                            f"addr={node.address}，请先注销该节点"
                        )

        cfg.MAX_NODES = new_max

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.set_config("max_nodes", str(new_max))
            except Exception as e:
                logger.warning(f"max_nodes DB 持久化失败: {e}")

        logger.info(
            f"最大节点数已更新: {old_max} → {new_max} "
            f"(添加={added}, 移除={removed})"
        )

        return {
            "status": "ok",
            "max_nodes": new_max,
            "old_max": old_max,
            "nodes_added": added,
            "nodes_removed": removed,
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
            if "master" in self.nodes:
                self.nodes["master"].last_heartbeat = time.time()
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
        处理从节点转发的推理请求。

        主节点收到 INFER_FORWARD 后:
          1. 创建推理任务
          2. 在主节点本地执行完整模型推理（全部层）
          3. 将结果通过 INFER_RESULT 回传给请求方

        TODO: 真正的分布式流水线（模型层拆分到多个节点并行推理）
              当前先实现「全部层都在主节点」模式，让从节点转发可用。
        """
        data = msg.get("data", {})
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        show_thinking = data.get("show_thinking", False)
        session_id = data.get("session_id")

        import threading as _thr

        def _run_inference():
            try:
                task_id = self.start_infer_task(prompt)
                logger.info(
                    f"📨 收到从节点 {client_id} 转发的推理请求: "
                    f"task={task_id}, prompt_len={len(prompt)}, "
                    f"max_tokens={max_new_tokens}, temp={temperature}"
                )

                # 在主节点本地执行推理（全部层）
                try:
                    from model_module import ModelManager
                except ImportError:
                    self._send_infer_result(
                        client_id, task_id, "",
                        {"error": "模型模块未加载，请确认模型文件已就绪"}
                    )
                    return

                # api_server 中的全局 model_manager 实例
                import api_server as _api
                mgr = getattr(_api, 'model_manager', None)
                if not mgr or not mgr.is_loaded:
                    self._send_infer_result(
                        client_id, task_id, "",
                        {"error": "模型未就绪，请确认模型文件已下载并加载成功"}
                    )
                    return

                # 使用相同的 chat() 方法（与直连 localhost 一致）
                messages = [{"role": "user", "content": prompt}]
                output = mgr.chat(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

                content = output.get("response", "") if isinstance(output, dict) else str(output)
                thinking = output.get("thinking") if isinstance(output, dict) else None
                followups = output.get("followups") if isinstance(output, dict) else None
                metrics = output.get("metrics", {}) if isinstance(output, dict) else {}

                # 保存到对话历史（主节点侧）
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

                self.complete_infer_task(task_id, content)
                self._send_infer_result(
                    client_id, task_id, content, metrics,
                    thinking_content=thinking,
                    followups=followups,
                )
                logger.info(
                    f"✅ 推理完成 → {client_id}: task={task_id}, "
                    f"len={len(content)}"
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
            if use_kv_cache and task_id in self._kv_cache:
                past_kv = self._kv_cache[task_id]
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

            # ---- KV Cache: 存储更新后的 past_key_values ----
            if result.get("past_key_values"):
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

            # ---- 链式直连：转发给下一个从节点（P2 优化）----
            chain_next = data.get("chain_next")
            chain_remaining = data.get("chain_remaining", [])

            if chain_next and isinstance(chain_next, dict) and chain_next.get("node_id"):
                # 非末节点：通过 TCP 直连转发 hidden_states 给下一个节点
                chain_data = {
                    "task_id": task_id,
                    "step": step,
                    "hidden_states": response.get("hidden_states"),
                    "hidden_shape": response.get("hidden_shape"),
                    "chain_next": chain_remaining[0] if chain_remaining else None,
                    "chain_remaining": chain_remaining[1:] if len(chain_remaining) > 1 else [],
                    "use_kv_cache": use_kv_cache,
                    "temperature": data.get("temperature", 0.7),
                    "top_p": data.get("top_p", 0.9),
                }
                ok = self._send_chain_forward(chain_next["node_id"], chain_data)
                if not ok:
                    # 链式转发失败 → 向主节点报错
                    self._send_layer_result("master", task_id, error=f"链式转发到 {chain_next['node_id']} 失败")
            else:
                # 末节点（或无链配置）：发送 LAYER_RESULT 回主节点
                self._send_layer_result("master", task_id, result_data=response)

        except Exception as e:
            logger.error(f"层前向传播失败: task={task_id}, error={e}")
            import traceback
            traceback.print_exc()
            self._send_layer_result("master", task_id, error=str(e))

    def _handle_chain_forward(self, client_id: str, msg: dict) -> None:
        """
        从节点：收到另一从节点的 CHAIN_FORWARD → 执行本节点层前向 → 继续转发或回传。

        CHAIN_FORWARD 的消息结构与 LAYER_FORWARD 一致（均为 hidden_states + chain 信息），
        直接委托 _handle_layer_forward 处理（其内部根据 chain_next 决定下一步动作）。
        """
        logger.info(f"🔗 收到链式转发: from={client_id}, task={msg.get('data', {}).get('task_id', '?')}")
        self._handle_layer_forward(client_id, msg)

    def _send_layer_result(self, client_id: str, task_id: str,
                           result_data: dict = None, error: str = None) -> None:
        """从节点 → 主节点：发送层前向传播结果"""
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            return

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
        except Exception as e:
            logger.error(f"发送层前向结果失败: {e}")

    def _handle_layer_result(self, client_id: str, msg: dict) -> None:
        """
        主节点：收到从节点的 LAYER_RESULT → 存储到流水线结果字典，
        唤醒正在等待的 run_pipeline() 主循环。
        """
        data = msg.get("data", {})
        task_id = data.get("task_id", "")
        node_id = data.get("node_id", client_id)

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
        检查所有流水线节点是否就绪：
        1. 节点在线
        2. 已分配层范围
        3. TCP 连接正常
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

        for node in pipeline_nodes:
            node_id = node["node_id"]
            if node_id not in self.nodes:
                logger.warning(f"流水线节点 {node_id} 未注册")
                return False
            if not self.nodes[node_id].is_available():
                logger.warning(f"流水线节点 {node_id} 不在线")
                return False
            if node_id not in (self._tcp_server.clients if self._tcp_server else {}):
                logger.warning(f"流水线节点 {node_id} TCP 未连接")
                return False

        logger.info(f"✅ 所有流水线节点就绪: {[n['node_id'] for n in pipeline_nodes]}")
        return True

    def _broadcast_pipeline_abort(self, pipeline_nodes: list, task_id: str,
                                   reason: str) -> None:
        """向所有流水线节点广播 PIPELINE_ABORT（清理 KV cache）。"""
        from tcp_comm import MessageType
        try:
            for n in pipeline_nodes:
                self._send_to_worker(
                    n["node_id"],
                    {"task_id": task_id, "reason": reason},
                    MessageType.PIPELINE_ABORT,
                )
        except Exception as e:
            logger.warning(f"广播 PIPELINE_ABORT 失败: {e}")

    def _get_node_address(self, node_id: str) -> Optional[dict]:
        """
        获取节点的 (host, port) 地址信息。

        返回 {"host": str, "port": int} 或 None（节点未知/离线）。
        """
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
            ok = client.connect(timeout_ms=5000)
            if not ok:
                logger.error(f"链式转发: 连接 {target_node_id} ({addr['host']}:{addr['port']}) 失败")
                return False

            client.send_data(data, MessageType.CHAIN_FORWARD)
            client.close()
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
                               timeout: float = 30.0) -> Optional[dict]:
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

        # 为每个可能的节点创建 event
        events = []
        with self._pipeline_lock:
            for key in keys:
                event = threading.Event()
                self._pipeline_events[key] = event
                events.append((key, event))

        # 等待任一 event 触发
        deadline = time.time() + timeout
        signaled_key = None
        while time.time() < deadline:
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

            # 查找第一个有结果或超时的 key
            result = None
            for key in keys:
                data = self._pipeline_results.pop(key, None)
                if data is not None:
                    result = data
                    break

        if signaled_key is None:
            logger.error(f"⏰ 等待流水线结果超时 ({timeout}s), task={task_id}")
            return None

        if result is None:
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

    def run_pipeline(self, prompt: str, max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.9,
                     session_id: str = None) -> dict:
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
        from tcp_comm import MessageType, deserialize_tensor_fast

        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.tokenizer:
            return {"response": "", "error": "模型未加载"}

        tokenizer = mgr.tokenizer
        device = mgr.get_device()

        # ---- Step 1: 获取分层配置 ----
        layer_info = self.get_layer_assignments()
        pipeline_nodes = [
            a for a in layer_info.get("assignments", [])
            if a.get("node_id") != "master"
        ]
        # 按 start_layer 排序，确保流水线顺序正确
        pipeline_nodes.sort(key=lambda a: a.get("start_layer", 0))

        if not pipeline_nodes:
            return {"response": "", "error": "没有可用的流水线从节点"}

        logger.info(
            f"🚀 启动流水线推理: prompt_len={len(prompt)}, "
            f"max_tokens={max_new_tokens}, 节点数={len(pipeline_nodes)}, "
            f"顺序: {' → '.join(n['node_id'] for n in pipeline_nodes)}, "
            f"KV Cache: ✅"
        )

        # ---- Step 2: Tokenize ----
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # (1, prompt_len)
        attention_mask = inputs.get("attention_mask")
        prompt_len = input_ids.shape[1]

        # ---- Step 3: 自回归生成 ----
        task_id = uuid.uuid4().hex[:12]
        generated_ids = []
        pipeline_metrics = {"steps": [], "total_time_ms": 0, "kv_cache": True,
                            "chain_topology": True}
        t_pipeline_start = time.time()

        # 仅用于最终解码，不再用于发送
        full_input_ids = input_ids

        for step in range(max_new_tokens):
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

            # ---- 构建 LAYER_FORWARD 消息（仅发给首节点）----
            forward_data = {
                "task_id": task_id,
                "step": step,
                "temperature": temperature,
                "top_p": top_p,
                "use_kv_cache": not is_prefill,  # ★ Prefill=False, Decode=True
            }

            if is_prefill:
                # Prefill: 发送完整 prompt input_ids
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
                    f"🔗 Step {step} 链式路由: {' → '.join(c['node_id'] for c in chain_info)}"
                )
            else:
                forward_data["chain_next"] = None
                forward_data["chain_remaining"] = []

            # ---- 发送给首节点 ----
            try:
                self._send_to_worker(first_node_id, forward_data, MessageType.LAYER_FORWARD)
            except Exception as e:
                step_error = f"发送到首节点 {first_node_id} 失败: {e}"
                logger.error(step_error)

            if step_error:
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                return {"response": "", "error": step_error}

            # ---- 等待链上任一节点返回结果（末节点=成功，其他=错误）----
            result = self._wait_for_layer_result(
                task_id,
                [n["node_id"] for n in pipeline_nodes],  # 任一节点都可能报错
                timeout=30.0,
            )
            if result is None:
                step_error = f"末节点 {last_node_id} 响应超时"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                return {"response": "", "error": step_error}

            if result.get("error"):
                step_error = f"流水线错误: {result['error']}"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                return {"response": "", "error": step_error}

            # 提取 logits（链式模式下仅末节点返回 logits）
            if "logits" in result and result["logits"] is not None:
                logits_data = result["logits"]
                if isinstance(logits_data, bytes):
                    logits = deserialize_tensor_fast(logits_data)
                logits = logits.to(device=device)
            else:
                step_error = "末节点未返回 logits"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                return {"response": "", "error": step_error}

            if step_error:
                # 广播 ABORT（清理各节点的 KV cache）
                try:
                    for n in pipeline_nodes:
                        self._send_to_worker(
                            n["node_id"],
                            {"task_id": task_id, "reason": step_error},
                            MessageType.PIPELINE_ABORT,
                        )
                except Exception:
                    pass
                return {"response": "", "error": step_error}

            # ---- Step 4: 从 logits 采样下一个 token ----
            # logits shape: prefill=(1, prompt_len, vocab), decode=(1, 1, vocab)
            next_logits = logits[:, -1, :] / temperature
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
        pipeline_metrics["nodes_used"] = len(pipeline_nodes)

        tokens_per_sec = (
            len(generated_ids) / (pipeline_metrics["total_time_ms"] / 1000)
            if pipeline_metrics["total_time_ms"] > 0 and generated_ids
            else 0
        )
        pipeline_metrics["tokens_per_second"] = round(tokens_per_sec, 1)

        logger.info(
            f"✅ 流水线推理完成: {len(generated_ids)} tokens, "
            f"{pipeline_metrics['total_time_ms']:.0f}ms, "
            f"{tokens_per_sec:.1f} tok/s (KV Cache: ✅)"
        )

        return {
            "response": new_text,
            "full_text": response_text,
            "metrics": pipeline_metrics,
        }

    # ================================================================
    # 流水线请求队列集成（Phase 4 — 多请求排队）
    # ================================================================

    def _process_queued_pipeline_task(self, prompt: str, **kwargs) -> dict:
        """
        队列工作线程的回调：执行流水线推理并返回结果。

        ★ 直接调用 run_pipeline（绕过 run_pipeline_safe 的排队检查），
           避免死锁：队列 worker 已设置 _current_task_id，若走 run_pipeline_safe
           会再次检测 is_busy=True → enqueue → 永久等待自己完成。
        """
        try:
            # 检查节点是否就绪
            if not self._all_pipeline_nodes_ready():
                logger.warning("流水线节点不可用，队列任务回退到全模型推理")
                return self._run_full_model_inference(prompt, **kwargs)
            return self.run_pipeline(prompt, **kwargs)
        except Exception as e:
            logger.error(f"队列任务流水线推理失败: {e}，回退到全模型推理")
            import traceback
            traceback.print_exc()
            return self._run_full_model_inference(prompt, **kwargs)

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
        # ---- 自动回退：节点不可用 → 全模型推理 ----
        try:
            pipeline_ready = self._all_pipeline_nodes_ready()
        except Exception:
            pipeline_ready = False

        if not pipeline_ready:
            logger.warning("部分流水线节点未就绪，回退到全层主节点模式")
            return self._run_full_model_inference(prompt, **kwargs)

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
                return result.get("result", {})
            elif result.get("status") == "timeout":
                return {"response": "", "error": f"排队超时 ({PIPELINE_TIMEOUT}s)"}
            else:
                return {"response": "", "error": result.get("error", "排队请求失败")}

        # ---- 立即执行（已通过原子检查）----
        try:
            return self.run_pipeline(prompt, **kwargs)
        except Exception as e:
            logger.error(f"流水线推理失败: {e}，回退到全层主节点模式")
            import traceback
            traceback.print_exc()
            return self._run_full_model_inference(prompt, **kwargs)
        finally:
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
        """
        import api_server as _api

        mgr = getattr(_api, 'model_manager', None)
        if not mgr or not mgr.is_loaded:
            return {"response": "", "error": "模型未加载"}

        try:
            messages = [{"role": "user", "content": prompt}]
            result = mgr.chat(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return {
                "response": result.get("content", ""),
                "thinking": result.get("thinking_content", ""),
                "metrics": {
                    "mode": "fallback_full_model",
                    "tokens_per_second": result.get("tokens_per_second", 0),
                    "usage": result.get("usage", {}),
                },
            }
        except Exception as e:
            logger.error(f"全模型回退推理失败: {e}")
            return {"response": "", "error": str(e)}

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
        for w in workers:
            nid = w["node_id"]
            node = self.nodes.get(nid)
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
                             address: str = "", network_type: str = "unknown") -> dict:
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

        if node_id in self.nodes:
            existing = self.nodes[node_id]
            if existing.role == "master":
                return {"status": "invalid", "reason": f"'{node_id}' 是主节点，不可覆盖"}
            return {"status": "exists", "node_id": node_id,
                    "message": f"节点 '{node_id}' 已存在 (state={existing.state.value})",
                    "state": existing.state.value}

        # 检查容量
        non_master = [n for n in self.nodes.values() if n.role != "master"]
        if len(non_master) >= MAX_NODES - 1:
            return {"status": "full", "reason": f"已达到最大从节点数量 ({MAX_NODES - 1})"}

        # 创建节点（初始 offline）
        node = NodeInfo(
            node_id=node_id,
            role=NodeRole.CLIENT,
            state=NodeState.OFFLINE,
            hostname=hostname or node_id,
            address=address,
            network_type=network_type,
        )
        self.nodes[node_id] = node

        # 持久化到数据库
        db = _get_db()
        if db and _db_available:
            try:
                db.upsert_node(
                    node_id=node_id, role="client", state="offline",
                    address=address, hostname=node.hostname,
                    network_type=network_type,
                )
            except Exception as e:
                logger.warning(f"手动注册节点 DB 持久化失败: {e}")

        logger.info(f"📝 主节点手动注册从节点: {node_id} (hostname={hostname}, addr={address})")
        return {
            "status": "registered",
            "node_id": node_id,
            "message": f"节点 '{node_id}' 已手动注册，等待 TCP 连接激活",
            "state": "offline",
        }

    def check_master_health(self) -> dict:
        """
        检查主节点是否在线（通过数据库心跳时间戳判断）。

        从节点可以周期性调用此方法监控主节点状态。
        如果主节点心跳超过 120 秒未更新，视为宕机。

        Returns:
            {
                "master_online": bool,
                "last_seen_seconds_ago": float | None,
                "stale": bool,
                "master_host": str,
                "master_port": int,
            }
        """
        db = _get_db()
        if not db or not _db_available:
            return {
                "master_online": False,
                "last_seen_seconds_ago": None,
                "stale": True,
                "master_host": "",
                "master_port": 0,
                "source": "db_unavailable",
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
                }

            now = time.time()
            last_seen = info.get("last_seen", 0)
            ago = now - last_seen if last_seen > 0 else None
            stale = info.get("stale", True)

            return {
                "master_online": not stale,
                "last_seen_seconds_ago": round(ago, 1) if ago else None,
                "stale": stale,
                "master_host": info["master_host"],
                "master_port": info["master_port"],
                "source": "database",
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
