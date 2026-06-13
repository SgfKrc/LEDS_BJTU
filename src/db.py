"""
数据库访问层 — PostgreSQL 持久化存储
=====================================
功能:
1. 连接池管理（psycopg2 ThreadedConnectionPool）
2. 表结构自动创建（首次运行）
3. 节点注册表 CRUD — 替代内存 dict，重启不丢失
4. 对话历史持久化 — 替代 localStorage
5. 集群配置键值存储 — 替代 config.py 常量

依赖: psycopg2-binary

数据库: 8.160.161.53:5432 / postgres / 123456
"""

from __future__ import annotations

import logging
import json
import time
import threading
from contextlib import contextmanager
from typing import Optional, Any

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# ================================================================
# 数据库连接配置（部署时修改）
# ================================================================
DB_HOST = "8.160.161.53"
DB_PORT = 5432
DB_NAME = "qlh_edge_inference"
DB_USER = "postgres"
DB_PASSWORD = "123456"
DB_MIN_CONN = 2
DB_MAX_CONN = 8


# ================================================================
# 连接池管理
# ================================================================

_connection_pool: Optional[pool.ThreadedConnectionPool] = None
_init_lock = threading.Lock()


def get_pool() -> pool.ThreadedConnectionPool:
    """获取数据库连接池（懒初始化，线程安全）"""
    global _connection_pool
    if _connection_pool is not None:
        return _connection_pool

    with _init_lock:
        if _connection_pool is not None:
            return _connection_pool

        # 尝试连接并自动创建数据库
        _ensure_database()

        _connection_pool = pool.ThreadedConnectionPool(
            DB_MIN_CONN,
            DB_MAX_CONN,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info(f"数据库连接池已创建: {DB_HOST}:{DB_PORT}/{DB_NAME} "
                     f"(min={DB_MIN_CONN}, max={DB_MAX_CONN})")

        # 初始化表结构
        _init_schema()

        return _connection_pool


def _ensure_database() -> None:
    """确保目标数据库存在，不存在则创建"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname="postgres", user=DB_USER, password=DB_PASSWORD,
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_NAME,)
        )
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_NAME)
            ))
            logger.info(f"数据库已创建: {DB_NAME}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"数据库自动创建失败（可能已存在）: {e}")


def _init_schema() -> None:
    """首次运行时创建所有表"""
    with get_conn() as conn:
        cur = conn.cursor()

        # ---- 节点注册表 ----
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id       VARCHAR(64) PRIMARY KEY,
                role          VARCHAR(16) NOT NULL DEFAULT 'client',
                state         VARCHAR(16) NOT NULL DEFAULT 'offline',
                address       VARCHAR(128) NOT NULL DEFAULT '',
                hostname      VARCHAR(256) NOT NULL DEFAULT '',
                device_info   JSONB NOT NULL DEFAULT '{}',
                network_type  VARCHAR(16) NOT NULL DEFAULT 'unknown',
                connected_at  DOUBLE PRECISION NOT NULL DEFAULT 0,
                last_heartbeat DOUBLE PRECISION NOT NULL DEFAULT 0,
                task_count    INTEGER NOT NULL DEFAULT 0,
                error_count   INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMP DEFAULT NOW(),
                updated_at    TIMESTAMP DEFAULT NOW()
            )
        """)

        # ---- 对话历史表 ----
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id            SERIAL PRIMARY KEY,
                session_id    VARCHAR(64) NOT NULL DEFAULT 'default',
                role          VARCHAR(16) NOT NULL,
                content       TEXT NOT NULL,
                metrics       JSONB DEFAULT '{}',
                file_context  JSONB DEFAULT NULL,
                created_at    TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations(session_id, created_at)
        """)

        # ---- 集群配置表（键值对） ----
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cluster_config (
                key           VARCHAR(128) PRIMARY KEY,
                value         TEXT NOT NULL DEFAULT '',
                updated_at    TIMESTAMP DEFAULT NOW()
            )
        """)

        # ---- 会话元数据表 ----
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            VARCHAR(64) PRIMARY KEY,
                title         VARCHAR(256) NOT NULL DEFAULT '新对话',
                created_at    TIMESTAMP DEFAULT NOW(),
                updated_at    TIMESTAMP DEFAULT NOW(),
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        conn.commit()
        logger.info("数据库表结构初始化完成 (nodes, conversations, cluster_config, sessions)")


# ================================================================
# 连接上下文管理器
# ================================================================

@contextmanager
def get_conn():
    """从连接池获取一个连接，事务结束后自动归还"""
    pool_ = get_pool()
    conn = pool_.getconn()
    try:
        yield conn
    finally:
        pool_.putconn(conn)


# ================================================================
# 节点管理
# ================================================================

def upsert_node(node_id: str, role: str = "client", state: str = "offline",
                address: str = "", hostname: str = "",
                device_info: dict = None, network_type: str = "unknown",
                connected_at: float = 0.0, last_heartbeat: float = 0.0,
                task_count: int = 0, error_count: int = 0) -> dict:
    """插入或更新节点记录，返回完整节点 dict"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO nodes (node_id, role, state, address, hostname,
                               device_info, network_type, connected_at,
                               last_heartbeat, task_count, error_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                role = EXCLUDED.role,
                state = EXCLUDED.state,
                address = EXCLUDED.address,
                hostname = EXCLUDED.hostname,
                device_info = EXCLUDED.device_info,
                network_type = EXCLUDED.network_type,
                connected_at = EXCLUDED.connected_at,
                last_heartbeat = EXCLUDED.last_heartbeat,
                task_count = EXCLUDED.task_count,
                error_count = EXCLUDED.error_count,
                updated_at = NOW()
            RETURNING *
        """, (
            node_id, role, state, address, hostname,
            json.dumps(device_info or {}),
            network_type, connected_at,
            last_heartbeat, task_count, error_count,
        ))
        conn.commit()
        row = cur.fetchone()
        return dict(row) if row else {}


def get_all_nodes() -> list[dict]:
    """获取所有节点"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM nodes ORDER BY node_id")
        return [dict(r) for r in cur.fetchall()]


def get_node(node_id: str) -> Optional[dict]:
    """获取单个节点"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM nodes WHERE node_id = %s", (node_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_node(node_id: str) -> bool:
    """删除节点记录"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM nodes WHERE node_id = %s", (node_id,))
        conn.commit()
        return cur.rowcount > 0


def update_node_state(node_id: str, state: str,
                      last_heartbeat: float = None,
                      task_count: int = None,
                      error_count: int = None) -> Optional[dict]:
    """部分更新节点状态字段"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        sets = ["state = %s", "updated_at = NOW()"]
        params = [state]
        if last_heartbeat is not None:
            sets.append("last_heartbeat = %s")
            params.append(last_heartbeat)
        if task_count is not None:
            sets.append("task_count = %s")
            params.append(task_count)
        if error_count is not None:
            sets.append("error_count = %s")
            params.append(error_count)
        params.append(node_id)
        cur.execute(
            f"UPDATE nodes SET {', '.join(sets)} WHERE node_id = %s RETURNING *",
            params,
        )
        conn.commit()
        row = cur.fetchone()
        return dict(row) if row else None


# ================================================================
# 对话历史
# ================================================================

def save_message(session_id: str, role: str, content: str,
                 metrics: dict = None, file_context: dict = None) -> dict:
    """保存一条对话消息"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO conversations (session_id, role, content, metrics, file_context)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """, (
            session_id, role, content,
            json.dumps(metrics or {}),
            json.dumps(file_context) if file_context else None,
        ))
        conn.commit()
        row = cur.fetchone()
        d = dict(row) if row else {}
        # 转换 JSONB 字段
        if "metrics" in d and isinstance(d["metrics"], str):
            d["metrics"] = json.loads(d["metrics"])
        if "file_context" in d and isinstance(d["file_context"], str):
            d["file_context"] = json.loads(d["file_context"])
        # 转换 datetime 为时间戳
        if "created_at" in d and d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        return d


def get_conversation(session_id: str = "default",
                     limit: int = 200) -> list[dict]:
    """获取指定会话的对话历史（最近 N 条）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, session_id, role, content, metrics, file_context, created_at
            FROM conversations
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (session_id, limit))
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if "metrics" in d and isinstance(d["metrics"], str):
                d["metrics"] = json.loads(d["metrics"])
            if "file_context" in d and isinstance(d["file_context"], str):
                d["file_context"] = json.loads(d["file_context"])
            if "created_at" in d and d["created_at"]:
                d["created_at"] = d["created_at"].isoformat()
            result.append(d)
        return result


def clear_conversation(session_id: str = "default") -> int:
    """清空指定会话的对话历史，返回删除条数"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversations WHERE session_id = %s",
            (session_id,)
        )
        conn.commit()
        return cur.rowcount


def get_conversation_count(session_id: str = "default") -> int:
    """获取会话消息条数"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversations WHERE session_id = %s",
            (session_id,)
        )
        return cur.fetchone()[0]


def delete_message_range(session_id: str, turn_index: int) -> int:
    """
    删除指定会话中某轮对话（user + assistant 两条消息）。

    turn_index: 0-based 对话轮次索引。
    turn 0 = 第1个user + 第1个assistant，
    turn 1 = 第2个user + 第2个assistant，以此类推。

    使用 CTE 定位要删除的行 ID，然后 DELETE。
    返回实际删除的行数。
    """
    with get_conn() as conn:
        cur = conn.cursor()
        # 使用子查询定位第 2*turn_index 和 2*turn_index+1 条消息的 ID
        cur.execute("""
            WITH ordered AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC) - 1 AS rn
                FROM conversations
                WHERE session_id = %s
            )
            DELETE FROM conversations
            WHERE id IN (
                SELECT id FROM ordered
                WHERE rn IN (%s, %s)
            )
        """, (session_id, 2 * turn_index, 2 * turn_index + 1))
        conn.commit()
        return cur.rowcount


# ================================================================
# 会话管理（多会话支持）
# ================================================================

def create_session(session_id: str, title: str = "新对话") -> dict:
    """创建新会话记录"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO sessions (id, title)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = NOW()
            RETURNING *
        """, (session_id, title))
        conn.commit()
        row = cur.fetchone()
        d = dict(row) if row else {}
        if "created_at" in d and d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d and d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def get_all_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """获取所有会话列表（按 updated_at DESC 排序）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM sessions
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if "created_at" in d and d["created_at"]:
                d["created_at"] = d["created_at"].isoformat()
            if "updated_at" in d and d["updated_at"]:
                d["updated_at"] = d["updated_at"].isoformat()
            result.append(d)
        return result


def get_session(session_id: str) -> Optional[dict]:
    """获取单个会话元数据"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if "created_at" in d and d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d and d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def get_session_count() -> int:
    """获取会话总数"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        return cur.fetchone()[0]


def update_session_title(session_id: str, title: str) -> Optional[dict]:
    """更新会话标题"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            UPDATE sessions
            SET title = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING *
        """, (title, session_id))
        conn.commit()
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        if "created_at" in d and d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d and d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def delete_session(session_id: str) -> int:
    """
    删除会话及其所有对话消息（事务内完成）。
    返回删除的会话数（0 或 1）。
    """
    with get_conn() as conn:
        cur = conn.cursor()
        # 先删消息（外键未定义，需手动级联）
        cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        # 再删会话
        cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        conn.commit()
        return cur.rowcount


def increment_session_message_count(session_id: str) -> None:
    """会话消息数 +1，同时更新 updated_at"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE sessions
            SET message_count = message_count + 1, updated_at = NOW()
            WHERE id = %s
        """, (session_id,))
        conn.commit()


def decrement_session_message_count(session_id: str, count: int = 2) -> None:
    """会话消息数 -count（不小于0），同时更新 updated_at"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE sessions
            SET message_count = GREATEST(0, message_count - %s),
                updated_at = NOW()
            WHERE id = %s
        """, (count, session_id,))
        conn.commit()


# ================================================================
# 集群配置
# ================================================================

def get_config(key: str, default: str = "") -> str:
    """读取配置项"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM cluster_config WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else default


def set_config(key: str, value: str) -> None:
    """写入配置项"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cluster_config (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
        """, (key, value))
        conn.commit()


def get_all_configs() -> dict[str, str]:
    """获取所有配置项"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM cluster_config ORDER BY key")
        return {r[0]: r[1] for r in cur.fetchall()}


def set_configs_batch(configs: dict[str, str]) -> None:
    """批量写入配置"""
    with get_conn() as conn:
        cur = conn.cursor()
        for key, value in configs.items():
            cur.execute("""
                INSERT INTO cluster_config (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
            """, (key, str(value)))
        conn.commit()


# ================================================================
# 主节点服务注册（用于从节点自动发现主节点）
# ================================================================

def register_master(host: str, port: int, node_id: str = "master",
                    mac_addresses: list = None) -> None:
    """
    将主节点连接信息写入数据库，供从节点启动时自动发现。

    写入 cluster_config 表：
      - master_host:           主节点局域网 IP（可变动）
      - master_port:           主节点 TCP 监听端口
      - master_node_id:        主节点标识
      - master_mac_addresses:  主节点物理网卡 MAC 地址集合（逗号分隔，不可变身份标识）
      - master_last_seen:      Unix 时间戳（心跳刷新）

    从节点启动时调用 get_master_info() 读取这些信息，
    即可在后台管理中自动发现主节点。

    MAC 地址的作用：
    - IP 可能随 DHCP/网络切换变化，MAC 是硬件级不变标识
    - 主节点重启后通过 MAC 验证身份，防止其他机器冒充
    - 首次启动记录，后续启动校验
    """
    now_ts = str(time.time())
    configs = {
        "master_host": host,
        "master_port": str(port),
        "master_node_id": node_id,
        "master_last_seen": now_ts,
    }
    if mac_addresses:
        configs["master_mac_addresses"] = ",".join(mac_addresses)
    set_configs_batch(configs)
    mac_info = f", MAC={mac_addresses}" if mac_addresses else ""
    logger.info(f"主节点已注册到数据库: {host}:{port} (node_id={node_id}{mac_info})")


def get_master_info() -> dict:
    """
    从数据库获取主节点的连接信息。

    用于从节点启动时自动发现主节点。
    如果 master_last_seen 超过 120 秒未更新，视为过期（stale）。

    Returns:
        {
            "found": bool,
            "master_host": str,
            "master_port": int,
            "master_node_id": str,
            "master_mac_addresses": [str],   # 主节点物理网卡 MAC 列表
            "last_seen": float,
            "stale": bool,
        }
    """
    host = get_config("master_host", "")
    port_str = get_config("master_port", "")
    node_id = get_config("master_node_id", "master")
    last_seen_str = get_config("master_last_seen", "")
    mac_str = get_config("master_mac_addresses", "")

    if not host or not port_str:
        return {"found": False}

    try:
        last_seen = float(last_seen_str) if last_seen_str else 0.0
    except ValueError:
        last_seen = 0.0

    stale = (time.time() - last_seen) > 120 if last_seen > 0 else True

    mac_addresses = [m.strip() for m in mac_str.split(",") if m.strip()] if mac_str else []

    return {
        "found": True,
        "master_host": host,
        "master_port": int(port_str),
        "master_node_id": node_id,
        "master_mac_addresses": mac_addresses,
        "last_seen": last_seen,
        "stale": stale,
    }


def verify_master_identity(local_macs: list[str]) -> dict:
    """
    验证本机 MAC 地址是否与数据库中记录的主节点 MAC 匹配。

    用于主节点重启时：检测本机所有物理网卡 MAC，与 DB 中存储的
    master_mac_addresses 逐一比对，只要有交集即验证通过。

    如果 DB 中尚无 master_mac_addresses 记录（首次启动），
    返回 {"verified": false, "reason": "first_run"}，
    调用方应随后写入 MAC 记录。

    Args:
        local_macs: 本机检测到的物理网卡 MAC 地址列表

    Returns:
        {
            "verified": bool,        # 是否通过验证
            "reason": str,           # "match" | "first_run" | "mac_mismatch" | "no_db_record"
            "db_macs": [str],        # DB 中记录的 MAC 列表
            "local_macs": [str],     # 本机检测到的 MAC 列表
            "matched": [str],        # 匹配上的 MAC 列表
        }
    """
    db_macs_str = get_config("master_mac_addresses", "")
    db_macs = [m.strip() for m in db_macs_str.split(",") if m.strip()] if db_macs_str else []

    if not db_macs:
        return {
            "verified": False,
            "reason": "first_run",
            "db_macs": [],
            "local_macs": local_macs,
            "matched": [],
        }

    local_set = set(local_macs)
    db_set = set(db_macs)
    matched = list(local_set & db_set)

    if matched:
        return {
            "verified": True,
            "reason": "match",
            "db_macs": db_macs,
            "local_macs": local_macs,
            "matched": matched,
        }
    else:
        return {
            "verified": False,
            "reason": "mac_mismatch",
            "db_macs": db_macs,
            "local_macs": local_macs,
            "matched": [],
        }


def reset_master_identity(new_macs: list[str] = None) -> bool:
    """
    重置主节点身份标识（仅在需要更换主节点机器时使用）。

    清除 DB 中的 master_mac_addresses，下次启动时重新记录。
    如果提供了 new_macs，则直接写入新的 MAC 地址集合。

    Args:
        new_macs: 新的 MAC 地址列表（可选）
    """
    if new_macs:
        set_config("master_mac_addresses", ",".join(new_macs))
        logger.info(f"主节点身份已重置: MAC={new_macs}")
    else:
        set_config("master_mac_addresses", "")
        logger.info("主节点身份已清除，下次启动将重新记录")
    return True


def update_master_heartbeat() -> bool:
    """
    更新主节点心跳时间戳（由主节点周期性调用）。

    仅更新 master_last_seen，不修改其他字段。
    如果已有 master_host 记录则更新，否则返回 False（应先调用 register_master）。

    Returns:
        是否更新成功
    """
    host = get_config("master_host", "")
    if not host:
        return False
    set_config("master_last_seen", str(time.time()))
    return True


# ================================================================
# 分布式推理开关
# ================================================================

def get_distributed_inference_enabled() -> bool:
    """读取分布式推理开关状态（默认 True）"""
    val = get_config("distributed_inference_enabled", "")
    if val == "":
        return True
    return val.lower() == "true"


def set_distributed_inference_enabled(enabled: bool) -> None:
    """设置分布式推理开关状态"""
    set_config("distributed_inference_enabled", "true" if enabled else "false")
    logger.info(f"分布式推理已{'启用' if enabled else '禁用'}")


# ================================================================
# 动态分层配置
# ================================================================

def get_layer_strategy() -> str:
    """获取分层策略: 'dynamic' | 'manual'（默认 'dynamic'）"""
    val = get_config("layer_strategy", "")
    return val if val in ("dynamic", "manual") else "dynamic"


def set_layer_strategy(strategy: str) -> None:
    """设置分层策略"""
    if strategy not in ("dynamic", "manual"):
        raise ValueError(f"无效的分层策略: {strategy}")
    set_config("layer_strategy", strategy)


def get_layer_assignments() -> Optional[dict]:
    """
    获取动态计算的分层分配方案（JSON）。

    Returns:
        {
            "total": 24,
            "strategy": "dynamic",
            "computed_at": 1234567890.0,
            "assignments": [
                {node_id, role, start_layer, end_layer, has_embedding, has_lm_head, score},
                ...
            ]
        }
        若无记录返回 None
    """
    val = get_config("layer_assignments", "")
    if not val:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def set_layer_assignments(assignments: dict) -> None:
    """存储动态计算的分层分配方案（JSON）"""
    set_config("layer_assignments", json.dumps(assignments, ensure_ascii=False))


def get_layer_override() -> Optional[list]:
    """
    获取手动覆盖的分层配置。

    Returns:
        [{node_id, start_layer, end_layer}, ...] 或无记录时返回 None
    """
    val = get_config("layer_override", "")
    if not val:
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def set_layer_override(overrides: list) -> None:
    """存储手动覆盖的分层配置（JSON）"""
    set_config("layer_override", json.dumps(overrides, ensure_ascii=False))


# ================================================================
# 用户偏好设置（JSON 云同步）
# ================================================================

def get_user_settings() -> dict:
    """
    从云数据库读取用户偏好设置（JSON）。

    存储所有前端设置项：maxNewTokens, temperature, topP,
    saveHistory, distributedInference, theme 等。

    Returns:
        settings dict，若无记录返回 {}
    """
    val = get_config("user_settings", "")
    if not val:
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def set_user_settings(settings: dict) -> None:
    """
    将用户偏好设置存储到云数据库（JSON）。

    与 localStorage 同步写入，确保云端/本地一致。
    """
    set_config("user_settings", json.dumps(settings, ensure_ascii=False))
    logger.info(f"用户设置已同步到云端: {len(settings)} 项")


# ================================================================
# 对话云同步开关
# ================================================================

def get_save_history() -> bool:
    """读取对话历史保存开关（默认 True — 云端持久化开启）"""
    val = get_config("save_history", "")
    if val == "":
        return True  # 默认开启，确保跨设备数据共享
    return val.lower() == "true"


def set_save_history(enabled: bool) -> None:
    """设置对话历史保存开关"""
    set_config("save_history", "true" if enabled else "false")
    logger.info(f"对话云同步已{'启用' if enabled else '禁用'}")


# ================================================================
# 角色转让日志
# ================================================================

def append_transfer_log(direction: str, from_role: str, to_role: str,
                        related_node: str, details: dict = None) -> dict:
    """
    追加一条角色转让日志。

    Args:
        direction: "demotion"（降级日志，原主节点记录）| "promotion"（升级日志，新主节点记录）
        from_role: 转让前角色
        to_role: 转让后角色
        related_node: 关联节点 ID（对方的 node_id）
        details: 额外详情（集群状态等）

    Returns:
        写入的日志条目
    """
    import time as _time
    entry = {
        "direction": direction,
        "from_role": from_role,
        "to_role": to_role,
        "related_node": related_node,
        "timestamp": _time.time(),
        "timestamp_iso": _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime()),
        "details": details or {},
    }
    # 追加到 JSON 数组
    key = f"transfer_log_{direction}"
    existing_json = get_config(key, "")
    logs = []
    if existing_json:
        try:
            logs = json.loads(existing_json)
        except (json.JSONDecodeError, TypeError):
            logs = []
    logs.append(entry)
    set_config(key, json.dumps(logs, ensure_ascii=False))
    logger.info(f"角色转让日志已记录: {direction} {from_role}→{to_role} (关联节点: {related_node})")
    return entry


def get_transfer_logs(direction: str = None) -> list:
    """
    获取角色转让日志。

    Args:
        direction: "demotion" | "promotion" | None（返回全部）

    Returns:
        日志条目列表
    """
    if direction:
        keys = [f"transfer_log_{direction}"]
    else:
        keys = ["transfer_log_demotion", "transfer_log_promotion"]

    all_logs = []
    for key in keys:
        val = get_config(key, "")
        if val:
            try:
                all_logs.extend(json.loads(val))
            except (json.JSONDecodeError, TypeError):
                pass
    all_logs.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    return all_logs


# ================================================================
# 备用主节点管理
# ================================================================

def get_spare_master() -> Optional[dict]:
    """
    获取当前备用主节点信息。

    Returns:
        {node_id, hostname, address, designated_at, ...} 或 None
    """
    val = get_config("spare_master", "")
    if val:
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def set_spare_master(node_id: str, hostname: str = "", address: str = "") -> None:
    """
    指定备用主节点。

    Args:
        node_id: 从节点 ID
        hostname: 主机名
        address: 网络地址
    """
    data = {
        "node_id": node_id,
        "hostname": hostname,
        "address": address,
        "designated_at": time.time(),
    }
    set_config("spare_master", json.dumps(data, ensure_ascii=False))
    logger.info(f"备用主节点已设置: {node_id}")


def clear_spare_master() -> None:
    """清除备用主节点指定"""
    set_config("spare_master", "")
    logger.info("备用主节点已清除")


def append_spare_master_log(direction: str, details: dict) -> None:
    """
    追加备用主节点操作日志。

    Args:
        direction: "designated"（被指定为备用）| "undesignated"（取消备用）
        details: 日志详情
    """
    key = "spare_master_log"
    existing = get_config(key, "[]")
    try:
        logs = json.loads(existing)
    except (json.JSONDecodeError, TypeError):
        logs = []
    logs.append({
        "direction": direction,
        "timestamp": time.time(),
        "details": details,
    })
    # 只保留最近 200 条
    set_config(key, json.dumps(logs[-200:], ensure_ascii=False))


def get_spare_master_logs() -> list:
    """获取备用主节点操作日志"""
    val = get_config("spare_master_log", "[]")
    try:
        logs = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        logs = []
    logs.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    return logs


# ================================================================
# 健康检查
# ================================================================

def db_health() -> dict:
    """数据库连接健康检查"""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            return {"status": "ok", "host": DB_HOST, "port": DB_PORT, "db": DB_NAME}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================================================================
# 初始化入口
# ================================================================

def init_db():
    """应用启动时调用：初始化连接池 + 表结构"""
    get_pool()
    logger.info("数据库初始化完成")


def close_db():
    """应用关闭时调用：关闭连接池"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("数据库连接池已关闭")
