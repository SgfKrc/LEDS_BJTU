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

数据库: 8.160.161.53:5432 / postgres / WUTqw6bLkK3Hn5Va
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
DB_PASSWORD = "WUTqw6bLkK3Hn5Va"
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
                     f"(min={DB_MIN_CONN}, max={DB_MAX_CONN}) "
                     f"user={DB_USER} pwd={DB_PASSWORD[:4]}...{DB_PASSWORD[-4:]}")

        # 初始化表结构
        _init_schema()

        return _connection_pool


# ================================================================
# 活跃节点 ID（用于数据隔离）
# 由 api_server 在启动时调用 set_active_node_id() 设置
# ================================================================
_active_node_id: str = "master"


def set_active_node_id(node_id: str) -> None:
    """设置当前活跃节点 ID，用于 conversations/sessions 数据隔离"""
    global _active_node_id
    _active_node_id = node_id or "master"
    logger.info(f"活跃节点 ID: {_active_node_id}")


def get_active_node_id() -> str:
    return _active_node_id


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
                node_type     VARCHAR(16) NOT NULL DEFAULT 'pc',
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
                node_id       VARCHAR(64) NOT NULL DEFAULT 'master',
                created_at    TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations(session_id, created_at)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_node
            ON conversations(node_id, created_at)
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
                node_id       VARCHAR(64) NOT NULL DEFAULT 'master',
                created_at    TIMESTAMP DEFAULT NOW(),
                updated_at    TIMESTAMP DEFAULT NOW(),
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_node
            ON sessions(node_id, updated_at DESC)
        """)

        # ---- 审查票表 (P3) ----
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_tickets (
                ticket_id         VARCHAR(64) PRIMARY KEY,
                status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                created_at        DOUBLE PRECISION NOT NULL,
                created_by        VARCHAR(64) NOT NULL,
                target_node_id    VARCHAR(64) NOT NULL,
                transfer_reason   TEXT NOT NULL DEFAULT '',
                votes             JSONB NOT NULL DEFAULT '[]',
                score             INTEGER NOT NULL DEFAULT 0,
                expires_at        DOUBLE PRECISION NOT NULL,
                resolved_at       DOUBLE PRECISION DEFAULT NULL,
                notification_sent BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at        TIMESTAMP DEFAULT NOW()
            )
        """)

        conn.commit()

    # ★ 迁移：为旧版表添加 node_id / node_type / model_sha256 列（在事务外执行，避免回滚影响建表）
    #    使用连接池而非独立连接，避免独立连接因网络/DNS 问题静默失败
    _migrate_add_node_id_columns()
    _migrate_add_node_type_column()
    _migrate_add_model_sha256_column()

    logger.info("数据库表结构初始化完成 (nodes, conversations, cluster_config, sessions, review_tickets)")


def _migrate_add_node_id_columns() -> None:
    """
    为旧版表添加 node_id 列（幂等迁移）。

    旧版 conversations / sessions 表可能缺少 node_id 列。
    此迁移使用 ALTER TABLE ... ADD COLUMN IF NOT EXISTS，
    已存在的列不会被修改。

    使用连接池连接并设置 autocommit，避免：
    1. 独立 psycopg2.connect() 因网络/DNS 问题静默失败
    2. 迁移失败导致整个 _init_schema 事务回滚
    """
    pool_ = None
    conn = None
    try:
        pool_ = get_pool()
        conn = pool_.getconn()
        conn.autocommit = True
        cur = conn.cursor()
        try:
            cur.execute(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS "
                "node_id VARCHAR(64) NOT NULL DEFAULT 'master'"
            )
            logger.info("迁移: conversations.node_id 列已就绪")
        except Exception as e:
            logger.warning(f"迁移 conversations.node_id 跳过: {e}")
        try:
            cur.execute(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "
                "node_id VARCHAR(64) NOT NULL DEFAULT 'master'"
            )
            logger.info("迁移: sessions.node_id 列已就绪")
        except Exception as e:
            logger.warning(f"迁移 sessions.node_id 跳过: {e}")
        cur.close()
    except Exception as e:
        logger.warning(f"node_id 列迁移失败（非致命）: {e}")
    finally:
        if conn and pool_:
            try:
                pool_.putconn(conn)
            except Exception:
                pass


def _migrate_add_node_type_column() -> None:
    """
    为旧版 nodes 表添加 node_type 列（幂等迁移）。

    旧版 nodes 表可能缺少 node_type 列（默认 'pc'）。
    此迁移使用 ALTER TABLE ... ADD COLUMN IF NOT EXISTS。
    """
    pool_ = None
    conn = None
    try:
        pool_ = get_pool()
        conn = pool_.getconn()
        conn.autocommit = True
        cur = conn.cursor()
        try:
            cur.execute(
                "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS "
                "node_type VARCHAR(16) NOT NULL DEFAULT 'pc'"
            )
            logger.info("迁移: nodes.node_type 列已就绪")
        except Exception as e:
            logger.warning(f"迁移 nodes.node_type 跳过: {e}")
        cur.close()
    except Exception as e:
        logger.warning(f"node_type 列迁移失败（非致命）: {e}")
    finally:
        if conn and pool_:
            try:
                pool_.putconn(conn)
            except Exception:
                pass


def _migrate_add_model_sha256_column() -> None:
    """
    为 nodes 表添加 model_sha256 列（幂等迁移，阶段 7）。

    用于存储各节点上报的模型 SHA256 校验值，
    主节点在推送分层配置前对比校验，模型不一致的节点将被排除。
    """
    pool_ = None
    conn = None
    try:
        pool_ = get_pool()
        conn = pool_.getconn()
        conn.autocommit = True
        cur = conn.cursor()
        try:
            cur.execute(
                "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS "
                "model_sha256 VARCHAR(64) NOT NULL DEFAULT ''"
            )
            logger.info("迁移: nodes.model_sha256 列已就绪")
        except Exception as e:
            logger.warning(f"迁移 nodes.model_sha256 跳过: {e}")
        cur.close()
    except Exception as e:
        logger.warning(f"model_sha256 列迁移失败（非致命）: {e}")
    finally:
        if conn and pool_:
            try:
                pool_.putconn(conn)
            except Exception:
                pass


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
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        pool_.putconn(conn)


# ================================================================
# 节点管理
# ================================================================

def upsert_node(node_id: str, role: str = "client", state: str = "offline",
                node_type: str = "pc", address: str = "", hostname: str = "",
                device_info: dict = None, network_type: str = "unknown",
                connected_at: float = 0.0, last_heartbeat: float = 0.0,
                task_count: int = 0, error_count: int = 0,
                model_sha256: str = "") -> dict:
    """插入或更新节点记录，返回完整节点 dict"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO nodes (node_id, role, node_type, state, address, hostname,
                               device_info, network_type, connected_at,
                               last_heartbeat, task_count, error_count, model_sha256)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                role = EXCLUDED.role,
                node_type = EXCLUDED.node_type,
                state = EXCLUDED.state,
                address = EXCLUDED.address,
                hostname = EXCLUDED.hostname,
                device_info = EXCLUDED.device_info,
                network_type = EXCLUDED.network_type,
                connected_at = EXCLUDED.connected_at,
                last_heartbeat = EXCLUDED.last_heartbeat,
                task_count = EXCLUDED.task_count,
                error_count = EXCLUDED.error_count,
                model_sha256 = EXCLUDED.model_sha256,
                updated_at = NOW()
            RETURNING *
        """, (
            node_id, role, node_type, state, address, hostname,
            json.dumps(device_info or {}),
            network_type, connected_at,
            last_heartbeat, task_count, error_count, model_sha256,
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
            INSERT INTO conversations (session_id, role, content, metrics, file_context, node_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (
            session_id, role, content,
            json.dumps(metrics or {}),
            json.dumps(file_context) if file_context else None,
            _active_node_id,
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
    """获取指定会话的对话历史（最近 N 条，仅本节点）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, session_id, role, content, metrics, file_context, created_at
            FROM conversations
            WHERE session_id = %s AND node_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (session_id, _active_node_id, limit))
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
    """清空指定会话的对话历史（仅本节点），返回删除条数"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversations WHERE session_id = %s AND node_id = %s",
            (session_id, _active_node_id)
        )
        conn.commit()
        return cur.rowcount


def get_conversation_count(session_id: str = "default") -> int:
    """获取会话消息条数（仅本节点）"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM conversations WHERE session_id = %s AND node_id = %s",
            (session_id, _active_node_id)
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
                WHERE session_id = %s AND node_id = %s
            )
            DELETE FROM conversations
            WHERE id IN (
                SELECT id FROM ordered
                WHERE rn IN (%s, %s)
            )
        """, (session_id, _active_node_id, 2 * turn_index, 2 * turn_index + 1))
        conn.commit()
        return cur.rowcount


# ================================================================
# 会话管理（多会话支持）
# ================================================================

def create_session(session_id: str, title: str = "新对话") -> dict:
    """创建新会话记录（关联到当前节点）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            INSERT INTO sessions (id, title, node_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = NOW(),
                node_id = EXCLUDED.node_id
            RETURNING *
        """, (session_id, title, _active_node_id))
        conn.commit()
        row = cur.fetchone()
        d = dict(row) if row else {}
        if "created_at" in d and d["created_at"]:
            d["created_at"] = d["created_at"].isoformat()
        if "updated_at" in d and d["updated_at"]:
            d["updated_at"] = d["updated_at"].isoformat()
        return d


def get_all_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """获取本节点的所有会话列表（按 updated_at DESC 排序）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM sessions
            WHERE node_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """, (_active_node_id, limit, offset))
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
    """获取单个会话元数据（仅本节点）"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM sessions WHERE id = %s AND node_id = %s",
            (session_id, _active_node_id)
        )
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
    """获取本节点会话总数"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions WHERE node_id = %s", (_active_node_id,))
        return cur.fetchone()[0]


def update_session_title(session_id: str, title: str) -> Optional[dict]:
    """更新会话标题"""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            UPDATE sessions
            SET title = %s, updated_at = NOW()
            WHERE id = %s AND node_id = %s
            RETURNING *
        """, (title, session_id, _active_node_id))
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
        # 先删消息（仅本节点，防止误删其他节点同名会话）
        cur.execute("DELETE FROM conversations WHERE session_id = %s AND node_id = %s",
                    (session_id, _active_node_id))
        # 再删会话
        cur.execute("DELETE FROM sessions WHERE id = %s AND node_id = %s",
                    (session_id, _active_node_id))
        conn.commit()
        return cur.rowcount


def increment_session_message_count(session_id: str) -> None:
    """会话消息数 +1，同时更新 updated_at（仅本节点）"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE sessions
            SET message_count = message_count + 1, updated_at = NOW()
            WHERE id = %s AND node_id = %s
        """, (session_id, _active_node_id))
        conn.commit()


def decrement_session_message_count(session_id: str, count: int = 2) -> None:
    """会话消息数 -count（不小于0），同时更新 updated_at（仅本节点）"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE sessions
            SET message_count = GREATEST(0, message_count - %s),
                updated_at = NOW()
            WHERE id = %s AND node_id = %s
        """, (count, session_id, _active_node_id))
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


def clear_layer_override() -> None:
    """清除手动覆盖的分层配置，恢复自动策略"""
    set_config("layer_override", "")


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
# 实验模型注册（P3: 多模型实验支持）
# 存储: cluster_config 表，key 前缀 "experimental_model:"
# ================================================================

EXPERIMENTAL_MODEL_KEY_PREFIX = "experimental_model:"


def save_experimental_model(model_id: str, config_json: str) -> bool:
    """注册/更新一个实验模型配置。

    Args:
        model_id: 模型唯一标识
        config_json: JSON 编码的模型配置 dict

    Returns:
        是否保存成功
    """
    key = f"{EXPERIMENTAL_MODEL_KEY_PREFIX}{model_id}"
    try:
        set_config(key, config_json)
        logger.info(f"实验模型已注册: {model_id}")
        return True
    except Exception as e:
        logger.error(f"注册实验模型失败 ({model_id}): {e}")
        return False


def get_experimental_models() -> list[dict]:
    """获取所有用户注册的实验模型配置。

    Returns:
        模型配置 dict 列表（已解析 JSON）
    """
    try:
        all_configs = get_all_configs()
    except Exception as e:
        logger.warning(f"读取实验模型列表失败: {e}")
        return []

    models = []
    for key, value in all_configs.items():
        if not key.startswith(EXPERIMENTAL_MODEL_KEY_PREFIX):
            continue
        try:
            config = json.loads(value)
            if isinstance(config, dict):
                # P3修复: 不覆盖存储中已有的 model_id，使用单独的键
                if "model_id" not in config:
                    config["model_id"] = key[len(EXPERIMENTAL_MODEL_KEY_PREFIX):]
                models.append(config)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"解析实验模型配置失败: {key}")
            continue

    return models


def delete_experimental_model(model_id: str) -> bool:
    """删除一个实验模型注册。

    Args:
        model_id: 模型唯一标识

    Returns:
        是否删除成功
    """
    key = f"{EXPERIMENTAL_MODEL_KEY_PREFIX}{model_id}"
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM cluster_config WHERE key = %s", (key,))
            conn.commit()
            deleted = cur.rowcount > 0
            if deleted:
                logger.info(f"实验模型已删除: {model_id}")
            else:
                logger.warning(f"实验模型不存在: {model_id}")
            return deleted
    except Exception as e:
        logger.error(f"删除实验模型失败 ({model_id}): {e}")
        return False


def set_active_model(model_id: str) -> None:
    """记录当前活跃的模型 ID。"""
    set_config("active_model_id", model_id)


def get_active_model() -> str:
    """获取当前活跃的模型 ID（默认 qwen-1_8b）。"""
    return get_config("active_model_id", "qwen-1_8b")


# ================================================================
# 审查票 CRUD (P3: 主节点转让审查)
# ================================================================

def create_review_ticket(ticket: dict) -> dict:
    """创建审查工单。

    Args:
        ticket: 工单 dict，至少包含 ticket_id。

    Returns:
        写入的工单 dict。
    """
    votes_raw = ticket.get("votes")
    if votes_raw is None:
        votes_json = "[]"
    elif isinstance(votes_raw, (list, dict)):
        votes_json = json.dumps(votes_raw, ensure_ascii=False)
    elif isinstance(votes_raw, str):
        votes_json = votes_raw
    else:
        votes_json = json.dumps(votes_raw, ensure_ascii=False)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO review_tickets
                (ticket_id, status, created_at, created_by, target_node_id,
                 transfer_reason, votes, score, expires_at, resolved_at,
                 notification_sent)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticket_id) DO UPDATE SET
                status = EXCLUDED.status,
                votes = EXCLUDED.votes,
                score = EXCLUDED.score,
                updated_at = NOW()
            RETURNING *
        """, (
            ticket["ticket_id"],
            ticket.get("status", "pending"),
            ticket.get("created_at", 0.0),
            ticket.get("created_by", ""),
            ticket.get("target_node_id", ""),
            ticket.get("transfer_reason", ""),
            votes_json,
            ticket.get("score", 0),
            ticket.get("expires_at", 0.0),
            ticket.get("resolved_at"),
            ticket.get("notification_sent", False),
        ))
        row = cur.fetchone()
        conn.commit()
        if row:
            return _review_ticket_row_to_dict(row)
    return None


def get_review_ticket(ticket_id: str) -> Optional[dict]:
    """获取单个审查工单。"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM review_tickets WHERE ticket_id = %s",
            (ticket_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _review_ticket_row_to_dict(row)


def update_review_ticket(ticket_id: str, updates: dict) -> Optional[dict]:
    """更新审查工单的部分字段。

    Args:
        ticket_id: 工单 ID。
        updates: 要更新的字段 dict。

    Returns:
        更新后的工单 dict，或 None（不存在时）。
    """
    # 构建 SET 子句
    allowed_fields = {
        "status", "score", "votes", "resolved_at",
        "notification_sent", "transfer_reason", "expires_at",
    }
    set_clauses = []
    values = []

    for field, value in updates.items():
        if field in allowed_fields:
            set_clauses.append(f"{field} = %s")
            if field == "votes" and isinstance(value, list):
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(value)

    if not set_clauses:
        return None

    set_clauses.append("updated_at = NOW()")
    values.append(ticket_id)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE review_tickets SET {', '.join(set_clauses)} WHERE ticket_id = %s",
            values,
        )
        conn.commit()

        if cur.rowcount == 0:
            return None

        # 返回更新后的完整记录
        cur.execute("SELECT * FROM review_tickets WHERE ticket_id = %s", (ticket_id,))
        row = cur.fetchone()
        if row:
            return _review_ticket_row_to_dict(row)
    return None


def list_review_tickets(status: Optional[str] = None) -> list[dict]:
    """列出审查工单，可按状态过滤。

    Args:
        status: None（全部）或 "pending" / "approved" / "rejected" / "expired"。

    Returns:
        工单 dict 列表。
    """
    with get_conn() as conn:
        cur = conn.cursor()
        if status:
            cur.execute(
                "SELECT * FROM review_tickets WHERE status = %s ORDER BY created_at DESC",
                (status,)
            )
        else:
            cur.execute(
                "SELECT * FROM review_tickets ORDER BY created_at DESC"
            )
        rows = cur.fetchall()
        return [_review_ticket_row_to_dict(r) for r in rows]


def delete_review_ticket(ticket_id: str) -> bool:
    """删除一个审查工单（所有状态均可删除）。"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM review_tickets WHERE ticket_id = %s",
                    (ticket_id,),
                )
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        logger.warning(f"删除审查工单失败 ({ticket_id}): {e}")
        return False


def delete_resolved_review_tickets() -> int:
    """删除所有已解决（approved/rejected/expired）的审查工单，返回删除数量。"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM review_tickets WHERE status IN ('approved', 'rejected', 'expired')"
                )
                conn.commit()
                return cur.rowcount
    except Exception as e:
        logger.warning(f"批量删除已解决工单失败: {e}")
        return 0


def _review_ticket_row_to_dict(row) -> dict:
    """将数据库行转换为 dict。

    row 是 psycopg2 查询返回的 tuple（按列顺序）。
    列顺序与 CREATE TABLE 一致:
      ticket_id, status, created_at, created_by, target_node_id,
      transfer_reason, votes, score, expires_at, resolved_at,
      notification_sent, updated_at
    """
    idx = 0
    ticket_id = row[idx]; idx += 1
    status = row[idx]; idx += 1
    created_at = row[idx]; idx += 1
    created_by = row[idx]; idx += 1
    target_node_id = row[idx]; idx += 1
    transfer_reason = row[idx]; idx += 1
    votes_raw = row[idx]; idx += 1
    score = row[idx]; idx += 1
    expires_at = row[idx]; idx += 1
    resolved_at = row[idx]; idx += 1
    notification_sent = row[idx]; idx += 1
    # updated_at = row[idx] — not used in ReviewTicket

    # Parse votes
    votes = votes_raw
    if isinstance(votes_raw, str):
        try:
            votes = json.loads(votes_raw)
        except (json.JSONDecodeError, TypeError):
            votes = []

    return {
        "ticket_id": ticket_id,
        "status": status,
        "created_at": float(created_at) if created_at else 0.0,
        "created_by": created_by or "",
        "target_node_id": target_node_id or "",
        "transfer_reason": transfer_reason or "",
        "votes": votes,
        "score": int(score) if score is not None else 0,
        "expires_at": float(expires_at) if expires_at else 0.0,
        "resolved_at": float(resolved_at) if resolved_at else None,
        "notification_sent": bool(notification_sent),
    }


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
