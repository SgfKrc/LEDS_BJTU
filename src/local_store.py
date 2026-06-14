"""
本地持久化存储 — 云数据库不可用时的 JSON 文件降级方案
======================================================
当 PostgreSQL 连接失败（跨省 Tailscale、pg_hba.conf 拒绝等场景）时，
对话记录和会话元数据自动写入本地 JSON 文件，确保离线可用。

文件结构:
  logs/chat_history/
    _sessions.json          # [{id, title, created_at, updated_at, message_count}, ...]
    {session_id}.json       # [{role, content, created_at, metrics?}, ...]

线程安全: 所有写操作使用 threading.Lock 保护。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ================================================================
# 路径配置
# ================================================================

def _get_store_dir() -> str:
    """获取本地储存目录（与 config.LOG_DIR 同级）。"""
    # 尝试从 config 获取，失败则使用默认路径
    try:
        from config import LOG_DIR as _ld
        store_dir = os.path.join(os.path.dirname(_ld), "chat_history")
    except Exception:
        store_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "chat_history"
        )
    os.makedirs(store_dir, exist_ok=True)
    return store_dir


STORE_DIR = property(lambda self: _get_store_dir())  # noqa — 惰性求值


def _store_dir() -> str:
    return _get_store_dir()


SESSIONS_FILE = "_sessions.json"

# ================================================================
# 线程安全
# ================================================================

_lock = threading.Lock()


def _read_json(filepath: str, default=None):
    """读取 JSON 文件，不存在或损坏时返回 default。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else []
    except json.JSONDecodeError:
        logger.warning(f"JSON 损坏，重建: {filepath}")
        return default if default is not None else []


def _write_json(filepath: str, data):
    """原子写入 JSON 文件（先写临时文件再重命名）。"""
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, filepath)
    except Exception as e:
        logger.error(f"写入本地储存失败: {filepath}: {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass


# ================================================================
# 会话元数据
# ================================================================

def _sessions_path() -> str:
    return os.path.join(_store_dir(), SESSIONS_FILE)


def _load_sessions() -> list[dict]:
    return _read_json(_sessions_path(), default=[])


def _save_sessions(sessions: list[dict]):
    _write_json(_sessions_path(), sessions)


def create_local_session(session_id: str, title: str = "新对话") -> dict:
    """创建本地会话记录。"""
    with _lock:
        sessions = _load_sessions()
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        session = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
        sessions.insert(0, session)
        _save_sessions(sessions)
        logger.debug(f"本地会话已创建: {session_id} ({title})")
        return session


def get_all_local_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """获取本地会话列表（按 updated_at DESC）。"""
    sessions = _load_sessions()
    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions[offset:offset + limit]


def get_local_session_count() -> int:
    """获取本地会话总数。"""
    return len(_load_sessions())


def get_local_session(session_id: str) -> Optional[dict]:
    """获取单个本地会话元数据。"""
    for s in _load_sessions():
        if s["id"] == session_id:
            return s
    return None


def update_local_session_title(session_id: str, title: str) -> Optional[dict]:
    """更新本地会话标题。"""
    with _lock:
        sessions = _load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["title"] = title
                s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_sessions(sessions)
                return s
    return None


def delete_local_session(session_id: str) -> int:
    """
    删除本地会话及其所有消息。
    Returns: 删除的会话数 (0 或 1)
    """
    deleted = 0
    with _lock:
        sessions = _load_sessions()
        new_sessions = [s for s in sessions if s["id"] != session_id]
        if len(new_sessions) < len(sessions):
            deleted = 1
        _save_sessions(new_sessions)

    # 删除消息文件
    msg_path = os.path.join(_store_dir(), f"{session_id}.json")
    try:
        os.remove(msg_path)
        logger.debug(f"本地消息已删除: {session_id}")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"删除本地消息文件失败: {e}")

    return deleted


def increment_local_session_message_count(session_id: str):
    """增加本地会话消息计数 +1。"""
    with _lock:
        sessions = _load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["message_count"] = s.get("message_count", 0) + 1
                s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                break
        _save_sessions(sessions)


def decrement_local_session_message_count(session_id: str, count: int = 2):
    """减少本地会话消息计数。"""
    with _lock:
        sessions = _load_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["message_count"] = max(0, s.get("message_count", 0) - count)
                s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                break
        _save_sessions(sessions)


# ================================================================
# 对话消息
# ================================================================

def _messages_path(session_id: str) -> str:
    return os.path.join(_store_dir(), f"{session_id}.json")


def save_local_message(session_id: str, role: str, content: str,
                       metrics: dict = None):
    """向本地会话追加一条消息。"""
    with _lock:
        messages = _read_json(_messages_path(session_id), default=[])
        msg = {
            "role": role,
            "content": content,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if metrics:
            msg["metrics"] = metrics
        messages.append(msg)
        _write_json(_messages_path(session_id), messages)


def load_local_conversation(session_id: str, limit: int = 200) -> list[dict]:
    """从本地文件加载指定会话的对话历史。"""
    messages = _read_json(_messages_path(session_id), default=[])
    if limit and len(messages) > limit:
        messages = messages[-limit:]
    return messages


def get_local_conversation_count(session_id: str) -> int:
    """获取本地会话的消息数量。"""
    return len(_read_json(_messages_path(session_id), default=[]))


def clear_local_conversation(session_id: str) -> int:
    """
    清空本地会话的对话消息。
    Returns: 删除的消息数
    """
    count = 0
    with _lock:
        msg_path = _messages_path(session_id)
        try:
            messages = _read_json(msg_path, default=[])
            count = len(messages)
            _write_json(msg_path, [])
        except Exception:
            pass
    return count


def delete_local_message_range(session_id: str, turn_index: int) -> int:
    """
    删除本地会话中指定轮次的消息（user + assistant 两条）。
    Returns: 删除的消息数（0 或 2）
    """
    with _lock:
        msg_path = _messages_path(session_id)
        messages = _read_json(msg_path, default=[])
        idx = turn_index * 2
        if idx + 1 >= len(messages):
            return 0
        del messages[idx:idx + 2]
        _write_json(msg_path, messages)
        return 2


# ================================================================
# 工具
# ================================================================

def local_store_stats() -> dict:
    """获取本地储存统计信息。"""
    store = _store_dir()
    try:
        files = os.listdir(store)
        msg_files = [f for f in files if f.endswith(".json") and f != SESSIONS_FILE]
        total_messages = 0
        for mf in msg_files:
            msgs = _read_json(os.path.join(store, mf), default=[])
            total_messages += len(msgs)
        return {
            "store_dir": store,
            "session_count": len(_load_sessions()),
            "message_count": total_messages,
            "file_count": len(msg_files),
        }
    except Exception as e:
        return {"store_dir": store, "error": str(e)}
