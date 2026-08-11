"""Main-node local asset storage backed by SQLite.

The Python runtime shares the control-svc sessions, session_messages, and
cluster_settings tables.  This module never changes PRAGMA user_version;
schema ownership remains with control-svc.  Legacy chat_history JSON files are
imported once and retained as user-owned recovery material.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_init_lock = threading.Lock()
_initialized_paths: set[str] = set()
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LEGACY_MIGRATION_KEY = "legacy_chat_history_json_v1"
_MASTER_IDENTITY_KEY = "master_identity_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_store_dir() -> str:
    try:
        from config import STATE_DIR

        return os.path.abspath(STATE_DIR)
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "state",
        )


def _store_dir() -> str:
    return _get_store_dir()


def _legacy_store_dir() -> str:
    try:
        from config import LOG_DIR

        return os.path.join(os.path.dirname(LOG_DIR), "chat_history")
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "chat_history",
        )


def _sqlite_path() -> str:
    override = os.environ.get("QLH_SQLITE_PATH", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    state_path = os.path.join(_store_dir(), "qlh-control.sqlite3")
    legacy_path = os.path.abspath("qlh-control.sqlite3")
    if os.path.isfile(legacy_path) and not os.path.exists(state_path):
        return legacy_path
    return state_path


def _connect_raw(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _read_legacy_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("旧 JSON 无法导入，保留原文件: %s: %s", path, exc)
        return default


def _valid_session_id(value: object) -> str | None:
    session_id = str(value or "").strip()
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        return None
    return session_id


def _nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return default


def _import_legacy_json(connection: sqlite3.Connection) -> None:
    marker = connection.execute(
        "SELECT value FROM python_local_store_meta WHERE key = ?",
        (_LEGACY_MIGRATION_KEY,),
    ).fetchone()
    if marker:
        return

    legacy_dir = _legacy_store_dir()
    sessions_path = os.path.join(legacy_dir, "_sessions.json")
    sessions = _read_legacy_json(sessions_path, [])
    imported_sessions: set[str] = set()
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, dict):
                continue
            session_id = _valid_session_id(item.get("id"))
            if not session_id:
                continue
            created_at = str(item.get("created_at") or _now())
            updated_at = str(item.get("updated_at") or created_at)
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                  (session_id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(item.get("title") or "新对话")[:256],
                    created_at,
                    updated_at,
                    _nonnegative_int(item.get("message_count")),
                ),
            )
            imported_sessions.add(session_id)

    if os.path.isdir(legacy_dir):
        for name in os.listdir(legacy_dir):
            if name == "_sessions.json" or not name.endswith(".json"):
                continue
            session_id = _valid_session_id(name[:-5])
            if not session_id:
                continue
            messages = _read_legacy_json(os.path.join(legacy_dir, name), [])
            if not isinstance(messages, list):
                continue
            now = _now()
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions
                  (session_id, title, created_at, updated_at, message_count)
                VALUES (?, '新对话', ?, ?, 0)
                """,
                (session_id, now, now),
            )
            imported_sessions.add(session_id)
            existing = connection.execute(
                "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing and int(existing["count"]) > 0:
                continue
            imported_count = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip()
                content = message.get("content")
                if not role or not isinstance(content, str):
                    continue
                metrics = message.get("metrics")
                connection.execute(
                    """
                    INSERT INTO session_messages
                      (session_id, role, content, created_at, metrics)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        role,
                        content,
                        str(message.get("created_at") or now),
                        json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
                    ),
                )
                imported_count += 1
            if imported_count:
                connection.execute(
                    """
                    UPDATE sessions SET
                      message_count = MAX(message_count, ?), updated_at = ?
                    WHERE session_id = ?
                    """,
                    (imported_count, now, session_id),
                )

    connection.execute(
        """
        INSERT INTO python_local_store_meta(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (_LEGACY_MIGRATION_KEY, str(len(imported_sessions)), _now()),
    )


def initialize_local_store() -> str:
    path = _sqlite_path()
    if path in _initialized_paths and os.path.isfile(path):
        return path
    with _init_lock:
        if path in _initialized_paths and os.path.isfile(path):
            return path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        connection = _connect_raw(path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cluster_settings (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  message_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                  ON sessions(updated_at DESC);
                CREATE TABLE IF NOT EXISTS session_messages (
                  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at TEXT,
                  metrics TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_session_messages_session
                  ON session_messages(session_id, message_id);
                CREATE TABLE IF NOT EXISTS model_registry (
                  model_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL DEFAULT '',
                  model_path TEXT NOT NULL DEFAULT '',
                  gguf_path TEXT,
                  quantization TEXT,
                  sha256 TEXT,
                  payload TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_tickets (
                  ticket_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_tickets_created
                  ON review_tickets(created_at DESC);
                CREATE TABLE IF NOT EXISTS python_local_store_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )
            try:
                connection.execute("ALTER TABLE model_registry ADD COLUMN updated_at TEXT")
            except sqlite3.OperationalError:
                pass
            connection.execute("BEGIN IMMEDIATE")
            try:
                _import_legacy_json(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO python_local_store_meta(key, value, updated_at)
                VALUES ('__write_probe__', '1', ?)
                ON CONFLICT(key) DO UPDATE SET value='1', updated_at=excluded.updated_at
                """,
                (_now(),),
            )
            connection.rollback()
        finally:
            connection.close()
        _initialized_paths.add(path)
    return path


def _connect() -> sqlite3.Connection:
    return _connect_raw(initialize_local_store())


@contextmanager
def _write_connection() -> Iterator[sqlite3.Connection]:
    with _lock:
        connection = _connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _session_record(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {
        "id": row["session_id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": int(row["message_count"]),
    }


def create_local_session(session_id: str, title: str = "新对话") -> dict:
    session_id = _valid_session_id(session_id) or ""
    if not session_id:
        raise ValueError("session_id is invalid")
    now = _now()
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions(session_id, title, created_at, updated_at, message_count)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(session_id) DO UPDATE SET
              title=excluded.title, updated_at=excluded.updated_at
            """,
            (session_id, str(title or "新对话")[:256], now, now),
        )
    return get_local_session(session_id) or {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def get_all_local_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT session_id, title, created_at, updated_at, message_count
            FROM sessions ORDER BY updated_at DESC, session_id LIMIT ? OFFSET ?
            """,
            (max(0, int(limit)), max(0, int(offset))),
        ).fetchall()
        return [_session_record(row) for row in rows if row is not None]
    finally:
        connection.close()


def get_local_session_count() -> int:
    connection = _connect()
    try:
        row = connection.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()
        return int(row["count"] if row else 0)
    finally:
        connection.close()


def get_local_session(session_id: str) -> Optional[dict]:
    connection = _connect()
    try:
        row = connection.execute(
            """
            SELECT session_id, title, created_at, updated_at, message_count
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return _session_record(row)
    finally:
        connection.close()


def update_local_session_title(session_id: str, title: str) -> Optional[dict]:
    with _write_connection() as connection:
        result = connection.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (str(title)[:256], _now(), session_id),
        )
        if result.rowcount != 1:
            return None
    return get_local_session(session_id)


def delete_local_session(session_id: str) -> int:
    with _write_connection() as connection:
        connection.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        result = connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return int(result.rowcount)


def increment_local_session_message_count(session_id: str) -> None:
    with _write_connection() as connection:
        connection.execute(
            """
            UPDATE sessions SET message_count = message_count + 1, updated_at = ?
            WHERE session_id = ?
            """,
            (_now(), session_id),
        )


def decrement_local_session_message_count(session_id: str, count: int = 2) -> None:
    with _write_connection() as connection:
        connection.execute(
            """
            UPDATE sessions SET message_count = MAX(0, message_count - ?), updated_at = ?
            WHERE session_id = ?
            """,
            (max(0, int(count)), _now(), session_id),
        )


def save_local_message(
    session_id: str,
    role: str,
    content: str,
    metrics: dict | None = None,
) -> None:
    session_id = _valid_session_id(session_id) or ""
    if not session_id:
        raise ValueError("session_id is invalid")
    now = _now()
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO sessions
              (session_id, title, created_at, updated_at, message_count)
            VALUES (?, '新对话', ?, ?, 0)
            """,
            (session_id, now, now),
        )
        connection.execute(
            """
            INSERT INTO session_messages(session_id, role, content, created_at, metrics)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(role),
                str(content),
                now,
                json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
            ),
        )


def load_local_conversation(session_id: str, limit: int = 200) -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT role, content, created_at, metrics FROM session_messages
            WHERE message_id IN (
              SELECT message_id FROM session_messages WHERE session_id = ?
              ORDER BY message_id DESC LIMIT ?
            )
            ORDER BY message_id
            """,
            (session_id, max(0, int(limit))),
        ).fetchall()
        messages = []
        for row in rows:
            item = {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            if row["metrics"]:
                try:
                    item["metrics"] = json.loads(row["metrics"])
                except json.JSONDecodeError:
                    item["metrics"] = {}
            messages.append(item)
        return messages
    finally:
        connection.close()


def get_local_conversation_count(session_id: str) -> int:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["count"] if row else 0)
    finally:
        connection.close()


def clear_local_conversation(session_id: str) -> int:
    with _write_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM session_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = int(row["count"] if row else 0)
        connection.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        connection.execute(
            "UPDATE sessions SET message_count = 0, updated_at = ? WHERE session_id = ?",
            (_now(), session_id),
        )
        return count


def delete_local_message_range(session_id: str, turn_index: int) -> int:
    offset = max(0, int(turn_index)) * 2
    with _write_connection() as connection:
        rows = connection.execute(
            """
            SELECT message_id FROM session_messages WHERE session_id = ?
            ORDER BY message_id LIMIT 2 OFFSET ?
            """,
            (session_id, offset),
        ).fetchall()
        ids = [int(row["message_id"]) for row in rows]
        if len(ids) != 2:
            return 0
        connection.execute(
            "DELETE FROM session_messages WHERE message_id IN (?, ?)",
            (ids[0], ids[1]),
        )
        return 2


def get_local_user_settings() -> dict:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = 'user_settings'"
        ).fetchone()
        if not row or not row["value"]:
            return {}
        value = json.loads(row["value"])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}
    finally:
        connection.close()


def set_local_user_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    now = _now()
    with _write_connection() as connection:
        values = {
            "user_settings": json.dumps(settings, ensure_ascii=False, separators=(",", ":")),
        }
        if "saveHistory" in settings:
            values["save_history"] = "true" if bool(settings["saveHistory"]) else "false"
        if "distributedInference" in settings:
            values["distributed_inference_enabled"] = (
                "true" if bool(settings["distributedInference"]) else "false"
            )
        for key, value in values.items():
            connection.execute(
                """
                INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )


def get_local_setting(key: str, default=None):
    """Read a user-owned cluster setting, decoding JSON when possible."""
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = ?", (str(key),)
        ).fetchone()
        if not row:
            return default
        raw = row["value"]
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return raw
    finally:
        connection.close()


def set_local_setting(key: str, value) -> None:
    """Write a user-owned cluster setting as JSON."""
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (str(key), json.dumps(value, ensure_ascii=False), _now()),
        )


def get_local_master_identity() -> dict:
    """Read the main-node MAC identity from the user-owned SQLite store."""
    value = get_local_setting(_MASTER_IDENTITY_KEY, {})
    if not isinstance(value, dict):
        return {}
    macs = value.get("mac_addresses")
    if not isinstance(macs, list):
        return {}
    normalized = sorted({str(mac).strip().lower() for mac in macs if str(mac).strip()})
    if not normalized:
        return {}
    return {
        "version": int(value.get("version") or 1),
        "mac_addresses": normalized,
        "bound_at": str(value.get("bound_at") or ""),
    }


def set_local_master_identity(mac_addresses: list[str]) -> dict:
    """Persist the physical MAC set that owns this main-node SQLite store."""
    normalized = sorted({str(mac).strip().lower() for mac in mac_addresses if str(mac).strip()})
    if not normalized:
        raise ValueError("无法记录空的主节点 MAC 地址集合")
    value = {
        "version": 1,
        "mac_addresses": normalized,
        "bound_at": _now(),
    }
    set_local_setting(_MASTER_IDENTITY_KEY, value)
    return value


def save_local_experimental_model(model_id: str, config_json: str) -> bool:
    """Persist a user-registered model configuration in the main-node SQLite."""
    config = json.loads(config_json) if isinstance(config_json, str) else dict(config_json)
    now = _now()
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_registry
              (model_id, name, model_path, gguf_path, quantization, sha256, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
              name=excluded.name, model_path=excluded.model_path, gguf_path=excluded.gguf_path,
              quantization=excluded.quantization, sha256=excluded.sha256,
              payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (
                str(model_id), str(config.get("name", "")), str(config.get("model_path", "")),
                str(config.get("gguf_path", "")), str(config.get("quantization", "")),
                str(config.get("sha256", "")), json.dumps(config, ensure_ascii=False), now, now,
            ),
        )
    return True


def get_local_experimental_models() -> list[dict]:
    connection = _connect()
    try:
        rows = connection.execute("SELECT payload FROM model_registry ORDER BY model_id").fetchall()
        result = []
        for row in rows:
            try:
                value = json.loads(row["payload"])
                if isinstance(value, dict):
                    result.append(value)
            except (TypeError, json.JSONDecodeError):
                continue
        return result
    finally:
        connection.close()


def delete_local_experimental_model(model_id: str) -> bool:
    with _write_connection() as connection:
        result = connection.execute("DELETE FROM model_registry WHERE model_id = ?", (str(model_id),))
        return int(result.rowcount) > 0


def upsert_local_review_ticket(ticket: dict) -> dict:
    payload = dict(ticket)
    payload["ticket_id"] = str(payload.get("ticket_id", ""))
    payload["status"] = str(payload.get("status", "pending"))
    payload["created_at"] = float(payload.get("created_at", 0.0) or 0.0)
    with _write_connection() as connection:
        connection.execute(
            """
            INSERT INTO review_tickets(ticket_id, status, created_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
              status=excluded.status, created_at=excluded.created_at, payload=excluded.payload
            """,
            (payload["ticket_id"], payload["status"], payload["created_at"],
             json.dumps(payload, ensure_ascii=False)),
        )
    return payload


def get_local_review_ticket(ticket_id: str) -> Optional[dict]:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT payload FROM review_tickets WHERE ticket_id = ?", (str(ticket_id),)
        ).fetchone()
        if not row:
            return None
        value = json.loads(row["payload"])
        return value if isinstance(value, dict) else None
    except (TypeError, json.JSONDecodeError):
        return None
    finally:
        connection.close()


def update_local_review_ticket(ticket_id: str, updates: dict) -> Optional[dict]:
    current = get_local_review_ticket(ticket_id)
    if current is None:
        return None
    current.update({key: value for key, value in updates.items() if key in {
        "status", "score", "votes", "resolved_at", "notification_sent",
        "transfer_reason", "expires_at",
    }})
    return upsert_local_review_ticket(current)


def list_local_review_tickets(status: str | None = None) -> list[dict]:
    connection = _connect()
    try:
        if status:
            rows = connection.execute(
                "SELECT payload FROM review_tickets WHERE status = ? ORDER BY created_at DESC",
                (str(status),),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT payload FROM review_tickets ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            try:
                value = json.loads(row["payload"])
                if isinstance(value, dict):
                    result.append(value)
            except (TypeError, json.JSONDecodeError):
                continue
        return result
    finally:
        connection.close()


def delete_local_review_ticket(ticket_id: str) -> bool:
    with _write_connection() as connection:
        result = connection.execute("DELETE FROM review_tickets WHERE ticket_id = ?", (str(ticket_id),))
        return int(result.rowcount) > 0


def delete_local_resolved_review_tickets() -> int:
    with _write_connection() as connection:
        result = connection.execute(
            "DELETE FROM review_tickets WHERE status IN ('approved', 'rejected', 'expired')"
        )
        return int(result.rowcount)
        for key, value in values.items():
            connection.execute(
                """
                INSERT INTO cluster_settings(key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )


def get_local_save_history() -> bool:
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT value FROM cluster_settings WHERE key = 'save_history'"
        ).fetchone()
        if not row or not row["value"]:
            settings_row = connection.execute(
                "SELECT value FROM cluster_settings WHERE key = 'user_settings'"
            ).fetchone()
            try:
                settings = json.loads(settings_row["value"]) if settings_row else {}
            except (json.JSONDecodeError, TypeError):
                settings = {}
            if not isinstance(settings, dict):
                settings = {}
            return bool(settings.get("saveHistory", True))
        return str(row["value"]).strip().lower() in {"1", "true", "yes", "on"}
    finally:
        connection.close()


def local_store_health() -> dict:
    path = initialize_local_store()
    connection = _connect_raw(path)
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO python_local_store_meta(key, value, updated_at)
            VALUES ('__health_probe__', '1', ?)
            ON CONFLICT(key) DO UPDATE SET value='1', updated_at=excluded.updated_at
            """,
            (_now(),),
        )
        connection.rollback()
        return {
            "status": "ok" if quick and str(quick[0]).lower() == "ok" else "unavailable",
            "backend": "sqlite",
            "writable": True,
            "path": path,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "sqlite",
            "writable": False,
            "path": path,
            "error": str(exc),
        }
    finally:
        connection.close()


def local_store_stats() -> dict:
    path = initialize_local_store()
    connection = _connect_raw(path)
    try:
        sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = connection.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0]
        return {
            "backend": "sqlite",
            "sqlite_path": path,
            "store_dir": os.path.dirname(path),
            "session_count": int(sessions),
            "message_count": int(messages),
            "file_count": 1,
        }
    except Exception as exc:
        return {
            "backend": "sqlite",
            "sqlite_path": path,
            "store_dir": os.path.dirname(path),
            "error": str(exc),
        }
    finally:
        connection.close()
