"""Small SQLite session store with explicit ownership and asset references."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    owner_scope: str
    title: str
    created_at: float
    updated_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "owner_scope": self.owner_scope,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SessionMessage:
    message_id: str
    session_id: str
    ordinal: int
    role: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "ordinal": self.ordinal,
            "role": self.role,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SessionAssetRef:
    asset_id: str
    session_id: str
    ordinal: int
    kind: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "session_id": self.session_id,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }


class SessionStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_scope TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(session_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS session_assets (
                    asset_id TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(asset_id, session_id)
                );
                """
            )

    def create(self, *, owner_scope: str = "local", title: str = "New session") -> SessionRecord:
        owner_scope = _scope(owner_scope)
        title = _title(title)
        now = time.time()
        record = SessionRecord(f"sess_{uuid.uuid4().hex}", owner_scope, title, now, now)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, owner_scope, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (record.session_id, record.owner_scope, record.title, now, now),
            )
        return record

    def get(self, session_id: str, *, owner_scope: str | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None or (owner_scope is not None and row["owner_scope"] != _scope(owner_scope)):
                raise KeyError("session not found")
            messages = connection.execute("SELECT * FROM session_messages WHERE session_id = ? ORDER BY ordinal", (session_id,)).fetchall()
            assets = connection.execute("SELECT * FROM session_assets WHERE session_id = ? ORDER BY ordinal", (session_id,)).fetchall()
        return {
            "session": SessionRecord(row["session_id"], row["owner_scope"], row["title"], row["created_at"], row["updated_at"]).as_dict(),
            "messages": [_message_from_row(item).as_dict() for item in messages],
            "assets": [_asset_from_row(item).as_dict() for item in assets],
        }

    def append_message(self, session_id: str, *, role: str, content: str, metadata: Mapping[str, Any] | None = None) -> SessionMessage:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported session message role")
        if not isinstance(content, str) or not content.strip() or len(content) > 120_000:
            raise ValueError("session message content is empty or too large")
        metadata = dict(metadata or {})
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        if len(encoded) > 32_000:
            raise ValueError("session message metadata is too large")
        message_id = f"msg_{uuid.uuid4().hex}"
        with self._connect() as connection:
            session = connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if session is None:
                raise KeyError("session not found")
            ordinal = connection.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM session_messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            now = time.time()
            connection.execute(
                "INSERT INTO session_messages(message_id, session_id, ordinal, role, content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, session_id, ordinal, role, content, encoded),
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        return SessionMessage(message_id, session_id, ordinal, role, content, metadata)

    def attach_asset(self, session_id: str, *, asset_id: str, kind: str = "image", metadata: Mapping[str, Any] | None = None) -> SessionAssetRef:
        if not isinstance(asset_id, str) or not asset_id.strip() or len(asset_id) > 128:
            raise ValueError("asset_id is invalid")
        if not isinstance(kind, str) or not kind.strip() or len(kind) > 64:
            raise ValueError("asset kind is invalid")
        metadata = dict(metadata or {})
        encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
                raise KeyError("session not found")
            ordinal = connection.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM session_assets WHERE session_id = ?", (session_id,)).fetchone()[0]
            connection.execute(
                "INSERT OR REPLACE INTO session_assets(asset_id, session_id, ordinal, kind, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (asset_id, session_id, ordinal, kind, encoded),
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (time.time(), session_id))
        return SessionAssetRef(asset_id, session_id, ordinal, kind, metadata)


def _scope(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 128 or any(char in value for char in "\\/\r\n"):
        raise ValueError("owner_scope is invalid")
    return value


def _title(value: str) -> str:
    value = str(value or "New session").strip()
    if not value or len(value) > 200:
        raise ValueError("session title is invalid")
    return value


def _metadata(value: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    return parsed if isinstance(parsed, Mapping) else {}


def _message_from_row(row: sqlite3.Row) -> SessionMessage:
    return SessionMessage(row["message_id"], row["session_id"], row["ordinal"], row["role"], row["content"], _metadata(row["metadata_json"]))


def _asset_from_row(row: sqlite3.Row) -> SessionAssetRef:
    return SessionAssetRef(row["asset_id"], row["session_id"], row["ordinal"], row["kind"], _metadata(row["metadata_json"]))


__all__ = ["SessionAssetRef", "SessionMessage", "SessionRecord", "SessionStore"]
