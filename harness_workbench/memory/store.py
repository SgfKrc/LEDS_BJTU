"""SQLite-backed user-owned memory entries with explicit lifecycle states.

This slice deliberately stores facts only.  Retrieval, embeddings and context
budgeting are separate tickets; callers can already persist and manage entries
without giving the harness ownership of a session database or a filesystem.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


MemoryKind = Literal["fact", "preference", "decision"]
_MEMORY_KINDS = frozenset(("fact", "preference", "decision"))
_MAX_CONTENT = 32_000
_MAX_METADATA = 32_000


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A user-owned memory item, including its auditable lifecycle state."""

    entry_id: str
    owner_scope: str
    kind: MemoryKind
    content: str
    fingerprint: str
    source_session_id: str | None
    source_message_id: str | None
    created_at: float
    updated_at: float
    valid_until: float | None = None
    invalidated_at: float | None = None
    invalidated_reason: str | None = None
    deleted_at: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.deleted_at is not None:
            return "deleted"
        if self.invalidated_at is not None:
            return "invalidated"
        if self.valid_until is not None and self.valid_until <= time.time():
            return "expired"
        return "active"

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "owner_scope": self.owner_scope,
            "kind": self.kind,
            "content": self.content,
            "fingerprint": self.fingerprint,
            "source_session_id": self.source_session_id,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_until": self.valid_until,
            "invalidated_at": self.invalidated_at,
            "invalidated_reason": self.invalidated_reason,
            "deleted_at": self.deleted_at,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


class MemoryStore:
    """Persist memory entries in a user-selected SQLite database.

    ``path`` is the ownership boundary: the harness creates and manages only
    the file supplied by the user.  Every read and lifecycle mutation requires
    the same ``owner_scope`` used at creation time.
    """

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
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    owner_scope TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('fact', 'preference', 'decision')),
                    content TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    source_session_id TEXT,
                    source_message_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    valid_until REAL,
                    invalidated_at REAL,
                    invalidated_reason TEXT,
                    deleted_at REAL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_scope_state
                    ON memory_entries(owner_scope, deleted_at, invalidated_at, valid_until, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_scope_kind
                    ON memory_entries(owner_scope, kind, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_fingerprint
                    ON memory_entries(owner_scope, fingerprint);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_fts USING fts5(
                    entry_id UNINDEXED,
                    owner_scope UNINDEXED,
                    kind UNINDEXED,
                    content
                );
                """
            )
            connection.execute("DELETE FROM memory_entries_fts")
            connection.execute(
                """
                INSERT INTO memory_entries_fts(entry_id, owner_scope, kind, content)
                SELECT entry_id, owner_scope, kind, content FROM memory_entries
                """
            )

    def add(
        self,
        *,
        kind: MemoryKind,
        content: str,
        owner_scope: str = "local",
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        valid_until: float | None = None,
        deduplicate: bool = True,
    ) -> MemoryEntry:
        owner_scope = _scope(owner_scope)
        kind = _kind(kind)
        content = _content(content)
        source_session_id = _identifier(source_session_id, "source_session_id")
        source_message_id = _identifier(source_message_id, "source_message_id")
        valid_until = _valid_until(valid_until)
        metadata = _metadata(metadata)
        encoded_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded_metadata) > _MAX_METADATA:
            raise ValueError("memory metadata is too large")
        fingerprint = _fingerprint(kind, content)
        now = time.time()
        entry = MemoryEntry(
            entry_id=f"mem_{uuid.uuid4().hex}",
            owner_scope=owner_scope,
            kind=kind,
            content=content,
            fingerprint=fingerprint,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            created_at=now,
            updated_at=now,
            valid_until=valid_until,
            metadata=metadata,
        )
        with self._connect() as connection:
            if deduplicate:
                existing = connection.execute(
                    """
                    SELECT * FROM memory_entries
                    WHERE owner_scope = ? AND kind = ? AND fingerprint = ?
                      AND deleted_at IS NULL
                      AND invalidated_at IS NULL
                      AND (valid_until IS NULL OR valid_until > ?)
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (owner_scope, kind, fingerprint, now),
                ).fetchone()
                if existing is not None:
                    return _entry_from_row(existing)
            connection.execute(
                """
                INSERT INTO memory_entries(
                    entry_id, owner_scope, kind, content, fingerprint,
                    source_session_id, source_message_id, created_at, updated_at,
                    valid_until, invalidated_at, invalidated_reason, deleted_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    entry.entry_id,
                    entry.owner_scope,
                    entry.kind,
                    entry.content,
                    entry.fingerprint,
                    entry.source_session_id,
                    entry.source_message_id,
                    entry.created_at,
                    entry.updated_at,
                    entry.valid_until,
                    encoded_metadata,
                ),
            )
            connection.execute(
                "INSERT INTO memory_entries_fts(entry_id, owner_scope, kind, content) VALUES (?, ?, ?, ?)",
                (entry.entry_id, entry.owner_scope, entry.kind, entry.content),
            )
        return entry

    def search(self, query: str, *, owner_scope: str = "local", limit: int = 8):
        """Search active entries using FTS5 and return scored memory hits."""
        from .retrieve import MemoryHit

        owner_scope = _scope(owner_scope)
        if not isinstance(query, str) or not query.strip():
            return []
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory search limit is invalid") from exc
        if not 1 <= limit <= 50:
            raise ValueError("memory search limit must be between 1 and 50")
        match = _fts_query(query)
        if not match:
            return []
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.*, bm25(memory_entries_fts) AS score
                FROM memory_entries_fts f
                JOIN memory_entries e ON e.entry_id = f.entry_id
                WHERE memory_entries_fts MATCH ?
                  AND f.owner_scope = ?
                  AND e.owner_scope = ?
                  AND e.deleted_at IS NULL
                  AND e.invalidated_at IS NULL
                  AND (e.valid_until IS NULL OR e.valid_until > ?)
                ORDER BY score, e.updated_at DESC
                LIMIT ?
                """,
                (match, owner_scope, owner_scope, now, limit),
            ).fetchall()
        return [
            MemoryHit(
                entry_id=row["entry_id"],
                owner_scope=row["owner_scope"],
                kind=row["kind"],
                content=row["content"],
                fingerprint=row["fingerprint"],
                source_session_id=row["source_session_id"],
                source_message_id=row["source_message_id"],
                created_at=row["created_at"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    def get(
        self,
        entry_id: str,
        *,
        owner_scope: str = "local",
        include_deleted: bool = False,
        include_invalidated: bool = False,
    ) -> MemoryEntry:
        entry_id = _identifier(entry_id, "entry_id")
        owner_scope = _scope(owner_scope)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_entries WHERE entry_id = ? AND owner_scope = ?",
                (entry_id, owner_scope),
            ).fetchone()
        if row is None or not _visible(row, include_deleted=include_deleted, include_invalidated=include_invalidated):
            raise KeyError("memory entry not found")
        return _entry_from_row(row)

    def list(
        self,
        *,
        owner_scope: str = "local",
        kind: MemoryKind | None = None,
        limit: int = 50,
        include_deleted: bool = False,
        include_invalidated: bool = False,
    ) -> list[MemoryEntry]:
        owner_scope = _scope(owner_scope)
        kind = _kind(kind) if kind is not None else None
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory limit is invalid") from exc
        if not 1 <= limit <= 200:
            raise ValueError("memory limit must be between 1 and 200")
        clauses = ["owner_scope = ?"]
        params: list[Any] = [owner_scope]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if not include_invalidated:
            clauses.append("invalidated_at IS NULL")
            clauses.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(time.time())
        params.append(limit)
        query = (
            "SELECT * FROM memory_entries WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_entry_from_row(row) for row in rows]

    def invalidate(
        self,
        entry_id: str,
        *,
        owner_scope: str = "local",
        reason: str | None = None,
    ) -> MemoryEntry:
        """Mark an entry as no longer valid without deleting its audit record."""
        entry_id = _identifier(entry_id, "entry_id")
        owner_scope = _scope(owner_scope)
        reason = _reason(reason)
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET invalidated_at = COALESCE(invalidated_at, ?),
                    invalidated_reason = COALESCE(invalidated_reason, ?),
                    updated_at = ?
                WHERE entry_id = ? AND owner_scope = ? AND deleted_at IS NULL
                """,
                (now, reason, now, entry_id, owner_scope),
            )
            if cursor.rowcount == 0:
                raise KeyError("memory entry not found")
            row = connection.execute(
                "SELECT * FROM memory_entries WHERE entry_id = ? AND owner_scope = ?",
                (entry_id, owner_scope),
            ).fetchone()
        return _entry_from_row(row)

    def delete(
        self,
        entry_id: str,
        *,
        owner_scope: str = "local",
        confirm: bool = False,
        reason: str | None = None,
    ) -> MemoryEntry:
        """Soft-delete an entry; explicit confirmation is required."""
        if confirm is not True:
            raise ValueError("memory deletion requires confirm=True")
        entry_id = _identifier(entry_id, "entry_id")
        owner_scope = _scope(owner_scope)
        reason = _reason(reason)
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE memory_entries
                SET deleted_at = COALESCE(deleted_at, ?),
                    invalidated_reason = COALESCE(invalidated_reason, ?),
                    updated_at = ?
                WHERE entry_id = ? AND owner_scope = ?
                """,
                (now, reason, now, entry_id, owner_scope),
            )
            if cursor.rowcount == 0:
                raise KeyError("memory entry not found")
            row = connection.execute(
                "SELECT * FROM memory_entries WHERE entry_id = ? AND owner_scope = ?",
                (entry_id, owner_scope),
            ).fetchone()
        return _entry_from_row(row)

    def health(self, *, owner_scope: str | None = None) -> dict[str, Any]:
        scope = _scope(owner_scope) if owner_scope is not None else None
        with self._connect() as connection:
            where = "WHERE owner_scope = ?" if scope is not None else ""
            params = (scope,) if scope is not None else ()
            total = int(connection.execute(f"SELECT COUNT(*) FROM memory_entries {where}", params).fetchone()[0])
            active = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM memory_entries {where + (' AND' if where else 'WHERE')} deleted_at IS NULL AND invalidated_at IS NULL AND (valid_until IS NULL OR valid_until > ?)",
                    (*params, time.time()),
                ).fetchone()[0]
            )
        return {
            "backend": "sqlite_memory",
            "retrieval": "fts5",
            "entries": total,
            "active_entries": active,
            "owner_scope": scope,
        }


def _scope(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 128 or any(char in value for char in "\\/\r\n"):
        raise ValueError("owner_scope is invalid")
    return value


def _kind(value: str) -> MemoryKind:
    if value not in _MEMORY_KINDS:
        raise ValueError("memory kind must be fact, preference or decision")
    return value  # type: ignore[return-value]


def _content(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > _MAX_CONTENT:
        raise ValueError("memory content is empty or too large")
    return value


def _identifier(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or len(value) > 256 or any(char in value for char in "\\/\r\n"):
        raise ValueError(f"{field_name} is invalid")
    return value


def _valid_until(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("valid_until is invalid") from exc
    if value <= 0:
        raise ValueError("valid_until must be positive")
    return value


def _reason(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if len(value) > 512:
        raise ValueError("memory lifecycle reason is too large")
    return value or None


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("memory metadata must be an object")
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        parsed = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory metadata must be JSON serializable") from exc
    return parsed


def _fingerprint(kind: MemoryKind, content: str) -> str:
    return hashlib.sha256(f"{kind}\0{content}".encode("utf-8")).hexdigest()


def _fts_query(value: str) -> str:
    terms = re.findall(r"[\w\u3400-\u9fff]+", value, flags=re.UNICODE)
    return " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:16])


def _visible(row: sqlite3.Row, *, include_deleted: bool, include_invalidated: bool) -> bool:
    if row["deleted_at"] is not None and not include_deleted:
        return False
    if row["invalidated_at"] is not None and not include_invalidated:
        return False
    if not include_invalidated and row["valid_until"] is not None and row["valid_until"] <= time.time():
        return False
    return True


def _entry_from_row(row: sqlite3.Row) -> MemoryEntry:
    metadata = json.loads(row["metadata_json"])
    if not isinstance(metadata, Mapping):
        metadata = {}
    return MemoryEntry(
        entry_id=row["entry_id"],
        owner_scope=row["owner_scope"],
        kind=row["kind"],
        content=row["content"],
        fingerprint=row["fingerprint"],
        source_session_id=row["source_session_id"],
        source_message_id=row["source_message_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        valid_until=row["valid_until"],
        invalidated_at=row["invalidated_at"],
        invalidated_reason=row["invalidated_reason"],
        deleted_at=row["deleted_at"],
        metadata=metadata,
    )


__all__ = ["MemoryEntry", "MemoryKind", "MemoryStore"]
