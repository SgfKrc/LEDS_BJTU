"""SQLite WAL + FTS5 retrieval store for user-owned harness data."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk_text


@dataclass(frozen=True, slots=True)
class RagHit:
    source_id: str
    chunk_id: str
    title: str
    ordinal: int
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "ordinal": self.ordinal,
            "text": self.text,
            "score": self.score,
        }


class RagStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_sources (
                    source_id TEXT PRIMARY KEY,
                    owner_scope TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES rag_sources(source_id) ON DELETE CASCADE,
                    owner_scope TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(source_id, ordinal)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    source_id UNINDEXED,
                    owner_scope UNINDEXED,
                    title,
                    text
                );
                """
            )

    def add_document(
        self,
        *,
        source_ref: str,
        title: str,
        text: str,
        owner_scope: str = "local",
        max_chars: int = 1200,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        owner_scope = _scope(owner_scope)
        source_ref = _relative_ref(source_ref)
        title = _title(title)
        chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            raise ValueError("document text is empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = f"src_{hashlib.sha256((owner_scope + "\0" + source_ref + "\0" + digest).encode()).hexdigest()[:24]}"
        now = time.time()
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM rag_chunks_fts WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute("DELETE FROM rag_chunks WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute(
                "INSERT INTO rag_sources(source_id, owner_scope, source_ref, title, content_sha256, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source_id) DO UPDATE SET title=excluded.title, content_sha256=excluded.content_sha256, updated_at=excluded.updated_at",
                (source_id, owner_scope, source_ref, title, digest, now, now),
            )
            for chunk in chunks:
                chunk_id = f"chk_{hashlib.sha256((source_id + "\0" + str(chunk.ordinal) + "\0" + chunk.text).encode()).hexdigest()[:24]}"
                connection.execute(
                    "INSERT INTO rag_chunks(chunk_id, source_id, owner_scope, ordinal, start_offset, end_offset, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, chunk.ordinal, chunk.start_offset, chunk.end_offset, chunk.text),
                )
                connection.execute(
                    "INSERT INTO rag_chunks_fts(chunk_id, source_id, owner_scope, title, text) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, title, chunk.text),
                )
        return {"source_id": source_id, "owner_scope": owner_scope, "title": title, "chunk_count": len(chunks), "content_sha256": digest}

    def search(self, query: str, *, owner_scope: str = "local", limit: int = 8) -> list[RagHit]:
        owner_scope = _scope(owner_scope)
        if not isinstance(query, str) or not query.strip():
            return []
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        match = _fts_query(query)
        if not match:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT f.chunk_id, f.source_id, s.title, c.ordinal, c.text, bm25(rag_chunks_fts) AS score FROM rag_chunks_fts f JOIN rag_chunks c ON c.chunk_id = f.chunk_id JOIN rag_sources s ON s.source_id = f.source_id WHERE rag_chunks_fts MATCH ? AND f.owner_scope = ? ORDER BY score, c.ordinal LIMIT ?",
                (match, owner_scope, limit),
            ).fetchall()
        return [RagHit(row["source_id"], row["chunk_id"], row["title"], row["ordinal"], row["text"], float(row["score"])) for row in rows]

    def delete_source(self, source_id: str, *, owner_scope: str = "local") -> bool:
        owner_scope = _scope(owner_scope)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM rag_chunks_fts WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            cursor = connection.execute("DELETE FROM rag_sources WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
        return cursor.rowcount > 0

    def rebuild_fts(self) -> int:
        with self._connect() as connection:
            connection.execute("DELETE FROM rag_chunks_fts")
            connection.execute(
                "INSERT INTO rag_chunks_fts(chunk_id, source_id, owner_scope, title, text) SELECT c.chunk_id, c.source_id, c.owner_scope, s.title, c.text FROM rag_chunks c JOIN rag_sources s ON s.source_id = c.source_id"
            )
            return int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            sources = int(connection.execute("SELECT COUNT(*) FROM rag_sources").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
        return {"backend": "sqlite_fts5", "sources": sources, "chunks": chunks, "embeddings": "optional_not_configured"}


def _scope(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 128 or any(char in value for char in "\\/\r\n"):
        raise ValueError("owner_scope is invalid")
    return value


def _relative_ref(value: str) -> str:
    value = str(value or "").strip().replace("\\", "/")
    if not value or value.startswith("/") or ":" in value.split("/", 1)[0] or ".." in value.split("/"):
        raise ValueError("source_ref must be a relative, path-free reference")
    return value[:512]


def _title(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 200:
        raise ValueError("source title is invalid")
    return value


def _fts_query(value: str) -> str:
    terms = re.findall(r"[\w\u3400-\u9fff]+", value, flags=re.UNICODE)
    return " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:16])


__all__ = ["RagHit", "RagStore"]
