"""SQLite WAL + FTS5 retrieval store for user-owned harness data."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .chunking import chunk_text


RAG_METADATA_FIELDS = frozenset({"source", "scope", "type", "tag", "time"})
_MAX_METADATA_FILTERS = 5
_MAX_METADATA_VALUES = 20
_MAX_METADATA_TEXT = 256


@dataclass(frozen=True, slots=True)
class RagHit:
    source_id: str
    chunk_id: str
    title: str
    ordinal: int
    text: str
    score: float
    granularity: str = "fixed"
    routes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "ordinal": self.ordinal,
            "text": self.text,
            "score": self.score,
        }
        if self.granularity != "fixed":
            value["granularity"] = self.granularity
        if self.routes:
            value["routes"] = list(self.routes)
        return value


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
                    metadata_json TEXT NOT NULL DEFAULT '{}',
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
                    granularity TEXT NOT NULL DEFAULT 'fixed',
                    UNIQUE(source_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS rag_metadata_index (
                    source_id TEXT NOT NULL REFERENCES rag_sources(source_id) ON DELETE CASCADE,
                    owner_scope TEXT NOT NULL,
                    field TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(source_id, owner_scope, field, value)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_metadata_filter
                    ON rag_metadata_index(owner_scope, field, value, source_id);
                CREATE TABLE IF NOT EXISTS rag_query_cache (
                    cache_key TEXT PRIMARY KEY,
                    owner_scope TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
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
            columns = {row[1] for row in connection.execute("PRAGMA table_info(rag_chunks)")}
            if "granularity" not in columns:
                connection.execute("ALTER TABLE rag_chunks ADD COLUMN granularity TEXT NOT NULL DEFAULT 'fixed'")
            source_columns = {row[1] for row in connection.execute("PRAGMA table_info(rag_sources)")}
            if "metadata_json" not in source_columns:
                connection.execute("ALTER TABLE rag_sources ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def add_document(
        self,
        *,
        source_ref: str,
        title: str,
        text: str,
        owner_scope: str = "local",
        max_chars: int = 1200,
        overlap_chars: int = 120,
        strategy: str = "fixed",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_scope = _scope(owner_scope)
        source_ref = _relative_ref(source_ref)
        title = _title(title)
        metadata_value = _normalize_metadata(metadata)
        metadata_json = json.dumps(metadata_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        metadata_entries = _metadata_entries(metadata_value)
        chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars, strategy=strategy)
        if not chunks:
            raise ValueError("document text is empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = f"src_{hashlib.sha256((owner_scope + "\0" + source_ref + "\0" + digest).encode()).hexdigest()[:24]}"
        now = time.time()
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM rag_chunks_fts WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute("DELETE FROM rag_chunks WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute("DELETE FROM rag_metadata_index WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute(
                "INSERT INTO rag_sources(source_id, owner_scope, source_ref, title, metadata_json, content_sha256, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source_id) DO UPDATE SET title=excluded.title, metadata_json=excluded.metadata_json, content_sha256=excluded.content_sha256, updated_at=excluded.updated_at",
                (source_id, owner_scope, source_ref, title, metadata_json, digest, now, now),
            )
            for field, value in metadata_entries:
                connection.execute(
                    "INSERT INTO rag_metadata_index(source_id, owner_scope, field, value) VALUES (?, ?, ?, ?)",
                    (source_id, owner_scope, field, value),
                )
            for chunk in chunks:
                chunk_id = f"chk_{hashlib.sha256((source_id + "\0" + str(chunk.ordinal) + "\0" + chunk.text).encode()).hexdigest()[:24]}"
                connection.execute(
                    "INSERT INTO rag_chunks(chunk_id, source_id, owner_scope, ordinal, start_offset, end_offset, text, granularity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, chunk.ordinal, chunk.start_offset, chunk.end_offset, chunk.text, chunk.granularity),
                )
                connection.execute(
                    "INSERT INTO rag_chunks_fts(chunk_id, source_id, owner_scope, title, text) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, title, chunk.text),
                )
        return {"source_id": source_id, "owner_scope": owner_scope, "title": title, "metadata": metadata_value, "chunk_count": len(chunks), "content_sha256": digest}

    def search(
        self,
        query: str,
        *,
        owner_scope: str = "local",
        limit: int = 8,
        source_ids: tuple[str, ...] | list[str] = (),
        title_prefix: str | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[RagHit]:
        owner_scope = _scope(owner_scope)
        normalized_filters = _normalize_metadata_filters(metadata_filters)
        if not isinstance(query, str) or not query.strip():
            return []
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        match = _fts_query(query)
        if not match:
            return []
        source_ids = tuple(str(item) for item in source_ids)
        if len(source_ids) > 50:
            raise ValueError("source_ids must contain at most 50 values")
        if title_prefix is not None and (not isinstance(title_prefix, str) or len(title_prefix) > 200):
            raise ValueError("title_prefix is invalid")
        with self._connect() as connection:
            clauses = ["rag_chunks_fts MATCH ?", "f.owner_scope = ?"]
            params: list[Any] = [match, owner_scope]
            if source_ids:
                clauses.append("f.source_id IN (" + ",".join("?" for _ in source_ids) + ")")
                params.extend(source_ids)
            if title_prefix:
                clauses.append("s.title LIKE ? ESCAPE '\\'")
                params.append(_like_prefix(title_prefix))
            _append_metadata_clauses(clauses, params, normalized_filters, source_alias="f")
            params.append(limit)
            rows = connection.execute(
                "SELECT f.chunk_id, f.source_id, s.title, c.ordinal, c.text, c.granularity, bm25(rag_chunks_fts) AS score FROM rag_chunks_fts f JOIN rag_chunks c ON c.chunk_id = f.chunk_id JOIN rag_sources s ON s.source_id = f.source_id WHERE " + " AND ".join(clauses) + " ORDER BY score, c.ordinal LIMIT ?",
                params,
            ).fetchall()
        return [RagHit(row["source_id"], row["chunk_id"], row["title"], row["ordinal"], row["text"], float(row["score"]), row["granularity"] or "fixed") for row in rows]

    def list_chunks(
        self,
        *,
        owner_scope: str = "local",
        limit: int = 1000,
        source_ids: tuple[str, ...] | list[str] = (),
        title_prefix: str | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[RagHit]:
        owner_scope = _scope(owner_scope)
        normalized_filters = _normalize_metadata_filters(metadata_filters)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        source_ids = tuple(str(item) for item in source_ids)
        if len(source_ids) > 50:
            raise ValueError("source_ids must contain at most 50 values")
        if title_prefix is not None and (not isinstance(title_prefix, str) or len(title_prefix) > 200):
            raise ValueError("title_prefix is invalid")
        clauses = ["c.owner_scope = ?"]
        params: list[Any] = [owner_scope]
        if source_ids:
            clauses.append("c.source_id IN (" + ",".join("?" for _ in source_ids) + ")")
            params.extend(source_ids)
        if title_prefix:
            clauses.append("s.title LIKE ? ESCAPE '\\'")
            params.append(_like_prefix(title_prefix))
        _append_metadata_clauses(clauses, params, normalized_filters, source_alias="c")
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.source_id, c.chunk_id, s.title, c.ordinal, c.text, c.granularity FROM rag_chunks c JOIN rag_sources s ON s.source_id = c.source_id WHERE " + " AND ".join(clauses) + " ORDER BY s.updated_at DESC, c.ordinal LIMIT ?",
                params,
            ).fetchall()
        return [RagHit(row["source_id"], row["chunk_id"], row["title"], row["ordinal"], row["text"], 0.0, row["granularity"] or "fixed") for row in rows]

    def cache_get(self, cache_key: str, *, owner_scope: str = "local") -> dict[str, Any] | None:
        owner_scope = _scope(owner_scope)
        cache_key = _cache_key(cache_key)
        snapshot = self._snapshot_digest(owner_scope)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_digest, payload_json FROM rag_query_cache WHERE cache_key = ? AND owner_scope = ?",
                (cache_key, owner_scope),
            ).fetchone()
            if row is None:
                return None
            if row["snapshot_digest"] != snapshot:
                connection.execute("DELETE FROM rag_query_cache WHERE cache_key = ?", (cache_key,))
                return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def cache_put(self, cache_key: str, payload: dict[str, Any], *, owner_scope: str = "local") -> None:
        owner_scope = _scope(owner_scope)
        cache_key = _cache_key(cache_key)
        if not isinstance(payload, dict):
            raise ValueError("cache payload must be an object")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > 2_000_000:
            raise ValueError("cache payload is too large")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO rag_query_cache(cache_key, owner_scope, snapshot_digest, payload_json, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET owner_scope=excluded.owner_scope, snapshot_digest=excluded.snapshot_digest, payload_json=excluded.payload_json, created_at=excluded.created_at",
                (cache_key, owner_scope, self._snapshot_digest(owner_scope, connection), encoded, time.time()),
            )

    def delete_source(self, source_id: str, *, owner_scope: str = "local") -> bool:
        owner_scope = _scope(owner_scope)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("DELETE FROM rag_chunks_fts WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
            connection.execute("DELETE FROM rag_metadata_index WHERE source_id = ? AND owner_scope = ?", (source_id, owner_scope))
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

    def _snapshot_digest(self, owner_scope: str, connection: sqlite3.Connection | None = None) -> str:
        owner_scope = _scope(owner_scope)
        owns_connection = connection is None
        if connection is None:
            connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT source_id, content_sha256, metadata_json, updated_at FROM rag_sources WHERE owner_scope = ? ORDER BY source_id",
                (owner_scope,),
            ).fetchall()
            material = "|".join(f"{row['source_id']}:{row['content_sha256']}:{row['metadata_json']}:{row['updated_at']:.6f}" for row in rows)
            return hashlib.sha256(material.encode("utf-8")).hexdigest()
        finally:
            if owns_connection:
                connection.close()


def _normalize_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object")
    if any(not isinstance(key, str) or not key.strip() or len(key) > 64 for key in value):
        raise ValueError("metadata keys are invalid")
    try:
        normalized = json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON serializable") from exc
    if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > 16 * 1024:
        raise ValueError("metadata exceeds the local limit")
    return normalized


def _metadata_scalar(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip()
        if not value or len(value) > _MAX_METADATA_TEXT:
            raise ValueError("metadata filter value is invalid")
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    raise ValueError("metadata filter value must be a scalar")


def _metadata_entries(metadata: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for field in sorted(RAG_METADATA_FIELDS):
        if field not in metadata:
            continue
        raw = metadata[field]
        values = raw if field == "tag" and isinstance(raw, (list, tuple)) else (raw,)
        for value in values:
            try:
                entries.append((field, _metadata_scalar(value)))
            except ValueError:
                continue
    return tuple(dict.fromkeys(entries))


def _normalize_metadata_filters(value: Mapping[str, Any] | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping) or len(value) > _MAX_METADATA_FILTERS:
        raise ValueError("metadata_filters must contain at most five fields")
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw_field, raw_value in value.items():
        field = str(raw_field).strip()
        if field not in RAG_METADATA_FIELDS:
            raise ValueError(f"metadata filter field is unsupported: {field}")
        values = raw_value if field == "tag" and isinstance(raw_value, (list, tuple)) else (raw_value,)
        if not values or len(values) > _MAX_METADATA_VALUES:
            raise ValueError("metadata filter has too many values")
        scalars = tuple(dict.fromkeys(_metadata_scalar(item) for item in values))
        normalized.append((field, scalars))
    return tuple(sorted(normalized))


def _append_metadata_clauses(
    clauses: list[str], params: list[Any], filters: tuple[tuple[str, tuple[str, ...]], ...], *, source_alias: str
) -> None:
    for field, values in filters:
        placeholders = ",".join("?" for _ in values)
        clauses.append(
            "EXISTS (SELECT 1 FROM rag_metadata_index mi "
            f"WHERE mi.source_id = {source_alias}.source_id AND mi.owner_scope = {source_alias}.owner_scope "
            f"AND mi.field = ? AND mi.value IN ({placeholders}))"
        )
        params.extend((field, *values))


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


def _cache_key(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 256 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("cache_key is invalid")
    return value


def _like_prefix(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


__all__ = ["RAG_METADATA_FIELDS", "RagHit", "RagStore"]
