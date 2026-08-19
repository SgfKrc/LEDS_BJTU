"""User-owned local RAG storage boundary for RAG-S0/RAG-S1.

This module intentionally has no HTTP/API integration and no embedding
dependency.  It owns a separate SQLite file, rejects secret-like material at
ingest, and provides atomic revision, deletion, and index-reset operations.
RAG-S1 adds a local FTS5 index over the chunk table without changing these
boundaries. No embedding provider or network/API integration is included.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


RAG_SCHEMA = "qlh.rag_store.v2"
_SCHEMA_VERSION = "2"
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIME = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_OWNER_SCOPES = frozenset({"local_user", "local_system", "project"})
_ACCESS_SCOPES = frozenset({"owner", "local_system", "project"})
_EMBEDDING_PROVIDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_EMBEDDING_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_ALLOWED_MIME_PREFIXES = ("text/",)
_ALLOWED_MIMES = frozenset({
    "application/json", "application/xml", "application/yaml", "application/x-yaml",
})
_SENSITIVE_REF_PARTS = frozenset({
    ".env", ".ssh", "authorized_keys", "credentials", "credential", "password",
    "passwd", "secret", "secrets", "token", "tokens", "recovery", "totp",
})
_SENSITIVE_CONTENT = re.compile(
    r"(?im)^\s*(?:password|passwd|token|access[_-]?token|refresh[_-]?token|"
    r"secret|api[_-]?key|private[_-]?key|client[_-]?secret|totp[_-]?secret|"
    r"recovery[_-]?codes?)\s*[:=]"
)
_PRIVATE_KEY_MARKER = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_MAX_EMBEDDING_BATCH = 64
_DEFAULT_VECTOR_SCAN_LIMIT = 1_024
_MAX_INDEX_CHUNKS = 10_000
_MAX_CJK_QUERY_TERMS = 24
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_JOB_STATES = frozenset({"queued", "running", "paused", "completed", "failed", "cancelled"})
_MIN_JOB_LEASE_SECONDS = 5
_MAX_JOB_LEASE_SECONDS = 3_600


class RagStoreError(ValueError):
    """Stable local RAG boundary error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IngestResult:
    source_id: str
    document_id: str
    revision: str
    text_digest: str
    chunk_count: int
    status: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_id(value: object, pattern: re.Pattern[str], *, code: str, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RagStoreError(code, f"{label} is invalid")
    return value


def _sha(value: object, *, code: str = "source_digest_invalid") -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RagStoreError(code, "sha256 must be a lowercase 64-character digest")
    return value


def _validate_ref(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagStoreError("source_ref_invalid", "source reference is required")
    ref = value.strip().replace("\\", "/")
    path = Path(ref)
    if ref.startswith("/") or re.match(r"^[A-Za-z]:", ref) or "//" in ref:
        raise RagStoreError("source_ref_invalid", "source reference must be relative")
    parts = [part for part in ref.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise RagStoreError("source_ref_invalid", "source reference contains traversal")
    lowered = {part.lower() for part in parts}
    if lowered & _SENSITIVE_REF_PARTS or any(
        part.lower().endswith((".key", ".pem", ".p12", ".pfx")) for part in parts
    ):
        raise RagStoreError("sensitive_source_rejected", "source reference is sensitive material")
    return "/".join(parts)


def _validate_content(text: object, *, max_bytes: int) -> str:
    if not isinstance(text, str) or not text:
        raise RagStoreError("document_empty", "document text is required")
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise RagStoreError("document_too_large", "document exceeds the local ingest limit")
    if "\x00" in text:
        raise RagStoreError("document_invalid", "document contains a NUL byte")
    if _PRIVATE_KEY_MARKER.search(text) or _SENSITIVE_CONTENT.search(text):
        raise RagStoreError("sensitive_content_rejected", "document contains secret-like material")
    return text


def _validate_mime(value: object) -> str:
    if not isinstance(value, str) or not _MIME.fullmatch(value.lower()):
        raise RagStoreError("mime_invalid", "document MIME type is invalid")
    mime = value.lower()
    if not mime.startswith(_ALLOWED_MIME_PREFIXES) and mime not in _ALLOWED_MIMES:
        raise RagStoreError("mime_rejected", "binary or non-text assets are not indexed")
    return mime


def _validate_scope(value: object, allowed: frozenset[str], *, code: str, label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise RagStoreError(code, f"{label} is outside the local RAG boundary")
    return value


class RagStore:
    """Independent SQLite store whose rows never leave the user's machine."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_document_bytes: int = 4 * 1024 * 1024,
        max_chunk_chars: int = 16_384,
        chunk_overlap_chars: int | None = None,
    ):
        if isinstance(max_document_bytes, bool) or not 1 <= int(max_document_bytes) <= 64 * 1024 * 1024:
            raise RagStoreError("limit_invalid", "max_document_bytes is outside the allowed range")
        if isinstance(max_chunk_chars, bool) or not 256 <= int(max_chunk_chars) <= 256 * 1024:
            raise RagStoreError("limit_invalid", "max_chunk_chars is outside the allowed range")
        normalized_max_chunk_chars = int(max_chunk_chars)
        if chunk_overlap_chars is None:
            chunk_overlap_chars = min(1_024, normalized_max_chunk_chars // 8)
        if (
            isinstance(chunk_overlap_chars, bool)
            or not 0 <= int(chunk_overlap_chars) < normalized_max_chunk_chars
        ):
            raise RagStoreError("limit_invalid", "chunk_overlap_chars is outside the allowed range")
        self.path = Path(path).expanduser().resolve()
        self.max_document_bytes = int(max_document_bytes)
        self.max_chunk_chars = normalized_max_chunk_chars
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self._lock = threading.RLock()
        self._embedding_job_lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            connection.close()
            raise RagStoreError("sqlite_wal_required", "RAG SQLite requires WAL mode")
        return connection

    def initialize(self) -> Path:
        if self._initialized and self.path.exists():
            return self.path
        with self._lock:
            if self._initialized and self.path.exists():
                return self.path
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS rag_meta (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS rag_sources (
                      source_id TEXT PRIMARY KEY,
                      owner_scope TEXT NOT NULL,
                      relative_ref TEXT NOT NULL,
                      sha256 TEXT NOT NULL,
                      mime TEXT NOT NULL,
                      title TEXT NOT NULL,
                      status TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS rag_documents (
                      document_id TEXT PRIMARY KEY,
                      source_id TEXT NOT NULL REFERENCES rag_sources(source_id) ON DELETE CASCADE,
                      revision TEXT NOT NULL,
                      text_digest TEXT NOT NULL,
                      language TEXT NOT NULL,
                      access_scope TEXT NOT NULL,
                      status TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL,
                      UNIQUE(source_id, revision)
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_documents_source_status
                      ON rag_documents(source_id, status, updated_at DESC);
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                      chunk_id TEXT PRIMARY KEY,
                      document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                      ordinal INTEGER NOT NULL,
                      text_digest TEXT NOT NULL,
                      token_count INTEGER NOT NULL,
                      start_offset INTEGER NOT NULL,
                      end_offset INTEGER NOT NULL,
                      text_content TEXT NOT NULL,
                      metadata_json TEXT NOT NULL,
                      UNIQUE(document_id, ordinal)
                    );
                    CREATE TABLE IF NOT EXISTS rag_embeddings (
                      chunk_id TEXT NOT NULL REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE,
                      provider TEXT NOT NULL,
                      model_id TEXT NOT NULL,
                      model_sha256 TEXT NOT NULL,
                      dimensions INTEGER NOT NULL,
                      dtype TEXT NOT NULL,
                      vector_blob BLOB NOT NULL,
                      created_at TEXT NOT NULL,
                      PRIMARY KEY(chunk_id, provider, model_id, model_sha256)
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_embeddings_identity
                      ON rag_embeddings(provider, model_id, model_sha256, dimensions, chunk_id);
                    CREATE TABLE IF NOT EXISTS rag_jobs (
                      job_id TEXT PRIMARY KEY,
                      kind TEXT NOT NULL,
                      state TEXT NOT NULL,
                      cursor TEXT NOT NULL,
                      error_code TEXT,
                      lease_owner TEXT,
                      lease_expires_at REAL,
                      attempts INTEGER NOT NULL DEFAULT 0,
                      created_at TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS rag_query_events (
                      event_id TEXT PRIMARY KEY,
                      query_digest TEXT NOT NULL,
                      filters_json TEXT NOT NULL,
                      result_ids_json TEXT NOT NULL,
                      created_at TEXT NOT NULL
                    );
                    """
                )
                # S5B: upgrade S5A job rows without replacing the user-owned DB.
                job_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(rag_jobs)").fetchall()
                }
                if "lease_owner" not in job_columns:
                    connection.execute("ALTER TABLE rag_jobs ADD COLUMN lease_owner TEXT")
                if "lease_expires_at" not in job_columns:
                    connection.execute("ALTER TABLE rag_jobs ADD COLUMN lease_expires_at REAL")
                if "attempts" not in job_columns:
                    connection.execute("ALTER TABLE rag_jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
                fts_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_chunks_fts'"
                ).fetchone()
                try:
                    connection.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5("
                        "chunk_id UNINDEXED, document_id UNINDEXED, text_content, tokenize='unicode61')"
                    )
                except sqlite3.OperationalError as exc:
                    raise RagStoreError("fts5_unavailable", "RAG SQLite requires the FTS5 extension") from exc
                if not fts_exists:
                    connection.execute(
                        "INSERT INTO rag_chunks_fts(chunk_id, document_id, text_content) "
                        "SELECT chunk_id, document_id, text_content FROM rag_chunks"
                    )
                cjk_fts_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_chunks_cjk_fts'"
                ).fetchone()
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_cjk_fts USING fts5("
                    "chunk_id UNINDEXED, document_id UNINDEXED, cjk_terms, tokenize='unicode61')"
                )
                cjk_chunk_count = int(connection.execute("SELECT COUNT(*) FROM rag_chunks_cjk_fts").fetchone()[0])
                chunk_count = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
                if not cjk_fts_exists or cjk_chunk_count != chunk_count:
                    connection.execute("DELETE FROM rag_chunks_cjk_fts")
                    for row in connection.execute("SELECT chunk_id, document_id, text_content FROM rag_chunks"):
                        self._cjk_fts_insert(connection, str(row[0]), str(row[1]), str(row[2]))
                connection.execute(
                    "INSERT INTO rag_meta(key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    ("schema_version", _SCHEMA_VERSION, _now()),
                )
            finally:
                connection.close()
            self._initialized = True
        return self.path

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _chunks(self, text: str) -> list[tuple[int, int, str]]:
        """Return bounded, overlapping chunks without splitting near a natural boundary.

        Character offsets intentionally remain offsets into the original text,
        including the overlap.  This keeps citations stable while letting a
        query that straddles a paragraph or sentence boundary retain context.
        """
        if len(text) <= self.max_chunk_chars:
            return [(0, len(text), text)]
        chunks: list[tuple[int, int, str]] = []
        start = 0
        boundary_floor = max(1, int(self.max_chunk_chars * 0.60))
        boundaries = frozenset("\n\r。！？.!?;；")
        while start < len(text):
            hard_end = min(len(text), start + self.max_chunk_chars)
            end = hard_end
            if hard_end < len(text):
                for index in range(hard_end - 1, start + boundary_floor - 1, -1):
                    if text[index] in boundaries or text[index].isspace():
                        end = index + 1
                        break
            chunks.append((start, end, text[start:end]))
            if end >= len(text):
                break
            next_start = end - self.chunk_overlap_chars
            start = next_start if next_start > start else end
        return chunks

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", query).split())

    @staticmethod
    def _cjk_terms(text: str) -> str:
        terms: list[str] = []
        for match in _CJK_RUN.finditer(text):
            value = match.group(0)
            terms.extend(value[index:index + 2] for index in range(len(value) - 1))
        return " ".join(terms)

    @classmethod
    def _cjk_query(cls, query: str) -> str | None:
        terms = cls._cjk_terms(query).split()
        if not terms:
            return None
        unique_terms = list(dict.fromkeys(terms))[:_MAX_CJK_QUERY_TERMS]
        # Terms are generated from fixed-width CJK n-grams, not supplied as
        # FTS syntax. The OR form lets a partial phrase remain discoverable.
        return " OR ".join(unique_terms)

    @staticmethod
    def _fts_delete(connection: sqlite3.Connection, chunk_ids: Iterator[str]) -> None:
        for chunk_id in chunk_ids:
            connection.execute("DELETE FROM rag_chunks_fts WHERE chunk_id = ?", (chunk_id,))

    @staticmethod
    def _fts_insert(connection: sqlite3.Connection, chunk_id: str, document_id: str, text: str) -> None:
        connection.execute(
            "INSERT INTO rag_chunks_fts(chunk_id, document_id, text_content) VALUES (?, ?, ?)",
            (chunk_id, document_id, text),
        )

    @staticmethod
    def _cjk_fts_delete(connection: sqlite3.Connection, chunk_ids: Iterator[str]) -> None:
        for chunk_id in chunk_ids:
            connection.execute("DELETE FROM rag_chunks_cjk_fts WHERE chunk_id = ?", (chunk_id,))

    @classmethod
    def _cjk_fts_insert(cls, connection: sqlite3.Connection, chunk_id: str, document_id: str, text: str) -> None:
        connection.execute(
            "INSERT INTO rag_chunks_cjk_fts(chunk_id, document_id, cjk_terms) VALUES (?, ?, ?)",
            (chunk_id, document_id, cls._cjk_terms(text)),
        )

    @staticmethod
    def _embedding_identity(
        provider: object, model_id: object, model_sha256: object, dimensions: object,
    ) -> tuple[str, str, str, int]:
        if not isinstance(provider, str) or not _EMBEDDING_PROVIDER.fullmatch(provider):
            raise RagStoreError("embedding_provider_invalid", "embedding provider identity is invalid")
        if not isinstance(model_id, str) or not _EMBEDDING_MODEL_ID.fullmatch(model_id):
            raise RagStoreError("embedding_model_invalid", "embedding model identity is invalid")
        digest = _sha(model_sha256, code="embedding_model_digest_invalid")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 32_768:
            raise RagStoreError("embedding_dimensions_invalid", "embedding dimensions are invalid")
        return provider, model_id, digest, dimensions

    @staticmethod
    def _pack_vector(vector: Sequence[object], dimensions: int) -> bytes:
        if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
            raise RagStoreError("embedding_dimensions_invalid", "embedding vector dimensions do not match")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise RagStoreError("embedding_vector_invalid", "embedding vector contains a non-number") from exc
        if any(not math.isfinite(value) for value in values):
            raise RagStoreError("embedding_vector_invalid", "embedding vector contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise RagStoreError("embedding_vector_invalid", "embedding vector must not be zero")
        try:
            return struct.pack(f"<{dimensions}f", *values)
        except struct.error as exc:
            raise RagStoreError("embedding_vector_invalid", "embedding vector cannot be stored as float32") from exc

    @staticmethod
    def _unpack_vector(blob: object, dimensions: int) -> tuple[float, ...]:
        if not isinstance(blob, bytes) or len(blob) != dimensions * 4:
            raise RagStoreError("embedding_corrupt", "stored embedding has an invalid float32 payload")
        try:
            values = struct.unpack(f"<{dimensions}f", blob)
        except struct.error as exc:
            raise RagStoreError("embedding_corrupt", "stored embedding cannot be decoded") from exc
        if any(not math.isfinite(value) for value in values):
            raise RagStoreError("embedding_corrupt", "stored embedding contains a non-finite value")
        return values

    def ingest_document(
        self, *, source_id: str, relative_ref: str, sha256: str | None, mime: str, title: str,
        text: str, revision: str, language: str = "und", owner_scope: str = "local_user",
        access_scope: str = "owner", metadata: Mapping[str, Any] | None = None,
    ) -> IngestResult:
        source_id = _safe_id(source_id, _SOURCE_ID, code="source_id_invalid", label="source_id")
        revision = _safe_id(revision, _REVISION, code="revision_invalid", label="revision")
        relative_ref = _validate_ref(relative_ref)
        mime = _validate_mime(mime)
        owner_scope = _validate_scope(owner_scope, _OWNER_SCOPES, code="owner_scope_invalid", label="owner_scope")
        access_scope = _validate_scope(access_scope, _ACCESS_SCOPES, code="access_scope_invalid", label="access_scope")
        if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z0-9-]{2,16}", language):
            raise RagStoreError("language_invalid", "language is invalid")
        if not isinstance(title, str) or not title.strip() or len(title) > 256:
            raise RagStoreError("title_invalid", "title is invalid")
        text = _validate_content(text, max_bytes=self.max_document_bytes)
        text_bytes = text.encode("utf-8")
        text_digest = _digest(text_bytes)
        source_digest = _sha(sha256) if sha256 is not None else text_digest
        if metadata is None:
            metadata_value: dict[str, Any] = {}
        elif isinstance(metadata, Mapping):
            try:
                metadata_value = json.loads(json.dumps(dict(metadata), ensure_ascii=True))
            except (TypeError, ValueError) as exc:
                raise RagStoreError("metadata_invalid", "metadata must be JSON serializable") from exc
        else:
            raise RagStoreError("metadata_invalid", "metadata must be a JSON object")
        metadata_json = json.dumps(metadata_value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        if len(metadata_json.encode("utf-8")) > 16 * 1024:
            raise RagStoreError("metadata_too_large", "metadata exceeds the local limit")
        document_id = _digest(f"{source_id}\0{revision}\0{text_digest}".encode("utf-8"))
        chunks = self._chunks(text)
        now = _now()
        try:
            with self._write() as connection:
                existing_source = connection.execute(
                    "SELECT owner_scope, relative_ref, mime FROM rag_sources WHERE source_id = ?", (source_id,)
                ).fetchone()
                if existing_source and (
                    existing_source["owner_scope"] != owner_scope
                    or existing_source["relative_ref"] != relative_ref
                    or existing_source["mime"] != mime
                ):
                    raise RagStoreError("source_conflict", "source identity or boundary changed")
                existing_doc = connection.execute(
                    "SELECT document_id, text_digest, status FROM rag_documents WHERE source_id = ? AND revision = ?",
                    (source_id, revision),
                ).fetchone()
                if existing_doc:
                    if existing_doc["text_digest"] != text_digest:
                        raise RagStoreError("revision_conflict", "revision already contains different text")
                    return IngestResult(source_id, existing_doc["document_id"], revision, text_digest, 0, "duplicate")
                connection.execute(
                    "INSERT INTO rag_sources(source_id, owner_scope, relative_ref, sha256, mime, title, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?) "
                    "ON CONFLICT(source_id) DO UPDATE SET sha256=excluded.sha256, title=excluded.title, status='active', updated_at=excluded.updated_at",
                    (source_id, owner_scope, relative_ref, source_digest, mime, title.strip(), now, now),
                )
                old_chunk_ids = connection.execute(
                    "SELECT c.chunk_id FROM rag_chunks c "
                    "JOIN rag_documents d ON d.document_id = c.document_id "
                    "WHERE d.source_id = ? AND d.status = 'active'",
                    (source_id,),
                ).fetchall()
                self._fts_delete(connection, (row[0] for row in old_chunk_ids))
                self._cjk_fts_delete(connection, (row[0] for row in old_chunk_ids))
                connection.execute(
                    "UPDATE rag_documents SET status='superseded', updated_at=? WHERE source_id=? AND status='active'",
                    (now, source_id),
                )
                connection.execute(
                    "INSERT INTO rag_documents(document_id, source_id, revision, text_digest, language, access_scope, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                    (document_id, source_id, revision, text_digest, language, access_scope, now, now),
                )
                for ordinal, (start, end, chunk) in enumerate(chunks):
                    chunk_id = _digest(f"{document_id}\0{ordinal}\0{_digest(chunk.encode('utf-8'))}".encode("utf-8"))
                    connection.execute(
                        "INSERT INTO rag_chunks(chunk_id, document_id, ordinal, text_digest, token_count, start_offset, end_offset, text_content, metadata_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (chunk_id, document_id, ordinal, _digest(chunk.encode("utf-8")), len(chunk.split()), start, end, chunk, metadata_json),
                    )
                    self._fts_insert(connection, chunk_id, document_id, chunk)
                    self._cjk_fts_insert(connection, chunk_id, document_id, chunk)
        except sqlite3.IntegrityError as exc:
            raise RagStoreError("storage_conflict", "RAG document transaction conflicted") from exc
        return IngestResult(source_id, document_id, revision, text_digest, len(chunks), "ingested")

    def delete_source(self, source_id: str) -> bool:
        source_id = _safe_id(source_id, _SOURCE_ID, code="source_id_invalid", label="source_id")
        with self._write() as connection:
            chunk_ids = connection.execute(
                "SELECT c.chunk_id FROM rag_chunks c "
                "JOIN rag_documents d ON d.document_id = c.document_id WHERE d.source_id = ?",
                (source_id,),
            ).fetchall()
            self._fts_delete(connection, (row[0] for row in chunk_ids))
            self._cjk_fts_delete(connection, (row[0] for row in chunk_ids))
            result = connection.execute("DELETE FROM rag_sources WHERE source_id = ?", (source_id,))
            return bool(result.rowcount)

    def reset_index(self) -> dict[str, int]:
        """Remove all materialized documents/chunks while retaining source declarations."""
        with self._write() as connection:
            counts = {
                "embeddings": int(connection.execute("SELECT COUNT(*) FROM rag_embeddings").fetchone()[0]),
                "chunks": int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]),
                "documents": int(connection.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]),
                "jobs": int(connection.execute("SELECT COUNT(*) FROM rag_jobs").fetchone()[0]),
                "query_events": int(connection.execute("SELECT COUNT(*) FROM rag_query_events").fetchone()[0]),
            }
            connection.execute("DELETE FROM rag_query_events")
            connection.execute("DELETE FROM rag_jobs")
            connection.execute("DELETE FROM rag_embeddings")
            connection.execute("DELETE FROM rag_chunks_fts")
            connection.execute("DELETE FROM rag_chunks_cjk_fts")
            connection.execute("DELETE FROM rag_documents")
            connection.execute("UPDATE rag_sources SET status='pending', updated_at=?", (_now(),))
            return counts

    def rebuild_fts(self) -> int:
        """Recreate the FTS index from materialized chunks in one transaction."""
        with self._write() as connection:
            connection.execute("DELETE FROM rag_chunks_fts")
            connection.execute(
                "INSERT INTO rag_chunks_fts(chunk_id, document_id, text_content) "
                "SELECT chunk_id, document_id, text_content FROM rag_chunks"
            )
            connection.execute("DELETE FROM rag_chunks_cjk_fts")
            for row in connection.execute("SELECT chunk_id, document_id, text_content FROM rag_chunks"):
                self._cjk_fts_insert(connection, str(row[0]), str(row[1]), str(row[2]))
            return int(connection.execute("SELECT COUNT(*) FROM rag_chunks_fts").fetchone()[0])

    def embedding_capacity(self, *, dimensions: int = 768) -> dict[str, int | str]:
        """Return a conservative local capacity estimate without materializing vectors."""
        if isinstance(dimensions, bool) or not 1 <= int(dimensions) <= 32_768:
            raise RagStoreError("embedding_dimensions_invalid", "embedding dimensions are invalid")
        self.initialize()
        connection = self._connect()
        try:
            active_chunks = int(connection.execute(
                "SELECT COUNT(*) FROM rag_chunks c "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id "
                "WHERE d.status='active' AND s.status='active'"
            ).fetchone()[0])
            embedding_count = int(connection.execute("SELECT COUNT(*) FROM rag_embeddings").fetchone()[0])
            db_bytes = self.path.stat().st_size if self.path.exists() else 0
            wal_path = Path(f"{self.path}-wal")
            wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
            vector_bytes = active_chunks * int(dimensions) * 4
            return {
                "dimensions": int(dimensions),
                "active_chunk_count": active_chunks,
                "embedding_count": embedding_count,
                "estimated_vector_bytes": vector_bytes,
                "database_bytes": int(db_bytes),
                "wal_bytes": int(wal_bytes),
                "estimated_total_bytes": int(db_bytes + wal_bytes + vector_bytes),
                "max_index_chunks": _MAX_INDEX_CHUNKS,
            }
        finally:
            connection.close()

    @staticmethod
    def _job_id(value: object) -> str:
        return _safe_id(value, _JOB_ID, code="rag_job_id_invalid", label="job_id")

    def _read_job(self, connection: sqlite3.Connection, job_id: str) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM rag_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise RagStoreError("rag_job_not_found", "RAG embedding job was not found")
        item = dict(row)
        try:
            item["cursor"] = json.loads(item["cursor"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise RagStoreError("rag_job_corrupt", "RAG embedding job cursor is corrupt") from exc
        if item["state"] not in _JOB_STATES or not isinstance(item["cursor"], dict):
            raise RagStoreError("rag_job_corrupt", "RAG embedding job state is corrupt")
        return item

    def _claim_embedding_job(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        owner: str,
        lease_seconds: int,
    ) -> dict[str, Any]:
        now = time.time()
        job = self._read_job(connection, job_id)
        if job["state"] in {"completed", "cancelled"}:
            return job
        current_owner = job.get("lease_owner")
        current_expiry = float(job.get("lease_expires_at") or 0.0)
        if current_owner and current_owner != owner and current_expiry > now:
            raise RagStoreError("rag_job_leased", "RAG embedding job is leased by another worker")
        expires = now + int(lease_seconds)
        updated = connection.execute(
            "UPDATE rag_jobs SET state='running', lease_owner=?, lease_expires_at=?, "
            "attempts=attempts+1, error_code=NULL, updated_at=? "
            "WHERE job_id=? AND (lease_owner IS NULL OR lease_expires_at IS NULL "
            "OR lease_expires_at<=? OR lease_owner=?)",
            (owner, expires, _now(), job_id, now, owner),
        )
        if updated.rowcount != 1:
            raise RagStoreError("rag_job_leased", "RAG embedding job lease was taken by another worker")
        return self._read_job(connection, job_id)

    def _mark_embedding_job_error(self, job_id: str, owner: str, exc: BaseException, *, retryable: bool = False) -> None:
        code = str(getattr(exc, "code", type(exc).__name__))
        with self._write() as connection:
            current = self._read_job(connection, job_id)
            if current.get("lease_owner") == owner:
                connection.execute(
                    "UPDATE rag_jobs SET state=?, error_code=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=?",
                    ("paused" if retryable else "failed", code, _now(), job_id),
                )

    def create_embedding_job(
        self,
        *,
        provider: str,
        model_sha256: str,
        source_id: str | None = None,
        batch_size: int = _MAX_EMBEDDING_BATCH,
    ) -> dict[str, Any]:
        """Create a deterministic, user-owned embedding job snapshot."""
        provider = _safe_id(provider, _EMBEDDING_PROVIDER, code="embedding_provider_invalid", label="provider")
        model_sha256 = _sha(model_sha256, code="embedding_model_digest_invalid")
        if source_id is not None:
            source_id = _safe_id(source_id, _SOURCE_ID, code="source_id_invalid", label="source_id")
        if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= _MAX_EMBEDDING_BATCH:
            raise RagStoreError("embedding_batch_invalid", "embedding batch size must be between 1 and 64")
        self.initialize()
        with self._write() as connection:
            sql = (
                "SELECT c.chunk_id FROM rag_chunks c "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id "
                "WHERE d.status='active' AND s.status='active'"
            )
            params: tuple[str, ...] = ()
            if source_id is not None:
                sql += " AND d.source_id=?"
                params = (source_id,)
            sql += " ORDER BY s.source_id, d.revision, c.ordinal, c.chunk_id LIMIT ?"
            rows = connection.execute(sql, (*params, _MAX_INDEX_CHUNKS + 1)).fetchall()
            if len(rows) > _MAX_INDEX_CHUNKS:
                raise RagStoreError("embedding_index_budget_exceeded", "active chunk count exceeds the local indexing budget")
            job_id = uuid.uuid4().hex
            cursor = {
                "schema": 1,
                "provider": provider,
                "model_sha256": model_sha256,
                "source_id": source_id,
                "batch_size": int(batch_size),
                "chunk_ids": [str(row[0]) for row in rows],
                "position": 0,
                "indexed": 0,
                "model_id": None,
                "dimensions": None,
            }
            now = _now()
            connection.execute(
                "INSERT INTO rag_jobs(job_id, kind, state, cursor, error_code, lease_owner, lease_expires_at, attempts, created_at, updated_at) "
                "VALUES (?, 'embedding_index', 'queued', ?, NULL, NULL, NULL, 0, ?, ?)",
                (job_id, json.dumps(cursor, ensure_ascii=True, separators=(",", ":")), now, now),
            )
            return {"job_id": job_id, "kind": "embedding_index", "state": "queued", "cursor": cursor, "created_at": now, "updated_at": now}

    def get_embedding_job(self, job_id: str) -> dict[str, Any]:
        job_id = self._job_id(job_id)
        self.initialize()
        connection = self._connect()
        try:
            return self._read_job(connection, job_id)
        finally:
            connection.close()

    def cancel_embedding_job(self, job_id: str) -> dict[str, Any]:
        job_id = self._job_id(job_id)
        with self._write() as connection:
            job = self._read_job(connection, job_id)
            if job["state"] in {"queued", "running", "paused"}:
                connection.execute(
                    "UPDATE rag_jobs SET state='cancelled', lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=?",
                    (_now(), job_id),
                )
            return self._read_job(connection, job_id)

    def run_embedding_job(
        self,
        job_id: str,
        provider: object,
        *,
        max_batches: int | None = None,
        lease_seconds: int = 120,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Run or resume a bounded job; each committed batch advances its cursor atomically."""
        job_id = self._job_id(job_id)
        if max_batches is not None and (isinstance(max_batches, bool) or not 1 <= int(max_batches) <= 10_000):
            raise RagStoreError("embedding_batch_invalid", "max_batches is outside the allowed range")
        if isinstance(lease_seconds, bool) or not _MIN_JOB_LEASE_SECONDS <= int(lease_seconds) <= _MAX_JOB_LEASE_SECONDS:
            raise RagStoreError("rag_job_lease_invalid", "RAG embedding job lease is outside the allowed range")
        if isinstance(max_retries, bool) or not 0 <= int(max_retries) <= 5:
            raise RagStoreError("embedding_retry_invalid", "embedding retry count is outside the allowed range")
        embed = getattr(provider, "embed", None)
        if not callable(embed):
            raise RagStoreError("embedding_provider_invalid", "embedding provider does not expose embed()")
        self.initialize()
        batches = 0
        owner = f"pid-{os.getpid()}-{uuid.uuid4().hex}"
        with self._embedding_job_lock:
            return self._run_embedding_job_locked(
                job_id, embed, owner=owner, lease_seconds=int(lease_seconds),
                max_batches=max_batches, max_retries=int(max_retries),
            )

    def _run_embedding_job_locked(
        self,
        job_id: str,
        embed: Any,
        *,
        owner: str,
        lease_seconds: int,
        max_batches: int | None,
        max_retries: int,
    ) -> dict[str, Any]:
        batches = 0
        while True:
            with self._write() as connection:
                job = self._claim_embedding_job(connection, job_id, owner, lease_seconds)
                if job["state"] in {"completed", "cancelled"}:
                    return job
                cursor = dict(job["cursor"])
                position = int(cursor.get("position", 0))
                chunk_ids = cursor.get("chunk_ids")
                if not isinstance(chunk_ids, list) or not 0 <= position <= len(chunk_ids):
                    raise RagStoreError("rag_job_corrupt", "RAG embedding job chunk snapshot is corrupt")
                if position >= len(chunk_ids):
                    connection.execute("UPDATE rag_jobs SET state='completed', updated_at=? WHERE job_id=?", (_now(), job_id))
                    return self._read_job(connection, job_id)
                batch_ids = [str(value) for value in chunk_ids[position:position + int(cursor["batch_size"])] ]
                placeholders = ",".join("?" for _ in batch_ids)
                rows = connection.execute(
                    "SELECT c.chunk_id, c.text_content FROM rag_chunks c "
                    "JOIN rag_documents d ON d.document_id=c.document_id "
                    "JOIN rag_sources s ON s.source_id=d.source_id "
                    f"WHERE c.chunk_id IN ({placeholders}) AND d.status='active' AND s.status='active'",
                    batch_ids,
                ).fetchall()
                by_id = {str(row["chunk_id"]): row for row in rows}
                texts = [str(by_id[chunk_id]["text_content"]) for chunk_id in batch_ids if chunk_id in by_id]
            if not texts:
                encoded = None
                provider_name = cursor.get("provider")
                model_id = cursor.get("model_id")
                dimensions = cursor.get("dimensions")
                indexed_delta = 0
            else:
                result = None
                last_error: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        result = embed(texts)
                        break
                    except Exception as exc:
                        last_error = exc
                        if not bool(getattr(exc, "retryable", False)) or attempt >= max_retries:
                            break
                if result is None:
                    exc = last_error or RuntimeError("embedding provider returned no result")
                    retryable = bool(getattr(exc, "retryable", False))
                    self._mark_embedding_job_error(job_id, owner, exc, retryable=retryable)
                    if isinstance(exc, RagStoreError):
                        raise
                    raise RagStoreError(
                        "embedding_provider_retry_exhausted" if bool(getattr(exc, "retryable", False)) else "embedding_provider_failed",
                        "embedding provider failed during resumable indexing",
                    ) from exc
                provider_name = str(getattr(result, "provider"))
                model_id = str(getattr(result, "model_id"))
                dimensions = int(getattr(result, "dimensions"))
                vectors = getattr(result, "vectors")
                try:
                    if provider_name != cursor["provider"]:
                        raise RagStoreError("embedding_provider_mismatch", "provider identity changed during embedding job")
                    if cursor.get("model_id") not in (None, model_id) or cursor.get("dimensions") not in (None, dimensions):
                        raise RagStoreError("embedding_model_mismatch", "embedding model identity changed during embedding job")
                    if not isinstance(vectors, list) or len(vectors) != len(texts):
                        raise RagStoreError("embedding_provider_invalid", "embedding provider returned the wrong vector count")
                    encoded = {chunk_id: vector for chunk_id, vector in zip((chunk_id for chunk_id in batch_ids if chunk_id in by_id), vectors)}
                    indexed_delta = len(encoded)
                    self.upsert_embeddings(
                        provider=provider_name, model_id=model_id, model_sha256=cursor["model_sha256"],
                        dimensions=dimensions, vectors=encoded,
                    )
                except RagStoreError as exc:
                    self._mark_embedding_job_error(job_id, owner, exc)
                    raise
            cursor["position"] = position + len(batch_ids)
            cursor["indexed"] = int(cursor.get("indexed", 0)) + int(indexed_delta)
            cursor["model_id"] = model_id
            cursor["dimensions"] = dimensions
            next_state = "completed" if cursor["position"] >= len(chunk_ids) else "paused"
            with self._write() as connection:
                current = self._read_job(connection, job_id)
                if current["state"] == "cancelled":
                    return current
                if current.get("lease_owner") != owner or float(current.get("lease_expires_at") or 0.0) <= time.time():
                    raise RagStoreError("rag_job_lease_lost", "RAG embedding job lease expired before commit")
                connection.execute(
                    "UPDATE rag_jobs SET state=?, cursor=?, error_code=NULL, lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE job_id=? AND lease_owner=?",
                    (next_state, json.dumps(cursor, ensure_ascii=True, separators=(",", ":")), _now(), job_id, owner),
                )
            batches += 1
            if next_state == "completed" or (max_batches is not None and batches >= int(max_batches)):
                return self.get_embedding_job(job_id)

    def upsert_embeddings(
        self,
        *,
        provider: str,
        model_id: str,
        model_sha256: str,
        dimensions: int,
        vectors: Mapping[str, Sequence[object]],
    ) -> int:
        """Atomically bind a bounded batch of vectors to active local chunks.

        The model digest is mandatory because a mutable provider tag, such as
        ``nomic-embed-text:latest``, is not sufficient evidence that vectors
        remain compatible with a future query embedding.
        """
        provider, model_id, model_sha256, dimensions = self._embedding_identity(
            provider, model_id, model_sha256, dimensions,
        )
        if not isinstance(vectors, Mapping) or not vectors or len(vectors) > _MAX_EMBEDDING_BATCH:
            raise RagStoreError("embedding_batch_invalid", "embedding batch must contain 1-64 chunk vectors")
        encoded: list[tuple[str, bytes]] = []
        for chunk_id, vector in vectors.items():
            chunk_id = _safe_id(chunk_id, _SHA256, code="chunk_id_invalid", label="chunk_id")
            encoded.append((chunk_id, self._pack_vector(vector, dimensions)))
        with self._write() as connection:
            placeholders = ",".join("?" for _ in encoded)
            active = connection.execute(
                "SELECT c.chunk_id FROM rag_chunks c "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id "
                f"WHERE c.chunk_id IN ({placeholders}) AND d.status='active' AND s.status='active'",
                [chunk_id for chunk_id, _ in encoded],
            ).fetchall()
            active_ids = {str(row[0]) for row in active}
            if active_ids != {chunk_id for chunk_id, _ in encoded}:
                raise RagStoreError("embedding_chunk_invalid", "embedding targets must be active local chunks")
            now = _now()
            for chunk_id, blob in encoded:
                connection.execute(
                    "INSERT INTO rag_embeddings(chunk_id, provider, model_id, model_sha256, dimensions, dtype, vector_blob, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 'float32le', ?, ?) "
                    "ON CONFLICT(chunk_id, provider, model_id, model_sha256) DO UPDATE SET "
                    "dimensions=excluded.dimensions, dtype=excluded.dtype, vector_blob=excluded.vector_blob, created_at=excluded.created_at",
                    (chunk_id, provider, model_id, model_sha256, dimensions, blob, now),
                )
            return len(encoded)

    def index_active_embeddings(
        self,
        provider: object,
        *,
        model_sha256: str,
        source_id: str | None = None,
        batch_size: int = _MAX_EMBEDDING_BATCH,
    ) -> int:
        """Embed active chunks through one explicit provider without holding a DB lock.

        The provider result declares its provider/model/dimension identity. The
        mutable provider label alone is never persisted as compatibility proof;
        callers must supply the separately frozen model digest.
        """
        model_sha256 = _sha(model_sha256, code="embedding_model_digest_invalid")
        if source_id is not None:
            source_id = _safe_id(source_id, _SOURCE_ID, code="source_id_invalid", label="source_id")
        if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= _MAX_EMBEDDING_BATCH:
            raise RagStoreError("embedding_batch_invalid", "embedding batch size must be between 1 and 64")
        embed = getattr(provider, "embed", None)
        if not callable(embed):
            raise RagStoreError("embedding_provider_invalid", "embedding provider does not expose embed()")
        self.initialize()
        connection = self._connect()
        try:
            sql = (
                "SELECT c.chunk_id, c.text_content FROM rag_chunks c "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id "
                "WHERE d.status='active' AND s.status='active'"
            )
            params: tuple[str, ...] = ()
            if source_id is not None:
                sql += " AND d.source_id=?"
                params = (source_id,)
            sql += " ORDER BY d.source_id, c.ordinal LIMIT ?"
            rows = connection.execute(sql, (*params, _MAX_INDEX_CHUNKS + 1)).fetchall()
        finally:
            connection.close()
        if len(rows) > _MAX_INDEX_CHUNKS:
            raise RagStoreError("embedding_index_budget_exceeded", "active chunk count exceeds the local indexing budget")
        total = 0
        for offset in range(0, len(rows), int(batch_size)):
            batch = rows[offset:offset + int(batch_size)]
            try:
                result = embed([str(row["text_content"]) for row in batch])
                provider_name = getattr(result, "provider")
                model_id = getattr(result, "model_id")
                dimensions = getattr(result, "dimensions")
                vectors = getattr(result, "vectors")
            except RagStoreError:
                raise
            except Exception as exc:
                raise RagStoreError("embedding_provider_failed", "embedding provider failed during local indexing") from exc
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise RagStoreError("embedding_provider_invalid", "embedding provider returned the wrong vector count")
            total += self.upsert_embeddings(
                provider=provider_name,
                model_id=model_id,
                model_sha256=model_sha256,
                dimensions=dimensions,
                vectors={str(row["chunk_id"]): vector for row, vector in zip(batch, vectors)},
            )
        return total

    def semantic_search(
        self,
        query_vector: Sequence[object],
        *,
        provider: str,
        model_id: str,
        model_sha256: str,
        dimensions: int,
        access_scope: str = "owner",
        limit: int = 20,
        max_scan: int = _DEFAULT_VECTOR_SCAN_LIMIT,
    ) -> list[dict[str, Any]]:
        """Bounded cosine search over a single frozen local embedding identity."""
        provider, model_id, model_sha256, dimensions = self._embedding_identity(
            provider, model_id, model_sha256, dimensions,
        )
        if isinstance(limit, bool) or not 1 <= int(limit) <= 100:
            raise RagStoreError("limit_invalid", "query limit must be between 1 and 100")
        if isinstance(max_scan, bool) or not 1 <= int(max_scan) <= 10_000:
            raise RagStoreError("vector_scan_limit_invalid", "vector scan limit must be between 1 and 10000")
        access_scope = _validate_scope(access_scope, _ACCESS_SCOPES, code="access_scope_invalid", label="access_scope")
        query = self._unpack_vector(self._pack_vector(query_vector, dimensions), dimensions)
        query_norm = math.sqrt(sum(value * value for value in query))
        with self._write() as connection:
            where = (
                "e.provider=? AND e.model_id=? AND e.model_sha256=? AND e.dimensions=? "
                "AND d.status='active' AND s.status='active' AND d.access_scope=?"
            )
            params = (provider, model_id, model_sha256, dimensions, access_scope)
            count = int(connection.execute(
                "SELECT COUNT(*) FROM rag_embeddings e "
                "JOIN rag_chunks c ON c.chunk_id=e.chunk_id "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id WHERE " + where,
                params,
            ).fetchone()[0])
            if count > int(max_scan):
                raise RagStoreError(
                    "vector_scan_budget_exceeded",
                    "vector corpus exceeds the configured local scan budget; use FTS5 or raise the explicit budget",
                )
            rows = connection.execute(
                "SELECT e.chunk_id, e.vector_blob, c.document_id, d.source_id, d.revision, d.access_scope, "
                "s.relative_ref, c.text_content, c.ordinal FROM rag_embeddings e "
                "JOIN rag_chunks c ON c.chunk_id=e.chunk_id "
                "JOIN rag_documents d ON d.document_id=c.document_id "
                "JOIN rag_sources s ON s.source_id=d.source_id WHERE " + where,
                params,
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                vector = self._unpack_vector(row["vector_blob"], dimensions)
                norm = math.sqrt(sum(value * value for value in vector))
                if norm == 0.0:
                    raise RagStoreError("embedding_corrupt", "stored embedding vector must not be zero")
                score = sum(left * right for left, right in zip(query, vector)) / (query_norm * norm)
                result = dict(row)
                result.pop("vector_blob", None)
                result["vector_score"] = score
                results.append(result)
            results.sort(key=lambda row: (-float(row["vector_score"]), int(row["ordinal"]), str(row["chunk_id"])))
            return results[:int(limit)]

    def hybrid_search(
        self,
        query: str,
        query_vector: Sequence[object],
        *,
        provider: str,
        model_id: str,
        model_sha256: str,
        dimensions: int,
        access_scope: str = "owner",
        limit: int = 20,
        max_scan: int = _DEFAULT_VECTOR_SCAN_LIMIT,
    ) -> list[dict[str, Any]]:
        """Merge FTS5 and bounded cosine candidates with explicit FTS fallback."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise RagStoreError("limit_invalid", "query limit must be between 1 and 100")
        lexical = self.search(query, access_scope=access_scope, limit=max(20, min(100, int(limit) * 4)))
        fallback_reason = ""
        try:
            semantic = self.semantic_search(
                query_vector, provider=provider, model_id=model_id, model_sha256=model_sha256,
                dimensions=dimensions, access_scope=access_scope, limit=max(20, min(100, int(limit) * 4)),
                max_scan=max_scan,
            )
        except RagStoreError as exc:
            if exc.code != "vector_scan_budget_exceeded":
                raise
            semantic = []
            fallback_reason = exc.code
        merged: dict[str, dict[str, Any]] = {}
        lexical_count = len(lexical)
        for index, row in enumerate(lexical):
            item = dict(row)
            # bm25 values are implementation-scale dependent (and negative in
            # SQLite FTS5). Normalize only the candidate ordering instead.
            item["lexical_score"] = 1.0 - (index / max(1, lexical_count - 1))
            item["vector_score"] = 0.0
            merged[str(item["chunk_id"])] = item
        for row in semantic:
            item = merged.get(str(row["chunk_id"]))
            if item is None:
                item = dict(row)
                item["lexical_score"] = 0.0
                merged[str(item["chunk_id"])] = item
            item["vector_score"] = float(row["vector_score"])
        output = list(merged.values())
        for item in output:
            item["hybrid_score"] = 0.55 * float(item["lexical_score"]) + 0.45 * ((float(item["vector_score"]) + 1.0) / 2.0)
            item["hybrid_mode"] = "fts_fallback" if fallback_reason else "fts_vector"
            if fallback_reason:
                item["vector_reason_code"] = fallback_reason
        output.sort(key=lambda row: (-float(row["hybrid_score"]), int(row["ordinal"]), str(row["chunk_id"])))
        return output[:int(limit)]

    @staticmethod
    def _search_cjk(connection: sqlite3.Connection, query: str, access_scope: str, limit: int) -> list[sqlite3.Row]:
        cjk_query = RagStore._cjk_query(query)
        if not cjk_query:
            return []
        return connection.execute(
            "SELECT f.chunk_id, c.document_id, d.source_id, d.revision, "
            "d.access_scope, s.relative_ref, c.text_content, c.ordinal, "
            "bm25(rag_chunks_cjk_fts) AS rank "
            "FROM rag_chunks_cjk_fts AS f "
            "JOIN rag_chunks AS c ON c.chunk_id = f.chunk_id "
            "JOIN rag_documents AS d ON d.document_id = c.document_id "
            "JOIN rag_sources AS s ON s.source_id = d.source_id "
            "WHERE rag_chunks_cjk_fts MATCH ? AND d.status='active' AND s.status='active' "
            "AND d.access_scope = ? ORDER BY rank ASC, c.ordinal ASC LIMIT ?",
            (cjk_query, access_scope, int(limit)),
        ).fetchall()

    def search(self, query: str, *, access_scope: str = "owner", limit: int = 20) -> list[dict[str, Any]]:
        """Search active chunks with an exact local access-scope filter.

        The raw query never enters the audit table; only its SHA-256 digest and
        returned chunk IDs are retained.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 512 or "\x00" in query:
            raise RagStoreError("query_invalid", "query must be non-empty, bounded, and NUL-free")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 100:
            raise RagStoreError("limit_invalid", "query limit must be between 1 and 100")
        access_scope = _validate_scope(access_scope, _ACCESS_SCOPES, code="access_scope_invalid", label="access_scope")
        query = self._normalize_query(query)
        if not query:
            raise RagStoreError("query_invalid", "query must contain searchable text")
        with self._write() as connection:
            try:
                rows = connection.execute(
                    "SELECT f.chunk_id, c.document_id, d.source_id, d.revision, "
                    "d.access_scope, s.relative_ref, f.text_content, c.ordinal, bm25(rag_chunks_fts) AS rank "
                    "FROM rag_chunks_fts AS f "
                    "JOIN rag_chunks AS c ON c.chunk_id = f.chunk_id "
                    "JOIN rag_documents AS d ON d.document_id = c.document_id "
                    "JOIN rag_sources AS s ON s.source_id = d.source_id "
                    "WHERE rag_chunks_fts MATCH ? AND d.status='active' AND s.status='active' "
                    "AND d.access_scope = ? ORDER BY rank ASC, c.ordinal ASC LIMIT ?",
                    (query, access_scope, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if any(marker in query for marker in ('"', "'", "*", "^", ":", "(", ")", "{", "}")):
                    raise RagStoreError("query_invalid", "query is not valid FTS5 syntax") from exc
                rows = self._search_cjk(connection, query, access_scope, int(limit))
                if not rows:
                    raise RagStoreError("query_invalid", "query is not valid FTS5 syntax") from exc
            if not rows:
                rows = self._search_cjk(connection, query, access_scope, int(limit))
            results = [dict(row) for row in rows]
            event_filters = json.dumps(
                {"access_scope": access_scope, "limit": int(limit)},
                ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            )
            result_ids = json.dumps(
                [row["chunk_id"] for row in results], ensure_ascii=True, separators=(",", ":")
            )
            connection.execute(
                "INSERT INTO rag_query_events(event_id, query_digest, filters_json, result_ids_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, _digest(query.encode("utf-8")), event_filters, result_ids, _now()),
            )
            return results

    def list_sources(self, *, owner_scope: str | None = None) -> list[dict[str, Any]]:
        self.initialize()
        if owner_scope is not None:
            owner_scope = _validate_scope(owner_scope, _OWNER_SCOPES, code="owner_scope_invalid", label="owner_scope")
        connection = self._connect()
        try:
            if owner_scope is None:
                rows = connection.execute("SELECT * FROM rag_sources ORDER BY updated_at DESC, source_id").fetchall()
            else:
                rows = connection.execute("SELECT * FROM rag_sources WHERE owner_scope=? ORDER BY updated_at DESC, source_id", (owner_scope,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def health(self) -> dict[str, Any]:
        self.initialize()
        connection = self._connect()
        try:
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower()
            synchronous_value = int(connection.execute("PRAGMA synchronous").fetchone()[0])
            synchronous = {0: "off", 1: "normal", 2: "full", 3: "extra"}.get(
                synchronous_value, str(synchronous_value)
            )
            return {
                "schema": RAG_SCHEMA,
                "status": "ok" if quick == "ok" else "error",
                "path": str(self.path),
                "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "synchronous": synchronous,
                "source_count": int(connection.execute("SELECT COUNT(*) FROM rag_sources").fetchone()[0]),
                "document_count": int(connection.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]),
                "chunk_count": int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]),
                "fts_chunk_count": int(connection.execute("SELECT COUNT(*) FROM rag_chunks_fts").fetchone()[0]),
                "embedding_count": int(connection.execute("SELECT COUNT(*) FROM rag_embeddings").fetchone()[0]),
                "query_event_count": int(connection.execute("SELECT COUNT(*) FROM rag_query_events").fetchone()[0]),
            }
        finally:
            connection.close()
