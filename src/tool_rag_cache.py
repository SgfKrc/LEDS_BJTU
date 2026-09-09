"""Explicit, user-owned persistence for normalized web-tool results.

G5 keeps network results out of RAG unless a caller passes ``persist=True``.
The cache ledger shares the user's RagStore SQLite file, while document text
continues through RagStore's secret checks, FTS index, access scope, and
revision rules.  Raw provider responses are never stored.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

try:  # Package import for tests; top-level imports match the application's src path.
    from .rag_store import RagStore, RagStoreError
    from .tool_gateway import ToolGatewayPolicy, normalize_tool_result
except ImportError:  # pragma: no cover - exercised by the application import layout.
    from rag_store import RagStore, RagStoreError
    from tool_gateway import ToolGatewayPolicy, normalize_tool_result


TOOL_CACHE_SCHEMA = "qlh.tool_cache.v1"
_CACHE_ID = re.compile(r"^tc_[0-9a-f]{32}$")
_TOOL_NAME = frozenset({"web_search", "web_fetch"})
_OWNER_SCOPES = frozenset({"local_user", "local_system", "project"})
_ACCESS_SCOPES = frozenset({"owner", "local_system", "project"})
_INJECTION_MARKERS = re.compile(
    r"(?is)(?:ignore\s+(?:all|any|the)\s+previous|system\s+message|"
    r"developer\s+message|jailbreak|<\|(?:system|assistant|tool)\|>)"
)


class ToolRagCacheError(ValueError):
    """Stable cache boundary error."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(message)


def _json(value: Any, *, code: str, max_bytes: int) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolRagCacheError(code, "cache metadata must be JSON serializable") from exc
    if len(encoded) > max_bytes:
        raise ToolRagCacheError("cache_metadata_too_large", "cache metadata exceeds the local limit")
    return encoded.decode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ToolRagCache:
    """TTL/capacity governed ledger over a user-owned :class:`RagStore`."""

    def __init__(
        self,
        store: RagStore,
        *,
        policy: ToolGatewayPolicy | None = None,
        max_entries: int = 256,
        max_bytes: int = 64 * 1024 * 1024,
        default_ttl_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        if not isinstance(store, RagStore):
            raise TypeError("store must be a RagStore")
        if isinstance(max_entries, bool) or not 1 <= int(max_entries) <= 10_000:
            raise ValueError("max_entries must be between 1 and 10000")
        if isinstance(max_bytes, bool) or not 1_024 <= int(max_bytes) <= 1 << 30:
            raise ValueError("max_bytes is outside the local cache limit")
        if isinstance(default_ttl_seconds, bool) or not 1 <= int(default_ttl_seconds) <= 90 * 24 * 60 * 60:
            raise ValueError("default_ttl_seconds is outside the local cache limit")
        self.store = store
        self.policy = policy or ToolGatewayPolicy()
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.default_ttl_seconds = int(default_ttl_seconds)
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.store.initialize()
        connection = sqlite3.connect(str(self.store.path), timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=FULL")
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            connection.close()
            raise ToolRagCacheError("sqlite_wal_required", "tool cache requires SQLite WAL mode")
        return connection

    def initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS rag_tool_cache (
                      cache_id TEXT PRIMARY KEY,
                      source_id TEXT NOT NULL UNIQUE,
                      request_id TEXT NOT NULL,
                      tool_name TEXT NOT NULL,
                      owner_scope TEXT NOT NULL,
                      access_scope TEXT NOT NULL,
                      title TEXT NOT NULL,
                      items_json TEXT NOT NULL,
                      citations_json TEXT NOT NULL,
                      content_bytes INTEGER NOT NULL,
                      content_digest TEXT NOT NULL,
                      prompt_injection_suspected INTEGER NOT NULL DEFAULT 0,
                      created_at REAL NOT NULL,
                      updated_at REAL NOT NULL,
                      expires_at REAL NOT NULL,
                      last_accessed_at REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_tool_cache_expiry
                      ON rag_tool_cache(expires_at, updated_at);
                    CREATE INDEX IF NOT EXISTS idx_rag_tool_cache_scope
                      ON rag_tool_cache(access_scope, owner_scope, updated_at);
                    """
                )
            finally:
                connection.close()

    @staticmethod
    def _validate_scope(owner_scope: str, access_scope: str) -> tuple[str, str]:
        if owner_scope not in _OWNER_SCOPES:
            raise ToolRagCacheError("owner_scope_invalid", "owner scope is outside the local boundary")
        if access_scope not in _ACCESS_SCOPES:
            raise ToolRagCacheError("access_scope_invalid", "access scope is outside the local boundary")
        return owner_scope, access_scope

    @staticmethod
    def _validate_cache_id(cache_id: str) -> str:
        if not isinstance(cache_id, str) or _CACHE_ID.fullmatch(cache_id) is None:
            raise ToolRagCacheError("cache_id_invalid", "cache id is invalid")
        return cache_id

    @staticmethod
    def _validate_ttl(ttl_seconds: int | None, default: int) -> int:
        value = default if ttl_seconds is None else ttl_seconds
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 90 * 24 * 60 * 60:
            raise ToolRagCacheError("cache_ttl_invalid", "cache TTL is outside the local limit")
        return int(value)

    @staticmethod
    def _result_parts(
        result: Mapping[str, Any],
        *,
        request_id: str,
        tool_name: str,
        policy: ToolGatewayPolicy,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, str]], bool]:
        if tool_name not in _TOOL_NAME:
            raise ToolRagCacheError("tool_name_invalid", "tool name is not persistable")
        try:
            normalized = normalize_tool_result(result, request_id=request_id, policy=policy)
        except Exception as exc:
            code = str(getattr(exc, "code", "tool_result_invalid"))
            raise ToolRagCacheError(code, "tool result does not satisfy the cache contract") from exc
        if normalized.get("status") != "ok":
            raise ToolRagCacheError("tool_result_error", "failed tool results are not persisted")
        items = normalized.get("items")
        citations = normalized.get("citations")
        if not isinstance(items, list) or not isinstance(citations, list) or not items:
            raise ToolRagCacheError("citation_incomplete", "a persisted result must contain items and citations")
        citation_urls = {str(item.get("url")) for item in citations if isinstance(item, Mapping)}
        if any(not isinstance(item, Mapping) or str(item.get("url")) not in citation_urls for item in items):
            raise ToolRagCacheError("citation_incomplete", "every persisted result item needs a citation")
        text_parts: list[str] = []
        for item in items:
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            url = str(item.get("url", "")).strip()
            if not title or not snippet or not url:
                raise ToolRagCacheError("tool_result_invalid", "persisted item is incomplete")
            text_parts.append(f"{title}\n{snippet}\nSource: {url}")
        text = "\n\n".join(text_parts)
        content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return normalized, text, [dict(item) for item in items], [dict(item) for item in citations], bool(_INJECTION_MARKERS.search(text))

    def _rows(self, *, access_scope: str | None = None) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            if access_scope is None:
                rows = connection.execute("SELECT * FROM rag_tool_cache ORDER BY updated_at DESC, cache_id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM rag_tool_cache WHERE access_scope=? ORDER BY updated_at DESC, cache_id",
                    (access_scope,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            items = json.loads(str(row.get("items_json", "[]")))
            citations = json.loads(str(row.get("citations_json", "[]")))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ToolRagCacheError("cache_corrupt", "cache citation metadata is corrupt") from exc
        return {
            "schema": TOOL_CACHE_SCHEMA,
            "cache_id": str(row["cache_id"]),
            "source_id": str(row["source_id"]),
            "request_id": str(row["request_id"]),
            "tool_name": str(row["tool_name"]),
            "owner_scope": str(row["owner_scope"]),
            "access_scope": str(row["access_scope"]),
            "title": str(row["title"]),
            "items": items,
            "citations": citations,
            "content_bytes": int(row["content_bytes"]),
            "content_digest": str(row["content_digest"]),
            "prompt_injection_suspected": bool(row["prompt_injection_suspected"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "expires_at": float(row["expires_at"]),
            "expired": float(row["expires_at"]) <= time.time(),
        }

    def _delete_row(self, cache_id: str, row: Mapping[str, Any]) -> bool:
        # RagStore owns the document/chunk/FTS transaction.  The ledger row is
        # removed only after that boundary succeeds, so failed deletion stays
        # visible and can be retried safely.
        self.store.delete_source(str(row["source_id"]))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            deleted = connection.execute("DELETE FROM rag_tool_cache WHERE cache_id=?", (cache_id,)).rowcount
            connection.commit()
            return bool(deleted)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def purge_expired(self, *, now: float | None = None, limit: int = 100) -> dict[str, int]:
        if isinstance(limit, bool) or not 1 <= int(limit) <= 1_000:
            raise ToolRagCacheError("cache_limit_invalid", "purge limit is invalid")
        cutoff = time.time() if now is None else float(now)
        with self._lock:
            self.initialize()
            rows = [row for row in self._rows() if float(row["expires_at"]) <= cutoff][: int(limit)]
            deleted = 0
            for row in rows:
                if self._delete_row(str(row["cache_id"]), row):
                    deleted += 1
            return {"expired": len(rows), "deleted": deleted}

    def save_tool_result(
        self,
        result: Mapping[str, Any],
        *,
        tool_name: str,
        persist: bool = False,
        owner_scope: str = "local_user",
        access_scope: str = "owner",
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        if persist is not True:
            raise ToolRagCacheError("persistence_not_explicit", "tool result persistence requires persist=true")
        if not isinstance(result, Mapping):
            raise ToolRagCacheError("tool_result_invalid", "tool result must be an object")
        request_id = str(result.get("request_id", ""))
        if not request_id:
            raise ToolRagCacheError("request_id_invalid", "tool result request id is required")
        owner_scope, access_scope = self._validate_scope(owner_scope, access_scope)
        ttl = self._validate_ttl(ttl_seconds, self.default_ttl_seconds)
        normalized, text, items, citations, injection_suspected = self._result_parts(
            result, request_id=request_id, tool_name=tool_name, policy=self.policy,
        )
        key = {"request_id": request_id, "tool_name": tool_name, "urls": sorted(str(item["url"]) for item in citations)}
        cache_id = f"tc_{_digest(key)[:32]}"
        source_id = cache_id
        content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        content_bytes = len(text.encode("utf-8"))
        title = str(items[0]["title"])[:256]
        now = time.time()
        expires_at = now + ttl
        items_json = _json(items, code="cache_items_invalid", max_bytes=16 * 1024)
        citations_json = _json(citations, code="cache_citations_invalid", max_bytes=16 * 1024)
        with self._lock:
            self.initialize()
            self.purge_expired(now=now, limit=1_000)
            connection = self._connect()
            try:
                existing = connection.execute("SELECT * FROM rag_tool_cache WHERE cache_id=?", (cache_id,)).fetchone()
                existing_bytes = int(existing["content_bytes"]) if existing is not None else 0
                existing_count = 1 if existing is not None else 0
                current_bytes = int(connection.execute("SELECT COALESCE(SUM(content_bytes),0) FROM rag_tool_cache").fetchone()[0])
                current_count = int(connection.execute("SELECT COUNT(*) FROM rag_tool_cache").fetchone()[0])
                if current_count - existing_count + 1 > self.max_entries:
                    raise ToolRagCacheError("cache_capacity_entries", "tool cache entry capacity is exhausted")
                if current_bytes - existing_bytes + content_bytes > self.max_bytes:
                    raise ToolRagCacheError("cache_capacity_bytes", "tool cache byte capacity is exhausted")
                if existing is not None and (
                    str(existing["owner_scope"]) != owner_scope or str(existing["access_scope"]) != access_scope
                ):
                    raise ToolRagCacheError("cache_scope_conflict", "cache identity cannot change scope")
                if existing is not None and str(existing["content_digest"]) != content_digest:
                    raise ToolRagCacheError("cache_identity_conflict", "cache identity cannot replace existing content")
            finally:
                connection.close()

            metadata = {
                "tool_cache_id": cache_id,
                "request_id": request_id,
                "tool_name": tool_name,
                "citations": citations,
                "prompt_injection_suspected": injection_suspected,
                "untrusted_external_content": True,
            }
            try:
                ingested = self.store.ingest_document(
                    source_id=source_id,
                    relative_ref=f"tool-cache/{cache_id}.json",
                    sha256=content_digest,
                    mime="application/json",
                    title=title,
                    text=text,
                    revision=f"r_{content_digest[:32]}",
                    owner_scope=owner_scope,
                    access_scope=access_scope,
                    metadata=metadata,
                )
            except RagStoreError as exc:
                raise ToolRagCacheError(exc.code, str(exc)) from exc
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO rag_tool_cache(cache_id, source_id, request_id, tool_name, owner_scope, access_scope, title, items_json, citations_json, content_bytes, content_digest, prompt_injection_suspected, created_at, updated_at, expires_at, last_accessed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(cache_id) DO UPDATE SET source_id=excluded.source_id, request_id=excluded.request_id, tool_name=excluded.tool_name, title=excluded.title, items_json=excluded.items_json, citations_json=excluded.citations_json, content_bytes=excluded.content_bytes, content_digest=excluded.content_digest, prompt_injection_suspected=excluded.prompt_injection_suspected, updated_at=excluded.updated_at, expires_at=excluded.expires_at",
                    (cache_id, source_id, request_id, tool_name, owner_scope, access_scope, title, items_json, citations_json, content_bytes, content_digest, int(injection_suspected), float(existing["created_at"]) if existing is not None else now, now, expires_at, None),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                # A newly ingested source has no prior user asset to preserve.
                # Remove it when the ledger commit fails; existing identities
                # are content-locked above and therefore remain idempotent.
                if existing is None:
                    try:
                        self.store.delete_source(source_id)
                    except Exception:
                        pass
                raise
            finally:
                connection.close()
            result_row = self.get_cache(cache_id, access_scope=access_scope)
            result_row["ingest_status"] = ingested.status
            return result_row

    def get_cache(self, cache_id: str, *, access_scope: str = "owner") -> dict[str, Any]:
        self._validate_cache_id(cache_id)
        if access_scope not in _ACCESS_SCOPES:
            raise ToolRagCacheError("access_scope_invalid", "access scope is outside the local boundary")
        with self._lock:
            self.initialize()
            row_list = [row for row in self._rows(access_scope=access_scope) if str(row["cache_id"]) == cache_id]
            if not row_list:
                raise ToolRagCacheError("cache_not_found", "tool cache entry was not found")
            row = row_list[0]
            if float(row["expires_at"]) <= time.time():
                self._delete_row(cache_id, row)
                raise ToolRagCacheError("cache_expired", "tool cache entry has expired")
            connection = self._connect()
            try:
                connection.execute("UPDATE rag_tool_cache SET last_accessed_at=? WHERE cache_id=?", (time.time(), cache_id))
            finally:
                connection.close()
            return self._public_row(row)

    def list_cache(self, *, access_scope: str = "owner") -> list[dict[str, Any]]:
        if access_scope not in _ACCESS_SCOPES:
            raise ToolRagCacheError("access_scope_invalid", "access scope is outside the local boundary")
        self.purge_expired(limit=1_000)
        return [self._public_row(row) for row in self._rows(access_scope=access_scope)]

    def search(self, query: str, *, access_scope: str = "owner", limit: int = 20) -> list[dict[str, Any]]:
        if access_scope not in _ACCESS_SCOPES:
            raise ToolRagCacheError("access_scope_invalid", "access scope is outside the local boundary")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 100:
            raise ToolRagCacheError("cache_limit_invalid", "cache search limit is invalid")
        self.purge_expired(limit=1_000)
        try:
            rows = self.store.search(query, access_scope=access_scope, limit=limit)
        except RagStoreError as exc:
            raise ToolRagCacheError(exc.code, str(exc)) from exc
        caches = {str(row["source_id"]): row for row in self._rows(access_scope=access_scope)}
        output: list[dict[str, Any]] = []
        for row in rows:
            cache = caches.get(str(row.get("source_id")))
            if cache is None:
                continue
            item = self._public_row(cache)
            item.update({
                "chunk_id": str(row.get("chunk_id", "")),
                "revision": str(row.get("revision", "")),
                "snippet": str(row.get("text_content", ""))[:512],
                "lexical_rank": float(row.get("rank", 0.0) or 0.0),
            })
            output.append(item)
        return output

    def delete_cache(self, cache_id: str, *, access_scope: str = "owner") -> bool:
        self._validate_cache_id(cache_id)
        if access_scope not in _ACCESS_SCOPES:
            raise ToolRagCacheError("access_scope_invalid", "access scope is outside the local boundary")
        with self._lock:
            self.initialize()
            rows = [row for row in self._rows(access_scope=access_scope) if str(row["cache_id"]) == cache_id]
            if not rows:
                return False
            return self._delete_row(cache_id, rows[0])

    def capacity(self) -> dict[str, Any]:
        self.initialize()
        self.purge_expired(limit=1_000)
        connection = self._connect()
        try:
            count, bytes_used = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(content_bytes),0) FROM rag_tool_cache"
            ).fetchone()
        finally:
            connection.close()
        return {
            "schema": TOOL_CACHE_SCHEMA,
            "entries": int(count),
            "bytes": int(bytes_used),
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "remaining_entries": max(0, self.max_entries - int(count)),
            "remaining_bytes": max(0, self.max_bytes - int(bytes_used)),
            "default_ttl_seconds": self.default_ttl_seconds,
        }

    def rebuild(self) -> dict[str, Any]:
        purged = self.purge_expired(limit=1_000)
        try:
            fts_count = self.store.rebuild_fts()
        except RagStoreError as exc:
            raise ToolRagCacheError(exc.code, str(exc)) from exc
        return {"status": "ok", "fts_chunk_count": int(fts_count), "purged": purged}


__all__ = ["TOOL_CACHE_SCHEMA", "ToolRagCache", "ToolRagCacheError"]
