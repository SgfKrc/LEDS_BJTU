"""SQLite WAL + FTS5 retrieval store for user-owned harness data."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .chunking import CHUNK_STRATEGIES, chunk_text


RAG_METADATA_FIELDS = frozenset({"source", "scope", "type", "tag", "time"})
INDEX_GRANULARITIES = frozenset({"document", "paragraph", "sentence", *CHUNK_STRATEGIES})
_INDEX_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{1,63}|[\u3400-\u9fff]{2,32}")
_INDEX_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9_.:-]{2,63}\b|[\u3400-\u9fff]{2,16}")
_INDEX_STOPWORDS = frozenset({"the", "and", "or", "for", "with", "from", "this", "that", "uses", "use", "depends", "on"})
_MAX_METADATA_FILTERS = 5
_MAX_METADATA_VALUES = 20
_MAX_METADATA_TEXT = 256


def _index_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _INDEX_TOKEN.finditer(normalized):
        token = match.group(0)
        if token in _INDEX_STOPWORDS or len(token) < 2:
            continue
        if re.fullmatch(r"[\u3400-\u9fff]{2,32}", token) and len(token) > 2:
            tokens.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _index_phrases(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    words = [item for item in re.findall(r"[A-Za-z][A-Za-z0-9_.:-]{1,63}|[\u3400-\u9fff]{2,16}", normalized) if item not in _INDEX_STOPWORDS]
    phrases = [
        " ".join(words[index:index + size])
        for size in range(2, min(4, len(words)) + 1)
        for index in range(len(words) - size + 1)
    ]
    return tuple(dict.fromkeys(phrases))


def _index_entities(text: str) -> tuple[str, ...]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in _INDEX_ENTITY.finditer(unicodedata.normalize("NFKC", text)):
        name = match.group(0).strip()
        normalized = name.casefold()
        if len(name) >= 2 and normalized not in _INDEX_STOPWORDS and normalized not in seen:
            seen.add(normalized)
            entities.append(name)
    return tuple(entities[:64])


def _index_relations(text: str, entities: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    if len(entities) < 2:
        return ()
    pattern = re.compile(
        r"(?P<left>[A-Za-z][A-Za-z0-9_.:-]{2,63}|[\u3400-\u9fff]{2,16})\s*"
        r"(?P<relation>uses?|depends?\s+on|->|浣跨敤|渚濊禆|鍏宠仈|璋冪敤)\s*"
        r"(?P<right>[A-Za-z][A-Za-z0-9_.:-]{2,63}|[\u3400-\u9fff]{2,16})",
        re.IGNORECASE,
    )
    known = {item.casefold(): item for item in entities}
    relations: list[tuple[str, str, str]] = []
    for match in pattern.finditer(text):
        left = known.get(match.group("left").casefold())
        right = known.get(match.group("right").casefold())
        if left and right and left.casefold() != right.casefold():
            relations.append((left, right, match.group("relation").casefold()))
    if not relations:
        relations.extend((entities[index], entities[index + 1], "cooccurs") for index in range(len(entities) - 1))
    return tuple(dict.fromkeys(relations))


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
        connection.execute("PRAGMA foreign_keys=ON")
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
                CREATE TABLE IF NOT EXISTS rag_index_chunks (
                    index_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES rag_sources(source_id) ON DELETE CASCADE,
                    base_chunk_id TEXT REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE,
                    granularity TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(source_id, granularity, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_index_chunks_lookup
                    ON rag_index_chunks(source_id, granularity, ordinal);
                CREATE TABLE IF NOT EXISTS rag_keyword_index (
                    index_id TEXT NOT NULL REFERENCES rag_index_chunks(index_id) ON DELETE CASCADE,
                    term TEXT NOT NULL,
                    term_kind TEXT NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(index_id, term, term_kind, position)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_keyword_term
                    ON rag_keyword_index(term, term_kind, index_id);
                CREATE TABLE IF NOT EXISTS rag_entities (
                    entity_id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL REFERENCES rag_index_chunks(index_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL DEFAULT 'rule',
                    UNIQUE(index_id, normalized_name)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_entities_name
                    ON rag_entities(normalized_name, index_id);
                CREATE TABLE IF NOT EXISTS rag_relations (
                    relation_id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL REFERENCES rag_index_chunks(index_id) ON DELETE CASCADE,
                    source_entity_id TEXT NOT NULL REFERENCES rag_entities(entity_id) ON DELETE CASCADE,
                    target_entity_id TEXT NOT NULL REFERENCES rag_entities(entity_id) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    UNIQUE(index_id, source_entity_id, target_entity_id, relation)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_relations_source
                    ON rag_relations(source_entity_id, target_entity_id);
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
            index_count = int(connection.execute("SELECT COUNT(*) FROM rag_index_chunks").fetchone()[0])
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
            if chunk_count and not index_count:
                for row in connection.execute("SELECT chunk_id, source_id, ordinal, start_offset, end_offset, text, granularity FROM rag_chunks"):
                    self._populate_index_row(
                        connection, source_id=str(row[1]), base_chunk_id=str(row[0]),
                        granularity=str(row[6] or "fixed"), ordinal=int(row[2]),
                        start_offset=int(row[3]), end_offset=int(row[4]), text=str(row[5]),
                    )

    @staticmethod
    def _index_id(source_id: str, granularity: str, ordinal: int, text: str) -> str:
        return hashlib.sha256(f"idx\0{source_id}\0{granularity}\0{ordinal}\0{hashlib.sha256(text.encode()).hexdigest()}".encode()).hexdigest()

    @classmethod
    def _populate_index_row(
        cls, connection: sqlite3.Connection, *, source_id: str, base_chunk_id: str | None,
        granularity: str, ordinal: int, start_offset: int, end_offset: int, text: str,
    ) -> str:
        index_id = cls._index_id(source_id, granularity, ordinal, text)
        connection.execute(
            "INSERT OR REPLACE INTO rag_index_chunks(index_id, source_id, base_chunk_id, granularity, ordinal, start_offset, end_offset, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (index_id, source_id, base_chunk_id, granularity, ordinal, start_offset, end_offset, text),
        )
        connection.execute("DELETE FROM rag_keyword_index WHERE index_id=?", (index_id,))
        terms = _index_tokens(text)
        phrases = _index_phrases(text)
        entries = [(index_id, term, "token", position) for position, term in enumerate(terms)]
        entries.extend(
            (index_id, term[:size], "prefix", position)
            for position, term in enumerate(terms)
            for size in range(2, min(32, len(term)) + 1)
        )
        entries.extend((index_id, phrase, "phrase", position) for position, phrase in enumerate(phrases))
        connection.executemany(
            "INSERT OR IGNORE INTO rag_keyword_index(index_id, term, term_kind, position) VALUES (?, ?, ?, ?)", entries,
        )
        connection.execute("DELETE FROM rag_relations WHERE index_id=?", (index_id,))
        connection.execute("DELETE FROM rag_entities WHERE index_id=?", (index_id,))
        entities = _index_entities(text)
        entity_ids: dict[str, str] = {}
        for name in entities:
            normalized = name.casefold()
            entity_id = hashlib.sha256(f"ent\0{index_id}\0{normalized}".encode()).hexdigest()
            entity_ids[normalized] = entity_id
            connection.execute(
                "INSERT OR IGNORE INTO rag_entities(entity_id, index_id, name, normalized_name, entity_type) VALUES (?, ?, ?, ?, 'rule')",
                (entity_id, index_id, name, normalized),
            )
        for left, right, relation in _index_relations(text, entities):
            left_id, right_id = entity_ids.get(left.casefold()), entity_ids.get(right.casefold())
            if left_id and right_id:
                relation_id = hashlib.sha256(f"rel\0{index_id}\0{left_id}\0{right_id}\0{relation}".encode()).hexdigest()
                connection.execute(
                    "INSERT OR IGNORE INTO rag_relations(relation_id, index_id, source_entity_id, target_entity_id, relation) VALUES (?, ?, ?, ?, ?)",
                    (relation_id, index_id, left_id, right_id, relation),
                )
        return index_id

    def _index_document_views(self, connection: sqlite3.Connection, *, source_id: str, text: str, chunks: list[Any], chunk_ids: list[str], max_chars: int, overlap_chars: int) -> None:
        for ordinal, chunk in enumerate(chunks):
            self._populate_index_row(
                connection, source_id=source_id, base_chunk_id=chunk_ids[ordinal], granularity=chunk.granularity,
                ordinal=ordinal, start_offset=chunk.start_offset, end_offset=chunk.end_offset, text=chunk.text,
            )
        self._populate_index_row(connection, source_id=source_id, base_chunk_id=None, granularity="document", ordinal=0, start_offset=0, end_offset=len(text), text=text)
        for granularity in ("paragraph", "sentence"):
            views = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars, strategy=granularity)
            for ordinal, chunk in enumerate(views):
                self._populate_index_row(
                    connection, source_id=source_id, base_chunk_id=None, granularity=granularity,
                    ordinal=ordinal, start_offset=chunk.start_offset, end_offset=chunk.end_offset, text=chunk.text,
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
        strategy: str = "fixed",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_scope = _scope(owner_scope)
        source_ref = _relative_ref(source_ref)
        title = _title(title)
        if strategy not in CHUNK_STRATEGIES:
            raise ValueError("unsupported chunk strategy")
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
            connection.execute("DELETE FROM rag_index_chunks WHERE source_id = ?", (source_id,))
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
            inserted_chunk_ids: list[str] = []
            for chunk in chunks:
                chunk_id = f"chk_{hashlib.sha256((source_id + "\0" + str(chunk.ordinal) + "\0" + chunk.text).encode()).hexdigest()[:24]}"
                inserted_chunk_ids.append(chunk_id)
                connection.execute(
                    "INSERT INTO rag_chunks(chunk_id, source_id, owner_scope, ordinal, start_offset, end_offset, text, granularity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, chunk.ordinal, chunk.start_offset, chunk.end_offset, chunk.text, chunk.granularity),
                )
                connection.execute(
                    "INSERT INTO rag_chunks_fts(chunk_id, source_id, owner_scope, title, text) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, source_id, owner_scope, title, chunk.text),
                )
            self._index_document_views(
                connection, source_id=source_id, text=text, chunks=chunks,
                chunk_ids=inserted_chunk_ids, max_chars=max_chars, overlap_chars=overlap_chars,
            )
        return {"source_id": source_id, "owner_scope": owner_scope, "title": title, "metadata": metadata_value, "strategy": strategy, "chunk_count": len(chunks), "content_sha256": digest}

    def list_index_chunks(self, *, owner_scope: str = "local", granularity: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        owner_scope = _scope(owner_scope)
        if granularity is not None and granularity not in INDEX_GRANULARITIES:
            raise ValueError("index granularity is unsupported")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 10_000:
            raise ValueError("index limit must be between 1 and 10000")
        clauses = ["s.owner_scope=?"]
        params: list[Any] = [owner_scope]
        if granularity:
            clauses.append("i.granularity=?")
            params.append(granularity)
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT i.index_id, i.base_chunk_id, i.source_id, i.granularity, i.ordinal, i.start_offset, i.end_offset, i.text "
                "FROM rag_index_chunks i JOIN rag_sources s ON s.source_id=i.source_id WHERE " + " AND ".join(clauses) +
                " ORDER BY i.source_id, i.granularity, i.ordinal LIMIT ?", params,
            ).fetchall()
        return [dict(row) for row in rows]

    def keyword_search(
        self, query: str, *, owner_scope: str = "local", limit: int = 20, granularity: str | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        owner_scope = _scope(owner_scope)
        if not isinstance(query, str) or not query.strip() or len(query) > 512 or "\x00" in query:
            raise ValueError("query must be non-empty, bounded, and NUL-free")
        if isinstance(limit, bool) or not 1 <= int(limit) <= 100:
            raise ValueError("query limit must be between 1 and 100")
        if granularity is not None and granularity not in INDEX_GRANULARITIES:
            raise ValueError("index granularity is unsupported")
        filters = _normalize_metadata_filters(metadata_filters)
        terms, phrases = _index_tokens(query), _index_phrases(query)
        if not terms and not phrases:
            return []
        clauses = ["s.owner_scope=?"]
        params: list[Any] = [owner_scope]
        route_clauses: list[str] = []
        if terms:
            route_clauses.append("(k.term_kind='token' AND k.term IN (" + ",".join("?" for _ in terms) + "))")
            params.extend(terms)
            route_clauses.append("(k.term_kind='prefix' AND k.term IN (" + ",".join("?" for _ in terms) + "))")
            params.extend(term[:32] for term in terms)
            route_clauses.append("(k.term_kind='token' AND (" + " OR ".join("k.term LIKE ? ESCAPE '\\'" for _ in terms) + "))")
            params.extend(term.replace("%", "\\%").replace("_", "\\_") + "%" for term in terms)
        if phrases:
            route_clauses.append("(k.term_kind='phrase' AND k.term IN (" + ",".join("?" for _ in phrases) + "))")
            params.extend(phrases)
        clauses.append("(" + " OR ".join(route_clauses) + ")")
        if granularity:
            clauses.append("i.granularity=?")
            params.append(granularity)
        for field, values in filters:
            placeholders = ",".join("?" for _ in values)
            clauses.append(
                "EXISTS (SELECT 1 FROM rag_metadata_index mi WHERE mi.source_id=i.source_id AND mi.owner_scope=s.owner_scope "
                "AND mi.field=? AND mi.value IN (" + placeholders + "))"
            )
            params.extend((field, *values))
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT i.index_id AS chunk_id, i.base_chunk_id, i.source_id, s.title, i.granularity, i.ordinal, "
                "i.start_offset, i.end_offset, i.text, SUM(CASE WHEN k.term_kind='phrase' THEN 2.0 ELSE 1.0 END) AS keyword_score "
                "FROM rag_keyword_index k JOIN rag_index_chunks i ON i.index_id=k.index_id JOIN rag_sources s ON s.source_id=i.source_id "
                "WHERE " + " AND ".join(clauses) + " GROUP BY i.index_id ORDER BY keyword_score DESC, i.ordinal ASC, i.index_id ASC LIMIT ?", params,
            ).fetchall()
        return [dict(row) for row in rows]

    def graph_search(self, query: str, *, owner_scope: str = "local", limit: int = 20, granularity: str | None = None, depth: int = 1) -> list[dict[str, Any]]:
        if isinstance(depth, bool) or not 0 <= int(depth) <= 2:
            raise ValueError("graph depth must be between 0 and 2")
        results = self.keyword_search(query, owner_scope=owner_scope, limit=limit, granularity=granularity)
        names = {item.casefold() for item in _index_entities(query)}
        if not names or depth == 0:
            for row in results:
                row["graph_score"], row["graph_entities"] = 0.0, []
            return results
        with self._connect() as connection:
            matched = connection.execute(
                "SELECT entity_id FROM rag_entities WHERE normalized_name IN (" + ",".join("?" for _ in names) + ")", tuple(names)
            ).fetchall()
            frontier = {str(row[0]) for row in matched}
            reached = set(frontier)
            for _ in range(int(depth)):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                adjacent = connection.execute(
                    "SELECT target_entity_id FROM rag_relations WHERE source_entity_id IN (" + placeholders + ") UNION SELECT source_entity_id FROM rag_relations WHERE target_entity_id IN (" + placeholders + ")",
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                frontier = {str(row[0]) for row in adjacent} - reached
                reached.update(frontier)
            related = []
            if reached:
                placeholders = ",".join("?" for _ in reached)
                related = connection.execute(
                    "SELECT DISTINCT i.index_id, e.name, i.base_chunk_id, i.source_id, s.title, i.ordinal, i.granularity, i.text "
                    "FROM rag_entities e JOIN rag_index_chunks i ON i.index_id=e.index_id JOIN rag_sources s ON s.source_id=i.source_id "
                    "WHERE e.entity_id IN (" + placeholders + ") AND s.owner_scope=?", tuple(reached) + (owner_scope,),
                ).fetchall()
        by_id: dict[str, list[str]] = {}
        for row in related:
            by_id.setdefault(str(row[0]), []).append(str(row[1]))
        for row in results:
            matched_names = by_id.get(str(row["chunk_id"]), [])
            row["graph_score"] = 0.25 * len(matched_names)
            row["graph_entities"] = matched_names
        seen = {str(row["chunk_id"]) for row in results}
        for row in related:
            index_id = str(row[0])
            if index_id in seen:
                continue
            seen.add(index_id)
            results.append({
                "chunk_id": index_id, "base_chunk_id": row[2], "source_id": row[3], "title": row[4],
                "ordinal": row[5], "granularity": row[6] or "fixed", "text": row[7],
                "keyword_score": 0.0, "graph_score": 0.25, "graph_entities": by_id.get(index_id, []),
            })
        results.sort(key=lambda row: (-float(row.get("graph_score", 0.0)), -float(row.get("keyword_score", 0.0)), str(row["chunk_id"])))
        return results[:int(limit)]

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
            index_chunks = int(connection.execute("SELECT COUNT(*) FROM rag_index_chunks").fetchone()[0])
            keyword_terms = int(connection.execute("SELECT COUNT(*) FROM rag_keyword_index").fetchone()[0])
            entities = int(connection.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0])
            relations = int(connection.execute("SELECT COUNT(*) FROM rag_relations").fetchone()[0])
        return {
            "backend": "sqlite_fts5", "sources": sources, "chunks": chunks,
            "index_chunks": index_chunks, "keyword_terms": keyword_terms,
            "entities": entities, "relations": relations, "embeddings": "optional_not_configured",
        }

    def index_health(self) -> dict[str, Any]:
        """Return counts for model-free materialized index layers."""
        with self._connect() as connection:
            granularities = {
                str(row[0]): int(row[1])
                for row in connection.execute("SELECT granularity, COUNT(*) FROM rag_index_chunks GROUP BY granularity")
            }
            return {
                "schema": "qlh.rag_index.v1", "model_free": True, "granularities": granularities,
                "index_chunks": int(connection.execute("SELECT COUNT(*) FROM rag_index_chunks").fetchone()[0]),
                "keyword_terms": int(connection.execute("SELECT COUNT(*) FROM rag_keyword_index").fetchone()[0]),
                "entities": int(connection.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0]),
                "relations": int(connection.execute("SELECT COUNT(*) FROM rag_relations").fetchone()[0]),
            }

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


__all__ = ["INDEX_GRANULARITIES", "RAG_METADATA_FIELDS", "RagHit", "RagStore"]
