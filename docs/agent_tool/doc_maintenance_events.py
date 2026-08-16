#!/usr/bin/env python3
"""文档维护 Agent M3.1：本地事件库与当前快照索引。"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_meta (
  doc_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  status_line TEXT NOT NULL,
  updated_at TEXT,
  last_commit TEXT,
  last_commit_ts TEXT,
  sha256 TEXT NOT NULL,
  indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('scan', 'manual_edit', 'llm_suggestion', 'applied')),
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_events_doc_ts
  ON doc_events(doc_id, ts DESC, event_id DESC);
CREATE TABLE IF NOT EXISTS check_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  rules TEXT NOT NULL,
  findings TEXT NOT NULL,
  decisions TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  chunk_no INTEGER NOT NULL,
  text TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  indexed_at TEXT NOT NULL,
  UNIQUE(doc_id, chunk_no)
);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_sha
  ON doc_chunks(doc_id, sha256);
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
  doc_id UNINDEXED,
  text
);
CREATE TABLE IF NOT EXISTS doc_embeddings (
  chunk_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  sha256 TEXT NOT NULL,
  embedded_at TEXT NOT NULL
);
"""
GIT_EVENT_MARKER = "@@DOCAGENT_EVENT@@"
CHUNK_MAX_CHARS = 1800


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def _chunk_markdown(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """按段落优先的确定性分块，过长段落才按字符切分。"""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[index:index + max_chars]
                  for index in range(0, len(paragraph), max_chars)] or [""]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _fts_query(query: str) -> str | None:
    """将用户查询收敛为纯 token AND 查询，避免 FTS 运算符注入与语法失败。"""
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
    if not terms:
        return None
    return " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:12] if term)


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(vector), *vector)


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    if len(blob) != 4 * dim:
        raise ValueError("embedding blob dimension mismatch")
    return list(struct.unpack("<" + "f" * dim, blob))


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _last_commit(repo_root: Path, doc_id: str) -> tuple[str | None, str | None]:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H%x09%cI", "--", doc_id],
        cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return None, None
    fields = output.split("\t", 1)
    return (fields[0], fields[1] if len(fields) == 2 else None)


class DocEventStore:
    """派生索引；可删除重建，git 始终是事实源。"""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DocEventStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def index_snapshot(self, audit: dict, repo_root: Path) -> dict:
        """原子写入当前 doc_meta，并追加本次 scan/LLM 事件与 check_run。"""
        ts = str(audit.get("run_ts") or _now())
        findings_summary = [
            {"doc": record.get("doc"), "findings": record.get("findings") or []}
            for record in audit.get("docs", ()) if record.get("findings")
        ]
        llm_by_doc = {
            str(item.get("doc")): item
            for item in (audit.get("llm") or {}).get("judgements", ())
        }
        scan_events = 0
        llm_events = 0
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO check_runs (ts, rules, findings, decisions) VALUES (?, ?, ?, ?)",
                (ts, _json(audit.get("rules") or {}), _json(findings_summary), _json({})),
            )
            run_id = int(cursor.lastrowid)
            for record in audit.get("docs", ()):
                doc_id = str(record.get("doc", ""))
                path = (repo_root / doc_id).resolve()
                docs_root = (repo_root / "docs").resolve()
                if path != docs_root and docs_root not in path.parents:
                    raise ValueError(f"audit document escapes docs root: {doc_id}")
                text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
                commit, commit_ts = _last_commit(repo_root, doc_id)
                self.conn.execute(
                    """
                    INSERT INTO doc_meta (
                      doc_id, title, status_line, updated_at, last_commit,
                      last_commit_ts, sha256, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                      title=excluded.title, status_line=excluded.status_line,
                      updated_at=excluded.updated_at, last_commit=excluded.last_commit,
                      last_commit_ts=excluded.last_commit_ts, sha256=excluded.sha256,
                      indexed_at=excluded.indexed_at
                    """,
                    (
                        doc_id, _title(text), str(record.get("status_line") or ""),
                        record.get("updated_at"), commit, commit_ts,
                        str(record.get("sha256") or ""), ts,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO doc_events (doc_id, ts, kind, payload) VALUES (?, ?, 'scan', ?)",
                    (doc_id, ts, _json({
                        "run_id": run_id,
                        "rules": [finding.get("rule") for finding in record.get("findings") or []],
                        "finding_count": len(record.get("findings") or []),
                        "sha256": record.get("sha256"),
                    })),
                )
                scan_events += 1
                llm_item = llm_by_doc.get(doc_id)
                if llm_item is not None:
                    self.conn.execute(
                        """
                        INSERT INTO doc_events (doc_id, ts, kind, payload)
                        VALUES (?, ?, 'llm_suggestion', ?)
                        """,
                        (doc_id, ts, _json({
                            "run_id": run_id,
                            "judgement": llm_item.get("judgement"),
                            "confidence": llm_item.get("confidence"),
                            "suggestion": llm_item.get("suggestion"),
                            "source": llm_item.get("source"),
                            "provider": llm_item.get("provider"),
                            "model": llm_item.get("model"),
                        })),
                    )
                    llm_events += 1
        return {
            "run_id": run_id,
            "docs_indexed": scan_events,
            "scan_events": scan_events,
            "llm_events": llm_events,
            "database": self.path.relative_to(repo_root).as_posix()
            if self.path.is_relative_to(repo_root) else self.path.name,
        }

    def replay_git_history(self, repo_root: Path) -> dict:
        """按提交时间正序把 docs/*.md 变更重放为 manual_edit 派生事件。"""
        result = subprocess.run(
            [
                "git", "-c", "core.quotepath=false", "log", "--reverse",
                "--date=iso-strict",
                f"--format={GIT_EVENT_MARKER}%H%x09%cI%x09%s",
                "--name-status", "--find-renames", "--", "docs/",
            ],
            cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError("git history replay failed")
        commit: dict | None = None
        commits: set[str] = set()
        events = 0
        with self.conn:
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(GIT_EVENT_MARKER):
                    fields = line[len(GIT_EVENT_MARKER):].split("\t", 2)
                    if len(fields) != 3:
                        commit = None
                        continue
                    commit = {"commit": fields[0], "ts": fields[1], "subject": fields[2]}
                    commits.add(fields[0])
                    continue
                if commit is None:
                    continue
                fields = raw_line.split("\t")
                if len(fields) < 2:
                    continue
                status = fields[0]
                paths = fields[1:]
                doc_id = paths[-1].replace("\\", "/")
                if not doc_id.startswith("docs/") or not doc_id.endswith(".md"):
                    continue
                payload = {
                    "commit": commit["commit"],
                    "subject": commit["subject"],
                    "status": status,
                }
                if status.startswith(("R", "C")) and len(paths) >= 2:
                    payload["previous_doc_id"] = paths[-2].replace("\\", "/")
                self.conn.execute(
                    """
                    INSERT INTO doc_events (doc_id, ts, kind, payload)
                    VALUES (?, ?, 'manual_edit', ?)
                    """,
                    (doc_id, commit["ts"], _json(payload)),
                )
                events += 1
        return {"commits_replayed": len(commits), "git_events": events}

    def index_chunks(self, audit: dict, repo_root: Path) -> dict:
        """写入当前文档分块；内容/sha 未变化的文档不重复建索引。"""
        ts = str(audit.get("run_ts") or _now())
        indexed = 0
        skipped = 0
        chunk_count = 0
        docs_root = (repo_root / "docs").resolve()
        with self.conn:
            for record in audit.get("docs", ()):
                doc_id = str(record.get("doc", ""))
                path = (repo_root / doc_id).resolve()
                if path != docs_root and docs_root not in path.parents:
                    raise ValueError(f"audit document escapes docs root: {doc_id}")
                if not path.is_file():
                    continue
                sha256 = str(record.get("sha256") or "")
                existing = self.conn.execute(
                    "SELECT DISTINCT sha256 FROM doc_chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
                if existing and len(existing) == 1 and existing[0][0] == sha256:
                    skipped += 1
                    continue
                old_ids = [row[0] for row in self.conn.execute(
                    "SELECT chunk_id FROM doc_chunks WHERE doc_id = ?", (doc_id,)
                )]
                if old_ids:
                    placeholders = ",".join("?" for _ in old_ids)
                    self.conn.execute(
                        f"DELETE FROM doc_chunks_fts WHERE rowid IN ({placeholders})", old_ids
                    )
                    self.conn.execute(
                        f"DELETE FROM doc_embeddings WHERE chunk_id IN ({placeholders})", old_ids
                    )
                self.conn.execute("DELETE FROM doc_chunks WHERE doc_id = ?", (doc_id,))
                chunks = _chunk_markdown(path.read_text(encoding="utf-8", errors="replace"))
                for chunk_no, chunk_text in enumerate(chunks):
                    cursor = self.conn.execute(
                        """
                        INSERT INTO doc_chunks (doc_id, chunk_no, text, sha256, indexed_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (doc_id, chunk_no, chunk_text, sha256, ts),
                    )
                    self.conn.execute(
                        "INSERT INTO doc_chunks_fts (rowid, doc_id, text) VALUES (?, ?, ?)",
                        (int(cursor.lastrowid), doc_id, chunk_text),
                    )
                indexed += 1
                chunk_count += len(chunks)
        return {"documents_indexed": indexed, "documents_unchanged": skipped, "chunks": chunk_count}

    def search_chunks(self, query: str, limit: int = 5) -> list[dict]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        match = _fts_query(query)
        if match is None:
            return []
        rows = self.conn.execute(
            """
            SELECT chunks.doc_id, chunks.chunk_no,
                   snippet(doc_chunks_fts, 1, '[', ']', '...', 18) AS snippet,
                   bm25(doc_chunks_fts) AS score
            FROM doc_chunks_fts
            JOIN doc_chunks AS chunks ON chunks.chunk_id = doc_chunks_fts.rowid
            WHERE doc_chunks_fts MATCH ?
            ORDER BY score ASC, chunks.doc_id ASC, chunks.chunk_no ASC
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def index_embeddings(self, audit: dict, provider, repo_root: Path,
                         batch_size: int = 32) -> dict:
        """按 chunks 的 sha/model 增量生成向量；provider 失败时事务整体回滚。"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        doc_ids = [str(record.get("doc", "")) for record in audit.get("docs", ())]
        if not doc_ids:
            return {"chunks_embedded": 0, "chunks_unchanged": 0, "provider_batches": 0, "dim": 0}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, doc_id, text, sha256 FROM doc_chunks
            WHERE doc_id IN ({placeholders}) ORDER BY chunk_id
            """,
            doc_ids,
        ).fetchall()
        todo = []
        unchanged = 0
        for row in rows:
            existing = self.conn.execute(
                "SELECT model, sha256 FROM doc_embeddings WHERE chunk_id = ?", (row["chunk_id"],)
            ).fetchone()
            if existing is not None and existing["model"] == provider.model and existing["sha256"] == row["sha256"]:
                unchanged += 1
            else:
                todo.append(row)
        embedded = 0
        batches = 0
        dimension = 0
        with self.conn:
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                vectors = provider.embed([row["text"] for row in batch])
                if len(vectors) != len(batch) or any(not vector for vector in vectors):
                    raise ValueError("embedding provider returned wrong batch size")
                batch_dim = len(vectors[0])
                if batch_dim == 0 or any(len(vector) != batch_dim for vector in vectors):
                    raise ValueError("embedding provider returned inconsistent dimensions")
                dimension = dimension or batch_dim
                if dimension != batch_dim:
                    raise ValueError("embedding dimensions changed within run")
                for row, vector in zip(batch, vectors):
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO doc_embeddings
                          (chunk_id, model, dim, vector, sha256, embedded_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (row["chunk_id"], provider.model, batch_dim, _pack_vector(vector),
                         row["sha256"], _now()),
                    )
                    embedded += 1
                batches += 1
        return {
            "chunks_embedded": embedded,
            "chunks_unchanged": unchanged,
            "provider_batches": batches,
            "dim": dimension,
        }

    def semantic_search(self, query: str, provider, limit: int = 5) -> list[dict]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_vectors = provider.embed([query])
        if len(query_vectors) != 1 or not query_vectors[0]:
            raise ValueError("embedding provider returned invalid query vector")
        query_vector = query_vectors[0]
        rows = self.conn.execute(
            """
            SELECT embeddings.chunk_id, chunks.doc_id, chunks.chunk_no, chunks.text,
                   embeddings.dim, embeddings.vector
            FROM doc_embeddings AS embeddings
            JOIN doc_chunks AS chunks ON chunks.chunk_id = embeddings.chunk_id
            WHERE embeddings.model = ?
            """,
            (provider.model,),
        ).fetchall()
        ranked = []
        for row in rows:
            vector = _unpack_vector(row["vector"], row["dim"])
            if len(vector) != len(query_vector):
                continue
            ranked.append({
                "doc_id": row["doc_id"], "chunk_no": row["chunk_no"],
                "score": _cosine(query_vector, vector),
                "snippet": row["text"][:300],
            })
        ranked.sort(key=lambda item: (-item["score"], item["doc_id"], item["chunk_no"]))
        return ranked[:limit]

    def record_decisions(self, run_id: int, decisions: dict) -> None:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE check_runs SET decisions = ? WHERE run_id = ?",
                (_json(decisions), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown check run: {run_id}")

    def get_doc_meta(self, doc_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM doc_meta WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def recent_events(self, doc_id: str, limit: int = 10) -> list[dict]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = self.conn.execute(
            """
            SELECT event_id, doc_id, ts, kind, payload FROM doc_events
            WHERE doc_id = ? ORDER BY event_id DESC LIMIT ?
            """,
            (doc_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def rebuild_event_database(path: Path, audit: dict, repo_root: Path) -> dict:
    """备份旧库，在临时库完成 git 重放与当前快照后原子替换。"""
    repo_root = repo_root.resolve()
    build_root = (repo_root / "build").resolve()
    target = path.resolve()
    if target != build_root and build_root not in target.parents:
        raise ValueError("event database must stay under build/")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if target.is_file():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = target.with_name(f"{target.stem}.bak-{stamp}{target.suffix}")
        shutil.copy2(target, backup_path)

    with tempfile.TemporaryDirectory(prefix="docagent-rebuild-", dir=target.parent) as temp_dir:
        temp_path = Path(temp_dir) / target.name
        with DocEventStore(temp_path) as store:
            replay = store.replay_git_history(repo_root)
            snapshot = store.index_snapshot(audit, repo_root)
            chunks = store.index_chunks(audit, repo_root)
        os.replace(temp_path, target)
    return {
        **replay,
        **snapshot,
        "chunks": chunks,
        "database": target.relative_to(repo_root).as_posix(),
        "backup": backup_path.relative_to(repo_root).as_posix() if backup_path else None,
    }
