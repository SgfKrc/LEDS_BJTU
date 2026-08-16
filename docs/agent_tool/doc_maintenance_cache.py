#!/usr/bin/env python3
"""文档维护 Agent M2.2：本地 SQLite 判定缓存与成本账本。"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from doc_maintenance_llm import Judgement, JudgementBatch, sanitize_text

VALID_SOURCES = {"llm", "human", "semantic"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_judgements (
  cache_key TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  doc_sha256 TEXT NOT NULL,
  related_commits TEXT NOT NULL,
  judgement TEXT NOT NULL CHECK (judgement IN ('stale', 'accurate', 'needs_review')),
  confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  suggestion TEXT NOT NULL,
  input_excerpt TEXT NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('llm', 'human', 'semantic')),
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
  completion_tokens INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
  judged_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_judgements_state
  ON llm_judgements(doc_id, rule_id, doc_sha256, related_commits, source, judged_at DESC);
CREATE TABLE IF NOT EXISTS cost_log (
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
  completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
  hits INTEGER NOT NULL CHECK (hits >= 0),
  misses INTEGER NOT NULL CHECK (misses >= 0)
);
"""


@dataclass(frozen=True)
class CacheHit:
    tier: str
    judgement: Judgement
    source: str
    provider: str
    model: str


def _rules_key(batch: JudgementBatch) -> str:
    return ",".join(sorted({str(finding["rule"]) for finding in batch.findings}))


def _commits_key(batch: JudgementBatch) -> str:
    return json.dumps(sorted(batch.related_commits), ensure_ascii=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JudgementCache:
    """M2 的 L1/L2 缓存；所有数据保留在用户工作区本地。"""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "JudgementCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _hit_from_row(self, row: sqlite3.Row, tier: str) -> CacheHit:
        return CacheHit(
            tier=tier,
            judgement=Judgement(
                doc_ref=row["doc_id"], judgement=row["judgement"],
                confidence=float(row["confidence"]), suggestion=row["suggestion"],
            ),
            source=row["source"], provider=row["provider"], model=row["model"],
        )

    def lookup(self, batch: JudgementBatch) -> CacheHit | None:
        """按 human → L1 精确 → L2 状态的优先级查找可复用判定。"""
        params = (batch.doc, _rules_key(batch), batch.doc_sha256, _commits_key(batch))
        # 人工结论是最高优先级，输入文本的格式变化不应使它失效。
        row = self.conn.execute(
            """
            SELECT * FROM llm_judgements
            WHERE doc_id = ? AND rule_id = ? AND doc_sha256 = ? AND related_commits = ?
              AND source = 'human'
            ORDER BY judged_at DESC LIMIT 1
            """,
            params,
        ).fetchone()
        if row is not None:
            return self._hit_from_row(row, "human")

        row = self.conn.execute(
            "SELECT * FROM llm_judgements WHERE cache_key = ?", (batch.cache_key,)
        ).fetchone()
        # L1 输入完全一致时，合法的 needs_review 也必须复用；否则低置信度
        # 样本会在每次扫描重复付费。仅 L2 状态复用排除 needs_review。
        if row is not None:
            return self._hit_from_row(row, "l1")

        row = self.conn.execute(
            """
            SELECT * FROM llm_judgements
            WHERE doc_id = ? AND rule_id = ? AND doc_sha256 = ? AND related_commits = ?
              AND source != 'human' AND judgement != 'needs_review'
            ORDER BY judged_at DESC LIMIT 1
            """,
            params,
        ).fetchone()
        return self._hit_from_row(row, "l2") if row is not None else None

    def store(self, batch: JudgementBatch, judgement: Judgement, *, source: str,
              provider: str, model: str, prompt_tokens: int = 0,
              completion_tokens: int = 0) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid cache source: {source}")
        if judgement.doc_ref != batch.doc_ref:
            raise ValueError("judgement doc_ref does not match batch")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.conn.execute(
            """
            INSERT OR REPLACE INTO llm_judgements (
              cache_key, doc_id, rule_id, doc_sha256, related_commits,
              judgement, confidence, suggestion, input_excerpt, source,
              provider, model, prompt_tokens, completion_tokens, judged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch.cache_key, batch.doc, _rules_key(batch), batch.doc_sha256,
                _commits_key(batch), judgement.judgement, judgement.confidence,
                sanitize_text(judgement.suggestion, 1024), batch.excerpt, source,
                sanitize_text(provider, 128), sanitize_text(model, 256),
                prompt_tokens, completion_tokens, _now(),
            ),
        )
        self.conn.commit()

    def log_cost(self, *, run_id: str, provider: str, model: str,
                 prompt_tokens: int, completion_tokens: int, hits: int,
                 misses: int) -> None:
        if min(prompt_tokens, completion_tokens, hits, misses) < 0:
            raise ValueError("cost values must be non-negative")
        self.conn.execute(
            """
            INSERT INTO cost_log (
              run_id, ts, provider, model, prompt_tokens, completion_tokens, hits, misses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, _now(), sanitize_text(provider, 128), sanitize_text(model, 256),
                prompt_tokens, completion_tokens, hits, misses,
            ),
        )
        self.conn.commit()

    def cost_rows(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT run_id, provider, model, prompt_tokens, completion_tokens, hits, misses
            FROM cost_log ORDER BY rowid
            """
        ).fetchall()
        return [dict(row) for row in rows]
