"""文档维护 Agent M2.2：SQLite L1/L2 缓存、人工结论与成本账本。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_cache import JudgementCache  # noqa: E402
from doc_maintenance_llm import Judgement, prepare_judgement_batches  # noqa: E402


def _batch(tmp_path: Path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("> 状态：规划\n", encoding="utf-8")
    audit = {"docs": [{
        "doc": "docs/x.md",
        "status_line": "> 状态：规划",
        "sha256": "a" * 64,
        "findings": [{"rule": "R1", "level": "warn"}],
        "related_commits": ["feat: x"],
    }]}
    return prepare_judgement_batches(audit, repo)[0]


def _judgement(batch, value="accurate"):
    return Judgement(batch.doc_ref, value, 0.9, "状态行准确")


def test_l1_exact_hit_and_sanitized_cache_input(tmp_path):
    batch = _batch(tmp_path)
    db = tmp_path / "cache.sqlite"
    with JudgementCache(db) as cache:
        cache.store(batch, _judgement(batch), source="llm", provider="ollama", model="test")
        hit = cache.lookup(batch)
        assert hit is not None and hit.tier == "l1"
        row = cache.conn.execute("SELECT input_excerpt FROM llm_judgements").fetchone()
        assert "docs/x.md" not in row["input_excerpt"]


def test_l2_state_hit_when_exact_input_changes(tmp_path):
    batch = _batch(tmp_path)
    changed_input = replace(batch, status_line="状态行格式已规范化")
    with JudgementCache(tmp_path / "cache.sqlite") as cache:
        cache.store(batch, _judgement(batch), source="llm", provider="ollama", model="test")
        hit = cache.lookup(changed_input)
        assert hit is not None and hit.tier == "l2"


def test_needs_review_reuses_l1_but_not_l2_state_cache(tmp_path):
    batch = _batch(tmp_path)
    changed_input = replace(batch, status_line="状态行格式已规范化")
    with JudgementCache(tmp_path / "cache.sqlite") as cache:
        cache.store(
            batch, _judgement(batch, "needs_review"), source="llm",
            provider="ollama", model="test",
        )
        exact = cache.lookup(batch)
        assert exact is not None and exact.tier == "l1"
        assert cache.lookup(changed_input) is None


def test_human_result_has_priority_over_llm_result(tmp_path):
    batch = _batch(tmp_path)
    changed_input = replace(batch, status_line="状态行格式已规范化")
    with JudgementCache(tmp_path / "cache.sqlite") as cache:
        cache.store(batch, _judgement(batch, "stale"), source="llm", provider="ollama", model="test")
        cache.store(
            changed_input, _judgement(changed_input, "accurate"), source="human",
            provider="human", model="manual",
        )
        hit = cache.lookup(batch)
        assert hit is not None
        assert hit.tier == "human"
        assert hit.judgement.judgement == "accurate"


def test_changed_document_hash_lazily_invalidates_cache(tmp_path):
    batch = _batch(tmp_path)
    with JudgementCache(tmp_path / "cache.sqlite") as cache:
        cache.store(batch, _judgement(batch), source="llm", provider="ollama", model="test")
        changed_document = replace(batch, doc_sha256="b" * 64)
        assert cache.lookup(changed_document) is None


def test_cost_log_and_invalid_input_validation(tmp_path):
    batch = _batch(tmp_path)
    with JudgementCache(tmp_path / "cache.sqlite") as cache:
        cache.log_cost(
            run_id="run-1", provider="deepseek", model="test",
            prompt_tokens=123, completion_tokens=45, hits=2, misses=1,
        )
        assert cache.cost_rows() == [{
            "run_id": "run-1", "provider": "deepseek", "model": "test",
            "prompt_tokens": 123, "completion_tokens": 45, "hits": 2, "misses": 1,
        }]
        with pytest.raises(ValueError):
            cache.log_cost(
                run_id="bad", provider="x", model="x", prompt_tokens=-1,
                completion_tokens=0, hits=0, misses=0,
            )
