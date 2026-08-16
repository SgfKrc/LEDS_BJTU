"""文档维护 Agent M3.4：向量存储、增量 embedding 与语义检索。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_embeddings import OllamaEmbeddingProvider  # noqa: E402
from doc_maintenance_events import DocEventStore  # noqa: E402


class FakeEmbedding:
    model = "fake-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[
            float("task" in text.lower()),
            float("auth" in text.lower()),
            float("scheduler" in text.lower()),
        ] for text in texts]


def _repo(tmp_path: Path):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text("# X\n\ntask graph scheduler\n", encoding="utf-8")
    (docs / "y.md").write_text("# Y\n\nauthentication notes\n", encoding="utf-8")
    audit = {"run_ts": "2026-08-17T10:00:00", "docs": [
        {"doc": "docs/x.md", "sha256": "a" * 64, "status_line": "", "findings": []},
        {"doc": "docs/y.md", "sha256": "b" * 64, "status_line": "", "findings": []},
    ]}
    return repo, audit


def test_embedding_index_and_semantic_search(tmp_path):
    repo, audit = _repo(tmp_path)
    provider = FakeEmbedding()
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        store.index_chunks(audit, repo)
        result = store.index_embeddings(audit, provider, repo)
        hits = store.semantic_search("task scheduler", provider, limit=2)
    assert result["chunks_embedded"] == 2
    assert result["dim"] == 3
    assert hits[0]["doc_id"] == "docs/x.md"
    assert hits[0]["score"] > hits[1]["score"]


def test_embedding_index_skips_same_sha_and_replaces_changed_chunks(tmp_path):
    repo, audit = _repo(tmp_path)
    provider = FakeEmbedding()
    with DocEventStore(repo / "build" / "events.sqlite") as store:
        store.index_chunks(audit, repo)
        first = store.index_embeddings(audit, provider, repo)
        second = store.index_embeddings(audit, provider, repo)
        assert first["chunks_embedded"] == 2
        assert second["chunks_embedded"] == 0
        assert second["chunks_unchanged"] == 2
        audit["docs"][0]["sha256"] = "c" * 64
        (repo / "docs" / "x.md").write_text("# X\n\nauth scheduler\n", encoding="utf-8")
        store.index_chunks(audit, repo)
        changed = store.index_embeddings(audit, provider, repo)
        assert changed["chunks_embedded"] == 1


def test_embedding_dimension_mismatch_fails(tmp_path):
    repo, audit = _repo(tmp_path)

    class Bad:
        model = "bad"
        def embed(self, texts):
            return [[1.0, 2.0] for _ in texts[:1]]

    with DocEventStore(repo / "build" / "events.sqlite") as store:
        store.index_chunks(audit, repo)
        with pytest.raises(ValueError, match="wrong batch size"):
            store.index_embeddings(audit, Bad(), repo)
        assert store.conn.execute("SELECT COUNT(*) FROM doc_embeddings").fetchone()[0] == 0


def test_ollama_provider_normalizes_v1_base_url_and_validates_timeout():
    provider = OllamaEmbeddingProvider("http://127.0.0.1:11434/v1", "fake", 5)
    assert provider.base_url.endswith("/v1")
    with pytest.raises(ValueError):
        OllamaEmbeddingProvider(timeout_seconds=61)
