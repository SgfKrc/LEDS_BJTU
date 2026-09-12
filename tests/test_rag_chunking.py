from __future__ import annotations

import sqlite3
import unicodedata

import pytest

from harness_workbench.rag import CHUNK_STRATEGIES, INDEX_GRANULARITIES, RagStore as HarnessRagStore, chunk_text
from harness_workbench.tools import (
    RAG_CHUNK_COMPARISON_SCHEMA,
    RAG_QUERY_COMPARISON_SCHEMA,
    run_rag_chunk_comparison,
    run_rag_query_comparison,
)
from src.rag_store import CHUNK_STRATEGIES as MAIN_CHUNK_STRATEGIES
from src.rag_store import RagStore, RagStoreError


def _fixture_text() -> str:
    sections = []
    for index in range(8):
        sections.append(
            f"# Section {index}\n\n"
            + ("This sentence carries a stable retrieval marker and enough context. " * 10)
            + "\n\n"
        )
    return "".join(sections)


def test_chunk_strategies_are_explicit_and_deterministic():
    text = _fixture_text()
    assert CHUNK_STRATEGIES == MAIN_CHUNK_STRATEGIES
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()
    for strategy in sorted(CHUNK_STRATEGIES):
        first = chunk_text(text, max_chars=128, overlap_chars=16, strategy=strategy)
        second = chunk_text(text, max_chars=128, overlap_chars=16, strategy=strategy)
        assert first == second
        assert len(first) > 1
        assert all(item.granularity == strategy for item in first)
        assert all(len(item.text) <= 128 for item in first)
        assert all(normalized[item.start_offset:item.end_offset] == item.text for item in first)
        assert all(item.start_offset < item.end_offset <= len(normalized) for item in first)


@pytest.mark.parametrize("strategy", sorted(CHUNK_STRATEGIES))
def test_main_store_persists_strategy_and_offsets(tmp_path, strategy):
    store = RagStore(tmp_path / f"{strategy}.sqlite3", max_chunk_chars=256, chunk_overlap_chars=32)
    text = _fixture_text()
    result = store.ingest_document(
        source_id=f"chunk-{strategy}", relative_ref=f"docs/{strategy}.md", sha256=None,
        mime="text/markdown", title="Chunk strategy", text=text, revision="r1", strategy=strategy,
    )
    assert result.chunk_count > 1
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT start_offset, end_offset, text_content, granularity FROM rag_chunks ORDER BY ordinal"
        ).fetchall()
    assert all(row[3] == strategy for row in rows)
    assert all(text[start:end] == chunk for start, end, chunk, _ in rows)
    assert all(0 <= start < end <= len(text) <= 4 * 1024 * 1024 for start, end, _, _ in rows)
    assert store.search("stable retrieval", access_scope="owner")[0]["granularity"] == strategy


def test_main_store_rejects_invalid_and_conflicting_strategy(tmp_path):
    store = RagStore(tmp_path / "strategy.sqlite3", max_chunk_chars=256)
    kwargs = {
        "source_id": "strategy-doc", "relative_ref": "docs/strategy.md", "sha256": None,
        "mime": "text/markdown", "title": "Strategy", "text": _fixture_text(), "revision": "r1",
    }
    with pytest.raises(RagStoreError) as invalid:
        store.ingest_document(**kwargs, strategy="unknown")
    assert invalid.value.code == "chunk_strategy_invalid"
    assert store.ingest_document(**kwargs, strategy="fixed").status == "ingested"
    assert store.ingest_document(**kwargs, strategy="fixed").status == "duplicate"
    with pytest.raises(RagStoreError) as conflict:
        store.ingest_document(**kwargs, strategy="sentence")
    assert conflict.value.code == "revision_conflict"
    assert "chunk strategy" in str(conflict.value)


def test_harness_store_returns_strategy_and_granularity(tmp_path):
    store = HarnessRagStore(tmp_path / "harness.sqlite3")
    created = store.add_document(
        source_ref="docs/section.md", title="Section", text=_fixture_text(),
        owner_scope="project", max_chars=128, overlap_chars=16, strategy="section",
    )
    assert created["strategy"] == "section"
    chunks = store.list_chunks(owner_scope="project", source_ids=[created["source_id"]], limit=100)
    assert chunks
    assert all(hit.granularity == "section" for hit in chunks)
    assert all(hit.as_dict()["granularity"] == "section" for hit in chunks)
    with pytest.raises(ValueError, match="unsupported chunk strategy"):
        store.add_document(
            source_ref="docs/bad.md", title="Bad", text="text", strategy="unknown",
        )


def test_index_layers_keyword_prefix_phrase_and_rule_graph_are_model_free(tmp_path):
    text = "AlphaService uses BetaStore. BetaStore depends on GammaIndex. retrieval pipeline remains local."
    main = RagStore(tmp_path / "indexed-main.sqlite3", max_chunk_chars=256)
    main.ingest_document(
        source_id="indexed-main", relative_ref="docs/indexed.md", sha256=None,
        mime="text/markdown", title="Indexed", text=text, revision="r1",
    )
    assert {row["granularity"] for row in main.list_index_chunks()} >= {"document", "paragraph", "sentence"}
    assert set(INDEX_GRANULARITIES) >= {"document", "paragraph", "sentence"}
    assert main.keyword_search("AlphaServ", access_scope="owner", granularity="sentence")
    assert main.keyword_search("retrieval pipeline", access_scope="owner", granularity="document")[0]["keyword_score"] >= 2
    with sqlite3.connect(main.path) as connection:
        assert {row[0] for row in connection.execute("SELECT DISTINCT term_kind FROM rag_keyword_index")} >= {"token", "prefix", "phrase"}
    graph = main.graph_search("AlphaService", access_scope="owner", granularity="document")
    assert graph and graph[0]["graph_entities"]
    assert main.health()["keyword_term_count"] > 0
    harness = HarnessRagStore(tmp_path / "indexed-harness.sqlite3")
    harness.add_document(source_ref="docs/indexed.md", title="Indexed", text=text, max_chars=256)
    assert {row["granularity"] for row in harness.list_index_chunks()} >= {"document", "paragraph", "sentence"}
    assert harness.keyword_search("AlphaServ", granularity="sentence")
    assert harness.keyword_search("retrieval pipeline", granularity="document")[0]["keyword_score"] >= 2
    with sqlite3.connect(harness.path) as connection:
        assert {row[0] for row in connection.execute("SELECT DISTINCT term_kind FROM rag_keyword_index")} >= {"token", "prefix", "phrase"}
    graph = harness.graph_search("AlphaService", granularity="document")
    assert graph and graph[0]["graph_entities"]
    assert harness.health()["keyword_terms"] > 0


def test_frozen_baseline_compares_all_chunk_strategies():
    report = run_rag_chunk_comparison(top_k=5)
    assert report["schema"] == RAG_CHUNK_COMPARISON_SCHEMA
    assert report["valid"] is True
    assert set(report["strategies"]) == set(CHUNK_STRATEGIES)
    for comparison in report["strategies"].values():
        assert comparison["details_match"] is True
        assert comparison["main_project"]["hit_at_k"] == 1.0
        assert comparison["harness"]["hit_at_k"] == 1.0


def test_frozen_baseline_compares_original_and_rewritten_queries():
    report = run_rag_query_comparison(top_k=5)
    assert report["schema"] == RAG_QUERY_COMPARISON_SCHEMA
    assert report["valid"] is True
    for side in report["sides"].values():
        assert side["details_match"] is True
        assert side["baseline"]["hit_at_k"] == 1.0
        assert side["rewritten"]["hit_at_k"] == 1.0
        assert side["delta"] == {"hit_at_k": 0.0, "mean_reciprocal_rank": 0.0}
