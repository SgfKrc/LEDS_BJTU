from pathlib import Path

from harness_workbench.rag import (
    EmbeddingResult,
    HybridRagRetriever,
    RagSearchConfig,
    RagStore,
    build_context,
    chunk_text,
    rewrite_query,
)


class _KeywordEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            value = text.lower()
            vectors.append((1.0, 0.0) if "latency" in value else (0.0, 1.0) if "policy" in value else (0.1, 0.1))
        return EmbeddingResult("fixture", "keyword-v1", 2, tuple(vectors))


def _store(tmp_path: Path) -> RagStore:
    store = RagStore(tmp_path / "rag.sqlite3")
    store.add_document(
        source_ref="docs/latency.md",
        title="Latency",
        text="The latency budget is explicit and measured. " * 8,
        owner_scope="user-a",
        max_chars=128,
        overlap_chars=16,
        strategy="sentence",
    )
    store.add_document(
        source_ref="docs/policy.md",
        title="Policy",
        text="The policy requires bounded citations and owner isolation. " * 8,
        owner_scope="user-a",
        max_chars=128,
        overlap_chars=16,
    )
    store.add_document(
        source_ref="docs/other.md",
        title="Other",
        text="latency is private to another owner. " * 8,
        owner_scope="user-b",
        max_chars=128,
        overlap_chars=16,
    )
    return store


def test_query_rewrite_is_normalized_deterministic_and_bounded():
    first = rewrite_query("  SQLite\n检索  ")
    second = rewrite_query("SQLite 检索")
    assert first.normalized == second.normalized
    assert first.variants == second.variants
    assert first.normalized == "SQLite 检索"
    assert first.variants[0] == first.normalized
    assert len(first.variants) <= 4
    assert "SQLite" in rewrite_query("SQLite and 检索", max_variants=4).variants
    assert "检索" in rewrite_query("SQLite and 检索", max_variants=4).variants


def test_chunking_supports_sentence_and_paragraph_granularity(tmp_path):
    text = "First sentence. Second sentence.\n\nA second paragraph. " * 8
    sentence = chunk_text(text, max_chars=128, overlap_chars=16, strategy="sentence")
    paragraph = chunk_text(text, max_chars=128, overlap_chars=16, strategy="paragraph")
    assert sentence and paragraph
    assert all(len(item.text) <= 128 for item in sentence + paragraph)
    assert {item.granularity for item in sentence} == {"sentence"}
    assert {item.granularity for item in paragraph} == {"paragraph"}


def test_hybrid_retriever_fuses_routes_and_keeps_owner_scope(tmp_path):
    provider = _KeywordEmbeddings()
    retriever = HybridRagRetriever(
        _store(tmp_path),
        embedding_provider=provider,
        config=RagSearchConfig(top_k=5, per_route_k=5),
    )
    result = retriever.search("latency", owner_scope="user-a")
    assert result.hits
    assert all(hit.source_id != "" for hit in result.hits)
    assert all(hit.source_id != "" for hit in result.hits)
    assert any(hit.title == "Latency" for hit in result.hits)
    assert result.route_counts["fts"] > 0
    assert result.route_counts["embedding"] > 0
    assert len({hit.chunk_id for hit in result.hits}) == len(result.hits)
    assert any(set(hit.routes) == {"embedding", "fts"} for hit in result.hits)
    assert provider.calls == 1


def test_rewritten_variants_are_embedded_and_fused_in_one_provider_call(tmp_path):
    class RecordingProvider:
        def __init__(self):
            self.inputs = []

        def embed(self, texts):
            self.inputs.append(list(texts))
            vectors = [(1.0, 0.0) if "latency" in text.lower() else (0.0, 1.0) for text in texts]
            return EmbeddingResult("fixture", "rewrite-v1", 2, tuple(vectors))

    provider = RecordingProvider()
    retriever = HybridRagRetriever(
        _store(tmp_path), embedding_provider=provider,
        config=RagSearchConfig(top_k=3, per_route_k=2, rewrite_limit=2),
        expansions={"latency": ("delay",)},
    )
    result = retriever.search("latency", owner_scope="user-a")
    assert result.hits
    assert result.route_counts["fts_variants"] == 2
    assert result.route_counts["embedding_variants"] == 2
    assert len(provider.inputs) == 1
    assert provider.inputs[0][:2] == ["latency", "delay"]


def test_embedding_failure_falls_back_to_fts(tmp_path):
    class Broken:
        def embed(self, texts):
            raise RuntimeError("fixture provider unavailable")

    result = HybridRagRetriever(_store(tmp_path), embedding_provider=Broken()).search("policy", owner_scope="user-a")
    assert result.hits
    assert result.route_counts["fts"] > 0
    assert result.route_counts["embedding_error"] == 1


def test_rag_cache_reuses_across_calls_and_invalidates_on_snapshot_change(tmp_path):
    store = _store(tmp_path)
    provider = _KeywordEmbeddings()
    retriever = HybridRagRetriever(store, embedding_provider=provider)
    first = retriever.search("latency", owner_scope="user-a", session_id="s1")
    second = retriever.search("latency", owner_scope="user-a", session_id="s2")
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert provider.calls == 1
    store.add_document(source_ref="docs/new.md", title="New", text="latency " * 30, owner_scope="user-a", max_chars=128, overlap_chars=16)
    third = retriever.search("latency", owner_scope="user-a")
    assert third.cache_hit is False
    assert provider.calls == 2


def test_metadata_filters_are_applied_before_hybrid_fusion(tmp_path):
    result = HybridRagRetriever(_store(tmp_path), config=RagSearchConfig(embedding_weight=0)).search(
        "latency", owner_scope="user-a", title_prefix="Policy"
    )
    assert result.hits == ()
    result = HybridRagRetriever(_store(tmp_path), config=RagSearchConfig(embedding_weight=0)).search(
        "citations", owner_scope="user-a", title_prefix="Policy"
    )
    assert result.hits and all(hit.title == "Policy" for hit in result.hits)


def test_context_enforces_token_budget_without_partial_citations():
    context = build_context(
        [
            {"source_id": "s", "chunk_id": "c1", "title": "A", "ordinal": 0, "text": "one two three"},
            {"source_id": "s", "chunk_id": "c2", "title": "A", "ordinal": 1, "text": "four five six"},
        ],
        max_chars=1000,
        max_tokens=6,
        tokenizer=lambda value: len(value.split()),
    )
    assert context.included_count == 1
    assert context.omitted_count == 1
    assert context.token_count <= 6
    assert [item["chunk_id"] for item in context.citations] == ["c1"]
    assert context.as_dict()["omission_reasons"] == ["tokens"]
