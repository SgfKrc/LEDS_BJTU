import pytest

from harness_workbench.memory import LayeredBudget, MemoryRetriever, MemoryStore, build_layered_context, build_memory_context
from harness_workbench.rag.providers import EmbeddingResult


def test_memory_search_uses_fts_and_preserves_scope_and_lifecycle_filters(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    keep = store.add(kind="fact", content="The primary node owns the local SQLite database.", owner_scope="user-a")
    store.add(kind="fact", content="The remote node owns a temporary cache.", owner_scope="user-b")
    removed = store.add(kind="fact", content="The primary node used a retired database.", owner_scope="user-a")
    store.delete(removed.entry_id, owner_scope="user-a", confirm=True)
    hits = store.search("primary SQLite", owner_scope="user-a")
    assert [hit.entry_id for hit in hits] == [keep.entry_id]
    assert hits[0].citation()["entry_id"] == keep.entry_id
    assert store.search("primary SQLite", owner_scope="user-b") == []
    assert store.health()["backend"] == "sqlite_memory"
    assert store.health()["retrieval"] == "fts5"


def test_layered_budget_is_shared_and_complete_blocks_are_omitted(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    entry = store.add(kind="fact", content="alpha primary node", owner_scope="user-a")
    hits = store.search("alpha", owner_scope="user-a")
    budget = LayeredBudget.from_input_budget(80, memory_ratio=0.5, rag_ratio=0.25)
    context = build_layered_context(
        hits,
        rag_hits=({"chunk_id": "chunk-1", "source_id": "source-1", "text": "rag " * 5},),
        context_messages=("recent " * 30,),
        budget=budget,
    )
    assert budget.memory_budget + budget.rag_budget + budget.context_budget == budget.input_budget
    assert context.input_tokens <= context.input_budget
    assert context.truncated is True
    assert context.omitted_count >= 1
    assert all("alpha" not in citation.get("content", "") for citation in context.citations)
    assert context.citations[0]["entry_id"] == entry.entry_id
    assert context.as_dict()["budget"]["memory_budget"] == budget.memory_budget


def test_memory_context_does_not_silently_clip_a_large_entry():
    context = build_memory_context(
        ({"entry_id": "large", "content": "x " * 200},),
        input_budget=16,
    )
    assert context.text == ""
    assert context.included_count == 0
    assert context.omitted_count == 1
    assert context.truncated is True


def test_layered_budget_rejects_invalid_ratios_and_zero_budget():
    with pytest.raises(ValueError):
        LayeredBudget.from_input_budget(0)
    with pytest.raises(ValueError):
        LayeredBudget.from_input_budget(100, memory_ratio=0.8, rag_ratio=0.4)


def test_optional_embedding_provider_only_reranks_fts_candidates_and_falls_back(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    first = store.add(kind="fact", content="alpha one", owner_scope="user-a")
    second = store.add(kind="fact", content="alpha two", owner_scope="user-a")

    class Provider:
        def embed(self, texts):
            assert texts[0] == "alpha"
            vectors = [(1.0, 0.0)]
            vectors.extend((0.2, 0.9) if text == "alpha one" else (1.0, 0.0) for text in texts[1:])
            return EmbeddingResult("fixture", "fixture-v1", 2, tuple(vectors))

    hits = MemoryRetriever(store, embedding_provider=Provider()).search("alpha", owner_scope="user-a")
    assert [hit.entry_id for hit in hits] == [second.entry_id, first.entry_id]

    class BrokenProvider:
        def embed(self, texts):
            raise RuntimeError("provider unavailable")

    fallback = MemoryRetriever(store, embedding_provider=BrokenProvider()).search("alpha", owner_scope="user-a")
    assert {hit.entry_id for hit in fallback} == {first.entry_id, second.entry_id}
