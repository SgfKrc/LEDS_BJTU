import json

import pytest

from harness_workbench.adapters import (
    AdapterCapabilities,
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    QLHAdapter,
    QLHAdapterConfig,
    StreamChunk,
)
from harness_workbench.adapters.qlh import _bounded_transcript
from harness_workbench.api_layer import create_app
from harness_workbench.rag import HybridRagRetriever, RagSearchConfig, RagStore, build_context, chunk_text
from harness_workbench.session import SessionStore


class _FakeQLHTransport:
    def __init__(self):
        self.posts = []

    def get_json(self, path):
        if path == "/api/status":
            return {"model_loaded": True, "current_model": "QW1.8B"}
        if path == "/api/models":
            return {
                "models": [
                    {
                        "model_id": "Qwen2.5-0.5B",
                        "is_available": False,
                        "unavailable_reason": "download_required",
                    }
                ]
            }
        if path == "/api/models/presets":
            return {
                "presets": [
                    {
                        "id": "qwen2.5-0.5b",
                        "name": "Qwen2.5-0.5B",
                        "default_model_id": "Qwen2.5-0.5B",
                        "installable": True,
                    }
                ]
            }
        if path == "/api/models/downloads":
            return {"jobs": []}
        raise AssertionError(f"unexpected GET path: {path}")

    def post_json(self, path, payload):
        self.posts.append((path, payload))
        if path == "/api/models/downloads":
            return {"job": {"id": "job-1", "preset_id": payload["preset_id"], "status": "queued"}}
        if path == "/api/models/load":
            return {"loaded": True, "model_id": payload["model_id"], "engine": payload["engine"]}
        return {"generation_id": "gen-1", "content": "answer", "metrics": {"total_prompt_tokens": 4, "total_generated_tokens": 2}}

    def post_stream(self, path, payload):
        self.posts.append((path, payload))
        return [
            'data: {"start": true}\n',
            'data: {"token": "hello"}\n',
            'data: {"done": true, "metrics": {"generated_tokens": 1}}\n',
        ]

    def close(self):
        return None


def _request(stream=False):
    return AdapterRequest(
        model="QW1.8B",
        messages=(
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ),
        stream=stream,
        max_tokens=64,
        request_id="req-1",
    )


def test_qlh_adapter_probes_status_and_maps_bounded_transcript():
    transport = _FakeQLHTransport()
    adapter = QLHAdapter(QLHAdapterConfig("http://master:8000", model_id="QW1.8B"), transport=transport)
    assert adapter.capabilities().model_ids == ("QW1.8B",)
    result = adapter.complete(_request())
    assert result.content == "answer"
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    path, payload = transport.posts[-1]
    assert path == "/api/chat"
    assert payload["model"] == "QW1.8B"
    assert "[system]" in payload["message"] and "[user]" in payload["message"]
    assert payload["routing_preference"] == "auto"


def test_qlh_adapter_exposes_model_assets_downloads_and_load():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    transport = _FakeQLHTransport()
    adapter = QLHAdapter(QLHAdapterConfig("http://master:8000"), transport=transport)
    models = adapter.models()
    assert models[0].id == "Qwen2.5-0.5B"
    assert models[0].available is False
    assert adapter.model_presets()["presets"][0]["id"] == "qwen2.5-0.5b"
    assert adapter.queue_model_download("qwen2.5-0.5b")["job"]["status"] == "queued"
    assert adapter.load_model_asset("Qwen2.5-0.5B")["loaded"] is True

    client = TestClient(create_app(adapter))
    assert client.get("/v1/model-assets").json()["models"][0]["available"] is False
    assert client.get("/v1/model-presets").json()["presets"]
    assert client.get("/v1/model-downloads").json()["jobs"] == []
    queued = client.post("/v1/model-downloads", json={"preset_id": "qwen2.5-0.5b"})
    assert queued.status_code == 200
    loaded = client.post("/v1/models/load", json={"model_id": "Qwen2.5-0.5B"})
    assert loaded.status_code == 200


def test_qlh_adapter_maps_sse_tokens_and_done():
    transport = _FakeQLHTransport()
    adapter = QLHAdapter(QLHAdapterConfig("http://master:8000"), transport=transport)
    chunks = list(adapter.stream(_request(stream=True)))
    assert chunks[0].delta == {"content": "hello"}
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage == {"completion_tokens": 1}
    assert transport.posts[-1][0] == "/api/chat/stream"


def test_qlh_full_sse_response_is_not_dropped_and_transcript_is_bounded():
    class FullTransport(_FakeQLHTransport):
        def post_stream(self, path, payload):
            return ['data: {"done": true, "response": "full answer"}\n']

    adapter = QLHAdapter(QLHAdapterConfig("http://master:8000", streaming_mode="full"), transport=FullTransport())
    chunks = list(adapter.stream(_request(stream=True)))
    assert chunks[0].delta == {"content": "full answer"}
    assert len(_bounded_transcript([{"role": "system", "content": "x" * 1000}], max_chars=512)) <= 512

    class DuplicateTransport(_FakeQLHTransport):
        def post_stream(self, path, payload):
            return [
                'data: {"token": "part"}\n',
                'data: {"done": true, "response": "part"}\n',
            ]

    chunks = list(QLHAdapter(QLHAdapterConfig("http://master:8000"), transport=DuplicateTransport()).stream(_request(stream=True)))
    assert [chunk.delta for chunk in chunks if chunk.delta] == [{"content": "part"}]


def test_chunking_is_bounded_and_explicit():
    chunks = chunk_text("alpha " * 100, max_chars=128, overlap_chars=16)
    assert len(chunks) > 1
    assert all(len(item.text) <= 128 for item in chunks)
    assert chunks[0].ordinal == 0
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=64)


def test_rag_store_uses_owner_scope_and_returns_citations(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3")
    created = store.add_document(
        source_ref="docs/guide.md",
        title="Guide",
        text="alpha retrieval contract. " * 20,
        owner_scope="user-a",
        max_chars=128,
        overlap_chars=16,
    )
    assert created["chunk_count"] > 1
    assert store.search("retrieval", owner_scope="user-a")
    assert store.search("retrieval", owner_scope="user-b") == []
    assert store.health()["backend"] == "sqlite_fts5"
    assert store.delete_source(created["source_id"], owner_scope="user-a") is True
    assert store.search("retrieval", owner_scope="user-a") == []


def test_harness_metadata_filters_are_indexed_and_cache_keyed(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3")
    store.add_document(
        source_ref="docs/team.md", title="Team", text="shared retrieval",
        owner_scope="user-a", metadata={"source": "handbook", "scope": "team", "tag": ["stable", "review"]},
    )
    store.add_document(
        source_ref="docs/public.md", title="Public", text="shared retrieval",
        owner_scope="user-a", metadata={"source": "handbook", "scope": "public", "tag": ["stable"]},
    )
    assert len(store.search("retrieval", owner_scope="user-a", metadata_filters={"scope": "team", "tag": "review"})) == 1
    with pytest.raises(ValueError, match="unsupported"):
        store.search("retrieval", owner_scope="user-a", metadata_filters={"unknown": "x"})

    retriever = HybridRagRetriever(store, config=RagSearchConfig(embedding_weight=0.0))
    first = retriever.search("retrieval", owner_scope="user-a", metadata_filters={"scope": "team"})
    second = retriever.search("retrieval", owner_scope="user-a", metadata_filters={"scope": "team"})
    assert first.hits and second.cache_hit
    assert len(second.hits) == 1


def test_rag_context_reports_omitted_chunks_instead_of_silent_truncation():
    context = build_context(
        [
            {"source_id": "s1", "chunk_id": "c1", "title": "A", "ordinal": 0, "text": "a" * 300},
            {"source_id": "s1", "chunk_id": "c2", "title": "A", "ordinal": 1, "text": "b" * 300},
        ],
        max_chars=600,
    )
    assert context.included_count == 1
    assert context.omitted_count == 1
    assert context.truncated is True


def test_session_store_persists_messages_and_asset_refs(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    session = store.create(owner_scope="user-a", title="Demo")
    message = store.append_message(session.session_id, role="user", content="hello", metadata={"source": "test"})
    asset = store.attach_asset(session.session_id, asset_id="img_abc", metadata={"mime_type": "image/png"})
    loaded = store.get(session.session_id, owner_scope="user-a")
    assert message.ordinal == 0
    assert asset.ordinal == 0
    assert loaded["messages"][0]["content"] == "hello"
    assert loaded["assets"][0]["asset_id"] == "img_abc"
    with pytest.raises(KeyError):
        store.get(session.session_id, owner_scope="other")


class _ChatAdapter:
    def capabilities(self):
        return AdapterCapabilities(backend="fake")

    def models(self):
        return (AdapterModel("fake"),)

    def complete(self, request):
        return AdapterResponse("id", request.model, "ok")

    def stream(self, request):
        yield StreamChunk("id", request.model, {"content": "ok"}, "stop")

    def close(self):
        return None


def test_s4_api_wires_rag_and_session_without_exposing_paths(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(
        _ChatAdapter(),
        rag_store=RagStore(tmp_path / "rag.sqlite3"),
        session_store=SessionStore(tmp_path / "sessions.sqlite3"),
    )
    client = TestClient(app)
    source = client.post(
        "/v1/rag/sources",
        json={"source_ref": "docs/a.md", "title": "A", "text": "alpha retrieval " * 30},
    )
    assert source.status_code == 201
    search = client.post("/v1/rag/search", json={"query": "retrieval"})
    assert search.status_code == 200
    assert search.json()["hits"]
    assert "path" not in json.dumps(search.json())
    created = client.post("/v1/sessions", json={"title": "S"})
    assert created.status_code == 201
    listed = client.get("/v1/sessions?owner_scope=local&limit=10")
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["session_id"] == created.json()["session_id"]
    loaded = client.get(f"/v1/sessions/{created.json()['session_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["session"]["title"] == "S"
    session_id = created.json()["session_id"]
    message = client.post(f"/v1/sessions/{session_id}/messages", json={"content": "hello"})
    assert message.status_code == 201
    asset = client.post(f"/v1/sessions/{session_id}/assets", json={"asset_id": "img_demo"})
    assert asset.status_code == 201


def test_s4_api_allows_per_request_rewrite_and_route_tuning(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    rag_store = RagStore(tmp_path / "rag.sqlite3")
    app = create_app(
        _ChatAdapter(),
        rag_store=rag_store,
        rag_retriever=HybridRagRetriever(rag_store, config=RagSearchConfig(embedding_weight=0.0)),
    )
    client = TestClient(app)
    assert client.post(
        "/v1/rag/sources",
        json={"source_ref": "docs/rewrite.md", "title": "Rewrite", "text": "retrieval contract"},
    ).status_code == 201
    response = client.post(
        "/v1/rag/search",
        json={"query": "检索", "rewrite_limit": 2, "per_route_k": 1, "fts_weight": 1.0, "embedding_weight": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hits"]
    assert body["retrieval"]["rewritten_queries"][0] == "检索"
    assert body["retrieval"]["route_counts"]["fts_variants"] == 2
