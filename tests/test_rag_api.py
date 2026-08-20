from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from rag_store import RagStore


def _store(tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    store.ingest_document(
        source_id="api-doc", relative_ref="docs/api.md", sha256=None,
        mime="text/markdown", title="API", text="cluster retrieval citation", revision="r1",
    )
    return store


def test_rag_api_returns_citations_and_local_health(monkeypatch, tmp_path):
    store = _store(tmp_path)
    monkeypatch.setattr(api_server, "_rag_store_instance", store)
    client = TestClient(api_server.app)
    health = client.get("/api/rag/health")
    assert health.status_code == 200
    assert health.json()["journal_mode"] == "wal"
    response = client.post("/api/rag/search", json={"query": "retrieval", "mode": "fts"})
    assert response.status_code == 200
    body = response.json()
    assert body["storage"] == "sqlite"
    assert body["results"][0]["source_id"] == "api-doc"
    assert body["results"][0]["relative_ref"] == "docs/api.md"
    assert "text_content" not in body["results"][0]
    assert "vector_blob" not in body["results"][0]


def test_rag_api_rebuild_delete_and_hybrid_contract(monkeypatch, tmp_path):
    store = _store(tmp_path)
    monkeypatch.setattr(api_server, "_rag_store_instance", store)
    client = TestClient(api_server.app)
    rebuild = client.post("/api/rag/rebuild", json={})
    assert rebuild.status_code == 200
    assert rebuild.json()["status"] == "ok"
    missing = client.post("/api/rag/search", json={"query": "retrieval", "mode": "hybrid"})
    assert missing.status_code == 422
    unsupported = client.post("/api/rag/rebuild", json={"include_embeddings": True})
    assert unsupported.status_code == 422
    deleted = client.delete("/api/rag/sources/api-doc")
    assert deleted.status_code == 200
    assert client.post("/api/rag/search", json={"query": "retrieval"}).json()["count"] == 0


def test_rag_api_capacity_and_job_cursor_are_redacted(monkeypatch, tmp_path):
    store = _store(tmp_path)
    monkeypatch.setattr(api_server, "_rag_store_instance", store)
    client = TestClient(api_server.app)
    capacity = client.get("/api/rag/capacity?dimensions=768")
    assert capacity.status_code == 200
    assert capacity.json()["estimated_vector_bytes"] >= 768 * 4
    created = client.post(
        "/api/rag/embedding-jobs",
        json={"provider": "ollama", "model_sha256": "e" * 64, "batch_size": 1},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["state"] == "queued"
    assert body["cursor"]["total"] == 1
    assert "chunk_ids" not in body["cursor"]
    assert "model_sha256" not in body["cursor"]
    cancelled = client.post(f"/api/rag/embedding-jobs/{body['job_id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"


def test_rag_api_ann_decision_is_path_free(monkeypatch, tmp_path):
    store = _store(tmp_path)
    monkeypatch.setattr(api_server, "_rag_store_instance", store)
    client = TestClient(api_server.app)
    response = client.get("/api/rag/ann-decision?scan_budget=100")
    assert response.status_code == 200
    body = response.json()
    assert body["storage"] == "sqlite"
    assert body["decision"]["decision"] == "NO_GO"
    assert "path" not in body["decision"]
