"""G5 tool-cache API keeps persistence explicit and path-free."""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from rag_store import RagStore
from tool_rag_cache import ToolRagCache


def _result(request_id: str = "req_cache_api_01") -> dict:
    return {
        "schema": "qlh.tool_result.v1",
        "request_id": request_id,
        "status": "ok",
        "items": [{"title": "API page", "url": "https://example.com/api", "snippet": "cacheable API content"}],
        "citations": [{"url": "https://example.com/api", "sha256": "a" * 64}],
        "truncated": False,
        "policy": {"redirects": 0, "bytes": 96, "content_type": "text/html"},
    }


def _client(monkeypatch, tmp_path):
    store = RagStore(tmp_path / "rag.sqlite3", max_chunk_chars=256)
    cache = ToolRagCache(store)
    monkeypatch.setattr(api_server, "_rag_store_instance", store)
    monkeypatch.setattr(api_server, "_tool_rag_cache_instance", cache)
    return TestClient(api_server.app)


def test_tool_cache_api_requires_explicit_persist_and_returns_citations(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    body = {"tool_name": "web_fetch", "result": _result()}
    denied = client.post("/api/tool-cache", json=body)
    assert denied.status_code == 422
    assert denied.json()["detail"] == "persistence_not_explicit"

    saved = client.post("/api/tool-cache", json={**body, "persist": True, "ttl_seconds": 120})
    assert saved.status_code == 200
    entry = saved.json()["entry"]
    assert entry["schema"] == "qlh.tool_cache.v1"
    assert entry["citations"][0]["url"] == "https://example.com/api"
    health = client.get("/api/tool-cache/health")
    assert health.status_code == 200
    assert health.json()["entries"] == 1

    search = client.post("/api/tool-cache/search", json={"query": "cacheable API content"})
    assert search.status_code == 200
    assert search.json()["entries"][0]["cache_id"] == entry["cache_id"]

    deleted = client.delete(f"/api/tool-cache/{entry['cache_id']}")
    assert deleted.status_code == 200
    assert client.get("/api/tool-cache/health").json()["entries"] == 0


def test_tool_cache_api_does_not_expose_local_path_or_raw_body(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    saved = client.post(
        "/api/tool-cache",
        json={"tool_name": "web_search", "result": _result("req_cache_api_02"), "persist": True},
    )
    assert saved.status_code == 200
    for payload in (saved.json(), client.get("/api/tool-cache/health").json()):
        text = str(payload)
        assert str(tmp_path) not in text
        assert "vector_blob" not in text

