"""P1a model search API contract tests without external network."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
import model_search


@pytest.fixture(autouse=True)
def _iso_db(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "test.sqlite3"))
    import local_store
    local_store._initialized_paths.clear()
    yield
    local_store._initialized_paths.clear()


def test_search_endpoint_returns_provider_neutral_projection(monkeypatch):
    captured = {}

    def fake_search(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return {
            "query": query,
            "source": kwargs["source"],
            "provider": "ms",
            "results": [{"id": "Qwen/Qwen3-4B", "source": "ms"}],
            "total": 1,
            "fallback_used": True,
            "attempts": [
                {"provider": "hf", "transport": "direct", "status": "failed", "code": "timeout"},
                {"provider": "ms", "transport": "direct", "status": "ok"},
            ],
        }

    monkeypatch.setattr(api_server.model_search, "search_models", fake_search)
    with TestClient(api_server.app) as client:
        response = client.get("/api/models/search", params={
            "q": "qwen", "source": "all", "page": 2, "limit": 7,
            "proxy": "http://127.0.0.1:7897",
        })
    assert response.status_code == 200, response.text
    assert response.json()["provider"] == "ms"
    assert captured == {
        "query": "qwen", "source": "all", "page": 2, "limit": 7,
        "proxy": "http://127.0.0.1:7897",
    }


def test_search_endpoint_exposes_coded_fail_closed_error(monkeypatch):
    def fail(*args, **kwargs):
        raise model_search.ModelSearchError(
            "SEARCH_UNAVAILABLE", "模型源搜索不可用",
            attempts=[{"provider": "hf", "status": "failed", "code": "timeout"}],
        )

    monkeypatch.setattr(api_server.model_search, "search_models", fail)
    with TestClient(api_server.app) as client:
        response = client.get("/api/models/search?q=qwen")
    assert response.status_code == 502
    assert response.headers.get("X-QLH-Error-Code") == "SEARCH_UNAVAILABLE"
    body = response.json()
    assert body["error_code"] == "SEARCH_UNAVAILABLE"
    assert body["detail"]["attempts"][0]["code"] == "timeout"


@pytest.mark.parametrize("params, code", [
    ({}, "QUERY_REQUIRED"),
    ({"q": "qwen", "source": "bad"}, "SOURCE_INVALID"),
])
def test_search_endpoint_rejects_invalid_query(params, code):
    with TestClient(api_server.app) as client:
        response = client.get("/api/models/search", params=params)
    assert response.status_code == 400
    assert response.headers.get("X-QLH-Error-Code") == code
