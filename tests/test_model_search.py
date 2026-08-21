"""P1a model repository search tests (provider calls are mocked)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_search as ms


def _item(repo_id="Qwen/Qwen3-4B"):
    return {
        "id": repo_id,
        "display_name": "Qwen 3 4B",
        "description": "public model",
        "tasks": ["text-generation"],
        "tags": ["library:transformers"],
        "downloads": 12,
        "likes": 3,
        "file_size": 1234,
        "license": "apache-2.0",
        "private": False,
        "gated": False,
    }


def test_normalise_projection_is_bounded_and_provider_neutral():
    item = _item()
    item["description"] = "x" * 1000
    result = ms._normalise_item(item, "hf")
    assert result["id"] == "Qwen/Qwen3-4B"
    assert result["source"] == "hf"
    assert len(result["description"]) == 500
    assert "files" not in result and "siblings" not in result
    assert result["url"].startswith("https://huggingface.co/")


def test_all_hf_direct_success_does_not_try_proxy_or_modelscope(monkeypatch):
    calls = []

    def hf(query, page, limit, *, proxy):
        calls.append(("hf", proxy))
        return [_item()], 1

    def fail_ms(*args, **kwargs):
        raise AssertionError("ModelScope must not be called")

    monkeypatch.setattr(ms, "_search_hf", hf)
    monkeypatch.setattr(ms, "_search_modelscope", fail_ms)
    result = ms.search_models("qwen", source="all")
    assert result["provider"] == "hf"
    assert result["fallback_used"] is False
    assert calls == [("hf", "")]


def test_all_fails_hf_then_uses_proxy(monkeypatch):
    calls = []

    def hf(query, page, limit, *, proxy):
        calls.append(proxy)
        if not proxy:
            raise ms._ProviderSearchError("network_error")
        return [_item()], 1

    monkeypatch.setattr(ms, "_search_hf", hf)
    result = ms.search_models("qwen", source="all", proxy="http://127.0.0.1:7897")
    assert result["provider"] == "hf"
    assert result["fallback_used"] is True
    assert calls == ["", "http://127.0.0.1:7897"]
    assert [attempt["transport"] for attempt in result["attempts"]] == ["direct", "proxy"]


def test_all_fails_hf_direct_and_proxy_then_modelscope(monkeypatch):
    calls = []

    def fail_hf(query, page, limit, *, proxy):
        calls.append(("hf", proxy))
        raise ms._ProviderSearchError("timeout")

    def model_scope(query, page, limit, *, proxy):
        calls.append(("ms", proxy))
        return [_item("Qwen/Qwen3-4B-MS")], 4

    monkeypatch.setattr(ms, "_search_hf", fail_hf)
    monkeypatch.setattr(ms, "_search_modelscope", model_scope)
    result = ms.search_models("qwen", source="all", proxy="http://127.0.0.1:7897")
    assert result["provider"] == "ms"
    assert result["total"] == 4
    assert calls == [
        ("hf", ""), ("hf", "http://127.0.0.1:7897"), ("ms", ""),
    ]
    assert result["attempts"][-1] == {"provider": "ms", "transport": "direct", "status": "ok"}


def test_source_ms_only_calls_modelscope(monkeypatch):
    calls = []
    monkeypatch.setattr(ms, "_search_hf", lambda *a, **k: calls.append("hf"))
    monkeypatch.setattr(ms, "_search_modelscope", lambda *a, **k: ([_item()], 1))
    result = ms.search_models("qwen", source="ms")
    assert result["provider"] == "ms"
    assert calls == []


@pytest.mark.parametrize("kwargs, code", [
    ({}, "QUERY_REQUIRED"),
    ({"source": "bad"}, "SOURCE_INVALID"),
    ({"proxy": "socks5://127.0.0.1:1"}, "PROXY_INVALID"),
])
def test_invalid_request_fails_closed(kwargs, code):
    with pytest.raises(ms.ModelSearchError) as exc:
        ms.search_models("" if not kwargs else "qwen", **kwargs)
    assert exc.value.code == code
