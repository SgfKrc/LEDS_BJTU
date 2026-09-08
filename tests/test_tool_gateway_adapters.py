"""Provider adapters stay bounded, deterministic, and fail closed."""

from __future__ import annotations

import json

import pytest

from src.tool_gateway import TOOL_REQUEST_SCHEMA, TOOL_RESULT_SCHEMA, ToolGatewayError, ToolGatewayPolicy
from src.tool_gateway_adapters import (
    FakeToolProvider,
    RestrictedFetchAdapter,
    SearxSearchAdapter,
    ToolGatewayExecutor,
    ToolProviderError,
    TransportResponse,
)


def _request(tool_name: str, arguments: dict, request_id: str = "req_adapter_01") -> dict:
    return {
        "schema": TOOL_REQUEST_SCHEMA,
        "request_id": request_id,
        "tool_name": tool_name,
        "arguments": arguments,
        "user_scope": "local_user",
        "network_scope": "explicit_opt_in",
        "deadline_ms": 5000,
    }


class FakeTransport:
    def __init__(self, response: TransportResponse):
        self.response = response
        self.calls: list[dict] = []

    def get(self, url, *, headers, proxy, timeout_seconds, max_bytes, policy):
        self.calls.append({"url": url, "headers": dict(headers), "proxy": proxy, "timeout_seconds": timeout_seconds, "max_bytes": max_bytes})
        return self.response


def _response(body: bytes, content_type: str, *, status_code: int = 200, url: str = "https://provider.example/response") -> TransportResponse:
    return TransportResponse(status_code, {"Content-Type": content_type}, body, url, (url,), 4)


def _allow_all() -> ToolGatewayPolicy:
    return ToolGatewayPolicy(data_scope="allow_all")


def test_searx_adapter_normalizes_results_and_passes_explicit_proxy():
    payload = {"results": [{"title": "Example", "url": "https://example.com/result", "content": "A short result."}]}
    transport = FakeTransport(_response(json.dumps(payload).encode(), "application/json"))
    adapter = SearxSearchAdapter(
        "https://search.example/search",
        transport=transport,
        policy=_allow_all(),
        proxy="http://127.0.0.1:7897",
    )

    result = adapter.execute(_request("web_search", {"query": "QLH", "top_k": 3}), allow_external=True)

    assert result["schema"] == TOOL_RESULT_SCHEMA
    assert result["status"] == "ok"
    assert result["items"][0]["title"] == "Example"
    assert result["citations"][0]["url"] == "https://example.com/result"
    assert "q=QLH" in transport.calls[0]["url"]
    assert transport.calls[0]["proxy"] == "http://127.0.0.1:7897"
    assert transport.calls[0]["headers"]["User-Agent"]


def test_searx_adapter_filters_unsafe_result_urls():
    payload = {"results": [{"title": "Private", "url": "https://127.0.0.1/admin", "content": "secret"}]}
    adapter = SearxSearchAdapter(
        "https://search.example/search",
        transport=FakeTransport(_response(json.dumps(payload).encode(), "application/json")),
        policy=_allow_all(),
    )

    with pytest.raises(ToolProviderError) as exc:
        adapter.execute(_request("web_search", {"query": "private", "top_k": 1}), allow_external=True)
    assert exc.value.code == "no_safe_results"


def test_fetch_adapter_extracts_text_and_drops_script_content():
    body = b"<html><title>Example title</title><body>Hello <script>evil()</script>world</body></html>"
    adapter = RestrictedFetchAdapter(
        transport=FakeTransport(_response(body, "text/html; charset=utf-8", url="https://example.com/page")),
        policy=_allow_all(),
    )

    result = adapter.execute(_request("web_fetch", {"url": "https://example.com/page", "max_chars": 500}), allow_external=True)

    assert result["items"][0]["title"] == "Example title"
    assert result["items"][0]["snippet"] == "Hello world"
    assert "evil" not in result["items"][0]["snippet"]
    assert len(result["citations"][0]["sha256"]) == 64


def test_fetch_adapter_rejects_unsupported_content_type():
    adapter = RestrictedFetchAdapter(
        transport=FakeTransport(_response(b"binary", "application/octet-stream")),
        policy=_allow_all(),
    )

    with pytest.raises(ToolProviderError) as exc:
        adapter.execute(_request("web_fetch", {"url": "https://example.com/file", "max_chars": 500}), allow_external=True)
    assert exc.value.code == "unsupported_content_type"


def _ok_result(request_id: str) -> dict:
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "ok",
        "items": [{"title": "Fallback", "url": "https://example.com/fallback", "snippet": "fallback result"}],
        "citations": [{"url": "https://example.com/fallback", "sha256": "a" * 64}],
        "truncated": False,
        "policy": {"redirects": 0, "bytes": 32, "content_type": "text/plain"},
    }


def test_executor_falls_back_only_after_retryable_provider_error():
    request = _request("web_search", {"query": "QLH", "top_k": 1}, request_id="req_fallback_01")
    first = FakeToolProvider("web_search", "primary", lambda _: ToolProviderError("timeout", "slow", retryable=True))
    second = FakeToolProvider("web_search", "secondary", lambda value: _ok_result(value["request_id"]))

    report = ToolGatewayExecutor({"primary": first, "secondary": second}, policy=_allow_all()).execute(request, allow_external=True)

    assert report.provider_id == "secondary"
    assert report.fallback_used is True
    assert [attempt["status"] for attempt in report.attempts] == ["failed", "ok"]
    assert report.result["items"][0]["title"] == "Fallback"


def test_executor_does_not_fallback_after_non_retryable_error():
    request = _request("web_search", {"query": "QLH", "top_k": 1}, request_id="req_no_fallback_01")
    first = FakeToolProvider("web_search", "primary", lambda _: ToolProviderError("bad_request", "invalid", retryable=False))
    second = FakeToolProvider("web_search", "secondary", lambda value: _ok_result(value["request_id"]))

    report = ToolGatewayExecutor({"primary": first, "secondary": second}, policy=_allow_all()).execute(request, allow_external=True)

    assert report.provider_id == "primary"
    assert report.fallback_used is False
    assert report.result["status"] == "error"
    assert report.result["error"]["code"] == "bad_request"
    assert len(report.attempts) == 1


def test_executor_preserves_gateway_scope_denial():
    provider = FakeToolProvider("web_search", "primary", lambda value: _ok_result(value["request_id"]))
    request = _request("web_search", {"query": "QLH", "top_k": 1})

    with pytest.raises(ToolGatewayError) as exc:
        ToolGatewayExecutor({"primary": provider}).execute(request, allow_external=False)
    assert exc.value.code == "scope_denied"

