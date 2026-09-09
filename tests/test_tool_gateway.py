"""Fail-closed Tool Gateway contract and SSRF policy tests."""

from __future__ import annotations

import ipaddress

import pytest

from src.tool_gateway import (
    TOOL_REQUEST_SCHEMA,
    TOOL_RESULT_SCHEMA,
    ToolGateway,
    ToolGatewayError,
    ToolGatewayPolicy,
    error_result,
    normalize_tool_result,
    prepare_tool_request,
    validate_proxy_url,
    validate_redirect_chain,
    validate_resolved_addresses,
    validate_url,
)


def _request(tool_name="web_search", arguments=None, **overrides):
    payload = {
        "schema": TOOL_REQUEST_SCHEMA,
        "request_id": "req_gateway_01",
        "tool_name": tool_name,
        "arguments": arguments or {"query": "QLH", "top_k": 3},
        "user_scope": "local_user",
        "network_scope": "explicit_opt_in",
        "deadline_ms": 5000,
    }
    payload.update(overrides)
    return payload


def _result(request_id="req_gateway_01"):
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "ok",
        "items": [{"title": "QLH", "url": "https://example.com/result", "snippet": "summary"}],
        "citations": [{"url": "https://example.com/result", "sha256": "a" * 64}],
        "truncated": False,
        "policy": {"redirects": 0, "bytes": 128, "content_type": "text/html; charset=utf-8"},
    }


def test_scope_is_fail_closed_and_request_cannot_widen_policy():
    with pytest.raises(ToolGatewayError) as denied:
        prepare_tool_request(_request(), policy=ToolGatewayPolicy(data_scope="deny"), allow_external=True)
    assert denied.value.code == "scope_denied"
    with pytest.raises(ToolGatewayError) as missing:
        prepare_tool_request(_request(), policy=ToolGatewayPolicy(data_scope="opt_in"), allow_external=False)
    assert missing.value.code == "scope_denied"
    prepared = prepare_tool_request(_request(), policy=ToolGatewayPolicy(data_scope="allow_all"))
    assert prepared["arguments"] == {"query": "QLH", "top_k": 3}
    with pytest.raises(ToolGatewayError) as override:
        prepare_tool_request(_request(network_scope="allow_all"), policy=ToolGatewayPolicy(data_scope="allow_all"))
    assert override.value.code == "scope_override"


@pytest.mark.parametrize(
    "url,code",
    [
        ("http://example.com", "unsafe_url_scheme"),
        ("file:///etc/passwd", "unsafe_url_scheme"),
        ("https://127.0.0.1/admin", "unsafe_url_host"),
        ("https://169.254.169.254/latest", "unsafe_url_host"),
        ("https://[::1]/", "unsafe_url_host"),
        ("https://user:pass@example.com/", "unsafe_url_host"),
        ("https://example.com/#fragment", "invalid_url"),
    ],
)
def test_url_policy_rejects_unsafe_targets(url, code):
    with pytest.raises(ToolGatewayError) as exc:
        validate_url(url)
    assert exc.value.code == code


def test_ipv6_and_http_opt_in_are_explicit():
    public_ipv6 = validate_url("https://[2001:4860:4860::8888]/dns-query")
    assert public_ipv6 == "https://[2001:4860:4860::8888]/dns-query"
    with pytest.raises(ToolGatewayError):
        validate_url("http://example.com")
    assert validate_url("http://example.com:8080/path", policy=ToolGatewayPolicy(allow_http=True)) == "http://example.com:8080/path"


def test_post_dns_ssrf_gate_rejects_mixed_addresses():
    assert validate_resolved_addresses("example.com", [ipaddress.ip_address("93.184.216.34")]) == ("93.184.216.34",)
    with pytest.raises(ToolGatewayError) as private:
        validate_resolved_addresses("example.com", ["93.184.216.34", "10.0.0.1"])
    assert private.value.code == "unsafe_url_host"
    with pytest.raises(ToolGatewayError) as empty:
        validate_resolved_addresses("example.com", [])
    assert empty.value.code == "dns_no_public_address"


def test_redirect_limit_scheme_and_each_hop_are_checked():
    policy = ToolGatewayPolicy(max_redirects=2)
    assert validate_redirect_chain(["https://example.com/a", "https://example.org/b"], policy=policy)[-1].endswith("/b")
    with pytest.raises(ToolGatewayError) as too_many:
        validate_redirect_chain(["https://example.com/1", "https://example.com/2", "https://example.com/3", "https://example.com/4"], policy=policy)
    assert too_many.value.code == "redirect_limit_exceeded"
    with pytest.raises(ToolGatewayError) as scheme:
        validate_redirect_chain(["https://example.com/1", "http://example.com/2"], policy=ToolGatewayPolicy(allow_http=True))
    assert scheme.value.code == "unsafe_redirect"


def test_request_contract_rejects_unknown_tools_and_limits():
    gateway = ToolGateway(ToolGatewayPolicy(data_scope="allow_all"))
    with pytest.raises(ToolGatewayError) as unknown:
        gateway.prepare(_request("shell", {}))
    assert unknown.value.code == "unknown_tool"
    with pytest.raises(ToolGatewayError) as bad_args:
        gateway.prepare(_request("web_fetch", {"url": "https://example.com", "max_chars": 999999}))
    assert bad_args.value.code == "invalid_arguments"
    with pytest.raises(ToolGatewayError) as bad_id:
        gateway.prepare(_request(request_id="short"))
    assert bad_id.value.code == "invalid_request_id"


def test_result_contract_redacts_raw_fields_and_enforces_limits():
    normalized = normalize_tool_result(_result(), request_id="req_gateway_01")
    assert normalized["policy"]["content_type"] == "text/html"
    assert "body" not in normalized
    with pytest.raises(ToolGatewayError) as bad_type:
        normalize_tool_result({**_result(), "policy": {"redirects": 0, "bytes": 1, "content_type": "application/octet-stream"}}, request_id="req_gateway_01")
    assert bad_type.value.code == "unsupported_content_type"
    with pytest.raises(ToolGatewayError) as bad_digest:
        normalize_tool_result({**_result(), "citations": [{"url": "https://example.com", "sha256": "bad"}]}, request_id="req_gateway_01")
    assert bad_digest.value.code == "invalid_result"


def test_proxy_is_explicit_user_configuration_only():
    assert validate_proxy_url("http://127.0.0.1:7897") == "http://127.0.0.1:7897"
    with pytest.raises(ToolGatewayError) as credentials:
        validate_proxy_url("http://user:pass@127.0.0.1:7897")
    assert credentials.value.code == "invalid_proxy"
    with pytest.raises(ToolGatewayError) as invalid:
        validate_proxy_url("ftp://127.0.0.1:7897")
    assert invalid.value.code == "invalid_proxy"


def test_error_result_is_stable_and_non_sensitive():
    error = ToolGatewayError("unsafe_url_host", "URL resolves to a non-public address")
    result = error_result("req_gateway_01", error)
    assert result == {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": "req_gateway_01",
        "status": "error",
        "error": {"code": "unsafe_url_host", "message": "URL resolves to a non-public address", "retryable": False},
    }
