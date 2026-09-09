import json

import pytest

from harness_workbench.tools import NetworkClient, NetworkPolicy, NetworkResponse, NetworkToolError


class FakeTransport:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def get(self, url, *, headers, proxy, timeout_seconds, max_bytes, resolved_addresses):
        self.calls.append({"url": url, "proxy": proxy, "addresses": tuple(resolved_addresses), "max_bytes": max_bytes})
        return self.responses[url]


def _resolver(host, port):
    assert port in {443, 8443}
    return ("93.184.216.34", "2606:4700:4700::1111")


def test_production_network_is_off_by_default_but_fake_transport_is_testable():
    with pytest.raises(NetworkToolError, match="production network access is disabled"):
        NetworkClient(resolver=_resolver).fetch("https://example.com", allow_external=True)

    transport = FakeTransport({"https://example.com/": NetworkResponse(200, {"content-type": "text/plain"}, b"hello")})
    result = NetworkClient(transport=transport, resolver=_resolver).fetch("https://example.com", allow_external=True)
    assert result.text == "hello"
    assert transport.calls[0]["addresses"] == ("2606:4700:4700::1111", "93.184.216.34")


def test_ssrf_dns_scheme_and_scope_gates_are_fail_closed():
    transport = FakeTransport({})
    client = NetworkClient(transport=transport, resolver=lambda host, port: ("127.0.0.1",))
    with pytest.raises(NetworkToolError, match="non-public"):
        client.fetch("https://example.com", allow_external=True)
    with pytest.raises(NetworkToolError, match="explicit user opt-in"):
        NetworkClient(transport=transport, resolver=_resolver).fetch("https://example.com")
    with pytest.raises(NetworkToolError, match="only HTTPS"):
        NetworkClient(transport=transport, resolver=_resolver, policy=NetworkPolicy(allow_http=False)).fetch("http://example.com", allow_external=True)
    with pytest.raises(NetworkToolError, match="not allowed"):
        NetworkClient(transport=transport, resolver=_resolver).fetch("https://localhost", allow_external=True)


def test_each_redirect_is_revalidated_and_response_gates_are_enforced():
    responses = {
        "https://example.com/": NetworkResponse(302, {"location": "https://example.org/next"}, b""),
        "https://example.org/next": NetworkResponse(200, {"content-type": "text/html; charset=utf-8"}, b"<script>bad()</script><p>safe page</p>"),
    }
    transport = FakeTransport(responses)
    client = NetworkClient(transport=transport, resolver=_resolver)
    result = client.fetch("https://example.com", allow_external=True)
    assert result.final_url == "https://example.org/next"
    assert result.redirects == ("https://example.com/", "https://example.org/next")
    assert result.text == "safe page"
    assert len(transport.calls) == 2

    unsafe = FakeTransport({"https://example.com/": NetworkResponse(302, {"location": "http://127.0.0.1/private"}, b"")})
    with pytest.raises(NetworkToolError, match="redirect target is not allowed"):
        NetworkClient(transport=unsafe, resolver=_resolver).fetch("https://example.com", allow_external=True)

    too_large = FakeTransport({"https://example.com/": NetworkResponse(200, {"content-type": "text/plain"}, b"x" * 2_000)})
    with pytest.raises(NetworkToolError, match="byte limit"):
        NetworkClient(transport=too_large, resolver=_resolver, policy=NetworkPolicy(max_response_bytes=1024)).fetch("https://example.com", allow_external=True)

    binary = FakeTransport({"https://example.com/": NetworkResponse(200, {"content-type": "application/octet-stream"}, b"x")})
    with pytest.raises(NetworkToolError, match="content type"):
        NetworkClient(transport=binary, resolver=_resolver).fetch("https://example.com", allow_external=True)


def test_search_returns_bounded_safe_items_and_citations():
    endpoint = "https://search.example/search"
    body = json.dumps(
        {
            "results": [
                {"title": "Safe", "url": "https://docs.example/a", "content": "summary"},
                {"title": "Private", "url": "https://127.0.0.1/private", "content": "do not use"},
            ]
        }
    ).encode()
    transport = FakeTransport({endpoint + "?q=QLH&format=json": NetworkResponse(200, {"content-type": "application/json"}, body)})
    client = NetworkClient(transport=transport, resolver=_resolver, search_endpoint=endpoint)
    result = client.search("QLH", top_k=2, allow_external=True)
    assert len(result.items) == 1
    assert result.items[0]["url"] == "https://docs.example/a"
    assert len(result.citations[0]["sha256"]) == 64
    assert result.truncated is True


def test_redirect_limit_and_proxy_credentials_are_rejected():
    policy = NetworkPolicy(max_redirects=1)
    responses = {
        "https://example.com/": NetworkResponse(302, {"location": "https://example.org/one"}, b""),
        "https://example.org/one": NetworkResponse(302, {"location": "https://example.net/two"}, b""),
    }
    with pytest.raises(NetworkToolError, match="redirect chain"):
        NetworkClient(transport=FakeTransport(responses), resolver=_resolver, policy=policy).fetch("https://example.com", allow_external=True)
    with pytest.raises(NetworkToolError, match="without credentials"):
        NetworkPolicy(proxy="http://user:pass@127.0.0.1:7897")
