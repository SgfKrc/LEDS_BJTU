import pytest

from harness_workbench.tools import (
    QLHToolAdapter,
    QLHToolAdapterConfig,
    RemoteToolError,
    REMOTE_RESULT_SCHEMA,
    TOOL_REQUEST_SCHEMA,
)


def _request(tool_name="web_search", arguments=None, request_id="req_remote_01"):
    return {
        "schema": TOOL_REQUEST_SCHEMA,
        "request_id": request_id,
        "tool_name": tool_name,
        "arguments": arguments or {"query": "QLH", "top_k": 2},
        "user_scope": "local_user",
        "network_scope": "explicit_opt_in",
        "deadline_ms": 5000,
    }


def _result(request_id="req_remote_01"):
    return {
        "schema": REMOTE_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "ok",
        "items": [{"title": "QLH", "url": "https://docs.example/qlh", "snippet": "bounded result"}],
        "citations": [{"url": "https://docs.example/qlh", "sha256": "a" * 64}],
        "truncated": False,
        "policy": {"redirects": 0, "bytes": 128, "content_type": "application/json"},
    }


class FakeQLH:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.closed = False

    def post_json(self, path, payload):
        self.calls.append((path, payload))
        return self.result

    def close(self):
        self.closed = True


def test_remote_adapter_maps_request_and_result_without_network_side_effects():
    transport = FakeQLH(_result())
    adapter = QLHToolAdapter(QLHToolAdapterConfig("http://master:8888"), transport=transport)

    result = adapter.execute(_request(), allow_external=True)

    assert transport.calls[0][0] == "/api/tool"
    sent = transport.calls[0][1]
    assert sent["schema"] == TOOL_REQUEST_SCHEMA
    assert sent["network_scope"] == "explicit_opt_in"
    assert result["schema"] == "qlh.harness.tool_result.v1"
    assert result["tool_name"] == "web_search"
    assert result["items"][0]["snippet"] == "bounded result"
    assert result["production_network_enabled"] is False
    adapter.close()
    assert transport.closed is True


def test_remote_adapter_requires_explicit_scope_and_rejects_argument_widening():
    transport = FakeQLH(_result())
    adapter = QLHToolAdapter(QLHToolAdapterConfig("https://master.example"), transport=transport)

    with pytest.raises(RemoteToolError, match="explicit user opt-in") as denied:
        adapter.execute(_request(), allow_external=False)
    assert denied.value.code == "scope_denied"
    with pytest.raises(RemoteToolError, match="unknown fields") as invalid:
        adapter.execute(_request(arguments={"query": "QLH", "top_k": 2, "persist": True}), allow_external=True)
    assert invalid.value.code == "invalid_arguments"
    assert transport.calls == []


def test_remote_error_preserves_stable_code_and_retryability():
    remote_error = {
        "schema": REMOTE_RESULT_SCHEMA,
        "request_id": "req_remote_01",
        "status": "error",
        "error": {"code": "timeout", "message": "provider timed out", "retryable": True},
    }
    adapter = QLHToolAdapter(QLHToolAdapterConfig("https://master.example"), transport=FakeQLH(remote_error))

    with pytest.raises(RemoteToolError, match="provider timed out") as exc:
        adapter.execute(_request(), allow_external=True)
    assert exc.value.code == "timeout"
    assert exc.value.retryable is True


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda value: value.update({"schema": "wrong.schema"}), "invalid_remote_result"),
        (lambda value: value.update({"request_id": "req_other_01"}), "invalid_remote_result"),
        (lambda value: value["items"].__setitem__(0, {"title": "bad", "url": "https://127.0.0.1/private", "snippet": "x"}), "unsafe_url_host"),
        (lambda value: value["policy"].update({"bytes": 2_000_000}), "response_too_large"),
    ],
)
def test_remote_result_is_fail_closed(mutate, code):
    value = _result()
    mutate(value)
    adapter = QLHToolAdapter(QLHToolAdapterConfig("https://master.example"), transport=FakeQLH(value))

    with pytest.raises(RemoteToolError) as exc:
        adapter.execute(_request(), allow_external=True)
    assert exc.value.code == code


def test_remote_transport_failure_is_retryable_and_default_transport_is_offline():
    adapter = QLHToolAdapter(QLHToolAdapterConfig("https://master.example"))
    with pytest.raises(RemoteToolError) as exc:
        adapter.execute(_request(), allow_external=True)
    assert exc.value.code == "remote_transport_unavailable"
    assert exc.value.retryable is True

    class Broken:
        def post_json(self, path, payload):
            raise OSError("offline")

        def close(self):
            pass

    broken = QLHToolAdapter(QLHToolAdapterConfig("https://master.example"), transport=Broken())
    with pytest.raises(RemoteToolError) as broken_error:
        broken.execute(_request(), allow_external=True)
    assert broken_error.value.code == "remote_transport_error"
    assert broken_error.value.retryable is True
