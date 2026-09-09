"""S2 API mapping and llama-server adapter contract tests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pytest

from harness_workbench.adapters.base import (
    AdapterCapabilities,
    AdapterError,
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    StreamChunk,
)
from harness_workbench.adapters.llama_server import (
    LlamaServerAdapter,
    LlamaServerConfig,
    LlamaServerProcess,
)
from harness_workbench.api_layer import create_app
from harness_workbench.api_layer.mapping import APIRequestError, parse_chat_request


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    def get_json(self, path: str) -> Mapping[str, Any]:
        self.calls.append(("get", path))
        if path == "/props":
            return {"n_ctx": 4096, "cache_prompt": True, "mmproj": False, "chat_template": "fixture"}
        if path == "/v1/models":
            return {"data": [{"id": "QW1.8B", "owned_by": "llama-server", "created": 1}]}
        raise AssertionError(path)

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(("post", path, dict(payload)))
        return {
            "id": "cmpl_fixture",
            "model": payload["model"],
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }

    def post_stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[str]:
        self.calls.append(("stream", path, dict(payload)))
        yield 'data: {"id":"cmpl_stream","model":"QW1.8B","choices":[{"delta":{"role":"assistant","content":"hel"},"finish_reason":null}]}'
        yield 'data: {"id":"cmpl_stream","model":"QW1.8B","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}'
        yield "data: [DONE]"

    def close(self) -> None:
        self.closed = True


def _config() -> LlamaServerConfig:
    return LlamaServerConfig(model="models/QW1.8B.gguf", context_size=4096, max_new_tokens=256)


def test_parse_chat_request_maps_explicit_context_and_developer_role() -> None:
    request = parse_chat_request(
        {
            "model": "QW1.8B",
            "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}],
            "max_completion_tokens": 128,
            "extra_body": {"num_ctx": 4096, "cache_prompt": True},
            "stream": True,
        },
        request_id="req-1",
    )
    assert request.messages[0]["role"] == "system"
    assert request.max_tokens == 128
    assert request.num_ctx == 4096
    assert request.cache_prompt is True
    assert request.request_id == "req-1"


def test_parse_chat_request_rejects_unknown_extra_body_and_conflicts() -> None:
    with pytest.raises(APIRequestError, match="unsupported extra_body"):
        parse_chat_request({"model": "m", "messages": [{"role": "user", "content": "x"}], "extra_body": {"n_ctx": 1}})
    with pytest.raises(APIRequestError, match="num_ctx values disagree"):
        parse_chat_request({"model": "m", "messages": [{"role": "user", "content": "x"}], "num_ctx": 1, "extra_body": {"num_ctx": 2}})


def test_llama_adapter_maps_completion_and_stream_contract() -> None:
    transport = FakeTransport()
    adapter = LlamaServerAdapter(_config(), transport=transport)
    models = adapter.models()
    assert models[0].id == "QW1.8B"
    caps = adapter.capabilities()
    assert caps.supports_num_ctx is False
    request = AdapterRequest(model="QW1.8B", messages=({"role": "user", "content": "hi"},), max_tokens=8, num_ctx=4096)
    response = adapter.complete(request)
    assert response.content == "hello"
    chunks = list(adapter.stream(request))
    assert "".join(str(chunk.delta.get("content", "")) for chunk in chunks) == "hello"
    post = [call for call in transport.calls if call[0] == "post"]
    assert post[0][2]["max_tokens"] == 8
    assert post[0][2]["stream"] is False
    assert transport.calls[-1][0] == "stream"


def test_llama_adapter_rejects_context_size_that_requires_restart() -> None:
    adapter = LlamaServerAdapter(_config(), transport=FakeTransport())
    request = AdapterRequest(model="QW1.8B", messages=({"role": "user", "content": "hi"},), num_ctx=8192)
    with pytest.raises(AdapterError) as exc_info:
        adapter.complete(request)
    assert exc_info.value.code == "num_ctx_process_scoped"


def test_process_builds_explicit_command_and_waits_for_health() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            self.terminated = True

    process = FakeProcess()

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    server = LlamaServerProcess(
        _config(),
        popen_factory=popen,
        health_checker=lambda _url: True,
        sleep=lambda _seconds: None,
    )
    server.start()
    assert server.is_running
    command = calls[0][0]
    assert "--ctx-size" in command and "4096" in command
    assert "--cache-prompt" in command and "--jinja" in command
    server.stop()
    assert process.terminated is True


class FakeAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[AdapterRequest] = []

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(backend="fake", model_ids=("fixture",))

    def models(self) -> tuple[AdapterModel, ...]:
        return (AdapterModel(id="fixture"),)

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        self.requests.append(request)
        if self.fail:
            raise AdapterError("backend unavailable", code="backend_down", status_code=503, retryable=True)
        return AdapterResponse(id="fixture-id", model=request.model, content="ok")

    def stream(self, request: AdapterRequest) -> Iterable[StreamChunk]:
        self.requests.append(request)
        if self.fail:
            raise AdapterError("backend unavailable", code="backend_down", status_code=503, retryable=True)
        yield StreamChunk(id="fixture-stream", model=request.model, delta={"role": "assistant", "content": "ok"})
        yield StreamChunk(id="fixture-stream", model=request.model, delta={}, finish_reason="stop")

    def close(self) -> None:
        return None


def test_fastapi_app_exposes_models_completion_and_sse() -> None:
    from fastapi.testclient import TestClient

    adapter = FakeAdapter()
    client = TestClient(create_app(adapter))
    assert client.get("/v1/models").json()["data"][0]["id"] == "fixture"
    profiles = client.get("/v1/model-profiles")
    assert profiles.status_code == 200
    assert "Qwen2.5-0.5B" in {item["model_id"] for item in profiles.json()["profiles"]}
    response = client.post("/v1/chat/completions", json={"model": "fixture", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    stream = client.post("/v1/chat/completions", json={"model": "fixture", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert stream.status_code == 200
    assert "chat.completion.chunk" in stream.text
    assert "data: [DONE]" in stream.text
    assert len(adapter.requests) == 2


def test_fastapi_app_returns_stable_backend_error() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(FakeAdapter(fail=True)))
    response = client.post("/v1/chat/completions", json={"model": "fixture", "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_down"
