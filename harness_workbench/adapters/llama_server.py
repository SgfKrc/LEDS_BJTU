"""OpenAI-compatible llama-server adapter and local process lifecycle."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .base import (
    AdapterCapabilities,
    AdapterError,
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    StreamChunk,
)


class LlamaTransport(Protocol):
    def get_json(self, path: str) -> Mapping[str, Any]:
        ...

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def post_stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[str | bytes]:
        ...

    def close(self) -> None:
        ...


class HttpxTransport:
    """Small synchronous httpx wrapper kept behind an injectable protocol."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise AdapterError(
                "httpx is required for the llama-server adapter",
                code="adapter_dependency_missing",
                status_code=503,
            ) from exc
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def get_json(self, path: str) -> Mapping[str, Any]:
        response = self._client.get(path)
        return self._decode(response)

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self._client.post(path, json=dict(payload))
        return self._decode(response)

    def post_stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[str | bytes]:
        def iterator() -> Iterable[str]:
            try:
                with self._client.stream("POST", path, json=dict(payload)) as response:
                    response.raise_for_status()
                    yield from response.iter_lines()
            except Exception as exc:
                if isinstance(exc, AdapterError):
                    raise
                raise AdapterError(
                    "llama-server rejected the streaming request",
                    code="backend_http_error",
                    status_code=502,
                    retryable=True,
                ) from exc

        return iterator()

    @staticmethod
    def _decode(response: Any) -> Mapping[str, Any]:
        if response.status_code >= 400:
            raise AdapterError(
                "llama-server rejected the request",
                code="backend_http_error",
                status_code=502,
                retryable=response.status_code >= 500,
            )
        try:
            value = response.json()
        except (ValueError, TypeError) as exc:
            raise AdapterError(
                "llama-server returned invalid JSON",
                code="backend_invalid_json",
                status_code=502,
            ) from exc
        if not isinstance(value, Mapping):
            raise AdapterError("llama-server returned a non-object response", code="backend_invalid_shape")
        return value

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True, slots=True)
class LlamaServerConfig:
    executable: str | Path = "llama-server"
    model: str | Path = ""
    host: str = "127.0.0.1"
    port: int = 8080
    context_size: int = 4096
    max_new_tokens: int = 768
    mmproj: str | Path | None = None
    enable_jinja: bool = True
    cache_prompt: bool = True
    extra_args: tuple[str, ...] = ()
    startup_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("llama-server model path is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.context_size <= 0 or self.max_new_tokens <= 0:
            raise ValueError("context_size and max_new_tokens must be positive")
        if self.startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

    @property
    def base_url(self) -> str:
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    @property
    def model_name(self) -> str:
        return Path(str(self.model)).stem or "local-model"

    def command(self) -> tuple[str, ...]:
        command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--n-predict",
            str(self.max_new_tokens),
        ]
        if self.enable_jinja:
            command.append("--jinja")
        if self.cache_prompt:
            command.append("--cache-prompt")
        if self.mmproj is not None:
            command.extend(("--mmproj", str(self.mmproj)))
        command.extend(self.extra_args)
        return tuple(command)


PopenFactory = Callable[..., subprocess.Popen[Any]]
HealthChecker = Callable[[str], bool]


class LlamaServerProcess:
    """Start/stop a llama-server child without hiding process failures."""

    def __init__(
        self,
        config: LlamaServerConfig,
        *,
        popen_factory: PopenFactory | None = None,
        health_checker: HealthChecker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._popen_factory = popen_factory or subprocess.Popen
        self._health_checker = health_checker or self._default_health_check
        self._sleep = sleep
        self._clock = clock
        self._process: subprocess.Popen[Any] | None = None

    @property
    def process(self) -> subprocess.Popen[Any] | None:
        return self._process

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, wait_ready: bool = True) -> None:
        if self.is_running:
            return
        try:
            self._process = self._popen_factory(
                list(self.config.command()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise AdapterError(
                "could not start llama-server",
                code="backend_process_start_failed",
                status_code=503,
            ) from exc
        if not wait_ready:
            return
        deadline = self._clock() + self.config.startup_timeout
        while self._clock() < deadline:
            if self._process.poll() is not None:
                self.stop()
                raise AdapterError(
                    "llama-server exited before becoming ready",
                    code="backend_process_exited",
                    status_code=503,
                )
            try:
                if self._health_checker(self.config.base_url):
                    return
            except Exception:
                pass
            self._sleep(0.05)
        self.stop()
        raise AdapterError(
            "llama-server did not become ready before the startup deadline",
            code="backend_start_timeout",
            status_code=503,
            retryable=True,
        )

    def stop(self, *, timeout: float = 5.0) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)

    def __enter__(self) -> "LlamaServerProcess":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @staticmethod
    def _default_health_check(base_url: str) -> bool:
        request = urllib.request.Request(base_url.rstrip("/") + "/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=0.5) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError):
            return False


class LlamaServerAdapter:
    """Adapt llama-server's OpenAI-compatible surface to the harness contract."""

    def __init__(
        self,
        config: LlamaServerConfig,
        *,
        transport: LlamaTransport | None = None,
        process: LlamaServerProcess | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or HttpxTransport(config.base_url)
        self._process = process

    def start(self) -> None:
        if self._process is not None:
            self._process.start()

    def capabilities(self) -> AdapterCapabilities:
        props = self._transport.get_json("/props")
        if not isinstance(props, Mapping):
            raise AdapterError("llama-server /props response is invalid", code="backend_invalid_shape")
        return AdapterCapabilities(
            backend="llama_server",
            model_ids=(self.config.model_name,),
            supports_stream=True,
            supports_num_ctx=False,
            supports_multimodal=bool(props.get("mmproj") or props.get("supports_multimodal")),
            supports_images=False,
            supports_cache_prompt=bool(props.get("cache_prompt", self.config.cache_prompt)),
            evidence={
                "source": "/props",
                "context_size": props.get("n_ctx", self.config.context_size),
                "chat_template": bool(props.get("chat_template")),
            },
        )

    def models(self) -> tuple[AdapterModel, ...]:
        value = self._transport.get_json("/v1/models")
        data = value.get("data") if isinstance(value, Mapping) else None
        if not isinstance(data, list):
            raise AdapterError("llama-server /v1/models response is invalid", code="backend_invalid_shape")
        models: list[AdapterModel] = []
        for item in data:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                raise AdapterError("llama-server returned an invalid model entry", code="backend_invalid_shape")
            models.append(
                AdapterModel(
                    id=item["id"],
                    owned_by=str(item.get("owned_by", "llama-server")),
                    created=item.get("created") if isinstance(item.get("created"), int) else None,
                )
            )
        return tuple(models)

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        payload = self._payload(request, stream=False)
        value = self._transport.post_json("/v1/chat/completions", payload)
        return self._decode_response(value, request.model)

    def stream(self, request: AdapterRequest) -> Iterable[StreamChunk]:
        payload = self._payload(request, stream=True)
        for line in self._transport.post_stream("/v1/chat/completions", payload):
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="strict")
            text = line.strip()
            if not text:
                continue
            if text.startswith("data:"):
                text = text[5:].strip()
            if text == "[DONE]":
                return
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AdapterError("llama-server returned invalid SSE JSON", code="backend_invalid_sse") from exc
            if not isinstance(value, Mapping):
                raise AdapterError("llama-server returned an invalid SSE chunk", code="backend_invalid_shape")
            choices = value.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise AdapterError("llama-server returned an invalid SSE choice", code="backend_invalid_shape")
            delta = choice.get("delta", {})
            if not isinstance(delta, Mapping):
                raise AdapterError("llama-server returned an invalid SSE delta", code="backend_invalid_shape")
            yield StreamChunk(
                id=str(value.get("id", "")),
                model=str(value.get("model", request.model)),
                delta=dict(delta),
                finish_reason=choice.get("finish_reason"),
                created=value.get("created") if isinstance(value.get("created"), int) else None,
                usage=value.get("usage", {}) if isinstance(value.get("usage"), Mapping) else {},
            )

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            if self._process is not None:
                self._process.stop()

    def _payload(self, request: AdapterRequest, *, stream: bool) -> dict[str, Any]:
        if request.num_ctx is not None and request.num_ctx != self.config.context_size:
            raise AdapterError(
                "num_ctx is process-scoped; restart llama-server with the requested context size",
                code="num_ctx_process_scoped",
                status_code=400,
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [dict(message) for message in request.messages],
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.cache_prompt is not None:
            payload["cache_prompt"] = request.cache_prompt
        return payload

    @staticmethod
    def _decode_response(value: Mapping[str, Any], requested_model: str) -> AdapterResponse:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise AdapterError("llama-server returned an invalid completion", code="backend_invalid_shape")
        message = choices[0].get("message", {})
        if not isinstance(message, Mapping):
            raise AdapterError("llama-server returned an invalid completion message", code="backend_invalid_shape")
        content = message.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise AdapterError("llama-server completion content is not text", code="backend_invalid_shape")
        usage = value.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        return AdapterResponse(
            id=str(value.get("id", "")),
            model=str(value.get("model", requested_model)),
            content=content,
            finish_reason=choices[0].get("finish_reason"),
            usage={key: int(item) for key, item in usage.items() if isinstance(item, int) and not isinstance(item, bool)},
            created=value.get("created") if isinstance(value.get("created"), int) else None,
        )
