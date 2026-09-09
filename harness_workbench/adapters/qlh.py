"""Small, explicit adapter for the QLH ``/api/chat`` contract."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .base import (
    AdapterCapabilities,
    AdapterError,
    AdapterModel,
    AdapterRequest,
    AdapterResponse,
    StreamChunk,
)


class QLHTransport(Protocol):
    def get_json(self, path: str) -> Mapping[str, Any]:
        ...

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def post_stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[str | bytes]:
        ...

    def close(self) -> None:
        ...


class QLHHttpTransport:
    """Lazy httpx transport; proxy/TLS policy remains caller-owned."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("QLH base_url must use http or https")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise AdapterError("httpx is required for the QLH adapter", code="adapter_dependency_missing", status_code=503) from exc
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def get_json(self, path: str) -> Mapping[str, Any]:
        return self._decode(self._client.get(path))

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._decode(self._client.post(path, json=dict(payload)))

    def post_stream(self, path: str, payload: Mapping[str, Any]) -> Iterable[str | bytes]:
        def iterator() -> Iterable[str]:
            try:
                with self._client.stream("POST", path, json=dict(payload)) as response:
                    response.raise_for_status()
                    yield from response.iter_lines()
            except Exception as exc:
                raise AdapterError("QLH streaming request failed", code="backend_http_error", status_code=502, retryable=True) from exc

        return iterator()

    @staticmethod
    def _decode(response: Any) -> Mapping[str, Any]:
        if response.status_code >= 400:
            raise AdapterError("QLH rejected the request", code="backend_http_error", status_code=502, retryable=response.status_code >= 500)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise AdapterError("QLH returned invalid JSON", code="backend_invalid_json", status_code=502) from exc
        if not isinstance(payload, Mapping):
            raise AdapterError("QLH returned a non-object response", code="backend_invalid_shape", status_code=502)
        return payload

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True, slots=True)
class QLHAdapterConfig:
    base_url: str
    model_id: str = "qlh-default"
    routing_preference: str = "auto"
    streaming_mode: str = "full"
    show_thinking: bool = False
    allow_external: bool = False
    prefer_external: bool = False
    max_context_chars: int = 30_000
    client_node_type: str = "harness"

    def __post_init__(self) -> None:
        if self.routing_preference not in {"auto", "local_only", "distributed_preferred", "distributed_required"}:
            raise ValueError("unsupported QLH routing_preference")
        if self.streaming_mode not in {"full", "fast", "interactive"}:
            raise ValueError("unsupported QLH streaming_mode")
        if not 512 <= self.max_context_chars <= 120_000:
            raise ValueError("max_context_chars must be between 512 and 120000")


class QLHAdapter:
    """Adapt harness chat requests without pretending QLH is OpenAI-native.

    QLH accepts one ``message`` rather than a messages array.  The adapter
    therefore serializes the explicit conversation into a bounded transcript;
    the transformation is included in capability evidence and never silently
    drops system/assistant turns.
    """

    def __init__(self, config: QLHAdapterConfig, *, transport: QLHTransport | None = None) -> None:
        self.config = config
        self._transport = transport or QLHHttpTransport(config.base_url)
        self._capabilities: AdapterCapabilities | None = None

    def capabilities(self) -> AdapterCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        try:
            status = self._transport.get_json("/api/status")
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("QLH capability probe failed", code="backend_probe_failed", status_code=502, retryable=True) from exc
        model_id = _status_model_id(status) or self.config.model_id
        self._capabilities = AdapterCapabilities(
            backend="qlh",
            model_ids=(model_id,),
            supports_stream=True,
            supports_num_ctx=False,
            supports_multimodal=False,
            supports_images=False,
            supports_cache_prompt=False,
            evidence={
                "source": "/api/status",
                "model_loaded": bool(status.get("model_loaded", status.get("loaded", False))),
                "routing_preference": self.config.routing_preference,
                "conversation_mapping": "bounded_role_transcript",
            },
        )
        return self._capabilities

    def models(self) -> tuple[AdapterModel, ...]:
        try:
            payload = self.model_catalog()
            rows = payload.get("models", [])
            if isinstance(rows, list):
                return tuple(
                    AdapterModel(
                        str(row.get("model_id") or row.get("id") or ""),
                        owned_by="qlh",
                        available=bool(row.get("is_available", True)),
                        unavailable_reason=str(row.get("unavailable_reason") or "") or None,
                    )
                    for row in rows
                    if isinstance(row, Mapping) and str(row.get("model_id") or row.get("id") or "").strip()
                )
        except AdapterError:
            pass
        capability = self.capabilities()
        return tuple(AdapterModel(model_id, owned_by="qlh") for model_id in capability.model_ids)

    def model_catalog(self) -> Mapping[str, Any]:
        """Return QLH's full model catalog, including assets not downloaded yet."""
        payload = self._transport.get_json("/api/models")
        models = payload.get("models", [])
        normalized: list[dict[str, Any]] = []
        if isinstance(models, list):
            for row in models:
                if not isinstance(row, Mapping):
                    continue
                item = dict(row)
                model_id = str(item.get("model_id") or item.get("id") or "").strip()
                if not model_id:
                    continue
                item["id"] = model_id
                item["available"] = bool(item.get("is_available", item.get("available", True)))
                if item.get("unavailable_reason"):
                    item["unavailable_reason"] = str(item["unavailable_reason"])
                normalized.append(item)
        result = dict(payload)
        result["models"] = normalized
        return result

    def model_presets(self) -> Mapping[str, Any]:
        return self._transport.get_json("/api/models/presets")

    def model_downloads(self) -> Mapping[str, Any]:
        return self._transport.get_json("/api/models/downloads")

    def queue_model_download(self, preset_id: str) -> Mapping[str, Any]:
        if not isinstance(preset_id, str) or not preset_id.strip():
            raise AdapterError("model preset is required", code="invalid_model_preset", status_code=400)
        return self._transport.post_json("/api/models/downloads", {"preset_id": preset_id.strip()})

    def load_model_asset(self, model_id: str, *, engine: str = "auto", quant_type: str = "int4") -> Mapping[str, Any]:
        if not isinstance(model_id, str) or not model_id.strip():
            raise AdapterError("model id is required", code="invalid_model_id", status_code=400)
        return self._transport.post_json(
            "/api/models/load",
            {"model_id": model_id.strip(), "engine": engine or "auto", "quant_type": quant_type or "int4"},
        )

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        payload = self._payload(request, streaming_mode="full")
        try:
            value = self._transport.post_json("/api/chat", payload)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("QLH chat request failed", code="backend_http_error", status_code=502, retryable=True) from exc
        return self._decode_response(value, request.model)

    def stream(self, request: AdapterRequest) -> Iterable[StreamChunk]:
        payload = self._payload(request, streaming_mode=self.config.streaming_mode)
        emitted_token = False
        try:
            lines = self._transport.post_stream("/api/chat/stream", payload)
            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="strict") if isinstance(raw_line, bytes) else raw_line
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AdapterError("QLH returned invalid SSE JSON", code="backend_invalid_sse", status_code=502) from exc
                if not isinstance(event, Mapping):
                    raise AdapterError("QLH returned an invalid SSE event", code="backend_invalid_shape", status_code=502)
                if event.get("error"):
                    raise AdapterError(str(event["error"]), code="backend_generation_failed", status_code=502)
                token = event.get("token")
                if isinstance(token, str) and token:
                    emitted_token = True
                    yield StreamChunk(
                        id=str(event.get("generation_id") or request.request_id or ""),
                        model=request.model,
                        delta={"content": token},
                        created=int(time.time()),
                    )
                if event.get("done"):
                    response_text = event.get("response")
                    if isinstance(response_text, str) and response_text and not emitted_token:
                        yield StreamChunk(
                            id=str(event.get("generation_id") or request.request_id or ""),
                            model=request.model,
                            delta={"content": response_text},
                            created=int(time.time()),
                        )
                    yield StreamChunk(
                        id=str(event.get("generation_id") or request.request_id or ""),
                        model=request.model,
                        delta={},
                        finish_reason="stop",
                        usage=_usage_from_metrics(event.get("metrics")),
                        created=int(time.time()),
                    )
                    return
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError("QLH streaming request failed", code="backend_http_error", status_code=502, retryable=True) from exc

    def close(self) -> None:
        self._transport.close()

    def _payload(self, request: AdapterRequest, *, streaming_mode: str) -> dict[str, Any]:
        transcript = _bounded_transcript(request.messages, max_chars=self.config.max_context_chars)
        payload: dict[str, Any] = {
            "model": request.model,
            "message": transcript,
            "max_new_tokens": request.max_tokens or 1024,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "top_p": request.top_p if request.top_p is not None else 0.9,
            "show_thinking": self.config.show_thinking,
            "streaming_mode": streaming_mode,
            "routing_preference": self.config.routing_preference,
            "client_node_type": self.config.client_node_type,
            "allow_external": self.config.allow_external,
            "prefer_external": self.config.prefer_external,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    @staticmethod
    def _decode_response(value: Mapping[str, Any], requested_model: str) -> AdapterResponse:
        content = value.get("content")
        if not isinstance(content, str):
            raise AdapterError("QLH chat response has no text content", code="backend_invalid_shape", status_code=502)
        metrics = value.get("metrics", {})
        return AdapterResponse(
            id=str(value.get("generation_id") or value.get("request_id") or ""),
            model=str(value.get("model") or requested_model),
            content=content,
            finish_reason="stop",
            usage=_usage_from_metrics(metrics),
            created=int(time.time()),
        )


def _bounded_transcript(messages: Iterable[Mapping[str, Any]], *, max_chars: int) -> str:
    blocks: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        if not isinstance(content, str):
            content = str(content)
        blocks.append(f"[{role}]\n{content}")
    transcript = "\n\n".join(blocks)
    if len(transcript) <= max_chars:
        return transcript
    # Keep the system message and newest turns.  This is explicit rather than
    # a silent backend truncation, and the bounded marker is visible to QLH.
    first = blocks[0] if blocks and blocks[0].startswith("[system]") else ""
    suffix = "\n\n[context] older turns omitted by harness budget\n\n"
    if len(first) + len(suffix) >= max_chars:
        return (suffix + first)[-max_chars:]
    tail_budget = max_chars - len(first) - len(suffix)
    tail = "\n\n".join(blocks[1:] if first else blocks)
    return (first + suffix + tail[-max(0, tail_budget):]).strip()


def _status_model_id(status: Mapping[str, Any]) -> str:
    for key in ("model_id", "current_model", "model"):
        value = status.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("id") or value.get("model_id")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _usage_from_metrics(metrics: Any) -> dict[str, int]:
    if not isinstance(metrics, Mapping):
        return {}
    mapping = {
        "prompt_tokens": ("prompt_tokens", "total_prompt_tokens"),
        "completion_tokens": ("completion_tokens", "generated_tokens", "total_generated_tokens"),
    }
    result: dict[str, int] = {}
    for output_name, keys in mapping.items():
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                result[output_name] = value
                break
    if "prompt_tokens" in result and "completion_tokens" in result:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


__all__ = ["QLHAdapter", "QLHAdapterConfig", "QLHHttpTransport", "QLHTransport"]
