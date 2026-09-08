"""Strict request/response mapping at the harness API boundary."""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

from ..adapters.base import AdapterRequest, AdapterResponse, StreamChunk


class APIRequestError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_request", status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


_MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}
_EXTRA_BODY_FIELDS = {"num_ctx", "cache_prompt"}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise APIRequestError(f"{name} must be a positive integer", code="invalid_parameter")
    return value


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= float(value) <= upper:
        raise APIRequestError(f"{name} must be between {lower} and {upper}", code="invalid_parameter")
    return float(value)


def _messages(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise APIRequestError("messages must be a non-empty array", code="invalid_messages")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise APIRequestError(f"messages[{index}] must be an object", code="invalid_messages")
        role = item.get("role")
        if role not in _MESSAGE_ROLES:
            raise APIRequestError(f"messages[{index}] has an unsupported role", code="invalid_messages")
        content = item.get("content", "")
        if not isinstance(content, (str, list, type(None))):
            raise APIRequestError(f"messages[{index}].content must be text or content parts", code="invalid_messages")
        normalized = dict(item)
        if role == "developer":
            normalized["role"] = "system"
        if content is None:
            normalized["content"] = ""
        result.append(normalized)
    return tuple(result)


def parse_chat_request(payload: Mapping[str, Any], *, request_id: str | None = None) -> AdapterRequest:
    if not isinstance(payload, Mapping):
        raise APIRequestError("request body must be an object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise APIRequestError("model is required", code="missing_model")
    extra_body = payload.get("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, Mapping):
        raise APIRequestError("extra_body must be an object", code="invalid_parameter")
    unknown_extra = set(extra_body) - _EXTRA_BODY_FIELDS
    if unknown_extra:
        raise APIRequestError(
            "unsupported extra_body fields: " + ", ".join(sorted(str(item) for item in unknown_extra)),
            code="unsupported_parameter",
        )

    max_tokens_value = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if "max_tokens" in payload and "max_completion_tokens" in payload and payload["max_tokens"] != payload["max_completion_tokens"]:
        raise APIRequestError("max_tokens and max_completion_tokens disagree", code="invalid_parameter")
    max_tokens = _positive_int(max_tokens_value, "max_tokens") if max_tokens_value is not None else None
    temperature = _number(payload["temperature"], "temperature", lower=0.0, upper=2.0) if "temperature" in payload else None
    top_p = _number(payload["top_p"], "top_p", lower=0.0, upper=1.0) if "top_p" in payload else None
    stop_value = payload.get("stop", ())
    if isinstance(stop_value, str):
        stop = (stop_value,)
    elif isinstance(stop_value, list) and all(isinstance(item, str) for item in stop_value):
        stop = tuple(stop_value)
    elif stop_value in (None, ()):
        stop = ()
    else:
        raise APIRequestError("stop must be a string or string array", code="invalid_parameter")
    if len(stop) > 4:
        raise APIRequestError("stop may contain at most four strings", code="invalid_parameter")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise APIRequestError("stream must be boolean", code="invalid_parameter")
    num_ctx = extra_body.get("num_ctx", payload.get("num_ctx"))
    if "num_ctx" in extra_body and "num_ctx" in payload and extra_body["num_ctx"] != payload["num_ctx"]:
        raise APIRequestError("num_ctx values disagree", code="invalid_parameter")
    if num_ctx is not None:
        num_ctx = _positive_int(num_ctx, "num_ctx")
    cache_prompt = extra_body.get("cache_prompt", payload.get("cache_prompt"))
    if cache_prompt is not None and not isinstance(cache_prompt, bool):
        raise APIRequestError("cache_prompt must be boolean", code="invalid_parameter")
    return AdapterRequest(
        model=model,
        messages=_messages(payload.get("messages")),
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        num_ctx=num_ctx,
        cache_prompt=cache_prompt,
        request_id=request_id or uuid.uuid4().hex,
    )


def response_to_openai(response: AdapterResponse, *, fallback_id: str | None = None) -> dict[str, Any]:
    response_id = response.id or fallback_id or f"chatcmpl-{uuid.uuid4().hex}"
    created = response.created or int(time.time())
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": response.finish_reason,
            }
        ],
        "usage": dict(response.usage),
    }


def chunk_to_openai(chunk: StreamChunk, *, fallback_id: str, fallback_model: str) -> dict[str, Any]:
    return {
        "id": chunk.id or fallback_id,
        "object": "chat.completion.chunk",
        "created": chunk.created or int(time.time()),
        "model": chunk.model or fallback_model,
        "choices": [
            {
                "index": 0,
                "delta": dict(chunk.delta),
                "finish_reason": chunk.finish_reason,
            }
        ],
        **({"usage": dict(chunk.usage)} if chunk.usage else {}),
    }
