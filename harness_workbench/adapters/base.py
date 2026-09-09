"""Backend-neutral request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol


class AdapterError(RuntimeError):
    """Stable error returned by an adapter without leaking backend internals."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "adapter_error",
        status_code: int = 502,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class AdapterModel:
    id: str
    owned_by: str = "harness"
    created: int | None = None
    available: bool = True
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.id, "object": "model", "owned_by": self.owned_by}
        if self.created is not None:
            value["created"] = self.created
        value["available"] = self.available
        if self.unavailable_reason:
            value["unavailable_reason"] = self.unavailable_reason
        return value


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    backend: str
    model_ids: tuple[str, ...] = ()
    supports_stream: bool = True
    supports_num_ctx: bool = False
    supports_multimodal: bool = False
    supports_images: bool = False
    supports_cache_prompt: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_ids": list(self.model_ids),
            "supports_stream": self.supports_stream,
            "supports_num_ctx": self.supports_num_ctx,
            "supports_multimodal": self.supports_multimodal,
            "supports_images": self.supports_images,
            "supports_cache_prompt": self.supports_cache_prompt,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class AdapterRequest:
    model: str
    messages: tuple[Mapping[str, Any], ...]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    num_ctx: int | None = None
    cache_prompt: bool | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("adapter request model is required")
        if not self.messages:
            raise ValueError("adapter request messages are required")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.num_ctx is not None and self.num_ctx <= 0:
            raise ValueError("num_ctx must be positive")


@dataclass(frozen=True, slots=True)
class AdapterResponse:
    id: str
    model: str
    content: str
    finish_reason: str | None = "stop"
    usage: Mapping[str, int] = field(default_factory=dict)
    created: int | None = None


@dataclass(frozen=True, slots=True)
class StreamChunk:
    id: str
    model: str
    delta: Mapping[str, Any]
    finish_reason: str | None = None
    created: int | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


class ChatAdapter(Protocol):
    def capabilities(self) -> AdapterCapabilities:
        ...

    def models(self) -> tuple[AdapterModel, ...]:
        ...

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        ...

    def stream(self, request: AdapterRequest) -> Iterable[StreamChunk]:
        ...

    def close(self) -> None:
        ...
