"""Backend-neutral image contracts for the harness workbench."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..adapters.base import AdapterError


class ImageRequestError(ValueError):
    """Raised when an image request does not satisfy the public contract."""

    def __init__(self, message: str, *, code: str = "invalid_image_request") -> None:
        super().__init__(message)
        self.code = code


class ImageAdapterError(AdapterError):
    """Stable error from a local or remote image adapter."""


@dataclass(frozen=True, slots=True)
class ImageRequest:
    prompt: str
    model: str | None = None
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 28
    guidance_scale: float = 7.5
    seed: int | None = None
    response_format: str = "b64_json"
    user: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ImageRequest":
        if not isinstance(payload, Mapping):
            raise ImageRequestError("request body must be an object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageRequestError("prompt is required", code="missing_prompt")
        if len(prompt) > 4000:
            raise ImageRequestError("prompt must be at most 4000 characters")
        negative = payload.get("negative_prompt", "")
        if negative is None:
            negative = ""
        if not isinstance(negative, str) or len(negative) > 4000:
            raise ImageRequestError("negative_prompt must be text of at most 4000 characters")

        width, height = _size(payload)
        steps = _int(payload.get("steps", 28), "steps", lower=1, upper=100)
        guidance = _number(payload.get("guidance_scale", 7.5), "guidance_scale", lower=0, upper=30)
        seed = payload.get("seed")
        if seed is not None:
            seed = _int(seed, "seed", lower=-(2**63), upper=2**63 - 1)
        response_format = payload.get("response_format", "b64_json")
        if response_format not in {"b64_json", "url"}:
            raise ImageRequestError("response_format must be b64_json or url")
        user = payload.get("user")
        if user is not None and (not isinstance(user, str) or len(user) > 128):
            raise ImageRequestError("user must be a string of at most 128 characters")
        model = payload.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 128):
            raise ImageRequestError("model must be a non-empty string of at most 128 characters")
        metadata = payload.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ImageRequestError("metadata must be an object")
        return cls(
            prompt=prompt,
            model=model.strip() if isinstance(model, str) else None,
            negative_prompt=negative,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            response_format=response_format,
            user=user,
            metadata=dict(metadata),
        )

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ImageRequestError("prompt is required", code="missing_prompt")
        _validate_dimensions(self.width, self.height)
        _int(self.steps, "steps", lower=1, upper=100)
        _number(self.guidance_scale, "guidance_scale", lower=0, upper=30)
        if self.response_format not in {"b64_json", "url"}:
            raise ImageRequestError("response_format must be b64_json or url")

    def as_remote_payload(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "preset_id": self.model,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    data: bytes
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("generated image data must be non-empty bytes")
        if not self.mime_type.startswith("image/"):
            raise ValueError("generated image mime_type must be an image type")


@dataclass(frozen=True, slots=True)
class ImageAdapterCapabilities:
    backend: str
    model_ids: tuple[str, ...] = ()
    supports_txt2img: bool = False
    supports_edit: bool = False
    supports_distributed: bool = False
    runtime_available: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_ids": list(self.model_ids),
            "supports_txt2img": self.supports_txt2img,
            "supports_edit": self.supports_edit,
            "supports_distributed": self.supports_distributed,
            "runtime_available": self.runtime_available,
            "evidence": dict(self.evidence),
        }


class ImageAdapter(Protocol):
    def capabilities(self) -> ImageAdapterCapabilities:
        ...

    def generate(self, request: ImageRequest) -> GeneratedImage:
        ...

    def close(self) -> None:
        ...


def _int(value: Any, name: str, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ImageRequestError(f"{name} must be an integer between {lower} and {upper}", code="invalid_parameter")
    return value


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ImageRequestError(f"{name} must be a number between {lower} and {upper}", code="invalid_parameter")
    value = float(value)
    if not lower <= value <= upper:
        raise ImageRequestError(f"{name} must be between {lower} and {upper}", code="invalid_parameter")
    return value


def _size(payload: Mapping[str, Any]) -> tuple[int, int]:
    if "size" in payload:
        size = payload["size"]
        if not isinstance(size, str) or "x" not in size.lower():
            raise ImageRequestError("size must use WIDTHxHEIGHT syntax")
        left, right = size.lower().split("x", 1)
        try:
            width, height = int(left), int(right)
        except ValueError as exc:
            raise ImageRequestError("size must use WIDTHxHEIGHT syntax") from exc
    else:
        width = payload.get("width", 512)
        height = payload.get("height", 512)
    if isinstance(width, bool) or not isinstance(width, int) or isinstance(height, bool) or not isinstance(height, int):
        raise ImageRequestError("width and height must be integers", code="invalid_parameter")
    _validate_dimensions(width, height)
    return width, height


def _validate_dimensions(width: int, height: int) -> None:
    if not 64 <= width <= 768 or not 64 <= height <= 768 or width % 8 or height % 8:
        raise ImageRequestError("width and height must be multiples of 8 between 64 and 768", code="invalid_parameter")
