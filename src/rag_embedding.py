"""Bounded local embedding providers for RAG-S2.

The provider contract is deliberately independent of the RAG SQLite store.
Ollama and the native llama.cpp engine produce the same validated result;
neither provider silently downloads models, changes the main runtime venv, or
falls back from one backend without an explicit caller choice.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_TEXTS = 64
MAX_TEXT_CHARS = 16_384
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"


class EmbeddingProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model_id: str
    dimensions: int
    vectors: list[list[float]]


def _validate_inputs(texts: Sequence[str]) -> list[str]:
    if not isinstance(texts, (list, tuple)) or not texts or len(texts) > MAX_TEXTS:
        raise EmbeddingProviderError("input_invalid", "embedding input must contain 1-64 texts")
    normalized = []
    for text in texts:
        if not isinstance(text, str) or not text or len(text) > MAX_TEXT_CHARS or "\x00" in text:
            raise EmbeddingProviderError("input_invalid", "embedding text is empty, oversized, or contains NUL")
        normalized.append(text)
    return normalized


def _validate_vectors(raw: Any, expected_dimensions: int | None, count: int) -> tuple[list[list[float]], int]:
    if not isinstance(raw, list) or len(raw) != count:
        raise EmbeddingProviderError("response_invalid", "embedding provider returned the wrong vector count")
    vectors: list[list[float]] = []
    dimensions: int | None = None
    for vector in raw:
        if not isinstance(vector, list) or not vector:
            raise EmbeddingProviderError("response_invalid", "embedding provider returned an invalid vector")
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions:
            raise EmbeddingProviderError("dimension_mismatch", "embedding vectors have inconsistent dimensions")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError("response_invalid", "embedding vector contains a non-number") from exc
        if any(not math.isfinite(value) for value in values):
            raise EmbeddingProviderError("response_invalid", "embedding vector contains a non-finite value")
        vectors.append(values)
    assert dimensions is not None
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise EmbeddingProviderError(
            "dimension_mismatch",
            f"embedding dimension {dimensions} does not match expected {expected_dimensions}",
        )
    return vectors, dimensions


class OllamaEmbeddingProvider:
    """Local Ollama `/api/embed` provider with bounded, injectable transport."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
        base_url: str = "http://127.0.0.1:11434",
        expected_dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        requester: Callable[[str, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 128:
            raise EmbeddingProviderError("config_invalid", "Ollama model_id is required")
        if expected_dimensions is not None and (isinstance(expected_dimensions, bool) or not 1 <= int(expected_dimensions) <= 32_768):
            raise EmbeddingProviderError("config_invalid", "expected embedding dimensions are invalid")
        if not 0 < float(timeout_seconds) <= 300:
            raise EmbeddingProviderError("config_invalid", "embedding timeout is outside the allowed range")
        base_url = str(base_url).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise EmbeddingProviderError("config_invalid", "Ollama base_url must be HTTP(S)")
        self.model_id = model_id.strip()
        self.base_url = base_url
        self.expected_dimensions = int(expected_dimensions) if expected_dimensions is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self._requester = requester

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._requester is not None:
            try:
                result = self._requester(f"{self.base_url}/api/embed", payload, self.timeout_seconds)
            except EmbeddingProviderError:
                raise
            except TimeoutError as exc:
                raise EmbeddingProviderError("provider_timeout", "Ollama embedding request timed out", retryable=True) from exc
            except OSError as exc:
                raise EmbeddingProviderError("provider_unavailable", "Ollama embedding endpoint is unavailable", retryable=True) from exc
            if not isinstance(result, Mapping):
                raise EmbeddingProviderError("response_invalid", "Ollama returned a non-object response")
            return result
        encoded = json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed", data=encoded,
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError as exc:
            raise EmbeddingProviderError("provider_timeout", "Ollama embedding request timed out", retryable=True) from exc
        except HTTPError as exc:
            raise EmbeddingProviderError("provider_http_error", "Ollama embedding endpoint rejected the request", retryable=exc.code >= 500) from exc
        except (URLError, OSError) as exc:
            raise EmbeddingProviderError("provider_unavailable", "Ollama embedding endpoint is unavailable", retryable=True) from exc
        if len(body) > MAX_RESPONSE_BYTES:
            raise EmbeddingProviderError("response_oversize", "Ollama embedding response exceeds the local limit")
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingProviderError("response_invalid", "Ollama returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            raise EmbeddingProviderError("response_invalid", "Ollama returned a non-object response")
        return result

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        texts = _validate_inputs(texts)
        payload = {"model": self.model_id, "input": texts}
        result = self._request(payload)
        if result.get("error"):
            raise EmbeddingProviderError("provider_rejected", "Ollama rejected the embedding request")
        vectors, dimensions = _validate_vectors(result.get("embeddings"), self.expected_dimensions, len(texts))
        return EmbeddingResult(self.provider_name, self.model_id, dimensions, vectors)


class NativeLlamaEmbeddingProvider:
    """Adapter for an explicitly embedding-enabled ``LlamaCppEngine``."""

    provider_name = "llama.cpp"

    def __init__(self, engine: Any, *, model_id: str, expected_dimensions: int) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise EmbeddingProviderError("config_invalid", "native embedding model_id is required")
        if isinstance(expected_dimensions, bool) or not 1 <= int(expected_dimensions) <= 32_768:
            raise EmbeddingProviderError("config_invalid", "native embedding dimensions are invalid")
        self.engine = engine
        self.model_id = model_id.strip()
        self.expected_dimensions = int(expected_dimensions)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        texts = _validate_inputs(texts)
        if not callable(getattr(self.engine, "embed_texts", None)):
            raise EmbeddingProviderError("capability_unavailable", "native llama engine has no embedding capability")
        try:
            vectors = self.engine.embed_texts(texts)
        except EmbeddingProviderError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise EmbeddingProviderError("provider_failed", "native llama embedding failed", retryable=False) from exc
        checked, dimensions = _validate_vectors(vectors, self.expected_dimensions, len(texts))
        return EmbeddingResult(self.provider_name, self.model_id, dimensions, checked)


class EmbeddingRouter:
    """Explicit backend selection; no implicit provider fallback."""

    def __init__(self, providers: Mapping[str, Any]) -> None:
        self.providers = dict(providers)

    def embed(self, provider: str, texts: Sequence[str]) -> EmbeddingResult:
        if provider not in self.providers:
            raise EmbeddingProviderError("provider_not_configured", "requested embedding provider is not configured")
        embed = getattr(self.providers[provider], "embed", None)
        if not callable(embed):
            raise EmbeddingProviderError("provider_invalid", "configured embedding provider is invalid")
        return embed(texts)
