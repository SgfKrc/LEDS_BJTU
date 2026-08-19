from __future__ import annotations

import pytest

from src.rag_embedding import (
    EmbeddingProviderError,
    EmbeddingRouter,
    NativeLlamaEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from src.llama_engine import LlamaCppEngine


def test_ollama_api_embed_batch_and_metadata():
    calls = []

    def requester(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    provider = OllamaEmbeddingProvider(
        model_id="embeddinggemma", expected_dimensions=2, requester=requester,
    )
    result = provider.embed(["one", "two"])
    assert result.provider == "ollama"
    assert result.model_id == "embeddinggemma"
    assert result.dimensions == 2
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert calls[0][0].endswith("/api/embed")
    assert calls[0][1]["model"] == "embeddinggemma"


def test_ollama_timeout_and_dimension_mismatch_are_structured():
    def timeout(url, payload, limit):
        raise TimeoutError()

    provider = OllamaEmbeddingProvider(model_id="embeddinggemma", requester=timeout)
    with pytest.raises(EmbeddingProviderError) as exc:
        provider.embed(["hello"])
    assert exc.value.code == "provider_timeout"
    assert exc.value.retryable is True

    mismatch = OllamaEmbeddingProvider(
        model_id="embeddinggemma", expected_dimensions=3,
        requester=lambda url, payload, limit: {"embeddings": [[1.0, 2.0]]},
    )
    with pytest.raises(EmbeddingProviderError) as exc:
        mismatch.embed(["hello"])
    assert exc.value.code == "dimension_mismatch"


def test_native_provider_requires_explicit_engine_capability_and_router():
    class FakeEngine:
        def embed_texts(self, texts):
            return [[0.0, 1.0] for _ in texts]

    native = NativeLlamaEmbeddingProvider(FakeEngine(), model_id="local-embed", expected_dimensions=2)
    result = EmbeddingRouter({"native": native}).embed("native", ["hello"])
    assert result.provider == "llama.cpp"
    assert result.vectors == [[0.0, 1.0]]
    with pytest.raises(EmbeddingProviderError) as exc:
        EmbeddingRouter({}).embed("ollama", ["hello"])
    assert exc.value.code == "provider_not_configured"

    with pytest.raises(EmbeddingProviderError) as exc:
        NativeLlamaEmbeddingProvider(object(), model_id="bad", expected_dimensions=2).embed(["hello"])
    assert exc.value.code == "capability_unavailable"


def test_llama_engine_embedding_capability_is_explicit_and_dimension_checked():
    class FakeModel:
        def create_embedding(self, text):
            return {"data": [{"embedding": [0.5, -0.5]}]}

    engine = LlamaCppEngine()
    engine._model = FakeModel()
    engine._loaded = True
    with pytest.raises(RuntimeError, match="not registered"):
        engine.embed_texts(["hello"])
    engine._embedding_enabled = True
    engine._embedding_dimension = 2
    engine._embedding_model_id = "local-embed"
    assert engine.embed_texts(["hello"]) == [[0.5, -0.5]]
    assert engine.get_capabilities()["embedding"] is True
