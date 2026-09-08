"""Independent, user-owned FTS-first retrieval for the harness."""

from .chunking import TextChunk, chunk_text
from .context import RetrievalContext, build_context
from .providers import EmbeddingProvider, EmbeddingResult
from .store import RagHit, RagStore

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "RagHit",
    "RagStore",
    "RetrievalContext",
    "TextChunk",
    "build_context",
    "chunk_text",
]
