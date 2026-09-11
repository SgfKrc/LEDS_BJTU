"""Independent, user-owned FTS-first retrieval for the harness."""

from .chunking import TextChunk, chunk_text
from .context import RetrievalContext, build_context
from .providers import EmbeddingProvider, EmbeddingResult
from .query import QueryPlan, normalize_query, rewrite_query
from .retriever import HybridRagRetriever, RagSearchConfig, RagSearchResult
from .store import RagHit, RagStore

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "HybridRagRetriever",
    "QueryPlan",
    "RagHit",
    "RagStore",
    "RagSearchConfig",
    "RagSearchResult",
    "RetrievalContext",
    "TextChunk",
    "build_context",
    "chunk_text",
    "normalize_query",
    "rewrite_query",
]
