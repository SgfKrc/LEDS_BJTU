"""FTS retrieval and bounded, citation-preserving memory injection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Any, Iterable, Mapping, Protocol

from ..context_engine.tokenizer import HeuristicTokenizer, TokenCounter


class EmbeddingProvider(Protocol):
    """Compatible subset of the S4 embedding provider contract."""

    def embed(self, texts: list[str]):
        ...


@dataclass(frozen=True, slots=True)
class MemoryHit:
    entry_id: str
    owner_scope: str
    kind: str
    content: str
    fingerprint: str
    source_session_id: str | None
    source_message_id: str | None
    created_at: float
    score: float

    @property
    def text(self) -> str:
        return self.content

    def citation(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "owner_scope": self.owner_scope,
            "source_session_id": self.source_session_id,
            "source_message_id": self.source_message_id,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.citation(),
            "content": self.content,
            "text": self.content,
            "score": self.score,
        }


class MemoryRetriever:
    """FTS-first retriever with optional provider-backed candidate reranking."""

    def __init__(self, store: Any, *, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def search(self, query: str, *, owner_scope: str = "local", limit: int = 8) -> list[MemoryHit]:
        hits = list(self.store.search(query, owner_scope=owner_scope, limit=limit))
        provider = self.embedding_provider
        if provider is None or not hits:
            return hits
        try:
            result = provider.embed([query, *(hit.content for hit in hits)])
            vectors = tuple(result.vectors)
            if len(vectors) != len(hits) + 1:
                return hits
            query_vector = vectors[0]
            scored = [
                replace(hit, score=_cosine_similarity(query_vector, vector))
                for hit, vector in zip(hits, vectors[1:])
            ]
            return sorted(scored, key=lambda hit: (-hit.score, hit.entry_id))
        except Exception:
            # FTS is the safe baseline; an optional provider must not make
            # local memory unavailable when its model is absent or unhealthy.
            return hits

@dataclass(frozen=True, slots=True)
class LayeredBudget:
    """One input budget divided among memory, RAG and recent context."""

    input_budget: int
    memory_budget: int
    rag_budget: int
    context_budget: int

    def __post_init__(self) -> None:
        values = (self.input_budget, self.memory_budget, self.rag_budget, self.context_budget)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("layered budgets must be integers")
        if self.input_budget <= 0 or any(value < 0 for value in values[1:]):
            raise ValueError("layered budgets must be non-negative with a positive input budget")
        if self.memory_budget + self.rag_budget + self.context_budget != self.input_budget:
            raise ValueError("layer budgets must sum to input_budget")

    @classmethod
    def from_input_budget(
        cls,
        input_budget: int,
        *,
        memory_ratio: float = 0.25,
        rag_ratio: float = 0.25,
    ) -> "LayeredBudget":
        if not isinstance(input_budget, int) or isinstance(input_budget, bool) or input_budget <= 0:
            raise ValueError("input_budget must be a positive integer")
        if not 0 <= memory_ratio <= 1 or not 0 <= rag_ratio <= 1 or memory_ratio + rag_ratio > 1:
            raise ValueError("memory_ratio and rag_ratio must leave context budget")
        memory_budget = int(input_budget * memory_ratio)
        rag_budget = int(input_budget * rag_ratio)
        context_budget = input_budget - memory_budget - rag_budget
        return cls(input_budget, memory_budget, rag_budget, context_budget)

    def as_dict(self) -> dict[str, int]:
        return {
            "input_budget": self.input_budget,
            "memory_budget": self.memory_budget,
            "rag_budget": self.rag_budget,
            "context_budget": self.context_budget,
        }


@dataclass(frozen=True, slots=True)
class LayeredContext:
    text: str
    citations: tuple[Mapping[str, Any], ...]
    included_count: int
    omitted_count: int
    input_budget: int
    budget: LayeredBudget
    layer_tokens: Mapping[str, int]
    layer_omitted: Mapping[str, int]

    @property
    def truncated(self) -> bool:
        return self.omitted_count > 0

    @property
    def input_tokens(self) -> int:
        return sum(self.layer_tokens.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [dict(item) for item in self.citations],
            "included_count": self.included_count,
            "omitted_count": self.omitted_count,
            "truncated": self.truncated,
            "input_budget": self.input_budget,
            "budget": self.budget.as_dict(),
            "input_tokens": self.input_tokens,
            "layer_tokens": dict(self.layer_tokens),
            "layer_omitted": dict(self.layer_omitted),
        }


def build_memory_context(
    hits: Iterable[MemoryHit | Mapping[str, Any]],
    *,
    input_budget: int,
    tokenizer: TokenCounter | None = None,
) -> LayeredContext:
    """Build a memory-only context while preserving complete entries."""
    if not isinstance(input_budget, int) or isinstance(input_budget, bool) or input_budget <= 0:
        raise ValueError("input_budget must be a positive integer")
    budget = LayeredBudget(input_budget, input_budget, 0, 0)
    return build_layered_context(hits, rag_hits=(), context_messages=(), budget=budget, tokenizer=tokenizer)


def build_layered_context(
    memory_hits: Iterable[MemoryHit | Mapping[str, Any]],
    *,
    rag_hits: Iterable[Mapping[str, Any]] = (),
    context_messages: Iterable[str | Mapping[str, Any]] = (),
    budget: LayeredBudget,
    tokenizer: TokenCounter | None = None,
) -> LayeredContext:
    """Assemble three bounded layers from one shared input budget.

    Every candidate is admitted or omitted as a complete block.  The function
    never slices content to fit a budget, so callers can surface truncation and
    ask for a larger context window or fewer retrieval hits.
    """
    tokenizer = tokenizer or HeuristicTokenizer()
    blocks: list[str] = []
    citations: list[Mapping[str, Any]] = []
    layer_tokens: dict[str, int] = {}
    layer_omitted: dict[str, int] = {}

    for name, values, layer_budget in (
        ("memory", memory_hits, budget.memory_budget),
        ("rag", rag_hits, budget.rag_budget),
        ("context", context_messages, budget.context_budget),
    ):
        layer_blocks, layer_citations, used, omitted = _fit_layer(
            name,
            values,
            layer_budget,
            tokenizer,
        )
        blocks.extend(layer_blocks)
        citations.extend(layer_citations)
        layer_tokens[name] = used
        layer_omitted[name] = omitted

    text = "\n\n".join(blocks)
    return LayeredContext(
        text=text,
        citations=tuple(citations),
        included_count=len(blocks),
        omitted_count=sum(layer_omitted.values()),
        input_budget=budget.input_budget,
        budget=budget,
        layer_tokens=layer_tokens,
        layer_omitted=layer_omitted,
    )


def _fit_layer(
    name: str,
    values: Iterable[MemoryHit | Mapping[str, Any] | str],
    layer_budget: int,
    tokenizer: TokenCounter,
) -> tuple[list[str], list[Mapping[str, Any]], int, int]:
    blocks: list[str] = []
    citations: list[Mapping[str, Any]] = []
    used = 0
    omitted = 0
    for index, value in enumerate(values):
        content, citation = _candidate(value, name, index)
        if not content.strip():
            omitted += 1
            continue
        marker = citation.get("entry_id") or citation.get("chunk_id") or citation.get("message_id") or str(index)
        block = f"[{name}:citation:{marker}] {content.strip()}"
        cost = tokenizer.count(block + "\n\n")
        if used + cost > layer_budget:
            omitted += 1
            continue
        blocks.append(block)
        used += cost
        if name != "context":
            citations.append(citation)
    return blocks, citations, used, omitted


def _candidate(value: MemoryHit | Mapping[str, Any] | str, name: str, index: int) -> tuple[str, dict[str, Any]]:
    if isinstance(value, MemoryHit):
        return value.content, {"layer": name, **value.citation()}
    if isinstance(value, str):
        return value, {"layer": name, "message_id": str(index)}
    if not isinstance(value, Mapping):
        return "", {"layer": name}
    content = value.get("content", value.get("text", ""))
    content = content if isinstance(content, str) else ""
    citation = {"layer": name}
    for key in ("entry_id", "chunk_id", "source_id", "title", "ordinal", "message_id", "kind"):
        if key in value:
            citation[key] = value[key]
    return content, citation


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


__all__ = [
    "LayeredBudget",
    "LayeredContext",
    "MemoryHit",
    "MemoryRetriever",
    "build_layered_context",
    "build_memory_context",
]
