"""Optional embedding provider contract; FTS remains the safe default."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    provider: str
    model_id: str
    dimensions: int
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.provider or not self.model_id or self.dimensions <= 0:
            raise ValueError("embedding identity is invalid")
        if any(len(vector) != self.dimensions for vector in self.vectors):
            raise ValueError("embedding vector dimension mismatch")


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        ...


__all__ = ["EmbeddingProvider", "EmbeddingResult"]
