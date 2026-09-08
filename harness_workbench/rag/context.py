"""Budgeted retrieval context assembly with explicit omissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    text: str
    citations: tuple[Mapping[str, Any], ...]
    included_count: int
    omitted_count: int

    @property
    def truncated(self) -> bool:
        return self.omitted_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [dict(item) for item in self.citations],
            "included_count": self.included_count,
            "omitted_count": self.omitted_count,
            "truncated": self.truncated,
        }


def build_context(hits: Iterable[Mapping[str, Any]], *, max_chars: int = 8_000) -> RetrievalContext:
    if not 256 <= max_chars <= 120_000:
        raise ValueError("max_chars must be between 256 and 120000")
    blocks: list[str] = []
    citations: list[Mapping[str, Any]] = []
    omitted = 0
    for index, hit in enumerate(hits):
        text = hit.get("text", "")
        if not isinstance(text, str) or not text.strip():
            omitted += 1
            continue
        citation = {
            "source_id": str(hit.get("source_id", "")),
            "chunk_id": str(hit.get("chunk_id", "")),
            "title": str(hit.get("title", "")),
            "ordinal": hit.get("ordinal", index),
        }
        block = f"[citation:{citation['chunk_id']}] {text.strip()}"
        separator = "\n\n" if blocks else ""
        if len("".join(blocks)) + len(separator) + len(block) > max_chars:
            omitted += 1
            continue
        blocks.append(block)
        citations.append(citation)
    return RetrievalContext("\n\n".join(blocks), tuple(citations), len(citations), omitted)


__all__ = ["RetrievalContext", "build_context"]
