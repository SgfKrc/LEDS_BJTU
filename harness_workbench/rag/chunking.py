"""Deterministic, bounded text chunking with overlap metadata."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    text: str
    start_offset: int
    end_offset: int
    granularity: str = "fixed"


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 120,
    strategy: str = "fixed",
) -> tuple[TextChunk, ...]:
    if not isinstance(text, str) or not text.strip():
        return ()
    if not 128 <= max_chars <= 32_000:
        raise ValueError("max_chars must be between 128 and 32000")
    if not 0 <= overlap_chars < max_chars // 2:
        raise ValueError("overlap_chars must be non-negative and less than half max_chars")
    if strategy not in {"fixed", "paragraph", "sentence"}:
        raise ValueError("strategy must be fixed, paragraph, or sentence")
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks: list[TextChunk] = []
    start = 0
    ordinal = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = _boundary(normalized, start, end, strategy)
            if boundary > start:
                end = boundary
        value = normalized[start:end].strip()
        if value:
            actual_start = start + len(normalized[start:end]) - len(normalized[start:end].lstrip())
            actual_end = actual_start + len(value)
            chunks.append(TextChunk(ordinal, value, actual_start, actual_end, strategy))
            ordinal += 1
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)
    return tuple(chunks)


def _boundary(value: str, start: int, end: int, strategy: str) -> int:
    floor = start + (end - start) // 2
    candidates = [value.rfind("\n", floor, end), value.rfind(" ", floor, end)]
    if strategy == "paragraph":
        candidates.append(value.rfind("\n\n", floor, end))
    elif strategy == "sentence":
        candidates.extend(floor + match.start() + 1 for match in re.finditer(r"[.!?。！？](?:\s|$)", value[floor:end]))
        candidates.extend(value.rfind(mark, floor, end) for mark in ("\n", " "))
    return max(candidates)


__all__ = ["TextChunk", "chunk_text"]
