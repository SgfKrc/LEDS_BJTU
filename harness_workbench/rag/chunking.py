"""Deterministic, bounded text chunking with explicit strategy metadata."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


CHUNK_STRATEGIES = frozenset({"fixed", "paragraph", "sentence", "section", "adaptive", "semantic"})


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
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError("unsupported chunk strategy")
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


def _last_boundary(value: str, start: int, end: int, markers: tuple[str, ...], *, floor_ratio: float = 0.5) -> int:
    floor = start + max(1, int((end - start) * floor_ratio))
    positions = [value.rfind(marker, floor, end) + len(marker) for marker in markers if value.rfind(marker, floor, end) >= 0]
    return max(positions, default=-1)


def _section_boundary(value: str, start: int, end: int) -> int:
    floor = start + max(1, int((end - start) * 0.35))
    candidates: list[int] = []
    for match in re.finditer(r"(?m)^(?:#{1,6}\s+|[A-Z][A-Z0-9 _-]{3,}:\s*$)", value[floor:end]):
        position = floor + match.start()
        if position > start:
            candidates.append(position)
    return max(candidates, default=-1)


def _boundary(value: str, start: int, end: int, strategy: str) -> int:
    if strategy == "fixed":
        return _last_boundary(value, start, end, ("\n", " "))
    if strategy == "paragraph":
        return max(_last_boundary(value, start, end, ("\n\n",), floor_ratio=0.3), _last_boundary(value, start, end, ("\n", " ")))
    if strategy == "sentence":
        return _last_boundary(value, start, end, (". ", "! ", "? ", "。", "！", "？", "\n", " "))
    if strategy == "section":
        return max(_section_boundary(value, start, end), _last_boundary(value, start, end, ("\n\n", "\n", " "), floor_ratio=0.35))
    if strategy == "adaptive":
        return max(_section_boundary(value, start, end), _last_boundary(value, start, end, ("\n\n",), floor_ratio=0.35), _last_boundary(value, start, end, (". ", "。", "！", "？", "\n", " "), floor_ratio=0.5))
    # semantic remains deterministic: use punctuation and headings, never a model.
    return max(_section_boundary(value, start, end), _last_boundary(value, start, end, (". ", "! ", "? ", "。", "！", "？"), floor_ratio=0.4), _last_boundary(value, start, end, ("\n\n", "\n", " "), floor_ratio=0.5))


__all__ = ["CHUNK_STRATEGIES", "TextChunk", "chunk_text"]
