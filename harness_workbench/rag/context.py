"""Budgeted retrieval context assembly with explicit omissions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    text: str
    citations: tuple[Mapping[str, Any], ...]
    included_count: int
    omitted_count: int
    token_count: int = 0
    max_chars: int = 0
    max_tokens: int | None = None
    omission_reasons: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        return self.omitted_count > 0

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "text": self.text,
            "citations": [dict(item) for item in self.citations],
            "included_count": self.included_count,
            "omitted_count": self.omitted_count,
            "truncated": self.truncated,
        }
        if self.max_tokens is not None:
            value.update({"token_count": self.token_count, "max_chars": self.max_chars, "max_tokens": self.max_tokens, "omission_reasons": list(self.omission_reasons)})
        return value


def build_context(
    hits: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = 8_000,
    max_tokens: int | None = None,
    tokenizer: Callable[[str], int] | None = None,
) -> RetrievalContext:
    if not 256 <= max_chars <= 120_000:
        raise ValueError("max_chars must be between 256 and 120000")
    if max_tokens is not None and not 1 <= max_tokens <= 100_000:
        raise ValueError("max_tokens must be between 1 and 100000")
    blocks: list[str] = []
    citations: list[Mapping[str, Any]] = []
    omitted = 0
    token_count = 0
    reasons: list[str] = []
    for index, hit in enumerate(hits):
        text = hit.get("text", "")
        if not isinstance(text, str) or not text.strip():
            omitted += 1
            reasons.append("invalid")
            continue
        citation = {
            "source_id": str(hit.get("source_id", "")),
            "chunk_id": str(hit.get("chunk_id", "")),
            "title": str(hit.get("title", "")),
            "ordinal": hit.get("ordinal", index),
        }
        if hit.get("granularity") not in (None, "", "fixed"):
            citation["granularity"] = str(hit["granularity"])
        block = f"[citation:{citation['chunk_id']}] {text.strip()}"
        separator = "\n\n" if blocks else ""
        if len("".join(blocks)) + len(separator) + len(block) > max_chars:
            omitted += 1
            reasons.append("chars")
            continue
        block_tokens = _count_tokens(block, tokenizer)
        if max_tokens is not None and token_count + block_tokens > max_tokens:
            omitted += 1
            reasons.append("tokens")
            continue
        blocks.append(block)
        citations.append(citation)
        token_count += block_tokens
    return RetrievalContext("\n\n".join(blocks), tuple(citations), len(citations), omitted, token_count, max_chars, max_tokens, tuple(reasons))


def _count_tokens(text: str, tokenizer: Callable[[str], int] | None) -> int:
    if tokenizer is not None:
        value = int(tokenizer(text))
        if value < 0:
            raise ValueError("tokenizer returned a negative count")
        return value
    return len(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\w\s]", text, flags=re.UNICODE))


__all__ = ["RetrievalContext", "build_context"]
