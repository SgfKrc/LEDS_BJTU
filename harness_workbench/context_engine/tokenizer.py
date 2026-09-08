"""Dependency-free token estimation for the first harness ticket."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .types import ContextMessage


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Return a deterministic token estimate for text."""


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HeuristicTokenizer:
    """A conservative tokenizer substitute until a backend tokenizer is known.

    ASCII words are counted as one token, punctuation and CJK characters as
    individual tokens.  The estimate is intentionally stable rather than
    pretending to match every BPE implementation.
    """

    message_overhead: int = 4

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(_TOKEN_RE.findall(text))

    def count_message(self, message: ContextMessage) -> int:
        return self.message_overhead + self.count(message.content) + self.count(message.role)


@dataclass(frozen=True, slots=True)
class FixedTokenCounter:
    """Small test/tool helper that charges a fixed cost per character."""

    per_character: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.per_character, int) or self.per_character <= 0:
            raise ValueError("per_character must be positive")

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(text) * self.per_character

    def count_message(self, message: ContextMessage) -> int:
        return self.count(message.content) + self.count(message.role) + 1


def count_message(tokenizer: TokenCounter, message: ContextMessage) -> int:
    """Call an optional message-aware counter, falling back to text cost."""

    method = getattr(tokenizer, "count_message", None)
    if method is not None:
        value = method(message)
    else:
        value = tokenizer.count(message.content) + tokenizer.count(message.role) + 4
    if not isinstance(value, int) or value < 0:
        raise ValueError("token counter must return a non-negative integer")
    return value
