"""Deterministic compression primitives and degradation accounting.

The context engine deliberately keeps compression choices explicit.  These
helpers do not call a model and never cut a message in the middle of a turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .types import ContextMessage


COMPRESSION_STRATEGIES = frozenset({"adaptive", "mask", "state", "verbatim"})
STATE_VARIANTS = frozenset({"compact", "lines", "nonempty"})


@dataclass(frozen=True, slots=True)
class CompressionStep:
    """One observable step in the over-budget degradation curve."""

    strategy: str
    before_tokens: int
    after_tokens: int
    changed: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "changed": self.changed,
            "details": dict(self.details),
        }


def compact_verbatim(messages: Sequence[ContextMessage]) -> str:
    """Render complete old messages without paraphrasing their content."""

    lines: list[str] = []
    for message in messages:
        content = re.sub(r"\s+", " ", message.content).strip()
        if content:
            lines.append(f"[{message.role}] {content}")
    return "\n".join(lines)


def render_state(state: Mapping[str, Sequence[str]], *, variant: str = "compact") -> str:
    """Render a validated STATE in one of the stable, bounded variants."""

    if variant not in STATE_VARIANTS:
        raise ValueError(f"unsupported STATE variant: {variant}")
    if variant == "lines":
        return "\n".join(
            f"{field}: " + " | ".join(str(item) for item in values)
            for field, values in state.items()
            if values
        ) or "STATE: empty"
    if variant == "nonempty":
        value = {field: list(values) for field, values in state.items() if values}
    else:
        value = {field: list(values) for field, values in state.items()}
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "COMPRESSION_STRATEGIES",
    "STATE_VARIANTS",
    "CompressionStep",
    "compact_verbatim",
    "render_state",
]
