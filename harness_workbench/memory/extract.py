"""Conservative, auditable candidates for ContextPolicy memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..context_engine.summarize import SummaryResult
from ..context_engine.types import ContextMessage
from .store import MemoryKind


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    kind: MemoryKind
    content: str
    source_message_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            "source_message_ids": list(self.source_message_ids),
        }


_PREFERENCE_MARKERS = (
    "i prefer",
    "my preference",
    "prefer:",
    "我偏好",
    "我喜欢",
    "偏好",
)
_DECISION_MARKERS = (
    "i decided",
    "we decided",
    "decided to",
    "decision:",
    "我决定",
    "决定",
    "选择",
    "采用",
)
_FACT_MARKERS = (
    "my name is",
    "i am ",
    "remember that",
    "fact:",
    "我的名字",
    "我是",
    "记住",
    "事实:",
)


def extract_memory_candidates(
    messages: Iterable[ContextMessage],
    summary: SummaryResult,
) -> tuple[MemoryCandidate, ...]:
    """Extract only explicit/high-signal candidates from omitted messages.

    The summary is used as a normalized representation, but it cannot create
    memory evidence by itself.  A candidate must have a matching user message,
    or an explicit ``metadata.memory`` opt-in supplied by the host.
    """
    values = tuple(messages)
    source_ids = tuple(message.message_id for message in values if message.message_id)
    candidates: list[MemoryCandidate] = []
    explicit_source_ids: set[str] = set()
    for message in values:
        explicit = _explicit_candidate(message)
        if explicit is not None:
            candidates.append(explicit)
            explicit_source_ids.update(explicit.source_message_ids)
            continue
        if message.role != "user":
            continue
        content = _clean(message.content)
        lowered = content.casefold()
        if _contains_any(lowered, _PREFERENCE_MARKERS):
            candidates.append(MemoryCandidate("preference", content, _ids(message)))
        elif _contains_any(lowered, _DECISION_MARKERS):
            candidates.append(MemoryCandidate("decision", content, _ids(message)))
        elif _contains_any(lowered, _FACT_MARKERS):
            candidates.append(MemoryCandidate("fact", content, _ids(message)))

    validated = summary.validated_state()
    if source_ids and any(message.role == "user" for message in values):
        user_ids = tuple(
            message.message_id
            for message in values
            if message.role == "user" and message.message_id and message.message_id not in explicit_source_ids
        )
        user_text = " ".join(
            _clean(message.content).casefold()
            for message in values
            if message.role == "user" and message.message_id not in explicit_source_ids
        )
        if user_ids and _contains_any(user_text, _FACT_MARKERS):
            candidates.extend(MemoryCandidate("fact", item, user_ids) for item in validated["what"] if _clean(item))
        if user_ids and _contains_any(user_text, _DECISION_MARKERS):
            candidates.extend(MemoryCandidate("decision", item, user_ids) for item in validated["decisions"] if _clean(item))

    unique: dict[tuple[str, str], MemoryCandidate] = {}
    for candidate in candidates:
        normalized = _clean(candidate.content)
        if normalized:
            unique.setdefault((candidate.kind, normalized.casefold()), MemoryCandidate(candidate.kind, normalized, candidate.source_message_ids))
    return tuple(unique.values())


def _explicit_candidate(message: ContextMessage) -> MemoryCandidate | None:
    raw = message.metadata.get("memory") if isinstance(message.metadata, Mapping) else None
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("kind")
    content = raw.get("content")
    if kind not in {"fact", "preference", "decision"} or not isinstance(content, str):
        return None
    content = _clean(content)
    if not content:
        return None
    return MemoryCandidate(kind, content, _ids(message))  # type: ignore[arg-type]


def _ids(message: ContextMessage) -> tuple[str, ...]:
    return (message.message_id,) if message.message_id else ()


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:2_000]


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


__all__ = ["MemoryCandidate", "extract_memory_candidates"]
