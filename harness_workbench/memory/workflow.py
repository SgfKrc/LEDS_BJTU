"""Cross-session memory workflow with safety and lifecycle gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from .retrieve import LayeredBudget, LayeredContext, MemoryHit, MemoryRetriever, build_layered_context
from .store import MemoryEntry, MemoryKind, MemoryStore

if TYPE_CHECKING:
    from ..eval.red_team import RedTeamGate


class MemorySafetyError(ValueError):
    """Raised when content is rejected by the deterministic red-team gate."""


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    query: str
    owner_scope: str
    hits: tuple[MemoryHit, ...]
    context: LayeredContext

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "owner_scope": self.owner_scope,
            "hits": [hit.as_dict() for hit in self.hits],
            "context": self.context.as_dict(),
        }


class MemoryWorkflow:
    """User-owned memory facade used by cross-session flows and E2E tests."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        retriever: MemoryRetriever | None = None,
        gate: "RedTeamGate | None" = None,
    ) -> None:
        self.store = store
        self.retriever = retriever or MemoryRetriever(store)
        if gate is None:
            from ..eval.red_team import RedTeamGate

            gate = RedTeamGate()
        self.gate = gate

    def remember(
        self,
        *,
        kind: MemoryKind,
        content: str,
        owner_scope: str = "local",
        source_session_id: str | None = None,
        source_message_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntry:
        content = _memory_content(content)
        from ..eval.red_team import RedTeamFixture

        decision = self.gate.evaluate(
            RedTeamFixture(
                id="memory-content-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                family="prompt_injection",
                payload={"untrusted": True, "messages": [{"role": "user", "content": content}]},
                expected_reason="memory_content_injection",
                expected_blocked=_looks_like_injection(content),
            )
        )
        if decision.blocked:
            raise MemorySafetyError("memory content rejected: " + decision.reason)
        return self.store.add(
            kind=kind,
            content=content,
            owner_scope=owner_scope,
            source_session_id=source_session_id,
            source_message_id=source_message_id,
            metadata=metadata,
            deduplicate=True,
        )

    def recall(
        self,
        query: str,
        *,
        owner_scope: str = "local",
        input_budget: int,
        limit: int = 8,
        budget: LayeredBudget | None = None,
        rag_hits: Iterable[Mapping[str, Any]] = (),
        context_messages: Iterable[str | Mapping[str, Any]] = (),
    ) -> MemoryRecall:
        hits = tuple(self.retriever.search(query, owner_scope=owner_scope, limit=limit))
        budget = budget or LayeredBudget(input_budget, input_budget, 0, 0)
        context = build_layered_context(
            hits,
            rag_hits=rag_hits,
            context_messages=context_messages,
            budget=budget,
        )
        return MemoryRecall(query, owner_scope, hits, context)

    def delete(
        self,
        entry_id: str,
        *,
        owner_scope: str = "local",
        confirm: bool = False,
        reason: str | None = None,
    ) -> MemoryEntry:
        return self.store.delete(entry_id, owner_scope=owner_scope, confirm=confirm, reason=reason)

    def invalidate(
        self,
        entry_id: str,
        *,
        owner_scope: str = "local",
        reason: str | None = None,
    ) -> MemoryEntry:
        return self.store.invalidate(entry_id, owner_scope=owner_scope, reason=reason)


def _memory_content(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("memory content is required")
    if len(value) > 32_000:
        raise ValueError("memory content is too large")
    return value.strip()


def _looks_like_injection(value: str) -> bool:
    lowered = value.casefold()
    return any(
        marker in lowered
        for marker in (
            "ignore previous",
            "ignore all prior",
            "reveal the system prompt",
            "you are now",
            "developer message",
            "jailbreak",
        )
    )


__all__ = ["MemoryRecall", "MemorySafetyError", "MemoryWorkflow"]
