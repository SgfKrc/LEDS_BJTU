"""Context selection policy for small-model conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .budget import ContextBudget
from .compression import (
    COMPRESSION_STRATEGIES,
    STATE_VARIANTS,
    CompressionStep,
    compact_verbatim,
    render_state,
)
from .notices import ContextNotice
from .summarize import (
    RuleBasedSummarizer,
    StateValidationError,
    SummaryProvider,
    SummaryResult,
    apply_state_patch,
    validate_state,
)
from .tokenizer import HeuristicTokenizer, TokenCounter, count_message
from .types import ContextLedgerEntry, ContextMessage
from ..memory.extract import MemoryCandidate, extract_memory_candidates


class ContextBuildError(ValueError):
    """Base error for an unsafe or impossible context build."""


class PinnedContentOverflow(ContextBuildError):
    """Raised rather than clipping pinned or system content."""


@dataclass(frozen=True, slots=True)
class ContextPolicyConfig:
    recent_turns: int = 12
    summary_trigger_ratio: float = 0.70
    recent_turn_ratio: float = 0.35
    max_tool_outputs: int = 1
    compression_strategy: str = "adaptive"
    state_variant: str = "compact"
    verbatim_max_characters: int = 8_000
    memory_recall_ratio: float = 0.10
    memory_recall_limit: int = 4

    def __post_init__(self) -> None:
        if self.recent_turns < 0:
            raise ValueError("recent_turns must be non-negative")
        if not 0 < self.summary_trigger_ratio <= 1:
            raise ValueError("summary_trigger_ratio must be between zero and one")
        if not 0 < self.recent_turn_ratio <= 1:
            raise ValueError("recent_turn_ratio must be between zero and one")
        if self.max_tool_outputs < 0:
            raise ValueError("max_tool_outputs must be non-negative")
        if self.compression_strategy not in COMPRESSION_STRATEGIES:
            raise ValueError("compression_strategy must be adaptive, mask, state, or verbatim")
        if self.state_variant not in STATE_VARIANTS:
            raise ValueError("state_variant must be compact, lines, or nonempty")
        if not isinstance(self.verbatim_max_characters, int) or self.verbatim_max_characters <= 0:
            raise ValueError("verbatim_max_characters must be positive")
        if not 0 <= self.memory_recall_ratio < 1:
            raise ValueError("memory_recall_ratio must be between zero and one")
        if not isinstance(self.memory_recall_limit, int) or self.memory_recall_limit < 0:
            raise ValueError("memory_recall_limit must be non-negative")


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Result of one context build, including an auditable token ledger."""

    messages: tuple[ContextMessage, ...]
    budget: ContextBudget
    input_tokens: int
    state: dict[str, list[str]]
    notices: tuple[ContextNotice, ...] = ()
    ledger: tuple[ContextLedgerEntry, ...] = ()
    summarized_message_ids: tuple[str, ...] = ()
    memory_entry_ids: tuple[str, ...] = ()
    memory_candidates: tuple[Mapping[str, object], ...] = ()
    memory_write_error: str | None = None
    memory_recall_ids: tuple[str, ...] = ()
    compression_strategy: str = "none"
    degradation: tuple[CompressionStep, ...] = ()

    @property
    def input_budget(self) -> int:
        return self.budget.input_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "messages": [message.as_dict() for message in self.messages],
            "budget": self.budget.as_dict(),
            "input_tokens": self.input_tokens,
            "state": self.state,
            "notices": [notice.as_dict() for notice in self.notices],
            "ledger": [entry.as_dict() for entry in self.ledger],
            "summarized_message_ids": list(self.summarized_message_ids),
            "memory_entry_ids": list(self.memory_entry_ids),
            "memory_candidates": [dict(item) for item in self.memory_candidates],
            "memory_write_error": self.memory_write_error,
            "memory_recall_ids": list(self.memory_recall_ids),
            "compression_strategy": self.compression_strategy,
            "degradation": [step.as_dict() for step in self.degradation],
        }


@dataclass(slots=True)
class ContextPolicy:
    """Apply masking, round-aware clipping and structured summarization."""

    config: ContextPolicyConfig = field(default_factory=ContextPolicyConfig)
    tokenizer: TokenCounter = field(default_factory=HeuristicTokenizer)
    summarizer: SummaryProvider = field(default_factory=RuleBasedSummarizer)

    def build(
        self,
        messages: Iterable[ContextMessage | Mapping[str, object]],
        budget: ContextBudget,
        *,
        state: Mapping[str, object] | None = None,
        state_patch: Mapping[str, object] | None = None,
        allow_state_delete: bool = False,
        memory_store: Any | None = None,
        memory_owner_scope: str = "local",
        memory_source_session_id: str | None = None,
        memory_query: str | None = None,
    ) -> ContextSnapshot:
        normalized = [ContextMessage.from_value(value) for value in messages]
        if not normalized:
            return ContextSnapshot(
                messages=(),
                budget=budget,
                input_tokens=0,
                state=validate_state(state or {}),
                compression_strategy="none",
            )
        current_state = validate_state(state or {})
        if state_patch is not None:
            current_state = apply_state_patch(
                current_state,
                state_patch,
                allow_delete=allow_state_delete,
            )

        token_costs = {id(message): count_message(self.tokenizer, message) for message in normalized}
        notices: list[ContextNotice] = []
        degradation: list[CompressionStep] = []
        original_tokens = sum(token_costs.values())
        masked_ids: set[int] = set()
        output_messages = [message for message in normalized if message.is_output and not message.pinned]
        keep_outputs = (
            set(id(message) for message in output_messages[-self.config.max_tool_outputs :])
            if self.config.max_tool_outputs
            else set()
        )
        for message in output_messages:
            if id(message) not in keep_outputs:
                masked_ids.add(id(message))
        if masked_ids:
            masked_tokens = sum(token_costs[id(message)] for message in normalized if id(message) not in masked_ids)
            degradation.append(
                CompressionStep(
                    "mask",
                    original_tokens,
                    masked_tokens,
                    details={"masked_messages": len(masked_ids)},
                )
            )
            notices.append(
                ContextNotice(
                    code="context.masked_tool_output",
                    message="Older tool or image output was masked to protect the context budget.",
                    details={"masked_messages": len(masked_ids)},
                )
            )
        candidates = [message for message in normalized if id(message) not in masked_ids]
        input_budget = budget.input_budget
        full_tokens = sum(token_costs[id(message)] for message in candidates)
        summary_pressure = self._summary_pressure(candidates, token_costs, input_budget)
        if full_tokens <= input_budget and not summary_pressure and not (memory_store is not None and memory_query):
            ledger = tuple(
                ContextLedgerEntry(
                    message_id=message.message_id,
                    role=message.role,
                    turn_id=message.turn_id,
                    tokens=token_costs[id(message)],
                    retained=id(message) not in masked_ids,
                    reason="masked_output" if id(message) in masked_ids else "within_budget",
                )
                for message in normalized
            )
            return ContextSnapshot(
                messages=tuple(candidates),
                budget=budget,
                input_tokens=full_tokens,
                state=current_state,
                notices=tuple(notices),
                ledger=ledger,
                compression_strategy="mask" if masked_ids else "none",
                degradation=tuple(degradation),
            )

        fixed = [message for message in candidates if message.role == "system" or message.pinned]
        fixed_ids = {id(message) for message in fixed}
        fixed_tokens = sum(token_costs[id(message)] for message in fixed)
        if fixed_tokens > input_budget:
            raise PinnedContentOverflow(
                "system and pinned content exceed the input budget; refusing to clip it"
            )
        variable = [message for message in candidates if id(message) not in fixed_ids]
        groups = self._group_turns(variable)
        recent_groups = groups[-self.config.recent_turns :] if self.config.recent_turns else []
        old_groups = groups[: len(groups) - len(recent_groups)] if recent_groups else groups
        old_messages = [message for group in old_groups for message in group]
        strategy = self.config.compression_strategy
        use_state = strategy in {"adaptive", "state"}
        use_verbatim = strategy == "verbatim"
        summary_result = (
            self._summarize(old_messages, current_state)
            if old_messages and use_state
            else SummaryResult(state=validate_state(current_state))
        )
        summary_message: ContextMessage | None = None
        summary_tokens = 0
        available_without_summary = input_budget - fixed_tokens
        # Reserve the configured recent-turn share before fitting STATE.  This
        # keeps a compact recent conversation usable even when the summary is
        # too verbose for a very small context window.
        recent_reserve = min(
            available_without_summary,
            max(0, int(input_budget * self.config.recent_turn_ratio)),
        )
        memory_reserve = (
            min(available_without_summary, max(0, int(input_budget * self.config.memory_recall_ratio)))
            if memory_store is not None and memory_query and self.config.memory_recall_limit
            else 0
        )
        if old_messages:
            summary_budget = max(0, available_without_summary - recent_reserve)
            summary_budget = max(0, summary_budget - memory_reserve)
            if use_state:
                summary_message, summary_result, summary_tokens = self._shrink_summary(
                    summary_result,
                    summary_budget,
                )
            elif use_verbatim:
                summary_message, summary_tokens = self._verbatim_message(old_messages, summary_budget)
            if summary_tokens > summary_budget:
                if strategy == "adaptive":
                    summary_message, summary_tokens = self._verbatim_message(old_messages, summary_budget)
                    if summary_message is not None:
                        notices.append(
                            ContextNotice(
                                code="context.verbatim_fallback",
                                message="Structured STATE did not fit; complete older messages were compacted verbatim.",
                                severity="warning",
                                details={"summary_budget": summary_budget},
                            )
                        )
                if summary_message is None or summary_tokens > summary_budget:
                    summary_message = None
                    summary_tokens = 0
                notices.append(
                    ContextNotice(
                        code="context.summary_omitted",
                        message="The structured summary could not fit; recent turns were retained.",
                        severity="warning",
                        details={"summary_budget": summary_budget},
                    )
                )
        available_for_recent = available_without_summary - summary_tokens - memory_reserve

        retained_recent: list[ContextMessage] = []
        retained_group_ids: set[int] = set()
        for group in reversed(recent_groups):
            group_tokens = sum(token_costs[id(message)] for message in group)
            if sum(token_costs[id(message)] for message in retained_recent) + group_tokens > available_for_recent:
                continue
            retained_recent[0:0] = group
            retained_group_ids.add(id(group))

        omitted_groups = [group for group in groups if id(group) not in retained_group_ids]
        omitted_messages = [message for group in omitted_groups for message in group]
        if omitted_messages:
            # If the recent-turn cap itself caused omission, include those
            # messages in STATE as well, preserving round boundaries in output.
            if use_state:
                summary_result = self._summarize(omitted_messages, current_state)
            summary_budget = input_budget - fixed_tokens - sum(
                token_costs[id(message)] for message in retained_recent
            )
            summary_budget = max(0, summary_budget - memory_reserve)
            if use_state:
                summary_message, summary_result, summary_tokens = self._shrink_summary(
                    summary_result,
                    summary_budget,
                )
            elif use_verbatim or strategy == "adaptive":
                summary_message, summary_tokens = self._verbatim_message(omitted_messages, summary_budget)
            if summary_tokens > summary_budget:
                summary_message = None
                summary_tokens = 0
            if omitted_messages and summary_message is None and strategy == "adaptive":
                summary_message, summary_tokens = self._verbatim_message(
                    omitted_messages,
                    summary_budget,
                )
                if summary_message is not None:
                    notices.append(
                        ContextNotice(
                            code="context.verbatim_fallback",
                            message="Older messages were compacted verbatim after the STATE budget was exhausted.",
                            severity="warning",
                        )
                    )
        memory_entry_ids: tuple[str, ...] = ()
        memory_candidates: tuple[Mapping[str, object], ...] = ()
        memory_write_error: str | None = None
        if memory_store is not None and omitted_messages:
            memory_entry_ids, memory_candidates, memory_write_error = self._persist_memory(
                memory_store,
                memory_owner_scope,
                memory_source_session_id,
                omitted_messages,
                summary_result,
            )
            if memory_candidates:
                notices.append(
                    ContextNotice(
                        code="context.memory_persisted" if not memory_write_error else "context.memory_write_failed",
                        message=(
                            "High-signal facts from the compressed turns were persisted to user-owned memory."
                            if memory_entry_ids and not memory_write_error
                            else "Some high-signal memory candidates were persisted, but others could not be written."
                            if memory_entry_ids
                            else "High-signal memory candidates were found but could not be persisted."
                        ),
                        severity="info" if memory_entry_ids and not memory_write_error else "warning",
                        details={
                            "candidate_count": len(memory_candidates),
                            "written_count": len(memory_entry_ids),
                            "error": memory_write_error,
                        },
                    )
                )
        memory_message: ContextMessage | None = None
        memory_recall_ids: tuple[str, ...] = ()
        if memory_store is not None and memory_query and self.config.memory_recall_limit:
            memory_message, memory_recall_ids = self._recall_memory(
                memory_store,
                memory_owner_scope,
                memory_query,
                memory_reserve,
            )
            if memory_query and memory_message is None:
                notices.append(
                    ContextNotice(
                        code="context.memory_recall_omitted",
                        message="Long-term memory was available but no complete entry fit the reserved budget.",
                        severity="warning",
                        details={"memory_budget": memory_reserve},
                    )
                )
        final_messages = self._order_fixed_and_recent(fixed, summary_message, memory_message, retained_recent)
        final_tokens = sum(token_costs.get(id(message), count_message(self.tokenizer, message)) for message in final_messages)
        if final_tokens > input_budget:
            # This can only happen when a custom tokenizer is inconsistent or
            # a summary was changed after selection; fail closed.
            raise ContextBuildError(
                f"context policy produced {final_tokens} tokens for budget {input_budget}"
            )

        if summary_message is not None:
            strategy_name = "state" if summary_message.kind == "summary" else "verbatim"
            degradation.append(
                CompressionStep(
                    strategy_name,
                    sum(token_costs[id(message)] for message in omitted_messages),
                    count_message(self.tokenizer, summary_message),
                    details={"omitted_messages": len(omitted_messages), "state_variant": self.config.state_variant},
                )
            )
        elif omitted_messages:
            degradation.append(
                CompressionStep(
                    "window",
                    sum(token_costs[id(message)] for message in omitted_messages),
                    0,
                    details={"omitted_messages": len(omitted_messages)},
                )
            )
        if memory_message is not None:
            degradation.append(
                CompressionStep(
                    "memory_recall",
                    0,
                    count_message(self.tokenizer, memory_message),
                    details={"entries": len(memory_recall_ids)},
                )
            )

        if omitted_messages:
            notices.append(
                ContextNotice(
                    code="context.summarized",
                    message="Older conversation turns were folded into a structured STATE summary.",
                    details={
                        "omitted_messages": len(omitted_messages),
                        "retained_turns": len(retained_group_ids),
                        "input_tokens": final_tokens,
                        "input_budget": input_budget,
                        "trigger": "summary_ratio" if summary_pressure else "budget",
                        "strategy": "state" if summary_message is not None and summary_message.kind == "summary" else "verbatim" if summary_message is not None else "window",
                    },
                )
            )
        masked_message_ids = {id(message) for message in normalized if id(message) in masked_ids}
        retained_ids = {id(item) for item in final_messages}
        fixed_message_ids = {id(item) for item in fixed}
        recent_message_ids = {id(item) for item in retained_recent}
        ledger = tuple(
            ContextLedgerEntry(
                message_id=message.message_id,
                role=message.role,
                turn_id=message.turn_id,
                tokens=token_costs[id(message)],
                retained=id(message) in retained_ids,
                reason=(
                    "masked_output"
                    if id(message) in masked_message_ids
                    else "retained_fixed"
                    if id(message) in fixed_message_ids
                    else "retained_recent"
                    if id(message) in recent_message_ids
                    else "summarized"
                ),
            )
            for message in normalized
        )
        return ContextSnapshot(
            messages=tuple(final_messages),
            budget=budget,
            input_tokens=final_tokens,
            state=summary_result.validated_state() if omitted_messages and use_state else current_state,
            notices=tuple(notices),
            ledger=ledger,
            summarized_message_ids=summary_result.source_message_ids if omitted_messages else (),
            memory_entry_ids=memory_entry_ids,
            memory_candidates=memory_candidates,
            memory_write_error=memory_write_error,
            memory_recall_ids=memory_recall_ids,
            compression_strategy=(
                "state" if summary_message is not None and summary_message.kind == "summary"
                else "verbatim" if summary_message is not None
                else "mask" if masked_ids
                else "memory_recall" if memory_message is not None
                else "window" if omitted_messages else "none"
            ),
            degradation=tuple(degradation),
        )

    def _persist_memory(
        self,
        memory_store: Any,
        owner_scope: str,
        source_session_id: str | None,
        messages: Sequence[ContextMessage],
        summary: SummaryResult,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...], str | None]:
        candidates = extract_memory_candidates(messages, summary)
        if not candidates:
            return (), (), None
        entry_ids: list[str] = []
        errors: list[str] = []
        for candidate in candidates:
            try:
                entry = memory_store.add(
                    kind=candidate.kind,
                    content=candidate.content,
                    owner_scope=owner_scope,
                    source_session_id=source_session_id,
                    source_message_id=candidate.source_message_ids[0] if candidate.source_message_ids else None,
                    metadata={
                        "extraction": "context_policy_v1",
                        "source_message_ids": list(candidate.source_message_ids),
                    },
                    deduplicate=True,
                )
                entry_ids.append(str(entry.entry_id))
            except Exception as exc:  # pragma: no cover - defensive boundary for optional persistence
                errors.append(type(exc).__name__)
        return (
            tuple(entry_ids),
            tuple(candidate.as_dict() for candidate in candidates),
            ",".join(sorted(set(errors))) if errors else None,
        )

    def _group_turns(self, messages: Sequence[ContextMessage]) -> list[list[ContextMessage]]:
        groups: list[list[ContextMessage]] = []
        current: list[ContextMessage] = []
        current_key: object = object()
        for message in messages:
            key = message.turn_id
            if key is None:
                # A user message starts a new implicit round.  Assistant and
                # tool messages remain attached to that round.
                if message.role == "user" and current:
                    groups.append(current)
                    current = []
                current.append(message)
                continue
            if current and key != current_key:
                groups.append(current)
                current = []
            current_key = key
            current.append(message)
        if current:
            groups.append(current)
        return groups

    def _summary_pressure(
        self,
        messages: Sequence[ContextMessage],
        token_costs: Mapping[int, int],
        input_budget: int,
    ) -> bool:
        """Return whether the old-message region crossed the 70% guardrail."""

        variable = [
            message
            for message in messages
            if message.role != "system" and not message.pinned
        ]
        groups = self._group_turns(variable)
        if not groups or self.config.recent_turns <= 0:
            return False
        recent_count = min(self.config.recent_turns, len(groups))
        old_messages = [
            message
            for group in groups[:-recent_count]
            for message in group
        ]
        old_tokens = sum(token_costs[id(message)] for message in old_messages)
        return bool(old_messages) and old_tokens > input_budget * self.config.summary_trigger_ratio

    def _summarize(
        self,
        messages: Sequence[ContextMessage],
        current_state: Mapping[str, object],
    ) -> SummaryResult:
        if not messages:
            return SummaryResult(state=validate_state(current_state))
        result = self.summarizer.summarize(messages)
        if not isinstance(result, SummaryResult):
            raise ContextBuildError("summary provider must return SummaryResult")
        # A provider may use current state as context, but its output itself is
        # always schema-checked before it reaches the model prompt.
        try:
            result.validated_state()
        except StateValidationError as exc:
            raise ContextBuildError(f"invalid structured summary: {exc}") from exc
        return result

    def _summary_message(self, result: SummaryResult) -> ContextMessage:
        state = result.validated_state()
        return ContextMessage(
            role="system",
            kind="summary",
            message_id="context-state-summary",
            content="[STATE] " + render_state(state, variant=self.config.state_variant),
            metadata={"schema": "qlh.harness.state.v1", "variant": self.config.state_variant},
        )

    def _verbatim_message(
        self,
        messages: Sequence[ContextMessage],
        token_budget: int,
    ) -> tuple[ContextMessage | None, int]:
        """Fit a verbatim block by dropping complete oldest turn groups."""

        if token_budget <= 0 or not messages:
            return None, 0
        groups = self._group_turns(messages)
        selected = [message for group in groups for message in group]
        content = compact_verbatim(selected)
        if self.config.verbatim_max_characters and len(content) > self.config.verbatim_max_characters:
            # Character bounding is only a guard against pathological input;
            # whole turn groups are removed below, never sliced.
            while groups and len(compact_verbatim(selected)) > self.config.verbatim_max_characters:
                groups.pop(0)
                selected = [message for group in groups for message in group]
            content = compact_verbatim(selected)
        while groups:
            message = ContextMessage(
                role="system",
                kind="verbatim",
                message_id="context-verbatim",
                content="[VERBATIM]\n" + content,
                metadata={
                    "source_message_ids": [item.message_id for item in selected if item.message_id],
                    "dropped_oldest": len(messages) - len(selected),
                },
            )
            tokens = count_message(self.tokenizer, message)
            if tokens <= token_budget:
                return message, tokens
            groups.pop(0)
            selected = [item for group in groups for item in group]
            content = compact_verbatim(selected)
        return None, 0

    def _recall_memory(
        self,
        memory_store: Any,
        owner_scope: str,
        query: str,
        token_budget: int,
    ) -> tuple[ContextMessage | None, tuple[str, ...]]:
        if token_budget <= 0 or not query.strip():
            return None, ()
        try:
            hits = tuple(memory_store.search(query, owner_scope=owner_scope, limit=self.config.memory_recall_limit))
        except Exception:
            return None, ()
        selected: list[Any] = []
        for hit in hits:
            content = getattr(hit, "content", None)
            entry_id = getattr(hit, "entry_id", None)
            if not isinstance(content, str) or not content.strip() or not entry_id:
                continue
            candidate = ContextMessage(
                role="system",
                kind="memory",
                message_id=f"context-memory-{entry_id}",
                content="[MEMORY] " + content.strip(),
                metadata={"entry_id": str(entry_id), "owner_scope": owner_scope},
            )
            candidate_content = "\n".join(item.content for item in (*selected, candidate))
            aggregate = ContextMessage(
                role="system",
                kind="memory",
                message_id="context-memory-recall",
                content=candidate_content,
                metadata={"entry_ids": [item.metadata["entry_id"] for item in (*selected, candidate)], "owner_scope": owner_scope},
            )
            if count_message(self.tokenizer, aggregate) > token_budget:
                break
            selected.append(candidate)
        if not selected:
            return None, ()
        content = "\n".join(item.content for item in selected)
        message = ContextMessage(
            role="system",
            kind="memory",
            message_id="context-memory-recall",
            content=content,
            metadata={"entry_ids": [item.metadata["entry_id"] for item in selected], "owner_scope": owner_scope},
        )
        return message, tuple(str(item.metadata["entry_id"]) for item in selected)

    def _shrink_summary(
        self,
        result: SummaryResult,
        token_budget: int,
    ) -> tuple[ContextMessage, SummaryResult, int]:
        if token_budget <= 0:
            empty = SummaryResult(
                state={"what": [], "decisions": [], "artifacts": [], "open": [], "next": []},
                source_message_ids=result.source_message_ids,
            )
            message = self._summary_message(empty)
            return message, empty, count_message(self.tokenizer, message)
        state = result.validated_state()
        # Remove least important tail entries until the summary fits.  STATE
        # remains valid at every step, and no arbitrary raw text is injected.
        fields = ("next", "open", "artifacts", "decisions", "what")
        while True:
            candidate = SummaryResult(state=state, source_message_ids=result.source_message_ids)
            message = self._summary_message(candidate)
            tokens = count_message(self.tokenizer, message)
            if tokens <= token_budget:
                return message, candidate, tokens
            removed = False
            for field_name in fields:
                if state[field_name]:
                    state[field_name] = state[field_name][:-1]
                    removed = True
                    break
            if not removed:
                return message, candidate, tokens

    @staticmethod
    def _order_fixed_and_recent(
        fixed: Sequence[ContextMessage],
        summary: ContextMessage | None,
        memory: ContextMessage | None,
        recent: Sequence[ContextMessage],
    ) -> list[ContextMessage]:
        system = [message for message in fixed if message.role == "system"]
        pinned = [message for message in fixed if message.role != "system"]
        return system + pinned + ([summary] if summary is not None else []) + ([memory] if memory is not None else []) + list(recent)
