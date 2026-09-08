"""Context selection policy for small-model conversations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .budget import ContextBudget
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

    def __post_init__(self) -> None:
        if self.recent_turns < 0:
            raise ValueError("recent_turns must be non-negative")
        if not 0 < self.summary_trigger_ratio <= 1:
            raise ValueError("summary_trigger_ratio must be between zero and one")
        if not 0 < self.recent_turn_ratio <= 1:
            raise ValueError("recent_turn_ratio must be between zero and one")
        if self.max_tool_outputs < 0:
            raise ValueError("max_tool_outputs must be non-negative")


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
    ) -> ContextSnapshot:
        normalized = [ContextMessage.from_value(value) for value in messages]
        if not normalized:
            return ContextSnapshot(
                messages=(),
                budget=budget,
                input_tokens=0,
                state=validate_state(state or {}),
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
        if full_tokens <= input_budget and not summary_pressure:
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
        summary_result = self._summarize(old_messages, current_state)
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
        if old_messages:
            summary_budget = max(0, available_without_summary - recent_reserve)
            summary_message, summary_result, summary_tokens = self._shrink_summary(
                summary_result,
                summary_budget,
            )
            if summary_tokens > summary_budget:
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
        available_for_recent = available_without_summary - summary_tokens

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
            summary_result = self._summarize(omitted_messages, current_state)
            if summary_message is not None:
                summary_budget = input_budget - fixed_tokens - sum(
                    token_costs[id(message)] for message in retained_recent
                )
                summary_message, summary_result, summary_tokens = self._shrink_summary(
                    summary_result,
                    summary_budget,
                )
                if summary_tokens > summary_budget:
                    summary_message = None
                    summary_tokens = 0
        final_messages = self._order_fixed_and_recent(fixed, summary_message, retained_recent)
        final_tokens = sum(token_costs.get(id(message), count_message(self.tokenizer, message)) for message in final_messages)
        if final_tokens > input_budget:
            # This can only happen when a custom tokenizer is inconsistent or
            # a summary was changed after selection; fail closed.
            raise ContextBuildError(
                f"context policy produced {final_tokens} tokens for budget {input_budget}"
            )

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
            state=summary_result.validated_state() if omitted_messages else current_state,
            notices=tuple(notices),
            ledger=ledger,
            summarized_message_ids=summary_result.source_message_ids if omitted_messages else (),
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
            content="[STATE] " + json.dumps(state, ensure_ascii=False, sort_keys=True),
            metadata={"schema": "qlh.harness.state.v1"},
        )

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
        recent: Sequence[ContextMessage],
    ) -> list[ContextMessage]:
        system = [message for message in fixed if message.role == "system"]
        pinned = [message for message in fixed if message.role != "system"]
        return system + pinned + ([summary] if summary is not None else []) + list(recent)
