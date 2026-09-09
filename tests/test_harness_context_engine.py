"""S1 context-engine contract tests for the independent harness workbench."""

from __future__ import annotations

import pytest

from harness_workbench.context_engine import (
    ContextBudget,
    ContextBudgetError,
    ContextMessage,
    ContextPolicy,
    ContextPolicyConfig,
    PinnedContentOverflow,
    StateValidationError,
    apply_state_patch,
)
from harness_workbench.context_engine.tokenizer import FixedTokenCounter


def _budget(tokens: int) -> ContextBudget:
    return ContextBudget(n_ctx=tokens + 100, max_new_tokens=50, overhead=50)


def _rounds(count: int) -> list[ContextMessage]:
    result: list[ContextMessage] = []
    for index in range(count):
        result.extend(
            [
                ContextMessage("user", f"question-{index}", f"u-{index}", index),
                ContextMessage("assistant", f"answer-{index}", f"a-{index}", index),
            ]
        )
    return result


def test_budget_rejects_generation_that_leaves_no_input_room() -> None:
    with pytest.raises(ContextBudgetError):
        ContextBudget(n_ctx=512, max_new_tokens=256, overhead=256).as_dict()


def test_context_within_budget_keeps_original_order() -> None:
    messages = [ContextMessage("system", "rules"), ContextMessage("user", "hello")]
    snapshot = ContextPolicy(tokenizer=FixedTokenCounter()).build(messages, _budget(100))
    assert [message.content for message in snapshot.messages] == ["rules", "hello"]
    assert snapshot.input_tokens <= snapshot.input_budget
    assert not snapshot.notices


def test_long_history_summarizes_at_round_boundaries_and_emits_notice() -> None:
    policy = ContextPolicy(
        config=ContextPolicyConfig(recent_turns=2, recent_turn_ratio=0.6),
        tokenizer=FixedTokenCounter(),
    )
    snapshot = policy.build(_rounds(8), _budget(220))
    assert snapshot.input_tokens <= snapshot.input_budget
    assert any(notice.code == "context.summarized" for notice in snapshot.notices)
    assert any(message.kind == "summary" for message in snapshot.messages)
    retained_turns = {message.turn_id for message in snapshot.messages if message.kind != "summary"}
    assert retained_turns.issubset({6, 7})
    assert snapshot.summarized_message_ids


def test_summary_ratio_can_trigger_before_hard_budget_limit() -> None:
    messages = [
        ContextMessage("user", "old " * 25, "old-user", 0),
        ContextMessage("assistant", "decision", "old-assistant", 0),
        ContextMessage("user", "new", "new-user", 1),
        ContextMessage("assistant", "answer", "new-assistant", 1),
    ]
    policy = ContextPolicy(
        config=ContextPolicyConfig(recent_turns=1, recent_turn_ratio=0.5),
        tokenizer=FixedTokenCounter(),
    )
    snapshot = policy.build(messages, _budget(130))
    assert snapshot.input_tokens <= snapshot.input_budget
    summary_notice = next(
        notice for notice in snapshot.notices if notice.code == "context.summarized"
    )
    assert summary_notice.details["trigger"] == "summary_ratio"
    assert {message.turn_id for message in snapshot.messages if message.kind != "summary"} == {1}


def test_pinned_messages_are_never_clipped() -> None:
    messages = [
        ContextMessage("system", "rules"),
        ContextMessage("user", "keep this fact", pinned=True),
        ContextMessage("user", "old context"),
    ]
    snapshot = ContextPolicy(tokenizer=FixedTokenCounter()).build(messages, _budget(45))
    assert any(message.content == "keep this fact" for message in snapshot.messages)


def test_pinned_overflow_fails_closed() -> None:
    messages = [ContextMessage("user", "x" * 100, pinned=True)]
    with pytest.raises(PinnedContentOverflow):
        ContextPolicy(tokenizer=FixedTokenCounter()).build(messages, _budget(10))


def test_older_tool_outputs_are_masked_but_latest_is_retained() -> None:
    messages = [
        ContextMessage("user", "first"),
        ContextMessage("tool", "old tool output", kind="tool"),
        ContextMessage("user", "second"),
        ContextMessage("tool", "latest tool output", kind="tool"),
    ]
    snapshot = ContextPolicy(tokenizer=FixedTokenCounter()).build(messages, _budget(100))
    contents = [message.content for message in snapshot.messages]
    assert "old tool output" not in contents
    assert "latest tool output" in contents
    assert any(notice.code == "context.masked_tool_output" for notice in snapshot.notices)
    old_entry = next(entry for entry in snapshot.ledger if entry.message_id is None and entry.role == "tool")
    assert old_entry.retained is False
    assert old_entry.reason == "masked_output"


def test_state_patch_rejects_unknown_fields_and_requires_delete_confirmation() -> None:
    with pytest.raises(StateValidationError):
        apply_state_patch({}, {"secret": ["do not accept"]})
    with pytest.raises(StateValidationError):
        apply_state_patch({}, {"delete": ["what"]})
    state = apply_state_patch(
        {"what": ["old"]},
        {"what": ["new"], "delete": ["open"]},
        allow_delete=True,
    )
    assert state["what"] == ["new"]
    assert state["open"] == []
