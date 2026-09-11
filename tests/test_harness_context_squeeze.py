"""Offline development gate for HW-CTX-SQZ-01.

The fixture deliberately uses a deterministic tokenizer and no model runner.
It measures strategy behaviour, not model quality.
"""

from __future__ import annotations

from harness_workbench.context_engine import (
    ContextBudget,
    ContextMessage,
    ContextPolicy,
    ContextPolicyConfig,
)
from harness_workbench.context_engine.tokenizer import FixedTokenCounter
from harness_workbench.memory import MemoryStore


def _budget(tokens: int = 220) -> ContextBudget:
    return ContextBudget(n_ctx=tokens + 100, max_new_tokens=50, overhead=50)


def _thirty_rounds() -> list[ContextMessage]:
    messages: list[ContextMessage] = []
    for index in range(30):
        messages.extend(
            (
                ContextMessage("user", f"early fact {index} " + "context " * 8, f"u-{index}", index),
                ContextMessage("assistant", f"decision {index}", f"a-{index}", index),
            )
        )
    return messages


def test_strategy_matrix_is_bounded_and_round_safe() -> None:
    for strategy in ("adaptive", "state", "verbatim", "mask"):
        policy = ContextPolicy(
            config=ContextPolicyConfig(
                recent_turns=4,
                recent_turn_ratio=0.40,
                compression_strategy=strategy,
                state_variant="lines",
            ),
            tokenizer=FixedTokenCounter(),
        )
        snapshot = policy.build(_thirty_rounds(), _budget())

        assert snapshot.input_tokens <= snapshot.input_budget
        assert snapshot.as_dict()["degradation"]
        retained_turns = {
            message.turn_id
            for message in snapshot.messages
            if message.kind not in {"summary", "verbatim"}
        }
        assert retained_turns.issubset({26, 27, 28, 29})
        assert all(entry.reason != "summarized" or not entry.retained for entry in snapshot.ledger)
        verbatim_messages = [message for message in snapshot.messages if message.kind == "verbatim"]
        if verbatim_messages:
            source_ids = verbatim_messages[0].metadata["source_message_ids"]
            assert len(source_ids) % 2 == 0


def test_adaptive_records_state_before_verbatim_fallback() -> None:
    policy = ContextPolicy(
        config=ContextPolicyConfig(
            recent_turns=2,
            recent_turn_ratio=0.8,
            compression_strategy="adaptive",
            state_variant="compact",
        ),
        tokenizer=FixedTokenCounter(),
    )
    snapshot = policy.build(_thirty_rounds(), _budget(90))

    assert snapshot.input_tokens <= snapshot.input_budget
    assert snapshot.compression_strategy in {"state", "verbatim", "window"}
    assert any(step.strategy in {"state", "verbatim", "window"} for step in snapshot.degradation)


def test_memory_recall_is_scoped_bounded_and_cited(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    owned = store.add(kind="fact", content="The primary node owns SQLite.", owner_scope="user-a")
    store.add(kind="fact", content="The other user owns a cache.", owner_scope="user-b")
    policy = ContextPolicy(
        config=ContextPolicyConfig(recent_turns=1, memory_recall_ratio=0.30, memory_recall_limit=3),
        tokenizer=FixedTokenCounter(),
    )

    snapshot = policy.build(
        [ContextMessage("user", "What does the primary node own?", "u-new", 1)],
        _budget(160),
        memory_store=store,
        memory_owner_scope="user-a",
        memory_query="primary SQLite",
    )

    assert snapshot.memory_recall_ids == (owned.entry_id,)
    memory = next(message for message in snapshot.messages if message.kind == "memory")
    assert owned.entry_id in memory.metadata["entry_ids"]
    assert "other user" not in memory.content
    assert not any(notice.code == "context.summarized" for notice in snapshot.notices)


def test_mask_strategy_reports_mask_and_window_degradation() -> None:
    messages = _thirty_rounds()
    messages.insert(1, ContextMessage("tool", "large output", "tool-old", 0, kind="tool"))
    messages.insert(3, ContextMessage("tool", "latest output", "tool-new", 1, kind="tool"))
    policy = ContextPolicy(
        config=ContextPolicyConfig(recent_turns=2, compression_strategy="mask"),
        tokenizer=FixedTokenCounter(),
    )
    snapshot = policy.build(messages, _budget(220))

    assert any(step.strategy == "mask" for step in snapshot.degradation)
    assert "large output" not in {message.content for message in snapshot.messages}


def test_layered_budget_exposes_state_reservation() -> None:
    from harness_workbench.memory import LayeredBudget

    budget = LayeredBudget.from_input_budget(100, memory_ratio=0.20, rag_ratio=0.20, state_ratio=0.10)
    assert budget.state_budget == 10
    assert sum((budget.memory_budget, budget.rag_budget, budget.state_budget, budget.context_budget)) == 100
    assert budget.as_dict()["state_budget"] == 10


def test_state_layer_is_complete_block_bounded() -> None:
    from harness_workbench.memory import LayeredBudget, build_layered_context

    budget = LayeredBudget.from_input_budget(100, memory_ratio=0, rag_ratio=0, state_ratio=0.5)
    context = build_layered_context(
        (),
        state_items=({"text": "state block"}, {"text": "x" * 200}),
        budget=budget,
        tokenizer=FixedTokenCounter(),
    )

    assert context.layer_tokens["state"] <= budget.state_budget
    assert context.layer_omitted["state"] == 1
    assert "state block" in context.text
