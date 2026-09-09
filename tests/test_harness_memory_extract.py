from harness_workbench.context_engine import ContextBudget, ContextMessage, ContextPolicy, ContextPolicyConfig
from harness_workbench.context_engine.tokenizer import FixedTokenCounter
from harness_workbench.memory import MemoryStore


def _budget(tokens: int) -> ContextBudget:
    return ContextBudget(n_ctx=tokens + 100, max_new_tokens=50, overhead=50)


def test_context_compression_double_writes_explicit_memory_and_deduplicates(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    messages = [
        ContextMessage(
            "user",
            "My name is Ada. " + "old context " * 30,
            "u-old",
            0,
            metadata={"memory": {"kind": "fact", "content": "The user is named Ada."}},
        ),
        ContextMessage("assistant", "acknowledged", "a-old", 0),
        ContextMessage("user", "new question", "u-new", 1),
        ContextMessage("assistant", "new answer", "a-new", 1),
    ]
    policy = ContextPolicy(config=ContextPolicyConfig(recent_turns=1), tokenizer=FixedTokenCounter())

    first = policy.build(
        messages,
        _budget(100),
        memory_store=store,
        memory_owner_scope="user-a",
        memory_source_session_id="sess-1",
    )
    second = policy.build(
        messages,
        _budget(100),
        memory_store=store,
        memory_owner_scope="user-a",
        memory_source_session_id="sess-1",
    )

    entries = store.list(owner_scope="user-a")
    assert len(entries) == 1
    assert entries[0].kind == "fact"
    assert entries[0].content == "The user is named Ada."
    assert first.memory_entry_ids == second.memory_entry_ids == (entries[0].entry_id,)
    assert any(notice.code == "context.memory_persisted" for notice in first.notices)
    assert first.as_dict()["memory_candidates"][0]["kind"] == "fact"


def test_context_compression_does_not_store_unproven_assistant_summary(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    messages = [
        ContextMessage("assistant", "The user probably likes blue. " * 20, "a-old", 0),
        ContextMessage("user", "Please summarize this", "u-old", 0),
        ContextMessage("user", "new question", "u-new", 1),
    ]
    policy = ContextPolicy(config=ContextPolicyConfig(recent_turns=1), tokenizer=FixedTokenCounter())
    snapshot = policy.build(messages, _budget(70), memory_store=store, memory_owner_scope="user-a")

    assert store.list(owner_scope="user-a") == []
    assert snapshot.memory_entry_ids == ()
    assert not any(notice.code == "context.memory_persisted" for notice in snapshot.notices)


def test_user_language_markers_extract_preference_and_decision(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    messages = [
        ContextMessage("user", "I prefer concise answers. " + "old " * 30, "u-pref", 0),
        ContextMessage("user", "I decided to keep SQLite local. " + "old " * 30, "u-decision", 1),
        ContextMessage("user", "new question", "u-new", 2),
    ]
    policy = ContextPolicy(config=ContextPolicyConfig(recent_turns=1), tokenizer=FixedTokenCounter())
    policy.build(messages, _budget(100), memory_store=store, memory_owner_scope="user-a")

    entries = store.list(owner_scope="user-a")
    assert {entry.kind for entry in entries} == {"preference", "decision"}
