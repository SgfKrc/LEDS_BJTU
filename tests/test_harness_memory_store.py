import sqlite3

import pytest

from harness_workbench.memory import MemoryStore


def test_memory_store_is_scope_isolated_and_user_owned(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    fact = store.add(
        kind="fact",
        content="The user's preferred language is Chinese.",
        owner_scope="user-a",
        source_session_id="sess_a",
        metadata={"source": "conversation"},
    )
    preference = store.add(kind="preference", content="Use concise answers.", owner_scope="user-a")
    decision = store.add(kind="decision", content="Keep memory on the primary node.", owner_scope="user-b")

    assert store.get(fact.entry_id, owner_scope="user-a").as_dict()["status"] == "active"
    assert [entry.entry_id for entry in store.list(owner_scope="user-a")] == [preference.entry_id, fact.entry_id]
    assert store.list(owner_scope="user-b")[0].entry_id == decision.entry_id
    with pytest.raises(KeyError):
        store.get(fact.entry_id, owner_scope="user-b")
    assert store.health(owner_scope="user-a")["active_entries"] == 2


def test_memory_store_soft_delete_requires_confirmation_and_keeps_audit_row(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    entry = store.add(kind="fact", content="A durable fact", owner_scope="user-a")
    with pytest.raises(ValueError, match="confirm=True"):
        store.delete(entry.entry_id, owner_scope="user-a")

    deleted = store.delete(entry.entry_id, owner_scope="user-a", confirm=True, reason="user requested removal")
    assert deleted.status == "deleted"
    assert store.list(owner_scope="user-a") == []
    assert store.get(entry.entry_id, owner_scope="user-a", include_deleted=True).invalidated_reason == "user requested removal"


def test_memory_store_invalidation_and_expiry_are_not_returned_as_active(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    invalidated = store.add(kind="decision", content="Old decision", owner_scope="user-a")
    expired = store.add(kind="fact", content="Temporary fact", owner_scope="user-a", valid_until=1)
    store.invalidate(invalidated.entry_id, owner_scope="user-a", reason="superseded")

    assert store.list(owner_scope="user-a") == []
    assert store.get(invalidated.entry_id, owner_scope="user-a", include_invalidated=True).status == "invalidated"
    assert store.get(expired.entry_id, owner_scope="user-a", include_invalidated=True).status == "expired"


def test_memory_store_rejects_unsafe_fields_and_schema_is_sqlite(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError):
        store.add(kind="other", content="x")
    with pytest.raises(ValueError):
        store.add(kind="fact", content="x", owner_scope="../other")
    with pytest.raises(ValueError):
        store.add(kind="fact", content="x", source_session_id="C:\\secret")

    with sqlite3.connect(store.path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_entries)")}
    assert {"entry_id", "owner_scope", "kind", "content", "fingerprint", "deleted_at", "invalidated_at"} <= columns
