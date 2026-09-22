"""HA-RECOVERY-01 journal and fenced-handoff recovery gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cluster_recovery import decide_recovery  # noqa: E402
from task_graph import TaskGraphCoordinator  # noqa: E402
from task_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402


def _append(journal, workflow_id, snapshot, event_type="workflow_state_changed"):
    journal.append_event(
        JournalEvent(
            event_id=f"evt_{workflow_id}_{snapshot['last_sequence']}",
            workflow_id=workflow_id,
            sequence=snapshot["last_sequence"],
            entity_type="workflow",
            entity_id=workflow_id,
            event_type=event_type,
            occurred_at=float(snapshot.get("updated_at", 100.0)),
            payload={"state": snapshot["state"]},
        ),
        snapshot,
    )


def _running_snapshot(workflow_id: str, **changes):
    snapshot = {
        "workflow_id": workflow_id,
        "last_sequence": 1,
        "state": "running",
        "created_at": 90.0,
        "started_at": 95.0,
        "stages": [{
            "stage_id": "pure-stage",
            "state": "running",
            "pure": True,
            "retry_safe": True,
            "attempts": [{
                "attempt_id": "attempt-1",
                "state": "running",
                "started_at": 96.0,
                "reservation_active": True,
            }],
        }],
    }
    snapshot.update(changes)
    return snapshot


def test_recovery_policy_retries_only_explicitly_safe_stage():
    decision = decide_recovery(_running_snapshot("wf_retry01"))
    assert decision.action == "retry"
    assert decision.retry_stage_ids == ("pure-stage",)
    assert decision.expired_attempts == 1


def test_recovery_policy_prefers_cancel_request_over_retry():
    snapshot = _running_snapshot("wf_cancel01", cancel_requested=True)
    decision = decide_recovery(snapshot)
    assert decision.action == "cancel"
    assert decision.reason == "recovered_cancel_request"


def test_recovery_policy_marks_fenced_handoff_timeout():
    snapshot = _running_snapshot(
        "wf_timeout01", handoff_pending=True, handoff_state="awaiting_quorum",
    )
    decision = decide_recovery(snapshot)
    assert decision.action == "handoff_timeout"
    assert decision.reason == "handoff_timeout"


def test_recovery_policy_can_continue_an_explicitly_continuable_projection():
    snapshot = _running_snapshot(
        "wf_continue01", recovery_continuable=True,
    )
    decision = decide_recovery(snapshot)
    assert decision.action == "continue"
    assert decision.reason == "journal_state_continuable"


def test_journal_retry_is_recorded_once_without_duplicate_submission(tmp_path):
    path = str(tmp_path / "recovery.sqlite3")
    seed = SQLiteTaskJournal(path)
    _append(seed, "wf_retry0201", _running_snapshot("wf_retry0201"))
    seed.close()

    journal = SQLiteTaskJournal(path)
    coordinator = TaskGraphCoordinator(journal=journal)
    summary = coordinator.recover_persisted_workflows()
    recovered = coordinator.get("wf_retry0201")
    assert summary["retried_workflows"] == 1
    assert recovered["state"] == "running"
    assert recovered["recovery_action"] == "retry"
    assert recovered["recovery_pending"] is True
    assert recovered["stages"][0]["state"] == "ready"
    assert recovered["stages"][0]["attempts"][0]["state"] == "expired"
    assert [event["event_type"] for event in journal.list_events("wf_retry0201")] == [
        "workflow_state_changed", "workflow_recovery_retry",
    ]

    replay = coordinator.recover_persisted_workflows()
    assert replay["recovered_workflows"] == 0
    assert len(journal.list_events("wf_retry0201")) == 2
    coordinator.close()


def test_journal_cancel_and_handoff_timeout_are_terminal_and_auditable(tmp_path):
    path = str(tmp_path / "recovery-terminal.sqlite3")
    journal = SQLiteTaskJournal(path)
    _append(
        journal, "wf_cancel02",
        _running_snapshot("wf_cancel02", cancel_requested=True),
        event_type="workflow_cancel_requested",
    )
    _append(
        journal, "wf_timeout02",
        _running_snapshot(
            "wf_timeout02", handoff_pending=True, handoff_state="fenced",
        ),
    )
    coordinator = TaskGraphCoordinator(journal=journal)

    summary = coordinator.recover_persisted_workflows()
    assert summary["cancelled_workflows"] == 1
    assert summary["handoff_timeouts"] == 1
    assert coordinator.get("wf_cancel02")["state"] == "cancelled"
    assert coordinator.get("wf_cancel02")["error_code"] == "recovered_cancel_request"
    assert coordinator.get("wf_timeout02")["state"] == "failed"
    assert coordinator.get("wf_timeout02")["error_code"] == "handoff_timeout"
    assert journal.list_events("wf_cancel02")[-1]["event_type"] == (
        "workflow_recovery_cancel"
    )
    assert journal.list_events("wf_timeout02")[-1]["event_type"] == (
        "workflow_recovery_handoff_timeout"
    )
    coordinator.close()


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled"])
def test_terminal_journal_state_is_never_recovered(tmp_path, state):
    journal = SQLiteTaskJournal(str(tmp_path / f"{state}.sqlite3"))
    _append(
        journal,
        f"wf_terminal_{state}",
        {
            "workflow_id": f"wf_terminal_{state}",
            "last_sequence": 1,
            "state": state,
            "created_at": 1.0,
            "stages": [],
        },
    )
    coordinator = TaskGraphCoordinator(journal=journal)
    assert coordinator.recover_persisted_workflows() == {
        "recovered_workflows": 0,
        "expired_attempts": 0,
        "failed_stages": 0,
        "skipped_stages": 0,
    }
    assert len(journal.list_events(f"wf_terminal_{state}")) == 1
    coordinator.close()
