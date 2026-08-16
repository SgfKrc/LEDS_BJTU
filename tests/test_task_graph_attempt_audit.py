import copy
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, TaskGraphCoordinator
from task_graph_attempt_audit import (
    TaskGraphAttemptAuditError,
    audit_task_graph_attempts,
    validate_attempt_graph_audit,
)
from task_provider import DeterministicFakeProvider, ProviderRegistry


def _run_single(*, provider_registry=None, stages=None, workflow_id="wf_audit001"):
    coordinator = TaskGraphCoordinator(provider_registry=provider_registry)
    try:
        _, snapshot = coordinator.run(
            stages or [StageSpec("answer", "full_inference", pure=True)],
            "answer",
            {"message": "question"},
            execute_stage=(
                None
                if provider_registry is not None
                else lambda stage, dependencies, root, cancel: {
                    "content": "answer",
                }
            ),
            workflow_id=workflow_id,
        )
        return copy.deepcopy(snapshot)
    finally:
        coordinator.close()


def _stage(snapshot):
    return snapshot["stages"][0]


def _rehash(report):
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    report["digest"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return report


def _recovery_events(last_sequence, *, reason):
    events = [
        {
            "sequence": sequence,
            "event_type": "workflow_state_changed",
            "payload": {},
        }
        for sequence in range(1, last_sequence)
    ]
    events.append({
        "sequence": last_sequence,
        "event_type": "workflow_recovered_after_restart",
        "payload": {
            "previous_state": "result_ready",
            "recovery_reason": reason,
            "expired_attempts": 0,
            "failed_stages": 0,
            "skipped_stages": 0,
        },
    })
    return events


def test_g5_1_clean_snapshot_passes_and_is_detached():
    snapshot = _run_single()
    report = audit_task_graph_attempts(snapshot)

    assert report["status"] == "passed"
    assert report["mode"] == "read_only"
    assert report["runtime_actions_enabled"] is False
    assert report["summary"]["gap_count"] == 0
    assert report["recovery_audit"]["status"] == "not_applicable"
    report["attempt_graph"]["nodes"].clear()
    assert snapshot["stages"]


def test_g5_1_fallback_attempt_sequence_passes():
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider("primary", execution_failures=1))
    registry.register(DeterministicFakeProvider("fallback"))
    snapshot = _run_single(
        provider_registry=registry,
        stages=[StageSpec(
            "answer",
            "full_inference",
            provider="primary",
            fallback_providers=("fallback",),
            pure=True,
        )],
        workflow_id="wf_auditfallback",
    )

    report = audit_task_graph_attempts(snapshot)

    assert report["status"] == "passed"
    assert report["summary"]["retry_count"] == 1
    assert report["stage_audits"][0]["lease_epoch"] == 2


def test_g5_1_epoch_and_stage_epoch_gaps_are_reported():
    snapshot = _run_single()
    stage = _stage(snapshot)
    stage["attempts"][0]["lease_epoch"] = 2
    stage["lease_epoch"] = 1

    report = audit_task_graph_attempts(snapshot)
    reasons = {gap["reason_code"] for gap in report["gaps"]}

    assert report["status"] == "gaps_found"
    assert "stage_epoch_mismatch" in reasons


def test_g5_1_rejects_non_admitted_provider_and_fallback_regression():
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider("primary", execution_failures=1))
    registry.register(DeterministicFakeProvider("fallback"))
    snapshot = _run_single(
        provider_registry=registry,
        stages=[StageSpec(
            "answer",
            "full_inference",
            provider="primary",
            fallback_providers=("fallback",),
            pure=True,
        )],
        workflow_id="wf_auditprovider",
    )
    stage = _stage(snapshot)
    stage["attempts"][0]["provider"] = "rogue"
    stage["attempts"][1]["provider"] = "primary"
    stage["selected_provider"] = "primary"

    report = audit_task_graph_attempts(snapshot)
    reasons = {gap["reason_code"] for gap in report["gaps"]}

    assert "attempt_provider_not_admitted" in reasons
    assert "fallback_sequence_regressed" not in reasons

    stage["attempts"][0]["provider"] = "fallback"
    report = audit_task_graph_attempts(snapshot)
    assert "fallback_sequence_regressed" in {
        gap["reason_code"] for gap in report["gaps"]
    }


def test_g5_1_winner_ownership_digest_and_terminal_state_are_checked():
    snapshot = _run_single()
    stage = _stage(snapshot)
    stage["winner_attempt_id"] = "att_missing01"
    report = audit_task_graph_attempts(snapshot)
    assert "winner_not_owned_by_stage" in {
        gap["reason_code"] for gap in report["gaps"]
    }

    snapshot = _run_single(workflow_id="wf_auditwinner2")
    stage = _stage(snapshot)
    stage["output_sha256"] = "a" * 64
    stage["attempts"][0]["result_sha256"] = "b" * 64
    report = audit_task_graph_attempts(snapshot)
    assert "winner_result_digest_mismatch" in {
        gap["reason_code"] for gap in report["gaps"]
    }

    snapshot = _run_single(workflow_id="wf_auditwinner3")
    stage = _stage(snapshot)
    stage["state"] = "failed"
    report = audit_task_graph_attempts(snapshot)
    assert "winner_on_noncompleted_stage" in {
        gap["reason_code"] for gap in report["gaps"]
    }


def test_g5_1_running_and_terminal_reservation_invariants_are_checked():
    snapshot = _run_single()
    stage = _stage(snapshot)
    stage["attempts"][0]["reservation_active"] = True
    report = audit_task_graph_attempts(snapshot)

    assert "terminal_attempt_reservation_active" in {
        gap["reason_code"] for gap in report["gaps"]
    }


def test_g5_1_workflow_counter_mismatch_is_reported():
    snapshot = _run_single()
    snapshot["attempt_count"] = 99
    snapshot["retry_count"] = 4
    report = audit_task_graph_attempts(snapshot)
    reasons = {gap["reason_code"] for gap in report["gaps"]}

    assert "workflow_attempt_count_mismatch" in reasons
    assert "workflow_retry_count_mismatch" in reasons


def test_g5_1_cross_stage_attempt_id_reuse_is_reported():
    coordinator = TaskGraphCoordinator()
    try:
        _, snapshot = coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            lambda stage, dependencies, root, cancel: {
                "content": stage.stage_id,
            },
            workflow_id="wf_auditreuse",
        )
        snapshot = copy.deepcopy(snapshot)
    finally:
        coordinator.close()
    stages = {stage["stage_id"]: stage for stage in snapshot["stages"]}
    stages["candidate_b"]["attempts"][0]["attempt_id"] = (
        stages["candidate_a"]["attempts"][0]["attempt_id"]
    )

    report = audit_task_graph_attempts(snapshot)

    assert "attempt_id_reused_across_stages" in {
        gap["reason_code"] for gap in report["gaps"]
    }


def test_g5_1_recovery_commit_facts_pass_with_journal_evidence():
    snapshot = _run_single(workflow_id="wf_auditrecovery")
    reason = "coordinator_restarted_before_result_commit"
    snapshot["state"] = "failed"
    snapshot["recovered_after_restart"] = True
    snapshot["recovery_reason"] = reason
    snapshot["recovery_pending"] = False
    snapshot["runtime_status"] = "terminal"
    snapshot["last_sequence"] += 1

    report = audit_task_graph_attempts(
        snapshot,
        journal_events=_recovery_events(snapshot["last_sequence"], reason=reason),
    )

    assert report["status"] == "passed"
    assert report["recovery_audit"]["status"] == "passed"
    assert report["recovery_audit"]["reason_code"] == "recovery_commit_verified"


def test_g5_1_recovery_requires_event_evidence_and_contiguous_sequence():
    snapshot = _run_single(workflow_id="wf_auditrecovery2")
    reason = "coordinator_restarted_during_execution"
    snapshot["state"] = "failed"
    snapshot["recovered_after_restart"] = True
    snapshot["recovery_reason"] = reason
    snapshot["last_sequence"] = 3

    missing = audit_task_graph_attempts(snapshot)
    assert "recovery_event_evidence_missing" in {
        gap["reason_code"] for gap in missing["gaps"]
    }

    events = _recovery_events(3, reason=reason)
    events[1]["sequence"] = 3
    report = audit_task_graph_attempts(snapshot, journal_events=events)
    reasons = {gap["reason_code"] for gap in report["gaps"]}
    assert "journal_sequence_not_contiguous" in reasons


def test_g5_1_recovery_expired_attempt_and_stage_counts_are_bound_to_event():
    snapshot = _run_single(workflow_id="wf_auditrecovery3")
    reason = "coordinator_restarted_during_execution"
    stage = _stage(snapshot)
    stage["state"] = "failed"
    stage["attempts"][0]["state"] = "expired"
    stage["attempts"][0]["reservation_active"] = False
    stage["attempts"][0]["recovery_reason"] = reason
    stage["recovery_reason"] = reason
    snapshot["state"] = "failed"
    snapshot["recovered_after_restart"] = True
    snapshot["recovery_reason"] = reason
    snapshot["last_sequence"] = 2
    events = _recovery_events(2, reason=reason)
    events[-1]["payload"]["expired_attempts"] = 0
    events[-1]["payload"]["failed_stages"] = 0

    report = audit_task_graph_attempts(snapshot, journal_events=events)
    reasons = {gap["reason_code"] for gap in report["gaps"]}

    assert "recovery_expired_attempt_count_mismatch" in reasons
    assert "recovery_failed_stage_count_mismatch" in reasons


def test_g5_1_unsafe_source_fields_are_not_projected():
    snapshot = _run_single(workflow_id="wf_auditprivacy")
    snapshot["root_input"] = {"secret": "do-not-persist"}
    snapshot["error"] = "sensitive failure detail"

    report = audit_task_graph_attempts(snapshot)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert "do-not-persist" not in serialized
    assert "sensitive failure detail" not in serialized
    assert "root_input" not in serialized


def test_g5_1_validator_rejects_forbidden_or_rehashed_tampering():
    snapshot = _run_single(workflow_id="wf_audittamper")
    report = audit_task_graph_attempts(snapshot)

    forbidden = copy.deepcopy(report)
    forbidden["output"] = "not allowed"
    with pytest.raises(TaskGraphAttemptAuditError, match="forbidden"):
        validate_attempt_graph_audit(forbidden)

    changed = copy.deepcopy(report)
    changed["stage_audits"][0]["retry_count"] = 8
    _rehash(changed)
    with pytest.raises(TaskGraphAttemptAuditError, match="does not match evidence"):
        validate_attempt_graph_audit(changed)


def test_g5_1_report_validation_rejects_runtime_enable_and_digest_tamper():
    snapshot = _run_single(workflow_id="wf_auditvalidation")
    report = audit_task_graph_attempts(snapshot)

    enabled = copy.deepcopy(report)
    enabled["runtime_actions_enabled"] = True
    _rehash(enabled)
    with pytest.raises(TaskGraphAttemptAuditError, match="runtime actions"):
        validate_attempt_graph_audit(enabled)

    broken = copy.deepcopy(report)
    broken["digest"] = "0" * 64
    with pytest.raises(TaskGraphAttemptAuditError, match="digest mismatch"):
        validate_attempt_graph_audit(broken)
