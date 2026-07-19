import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, TaskGraphCoordinator, WorkflowExecutionError
from task_journal import SQLiteTaskJournal
from task_provider import (
    DeterministicFakeProvider,
    InProcessWorkerProvider,
    LocalFullModelProvider,
    ProviderRegistry,
    StageResult,
)


def _single_stage(*, pure=True, lease_timeout_seconds=1.0):
    return [
        StageSpec(
            "answer",
            "full_inference",
            provider="primary",
            fallback_providers=("fallback",),
            pure=pure,
            lease_timeout_seconds=lease_timeout_seconds,
        ),
    ]


def _assert_no_active_reservations(coordinator):
    assert all(
        status["active_reservations"] == 0
        for status in coordinator.provider_status()
    )


def test_accept_timeout_falls_back_before_creating_a_lease():
    registry = ProviderRegistry()
    primary = DeterministicFakeProvider("primary", accept_failures=1)
    fallback = DeterministicFakeProvider("fallback")
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    output, workflow = coordinator.run(
        _single_stage(),
        "answer",
        {"message": "question"},
        workflow_id="wf_acceptretry",
    )

    stage = workflow["stages"][0]
    assert output == {"content": "answer"}
    assert primary.call_records() == []
    assert len(fallback.call_records()) == 1
    assert stage["selected_provider"] == "fallback"
    assert stage["lease_epoch"] == 1
    assert stage["retry_count"] == 1
    assert workflow["retry_count"] == 1
    assert workflow["result_rejection_count"] == 0
    assert [attempt["provider"] for attempt in stage["attempts"]] == [
        "fallback",
    ]
    assert stage["winner_attempt_id"] == stage["attempts"][0]["attempt_id"]
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_retryable_disconnect_expires_old_attempt_and_fences_late_result():
    registry = ProviderRegistry()
    primary = DeterministicFakeProvider("primary", execution_failures=1)
    fallback = DeterministicFakeProvider(
        "fallback",
        output_factory=lambda request, cancel_event: {"content": "winner"},
    )
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    output, workflow = coordinator.run(
        _single_stage(),
        "answer",
        {"message": "question"},
        workflow_id="wf_disconnect1",
    )
    stage = workflow["stages"][0]
    old_attempt, winner = stage["attempts"]

    assert output == {"content": "winner"}
    assert stage["lease_epoch"] == 2
    assert old_attempt["state"] == "expired"
    assert old_attempt["lease_epoch"] == 1
    assert winner["state"] == "completed"
    assert winner["lease_epoch"] == 2
    assert stage["winner_attempt_id"] == winner["attempt_id"]

    late = coordinator.submit_stage_result(
        "wf_disconnect1",
        "answer",
        StageResult(
            output={"content": "late-old-result"},
            provider_id="primary",
            attempt_id=old_attempt["attempt_id"],
            lease_epoch=old_attempt["lease_epoch"],
        ),
    )
    duplicate = coordinator.submit_stage_result(
        "wf_disconnect1",
        "answer",
        StageResult(
            output={"content": "winner"},
            provider_id="fallback",
            attempt_id=winner["attempt_id"],
            lease_epoch=winner["lease_epoch"],
        ),
    )
    conflicting_duplicate = coordinator.submit_stage_result(
        "wf_disconnect1",
        "answer",
        StageResult(
            output={"content": "different"},
            provider_id="fallback",
            attempt_id=winner["attempt_id"],
            lease_epoch=winner["lease_epoch"],
        ),
    )

    assert late["status"] == "rejected"
    assert late["reason"] == "winner_already_committed"
    assert duplicate["status"] == "idempotent"
    assert conflicting_duplicate["status"] == "rejected"
    assert conflicting_duplicate["reason"] == "winner_digest_mismatch"
    current = coordinator.get("wf_disconnect1")
    assert current["stages"][0]["winner_attempt_id"] == winner["attempt_id"]
    assert current["retry_count"] == 1
    assert current["result_rejection_count"] == 2
    assert current["stages"][0]["last_result_rejection_reason"] == (
        "winner_digest_mismatch"
    )
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_execution_timeout_rejects_result_and_reassigns_with_new_epoch(tmp_path):
    journal = SQLiteTaskJournal(str(tmp_path / "fencing.sqlite3"))
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider(
        "primary",
        delay_seconds=0.3,
        output_factory=lambda request, cancel_event: {"content": "too-late"},
    ))
    registry.register(DeterministicFakeProvider(
        "fallback",
        output_factory=lambda request, cancel_event: {"content": "on-time"},
    ))
    coordinator = TaskGraphCoordinator(
        journal=journal,
        provider_registry=registry,
    )

    output, workflow = coordinator.run(
        _single_stage(lease_timeout_seconds=0.15),
        "answer",
        {"message": "question"},
        workflow_id="wf_leasetimeout",
    )

    stage = workflow["stages"][0]
    assert output == {"content": "on-time"}
    assert [attempt["state"] for attempt in stage["attempts"]] == [
        "expired",
        "completed",
    ]
    assert [attempt["lease_epoch"] for attempt in stage["attempts"]] == [1, 2]
    assert stage["winner_attempt_id"] == stage["attempts"][1]["attempt_id"]
    assert stage["retry_count"] == 1
    assert stage["result_rejection_count"] == 1
    rejection_events = [
        event
        for event in journal.list_events("wf_leasetimeout")
        if event["event_type"] == "stage_result_rejected"
    ]
    assert len(rejection_events) == 1
    assert rejection_events[0]["payload"]["reason"] == "lease_expired"
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_local_model_is_fenced_by_epoch_but_not_forcibly_timed_out():
    registry = ProviderRegistry()

    def slow_local(request, cancel_event):
        time.sleep(0.02)
        return {"content": "local-result"}

    registry.register(LocalFullModelProvider(
        slow_local,
        provider_id="primary",
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    stages = [
        StageSpec(
            "answer",
            "full_inference",
            provider="primary",
            lease_timeout_seconds=0.001,
        ),
    ]

    output, workflow = coordinator.run(
        stages,
        "answer",
        {"message": "question"},
        workflow_id="wf_localnolease",
    )

    attempt = workflow["stages"][0]["attempts"][0]
    assert output == {"content": "local-result"}
    assert attempt["lease_epoch"] == 1
    assert attempt["lease_enforced"] is False
    assert attempt["state"] == "completed"
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_invalid_output_schema_is_rejected_without_fallback():
    registry = ProviderRegistry()
    primary = DeterministicFakeProvider(
        "primary",
        output_factory=lambda request, cancel_event: {"content": {"not-json"}},
    )
    fallback = DeterministicFakeProvider("fallback")
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError, match="invalid_result_schema"):
        coordinator.run(
            _single_stage(),
            "answer",
            {"message": "question"},
            workflow_id="wf_invalidschema",
        )

    stage = coordinator.get("wf_invalidschema")["stages"][0]
    assert stage["attempts"][0]["state"] == "failed"
    assert stage["winner_attempt_id"] == ""
    assert fallback.call_records() == []
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_wrong_epoch_is_rejected_before_winner_and_valid_result_can_commit():
    start_barrier = threading.Barrier(2)
    release_provider = threading.Event()
    registry = ProviderRegistry()
    registry.register(DeterministicFakeProvider(
        "primary",
        start_barrier=start_barrier,
        block_event=release_provider,
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    completed = []
    errors = []

    def run_workflow():
        try:
            completed.append(coordinator.run(
                [StageSpec("answer", "full_inference", provider="primary")],
                "answer",
                {"message": "question"},
                workflow_id="wf_wrongepoch",
            ))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    start_barrier.wait(timeout=5)
    attempt = coordinator.get("wf_wrongepoch")["stages"][0]["attempts"][0]

    rejected = coordinator.submit_stage_result(
        "wf_wrongepoch",
        "answer",
        StageResult(
            output={"content": "wrong-epoch"},
            provider_id="primary",
            attempt_id=attempt["attempt_id"],
            lease_epoch=attempt["lease_epoch"] + 1,
        ),
    )
    release_provider.set()
    thread.join(5)

    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "attempt_epoch_mismatch"
    assert not errors
    assert completed[0][0] == {"content": "answer"}
    assert completed[0][1]["stages"][0]["winner_attempt_id"] == attempt[
        "attempt_id"
    ]
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_provider_result_attempt_identity_is_not_silently_rebound():
    class WrongAttemptProvider(LocalFullModelProvider):
        def execute(self, attempt, reservation, cancel_event):
            result = super().execute(attempt, reservation, cancel_event)
            return StageResult(
                output=result.output,
                provider_id=result.provider_id,
                metadata=result.metadata,
                attempt_id="att_foreign01",
                lease_epoch=attempt.lease_epoch,
            )

    registry = ProviderRegistry()
    registry.register(WrongAttemptProvider(
        lambda request, cancel_event: {"content": "foreign"},
        provider_id="primary",
        provider_kind="in_process_worker",
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(
        WorkflowExecutionError, match="attempt_not_owned_by_stage",
    ):
        coordinator.run(
            [StageSpec("answer", "full_inference", provider="primary")],
            "answer",
            {"message": "question"},
            workflow_id="wf_wrongattempt",
        )

    stage = coordinator.get("wf_wrongattempt")["stages"][0]
    assert stage["winner_attempt_id"] == ""
    assert stage["attempts"][0]["state"] == "failed"
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_partial_provider_result_identity_is_rejected_not_completed():
    class PartialIdentityProvider(LocalFullModelProvider):
        def execute(self, attempt, reservation, cancel_event):
            result = super().execute(attempt, reservation, cancel_event)
            return StageResult(
                output=result.output,
                provider_id=result.provider_id,
                metadata=result.metadata,
                attempt_id="",
                lease_epoch=attempt.lease_epoch,
            )

    registry = ProviderRegistry()
    registry.register(PartialIdentityProvider(
        lambda request, cancel_event: {"content": "partial"},
        provider_id="primary",
        provider_kind="in_process_worker",
    ))
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(
        WorkflowExecutionError, match="attempt_not_owned_by_stage",
    ):
        coordinator.run(
            [StageSpec("answer", "full_inference", provider="primary")],
            "answer",
            {"message": "question"},
            workflow_id="wf_partialidentity",
        )

    stage = coordinator.get("wf_partialidentity")["stages"][0]
    assert stage["winner_attempt_id"] == ""
    assert stage["attempts"][0]["state"] == "failed"
    _assert_no_active_reservations(coordinator)
    coordinator.close()


@pytest.mark.parametrize(
    ("pure", "retryable", "expected_code"),
    [
        (False, True, "fake_worker_disconnected"),
        (True, False, "deterministic_provider_failure"),
    ],
)
def test_fallback_requires_both_pure_stage_and_retryable_error(
    pure,
    retryable,
    expected_code,
):
    registry = ProviderRegistry()
    if retryable:
        primary = DeterministicFakeProvider("primary", execution_failures=1)
    else:
        primary = DeterministicFakeProvider(
            "primary", fail_stage_ids=("answer",),
        )
    fallback = DeterministicFakeProvider("fallback")
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError, match=expected_code):
        coordinator.run(
            _single_stage(pure=pure),
            "answer",
            {"message": "question"},
            workflow_id=f"wf_guardcase{int(pure)}{int(retryable)}",
        )

    snapshot = coordinator.get(f"wf_guardcase{int(pure)}{int(retryable)}")
    stage = snapshot["stages"][0]
    assert len(stage["attempts"]) == 1
    assert stage["attempts"][0]["state"] == "failed"
    assert fallback.call_records() == []
    _assert_no_active_reservations(coordinator)
    coordinator.close()


def test_fault_loop_commits_each_aggregate_once_and_never_accepts_old_epoch():
    loop_count = 20
    registry = ProviderRegistry()
    faulty = DeterministicFakeProvider(
        "faulty",
        execution_failures=loop_count,
    )
    backup = DeterministicFakeProvider("backup")
    stable = DeterministicFakeProvider("stable")
    aggregate_calls = []

    def aggregate(request, cancel_event):
        aggregate_calls.append(request.workflow_id)
        return {
            "content": "+".join(
                request.dependencies[stage_id]["content"]
                for stage_id in sorted(request.dependencies)
            ),
        }

    registry.register(faulty)
    registry.register(backup)
    registry.register(stable)
    registry.register(InProcessWorkerProvider("aggregate", aggregate))
    coordinator = TaskGraphCoordinator(
        provider_registry=registry,
        max_parallel_stages=2,
    )
    stages = [
        StageSpec(
            "candidate_a",
            "full_inference",
            provider="faulty",
            fallback_providers=("backup",),
            pure=True,
        ),
        StageSpec("candidate_b", "full_inference", provider="stable"),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
            provider="aggregate",
        ),
    ]

    for index in range(loop_count):
        workflow_id = f"wf_faultloop{index:02d}"
        output, workflow = coordinator.run(
            stages,
            "aggregate",
            {"message": "question"},
            workflow_id=workflow_id,
        )
        by_id = {stage["stage_id"]: stage for stage in workflow["stages"]}
        old_attempt, winner = by_id["candidate_a"]["attempts"]
        rejected = coordinator.submit_stage_result(
            workflow_id,
            "candidate_a",
            StageResult(
                output={"content": f"stale-{index}"},
                provider_id="faulty",
                attempt_id=old_attempt["attempt_id"],
                lease_epoch=old_attempt["lease_epoch"],
            ),
        )

        assert output == {"content": "candidate_a+candidate_b"}
        assert old_attempt["state"] == "expired"
        assert winner["lease_epoch"] == 2
        assert rejected["status"] == "rejected"
        assert rejected["winner_attempt_id"] == winner["attempt_id"]
        assert aggregate_calls.count(workflow_id) == 1
        _assert_no_active_reservations(coordinator)

    assert len(aggregate_calls) == loop_count
    coordinator.close()
