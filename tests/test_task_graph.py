import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (
    StageSpec,
    TaskGraphCoordinator,
    TaskGraphError,
    WorkflowCancelled,
    WorkflowExecutionError,
)


def test_dual_candidate_template_executes_dependencies_in_order():
    coordinator = TaskGraphCoordinator()
    order = []

    def execute(stage, dependencies, root_input, cancel_event):
        assert not cancel_event.is_set()
        order.append(stage.stage_id)
        if stage.stage_type == "aggregate":
            assert set(dependencies) == {"candidate_a", "candidate_b"}
            return {
                "content": dependencies["candidate_a"]["content"]
                + dependencies["candidate_b"]["content"]
            }
        assert dependencies == {}
        return {"content": stage.stage_id + root_input["message"]}

    output, workflow = coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        request_id="req-1",
        workflow_id="wf_12345678",
    )

    assert order == ["candidate_a", "candidate_b", "aggregate"]
    assert output["content"] == "candidate_aquestioncandidate_bquestion"
    assert workflow["state"] == "result_ready"
    assert workflow["result_ready_at"] is not None
    assert workflow["finished_at"] is None
    assert workflow["stage_count"] == 3
    assert workflow["completed_stage_count"] == 3
    assert workflow["attempt_count"] == 3
    assert all(stage["output_available"] for stage in workflow["stages"])
    assert all("content" not in stage for stage in workflow["stages"])

    committed = coordinator.commit_result("wf_12345678")
    assert committed["state"] == "completed"
    assert committed["finished_at"] is not None


@pytest.mark.parametrize(
    "stages, final_stage, message",
    [
        ([StageSpec("a", "x"), StageSpec("a", "y")], "a", "unique"),
        ([StageSpec("a", "x", ("missing",))], "a", "missing"),
        ([StageSpec("a", "x", ("a",))], "a", "itself"),
        (
            [StageSpec("a", "x", ("b",)), StageSpec("b", "x", ("a",))],
            "a",
            "acyclic",
        ),
    ],
)
def test_validate_rejects_invalid_graphs(stages, final_stage, message):
    with pytest.raises(TaskGraphError, match=message):
        TaskGraphCoordinator.validate(stages, final_stage)


def test_stage_failure_marks_downstream_skipped_and_workflow_failed():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "candidate_b":
            raise RuntimeError("candidate failed")
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowExecutionError) as exc_info:
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_failure1",
        )

    workflow = coordinator.get(exc_info.value.workflow_id)
    by_id = {stage["stage_id"]: stage for stage in workflow["stages"]}
    assert workflow["state"] == "failed"
    assert by_id["candidate_a"]["state"] == "completed"
    assert by_id["candidate_b"]["state"] == "failed"
    assert by_id["aggregate"]["state"] == "skipped"
    assert by_id["candidate_b"]["attempts"][0]["state"] == "failed"


def test_cancel_running_workflow_discards_stage_result():
    coordinator = TaskGraphCoordinator()
    started = threading.Event()
    release = threading.Event()
    error = []

    def execute(stage, dependencies, root_input, cancel_event):
        started.set()
        assert release.wait(5)
        return {"content": "late"}

    def run():
        try:
            coordinator.run_template(
                "dual_candidate",
                {"message": "question"},
                execute,
                workflow_id="wf_cancel12",
            )
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    snapshot = coordinator.cancel("wf_cancel12")
    assert snapshot["cancel_requested"] is True
    release.set()
    thread.join(5)

    assert not thread.is_alive()
    assert isinstance(error[0], WorkflowCancelled)
    workflow = coordinator.get("wf_cancel12")
    assert workflow["state"] == "cancelled"
    assert workflow["completed_stage_count"] == 0
    assert workflow["cancelled_stage_count"] == 3
    candidate = workflow["stages"][0]
    assert candidate["output_available"] is False
    assert candidate["attempts"][0]["state"] == "cancelled"


def test_cancel_before_registration_fences_future_workflow():
    coordinator = TaskGraphCoordinator()
    called = []

    assert coordinator.request_cancel("wf_cancelbefore") is None

    def execute(stage, dependencies, root_input, cancel_event):
        called.append(stage.stage_id)
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_cancelbefore",
        )

    assert called == []
    assert coordinator.get("wf_cancelbefore")["state"] == "cancelled"


def test_duplicate_or_unsafe_workflow_id_is_rejected():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_duplicate",
    )
    with pytest.raises(TaskGraphError, match="already exists"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_duplicate",
        )
    with pytest.raises(TaskGraphError, match="workflow_id"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="../unsafe",
        )


def test_registry_prunes_old_terminal_workflows():
    coordinator = TaskGraphCoordinator(max_records=2)

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    for workflow_id in ("wf_record01", "wf_record02", "wf_record03"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id=workflow_id,
        )
        coordinator.commit_result(workflow_id)

    workflows = coordinator.list(limit=10)
    assert len(workflows) == 2
    assert {item["workflow_id"] for item in workflows} == {
        "wf_record02", "wf_record03",
    }


def test_workflow_list_filters_by_session():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    for workflow_id, session_id in (
        ("wf_sessiona1", "session-a"),
        ("wf_sessionb1", "session-b"),
    ):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            session_id=session_id,
            workflow_id=workflow_id,
        )
        coordinator.commit_result(workflow_id)

    workflows = coordinator.list(limit=10, session_id="session-a")
    assert [item["workflow_id"] for item in workflows] == ["wf_sessiona1"]
    coordinator.close()


def test_cancel_during_final_stage_wins_over_completed_state():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        if stage.stage_id == "aggregate":
            cancel_event.set()
        return {"content": stage.stage_id}

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_finalcancel",
        )

    snapshot = coordinator.get("wf_finalcancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["stages"][-1]["output_available"] is False


def test_provider_error_after_cancel_is_reported_as_cancelled():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        cancel_event.set()
        raise RuntimeError("provider stopped while cancelling")

    with pytest.raises(WorkflowCancelled):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            execute,
            workflow_id="wf_errorcancel",
        )

    snapshot = coordinator.get("wf_errorcancel")
    assert snapshot["state"] == "cancelled"
    assert snapshot["stages"][0]["state"] == "cancelled"
    assert snapshot["stages"][0]["attempts"][0]["state"] == "cancelled"


def test_result_ready_workflow_can_be_discarded_before_result_commit():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_discarded1",
    )
    assert coordinator.get("wf_discarded1")["state"] == "result_ready"
    snapshot = coordinator.discard_result("wf_discarded1")

    assert snapshot["state"] == "cancelled"
    assert snapshot["cancel_requested"] is True
    assert snapshot["completed_stage_count"] == 3
    assert snapshot["error"] == "cancelled before result commit"


def test_completed_workflow_is_idempotent_and_cannot_be_discarded():
    coordinator = TaskGraphCoordinator()
    cancel_event = threading.Event()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_terminal1",
        cancel_event=cancel_event,
    )
    first = coordinator.commit_result("wf_terminal1")
    second = coordinator.commit_result("wf_terminal1")

    assert first["state"] == "completed"
    assert second["state"] == "completed"
    assert second["finished_at"] == first["finished_at"]
    with pytest.raises(TaskGraphError, match="cannot be discarded"):
        coordinator.discard_result("wf_terminal1")
    assert coordinator.cancel("wf_terminal1")["cancel_requested"] is False
    cancel_event.set()
    assert coordinator.get("wf_terminal1")["cancel_requested"] is False
    assert coordinator.get("wf_terminal1")["state"] == "completed"


def test_cancel_result_ready_is_terminal_and_prevents_commit():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        return {"content": stage.stage_id}

    coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        execute,
        workflow_id="wf_terminal2",
    )
    cancelled = coordinator.cancel("wf_terminal2")

    assert cancelled["state"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert coordinator.discard_result("wf_terminal2")["state"] == "cancelled"
    with pytest.raises(WorkflowCancelled):
        coordinator.commit_result("wf_terminal2")
