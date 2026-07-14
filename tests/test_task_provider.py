import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import task_graph as task_graph_module
from task_graph import (
    TaskGraphCoordinator,
    WorkflowCancelled,
    WorkflowExecutionError,
)
from task_provider import (
    LocalFullModelProvider,
    ProviderBusy,
    ProviderNotFound,
    ProviderRegistrationError,
    ProviderReservationError,
    ProviderRegistry,
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)


def _request(
    *,
    workflow_id="wf_provider01",
    stage_id="candidate_a",
    stage_type="full_inference",
    provider_id="local_full_model",
):
    return StageRequest(
        workflow_id=workflow_id,
        request_id="req-provider",
        stage_id=stage_id,
        stage_type=stage_type,
        provider_id=provider_id,
        dependencies={},
        root_input={"message": "question"},
    )


def test_registry_enforces_unique_identity_and_stage_capabilities():
    registry = ProviderRegistry()
    provider = LocalFullModelProvider(
        lambda request, cancel_event: {"content": request.stage_id},
        node_id="node-local",
    )
    registry.register(provider)

    status = registry.inspect()[0]
    assert status["provider_id"] == "local_full_model"
    assert status["provider_kind"] == "local_full_model"
    assert status["node_id"] == "node-local"
    assert status["max_concurrency"] == 1
    assert status["available"] is True
    with pytest.raises(ProviderRegistrationError) as duplicate:
        registry.register(provider)
    assert duplicate.value.code == "duplicate_provider_id"
    with pytest.raises(ProviderNotFound) as missing:
        registry.reserve(_request(provider_id="missing_provider"))
    assert missing.value.code == "provider_not_found"
    with pytest.raises(ProviderUnavailable) as unsupported:
        registry.reserve(_request(stage_type="unsupported"))
    assert unsupported.value.code == "no_compatible_provider"
    reservation = registry.reserve(_request())
    with pytest.raises(ProviderBusy) as active:
        registry.unregister("local_full_model")
    assert active.value.code == "provider_has_active_reservations"
    registry.release(reservation.reservation_id)
    assert registry.unregister("local_full_model") is True
    assert registry.unregister("local_full_model") is False
    registry.close()


def test_local_provider_reservation_is_atomic_across_threads():
    registry = ProviderRegistry()
    registry.register(LocalFullModelProvider(
        lambda request, cancel_event: {"content": "unused"},
    ))
    barrier = threading.Barrier(3)
    winner_reserved = threading.Event()
    loser_finished = threading.Event()
    release_winner = threading.Event()
    reservations = []
    errors = []

    def reserve_slot(index):
        barrier.wait()
        try:
            reservation = registry.reserve(_request(
                workflow_id=f"wf_atomic0{index}",
            ))
            reservations.append(reservation)
            winner_reserved.set()
            assert release_winner.wait(5)
            registry.release(reservation.reservation_id)
        except ProviderBusy as exc:
            errors.append(exc)
            loser_finished.set()

    threads = [
        threading.Thread(target=reserve_slot, args=(index,))
        for index in (1, 2)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    assert winner_reserved.wait(5)
    assert loser_finished.wait(5)
    release_winner.set()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(reservations) == 1
    assert len(errors) == 1
    assert errors[0].retryable is True
    assert registry.inspect()[0]["active_reservations"] == 0
    registry.close()


def test_provider_inspection_failure_is_isolated_and_selection_continues():
    class ToggleInspectProvider(LocalFullModelProvider):
        fail_inspection = False

        def inspect(self):
            if self.fail_inspection:
                raise RuntimeError("inspection failed")
            return super().inspect()

    registry = ProviderRegistry()
    broken = ToggleInspectProvider(
        lambda request, cancel_event: {"content": "broken"},
        provider_id="a_broken",
    )
    healthy = LocalFullModelProvider(
        lambda request, cancel_event: {"content": "healthy"},
        provider_id="b_healthy",
    )
    registry.register(broken)
    registry.register(healthy)
    broken.fail_inspection = True

    statuses = {item["provider_id"]: item for item in registry.inspect()}
    assert statuses["a_broken"]["healthy"] is False
    assert statuses["a_broken"]["error_code"] == "provider_inspection_failed"
    assert statuses["b_healthy"]["available"] is True
    automatic = registry.reserve(_request(provider_id=""))
    assert automatic.provider_id == "b_healthy"
    registry.release(automatic.reservation_id)
    with pytest.raises(ProviderUnavailable) as explicit:
        registry.reserve(_request(provider_id="a_broken"))
    assert explicit.value.code == "provider_inspection_failed"
    registry.close()


def test_local_provider_result_metadata_excludes_output_content():
    registry = ProviderRegistry()
    provider = LocalFullModelProvider(
        lambda request, cancel_event: {
            "content": "secret candidate text",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "untrusted": "secret",
            },
            "tokens_per_second": 2.5,
            "model": "test-model",
        },
    )
    registry.register(provider)
    request = _request()
    reservation = registry.reserve(request)
    result = registry.execute(
        StageAttempt("att_provider01", request, reservation.provider_id),
        reservation,
        threading.Event(),
    )

    assert result.output["content"] == "secret candidate text"
    assert result.metadata == {
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "usage_estimated": False,
        "tokens_per_second": 2.5,
        "model": "test-model",
    }
    assert "secret candidate text" not in str(result.metadata)
    with pytest.raises(ProviderReservationError) as duplicate_execute:
        registry.execute(
            StageAttempt("att_provider02", request, reservation.provider_id),
            reservation,
            threading.Event(),
        )
    assert duplicate_execute.value.code == "reservation_already_executed"
    registry.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0
    registry.close()


def test_provider_cancel_reaches_active_attempt():
    started = threading.Event()

    def execute(request, cancel_event):
        started.set()
        assert cancel_event.wait(5)
        return {"content": "cancel observed"}

    registry = ProviderRegistry()
    registry.register(LocalFullModelProvider(execute))
    request = _request()
    reservation = registry.reserve(request)
    result = []

    def run_provider():
        result.append(registry.execute(
            StageAttempt("att_cancel01", request, reservation.provider_id),
            reservation,
            threading.Event(),
        ))

    thread = threading.Thread(target=run_provider)
    thread.start()
    assert started.wait(5)
    registry.cancel("local_full_model", "att_cancel01")
    thread.join(5)

    assert not thread.is_alive()
    assert result[0].output["content"] == "cancel observed"
    registry.release(reservation.reservation_id)
    registry.close()


def test_coordinator_executes_through_registered_provider_and_releases_slots():
    calls = []

    def execute(request, cancel_event):
        calls.append(request.stage_id)
        if request.stage_type == "aggregate":
            assert set(request.dependencies) == {"candidate_a", "candidate_b"}
        return {
            "content": request.stage_id,
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "model": "provider-model",
        }

    registry = ProviderRegistry()
    provider = LocalFullModelProvider(execute, node_id="master-node")
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    output, workflow = coordinator.run_template(
        "dual_candidate",
        {"message": "question"},
        workflow_id="wf_realprovider",
    )

    assert output["content"] == "aggregate"
    assert calls == ["candidate_a", "candidate_b", "aggregate"]
    attempts = [
        attempt
        for stage in workflow["stages"]
        for attempt in stage["attempts"]
    ]
    assert {attempt["provider"] for attempt in attempts} == {
        "local_full_model",
    }
    assert {attempt["provider_kind"] for attempt in attempts} == {
        "local_full_model",
    }
    assert {attempt["provider_node_id"] for attempt in attempts} == {
        "master-node",
    }
    assert all(attempt["reservation_id"].startswith("res_") for attempt in attempts)
    assert all(
        attempt["result_metadata"]["usage"]["prompt_tokens"] == 2
        for attempt in attempts
    )
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_coordinator_provider_failure_is_terminal_and_releases_reservation():
    def fail(request, cancel_event):
        raise RuntimeError("provider failure")

    registry = ProviderRegistry()
    provider = LocalFullModelProvider(fail)
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError, match="provider failure"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            workflow_id="wf_providerfail",
        )

    workflow = coordinator.get("wf_providerfail")
    assert workflow["state"] == "failed"
    assert workflow["stages"][0]["state"] == "failed"
    assert workflow["stages"][0]["attempts"][0]["state"] == "failed"
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_release_failure_does_not_hide_the_primary_execution_error():
    class FailingReleaseProvider(LocalFullModelProvider):
        def release(self, reservation_id):
            super().release(reservation_id)
            raise RuntimeError("release failure")

    def fail(request, cancel_event):
        raise RuntimeError("primary execution failure")

    registry = ProviderRegistry()
    provider = FailingReleaseProvider(fail)
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            workflow_id="wf_releasefail",
        )

    assert captured.value.stage_id == "candidate_a"
    assert "primary execution failure" in str(captured.value)
    assert any(
        "reservation cleanup also failed" in note
        for note in getattr(captured.value, "__notes__", [])
    )
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_release_failure_after_success_fails_the_specific_stage():
    class FailingReleaseProvider(LocalFullModelProvider):
        def release(self, reservation_id):
            super().release(reservation_id)
            raise RuntimeError("release failure")

    registry = ProviderRegistry()
    provider = FailingReleaseProvider(
        lambda request, cancel_event: {"content": request.stage_id},
    )
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            workflow_id="wf_releasesuccess",
        )

    assert captured.value.stage_id == "candidate_a"
    assert "reservation cleanup failed" in str(captured.value)
    snapshot = coordinator.get("wf_releasesuccess")
    assert snapshot["state"] == "failed"
    assert snapshot["stages"][0]["state"] == "completed"
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_coordinator_cancellation_reaches_provider_and_releases_reservation():
    started = threading.Event()
    cancel_called = threading.Event()

    class ObservableLocalProvider(LocalFullModelProvider):
        def cancel(self, attempt_id):
            cancel_called.set()
            super().cancel(attempt_id)

    def execute(request, cancel_event):
        started.set()
        assert cancel_event.wait(5)
        return {"content": "late result"}

    registry = ProviderRegistry()
    provider = ObservableLocalProvider(execute)
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    errors = []

    def run_workflow():
        try:
            coordinator.run_template(
                "dual_candidate",
                {"message": "question"},
                workflow_id="wf_provcancel",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    assert started.wait(5)
    requested = coordinator.cancel("wf_provcancel")
    thread.join(5)

    assert requested["cancel_requested"] is True
    assert cancel_called.is_set()
    assert not thread.is_alive()
    assert isinstance(errors[0], WorkflowCancelled)
    assert coordinator.get("wf_provcancel")["state"] == "cancelled"
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_cancel_after_reservation_does_not_create_a_new_attempt():
    reservation_created = threading.Event()
    return_reservation = threading.Event()

    class PausedReservationProvider(LocalFullModelProvider):
        def reserve(self, request):
            reservation = super().reserve(request)
            reservation_created.set()
            assert return_reservation.wait(5)
            return reservation

    registry = ProviderRegistry()
    provider = PausedReservationProvider(
        lambda request, cancel_event: {"content": "must not execute"},
    )
    registry.register(provider)
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    errors = []

    def run_workflow():
        try:
            coordinator.run_template(
                "dual_candidate",
                {"message": "question"},
                workflow_id="wf_rescancel1",
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    assert reservation_created.wait(5)
    coordinator.cancel("wf_rescancel1")
    return_reservation.set()
    thread.join(5)

    assert not thread.is_alive()
    assert isinstance(errors[0], WorkflowCancelled)
    snapshot = coordinator.get("wf_rescancel1")
    assert snapshot["state"] == "cancelled"
    assert snapshot["attempt_count"] == 0
    assert provider.inspect().active_reservations == 0
    coordinator.close()


def test_coordinator_missing_provider_fails_without_creating_attempt():
    coordinator = TaskGraphCoordinator(provider_registry=ProviderRegistry())

    with pytest.raises(WorkflowExecutionError, match="provider_not_found"):
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            workflow_id="wf_noprovider1",
        )

    workflow = coordinator.get("wf_noprovider1")
    assert workflow["state"] == "failed"
    assert workflow["stages"][0]["state"] == "failed"
    assert workflow["attempt_count"] == 0
    coordinator.close()


def test_callback_provider_setup_failure_terminalizes_workflow(
    monkeypatch,
):
    def fail_provider_setup(*args, **kwargs):
        raise RuntimeError("provider setup failed")

    monkeypatch.setattr(
        task_graph_module,
        "CallbackExecutionProvider",
        fail_provider_setup,
    )
    coordinator = TaskGraphCoordinator()

    with pytest.raises(WorkflowExecutionError) as captured:
        coordinator.run_template(
            "dual_candidate",
            {"message": "question"},
            lambda *args: {"content": "unused"},
            workflow_id="wf_setupfail1",
        )

    assert captured.value.stage_id == "provider_setup"
    assert coordinator.get("wf_setupfail1")["state"] == "failed"
    coordinator.close()
