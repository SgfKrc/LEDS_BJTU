import json

import pytest

from tests.simulation.task_graph_harness import (
    SIMULATION_SCHEMA_VERSION,
    SimulationScenarioError,
    TaskGraphSimulationHarness,
    available_scenarios,
)


def test_simulation_catalog_is_fixed_and_explicit():
    assert [item.scenario_id for item in available_scenarios()] == [
        "cancellation_during_worker_stage",
        "dual_remote_success",
        "fallback_after_worker_loss",
    ]


@pytest.mark.parametrize(
    ("scenario_id", "expected_state"),
    [
        ("dual_remote_success", "completed"),
        ("fallback_after_worker_loss", "completed"),
        ("cancellation_during_worker_stage", "cancelled"),
    ],
)
def test_task_graph_simulation_runs_without_external_dependencies(
    scenario_id,
    expected_state,
):
    report = TaskGraphSimulationHarness().run(scenario_id)

    assert report["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert report["scenario_id"] == scenario_id
    assert report["workflow"]["state"] == expected_state
    assert report["execution_environment"] == {
        "kind": "in_memory_provider_simulation",
        "network_io": False,
        "subprocesses_started": False,
        "real_model_loaded": False,
        "physical_nodes": False,
    }
    assert all(item["active_reservations"] == 0 for item in report["providers"])


def test_simulation_success_preserves_distinct_nodes_and_no_output_content():
    report = TaskGraphSimulationHarness().run("dual_remote_success")
    stages = {item["stage_id"]: item for item in report["workflow"]["stages"]}

    assert stages["candidate_a"]["attempts"][0]["node_id"] == "sim-node-a"
    assert stages["candidate_b"]["attempts"][0]["node_id"] == "sim-node-b"
    assert stages["aggregate"]["attempts"][0]["node_id"] == "sim-master"
    serialized = json.dumps(report, sort_keys=True)
    assert "simulation-output" not in serialized


def test_simulation_fallback_records_retry_without_error_body():
    report = TaskGraphSimulationHarness().run("fallback_after_worker_loss")
    stage = report["workflow"]["stages"][0]

    assert stage["retry_count"] == 1
    assert [item["provider_id"] for item in stage["attempts"]] == [
        "sim-worker-primary",
        "sim-worker-fallback",
    ]
    assert stage["last_retry_error_code"] == "fake_worker_disconnected"
    assert "error" not in stage


def test_simulation_rejects_unknown_scenario():
    with pytest.raises(SimulationScenarioError):
        TaskGraphSimulationHarness().run("not-a-scenario")
