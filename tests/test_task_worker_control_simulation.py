import json

import pytest

from tests.simulation.task_worker_harness import (
    SIMULATION_SCHEMA_VERSION,
    SimulationScenarioError,
    TaskWorkerControlSimulationHarness,
    available_scenarios,
)


def test_v2_control_simulation_catalog_is_fixed():
    assert [item.scenario_id for item in available_scenarios()] == [
        "cancel_acknowledgement",
        "disconnect_fences_late_result",
        "hello_replay_and_conflict",
        "lease_renewal",
        "model_identity_mismatch",
        "result_replay_and_epoch_fencing",
    ]


@pytest.mark.parametrize(
    "scenario_id",
    [item.scenario_id for item in available_scenarios()],
)
def test_v2_control_simulation_is_in_memory_and_releases_slots(scenario_id):
    report = TaskWorkerControlSimulationHarness().run(scenario_id)

    assert report["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert report["scenario_id"] == scenario_id
    assert report["execution_environment"] == {
        "kind": "in_memory_v2_control_simulation",
        "network_io": False,
        "subprocesses_started": False,
        "real_model_loaded": False,
        "physical_nodes": False,
    }
    provider = report["contract"].get("provider")
    if provider is not None:
        assert provider["active_reservations"] == 0


def test_v2_control_simulation_records_fencing_without_message_bodies():
    report = TaskWorkerControlSimulationHarness().run(
        "result_replay_and_epoch_fencing",
    )

    assert report["contract"]["rejected_codes"] == [
        "attempt_identity_mismatch",
    ]
    assert report["contract"]["replay_idempotent"] is True
    assert "stage_offer" in report["contract"]["outbound_message_types"]
    serialized = json.dumps(report, sort_keys=True)
    assert "not-retained" not in serialized
    assert "root_input" not in serialized


def test_v2_control_simulation_rejects_unknown_scenario():
    with pytest.raises(SimulationScenarioError):
        TaskWorkerControlSimulationHarness().run("unknown-v2-scenario")
