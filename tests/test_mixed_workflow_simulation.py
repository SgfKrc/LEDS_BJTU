import json

import pytest

from tests.simulation.mixed_workflow_harness import (
    SIMULATION_SCHEMA_VERSION,
    MixedWorkflowSimulationHarness,
    SimulationScenarioError,
    available_scenarios,
)


def test_mixed_workflow_simulation_catalog_is_fixed():
    assert [item.scenario_id for item in available_scenarios()] == [
        "cancelled_image_cleanup",
        "image_manifest_ambiguity",
        "same_node_role_conflict",
        "successful_local_binding",
        "text_contract_limit",
        "text_model_ambiguity",
    ]


@pytest.mark.parametrize(
    "scenario_id",
    [item.scenario_id for item in available_scenarios()],
)
def test_mixed_workflow_simulation_is_local_and_converges_storage(scenario_id):
    report = MixedWorkflowSimulationHarness().run(scenario_id)

    assert report["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert report["scenario_id"] == scenario_id
    assert report["execution_environment"] == {
        "kind": "temporary_local_mixed_workflow_simulation",
        "network_io": False,
        "subprocesses_started": False,
        "real_model_loaded": False,
        "physical_nodes": False,
        "persistent_state_scope": "temporary_local_only",
    }
    assert report["contract"]["store"] == {
        "blobs": 0,
        "objects": 0,
        "uploads": 0,
        "active_leases": 0,
    }


def test_mixed_workflow_simulation_records_fixed_contract_boundaries_without_bodies():
    harness = MixedWorkflowSimulationHarness()
    success = harness.run("successful_local_binding")
    text_ambiguity = harness.run("text_model_ambiguity")
    image_ambiguity = harness.run("image_manifest_ambiguity")
    role_conflict = harness.run("same_node_role_conflict")
    text_limit = harness.run("text_contract_limit")
    cancelled = harness.run("cancelled_image_cleanup")

    assert success["contract"] == {
        "terminal_state": "completed",
        "v3_dependencies_omitted": True,
        "prompt_bound_locally": True,
        "prompt_output_sha256_present": True,
        "stage_binding_declared": True,
        "cleanup": {"blobs_removed": 1},
        "store": {"blobs": 0, "objects": 0, "uploads": 0, "active_leases": 0},
    }
    assert text_ambiguity["contract"]["rejected_codes"] == [
        "MIXED_TEXT_MODEL_SELECTION_REQUIRED",
    ]
    assert image_ambiguity["contract"]["rejected_codes"] == [
        "DIFFUSION_ARTIFACT_SELECTION_REQUIRED",
    ]
    assert role_conflict["contract"]["rejected_codes"] == [
        "MIXED_WORKER_ROLE_CONFLICT",
    ]
    assert text_limit["contract"] == {
        "terminal_state": "failed",
        "rejected_codes": ["MIXED_WORKFLOW_FAILED"],
        "image_dispatches": 0,
        "store": {"blobs": 0, "objects": 0, "uploads": 0, "active_leases": 0},
    }
    assert cancelled["contract"] == {
        "terminal_state": "cancelled",
        "rejected_codes": ["MIXED_WORKFLOW_CANCELLED"],
        "image_dispatches": 1,
        "store": {"blobs": 0, "objects": 0, "uploads": 0, "active_leases": 0},
    }
    serialized = json.dumps([
        success, text_ambiguity, image_ambiguity, role_conflict, text_limit, cancelled,
    ], sort_keys=True)
    assert "root_input" not in serialized
    assert "blob_id" not in serialized
    assert "transfer_plan" not in serialized
    assert "simulated visual prompt" not in serialized


def test_mixed_workflow_simulation_rejects_unknown_scenario():
    with pytest.raises(SimulationScenarioError):
        MixedWorkflowSimulationHarness().run("unknown-mixed-scenario")
