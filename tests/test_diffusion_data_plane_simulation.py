import json

import pytest

from tests.simulation.diffusion_data_plane_harness import (
    SIMULATION_SCHEMA_VERSION,
    DiffusionDataPlaneSimulationHarness,
    SimulationScenarioError,
    available_scenarios,
)


def test_v3_data_plane_simulation_catalog_is_fixed():
    assert [item.scenario_id for item in available_scenarios()] == [
        "cancelled_output_cleanup",
        "chunk_replay_and_cas_dedup",
        "manifest_mismatch_rejected",
        "owner_scope_isolation",
        "restart_recovery_reclaims_scope",
    ]


@pytest.mark.parametrize(
    "scenario_id",
    [item.scenario_id for item in available_scenarios()],
)
def test_v3_data_plane_simulation_has_no_network_or_persistent_test_residue(scenario_id):
    report = DiffusionDataPlaneSimulationHarness().run(scenario_id)

    assert report["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert report["scenario_id"] == scenario_id
    assert report["execution_environment"] == {
        "kind": "temporary_local_v3_data_plane_simulation",
        "network_io": False,
        "subprocesses_started": False,
        "real_model_loaded": False,
        "physical_nodes": False,
        "persistent_state_scope": "temporary_local_only",
    }
    store = report["contract"].get("store")
    if store is not None:
        assert store == {
            "blobs": 0,
            "objects": 0,
            "uploads": 0,
            "active_leases": 0,
        }


def test_v3_data_plane_simulation_records_boundaries_without_blob_or_input_bodies():
    harness = DiffusionDataPlaneSimulationHarness()
    chunk = harness.run("chunk_replay_and_cas_dedup")
    owner = harness.run("owner_scope_isolation")
    recovery = harness.run("restart_recovery_reclaims_scope")

    assert chunk["contract"] == {
        "rejected_codes": ["upload_replay_mismatch"],
        "chunk_replay_idempotent": True,
        "commit_replay_idempotent": True,
        "cas_deduplicated": True,
        "store": {"blobs": 0, "objects": 0, "uploads": 0, "active_leases": 0},
    }
    assert owner["contract"]["rejected_codes"] == ["blob_owner_mismatch"]
    assert owner["contract"]["cleanup"]["owner_b_survived_foreign_cleanup"] is True
    assert recovery["contract"]["recovery"] == {
        "workflows_reconciled": 1,
        "blobs_removed": 1,
        "blobs_blocked": 0,
        "leases_revoked": 1,
    }
    serialized = json.dumps([chunk, owner, recovery], sort_keys=True)
    assert "blob_id" not in serialized
    assert "root_input" not in serialized
    assert "transfer_plan" not in serialized


def test_v3_data_plane_simulation_rejects_unknown_scenario():
    with pytest.raises(SimulationScenarioError):
        DiffusionDataPlaneSimulationHarness().run("unknown-v3-scenario")
