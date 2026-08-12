import json

import pytest

from tests.simulation.capacity_harness import (
    SIMULATION_SCHEMA_VERSION,
    CapacitySimulationHarness,
    SimulationScenarioError,
    available_scenarios,
)


def test_capacity_simulation_catalog_is_fixed():
    assert [item.scenario_id for item in available_scenarios()] == [
        "busy_reservation_fallback",
        "cas_capacity_recovery",
        "global_parallel_bound",
        "parallel_cancellation_releases_slots",
        "single_provider_serialization",
    ]


@pytest.mark.parametrize(
    "scenario_id",
    [item.scenario_id for item in available_scenarios()],
)
def test_capacity_simulation_is_local_and_releases_slots(scenario_id):
    report = CapacitySimulationHarness().run(scenario_id)

    assert report["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert report["scenario_id"] == scenario_id
    assert report["execution_environment"] == {
        "kind": "temporary_local_capacity_simulation",
        "network_io": False,
        "subprocesses_started": False,
        "real_model_loaded": False,
        "physical_nodes": False,
        "persistent_state_scope": "temporary_local_only",
        "performance_claim": False,
    }
    for provider in report["contract"].get("providers", []):
        assert provider["active_reservations"] == 0
    if "store" in report["contract"]:
        assert report["contract"]["store"] == {
            "blobs": 0,
            "objects": 0,
            "uploads": 0,
            "active_leases": 0,
        }


def test_capacity_simulation_records_bounded_contracts_without_bodies():
    harness = CapacitySimulationHarness()
    global_bound = harness.run("global_parallel_bound")
    serialized = harness.run("single_provider_serialization")
    fallback = harness.run("busy_reservation_fallback")
    cancelled = harness.run("parallel_cancellation_releases_slots")
    cas = harness.run("cas_capacity_recovery")

    assert global_bound["contract"] == {
        "terminal_state": "result_ready",
        "configured_global_parallel_stages": 2,
        "observed_active_peak": 2,
        "third_stage_deferred_until_capacity_released": True,
        "providers": [
            {"provider_id": "sim_global_a", "active_reservations": 0, "healthy": True},
            {"provider_id": "sim_global_b", "active_reservations": 0, "healthy": True},
            {"provider_id": "sim_global_c", "active_reservations": 0, "healthy": True},
        ],
    }
    assert serialized["contract"] == {
        "terminal_state": "result_ready",
        "provider_max_concurrency": 1,
        "observed_active_peak": 1,
        "second_stage_deferred_until_slot_released": True,
        "providers": [
            {"provider_id": "sim_shared_slot", "active_reservations": 0, "healthy": True},
        ],
    }
    assert fallback["contract"] == {
        "terminal_state": "result_ready",
        "retry_count": 1,
        "last_retry_error_code": "all_providers_busy",
        "primary_execution_count": 0,
        "fallback_execution_count": 1,
        "selected_provider": "sim_busy_fallback",
        "providers": [
            {"provider_id": "sim_busy_fallback", "active_reservations": 0, "healthy": True},
            {"provider_id": "sim_busy_primary", "active_reservations": 0, "healthy": True},
        ],
    }
    assert cancelled["contract"] == {
        "terminal_state": "cancelled",
        "active_slots_before_cancel": [1, 1],
        "observed_active_peak": 2,
        "providers": [
            {"provider_id": "sim_cancel_a", "active_reservations": 0, "healthy": True},
            {"provider_id": "sim_cancel_b", "active_reservations": 0, "healthy": True},
        ],
    }
    assert cas["contract"] == {
        "rejected_codes": ["blob_store_full"],
        "cleanup": {"blobs_removed": 1, "leases_revoked": 0},
        "store": {"blobs": 0, "objects": 0, "uploads": 0, "active_leases": 0},
    }
    evidence = json.dumps([global_bound, serialized, fallback, cancelled, cas], sort_keys=True)
    assert "root_input" not in evidence
    assert "blob_id" not in evidence
    assert "content" not in evidence
    assert "transfer_plan" not in evidence


def test_capacity_simulation_rejects_unknown_scenario():
    with pytest.raises(SimulationScenarioError):
        CapacitySimulationHarness().run("unknown-capacity-scenario")
