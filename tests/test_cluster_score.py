from __future__ import annotations

from cluster_score import (
    MANAGEMENT_SCORE_ALGORITHM_VERSION,
    MANAGEMENT_SCORE_SCHEMA_VERSION,
    build_management_score_snapshot,
)


NOW = 1_000.0


def _node(
    node_id: str,
    *,
    state: str = "online",
    last_heartbeat: float = NOW - 1,
    avg_rtt_ms: float = 10,
    score_hint: float = 0.0,
    **info,
):
    del score_hint
    return {
        "node_id": node_id,
        "role": "client",
        "node_type": "pc",
        "state": state,
        "last_heartbeat": last_heartbeat,
        "avg_rtt_ms": avg_rtt_ms,
        "device_info": {
            "cluster_voter": True,
            "control_capabilities": ["cluster_control"],
            "cpu": {
                "physical_cores": 8,
                "freq_max_mhz": 3200,
                "usage_percent": 20,
            },
            "ram": {"total_gb": 32, "available_gb": 24},
            **info,
        },
    }


def test_snapshot_is_versioned_explainable_and_signed_without_authority_side_effect():
    snapshot = build_management_score_snapshot(
        [_node("b"), _node("a")], now=NOW, signing_secret="test-secret"
    )

    assert snapshot["schema_version"] == MANAGEMENT_SCORE_SCHEMA_VERSION
    assert snapshot["algorithm_version"] == MANAGEMENT_SCORE_ALGORITHM_VERSION
    assert snapshot["signature_algorithm"] == "hmac-sha256"
    assert snapshot["signature_status"] == "signed"
    assert snapshot["snapshot_digest"]
    assert [item["node_id"] for item in snapshot["nodes"]] == ["a", "b"]
    assert snapshot["nodes"][0]["rank"] == 1
    assert set(snapshot["nodes"][0]["score_breakdown"]) == {
        "capacity", "memory_headroom", "load_headroom", "stability", "network",
    }
    assert snapshot["nodes"][0]["minimum_qualified"] is True
    assert snapshot["nodes"][0]["eligible_voter"] is True


def test_same_observation_replays_identically_and_ties_break_by_node_id():
    nodes = [_node("node-b"), _node("node-a")]
    first = build_management_score_snapshot(nodes, now=NOW)
    second = build_management_score_snapshot(nodes, now=NOW)

    assert first == second
    assert [item["node_id"] for item in first["nodes"]] == ["node-a", "node-b"]
    assert first["signature"] is None
    assert first["signature_status"] == "unsigned_no_cluster_secret"


def test_missing_liveness_and_capability_fail_closed_without_hiding_reason():
    node = _node("offline", state="offline", last_heartbeat=0, avg_rtt_ms=-1)
    node["device_info"].pop("cluster_voter")
    node["device_info"].pop("control_capabilities")
    snapshot = build_management_score_snapshot([node], now=NOW)
    item = snapshot["nodes"][0]

    assert item["minimum_qualified"] is False
    assert item["eligible_voter"] is False
    assert "heartbeat_missing" in item["reasons"]
    assert "control_capability_missing" in item["reasons"]
    assert "network_rtt_missing" in item["reasons"]


def test_management_score_is_registered_as_read_only_get_route():
    from api_server import app

    routes = [route for route in app.routes if route.path == "/api/cluster/management-score"]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}
