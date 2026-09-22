from __future__ import annotations

from cluster_device_qualification import qualify_device, summarize_device_composition
from koakuma_engine import Capability


NOW = 1_000.0


def _node(node_id: str, platform: str, *, state: str = "online", **info):
    return {
        "node_id": node_id,
        "node_type": platform,
        "state": state,
        "last_heartbeat": NOW - 1,
        "device_info": info,
    }


def test_android_and_pc_use_the_same_explicit_voter_contract():
    common = {"cluster_voter": True, "capabilities": ["cluster_control", Capability.FORWARD_LAYERS]}
    pc = qualify_device(_node("pc-1", "pc", **common), now=NOW)
    android = qualify_device(_node("y700", "android", **common), now=NOW)

    assert pc.eligible_voter is True
    assert android.eligible_voter is True
    assert android.forward_layers is True


def test_platform_alone_never_qualifies_an_android_voter():
    result = qualify_device(
        _node("android-1", "android", backend_id="llama_cpp", forward_layers=True),
        now=NOW,
    )

    assert result.forward_layers is True
    assert result.eligible_voter is False
    assert "voter_not_explicit" in result.reasons
    assert "control_capability_missing" in result.reasons


def test_sleeping_or_expired_node_cannot_be_eligible():
    result = qualify_device(
        _node(
            "sleeping-1",
            "android",
            state="sleeping",
            cluster_voter=True,
            capabilities=["cluster_control"],
        ),
        now=NOW,
    )

    assert result.eligible_voter is False
    assert result.reasons == ("state_sleeping",)


def test_expired_heartbeat_is_fail_closed():
    node = _node("stale-1", "pc", cluster_voter=True, capabilities=["cluster_control"])
    node["last_heartbeat"] = NOW - 60

    result = qualify_device(node, now=NOW)

    assert result.eligible_voter is False
    assert result.reasons == ("heartbeat_expired",)


def test_composition_is_deterministic_and_reports_duplicates_without_election():
    common = {"cluster_voter": True, "capabilities": ["cluster_control"]}
    snapshot = summarize_device_composition(
        [
            _node("y700", "android", **common),
            _node("pc-1", "pc", **common),
            _node("y700", "android", state="sleeping", **common),
        ],
        now=NOW,
    )

    assert snapshot["platform_counts"] == {"android": 2, "pc": 1}
    assert snapshot["eligible_voter_ids"] == ["y700", "pc-1"]
    assert snapshot["sleeping_or_unavailable_ids"] == ["y700"]
    assert snapshot["duplicate_node_ids"] == ["y700"]
    assert snapshot["platform_affects_eligibility"] is False
