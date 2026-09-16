"""Planner tests for the L -> L relay cut point (CORE-RELAY-01)."""

from src.relay_planner import (
    RelayDownstreamProfile,
    plan_relay_cut,
)


def _profile(**overrides) -> RelayDownstreamProfile:
    base = dict(
        node_id="relay-downstream",
        profile_available=True,
        ram_available_gb=8.0,
    )
    base.update(overrides)
    return RelayDownstreamProfile(**base)


def test_requested_cut_is_admitted_and_keeps_last_layer_on_a_single_side():
    decision = plan_relay_cut(24, n_embd=2048, requested_cut=4)

    assert decision.admitted is True
    assert decision.reason == "requested_cut_admitted"
    assert decision.cut_layer == 4
    assert decision.upstream_layers == 4
    assert decision.downstream_layers == 20
    assert decision.hidden_bytes_per_token == 8192


def test_requested_cut_bounds_never_hand_off_the_last_layer():
    # cut == total - 1 is the highest legal value: handed-off layer == total - 2.
    highest = plan_relay_cut(24, n_embd=2048, requested_cut=23)
    too_high = plan_relay_cut(24, n_embd=2048, requested_cut=24)
    too_low = plan_relay_cut(24, n_embd=2048, requested_cut=0)

    assert highest.admitted is True
    assert highest.cut_layer == 23
    assert highest.upstream_layers - 1 == 22
    assert too_high.admitted is False
    assert too_low.admitted is False
    assert too_high.reason == too_low.reason == "cut_layer_out_of_range"


def test_short_models_and_unsupported_hidden_formats_are_rejected():
    short = plan_relay_cut(1, n_embd=2048, requested_cut=1)
    fmt = plan_relay_cut(24, n_embd=2048, requested_cut=4, hidden_dtype="int8")

    assert short.admitted is False
    assert short.reason == "model_has_no_relay_range"
    assert fmt.admitted is False
    assert fmt.reason == "unsupported_hidden_format"


def test_auto_split_requires_downstream_telemetry():
    missing = plan_relay_cut(24, n_embd=2048)  # no profile at all

    assert missing.admitted is False
    assert missing.reason == "downstream_profile_missing"
    assert missing.cut_layer == 0
    assert missing.upstream_layers == 0


def test_auto_split_is_conservative_and_leaves_both_sides_with_layers():
    decision = plan_relay_cut(24, n_embd=2048, downstream_profile=_profile(ram_available_gb=2.0))

    assert decision.admitted is True
    assert decision.reason == "auto_capacity_split"
    assert 1 <= decision.cut_layer <= 23
    assert decision.upstream_layers >= 1
    assert decision.downstream_layers >= 1


def test_auto_split_accepts_device_info_mapping():
    decision = plan_relay_cut(
        24,
        n_embd=2048,
        downstream_profile={"ram": {"total_gb": 16.0, "available_gb": 6.0}},
    )

    assert decision.admitted is True
    assert decision.profile.profile_available is True
    assert decision.profile.ram_available_gb == 6.0
    assert decision.to_dict()["profile"]["node_id"] == "relay-downstream"


def test_profile_from_device_info_marks_missing_telemetry():
    profile = RelayDownstreamProfile.from_device_info({})

    assert profile.profile_available is False
    assert profile.ram_available_gb == 0.0
    assert profile.rtt_ms == 0.0
