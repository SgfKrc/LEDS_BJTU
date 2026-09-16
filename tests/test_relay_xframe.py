"""Fail-closed policy tests for CORE-RELAY-XFRAME-01."""

from src.relay_contract import (
    RelayXFrameRequest,
    admit_relay_xframe,
)


def _request(**overrides):
    values = {
        "upstream_engine": "llama.cpp",
        "downstream_engine": "llama.cpp",
        "network_mode": "loopback",
        "sequence_length": 16,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    values.update(overrides)
    return RelayXFrameRequest(**values)


def test_xframe_never_turns_into_production_admission_without_evidence():
    decision = admit_relay_xframe(_request())

    assert decision.admitted is False
    assert decision.reason == "xframe_evidence_required"
    assert decision.fallback.engaged is True


def test_cross_engine_and_public_network_are_rejected_explicitly():
    assert admit_relay_xframe(_request(downstream_engine="pytorch")).reason == "cross_engine_not_admitted"
    assert admit_relay_xframe(_request(network_mode="tailnet_direct")).reason == "network_scope_not_admitted"


def test_sampling_and_invalid_length_are_separate_gates():
    assert admit_relay_xframe(_request(temperature=0.7)).reason == "sampling_matrix_not_admitted"
    assert admit_relay_xframe(_request(sequence_length=0)).reason == "sequence_length_invalid"
    assert admit_relay_xframe(_request(sequence_length="bad")).reason == "sequence_length_invalid"
    assert admit_relay_xframe(_request(top_p=float("nan"))).reason == "sampling_matrix_not_admitted"
