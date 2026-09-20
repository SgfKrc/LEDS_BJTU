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


# ---------------------------------------------------------------- 收口：证据范围
# 2026-09-19 收口：把「还没验证」与「已验证但性能不具优势（终态）」区分开。
# **两者 admitted 都是 False** —— 准入判据是性能，不是正确性。

def test_verified_evidence_narrows_reason_but_never_admits():
    from src.relay_contract import RelayXFrameEvidence

    evidence = RelayXFrameEvidence(
        correctness_verified=True,
        correctness_cases=8,
        max_tested_prefill=547,
        prompt_distribution_verified=True,
        long_sequence_verified=True,
        weak_network_verified=True,
        protocol_consistency_verified=True,
        performance_verdict="not_advantageous",
    )
    decision = admit_relay_xframe(_request(), evidence)

    assert decision.admitted is False, "正确性已验证也不得准入：判据是性能"
    assert decision.reason == "xframe_correctness_verified_performance_not_advantageous"
    assert decision.fallback.engaged is True
    # 证据对象本身也不允许被当成准入许可
    assert evidence.admits_production() is False
    assert evidence.to_dict()["admits_production"] is False
    assert evidence.to_dict()["correctness_cases"] == 8


def test_evidence_without_correctness_keeps_the_pending_reason():
    """只填了部分字段（正确性未验证）⇒ 仍按「待验证」处理，不得提前精确化。"""
    from src.relay_contract import RelayXFrameEvidence

    pending = RelayXFrameEvidence(protocol_consistency_verified=True, performance_verdict="unknown")
    assert admit_relay_xframe(_request(), pending).reason == "xframe_evidence_required"


def test_illegal_request_is_rejected_before_evidence_is_considered():
    """非法请求（跨引擎等）优先于证据判定 —— 有证据也不能绕过硬门。"""
    from src.relay_contract import RelayXFrameEvidence

    evidence = RelayXFrameEvidence(correctness_verified=True, performance_verdict="not_advantageous")
    assert admit_relay_xframe(
        _request(downstream_engine="pytorch"), evidence
    ).reason == "cross_engine_not_admitted"
