"""Fail-closed policy tests for CORE-RELAY-XFRAME-01.

★ 2026-09-21（用户裁定 `dec-6d91cd100fecc798`）：准入判据由「性能（=速度）不占优 ⇒
永不允许」改为「**正确性已验证即准入**」—— 速度与容量同属性能，速度不占优只影响
**默认路由倾向**，不构成否决。下列用例已按新口径更新。
"""

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
# 2026-09-19 收口：把「还没验证」与「已验证」区分开。
# ★ 2026-09-21：两者含义**相反** —— 已验证 ⇒ **准入**；未验证 ⇒ 待验证（fail-closed）。

def test_verified_evidence_admits_production_and_disengages_fallback():
    """正确性证据齐备 ⇒ **准入**，且 fallback **不启用**（handoff 才是预期路径）。

    ⚠️ 注意 `performance_verdict="not_advantageous"`（速度比整模 GPU 慢约 20×）
    **不影响准入** —— 速度与容量同属性能，不能只取速度作否决。
    """
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

    assert decision.admitted is True
    assert decision.reason == "admitted"
    assert decision.fallback.engaged is False, "准入时不应回退到单进程路径"
    # 证据对象自身也反映准入
    assert evidence.admits_production() is True
    assert evidence.to_dict()["admits_production"] is True
    assert evidence.to_dict()["correctness_cases"] == 8
    # 速度结论仍被**如实记录**，只是不参与准入
    assert evidence.to_dict()["performance_verdict"] == "not_advantageous"


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
