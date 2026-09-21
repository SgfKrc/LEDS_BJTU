"""P2 切点求解器的守卫用例（容量 / 延迟 / 风险 三输出 + 合法切点约束）。

全部为纯计算用例：不需要模型、GPU 或网络，任何环境都可跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_cut_objective import (  # noqa: E402
    SegmentProfile,
    fit_segment_profile,
    fit_two_segment,
    legal_cuts,
    plan_relay_cut_n_segments,
)

MB = 1024 ** 2


def _layers(total: int = 24, per_layer_bytes: int = 20 * MB, tail_extra: int = 8 * MB):
    return [per_layer_bytes] * total, tail_extra


def _sym_segments(count: int = 2, *, ms_per_layer: float = 4.0, capacity_gb: float = 8.0,
                  bandwidth_mbps: float = 1000.0, rtt_ms: float = 0.5):
    return [SegmentProfile(node_id=f"seg{i}", engine="llama.cpp",
                           capacity_bytes=int(capacity_gb * 1024 ** 3),
                           ms_per_layer_decode=ms_per_layer, ms_per_layer_prefill=0.0,
                           ms_fixed_decode=0.0, bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms)
            for i in range(count)]


# --------------------------------------------------------------------- 合法切点
def test_legal_cuts_respects_multiple_and_minimum_layers():
    assert legal_cuts(24, cut_multiple=1, min_layers_per_segment=1) == tuple(range(1, 24))
    assert legal_cuts(24, cut_multiple=4, min_layers_per_segment=4) == (4, 8, 12, 16, 20)
    assert legal_cuts(24, cut_multiple=4, min_layers_per_segment=8) == (8, 12, 16)
    assert legal_cuts(3, cut_multiple=4) == ()


def test_qwen35_style_multiple_is_enforced_in_the_plan():
    """Qwen3.5 的硬约束：切点必须是 full_attention_interval(=4) 的整倍数。"""
    layer_bytes, extra = _layers()
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=_sym_segments(), non_split_bytes=extra,
                                     cut_multiple=4)
    assert plan.admitted
    assert all(cut % 4 == 0 for cut in plan.cuts), f"切点必须落在 4 的倍数：{plan.cuts}"


# --------------------------------------------------------------------- 目标函数
def test_symmetric_devices_pick_the_middle_cut():
    layer_bytes, extra = _layers()
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=_sym_segments(), non_split_bytes=extra)
    assert plan.admitted
    assert plan.cuts == (12,), f"对称设备的最优切点应在 N/2，实得 {plan.cuts}"
    assert plan.capacity_feasible is True
    assert plan.capacity_gain_x and plan.capacity_gain_x > 1.0


def test_fast_segment_takes_more_layers():
    """纯延迟目标下，更快的段应拿更多层（容量项会被显式关掉）。"""
    layer_bytes, extra = _layers()
    latency_only = {"capacity": 0.0, "latency": 1.0, "risk": 0.0}
    slow_up = [SegmentProfile(node_id="up", ms_per_layer_decode=10.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0),
               SegmentProfile(node_id="down", ms_per_layer_decode=2.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0)]
    fast_up = [SegmentProfile(node_id="up", ms_per_layer_decode=2.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0),
               SegmentProfile(node_id="down", ms_per_layer_decode=10.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0)]
    slow_plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes,
                                          n_embd=896, segments=slow_up, non_split_bytes=extra,
                                          weights=latency_only)
    fast_plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes,
                                          n_embd=896, segments=fast_up, non_split_bytes=extra,
                                          weights=latency_only)
    assert slow_plan.cuts[0] < fast_plan.cuts[0], (
        f"更快的上游应拿更多层：slow={slow_plan.cuts} fast={fast_plan.cuts}")


def test_default_weights_are_capacity_first_then_latency():
    """默认权重下容量项占主导：对称设备仍取中间切点，即使延迟项偏好别的切点。"""
    layer_bytes, extra = _layers()
    fast_up = [SegmentProfile(node_id="up", ms_per_layer_decode=2.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0),
               SegmentProfile(node_id="down", ms_per_layer_decode=10.0,
                              capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0)]
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=fast_up, non_split_bytes=extra)
    assert plan.cuts == (12,), "默认权重（capacity=latency=risk=1）应按容量收益取中间切点"
    latency_only = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes,
                                             n_embd=896, segments=fast_up,
                                             non_split_bytes=extra, cut_multiple=4,
                                             weights={"capacity": 0.0, "latency": 1.0,
                                                      "risk": 0.0})
    assert latency_only.cuts == (20,), "纯延迟目标应把层尽量给更快的上游（4 的倍数约束下为 20）"


def test_capacity_infeasible_is_reported_not_hidden():
    layer_bytes, extra = _layers(per_layer_bytes=100 * MB)
    tiny = SegmentProfile(node_id="tiny", capacity_bytes=200 * MB,
                          ms_per_layer_decode=1.0, bandwidth_mbps=1000.0)
    other = SegmentProfile(node_id="other", capacity_bytes=200 * MB,
                           ms_per_layer_decode=1.0, bandwidth_mbps=1000.0)
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=[tiny, other], non_split_bytes=extra)
    assert plan.admitted is False
    assert plan.capacity_feasible is False
    assert plan.reason == "all_placements_infeasible"
    assert plan.candidates, "被拒的方案也要留下候选，便于复算"


def test_n_segment_plan_keeps_contiguous_full_coverage():
    layer_bytes, extra = _layers()
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=_sym_segments(3), non_split_bytes=extra,
                                     cut_multiple=4)
    assert plan.admitted
    assert len(plan.segment_layers) == 3
    assert sum(plan.segment_layers) == 24
    assert list(plan.cuts) == sorted(plan.cuts)
    # 三段：只有首段带 embedding、末段带 lm_head，且中段不吃 non_split_bytes
    expected_bytes = [sum(layer_bytes[:plan.cuts[0]]) + extra,
                      sum(layer_bytes[plan.cuts[0]:plan.cuts[1]]),
                      sum(layer_bytes[plan.cuts[1]:]) + extra]
    assert list(plan.segment_bytes) == expected_bytes


# --------------------------------------------------------------------- 风险
def test_risk_items_reach_the_objective_and_the_report():
    layer_bytes, extra = _layers()
    segments = _sym_segments(2, bandwidth_mbps=1.0, rtt_ms=200.0)
    segments[0] = SegmentProfile(**{**segments[0].to_dict(), "thermal_throttled": True})
    segments[1] = SegmentProfile(**{**segments[1].to_dict(), "artifacts_ready": False})
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=segments, non_split_bytes=extra)
    risk = plan.risk_penalty
    assert risk["total"] > 0
    kinds = {k for item in risk["items"] for k in item}
    assert {"bandwidth", "rtt", "thermal", "artifacts"} <= kinds
    assert any(item["thermal"] > 0 for item in risk["items"])
    assert any(item["artifacts"] > 0 for item in risk["items"])
    # 风险必须真的进入分数：风险高时分数低于无风险同构场景
    clean = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                      segments=_sym_segments(), non_split_bytes=extra)
    assert plan.score < clean.score


def test_zero_bandwidth_marks_latency_unavailable_and_scores_worst():
    layer_bytes, extra = _layers()
    segments = _sym_segments(2, bandwidth_mbps=0.0)
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=segments, non_split_bytes=extra)
    assert plan.risk_penalty["total"] == 1.0, "带宽为 0 时风险应顶格"
    assert plan.latency_estimate["transfer_decode_ms"] is None or plan.admitted is False


# --------------------------------------------------------------------- 拒绝路径
@pytest.mark.parametrize(("kwargs", "reason"), [
    ({"segments": []}, "no_usable_segments"),
    ({"segments": [SegmentProfile(node_id="only")]}, "segment_count_mismatch"),
    ({"cut_multiple": 100}, "no_legal_cut_point"),
    ({"total_layers": 24, "layer_bytes": [1] * 5}, "layer_bytes_length_mismatch"),
])
def test_rejection_paths_are_named(kwargs, reason):
    layer_bytes, extra = _layers()
    base = {"total_layers": 24, "layer_bytes": layer_bytes, "n_embd": 896,
            "segments": _sym_segments(), "non_split_bytes": extra}
    base.update(kwargs)
    plan = plan_relay_cut_n_segments(**base)
    assert plan.admitted is False
    assert plan.reason == reason


def test_unsupported_hidden_dtype_is_rejected():
    layer_bytes, extra = _layers()
    plan = plan_relay_cut_n_segments(total_layers=24, layer_bytes=layer_bytes, n_embd=896,
                                     segments=_sym_segments(), non_split_bytes=extra,
                                     hidden_dtype="bfloat16")
    assert plan.admitted is False
    assert plan.reason == "unsupported_hidden_format"


# --------------------------------------------------------------------- 实测拟合
def test_fit_segment_profile_recovers_known_coefficients():
    # 真值：固定 5 ms + 每层 3 ms
    measurements = [{"k": k, "ms": 5.0 + 3.0 * k} for k in (4, 8, 12, 16, 20)]
    profile, fit = fit_segment_profile(measurements, node_id="down", engine="llama.cpp",
                                       layers_key="k", ms_key="ms")
    assert fit["samples"] == 5
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert profile.ms_per_layer_decode == pytest.approx(3.0, abs=1e-6)
    assert profile.ms_fixed_decode == pytest.approx(5.0, abs=1e-6)


def test_fit_two_segment_splits_upstream_and_downstream():
    total = 24
    measurements = []
    for k in (4, 8, 12, 16, 20):
        measurements.append({
            "upstream_layers": k,
            "upstream_decode_ms": 1.0 + 2.0 * k,                 # 上游：1 + 2K
            "downstream_decode_ms": 4.0 + 0.5 * (total - k),     # 下游：4 + 0.5(N-K)
        })
    fitted = fit_two_segment(measurements, total_layers=total)
    assert fitted["fit"]["upstream"]["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fitted["upstream"].ms_per_layer_decode == pytest.approx(2.0, abs=1e-6)
    assert fitted["upstream"].ms_fixed_decode == pytest.approx(1.0, abs=1e-6)
    assert fitted["downstream"].ms_per_layer_decode == pytest.approx(0.5, abs=1e-6)
    assert fitted["downstream"].ms_fixed_decode == pytest.approx(4.0, abs=1e-6)


def test_fit_then_plan_predicts_the_measured_optimum():
    """端到端闭环：从实测点拟合 → 求解器预测切点 = 解析最优（更慢的段拿更少层）。

    两个前置都显式给出，避免把「未知」当成「可行」：容量来自设备画像（拟合只给耗时），
    权重取纯延迟（容量项在这个断言里无关）。
    """
    total = 24
    measurements = []
    for k in (4, 8, 12, 16, 20):
        measurements.append({
            "upstream_layers": k,
            "upstream_decode_ms": 1.0 + 2.0 * k,                 # 上游：1 + 2K（更慢）
            "downstream_decode_ms": 4.0 + 0.5 * (total - k),     # 下游：4 + 0.5(N-K)
        })
    fitted = fit_two_segment(
        measurements, total_layers=total,
        upstream_profile=SegmentProfile(node_id="up", engine="pytorch",
                                        capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0),
        downstream_profile=SegmentProfile(node_id="down", engine="llama.cpp",
                                          capacity_bytes=8 * 1024 ** 3, bandwidth_mbps=1000.0))
    assert fitted["upstream"].capacity_bytes > 0
    layer_bytes = [20 * MB] * total
    plan = plan_relay_cut_n_segments(
        total_layers=total, layer_bytes=layer_bytes, n_embd=896,
        segments=[fitted["upstream"], fitted["downstream"]],
        non_split_bytes=8 * MB, cut_multiple=4,
        weights={"capacity": 0.0, "latency": 1.0, "risk": 0.0})
    assert plan.admitted
    # 上游 2 ms/层 vs 下游 0.5 ms/层 ⇒ 下游应拿更多层（切点取最小合法值 4）
    assert plan.cuts == (4,), f"按拟合系数应把更多层给更快的下游，实得 {plan.cuts}"
