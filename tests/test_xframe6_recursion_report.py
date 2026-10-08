"""XFRAME-6 可复跑工具的纯函数回归（不加载模型）。

判定核心是「逐位置 rel 对位置的线性回归斜率」—— 它是"递推是否引入额外不可消除项"
的判据，所以必须被单测钉住：常数序列斜率=0、线性序列斜率精确、噪声不产生假斜率。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe6_recursion_report import (  # noqa: E402
    rel_series,
    slope_per_position,
    summarise,
)


def test_slope_is_zero_for_constant_series():
    assert slope_per_position([1e-6] * 32) == 0.0


def test_slope_recovers_known_linear_trend():
    # rel = 1e-7 * (pos + 1) ⇒ 斜率恰为 1e-7
    rel = [1e-7 * (i + 1) for i in range(16)]
    assert abs(slope_per_position(rel) - 1e-7) < 1e-12


def test_slope_of_noise_is_far_below_threshold():
    # 交替噪声：不构成趋势 ⇒ 判据必须判成"无额外累积"
    rel = [5e-7 if i % 2 else 1e-6 for i in range(64)]
    assert abs(slope_per_position(rel)) < 1e-8


def test_rel_series_uses_top1_magnitude():
    """标量序列的相对差：分母取 |ref|，相同则为 0（长上下文扫描依赖此语义）。"""
    rel = rel_series([2.0, -4.0], [2.0 + 1e-6, -4.0])
    assert len(rel) == 2
    assert rel[1] == 0.0
    assert rel[0] > 0.0 and rel[0] < 1e-5


def test_rel_series_zips_shorter_input():
    """长度不等时按 zip 截断（不抛异常）—— 长序列扫描里两侧长度应一致，但接口要稳。"""
    assert rel_series([1.0], [1.0, 2.0]) == [0.0]


def test_summarise_flags_extra_accumulation():
    flat = summarise([1e-6] * 80, [True] * 80)
    assert flat["argmax_all_same"] is True
    assert flat["no_extra_accumulation"] is True          # 常数 ⇒ 首尾比 1.0
    assert flat["head_mean_rel"] == flat["tail_mean_rel"]

    growing = summarise([1e-7 * (i + 1) for i in range(80)], [True] * 80)
    assert growing["tail_over_head"] > 2.0
    assert growing["no_extra_accumulation"] is False       # 线性增长 ⇒ 首尾比 ≈2.6


def test_summarise_refuses_verdict_on_short_series():
    # 样本不足（<48 位置）⇒ 不下结论，避免用噪声斜率误判
    short = summarise([1e-7 * (i + 1) for i in range(35)], [True] * 35)
    assert short["no_extra_accumulation"] is None
    assert short["positions"] == 35
    # 48 位置起给结论（60 位置的真实 prompt 因此可判）
    borderline = summarise([1e-6] * 48, [True] * 48)
    assert borderline["no_extra_accumulation"] is True
