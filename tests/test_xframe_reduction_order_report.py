"""XFRAME-4/5 可复跑工具的纯函数回归（不加载共享库）。

判据是"归约顺序差异的**阶**"与"换 f64 累加后是否**归零**"，所以这里只钉住统计/汇总
逻辑：相对差的分母取参照的 `max|·|`；汇总必须识别"随 K 增长"与"f64 恒为 0"两件事。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_reduction_order_report import comparison_stats, summarise  # noqa: E402


def test_comparison_stats_uses_reference_magnitude():
    # 用 float32 **可精确表示**的差值（2.0 处 ULP = 2**-22），避免测试本身引入舍入
    delta = np.float32(2.0 ** -20)
    ref = np.array([[2.0, -4.0]], dtype=np.float32)
    got = np.array([[np.float32(2.0) + delta, -4.0]], dtype=np.float32)
    stats = comparison_stats(ref, got)
    assert abs(stats["max_abs"] - float(delta)) < 1e-15
    # 分母 = max|ref| = 4.0
    assert abs(stats["rel"] - float(delta) / 4.0) < 1e-15


def test_comparison_stats_zero_when_identical():
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    stats = comparison_stats(a, a.copy())
    assert stats["max_abs"] == 0.0 and stats["rel"] == 0.0


def _report(rows):
    return {"orders": [0, 8], "ks": [128, 2048], "rows": rows}


def test_summarise_detects_growth_and_f64_zero():
    rows = [
        {"K": 128, "order": 0, "f32_vs_order0": {"rel": 0.0}, "acc64_vs_order0": {"rel": 0.0}},
        {"K": 128, "order": 8, "f32_vs_order0": {"rel": 1e-7}, "acc64_vs_order0": {"rel": 0.0}},
        {"K": 2048, "order": 0, "f32_vs_order0": {"rel": 0.0}, "acc64_vs_order0": {"rel": 0.0}},
        {"K": 2048, "order": 8, "f32_vs_order0": {"rel": 1e-6}, "acc64_vs_order0": {"rel": 0.0}},
    ]
    s = summarise(_report(rows))
    assert s["f32_grows_with_K"] is True
    assert s["acc64_all_zero"] is True
    assert s["f32_worst_rel_by_K"][2048] > s["f32_worst_rel_by_K"][128]


def test_summarise_flags_nonzero_acc64_as_not_bypassed():
    """若 f64 累加下差异非 0，则**不得**声称"墙可绕过"（防谎报）。"""
    rows = [
        {"K": 128, "order": 0, "f32_vs_order0": {"rel": 0.0}, "acc64_vs_order0": {"rel": 0.0}},
        {"K": 128, "order": 8, "f32_vs_order0": {"rel": 1e-7}, "acc64_vs_order0": {"rel": 3e-9}},
    ]
    s = summarise(_report(rows))
    assert s["acc64_all_zero"] is False


def test_summarise_ignores_order_zero_baseline():
    """`order=0` 是基准自身，不得参与"最大 rel"统计（否则恒 0 会稀释结论）。"""
    rows = [
        {"K": 128, "order": 0, "f32_vs_order0": {"rel": 0.0}, "acc64_vs_order0": {"rel": 0.0}},
    ]
    s = summarise(_report(rows))
    assert s["f32_worst_rel_by_K"] == {}
    assert s["acc64_all_zero"] is True
