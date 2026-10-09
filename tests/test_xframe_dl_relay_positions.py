"""XFRAME-6 复验工具（真 D→L 链路）的纯函数回归（不加载模型）。

判据本身复用 `xframe6_recursion_report.summarise`（已在 `test_xframe6_recursion_report.py`
钉住），这里只钉本工具新增的两个纯函数：分段趋势与首个分歧位置。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_dl_relay_positions import (  # noqa: E402
    first_divergence,
    segment_means,
)


def test_segment_means_splits_evenly_and_preserves_order():
    segs = segment_means([0.0, 1.0, 2.0, 3.0], 4)
    assert segs == [0.0, 1.0, 2.0, 3.0]


def test_segment_means_handles_ragged_lengths():
    # 10 个元素分 4 段：边界按 (i*n)//segments ⇒ 2/3/2/3 ⇒ 均值 0.5/3.0/5.5/8.0
    # （钉住切分规则本身：段长不等，段均值之平均**不**等于整体均值 4.5）
    assert segment_means([float(i) for i in range(10)], 4) == [0.5, 3.0, 5.5, 8.0]


def test_segment_means_empty_input_is_empty():
    assert segment_means([], 4) == []
    assert segment_means([1.0], 0) == []


def test_segment_means_detects_growth_trend():
    """线性增长序列 ⇒ 段均值单调递增（本工具用它替代不稳定的单点 first/last）。"""
    segs = segment_means([float(i) for i in range(64)], 4)
    assert segs == sorted(segs)
    assert segs[-1] > segs[0] * 3


def test_first_divergence_reports_first_false_index():
    assert first_divergence([True, True, False, False]) == 2
    assert first_divergence([False, True]) == 0


def test_first_divergence_none_when_all_same():
    assert first_divergence([True, True, True]) is None
    assert first_divergence([]) is None
