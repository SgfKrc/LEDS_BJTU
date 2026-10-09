"""XFRAME-1 补充：层段等价性工具的纯函数回归（不加载模型）。

判据是「主仓 `load_layer_range` + `forward_layers` 与整模是否逐元素一致」，
其中 `_metrics` 的比较口径必须与 XFRAME-1 的 `comparison_stats` 同源：
`rel_err = ||r-a|| / ||r||`、`cos`、`max_abs`，并给出 `identical`。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_layer_range_equivalence import _metrics  # noqa: E402


def test_identical_arrays_report_zero_error():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    st = _metrics(a, a.copy())
    assert st["identical"] is True
    assert st["rel_err"] == 0.0
    assert st["max_abs"] == 0.0
    assert abs(st["cos"] - 1.0) < 1e-12


def test_perturbation_rel_err_matches_norm_ratio():
    ref = np.array([[3.0, 4.0]], dtype=np.float32)          # ||ref|| = 5
    got = np.array([[3.0, 4.005]], dtype=np.float32)        # 差 ≈ 5e-3（含 float32 舍入）
    st = _metrics(ref, got)
    assert st["identical"] is False
    assert abs(st["rel_err"] - 5e-3 / 5.0) < 1e-6
    assert abs(st["max_abs"] - 5e-3) < 1e-6
    assert st["cos"] < 1.0


def test_shape_mismatch_is_rejected():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    try:
        _metrics(a, b)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch 必须显式抛错，不得静默对齐")
