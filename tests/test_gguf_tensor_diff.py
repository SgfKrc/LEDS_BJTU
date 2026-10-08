"""`scripts/gguf_tensor_diff.py` 的纯函数回归（不读 GGUF 文件）。

它是"反量化实现是否语义等价"的直接判据：`rel` 的口径与按 qtype 的汇总必须被钉住 ——
否则会把"f16 舍入量级"误读成"实现不一致"（本项目就踩过一次这种误判）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gguf_tensor_diff import compare_arrays, summarise_by_qtype  # noqa: E402


def test_compare_arrays_identical():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    st = compare_arrays(a, a.copy())
    assert st["identical"] is True
    assert st["rel"] == 0.0
    assert st["max_abs"] == 0.0


def test_compare_arrays_rel_uses_reference_norm():
    ref = np.array([[3.0, 4.0]])            # ||ref|| = 5
    got = np.array([[3.0, 4.0 + 0.5]])
    st = compare_arrays(ref, got)
    assert abs(st["rel"] - 0.5 / 5.0) < 1e-12
    assert abs(st["max_abs"] - 0.5) < 1e-12


def test_compare_arrays_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(np.zeros((2, 3)), np.zeros((3, 2)))


def test_summarise_by_qtype_groups_and_maxes():
    rows = [
        {"qtype": "6", "rel": 1.0e-4},
        {"qtype": "6", "rel": 3.0e-4},
        {"qtype": "0", "rel": 0.0},
    ]
    st = summarise_by_qtype(rows)
    assert st["6"]["n"] == 2
    assert st["6"]["max_rel"] == 3.0e-4
    assert abs(st["6"]["mean_rel"] - 2.0e-4) < 1e-12
    assert st["0"]["max_rel"] == 0.0


def test_summarise_by_qtype_empty():
    assert summarise_by_qtype([]) == {}
