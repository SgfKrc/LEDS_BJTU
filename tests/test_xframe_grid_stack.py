"""`scripts/xframe_grid_stack.py` 的纯函数回归（不加载模型）。

钉住三件回答"优化能不能叠加"所依赖的代数事实：
1. **整数累加与分块顺序无关**（`int8_gemm` 的立足点）—— 用"同一输入两次调用逐位相同" +
   "分段累加与整体累加逐位相同"两条断言；
2. `quantize_shared_per_channel` 必须用**调用方给定**的 scale（共享网格），且零 scale 兜底；
3. `int8_gemm` 的误差有界（int8 档 ~1e-2 量级，不是 f32 的 1e-7）—— 这正是"档位不足时
   上游量化会破坏交界处对齐"的原因。
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_grid_stack import (  # noqa: E402
    group_name,
    int8_gemm,
    quantize_shared_per_channel,
)


def _data(m=6, k=256, n=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * 0.2
    x[0, 5] = 9.0                     # 注入一个 massive activation，复现真实分布
    w = torch.randn(n, k, generator=g) * 0.02
    return x, w


def test_quantize_shared_per_channel_uses_given_scale_only():
    x = np.array([[0.5, -1.0, 0.0]], dtype=np.float32)
    scale = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    q = quantize_shared_per_channel(x, scale, 127.0)
    # 0.5/1.0*127 = 63.5 -> 64（round-half-even 由 np.round 决定）⇒ 回乘后 64/127
    assert abs(float(q[0, 0]) - 64.0 / 127.0) < 1e-6
    assert float(q[0, 2]) == 0.0


def test_quantize_shared_per_channel_zero_scale_falls_back_to_one():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    q = quantize_shared_per_channel(x, np.array([0.0, 0.0], dtype=np.float32), 127.0)
    assert np.all(np.isfinite(q))


def test_int8_gemm_shape_and_bounded_error():
    x, w = _data()
    ref = x @ w.T
    got = int8_gemm(x, w)
    assert got.shape == ref.shape
    rel = float((got - ref).norm() / ref.norm())
    assert 0.0 < rel < 1.5e-1, f"int8 档误差应在 1e-2 量级，实得 {rel:.3e}"


def test_int8_gemm_is_deterministic_bitwise():
    x, w = _data()
    a = int8_gemm(x, w)
    b = int8_gemm(x, w)
    assert torch.equal(a, b), "整数累加必须逐位确定（与分块/线程划分无关）"


def test_int8_accumulation_is_order_independent_bitwise():
    """**先量化整体、再在整数域分段累加** ⇒ 必须与整体一次算**逐位相同**（整数加结合律）。

    注意：不能在量化**前**分段 —— `int8_gemm` 的 per-token scale 是按整行 `amax` 取的，
    分段会改变 scale，那是"量化网格变了"而不是"累加顺序变了"（两者必须分开测）。
    """
    x, w = _data()
    flat = x.float()
    xs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    ws = w.abs().amax(dim=1).clamp(min=1e-30)
    qx = torch.round(flat / xs * 127.0).clamp(-127, 127).to(torch.int32)
    qw = torch.round(w / ws[:, None] * 127.0).clamp(-127, 127).to(torch.int32)
    whole = qx @ qw.T
    k = x.shape[1]
    half = k // 2
    split = qx[:, :half] @ qw[:, :half].T + qx[:, half:] @ qw[:, half:].T
    assert torch.equal(whole, split), "整数累加必须与分段顺序无关（这是 L3 的全部立足点）"


def test_int8_gemm_handles_single_token_row():
    x, w = _data(m=1)
    assert int8_gemm(x, w).shape == (1, w.shape[0])


def test_group_name_covers_all_groups():
    assert group_name("A") == "整模基线"
    assert group_name("C").startswith("跨引擎")
    assert group_name("E") != group_name("F")
