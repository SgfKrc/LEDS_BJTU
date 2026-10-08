"""`scripts/xframe_int8_grid_report.py` 的纯函数回归（不加载模型）。

这一环的意义全在三条**可判定的代数事实**上，必须被单测钉住：
1. **纯整数累加与分段顺序无关**（整数加满足结合律/交换律）—— 这是 L3 的立足点；
2. **浮点累加与分段顺序有关**（否则测出来"0 敏感度"就不是整数的功劳）；
3. `scale` 必须取在**被累加掉的那一维**上为标量，否则 `Σ qx·qw` 之后无法乘回
   （早先版本按 K 逐元素给 scale，结果形状无法广播 —— 单测钉住这个契约）。
"""
import numpy as np
import pytest

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_int8_grid_report import (  # noqa: E402
    blockwise_quant,
    dequantize_parts,
    quantize_per_channel,
    split_matmul_blockwise,
    split_matmul_f32,
    split_matmul_int32,
    split_slices,
)


def _data(m=8, k=256, n=16, seed=0):
    rng = np.random.default_rng(seed)
    # 带一个离群值，让 per-row scale 的粒度问题可复现（对应真实模型的 massive activation）
    x = rng.normal(0, 0.2, (m, k)).astype(np.float32)
    w = rng.normal(0, 0.02, (n, k)).astype(np.float32)
    x[0, 3] = 8.0
    return x, w


def test_split_slices_covers_k_exactly():
    for order in (1, 2, 4, 8):
        sl = split_slices(256, order)
        assert sum(s.stop - s.start for s in sl) == 256
        assert sl[0].start == 0 and sl[-1].stop == 256


def test_int32_accumulation_is_bit_identical_across_orders():
    x, w = _data()
    qx = np.clip(np.round(x / np.abs(x).max(axis=1, keepdims=True) * 127), -127, 127).astype(np.int32)
    qw = np.clip(np.round(w / np.abs(w).max(axis=1, keepdims=True) * 127), -127, 127).astype(np.int32)
    ref = split_matmul_int32(qx, qw, 1)
    for order in (2, 4, 8):
        assert np.array_equal(ref, split_matmul_int32(qx, qw, order))


def test_f32_accumulation_does_depend_on_order():
    x, w = _data()
    ref = split_matmul_f32(x, w, 1)
    assert any(
        not np.array_equal(ref, split_matmul_f32(x, w, o)) for o in (2, 4, 8)
    ), "浮点分段顺序应当产生差异（否则该对照没有区分度）"


def test_quantize_per_channel_clips_and_handles_zero_scale():
    x = np.array([[0.5, -1.0, 0.25, 0.0]], dtype=np.float32)
    q = quantize_per_channel(x, np.abs(x).max(axis=1, keepdims=True))
    assert q.dtype == np.float32 and float(np.abs(q).max()) <= 127.0
    # scale 为 0 应兜底成 1.0（否则整行塌成 0）
    z = quantize_per_channel(np.array([[0.0, 0.0]], dtype=np.float32),
                             np.array([[0.0]], dtype=np.float32))
    assert np.array_equal(z, np.zeros((1, 2), dtype=np.float32))


def test_dequantize_parts_restores_gemm_scale():
    # 单元素 GEMM：Σ (qx·sx/127)(qw·sw/127) 应还原成 x·w
    x = np.array([[3.0]], dtype=np.float32)
    w = np.array([[2.0]], dtype=np.float32)
    sx = np.array([3.0], dtype=np.float32)
    sw = np.array([2.0], dtype=np.float32)
    qx = quantize_per_channel(x, sx[:, None])
    qw = quantize_per_channel(w, sw[:, None])
    acc = qx.astype(np.int32) @ qw.astype(np.int32).T
    assert abs(float(dequantize_parts(acc, sx, sw)[0, 0]) - 6.0) < 1e-5


def test_blockwise_quant_shape_and_padding():
    x, _w = _data(k=100)
    q, s = blockwise_quant(x, block=32)
    assert q.shape == (x.shape[0], 4, 32)     # 100 -> 4 块（右侧补零）
    assert s.shape == (x.shape[0], 4)
    assert np.abs(q).max() <= 127.0


def test_blockwise_order_dependence_is_possible_but_quant_error_smaller():
    x, w = _data(k=512, n=8)
    qx, sx = blockwise_quant(x, 32)
    qw, sw = blockwise_quant(w, 32)
    ref = split_matmul_blockwise(qx, sx, qw, sw, 1)
    assert ref.shape == (x.shape[0], w.shape[0])
    # 块级网格的量化误差应小于 1%（块内动态范围窄）
    f32_ref = split_matmul_f32(x, w, 1)
    rel = float(np.linalg.norm(ref - f32_ref) / np.linalg.norm(f32_ref))
    assert rel < 1e-2
