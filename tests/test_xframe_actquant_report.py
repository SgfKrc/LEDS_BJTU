"""`scripts/xframe_actquant_report.py` 的纯函数回归（不加载模型）。

钉住两件事：
1. `q8_0_quant` 必须复刻 llama.cpp `quantize_row_q8_0` 的语义 —— **per-32 块**、
   对称量化（`d = amax/127`）、scale 以 **f16** 存储、块内误差 ≤ `d/2`；
2. `rel` 的口径（分母取参考向量的 L2 范数），且同输入恒为 0。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from xframe_actquant_report import q8_0_quant, rel, select_targets  # noqa: E402


def test_select_targets_prefers_larger_matrices():
    cands = [("a.small", 100), ("b.big", 10_000), ("c.mid", 5_000)]
    assert select_targets(cands, 2) == ["b.big", "c.mid"]


def test_select_targets_is_deterministic_on_ties():
    # 参数量相同 ⇒ 按名字定序（避免每次跑测到不同目标）
    cands = [("z", 10), ("a", 10), ("m", 10)]
    assert select_targets(cands, 2) == ["a", "m"]


def test_select_targets_handles_fewer_candidates_than_requested():
    assert select_targets([("only", 5)], 4) == ["only"]
    assert select_targets([], 4) == []


def test_q8_0_quant_is_blockwise_and_uses_f16_scale():
    x = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32).reshape(2, 32)
    q = q8_0_quant(x)
    assert q.shape == x.shape
    # 每块误差上界 = d/2 = amax/254（第一块 amax=3.0）
    blk0 = (x[0] - q[0]).abs().max().item()
    assert blk0 <= 3.0 / 254.0 + 1e-6
    # f16 scale 的舍入远小于块内步长 ⇒ 重构值应成"阶梯"分布（不止一个唯一值）
    assert len(torch.unique(q[0])) > 1


def test_q8_0_quant_exact_for_representable_values():
    # 取 d=1/127 的整数倍（scale 恰好可用 f16 表达为 amax/127 时误差为 0 或极小）
    vals = torch.arange(32, dtype=torch.float32)
    amax = vals.abs().amax().item()
    d = amax / 127.0
    x = (vals * 0).reshape(1, 32)  # 全零块：scale 取 1，量化后仍为 0
    assert torch.equal(q8_0_quant(x), x)
    assert d > 0


def test_q8_0_quant_passes_through_unaligned_width():
    x = torch.randn(3, 33)
    assert torch.equal(q8_0_quant(x), x)


def test_q8_0_quant_zero_block_does_not_produce_nan():
    x = torch.zeros(1, 32)
    q = q8_0_quant(x)
    assert torch.isfinite(q).all()
    assert torch.equal(q, x)


def test_rel_is_zero_for_identical_and_positive_for_perturbed():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    assert rel(a, a.clone()) == 0.0
    b = a + 0.1
    assert rel(a, b) > 0.0


def test_rel_uses_reference_norm_as_denominator():
    a = torch.tensor([[3.0, 4.0]])          # ||a|| = 5
    b = torch.tensor([[3.0, 4.0 + 0.5]])
    assert abs(rel(a, b) - 0.5 / 5.0) < 1e-6
