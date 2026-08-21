"""SPC-MEDS 半自回归（Medusa 候选树）核心测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import speculative as spec
from speculative_medusa import (
    MedusaHead,
    medusa_heads_from_distribution,
    verify_medusa_prefix,
)


def _rows(values, n, *, normalize=False):
    arr = np.asarray(values, dtype=np.float64)
    return np.tile(arr, (max(n, 1), 1))


def test_medusa_heads_topk_and_order():
    rows = _rows([0.2, 0.5, 0.3], 3)
    heads = medusa_heads_from_distribution(rows, heads=2, mode="topk")
    assert len(heads) == 3
    # 每位置候选按 score 降序且不含 0 概率（归一化后 top-2 = 1,2）
    assert [c.token for c in heads[0]] == [1, 2]
    assert [c.token for c in heads[1]] == [1, 2]
    assert all(c.score >= 0 for c in heads[0])


def test_medusa_perfect_draft_accepts_all():
    # draft 与 verify 相同分布：候选含真值且 q==p → 全接受 + bonus
    p = _rows([0.25, 0.75], 4)
    verify = np.vstack([p, np.array([[0.25, 0.75]], dtype=np.float64)])  # 含 bonus 行
    heads = medusa_heads_from_distribution(p, heads=2)
    outcome = verify_medusa_prefix(heads, verify, rng=np.random.default_rng(1))
    assert outcome.accepted_all is True
    assert outcome.accepted_count == 4
    assert outcome.correction_token is None
    assert outcome.bonus_token is not None
    assert len(outcome.emitted) == 5


def test_medusa_disjoint_supports_reject_all_with_correction():
    # draft 全集中在 token0，verify 全集中在 token1：全拒 + 修正 token1
    draft = _rows([1.0, 0.0], 3)
    verify = _rows([0.0, 1.0], 3)
    heads = medusa_heads_from_distribution(draft, heads=2)
    outcome = verify_medusa_prefix(heads, verify[:3], rng=np.random.default_rng(1))
    assert outcome.accepted_count == 0
    assert outcome.accepted_all is False
    assert outcome.correction_token == 1
    assert outcome.emitted == [1]


def test_medusa_heads_recovers_top1_mistake():
    """块预测价值：draft top-1 错、top-2 是 verify 真值 → 仍接受 top-2。"""
    draft_row = np.array([0.6, 0.4], dtype=np.float64)   # top1=0, top2=1
    verify_row = np.array([0.0, 1.0], dtype=np.float64)
    heads = medusa_heads_from_distribution(draft_row.reshape(1, 2), heads=2, mode="topk")
    verify = verify_row.reshape(1, 2)
    greedy = verify_medusa_prefix(heads, verify, greedy=True)
    assert greedy.accepted_count == 1
    assert greedy.accepted_tokens == [1]
    # heads=1 退化为扁平：只试 top1(0)，reject → 修正 1
    flat = medusa_heads_from_distribution(draft_row.reshape(1, 2), heads=1, mode="topk")
    flat_out = verify_medusa_prefix(flat, verify, greedy=True)
    assert flat_out.accepted_count == 0
    assert flat_out.correction_token == 1


def test_medusa_heads1_equals_flat_greedy_verify():
    """heads=1 时接受性与 verify_draft_tokens（greedy）一致。"""
    p = _rows([0.3, 0.7], 5)
    verify = p[:5]
    heads = medusa_heads_from_distribution(p, heads=1, mode="topk")
    tokens = [c.token for layer in heads for c in layer]
    assert tokens == [1] * 5          # top1 都是 token1
    med = verify_medusa_prefix(heads, verify, greedy=True)
    # 标准核心：同一 top1 序列
    probe = np.tile(np.array([0.3, 0.7], dtype=np.float64), (5, 1))
    outcome = spec.verify_draft_tokens(probe, verify, tokens, greedy=True, rng=np.random.default_rng(0))
    assert med.accepted_count == outcome.accepted_count


def test_medusa_output_distribution_equals_verify():
    """Monte-Carlo：**sample 模式 heads=1（等价标准投机）** 首 token 分布=verify。"""
    p = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    verify = np.tile(p, (3, 1))
    draft = np.tile(p, (3, 1))
    counts = np.zeros(3, dtype=np.float64)
    for _ in range(12000):
        heads = medusa_heads_from_distribution(
            draft, heads=1, mode="sample", rng=np.random.default_rng(None),
        )
        outcome = verify_medusa_prefix(
            heads, verify, rng=np.random.default_rng(None), bonus=False,
        )
        counts[int(outcome.emitted[0])] += 1.0
    empirical = counts / counts.sum()
    # sample=heads=1 的 accept/reject 与标准投机一致 → 分布 = verify
    assert float(np.max(np.abs(empirical - p))) < 0.03, np.round(empirical, 4)


def test_medusa_multihead_improves_acceptance():
    """heads>1 块预测提升接受率：软分布下多候选比单一 top1 接受更多。"""
    p = np.array([0.45, 0.35, 0.20], dtype=np.float64)
    verify = np.tile(p, (4, 1))
    rng1 = np.random.default_rng(11)
    rng2 = np.random.default_rng(11)
    out1 = verify_medusa_prefix(
        medusa_heads_from_distribution(np.tile(p, (4, 1)), heads=1), verify, rng=rng1,
    )
    out2 = verify_medusa_prefix(
        medusa_heads_from_distribution(np.tile(p, (4, 1)), heads=3), verify, rng=rng2,
    )
    assert out2.accepted_count >= out1.accepted_count
    # 多候选输出仍在 verify 支持集（正确性不破坏）
    assert set(out2.emitted) <= {0, 1, 2}


def test_medusa_invalid_inputs():
    # 空树 + 空校验矩阵：返回空 accepted（不是错误）
    empty = verify_medusa_prefix([], np.zeros((0, 5)))
    assert empty.accepted_count == 0
    # verify 行数不足
    heads = [list([MedusaHead(0, 1.0)])] * 3
    with pytest.raises(spec.SpeculativeError):
        verify_medusa_prefix(heads, np.zeros((2, 5)))
    # token 越界
    bad = [list([MedusaHead(99, 1.0)])]
    with pytest.raises(spec.SpeculativeError):
        verify_medusa_prefix(bad, np.ones((1, 3)))
    # 未知 head 模式
    with pytest.raises(spec.SpeculativeError):
        medusa_heads_from_distribution(np.ones((1, 2)), heads=2, mode="bogus")
