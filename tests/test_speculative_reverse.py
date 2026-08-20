"""SPC-REV 逆向投机解码（形态 B：置信驱动回退）核心测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

import speculative as spec
from speculative_reverse import (
    confidence_max,
    reverse_step,
    should_fallback,
)


def test_confidence_max_values():
    onehot = np.array([[0.0, 1.0]], dtype=np.float64)
    assert confidence_max(onehot) == pytest.approx(1.0)
    uniform = np.tile(np.array([0.5, 0.5]), (3, 1))
    assert confidence_max(uniform) == pytest.approx(0.5)
    assert confidence_max(np.zeros((0, 4))) == 1.0
    # 全零行 → 0 置信
    assert confidence_max(np.zeros((2, 4))) == 0.0


def test_should_fallback_threshold():
    assert should_fallback(0.4, 0.6) is True
    assert should_fallback(0.6, 0.6) is False   # 阈值处不回退（走投机）
    assert should_fallback(0.9, 0.6) is False


def _mk(conf_p, verify_p):
    def draft(context, gamma):
        g = max(0, int(gamma))
        row = np.asarray(conf_p, dtype=np.float64)
        rows = np.tile(row, (max(g, 1), 1))
        rng = np.random.default_rng(0)
        tokens = [spec.sample_from_probs(rows[i], rng) for i in range(g)]
        return tokens, rows[:g]

    vrow = np.asarray(verify_p, dtype=np.float64)
    vrows = np.tile(vrow, (32, 1))

    def verify(context, draft_tokens):
        return vrows[:len(draft_tokens) + 1], {}

    return draft, verify, vrow


def test_high_confidence_takes_speculate_path():
    # draft 与 verify 相同且很自信 → 投机接受
    draft, verify, _ = _mk([0.0, 1.0], [0.0, 1.0])
    token, metrics = reverse_step(
        context_ids=[1], draft_fn=draft, verify_fn=verify,
        gamma=3, threshold=0.8, rng=np.random.default_rng(1),
    )
    assert metrics["mode"] == "speculate"
    assert token == 1
    assert metrics["confidence"] == pytest.approx(1.0)


def test_low_confidence_falls_back_to_verify():
    # draft 均匀（conf 0.5）低于阈值 → 回退大模型直采
    draft, verify, vrow = _mk([0.5, 0.5], [0.2, 0.8])
    token, metrics = reverse_step(
        context_ids=[1], draft_fn=draft, verify_fn=verify,
        gamma=3, threshold=0.9, rng=np.random.default_rng(3),
    )
    assert metrics["mode"] == "fallback"
    assert metrics["fallback"] is True
    # 回退输出由 verify 分布采样（大 token 1 更常见，且 token ∈ verify 支持集）
    assert token in (0, 1)


def test_greedy_reverse_step():
    draft, verify, vrow = _mk([0.2, 0.8], [0.0, 1.0])
    fall = reverse_step(
        context_ids=[1], draft_fn=draft, verify_fn=verify,
        gamma=3, threshold=0.99, greedy=True,
    )
    assert fall[1]["mode"] == "fallback"
    assert fall[0] == 1                                # verify argmax
    spec_ = reverse_step(
        context_ids=[1], draft_fn=draft, verify_fn=verify,
        gamma=3, threshold=0.1, greedy=True,
    )
    assert spec_[1]["mode"] == "speculate"
    assert spec_[0] == 1                               # 投机贪心 = verify argmax


def test_reverse_output_distribution_equals_verify():
    """MC：混合（部分 fallback 部分 speculate）下 token 分布仍≈verify。"""
    verify_p = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    draft_state = {"n": 0}
    draw_rng = np.random.default_rng(424242)

    def draft(context, gamma):
        draft_state["n"] += 1
        g = max(0, int(gamma))
        # Dirichlet：置信度（max）在阈值附近波动 → 制造混合路径
        row = draw_rng.dirichlet([0.4, 0.4, 1.2])
        rows = np.tile(row, (max(g, 1), 1))
        rng = np.random.default_rng(draft_state["n"])
        tokens = [spec.sample_from_probs(rows[i], rng) for i in range(g)]
        return tokens, rows[:g]

    vrows = np.tile(verify_p, (32, 1))

    def verify(context, draft_tokens):
        return vrows[:len(draft_tokens) + 1], {}

    counts = np.zeros(3, dtype=np.float64)
    modes = []
    rng = np.random.default_rng(20260821)
    for _ in range(12000):
        token, metrics = reverse_step(
            context_ids=[7], draft_fn=draft, verify_fn=verify,
            gamma=2, threshold=0.6, rng=rng,
        )
        counts[token] += 1.0
        modes.append(metrics["mode"])
    empirical = counts / counts.sum()
    assert {m for m in modes} <= {"fallback", "speculate"}
    assert float(np.max(np.abs(empirical - verify_p))) < 0.05, np.round(empirical, 4)
    # 确实发生混合（不是全在一侧）
    assert "fallback" in modes and "speculate" in modes


def test_reverse_invalid_verify_all_zero_greedy():
    draft, _, _ = _mk([0.5, 0.5], [0.5, 0.5])

    def bad_verify(context, draft_tokens):
        return np.zeros((1, 2)), {}

    with pytest.raises(spec.SpeculativeError):
        reverse_step(
            context_ids=[1], draft_fn=draft, verify_fn=bad_verify,
            gamma=2, threshold=0.9, greedy=True,
        )
