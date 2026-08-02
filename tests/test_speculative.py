"""
单元测试 — 投机解码外部辅助（路线 C-1 阶段 0-1 PoC，speculative）
==================================================================
覆盖:

1. **分布等价（头号正确性测试）**：Monte-Carlo 校验 verify_draft_tokens 的
   产出分布严格等于 verify 模型分布 —— 这是 §2.3 "输出分布与 verify 模型
   一致"的全部意义所在。含单轮首 token 等价与跨轮联合分布等价。
2. 接受率行为：draft==verify 全接受；支撑集不相交则全拒绝。
3. 全部边界：γ=0/1、draft 概率为 0、残差全零回落、NaN/inf/负数、
   未归一化输入、贪心模式、词表越界。
4. SpeculativeSession：轮次 / 停止词 / max_new_tokens / 取消 / 指标算术。
5. ExternalVerifyClient：mock OpenAI 兼容后端（stdlib http.server，
   与 test_external_provider 同模式）——logprobs 解析、缺 logprobs 的中文报错、
   token id 不可还原的中文报错、scope=deny 零外发请求、凭据脱敏。
6. 门控：QLH_SPEC_ENABLED=False → 实验端点 404、/api/status 无 speculative 段、
   主聊天路径的 execution_mode 校验集合未被污染。
"""

import json
import math
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_host import model_host

import speculative as spec
from external_provider import ExternalScopeDeniedError
from speculative import (
    ByteTokenizer,
    ExternalVerifyClient,
    SpeculativeCapabilityError,
    SpeculativeConfigError,
    SpeculativeError,
    SpeculativeSession,
    SpeculativeVerifyError,
    StubDraftModel,
    parse_token_key,
    residual_distribution,
    resolve_verify_config,
    sample_from_probs,
    speculative_model_identity,
    verify_draft_tokens,
)


# ================================================================
# 测试工具
# ================================================================

class FixedRng:
    """返回预设序列的假 RNG（耗尽后重复最后一个值），把分支逼到确定位置。"""

    def __init__(self, values):
        self._values = [float(v) for v in values]
        self._index = 0
        self.draws = []

    def random(self):
        if self._index < len(self._values):
            value = self._values[self._index]
            self._index += 1
        else:
            value = self._values[-1] if self._values else 0.0
        self.draws.append(value)
        return value


def _rows(*rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def _total_variation(left, right) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(left) - np.asarray(right))))


# ================================================================
# 1. 分布等价 —— Monte-Carlo（头号正确性测试）
# ================================================================

# (说明, draft 分布, verify 分布)
_DISTRIBUTION_PAIRS = [
    (
        "完全一致",
        [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
        [0.10, 0.20, 0.30, 0.15, 0.15, 0.10],
    ),
    (
        "轻度失配",
        [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
        [0.10, 0.15, 0.35, 0.20, 0.15, 0.05],
    ),
    (
        "严重失配（draft 峰值恰在 verify 谷底）",
        [0.70, 0.20, 0.05, 0.03, 0.01, 0.01],
        [0.02, 0.03, 0.25, 0.30, 0.20, 0.20],
    ),
    (
        "支撑集不相交",
        [0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.25, 0.25, 0.25, 0.25],
    ),
    (
        "draft 均匀 / verify 尖峰",
        [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
        [0.01, 0.01, 0.01, 0.02, 0.05, 0.90],
    ),
]

MC_DRAWS = 20000
MC_SEED = 20260726
MC_TOLERANCE = 0.02          # 绝对误差上界（N=20000 时 ≈5.7σ）
MC_GAMMA = 3


@pytest.mark.parametrize(
    "label,draft,verify",
    _DISTRIBUTION_PAIRS,
    ids=[item[0] for item in _DISTRIBUTION_PAIRS],
)
def test_monte_carlo_output_distribution_equals_verify(label, draft, verify):
    """
    投机采样的**唯一意义**：无论 draft 多差，产出 token 的分布都等于 verify。

    做法：按 draft 分布抽草稿（这是投机解码里草稿的真实来源），跑一轮
    draft-verify，统计**首个产出 token**的经验分布，与 p_verify[0] 比较。
    """
    vocab = len(verify)
    draft_rows = np.tile(np.asarray(draft, dtype=np.float64), (MC_GAMMA, 1))
    verify_rows = np.tile(np.asarray(verify, dtype=np.float64), (MC_GAMMA + 1, 1))
    rng = np.random.default_rng(MC_SEED)

    counts = np.zeros(vocab, dtype=np.float64)
    for _ in range(MC_DRAWS):
        tokens = [sample_from_probs(draft_rows[i], rng) for i in range(MC_GAMMA)]
        outcome = verify_draft_tokens(draft_rows, verify_rows, tokens, rng=rng)
        assert outcome.emitted >= 1
        counts[outcome.tokens[0]] += 1.0

    empirical = counts / MC_DRAWS
    expected = np.asarray(verify, dtype=np.float64)
    max_error = float(np.max(np.abs(empirical - expected)))
    tv_distance = _total_variation(empirical, expected)
    assert max_error < MC_TOLERANCE, (
        f"{label}: 最大逐 token 偏差 {max_error:.4f} 超过容差 {MC_TOLERANCE}; "
        f"经验分布={np.round(empirical, 4).tolist()}"
    )
    assert tv_distance < MC_TOLERANCE, (
        f"{label}: 总变差 {tv_distance:.4f} 超过容差 {MC_TOLERANCE}"
    )


def test_monte_carlo_joint_distribution_across_rounds():
    """
    跨轮联合分布等价：连续两个产出 token 的联合分布 = q0 ⊗ q1。

    这比单 token 等价更强——它证明"接受则前进、拒绝则回滚"的状态机
    没有在轮次拼接处引入偏差。
    """
    vocab = 3
    draws = 8000
    seed = 4242
    q0 = np.asarray([0.5, 0.3, 0.2])
    q1 = np.asarray([0.2, 0.5, 0.3])
    draft_row = np.asarray([0.8, 0.1, 0.1])     # 与 verify 明显不同
    prompt = [0]

    def verify_fn(context_ids, draft_tokens):
        # 绝对位置 → 分布：位置 len(prompt)+i 决定第 i 个产出 token
        table = [q0, q1, q1]
        start = len(context_ids) - len(prompt)
        rows = []
        for offset in range(len(draft_tokens) + 1):
            index = min(start + offset, len(table) - 1)
            rows.append(table[index])
        return np.asarray(rows, dtype=np.float64)

    class _FixedDraft:
        def __init__(self, rng):
            self._rng = rng

        def __call__(self, context_ids, gamma):
            rows = np.tile(draft_row, (gamma, 1))
            tokens = [sample_from_probs(rows[i], self._rng) for i in range(gamma)]
            return tokens, rows

    rng = np.random.default_rng(seed)
    joint = np.zeros((vocab, vocab), dtype=np.float64)
    for _ in range(draws):
        session = SpeculativeSession(
            prompt,
            _FixedDraft(rng),
            verify_fn,
            gamma=1,
            max_new_tokens=2,
            rng=rng,
        )
        result = session.run()
        assert len(result.tokens) == 2
        joint[result.tokens[0], result.tokens[1]] += 1.0

    empirical = joint / draws
    expected = np.outer(q0, q1)
    max_error = float(np.max(np.abs(empirical - expected)))
    assert max_error < 0.03, (
        f"跨轮联合分布偏差 {max_error:.4f} 过大; 经验={np.round(empirical, 4).tolist()}"
    )


# ================================================================
# 2. 接受率行为
# ================================================================

def test_identical_distributions_accept_everything():
    row = [0.25, 0.25, 0.25, 0.25]
    draft_rows = _rows(row, row, row, row)
    verify_rows = _rows(row, row, row, row, row)
    rng = np.random.default_rng(7)
    for _ in range(200):
        tokens = [int(rng.integers(0, 4)) for _ in range(4)]
        outcome = verify_draft_tokens(draft_rows, verify_rows, tokens, rng=rng)
        assert outcome.accepted_count == 4
        assert outcome.rejected_count == 0
        assert outcome.correction_token is None
        assert outcome.bonus_token is not None
        assert outcome.tokens[:4] == tuple(tokens)


def test_disjoint_supports_accept_nothing():
    draft_rows = _rows([0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0])
    verify_rows = _rows([0.0, 0.0, 0.5, 0.5]) .repeat(3, axis=0)
    rng = np.random.default_rng(11)
    for _ in range(100):
        tokens = [int(rng.integers(0, 2)) for _ in range(2)]
        outcome = verify_draft_tokens(draft_rows, verify_rows, tokens, rng=rng)
        assert outcome.accepted_count == 0
        assert outcome.rejected_count == 2
        assert outcome.correction_token in (2, 3)   # 只能来自 verify 的支撑集
        assert outcome.tokens == (outcome.correction_token,)


# ================================================================
# 3. 边界条件
# ================================================================

def test_gamma_zero_returns_bonus_token_only():
    verify_rows = _rows([0.0, 1.0, 0.0])
    outcome = verify_draft_tokens(None, verify_rows, [], rng=FixedRng([0.5]))
    assert outcome.accepted_count == 0
    assert outcome.rejected_count == 0
    assert outcome.correction_token is None
    assert outcome.bonus_token == 1
    assert outcome.tokens == (1,)


def test_gamma_zero_without_bonus_row_emits_nothing():
    outcome = verify_draft_tokens(None, np.zeros((0, 3)), [], rng=FixedRng([0.5]))
    assert outcome.tokens == ()
    assert outcome.emitted == 0


def test_gamma_one_accept_and_reject():
    draft_rows = _rows([0.5, 0.5, 0.0])
    verify_rows = _rows([0.25, 0.75, 0.0], [0.0, 0.0, 1.0])
    # ratio = 0.75/0.5 → min(1, 1.5) = 1 → 必接受，并取奖励 token
    accepted = verify_draft_tokens(draft_rows, verify_rows, [1], rng=FixedRng([0.99, 0.5]))
    assert accepted.accepted_count == 1
    assert accepted.tokens == (1, 2)
    # ratio = 0.25/0.5 = 0.5，抽 0.9 → 拒绝
    rejected = verify_draft_tokens(draft_rows, verify_rows, [0], rng=FixedRng([0.9, 0.1]))
    assert rejected.accepted_count == 0
    assert rejected.correction_token == 1     # 残差 max(0, q-p) 只在 token1 上有质量
    assert rejected.tokens == (1,)


def test_zero_draft_probability_is_guarded():
    """draft 概率为 0（top-k 截断/浮点误差会造成）不得触发除零。"""
    draft_rows = _rows([1.0, 0.0, 0.0])
    verify_rows = _rows([0.2, 0.8, 0.0], [0.0, 0.0, 1.0])
    outcome = verify_draft_tokens(draft_rows, verify_rows, [1], rng=FixedRng([0.999]))
    assert outcome.accepted_count == 1        # p=0 且 q>0 → 比值取 1，接受
    assert math.isfinite(outcome.positions[0].accept_ratio)
    assert outcome.positions[0].accept_ratio == 1.0

    # p=0 且 q=0 → 比值 0，必拒绝
    verify_zero = _rows([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    rejected = verify_draft_tokens(draft_rows, verify_zero, [1], rng=FixedRng([0.0]))
    assert rejected.accepted_count == 0
    assert rejected.positions[0].accept_ratio == 0.0
    assert rejected.correction_token == 0


def test_residual_all_zero_falls_back_to_verify_distribution():
    """残差退化（p_verify 处处不高于 p_draft）→ 回落到直接从 verify 采样。"""
    row = [0.5, 0.5, 0.0]
    draft_rows = _rows(row)
    verify_rows = _rows(row, [0.0, 0.0, 1.0])
    # rng 返回 1.0：draw < ratio 恒不成立，即使 ratio=1 也强制走到拒绝分支
    rng = FixedRng([1.0])
    outcome = verify_draft_tokens(draft_rows, verify_rows, [0], rng=rng)
    assert outcome.accepted_count == 0
    assert outcome.residual_fallback is True
    assert outcome.correction_token in (0, 1)     # 来自 verify 分布本身
    residual, total = residual_distribution(np.asarray(row), np.asarray(row))
    assert total <= spec.PROB_EPS
    assert float(np.sum(residual)) == pytest.approx(0.0)


def test_non_finite_and_negative_values_are_sanitized():
    draft_rows = _rows([0.5, float("nan"), 0.5])
    verify_rows = _rows(
        [0.5, float("inf"), -0.25],
        [float("-inf"), 1.0, float("nan")],
    )
    outcome = verify_draft_tokens(draft_rows, verify_rows, [0], rng=FixedRng([0.999, 0.5]))
    # verify 行清洗后 = [1, 0, 0]；draft 行清洗后 = [0.5, 0, 0.5]
    assert outcome.positions[0].p_verify == pytest.approx(1.0)
    assert outcome.positions[0].p_draft == pytest.approx(0.5)
    assert outcome.accepted_count == 1
    assert outcome.bonus_token == 1


def test_unnormalized_rows_are_renormalized():
    normalized = verify_draft_tokens(
        _rows([0.25, 0.75]), _rows([0.5, 0.5], [1.0, 0.0]), [0],
        rng=FixedRng([0.9, 0.1]),
    )
    scaled = verify_draft_tokens(
        _rows([2.5, 7.5]), _rows([50.0, 50.0], [3.0, 0.0]), [0],
        rng=FixedRng([0.9, 0.1]),
    )
    assert normalized.accepted_count == scaled.accepted_count
    assert normalized.tokens == scaled.tokens
    assert scaled.positions[0].p_draft == pytest.approx(0.25)
    assert scaled.positions[0].p_verify == pytest.approx(0.5)


def test_greedy_mode_accepts_only_verify_argmax():
    draft_rows = _rows([0.5, 0.5, 0.0], [0.5, 0.5, 0.0])
    verify_rows = _rows([0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.0, 0.0, 1.0])
    hit = verify_draft_tokens(draft_rows, verify_rows, [1, 0], greedy=True)
    assert hit.accepted_count == 2
    assert hit.bonus_token == 2
    assert all(pos.random_draw == -1.0 for pos in hit.positions)   # 未消耗随机数

    miss = verify_draft_tokens(draft_rows, verify_rows, [0, 0], greedy=True)
    assert miss.accepted_count == 0
    assert miss.correction_token == 1        # verify argmax
    assert miss.tokens == (1,)


def test_out_of_range_token_and_short_verify_matrix_raise_chinese_errors():
    with pytest.raises(SpeculativeError, match="超出词表范围"):
        verify_draft_tokens(_rows([0.5, 0.5]), _rows([0.5, 0.5], [0.5, 0.5]), [5])
    with pytest.raises(SpeculativeError, match="行数不足|行数不匹配"):
        verify_draft_tokens(
            _rows([0.5, 0.5], [0.5, 0.5]), _rows([0.5, 0.5]), [0, 1],
        )


def test_all_zero_verify_row_raises_on_sampling():
    with pytest.raises(SpeculativeError, match="全零概率分布"):
        sample_from_probs(np.zeros(4), FixedRng([0.5]))


def test_sample_from_probs_is_deterministic_with_injected_rng():
    row = np.asarray([0.2, 0.3, 0.5])
    assert sample_from_probs(row, FixedRng([0.0])) == 0
    assert sample_from_probs(row, FixedRng([0.25])) == 1
    assert sample_from_probs(row, FixedRng([0.9])) == 2
    # 边界：抽到 1.0（真实 rng 不会，但假 rng 会）落在最后一个正概率 token
    assert sample_from_probs(np.asarray([0.5, 0.5, 0.0]), FixedRng([1.0])) == 1


# ================================================================
# 4. SpeculativeSession
# ================================================================

def _make_verify_fn(table, prompt_len, meta=None):
    """按绝对位置查表的假 verify 模型。"""
    def _verify(context_ids, draft_tokens):
        start = len(context_ids) - prompt_len
        rows = []
        for offset in range(len(draft_tokens) + 1):
            index = min(start + offset, len(table) - 1)
            rows.append(table[index])
        matrix = np.asarray(rows, dtype=np.float64)
        return (matrix, dict(meta)) if meta is not None else matrix
    return _verify


def _one_hot(vocab, token):
    row = np.zeros(vocab, dtype=np.float64)
    row[token] = 1.0
    return row


def test_session_perfect_draft_reaches_full_acceptance():
    vocab = 8
    target = [3, 4, 5, 6, 7, 3, 4, 5]
    table = [_one_hot(vocab, t) for t in target]
    prompt = [1, 2]

    def draft_fn(context_ids, gamma):
        start = len(context_ids) - len(prompt)
        tokens, rows = [], []
        for offset in range(gamma):
            index = min(start + offset, len(target) - 1)
            tokens.append(target[index])
            rows.append(_one_hot(vocab, target[index]))
        return tokens, np.asarray(rows)

    session = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=4, max_new_tokens=8, rng=np.random.default_rng(0),
    )
    result = session.run()
    assert result.tokens == target
    assert result.finish_reason == "length"
    metrics = result.metrics
    assert metrics["acceptance_rate"] == 1.0
    assert metrics["rejected_tokens"] == 0
    # γ=4 时每轮产出 5 个 token（4 接受 + 1 奖励）→ 8 token 需 2 轮
    assert metrics["rounds"] == 2
    assert metrics["tokens_per_round"] == pytest.approx(4.0)
    assert metrics["bonus_tokens"] == 2
    assert metrics["correction_tokens"] == 0


def test_session_stop_token_and_metric_arithmetic():
    vocab = 6
    target = [2, 3, 4, 5]          # 5 作为停止 token
    table = [_one_hot(vocab, t) for t in target]
    prompt = [0, 1]

    def draft_fn(context_ids, gamma):
        start = len(context_ids) - len(prompt)
        tokens, rows = [], []
        for offset in range(gamma):
            index = min(start + offset, len(target) - 1)
            tokens.append(target[index])
            rows.append(_one_hot(vocab, target[index]))
        return tokens, np.asarray(rows)

    session = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=4, max_new_tokens=32, stop_token_ids=[5],
        rng=np.random.default_rng(0),
    )
    result = session.run()
    assert result.tokens == [2, 3, 4, 5]
    assert result.finish_reason == "stop"

    metrics = result.metrics
    rounds = metrics["rounds"]
    # 指标算术自洽性
    assert metrics["accepted_tokens"] + metrics["rejected_tokens"] == metrics["drafted_tokens"]
    assert metrics["acceptance_rate"] == pytest.approx(
        metrics["accepted_tokens"] / metrics["drafted_tokens"], rel=1e-4,
    )
    assert metrics["tokens_per_round"] == pytest.approx(
        metrics["emitted_tokens"] / rounds, rel=1e-4,
    )
    assert metrics["verify_calls"] == rounds
    assert metrics["verify_latency_ms_mean"] == pytest.approx(
        metrics["verify_latency_ms_total"] / rounds, abs=0.05,
    )
    assert metrics["emitted_tokens"] == len(result.tokens)
    assert metrics["prompt_tokens"] == len(prompt)
    assert len(result.rounds) == rounds
    assert sum(item["emitted"] for item in result.rounds) == metrics["emitted_tokens"]
    # 无状态 verify：上行含整段 context，必然大于有状态理想值
    assert metrics["bytes_up"] > metrics["bytes_up_ideal_stateful"]
    assert metrics["bytes_per_round"] == pytest.approx(
        (metrics["bytes_up"] + metrics["bytes_down"]) / rounds, rel=1e-3,
    )


def test_session_respects_max_new_tokens_and_max_rounds():
    vocab = 5
    table = [_one_hot(vocab, 4)] * 4
    prompt = [0]

    def draft_fn(context_ids, gamma):
        rows = np.tile(_one_hot(vocab, 4), (gamma, 1))
        return [4] * gamma, rows

    capped = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=4, max_new_tokens=3, rng=np.random.default_rng(0),
    ).run()
    assert len(capped.tokens) == 3
    assert capped.finish_reason == "length"

    rounds_capped = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=1, max_new_tokens=100, max_rounds=3, rng=np.random.default_rng(0),
    ).run()
    assert rounds_capped.metrics["rounds"] == 3
    assert rounds_capped.finish_reason == "max_rounds"


def test_session_cancel_event_stops_between_rounds():
    vocab = 4
    table = [_one_hot(vocab, 3)] * 4
    prompt = [0]
    cancel = threading.Event()
    calls = {"n": 0}

    def draft_fn(context_ids, gamma):
        calls["n"] += 1
        if calls["n"] >= 2:
            cancel.set()
        return [3] * gamma, np.tile(_one_hot(vocab, 3), (gamma, 1))

    session = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=1, max_new_tokens=64, cancel_event=cancel,
        rng=np.random.default_rng(0),
    )
    result = session.run()
    assert result.finish_reason == "cancelled"
    assert calls["n"] == 2                     # 取消在轮边界生效，未继续起草
    assert len(result.tokens) == 2             # 第 1 轮的 2 个 token 已产出

    # 起手即取消 → 一轮都不跑
    pre_cancel = threading.Event()
    pre_cancel.set()
    empty = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt)),
        gamma=1, max_new_tokens=8, cancel_event=pre_cancel,
    ).run()
    assert empty.tokens == []
    assert empty.metrics["rounds"] == 0
    assert empty.metrics["acceptance_rate"] == 0.0


def test_session_uses_verify_reported_bytes_and_latency():
    vocab = 4
    table = [_one_hot(vocab, 2)] * 3
    prompt = [0]
    meta = {"bytes_up": 1024, "bytes_down": 512, "latency_ms": 25.0}

    def draft_fn(context_ids, gamma):
        return [2] * gamma, np.tile(_one_hot(vocab, 2), (gamma, 1))

    result = SpeculativeSession(
        prompt, draft_fn, _make_verify_fn(table, len(prompt), meta=meta),
        gamma=2, max_new_tokens=6, rng=np.random.default_rng(0),
    ).run()
    rounds = result.metrics["rounds"]
    assert result.metrics["bytes_measured"] is True
    assert result.metrics["bytes_up"] == 1024 * rounds
    assert result.metrics["bytes_down"] == 512 * rounds
    assert result.metrics["verify_latency_ms_total"] == pytest.approx(25.0 * rounds)
    assert result.metrics["verify_latency_ms_max"] == pytest.approx(25.0)


def test_session_rejects_oversized_draft_and_guards_empty_round():
    vocab = 3
    prompt = [0]

    def greedy_draft(context_ids, gamma):
        return [1] * (gamma + 1), np.tile(_one_hot(vocab, 1), (gamma + 1, 1))

    with pytest.raises(SpeculativeError, match="超过请求的"):
        SpeculativeSession(
            prompt, greedy_draft, _make_verify_fn([_one_hot(vocab, 1)], 1),
            gamma=2, max_new_tokens=4,
        ).run()

    # verify 端不给奖励行 + γ=0 → 无 token 产出，必须跳出而不是死循环
    def empty_draft(context_ids, gamma):
        return [], np.zeros((0, vocab))

    def no_row_verify(context_ids, draft_tokens):
        return np.zeros((0, vocab))

    result = SpeculativeSession(
        prompt, empty_draft, no_row_verify, gamma=0, max_new_tokens=4,
    ).run()
    assert result.finish_reason == "verify_no_token"
    assert result.tokens == []


def test_session_metrics_recorded_for_status_snapshot():
    spec.reset_last_session_metrics()
    assert spec.get_last_session_metrics() == {}
    vocab = 3
    table = [_one_hot(vocab, 1)] * 3

    def draft_fn(context_ids, gamma):
        return [1] * gamma, np.tile(_one_hot(vocab, 1), (gamma, 1))

    SpeculativeSession(
        [0], draft_fn, _make_verify_fn(table, 1), gamma=2, max_new_tokens=4,
    ).run()
    snapshot = spec.get_last_session_metrics()
    assert snapshot["rounds"] >= 1
    assert "recorded_at" in snapshot
    spec.reset_last_session_metrics()


def test_stub_draft_model_hint_raises_acceptance():
    vocab = 32
    hint = [7, 8, 9, 10]
    model = StubDraftModel(vocab, seed=3, hint_tokens=hint, hint_weight=0.95)
    tokens, rows = model([0, 1], 4)
    assert tokens == hint                    # hint 权重 0.95，seed 固定下必然命中
    assert rows.shape == (4, vocab)
    assert float(np.sum(rows[0])) == pytest.approx(1.0)

    uniform = StubDraftModel(vocab, seed=3)
    _, uniform_rows = uniform([0, 1], 2)
    assert float(np.max(uniform_rows[0])) == pytest.approx(1.0 / vocab)


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("投机解码 hello")
    assert tokenizer.decode(ids) == "投机解码 hello"
    assert tokenizer.decode(ids + [tokenizer.EOS_ID]) == "投机解码 hello"
    assert tokenizer.vocab_size == 257


# ================================================================
# 5. ExternalVerifyClient + mock OpenAI 兼容后端
# ================================================================

class MockVerifyHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容 verify 端点：/v1/models + /v1/completions(echo+logprobs)。"""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        behavior = self.server.behavior
        self.server.requests.append({"method": "GET", "path": self.path,
                                     "headers": dict(self.headers)})
        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": behavior["model_id"], "object": "model"}],
            })
        else:
            self._send_json({"error": {"message": "not found"}}, status=404)

    def _distribution(self, position: int) -> dict:
        """位置 → {token_id: 概率}（目标序列上 0.9，另外两个候选各 0.05）。"""
        behavior = self.server.behavior
        target = behavior["target"]
        vocab = behavior.get("vocab_size", 257)
        main = target[position] if position < len(target) else behavior["eos_id"]
        alt_a = (main + 1) % vocab
        alt_b = (main + 2) % vocab
        return {int(main): 0.9, int(alt_a): 0.05, int(alt_b): 0.05}

    def do_POST(self):
        behavior = self.server.behavior
        length = int(self.headers.get("Content-Length", 0) or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append({"method": "POST", "path": self.path,
                                     "headers": dict(self.headers),
                                     "payload": payload})
        if self.path != "/v1/completions":
            self._send_json({"error": {"message": "not found"}}, status=404)
            return
        if behavior.get("post_status", 200) != 200:
            self._send_json({"error": {"message": "verify backend error"}},
                            status=behavior["post_status"])
            return

        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            prompt_ids = list(prompt.encode("utf-8"))
        else:
            prompt_ids = [int(t) for t in (prompt or [])]

        if behavior.get("mode") == "no_logprobs":
            self._send_json({
                "id": "cmpl-mock", "object": "text_completion",
                "model": behavior["model_id"],
                "choices": [{"index": 0, "text": "x", "finish_reason": "length"}],
            })
            return

        text_keys = behavior.get("mode") == "text_keys"
        truncated = behavior.get("mode") == "truncated"
        tokens, token_logprobs, top_logprobs = [], [], []
        limit = len(prompt_ids) + 1
        if truncated:
            limit = max(1, len(prompt_ids) - 1)
        for index in range(limit):
            distribution = self._distribution(index)
            tokens.append(f"token_id:{index}")
            if index == 0:
                token_logprobs.append(None)
                top_logprobs.append(None)
                continue
            if index < len(prompt_ids):
                actual = prompt_ids[index]
            else:
                actual = max(distribution, key=distribution.get)
            token_logprobs.append(math.log(distribution.get(actual, 1e-9)))
            if text_keys:
                entry = {
                    f"<tok{key}>": math.log(value)
                    for key, value in distribution.items()
                }
            else:
                entry = {
                    f"token_id:{key}": math.log(value)
                    for key, value in distribution.items()
                }
            top_logprobs.append(entry)

        self._send_json({
            "id": "cmpl-mock",
            "object": "text_completion",
            "model": behavior["model_id"],
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "length",
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs,
                    "text_offset": list(range(len(tokens))),
                },
            }],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": 1},
        })


@pytest.fixture()
def mock_verify_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockVerifyHandler)
    server.behavior = {
        "model_id": "qwen2.5-14b-verify",
        "target": [],
        "eos_id": ByteTokenizer.EOS_ID,
        "vocab_size": ByteTokenizer.vocab_size,
    }
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _base_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _patch_spec_config(
    monkeypatch,
    server=None,
    *,
    base_url=None,
    enabled=True,
    data_scope="opt_in",
    gamma=4,
    max_new_tokens=32,
    max_rounds=64,
    temperature=1.0,
    top_logprobs=8,
    verify_model="",
    verify_api_key="",
    external_base_url="",
    external_api_key="",
    stateful=False,
):
    import config

    resolved = base_url if base_url is not None else (
        _base_url(server) if server is not None else ""
    )
    monkeypatch.setattr(config, "SPEC_ENABLED", enabled, raising=False)
    monkeypatch.setattr(config, "SPEC_VERIFY_BASE_URL", resolved, raising=False)
    monkeypatch.setattr(config, "SPEC_VERIFY_API_KEY", verify_api_key, raising=False)
    monkeypatch.setattr(config, "SPEC_VERIFY_MODEL", verify_model, raising=False)
    monkeypatch.setattr(config, "SPEC_GAMMA", gamma, raising=False)
    monkeypatch.setattr(config, "SPEC_MAX_ROUNDS", max_rounds, raising=False)
    monkeypatch.setattr(config, "SPEC_MAX_NEW_TOKENS", max_new_tokens, raising=False)
    monkeypatch.setattr(config, "SPEC_TIMEOUT", 10, raising=False)
    monkeypatch.setattr(config, "SPEC_CONNECT_TIMEOUT", 3, raising=False)
    monkeypatch.setattr(config, "SPEC_TEMPERATURE", temperature, raising=False)
    monkeypatch.setattr(config, "SPEC_TOP_LOGPROBS", top_logprobs, raising=False)
    monkeypatch.setattr(config, "SPEC_STATEFUL_VERIFY", stateful, raising=False)
    monkeypatch.setattr(config, "SPEC_LABEL", "测试投机解码", raising=False)
    monkeypatch.setattr(config, "EXTERNAL_DATA_SCOPE", data_scope, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_BASE_URL", external_base_url, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_API_KEY", external_api_key, raising=False)
    monkeypatch.setattr(config, "EXTERNAL_MODEL", "", raising=False)
    monkeypatch.setattr(config, "EXTERNAL_ENABLED", False, raising=False)
    return resolved


@pytest.fixture(autouse=True)
def _reset_spec_state():
    spec.reset_last_session_metrics()
    yield
    spec.reset_last_session_metrics()


def test_verify_client_parses_logprobs_into_probability_rows(
    monkeypatch, mock_verify_server,
):
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["target"] = [10, 20, 30, 40, 50, 60]
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    client.connect(allow_external=True)
    assert client.capability["logprobs"] is True
    assert client.capability["token_ids_resolvable"] is True

    context = [10, 20]
    drafts = [30, 40]
    rows, meta = client.verify(context, drafts, allow_external=True)
    assert rows.shape == (3, ByteTokenizer.vocab_size)
    # 位置 2/3 的主候选 = target[2]/target[3]，概率 0.9（归一化前）
    assert rows[0][30] == pytest.approx(0.9, rel=1e-6)
    assert rows[1][40] == pytest.approx(0.9, rel=1e-6)
    assert rows[2][50] == pytest.approx(0.9, rel=1e-6)
    assert meta["bytes_up"] > 0 and meta["bytes_down"] > 0
    assert meta["latency_ms"] >= 0
    posts = [r for r in mock_verify_server.requests if r["method"] == "POST"]
    assert posts[-1]["path"] == "/v1/completions"
    assert posts[-1]["payload"]["prompt"] == context + drafts
    assert posts[-1]["payload"]["echo"] is True
    assert posts[-1]["payload"]["logprobs"] == 8
    client.close()


def test_verify_client_uses_exact_logprob_for_draft_token(
    monkeypatch, mock_verify_server,
):
    """草稿 token 的 verify 概率来自 token_logprobs（top-k 之外也准确）。"""
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["target"] = [1, 2, 3, 4]
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    client.connect(allow_external=True)
    # 草稿 200 不在 target 的 top-3 候选里 → 概率取兜底的 1e-9
    rows, _ = client.verify([1, 2], [200], allow_external=True)
    assert rows[0][200] == pytest.approx(1e-9, rel=1e-3)
    assert rows[0][3] == pytest.approx(0.9, rel=1e-6)      # top-k 给出的主候选
    client.close()


def test_verify_client_missing_logprobs_raises_chinese_capability_error(
    monkeypatch, mock_verify_server,
):
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["mode"] = "no_logprobs"
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    with pytest.raises(SpeculativeCapabilityError) as excinfo:
        client.connect(allow_external=True)
    message = str(excinfo.value)
    assert "logprobs" in message
    assert "投机" in message or "外部端点不满足" in message
    assert "vLLM" in message or "SGLang" in message
    client.close()


def test_verify_client_unresolvable_token_keys_raise_capability_error(
    monkeypatch, mock_verify_server,
):
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["target"] = [1, 2, 3]
    mock_verify_server.behavior["mode"] = "text_keys"
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    with pytest.raises(SpeculativeCapabilityError) as excinfo:
        client.connect(allow_external=True)
    assert "token id" in str(excinfo.value)
    assert "--return-tokens-as-token-ids" in str(excinfo.value)
    client.close()


def test_verify_client_accepts_token_id_resolver(monkeypatch, mock_verify_server):
    """给出 token_id_resolver 后，纯文本 key 的端点也可用。"""
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["target"] = [1, 2, 3]
    mock_verify_server.behavior["mode"] = "text_keys"

    def resolver(key: str):
        text = str(key)
        if text.startswith("<tok") and text.endswith(">"):
            try:
                return int(text[4:-1])
            except ValueError:
                return None
        return None

    client = ExternalVerifyClient(
        ByteTokenizer.vocab_size, token_id_resolver=resolver,
    )
    client.connect(allow_external=True)
    assert client.capability["token_ids_resolvable"] is True
    client.close()


def test_verify_client_truncated_response_raises_verify_error(
    monkeypatch, mock_verify_server,
):
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["target"] = [1, 2, 3, 4, 5, 6]
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    client.connect(allow_external=True)
    mock_verify_server.behavior["mode"] = "truncated"
    with pytest.raises(SpeculativeVerifyError, match="长度不足"):
        client.verify([1, 2, 3], [4, 5], allow_external=True)
    client.close()


def test_verify_client_scope_deny_sends_zero_requests(
    monkeypatch, mock_verify_server,
):
    """作用域 deny：连健康检查都不发，服务端一个请求都收不到。"""
    _patch_spec_config(monkeypatch, mock_verify_server, data_scope="deny")
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    with pytest.raises(ExternalScopeDeniedError):
        client.connect(allow_external=True)
    assert mock_verify_server.requests == []

    # opt_in 但未授权同样零外发
    _patch_spec_config(monkeypatch, mock_verify_server, data_scope="opt_in")
    with pytest.raises(ExternalScopeDeniedError):
        client.connect(allow_external=False)
    assert mock_verify_server.requests == []


def test_verify_client_masks_url_credentials(monkeypatch, mock_verify_server):
    port = mock_verify_server.server_address[1]
    _patch_spec_config(
        monkeypatch, base_url=f"http://specuser:specpass@127.0.0.1:{port}",
    )
    mock_verify_server.behavior["target"] = [1, 2, 3]
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    client.connect(allow_external=True)
    masked = client.masked_base_url
    assert "specpass" not in masked and "specuser" not in masked
    assert masked == f"http://127.0.0.1:{port}"
    # URL 内嵌账号已转成 BasicAuth 头发出
    assert any(
        "Authorization" in request["headers"]
        and request["headers"]["Authorization"].startswith("Basic ")
        for request in mock_verify_server.requests
    )
    client.close()


def test_verify_client_unreachable_endpoint_raises_chinese_error(monkeypatch):
    _patch_spec_config(monkeypatch, base_url=f"http://127.0.0.1:{_closed_port()}")
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    with pytest.raises(SpeculativeVerifyError) as excinfo:
        client.connect(allow_external=True)
    assert "外部推理服务" in str(excinfo.value)


def test_verify_client_missing_base_url_raises_config_error(monkeypatch):
    _patch_spec_config(monkeypatch, base_url="")
    client = ExternalVerifyClient(ByteTokenizer.vocab_size)
    with pytest.raises(SpeculativeConfigError, match="QLH_SPEC_VERIFY_BASE_URL"):
        client.connect(allow_external=True)


def test_end_to_end_session_against_mock_backend(monkeypatch, mock_verify_server):
    """完整多轮会话：draft 命中 hint → 高接受率，产出文本来自 verify 分布。"""
    tokenizer = ByteTokenizer()
    prompt = "请解释投机解码"
    answer = "投机解码把草稿交给大模型校验"
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    _patch_spec_config(
        monkeypatch, mock_verify_server, gamma=4,
        max_new_tokens=len(answer_ids), temperature=1.0,
    )
    mock_verify_server.behavior["target"] = prompt_ids + answer_ids

    result = spec.run_speculative_chat(
        prompt, allow_external=True, seed=17, draft_hint=answer,
    )
    metrics = result["metrics"]
    assert metrics["execution_mode"] == "speculative_assisted"
    assert metrics["engine"] == "speculative_assisted"
    assert metrics["rounds"] >= 2
    assert metrics["acceptance_rate"] > 0.5        # hint 命中 → 高接受率
    assert metrics["verify_calls"] == metrics["rounds"]
    assert metrics["bytes_measured"] is True
    assert metrics["verify_model"] == "qwen2.5-14b-verify"   # 关连接前已取到
    assert metrics["bytes_up"] > metrics["bytes_up_ideal_stateful"]
    # 标准 completions 的 echo=true 回传整段 prompt 的 logprobs，
    # 下行远超"只回 γ+1 行"的理想值（实施说明"已知限制"记录了此结论）
    assert metrics["bytes_down"] > metrics["bytes_down_ideal_suffix"]
    assert metrics["tokens_per_round"] > 1.0
    assert result["content"]                        # 产出真实文本
    # 产出分布 = verify 分布（mock 在目标 token 上放 0.9，另两候选各 0.05），
    # 因此约九成 token 应等于目标序列——这是"质量对齐外部端"的直接体现
    matched = sum(
        1 for produced, expected in zip(result["tokens"], answer_ids)
        if produced == expected
    )
    assert matched / len(answer_ids) > 0.7
    assert len(result["rounds"]) == metrics["rounds"]
    posts = [r for r in mock_verify_server.requests if r["method"] == "POST"]
    # 无状态 verify：每轮都重传整段 context（§2.3 的"重付 prefill"）
    assert len(posts) >= metrics["rounds"]
    assert len(posts[-1]["payload"]["prompt"]) > len(prompt_ids)


def test_end_to_end_greedy_mode(monkeypatch, mock_verify_server):
    tokenizer = ByteTokenizer()
    prompt = "你好"
    answer = "世界"
    _patch_spec_config(
        monkeypatch, mock_verify_server, gamma=2, max_new_tokens=6, temperature=0.0,
    )
    mock_verify_server.behavior["target"] = (
        tokenizer.encode(prompt) + tokenizer.encode(answer)
    )
    result = spec.run_speculative_chat(prompt, allow_external=True, seed=5)
    assert result["metrics"]["greedy"] is True
    # 贪心模式下 verify argmax 即目标序列
    assert result["content"].startswith(answer[0])


def test_run_speculative_chat_scope_denied_sends_nothing(
    monkeypatch, mock_verify_server,
):
    _patch_spec_config(monkeypatch, mock_verify_server, data_scope="deny")
    with pytest.raises(ExternalScopeDeniedError):
        spec.run_speculative_chat("敏感内容", allow_external=True)
    assert mock_verify_server.requests == []


# ================================================================
# 6. 配置解析 / 模型身份
# ================================================================

def test_resolve_verify_config_falls_back_to_external(monkeypatch):
    _patch_spec_config(
        monkeypatch, base_url="", external_base_url="http://ext:8000",
        external_api_key="ext-key",
    )
    import config
    monkeypatch.setattr(config, "EXTERNAL_MODEL", "ext-model", raising=False)
    resolved = resolve_verify_config()
    assert resolved["base_url"] == "http://ext:8000"
    assert resolved["api_key"] == "ext-key"
    assert resolved["model"] == "ext-model"
    assert resolved["inherited_from_external"] is True


def test_resolve_verify_config_does_not_leak_external_key_to_other_host(monkeypatch):
    """显式配置了另一个 verify 端点时，绝不复用路线 B 的凭据。"""
    _patch_spec_config(
        monkeypatch, base_url="http://verify-host:8000",
        external_base_url="http://ext:8000", external_api_key="ext-key",
    )
    resolved = resolve_verify_config()
    assert resolved["base_url"] == "http://verify-host:8000"
    assert resolved["api_key"] == ""
    assert resolved["inherited_from_external"] is False


def test_parse_token_key_variants():
    assert parse_token_key("token_id:1234") == 1234
    assert parse_token_key("5678") == 5678
    assert parse_token_key(42) == 42
    assert parse_token_key("hello") is None
    assert parse_token_key("") is None
    assert parse_token_key(None) is None


def test_speculative_model_identity_accepted_by_validators(monkeypatch):
    from task_provider import ModelIdentity
    from task_worker_protocol import _validate_capabilities, _validate_model_identity

    _patch_spec_config(monkeypatch, base_url="http://verify-host:8000")
    identity = speculative_model_identity(draft_model_id="qwen-1_8b")
    assert identity.engine == "speculative_assisted"
    assert isinstance(identity, ModelIdentity)

    _validate_model_identity(identity.snapshot(), "payload.model")
    _validate_capabilities({
        "stage_types": ["full_inference"],
        "engines": ["speculative_assisted"],
        "models": [identity.snapshot()],
        "max_concurrency": 1,
    })


def test_check_speculative_available(monkeypatch, mock_verify_server):
    _patch_spec_config(monkeypatch, mock_verify_server)
    assert spec.check_speculative_available() is True
    _patch_spec_config(monkeypatch, mock_verify_server, enabled=False)
    assert spec.check_speculative_available() is False
    _patch_spec_config(monkeypatch, base_url="")
    assert spec.check_speculative_available() is False


# ================================================================
# 7. API 层：门控 / 状态段 / 实验端点
# ================================================================

@pytest.fixture()
def api_env(monkeypatch, tmp_path):
    """TestClient + mock scheduler + 无本地模型的干净 api_server 环境。"""
    from fastapi.testclient import TestClient

    import api_server
    import config

    with patch("api_server.scheduler", MagicMock()) as mock_sched:
        mock_sched.get_effective_node_id.return_value = "test-node"
        mock_sched._effective_role.return_value = "master"
        mock_sched.get_distributed_inference_enabled.return_value = False
        mock_sched.has_pipeline_worker_reservation.return_value = False
        mock_sched._max_nodes = 3
        monkeypatch.setattr(model_host, "model_loaded", False)
        monkeypatch.setattr(model_host, "_db_available", False)
        monkeypatch.setattr(api_server, "_local_store", MagicMock())
        monkeypatch.setattr(
            config, "GGUF_MODEL_PATH", str(tmp_path / "no-model.gguf"),
        )
        monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "no-model-dir"))
        monkeypatch.setattr(config, "ISLAND_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "EXTERNAL_ENABLED", False, raising=False)
        client = TestClient(api_server.app)
        yield {"client": client, "api_server": api_server}


def test_status_omits_speculative_section_when_disabled(monkeypatch, api_env):
    _patch_spec_config(monkeypatch, base_url="", enabled=False)
    response = api_env["client"].get("/api/status")
    assert response.status_code == 200
    assert response.json()["speculative"] is None


def test_status_reports_speculative_section_masked(
    monkeypatch, mock_verify_server, api_env,
):
    port = mock_verify_server.server_address[1]
    _patch_spec_config(
        monkeypatch, base_url=f"http://specuser:specpass@127.0.0.1:{port}",
        gamma=6, verify_model="qwen2.5-14b-verify",
    )
    response = api_env["client"].get("/api/status")
    assert response.status_code == 200
    section = response.json()["speculative"]
    assert section["enabled"] is True
    assert section["label"] == "测试投机解码"
    assert section["gamma"] == 6
    assert section["verify_model"] == "qwen2.5-14b-verify"
    assert section["data_scope"] == "opt_in"
    assert "specpass" not in section["verify_base_url"]
    assert section["verify_base_url"] == f"http://127.0.0.1:{port}"
    # /api/status 绝不对 verify 端点发起任何网络请求
    assert mock_verify_server.requests == []


def test_experimental_endpoint_404_when_disabled(
    monkeypatch, mock_verify_server, api_env,
):
    _patch_spec_config(monkeypatch, mock_verify_server, enabled=False)
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": "你好", "allow_external": True,
    })
    assert response.status_code == 404
    assert "QLH_SPEC_ENABLED" in response.json()["detail"]
    assert mock_verify_server.requests == []


def test_main_chat_path_rejects_speculative_execution_mode(monkeypatch, api_env):
    """主聊天路径的 execution_mode 校验集未被污染：仍只认 auto / task_graph。"""
    _patch_spec_config(monkeypatch, base_url="", enabled=True)
    response = api_env["client"].post("/api/chat", json={
        "message": "你好", "execution_mode": "speculative_assisted",
    })
    assert response.status_code == 422


def test_experimental_endpoint_scope_denied_returns_403(
    monkeypatch, mock_verify_server, api_env,
):
    _patch_spec_config(monkeypatch, mock_verify_server, data_scope="deny")
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": "敏感内容", "allow_external": True,
    })
    assert response.status_code == 403
    assert "数据作用域" in response.json()["detail"]
    assert mock_verify_server.requests == []

    # opt_in 档位未携带 allow_external 同样 403 且零外发
    _patch_spec_config(monkeypatch, mock_verify_server, data_scope="opt_in")
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": "敏感内容",
    })
    assert response.status_code == 403
    assert mock_verify_server.requests == []


def test_experimental_endpoint_runs_full_session(
    monkeypatch, mock_verify_server, api_env,
):
    tokenizer = ByteTokenizer()
    prompt = "解释一下投机解码"
    answer = "本地起草外部校验"
    _patch_spec_config(
        monkeypatch, mock_verify_server, gamma=4,
        max_new_tokens=len(tokenizer.encode(answer)),
    )
    mock_verify_server.behavior["target"] = (
        tokenizer.encode(prompt) + tokenizer.encode(answer)
    )
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": prompt,
        "allow_external": True,
        "max_new_tokens": len(tokenizer.encode(answer)),
        "gamma": 4,
        "seed": 11,
        "draft_hint": answer,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"]["execution_mode"] == "speculative_assisted"
    assert body["metrics"]["rounds"] >= 1
    assert body["metrics"]["acceptance_rate"] > 0.5
    assert body["content"]
    assert body["finish_reason"] in {"length", "stop", "max_rounds"}
    assert "specpass" not in json.dumps(body, ensure_ascii=False)

    # 会话指标进入 /api/status 的 last_session（零网络 IO）
    status = api_env["client"].get("/api/status").json()["speculative"]
    assert status["last_session"]["rounds"] == body["metrics"]["rounds"]


def test_experimental_endpoint_capability_error_returns_502(
    monkeypatch, mock_verify_server, api_env,
):
    _patch_spec_config(monkeypatch, mock_verify_server)
    mock_verify_server.behavior["mode"] = "no_logprobs"
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": "你好", "allow_external": True,
    })
    assert response.status_code == 502
    assert "logprobs" in response.json()["detail"]


def test_experimental_endpoint_rejects_other_execution_mode(monkeypatch, api_env):
    _patch_spec_config(monkeypatch, base_url="", enabled=True)
    response = api_env["client"].post("/api/experimental/speculative", json={
        "message": "你好", "allow_external": True, "execution_mode": "auto",
    })
    assert response.status_code == 422


# ================================================================
# top-k 截断下的接受判定精确性（renormalize_verify）
# ----------------------------------------------------------------
# HTTP verify 端返回的行只覆盖 top-k ∪ {草稿 token}，行和 = 保留质量 M < 1，
# 但草稿 token 的 q(t) 由 token_logprobs 精确给出。若把这样的行按分布归一化，
# 接受比值会被整体放大 1/M —— 草稿被"橡皮图章"，接受率随 top-k 参数漂移，
# 输出质量退化回 draft 模型。以下用例锁定修复后的语义。
# ================================================================

def _truncated_verify_rows(q_full, topk_idx, draft_token):
    """构造 HTTP verify 端语义的行：top-k 概率 + 草稿 token 的精确概率。"""
    row = np.zeros_like(q_full)
    row[topk_idx] = q_full[topk_idx]
    row[draft_token] = q_full[draft_token]      # token_logprobs 精确给出
    bonus = np.zeros_like(q_full)
    bonus[topk_idx] = q_full[topk_idx]
    return np.vstack([row, bonus])


def _measure_acceptance(q_full, p_full, top_k, draws, seed=3):
    vocab = q_full.shape[0]
    topk_idx = np.argsort(q_full)[-top_k:]
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(draws):
        token = int(rng.choice(vocab, p=p_full))
        outcome = verify_draft_tokens(
            [p_full],
            _truncated_verify_rows(q_full, topk_idx, token),
            [token],
            rng=rng,
            renormalize_verify=False,
        )
        accepted += outcome.accepted_count
    return accepted / draws


def _zipf_pair(vocab=512, alpha=1.1, seed=11):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocab + 1)
    q = 1.0 / ranks ** alpha
    q = q / q.sum()
    rng.shuffle(q)
    p = 1.0 / ranks ** (alpha * 0.8)
    p = p / p.sum()
    rng.shuffle(p)
    return q, p


def test_truncated_rows_acceptance_matches_exact_theory():
    """截断行 + renormalize_verify=False → 接受率等于理论值 Σ p·min(1, q/p)。"""
    q_full, p_full = _zipf_pair()
    exact = float(np.sum(p_full * np.minimum(1.0, q_full / p_full)))
    measured = _measure_acceptance(q_full, p_full, top_k=20, draws=30000)
    assert abs(measured - exact) < 0.01, (
        f"接受率 {measured:.4f} 偏离理论值 {exact:.4f}，"
        f"说明 verify 行被错误归一化（q 被放大 1/M）"
    )


def test_truncated_rows_acceptance_is_topk_independent():
    """接受判定只依赖 q(t) 的精确值，因此与 top-k 大小无关。"""
    q_full, p_full = _zipf_pair()
    rates = [
        _measure_acceptance(q_full, p_full, top_k=k, draws=20000)
        for k in (20, 64, 200)
    ]
    assert max(rates) - min(rates) < 0.01, (
        f"接受率随 top-k 漂移: {rates}；"
        f"漂移即意味着接受比值掺入了截断保留质量 M"
    )


def test_renormalized_truncated_rows_inflate_acceptance():
    """反向锁定：若归一化截断行，接受率会被显著抬高（本用例记录旧行为的危害）。"""
    q_full, p_full = _zipf_pair()
    exact = float(np.sum(p_full * np.minimum(1.0, q_full / p_full)))
    vocab = q_full.shape[0]
    topk_idx = np.argsort(q_full)[-20:]
    rng = np.random.default_rng(3)
    accepted = 0
    draws = 20000
    for _ in range(draws):
        token = int(rng.choice(vocab, p=p_full))
        outcome = verify_draft_tokens(
            [p_full],
            _truncated_verify_rows(q_full, topk_idx, token),
            [token],
            rng=rng,
            renormalize_verify=True,     # 旧行为
        )
        accepted += outcome.accepted_count
    inflated = accepted / draws
    assert inflated > exact + 0.05, (
        f"预期归一化会抬高接受率，实测 {inflated:.4f} vs 精确 {exact:.4f}"
    )


def test_run_speculative_chat_disables_verify_renormalization():
    """run_speculative_chat 必须以 renormalize_verify=False 构造会话。"""
    import inspect
    source = inspect.getsource(spec.run_speculative_chat)
    assert "renormalize_verify=False" in source, (
        "run_speculative_chat 未关闭 verify 行归一化，"
        "HTTP verify 路径的接受判定会失真"
    )
