"""SPC-REV 逆向投机解码核心（形态 B：置信驱动回退）。

语义界定（§8.5.4 定稿为形态 B，A 留作探索）：
- 传统投机：快模型(draft)产候选，权威大模型(verify)校验接受。
- 本模块（逆向/回退）：当**快模型置信度低于阈值**时，不再用投机接受路径，
  而是**直接回退到目标大模型**在该位置采样；置信足够时继续走标准投机接受。
  核心是快模型置信度作为"是否值得投机"的门（拒绝器 = 快模型的自信评估），
  避免在 draft 不可信处浪费一次 RTT。

分布等价：\b
- fallback 路径：token 直接由 verify（目标）分布采样/argmax。
- speculate 路径：token = 投机接受或修正，均由 verify 分布决定。
两条路径都满足"输出由 verify 决定"，因此整体对 verify 分布等价（本机用
fake verify 验证）。confidence 只改变"走哪条路"，不改变输出分布。

语义注记：反向门以"下一个位"置信度 gate 整轮，但每步只推进 **1 个 token**
（γ>1 时其余被接受的草稿会在下一步骤重入 draft/verify，不一次吞掉）——
保持逐步推进语义，避免门控粒度与推进粒度错位；分布仍由 verify 决定。

本模块刻意不接生产解码循环（只读核心 + 指标），与 SPC-CS/SPC-MEDS 同层。
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from speculative import (
    PROB_EPS,
    SpeculativeError,
    _prob_matrix,
    sample_from_probs,
    verify_draft_tokens,
)


def confidence_max(probs: Any) -> float:
    """draft 置信度：最后一个位置的最大概率（对下一个 token 的自信），
    one-hot=1.0、均匀分布=1/vocab。"""
    rows = _prob_matrix(probs, "reverse_draft_probs")
    if rows.shape[0] == 0:
        return 1.0
    tail = rows[-1]
    if float(np.sum(tail)) <= PROB_EPS:
        return 0.0
    return float(np.max(tail))


def should_fallback(confidence: float, threshold: float) -> bool:
    """置信度低于阈值 → 回退目标大模型（不投机）。"""
    return float(confidence) < float(threshold)


def reverse_step(
    *,
    context_ids: Sequence[int],
    draft_fn: Callable[[Sequence[int], int], Tuple[Any, Any]],
    verify_fn: Callable[[Sequence[int], Sequence[int]], Any],
    gamma: int,
    threshold: float,
    greedy: bool = False,
    rng: Any = None,
    renormalize_verify: bool = True,
    vocab: Optional[int] = None,
) -> Tuple[int, dict]:
    """一步逆向解码：返回 (token, metrics)。

    - confidence >= threshold → 走投机接受（draft γ 个 + verify 校验）。
    - confidence <  threshold → 回退：直接由 verify 在 context 采样一个 token。
    metrics 含 mode/confidence/next_action/fallback 计数语义。
    """
    rng = np.random.default_rng() if rng is None else rng
    draft_tokens, draft_probs = draft_fn(list(context_ids), max(0, int(gamma)))
    tokens = [int(t) for t in (draft_tokens or ())]
    if len(tokens) > max(0, int(gamma)):
        raise SpeculativeError("draft_fn 返回草稿数超过请求的 γ")

    confidence = confidence_max(draft_probs)
    if should_fallback(confidence, threshold):
        verify_rows, _ = _split_verify(verify_fn(list(context_ids), []))
        row = verify_rows[0]
        if greedy:
            if float(np.sum(row)) <= PROB_EPS:
                raise SpeculativeError("verify 在回退位置返回全零分布")
            token = int(np.argmax(row))
        else:
            token = sample_from_probs(row, rng)
        return token, {
            "mode": "fallback", "confidence": round(confidence, 6),
            "fallback": True,
        }

    verify_rows, meta = _split_verify(verify_fn(list(context_ids), tokens))
    outcome = verify_draft_tokens(
        draft_probs, verify_rows, tokens, rng=rng, greedy=greedy,
        renormalize_verify=renormalize_verify,
    )
    if outcome.emitted == 0:
        raise SpeculativeError("投机路径未产出任何 token")
    return int(outcome.tokens[0]), {
        "mode": "speculate", "confidence": round(confidence, 6),
        "accepted": outcome.accepted_count > 0,
        "accepted_count": outcome.accepted_count,
        "fallback": False,
    }


def _split_verify(raw: Any) -> Tuple[np.ndarray, dict]:
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        return _prob_matrix(raw[0], "reverse_verify", normalize=False), dict(raw[1])
    if isinstance(raw, np.ndarray):
        return _prob_matrix(raw, "reverse_verify", normalize=False), {}
    raise SpeculativeError("verify_fn 必须返回 (verify_probs, meta) 或 np.ndarray")
