"""SPC-MEDS 半自回归（Medusa 风格）候选树接受核心。

仅纯核心（numpy only、无网络、RNG 可注入），不接生产解码循环。
draft 每位置给出 ``heads`` 个候选（块预测），verify 沿候选取精确概率，
按 ``min(1, q/p)`` 采样接受，找最长可接受前缀；全候选被拒则在该位置给出
verify 修正。不接受树的后端把 ``heads`` 设为 1（每位置取 top-1）即退化为
与 :func:`verify_draft_tokens` 相同的扁平逐 token 语义。

分布等价：接受与修正都由 verify 分布决定（accept 判定多次尝试只会提高
接受率，不改变"输出由 verify 采样决定"的性质）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from speculative import PROB_EPS, SpeculativeError, _prob_matrix, sample_from_probs


@dataclass(frozen=True)
class MedusaHead:
    """draft 侧一个候选 token 及其概率（score >= 0）。"""

    token: int
    score: float


@dataclass
class MedusaPrefixOutcome:
    """一次候选树校验的结果：沿最长可接受前缀前进。"""

    accepted_tokens: list[int]
    accepted_count: int
    correction_token: Optional[int] = None
    bonus_token: Optional[int] = None
    accepted_all: bool = False

    @property
    def emitted(self) -> list[int]:
        """本轮实际产出的 token（接受前缀 + 修正/奖励）。"""
        out = list(self.accepted_tokens)
        if self.correction_token is not None:
            out.append(self.correction_token)
        elif self.bonus_token is not None:
            out.append(self.bonus_token)
        return out


def medusa_heads_from_distribution(
    probs: Any, heads: int, *, mode: str = "sample", rng: Any = None,
) -> list[list[MedusaHead]]:
    """从 draft 分布每位置取 ``heads`` 个候选（score>0 降序）。

    - ``mode="sample"``（默认，完整投机语义）：每位置**按 draft 分布采样**
      （无放回）候选；``heads=1`` 时等价于标准投机中 draft 每位置采一个
      token，是**分布等价的严格路径**。
    - ``mode="topk"``：取 draft 每位置 top-``heads``（Medusa 用 logits 风格）；
      穷举式 top-k 会钉住高概率 token（q/p=1 必接受），不承诺分布等价，
      仅作句"提高接受率"的块预测参考。

    只保留 ``score > PROB_EPS`` 的候选：score≈0 的 token 是 draft 从未
    采样到的项，放入候选会触发 ``verify 认可即接受`` 的除零特判。行会被
    归一化，使 ``score`` 可作为接受比率 p。
    """
    rows = _prob_matrix(probs, "medusa_draft_probs")
    count = max(1, int(heads))
    rng = np.random.default_rng() if rng is None else rng
    vocab = int(rows.shape[1])
    result: list[list[MedusaHead]] = []
    for row in rows:
        if float(np.sum(row)) <= PROB_EPS:
            result.append([])          # 该位置无候选 → 直接修正
            continue
        if mode == "topk":
            order = np.argsort(row)[::-1]
        elif mode == "sample":
            support = int(np.sum(row > PROB_EPS))
            size = max(0, min(count, support))
            if size == 0:
                result.append([])
                continue
            order = rng.choice(vocab, size=size, replace=False, p=row)
        else:
            raise SpeculativeError(f"unsupported medusa head mode: {mode}")
        candidates = [
            MedusaHead(int(token), float(row[token]))
            for token in order
            if float(row[token]) > PROB_EPS
        ][:count]
        candidates.sort(key=lambda c: c.score, reverse=True)
        result.append(candidates)
    return result


def _correction_token(row: np.ndarray, *, greedy: bool, rng: Any) -> int:
    """从 verify 行给出修正 token（贪心取 argmax，否则按分布采样）。"""
    if float(np.sum(row)) <= PROB_EPS:
        raise SpeculativeError("verify 在拒绝位置返回全零分布，无法给出修正 token")
    if greedy:
        return int(np.argmax(row))
    return sample_from_probs(row, rng)


def verify_medusa_prefix(
    heads: list[list[MedusaHead]],
    verify_probs: Any,
    *,
    greedy: bool = False,
    rng: Any = None,
    bonus: bool = True,
) -> MedusaPrefixOutcome:
    """沿候选树找最长可接受前缀。

    :param heads: 每位置的候选列表（按 score 降序更高效，顺序无影响）。
    :param verify_probs: verify 在 (context+候选前缀) 各位置的分布，
        shape 至少 [len(heads), vocab]；多一行时用于全接受的 bonus。
    """
    rows = _prob_matrix(verify_probs, "medusa_verify_probs", normalize=False)
    vocab = int(rows.shape[1])
    depth = len(heads)
    if rows.shape[0] < depth:
        raise SpeculativeError(
            f"verify 行数不足：候选树 {depth} 个位置至少需要 {depth} 行"
            f"（全接受时需 {depth + 1} 行），实际 {rows.shape[0]} 行。"
        )
    has_bonus_row = rows.shape[0] >= depth + 1
    rng = np.random.default_rng() if rng is None else rng

    accepted: list[int] = []
    for pos, candidates in enumerate(heads):
        if not candidates:
            # 某位置无候选：直接修正（等价于该位 rejected）
            correction = _correction_token(rows[pos], greedy=greedy, rng=rng)
            return MedusaPrefixOutcome(
                list(accepted), len(accepted), correction_token=correction,
            )
        row = rows[pos]
        picked: Optional[int] = None
        for cand in candidates:
            if not 0 <= cand.token < vocab:
                raise SpeculativeError(
                    f"候选 token id {cand.token} 超出词表范围 [0, {vocab})。"
                )
            p_draft = float(cand.score)
            p_verify = float(row[cand.token])
            if greedy:
                target = int(np.argmax(row)) if float(np.sum(row)) > PROB_EPS else -1
                ok = target == cand.token and p_verify > PROB_EPS
            else:
                if p_draft <= PROB_EPS:
                    ok = p_verify > PROB_EPS
                else:
                    ok = float(rng.random()) < min(1.0, p_verify / p_draft)
            if ok:
                picked = int(cand.token)
                break
        if picked is None:
            correction = _correction_token(row, greedy=greedy, rng=rng)
            return MedusaPrefixOutcome(
                list(accepted), len(accepted), correction_token=correction,
            )
        accepted.append(picked)

    bonus_token: Optional[int] = None
    if bonus and has_bonus_row:
        bonus_token = sample_from_probs(rows[depth], rng)
    return MedusaPrefixOutcome(
        list(accepted), len(accepted), bonus_token=bonus_token, accepted_all=True,
    )
