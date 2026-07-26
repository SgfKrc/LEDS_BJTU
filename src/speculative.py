"""
投机解码外部辅助 — 本地 draft + 外部 verify（路线 C-1，阶段 0-1 探索性 PoC）
==============================================================================
对应《张量并行外部辅助与混合拆分调研方案》§2.3 路线 C-1：

    本地 (draft: 小模型)                        外部 (verify: 大模型 TP 实例)
      |-- 生成 γ 个草稿 token（本地 KV 推进）------|
      |== 上行: token ids ≈ γ×4B + 会话头 ========>|  一次前向并行校验 γ 个位置
      |<== 下行: 接受数 k + 修正 token + logprob ==|  (外部 KV 推进 k+1)
      |-- 接受则一次前进 k+1 token；拒绝处回滚 ----|
      每轮跨 mesh 代价: ~1 KB + 1 RTT，摊薄到 k+1 个 token

本模块的语义承诺（标准投机采样性质）:
    **给定精确的 verify 分布时，输出分布严格等于 verify 模型的分布**，
    本地 draft 只影响速度不影响质量。因此 C-1 是"以本地小模型加速外部大模型"，
    而不是"用小模型近似大模型"。该性质由 verify_draft_tokens() 的
    接受-重采样规则保证，并由 tests/test_speculative.py 的 Monte-Carlo
    分布等价测试实测校验。
    ★ 作用域限定：严格等价是**纯核心**的性质。经 ExternalVerifyClient 时
      verify 分布只能由 OpenAI 兼容接口的 top-k logprobs 重建并逐行归一化，
      于是 p_verify = q/M（M = top-k 概率质量），**接受判定同样被放大 1/M**，
      端到端输出等于 verify 分布的 "top-k 截断 + 重归一化" 版本，
      偏差与 1-M 同阶。详见 docs/投机解码外部辅助实施说明.md §7.4。

模块边界（阶段 0-1 的 PoC 纪律）:
  1. 纯核心（verify_draft_tokens / residual_distribution / sample_from_probs）
     只依赖 numpy，无 torch、无网络、RNG 可注入 —— 这是唯一必须"可证明正确"
     的部分，也是唯一被 Monte-Carlo 校验的部分。
  2. SpeculativeSession 是状态机，draft 侧是**注入的可调用对象**
     `(context_ids, gamma) -> (tokens, probs)`，因此既能用假 draft 模型单测，
     也能在具备真实 PyTorch 运行时后直接接上 model_module 的解码循环。
  3. ExternalVerifyClient 走 /v1/completions + logprobs，传输层**组合复用**
     island_engine.IslandEngine（凭据脱敏 / URL 内嵌账号→BasicAuth / 错误分类），
     数据作用域门控复用 external_provider.ensure_external_scope_allowed
     —— 路线 B 的安全边界在这里原样适用：草稿 token 由用户内容派生，
     本路径确实把用户数据送出集群。
  4. **本 PoC 未接入生产解码循环**（model_module 的 generate / chat_stream）。
     容器内无法运行真实 PyTorch 解码，改写最核心路径的风险远大于 PoC 收益。
     接入点与剩余工作见 docs/投机解码外部辅助实施说明.md。

已知前提（§2.3 明确点名，未解决则收益不成立）:
  - 外部端必须返回 per-token logprobs（vLLM/SGLang `logprobs`；纯 chat 接口
    不够）。且 top_logprobs 的 key 必须可还原为 token id
    （vLLM `--return-tokens-as-token-ids`），否则残差分布无法构造。
  - 外部端最好为会话维护 KV（有状态）；否则每轮 verify 重付 prefill —— 这是
    §2.3 点名的**最大工程难点**。实践替代品是 vLLM 自动前缀缓存
    （`--enable-prefix-caching`）：KV 仍在外部端复用，但上行仍需重传整段
    context（本模块的 bytes_up 与 bytes_up_ideal_stateful 两个指标即量化此差距）。
  - draft 与 verify 必须共享同一 tokenizer / 词表（如 Qwen-1.8B + Qwen-7B）。

配置（config.py / 环境变量，QLH_SPEC_*，与 QLH_ISLAND_*/QLH_EXTERNAL_* 同风格）:
  QLH_SPEC_ENABLED / GAMMA / MAX_ROUNDS / MAX_NEW_TOKENS
  QLH_SPEC_VERIFY_BASE_URL / VERIFY_API_KEY / VERIFY_MODEL（留空回落 QLH_EXTERNAL_*）
  QLH_SPEC_TIMEOUT / CONNECT_TIMEOUT / TEMPERATURE / TOP_LOGPROBS / STATEFUL_VERIFY
  数据作用域**不另设开关**，共用 QLH_EXTERNAL_DATA_SCOPE（路线 B 的硬门控）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from external_provider import (
    ExternalScopeDeniedError,        # noqa: F401  （供调用方 except 复用）
    ExternalServiceError,
    _map_transport_error,
    ensure_external_scope_allowed,
    mask_external_url,
)
from island_engine import IslandEngine, IslandEngineError
from task_provider import ModelIdentity

logger = logging.getLogger(__name__)

# 概率比较阈值：小于此值视为 0（除零保护 / 残差判空）
PROB_EPS = 1e-12

# 通信量估算常量（§2.3 "~1KB + 1 RTT/轮"）
BYTES_PER_TOKEN_ID = 4          # token id 按 4 字节计
SESSION_HEADER_BYTES = 256      # 会话头（模型名 / 会话 id / HTTP 头）粗估


# ================================================================
# 错误分类 — 中文文案，面向"投机解码外部校验"
# ================================================================

class SpeculativeError(RuntimeError):
    """投机解码统一错误基类。"""


class SpeculativeConfigError(SpeculativeError):
    """投机解码配置缺失或非法。"""


class SpeculativeCapabilityError(SpeculativeError):
    """外部端点不满足投机解码前提（不返回可用 logprobs / token id）。"""


class SpeculativeVerifyError(SpeculativeError):
    """外部校验请求失败或响应无法解析。"""


# ================================================================
# 1. 投机采样纯核心 —— numpy only，无 torch、无网络、RNG 可注入
# ================================================================

def _prob_matrix(
    values: Any,
    name: str,
    *,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    把任意二维概率输入整理成规范化的 float64 概率矩阵。

    健壮性处理（全部是投机解码在真实后端上会遇到的情况）:
      - NaN / ±inf → 0（后端 logprob 解析出脏值时不至于污染整轮）
      - 负数 → 0（浮点误差 / 差值构造出的微小负值）
      - 每行独立归一化（后端返回的 top-k 概率天然不精确求和为 1）
      - 整行和为 0 的行保持全零：调用方按语义决定是"该 token 概率为 0"
        还是"无法采样"（后者才报错）

    normalize=False：只做清洗不做逐行缩放。用于 verify 行本身就是**真实概率**
    的场景（HTTP 端 token_logprobs 给出的是精确 q(t)，行和 = top-k 保留质量
    M < 1）。此时若强行归一化，接受判定的 q(t) 会被放大 1/M 倍——草稿被系统性
    "橡皮图章"，接受率虚高、输出分布偏离 verify。详见 verify_draft_tokens。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise SpeculativeError(
            f"{name} 必须是二维概率矩阵（每行一个位置的分布），"
            f"实际维度 {array.ndim}。"
        )
    array = np.where(np.isfinite(array), array, 0.0)
    array = np.clip(array, 0.0, None)
    if normalize:
        totals = array.sum(axis=1, keepdims=True)
        safe = np.where(totals > PROB_EPS, totals, 1.0)
        array = array / safe
    # 和为 0 的行归一化后仍为全零（除以 1.0），语义保持
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise SpeculativeError(
            f"{name} 行数不匹配：期望 {expected_rows} 行，实际 {array.shape[0]} 行。"
        )
    if expected_cols is not None and array.shape[1] != expected_cols:
        raise SpeculativeError(
            f"{name} 词表维度不匹配：期望 {expected_cols}，实际 {array.shape[1]}。"
        )
    return array


def sample_from_probs(row: np.ndarray, rng: Any) -> int:
    """
    从一行（已归一化的）概率分布中采样一个 token id。

    只依赖 rng.random() -> [0,1)，因此测试里可以注入返回定值的假 RNG，
    把"接受/拒绝/残差回落"三条分支全部逼出来。
    """
    total = float(np.sum(row))
    if total <= PROB_EPS:
        raise SpeculativeError(
            "无法从全零概率分布中采样：verify 端返回的该位置分布不可用。"
        )
    draw = float(rng.random()) * total
    cumulative = np.cumsum(row)
    index = int(np.searchsorted(cumulative, draw, side="right"))
    if index >= row.shape[0]:
        index = int(row.shape[0]) - 1
    # 边界保护：浮点累加可能落在零概率 token 上，向后找到第一个正概率 token
    while index < row.shape[0] - 1 and row[index] <= PROB_EPS:
        index += 1
    while index > 0 and row[index] <= PROB_EPS:
        index -= 1
    return index


def residual_distribution(
    draft_row: np.ndarray, verify_row: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    残差分布 max(0, p_verify - p_draft)，标准投机采样的拒绝重采样分布。

    Returns:
        (残差向量, 残差质量总和)。总和 ≤ PROB_EPS 时表示残差退化，
        调用方须回落到直接从 p_verify 采样（否则会除零 / 采不出 token）。
    """
    residual = np.clip(
        np.asarray(verify_row, dtype=np.float64)
        - np.asarray(draft_row, dtype=np.float64),
        0.0,
        None,
    )
    return residual, float(np.sum(residual))


@dataclass(frozen=True)
class PositionDiagnostic:
    """单个草稿位置的校验诊断（用于指标与调试，不参与采样）。"""
    index: int
    token: int
    p_draft: float
    p_verify: float
    accept_ratio: float
    random_draw: float          # 贪心模式为 -1.0（未消耗随机数）
    accepted: bool


@dataclass(frozen=True)
class VerifyOutcome:
    """一轮 draft-verify 的结果。"""
    accepted_count: int                     # k：被接受的草稿 token 数
    tokens: Tuple[int, ...]                 # 本轮实际产出的 token（k 个草稿 + 修正/奖励）
    correction_token: Optional[int]         # 拒绝处的修正 token（无拒绝则 None）
    bonus_token: Optional[int]              # γ 个全接受时的奖励 token（否则 None）
    rejected_count: int                     # γ-k：本地 KV 需回滚的长度
    residual_fallback: bool                 # 残差全零 → 回落到 verify 分布采样
    positions: Tuple[PositionDiagnostic, ...]

    @property
    def emitted(self) -> int:
        return len(self.tokens)


def verify_draft_tokens(
    draft_probs: Any,
    verify_probs: Any,
    draft_tokens: Sequence[int],
    rng: Any = None,
    *,
    greedy: bool = False,
    renormalize_verify: bool = True,
) -> VerifyOutcome:
    """
    标准投机采样（speculative sampling）核心 —— 本模块唯一必须证明正确的函数。

    规则（Leviathan et al. 2023 / Chen et al. 2023，§2.3 引用的 SLED/DSSD 同源）:
      1. 逐位置 i 以概率 min(1, p_verify[i][t_i] / p_draft[i][t_i]) 接受草稿 t_i；
      2. 首个被拒绝的位置：从归一化的残差分布 max(0, p_verify - p_draft) 重采样
         一个**修正 token**，并**立即停止**（其后的草稿全部作废，需回滚）；
      3. γ 个全部接受：额外从 p_verify[γ]（第 γ+1 行）采一个**奖励 token**。
    该规则使得产出 token 的边缘分布**严格等于 p_verify**，与 draft 质量无关。

    Args:
        draft_probs:  形状 (γ, V) —— draft 模型在每个草稿位置的分布
        verify_probs: 形状 (γ+1, V) —— verify 模型的分布；前 γ 行用于校验，
                      第 γ+1 行用于全接受时的奖励采样。只给 γ 行也可用
                      （此时全接受不产出奖励 token）。
        draft_tokens: 长度 γ 的草稿 token id
        rng:          任何提供 .random() -> [0,1) 的对象；默认
                      numpy.random.default_rng()。注入即可复现。
        greedy:       温度 0 贪心模式：接受当且仅当草稿 token == verify argmax，
                      修正/奖励 token 取 verify argmax，不消耗随机数。
        renormalize_verify:
                      verify 行是否逐行归一化。默认 True（行是"分布"，
                      行和被视为 1）。当 verify 行来自 OpenAI 兼容端点的
                      top-k logprobs 时必须传 False：那些行和 = 保留质量
                      M < 1，但草稿 token 的 q(t) 由 token_logprobs 精确给出，
                      归一化会把接受比值整体放大 1/M（实测 M=0.32 时接受率
                      虚高 1.55 倍，输出分布 TV 偏离 0.34）。传 False 后
                      **接受判定精确**，只剩修正 token 的残差采样受 top-k 截断
                      影响——这正是《实施说明》§7.4 记录的取舍。

    边界处理:
        γ=0            → 不校验，直接取奖励 token（若给了第 1 行）
        p_draft ≈ 0    → 比值退化为 1（q>0 接受）/ 0（q=0 拒绝），不做除法
        残差全零       → 回落到直接从 p_verify 采样（residual_fallback=True）
        NaN/inf/负数   → 在 _prob_matrix 中清洗为 0
        行和 ≠ 1       → renormalize_verify=True 时逐行归一化后再比较
    """
    tokens = [int(t) for t in (draft_tokens or ())]
    gamma = len(tokens)
    if rng is None:
        rng = np.random.default_rng()

    verify_rows = _prob_matrix(
        verify_probs, "verify_probs", normalize=bool(renormalize_verify),
    )
    vocab = int(verify_rows.shape[1])
    if gamma > 0:
        draft_rows = _prob_matrix(
            draft_probs, "draft_probs", expected_rows=gamma, expected_cols=vocab,
        )
    else:
        draft_rows = np.zeros((0, vocab), dtype=np.float64)
    if verify_rows.shape[0] < gamma:
        raise SpeculativeError(
            f"verify_probs 行数不足：草稿 {gamma} 个 token 至少需要 {gamma} 行"
            f"（全接受时需 {gamma + 1} 行），实际 {verify_rows.shape[0]} 行。"
        )
    has_bonus_row = verify_rows.shape[0] >= gamma + 1

    for token in tokens:
        if token < 0 or token >= vocab:
            raise SpeculativeError(
                f"草稿 token id {token} 超出词表范围 [0, {vocab})。"
            )

    diagnostics: List[PositionDiagnostic] = []
    accepted_count = 0
    correction_token: Optional[int] = None
    bonus_token: Optional[int] = None
    residual_fallback = False

    for index in range(gamma):
        token = tokens[index]
        draft_row = draft_rows[index]
        verify_row = verify_rows[index]
        p_draft = float(draft_row[token])
        p_verify = float(verify_row[token])

        if greedy:
            # 温度 0：verify 的贪心 token 就是"正确答案"，草稿与之相同才接受
            target = int(np.argmax(verify_row)) if float(np.sum(verify_row)) > PROB_EPS else -1
            accepted = target == token
            ratio = 1.0 if accepted else 0.0
            draw = -1.0
        else:
            if p_draft <= PROB_EPS:
                # 除零保护：draft 认为不可能的 token（理论上采不出来，
                # 但后端 top-k 截断/浮点误差会造成）——verify 认可就接受
                ratio = 1.0 if p_verify > PROB_EPS else 0.0
            else:
                ratio = min(1.0, p_verify / p_draft)
            draw = float(rng.random())
            accepted = draw < ratio

        diagnostics.append(PositionDiagnostic(
            index=index,
            token=token,
            p_draft=p_draft,
            p_verify=p_verify,
            accept_ratio=ratio,
            random_draw=draw,
            accepted=accepted,
        ))

        if accepted:
            accepted_count += 1
            continue

        # ---- 首个拒绝点：重采样修正 token 并终止本轮 ----
        if greedy:
            if float(np.sum(verify_row)) <= PROB_EPS:
                raise SpeculativeError(
                    f"verify 端在草稿位置 {index} 返回了全零分布，无法给出修正 token。"
                )
            correction_token = int(np.argmax(verify_row))
        else:
            residual, total = residual_distribution(draft_row, verify_row)
            if total <= PROB_EPS:
                # 残差退化（p_verify 处处不高于 p_draft）：回落到 verify 分布
                residual_fallback = True
                correction_token = sample_from_probs(verify_row, rng)
            else:
                correction_token = sample_from_probs(residual, rng)
        break

    if correction_token is None and has_bonus_row:
        bonus_row = verify_rows[gamma]
        if greedy:
            if float(np.sum(bonus_row)) <= PROB_EPS:
                raise SpeculativeError(
                    "verify 端在奖励位置返回了全零分布，无法给出奖励 token。"
                )
            bonus_token = int(np.argmax(bonus_row))
        else:
            bonus_token = sample_from_probs(bonus_row, rng)

    emitted: List[int] = tokens[:accepted_count]
    if correction_token is not None:
        emitted = emitted + [int(correction_token)]
    elif bonus_token is not None:
        emitted = emitted + [int(bonus_token)]

    return VerifyOutcome(
        accepted_count=accepted_count,
        tokens=tuple(int(t) for t in emitted),
        correction_token=correction_token,
        bonus_token=bonus_token,
        rejected_count=gamma - accepted_count,
        residual_fallback=residual_fallback,
        positions=tuple(diagnostics),
    )


# ================================================================
# 2. SpeculativeSession —— draft-verify 轮次状态机
# ================================================================

DraftCallable = Callable[[Sequence[int], int], Tuple[Sequence[int], Any]]
VerifyCallable = Callable[[Sequence[int], Sequence[int]], Any]


@dataclass
class SpeculativeResult:
    """一次投机解码会话的产出。"""
    tokens: List[int] = field(default_factory=list)
    finish_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    rounds: List[Dict[str, Any]] = field(default_factory=list)


class SpeculativeSession:
    """
    draft-verify 轮次状态机（§2.3 的"本地解码循环插入点"在 PoC 中的等价物）。

    每轮:
        1. 向 draft 侧要 γ 个草稿 token 及其分布（本地 KV 前进 γ）
        2. 把 context + 草稿发给 verify 侧，取回 γ+1 行分布
        3. 用 verify_draft_tokens 判定接受数 k 与修正/奖励 token
        4. 前进 k+1 个 token；被拒绝的 γ-k 个草稿作废（本地 KV 回滚 γ-k）

    draft 侧是注入的可调用对象 `(context_ids, gamma) -> (tokens, probs)`，
    因此本类不依赖任何模型运行时，可用假 draft 完整单测。
    """

    def __init__(
        self,
        prompt_ids: Sequence[int],
        draft_fn: DraftCallable,
        verify_fn: VerifyCallable,
        *,
        gamma: int = 4,
        max_new_tokens: int = 64,
        max_rounds: int = 0,
        stop_token_ids: Iterable[int] = (),
        cancel_event: Optional[threading.Event] = None,
        rng: Any = None,
        greedy: bool = False,
        stateful_verify: bool = False,
        renormalize_verify: bool = True,
    ):
        if not callable(draft_fn) or not callable(verify_fn):
            raise SpeculativeError("draft_fn 与 verify_fn 必须是可调用对象。")
        self._prompt_ids = [int(t) for t in (prompt_ids or ())]
        self._draft_fn = draft_fn
        self._verify_fn = verify_fn
        self._gamma = max(0, int(gamma))
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._max_rounds = max(0, int(max_rounds))
        self._stop_token_ids = {int(t) for t in (stop_token_ids or ())}
        self._cancel_event = cancel_event
        self._rng = rng if rng is not None else np.random.default_rng()
        self._greedy = bool(greedy)
        self._stateful_verify = bool(stateful_verify)
        # verify 行是否归一化。HTTP verify 端返回的是真实概率（行和 = top-k
        # 保留质量 M < 1），必须传 False 才能保证接受判定精确，见
        # verify_draft_tokens 的 renormalize_verify 说明。
        self._renormalize_verify = bool(renormalize_verify)

        self._context: List[int] = list(self._prompt_ids)
        self._generated: List[int] = []
        self._rounds_log: List[Dict[str, Any]] = []
        self._counters: Dict[str, float] = {
            "rounds": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "bonus_tokens": 0,
            "correction_tokens": 0,
            "residual_fallbacks": 0,
            "verify_calls": 0,
            "verify_latency_ms_total": 0.0,
            "verify_latency_ms_max": 0.0,
            "bytes_up": 0,
            "bytes_down": 0,
            "bytes_up_ideal_stateful": 0,
            "bytes_down_ideal_suffix": 0,
        }
        self._bytes_measured = True
        self._wall_ms = 0.0
        self._finish_reason = ""

    # ---- 只读视图 ----

    @property
    def context_ids(self) -> List[int]:
        return list(self._context)

    @property
    def generated_ids(self) -> List[int]:
        return list(self._generated)

    @property
    def finish_reason(self) -> str:
        return self._finish_reason

    # ---- 内部工具 ----

    def _cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    @staticmethod
    def _split_verify_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
        """verify_fn 可以只返回概率矩阵，也可以返回 (概率矩阵, meta)。"""
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[0], dict(result[1])
        return result, {}

    def _account_bytes(
        self, context_len: int, gamma_eff: int, meta: Dict[str, Any],
    ) -> Tuple[int, int]:
        """
        通信量记账（§2.3 "~1KB + 1 RTT/轮"的实测口径）。

        - bytes_up / bytes_down：verify 侧报告的真实字节数优先，否则按
          token id 数估算。无状态 verify 端每轮必须重传整段 context ——
          这正是 §2.3 点名的"每轮重付 prefill"在通信侧的体现。
        - bytes_up_ideal_stateful：外部端有会话级 KV 时的理想上行
          （只需 γ 个 token id + 会话头），与 bytes_up 的差值就是
          "无状态 KV"这一工程难点的量化代价。
        - bytes_down_ideal_suffix：自建 verify 端点只回 γ+1 个位置的 logprobs
          时的理想下行。标准 OpenAI completions 的 echo=true 会把**整段
          prompt 每个位置**的 logprobs 都回传，下行是 O(context×top_k) 而非
          O(γ×top_k)——实测下行远超 §2.3 的"~1KB/轮"，见实施说明"已知限制"。
        """
        ideal_up = SESSION_HEADER_BYTES + gamma_eff * BYTES_PER_TOKEN_ID
        if self._stateful_verify:
            estimated_up = ideal_up
        else:
            estimated_up = (
                SESSION_HEADER_BYTES
                + (context_len + gamma_eff) * BYTES_PER_TOKEN_ID
            )
        estimated_down = SESSION_HEADER_BYTES + (gamma_eff + 1) * BYTES_PER_TOKEN_ID
        up = meta.get("bytes_up")
        down = meta.get("bytes_down")
        if up is None or down is None:
            self._bytes_measured = False
        up = int(estimated_up if up is None else up)
        down = int(estimated_down if down is None else down)
        self._counters["bytes_up_ideal_stateful"] += ideal_up
        self._counters["bytes_down_ideal_suffix"] += int(
            meta.get("bytes_down_ideal", estimated_down)
        )
        return up, down

    # ---- 主循环 ----

    def run(self) -> SpeculativeResult:
        """驱动全部轮次直到停止条件；返回 token、结束原因与指标。"""
        started = time.perf_counter()
        while True:
            if self._cancelled():
                self._finish_reason = "cancelled"
                break
            if len(self._generated) >= self._max_new_tokens:
                self._finish_reason = "length"
                break
            if self._max_rounds and self._counters["rounds"] >= self._max_rounds:
                self._finish_reason = "max_rounds"
                break

            remaining = self._max_new_tokens - len(self._generated)
            # 每轮最多产出 k+1 个 token，因此 γ 上限为 remaining-1，
            # 保证不会越过 max_new_tokens（remaining=1 时 γ=0，只取奖励 token）
            gamma_eff = max(0, min(self._gamma, remaining - 1))

            draft_tokens, draft_probs = self._call_draft(gamma_eff)
            if self._cancelled():
                self._finish_reason = "cancelled"
                break

            verify_started = time.perf_counter()
            raw = self._verify_fn(list(self._context), list(draft_tokens))
            verify_probs, meta = self._split_verify_result(raw)
            latency_ms = (time.perf_counter() - verify_started) * 1000.0
            if "latency_ms" in meta:
                latency_ms = float(meta["latency_ms"])

            outcome = verify_draft_tokens(
                draft_probs,
                verify_probs,
                draft_tokens,
                rng=self._rng,
                greedy=self._greedy,
                renormalize_verify=self._renormalize_verify,
            )

            up_bytes, down_bytes = self._account_bytes(
                len(self._context), len(draft_tokens), meta,
            )
            self._counters["rounds"] += 1
            self._counters["drafted_tokens"] += len(draft_tokens)
            self._counters["accepted_tokens"] += outcome.accepted_count
            self._counters["rejected_tokens"] += outcome.rejected_count
            self._counters["verify_calls"] += 1
            self._counters["verify_latency_ms_total"] += latency_ms
            self._counters["verify_latency_ms_max"] = max(
                float(self._counters["verify_latency_ms_max"]), latency_ms,
            )
            self._counters["bytes_up"] += up_bytes
            self._counters["bytes_down"] += down_bytes
            if outcome.bonus_token is not None:
                self._counters["bonus_tokens"] += 1
            if outcome.correction_token is not None:
                self._counters["correction_tokens"] += 1
            if outcome.residual_fallback:
                self._counters["residual_fallbacks"] += 1

            round_entry = {
                "round": int(self._counters["rounds"]),
                "drafted": len(draft_tokens),
                "accepted": outcome.accepted_count,
                "rejected": outcome.rejected_count,
                # produced = 本轮投机采样产出的 token 数（k + 修正/奖励）
                # emitted  = 实际写入输出的 token 数（停止词/上限可能截断本轮）
                "produced": outcome.emitted,
                "emitted": 0,
                "bonus": outcome.bonus_token is not None,
                "correction": outcome.correction_token is not None,
                "verify_latency_ms": round(latency_ms, 2),
                "bytes_up": up_bytes,
                "bytes_down": down_bytes,
            }
            self._rounds_log.append(round_entry)

            if outcome.emitted == 0:
                # verify 端未给出任何 token（γ=0 且缺奖励行）——避免死循环
                self._finish_reason = "verify_no_token"
                break

            stopped = False
            for token in outcome.tokens:
                self._generated.append(int(token))
                self._context.append(int(token))
                round_entry["emitted"] += 1
                if int(token) in self._stop_token_ids:
                    self._finish_reason = "stop"
                    stopped = True
                    break
                if len(self._generated) >= self._max_new_tokens:
                    self._finish_reason = "length"
                    stopped = True
                    break
            if stopped:
                break

        self._wall_ms = (time.perf_counter() - started) * 1000.0
        if not self._finish_reason:
            self._finish_reason = "length"
        metrics = self.metrics()
        record_last_session_metrics(metrics)
        return SpeculativeResult(
            tokens=list(self._generated),
            finish_reason=self._finish_reason,
            metrics=metrics,
            rounds=list(self._rounds_log),
        )

    def _call_draft(self, gamma_eff: int) -> Tuple[List[int], Any]:
        """调用注入的 draft 可调用对象并校验其返回形状。"""
        if gamma_eff <= 0:
            return [], np.zeros((0, 0), dtype=np.float64)
        result = self._draft_fn(list(self._context), gamma_eff)
        if not isinstance(result, tuple) or len(result) != 2:
            raise SpeculativeError(
                "draft_fn 必须返回 (tokens, probs) 二元组"
                "（probs 形状为 [len(tokens), 词表大小]）。"
            )
        tokens, probs = result
        tokens = [int(t) for t in (tokens or ())]
        if len(tokens) > gamma_eff:
            raise SpeculativeError(
                f"draft_fn 返回了 {len(tokens)} 个草稿 token，超过请求的 {gamma_eff} 个。"
            )
        return tokens, probs

    # ---- 指标 ----

    def metrics(self) -> Dict[str, Any]:
        """
        指标 schema（同时供 /api/status 与实验端点返回）。

        接受率与每轮 token 是判断 C-1 是否划算的两个核心量；
        verify_share 给出"RTT 占比"，是 §2.3 收益上界公式里的分母来源。
        """
        rounds = int(self._counters["rounds"])
        drafted = int(self._counters["drafted_tokens"])
        accepted = int(self._counters["accepted_tokens"])
        emitted = len(self._generated)
        verify_calls = int(self._counters["verify_calls"])
        latency_total = float(self._counters["verify_latency_ms_total"])
        bytes_up = int(self._counters["bytes_up"])
        bytes_down = int(self._counters["bytes_down"])
        return {
            "gamma": self._gamma,
            "greedy": self._greedy,
            "stateful_verify": self._stateful_verify,
            "rounds": rounds,
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "rejected_tokens": int(self._counters["rejected_tokens"]),
            "bonus_tokens": int(self._counters["bonus_tokens"]),
            "correction_tokens": int(self._counters["correction_tokens"]),
            "residual_fallbacks": int(self._counters["residual_fallbacks"]),
            "emitted_tokens": emitted,
            "prompt_tokens": len(self._prompt_ids),
            # 接受率 = 被接受的草稿 / 全部草稿（§2.3 收益公式的"接受率"）
            "acceptance_rate": round(accepted / drafted, 4) if drafted else 0.0,
            # 每轮有效 token = 每次 RTT 换回多少 token（理想值 γ+1）
            "tokens_per_round": round(emitted / rounds, 4) if rounds else 0.0,
            "verify_calls": verify_calls,
            "verify_latency_ms_total": round(latency_total, 2),
            "verify_latency_ms_mean": (
                round(latency_total / verify_calls, 2) if verify_calls else 0.0
            ),
            "verify_latency_ms_max": round(
                float(self._counters["verify_latency_ms_max"]), 2,
            ),
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "bytes_per_round": (
                round((bytes_up + bytes_down) / rounds, 1) if rounds else 0.0
            ),
            "bytes_per_emitted_token": (
                round((bytes_up + bytes_down) / emitted, 1) if emitted else 0.0
            ),
            # 有状态 KV 时的理想上行；与 bytes_up 的比值 = 无状态重传的代价
            "bytes_up_ideal_stateful": int(
                self._counters["bytes_up_ideal_stateful"]
            ),
            # 自建 verify 端点只回 γ+1 行 logprobs 时的理想下行；
            # 与 bytes_down 的比值 = 标准 completions echo 全量回传的代价
            "bytes_down_ideal_suffix": int(
                self._counters["bytes_down_ideal_suffix"]
            ),
            "bytes_measured": bool(self._bytes_measured),
            "wall_ms": round(self._wall_ms, 2),
            # verify（含 RTT）占整段解码墙钟的比例：> 0.5 时本地 draft 基本白干
            "verify_share": (
                round(latency_total / self._wall_ms, 4)
                if self._wall_ms > 0 else 0.0
            ),
            "finish_reason": self._finish_reason,
        }


# ---- 最近一次会话指标快照（/api/status 用，零网络 IO）----

_last_metrics_lock = threading.Lock()
_last_metrics: Dict[str, Any] = {}


def record_last_session_metrics(metrics: Dict[str, Any]) -> None:
    global _last_metrics
    with _last_metrics_lock:
        _last_metrics = dict(metrics or {})
        _last_metrics["recorded_at"] = time.time()


def get_last_session_metrics() -> Dict[str, Any]:
    with _last_metrics_lock:
        return dict(_last_metrics)


def reset_last_session_metrics() -> None:
    global _last_metrics
    with _last_metrics_lock:
        _last_metrics = {}


# ================================================================
# 3. PoC 用 draft 侧组件（真实 draft 模型接入前的占位实现）
# ================================================================

class ByteTokenizer:
    """
    UTF-8 字节级 tokenizer（PoC 兜底）。

    真实部署里 draft 与 verify 必须共享模型 tokenizer；本类只在**本地没有
    可用 tokenizer**（如本容器无 PyTorch 运行时）时给实验端点一个自洽的
    token 空间，让整条 draft-verify 链路可以端到端跑通并产出真实文本。
    """

    EOS_ID = 256
    vocab_size = 257

    def encode(self, text: str) -> List[int]:
        return list((text or "").encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        payload = bytes(int(t) for t in (token_ids or ()) if 0 <= int(t) < 256)
        return payload.decode("utf-8", errors="replace")


class StubDraftModel:
    """
    PoC 假 draft 模型：`(context_ids, gamma) -> (tokens, probs)`。

    没有真实小模型时用它驱动状态机。可选 hint 序列让接受率可控——
    hint 命中时接受率高（模拟"draft 与 verify 高度一致"），
    不给 hint 时退化为均匀分布（模拟最差情况，接受率≈1/V）。
    投机采样的性质保证：**无论 draft 多差，产出分布仍等于 verify**。
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        seed: int = 0,
        hint_tokens: Optional[Sequence[int]] = None,
        hint_weight: float = 0.9,
    ):
        self.vocab_size = int(vocab_size)
        self._rng = np.random.default_rng(int(seed))
        self._hint = [int(t) for t in (hint_tokens or ())]
        self._hint_weight = float(min(max(hint_weight, 0.0), 1.0))
        self._base_len: Optional[int] = None

    def __call__(
        self, context_ids: Sequence[int], gamma: int,
    ) -> Tuple[List[int], np.ndarray]:
        if self._base_len is None:
            self._base_len = len(context_ids)
        offset = max(0, len(context_ids) - self._base_len)
        gamma = max(0, int(gamma))
        rows = np.zeros((gamma, self.vocab_size), dtype=np.float64)
        tokens: List[int] = []
        for step in range(gamma):
            row = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
            position = offset + step
            if self._hint and position < len(self._hint):
                hint_token = self._hint[position] % self.vocab_size
                row = np.full(
                    self.vocab_size,
                    (1.0 - self._hint_weight) / max(1, self.vocab_size - 1),
                    dtype=np.float64,
                )
                row[hint_token] = self._hint_weight
            row = row / float(row.sum())
            rows[step] = row
            tokens.append(sample_from_probs(row, self._rng))
        return tokens, rows


# ================================================================
# 4. ExternalVerifyClient —— /v1/completions + logprobs 外部校验
# ================================================================

def resolve_verify_config() -> Dict[str, Any]:
    """
    解析 verify 端配置：QLH_SPEC_VERIFY_* 优先，留空回落 QLH_EXTERNAL_*。

    凭据回落有额外约束：**只有当 base_url 本身也是回落来的**（即 verify 与
    路线 B 用同一个端点）才复用 EXTERNAL_API_KEY。否则显式配置了另一个
    verify 端点时，会把路线 B 的凭据发给另一台主机——那是凭据泄露。
    """
    import config as _cfg

    spec_url = str(getattr(_cfg, "SPEC_VERIFY_BASE_URL", "") or "").strip()
    external_url = str(getattr(_cfg, "EXTERNAL_BASE_URL", "") or "").strip()
    base_url = (spec_url or external_url).rstrip("/")
    inherited = not spec_url

    spec_key = str(getattr(_cfg, "SPEC_VERIFY_API_KEY", "") or "")
    api_key = spec_key or (
        str(getattr(_cfg, "EXTERNAL_API_KEY", "") or "") if inherited else ""
    )
    spec_model = str(getattr(_cfg, "SPEC_VERIFY_MODEL", "") or "").strip()
    model = spec_model or (
        str(getattr(_cfg, "EXTERNAL_MODEL", "") or "").strip() if inherited else ""
    )
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "inherited_from_external": inherited,
        "timeout": float(getattr(_cfg, "SPEC_TIMEOUT", 60)),
        "connect_timeout": float(getattr(_cfg, "SPEC_CONNECT_TIMEOUT", 5)),
        "top_logprobs": int(getattr(_cfg, "SPEC_TOP_LOGPROBS", 20)),
        "gamma": int(getattr(_cfg, "SPEC_GAMMA", 4)),
        "max_rounds": int(getattr(_cfg, "SPEC_MAX_ROUNDS", 64)),
        "max_new_tokens": int(getattr(_cfg, "SPEC_MAX_NEW_TOKENS", 128)),
        "temperature": float(getattr(_cfg, "SPEC_TEMPERATURE", 0.7)),
        "stateful_verify": bool(getattr(_cfg, "SPEC_STATEFUL_VERIFY", False)),
    }


def check_speculative_available() -> bool:
    """投机解码是否已启用且能解析出 verify 端点（不发起网络请求）。"""
    try:
        import config as _cfg
        if not getattr(_cfg, "SPEC_ENABLED", False):
            return False
        return bool(resolve_verify_config()["base_url"])
    except Exception:
        return False


def parse_token_key(key: Any) -> Optional[int]:
    """
    把 logprobs 的 key 还原成 token id。

    支持三种形态:
      - vLLM `--return-tokens-as-token-ids`: "token_id:12345"
      - 直接给 id 的自建 verify 端点: 12345 / "12345"
      - 其他（纯 token 文本）: 返回 None —— 无法还原，由调用方按能力探测报错
    """
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return int(key)
    text = str(key or "").strip()
    if not text:
        return None
    if text.startswith("token_id:"):
        text = text[len("token_id:"):].strip()
    try:
        return int(text)
    except ValueError:
        return None


class ExternalVerifyClient:
    """
    外部 verify 客户端：POST /v1/completions（echo + logprobs）批量校验草稿。

    请求语义（一次前向覆盖 γ 个校验位置 + 1 个奖励位置，对应 §2.3 图示）:
        prompt      = context_ids + draft_tokens   （token id 数组）
        echo        = true    → 返回 prompt 各位置的 logprobs
        max_tokens  = 1       → 顺带拿到第 γ+1 行（全接受时的奖励分布）
        logprobs    = top_k   → 每个位置的 top-k 备选（构造残差分布用）

    传输层复用 IslandEngine（凭据脱敏 / BasicAuth / 错误分类），
    数据作用域复用 external_provider.ensure_external_scope_allowed。

    诚实说明：**无状态 KV 时每轮重付 prefill**——本客户端每轮都把整段
    context 作为 prompt 重传，外部端若未开前缀缓存（vLLM
    `--enable-prefix-caching`）就会重算整段 prefill，这正是 §2.3 点名的
    最大工程难点，也是 bytes_up 与 bytes_up_ideal_stateful 分开统计的原因。
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        token_id_resolver: Optional[Callable[[str], Optional[int]]] = None,
        temperature: float = 1.0,
    ):
        self.vocab_size = int(vocab_size)
        self._resolver = token_id_resolver
        self._temperature = float(temperature)
        self._engine: Optional[IslandEngine] = None
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._capability: Dict[str, Any] = {}

    # ---- 连接 + 能力探测 ----

    def connect(self, *, allow_external: bool, probe: bool = True) -> None:
        """
        连接 verify 端点并做能力探测。

        能力探测失败时**大声报错**而不是静默降级：logprobs 是 §2.3 写明的
        前提条件，静默退回"纯 chat 接口"会让输出分布不再等于 verify 模型，
        那已经不是投机解码了。
        """
        config = resolve_verify_config()
        if not config["base_url"]:
            raise SpeculativeConfigError(
                "投机解码 verify 端点未配置：请设置 QLH_SPEC_VERIFY_BASE_URL"
                "（留空则回落 QLH_EXTERNAL_BASE_URL）后重试。"
            )
        masked = mask_external_url(config["base_url"])
        engine = IslandEngine()
        # ★ 数据作用域最后关口：健康检查也是对外请求，deny 档位一个包都不发
        ensure_external_scope_allowed(allow_external)
        try:
            engine.load_model(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model"],
                timeout=config["timeout"],
                connect_timeout=config["connect_timeout"],
            )
        except IslandEngineError as exc:
            raise SpeculativeVerifyError(
                str(_map_transport_error(exc, masked))
            ) from None
        with self._lock:
            old = self._engine
            self._engine = engine
            self._config = config
        if old is not None:
            try:
                old.unload()
            except Exception:
                pass
        logger.info(
            "投机解码 verify 端就绪: endpoint=%s, model=%s, top_logprobs=%s",
            masked, engine.model_name, config["top_logprobs"],
        )
        if probe:
            self.probe_capability(allow_external=allow_external)

    def probe_capability(self, *, allow_external: bool) -> Dict[str, Any]:
        """
        探测端点是否真的返回可用的 per-token logprobs 与可还原的 token id。

        探测请求只发一个固定的英文单词，不含任何用户内容。
        """
        engine = self._require_engine()
        payload = {
            "model": engine.model_name,
            "prompt": "ping",
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": int(self._config.get("top_logprobs", 20)),
        }
        # ★ 数据作用域最后关口
        ensure_external_scope_allowed(allow_external)
        body, _ = self._post(engine, "/v1/completions", payload)
        logprobs = self._extract_logprobs_block(body)
        token_logprobs = logprobs.get("token_logprobs") or []
        top_logprobs = logprobs.get("top_logprobs") or []
        if not token_logprobs or not any(
            value is not None for value in token_logprobs
        ):
            raise SpeculativeCapabilityError(
                "外部端点不满足投机解码前提：POST /v1/completions 未返回 "
                "per-token logprobs。请改用支持 logprobs 的推理服务"
                "（vLLM / SGLang，启动后以 logprobs 参数验证），"
                "纯 chat 接口无法用于投机校验。"
            )
        resolvable = False
        for entry in top_logprobs:
            if not isinstance(entry, dict):
                continue
            for key in entry.keys():
                if self._resolve_token_id(key) is not None:
                    resolvable = True
                    break
            if resolvable:
                break
        if not resolvable:
            raise SpeculativeCapabilityError(
                "外部端点不满足投机解码前提：top_logprobs 的 key 无法还原为 "
                "token id（拒绝采样需要按 id 构造残差分布）。请为 vLLM 加上 "
                "--return-tokens-as-token-ids 启动参数，或提供 "
                "token_id_resolver 把 token 文本映射回 id。"
            )
        capability = {
            "logprobs": True,
            "token_ids_resolvable": True,
            "top_logprobs": int(self._config.get("top_logprobs", 20)),
            "model": engine.model_name,
        }
        with self._lock:
            self._capability = capability
        return capability

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            try:
                engine.unload()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._engine is not None and self._engine.is_loaded

    @property
    def masked_base_url(self) -> str:
        with self._lock:
            if self._engine is not None:
                return self._engine.masked_base_url
        return mask_external_url(str(self._config.get("base_url", "")))

    @property
    def model_name(self) -> str:
        with self._lock:
            if self._engine is not None:
                return self._engine.model_name
        return str(self._config.get("model", ""))

    @property
    def capability(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._capability)

    def _require_engine(self) -> IslandEngine:
        with self._lock:
            engine = self._engine
        if engine is None or not engine.is_loaded:
            raise SpeculativeVerifyError(
                "投机解码 verify 端未连接，请先调用 connect()。"
            )
        return engine

    # ---- 校验请求 ----

    def verify(
        self,
        context_ids: Sequence[int],
        draft_tokens: Sequence[int],
        *,
        allow_external: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        校验一轮草稿：返回 (形状 [γ+1, V] 的概率矩阵, meta)。

        meta 含真实通信量 bytes_up / bytes_down 与 latency_ms，
        直接进 SpeculativeSession 的指标（§2.3 "~1KB/轮"的实测口径）。
        """
        engine = self._require_engine()
        context = [int(t) for t in (context_ids or ())]
        drafts = [int(t) for t in (draft_tokens or ())]
        payload = {
            "model": engine.model_name,
            "prompt": context + drafts,
            "max_tokens": 1,
            "temperature": self._temperature,
            "echo": True,
            "logprobs": int(self._config.get("top_logprobs", 20)),
        }
        started = time.perf_counter()
        # ★ 数据作用域最后关口：草稿 token 由用户内容派生，
        #   本路径确实把用户数据送出集群，路线 B 的门控在此原样适用。
        ensure_external_scope_allowed(allow_external)
        body, bytes_down = self._post(engine, "/v1/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0

        rows = self._rows_from_response(body, len(context), len(drafts), drafts)
        top_k = int(self._config.get("top_logprobs", 20))
        meta = {
            "bytes_up": len(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
            ),
            "bytes_down": int(bytes_down),
            # 自建 verify 端点只回 γ+1 行 top-k logprobs 时的理想下行；
            # 标准 completions 的 echo=true 会回传整段 prompt 的 logprobs，
            # 实测 bytes_down 会是它的 (context/γ) 倍（见实施说明"已知限制"）。
            "bytes_down_ideal": SESSION_HEADER_BYTES + (len(drafts) + 1) * top_k * 24,
            "latency_ms": latency_ms,
        }
        return rows, meta

    def as_verify_callable(self, *, allow_external: bool) -> VerifyCallable:
        """包装成 SpeculativeSession 需要的 `(context, drafts) -> (probs, meta)`。"""
        def _verify(context_ids: Sequence[int], draft_tokens: Sequence[int]):
            return self.verify(
                context_ids, draft_tokens, allow_external=allow_external,
            )
        return _verify

    # ---- HTTP + 解析 ----

    @staticmethod
    def _post(
        engine: IslandEngine, path: str, payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], int]:
        try:
            return engine.post_json(path, payload)
        except IslandEngineError as exc:
            raise SpeculativeVerifyError(
                str(_map_transport_error(exc, engine.masked_base_url))
            ) from None
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise SpeculativeVerifyError(
                f"投机解码 verify 请求失败：{engine.masked_base_url}"
                f"（{type(exc).__name__}）。"
            ) from None

    @staticmethod
    def _extract_logprobs_block(body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise SpeculativeVerifyError(
                "投机解码 verify 响应格式异常：缺少 choices[0]。"
            ) from None
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            raise SpeculativeCapabilityError(
                "外部端点不满足投机解码前提：响应中没有 logprobs 字段。"
                "投机校验要求外部端返回 per-token logprobs"
                "（vLLM/SGLang 的 /v1/completions?logprobs=N）。"
            )
        return logprobs

    def _resolve_token_id(self, key: Any) -> Optional[int]:
        token_id = parse_token_key(key)
        if token_id is None and self._resolver is not None:
            try:
                token_id = self._resolver(str(key))
            except Exception:
                token_id = None
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.vocab_size:
            return None
        return token_id

    def _rows_from_response(
        self,
        body: Dict[str, Any],
        context_len: int,
        gamma: int,
        draft_tokens: Sequence[int],
    ) -> np.ndarray:
        """
        把 echo+logprobs 响应转成 [γ+1, V] 概率矩阵。

        逐位置的取数规则:
          - 草稿位置 i（prompt 下标 context_len+i）：该位置的 token 就是
            draft_tokens[i]，其 verify 概率由 token_logprobs 精确给出
            （与 top_logprobs 的 key 格式无关，接受判定因此永远可靠）；
            其余候选由 top_logprobs 补齐，用于拒绝时的残差重采样。
          - 奖励位置（prompt 下标 context_len+γ，即第一个生成 token）：
            只有 top_logprobs，全部来自 top-k。
        top-k 截断说明：残差分布只在 top-k ∪ {草稿 token} 的支撑集上构造，
        这是 OpenAI 兼容接口的固有限制（自建 verify 端返回完整 logits 可消除）。
        """
        logprobs = self._extract_logprobs_block(body)
        token_logprobs = logprobs.get("token_logprobs") or []
        top_logprobs = logprobs.get("top_logprobs") or []
        needed = context_len + gamma
        if len(token_logprobs) <= needed or len(top_logprobs) <= needed:
            raise SpeculativeVerifyError(
                f"投机解码 verify 响应长度不足：需要 {needed + 1} 个位置的 "
                f"logprobs（context {context_len} + 草稿 {gamma} + 奖励 1），"
                f"实际 token_logprobs={len(token_logprobs)}、"
                f"top_logprobs={len(top_logprobs)}。请确认请求带 echo=true "
                f"且外部端点按 OpenAI completions 语义回显 prompt logprobs。"
            )

        rows = np.zeros((gamma + 1, self.vocab_size), dtype=np.float64)
        for offset in range(gamma + 1):
            index = context_len + offset
            entry = top_logprobs[index] if index < len(top_logprobs) else None
            if isinstance(entry, dict):
                for key, value in entry.items():
                    token_id = self._resolve_token_id(key)
                    if token_id is None:
                        continue
                    try:
                        rows[offset, token_id] = float(np.exp(float(value)))
                    except (TypeError, ValueError):
                        continue
            if offset < gamma:
                # 草稿 token 自身的精确概率（top-k 之外也一定有）
                exact = token_logprobs[index] if index < len(token_logprobs) else None
                if exact is not None:
                    try:
                        rows[offset, int(draft_tokens[offset])] = float(
                            np.exp(float(exact))
                        )
                    except (TypeError, ValueError, IndexError):
                        pass
            if float(np.sum(rows[offset])) <= PROB_EPS:
                raise SpeculativeVerifyError(
                    f"投机解码 verify 响应第 {offset} 个位置没有任何可用概率"
                    f"（logprobs 为空或 token id 无法还原）。"
                )
        return rows


# ================================================================
# 5. 端点指纹 / 模型身份
# ================================================================

def speculative_endpoint_fingerprint(
    draft_model_id: str = "", verify_model: Optional[str] = None,
) -> str:
    """指纹 = sha256(draft 模型 :: 脱敏 verify 端点 :: verify 模型名)。"""
    config = resolve_verify_config()
    masked = mask_external_url(config["base_url"])
    served = config["model"] if verify_model is None else verify_model
    raw = f"{draft_model_id}::{masked}::{served}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def speculative_model_identity(
    model_id: str = "speculative-draft-verify",
    draft_model_id: str = "",
    verify_model: Optional[str] = None,
) -> ModelIdentity:
    """
    投机解码组合体的模型身份。

    engine="speculative_assisted"：本质是"本地 draft + 外部 verify"的组合，
    没有单一本地 artifact，因此和 external_api 一样用端点指纹替代文件摘要。
    输出分布等于 verify 模型，故 verify 端模型名进指纹。
    """
    digest = speculative_endpoint_fingerprint(draft_model_id, verify_model)
    return ModelIdentity(
        model_id=model_id,
        engine="speculative_assisted",
        format="openai_api",
        revision=f"spec-{digest[:12]}",
        sha256=digest,
    )


# ================================================================
# 6. 高层编排 —— 供实验端点调用的一体化入口
# ================================================================

def run_speculative_chat(
    message: str,
    *,
    allow_external: bool,
    max_new_tokens: Optional[int] = None,
    gamma: Optional[int] = None,
    max_rounds: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: int = 0,
    draft_hint: str = "",
    tokenizer: Any = None,
    draft_fn: Optional[DraftCallable] = None,
    verify_client: Optional[ExternalVerifyClient] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """
    跑一次完整的 draft-verify 会话并返回文本 + 指标（实验端点用）。

    tokenizer/draft 缺省时使用 ByteTokenizer + StubDraftModel：
    容器内没有 PyTorch 运行时，PoC 用字节级 token 空间把链路跑通；
    产出分布仍严格等于 verify 端分布（这正是投机采样的性质）。
    """
    config = resolve_verify_config()
    resolved_gamma = int(gamma if gamma is not None else config["gamma"])
    resolved_gamma = max(1, min(resolved_gamma, 16))
    resolved_max_new = int(
        max_new_tokens if max_new_tokens is not None else config["max_new_tokens"]
    )
    resolved_max_rounds = int(
        max_rounds if max_rounds is not None else config["max_rounds"]
    )
    resolved_temperature = float(
        temperature if temperature is not None else config["temperature"]
    )

    tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 257))
    stop_ids = {int(getattr(tokenizer, "EOS_ID", -1))} - {-1}
    prompt_ids = list(tokenizer.encode(message or ""))

    owns_client = verify_client is None
    client = verify_client or ExternalVerifyClient(
        vocab_size, temperature=max(resolved_temperature, 0.0) or 1.0,
    )
    if not client.is_connected:
        client.connect(allow_external=allow_external)

    if draft_fn is None:
        hint_ids = list(tokenizer.encode(draft_hint)) if draft_hint else None
        draft_fn = StubDraftModel(vocab_size, seed=seed, hint_tokens=hint_ids)

    session = SpeculativeSession(
        prompt_ids,
        draft_fn,
        client.as_verify_callable(allow_external=allow_external),
        gamma=resolved_gamma,
        max_new_tokens=resolved_max_new,
        max_rounds=resolved_max_rounds,
        stop_token_ids=stop_ids,
        cancel_event=cancel_event,
        rng=np.random.default_rng(seed),
        greedy=resolved_temperature <= 0.0,
        stateful_verify=bool(config["stateful_verify"]),
        # ★ ExternalVerifyClient 返回的是**真实概率**（草稿 token 的 q(t) 由
        #   token_logprobs 精确给出，行和 = top-k 保留质量 M < 1）。这里必须
        #   关闭归一化，否则接受比值被整体放大 1/M，草稿被橡皮图章。
        renormalize_verify=False,
    )
    try:
        result = session.run()
    finally:
        # 先取端点标识再关连接：close() 之后 model_name 会回落到配置值
        # （自动发现的后端模型名就丢了），指标里会出现空模型名
        verify_endpoint = client.masked_base_url
        verify_model = client.model_name
        if owns_client:
            client.close()

    metrics = dict(result.metrics)
    metrics.update({
        "engine": "speculative_assisted",
        "execution_mode": "speculative_assisted",
        "provider": "speculative_draft_verify",
        "verify_base_url": verify_endpoint,
        "verify_model": verify_model,
        "draft_source": "stub" if isinstance(draft_fn, StubDraftModel) else "injected",
        "tokenizer": type(tokenizer).__name__,
        "endpoint_fingerprint": speculative_endpoint_fingerprint(),
    })
    # 用补全后的指标覆盖 session.run() 里记下的裸指标，
    # 让 /api/status 的 last_session 也带上端点/模型标识
    record_last_session_metrics(metrics)
    return {
        "content": tokenizer.decode(result.tokens),
        "tokens": list(result.tokens),
        "finish_reason": result.finish_reason,
        "metrics": metrics,
        "rounds": result.rounds,
    }


def speculative_status_section() -> Optional[Dict[str, Any]]:
    """
    /api/status 的 speculative 段（**零网络 IO**）。

    注意：这里刻意不做端点探活。路线 B 曾因在 /api/status 里同步探活把
    事件循环堵死（后来用 run_in_threadpool 修掉），投机解码段直接不引入
    任何出网请求，从源头上避免重蹈覆辙。
    """
    try:
        import config as _cfg
    except Exception:
        return None
    if not getattr(_cfg, "SPEC_ENABLED", False):
        return None
    config = resolve_verify_config()
    return {
        "enabled": True,
        "label": str(getattr(_cfg, "SPEC_LABEL", "投机解码外部辅助")),
        "gamma": config["gamma"],
        "max_rounds": config["max_rounds"],
        "max_new_tokens": config["max_new_tokens"],
        "verify_base_url": mask_external_url(config["base_url"]),
        "verify_model": config["model"],
        "inherited_from_external": config["inherited_from_external"],
        "top_logprobs": config["top_logprobs"],
        "stateful_verify": config["stateful_verify"],
        # 作用域不另设开关，与路线 B 共用（见 config.py QLH_SPEC_* 段注释）
        "data_scope": str(getattr(_cfg, "EXTERNAL_DATA_SCOPE", "opt_in")),
        "last_session": get_last_session_metrics(),
    }
