"""Layer-handoff (relay) contract for llama.cpp -> llama.cpp (L -> L) pipelines.

This module freezes what the experimental PoC established
(``docs/跨框架层接力重启评估-2026-09-15.md`` §7.8-§7.11) so the main project can admit a
relay pipeline without re-deriving the rules:

* **trim** -- keep ``blk.K..blk.(N-1)`` and rewrite ``<arch>.block_count`` to ``N-K``;
  llama.cpp needs no source change (experiment §7.9);
* **indexing** -- in the trimmed instance ``layer i`` is the source model's
  ``layer i + K``, so the trim offset must travel with every handoff (§7.9);
* **boundary** -- the engine-specific last layer is never handed off; hidden crosses only
  for layer numbers ``0..n_layer-2`` (§7.3/§7.7);
* **acceptance** -- decided by per-token **argmax**, never by a cosine threshold;
  ``cosine``/``bitwise_equal`` are recorded as diagnostics only. The former apparent
  ``embd`` non-determinism was upstream issue #28963's caller-side ``pos`` overread and
  disappeared after supplying every M-RoPE position section (§7.11.3/§13);
* **fallback** -- any rejection or failure falls back to single-process llama.cpp.

Only stdlib is used and nothing is imported from the control plane, so this stays inside
the data-plane contract layer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

RELAY_ENGINE = "llama.cpp"
#: Strict acceptance criterion: every generated token must match by argmax.
RELAY_ACCEPTANCE = "per_token_argmax"
#: Diagnostic-only tolerant comparison retained for investigating rejected runs. It must
#: never admit Relay: the top-1/top-2 swaps observed before the #28963 position fix were
#: artifacts of an out-of-bounds read, and strict per-token acceptance is now attainable.
RELAY_ACCEPTANCE_TOLERANT = "top_k_tolerant_argmax"
#: Top-k used by the diagnostic comparator.
RELAY_TOLERANT_TOP_K = 2
CUT_LAYER_MIN = 1
RELAY_FALLBACK_STRATEGY = "single_process_llama_cpp"
XFRAME_SCOPE = "pc_explicit_only"
XFRAME_NETWORK_MODES = frozenset({"loopback", "ssh_tunnel"})
#: ★ 2026-09-21：`admit_relay_xframe` 准入通过时使用的 reason。
#: 与「拒绝」各理由并列，使调用方可直接用等值判断；`relay_fallback` 见此值**不启用**回退。
XFRAME_ADMITTED_REASON = "admitted"

_HIDDEN_WIDTH_BYTES = {"float32": 4, "float16": 2}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RelayModelIdentity:
    """Identity of one llama.cpp artifact taking part in a relay pipeline."""

    #: Canonical identity of the complete logical model. It is shared by full and cut GGUF.
    model_sha256: str
    architecture: str
    #: GGUF ``<arch>.block_count`` -- includes the multi-token-prediction layer.
    block_count: int
    n_embd: int
    #: GGUF ``<arch>.nextn_predict_layers``; normally 1 for qwen3.5-style models.
    nextn_predict_layers: int = 0
    #: Digest of this exact artifact. A cut GGUF normally differs from the upstream artifact.
    artifact_sha256: str = ""
    engine: str = RELAY_ENGINE

    @property
    def n_layer(self) -> int:
        """Layers participating in the main forward pass (``block_count`` minus nextn)."""
        return max(0, self.block_count - self.nextn_predict_layers)

    @property
    def last_handoff_layer(self) -> int:
        """Highest layer number whose *output* may cross the handoff boundary."""
        return self.n_layer - 2

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_layer"] = self.n_layer
        result["last_handoff_layer"] = self.last_handoff_layer
        return result


@dataclass(frozen=True)
class RelayTrimPlan:
    """Trim of the downstream artifact: drop the first ``trim_layers`` blocks."""

    trim_layers: int
    source: RelayModelIdentity

    @property
    def kept_block_count(self) -> int:
        return max(0, self.source.block_count - self.trim_layers)

    def local_to_source_layer(self, local_layer: int) -> int:
        """Map a trimmed-instance layer index back to the source model's layer index."""
        return _as_int(local_layer) + self.trim_layers

    def is_valid(self) -> bool:
        return CUT_LAYER_MIN <= self.trim_layers < self.source.n_layer

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = self.source.to_dict()
        result["kept_block_count"] = self.kept_block_count
        return result


@dataclass(frozen=True)
class RelayHiddenSpec:
    """Wire format of one handoff payload entry (one token position)."""

    n_embd: int
    dtype: str = "float32"

    @property
    def bytes_per_token(self) -> int:
        return max(0, self.n_embd) * _HIDDEN_WIDTH_BYTES.get(self.dtype, 0)

    @property
    def supported(self) -> bool:
        return self.n_embd > 0 and self.dtype in _HIDDEN_WIDTH_BYTES

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bytes_per_token"] = self.bytes_per_token
        result["supported"] = self.supported
        return result


@dataclass(frozen=True)
class RelayHandoff:
    """An admitted L -> L handoff: upstream computes ``0..cut-1``, hidden enters at ``cut``."""

    upstream: RelayModelIdentity
    downstream: RelayModelIdentity
    cut_layer: int
    trim: RelayTrimPlan
    hidden: RelayHiddenSpec

    @property
    def upstream_last_layer(self) -> int:
        """Source layer whose output crosses the boundary (must be <= ``n_layer - 2``)."""
        return self.cut_layer - 1

    @property
    def downstream_first_source_layer(self) -> int:
        return self.cut_layer

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["upstream"] = self.upstream.to_dict()
        result["downstream"] = self.downstream.to_dict()
        result["trim"] = self.trim.to_dict()
        result["hidden"] = self.hidden.to_dict()
        result["upstream_last_layer"] = self.upstream_last_layer
        result["downstream_first_source_layer"] = self.downstream_first_source_layer
        return result


@dataclass(frozen=True)
class RelayHandoffDecision:
    admitted: bool
    reason: str
    handoff: RelayHandoff | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"admitted": self.admitted, "reason": self.reason}
        result["handoff"] = self.handoff.to_dict() if self.handoff else None
        return result


def build_relay_handoff(
    upstream: RelayModelIdentity,
    downstream: RelayModelIdentity,
    cut_layer: int,
    *,
    hidden_dtype: str = "float32",
) -> RelayHandoffDecision:
    """Validate an L -> L handoff request and return the frozen contract when admitted.

    Rejections are explicit and each names the rule that failed, because the caller must
    fall back to the single-process path rather than guess.
    """
    if not isinstance(upstream, RelayModelIdentity) or not isinstance(downstream, RelayModelIdentity):
        return RelayHandoffDecision(False, "invalid_identity")
    if upstream.engine != RELAY_ENGINE or downstream.engine != RELAY_ENGINE:
        # PyTorch -> llama.cpp is CORE-RELAY-XFRAME-01 and is not admitted here.
        return RelayHandoffDecision(False, "cross_engine_not_admitted")
    if upstream.model_sha256 != downstream.model_sha256:
        return RelayHandoffDecision(False, "model_identity_mismatch")
    if upstream.architecture != downstream.architecture:
        return RelayHandoffDecision(False, "architecture_mismatch")
    if upstream.n_embd != downstream.n_embd:
        return RelayHandoffDecision(False, "shape_mismatch")

    n_layer = upstream.n_layer
    cut = _as_int(cut_layer)
    if n_layer < 2:
        return RelayHandoffDecision(False, "model_has_no_relay_range")
    if cut < CUT_LAYER_MIN or cut > n_layer - 1:
        # Upper bound keeps the handed-off layer <= n_layer-2 (never the last layer).
        return RelayHandoffDecision(False, "cut_layer_out_of_range")

    hidden = RelayHiddenSpec(n_embd=upstream.n_embd, dtype=hidden_dtype)
    if not hidden.supported:
        return RelayHandoffDecision(False, "unsupported_hidden_format")

    # A cut artifact is expected to differ in digest, but its GGUF layout must match the
    # declared boundary exactly.
    expected_block_count = upstream.block_count - cut
    if (
        downstream.nextn_predict_layers != upstream.nextn_predict_layers
        or downstream.block_count != expected_block_count
        or downstream.n_layer != upstream.n_layer - cut
    ):
        return RelayHandoffDecision(False, "trim_layout_mismatch")

    trim = RelayTrimPlan(trim_layers=cut, source=upstream)
    return RelayHandoffDecision(
        True,
        "admitted",
        RelayHandoff(
            upstream=upstream,
            downstream=downstream,
            cut_layer=cut,
            trim=trim,
            hidden=hidden,
        ),
    )


@dataclass(frozen=True)
class RelayComparisonVerdict:
    """Outcome of comparing a relay generation against the single-process baseline."""

    accepted: bool
    reason: str
    criterion: str
    matched_steps: int
    total_steps: int
    cosine: float | None = None
    bitwise_equal: bool | None = None
    diagnostics: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def judge_relay_generation(
    baseline_tokens: Sequence[int],
    relay_tokens: Sequence[int],
    *,
    cosine: float | None = None,
    bitwise_equal: bool | None = None,
) -> RelayComparisonVerdict:
    """Accept a relay run only when every generated token matches by argmax.

    ``cosine`` and ``bitwise_equal`` are recorded for the evidence trail but never gate the
    verdict. Strict token equality remains the only acceptance criterion.
    """
    baseline = [int(token) for token in baseline_tokens]
    relay = [int(token) for token in relay_tokens]
    total = min(len(baseline), len(relay))

    matched = 0
    for index in range(total):
        if baseline[index] != relay[index]:
            break
        matched += 1

    if total == 0:
        return RelayComparisonVerdict(
            False, "empty_sequence", RELAY_ACCEPTANCE, 0, 0, cosine, bitwise_equal
        )
    if len(baseline) != len(relay):
        return RelayComparisonVerdict(
            False,
            "length_mismatch",
            RELAY_ACCEPTANCE,
            matched,
            total,
            cosine,
            bitwise_equal,
            diagnostics=f"baseline={len(baseline)} relay={len(relay)}",
        )
    if matched != total:
        return RelayComparisonVerdict(
            False,
            "token_mismatch",
            RELAY_ACCEPTANCE,
            matched,
            total,
            cosine,
            bitwise_equal,
            diagnostics=f"first_divergence_step={matched}",
        )
    return RelayComparisonVerdict(
        True, "all_tokens_match", RELAY_ACCEPTANCE, matched, total, cosine, bitwise_equal
    )


def judge_relay_generation_tolerant(
    baseline_topk: Sequence[Sequence[int]],
    relay_tokens: Sequence[int],
    *,
    top_k: int = RELAY_TOLERANT_TOP_K,
    cosine: float | None = None,
    bitwise_equal: bool | None = None,
) -> RelayComparisonVerdict:
    """Diagnose whether relay tokens remain inside the baseline's top-k.

    This result is never a production or evidence admission. It is retained to explain a
    strict rejection; the pre-fix top-1/top-2 swaps came from #28963's position-array overread.

    ``baseline_topk[i]`` is the baseline's ranked candidate list at step ``i`` (best first);
    only its first ``top_k`` entries are consulted. ``cosine`` / ``bitwise_equal`` are recorded
    as diagnostics and never gate this diagnostic result.
    """
    baseline = [[int(candidate) for candidate in candidates] for candidates in baseline_topk]
    relay = [int(token) for token in relay_tokens]
    total = min(len(baseline), len(relay))

    if total == 0:
        return RelayComparisonVerdict(
            False, "empty_sequence", RELAY_ACCEPTANCE_TOLERANT, 0, 0, cosine, bitwise_equal
        )
    if len(baseline) != len(relay):
        return RelayComparisonVerdict(
            False,
            "length_mismatch",
            RELAY_ACCEPTANCE_TOLERANT,
            0,
            total,
            cosine,
            bitwise_equal,
            diagnostics=f"baseline={len(baseline)} relay={len(relay)}",
        )

    effective_k = max(1, _as_int(top_k, RELAY_TOLERANT_TOP_K))
    matched = 0
    for index in range(total):
        if relay[index] not in baseline[index][:effective_k]:
            return RelayComparisonVerdict(
                False,
                "token_outside_top_k",
                RELAY_ACCEPTANCE_TOLERANT,
                matched,
                total,
                cosine,
                bitwise_equal,
                diagnostics=f"first_divergence_step={matched} top_k={effective_k}",
            )
        matched += 1

    return RelayComparisonVerdict(
        True,
        "all_tokens_in_top_k",
        RELAY_ACCEPTANCE_TOLERANT,
        matched,
        total,
        cosine,
        bitwise_equal,
    )


@dataclass(frozen=True)
class RelayFallback:
    """What the caller must do when a handoff is rejected or fails mid-request."""

    engaged: bool
    reason: str
    strategy: str = RELAY_FALLBACK_STRATEGY
    note: str = "relay never replaces the single-process llama.cpp path implicitly"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def relay_fallback(reason: str) -> RelayFallback:
    """回退决策。

    ★ 2026-09-21：**准入通过时不启用回退** —— 此时调用方应当走 handoff，
    而不是退回单进程 llama.cpp。其余（拒绝）各理由一律 `engaged=True`。
    """
    if reason == XFRAME_ADMITTED_REASON:
        return RelayFallback(
            engaged=False,
            reason=reason,
            note="admitted: the relay handoff is the intended path for this request",
        )
    return RelayFallback(engaged=True, reason=str(reason))


@dataclass(frozen=True)
class RelayXFrameRequest:
    """Explicit gate for the unverified cross-process/cross-engine relay track."""

    upstream_engine: str
    downstream_engine: str
    network_mode: str = "loopback"
    sequence_length: int = 1
    temperature: float = 0.0
    top_p: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayXFrameEvidence:
    """CORE-RELAY-XFRAME-01 已**独立取得**的证据范围声明。

    加入动机（2026-09-19 收口）：`admit_relay_xframe` 原先无论正确性验证到什么程度都只回
    ``xframe_evidence_required``（「缺证据」），**无法区分**两种完全不同的状态：

    * ①「还没做验证」——需要继续投入；
    * ②「正确性已验证」——证据已够，按下面的准入判据决定路由。

    ## ★ 2026-09-21 纠正：准入判据**不包含「速度」**

    此前把「性能」窄化为「速度」（最优 333 ms/步 vs llama.cpp 原生整模 GPU
    13.5 ms/步，约 20×），据此判 `admits_production() == False`。**该推理不成立**：
    **性能必须同时看两类，不可只取其一作否决**。

    * **速度性能** —— D→L 约 20× 慢于整模 GPU。这个结论本身正确，但它的含义是
      **「Relay 的定位是能力组合、而非提速」**，只应影响**默认路由倾向**
      （能单机整模跑时优先单机），**不构成准入否决**；
    * **容量性能** —— 跨设备拆分让**单机装不下**的模型得以运行。这是 D→L 的
      **独有**优势，速度劣势**无法抵消**它。

    用户裁定（`dec-6d91cd100fecc798`，2026-09-21）：**速度性能与容量性能都算性能**
    ⇒ 原「性能不具优势 ⇒ 永不允许准入」不成立 ⇒ **改为按正确性证据准入**。

    见 `docs/跨框架层接力-项目报告.md` §7 与 `local_docs/CORE-RELAY-XFRAME-0*-*.json`。
    """

    correctness_verified: bool = False
    #: 逐 token 一致的用例数（每例一次完整贪心生成对照）。
    correctness_cases: int = 0
    #: 已验证的最长 prefill（token 数）。
    max_tested_prefill: int = 0
    prompt_distribution_verified: bool = False
    long_sequence_verified: bool = False
    weak_network_verified: bool = False
    protocol_consistency_verified: bool = False
    #: 速度性能判定 —— **仅作记录与路由倾向，不影响准入**。
    #: `not_advantageous` = 「正确但比整模 GPU 慢约 20×」⇒ 定位是**能力组合**而非提速。
    performance_verdict: str = "unknown"
    evidence_refs: tuple = ()

    def admits_production(self) -> bool:
        """准入判据：**正确性已验证**即准入。

        ★ 2026-09-21：**不以速度否决**（见类 docstring）。
        速度只决定路由**倾向**（能单机整模跑时优先单机）；**容量收益是 D→L 的独有优势**。
        二者同属「性能」，不可只取其一作为否决理由。
        """
        return self.correctness_verified

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["evidence_refs"] = list(self.evidence_refs)
        result["admits_production"] = self.admits_production()
        return result


@dataclass(frozen=True)
class RelayXFrameDecision:
    admitted: bool
    reason: str
    fallback: RelayFallback
    scope: str = XFRAME_SCOPE

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fallback"] = self.fallback.to_dict()
        return result


def admit_relay_xframe(
    request: RelayXFrameRequest | None,
    evidence: RelayXFrameEvidence | None = None,
) -> RelayXFrameDecision:
    """Policy gate for the D->L cross-process/cross-engine relay track.

    ★ 2026-09-21（用户裁定 `dec-6d91cd100fecc798`）：**准入判据改为「正确性已验证」**。

    此前传入 `evidence` 时回 ``xframe_correctness_verified_performance_not_advantageous``
    并恒 `admitted=False` —— 那是把「性能」窄化为「速度」（~20×）。但**速度与容量都属性能**：
    速度不占优只影响**默认路由倾向**（能单机整模跑时优先单机），
    而**容量收益（跨设备拆分让单机装不下的模型可跑）是 D→L 的独有优势**，不可被速度劣势抵消。

    ⇒ 证据齐备（`correctness_verified`）时回 ``admitted`` 且 **fallback 不启用**；
    证据不足仍回 ``xframe_evidence_required``（fail-closed，待验证）。
    非法请求（跨引擎 / 非 loopback·ssh_tunnel / 非确定性采样）**仍优先拒绝**，有证据也不能绕过硬门。
    """
    admitted = False
    if not isinstance(request, RelayXFrameRequest):
        reason = "invalid_xframe_request"
    elif request.upstream_engine != RELAY_ENGINE or request.downstream_engine != RELAY_ENGINE:
        reason = "cross_engine_not_admitted"
    elif str(request.network_mode).strip().lower() not in XFRAME_NETWORK_MODES:
        reason = "network_scope_not_admitted"
    else:
        try:
            sequence_length = int(request.sequence_length)
        except (TypeError, ValueError):
            sequence_length = 0
        if sequence_length < 1:
            reason = "sequence_length_invalid"
        else:
            try:
                temperature = float(request.temperature)
                top_p = float(request.top_p)
            except (TypeError, ValueError):
                temperature = math.nan
                top_p = math.nan
            if (
                not math.isfinite(temperature)
                or not math.isfinite(top_p)
                or temperature != 0.0
                or top_p != 1.0
            ):
                reason = "sampling_matrix_not_admitted"
            elif isinstance(evidence, RelayXFrameEvidence) and evidence.correctness_verified:
                # ★ 2026-09-21：正确性已独立验证 ⇒ **准入**。
                #   原先此处回 ``..._performance_not_advantageous`` 且恒 admitted=False，
                #   是把「性能」窄化为「速度」；速度不占优只影响**默认路由倾向**
                #   （见 RelayXFrameEvidence docstring），**不构成否决**。
                reason = XFRAME_ADMITTED_REASON
                admitted = True
            else:
                reason = "xframe_evidence_required"
    return RelayXFrameDecision(admitted, reason, relay_fallback(reason))
