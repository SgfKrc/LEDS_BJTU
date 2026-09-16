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
* **acceptance** -- decided by per-token **argmax**, never by a cosine threshold: the
  ``llama_batch.embd`` path is not bit-reproducible even inside a single process
  (§7.11.2), so ``cosine``/``bitwise_equal`` are recorded as diagnostics only;
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
#: Tolerant acceptance criterion for the embedding-input (``embd``) handoff path.
#:
#: Measured 2026-09-16 (``llama-relay-gen``, Qwen3.5-2B, L -> L, 141-step generation): at every
#: observed divergence (6/6 runs, steps 31-67) the two logits vectors agreed at cosine
#: 0.984-0.999 with a 4-5/5 top-5 overlap, and the two argmaxes were each ranked **2nd** in the
#: other vector -- every divergence was a top-1 <-> top-2 swap caused by the ``embd`` path's
#: known non-determinism, which autoregression then amplifies. A strict per-token criterion is
#: therefore unattainable on that path (even for L -> L, which fails by step 46); the relay
#: argmax merely has to fall inside the baseline's top-k (k = 2 covers all measured cases).
RELAY_ACCEPTANCE_TOLERANT = "top_k_tolerant_argmax"
#: Top-k used by :data:`RELAY_ACCEPTANCE_TOLERANT`.
RELAY_TOLERANT_TOP_K = 2
CUT_LAYER_MIN = 1
RELAY_FALLBACK_STRATEGY = "single_process_llama_cpp"
XFRAME_SCOPE = "pc_explicit_only"
XFRAME_NETWORK_MODES = frozenset({"loopback", "ssh_tunnel"})

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
    verdict: the embedding-input path is not bit-reproducible (§7.11.2), and a low cosine
    with a matching argmax is the expected shape of a healthy relay run.
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
    """Accept a relay run when every relay token falls inside the baseline's top-k.

    Use this instead of :func:`judge_relay_generation` on the ``embd`` handoff path: that path
    is not bit-reproducible inside a single process, and its observed divergence is always a
    top-1 <-> top-2 swap (see :data:`RELAY_ACCEPTANCE_TOLERANT`). Autoregression amplifies one
    swap into a different tail, so a strict per-token comparison fails even on healthy L -> L
    relays (measured: fails by step 46 of 141). Membership in the baseline's top-k is the
    strongest criterion the path actually satisfies.

    ``baseline_topk[i]`` is the baseline's ranked candidate list at step ``i`` (best first);
    only its first ``top_k`` entries are consulted. ``cosine`` / ``bitwise_equal`` are recorded
    for the evidence trail and never gate the verdict.
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
class RelayXFrameDecision:
    admitted: bool
    reason: str
    fallback: RelayFallback
    scope: str = XFRAME_SCOPE

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fallback"] = self.fallback.to_dict()
        return result


def admit_relay_xframe(request: RelayXFrameRequest | None) -> RelayXFrameDecision:
    """Fail closed until D->L and network handoff have independent evidence.

    This is deliberately a policy gate, not a claim that hidden-state transport is
    production-ready.  Only deterministic PC experiments over loopback/SSH are
    eligible; the normal caller must still provide a separate acceptance report.
    """
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
            else:
                reason = "xframe_evidence_required"
    return RelayXFrameDecision(False, reason, relay_fallback(reason))
