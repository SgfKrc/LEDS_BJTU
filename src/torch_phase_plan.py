"""Offline admission and transition contract for paired PyTorch phase plans.

This module never executes inference. A phase pair is admitted only when both
plans have matched runtime calibration, exact output evidence, a stable per-layer
KV owner/layout contract, and a verified whole-request fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Sequence

from src.torch_hetero_plan import HeterogeneousPlan


PHASE_PLAN_SCHEMA = "qlh.torch_phase_plan.v1"
PHASES = ("prefill", "decode")
FALLBACK_TARGETS = ("single_pytorch", "llama_cpp")
_ERROR_CODES = frozenset({
    "correctness_gate_failed",
    "device_unavailable",
    "execution_failed",
    "phase_timeout",
    "resource_limit",
})


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _finite(value: float, name: str, *, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phase_plan_fingerprint(plan: HeterogeneousPlan) -> str:
    if not isinstance(plan, HeterogeneousPlan) or not plan.admitted:
        raise ValueError("only an admitted heterogeneous plan can be fingerprinted")
    return _fingerprint(plan.to_dict())


@dataclass(frozen=True)
class LayerKVOwner:
    layer_index: int
    device_id: str
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("KV owner layer_index must be a non-negative integer")
        _text(self.device_id, "KV owner device_id")
        _text(self.layout_fingerprint, "KV layout_fingerprint")


def kv_contract_fingerprint(
    owners: Sequence[LayerKVOwner],
    *,
    total_layers: int,
) -> str:
    if isinstance(total_layers, bool) or not isinstance(total_layers, int) or total_layers < 1:
        raise ValueError("total_layers must be a positive integer")
    try:
        owner_rows = tuple(owners)
    except TypeError as exc:
        raise ValueError("KV ownership must be iterable") from exc
    if any(not isinstance(owner, LayerKVOwner) for owner in owner_rows):
        raise ValueError("KV ownership must contain LayerKVOwner values")
    ordered = tuple(sorted(owner_rows, key=lambda item: item.layer_index))
    if tuple(item.layer_index for item in ordered) != tuple(range(total_layers)):
        raise ValueError("KV owners must cover every layer exactly once")
    return _fingerprint({
        "schema_version": "qlh.torch_kv_contract.v1",
        "layers": [asdict(item) for item in ordered],
    })


@dataclass(frozen=True)
class PhaseAdmissionPolicy:
    max_runtime_cv: float
    max_schedule_error_ratio: float

    def __post_init__(self) -> None:
        _finite(self.max_runtime_cv, "max_runtime_cv")
        _finite(self.max_schedule_error_ratio, "max_schedule_error_ratio", maximum=1.0)


@dataclass(frozen=True)
class PhaseExecutionEvidence:
    phase: str
    model_fingerprint: str
    workload_fingerprint: str
    session_fingerprint: str
    plan_fingerprint: str
    reference_fingerprint: str
    kv_contract_fingerprint: str
    samples_ms: tuple[float, ...]
    warmup_consistent: bool
    instrumented: bool
    correctness_passed: bool
    argmax_exact: bool
    artifact_ref: str

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError("phase execution evidence phase is invalid")
        for name in (
            "model_fingerprint", "workload_fingerprint", "session_fingerprint",
            "plan_fingerprint", "reference_fingerprint",
            "kv_contract_fingerprint", "artifact_ref",
        ):
            _text(getattr(self, name), name)
        samples = tuple(self.samples_ms)
        if len(samples) < 3:
            raise ValueError("phase execution evidence requires at least three samples")
        for sample in samples:
            _finite(sample, "phase timing sample")
            if sample <= 0:
                raise ValueError("phase timing samples must be positive")
        object.__setattr__(self, "samples_ms", samples)
        for name in (
            "warmup_consistent", "instrumented", "correctness_passed", "argmax_exact",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class FallbackEvidence:
    target: str
    model_fingerprint: str
    session_fingerprint: str
    reference_fingerprint: str
    correctness_passed: bool
    argmax_exact: bool
    full_request_restart: bool
    artifact_ref: str

    def __post_init__(self) -> None:
        if self.target not in FALLBACK_TARGETS:
            raise ValueError("fallback target must be single_pytorch or llama_cpp")
        for name in (
            "model_fingerprint", "session_fingerprint", "reference_fingerprint",
            "artifact_ref",
        ):
            _text(getattr(self, name), name)
        for name in ("correctness_passed", "argmax_exact", "full_request_restart"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class PhaseCalibration:
    measured_median_ms: float
    runtime_cv: float
    schedule_error_ratio: float

    def __post_init__(self) -> None:
        _finite(self.measured_median_ms, "measured_median_ms")
        if self.measured_median_ms == 0:
            raise ValueError("measured_median_ms must be positive")
        _finite(self.runtime_cv, "runtime_cv")
        _finite(self.schedule_error_ratio, "schedule_error_ratio", maximum=1.0)


@dataclass(frozen=True)
class PhasePlanBundle:
    admitted: bool
    reason: str
    model_fingerprint: str
    session_fingerprint: str
    reference_fingerprint: str | None = None
    prefill_plan_fingerprint: str | None = None
    decode_plan_fingerprint: str | None = None
    kv_contract_fingerprint: str | None = None
    calibration: dict[str, PhaseCalibration] | None = None
    fallbacks: tuple[FallbackEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.admitted, bool):
            raise ValueError("admitted must be a boolean")
        _text(self.reason, "reason")
        _text(self.session_fingerprint, "session_fingerprint")
        if not self.admitted:
            return
        for name in (
            "model_fingerprint", "reference_fingerprint", "prefill_plan_fingerprint",
            "decode_plan_fingerprint", "kv_contract_fingerprint",
        ):
            _text(getattr(self, name), name)
        if not isinstance(self.calibration, dict) or set(self.calibration) != set(PHASES):
            raise ValueError("an admitted bundle requires prefill and decode calibration")
        if any(not isinstance(item, PhaseCalibration) for item in self.calibration.values()):
            raise ValueError("bundle calibration values must be PhaseCalibration")
        try:
            fallback_rows = tuple(self.fallbacks)
        except TypeError as exc:
            raise ValueError("bundle fallbacks must be iterable") from exc
        if not fallback_rows or any(not isinstance(item, FallbackEvidence) for item in fallback_rows):
            raise ValueError("an admitted bundle requires verified fallback evidence")
        targets = [item.target for item in fallback_rows]
        if len(targets) != len(set(targets)) or targets != sorted(
            targets, key=FALLBACK_TARGETS.index,
        ):
            raise ValueError("fallbacks must be unique and ordered by preference")
        if any(
            item.model_fingerprint != self.model_fingerprint
            or item.session_fingerprint != self.session_fingerprint
            or item.reference_fingerprint != self.reference_fingerprint
            or not item.correctness_passed
            or not item.argmax_exact
            or not item.full_request_restart
            for item in fallback_rows
        ):
            raise ValueError("bundle fallback evidence does not match the admitted request")
        object.__setattr__(self, "fallbacks", fallback_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_PLAN_SCHEMA,
            "admitted": self.admitted,
            "reason": self.reason,
            "model_fingerprint": self.model_fingerprint,
            "session_fingerprint": self.session_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "prefill_plan_fingerprint": self.prefill_plan_fingerprint,
            "decode_plan_fingerprint": self.decode_plan_fingerprint,
            "kv_contract_fingerprint": self.kv_contract_fingerprint,
            "calibration": {
                phase: asdict(item) for phase, item in (self.calibration or {}).items()
            },
            "fallbacks": [asdict(item) for item in self.fallbacks],
        }


def _calibrate(
    plan: HeterogeneousPlan,
    evidence: PhaseExecutionEvidence,
    *,
    expected_phase: str,
    model_fingerprint: str,
    session_fingerprint: str,
    kv_contract: str,
    reference_fingerprint: str | None,
    policy: PhaseAdmissionPolicy,
) -> tuple[PhaseCalibration | None, str | None]:
    if plan.phase != expected_phase or evidence.phase != expected_phase:
        return None, f"{expected_phase}_phase_identity_mismatch"
    if plan.model_fingerprint != model_fingerprint or evidence.model_fingerprint != model_fingerprint:
        return None, f"{expected_phase}_model_identity_mismatch"
    if evidence.workload_fingerprint != plan.workload_fingerprint:
        return None, f"{expected_phase}_workload_identity_mismatch"
    if evidence.session_fingerprint != session_fingerprint:
        return None, f"{expected_phase}_session_identity_mismatch"
    if evidence.plan_fingerprint != phase_plan_fingerprint(plan):
        return None, f"{expected_phase}_plan_evidence_mismatch"
    if evidence.kv_contract_fingerprint != kv_contract:
        return None, f"{expected_phase}_kv_evidence_mismatch"
    if reference_fingerprint is not None and evidence.reference_fingerprint != reference_fingerprint:
        return None, "phase_reference_identity_mismatch"
    if not evidence.warmup_consistent:
        return None, f"{expected_phase}_warmup_inconsistent"
    if evidence.instrumented:
        return None, f"{expected_phase}_measurement_instrumented"
    if not evidence.correctness_passed or not evidence.argmax_exact:
        return None, f"{expected_phase}_correctness_gate_failed"
    if plan.total_ms is None or not math.isfinite(plan.total_ms) or plan.total_ms <= 0:
        return None, f"{expected_phase}_schedule_estimate_missing"

    median = statistics.median(evidence.samples_ms)
    runtime_cv = statistics.pstdev(evidence.samples_ms) / median
    schedule_error = abs(median - plan.total_ms) / max(median, plan.total_ms)
    if runtime_cv > policy.max_runtime_cv:
        return None, f"{expected_phase}_runtime_variance_exceeds_policy"
    if schedule_error > policy.max_schedule_error_ratio:
        return None, f"{expected_phase}_schedule_calibration_exceeds_policy"
    return PhaseCalibration(
        measured_median_ms=round(median, 6),
        runtime_cv=round(runtime_cv, 6),
        schedule_error_ratio=round(schedule_error, 6),
    ), None


def admit_phase_pair(
    prefill_plan: HeterogeneousPlan,
    decode_plan: HeterogeneousPlan,
    *,
    session_fingerprint: str,
    total_layers: int,
    prefill_kv_owners: Sequence[LayerKVOwner],
    decode_kv_owners: Sequence[LayerKVOwner],
    prefill_evidence: PhaseExecutionEvidence,
    decode_evidence: PhaseExecutionEvidence,
    fallbacks: Sequence[FallbackEvidence],
    policy: PhaseAdmissionPolicy,
) -> PhasePlanBundle:
    """Admit phase-specific plans only with matched execution and cache evidence."""
    _text(session_fingerprint, "session_fingerprint")

    def reject(reason: str, model: str = "") -> PhasePlanBundle:
        return PhasePlanBundle(
            admitted=False,
            reason=reason,
            model_fingerprint=model,
            session_fingerprint=session_fingerprint,
        )

    if not isinstance(prefill_plan, HeterogeneousPlan) or not isinstance(
        decode_plan, HeterogeneousPlan,
    ):
        return reject("phase_plans_invalid")
    model_fingerprint = prefill_plan.model_fingerprint
    if not prefill_plan.admitted or not decode_plan.admitted:
        return reject("phase_plan_not_admitted", model_fingerprint)
    if not isinstance(prefill_evidence, PhaseExecutionEvidence) or not isinstance(
        decode_evidence, PhaseExecutionEvidence,
    ):
        return reject("phase_execution_evidence_invalid", model_fingerprint)
    if not isinstance(policy, PhaseAdmissionPolicy):
        return reject("phase_admission_policy_invalid", model_fingerprint)
    if prefill_plan.phase != "prefill" or decode_plan.phase != "decode":
        return reject("phase_plan_order_invalid", model_fingerprint)
    if not model_fingerprint or decode_plan.model_fingerprint != model_fingerprint:
        return reject("phase_model_identity_mismatch", model_fingerprint)
    try:
        prefill_owner_rows = tuple(prefill_kv_owners)
        decode_owner_rows = tuple(decode_kv_owners)
        prefill_kv_fingerprint = kv_contract_fingerprint(
            prefill_owner_rows, total_layers=total_layers,
        )
        decode_kv_fingerprint = kv_contract_fingerprint(
            decode_owner_rows, total_layers=total_layers,
        )
    except (TypeError, ValueError) as exc:
        return reject(f"kv_contract_invalid:{exc}", model_fingerprint)
    if prefill_kv_fingerprint != decode_kv_fingerprint:
        return reject("kv_contract_changed_between_phases", model_fingerprint)

    owner_by_layer = {owner.layer_index: owner.device_id for owner in prefill_owner_rows}
    for plan in (prefill_plan, decode_plan):
        attention_by_layer: dict[int, set[str]] = {}
        for placement in plan.placements:
            if placement.operator_id == "attention_core":
                attention_by_layer.setdefault(placement.layer_index, set()).add(
                    placement.device_id,
                )
        if set(attention_by_layer) != set(range(total_layers)):
            return reject("attention_plan_layer_coverage_incomplete", model_fingerprint)
        if any(
            attention_by_layer[layer] != {owner_by_layer[layer]}
            for layer in range(total_layers)
        ):
            return reject("kv_owner_not_backed_by_attention_plan", model_fingerprint)

    prefill_plan_fp = phase_plan_fingerprint(prefill_plan)
    decode_plan_fp = phase_plan_fingerprint(decode_plan)
    calibration: dict[str, PhaseCalibration] = {}
    reference_fingerprint: str | None = None
    for plan, evidence, phase in (
        (prefill_plan, prefill_evidence, "prefill"),
        (decode_plan, decode_evidence, "decode"),
    ):
        result, error = _calibrate(
            plan,
            evidence,
            expected_phase=phase,
            model_fingerprint=model_fingerprint,
            session_fingerprint=session_fingerprint,
            kv_contract=prefill_kv_fingerprint,
            reference_fingerprint=reference_fingerprint,
            policy=policy,
        )
        if error:
            return reject(error, model_fingerprint)
        if result is None:
            return reject(f"{phase}_calibration_missing", model_fingerprint)
        calibration[phase] = result
        if reference_fingerprint is None:
            reference_fingerprint = evidence.reference_fingerprint

    valid_fallbacks = [
        item for item in fallbacks
        if isinstance(item, FallbackEvidence)
        and item.model_fingerprint == model_fingerprint
        and item.session_fingerprint == session_fingerprint
        and item.reference_fingerprint == reference_fingerprint
        and item.correctness_passed
        and item.argmax_exact
        and item.full_request_restart
    ]
    valid_fallbacks.sort(key=lambda item: FALLBACK_TARGETS.index(item.target))
    if not valid_fallbacks:
        return reject("no_verified_full_request_fallback", model_fingerprint)
    return PhasePlanBundle(
        admitted=True,
        reason="paired_phase_plans_calibrated_and_safe_to_switch",
        model_fingerprint=model_fingerprint,
        session_fingerprint=session_fingerprint,
        reference_fingerprint=reference_fingerprint,
        prefill_plan_fingerprint=prefill_plan_fp,
        decode_plan_fingerprint=decode_plan_fp,
        kv_contract_fingerprint=prefill_kv_fingerprint,
        calibration=calibration,
        fallbacks=tuple(valid_fallbacks),
    )


@dataclass(frozen=True)
class PhaseTransitionDecision:
    allowed: bool
    state: str
    action: str
    phase: str | None = None
    plan_fingerprint: str | None = None
    fallback_target: str | None = None
    reason: str = ""


class PhaseSessionController:
    """Metadata-only state machine; callers remain responsible for execution."""

    def __init__(self, bundle: PhasePlanBundle) -> None:
        if not bundle.admitted:
            raise ValueError("cannot start a session with a rejected phase-plan bundle")
        self.bundle = bundle
        self.state = "ready"
        self._output_published = False
        self._fallback_target: str | None = None
        self._fallback_index = -1

    def _denied(self, reason: str) -> PhaseTransitionDecision:
        return PhaseTransitionDecision(False, self.state, "none", reason=reason)

    def _session_mismatch(self, session_fingerprint: str) -> bool:
        return session_fingerprint != self.bundle.session_fingerprint

    def start_prefill(self, session_fingerprint: str) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "ready":
            return self._denied("prefill_not_allowed_in_current_state")
        self.state = "prefill_running"
        return PhaseTransitionDecision(
            True, self.state, "run_phase", "prefill", self.bundle.prefill_plan_fingerprint,
        )

    def complete_prefill(
        self,
        session_fingerprint: str,
        kv_contract: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "prefill_running":
            return self._denied("prefill_completion_out_of_order")
        if kv_contract != self.bundle.kv_contract_fingerprint:
            return self._fallback("kv_contract_mismatch")
        self.state = "decode_ready"
        return PhaseTransitionDecision(True, self.state, "phase_switch_ready", "decode")

    def start_decode(
        self,
        session_fingerprint: str,
        kv_contract: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "decode_ready":
            return self._denied("decode_not_allowed_before_prefill")
        if kv_contract != self.bundle.kv_contract_fingerprint:
            return self._fallback("kv_contract_mismatch")
        self.state = "decode_running"
        return PhaseTransitionDecision(
            True, self.state, "run_phase", "decode", self.bundle.decode_plan_fingerprint,
        )

    def note_output_published(self, session_fingerprint: str) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state not in {"decode_running", "fallback_running"}:
            return self._denied("output_publication_out_of_order")
        self._output_published = True
        phase = "decode" if self.state == "decode_running" else "fallback"
        return PhaseTransitionDecision(True, self.state, "output_recorded", phase)

    def complete_decode(self, session_fingerprint: str) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "decode_running":
            return self._denied("decode_completion_out_of_order")
        self.state = "completed"
        return PhaseTransitionDecision(True, self.state, "complete", "decode")

    def fail(
        self,
        session_fingerprint: str,
        *,
        phase: str,
        error_code: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        expected_state = "prefill_running" if phase == "prefill" else "decode_running"
        if phase not in PHASES or self.state != expected_state:
            return self._denied("failure_out_of_order")
        if error_code not in _ERROR_CODES:
            return self._denied("unrecognized_error_code")
        return self._fallback(error_code)

    def confirm_fallback_completed(
        self,
        session_fingerprint: str,
        target: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "fallback_running" or target != self._fallback_target:
            return self._denied("fallback_completion_mismatch")
        self.state = "completed"
        return PhaseTransitionDecision(
            True, self.state, "fallback_completed", fallback_target=target,
        )

    def start_fallback(
        self,
        session_fingerprint: str,
        target: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "fallback_ready" or target != self._fallback_target:
            return self._denied("fallback_start_mismatch")
        self.state = "fallback_running"
        return PhaseTransitionDecision(
            True, self.state, "run_fallback", fallback_target=target,
        )

    def fail_fallback(
        self,
        session_fingerprint: str,
        *,
        target: str,
        error_code: str,
    ) -> PhaseTransitionDecision:
        if self._session_mismatch(session_fingerprint):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="session_identity_mismatch")
        if self.state != "fallback_running" or target != self._fallback_target:
            return self._denied("fallback_failure_mismatch")
        if error_code not in _ERROR_CODES:
            return self._denied("unrecognized_error_code")
        if self._output_published:
            self.state = "aborted"
            return PhaseTransitionDecision(
                False, self.state, "abort", reason="partial_output_already_published",
            )
        self._fallback_index += 1
        if self._fallback_index >= len(self.bundle.fallbacks):
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="no_fallback_remaining")
        self._fallback_target = self.bundle.fallbacks[self._fallback_index].target
        self.state = "fallback_ready"
        return PhaseTransitionDecision(
            True,
            self.state,
            "restart_full_request",
            fallback_target=self._fallback_target,
            reason=error_code,
        )

    def _fallback(self, reason: str) -> PhaseTransitionDecision:
        if self._output_published:
            self.state = "aborted"
            return PhaseTransitionDecision(
                False, self.state, "abort", reason="partial_output_already_published",
            )
        if not self.bundle.fallbacks:
            self.state = "aborted"
            return PhaseTransitionDecision(False, self.state, "abort", reason="fallback_unavailable")
        self._fallback_index = 0
        self._fallback_target = self.bundle.fallbacks[self._fallback_index].target
        self.state = "fallback_ready"
        return PhaseTransitionDecision(
            True,
            self.state,
            "restart_full_request",
            fallback_target=self._fallback_target,
            reason=reason,
        )


__all__ = [
    "FALLBACK_TARGETS",
    "FallbackEvidence",
    "LayerKVOwner",
    "PHASE_PLAN_SCHEMA",
    "PhaseAdmissionPolicy",
    "PhaseCalibration",
    "PhaseExecutionEvidence",
    "PhasePlanBundle",
    "PhaseSessionController",
    "PhaseTransitionDecision",
    "admit_phase_pair",
    "kv_contract_fingerprint",
    "phase_plan_fingerprint",
]
