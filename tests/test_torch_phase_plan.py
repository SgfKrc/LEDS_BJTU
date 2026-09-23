from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.torch_hetero_plan import HeterogeneousPlan, NodePlacement  # noqa: E402
from src.torch_phase_plan import (  # noqa: E402
    FALLBACK_TARGETS,
    FallbackEvidence,
    LayerKVOwner,
    PhaseAdmissionPolicy,
    PhaseExecutionEvidence,
    PhasePlanBundle,
    PhaseSessionController,
    admit_phase_pair,
    kv_contract_fingerprint,
    phase_plan_fingerprint,
)


MODEL_FP = "model:sha256:phase-test"
SESSION_FP = "session:sha256:phase-test"
REFERENCE_FP = "reference:sha256:phase-test"
POLICY = PhaseAdmissionPolicy(max_runtime_cv=0.1, max_schedule_error_ratio=0.1)
OWNERS = (
    LayerKVOwner(0, "cuda", "layout:layer-0"),
    LayerKVOwner(1, "cuda", "layout:layer-1"),
)


def _plan(phase: str, workload: str, devices: tuple[str, str] = ("cuda", "cuda")):
    return HeterogeneousPlan(
        admitted=True,
        reason="synthetic_admitted_plan",
        model_fingerprint=MODEL_FP,
        workload_fingerprint=workload,
        phase=phase,
        placements=tuple(
            NodePlacement(
                node_id=f"attention-{layer}",
                operator_id="attention_core",
                layer_index=layer,
                device_id=device,
                implementation_id="pytorch.eager.cuda.attention_core",
                measured_median_ms=4.0 if phase == "prefill" else 1.0,
                cost_source_ref=f"costs/{phase}-{layer}.json",
                correctness_evidence_ref=None,
            )
            for layer, device in enumerate(devices)
        ),
        compute_work_ms=8.0 if phase == "prefill" else 2.0,
        transfer_work_ms=0.0,
        total_ms=8.0 if phase == "prefill" else 2.0,
        states_explored=4,
        schedule_model="synthetic_test_schedule",
    )


def _evidence(plan, owners=OWNERS, **changes):
    phase = plan.phase
    expected_ms = plan.total_ms
    payload = {
        "phase": phase,
        "model_fingerprint": plan.model_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "session_fingerprint": SESSION_FP,
        "plan_fingerprint": phase_plan_fingerprint(plan),
        "reference_fingerprint": REFERENCE_FP,
        "kv_contract_fingerprint": kv_contract_fingerprint(owners, total_layers=2),
        "samples_ms": (expected_ms - 0.05, expected_ms, expected_ms + 0.05),
        "warmup_consistent": True,
        "instrumented": False,
        "correctness_passed": True,
        "argmax_exact": True,
        "artifact_ref": f"evidence/{phase}.json",
    }
    payload.update(changes)
    return PhaseExecutionEvidence(**payload)


def _fallback(target: str, **changes):
    payload = {
        "target": target,
        "model_fingerprint": MODEL_FP,
        "session_fingerprint": SESSION_FP,
        "reference_fingerprint": REFERENCE_FP,
        "correctness_passed": True,
        "argmax_exact": True,
        "full_request_restart": True,
        "artifact_ref": f"evidence/fallback-{target}.json",
    }
    payload.update(changes)
    return FallbackEvidence(**payload)


def _admit(
    *,
    prefill_plan=None,
    decode_plan=None,
    prefill_owners=OWNERS,
    decode_owners=OWNERS,
    prefill_evidence=None,
    decode_evidence=None,
    fallbacks=None,
):
    prefill_plan = prefill_plan or _plan("prefill", "workload:prefill:64")
    decode_plan = decode_plan or _plan("decode", "workload:decode:1")
    prefill_evidence = prefill_evidence or _evidence(prefill_plan, prefill_owners)
    decode_evidence = decode_evidence or _evidence(decode_plan, decode_owners)
    fallbacks = fallbacks if fallbacks is not None else (
        _fallback("llama_cpp"), _fallback("single_pytorch"),
    )
    bundle = admit_phase_pair(
        prefill_plan,
        decode_plan,
        session_fingerprint=SESSION_FP,
        total_layers=2,
        prefill_kv_owners=prefill_owners,
        decode_kv_owners=decode_owners,
        prefill_evidence=prefill_evidence,
        decode_evidence=decode_evidence,
        fallbacks=fallbacks,
        policy=POLICY,
    )
    return bundle, prefill_plan, decode_plan


def test_admits_two_calibrated_phase_plans_with_a_shared_kv_layout():
    bundle, prefill, decode = _admit()

    assert bundle.admitted
    assert bundle.reason == "paired_phase_plans_calibrated_and_safe_to_switch"
    assert bundle.prefill_plan_fingerprint == phase_plan_fingerprint(prefill)
    assert bundle.decode_plan_fingerprint == phase_plan_fingerprint(decode)
    assert bundle.kv_contract_fingerprint == kv_contract_fingerprint(OWNERS, total_layers=2)
    assert bundle.calibration["prefill"].schedule_error_ratio < 0.1
    assert tuple(item.target for item in bundle.fallbacks) == FALLBACK_TARGETS
    assert bundle.to_dict()["schema_version"] == "qlh.torch_phase_plan.v1"


def test_rejects_plan_evidence_bound_to_a_different_plan_digest():
    plan = _plan("prefill", "workload:prefill:64")
    evidence = _evidence(plan, plan_fingerprint="f" * 64)
    bundle, _, _ = _admit(prefill_plan=plan, prefill_evidence=evidence)

    assert not bundle.admitted
    assert bundle.reason == "prefill_plan_evidence_mismatch"


def test_rejects_mixed_model_phase_plans():
    decode = replace(_plan("decode", "workload:decode:1"), model_fingerprint="another-model")

    bundle, _, _ = _admit(decode_plan=decode)

    assert not bundle.admitted
    assert bundle.reason == "phase_model_identity_mismatch"


def test_rejects_kv_owner_changes_between_prefill_and_decode():
    decode_owners = (
        LayerKVOwner(0, "cuda", "layout:layer-0"),
        LayerKVOwner(1, "cpu", "layout:layer-1"),
    )
    decode = _plan("decode", "workload:decode:1", ("cuda", "cpu"))

    bundle, _, _ = _admit(decode_plan=decode, decode_owners=decode_owners)

    assert not bundle.admitted
    assert bundle.reason == "kv_contract_changed_between_phases"


def test_rejects_kv_layout_change_even_when_layer_owners_stay_the_same():
    decode_owners = (
        LayerKVOwner(0, "cuda", "layout:layer-0"),
        LayerKVOwner(1, "cuda", "layout:changed"),
    )

    bundle, _, _ = _admit(decode_owners=decode_owners)

    assert not bundle.admitted
    assert bundle.reason == "kv_contract_changed_between_phases"


def test_rejects_kv_owner_that_disagrees_with_attention_placement():
    decode = _plan("decode", "workload:decode:1", ("cuda", "cpu"))

    bundle, _, _ = _admit(decode_plan=decode)

    assert not bundle.admitted
    assert bundle.reason == "kv_owner_not_backed_by_attention_plan"


def test_rejects_incomplete_kv_layer_coverage():
    owners = (LayerKVOwner(0, "cuda", "layout:layer-0"),)
    prefill = _plan("prefill", "workload:prefill:64")
    decode = _plan("decode", "workload:decode:1")

    bundle, _, _ = _admit(
        prefill_plan=prefill,
        decode_plan=decode,
        prefill_owners=owners,
        decode_owners=owners,
        prefill_evidence=_evidence(prefill),
        decode_evidence=_evidence(decode),
    )

    assert not bundle.admitted
    assert bundle.reason.startswith("kv_contract_invalid:")


def test_admission_normalizes_one_shot_kv_owner_iterables():
    prefill = _plan("prefill", "workload:prefill:64")
    decode = _plan("decode", "workload:decode:1")

    bundle = admit_phase_pair(
        prefill,
        decode,
        session_fingerprint=SESSION_FP,
        total_layers=2,
        prefill_kv_owners=iter(OWNERS),
        decode_kv_owners=iter(OWNERS),
        prefill_evidence=_evidence(prefill),
        decode_evidence=_evidence(decode),
        fallbacks=(_fallback("single_pytorch"),),
        policy=POLICY,
    )

    assert bundle.admitted


def test_controller_rejects_forged_admitted_bundle():
    with pytest.raises(ValueError, match="calibration"):
        PhaseSessionController(PhasePlanBundle(
            admitted=True,
            reason="forged",
            model_fingerprint=MODEL_FP,
            session_fingerprint=SESSION_FP,
            reference_fingerprint=REFERENCE_FP,
            prefill_plan_fingerprint="prefill-plan",
            decode_plan_fingerprint="decode-plan",
            kv_contract_fingerprint="kv-contract",
        ))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("instrumented", True, "prefill_measurement_instrumented"),
        ("warmup_consistent", False, "prefill_warmup_inconsistent"),
        ("correctness_passed", False, "prefill_correctness_gate_failed"),
        ("argmax_exact", False, "prefill_correctness_gate_failed"),
    ],
)
def test_rejects_unqualified_phase_execution_evidence(field, value, reason):
    plan = _plan("prefill", "workload:prefill:64")
    evidence = _evidence(plan, **{field: value})

    bundle, _, _ = _admit(prefill_plan=plan, prefill_evidence=evidence)

    assert not bundle.admitted
    assert bundle.reason == reason


def test_rejects_runtime_samples_outside_explicit_stability_policy():
    plan = _plan("decode", "workload:decode:1")
    evidence = _evidence(plan, samples_ms=(1.0, 2.0, 3.0))

    bundle, _, _ = _admit(decode_plan=plan, decode_evidence=evidence)

    assert not bundle.admitted
    assert bundle.reason == "decode_runtime_variance_exceeds_policy"


def test_rejects_schedule_estimate_outside_explicit_calibration_policy():
    plan = _plan("decode", "workload:decode:1")
    evidence = _evidence(plan, samples_ms=(3.0, 3.02, 3.04))

    bundle, _, _ = _admit(decode_plan=plan, decode_evidence=evidence)

    assert not bundle.admitted
    assert bundle.reason == "decode_schedule_calibration_exceeds_policy"


def test_requires_a_verified_full_request_fallback():
    fallback = _fallback("single_pytorch", argmax_exact=False)

    bundle, _, _ = _admit(fallbacks=(fallback,))

    assert not bundle.admitted
    assert bundle.reason == "no_verified_full_request_fallback"


def test_session_controller_switches_only_after_prefill_kv_identity_matches():
    bundle, _, decode = _admit()
    controller = PhaseSessionController(bundle)
    kv_fingerprint = bundle.kv_contract_fingerprint

    prefill = controller.start_prefill(SESSION_FP)
    completed = controller.complete_prefill(SESSION_FP, kv_fingerprint)
    decode_decision = controller.start_decode(SESSION_FP, kv_fingerprint)
    finished = controller.complete_decode(SESSION_FP)

    assert prefill.plan_fingerprint == bundle.prefill_plan_fingerprint
    assert completed.action == "phase_switch_ready"
    assert decode_decision.plan_fingerprint == phase_plan_fingerprint(decode)
    assert finished.state == "completed"


def test_kv_mismatch_requests_whole_request_fallback_before_decode():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)
    controller.start_prefill(SESSION_FP)

    decision = controller.complete_prefill(SESSION_FP, "wrong-kv-layout")

    assert decision.allowed
    assert decision.action == "restart_full_request"
    assert decision.fallback_target == "single_pytorch"
    assert decision.state == "fallback_ready"


def test_execution_failure_falls_back_by_restarting_the_full_request():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)
    controller.start_prefill(SESSION_FP)
    controller.complete_prefill(SESSION_FP, bundle.kv_contract_fingerprint)
    controller.start_decode(SESSION_FP, bundle.kv_contract_fingerprint)

    decision = controller.fail(
        SESSION_FP, phase="decode", error_code="device_unavailable",
    )
    started = controller.start_fallback(SESSION_FP, "single_pytorch")
    completed = controller.confirm_fallback_completed(SESSION_FP, "single_pytorch")

    assert decision.action == "restart_full_request"
    assert decision.fallback_target == "single_pytorch"
    assert started.action == "run_fallback"
    assert completed.state == "completed"


def test_failed_fallback_advances_to_next_candidate_before_output():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)
    controller.start_prefill(SESSION_FP)

    first = controller.fail(SESSION_FP, phase="prefill", error_code="device_unavailable")
    controller.start_fallback(SESSION_FP, "single_pytorch")
    second = controller.fail_fallback(
        SESSION_FP, target="single_pytorch", error_code="device_unavailable",
    )
    started = controller.start_fallback(SESSION_FP, "llama_cpp")
    completed = controller.confirm_fallback_completed(SESSION_FP, "llama_cpp")

    assert first.fallback_target == "single_pytorch"
    assert second.action == "restart_full_request"
    assert second.fallback_target == "llama_cpp"
    assert started.action == "run_fallback"
    assert completed.state == "completed"


def test_failed_fallback_after_published_output_aborts():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)
    controller.start_prefill(SESSION_FP)
    controller.fail(SESSION_FP, phase="prefill", error_code="device_unavailable")
    controller.start_fallback(SESSION_FP, "single_pytorch")
    controller.note_output_published(SESSION_FP)

    decision = controller.fail_fallback(
        SESSION_FP, target="single_pytorch", error_code="device_unavailable",
    )

    assert not decision.allowed
    assert decision.action == "abort"
    assert decision.state == "aborted"
    assert decision.reason == "partial_output_already_published"


def test_no_fallback_after_any_decode_output_was_published():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)
    controller.start_prefill(SESSION_FP)
    controller.complete_prefill(SESSION_FP, bundle.kv_contract_fingerprint)
    controller.start_decode(SESSION_FP, bundle.kv_contract_fingerprint)
    controller.note_output_published(SESSION_FP)

    decision = controller.fail(
        SESSION_FP, phase="decode", error_code="phase_timeout",
    )

    assert not decision.allowed
    assert decision.state == "aborted"
    assert decision.reason == "partial_output_already_published"


def test_rejects_decode_before_prefill_and_does_not_mutate_state():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)

    decision = controller.start_decode(SESSION_FP, bundle.kv_contract_fingerprint)

    assert not decision.allowed
    assert decision.state == "ready"
    assert controller.state == "ready"


def test_session_identity_mismatch_aborts_the_controller():
    bundle, _, _ = _admit()
    controller = PhaseSessionController(bundle)

    decision = controller.start_prefill("another-session")

    assert not decision.allowed
    assert controller.state == "aborted"
    assert decision.reason == "session_identity_mismatch"


def test_phase_planner_has_no_torch_import():
    import ast

    source = (ROOT / "src" / "torch_phase_plan.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "torch"
    ]

    assert imports == []
