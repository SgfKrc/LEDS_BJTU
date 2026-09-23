from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch_operator_registry import (  # noqa: E402
    DEFAULT_TORCH_OPERATOR_REGISTRY,
    ImplementationKind,
    ImplementationSpec,
    LogicalOperator,
    NoSafeImplementationError,
    NumericalEvidence,
    OperatorContext,
    OperatorRegistry,
    SelectionPolicy,
)


def _context(**overrides) -> OperatorContext:
    values = {
        "device": "cuda",
        "dtype": "float16",
        "phase": "decode",
        "capabilities": frozenset({"cuda", "torch_compile"}),
        "enabled_feature_gates": frozenset({"torch_compile"}),
        "model_fingerprint": "model:sha256:abc",
        "workload_fingerprint": "workload:decode:1x896",
        "model_parameter_count": 1_500_000_000,
    }
    values.update(overrides)
    return OperatorContext(**values)


def _evidence(**overrides) -> NumericalEvidence:
    values = {
        "model_fingerprint": "model:sha256:abc",
        "workload_fingerprint": "workload:decode:1x896",
        "artifact_ref": "local_docs/evidence/profile.json",
        "passed": True,
        "argmax_exact": True,
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "sample_count": 16,
    }
    values.update(overrides)
    return NumericalEvidence(**values)


def test_default_registry_exposes_logical_catalog_without_runtime_kernels():
    catalog = DEFAULT_TORCH_OPERATOR_REGISTRY.catalog()

    assert catalog["schema_version"] == "qlh.torch_operator_registry.v1"
    assert {item["operator_id"] for item in catalog["logical_operators"]} == {
        "linear_projection",
        "attention_core",
        "normalization",
        "transformer_layer_loop",
    }
    assert {
        item["kind"] for item in catalog["implementations"]
    } == {"eager", "compiled"}


def test_eager_reference_is_selected_for_cpu_and_unprofiled_shapes():
    result = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "linear_projection",
        _context(
            device="cpu", dtype="float32", capabilities=frozenset(),
            enabled_feature_gates=frozenset(),
        ),
    )

    assert result.implementation.implementation_id == "pytorch.eager.cpu.linear_projection"
    assert result.implementation.is_reference
    assert not result.used_fallback


def test_compile_candidate_fails_closed_without_matching_numerical_evidence():
    result = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "transformer_layer_loop", _context(),
    )

    assert result.implementation.implementation_id == "pytorch.eager.cuda.transformer_layer_loop"
    assert result.used_fallback
    assert result.rejected_candidates == (
        "pytorch.compile.transformer_layer_loop: numerical evidence missing",
    )


def test_compile_candidate_requires_capability_gate_and_exact_evidence():
    candidate_id = "pytorch.compile.transformer_layer_loop"
    result = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "transformer_layer_loop",
        _context(),
        evidence={candidate_id: _evidence()},
    )

    assert result.implementation.implementation_id == candidate_id
    assert not result.used_fallback
    assert result.evidence.artifact_ref == "local_docs/evidence/profile.json"


@pytest.mark.parametrize(
    ("context", "evidence", "reason"),
    [
        (
            _context(device="cpu", dtype="float32"),
            _evidence(),
            "device unsupported",
        ),
        (_context(dtype="float32"), _evidence(), "dtype unsupported"),
        (_context(model_parameter_count=None), _evidence(), "model parameter count unknown"),
        (
            _context(model_parameter_count=1_000_000_000),
            _evidence(),
            "model has fewer than 1500000000 parameters",
        ),
        (
            _context(capabilities=frozenset({"cuda"})),
            _evidence(),
            "missing capabilities: torch_compile",
        ),
        (
            _context(enabled_feature_gates=frozenset()),
            _evidence(),
            "feature gate disabled",
        ),
        (
            _context(),
            _evidence(workload_fingerprint="other"),
            "numerical evidence scope mismatch",
        ),
        (
            _context(),
            _evidence(argmax_exact=False),
            "argmax-equivalence gate failed",
        ),
        (
            _context(),
            _evidence(passed=False),
            "numerical evidence failed",
        ),
    ],
)
def test_compile_ineligibility_uses_eager_fallback(context, evidence, reason):
    candidate_id = "pytorch.compile.transformer_layer_loop"
    result = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "transformer_layer_loop", context, evidence={candidate_id: evidence},
    )

    assert result.implementation.is_reference
    assert result.used_fallback
    assert result.rejected_candidates == (f"{candidate_id}: {reason}",)


def test_non_exact_candidate_needs_explicit_error_limits():
    with pytest.raises(ValueError, match="explicit error limits"):
        SelectionPolicy(require_argmax_exact=False)

    candidate_id = "pytorch.compile.transformer_layer_loop"
    strict = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "transformer_layer_loop",
        _context(),
        policy=SelectionPolicy(
            require_argmax_exact=False,
            max_abs_error=0.01,
            max_relative_error=0.02,
        ),
        evidence={candidate_id: _evidence(
            argmax_exact=False, max_abs_error=0.02, max_relative_error=0.01,
        )},
    )

    assert strict.implementation.is_reference
    assert strict.rejected_candidates == (
        f"{candidate_id}: numerical error exceeds policy limits",
    )


def test_exact_equivalence_does_not_bypass_explicit_error_caps():
    candidate_id = "pytorch.compile.transformer_layer_loop"
    result = DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
        "transformer_layer_loop",
        _context(),
        policy=SelectionPolicy(
            require_argmax_exact=True,
            max_abs_error=0.01,
            max_relative_error=0.02,
        ),
        evidence={candidate_id: _evidence(
            max_abs_error=0.02, max_relative_error=0.01,
        )},
    )

    assert result.implementation.is_reference
    assert result.rejected_candidates == (
        f"{candidate_id}: numerical error exceeds policy limits",
    )


def test_valid_candidate_is_not_shadowed_by_reference_priority():
    operator = LogicalOperator("transformer_layer_loop")
    reference = ImplementationSpec(
        "pytorch.eager.loop", "transformer_layer_loop", ImplementationKind.EAGER,
        frozenset({"cuda"}), frozenset({"float16"}), frozenset({"decode"}),
        priority=1000, is_reference=True,
    )
    candidate = ImplementationSpec(
        "pytorch.compile.loop", "transformer_layer_loop", ImplementationKind.COMPILED,
        frozenset({"cuda"}), frozenset({"float16"}), frozenset({"decode"}),
        required_capabilities=frozenset({"torch_compile"}),
        feature_gate="torch_compile", priority=1,
    )
    registry = OperatorRegistry((operator,), (reference, candidate))

    result = registry.resolve(
        "transformer_layer_loop", _context(),
        evidence={candidate.implementation_id: _evidence()},
    )

    assert result.implementation.implementation_id == candidate.implementation_id


def test_experimental_candidate_requires_explicit_policy_gate_and_evidence():
    operator = LogicalOperator("attention_core")
    reference = ImplementationSpec(
        "pytorch.eager.attention_core", "attention_core", ImplementationKind.EAGER,
        frozenset({"cuda"}), frozenset({"float16"}), frozenset({"decode"}),
        is_reference=True,
    )
    experimental = ImplementationSpec(
        "pytorch.experimental.attention_core", "attention_core",
        ImplementationKind.EXPERIMENTAL, frozenset({"cuda"}),
        frozenset({"float16"}), frozenset({"decode"}),
        required_capabilities=frozenset({"cuda"}),
        feature_gate="attention_experiment", priority=10,
    )
    registry = OperatorRegistry((operator,), (reference, experimental))
    context = _context(enabled_feature_gates=frozenset({"attention_experiment"}))
    evidence = {experimental.implementation_id: _evidence()}

    denied = registry.resolve("attention_core", context, evidence=evidence)
    allowed = registry.resolve(
        "attention_core", context,
        policy=SelectionPolicy(allow_experimental=True), evidence=evidence,
    )

    assert denied.implementation.is_reference
    assert allowed.implementation.implementation_id == experimental.implementation_id


def test_registry_rejects_unknown_and_duplicate_implementations():
    operator = LogicalOperator("linear_projection")
    registry = OperatorRegistry((operator,))
    eager = ImplementationSpec(
        "pytorch.eager.linear", "linear_projection", ImplementationKind.EAGER,
        frozenset({"cpu"}), frozenset({"float32"}), frozenset({"prefill"}),
        is_reference=True,
    )
    registry.register(eager)

    with pytest.raises(ValueError, match="duplicate implementation"):
        registry.register(eager)
    with pytest.raises(ValueError, match="unknown logical operator"):
        OperatorRegistry(
            (operator,),
            (ImplementationSpec(
                "pytorch.eager.unknown", "unknown", ImplementationKind.EAGER,
                frozenset({"cpu"}), frozenset({"float32"}), frozenset({"prefill"}),
                is_reference=True,
            ),),
        )


def test_no_compatible_reference_fails_closed():
    registry = OperatorRegistry((LogicalOperator("linear_projection"),))

    with pytest.raises(NoSafeImplementationError, match="no safe implementation"):
        registry.resolve("linear_projection", _context())


def test_unsupported_phase_fails_closed_instead_of_claiming_eager_fallback():
    with pytest.raises(NoSafeImplementationError, match="no safe implementation"):
        DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
            "transformer_layer_loop", _context(phase="training"),
        )


def test_cpu_fp16_has_no_declared_eager_fallback():
    with pytest.raises(NoSafeImplementationError, match="no safe implementation"):
        DEFAULT_TORCH_OPERATOR_REGISTRY.resolve(
            "linear_projection",
            _context(device="cpu", dtype="float16"),
        )


def test_evidence_rejects_non_finite_error_values():
    with pytest.raises(ValueError, match="finite and non-negative"):
        _evidence(max_abs_error=float("nan"))

    with pytest.raises(ValueError, match="must be booleans"):
        _evidence(passed="false")

    with pytest.raises(ValueError, match="sample_count must be an integer"):
        _evidence(sample_count=1.5)


def test_registry_module_does_not_import_torch_at_module_scope():
    import ast

    source = (ROOT / "src" / "torch_operator_registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "torch"
    ]

    assert imports == []
