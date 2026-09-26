"""Offline capability and correctness registry for PyTorch operator planning.

This module is deliberately dependency-free. It does not load torch, change the
Koakuma backend, or participate in inference dispatch; planners may use it to
reject unsupported or insufficiently validated implementation candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping


class ImplementationKind:
    EAGER = "eager"
    COMPILED = "compiled"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class LogicalOperator:
    operator_id: str
    observed_profile_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operator_id.strip():
            raise ValueError("operator_id must be non-empty")
        if any(not item.startswith("aten::") for item in self.observed_profile_operators):
            raise ValueError("observed profile operators must use aten names")


@dataclass(frozen=True)
class ImplementationSpec:
    implementation_id: str
    operator_id: str
    kind: str
    devices: frozenset[str]
    dtypes: frozenset[str]
    phases: frozenset[str]
    required_capabilities: frozenset[str] = frozenset()
    feature_gate: str | None = None
    priority: int = 0
    is_reference: bool = False
    min_model_parameters: int | None = None

    def __post_init__(self) -> None:
        if not self.implementation_id.strip() or not self.operator_id.strip():
            raise ValueError("implementation_id and operator_id must be non-empty")
        if self.kind not in {
            ImplementationKind.EAGER,
            ImplementationKind.COMPILED,
            ImplementationKind.EXPERIMENTAL,
        }:
            raise ValueError(f"unsupported implementation kind: {self.kind}")
        if not self.devices or not self.dtypes or not self.phases:
            raise ValueError("implementations must declare devices, dtypes, and phases")
        for name, values in (
            ("devices", self.devices),
            ("dtypes", self.dtypes),
            ("phases", self.phases),
            ("required_capabilities", self.required_capabilities),
        ):
            if any(not isinstance(value, str) or not value or value != value.lower()
                   for value in values):
                raise ValueError(f"{name} values must be non-empty lowercase strings")
        if not isinstance(self.is_reference, bool):
            raise ValueError("is_reference must be a boolean")
        if self.min_model_parameters is not None and (
            isinstance(self.min_model_parameters, bool)
            or not isinstance(self.min_model_parameters, int)
            or self.min_model_parameters < 1
        ):
            raise ValueError("min_model_parameters must be a positive integer")
        if self.kind == ImplementationKind.EAGER:
            if not self.is_reference or self.feature_gate:
                raise ValueError("eager implementations must be ungated references")
        elif self.is_reference or not self.feature_gate:
            raise ValueError("non-reference implementations require an explicit feature gate")


@dataclass(frozen=True)
class OperatorContext:
    device: str
    dtype: str
    phase: str
    capabilities: frozenset[str] = frozenset()
    enabled_feature_gates: frozenset[str] = frozenset()
    model_fingerprint: str = ""
    workload_fingerprint: str = ""
    model_parameter_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("device", "dtype", "phase"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.lower():
                raise ValueError(f"{name} must be a non-empty lowercase string")
        for name, values in (
            ("capabilities", self.capabilities),
            ("enabled_feature_gates", self.enabled_feature_gates),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} values must be non-empty strings")
        if self.model_parameter_count is not None and (
            isinstance(self.model_parameter_count, bool)
            or not isinstance(self.model_parameter_count, int)
            or self.model_parameter_count < 1
        ):
            raise ValueError("model_parameter_count must be a positive integer")


@dataclass(frozen=True)
class NumericalEvidence:
    model_fingerprint: str
    workload_fingerprint: str
    artifact_ref: str
    passed: bool
    argmax_exact: bool
    max_abs_error: float
    max_relative_error: float
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool) or not isinstance(self.argmax_exact, bool):
            raise ValueError("evidence pass and argmax fields must be booleans")
        if not self.model_fingerprint or not self.workload_fingerprint or not self.artifact_ref:
            raise ValueError("evidence must identify its model, workload, and artifact")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise ValueError("evidence sample_count must be an integer")
        if self.sample_count < 1:
            raise ValueError("evidence sample_count must be positive")
        for name, value in (
            ("max_abs_error", self.max_abs_error),
            ("max_relative_error", self.max_relative_error),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class SelectionPolicy:
    require_argmax_exact: bool = True
    max_abs_error: float | None = None
    max_relative_error: float | None = None
    allow_experimental: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_argmax_exact, bool) or not isinstance(
            self.allow_experimental, bool,
        ):
            raise ValueError("selection policy flags must be booleans")
        if not self.require_argmax_exact and (
            self.max_abs_error is None or self.max_relative_error is None
        ):
            raise ValueError("non-exact selection requires explicit error limits")
        for value in (self.max_abs_error, self.max_relative_error):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("error limits must be finite and non-negative")


@dataclass(frozen=True)
class OperatorResolution:
    implementation: ImplementationSpec
    used_fallback: bool
    rejected_candidates: tuple[str, ...]
    evidence: NumericalEvidence | None = None


class NoSafeImplementationError(RuntimeError):
    """No registered implementation satisfies the requested execution context."""


class OperatorRegistry:
    def __init__(
        self,
        logical_operators: Iterable[LogicalOperator],
        implementations: Iterable[ImplementationSpec] = (),
    ) -> None:
        self._operators: dict[str, LogicalOperator] = {}
        self._implementations: dict[str, ImplementationSpec] = {}
        for operator in logical_operators:
            if operator.operator_id in self._operators:
                raise ValueError(f"duplicate logical operator: {operator.operator_id}")
            self._operators[operator.operator_id] = operator
        if not self._operators:
            raise ValueError("registry requires at least one logical operator")
        for implementation in implementations:
            self.register(implementation)

    def register(self, implementation: ImplementationSpec) -> None:
        if implementation.operator_id not in self._operators:
            raise ValueError(f"unknown logical operator: {implementation.operator_id}")
        if implementation.implementation_id in self._implementations:
            raise ValueError(f"duplicate implementation: {implementation.implementation_id}")
        self._implementations[implementation.implementation_id] = implementation

    def compatible_implementations(
        self,
        operator_id: str,
        context: OperatorContext,
        *,
        policy: SelectionPolicy = SelectionPolicy(),
        evidence: Mapping[str, NumericalEvidence] | None = None,
    ) -> tuple[ImplementationSpec, ...]:
        """List every implementation that passes the same gates as ``resolve``.

        Cost planners need all safe alternatives to compare measured latency;
        ``resolve`` remains the simple priority-based single-choice API.
        """
        if operator_id not in self._operators:
            raise KeyError(f"unknown logical operator: {operator_id}")
        evidence = evidence or {}
        candidates = sorted(
            (item for item in self._implementations.values()
             if item.operator_id == operator_id),
            key=lambda item: (item.priority, item.implementation_id),
            reverse=True,
        )
        return tuple(
            item for item in candidates
            if not self._ineligibility(
                item, context, policy, evidence.get(item.implementation_id),
            )
        )

    def catalog(self) -> dict[str, object]:
        """Return a stable JSON-compatible snapshot for offline planner artifacts."""
        return {
            "schema_version": "qlh.torch_operator_registry.v1",
            "logical_operators": [
                asdict(operator) for operator in self._operators.values()
            ],
            "implementations": [
                {
                    **asdict(implementation),
                    "devices": sorted(implementation.devices),
                    "dtypes": sorted(implementation.dtypes),
                    "phases": sorted(implementation.phases),
                    "required_capabilities": sorted(implementation.required_capabilities),
                }
                for implementation in self._implementations.values()
            ],
        }

    def resolve(
        self,
        operator_id: str,
        context: OperatorContext,
        *,
        policy: SelectionPolicy = SelectionPolicy(),
        evidence: Mapping[str, NumericalEvidence] | None = None,
    ) -> OperatorResolution:
        if operator_id not in self._operators:
            raise KeyError(f"unknown logical operator: {operator_id}")
        evidence = evidence or {}
        candidates = [
            item for item in self._implementations.values()
            if item.operator_id == operator_id
        ]
        candidates.sort(key=lambda item: (item.priority, item.implementation_id), reverse=True)
        rejected: list[str] = []
        compatible: list[ImplementationSpec] = []
        for item in candidates:
            reason = self._ineligibility(item, context, policy, evidence.get(item.implementation_id))
            if reason:
                if not item.is_reference:
                    rejected.append(f"{item.implementation_id}: {reason}")
            else:
                compatible.append(item)
        if not compatible:
            raise NoSafeImplementationError(
                f"no safe implementation for {operator_id} on "
                f"{context.device}/{context.dtype}/{context.phase}"
            )
        selected = next((item for item in compatible if not item.is_reference), None)
        if selected is None:
            selected = next((item for item in compatible if item.is_reference), None)
        if selected is None:
            raise NoSafeImplementationError(
                f"no safe implementation for {operator_id} on "
                f"{context.device}/{context.dtype}/{context.phase}"
            )
        return OperatorResolution(
            implementation=selected,
            used_fallback=selected.is_reference and any(
                not item.is_reference for item in candidates
            ),
            rejected_candidates=tuple(rejected),
            evidence=evidence.get(selected.implementation_id),
        )

    @staticmethod
    def _ineligibility(
        implementation: ImplementationSpec,
        context: OperatorContext,
        policy: SelectionPolicy,
        evidence: NumericalEvidence | None,
    ) -> str:
        if context.device not in implementation.devices:
            return "device unsupported"
        if context.dtype not in implementation.dtypes:
            return "dtype unsupported"
        if context.phase not in implementation.phases:
            return "phase unsupported"
        missing = implementation.required_capabilities - context.capabilities
        if missing:
            return "missing capabilities: " + ", ".join(sorted(missing))
        if implementation.is_reference:
            return ""
        if implementation.min_model_parameters is not None:
            if context.model_parameter_count is None:
                return "model parameter count unknown"
            if context.model_parameter_count < implementation.min_model_parameters:
                return (
                    f"model has fewer than {implementation.min_model_parameters} parameters"
                )
        if implementation.feature_gate not in context.enabled_feature_gates:
            return "feature gate disabled"
        if implementation.kind == ImplementationKind.EXPERIMENTAL and not policy.allow_experimental:
            return "experimental implementations are not allowed"
        if evidence is None:
            return "numerical evidence missing"
        if not evidence.passed:
            return "numerical evidence failed"
        if (
            evidence.model_fingerprint != context.model_fingerprint
            or evidence.workload_fingerprint != context.workload_fingerprint
        ):
            return "numerical evidence scope mismatch"
        if policy.require_argmax_exact:
            if not evidence.argmax_exact:
                return "argmax-equivalence gate failed"
        if (
            policy.max_abs_error is not None
            and evidence.max_abs_error > policy.max_abs_error
        ) or (
            policy.max_relative_error is not None
            and evidence.max_relative_error > policy.max_relative_error
        ):
            return "numerical error exceeds policy limits"
        return ""


_ALL_PHASES = frozenset({"prefill", "decode"})
_CPU_DTYPES = frozenset({"float32"})
_CUDA_DTYPES = frozenset({"float32", "float16"})

LOGICAL_OPERATORS = (
    LogicalOperator("linear_projection", ("aten::mm", "aten::addmm")),
    LogicalOperator("attention_core", ("aten::bmm",)),
    LogicalOperator("normalization", ("aten::mean", "aten::pow", "aten::rsqrt")),
    LogicalOperator("transformer_layer_loop"),
)


def _eager(operator_id: str, device: str, dtypes: frozenset[str]) -> ImplementationSpec:
    return ImplementationSpec(
        implementation_id=f"pytorch.eager.{device}.{operator_id}",
        operator_id=operator_id,
        kind=ImplementationKind.EAGER,
        devices=frozenset({device}),
        dtypes=dtypes,
        phases=_ALL_PHASES,
        is_reference=True,
    )


DEFAULT_IMPLEMENTATIONS = tuple(
    _eager(operator.operator_id, device, dtypes)
    for operator in LOGICAL_OPERATORS
    for device, dtypes in (("cpu", _CPU_DTYPES), ("cuda", _CUDA_DTYPES))
) + (
    ImplementationSpec(
        implementation_id="pytorch.compile.transformer_layer_loop",
        operator_id="transformer_layer_loop",
        kind=ImplementationKind.COMPILED,
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float16"}),
        phases=_ALL_PHASES,
        required_capabilities=frozenset({"cuda", "torch_compile"}),
        feature_gate="torch_compile",
        priority=100,
        min_model_parameters=1_500_000_000,
    ),
)

DEFAULT_TORCH_OPERATOR_REGISTRY = OperatorRegistry(
    LOGICAL_OPERATORS, DEFAULT_IMPLEMENTATIONS,
)


__all__ = [
    "DEFAULT_IMPLEMENTATIONS",
    "DEFAULT_TORCH_OPERATOR_REGISTRY",
    "ImplementationKind",
    "ImplementationSpec",
    "LogicalOperator",
    "LOGICAL_OPERATORS",
    "NoSafeImplementationError",
    "NumericalEvidence",
    "OperatorContext",
    "OperatorRegistry",
    "OperatorResolution",
    "SelectionPolicy",
]
