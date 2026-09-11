"""Offline research contracts for bounded small-model ceiling studies."""

from .ceiling import (
    DEFAULT_MODEL_IDS,
    CeilingStudyPlan,
    EvidenceRecord,
    ExperimentCell,
    PublicEvidence,
    ResearchFactor,
    ResearchQuestion,
    build_ceiling_study,
    compare_factor,
    evaluate_evidence_gate,
    pareto_points,
)
from .context_measure import (
    CONTEXT_MEASURE_SCHEMA,
    DEFAULT_BUDGETS,
    STRATEGIES,
    ContextMeasureFixture,
    ContextMeasureObservation,
    ContextMeasureReport,
    build_context_measure_fixture,
    build_context_measure_report,
    run_context_measure,
)

__all__ = [
    "DEFAULT_MODEL_IDS",
    "CeilingStudyPlan",
    "EvidenceRecord",
    "ExperimentCell",
    "PublicEvidence",
    "ResearchFactor",
    "ResearchQuestion",
    "build_ceiling_study",
    "compare_factor",
    "evaluate_evidence_gate",
    "pareto_points",
    "CONTEXT_MEASURE_SCHEMA",
    "DEFAULT_BUDGETS",
    "STRATEGIES",
    "ContextMeasureFixture",
    "ContextMeasureObservation",
    "ContextMeasureReport",
    "build_context_measure_fixture",
    "build_context_measure_report",
    "run_context_measure",
]
