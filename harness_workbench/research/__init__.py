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
]
