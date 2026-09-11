from hashlib import sha256

import pytest

from harness_workbench.eval import builtin_fixtures, fixture_digest
from harness_workbench.research import (
    EvidenceRecord,
    build_ceiling_study,
    compare_factor,
    evaluate_evidence_gate,
    pareto_points,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def test_ceiling_study_is_bounded_deterministic_and_model_free():
    digest = fixture_digest(builtin_fixtures())
    first = build_ceiling_study(fixture_set_digest=digest)
    second = build_ceiling_study(fixture_set_digest=digest)
    assert first.digest == second.digest
    assert first.v2_policy == {"match": "loose_contains", "max_new_tokens": 512, "historical_policy": {"match": "normalized_contains", "max_new_tokens": 192}}
    assert len(first.models) == 5
    assert len(first.questions) == 6
    assert len(first.cells) == len(second.cells)
    assert len(first.cells) > len(first.models) * len(first.factors)
    assert all(cell.holdout_required for cell in first.cells)
    assert first.as_dict()["study_digest"] == first.digest


def test_study_rejects_unbounded_or_invalid_inputs():
    with pytest.raises(ValueError):
        build_ceiling_study(fixture_set_digest="not-a-digest")
    with pytest.raises(ValueError):
        build_ceiling_study(fixture_set_digest=_digest("fixtures"), models=("QW1.8B", "QW1.8B"))


def test_factor_comparison_is_directional_and_does_not_claim_significance():
    result = compare_factor(
        {"quality_rate": 0.4, "latency_p95_ms": 100.0},
        {"quality_rate": 0.6, "latency_p95_ms": 120.0},
    )
    assert result["deltas"] == {"latency_p95_ms": 20.0, "quality_rate": 0.19999999999999996}
    assert "quality_rate" in result["improvements"]
    assert "latency_p95_ms" in result["regressions"]
    assert result["evidence_only"] is True


def test_pareto_marks_dominated_points_without_selecting_winner():
    points = pareto_points(
        [
            {"id": "safe", "quality_rate": 0.8, "latency_p95_ms": 100, "rss_peak_bytes": 1000},
            {"id": "slow-worse", "quality_rate": 0.7, "latency_p95_ms": 120, "rss_peak_bytes": 1200},
            {"id": "quality", "quality_rate": 0.9, "latency_p95_ms": 150, "rss_peak_bytes": 1400},
        ]
    )
    marked = {item["id"]: item["dominated"] for item in points}
    assert marked == {"safe": False, "slow-worse": True, "quality": False}


def test_evidence_gate_keeps_fixture_results_candidate():
    record = EvidenceRecord(
        cell_id="baseline:QW1.8B",
        model_profile_digest=_digest("profile"),
        artifact_digest=_digest("artifact"),
        runtime="fixture-runner",
        fixture_digest=_digest("fixtures"),
        seed=17,
        metrics={"quality_rate": 1.0},
        runner_kind="injected",
        status="collected",
    )
    gate = evaluate_evidence_gate(record)
    assert gate["status"] == "candidate"
    assert gate["promotable"] is False
    assert set(gate["reasons"]) == {"weights_not_loaded", "non_production_runner"}


def test_real_evidence_requires_explicit_local_boundary():
    record = EvidenceRecord(
        cell_id="baseline:QW1.8B",
        model_profile_digest=_digest("profile"),
        artifact_digest=_digest("artifact"),
        runtime="local-sidecar-v1",
        fixture_digest=_digest("fixtures"),
        seed=17,
        metrics={"quality_rate": 0.5},
        runner_kind="real",
        weights_loaded=True,
        status="collected",
    )
    assert evaluate_evidence_gate(record) == {"status": "measured", "promotable": False, "reasons": ()}
