import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec
from task_graph_analysis import (
    analyze_task_graph_structure,
    build_stage_duration_estimate,
)
from task_graph_optimization import project_task_graph
from task_graph_ordering import (
    build_ordering_simulation_profile,
    simulate_task_graph_ordering,
)
from task_graph_resilience import (
    TaskGraphResilienceError,
    build_resilience_recommendation_profile,
    recommend_task_graph_resilience,
    validate_resilience_recommendation,
    validate_resilience_recommendation_profile,
)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _duration(stage_id, expected):
    return build_stage_duration_estimate(
        stage_id,
        estimate_source="stage_type_baseline",
        duration_lower_ms=expected,
        duration_expected_ms=expected,
        duration_upper_ms=expected,
        profile_version="g4-resilience-fixture.v1",
    )


def _fixture(*, partial_final=False, prep_pure=True):
    stages = [
        StageSpec("start", "transform", pure=True),
        StageSpec(
            "prep",
            "transform",
            depends_on=("start",),
            pure=prep_pure,
        ),
        StageSpec("left", "transform", depends_on=("prep",), pure=True),
        StageSpec("right", "transform", depends_on=("prep",), pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("left", "right"),
            minimum_successful_dependencies=1 if partial_final else None,
        ),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="g4-resilience-partial" if partial_final else "g4-resilience",
    )
    contracts = [
        _duration("start", 5),
        _duration("prep", 10),
        _duration("left", 3),
        _duration("right", 2),
        _duration("final", 1),
    ]
    analysis = analyze_task_graph_structure(projection, contracts)
    simulation = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=2,
        ),
    )
    return projection, analysis, simulation


def _profile(**overrides):
    return build_resilience_recommendation_profile(**overrides)


def _checkpoint(report, stage_id):
    return next(
        item for item in report["checkpoint_candidates"]
        if item["stage_id"] == stage_id
    )


def _rejection(report, stage_id):
    return next(
        item for item in report["checkpoint_rejections"]
        if item["stage_id"] == stage_id
    )


def _scope(report, trigger_stage_id):
    return next(
        item for item in report["cancellation_scopes"]
        if item["trigger_stage_id"] == trigger_stage_id
    )


def test_g4_3_checkpoint_candidates_rank_dominator_recompute_boundaries():
    projection, analysis, simulation = _fixture()

    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )

    assert report["status"] == "evaluated"
    assert report["selected_graph_kind"] == "logical_dag"
    assert report["runtime_actions_enabled"] is False
    assert [item["stage_id"] for item in report["checkpoint_candidates"]] == [
        "prep", "start",
    ]
    prep = _checkpoint(report, "prep")
    start = _checkpoint(report, "start")
    assert prep["priority_rank"] == 1
    assert prep["dominated_stage_ids"] == ["final", "left", "right"]
    assert prep["upstream_recompute_expected_ms"] == 15
    assert prep["affected_work_expected_ms"] == 6
    assert prep["benefit_score"] == 60
    assert prep["critical_path_member"] is True
    assert prep["post_dominator_boundary_node_id"] == "stage:final"
    assert start["priority_rank"] == 2
    assert start["dominated_stage_count"] == 4
    assert start["benefit_score"] == 25
    assert _rejection(report, "final")["reason_code"] == "exit_stage"
    assert _rejection(report, "left")["reason_code"] == (
        "dominated_scope_too_small"
    )
    assert report["summary"]["checkpoint_candidate_count"] == 2
    assert report["summary"]["runtime_action_count"] == 0
    assert validate_resilience_recommendation(report) == report


def test_g4_3_required_join_failure_builds_cascade_cancellation_scope():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )

    prep = _scope(report, "prep")
    left = _scope(report, "left")
    start = _scope(report, "start")
    assert start["affected_stage_ids"] == ["final", "left", "prep", "right"]
    assert start["affected_work_expected_ms"] == 16
    assert start["priority_rank"] == 1
    assert prep["affected_stage_ids"] == ["final", "left", "right"]
    assert prep["affected_work_expected_ms"] == 6
    assert left["affected_stage_ids"] == ["final"]
    assert left["reason_code"] == "dependency_success_threshold_exhausted"
    assert left["applies_to_state"] == "not_started"
    assert left["contains_side_effect_stage"] is True
    assert left["post_dominator_boundary_node_id"] == "stage:final"


def test_g4_3_partial_join_preserves_final_when_one_branch_can_succeed():
    projection, analysis, simulation = _fixture(partial_final=True)
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )

    triggers = {
        item["trigger_stage_id"] for item in report["cancellation_scopes"]
    }
    assert "left" not in triggers
    assert "right" not in triggers
    assert _scope(report, "prep")["affected_stage_ids"] == [
        "final", "left", "right",
    ]
    assert _scope(report, "start")["affected_stage_ids"] == [
        "final", "left", "prep", "right",
    ]


def test_g4_3_non_pure_stage_is_not_a_checkpoint_candidate():
    projection, analysis, simulation = _fixture(prep_pure=False)
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )

    assert "prep" not in {
        item["stage_id"] for item in report["checkpoint_candidates"]
    }
    assert _rejection(report, "prep")["reason_code"] == "stage_not_pure"


def test_g4_3_checkpoint_threshold_and_candidate_limit_are_explicit():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection,
        analysis,
        simulation,
        _profile(
            min_checkpoint_recompute_ms=1,
            max_checkpoint_candidates=1,
        ),
    )

    assert [item["stage_id"] for item in report["checkpoint_candidates"]] == [
        "prep",
    ]
    assert _rejection(report, "start")["reason_code"] == (
        "candidate_limit_reached"
    )

    threshold_report = recommend_task_graph_resilience(
        projection,
        analysis,
        simulation,
        _profile(min_checkpoint_recompute_ms=10),
    )
    assert _rejection(threshold_report, "start")["reason_code"] == (
        "recompute_cost_below_threshold"
    )


def test_g4_3_cancel_scope_limit_ranks_by_affected_work():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection,
        analysis,
        simulation,
        _profile(max_cancel_scopes=2),
    )

    assert [
        item["trigger_stage_id"] for item in report["cancellation_scopes"]
    ] == ["start", "prep"]
    assert [item["priority_rank"] for item in report["cancellation_scopes"]] == [
        1, 2,
    ]


def test_g4_3_partial_analysis_falls_back_without_recommendations():
    projection, _analysis, _simulation = _fixture()
    partial = analyze_task_graph_structure(projection, [])
    fallback_simulation = simulate_task_graph_ordering(
        projection,
        partial,
        build_ordering_simulation_profile(),
    )

    report = recommend_task_graph_resilience(
        projection, partial, fallback_simulation, _profile(),
    )

    assert report["status"] == "fallback"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "analysis_unavailable",
    }
    assert report["checkpoint_candidates"] == []
    assert report["checkpoint_rejections"] == []
    assert report["cancellation_scopes"] == []
    assert report["summary"]["runtime_action_count"] == 0
    assert validate_resilience_recommendation(report) == report


def test_g4_3_simulation_unavailable_falls_back():
    projection, analysis, _simulation = _fixture()
    partial = analyze_task_graph_structure(projection, [])
    fallback_simulation = simulate_task_graph_ordering(
        projection,
        partial,
        build_ordering_simulation_profile(),
    )

    report = recommend_task_graph_resilience(
        projection, analysis, fallback_simulation, _profile(),
    )

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "simulation_unavailable"


def test_g4_3_analysis_simulation_digest_mismatch_falls_back():
    projection, analysis, _simulation = _fixture()
    alternate_analysis = analyze_task_graph_structure(
        projection,
        [
            _duration("start", 6),
            _duration("prep", 10),
            _duration("left", 3),
            _duration("right", 2),
            _duration("final", 1),
        ],
    )
    alternate_simulation = simulate_task_graph_ordering(
        projection,
        alternate_analysis,
        build_ordering_simulation_profile(),
    )

    report = recommend_task_graph_resilience(
        projection, analysis, alternate_simulation, _profile(),
    )

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "evidence_digest_mismatch"


def test_g4_3_evidence_from_another_graph_falls_back():
    projection, analysis, simulation = _fixture()
    other = project_task_graph(
        [StageSpec("only", "transform", pure=True)],
        "only",
        graph_kind="logical_dag",
        graph_id="other-resilience-graph",
    )

    report = recommend_task_graph_resilience(
        other, analysis, simulation, _profile(),
    )

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "evidence_graph_mismatch"
    assert report["summary"]["stage_count"] == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_checkpoint_recompute_ms", 0),
        ("min_checkpoint_scope_size", 0),
        ("max_checkpoint_candidates", 0),
        ("min_cancel_scope_size", 0),
        ("max_cancel_scopes", 257),
    ],
)
def test_g4_3_profile_bounds(field_name, value):
    with pytest.raises(TaskGraphResilienceError, match="safe range"):
        build_resilience_recommendation_profile(**{field_name: value})


def test_g4_3_profile_and_report_are_deterministic_and_body_free():
    projection, analysis, simulation = _fixture()
    profile = _profile(profile_id="deterministic-resilience.v1")

    first = recommend_task_graph_resilience(
        projection, analysis, simulation, profile,
    )
    second = recommend_task_graph_resilience(
        projection, analysis, simulation, dict(profile),
    )

    assert first == second
    assert validate_resilience_recommendation_profile(profile) == profile
    rendered = json.dumps(first)
    assert "g4-resilience-fixture.v1" not in rendered
    assert "root_input" not in rendered


def test_g4_3_validation_rejects_runtime_enable_and_summary_tampering():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )

    runtime_enabled = json.loads(json.dumps(report))
    runtime_enabled["runtime_actions_enabled"] = True
    with pytest.raises(TaskGraphResilienceError, match="disabled"):
        validate_resilience_recommendation(runtime_enabled)

    wrong_summary = json.loads(json.dumps(report))
    wrong_summary["summary"]["checkpoint_candidate_count"] += 1
    with pytest.raises(TaskGraphResilienceError, match="summary"):
        validate_resilience_recommendation(wrong_summary)


def test_g4_3_validation_rejects_rehashed_false_checkpoint_score():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )
    tampered = json.loads(json.dumps(report))
    candidate = tampered["checkpoint_candidates"][0]
    candidate["benefit_score"] += 1
    candidate_base = {
        key: value for key, value in candidate.items()
        if key != "candidate_digest"
    }
    candidate["candidate_digest"] = _digest(candidate_base)
    tampered["summary"]["checkpoint_total_benefit_score"] += 1
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphResilienceError, match="benefit score"):
        validate_resilience_recommendation(tampered)


def test_g4_3_validation_rejects_rehashed_false_cancellation_scope():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )
    tampered = json.loads(json.dumps(report))
    scope = tampered["cancellation_scopes"][0]
    scope["affected_stage_count"] += 1
    scope_base = {
        key: value for key, value in scope.items() if key != "scope_digest"
    }
    scope["scope_digest"] = _digest(scope_base)
    tampered["summary"]["cancellation_affected_stage_count"] += 1
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphResilienceError, match="scope count"):
        validate_resilience_recommendation(tampered)


def test_g4_3_report_validation_rejects_forbidden_fields():
    projection, analysis, simulation = _fixture()
    report = recommend_task_graph_resilience(
        projection, analysis, simulation, _profile(),
    )
    tampered = json.loads(json.dumps(report))
    tampered["summary"]["path"] = "private"

    with pytest.raises(TaskGraphResilienceError, match="forbidden"):
        validate_resilience_recommendation(tampered)
