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
    TaskGraphOrderingError,
    build_ordering_simulation_profile,
    simulate_task_graph_ordering,
    validate_ordering_simulation,
    validate_ordering_simulation_profile,
)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _duration(stage_id, expected, *, lower=None, upper=None):
    return build_stage_duration_estimate(
        stage_id,
        estimate_source="stage_type_baseline",
        duration_lower_ms=expected if lower is None else lower,
        duration_expected_ms=expected,
        duration_upper_ms=expected if upper is None else upper,
        profile_version="g4-ordering-fixture.v1",
    )


def _benefit_fixture():
    stages = [
        StageSpec("a_side", "transform", pure=True),
        StageSpec("b_side", "transform", pure=True),
        StageSpec("z1", "transform", pure=True),
        StageSpec("z2", "transform", depends_on=("z1",), pure=True),
        StageSpec("z3", "transform", depends_on=("z2",), pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("a_side", "b_side", "z3"),
        ),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="g4-ordering-benefit",
    )
    contracts = [
        _duration("a_side", 9, lower=8, upper=10),
        _duration("b_side", 9, lower=8, upper=10),
        _duration("z1", 1, lower=1, upper=2),
        _duration("z2", 10, lower=8, upper=12),
        _duration("z3", 10, lower=8, upper=12),
        _duration("final", 1, lower=1, upper=2),
    ]
    analysis = analyze_task_graph_structure(projection, contracts)
    return projection, analysis


def _policy(report, policy):
    return next(
        result for result in report["policy_results"]
        if result["policy"] == policy
    )


def _scenario(report, policy, basis="expected"):
    return next(
        scenario for scenario in _policy(report, policy)["scenarios"]
        if scenario["duration_basis"] == basis
    )


def _record(report, policy, stage_id, basis="expected"):
    return next(
        record for record in _scenario(report, policy, basis)["stage_records"]
        if record["stage_id"] == stage_id
    )


def test_g4_2_critical_path_shadow_policy_reduces_fixture_makespan():
    projection, analysis = _benefit_fixture()
    profile = build_ordering_simulation_profile(
        max_concurrency=2,
        max_ready_bypass=100,
    )

    report = simulate_task_graph_ordering(projection, analysis, profile)

    topological = _policy(report, "topological")["summary"]
    critical = _policy(report, "critical_path")["summary"]
    assert report["status"] == "evaluated"
    assert report["selected_graph_kind"] == "logical_dag"
    assert report["selected_ordering_policy"] == "topological"
    assert topological["expected_makespan_ms"] == 31
    assert critical["expected_makespan_ms"] == 22
    assert report["comparison"] == {
        "available": True,
        "expected_makespan_delta_ms": -9,
        "p95_makespan_delta_ms": -10,
        "expected_improvement_ms": 9,
        "p95_improvement_ms": 10,
        "critical_non_regression": True,
        "recommendation": "continue_shadow",
        "comparison_digest": report["comparison"]["comparison_digest"],
    }
    assert _scenario(report, "topological")["dispatch_order"][:3] == [
        "a_side", "b_side", "z1",
    ]
    assert _scenario(report, "critical_path")["dispatch_order"][:3] == [
        "z1", "a_side", "z2",
    ]
    assert validate_ordering_simulation(report) == report


def test_g4_2_bounded_bypass_promotes_waiting_noncritical_stage():
    stages = [
        StageSpec("side", "transform", pure=True),
        StageSpec("z1", "transform", pure=True),
        StageSpec("z2", "transform", depends_on=("z1",), pure=True),
        StageSpec(
            "final", "aggregate", depends_on=("side", "z2"),
        ),
    ]
    projection = project_task_graph(
        stages, "final", graph_kind="logical_dag", graph_id="bounded-bypass",
    )
    analysis = analyze_task_graph_structure(
        projection,
        [
            _duration("side", 1),
            _duration("z1", 5),
            _duration("z2", 5),
            _duration("final", 1),
        ],
    )
    profile = build_ordering_simulation_profile(
        max_concurrency=1,
        max_ready_bypass=1,
    )

    report = simulate_task_graph_ordering(projection, analysis, profile)
    critical = _scenario(report, "critical_path")
    side = _record(report, "critical_path", "side")

    assert critical["dispatch_order"] == ["z1", "side", "z2", "final"]
    assert side["bypass_count"] == 1
    assert side["protection_applied"] is True
    assert critical["max_ready_bypass"] == 1
    assert critical["protection_dispatch_count"] >= 1
    assert all(
        record["bypass_count"] <= profile["max_ready_bypass"]
        for record in critical["stage_records"]
    )


def test_g4_2_single_concurrency_changes_order_but_not_total_makespan():
    projection, analysis = _benefit_fixture()
    profile = build_ordering_simulation_profile(
        max_concurrency=1,
        max_ready_bypass=100,
    )

    report = simulate_task_graph_ordering(projection, analysis, profile)

    assert _policy(report, "topological")["summary"][
        "expected_makespan_ms"
    ] == 40
    assert _policy(report, "critical_path")["summary"][
        "expected_makespan_ms"
    ] == 40
    assert report["comparison"]["expected_makespan_delta_ms"] == 0
    assert report["comparison"]["critical_non_regression"] is True


def test_g4_2_lower_expected_upper_scenarios_use_identical_policy_inputs():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=100,
        ),
    )

    for policy in ("topological", "critical_path"):
        result = _policy(report, policy)
        assert [item["duration_basis"] for item in result["scenarios"]] == [
            "lower", "expected", "upper",
        ]
        assert result["summary"]["scenario_count"] == 3
        assert result["summary"]["makespan_p95_ms"] == result[
            "scenarios"
        ][2]["makespan_ms"]
    for basis in ("lower", "expected", "upper"):
        topological_durations = {
            record["stage_id"]: record["duration_ms"]
            for record in _scenario(report, "topological", basis)[
                "stage_records"
            ]
        }
        critical_durations = {
            record["stage_id"]: record["duration_ms"]
            for record in _scenario(report, "critical_path", basis)[
                "stage_records"
            ]
        }
        assert topological_durations == critical_durations


def test_g4_2_capacity_and_wait_metrics_are_derived_from_stage_records():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=100,
        ),
    )
    scenario = _scenario(report, "critical_path")

    assert scenario["busy_slot_ms"] == 40
    assert scenario["capacity_slot_ms"] == 44
    assert scenario["utilization_ppm"] == 909090
    assert scenario["ready_wait_max_ms"] == max(
        record["ready_wait_ms"] for record in scenario["stage_records"]
    )
    assert report["summary"]["critical_expected_makespan_ms"] == 22
    assert report["summary"]["topological_expected_makespan_ms"] == 31


def test_g4_2_partial_g4_1_analysis_returns_no_candidate_simulation():
    projection, _analysis = _benefit_fixture()
    partial = analyze_task_graph_structure(projection, [])
    profile = build_ordering_simulation_profile()

    report = simulate_task_graph_ordering(projection, partial, profile)

    assert report["status"] == "fallback"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "analysis_timing_unavailable",
    }
    assert report["policy_results"] == []
    assert report["comparison"]["available"] is False
    assert report["summary"]["policy_count"] == 0
    assert report["selected_ordering_policy"] == "topological"
    assert validate_ordering_simulation(report) == report


def test_g4_2_analysis_from_another_graph_falls_back_without_simulation():
    projection, analysis = _benefit_fixture()
    other = project_task_graph(
        [StageSpec("only", "transform", pure=True)],
        "only",
        graph_kind="logical_dag",
        graph_id="other-graph",
    )

    report = simulate_task_graph_ordering(
        other, analysis, build_ordering_simulation_profile(),
    )

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "analysis_graph_mismatch"
    assert report["summary"]["stage_count"] == 1
    assert report["policy_results"] == []


@pytest.mark.parametrize(
    ("max_concurrency", "max_ready_bypass", "valid"),
    [
        (1, 0, True),
        (64, 1024, True),
        (0, 1, False),
        (65, 1, False),
        (2, -1, False),
        (2, 1025, False),
    ],
)
def test_g4_2_simulation_profile_bounds(
    max_concurrency, max_ready_bypass, valid,
):
    if valid:
        profile = build_ordering_simulation_profile(
            max_concurrency=max_concurrency,
            max_ready_bypass=max_ready_bypass,
        )
        assert validate_ordering_simulation_profile(profile) == profile
    else:
        with pytest.raises(TaskGraphOrderingError, match="safe range"):
            build_ordering_simulation_profile(
                max_concurrency=max_concurrency,
                max_ready_bypass=max_ready_bypass,
            )


def test_g4_2_profile_and_report_are_deterministic_and_body_free():
    projection, analysis = _benefit_fixture()
    profile = build_ordering_simulation_profile(
        profile_id="deterministic-shadow.v1",
        max_concurrency=2,
        max_ready_bypass=3,
    )

    first = simulate_task_graph_ordering(projection, analysis, profile)
    second = simulate_task_graph_ordering(projection, analysis, dict(profile))

    assert first == second
    rendered = json.dumps(first)
    assert "g4-ordering-fixture.v1" not in rendered
    assert "root_input" not in rendered
    assert "critical_path" in rendered


def test_g4_2_validation_rejects_policy_selection_and_summary_tampering():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=100,
        ),
    )

    wrong_policy = json.loads(json.dumps(report))
    wrong_policy["selected_ordering_policy"] = "critical_path"
    with pytest.raises(TaskGraphOrderingError, match="Scheduler"):
        validate_ordering_simulation(wrong_policy)

    wrong_summary = json.loads(json.dumps(report))
    wrong_summary["summary"]["critical_expected_makespan_ms"] += 1
    with pytest.raises(TaskGraphOrderingError, match="summary"):
        validate_ordering_simulation(wrong_summary)


def test_g4_2_validation_rejects_rehashed_false_stage_timing():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=100,
        ),
    )
    tampered = json.loads(json.dumps(report))
    record = tampered["policy_results"][1]["scenarios"][1][
        "stage_records"
    ][0]
    record["completed_at_ms"] += 1
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphOrderingError, match="timing"):
        validate_ordering_simulation(tampered)


def test_g4_2_validation_rejects_rehashed_bypass_bound_violation():
    stages = [
        StageSpec("side", "transform", pure=True),
        StageSpec("z1", "transform", pure=True),
        StageSpec("final", "aggregate", depends_on=("side", "z1")),
    ]
    projection = project_task_graph(
        stages, "final", graph_kind="logical_dag", graph_id="tamper-bypass",
    )
    analysis = analyze_task_graph_structure(
        projection,
        [_duration("side", 1), _duration("z1", 5), _duration("final", 1)],
    )
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=1,
            max_ready_bypass=1,
        ),
    )
    tampered = json.loads(json.dumps(report))
    scenario = tampered["policy_results"][1]["scenarios"][1]
    side = next(
        record for record in scenario["stage_records"]
        if record["stage_id"] == "side"
    )
    side["bypass_count"] = 2
    scenario["max_ready_bypass"] = 2
    policy_summary = tampered["policy_results"][1]["summary"]
    policy_summary["max_ready_bypass"] = 2
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphOrderingError, match="bypass bound"):
        validate_ordering_simulation(tampered)


def test_g4_2_validation_rejects_rehashed_false_comparison():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection,
        analysis,
        build_ordering_simulation_profile(
            max_concurrency=2,
            max_ready_bypass=100,
        ),
    )
    tampered = json.loads(json.dumps(report))
    comparison = tampered["comparison"]
    comparison["expected_improvement_ms"] = 99
    comparison_base = {
        key: value for key, value in comparison.items()
        if key != "comparison_digest"
    }
    comparison["comparison_digest"] = _digest(comparison_base)
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphOrderingError, match="comparison"):
        validate_ordering_simulation(tampered)


def test_g4_2_report_validation_rejects_forbidden_fields():
    projection, analysis = _benefit_fixture()
    report = simulate_task_graph_ordering(
        projection, analysis, build_ordering_simulation_profile(),
    )
    tampered = json.loads(json.dumps(report))
    tampered["summary"]["path"] = "private"

    with pytest.raises(TaskGraphOrderingError, match="forbidden"):
        validate_ordering_simulation(tampered)
