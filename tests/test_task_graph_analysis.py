import hashlib
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec
from task_graph_analysis import (
    TaskGraphAnalysisError,
    analyze_task_graph_structure,
    build_stage_duration_estimate,
    validate_stage_duration_estimate,
    validate_structure_analysis,
)
from task_graph_optimization import project_task_graph


def _digest(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _duration(
    stage_id,
    expected,
    *,
    lower=None,
    upper=None,
    source="stage_type_baseline",
    sample_count=0,
):
    return build_stage_duration_estimate(
        stage_id,
        estimate_source=source,
        duration_lower_ms=expected if lower is None else lower,
        duration_expected_ms=expected,
        duration_upper_ms=expected if upper is None else upper,
        sample_count=sample_count,
        profile_version="task-stage-baseline.v1",
    )


def _diamond_fixture():
    stages = [
        StageSpec("start", "transform", pure=True),
        StageSpec("left", "transform", depends_on=("start",), pure=True),
        StageSpec("right", "transform", depends_on=("start",), pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("left", "right"),
        ),
    ]
    projection = project_task_graph(
        stages, "final", graph_kind="logical_dag", graph_id="g4-diamond",
    )
    contracts = [
        _duration("start", 1, lower=1, upper=2),
        _duration("left", 5, lower=4, upper=7),
        _duration("right", 2, lower=1, upper=3),
        _duration("final", 1, lower=1, upper=2),
    ]
    return projection, contracts


def _stage(report, stage_id):
    return next(
        item for item in report["stage_analysis"]
        if item["stage_id"] == stage_id
    )


def _timing(report, stage_id):
    return next(
        item for item in report["stage_timings"]
        if item["stage_id"] == stage_id
    )


def test_g4_1_diamond_dominator_post_dominator_and_critical_path():
    projection, contracts = _diamond_fixture()
    original = json.loads(json.dumps(projection))

    report = analyze_task_graph_structure(projection, contracts)

    assert projection == original
    assert report["status"] == "evaluated"
    assert report["selected_graph_kind"] == "logical_dag"
    assert report["selected_ordering_policy"] == "topological"
    assert report["fallback"] == {
        "used": False,
        "reason_code": "analysis_ready",
    }
    assert report["virtual_nodes"] == {
        "entry_node_id": "stage:start",
        "exit_node_id": "stage:final",
        "entry_stage_ids": ["start"],
        "exit_stage_ids": ["final"],
        "entry_virtual": False,
        "exit_virtual": False,
    }
    assert _stage(report, "final")["dominator_node_ids"] == [
        "stage:final", "stage:start",
    ]
    assert _stage(report, "final")["immediate_dominator_node_id"] == (
        "stage:start"
    )
    assert _stage(report, "start")["post_dominator_node_ids"] == [
        "stage:final", "stage:start",
    ]
    assert _stage(report, "start")[
        "immediate_post_dominator_node_id"
    ] == "stage:final"
    assert report["critical_path"] == {
        "available": True,
        "stage_ids": ["start", "left", "final"],
        "duration_lower_ms": 6,
        "duration_expected_ms": 7,
        "duration_upper_ms": 11,
        "path_digest": report["critical_path"]["path_digest"],
    }
    assert _timing(report, "final")["earliest_start_expected_ms"] == 6
    assert _timing(report, "start")["remaining_expected_ms"] == 7
    assert report["summary"] == {
        "stage_count": 4,
        "edge_count": 4,
        "entry_count": 1,
        "exit_count": 1,
        "entry_virtual_count": 0,
        "exit_virtual_count": 0,
        "dominator_tree_edge_count": 3,
        "post_dominator_tree_edge_count": 3,
        "timing_stage_count": 4,
        "critical_path_stage_count": 3,
        "critical_path_expected_ms": 7,
    }
    assert validate_structure_analysis(report) == report


def test_g4_1_multi_entry_and_exit_use_analysis_only_virtual_nodes():
    stages = [
        StageSpec("left", "transform", pure=True),
        StageSpec("right", "transform", pure=True),
        StageSpec(
            "join", "aggregate", depends_on=("left", "right"), pure=True,
        ),
        StageSpec("done", "transform", depends_on=("join",), pure=True),
        StageSpec("orphan", "audit", pure=True),
    ]
    projection = project_task_graph(
        stages, "done", graph_kind="logical_dag", graph_id="multi-root",
    )
    contracts = [
        _duration("left", 2),
        _duration("right", 3),
        _duration("join", 4),
        _duration("done", 1),
        _duration("orphan", 20),
    ]

    report = analyze_task_graph_structure(projection, contracts)

    assert report["virtual_nodes"] == {
        "entry_node_id": "virtual:entry",
        "exit_node_id": "virtual:exit",
        "entry_stage_ids": ["left", "orphan", "right"],
        "exit_stage_ids": ["done", "orphan"],
        "entry_virtual": True,
        "exit_virtual": True,
    }
    assert _stage(report, "join")["dominator_node_ids"] == [
        "stage:join", "virtual:entry",
    ]
    assert _stage(report, "join")["immediate_dominator_node_id"] == (
        "virtual:entry"
    )
    assert _stage(report, "join")["post_dominator_node_ids"] == [
        "stage:done", "stage:join", "virtual:exit",
    ]
    assert report["critical_path"]["stage_ids"] == ["orphan"]
    assert report["critical_path"]["duration_expected_ms"] == 20
    assert all(
        node["node_id"] not in {"virtual:entry", "virtual:exit"}
        for node in projection["nodes"]
    )


def test_g4_1_equal_critical_branches_choose_lexicographically_stable_path():
    projection, contracts = _diamond_fixture()
    contracts = [
        _duration("start", 1),
        _duration("left", 5),
        _duration("right", 5),
        _duration("final", 1),
    ]

    first = analyze_task_graph_structure(projection, contracts)
    second = analyze_task_graph_structure(projection, list(reversed(contracts)))

    assert first == second
    assert first["critical_path"]["stage_ids"] == ["start", "left", "final"]


def test_g4_1_missing_duration_contract_keeps_structure_and_falls_back():
    projection, contracts = _diamond_fixture()

    report = analyze_task_graph_structure(projection, contracts[:-1])

    assert report["status"] == "partial"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "duration_contract_coverage_mismatch",
    }
    assert len(report["stage_analysis"]) == 4
    assert len(report["dominator_tree_edges"]) == 3
    assert report["stage_timings"] == []
    assert report["critical_path"]["available"] is False
    assert report["critical_path"]["stage_ids"] == []
    assert report["selected_ordering_policy"] == "topological"
    assert validate_structure_analysis(report) == report


def test_g4_1_invalid_or_sensitive_duration_input_is_redacted_partial_result():
    projection, contracts = _diamond_fixture()
    invalid = dict(contracts[0])
    invalid["duration_lower_ms"] = 100
    invalid["duration_expected_ms"] = 10
    sensitive = {**contracts[1], "body": "private historical samples"}

    invalid_report = analyze_task_graph_structure(
        projection, [invalid, *contracts[1:]],
    )
    sensitive_report = analyze_task_graph_structure(
        projection, [contracts[0], sensitive, *contracts[2:]],
    )

    assert invalid_report["fallback"]["reason_code"] == (
        "duration_contract_invalid"
    )
    assert sensitive_report["fallback"]["reason_code"] == (
        "duration_contract_invalid"
    )
    assert "private historical samples" not in json.dumps(sensitive_report)
    assert sensitive_report["stage_timings"] == []


@pytest.mark.parametrize(
    ("source", "sample_count", "valid"),
    [
        ("stage_type_baseline", 0, True),
        ("operator_profile", 0, True),
        ("historical_interval", 3, True),
        ("historical_interval", 2, False),
        ("stage_type_baseline", 1, False),
    ],
)
def test_g4_1_duration_source_sample_contract(source, sample_count, valid):
    kwargs = {
        "stage_id": "stage",
        "estimate_source": source,
        "duration_lower_ms": 10,
        "duration_expected_ms": 20,
        "duration_upper_ms": 30,
        "sample_count": sample_count,
        "profile_version": "profile.v1",
    }
    if valid:
        contract = build_stage_duration_estimate(**kwargs)
        assert validate_stage_duration_estimate(contract) == contract
    else:
        with pytest.raises(TaskGraphAnalysisError):
            build_stage_duration_estimate(**kwargs)


def test_g4_1_contract_rejects_interval_digest_and_unsafe_fields():
    with pytest.raises(TaskGraphAnalysisError, match="interval"):
        build_stage_duration_estimate(
            "stage",
            estimate_source="operator_profile",
            duration_lower_ms=30,
            duration_expected_ms=20,
            duration_upper_ms=40,
        )

    contract = _duration("stage", 20, lower=10, upper=30)
    tampered = dict(contract, duration_expected_ms=25)
    with pytest.raises(TaskGraphAnalysisError, match="digest"):
        validate_stage_duration_estimate(tampered)

    forbidden = {**contract, "body": "private"}
    with pytest.raises(TaskGraphAnalysisError, match="fields"):
        validate_stage_duration_estimate(forbidden)


def test_g4_1_rejects_non_logical_graph_types():
    projection, contracts = _diamond_fixture()
    projection = dict(projection)
    projection["graph_kind"] = "analysis_view"
    projection["digest"] = _digest({
        key: value for key, value in projection.items() if key != "digest"
    })

    with pytest.raises(Exception, match="cannot be used"):
        analyze_task_graph_structure(projection, contracts)


def test_g4_1_report_is_digest_only_and_contract_order_independent():
    projection, contracts = _diamond_fixture()
    first = analyze_task_graph_structure(projection, contracts)
    second = analyze_task_graph_structure(projection, list(reversed(contracts)))

    assert first == second
    rendered = json.dumps(first)
    assert "task-stage-baseline.v1" not in rendered
    assert "stage_type_baseline" not in rendered
    assert "root_input" not in rendered


def test_g4_1_validation_rejects_tree_policy_and_summary_tampering():
    projection, contracts = _diamond_fixture()
    report = analyze_task_graph_structure(projection, contracts)

    wrong_policy = json.loads(json.dumps(report))
    wrong_policy["selected_ordering_policy"] = "critical_path"
    with pytest.raises(TaskGraphAnalysisError, match="Scheduler"):
        validate_structure_analysis(wrong_policy)

    wrong_tree = json.loads(json.dumps(report))
    wrong_tree["dominator_tree_edges"][0]["parent_node_id"] = "stage:right"
    with pytest.raises(TaskGraphAnalysisError, match="dominator tree"):
        validate_structure_analysis(wrong_tree)

    wrong_summary = json.loads(json.dumps(report))
    wrong_summary["summary"]["critical_path_stage_count"] += 1
    with pytest.raises(TaskGraphAnalysisError, match="summary"):
        validate_structure_analysis(wrong_summary)


def test_g4_1_validation_rejects_rehashed_false_critical_path():
    projection, contracts = _diamond_fixture()
    report = analyze_task_graph_structure(projection, contracts)
    tampered = json.loads(json.dumps(report))
    critical = tampered["critical_path"]
    critical["stage_ids"] = ["left", "start", "final"]
    critical_base = {
        key: value for key, value in critical.items() if key != "path_digest"
    }
    critical["path_digest"] = _digest(critical_base)
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphAnalysisError, match="start at an entry"):
        validate_structure_analysis(tampered)


def test_g4_1_random_dag_dominator_sets_match_all_path_intersections():
    def all_paths(predecessors, source, target):
        if source == target:
            return [[source]]
        paths = []
        for parent in predecessors[target]:
            for prefix in all_paths(predecessors, source, parent):
                paths.append([*prefix, target])
        return paths

    for seed in range(12):
        rng = random.Random(seed)
        dependencies = {f"n{index}": set() for index in range(7)}
        for index in range(1, 7):
            dependencies[f"n{index}"].add(f"n{index - 1}")
            for source in range(index - 1):
                if rng.random() < 0.35:
                    dependencies[f"n{index}"].add(f"n{source}")
        stages = [
            StageSpec(
                stage_id,
                "transform",
                depends_on=tuple(sorted(dependencies[stage_id])),
                pure=True,
            )
            for stage_id in dependencies
        ]
        projection = project_task_graph(
            stages, "n6", graph_kind="logical_dag", graph_id=f"g4-random-{seed}",
        )
        report = analyze_task_graph_structure(
            projection,
            [_duration(stage.stage_id, 1) for stage in stages],
        )
        successors = {stage_id: set() for stage_id in dependencies}
        for target, parents in dependencies.items():
            for parent in parents:
                successors[parent].add(target)

        for stage_id in dependencies:
            paths_from_entry = all_paths(dependencies, "n0", stage_id)
            expected_dominators = set.intersection(*(
                set(path) for path in paths_from_entry
            ))

            def paths_to_exit(current):
                if current == "n6":
                    return [[current]]
                return [
                    [current, *suffix]
                    for child in successors[current]
                    for suffix in paths_to_exit(child)
                ]

            expected_post = set.intersection(*(
                set(path) for path in paths_to_exit(stage_id)
            ))
            item = _stage(report, stage_id)
            assert set(item["dominator_node_ids"]) == {
                f"stage:{value}" for value in expected_dominators
            }
            assert set(item["post_dominator_node_ids"]) == {
                f"stage:{value}" for value in expected_post
            }


def test_g4_1_report_validation_rejects_forbidden_fields():
    projection, contracts = _diamond_fixture()
    report = analyze_task_graph_structure(projection, contracts)
    tampered = json.loads(json.dumps(report))
    tampered["summary"]["path"] = "private"

    with pytest.raises(TaskGraphAnalysisError, match="forbidden"):
        validate_structure_analysis(tampered)
