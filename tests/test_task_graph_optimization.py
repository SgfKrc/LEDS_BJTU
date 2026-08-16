import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, TaskGraphCoordinator
from task_graph_optimization import (
    GraphTypeError,
    TaskGraphOptimizationError,
    TaskGraphProjectionError,
    optimize_task_graph,
    project_task_graph,
    project_workflow_snapshot,
    require_graph_kind,
    validate_optimization_result,
    validate_projection,
)


def _diamond_stages():
    return [
        StageSpec(
            "shared_input",
            "prepare",
            provider="local_prepare",
            pure=True,
            root_input_overrides={"secret_prompt": "must never be projected"},
        ),
        StageSpec(
            "candidate_a",
            "full_inference",
            depends_on=("shared_input",),
            provider="local_full_model",
        ),
        StageSpec(
            "candidate_b",
            "full_inference",
            depends_on=("shared_input",),
            provider="local_full_model",
        ),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
            input_bindings={"best": ("candidate_a", "content")},
        ),
    ]


def test_logical_projection_preserves_diamond_without_payload_body():
    projection = project_task_graph(
        _diamond_stages(),
        "aggregate",
        graph_id="diamond_v1",
    )

    assert projection["graph_kind"] == "logical_dag"
    assert {node["node_id"] for node in projection["nodes"]} == {
        "stage:shared_input",
        "stage:candidate_a",
        "stage:candidate_b",
        "stage:aggregate",
    }
    assert {
        (edge["source_node_id"], edge["target_node_id"])
        for edge in projection["edges"]
    } == {
        ("stage:shared_input", "stage:candidate_a"),
        ("stage:shared_input", "stage:candidate_b"),
        ("stage:candidate_a", "stage:aggregate"),
        ("stage:candidate_b", "stage:aggregate"),
    }
    aggregate = next(
        node for node in projection["nodes"] if node["stage_id"] == "aggregate"
    )
    assert aggregate["input_bindings"] == [{
        "target_key": "best",
        "dependency_stage_id": "candidate_a",
        "output_key": "content",
    }]
    assert "must never be projected" not in json.dumps(projection)
    assert "root_input_overrides" not in json.dumps(projection)


def test_projection_digest_and_trace_are_deterministic():
    trace = [{
        "rule": "cull_unreachable",
        "reason_code": "shadow_only",
        "affected_node_ids": ["stage:candidate_b", "stage:candidate_a"],
        "accepted": False,
    }]

    first = project_task_graph(
        _diamond_stages(), "aggregate", graph_kind="optimized_dag", trace=trace,
    )
    second = project_task_graph(
        _diamond_stages(), "aggregate", graph_kind="optimized_dag", trace=trace,
    )

    assert first["digest"] == second["digest"]
    assert first["trace"][0]["affected_node_ids"] == [
        "stage:candidate_a", "stage:candidate_b",
    ]
    assert validate_projection(first) == first


def test_analysis_view_is_a_separate_typed_structural_projection():
    projection = project_task_graph(
        _diamond_stages(),
        "aggregate",
        graph_kind="analysis_view",
        graph_id="diamond_analysis_v1",
    )

    assert projection["graph_kind"] == "analysis_view"
    assert {node["node_kind"] for node in projection["nodes"]} == {"stage"}
    with pytest.raises(GraphTypeError, match="cannot be used"):
        require_graph_kind(projection, "optimized_dag")


def test_attempt_projection_uses_winner_summary_without_errors_or_output_body():
    coordinator = TaskGraphCoordinator()

    def execute(stage, dependencies, root_input, cancel_event):
        assert not cancel_event.is_set()
        return {"content": f"private-output-{stage.stage_id}"}

    _, workflow = coordinator.run(
        [StageSpec("only", "full_inference")],
        "only",
        {"prompt": "private-root-input"},
        execute,
        workflow_id="wf_projection1",
    )
    projection = project_workflow_snapshot(workflow)
    coordinator.close()

    attempt_nodes = [
        node for node in projection["nodes"] if node["node_kind"] == "attempt"
    ]
    assert len(attempt_nodes) == 1
    assert attempt_nodes[0]["is_winner"] is True
    assert attempt_nodes[0]["result_digest_present"] is True
    rendered = json.dumps(projection)
    assert "private-output" not in rendered
    assert "private-root-input" not in rendered
    assert '"error"' not in rendered
    assert '"result_metadata"' not in rendered


def test_graph_types_have_distinct_namespaces_and_reject_mixing():
    logical = project_task_graph(_diamond_stages(), "aggregate")
    topology = project_task_graph(
        _diamond_stages(), "aggregate", graph_kind="provider_topology",
    )

    assert all(node["node_id"].startswith("provider:") for node in topology["nodes"])
    assert topology["edges"] == []
    with pytest.raises(GraphTypeError, match="cannot be used"):
        require_graph_kind(logical, "attempt_graph")

    mixed = json.loads(json.dumps(logical))
    mixed["nodes"][0]["node_kind"] = "provider"
    mixed["nodes"][0]["node_id"] = "provider:local_full_model"
    with pytest.raises(TaskGraphProjectionError):
        validate_projection(mixed)


def test_projection_fails_closed_for_unsafe_trace_and_body_fields():
    with pytest.raises(TaskGraphProjectionError, match="unsupported fields"):
        project_task_graph(
            _diamond_stages(),
            "aggregate",
            trace=[{
                "rule": "cull",
                "reason_code": "test",
                "affected_node_ids": [],
                "accepted": False,
                "detail": "do not store raw optimizer notes",
            }],
        )

    projection = project_task_graph(_diamond_stages(), "aggregate")
    projection["summary"]["output"] = "unsafe body"
    with pytest.raises(TaskGraphProjectionError, match="forbidden field"):
        validate_projection(projection)


def test_shadow_optimizer_culls_only_pure_unreachable_stages():
    stages = [
        *_diamond_stages(),
        StageSpec("unused_pure", "transform", pure=True),
    ]
    logical = project_task_graph(stages, "aggregate", graph_id="shadow_cull")
    result = optimize_task_graph(logical)

    assert result["mode"] == "shadow"
    assert result["summary"]["culled_stage_count"] == 1
    assert {
        node["stage_id"] for node in result["optimized_graph"]["nodes"]
    } == {"shared_input", "candidate_a", "candidate_b", "aggregate"}
    cull_events = [event for event in result["trace"] if event["rule"] == "cull_unreachable"]
    assert cull_events == [{
        "rule": "cull_unreachable",
        "reason_code": "pure_unreachable_only",
        "affected_node_ids": ["stage:unused_pure"],
        "accepted": True,
    }]
    assert validate_optimization_result(result) == result


def test_shadow_optimizer_preserves_unreachable_side_effect_boundary():
    stages = [
        *_diamond_stages(),
        StageSpec("audit_side_effect", "audit", pure=False),
    ]
    result = optimize_task_graph(
        project_task_graph(stages, "aggregate", graph_id="shadow_side_effect"),
    )

    assert result["summary"]["culled_stage_count"] == 0
    assert {
        node["stage_id"] for node in result["optimized_graph"]["nodes"]
    } == {
        "shared_input", "candidate_a", "candidate_b", "aggregate",
        "audit_side_effect",
    }
    assert any(
        event["rule"] == "cull_unreachable"
        and event["reason_code"] == "side_effect_boundary"
        and event["accepted"] is False
        for event in result["trace"]
    )


def test_shadow_optimizer_plans_one_immutable_payload_for_diamond_fanout():
    result = optimize_task_graph(
        project_task_graph(_diamond_stages(), "aggregate", graph_id="shadow_payload"),
    )

    assert result["payload_plan"] == [{
        "payload_ref": "payload:shared_input",
        "source_stage_id": "shared_input",
        "target_stage_ids": ["candidate_a", "candidate_b"],
        "reason_code": "immutable_fanout_source",
    }]
    assert result["summary"]["payload_share_candidate_count"] == 1
    assert any(
        event["rule"] == "share_payload" and event["accepted"] is True
        for event in result["trace"]
    )


def test_shadow_optimizer_is_deterministic_and_rejects_runtime_mode():
    logical = project_task_graph(_diamond_stages(), "aggregate")
    first = optimize_task_graph(logical)
    second = optimize_task_graph(logical)

    assert first["digest"] == second["digest"]
    with pytest.raises(TaskGraphOptimizationError, match="shadow"):
        optimize_task_graph(logical, mode="runtime")


def _transitive_stages(*, bound_direct_edge=False, minimum_successful=None):
    return [
        StageSpec("source", "transform", pure=True),
        StageSpec(
            "middle", "transform", depends_on=("source",), pure=True,
        ),
        StageSpec(
            "final",
            "transform",
            depends_on=("source", "middle"),
            pure=True,
            minimum_successful_dependencies=minimum_successful,
            input_bindings=(
                {"direct": ("source", "content")}
                if bound_direct_edge else {}
            ),
        ),
    ]


def test_g2_semantic_transitive_reduction_is_disabled_by_default():
    logical = project_task_graph(
        _transitive_stages(), "final", graph_id="g2_default_off",
    )
    result = optimize_task_graph(logical)

    assert result["enabled_rules"] == []
    assert result["summary"]["semantic_reduction_enabled"] is False
    assert result["summary"]["reduced_edge_count"] == 0
    assert len(result["optimized_graph"]["edges"]) == 3
    assert any(
        event["rule"] == "semantic_transitive_reduction"
        and event["reason_code"] == "disabled_by_default"
        for event in result["trace"]
    )


def test_g2_reduces_only_redundant_pure_dependency_and_keeps_fallback():
    logical = project_task_graph(
        _transitive_stages(), "final", graph_id="g2_safe_reduction",
    )
    result = optimize_task_graph(
        logical, rules=("semantic_transitive_reduction",),
    )

    assert result["enabled_rules"] == ["semantic_transitive_reduction"]
    assert result["summary"]["reduced_edge_count"] == 1
    assert {
        (edge["source_node_id"], edge["target_node_id"])
        for edge in result["optimized_graph"]["edges"]
    } == {
        ("stage:source", "stage:middle"),
        ("stage:middle", "stage:final"),
    }
    final = next(
        node for node in result["optimized_graph"]["nodes"]
        if node["stage_id"] == "final"
    )
    assert final["depends_on"] == ["middle"]
    assert final["execution_constraints"]["minimum_successful_dependencies"] == 1
    assert len(result["logical_graph"]["edges"]) == 3
    assert result["fallback"] == {
        "available": True,
        "graph_kind": "logical_dag",
        "reason_code": "shadow_only_candidate",
    }


def test_g2_preserves_direct_edge_with_unique_input_binding():
    logical = project_task_graph(
        _transitive_stages(bound_direct_edge=True),
        "final",
        graph_id="g2_binding_boundary",
    )
    result = optimize_task_graph(
        logical, rules=("semantic_transitive_reduction",),
    )

    assert result["summary"]["reduced_edge_count"] == 0
    assert len(result["optimized_graph"]["edges"]) == 3
    direct = next(
        edge for edge in result["optimized_graph"]["edges"]
        if edge["source_node_id"] == "stage:source"
        and edge["target_node_id"] == "stage:final"
    )
    assert direct["semantic_contract"]["kind"] == "bound_input"
    assert direct["semantic_contract"]["binding_targets"] == ["direct"]


def test_g2_preserves_partial_join_and_rejects_unknown_rule():
    logical = project_task_graph(
        _transitive_stages(minimum_successful=1),
        "final",
        graph_id="g2_partial_boundary",
    )
    result = optimize_task_graph(
        logical, rules=("semantic_transitive_reduction",),
    )

    assert result["summary"]["reduced_edge_count"] == 0
    assert len(result["optimized_graph"]["edges"]) == 3
    with pytest.raises(TaskGraphOptimizationError, match="unsupported"):
        optimize_task_graph(logical, rules=("unsafe_runtime_rewrite",))


def test_g2_edge_semantic_contract_rejects_bound_pure_dependency():
    projection = project_task_graph(_transitive_stages(), "final")
    projection["edges"][0]["semantic_contract"]["binding_targets"] = ["content"]

    with pytest.raises(TaskGraphProjectionError, match="cannot carry"):
        validate_projection(projection)


def test_g2_random_pure_dags_preserve_reachability_after_reduction():
    def reachable_pairs(projection):
        adjacency = {}
        for edge in projection["edges"]:
            adjacency.setdefault(edge["source_node_id"], set()).add(
                edge["target_node_id"],
            )
        pairs = set()
        for source in adjacency:
            pending = list(adjacency[source])
            visited = set()
            while pending:
                target = pending.pop()
                if target in visited:
                    continue
                visited.add(target)
                pairs.add((source, target))
                pending.extend(adjacency.get(target, ()))
        return pairs

    for seed in range(20):
        rng = random.Random(seed)
        dependencies = {f"n{index}": set() for index in range(8)}
        for index in range(1, 8):
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
            for stage_id in sorted(dependencies, key=lambda item: int(item[1:]))
        ]
        logical = project_task_graph(stages, "n7", graph_id=f"g2_random_{seed}")
        optimized = optimize_task_graph(
            logical, rules=("semantic_transitive_reduction",),
        )["optimized_graph"]

        assert reachable_pairs(optimized) == reachable_pairs(logical)
        assert len(optimized["edges"]) <= len(logical["edges"])
