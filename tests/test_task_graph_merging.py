import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, dual_candidate_template
from task_graph_merging import (
    TaskGraphMergeError,
    merge_shareable_stages,
    validate_merge_candidate,
)
from task_graph_optimization import project_task_graph, validate_projection
from task_graph_sharing import build_stage_semantics_contract


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _contract(
    stage_id,
    *,
    input_signature=None,
    operation_config=None,
    data_scope="workflow:wf1",
    determinism="deterministic",
    share_policy="allow",
):
    return build_stage_semantics_contract(
        stage_id,
        input_signature_sha256=input_signature or _sha("same-input"),
        input_schema_version="input.v1",
        output_schema_version="output.v1",
        operation_config_sha256=operation_config or _sha("same-config"),
        data_scope=data_scope,
        side_effect_class="none",
        determinism=determinism,
        share_policy=share_policy,
    )


def _safe_merge_fixture():
    stages = [
        StageSpec("left", "transform", provider="local", pure=True),
        StageSpec("right", "transform", provider="local", pure=True),
        StageSpec(
            "consume_left",
            "transform",
            depends_on=("left",),
            provider="local",
            pure=True,
        ),
        StageSpec(
            "consume_right",
            "transform",
            depends_on=("right",),
            provider="local",
            pure=True,
        ),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("consume_left", "consume_right"),
            provider="local",
        ),
    ]
    contracts = [
        _contract("left"),
        _contract("right"),
        _contract("consume_left", input_signature=_sha("left-output")),
        _contract("consume_right", input_signature=_sha("right-output")),
        _contract("final", share_policy="deny"),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="safe-common-subtask",
    )
    return projection, contracts


def _join_fixture(*, bound_left=False):
    bindings = {"first": ("left", "content")} if bound_left else {}
    stages = [
        StageSpec("left", "transform", provider="local", pure=True),
        StageSpec("right", "transform", provider="local", pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("left", "right"),
            provider="local",
            input_bindings=bindings,
        ),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="unsafe-common-subtask",
    )
    contracts = [
        _contract("left"),
        _contract("right"),
        _contract("final", share_policy="deny"),
    ]
    return projection, contracts


def _nodes_by_stage(report):
    return {
        node["stage_id"]: node
        for node in report["candidate_graph"]["nodes"]
    }


def test_g3_2_safe_common_subtask_merge_rebinds_consumers_and_provenance():
    projection, contracts = _safe_merge_fixture()
    original = json.loads(json.dumps(projection))

    report = merge_shareable_stages(projection, contracts)

    assert projection == original
    assert report["status"] == "evaluated"
    assert report["selected_graph_kind"] == "logical_dag"
    assert report["fallback"] == {
        "used": False,
        "reason_code": "merge_candidate_ready",
    }
    assert report["summary"] == {
        "logical_stage_count": 5,
        "optimized_stage_count": 4,
        "logical_edge_count": 4,
        "optimized_edge_count": 4,
        "merge_group_count": 1,
        "merged_source_stage_count": 2,
        "rejected_group_count": 0,
    }
    group = report["merge_groups"][0]
    assert group["source_stage_ids"] == ["left", "right"]
    assert group["merged_stage_id"] == f"shared:{group['fingerprint_sha256']}"
    assert report["provenance"] == [{
        "merged_stage_id": group["merged_stage_id"],
        "source_stage_ids": ["left", "right"],
    }]

    nodes = _nodes_by_stage(report)
    assert "left" not in nodes
    assert "right" not in nodes
    assert group["merged_stage_id"] in nodes
    assert nodes["consume_left"]["depends_on"] == [group["merged_stage_id"]]
    assert nodes["consume_right"]["depends_on"] == [group["merged_stage_id"]]
    assert report["candidate_graph"]["graph_kind"] == "optimized_dag"
    assert validate_projection(report["candidate_graph"]) == report["candidate_graph"]
    assert validate_merge_candidate(report) == report


def test_g3_2_candidate_is_stable_across_contract_order_and_body_free():
    projection, contracts = _safe_merge_fixture()
    first = merge_shareable_stages(projection, contracts)
    second = merge_shareable_stages(projection, list(reversed(contracts)))

    assert first == second
    rendered = json.dumps(first)
    assert "same-input" not in rendered
    assert "same-config" not in rendered
    assert "workflow:wf1" not in rendered
    assert "root_input" not in rendered


def test_g3_2_join_arity_change_falls_back_to_noop_candidate():
    projection, contracts = _join_fixture()
    report = merge_shareable_stages(projection, contracts)

    assert report["status"] == "fallback"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "join_arity_would_change",
    }
    assert report["merge_groups"] == []
    assert report["provenance"] == []
    assert report["rejections"][0]["source_stage_ids"] == ["left", "right"]
    assert report["rejections"][0]["reason_code"] == "join_arity_would_change"
    assert report["summary"]["logical_stage_count"] == 3
    assert report["summary"]["optimized_stage_count"] == 3
    assert report["summary"]["logical_edge_count"] == 2
    assert report["summary"]["optimized_edge_count"] == 2
    assert set(_nodes_by_stage(report)) == {"left", "right", "final"}
    assert report["candidate_graph"]["trace"][-1]["reason_code"] == (
        "join_arity_would_change"
    )


def test_g3_2_different_edge_contracts_fail_before_join_collapse():
    projection, contracts = _join_fixture(bound_left=True)
    report = merge_shareable_stages(projection, contracts)

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "semantic_contract_collision"
    assert report["rejections"][0]["reason_code"] == (
        "semantic_contract_collision"
    )


def test_g3_2_random_llm_candidates_produce_a_noop_shadow_graph():
    stages, final_stage_id = dual_candidate_template()
    projection = project_task_graph(
        stages,
        final_stage_id,
        graph_kind="logical_dag",
        graph_id="dual-candidate-no-merge",
    )
    contracts = [
        _contract(
            "candidate_a",
            determinism="nondeterministic",
            share_policy="independent",
        ),
        _contract(
            "candidate_b",
            determinism="nondeterministic",
            share_policy="independent",
        ),
        _contract("aggregate", share_policy="deny"),
    ]
    report = merge_shareable_stages(projection, contracts)

    assert report["status"] == "no_op"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "no_shareable_pair",
    }
    assert report["merge_groups"] == []
    assert report["summary"]["logical_stage_count"] == 3
    assert report["summary"]["optimized_stage_count"] == 3
    assert set(_nodes_by_stage(report)) == {
        "candidate_a", "candidate_b", "aggregate",
    }


def test_g3_2_three_identical_sources_collapse_to_one_candidate_stage():
    stages = [
        StageSpec(stage_id, "transform", provider="local", pure=True)
        for stage_id in ("source_a", "source_b", "source_c")
    ]
    stages.extend([
        StageSpec("use_a", "transform", depends_on=("source_a",), pure=True),
        StageSpec("use_b", "transform", depends_on=("source_b",), pure=True),
        StageSpec("use_c", "transform", depends_on=("source_c",), pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("use_a", "use_b", "use_c"),
        ),
    ])
    contracts = [
        _contract(stage_id)
        for stage_id in ("source_a", "source_b", "source_c")
    ] + [
        _contract("use_a", input_signature=_sha("use-a")),
        _contract("use_b", input_signature=_sha("use-b")),
        _contract("use_c", input_signature=_sha("use-c")),
        _contract("final", share_policy="deny"),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="three-way-common-subtask",
    )
    report = merge_shareable_stages(projection, contracts)

    assert report["status"] == "evaluated"
    assert report["merge_groups"][0]["source_stage_ids"] == [
        "source_a", "source_b", "source_c",
    ]
    assert report["summary"]["logical_stage_count"] == 7
    assert report["summary"]["optimized_stage_count"] == 5
    assert report["summary"]["merged_source_stage_count"] == 3


def test_g3_2_existing_synthetic_stage_id_forces_collision_fallback():
    base_projection, base_contracts = _safe_merge_fixture()
    base_report = merge_shareable_stages(base_projection, base_contracts)
    collision_stage_id = base_report["merge_groups"][0]["merged_stage_id"]
    stages = [
        StageSpec("left", "transform", provider="local", pure=True),
        StageSpec("right", "transform", provider="local", pure=True),
        StageSpec("consume_left", "transform", depends_on=("left",), pure=True),
        StageSpec("consume_right", "transform", depends_on=("right",), pure=True),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("consume_left", "consume_right"),
        ),
        StageSpec(collision_stage_id, "audit", provider="local"),
    ]
    contracts = base_contracts + [
        _contract(collision_stage_id, share_policy="deny"),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="shared-stage-id-collision",
    )
    report = merge_shareable_stages(projection, contracts)

    assert report["status"] == "fallback"
    assert report["fallback"]["reason_code"] == "shared_stage_id_collision"
    assert report["merge_groups"] == []
    assert collision_stage_id in _nodes_by_stage(report)


def test_g3_2_report_validation_rejects_provenance_and_selection_tampering():
    projection, contracts = _safe_merge_fixture()
    report = merge_shareable_stages(projection, contracts)

    wrong_selection = json.loads(json.dumps(report))
    wrong_selection["selected_graph_kind"] = "optimized_dag"
    with pytest.raises(TaskGraphMergeError, match="cannot select"):
        validate_merge_candidate(wrong_selection)

    wrong_provenance = json.loads(json.dumps(report))
    wrong_provenance["provenance"][0]["source_stage_ids"] = ["left", "other"]
    with pytest.raises(TaskGraphMergeError, match="provenance"):
        validate_merge_candidate(wrong_provenance)

    contradictory = json.loads(json.dumps(report))
    contradictory["summary"]["optimized_stage_count"] += 1
    with pytest.raises(TaskGraphMergeError, match="summary"):
        validate_merge_candidate(contradictory)


def test_g3_2_report_validation_rejects_forbidden_or_tampered_candidate_graph():
    projection, contracts = _safe_merge_fixture()
    report = merge_shareable_stages(projection, contracts)

    forbidden = json.loads(json.dumps(report))
    forbidden["candidate_graph"]["summary"]["path"] = "private"
    with pytest.raises(TaskGraphMergeError, match="forbidden"):
        validate_merge_candidate(forbidden)

    tampered = json.loads(json.dumps(report))
    tampered["candidate_graph"]["summary"]["edge_count"] += 1
    with pytest.raises(Exception, match="digest"):
        validate_merge_candidate(tampered)


def test_g3_2_report_validation_rejects_rehashed_false_merge_trace():
    projection, contracts = _safe_merge_fixture()
    report = merge_shareable_stages(projection, contracts)
    tampered = json.loads(json.dumps(report))
    event = tampered["candidate_graph"]["trace"][-1]
    event["accepted"] = False
    event["reason_code"] = "no_shareable_pair"
    candidate_unsigned = {
        key: value for key, value in tampered["candidate_graph"].items()
        if key != "digest"
    }
    tampered["candidate_graph"]["digest"] = _digest(candidate_unsigned)
    tampered["optimized_graph_digest"] = tampered["candidate_graph"]["digest"]
    report_unsigned = {
        key: value for key, value in tampered.items() if key != "digest"
    }
    tampered["digest"] = _digest(report_unsigned)

    with pytest.raises(TaskGraphMergeError, match="trace"):
        validate_merge_candidate(tampered)
