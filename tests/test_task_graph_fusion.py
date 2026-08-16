import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, dual_candidate_template
from task_graph_fusion import (
    TaskGraphFusionError,
    build_stage_fusion_contract,
    fuse_transform_chains,
    validate_fusion_candidate,
    validate_stage_fusion_contract,
)
from task_graph_optimization import project_task_graph, validate_projection
from task_graph_sharing import build_stage_semantics_contract
from task_provider import ModelIdentity


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _semantics(
    stage_id,
    *,
    input_schema="payload.v1",
    output_schema="payload.v1",
    data_scope="workflow:wf1",
    determinism="deterministic",
    share_policy="allow",
):
    return build_stage_semantics_contract(
        stage_id,
        input_signature_sha256=_sha(f"input:{stage_id}"),
        input_schema_version=input_schema,
        output_schema_version=output_schema,
        operation_config_sha256=_sha(f"operation:{stage_id}"),
        data_scope=data_scope,
        side_effect_class="none",
        determinism=determinism,
        share_policy=share_policy,
    )


def _fusion(stage_id, **overrides):
    return build_stage_fusion_contract(stage_id, **overrides)


def _linear_fixture(*, providers=("local", "local", "local")):
    stages = [
        StageSpec("decode", "transform", provider=providers[0], pure=True),
        StageSpec(
            "normalize",
            "transform",
            depends_on=("decode",),
            provider=providers[1],
            pure=True,
        ),
        StageSpec(
            "encode",
            "transform",
            depends_on=("normalize",),
            provider=providers[2],
            pure=True,
        ),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("encode",),
            provider="local",
        ),
    ]
    projection = project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="linear-fusion-fixture",
    )
    semantics = [
        _semantics("decode", input_schema="raw.v1", output_schema="decoded.v1"),
        _semantics(
            "normalize",
            input_schema="decoded.v1",
            output_schema="normalized.v1",
        ),
        _semantics(
            "encode",
            input_schema="normalized.v1",
            output_schema="encoded.v1",
        ),
        _semantics("final", share_policy="deny"),
    ]
    fusion = [_fusion(stage.stage_id) for stage in stages]
    return projection, semantics, fusion


def _nodes(report):
    return {
        node["stage_id"]: node
        for node in report["candidate_graph"]["nodes"]
    }


def _replace_semantics(contracts, stage_id, **changes):
    return [
        _semantics(stage_id, **changes)
        if contract["stage_id"] == stage_id else contract
        for contract in contracts
    ]


def _replace_fusion(contracts, stage_id, **changes):
    return [
        _fusion(stage_id, **changes)
        if contract["stage_id"] == stage_id else contract
        for contract in contracts
    ]


def test_g3_3_linear_transform_chain_builds_one_shadow_fused_stage():
    projection, semantics, fusion = _linear_fixture()
    original = json.loads(json.dumps(projection))

    report = fuse_transform_chains(projection, semantics, fusion)

    assert projection == original
    assert report["status"] == "evaluated"
    assert report["selected_graph_kind"] == "logical_dag"
    assert report["fallback"] == {
        "used": False,
        "reason_code": "fusion_candidate_ready",
    }
    group = report["fusion_groups"][0]
    assert group["source_stage_ids"] == ["decode", "normalize", "encode"]
    assert group["fused_stage_id"] == f"fused:{group['chain_digest']}"
    assert report["summary"] == {
        "logical_stage_count": 4,
        "optimized_stage_count": 2,
        "logical_edge_count": 3,
        "optimized_edge_count": 1,
        "fusion_group_count": 1,
        "fused_source_stage_count": 3,
        "preserved_boundary_count": 3,
        "rejected_boundary_count": 0,
    }
    nodes = _nodes(report)
    assert set(nodes) == {group["fused_stage_id"], "final"}
    assert nodes[group["fused_stage_id"]]["stage_type"] == "transform_fused"
    assert nodes["final"]["depends_on"] == [group["fused_stage_id"]]
    assert report["candidate_graph"]["graph_kind"] == "optimized_dag"
    assert validate_projection(report["candidate_graph"]) == report[
        "candidate_graph"
    ]
    assert validate_fusion_candidate(report) == report


def test_g3_3_boundary_map_preserves_each_source_stage_attribution():
    projection, semantics, fusion = _linear_fixture()
    report = fuse_transform_chains(projection, semantics, fusion)
    group = report["fusion_groups"][0]

    assert [item["source_stage_id"] for item in report["boundary_map"]] == [
        "decode", "normalize", "encode",
    ]
    for ordinal, item in enumerate(report["boundary_map"]):
        assert item["fused_stage_id"] == group["fused_stage_id"]
        assert item["ordinal"] == ordinal
        for field_name, prefix in (
            ("failure_boundary_id", "failure"),
            ("cancellation_boundary_id", "cancellation"),
            ("logging_boundary_id", "logging"),
            ("accounting_boundary_id", "accounting"),
        ):
            assert item[field_name] == (
                f"{prefix}:{group['chain_digest']}:{ordinal}"
            )


def test_g3_3_final_transform_is_rebound_to_the_fused_stage():
    stages = [
        StageSpec("first", "transform", pure=True),
        StageSpec("second", "transform", depends_on=("first",), pure=True),
    ]
    projection = project_task_graph(
        stages, "second", graph_kind="logical_dag", graph_id="fused-final",
    )
    semantics = [
        _semantics("first", input_schema="a.v1", output_schema="b.v1"),
        _semantics("second", input_schema="b.v1", output_schema="c.v1"),
    ]
    report = fuse_transform_chains(
        projection,
        semantics,
        [_fusion("first"), _fusion("second")],
    )

    fused_id = report["fusion_groups"][0]["fused_stage_id"]
    assert report["candidate_graph"]["summary"]["final_stage_id"] == fused_id
    assert report["candidate_graph"]["summary"]["stage_count"] == 1


def test_g3_3_contract_and_candidate_are_stable_and_body_free():
    projection, semantics, fusion = _linear_fixture()
    first = fuse_transform_chains(projection, semantics, fusion)
    second = fuse_transform_chains(
        projection,
        list(reversed(semantics)),
        list(reversed(fusion)),
    )

    assert first == second
    rendered = json.dumps(first)
    assert "workflow:wf1" not in rendered
    assert "operation:decode" not in rendered
    assert "root_input" not in rendered

    contract = _fusion("decode")
    assert validate_stage_fusion_contract(contract) == contract
    tampered = dict(contract, fusion_policy="deny")
    with pytest.raises(TaskGraphFusionError, match="digest"):
        validate_stage_fusion_contract(tampered)


def test_g3_3_fan_out_boundary_rejects_fusion():
    stages = [
        StageSpec("source", "transform", pure=True),
        StageSpec("left", "transform", depends_on=("source",), pure=True),
        StageSpec("right", "transform", depends_on=("source",), pure=True),
        StageSpec("final", "aggregate", depends_on=("left", "right")),
    ]
    projection = project_task_graph(
        stages, "final", graph_kind="logical_dag", graph_id="fan-out",
    )
    semantics = [_semantics(stage.stage_id) for stage in stages]
    report = fuse_transform_chains(
        projection,
        semantics,
        [_fusion(stage.stage_id) for stage in stages],
    )

    assert report["status"] == "no_op"
    assert report["fusion_groups"] == []
    assert {item["reason_code"] for item in report["rejections"]} == {
        "fan_out_boundary",
    }
    assert report["summary"]["logical_stage_count"] == 4
    assert report["summary"]["optimized_stage_count"] == 4


def test_g3_3_fan_in_boundary_rejects_fusion():
    stages = [
        StageSpec("left", "transform", pure=True),
        StageSpec("right", "transform", pure=True),
        StageSpec(
            "join", "transform", depends_on=("left", "right"), pure=True,
        ),
        StageSpec("final", "aggregate", depends_on=("join",)),
    ]
    projection = project_task_graph(
        stages, "final", graph_kind="logical_dag", graph_id="fan-in",
    )
    semantics = [_semantics(stage.stage_id) for stage in stages]
    report = fuse_transform_chains(
        projection,
        semantics,
        [_fusion(stage.stage_id) for stage in stages],
    )

    assert report["status"] == "no_op"
    assert {item["reason_code"] for item in report["rejections"]} == {
        "fan_in_boundary",
    }


def test_g3_3_bound_input_is_an_explicit_fusion_boundary():
    stages = [
        StageSpec("source", "transform", pure=True),
        StageSpec(
            "target",
            "transform",
            depends_on=("source",),
            input_bindings={"value": ("source", "content")},
            pure=True,
        ),
    ]
    projection = project_task_graph(
        stages, "target", graph_kind="logical_dag", graph_id="bound-input",
    )
    semantics = [_semantics(stage.stage_id) for stage in stages]
    report = fuse_transform_chains(
        projection,
        semantics,
        [_fusion(stage.stage_id) for stage in stages],
    )

    assert report["status"] == "no_op"
    assert report["rejections"][0]["reason_code"] == "binding_boundary"


@pytest.mark.parametrize(
    ("case", "reason_code"),
    [
        ("provider", "provider_boundary"),
        ("model", "model_boundary"),
        ("scope", "data_scope_boundary"),
        ("schema", "schema_boundary"),
    ],
)
def test_g3_3_provider_model_scope_and_schema_switches_are_boundaries(
    case, reason_code,
):
    identity_a = ModelIdentity(
        model_id="model-a",
        engine="pytorch",
        format="safetensors",
        revision="main",
        sha256=_sha("model-a"),
    )
    identity_b = ModelIdentity(
        model_id="model-b",
        engine="pytorch",
        format="safetensors",
        revision="main",
        sha256=_sha("model-b"),
    )
    stages = [
        StageSpec(
            "first",
            "transform",
            provider="local",
            pure=True,
            model_identity=identity_a if case == "model" else None,
        ),
        StageSpec(
            "second",
            "transform",
            depends_on=("first",),
            provider="remote" if case == "provider" else "local",
            pure=True,
            model_identity=identity_b if case == "model" else None,
        ),
    ]
    projection = project_task_graph(
        stages, "second", graph_kind="logical_dag", graph_id=f"{case}-boundary",
    )
    semantics = [
        _semantics("first", input_schema="a.v1", output_schema="b.v1"),
        _semantics(
            "second",
            input_schema="wrong.v1" if case == "schema" else "b.v1",
            output_schema="c.v1",
            data_scope="workflow:wf2" if case == "scope" else "workflow:wf1",
        ),
    ]
    report = fuse_transform_chains(
        projection, semantics, [_fusion("first"), _fusion("second")],
    )

    assert report["status"] == "no_op"
    assert report["rejections"][0]["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("changes", "reason_code"),
    [
        ({"fusion_policy": "deny"}, "fusion_not_allowed"),
        ({"barrier_class": "checkpoint"}, "checkpoint_boundary"),
        ({"barrier_class": "commit"}, "commit_boundary"),
        ({"barrier_class": "billing"}, "billing_boundary"),
        ({"failure_mapping": "unavailable"}, "failure_mapping_unavailable"),
        (
            {"cancellation_mapping": "unavailable"},
            "cancellation_mapping_unavailable",
        ),
        ({"logging_mapping": "unavailable"}, "logging_mapping_unavailable"),
        (
            {"accounting_mapping": "unavailable"},
            "accounting_mapping_unavailable",
        ),
    ],
)
def test_g3_3_contract_barriers_and_missing_mappings_reject_fusion(
    changes, reason_code,
):
    projection, semantics, fusion = _linear_fixture()
    fusion = _replace_fusion(fusion, "normalize", **changes)

    report = fuse_transform_chains(projection, semantics, fusion)

    assert report["status"] == "no_op"
    assert report["fusion_groups"] == []
    assert {item["reason_code"] for item in report["rejections"]} == {
        reason_code,
    }


def test_g3_3_nondeterministic_llm_graph_remains_a_noop_candidate():
    stages, final_stage_id = dual_candidate_template()
    projection = project_task_graph(
        stages,
        final_stage_id,
        graph_kind="logical_dag",
        graph_id="llm-no-fusion",
    )
    semantics = [
        _semantics(
            "candidate_a",
            determinism="nondeterministic",
            share_policy="independent",
        ),
        _semantics(
            "candidate_b",
            determinism="nondeterministic",
            share_policy="independent",
        ),
        _semantics("aggregate", share_policy="deny"),
    ]
    report = fuse_transform_chains(
        projection,
        semantics,
        [_fusion(stage.stage_id) for stage in stages],
    )

    assert report["status"] == "no_op"
    assert report["fallback"] == {
        "used": True,
        "reason_code": "no_fusible_chain",
    }
    assert report["fusion_groups"] == []
    assert report["boundary_map"] == []
    assert report["rejections"] == []


def test_g3_3_validation_rejects_boundary_and_selection_tampering():
    projection, semantics, fusion = _linear_fixture()
    report = fuse_transform_chains(projection, semantics, fusion)

    wrong_selection = json.loads(json.dumps(report))
    wrong_selection["selected_graph_kind"] = "optimized_dag"
    with pytest.raises(TaskGraphFusionError, match="cannot select"):
        validate_fusion_candidate(wrong_selection)

    wrong_boundary = json.loads(json.dumps(report))
    wrong_boundary["boundary_map"][0]["source_stage_id"] = "other"
    with pytest.raises(TaskGraphFusionError, match="boundary map"):
        validate_fusion_candidate(wrong_boundary)

    wrong_summary = json.loads(json.dumps(report))
    wrong_summary["summary"]["optimized_stage_count"] += 1
    with pytest.raises(TaskGraphFusionError, match="summary"):
        validate_fusion_candidate(wrong_summary)


def test_g3_3_validation_rejects_forbidden_and_rehashed_false_trace():
    projection, semantics, fusion = _linear_fixture()
    report = fuse_transform_chains(projection, semantics, fusion)

    forbidden = json.loads(json.dumps(report))
    forbidden["candidate_graph"]["summary"]["path"] = "private"
    with pytest.raises(TaskGraphFusionError, match="forbidden"):
        validate_fusion_candidate(forbidden)

    tampered = json.loads(json.dumps(report))
    event = tampered["candidate_graph"]["trace"][-1]
    event["accepted"] = False
    event["reason_code"] = "no_fusible_chain"
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

    with pytest.raises(TaskGraphFusionError, match="trace"):
        validate_fusion_candidate(tampered)
