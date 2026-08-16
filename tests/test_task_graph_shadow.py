import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import task_graph_shadow
from task_graph import (
    StageSpec,
    dual_candidate_template,
    image_prompt_sd15_template,
)
from task_graph_payloads import TaskPayloadStore
from task_graph_shadow import (
    TaskGraphShadowError,
    release_shadow_payloads,
    run_task_graph_shadow,
    validate_shadow_report,
)
from task_provider import ModelIdentity


def _fanout_stages():
    return [
        StageSpec("shared", "transform", provider="local", pure=True),
        StageSpec("left", "transform", depends_on=("shared",), provider="local"),
        StageSpec("right", "transform", depends_on=("shared",), provider="local"),
        StageSpec(
            "final", "aggregate", depends_on=("left", "right"), provider="local",
        ),
    ]


def test_g2_3_dual_candidate_fixed_template_is_stable_noop():
    stages, final_stage_id = dual_candidate_template()
    first = run_task_graph_shadow("dual_candidate", stages, final_stage_id)
    second = run_task_graph_shadow("dual_candidate", stages, final_stage_id)

    assert first == second
    assert first["status"] == "evaluated"
    assert first["selected_graph_kind"] == "logical_dag"
    assert first["fallback"] == {
        "used": False,
        "reason_code": "shadow_execution_unchanged",
    }
    assert first["metrics"]["logical_stage_count"] == 3
    assert first["metrics"]["optimized_stage_count"] == 3
    assert first["metrics"]["logical_edge_count"] == 2
    assert first["metrics"]["optimized_edge_count"] == 2
    assert first["metrics"]["payload_plan_count"] == 0
    assert validate_shadow_report(first) == first


def test_g2_3_llm_sd15_binding_boundary_remains_noop():
    stages, final_stage_id = image_prompt_sd15_template(
        text_provider_id="local_text",
        image_provider_id="local_image",
        text_model_identity=ModelIdentity(
            model_id="qwen-test",
            engine="pytorch",
            format="safetensors",
            revision="rev1",
            sha256="a" * 64,
        ),
    )
    report = run_task_graph_shadow("llm_sd15_v1", stages, final_stage_id)

    assert report["status"] == "evaluated"
    assert report["metrics"]["reduced_edge_count"] == 0
    assert report["metrics"]["payload_plan_count"] == 0
    assert report["metrics"]["logical_edge_count"] == 1
    assert report["metrics"]["optimized_edge_count"] == 1


def test_g2_3_image_grid_fixed_shape_allows_dynamic_providers():
    seed_ids = [f"seed_{index}" for index in range(4)]
    stages = [
        StageSpec(stage_id, "image_generate", provider=f"worker-{index}", pure=True)
        for index, stage_id in enumerate(seed_ids)
    ]
    stages.append(
        StageSpec(
            "image_grid",
            "image_grid",
            depends_on=tuple(seed_ids),
            provider="grid-worker",
        )
    )
    report = run_task_graph_shadow("image_grid_v1", stages, "image_grid")

    assert report["status"] == "evaluated"
    assert report["metrics"]["logical_stage_count"] == 5
    assert report["metrics"]["logical_edge_count"] == 4


def test_g2_3_fanout_shadow_binds_once_for_two_consumers(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    body = b"shared-payload"
    report = run_task_graph_shadow(
        "g2_fanout_shadow_v1",
        _fanout_stages(),
        "final",
        payloads={"shared": body},
        payload_store=store,
        data_scope="workflow:wf1",
    )

    assert report["status"] == "evaluated"
    assert report["metrics"]["payload_plan_count"] == 1
    assert report["metrics"]["payload_bound_count"] == 1
    assert report["metrics"]["payload_source_bytes"] == len(body)
    assert report["metrics"]["avoided_inline_bytes"] == len(body)
    assert report["metrics"]["reference_contract_bytes"] > 0
    assert store.stats()["object_count"] == 1
    reference = report["payload_references"][0]
    for consumer in ("left", "right"):
        with store.materialize(
            reference,
            consumer_stage_id=consumer,
            data_scope="workflow:wf1",
        ) as local_path:
            assert local_path.read_bytes() == body
    release_shadow_payloads(store, report)
    assert store.stats()["object_count"] == 0


def test_g2_3_fanout_report_is_root_independent_and_contains_no_body(tmp_path):
    body = b"private-shared-body"
    first = run_task_graph_shadow(
        "g2_fanout_shadow_v1",
        _fanout_stages(),
        "final",
        payloads={"shared": body},
        payload_store=TaskPayloadStore(tmp_path / "first"),
        data_scope="workflow:wf1",
    )
    second = run_task_graph_shadow(
        "g2_fanout_shadow_v1",
        _fanout_stages(),
        "final",
        payloads={"shared": body},
        payload_store=TaskPayloadStore(tmp_path / "second"),
        data_scope="workflow:wf1",
    )

    assert first == second
    rendered = json.dumps(first)
    assert "private-shared-body" not in rendered
    assert str(tmp_path) not in rendered


def test_g2_3_unadmitted_template_and_rule_fall_back_to_logical_graph():
    stages, final_stage_id = dual_candidate_template()
    unsupported = run_task_graph_shadow("custom_runtime", stages, final_stage_id)
    rule_rejected = run_task_graph_shadow(
        "dual_candidate",
        stages,
        final_stage_id,
        rules=("semantic_transitive_reduction",),
    )

    assert unsupported["status"] == "fallback"
    assert unsupported["fallback"]["reason_code"] == "template_not_admitted"
    assert unsupported["logical_graph_digest"] == ""
    assert rule_rejected["status"] == "fallback"
    assert rule_rejected["fallback"]["reason_code"] == "rule_not_admitted"
    assert rule_rejected["logical_graph_digest"]
    assert rule_rejected["selected_graph_kind"] == "logical_dag"

    shape_rejected = run_task_graph_shadow(
        "dual_candidate",
        _fanout_stages(),
        "final",
    )
    assert shape_rejected["status"] == "fallback"
    assert shape_rejected["fallback"]["reason_code"] == "template_shape_not_admitted"


def test_g2_3_projection_and_optimizer_failures_are_sanitized_fallbacks(monkeypatch):
    projection_failed = run_task_graph_shadow(
        "dual_candidate",
        [StageSpec("final", "aggregate", depends_on=("missing",))],
        "final",
    )

    def fail_optimizer(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("private optimizer details must not escape")

    monkeypatch.setattr(task_graph_shadow, "optimize_task_graph", fail_optimizer)
    stages, final_stage_id = dual_candidate_template()
    optimization_failed = run_task_graph_shadow(
        "dual_candidate", stages, final_stage_id,
    )

    assert projection_failed["fallback"]["reason_code"] == "projection_failed"
    assert optimization_failed["fallback"]["reason_code"] == "optimization_failed"
    assert "private optimizer" not in json.dumps(optimization_failed)


def test_g2_3_payload_contract_failures_fall_back_without_runtime_selection(tmp_path):
    store = TaskPayloadStore(tmp_path / "payloads")
    unplanned = run_task_graph_shadow(
        "g2_fanout_shadow_v1",
        _fanout_stages(),
        "final",
        payloads={"left": b"wrong source"},
        payload_store=store,
        data_scope="workflow:wf1",
    )
    invalid_scope = run_task_graph_shadow(
        "g2_fanout_shadow_v1",
        _fanout_stages(),
        "final",
        payloads={"shared": b"body"},
        payload_store=store,
        data_scope="../escape",
    )

    assert unplanned["fallback"]["reason_code"] == "payload_source_not_planned"
    assert invalid_scope["fallback"]["reason_code"] == "payload_binding_failed"
    assert unplanned["selected_graph_kind"] == "logical_dag"
    assert invalid_scope["selected_graph_kind"] == "logical_dag"
    assert store.stats()["object_count"] == 0


def test_g2_3_report_validation_rejects_forbidden_or_tampered_fields():
    stages, final_stage_id = dual_candidate_template()
    report = run_task_graph_shadow("dual_candidate", stages, final_stage_id)
    forbidden = json.loads(json.dumps(report))
    forbidden["metrics"]["path"] = "private"
    with pytest.raises(TaskGraphShadowError, match="forbidden"):
        validate_shadow_report(forbidden)

    tampered = json.loads(json.dumps(report))
    tampered["metrics"]["logical_stage_count"] += 1
    with pytest.raises(TaskGraphShadowError, match="digest"):
        validate_shadow_report(tampered)

    missing_candidate = json.loads(json.dumps(report))
    missing_candidate["metrics"]["candidate_available"] = False
    with pytest.raises(TaskGraphShadowError, match="requires a candidate"):
        validate_shadow_report(missing_candidate)

    fallback = run_task_graph_shadow("custom_runtime", stages, final_stage_id)
    leaked_candidate = json.loads(json.dumps(fallback))
    leaked_candidate["optimized_graph_digest"] = "a" * 64
    with pytest.raises(TaskGraphShadowError, match="cannot expose candidate state"):
        validate_shadow_report(leaked_candidate)
