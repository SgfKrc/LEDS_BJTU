import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import StageSpec, dual_candidate_template
from task_graph_optimization import project_task_graph
from task_graph_sharing import (
    TaskGraphSharingError,
    analyze_stage_sharing,
    build_stage_semantics_contract,
    digest_stage_input_references,
    validate_share_analysis,
    validate_stage_semantics_contract,
)
from task_provider import ModelIdentity


def _sha(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _contract(
    stage_id,
    *,
    input_signature=None,
    input_schema="input.v1",
    output_schema="output.v1",
    operation_config=None,
    data_scope="workflow:wf1",
    side_effect="none",
    determinism="deterministic",
    share_policy="allow",
):
    return build_stage_semantics_contract(
        stage_id,
        input_signature_sha256=input_signature or _sha("same-input"),
        input_schema_version=input_schema,
        output_schema_version=output_schema,
        operation_config_sha256=operation_config or _sha("same-config"),
        data_scope=data_scope,
        side_effect_class=side_effect,
        determinism=determinism,
        share_policy=share_policy,
    )


def _duplicate_projection(
    *,
    left_provider="local",
    right_provider="local",
    left_pure=True,
    left_identity=None,
    right_identity=None,
):
    stages = [
        StageSpec(
            "left",
            "transform",
            provider=left_provider,
            pure=left_pure,
            model_identity=left_identity,
        ),
        StageSpec(
            "right",
            "transform",
            provider=right_provider,
            pure=True,
            model_identity=right_identity,
        ),
        StageSpec(
            "final",
            "aggregate",
            depends_on=("left", "right"),
            provider="local",
        ),
    ]
    return project_task_graph(
        stages,
        "final",
        graph_kind="logical_dag",
        graph_id="sharing-fixture",
    )


def _contracts(*, left=None, right=None):
    return [
        left or _contract("left"),
        right or _contract("right"),
        _contract("final", share_policy="deny"),
    ]


def _stage_result(analysis, stage_id):
    return next(
        item for item in analysis["stage_fingerprints"]
        if item["stage_id"] == stage_id
    )


def _pair_result(analysis, left, right):
    left, right = sorted((left, right))
    return next(
        item for item in analysis["pair_decisions"]
        if item["left_stage_id"] == left and item["right_stage_id"] == right
    )


def test_g3_1_input_reference_digest_is_canonical_and_body_free():
    first = {
        "ref_id": "dependency:source:content",
        "schema_version": "payload.v1",
        "content_sha256": _sha("private-body-a"),
    }
    second = {
        "ref_id": "root:request:settings",
        "schema_version": "settings.v1",
        "content_sha256": _sha("private-body-b"),
    }

    assert digest_stage_input_references([first, second]) == (
        digest_stage_input_references([second, first])
    )
    assert digest_stage_input_references([]) == hashlib.sha256(b"[]").hexdigest()
    with pytest.raises(TaskGraphSharingError, match="unique"):
        digest_stage_input_references([first, first])
    with pytest.raises(TaskGraphSharingError, match="fields"):
        digest_stage_input_references([{**first, "body": "private-body-a"}])


def test_g3_1_identical_pure_transforms_share_a_stage_independent_fingerprint():
    projection = _duplicate_projection()
    original = json.loads(json.dumps(projection))
    contracts = _contracts()

    first = analyze_stage_sharing(projection, contracts)
    second = analyze_stage_sharing(projection, list(reversed(contracts)))

    assert first == second
    assert projection == original
    left = _stage_result(first, "left")
    right = _stage_result(first, "right")
    assert left["eligible"] is True
    assert right["eligible"] is True
    assert left["fingerprint_sha256"] == right["fingerprint_sha256"]
    assert left["contract_digest"] != right["contract_digest"]
    assert _pair_result(first, "left", "right") == {
        "left_stage_id": "left",
        "right_stage_id": "right",
        "shareable": True,
        "reason_code": "identical_stage_fingerprint",
        "fingerprint_sha256": left["fingerprint_sha256"],
    }
    assert first["summary"] == {
        "stage_count": 3,
        "eligible_stage_count": 2,
        "pair_count": 3,
        "shareable_pair_count": 1,
    }
    assert validate_share_analysis(first) == first


@pytest.mark.parametrize(
    ("right_contract", "reason_code"),
    [
        (_contract("right", input_signature=_sha("other-input")),
         "input_signature_mismatch"),
        (_contract("right", input_schema="input.v2"),
         "schema_version_mismatch"),
        (_contract("right", operation_config=_sha("other-config")),
         "operation_config_mismatch"),
        (_contract("right", data_scope="workflow:wf2"),
         "data_scope_mismatch"),
    ],
)
def test_g3_1_semantic_contract_differences_reject_sharing(
    right_contract,
    reason_code,
):
    analysis = analyze_stage_sharing(
        _duplicate_projection(),
        _contracts(right=right_contract),
    )

    decision = _pair_result(analysis, "left", "right")
    assert decision["shareable"] is False
    assert decision["reason_code"] == reason_code
    assert decision["fingerprint_sha256"] == ""


def test_g3_1_provider_and_locked_model_identity_are_fingerprint_dimensions():
    provider_analysis = analyze_stage_sharing(
        _duplicate_projection(right_provider="remote"),
        _contracts(),
    )
    assert _pair_result(provider_analysis, "left", "right")["reason_code"] == (
        "provider_requirements_mismatch"
    )

    left_identity = ModelIdentity(
        model_id="transform-a",
        engine="pytorch",
        format="safetensors",
        revision="rev1",
        sha256="a" * 64,
    )
    right_identity = ModelIdentity(
        model_id="transform-b",
        engine="pytorch",
        format="safetensors",
        revision="rev1",
        sha256="b" * 64,
    )
    model_analysis = analyze_stage_sharing(
        _duplicate_projection(
            left_identity=left_identity,
            right_identity=right_identity,
        ),
        _contracts(),
    )
    assert _pair_result(model_analysis, "left", "right")["reason_code"] == (
        "model_identity_mismatch"
    )


@pytest.mark.parametrize(
    ("projection_kwargs", "contract_kwargs", "reason_code"),
    [
        ({"left_pure": False}, {}, "stage_not_pure"),
        ({}, {"side_effect": "external_mutation"}, "side_effect_not_none"),
        ({}, {"determinism": "seeded"}, "determinism_not_deterministic"),
        ({}, {"share_policy": "deny"}, "share_policy_denied"),
        ({}, {"share_policy": "independent"}, "independent_result_required"),
    ],
)
def test_g3_1_eligibility_is_fail_closed(
    projection_kwargs,
    contract_kwargs,
    reason_code,
):
    analysis = analyze_stage_sharing(
        _duplicate_projection(**projection_kwargs),
        _contracts(left=_contract("left", **contract_kwargs)),
    )

    left = _stage_result(analysis, "left")
    assert left["eligible"] is False
    assert left["reason_code"] == reason_code
    assert _pair_result(analysis, "left", "right")["reason_code"] == (
        "stage_ineligible"
    )


def test_g3_1_random_llm_candidates_remain_independent_and_ineligible():
    stages, final_stage_id = dual_candidate_template()
    projection = project_task_graph(
        stages,
        final_stage_id,
        graph_kind="logical_dag",
        graph_id="dual-candidate-sharing",
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
    analysis = analyze_stage_sharing(projection, contracts)

    assert analysis["summary"]["eligible_stage_count"] == 0
    assert analysis["summary"]["shareable_pair_count"] == 0
    assert _pair_result(
        analysis, "candidate_a", "candidate_b",
    )["reason_code"] == "stage_ineligible"


def test_g3_1_partial_model_identity_is_ineligible():
    projection = project_task_graph(
        [
            {
                "stage_id": "transform",
                "stage_type": "transform",
                "provider": "local",
                "pure": True,
                "model_identity": {"model_id": "unlocked-model"},
            },
        ],
        "transform",
        graph_kind="logical_dag",
        graph_id="partial-model-identity",
    )
    analysis = analyze_stage_sharing(projection, [_contract("transform")])

    assert _stage_result(analysis, "transform")["reason_code"] == (
        "model_identity_incomplete"
    )


def test_g3_1_contract_set_and_contract_digest_are_strict():
    projection = _duplicate_projection()
    contract = _contract("left")
    tampered = json.loads(json.dumps(contract))
    tampered["operation_config_sha256"] = _sha("tampered-config")

    with pytest.raises(TaskGraphSharingError, match="digest mismatch"):
        validate_stage_semantics_contract(tampered)
    with pytest.raises(TaskGraphSharingError, match="match projected Stage IDs"):
        analyze_stage_sharing(projection, [_contract("left"), _contract("right")])
    with pytest.raises(TaskGraphSharingError, match="must be unique"):
        analyze_stage_sharing(
            projection,
            _contracts() + [_contract("left")],
        )


def test_g3_1_report_rejects_tampering_and_contains_only_safe_digests():
    analysis = analyze_stage_sharing(_duplicate_projection(), _contracts())
    rendered = json.dumps(analysis)
    assert "same-input" not in rendered
    assert "same-config" not in rendered
    assert "workflow:wf1" not in rendered
    assert "private-body" not in rendered

    contradictory = json.loads(json.dumps(analysis))
    pair = _pair_result(contradictory, "left", "right")
    pair["shareable"] = False
    pair["fingerprint_sha256"] = ""
    pair["reason_code"] = "fingerprint_mismatch"
    with pytest.raises(TaskGraphSharingError, match="contradicts"):
        validate_share_analysis(contradictory)

    forbidden = json.loads(json.dumps(analysis))
    forbidden["stage_fingerprints"][0]["path"] = "private"
    with pytest.raises(TaskGraphSharingError, match="forbidden"):
        validate_share_analysis(forbidden)
