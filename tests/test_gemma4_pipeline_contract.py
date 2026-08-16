from __future__ import annotations

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gemma4_pipeline_contract import (  # noqa: E402
    Gemma4PipelineContractError,
    build_gemma4_pipeline_contract,
    validate_gemma4_pipeline_contract,
)


def _segment(index, start, end, *, total=4):
    return {
        "node_id": f"node-{index}",
        "layer_range": [start, end],
        "has_embedding": start == 0,
        "has_lm_head": end == total,
        "assignment_manifest_sha256": f"{index + 1:064x}",
        "required_bytes": 1024,
        "execution_device": "cpu",
        "dtype": "float32",
    }


def _contract(segments=None):
    return build_gemma4_pipeline_contract(
        config_id="config-g4",
        plan_id="plan-g4",
        generation=1,
        model_id="gemma4",
        model_sha256="a" * 64,
        total_layers=4,
        hidden_size=3,
        layer_types=[
            "full_attention", "sliding_attention",
            "full_attention", "sliding_attention",
        ],
        num_kv_shared_layers=2,
        segments=segments or [_segment(0, 0, 2), _segment(1, 2, 4)],
    )


def test_contract_freezes_shared_kv_sources_and_cross_segment_handoff():
    contract = _contract()

    assert contract["model_type"] == "gemma4_unified"
    assert contract["shared_kv_source_layers"] == {
        "full_attention": 0,
        "sliding_attention": 1,
    }
    assert contract["segments"][0]["produces_shared_kv_types"] == [
        "full_attention", "sliding_attention",
    ]
    assert contract["segments"][1]["requires_shared_kv_types"] == [
        "full_attention", "sliding_attention",
    ]
    assert contract["handoffs"][0]["shared_kv_types"] == [
        "full_attention", "sliding_attention",
    ]
    assert contract["runtime_environment"] == ".venv-gemma4-pipeline"
    assert contract["production_admitted"] is False
    assert validate_gemma4_pipeline_contract(contract) == contract


def test_three_segment_contract_tracks_source_and_consumer_boundaries():
    contract = _contract([
        _segment(0, 0, 1),
        _segment(1, 1, 3),
        _segment(2, 3, 4),
    ])

    assert contract["handoffs"][0]["shared_kv_types"] == ["full_attention"]
    assert contract["handoffs"][1]["shared_kv_types"] == ["sliding_attention"]


def test_contract_rejects_missing_shared_kv_source_and_sensitive_fields():
    with pytest.raises(Gemma4PipelineContractError, match="no source layer"):
        build_gemma4_pipeline_contract(
            config_id="config-g4",
            plan_id="plan-g4",
            generation=1,
            model_id="gemma4",
            model_sha256="a" * 64,
            total_layers=4,
            hidden_size=3,
            layer_types=["full_attention"] * 2 + ["sliding_attention"] * 2,
            num_kv_shared_layers=2,
            segments=[_segment(0, 0, 2), _segment(1, 2, 4)],
        )
    segments = [_segment(0, 0, 2), _segment(1, 2, 4)]
    segments[0]["model_path"] = "G:/models/private"
    with pytest.raises(Gemma4PipelineContractError, match="model_path"):
        _contract(segments)


def test_contract_digest_and_derived_shared_kv_fields_are_fail_closed():
    contract = _contract()
    tampered = copy.deepcopy(contract)
    tampered["segments"][1]["requires_shared_kv_types"] = []
    with pytest.raises(Gemma4PipelineContractError, match="digest mismatch"):
        validate_gemma4_pipeline_contract(tampered)

    noncanonical = copy.deepcopy(contract)
    noncanonical["segments"][1]["requires_shared_kv_types"] = []
    payload = {key: value for key, value in noncanonical.items() if key != "contract_sha256"}
    import hashlib
    import json

    noncanonical["contract_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    with pytest.raises(Gemma4PipelineContractError, match="not canonical"):
        validate_gemma4_pipeline_contract(noncanonical)
