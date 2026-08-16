import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity


MIB = 1024 * 1024


def descriptor(layer_sizes=(100, 100, 100, 100)):
    return {
        "model_id": "synthetic-qwen2",
        "model_type": "qwen2",
        "model_sha256": "a" * 64,
        "total_layers": len(layer_sizes),
        "layer_weight_bytes": [value * MIB for value in layer_sizes],
        "component_weight_bytes": {
            "embedding": 40 * MIB,
            "final_norm": 5 * MIB,
            "lm_head": 40 * MIB,
            "visual": 0,
            "mtp": 0,
            "other": 0,
        },
    }


def node(node_id, capacity_mb, *, role="client", score=10):
    return {
        "node_id": node_id,
        "role": role,
        "capacity_bytes": capacity_mb * MIB,
        "reserve_bytes": 10 * MIB,
        "runtime_multiplier": 1.0,
        "score": score,
        "execution_device": "cuda",
        "capacity_source": "test",
    }


def test_aggregate_capacity_admits_when_no_single_node_fits():
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 80, role="master", score=100), node("worker-a", 300), node("worker-b", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    assert plan["aggregate_only"] is True
    assert plan["single_node_full_model_candidates"] == []
    assert sum(item["layers_count"] for item in plan["assignments"]) == 4
    assert plan["assignments"][0]["has_embedding"] is True
    assert plan["assignments"][-1]["has_lm_head"] is True
    assert "master" in plan["control_only_nodes"]


def test_capacity_failure_returns_no_partial_assignment():
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 180, role="master"), node("worker", 180)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_cluster_capacity_insufficient"
    assert plan["assignments"] == []


def test_cpu_runtime_multiplier_is_charged_to_required_bytes():
    cuda = node("cuda", 600)
    cpu = node("cpu", 600)
    cpu["runtime_multiplier"] = 2.0
    plan = solve_pipeline_capacity(descriptor((100,)), [cuda, cpu], safety_margin=1.0)

    assert plan["admitted"] is True
    assert plan["assignments"][0]["node_id"] == "cuda"
    assert plan["assignments"][0]["required_bytes"] == 195 * MIB


@pytest.mark.parametrize("component", ["visual", "mtp", "multimodal"])
def test_component_that_requires_an_unimplemented_runtime_is_rejected(component):
    item = descriptor((100,))
    item["component_weight_bytes"][component] = 20 * MIB

    with pytest.raises(PipelineCapacityError, match=component):
        solve_pipeline_capacity(item, [node("worker", 500)])


def test_capacity_plan_id_is_stable_for_same_inputs():
    nodes = [node("worker-a", 300), node("worker-b", 300)]
    first = solve_pipeline_capacity(descriptor(), nodes, safety_margin=1.0)
    second = solve_pipeline_capacity(descriptor(), list(reversed(nodes)), safety_margin=1.0)

    assert first["plan_id"] == second["plan_id"]


def test_tied_embedding_is_charged_to_output_capacity():
    item = descriptor((100,))
    item["tie_word_embeddings"] = True
    item["component_weight_bytes"]["lm_head"] = 0
    plan = solve_pipeline_capacity(
        item,
        [node("worker", 250)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    assert plan["raw_model_bytes"] == 185 * MIB
    assert plan["assignments"][0]["raw_weight_bytes"] == 185 * MIB
