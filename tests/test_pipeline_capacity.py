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


def test_declared_layer_budget_caps_that_node_layers():
    """设备自荐的层数上限必须被分配器尊重 —— 它是"本地裁层后能承载的上限"，
    不是"当前工件已就绪的区间"，所以分配器可以给它任意连续区间，但不能超过层数。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 2,
                    "local_cut": True,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["layers_count"] == 2
    assert sum(item["layers_count"] for item in plan["assignments"]) == 4


def test_layer_budget_rejects_malformed_payload():
    bad_budgets = (
        {"available_bytes": 1 * MIB, "per_layer_bytes": 0, "max_layers": 2},
        {"available_bytes": 1 * MIB, "per_layer_bytes": 1 * MIB, "max_layers": 0},
        {"available_bytes": 1 * MIB, "per_layer_bytes": 1 * MIB},
        {
            "available_bytes": 1 * MIB,
            "per_layer_bytes": 1 * MIB,
            "max_layers": 2,
            "local_cut": "yes",
        },
    )
    for bad in bad_budgets:
        with pytest.raises(PipelineCapacityError):
            solve_pipeline_capacity(
                descriptor(),
                [
                    {**node("worker-a", 500), "layer_budget": bad},
                    node("worker-b", 400),
                ],
                safety_margin=1.0,
            )


def test_layer_budget_supersedes_fixed_layer_ranges():
    """申报了 `layer_budget` 的节点：`layer_ranges` 只代表"当前已就绪"，
    不再作为分配硬约束 —— 否则设备永远被它预置的那一段钉死。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 1]],
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 3,
                    "local_cut": True,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["layers_count"] == 3
    assert by_node["worker-a"]["start_layer"] == 0


def test_layer_budget_without_local_cut_keeps_fixed_ranges():
    """设备没声明"能本地裁层"时，`layer_ranges` 仍是硬约束 —— 否则会派给它一个
    手上没有工件的区间，请求必然失败。"""
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 2]],
                "layer_budget": {
                    "available_bytes": 500 * MIB,
                    "per_layer_bytes": 100 * MIB,
                    "max_layers": 3,
                    "local_cut": False,
                },
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["start_layer"] == 0
    assert by_node["worker-a"]["end_layer"] == 2


def test_middle_segment_artifact_cannot_take_the_tail_slot():
    """★ DIST-3（2026-10-05 三机实测）：中间段工件不能接末段。

    `layer_ranges` 只声明"本节点覆盖哪些层"，**不区分工件含不含
    `embedding` / `lm_head`**。Y700 广告 `[0, 3]`（实为 `mid8-24`，
    `mode=middle`）时，旧判据认为末段落在区间内就照分 —— 而中间段工件没有
    `lm_head` / `final_norm`，请求必然失败（实测报
    `remote worker reported a Stage error`）。声明的 `segment_mode` 必须让求解器
    拒绝这种分配。
    """
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 3]],
                "segment_mode": "middle",
            },
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is False


def test_absent_segment_declaration_keeps_legacy_behaviour():
    """未声明 `segment_mode` 时**不得新增任何约束**（旧设备 / 旧 manifest 不变）。

    取形沿用 `test_layer_budget_without_local_cut_keeps_fixed_ranges`：`worker-a`
    只声明 `[0, 2]`，`worker-b` 不声明区间。这条同时是段类型约束的**反证** ——
    若约束无条件生效（把缺失当 `middle` 处理），`worker-a` 拿首段就会被拒。
    """
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            {**node("worker-a", 500), "layer_ranges": [[0, 2]]},
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True
    by_node = {item["node_id"]: item for item in plan["assignments"]}
    assert by_node["worker-a"]["start_layer"] == 0

    # 同一拓扑、但显式声明 `middle` ⇒ 首段必须被拒：段类型约束确实生效。
    rejected = solve_pipeline_capacity(
        descriptor(),
        [
            {
                **node("worker-a", 500),
                "layer_ranges": [[0, 2]],
                "segment_mode": "middle",
            },
            node("worker-b", 400),
        ],
        safety_margin=1.0,
    )
    assert rejected["admitted"] is False


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


# ── ★ #31 M3：被运行时忽略的分量可以不计入容量账（只有**显式声明**的架构才行）──────


def test_hybrid_visual_and_mtp_no_longer_block_capacity() -> None:
    """★ M3 的核心：`qwen3_5` 声明了 visual / mtp 不参与层执行 ⇒ 容量账**放行**。

    修复前这里必然 `PipelineCapacityError`（"separately placeable components without a
    runtime plan"）⇒ hybrid 在层流水线上完全不可用。
    """
    model = descriptor()
    model["model_type"] = "qwen3_5"
    model["component_weight_bytes"]["visual"] = 600 * MIB
    model["component_weight_bytes"]["mtp"] = 100 * MIB
    model["runtime_ignored_components"] = ["visual", "mtp"]
    model["runtime_ignored_component_bytes"] = {"visual": 600 * MIB, "mtp": 100 * MIB}

    plan = solve_pipeline_capacity(
        model,
        [node("master", 300, role="master", score=100), node("worker", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True


def test_unignored_component_still_blocks_capacity() -> None:
    """★ 闸门**没被削弱**：只声明了 visual / mtp，但 `multimodal` 也非零 ⇒ 仍必须拒绝。"""
    model = descriptor()
    model["component_weight_bytes"]["visual"] = 600 * MIB
    model["component_weight_bytes"]["multimodal"] = 1 * MIB
    model["runtime_ignored_components"] = ["visual", "mtp"]

    with pytest.raises(PipelineCapacityError, match="without a runtime plan"):
        solve_pipeline_capacity(
            model, [node("master", 300, role="master", score=100)], safety_margin=1.0
        )


def test_unknown_ignored_component_is_rejected() -> None:
    """★ 标记里写错名字必须报错 —— 否则一个笔误就等于**悄悄放行**了一个分量。"""
    model = descriptor()
    model["runtime_ignored_components"] = ["visualx"]

    with pytest.raises(PipelineCapacityError, match="unknown entries"):
        solve_pipeline_capacity(
            model, [node("master", 300, role="master", score=100)], safety_margin=1.0
        )


def test_descriptor_without_the_marker_behaves_as_before() -> None:
    """★ 兼容性：**没有**这个字段的 descriptor（老工件 / 其它架构）行为与修复前一致。"""
    model = descriptor()          # 不带 runtime_ignored_components
    assert "runtime_ignored_components" not in model

    plan = solve_pipeline_capacity(
        model,
        [node("master", 300, role="master", score=100), node("worker", 300)],
        safety_margin=1.0,
    )

    assert plan["admitted"] is True


def test_required_distributed_rejects_single_node_shortcut():
    plan = solve_pipeline_capacity(
        descriptor(),
        [
            node("master", 600, role="master", score=100),
            node("worker-a", 300, score=10),
        ],
        safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is True
    assert plan["reason_code"] == "distributed_forced"
    assert plan["require_distributed"] is True
    assert plan["participating_node_count"] == 2
    assert {item["node_id"] for item in plan["assignments"]} == {
        "master", "worker-a",
    }


def test_relay_exempt_assignment_counts_as_participating_node():
    relay = node("relay", 1)
    relay["capacity_source"] = "relay_exempt"
    plan = solve_pipeline_capacity(
        descriptor(),
        [node("master", 600, role="master", score=100), relay],
        safety_margin=1.0,
        require_distributed=True,
        local_layer_budget=4,
        relay_claims={"relay": (4, 4)},
    )

    assert plan["admitted"] is True
    assert plan["participating_node_count"] == 2
    assert plan["aggregate_only"] is True


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


def test_advertised_layer_ranges_constrain_worker_assignment():
    master = node("master", 160, role="master", score=100)
    android = node("android", 400, score=1)
    android["layer_ranges"] = [[1, 4]]

    plan = solve_pipeline_capacity(
        descriptor(), [master, android], safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is True
    assert [
        (item["node_id"], item["start_layer"], item["end_layer"])
        for item in plan["assignments"]
    ] == [("master", 0, 1), ("android", 1, 4)]
    assert plan["assignments"][1]["layer_ranges"] == [[1, 4]]


def test_advertised_layer_ranges_reject_uncovered_cursor():
    master = node("master", 160, role="master", score=100)
    android = node("android", 400, score=1)
    android["layer_ranges"] = [[2, 4]]

    plan = solve_pipeline_capacity(
        descriptor(), [master, android], safety_margin=1.0,
        require_distributed=True,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == "pipeline_layer_range_coverage_insufficient"
    assert plan["assignments"] == []


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
