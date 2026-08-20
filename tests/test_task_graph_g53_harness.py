"""TG-OPT-G5.3 完整版解锁：动态故障注入基础设 + 复验闭环测试。

对 shadow 候选准入（recommend_speculative_execution）注入确定扰动（尾延迟/
成本/候选 Provider/空闲 worker/准入/故障域），验证：
1. 扰动只影响决策输入，不改变 DAG 或运行时；
2. 同 seed 重放 → digest 可复现（复验闭环）；
3. 未知扰动键 / 非法类型 fail-closed 拒绝。
真实双 Worker 竞速/唯一提交仍属 G5.3 外部门，本门仅解锁本机影子级复验。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from task_graph_speculation import (
    TaskGraphSpeculationError,
    build_speculation_profile,
    inject_speculation_disturbance,
    recommend_speculative_execution,
    run_speculation_injection_matrix,
    verify_speculation_injection_closed_loop,
)


def _stage(**overrides):
    stage = {
        "stage_id": "s1",
        "pure": True,
        "cancellable": True,
        "result_arbitration": "single_winner",
        "primary_provider": "local_full_model",
        "candidate_provider": "remote_full_worker",
        "latency": {"p95_ms": 400, "elapsed_ms": 500, "deadline_ms": 2000},
        "resources": {
            "extra_cost_ratio": 0.2,
            "idle_compatible_workers": 2,
            "provider_admitted": True,
            "failure_domain_changed": True,
        },
        **overrides,
    }
    return stage


def _profile():
    return build_speculation_profile(
        min_tail_latency_ms=250, min_elapsed_ratio=1.0,
        max_extra_cost_ratio=0.5, max_candidates=8, require_dual_worker=True,
    )


def test_baseline_stage_is_candidate():
    report = recommend_speculative_execution([_stage()], profile=_profile())
    assert report["status"] == "candidate"
    assert report["runtime_actions_enabled"] is False
    assert len(report["candidates"]) == 1


def test_disturbance_changes_decision_without_runtime():
    # 尾延迟降到阈值下 → 拒绝；成本超额 → 拒绝；空转 worker 为 0 → 拒绝
    low_tail = recommend_speculative_execution(
        [inject_speculation_disturbance(_stage(), disturb={"tail_latency_ms": 100})],
        profile=_profile(),
    )
    assert low_tail["status"] == "no_candidate"
    assert any(r["reason_code"] == "tail_latency_below_threshold" for r in low_tail["rejections"])

    over_cost = recommend_speculative_execution(
        [inject_speculation_disturbance(_stage(), disturb={"extra_cost_ratio": 0.9})],
        profile=_profile(),
    )
    assert any(r["reason_code"] == "extra_cost_budget_exceeded" for r in over_cost["rejections"])

    no_worker = recommend_speculative_execution(
        [inject_speculation_disturbance(_stage(), disturb={"idle_compatible_workers": 0})],
        profile=_profile(),
    )
    assert any(r["reason_code"] == "no_idle_compatible_worker" for r in no_worker["rejections"])


def test_injection_matrix_fields_and_shadow_safety():
    matrix = run_speculation_injection_matrix(
        _stage(), profile=_profile(),
        disturbances=[
            {"tail_latency_ms": 100},
            {"extra_cost_ratio": 0.9},
            {"candidate_provider": "remote_worker_b"},
            {"fail_provider": "untrusted_provider"},
            {},
        ],
        seed="g53-matrix-1",
    )
    assert len(matrix["rows"]) == 5
    assert matrix["runtime_actions_enabled"] is False
    for row in matrix["rows"]:
        assert row["scenario"]
        assert row["recommend_digest"]
        assert row["runtime_actions_enabled"] is False


def test_closed_loop_replay_is_reproducible():
    disturbances = [
        {"tail_latency_ms": 100},
        {"extra_cost_ratio": 0.9},
        {"candidate_provider": "remote_worker_b"},
        {"fail_provider": "untrusted_provider"},
        {"elapsed_ms": 1000},
        {"deadline_ms": 30000},
        {},
    ]
    matrix = run_speculation_injection_matrix(
        _stage(), profile=_profile(), disturbances=disturbances, seed="g53-cl",
    )
    violations = verify_speculation_injection_closed_loop(
        matrix, stage=_stage(), profile=_profile(),
        disturbances=disturbances, seed="g53-cl",
    )
    assert violations == []


def test_unknown_disturbance_and_bad_types_fail_closed():
    with pytest.raises(TaskGraphSpeculationError):
        inject_speculation_disturbance(_stage(), disturb={"not_a_key": 1})
    with pytest.raises(TaskGraphSpeculationError):
        run_speculation_injection_matrix(
            _stage(), profile=_profile(),
            disturbances=[{"not_a_key": 1}], seed="x",
        )
    with pytest.raises(TaskGraphSpeculationError):
        inject_speculation_disturbance(_stage(), disturb={"tail_latency_ms": "abc"})
    with pytest.raises(TaskGraphSpeculationError):
        inject_speculation_disturbance(_stage(), disturb={"provider_admitted": 1})
    with pytest.raises(TaskGraphSpeculationError):
        inject_speculation_disturbance(_stage(), disturb={"failure_domain_changed": "yes"})
    with pytest.raises(TaskGraphSpeculationError):
        inject_speculation_disturbance(_stage(), disturb={"idle_compatible_workers": "abc"})


def test_original_stage_unchanged_by_injection():
    before = _stage()
    source = _stage()
    injected = inject_speculation_disturbance(source, disturb={"tail_latency_ms": 100})
    assert source == before            # 注入不改原始 stage
    assert injected["latency"]["p95_ms"] == 100
    assert injected["stage_id"] == "s1"
