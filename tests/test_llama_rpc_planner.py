from dataclasses import replace

import pytest

from src.llama_rpc_planner import plan_rpc_split, plan_to_tensor_split


def test_cpu_rpc_planner_reuses_profile_score_and_stays_conservative():
    decision = plan_rpc_split(
        25,
        1100,
        {
            "score_total": 18,
            "cpu_cores": 4,
            "cpu_freq_mhz": 1200,
            "cpu_load_percent": 25,
            "ram_total_gb": 16,
            "ram_available_gb": 8,
        },
        rtt_ms=20,
        bandwidth_mbps=50,
    )

    assert decision.admitted is True
    assert decision.rpc_layers == 1
    assert decision.local_layers == 24
    assert decision.profile.score_source == "DeviceProfiler.score_total"
    assert decision.reason == "cpu_rpc_conservative_cap"


def test_cpu_rpc_planner_allows_more_layers_for_a_strong_idle_worker():
    decision = plan_rpc_split(
        25,
        1100,
        {
            "score_total": 70,
            "cpu_cores": 16,
            "cpu_freq_mhz": 4000,
            "cpu_load_percent": 0,
            "ram_total_gb": 64,
            "ram_available_gb": 32,
        },
    )

    assert decision.admitted is True
    assert decision.rpc_layers > 1
    assert decision.rpc_layers < decision.total_layers


def test_cpu_rpc_planner_excludes_overloaded_or_unprofiled_workers():
    overloaded = plan_rpc_split(
        25,
        1100,
        {"cpu_cores": 8, "cpu_freq_mhz": 3000, "cpu_load_percent": 95},
    )
    missing = plan_rpc_split(25, 1100, {})

    assert overloaded.rpc_layers == 0
    assert overloaded.reason == "worker_cpu_overloaded"
    assert missing.rpc_layers == 0
    assert missing.reason == "worker_profile_missing"


def _planner_base():
    return plan_rpc_split(
        25,
        1100,
        {
            "score_total": 18,
            "cpu_cores": 4,
            "cpu_freq_mhz": 1200,
            "cpu_load_percent": 25,
            "ram_total_gb": 16,
            "ram_available_gb": 8,
        },
        rtt_ms=20,
        bandwidth_mbps=50,
    )


def test_plan_to_tensor_split_translates_layer_decision_to_device_ratio():
    """planner 的层段决策 -> 引擎的 tensor_split（[本机 CPU, RPC0]，顺序必须一致）。"""
    base = _planner_base()
    assert base.rpc_layers == 1 and base.local_layers == 24

    assert plan_to_tensor_split(base) == [0.96, 0.04]
    # 全部给远端：本机 0 份
    assert plan_to_tensor_split(replace(base, rpc_layers=25, local_layers=0, admitted=True)) == [0.0, 1.0]
    # 超出总层数时按总层数夹紧
    assert plan_to_tensor_split(replace(base, rpc_layers=40, local_layers=0, admitted=True)) == [0.0, 1.0]
    # total_layers 可覆盖（按 4 层规划时 1/4 给远端）
    assert plan_to_tensor_split(replace(base, rpc_layers=1), total_layers=4) == [0.75, 0.25]


def test_plan_to_tensor_split_returns_none_when_no_remote_layer():
    base = _planner_base()
    assert plan_to_tensor_split(replace(base, rpc_layers=0, local_layers=25, admitted=False)) is None
    assert plan_to_tensor_split(replace(base, rpc_layers=0, local_layers=25, admitted=True)) is None
    with pytest.raises(ValueError):
        plan_to_tensor_split(base, total_layers=0)
