from src.llama_rpc_planner import plan_rpc_split


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
