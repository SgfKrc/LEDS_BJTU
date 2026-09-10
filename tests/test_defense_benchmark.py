from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "demo"))

import defense_benchmark as benchmark


def test_latency_summary_uses_nearest_rank_p95_and_serial_throughput():
    metrics = benchmark.summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert metrics == {
        "min_ms": 1.0,
        "median_ms": 2.5,
        "mean_ms": 2.5,
        "p95_ms": 4.0,
        "max_ms": 4.0,
        "throughput_tasks_per_second": 400.0,
    }


def test_workload_digest_is_stable_and_changes_with_iteration_count():
    first = benchmark._workload(benchmark.BenchmarkConfig(iterations=3, warmup_iterations=1))
    repeat = benchmark._workload(benchmark.BenchmarkConfig(iterations=3, warmup_iterations=1))
    changed = benchmark._workload(benchmark.BenchmarkConfig(iterations=4, warmup_iterations=1))

    assert first == repeat
    assert first["sha256"] != changed["sha256"]
    assert first["payload_bytes"] == 288


def test_svg_has_claim_guard_and_escapes_labels():
    report = {
        "series": [
            {
                "label": "local <fixture>",
                "metrics": {"median_ms": 1.0, "p95_ms": 2.0},
            },
            {
                "label": "loopback & worker",
                "metrics": {"median_ms": 2.0, "p95_ms": 4.0},
            },
        ]
    }

    chart = benchmark.render_svg(report)

    assert "FIXTURE · SINGLE HOST · NOT MODEL PERFORMANCE" in chart
    assert "Physical dual-host data: NOT RUN" in chart
    assert "local &lt;fixture&gt;" in chart
    assert "loopback &amp; worker" in chart


def test_fixed_load_benchmark_runs_loopback_worker_and_writes_redacted_artifacts(
    monkeypatch,
    tmp_path,
):
    build_root = tmp_path / "build"
    report_path = build_root / "report.json"
    chart_path = build_root / "report.svg"
    monkeypatch.setattr(benchmark, "BUILD_ROOT", build_root)
    monkeypatch.setattr(benchmark, "DEFAULT_REPORT", build_root / "latest.json")
    monkeypatch.setattr(benchmark, "DEFAULT_CHART", build_root / "latest.svg")
    config = benchmark.BenchmarkConfig(
        iterations=3,
        warmup_iterations=1,
        startup_timeout=30.0,
        report_path=report_path,
        chart_path=chart_path,
    )

    report = benchmark.run_benchmark(config)
    benchmark.write_artifacts(report, report_path, chart_path)

    assert report["status"] == "passed"
    assert report["benchmark_class"] == "task_graph_control_plane_fixture"
    assert report["claim_guard"] == {
        "real_model_performance": False,
        "physical_dual_host_performance": False,
        "allowed_claim": "single-host TaskGraph control-plane fixture only",
    }
    assert report["physical_dual_host"]["status"] == "not_run"
    assert [item["topology"] for item in report["series"]] == [
        "single_host_single_process",
        "single_host_dual_process",
    ]
    assert [item["sample_count"] for item in report["series"]] == [3, 3]
    assert report["worker_exit_code"] == 0

    encoded = report_path.read_text(encoding="utf-8")
    payload = json.loads(encoded)
    assert payload["workload"]["id"] == benchmark.WORKLOAD_ID
    assert str(tmp_path) not in encoded
    assert "fixture-benchmark-result" not in encoded
    assert "QLH_CLUSTER_SECRET" not in encoded
    assert chart_path.is_file()
