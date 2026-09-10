from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import perf_report as perf


def control_payload() -> dict:
    return {
        "schema": "qlh.defense_benchmark.v1",
        "status": "passed",
        "created_at": "2026-09-10T07:18:41+00:00",
        "benchmark_class": "task_graph_control_plane_fixture",
        "workload": {
            "id": "def-p3-control-plane-v1",
            "iterations": 12,
            "warmup_iterations": 2,
            "payload_bytes": 288,
            "stage_count_per_iteration": 1,
            "stage_type": "full_inference",
            "sha256": "a" * 64,
        },
        "claim_guard": {
            "real_model_performance": False,
            "physical_dual_host_performance": False,
            "allowed_claim": "single-host TaskGraph control-plane fixture only",
        },
        "physical_dual_host": {
            "status": "not_run",
            "reason_code": "physical_dual_host_data_pending",
            "eligible_for_claim": False,
        },
        "series": [
            {
                "series_id": "in_process_fixture",
                "label": "In-process fixture",
                "topology": "single_host_single_process",
                "host_count": 1,
                "process_count": 1,
                "sample_count": 12,
                "metrics": {
                    "min_ms": 1.0,
                    "median_ms": 1.5,
                    "mean_ms": 1.6,
                    "p95_ms": 2.0,
                    "max_ms": 2.0,
                    "throughput_tasks_per_second": 625.0,
                },
            },
            {
                "series_id": "loopback_worker_fixture",
                "label": "Loopback worker fixture",
                "topology": "single_host_dual_process",
                "host_count": 1,
                "process_count": 2,
                "sample_count": 12,
                "metrics": {
                    "min_ms": 2.0,
                    "median_ms": 2.5,
                    "mean_ms": 2.6,
                    "p95_ms": 3.0,
                    "max_ms": 3.0,
                    "throughput_tasks_per_second": 384.6,
                },
            },
        ],
        "worker_exit_code": 0,
    }


def not_run_payload() -> dict:
    return json.loads(perf.DEFAULT_MODEL_REPORT.read_text(encoding="utf-8"))


def measured_model_payload(*, host_count: int = 1) -> dict:
    return {
        "schema": "qlh.real_model_performance.v1",
        "status": "passed",
        "created_at": "2026-10-01T08:00:00+00:00",
        "claim_guard": {
            "real_model_measurement": True,
            "physical_dual_host_measurement": host_count > 1,
            "eligible_for_model_claim": True,
        },
        "model": {
            "id": "model-fixture-id",
            "artifact_sha256": "b" * 64,
            "format": "gguf",
            "quantization": "q4_k_m",
            "tokenizer": "tokenizer-v1",
            "context_length": 4096,
        },
        "environment": {
            "os": "test-os",
            "cpu": "test-cpu",
            "accelerator": "test-gpu",
            "memory_gb": 16.0,
            "engine": "test-engine",
        },
        "topology": {
            "execution_mode": "local_full_model" if host_count == 1 else "pytorch_layer",
            "host_count": host_count,
            "participant_aliases": ["node-master"] if host_count == 1 else ["node-master", "node-worker-a"],
            "network_path": "local" if host_count == 1 else "controlled-lan",
        },
        "workload": {
            "prompt_set_id": "prompt-set-v1",
            "prompt_set_sha256": "c" * 64,
            "prompt_tokens": 64,
            "generated_tokens": 32,
            "concurrency": 1,
            "runs": 5,
            "streaming": True,
        },
        "metrics": {
            "ttft_ms": {"median": 120.0, "p95": 150.0},
            "inter_token_ms": {"median": 30.0, "p95": 42.0},
            "e2e_ms": {"median": 1080.0, "p95": 1350.0},
            "decode_tokens_per_second": {"median": 33.3, "p95": 35.0},
            "peak_memory_gb": 4.2,
            "model_load_seconds": 7.5,
        },
        "events": {
            "success_count": 5,
            "failure_count": 0,
            "retry_count": 0,
            "fallback_count": 0,
            "node_replacement_count": 0,
        },
    }


def test_not_run_report_keeps_control_plane_and_model_metrics_separate():
    control = perf.validate_control_plane(control_payload())
    model = perf.validate_real_model(not_run_payload())

    markdown = perf.render_markdown(
        control,
        model,
        benchmark_ref="build/defense-benchmark/latest.json",
        benchmark_sha256="d" * 64,
        model_ref="scripts/demo/real-model-performance-not-run.json",
        model_sha256="e" * 64,
    )

    assert "Serial tasks/s" in markdown
    assert "不是 tokens/s" in markdown
    assert "| TTFT (ms) | `NOT RUN` | `NOT RUN` |" in markdown
    assert "physical_dual_host_data_pending" in markdown
    assert "不得从本表推导模型 TTFT" in markdown


def test_control_plane_report_rejects_model_performance_claim():
    payload = control_payload()
    payload["claim_guard"]["real_model_performance"] = True

    with pytest.raises(perf.PerformanceReportError, match="声明边界"):
        perf.validate_control_plane(payload)


def test_control_plane_report_rejects_sample_count_drift():
    payload = control_payload()
    payload["series"][1]["sample_count"] = 11

    with pytest.raises(perf.PerformanceReportError, match="样本数"):
        perf.validate_control_plane(payload)


def test_not_run_model_record_rejects_hidden_fields_and_fake_metrics():
    payload = not_run_payload()
    payload["raw_output"] = "must not appear"

    with pytest.raises(perf.PerformanceReportError, match="白名单"):
        perf.validate_real_model(payload)

    payload = not_run_payload()
    payload["metrics"]["ttft_ms"] = {"median": 1, "p95": 2}
    with pytest.raises(perf.PerformanceReportError, match="空指标"):
        perf.validate_real_model(payload)


def test_measured_model_contract_requires_identity_and_matches_dual_host_guard():
    normalized = perf.validate_real_model(measured_model_payload(host_count=2))
    assert normalized["status"] == "passed"
    assert normalized["claim_guard"]["physical_dual_host_measurement"] is True

    payload = measured_model_payload(host_count=2)
    payload["claim_guard"]["physical_dual_host_measurement"] = False
    with pytest.raises(perf.PerformanceReportError, match="声明边界与拓扑"):
        perf.validate_real_model(payload)


def test_measured_model_contract_rejects_addresses_and_output_fields():
    payload = measured_model_payload()
    payload["environment"]["cpu"] = "host 10.0.0.1"
    with pytest.raises(perf.PerformanceReportError, match="IPv4"):
        perf.validate_real_model(payload)

    payload = measured_model_payload()
    payload["output"] = "generated text"
    with pytest.raises(perf.PerformanceReportError, match="白名单"):
        perf.validate_real_model(payload)


def test_build_report_writes_relative_refs_and_claim_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, "ROOT", tmp_path)
    benchmark_path = tmp_path / "build" / "benchmark.json"
    model_path = tmp_path / "inputs" / "model.json"
    output_path = tmp_path / "docs" / "report.md"
    validation_path = tmp_path / "build" / "report.json"
    benchmark_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    benchmark_path.write_text(json.dumps(control_payload()), encoding="utf-8")
    model_path.write_text(json.dumps(not_run_payload()), encoding="utf-8")

    report = perf.build_report(benchmark_path, model_path, output_path, validation_path)

    assert report["model_metrics_status"] == "not_run"
    assert report["claim_guard"]["control_plane_metrics_are_not_ttft_or_tokens_per_second"] is True
    encoded = validation_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in encoded
    assert "build/benchmark.json" in encoded
    assert output_path.is_file()
