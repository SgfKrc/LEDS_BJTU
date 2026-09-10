"""DEF-P3 fixed-load, model-free TaskGraph control-plane benchmark.

This benchmark compares an in-process fixture provider with a real loopback
task-worker process.  It deliberately does not claim model or physical
dual-host performance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demo_ownership import OwnershipLedger


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
BUILD_ROOT = ROOT / "build" / "defense-benchmark"
DEFAULT_REPORT = BUILD_ROOT / "latest.json"
DEFAULT_CHART = BUILD_ROOT / "latest.svg"
BENCHMARK_WORKER = Path(__file__).with_name("benchmark_worker.py")
BENCHMARK_SCHEMA = "qlh.defense_benchmark.v1"
WORKLOAD_ID = "def-p3-control-plane-v1"
PAYLOAD = {"message": "x" * 256, "fixture": True}


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int = 12
    warmup_iterations: int = 2
    startup_timeout: float = 30.0
    report_path: Path = DEFAULT_REPORT
    chart_path: Path = DEFAULT_CHART


def _validate_config(config: BenchmarkConfig) -> None:
    if not 3 <= config.iterations <= 200:
        raise ValueError("iterations 必须在 3 到 200 之间")
    if not 0 <= config.warmup_iterations <= 20:
        raise ValueError("warmup 必须在 0 到 20 之间")
    if not 1.0 <= config.startup_timeout <= 120.0:
        raise ValueError("timeout 必须在 1 到 120 秒之间")
    if not BENCHMARK_WORKER.is_file():
        raise RuntimeError("缺少 scripts/demo/benchmark_worker.py")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize_latencies(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency samples must not be empty")
    total_seconds = sum(values) / 1000.0
    return {
        "min_ms": round(min(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "max_ms": round(max(values), 3),
        "throughput_tasks_per_second": round(len(values) / total_seconds, 3),
    }


def _workload(config: BenchmarkConfig) -> dict[str, Any]:
    canonical = {
        "id": WORKLOAD_ID,
        "iterations": config.iterations,
        "warmup_iterations": config.warmup_iterations,
        "payload_bytes": len(json.dumps(PAYLOAD, sort_keys=True).encode("utf-8")),
        "stage_count_per_iteration": 1,
        "stage_type": "full_inference",
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**canonical, "sha256": digest}


def _run_series(
    coordinator,
    *,
    provider_id: str,
    identity,
    config: BenchmarkConfig,
    series_id: str,
) -> list[float]:
    from task_graph import StageSpec

    latencies: list[float] = []
    total = config.warmup_iterations + config.iterations
    for index in range(total):
        started = time.perf_counter_ns()
        output, snapshot = coordinator.run(
            stages=[StageSpec(
                "answer",
                "full_inference",
                provider=provider_id,
                fallback_providers=(),
                pure=False,
                lease_timeout_seconds=5.0,
            )],
            final_stage_id="answer",
            root_input=PAYLOAD,
            model_identity=identity,
            template="defense_benchmark_fixture_v1",
            workflow_id=f"wf_defbench_{series_id}_{index:03d}",
        )
        if output.get("content") != "fixture-benchmark-result":
            raise RuntimeError("benchmark fixture output marker mismatch")
        committed = coordinator.commit_result(snapshot["workflow_id"])
        if committed.get("state") != "completed":
            raise RuntimeError("benchmark workflow did not commit")
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if index >= config.warmup_iterations:
            latencies.append(elapsed_ms)
    return latencies


def _series_record(
    series_id: str,
    label: str,
    topology: str,
    process_count: int,
    latencies: list[float],
) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "label": label,
        "topology": topology,
        "host_count": 1,
        "process_count": process_count,
        "sample_count": len(latencies),
        "metrics": summarize_latencies(latencies),
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    _validate_config(config)
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import config as runtime_config
    from task_graph import TaskGraphCoordinator
    from task_provider import DeterministicFakeProvider, ModelIdentity, ProviderRegistry
    from task_worker_adapter import RemoteFullWorkerProvider, TaskWorkerControlPlane
    from tcp_comm import MessageType, TCPServer

    started_at = time.perf_counter()
    secret = secrets.token_hex(24)
    previous_secret = runtime_config.CLUSTER_SECRET
    runtime_config.CLUSTER_SECRET = secret
    worker_id = "demo-benchmark-worker"
    control = TaskWorkerControlPlane(health_timeout_seconds=max(5.0, config.startup_timeout))
    hello_received = threading.Event()
    disconnected = threading.Event()
    provider_holder: dict[str, Any] = {}
    server = TCPServer(host="127.0.0.1", port=0)
    server_started = False
    coordinator = None
    worker_process: subprocess.Popen[str] | None = None
    ownership = OwnershipLedger("defense_benchmark")
    log_path = BUILD_ROOT / "benchmark-worker.log"

    def on_server_message(client_id: str, outer: dict[str, Any]) -> None:
        if outer.get("type") == MessageType.REGISTER.value:
            server.confirm_registration(client_id)
            return
        if outer.get("type") != MessageType.TASK_WORKER.value:
            return
        inner = outer.get("data", {})
        if inner.get("message_type") == "hello":
            acknowledgement = control.receive_on_coordinator(
                client_id,
                inner,
                coordinator_node_id="demo-master",
            )
            server.send_to_client(client_id, acknowledgement.snapshot(), MessageType.TASK_WORKER)
            hello_received.set()
            return
        provider_holder["provider"].handle_message(inner)

    def on_disconnect(client_id: str) -> None:
        control.disconnect_worker(client_id)
        provider = provider_holder.get("provider")
        if provider is not None:
            provider.notify_disconnect()
        disconnected.set()

    remote_provider = RemoteFullWorkerProvider(
        node_id=worker_id,
        peer_snapshot=lambda: control.worker_snapshot(worker_id),
        send_message=lambda message: server.send_to_client(
            worker_id,
            message.snapshot(),
            MessageType.TASK_WORKER,
        ),
    )
    provider_holder["provider"] = remote_provider
    local_provider = DeterministicFakeProvider(
        "demo-benchmark-local",
        node_id="demo-master",
        supported_stage_types=("full_inference",),
        output_factory=lambda _request, _cancel: {"content": "fixture-benchmark-result"},
    )
    identity = ModelIdentity(
        model_id="defense-benchmark-fixture",
        engine="pytorch",
        format="safetensors",
        revision="defense-benchmark-v1",
        sha256="b" * 64,
    )

    try:
        server.start(on_message=on_server_message, on_disconnect=on_disconnect)
        server_started = True
        BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        worker_env = os.environ.copy()
        worker_env.update({
            "HF_HUB_OFFLINE": "1",
            "PYTHONUTF8": "1",
            "QLH_CLUSTER_SECRET": secret,
            "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "true",
            "TRANSFORMERS_OFFLINE": "1",
        })
        try:
            worker_process = subprocess.Popen(
                [
                    sys.executable,
                    str(BENCHMARK_WORKER),
                    "--port",
                    str(server.port),
                    "--node-id",
                    worker_id,
                    "--model-id",
                    identity.model_id,
                    "--model-sha256",
                    identity.sha256,
                    "--task-count",
                    str(config.iterations + config.warmup_iterations),
                ],
                cwd=ROOT,
                env=worker_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            ownership.register("benchmark-worker", worker_process.pid)
        finally:
            log_handle.close()
        if not hello_received.wait(config.startup_timeout):
            raise RuntimeError("benchmark worker protocol handshake timed out")

        registry = ProviderRegistry()
        registry.register(local_provider)
        registry.register(remote_provider)
        coordinator = TaskGraphCoordinator(provider_registry=registry)
        local_latencies = _run_series(
            coordinator,
            provider_id=local_provider.provider_id,
            identity=identity,
            config=config,
            series_id="local",
        )
        remote_latencies = _run_series(
            coordinator,
            provider_id=remote_provider.provider_id,
            identity=identity,
            config=config,
            series_id="loopback",
        )
        try:
            worker_exit_code = worker_process.wait(timeout=min(10.0, config.startup_timeout))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("benchmark worker did not exit") from exc
        if worker_exit_code != 0:
            raise RuntimeError("benchmark worker exited unsuccessfully")
        if not disconnected.wait(min(2.0, config.startup_timeout)):
            raise RuntimeError("benchmark worker disconnect was not observed")

        series = [
            _series_record(
                "in_process_fixture",
                "In-process fixture",
                "single_host_single_process",
                1,
                local_latencies,
            ),
            _series_record(
                "loopback_worker_fixture",
                "Loopback worker fixture",
                "single_host_dual_process",
                2,
                remote_latencies,
            ),
        ]
        return {
            "schema": BENCHMARK_SCHEMA,
            "status": "passed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_class": "task_graph_control_plane_fixture",
            "workload": _workload(config),
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
            "series": series,
            "worker_exit_code": worker_exit_code,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "logs": [log_path.name],
        }
    finally:
        if coordinator is not None:
            coordinator.close()
        else:
            remote_provider.close()
            local_provider.close()
        if worker_process is not None and worker_process.poll() is None:
            worker_process.terminate()
            try:
                worker_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                worker_process.kill()
                worker_process.wait(timeout=5.0)
        if server_started:
            server.stop()
        runtime_config.CLUSTER_SECRET = previous_secret
        ownership.close()


def render_svg(report: dict[str, Any]) -> str:
    series = report.get("series", [])
    if len(series) != 2:
        raise ValueError("benchmark chart requires exactly two series")
    max_value = max(float(item["metrics"]["p95_ms"]) for item in series) or 1.0
    colors = ("#c7ff3d", "#d8b4ff")
    rows = []
    for index, item in enumerate(series):
        median = float(item["metrics"]["median_ms"])
        p95 = float(item["metrics"]["p95_ms"])
        median_width = max(2.0, 430.0 * median / max_value)
        p95_width = max(2.0, 430.0 * p95 / max_value)
        y = 118 + index * 126
        label = html.escape(str(item["label"]))
        color = colors[index]
        rows.append(
            f'<text x="24" y="{y}" class="label">{label}</text>'
            f'<rect x="210" y="{y - 17}" width="{median_width:.2f}" height="18" fill="{color}"/>'
            f'<text x="650" y="{y - 3}" class="value">median {median:.3f} ms</text>'
            f'<rect x="210" y="{y + 13}" width="{p95_width:.2f}" height="18" fill="{color}" opacity="0.56"/>'
            f'<text x="650" y="{y + 27}" class="value">p95 {p95:.3f} ms</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390">'
        '<rect width="900" height="390" fill="#0b0b0d"/>'
        '<style>.title{font:700 22px sans-serif;fill:#f4f2ed}.guard{font:700 13px monospace;fill:#ffb454}'
        '.label{font:600 14px sans-serif;fill:#f4f2ed}.value{font:12px monospace;fill:#9c9aa2}'
        '.foot{font:12px sans-serif;fill:#6f6d76}</style>'
        '<text x="24" y="38" class="title">DEF-P3 TaskGraph control-plane benchmark</text>'
        '<text x="24" y="65" class="guard">FIXTURE · SINGLE HOST · NOT MODEL PERFORMANCE</text>'
        + "".join(rows)
        + '<text x="24" y="355" class="foot">Physical dual-host data: NOT RUN / pending controlled hardware window</text>'
        '</svg>\n'
    )


def write_artifacts(
    report: dict[str, Any],
    report_path: Path,
    chart_path: Path,
) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    chart = render_svg(report)
    for path in {DEFAULT_REPORT, report_path}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    for path in {DEFAULT_CHART, chart_path}:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(chart, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEF-P3 model-free benchmark")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2, dest="warmup_iterations")
    parser.add_argument("--timeout", type=float, default=30.0, dest="startup_timeout")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        startup_timeout=args.startup_timeout,
        report_path=args.report,
        chart_path=args.chart,
    )
    try:
        report = run_benchmark(config)
        write_artifacts(report, config.report_path, config.chart_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[QLH-BENCHMARK] ERROR {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"[QLH-BENCHMARK] OK iterations={config.iterations}")
    print("[QLH-BENCHMARK] SCOPE single-host control-plane fixture; dual-host=not-run")
    print(f"[QLH-BENCHMARK] REPORT {config.report_path}")
    print(f"[QLH-BENCHMARK] CHART {config.chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
