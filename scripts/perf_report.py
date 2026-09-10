"""Generate the DEF-A2 closeout tables from typed benchmark evidence.

The default real-model input is an explicit NOT RUN record. This tool never
loads a model and never turns TaskGraph fixture latency into TTFT or tok/s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "build" / "defense-benchmark" / "latest.json"
DEFAULT_MODEL_REPORT = ROOT / "scripts" / "demo" / "real-model-performance-not-run.json"
DEFAULT_OUTPUT = ROOT / "docs" / "答辩演示-性能结题表-2026-09-10.md"
DEFAULT_REPORT = ROOT / "build" / "defense-performance" / "latest.json"
BENCHMARK_SCHEMA = "qlh.defense_benchmark.v1"
MODEL_SCHEMA = "qlh.real_model_performance.v1"
REPORT_SCHEMA = "qlh.defense_performance_report.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
MODEL_METRICS = (
    "ttft_ms",
    "inter_token_ms",
    "e2e_ms",
    "decode_tokens_per_second",
    "peak_memory_gb",
    "model_load_seconds",
)


class PerformanceReportError(ValueError):
    """A stable, user-facing input or claim-boundary error."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerformanceReportError(f"{label} 不能为空")
    return value.strip()


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PerformanceReportError(f"{label} 必须是不小于 {minimum} 的整数")
    return value


def _number(value: object, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerformanceReportError(f"{label} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        raise PerformanceReportError(f"{label} 数值范围无效")
    return result


def _repo_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PerformanceReportError("输入与输出路径必须位于仓库内") from exc


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PerformanceReportError(f"{label}不可读取：{type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise PerformanceReportError(f"{label}必须是 JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_summary(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"median", "p95"}:
        raise PerformanceReportError(f"{label} 必须只含 median/p95")
    median = _number(value.get("median"), f"{label}.median")
    p95 = _number(value.get("p95"), f"{label}.p95")
    if p95 < median:
        raise PerformanceReportError(f"{label} p95 不得小于 median")
    return {"median": median, "p95": p95}


def validate_control_plane(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != BENCHMARK_SCHEMA:
        raise PerformanceReportError("P3 benchmark schema 无效")
    if payload.get("status") != "passed" or payload.get("benchmark_class") != "task_graph_control_plane_fixture":
        raise PerformanceReportError("P3 benchmark 未通过或类型无效")
    expected_guard = {
        "real_model_performance": False,
        "physical_dual_host_performance": False,
        "allowed_claim": "single-host TaskGraph control-plane fixture only",
    }
    if payload.get("claim_guard") != expected_guard:
        raise PerformanceReportError("P3 benchmark 声明边界无效")
    if payload.get("physical_dual_host") != {
        "status": "not_run",
        "reason_code": "physical_dual_host_data_pending",
        "eligible_for_claim": False,
    }:
        raise PerformanceReportError("P3 物理双机边界无效")

    workload = payload.get("workload")
    if not isinstance(workload, dict):
        raise PerformanceReportError("P3 workload 缺失")
    iterations = _integer(workload.get("iterations"), "workload.iterations", minimum=3)
    warmup = _integer(workload.get("warmup_iterations"), "workload.warmup_iterations")
    payload_bytes = _integer(workload.get("payload_bytes"), "workload.payload_bytes", minimum=1)
    if workload.get("id") != "def-p3-control-plane-v1" or workload.get("stage_type") != "full_inference":
        raise PerformanceReportError("P3 workload identity 无效")
    if workload.get("stage_count_per_iteration") != 1 or not HEX64.fullmatch(str(workload.get("sha256", ""))):
        raise PerformanceReportError("P3 workload digest 或 stage count 无效")

    series = payload.get("series")
    expected = (
        ("in_process_fixture", "single_host_single_process", 1),
        ("loopback_worker_fixture", "single_host_dual_process", 2),
    )
    if not isinstance(series, list) or len(series) != len(expected):
        raise PerformanceReportError("P3 benchmark series 数量无效")
    normalized_series: list[dict[str, Any]] = []
    for item, (series_id, topology, process_count) in zip(series, expected, strict=True):
        if not isinstance(item, dict):
            raise PerformanceReportError("P3 benchmark series 无效")
        if item.get("series_id") != series_id or item.get("topology") != topology:
            raise PerformanceReportError("P3 benchmark series identity 无效")
        if item.get("host_count") != 1 or item.get("process_count") != process_count:
            raise PerformanceReportError("P3 benchmark 拓扑声明无效")
        if item.get("sample_count") != iterations:
            raise PerformanceReportError("P3 benchmark 样本数与 workload 不一致")
        metrics = item.get("metrics")
        metric_keys = {"min_ms", "median_ms", "mean_ms", "p95_ms", "max_ms", "throughput_tasks_per_second"}
        if not isinstance(metrics, dict) or set(metrics) != metric_keys:
            raise PerformanceReportError("P3 benchmark metrics 字段无效")
        values = {key: _number(metrics.get(key), f"{series_id}.{key}") for key in metric_keys}
        if not values["min_ms"] <= values["median_ms"] <= values["p95_ms"] <= values["max_ms"]:
            raise PerformanceReportError("P3 benchmark 延迟分位次序无效")
        if not values["min_ms"] <= values["mean_ms"] <= values["max_ms"]:
            raise PerformanceReportError("P3 benchmark mean 超出 min/max")
        normalized_series.append({
            "series_id": series_id,
            "label": _text(item.get("label"), f"{series_id}.label"),
            "topology": topology,
            "host_count": 1,
            "process_count": process_count,
            "sample_count": iterations,
            "metrics": values,
        })
    if payload.get("worker_exit_code") != 0:
        raise PerformanceReportError("P3 benchmark worker 未正常退出")
    return {
        "created_at": _text(payload.get("created_at"), "P3 created_at"),
        "workload": {
            "id": workload["id"],
            "iterations": iterations,
            "warmup_iterations": warmup,
            "payload_bytes": payload_bytes,
            "stage_type": workload["stage_type"],
            "sha256": workload["sha256"],
        },
        "series": normalized_series,
    }


def validate_real_model(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
        raise PerformanceReportError("真实模型性能 schema 无效")
    status = payload.get("status")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(MODEL_METRICS):
        raise PerformanceReportError("真实模型 metrics 字段无效")
    if status == "not_run":
        if set(payload) != {"schema", "status", "reason_code", "claim_guard", "metrics"}:
            raise PerformanceReportError("真实模型 NOT RUN 顶层字段未通过白名单")
        expected_guard = {
            "real_model_measurement": False,
            "physical_dual_host_measurement": False,
            "eligible_for_model_claim": False,
        }
        if payload.get("reason_code") != "real_model_environment_unavailable":
            raise PerformanceReportError("真实模型 NOT RUN 原因无效")
        if payload.get("claim_guard") != expected_guard or any(value is not None for value in metrics.values()):
            raise PerformanceReportError("真实模型 NOT RUN 声明或空指标无效")
        return {"status": "not_run", "reason_code": payload["reason_code"], "metrics": metrics}
    if status != "passed":
        raise PerformanceReportError("真实模型性能状态必须是 passed 或 not_run")

    allowed_top = {
        "schema", "status", "created_at", "claim_guard", "model", "environment",
        "topology", "workload", "metrics", "events",
    }
    if set(payload) != allowed_top:
        raise PerformanceReportError("真实模型性能顶层字段未通过白名单")
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if IPV4.search(serialized):
        raise PerformanceReportError("真实模型性能记录包含 IPv4 地址")

    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {
        "id", "artifact_sha256", "format", "quantization", "tokenizer", "context_length",
    }:
        raise PerformanceReportError("真实模型 identity 字段无效")
    model_id = _text(model.get("id"), "model.id")
    artifact_sha = str(model.get("artifact_sha256", ""))
    if not HEX64.fullmatch(artifact_sha):
        raise PerformanceReportError("model.artifact_sha256 无效")
    for key in ("format", "quantization", "tokenizer"):
        _text(model.get(key), f"model.{key}")
    _integer(model.get("context_length"), "model.context_length", minimum=1)

    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != {"os", "cpu", "accelerator", "memory_gb", "engine"}:
        raise PerformanceReportError("真实模型 environment 字段无效")
    for key in ("os", "cpu", "accelerator", "engine"):
        _text(environment.get(key), f"environment.{key}")
    _number(environment.get("memory_gb"), "environment.memory_gb")

    topology = payload.get("topology")
    if not isinstance(topology, dict) or set(topology) != {"execution_mode", "host_count", "participant_aliases", "network_path"}:
        raise PerformanceReportError("真实模型 topology 字段无效")
    host_count = _integer(topology.get("host_count"), "topology.host_count", minimum=1)
    aliases = topology.get("participant_aliases")
    if not isinstance(aliases, list) or not aliases or any(not isinstance(item, str) or not item.startswith("node-") for item in aliases):
        raise PerformanceReportError("真实模型 participant aliases 无效")
    if len(set(aliases)) != len(aliases) or len(aliases) < host_count:
        raise PerformanceReportError("真实模型 participant aliases 与 host count 不一致")
    _text(topology.get("execution_mode"), "topology.execution_mode")
    _text(topology.get("network_path"), "topology.network_path")

    workload = payload.get("workload")
    expected_workload_keys = {
        "prompt_set_id", "prompt_set_sha256", "prompt_tokens", "generated_tokens",
        "concurrency", "runs", "streaming",
    }
    if not isinstance(workload, dict) or set(workload) != expected_workload_keys:
        raise PerformanceReportError("真实模型 workload 字段无效")
    _text(workload.get("prompt_set_id"), "workload.prompt_set_id")
    if not HEX64.fullmatch(str(workload.get("prompt_set_sha256", ""))):
        raise PerformanceReportError("workload.prompt_set_sha256 无效")
    for key in ("prompt_tokens", "generated_tokens", "concurrency", "runs"):
        _integer(workload.get(key), f"workload.{key}", minimum=1)
    if not isinstance(workload.get("streaming"), bool):
        raise PerformanceReportError("workload.streaming 必须是 boolean")

    normalized_metrics = {
        "ttft_ms": _validate_summary(metrics["ttft_ms"], "metrics.ttft_ms"),
        "inter_token_ms": _validate_summary(metrics["inter_token_ms"], "metrics.inter_token_ms"),
        "e2e_ms": _validate_summary(metrics["e2e_ms"], "metrics.e2e_ms"),
        "decode_tokens_per_second": _validate_summary(metrics["decode_tokens_per_second"], "metrics.decode_tokens_per_second"),
        "peak_memory_gb": _number(metrics["peak_memory_gb"], "metrics.peak_memory_gb"),
        "model_load_seconds": _number(metrics["model_load_seconds"], "metrics.model_load_seconds"),
    }
    events = payload.get("events")
    if not isinstance(events, dict) or set(events) != {"success_count", "failure_count", "retry_count", "fallback_count", "node_replacement_count"}:
        raise PerformanceReportError("真实模型 events 字段无效")
    for key in events:
        _integer(events.get(key), f"events.{key}")
    if events["success_count"] + events["failure_count"] != workload["runs"]:
        raise PerformanceReportError("真实模型运行计数与 workload 不一致")

    expected_guard = {
        "real_model_measurement": True,
        "physical_dual_host_measurement": host_count > 1,
        "eligible_for_model_claim": True,
    }
    if payload.get("claim_guard") != expected_guard:
        raise PerformanceReportError("真实模型 passed 声明边界与拓扑不一致")
    return {
        "status": "passed",
        "created_at": _text(payload.get("created_at"), "真实模型 created_at"),
        "model": {**model, "id": model_id},
        "environment": environment,
        "topology": topology,
        "workload": workload,
        "metrics": normalized_metrics,
        "events": events,
        "claim_guard": expected_guard,
    }


def _metric(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(
    control: dict[str, Any],
    model: dict[str, Any],
    *,
    benchmark_ref: str,
    benchmark_sha256: str,
    model_ref: str,
    model_sha256: str,
) -> str:
    workload = control["workload"]
    rows = []
    for item in control["series"]:
        metrics = item["metrics"]
        rows.append(
            f"| `{item['series_id']}` | `{item['topology']}` | {item['host_count']} | "
            f"{item['process_count']} | {item['sample_count']} | {_metric(metrics['median_ms'])} | "
            f"{_metric(metrics['p95_ms'])} | {_metric(metrics['throughput_tasks_per_second'])} |"
        )

    if model["status"] == "not_run":
        model_identity = "`NOT RUN`（当前机器缺少合适模型与运行条件）"
        model_rows = [
            f"| {label} | `NOT RUN` | `NOT RUN` | `real_model_environment_unavailable` |"
            for label in ("TTFT (ms)", "Inter-token latency (ms)", "E2E latency (ms)", "Decode (tokens/s)")
        ]
        resource_rows = (
            "| Peak memory (GB) | `NOT RUN` | `real_model_environment_unavailable` |\n"
            "| Model load (s) | `NOT RUN` | `real_model_environment_unavailable` |"
        )
        model_claim = "不可用；`AUD-RT-01` / `AUD-SW-01` 仍保持待开始"
        dual_host_claim = "不可用；`physical_dual_host_data_pending`"
        model_details = "无模型工件、设备或输出被采集。"
    else:
        identity = model["model"]
        model_identity = f"`{identity['id']}` / `{identity['format']}` / `{identity['quantization']}`"
        metric_labels = (
            ("TTFT (ms)", "ttft_ms"),
            ("Inter-token latency (ms)", "inter_token_ms"),
            ("E2E latency (ms)", "e2e_ms"),
            ("Decode (tokens/s)", "decode_tokens_per_second"),
        )
        model_rows = [
            f"| {label} | {_metric(model['metrics'][key]['median'])} | {_metric(model['metrics'][key]['p95'])} | `measured` |"
            for label, key in metric_labels
        ]
        resource_rows = (
            f"| Peak memory (GB) | {_metric(model['metrics']['peak_memory_gb'])} | `measured` |\n"
            f"| Model load (s) | {_metric(model['metrics']['model_load_seconds'])} | `measured` |"
        )
        model_claim = "可用，但仅限该工件、环境、负载和拓扑"
        dual_host_claim = "可用" if model["topology"]["host_count"] > 1 else "不可用；真实记录仅为单机"
        model_details = (
            f"工件 SHA-256：`{identity['artifact_sha256']}`；执行模式："
            f"`{model['topology']['execution_mode']}`；host 数：{model['topology']['host_count']}。"
        )

    return f"""# 答辩演示性能结题表

> 票号：`DEF-A2`  
> 报告口径：控制面 fixture 与真实模型指标严格分表，禁止跨表替代  
> 当前结论：P3 TaskGraph 控制面数据可引用；真实模型 TTFT/tok/s 与物理双机数据按输入状态明确标记

## 数据来源与完整性

| 数据源 | SHA-256 | 状态 | 可支持结论 |
| --- | --- | --- | --- |
| `{benchmark_ref}` | `{benchmark_sha256}` | `PASSED` | 单机 TaskGraph 控制面 fixture 延迟与串行任务率 |
| `{model_ref}` | `{model_sha256}` | `{model['status'].upper()}` | {model_claim} |

- P3 采集时间：`{control['created_at']}`。
- P3 workload：`{workload['id']}`，{workload['iterations']} 次计量 + {workload['warmup_iterations']} 次预热，每轮 1 个 `{workload['stage_type']}` fixture stage，固定载荷 {workload['payload_bytes']} bytes。
- Workload SHA-256：`{workload['sha256']}`。
- 真实模型身份：{model_identity}。{model_details}
- `scripts/sse_ttft_compare.py` 的控制台文本缺少本报告要求的完整身份、环境、拓扑和 JSON 契约，因此未被静默采纳。

## 表 1：TaskGraph 控制面 fixture

> 单位中的 `tasks/s` 是串行 fixture workflow 完成率，不是 tokens/s；两组均为 `host_count=1`。

| Series | Topology | Hosts | Processes | Samples | Median (ms) | P95 (ms) | Serial tasks/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

可引用结论：同一固定 workload 已分别走进程内 provider 与真实 loopback worker 进程路径，可比较本机控制面/进程通信开销。

不可引用结论：不得从本表推导模型 TTFT、decode tokens/s、生成质量、物理双机收益或模型加速比。

## 表 2：真实模型时延与吞吐

| Metric | Median | P95 | Evidence state |
| --- | ---: | ---: | --- |
{chr(10).join(model_rows)}

## 表 3：真实模型资源

| Metric | Value | Evidence state |
| --- | ---: | --- |
{resource_rows}

## 表 4：结论资格矩阵

| 结论 | 当前是否可用 | 原因/边界 |
| --- | --- | --- |
| 固定负载下 TaskGraph 控制面 fixture 数值 | 是 | P3 schema、workload SHA、样本数、拓扑和声明门均通过 |
| 第二个真实 worker 进程参与控制面路径 | 是 | `single_host_dual_process`，worker exit code 为 0 |
| 真实模型 TTFT / inter-token / E2E / tokens/s | {'是' if model['status'] == 'passed' else '否'} | {model_claim} |
| 物理双机性能 | {'是' if model['status'] == 'passed' and model['topology']['host_count'] > 1 else '否'} | {dual_host_claim} |
| 模型质量或通用加速比 | 否 | 本报告不采集输出质量，也不把异类指标作加速比 |

## 结题摘要

1. `DEF-P3` 的两组数字只进入“控制面 fixture”表，标签、单位和拓扑保持完整。
2. 当前真实模型输入为 `{model['status']}`；任何 `NOT RUN` 单元格都不得用控制面毫秒数补齐。
3. 只有未来输入通过 `qlh.real_model_performance.v1` 的模型工件、环境、拓扑、负载、事件计数和声明门校验后，表 2/3 才会出现真实数值。
4. 当前物理双机性能结论：{dual_host_claim}。

## 实现与契约证据

- [A2 汇总器](../scripts/perf_report.py)
- [真实模型 NOT RUN 输入](../scripts/demo/real-model-performance-not-run.json)
- [P3 benchmark 实现](../scripts/demo/defense_benchmark.py)
- [A2 正负向契约测试](../tests/test_defense_performance_report.py)

## 重建命令

```powershell
scripts\\demo\\performance_report.bat --benchmark {benchmark_ref}
```

该命令只读取 JSON 并生成 Markdown/校验报告，不启动后端、不访问网络、不加载模型。
"""


def build_report(
    benchmark_path: Path,
    model_path: Path,
    output_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    benchmark_ref = _repo_ref(benchmark_path)
    model_ref = _repo_ref(model_path)
    _repo_ref(output_path)
    _repo_ref(validation_path)
    benchmark_payload, benchmark_sha = _read_json(benchmark_path, "P3 benchmark")
    model_payload, model_sha = _read_json(model_path, "真实模型性能记录")
    control = validate_control_plane(benchmark_payload)
    model = validate_real_model(model_payload)
    markdown = render_markdown(
        control,
        model,
        benchmark_ref=benchmark_ref,
        benchmark_sha256=benchmark_sha,
        model_ref=model_ref,
        model_sha256=model_sha,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": _repo_ref(output_path),
        "inputs": {
            "control_plane": {"path": benchmark_ref, "sha256": benchmark_sha, "status": "passed"},
            "real_model": {"path": model_ref, "sha256": model_sha, "status": model["status"]},
        },
        "claim_guard": {
            "control_plane_fixture_only": True,
            "real_model_performance_available": model["status"] == "passed",
            "physical_dual_host_performance_available": (
                model["status"] == "passed" and model["topology"]["host_count"] > 1
            ),
            "control_plane_metrics_are_not_ttft_or_tokens_per_second": True,
        },
        "control_plane_series_count": len(control["series"]),
        "model_metrics_status": model["status"],
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DEF-A2 performance closeout tables")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args.benchmark, args.model_report, args.output, args.report)
    except PerformanceReportError as exc:
        print(f"[QLH-PERF-REPORT] ERROR {exc}", file=sys.stderr)
        return 2
    print(
        "[QLH-PERF-REPORT] OK "
        f"control_series={report['control_plane_series_count']} "
        f"model_metrics={report['model_metrics_status']}"
    )
    print(f"[QLH-PERF-REPORT] OUTPUT {report['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
