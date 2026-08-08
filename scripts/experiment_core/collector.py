"""采集器（EX-N1）：把单元结果组装为 schema v1 实验记录并落盘。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .plan import ExperimentUnit, GateSpec, PlanManifest
from .runner import UnitOutcome


class CollectorError(RuntimeError):
    """记录组装或落盘失败。"""


def current_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def git_describe() -> str:
    try:
        completed = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, timeout=5,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return current_commit()


def _pick(metrics: Mapping[str, Any], key: str) -> Any:
    return metrics.get(key)


def evaluate_gate(
    gate: GateSpec,
    metrics: Mapping[str, Any],
    *,
    baseline_metrics: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """按 gate 判定：返回 (status, threshold_desc)。status ∈ passed/failed/invalid。"""
    value = metrics.get(gate.metric)
    if value is None:
        return "invalid", f"缺少指标 {gate.metric}"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "invalid", f"指标 {gate.metric} 不是数字: {value!r}"
    if gate.threshold is not None:
        threshold = gate.threshold
    else:
        if baseline_metrics is None:
            return "invalid", "缺少基线记录，无法按 baseline_ratio 判定"
        baseline = baseline_metrics.get(gate.metric)
        if baseline is None:
            return "invalid", f"基线缺少指标 {gate.metric}"
        threshold = float(baseline) * float(gate.baseline_ratio)
    ok = {
        ">=": value >= threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        "<": value < threshold,
        "==": value == threshold,
    }[gate.op]
    desc = f"{gate.metric} {gate.op} {threshold:.4g}"
    return ("passed" if ok else "failed"), desc


def _env_from_plan(plan: PlanManifest) -> dict[str, Any]:
    declared = dict(plan.env)
    declared.setdefault("os", f"{platform.system()} {platform.release()}")
    declared.setdefault("cpu", platform.machine())
    declared.setdefault("ram_gb", None)
    declared.setdefault("engine", "")
    declared.setdefault("torch", "")
    declared.setdefault("cuda", "")
    return declared


def build_record(
    plan: PlanManifest,
    unit: ExperimentUnit,
    outcome: UnitOutcome,
    *,
    prompt_set_dir: Path,
    records: Mapping[str, dict],
) -> dict[str, Any]:
    """组装一条 schema v1 实验记录。records 为已完成记录（供基线引用）。"""
    baseline = records.get(unit.baseline_experiment_id or "") if unit.baseline_experiment_id else None
    baseline_metrics = baseline.get("metrics") if baseline else None
    gate_status = "invalid"
    gate_threshold = ""
    if unit.gate is not None:
        gate_status, gate_threshold = evaluate_gate(
            unit.gate, outcome.metrics,
            baseline_metrics=baseline_metrics,
        )
    elif outcome.status == "passed":
        gate_status = "passed"
    else:
        gate_status = "failed"
    if outcome.status != "passed":
        gate_status = "failed"

    prompt_set_sha = ""
    prompt_count = 0
    try:
        prompts = prompt_set_dir / "prompts.jsonl"
        if prompts.is_file():
            prompt_set_sha = _sha256(prompts)
            prompt_count = sum(1 for _ in prompts.open(encoding="utf-8"))
    except OSError:
        pass

    return {
        "experiment_id": unit.experiment_id,
        "experiment_name": unit.name,
        "plan_id": plan.plan_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": current_commit(),
        "env": _env_from_plan(plan),
        "model": dict(unit.model),
        "prompt_set": {
            "id": plan.prompt_set["id"],
            "sha256": prompt_set_sha or plan.prompt_set.get("sha256", ""),
            "count": prompt_count,
            **(dict(unit.prompt_set) if unit.prompt_set else {}),
        },
        "params": dict(unit.params),
        "runs": unit.runs,
        "metrics": dict(outcome.metrics),
        "quality": {"correct_rate": None, "format_rate": None},
        "baseline_experiment_id": unit.baseline_experiment_id,
        "gate": {
            "metric": unit.gate.metric if unit.gate else "",
            "threshold": gate_threshold,
            "status": gate_status,
        },
        "artifacts": {
            "raw_log": outcome.raw_log,
            "outputs": str(out_dir_of(outcome.raw_log)),
            "result_file": None,
        },
        "retries": list(outcome.retries),
        "error": outcome.error or None,
    }


def out_dir_of(raw_log: str) -> Path:
    if raw_log:
        return Path(raw_log).parent
    return Path(".")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_record(records_path: Path, record: dict[str, Any]) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with open(records_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_record(record: dict[str, Any]) -> list[str]:
    """schema 必填项检查（jsonschema 完整校验在 CLI 层可选启用）。"""
    missing = [key for key in (
        "experiment_id", "experiment_name", "plan_id", "timestamp",
        "commit", "env", "model", "prompt_set", "params", "runs",
        "metrics", "gate", "artifacts",
    ) if key not in record]
    return missing
