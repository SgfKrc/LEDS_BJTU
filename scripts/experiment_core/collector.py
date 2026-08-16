"""采集器（EX-N1）：把单元结果组装为 schema v1 实验记录并落盘。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .plan import ExperimentUnit, GateSpec, PlanManifest, QualitySpec
from .quality import (
    QualityEvidenceError,
    invalid_quality_evidence,
    normalize_quality_evidence,
)
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


def _quality_baseline_rates(record: Mapping[str, Any] | None) -> Mapping[str, float] | None:
    if not isinstance(record, Mapping):
        return None
    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        return None
    llm = quality.get("llm")
    if not isinstance(llm, Mapping):
        return None
    correctness = llm.get("correctness")
    formatting = llm.get("format")
    if not isinstance(correctness, Mapping) or not isinstance(formatting, Mapping):
        return None
    try:
        return {
            "correctness": float(correctness["rate"]),
            "format": float(formatting["rate"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_quality_gate(
    quality: Mapping[str, Any],
    quality_spec: QualitySpec | None,
    checks: tuple[str, ...],
    *,
    baseline_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate EX-N3 quality independently from the existing performance gate.

    The return value is deliberately summary-only.  It contains thresholds and
    counters-derived rates, never prompts, completions, images, paths, or judge
    explanations.  A plan with no requested check retains the EX-N3-S0 state.
    """
    if not checks:
        return {"status": "not_collected", "required": False, "checks": []}
    if quality_spec is None:
        return {"status": "invalid", "required": False, "checks": list(checks)}
    status = quality.get("status")
    if status == "invalid":
        return {
            "status": "invalid", "required": quality_spec.required,
            "checks": list(checks),
        }
    if status != "collected":
        return {
            "status": "not_collected", "required": quality_spec.required,
            "checks": list(checks),
        }

    criteria: list[str] = []
    failed = False
    missing = False

    if "llm" in checks:
        llm = quality.get("llm")
        if not isinstance(llm, Mapping) or quality_spec.llm is None:
            missing = True
        else:
            try:
                correctness = float(llm["correctness"]["rate"])
                formatting = float(llm["format"]["rate"])
            except (KeyError, TypeError, ValueError):
                return {
                    "status": "invalid", "required": quality_spec.required,
                    "checks": list(checks),
                }
            correct_floor = float(quality_spec.llm["correctness_rate_baseline"])
            format_floor = float(quality_spec.llm["format_rate_baseline"])
            criteria.extend((
                f"llm.correctness >= {correct_floor:.4g}",
                f"llm.format >= {format_floor:.4g}",
            ))
            failed = failed or correctness < correct_floor or formatting < format_floor
            baseline_rates = _quality_baseline_rates(baseline_record)
            if baseline_rates is not None:
                ratio = 0.9
                criteria.extend((
                    f"llm.correctness >= baseline*{ratio:.1f}",
                    f"llm.format >= baseline*{ratio:.1f}",
                ))
                failed = failed or correctness < baseline_rates["correctness"] * ratio
                failed = failed or formatting < baseline_rates["format"] * ratio

    if "sd" in checks:
        sd = quality.get("sd")
        automatic = sd.get("automatic_gate") if isinstance(sd, Mapping) else None
        if not isinstance(automatic, Mapping) or quality_spec.sd is None:
            missing = True
        else:
            criteria.append("sd.automatic_gate == passed")
            failed = failed or automatic.get("status") != "passed"
            asset_id = sd.get("asset_id") if isinstance(sd, Mapping) else None
            allowed_assets = quality_spec.sd.get("asset_ids", ())
            criteria.append("sd.asset_id is declared by the plan")
            failed = failed or asset_id not in allowed_assets
            # KIP-16: required 时人工审核必须 passed（双人目视证据绑定）。
            if quality_spec.required:
                manual = sd.get("manual_review") if isinstance(sd, Mapping) else None
                plan_manual = quality_spec.manual_review or {}
                required_reviewers = int(plan_manual.get("reviewers_required", 0) or 0)
                criteria.append(f"sd.manual_review == passed (reviewers >= {required_reviewers})")
                manual_ok = (
                    isinstance(manual, Mapping)
                    and manual.get("status") == "passed"
                    and int(manual.get("required_reviewers", 0) or 0) >= required_reviewers
                )
                failed = failed or not manual_ok

    if "gemma_judge" in checks:
        gemma = quality.get("gemma_judge")
        if not isinstance(gemma, Mapping) or quality_spec.gemma_judge is None:
            missing = True
        else:
            try:
                topic_hit = float(gemma["topic_hit"]["rate"])
                coverage = float(gemma["key_element_coverage"]["rate"])
            except (KeyError, TypeError, ValueError):
                return {
                    "status": "invalid", "required": quality_spec.required,
                    "checks": list(checks),
                }
            topic_floor = float(quality_spec.gemma_judge["topic_hit_rate_baseline"])
            coverage_floor = float(
                quality_spec.gemma_judge["key_element_coverage_baseline"],
            )
            criteria.extend((
                f"gemma_judge.topic_hit >= {topic_floor:.4g}",
                f"gemma_judge.key_element_coverage >= {coverage_floor:.4g}",
            ))
            failed = failed or topic_hit < topic_floor or coverage < coverage_floor
            # KIP-16: required 时判题结果必须已人工复核（evidence 绑定）。
            if quality_spec.required:
                plan_manual = quality_spec.manual_review or {}
                required_reviewers = int(plan_manual.get("reviewers_required", 0) or 0)
                criteria.append(f"gemma_judge.manual_review == passed (reviewers >= {required_reviewers})")
                manual = gemma.get("manual_review") if isinstance(gemma, Mapping) else None
                manual_ok = (
                    isinstance(manual, Mapping)
                    and manual.get("status") == "passed"
                    and int(manual.get("required_reviewers", 0) or 0) >= required_reviewers
                )
                failed = failed or not manual_ok

    return {
        "status": "failed" if failed else "not_collected" if missing else "passed",
        "required": quality_spec.required,
        "checks": list(checks),
        "criteria": criteria,
    }


def _combine_gate_status(performance: str, quality: Mapping[str, Any]) -> str:
    """Apply the manifest's opt-in EX-N3 total-gate rule."""
    if not quality.get("required"):
        return performance
    quality_status = quality.get("status")
    if performance == "failed" or quality_status == "failed":
        return "failed"
    if performance == "invalid" or quality_status == "invalid":
        return "invalid"
    # §6.2.4: required-but-not-collected is visible but not a failure.
    return performance


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
    metrics = dict(outcome.metrics)
    raw_quality = metrics.pop("quality_evidence", None)
    baseline = records.get(unit.baseline_experiment_id or "") if unit.baseline_experiment_id else None
    baseline_metrics = baseline.get("metrics") if baseline else None
    gate_status = "invalid"
    gate_threshold = ""
    if unit.gate is not None:
        performance_status, gate_threshold = evaluate_gate(
            unit.gate, metrics,
            baseline_metrics=baseline_metrics,
        )
    elif outcome.status == "passed":
        performance_status = "passed"
    else:
        performance_status = "failed"
    if outcome.status != "passed":
        performance_status = "failed"

    prompt_set_sha = ""
    prompt_count = 0
    try:
        prompts = prompt_set_dir / "prompts.jsonl"
        if prompts.is_file():
            prompt_set_sha = _sha256(prompts)
            prompt_count = sum(1 for _ in prompts.open(encoding="utf-8"))
    except OSError:
        pass

    try:
        quality = normalize_quality_evidence(
            raw_quality,
            expected_prompt_set={
                "id": plan.prompt_set["id"],
                "sha256": prompt_set_sha or plan.prompt_set.get("sha256", ""),
            },
            expected_gemma_judge=(
                plan.quality.gemma_judge if plan.quality else None
            ),
        )
    except QualityEvidenceError:
        # Result files are untrusted input.  Preserve only a stable redacted state.
        quality = invalid_quality_evidence()
    quality_gate = evaluate_quality_gate(
        quality, plan.quality, unit.quality_checks,
        baseline_record=baseline,
    )
    gate_status = _combine_gate_status(performance_status, quality_gate)

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
        "metrics": metrics,
        "quality": quality,
        "quality_gate": quality_gate,
        "performance_gate": {
            "metric": unit.gate.metric if unit.gate else "",
            "threshold": gate_threshold,
            "status": performance_status,
        },
        "calibration": (
            dict(plan.quality.calibration)
            if plan.quality and plan.quality.calibration else None
        ),
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
