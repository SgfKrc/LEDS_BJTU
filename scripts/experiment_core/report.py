"""报告生成器（EX-N1）：report.md（对照表/失败明细/结论模板）+ summary.json。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .collector import git_describe


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _comparison_row(record: Mapping, baseline: Mapping | None) -> list[str]:
    metric = record["gate"].get("metric") or ""
    value = record["metrics"].get(metric)
    base_value = baseline["metrics"].get(metric) if baseline else None
    diff = ""
    pct = ""
    if value is not None and base_value is not None:
        try:
            diff = f"{float(value) - float(base_value):+.4g}"
            pct = f"{(float(value) - float(base_value)) / float(base_value) * 100:+.2f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return [
        record["experiment_id"],
        record["experiment_name"],
        _fmt(value),
        _fmt(base_value),
        diff,
        pct,
        str(record["runs"]),
        record["gate"].get("status", "?"),
    ]


def _quality_cells(record: Mapping) -> list[str]:
    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        return ["not_collected", "-", "-", "-", "-", "-", "-", "not_collected"]
    llm = quality.get("llm")
    sd = quality.get("sd")
    gemma = quality.get("gemma_judge")
    correctness = llm.get("correctness") if isinstance(llm, Mapping) else None
    formatting = llm.get("format") if isinstance(llm, Mapping) else None
    automatic = sd.get("automatic_gate") if isinstance(sd, Mapping) else None
    manual = sd.get("manual_review") if isinstance(sd, Mapping) else None
    topic_hit = gemma.get("topic_hit") if isinstance(gemma, Mapping) else None
    coverage = gemma.get("key_element_coverage") if isinstance(gemma, Mapping) else None
    return [
        str(quality.get("status", "not_collected")),
        _fmt(correctness.get("rate") if isinstance(correctness, Mapping) else None),
        _fmt(formatting.get("rate") if isinstance(formatting, Mapping) else None),
        str(automatic.get("status", "-") if isinstance(automatic, Mapping) else "-"),
        str(manual.get("status", "-") if isinstance(manual, Mapping) else "-"),
        _fmt(topic_hit.get("rate") if isinstance(topic_hit, Mapping) else None),
        _fmt(coverage.get("rate") if isinstance(coverage, Mapping) else None),
        str(
            record.get("quality_gate", {}).get("status", "not_collected")
            if isinstance(record.get("quality_gate"), Mapping) else "not_collected"
        ),
    ]


def build_report(out_dir: Path, records: list[Mapping], plan_meta: Mapping) -> tuple[Path, Path]:
    """生成 report.md 与 summary.json，返回 (report_path, summary_path)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(r["experiment_id"]): r for r in records}

    status_counts = {"passed": 0, "failed": 0, "invalid": 0}
    quality_counts = {"collected": 0, "not_collected": 0, "invalid": 0}
    quality_gate_counts = {"passed": 0, "failed": 0, "not_collected": 0, "invalid": 0}
    for record in records:
        status = record["gate"].get("status")
        if status in status_counts:
            status_counts[status] += 1
        quality = record.get("quality")
        quality_status = quality.get("status") if isinstance(quality, Mapping) else "not_collected"
        if quality_status not in quality_counts:
            quality_status = "invalid"
        quality_counts[quality_status] += 1
        quality_gate = record.get("quality_gate")
        quality_gate_status = (
            quality_gate.get("status") if isinstance(quality_gate, Mapping)
            else "not_collected"
        )
        if quality_gate_status not in quality_gate_counts:
            quality_gate_status = "invalid"
        quality_gate_counts[quality_gate_status] += 1

    lines: list[str] = []
    lines.append(f"# 实验报告：{plan_meta.get('title', plan_meta.get('plan_id', ''))}")
    lines.append("")

    quality_policy = plan_meta.get("quality")
    has_ex_n3_policy = isinstance(quality_policy, Mapping)
    lines.append("## 质量证据与门（EX-N3）" if has_ex_n3_policy else "## 质量证据（EX-N3-S0）")
    lines.append("")
    if has_ex_n3_policy:
        if quality_policy.get("required"):
            lines.append("质量门已由 manifest 声明为 required；总 gate = 性能 gate ∧ 质量 gate，缺失证据单独标记。")
        else:
            lines.append("质量门独立记录，不改变既有性能 gate；仅 manifest 声明 required 时才参与总 gate。")
    else:
        lines.append("本节只汇总结构化证据，不改变既有性能 gate。")
    lines.append("| id | evidence | LLM correct rate | LLM format rate | SD automatic gate | manual review | Gemma topic hit | Gemma key coverage | quality gate |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for record in records:
        lines.append("| " + record["experiment_id"] + " | " + " | ".join(_quality_cells(record)) + " |")
    lines.append("")
    lines.append(f"- plan_id：`{plan_meta.get('plan_id')}`")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- commit：`{git_describe()}`")
    lines.append(f"- 门判定：passed {status_counts['passed']} / failed {status_counts['failed']} / invalid {status_counts['invalid']}")
    lines.append(
        "- 质量门："
        f"passed {quality_gate_counts['passed']} / failed {quality_gate_counts['failed']} / "
        f"not_collected {quality_gate_counts['not_collected']} / invalid {quality_gate_counts['invalid']}"
    )
    lines.append("")

    lines.append("## 环境与工件")
    lines.append("")
    env = plan_meta.get("env") or {}
    for key in ("os", "cpu", "gpu", "ram_gb", "engine", "torch", "cuda"):
        if env.get(key) not in (None, ""):
            lines.append(f"- {key}：{env[key]}")
    prompt_set = plan_meta.get("prompt_set") or {}
    lines.append(f"- prompt_set：`{prompt_set.get('id')}` sha256 `{prompt_set.get('sha256')}`")
    lines.append("")

    lines.append("## 实验清单")
    lines.append("")
    lines.append("| id | 名称 | 状态 | runs | 关键指标 | 错误 |")
    lines.append("|---|---|---|---|---|---|")
    for record in records:
        metric = record["gate"].get("metric") or ""
        value = record["metrics"].get(metric)
        error = record.get("error") or ""
        lines.append(
            f"| {record['experiment_id']} | {record['experiment_name']} "
            f"| {record['gate'].get('status')} | {record['runs']} "
            f"| {metric}={_fmt(value)} | {error} |"
        )
    lines.append("")

    baselined = [r for r in records if r.get("baseline_experiment_id")]
    if baselined:
        lines.append("## 对照表（实验组 vs 基线）")
        lines.append("")
        lines.append("| id | 名称 | 实验值 | 基线值 | 差值 | 百分比 | 样本数 | 门判定 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for record in baselined:
            baseline = by_id.get(record["baseline_experiment_id"] or "")
            lines.append("| " + " | ".join(_comparison_row(record, baseline)) + " |")
        lines.append("")

    retried = [r for r in records if r.get("retries")]
    if retried:
        lines.append("## 失败与重试明细")
        lines.append("")
        lines.append("| id | attempt | exit_code | 原因 | 日志 |")
        lines.append("|---|---|---|---|---|")
        for record in retried:
            for retry in record["retries"]:
                lines.append(
                    f"| {record['experiment_id']} | {retry.get('attempt')} "
                    f"| {retry.get('exit_code')} | {retry.get('reason')} "
                    f"| `{retry.get('log')}` |"
                )
        lines.append("")

    lines.append("## 结论模板（人工复核后填写）")
    lines.append("")
    lines.append("| 假设 | 结果 | 是否支持 | 遗留风险 |")
    lines.append("|---|---|---|---|")
    lines.append("|  |  |  |  |")
    lines.append("")
    lines.append("> 需人工复核的结论（如输出质量）按《自动化优化实验与报告方案》§6 双人目视登记后，报告方可标记 `passed`。")
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "plan_id": plan_meta.get("plan_id"),
        "title": plan_meta.get("title"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_describe(),
        "prompt_set": plan_meta.get("prompt_set"),
        "units": len(records),
        "status": status_counts,
        "quality_evidence": quality_counts,
        "quality_gate": quality_gate_counts,
        "records": f"records.jsonl",
        "report": "report.md",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path, summary_path
