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


def build_report(out_dir: Path, records: list[Mapping], plan_meta: Mapping) -> tuple[Path, Path]:
    """生成 report.md 与 summary.json，返回 (report_path, summary_path)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(r["experiment_id"]): r for r in records}

    status_counts = {"passed": 0, "failed": 0, "invalid": 0}
    for record in records:
        status = record["gate"].get("status")
        if status in status_counts:
            status_counts[status] += 1

    lines: list[str] = []
    lines.append(f"# 实验报告：{plan_meta.get('title', plan_meta.get('plan_id', ''))}")
    lines.append("")
    lines.append(f"- plan_id：`{plan_meta.get('plan_id')}`")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- commit：`{git_describe()}`")
    lines.append(f"- 门判定：passed {status_counts['passed']} / failed {status_counts['failed']} / invalid {status_counts['invalid']}")
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
        "records": f"records.jsonl",
        "report": "report.md",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path, summary_path
