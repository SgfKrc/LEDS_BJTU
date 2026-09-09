"""Stable text renderers for scanner reports."""

from __future__ import annotations

import json
from typing import Any, Mapping


def findings(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [finding for doc in report["docs"] for finding in doc["findings"]]


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=1) + "\n"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def render_markdown(report: Mapping[str, Any]) -> str:
    report_findings = findings(report)
    profile = report["profile"]
    lines = [
        "# Documentation maintenance audit",
        "",
        f"> run_ts: {_cell(report['run_ts'])} | profile: {_cell(profile['name'])} | "
        f"ruleset: v{_cell(report['ruleset_version'])} | fingerprint: `{_cell(report['rules_fingerprint'])}` | "
        f"report: `{_cell(report.get('report_fingerprint') or '')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Documents scanned | {len(report['docs'])} |",
        f"| Findings | {len(report_findings)} |",
        "",
        "## Findings",
        "",
        "| Document | Rule | Level | Evidence |",
        "|---|---|---|---|",
    ]
    if report_findings:
        for doc in report["docs"]:
            for finding in doc["findings"]:
                lines.append(
                    f"| {_cell(doc['doc'])} | {_cell(finding['rule'])} | "
                    f"{_cell(finding['level'])} | {_cell(finding['message'])} |"
                )
    else:
        lines.append("| _none_ |  |  | No findings |")
    lines.extend([
        "",
        "## Documents",
        "",
        "| Document | Status | Updated | Findings |",
        "|---|---|---|---:|",
    ])
    for doc in report["docs"]:
        lines.append(
            f"| {_cell(doc['doc'])} | {_cell(doc.get('status_line') or '')} | "
            f"{_cell(doc.get('updated_at') or '')} | {len(doc['findings'])} |"
        )
    return "\n".join(lines) + "\n"


def render_text(report: Mapping[str, Any]) -> str:
    report_findings = findings(report)
    lines = [f"Scan {len(report['docs'])} documents, {len(report_findings)} findings"]
    for doc in report["docs"]:
        for finding in doc["findings"]:
            lines.append(
                f"  [{finding['level']}] {doc['doc']} :: {finding['rule']} {finding['message']}"
            )
    lines.append(f"Ruleset v{report['ruleset_version']} fingerprint={report['rules_fingerprint']}")
    if report.get("report_fingerprint"):
        lines.append(f"Report fingerprint={report['report_fingerprint']}")
    return "\n".join(lines) + "\n"


def render_delta_markdown(delta: Mapping[str, Any]) -> str:
    summary = delta["summary"]
    lines = [
        "# Documentation maintenance dry-run",
        "",
        f"> baseline fingerprint: `{_cell(delta['baseline']['rules_fingerprint'])}` | "
        f"current fingerprint: `{_cell(delta['current']['rules_fingerprint'])}`",
        "",
        "## Summary",
        "",
        "| Delta | Count |",
        "|---|---:|",
        f"| New findings | {summary['new']} |",
        f"| Gone findings | {summary['gone']} |",
        f"| Changed findings | {summary['changed']} |",
        f"| Affected documents | {summary['affected_docs']} |",
        "",
    ]
    for title, key, columns in (
        ("New", "new", ("Document", "Rule", "Level", "Evidence")),
        ("Gone", "gone", ("Document", "Rule", "Level", "Evidence")),
    ):
        lines.extend([f"## {title}", "", "| " + " | ".join(columns) + " |", "|---|---|---|---|"])
        items = delta[key]
        if items:
            lines.extend(
                f"| {_cell(item['doc'])} | {_cell(item['rule'])} | {_cell(item['level'])} | {_cell(item['message'])} |"
                for item in items
            )
        else:
            lines.append("| _none_ |  |  |  |")
        lines.append("")
    lines.extend(["## Changed", "", "| Document | Rule | Old | New |", "|---|---|---|---|"])
    if delta["changed"]:
        lines.extend(
            f"| {_cell(item['doc'])} | {_cell(item['rule'])} | "
            f"{_cell(item['old']['level'])}: {_cell(item['old']['message'])} | "
            f"{_cell(item['new']['level'])}: {_cell(item['new']['message'])} |"
            for item in delta["changed"]
        )
    else:
        lines.append("| _none_ |  |  |  |")
    lines.extend(["", "## Affected Documents", ""])
    lines.extend(f"- {_cell(doc)}" for doc in delta["affected_docs"])
    if not delta["affected_docs"]:
        lines.append("- _none_")
    return "\n".join(lines) + "\n"


def render_delta_text(delta: Mapping[str, Any]) -> str:
    summary = delta["summary"]
    lines = [
        "Dry-run delta: "
        f"new={summary['new']} gone={summary['gone']} changed={summary['changed']} "
        f"affected_docs={summary['affected_docs']}"
    ]
    for key, label in (("new", "NEW"), ("gone", "GONE")):
        for item in delta[key]:
            lines.append(f"  [{label}] {item['doc']} :: {item['rule']} {item['message']}")
    for item in delta["changed"]:
        lines.append(f"  [CHANGED] {item['doc']} :: {item['rule']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "findings", "render_delta_markdown", "render_delta_text", "render_json",
    "render_markdown", "render_text",
]
