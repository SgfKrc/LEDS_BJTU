"""PATCH-04 编目诊断：委托 docagent scan（子进程），fail-soft 封装。

数据源：`tools/docagent/run.py scan --root <repo> --json`（只读）。
输出摘要：{"ok", "docs", "by_rule", "by_level", "findings", "error"}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT = 180


def find_docagent(repo_root: Path) -> Path | None:
    """定位子仓 docagent 的 run.py（免安装 fallback 入口）。"""
    candidate = repo_root / "tools" / "docagent" / "run.py"
    return candidate if candidate.is_file() else None


def summarize_scan(payload: dict[str, Any]) -> dict[str, Any]:
    """docagent scan JSON → Patchouli 摘要素养。"""
    findings: list[dict[str, str]] = []
    by_rule: dict[str, int] = {}
    by_level: dict[str, int] = {}
    docs = payload.get("docs") or []
    for entry in docs:
        doc = entry.get("doc", "?")
        for finding in entry.get("findings") or []:
            rule = finding.get("rule", "?")
            level = finding.get("level", "info")
            findings.append({"doc": doc, "rule": rule, "level": level, "message": finding.get("message", "")})
            by_rule[rule] = by_rule.get(rule, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1
    return {
        "ok": True,
        "run_ts": payload.get("run_ts"),
        "docs": len(docs),
        "rules": payload.get("rules", {}),
        "by_rule": dict(sorted(by_rule.items())),
        "by_level": dict(sorted(by_level.items())),
        "findings": findings,
        "error": None,
    }


def run_scan(repo_root: Path, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """运行 docagent scan；任何失败都 fail-soft 为 {"ok": False, "error": ...}。"""
    repo_root = Path(repo_root)
    runner = find_docagent(repo_root)
    if runner is None:
        return {
            "ok": False,
            "docs": 0,
            "by_rule": {},
            "by_level": {},
            "findings": [],
            "error": f"docagent not found: {repo_root / 'tools' / 'docagent' / 'run.py'}",
        }
    try:
        proc = subprocess.run(
            [sys.executable, str(runner), "scan", "--root", str(repo_root), "--json"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # noqa: BLE001
        return {"ok": False, "docs": 0, "by_rule": {}, "by_level": {}, "findings": [], "error": f"subprocess: {exc!r}"}
    if proc.returncode not in (0, 1):  # docagent 用 1 表示发现 findings 的常规退出
        return {
            "ok": False,
            "docs": 0,
            "by_rule": {},
            "by_level": {},
            "findings": [],
            "error": f"exit={proc.returncode}: {proc.stderr.strip()[:300]}",
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "docs": 0, "by_rule": {}, "by_level": {}, "findings": [], "error": f"json: {exc}"}
    return summarize_scan(payload)


def render_diag(result: dict[str, Any], max_lines: int = 40) -> str:
    """诊断摘要 → 纯文本（供 TUI 预览区显示）。"""
    if not result.get("ok"):
        return f"编目诊断失败：\n{result.get('error')}"
    rules = result.get("rules") or {}
    header = [
        "== 编目诊断（docagent scan）==",
        f"扫描文档: {result['docs']} · 问题: {len(result['findings'])}",
        f"按规则: {result['by_rule'] or '无'}",
        f"按级别: {result['by_level'] or '无'}",
        "",
    ]
    if rules:
        header.insert(2, "规则: " + " · ".join(f"{k}={v}" for k, v in rules.items()))
    rows = []
    for finding in result["findings"][:max_lines]:
        rows.append(f"[{finding['level']}/{finding['rule']}] {finding['doc']}\n    {finding['message']}")
    if len(result["findings"]) > max_lines:
        rows.append(f"… 其余 {len(result['findings']) - max_lines} 条省略")
    return "\n".join(header + rows)
