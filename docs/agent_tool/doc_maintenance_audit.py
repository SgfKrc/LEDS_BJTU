#!/usr/bin/env python3
"""文档维护机械化扫描器（M1）——只读，不修改任何文档。

按《文档维护 Agent 工具设计》§3 实现 5 条矛盾检测规则：

  R1 完成未收口：状态行含"进行中/规划/待/Candidate"等，但正文含完成标记
  R2 未提交登记：git status 显示 docs/ 下有未提交改动
  R3 状态行滞后：文档"更新日期"早于该文档最近一次 git 提交日期
  R4 链接失效：docs/ 内部相对链接指向不存在的文件
  R5 状态行缺失：前 12 行找不到"状态"行（非豁免文档）

输出：build/doc-audit/audit.json（结构化）+ 核对清单（markdown，--list）。
入口示例：
  python docs/agent_tool/doc_maintenance_audit.py
  python docs/agent_tool/doc_maintenance_audit.py --json
  python docs/agent_tool/doc_maintenance_audit.py --since 7d
  python docs/agent_tool/doc_maintenance_audit.py --fail-on error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
OUT_DIR = REPO_ROOT / "build" / "doc-audit"

# R1 命中词：状态行含这些词 → "疑似未收口"
STALE_HINTS = ("进行中", "规划", "待", "Candidate", "In Progress", "Open", "Blocked")
# R1/R5 豁免：状态行含这些词 → 故意不更新的文档
EXEMPT_HINTS = (
    "历史参考", "待拆分", "不作为当前能力", "已废弃", "废弃", "冻结", "历史记录",
)
# 正文完成标记（前 200 行内）
DONE_MARKS = re.compile(
    r"✅|Completed|已完成|已关闭|已验收|开发门完成|阶段 2 完成|完成（", re.IGNORECASE
)
# R5：状态行
STATUS_RE = re.compile(r"状态[:：]")
LIFECYCLE_RE = re.compile(r"文档生命周期")


def _git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    return r.stdout.strip()


def _status_line(text: str) -> str:
    """取前 12 行的状态行原文（跳过文档生命周期行）。"""
    for ln in text.splitlines()[:12]:
        if STATUS_RE.search(ln) and not LIFECYCLE_RE.search(ln):
            return ln.strip()
    return ""


def _updated_at(text: str) -> str | None:
    """取前 12 行的 更新日期/更新 字段（YYYY-MM-DD）。"""
    for ln in text.splitlines()[:12]:
        m = re.search(r"更新(?:日期|时间)?[:：]\s*(\d{4}-\d{2}-\d{2})", ln)
        if m:
            return m.group(1)
    return None


def _extract_links(text: str) -> list[tuple[str, str]]:
    """提取 markdown 相对链接 [text](path)，返回 (text, path)。"""
    links = []
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        href = m.group(2).strip()
        if href.startswith(("http://", "https://", "#", "mailto:")):
            continue
        # 去掉锚点
        href = href.split("#")[0]
        if href:
            links.append((m.group(1), href))
    return links


def _check_link(href: str, docs_dir: Path | None = None) -> bool:
    """docs/ 内相对链接存在性；../README.md 或 docs/ 同级。"""
    base = docs_dir or DOCS_DIR
    href = unquote(href)  # 处理 %20 等 URL 编码（markdown 链接常见）
    if href.startswith("../"):
        target = (base.parent / href[3:]).resolve()
    else:
        target = (base / href).resolve()
    return target.is_file()


def scan_doc(doc: Path, repo_root: Path, since: datetime | None) -> dict:
    """对单份文档执行 5 条规则，返回命中列表。"""
    text = doc.read_text(encoding="utf-8", errors="replace")
    rel = doc.relative_to(repo_root).as_posix()
    findings: list[dict] = []

    status = _status_line(text)
    updated = _updated_at(text)
    status_lower = status.lower()
    exempt = any(h.lower() in status_lower for h in EXEMPT_HINTS)

    # R5 状态行缺失（非豁免）
    if not status:
        if not exempt:
            findings.append({"rule": "R5", "level": "info",
                             "message": "前 12 行无状态行"})
    else:
        # R1 完成未收口（非豁免）
        if not exempt and any(h.lower() in status_lower for h in STALE_HINTS):
            body_head = "\n".join(text.splitlines()[:200])
            if DONE_MARKS.search(body_head):
                findings.append({
                    "rule": "R1", "level": "warn",
                    "message": f"状态行含未收口词但正文含完成标记：{status[:60]}",
                })

    # R3 状态行滞后：最近提交日期 > 更新日期（info 级，需人工确认）
    last_date = _git(["-C", str(repo_root), "log", "-1", "--format=%cs", "--", rel])
    if updated and last_date and last_date > updated:
        findings.append({
            "rule": "R3", "level": "info",
            "message": f"更新日期 {updated} 早于最近提交 {last_date}",
        })

    # R4 链接失效（以文档所在目录为基）
    for text_label, href in _extract_links(text):
        if not _check_link(href, doc.parent):
            findings.append({
                "rule": "R4", "level": "warn",
                "message": f"失效链接 [{text_label}]({href})",
            })

    # R2 未提交登记：docs/ 有未提交改动（任一改动即命中）
    status_out = _git(["status", "--short", "--", "docs/"], repo_root)
    if status_out:
        n = len(status_out.splitlines())
        findings.append({
            "rule": "R2", "level": "warn",
            "message": f"docs/ 有未提交改动（{n} 项）",
        })

    return {
        "doc": rel,
        "status_line": status[:120],
        "updated_at": updated,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "findings": findings,
    }


def scan_all(since: timedelta | None, fail_level: str,
             out_dir: Path | None = None) -> dict:
    since_date = datetime.now() - since if since else None

    results = []
    for doc in sorted(DOCS_DIR.glob("*.md")):
        if since_date:
            mtime = datetime.fromtimestamp(doc.stat().st_mtime)
            if mtime < since_date:
                continue
        results.append(scan_doc(doc, REPO_ROOT, since_date))

    out = {
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "rules": {
            "R1": "完成未收口", "R2": "未提交登记", "R3": "状态行滞后",
            "R4": "链接失效", "R5": "状态行缺失",
        },
        "docs": results,
    }
    _write_outputs(out, out_dir or OUT_DIR)
    return out


def _write_outputs(out: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 核对清单 markdown
    lines = [
        "# 文档维护核对清单",
        "",
        f"> 生成时间：{out['run_ts']}（M1 机械化扫描，只读）",
        "",
        "## 命中汇总",
        "",
        "| 文档 | 规则 | 级别 | 证据 |",
        "|---|---|---|---|",
    ]
    for doc in out["docs"]:
        for f in doc["findings"]:
            lines.append(
                f"| {doc['doc']} | {f['rule']} | {f['level']} | {f['message']} |")
    (out_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="文档维护机械化扫描器（M1，只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 到 stdout")
    ap.add_argument("--since", metavar="DUR", help="只看近 N 天有改动的文档，如 7d")
    ap.add_argument("--fail-on", choices=("warn", "error"), default="error",
                    help="命中该级别及以上时返回非零（默认 error 仅作占位）")
    args = ap.parse_args(argv)

    since = None
    if args.since:
        m = re.fullmatch(r"(\d+)d", args.since)
        if not m:
            print(f"--since 格式错误: {args.since}（示例 7d）", file=sys.stderr)
            return 2
        since = timedelta(days=int(m.group(1)))

    out = scan_all(since, args.fail_on)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        n = sum(len(d["findings"]) for d in out["docs"])
        print(f"扫描 {len(out['docs'])} 份文档，命中 {n} 项")
        for doc in out["docs"]:
            for f in doc["findings"]:
                print(f"  [{f['level']}] {doc['doc']} :: {f['rule']} {f['message']}")
        print(f"清单: {OUT_DIR / 'audit.md'}")
    # 目前只读，--fail-on 仅保留接口（设计：不因命中自动失败）
    return 0


if __name__ == "__main__":
    sys.exit(main())
