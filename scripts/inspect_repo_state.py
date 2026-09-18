#!/usr/bin/env python
"""inspect_repo_state.py — 用 UTF-8 正确核实票文件状态 + 断链上下文

背景：上一轮用 PowerShell `Get-Content`（默认 ANSI）读 UTF-8 中文文件，中文匹配全部失败，
导致误判「我的票更新被覆盖」。这里用 Python 显式 UTF-8 重读，给出可靠结论。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TICKET = ROOT / "docs" / "验收清单与资源限制登记.md"

MARKERS = ["P6", "P7", "P8", "P9", "P10", "P11", "P12", "B1/B2",
           "v4 策略", "v5 实施", "第五次更正", "第六次更正",
           "B6 完成", "B8 完成", "cache 机制归因", "issue 材料处置",
           "COMPILE_RECOMPILE_LIMIT", "TODO v2", "static cache", "上游调研"]


def main() -> int:
    t = TICKET.read_text(encoding="utf-8")
    print(f"=== 票文件（UTF-8 读取）共 {len(t.splitlines())} 行，{len(t)} 字符 ===")
    for k in MARKERS:
        print(f"  {k:26s} = {t.count(k)}")

    print("\n=== 章节结构 ===")
    for i, line in enumerate(t.splitlines(), 1):
        if line.startswith("#"):
            print(f"  {i:4d}: {line[:90]}")

    print("\n=== 末尾 4 行（前 200 字符）===")
    for line in t.splitlines()[-4:]:
        print(f"  {line[:200]}")

    print("\n=== 断链定位（check_doc_links 报的目标不存在）===")
    targets = ["Android验证替代路径", "抗弱网通信协议专项计划", "DistilQwen2.5-DS3"]
    for rel in ["README.md", "docs/README.en.md"]:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if any(x in line for x in targets):
                print(f"  {rel}:{i}: {line.strip()[:190]}")

    print("\n=== docs/ 下是否存在这些目标文件 ===")
    for name in ["Android验证替代路径", "抗弱网通信协议专项计划", "DistilQwen2.5-DS3"]:
        hits = [q.name for q in (ROOT / "docs").glob("*.md") if name.split("替代")[0][:4] in q.name]
        print(f"  {name}: {hits if hits else '无相近文件'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
