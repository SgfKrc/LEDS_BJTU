#!/usr/bin/env python
"""fix_broken_links.py — 机械修复「目标被归档到 docs/archive/ 但引用未更新」的链接

安全边界（刻意收窄，避免篡改他人意图）：
  * 只处理**目标文件名确实存在于 `docs/archive/`** 的情况 ⇒ 改写为 `archive/<name>`；
  * 目标**完全不存在**的链接（如 README 里的 `docs/Android验证替代路径-2026-09-18.md`）
    **不动** —— 那可能是尚未创建的文档，交由作者决定；
  * 默认 **dry-run**，加 `--apply` 才真正写入。

用法:
  python scripts/fix_broken_links.py            # 预览
  python scripts/fix_broken_links.py --apply    # 写入
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ARCHIVE = DOCS / "archive"
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
SKIP_DIRS = {".git", "_to_delete", "_archive", "build", "local_docs", ".venv", "node_modules"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    archived = {p.name for p in ARCHIVE.glob("*.md")}
    fixes: list[tuple[Path, int, str, str]] = []

    for path in sorted(ROOT.rglob("*.md")):
        # 刻意只处理 `docs/` **直属**文档：本次断链都源于「docs/ 下的计划文档被归档」
        # ⇒ 引用方也在 docs/ 直属。子目录（harness_workbench 子模块 / packaging / tools）
        # 与别仓路径（external/...）不归本脚本管，避免篡改语义。
        if path.parent != DOCS:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for _text, target in LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:", "#", "ssh:")):
                    continue
                clean = target.split("#")[0].strip()
                if not clean or clean.startswith("archive/"):
                    continue
                name = Path(clean).name
                if (path.parent / clean).exists():
                    continue
                # 只修「纯文件名」形式的引用（不带路径），且目标确实在 docs/archive/ 里
                if name in archived and "/" not in clean and "\\" not in clean:
                    fixes.append((path, lineno, target, f"archive/{name}"))

    print(f"可机械修复 {len(fixes)} 处（目标都在 docs/archive/）：")
    for path, lineno, old, new in fixes:
        print(f"  {path.relative_to(ROOT)}:{lineno}\n      {old}\n   -> {new}")

    if not fixes:
        return 0
    if not args.apply:
        print("\n（dry-run；加 --apply 才会写入）")
        return 0

    by_file: dict[Path, list[tuple[str, str]]] = {}
    for path, _lineno, old, new in fixes:
        by_file.setdefault(path, []).append((old, new))
    for path, pairs in by_file.items():
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            text = text.replace(f"]({old})", f"]({new})")
        path.write_text(text, encoding="utf-8")
        print(f"[written] {path.relative_to(ROOT)} ({len(pairs)} 处)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
