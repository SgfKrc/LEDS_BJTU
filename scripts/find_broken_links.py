#!/usr/bin/env python
"""find_broken_links.py — 精确列出仓库中「指向不存在文件」的 markdown 链接

与 scripts/check_doc_links.py 的关注点一致，但直接给出「文件:行:链接文本 -> 目标」三元组，
便于逐条修复。只读，不修改。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
SKIP_DIRS = {".git", "_to_delete", "_archive", "build", "local_docs", ".venv", "node_modules"}


def main() -> int:
    broken = []
    for path in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for text_part, target in LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:", "#", "ssh:")):
                    continue
                clean = target.split("#")[0].strip()
                if not clean:
                    continue
                resolved = (path.parent / clean).resolve()
                if not resolved.exists():
                    broken.append((path.relative_to(ROOT), lineno, text_part, target))

    if not broken:
        print("无断链")
        return 0
    print(f"共 {len(broken)} 处断链：")
    for rel, lineno, text_part, target in broken:
        print(f"  {rel}:{lineno}")
        print(f"      链接文本: {text_part}")
        print(f"      目标    : {target}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
