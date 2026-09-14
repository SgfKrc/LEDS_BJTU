"""Patchouli 无安装 fallback 入口。

用法（无需 pip install）：
    python tools/patchouli/run.py summary --root .
    python tools/patchouli/run.py json --root .
    python tools/patchouli/run.py shelf --root .
安装后（推荐）可直接：python -m patchouli / python -m patchouli.bookshelf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from patchouli.bookshelf import main as shelf_main  # noqa: E402
from patchouli.catalog import main as catalog_main  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in {"shelf", "bookshelf"}:
        return shelf_main(argv[1:])
    if argv and argv[0] == "summary":
        argv = argv[1:] + ["--summary"]
    elif argv and argv[0] == "json":
        argv = argv[1:] + ["--json"]
    return catalog_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
