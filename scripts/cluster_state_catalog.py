#!/usr/bin/env python
"""Emit the reviewed P4.5 state catalog without reading user state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ★ 2026-09-24 修复：**无条件**把 `src` / 仓库根放到 `sys.path` 最前，去掉原来的
#   `if str(candidate) not in sys.path:` 守卫。
#   那条守卫会让「`PYTHONPATH` 里**已经**有 `src`」的调用方**跳过插入** ⇒ `sys.path[0]` 仍是
#   本脚本所在的 `scripts/` 目录 ⇒ 下面那行**裸名** `import cluster_state_catalog` 会命中
#   **脚本自己** ⇒ 循环导入：
#     ImportError: cannot import name 'build_state_catalog' from partially initialized module
#   实测：`PYTHONPATH=""` ⇒ exit 0；`PYTHONPATH=<repo>/src` ⇒ exit 1。后者正是
#   `scripts/run_test_channels.py` 的 `_pytest_env()`（`:34-41`）注入的口径 ⇒ 于是 `unit` 通道
#   **每次**都在 `tests/test_cluster_state_catalog.py::test_catalog_cli_emits_metadata_only_json`
#   失败（且该用例丢弃 stderr ⇒ 根因不可观测）。
#   回归用例：`tests/test_cluster_state_catalog.py::test_catalog_cli_survives_pythonpath_pointing_at_src`。
for candidate in (ROOT / "src", ROOT):
    candidate_text = str(candidate)
    while candidate_text in sys.path:      # 去掉可能来自 PYTHONPATH 的既有同名条目
        sys.path.remove(candidate_text)
    sys.path.insert(0, candidate_text)

from cluster_state_catalog import build_state_catalog, validate_state_catalog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the P4.5 cluster state catalog")
    parser.add_argument("--out", default="-", help="JSON output path, or '-' for stdout")
    args = parser.parse_args(argv)
    catalog = build_state_catalog()
    payload = validate_state_catalog(catalog).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(rendered)
    else:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
