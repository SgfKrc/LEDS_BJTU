#!/usr/bin/env python3
"""Render the fixture-only CACHE-05 role asymmetry evidence report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_workbench.research import build_role_asymmetry_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def _write(path: str, content: str) -> None:
    if path == "-":
        print(content, end="" if content.endswith("\n") else "\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_role_asymmetry_report()
    if args.json_path:
        _write(args.json_path, json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n")
    if args.markdown_path:
        _write(args.markdown_path, report.to_markdown())
    if not args.json_path and not args.markdown_path:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
