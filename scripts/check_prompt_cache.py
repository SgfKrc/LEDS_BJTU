#!/usr/bin/env python3
"""Check system prompts for runtime values that invalidate a stable prefix.

The checker reports rule and line only. It deliberately never echoes matched
prompt content, so it can be used on prompts that may contain sensitive text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make direct `python scripts/check_prompt_cache.py` invocation work without
# requiring the repository package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_workbench.adaptation.cache_policy import find_prompt_cache_violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check prompts for dynamic cache-prefix content")
    parser.add_argument("paths", nargs="*", type=Path, help="UTF-8 prompt files to check")
    parser.add_argument("--text", help="check one prompt supplied directly")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit a machine-readable report")
    return parser


def _check_path(path: Path) -> dict[str, object]:
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"path": str(path), "ok": False, "error": f"read_error:{exc.__class__.__name__}"}
    violations = find_prompt_cache_violations(prompt)
    return {
        "path": str(path),
        "ok": not violations,
        "violations": [{"line": item.line, "rule": item.rule} for item in violations],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.text is not None and args.paths:
        raise SystemExit("use --text or file paths, not both")
    if args.text is None and not args.paths:
        raise SystemExit("provide at least one prompt file or --text")
    reports = [
        {
            "path": "<text>",
            "ok": not (violations := find_prompt_cache_violations(args.text)),
            "violations": [{"line": item.line, "rule": item.rule} for item in violations],
        }
    ] if args.text is not None else [_check_path(path) for path in args.paths]
    if args.as_json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            if report["ok"]:
                print(f"OK {report['path']}")
            elif "error" in report:
                print(f"ERROR {report['path']} {report['error']}")
            else:
                for violation in report["violations"]:
                    print(f"VIOLATION {report['path']}:{violation['line']} {violation['rule']}")
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
