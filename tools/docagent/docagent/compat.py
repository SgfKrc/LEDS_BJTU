"""Compatibility adapter for the main project's historical CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

from .profile import ProfileConfigError
from .report import findings, render_json, render_markdown, render_text
from .rules import RulesConfigError
from .scanner import ScanError, scan_repository


def _configure_stream(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _parse_since(value: str | None) -> timedelta | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)d", value)
    if not match:
        raise ValueError(f"--since format error: {value!r}; expected values such as 7d")
    return timedelta(days=int(match.group(1)))


def _exit_code(report: dict[str, Any], fail_on: str) -> int:
    report_findings = findings(report)
    if fail_on.startswith("R"):
        return int(any(finding["rule"] == fail_on for finding in report_findings))
    if fail_on == "warn":
        return int(any(finding["level"] in {"warn", "error"} for finding in report_findings))
    if fail_on == "error":
        return int(any(finding["level"] == "error" for finding in report_findings))
    return 0


def _write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(render_json(report), encoding="utf-8")
    (output_dir / "audit.md").write_text(render_markdown(report), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="read-only documentation maintenance scanner compatibility entrypoint")
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--json", action="store_true", help="write JSON to stdout")
    formats.add_argument("--markdown", action="store_true", help="write Markdown to stdout")
    parser.add_argument("--since", metavar="DUR", help="scan documents changed within N days, for example 7d")
    parser.add_argument("--profile", help="profile name or project profile file path")
    parser.add_argument(
        "--fail-on", choices=("none", "warn", "error", "R1", "R2", "R3", "R4", "R5"),
        default="error", help="return 1 when this severity or rule is found",
    )
    return parser


def run_compat(argv: list[str] | None, root: str | Path) -> int:
    """Run the historical mechanical CLI contract against ``root``."""
    _configure_stream(sys.stdout)
    _configure_stream(sys.stderr)
    args = _build_parser().parse_args(argv)
    repo_root = Path(root).expanduser().resolve()
    output_dir = repo_root / "build" / "doc-audit"
    try:
        report = scan_repository(repo_root, since=_parse_since(args.since), profile=args.profile)
    except (ProfileConfigError, RulesConfigError) as exc:
        print(f"docagent: configuration error: {exc}", file=sys.stderr)
        return 2
    except (ScanError, ValueError) as exc:
        print(f"docagent: {exc}", file=sys.stderr)
        return 2
    try:
        _write_outputs(report, output_dir)
    except OSError as exc:
        print(f"docagent: cannot write compatibility report: {exc}", file=sys.stderr)
        return 2
    if args.json:
        sys.stdout.write(render_json(report))
    elif args.markdown:
        sys.stdout.write(render_markdown(report))
    else:
        sys.stdout.write(render_text(report))
        print(f"清单: {output_dir / 'audit.md'}")
    return _exit_code(report, args.fail_on)


__all__ = ["run_compat"]
