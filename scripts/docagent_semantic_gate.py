#!/usr/bin/env python3
"""Run the four-sample DOCAGENT M2 semantic agreement gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_audit import scan_all  # noqa: E402
from doc_maintenance_llm import load_docagent_config  # noqa: E402
from doc_maintenance_runtime import run_llm_judgements  # noqa: E402
from doc_maintenance_semantic_gate import (  # noqa: E402
    SemanticBaselineError,
    evaluate_semantic_baseline,
    load_semantic_baseline,
    prepare_semantic_baseline_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DOCAGENT M2 four-sample semantic agreement gate",
    )
    parser.add_argument(
        "--baseline",
        default="fixtures/docagent/m2-semantic-baseline-v1.json",
        help="Human-reviewed four-sample baseline JSON",
    )
    parser.add_argument(
        "--audit",
        help="Existing M1 audit JSON; omit to run a fresh local M1 scan",
    )
    parser.add_argument(
        "--provider", choices=("opencode", "deepseek", "ollama"),
        help="Override DOCAGENT_PROVIDER for this explicit gate run",
    )
    return parser


def _load_audit(path: Path | None) -> dict:
    if path is None:
        return scan_all(None, "none")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticBaselineError("audit is unreadable") from exc
    if not isinstance(value, dict):
        raise SemanticBaselineError("audit is invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline = load_semantic_baseline(REPO_ROOT / args.baseline)
        selected_audit = prepare_semantic_baseline_audit(
            baseline,
            _load_audit(Path(args.audit).expanduser() if args.audit else None),
        )
        config = load_docagent_config(REPO_ROOT / ".env.docagent", args.provider)
    except (SemanticBaselineError, ValueError):
        print(json.dumps({
            "schema_version": "qlh.docagent.semantic_gate.v1",
            "status": "invalid",
            "reason": "baseline_or_configuration_invalid",
        }, ensure_ascii=False))
        return 2

    llm_report = run_llm_judgements(
        selected_audit,
        REPO_ROOT,
        config,
        REPO_ROOT / "build" / "docagent-cache.sqlite",
    )
    result = evaluate_semantic_baseline(baseline, llm_report)
    result["provider_calls"] = llm_report["cost"]["provider_calls"]
    result["cache_hits"] = llm_report["cost"]["cache_hits"]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
