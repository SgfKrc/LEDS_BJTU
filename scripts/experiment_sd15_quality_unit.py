#!/usr/bin/env python3
"""Collect a redacted EX-N3 SD quality-evidence result from a fixed report.

This adapter deliberately does not import Diffusers, load a model, or inspect
an image.  It only turns a hash-pinned ``quality_gate_sd15*.py`` report into
the narrow counters/status evidence accepted by ``experiment_core``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from experiment_core.quality import (
    QualityEvidenceError,
    normalize_quality_evidence,
    sd_evidence_from_gate_report,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_report(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError("quality report SHA-256 does not match the plan")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("quality report root must be an object")
    return parsed


def collect_evidence(
    report: Mapping[str, Any],
    *,
    expected_asset_id: str,
    expected_artifact_id: str,
    expected_mode: str,
) -> dict[str, Any]:
    """Validate fixed report identity and return only normalized SD evidence."""
    if report.get("asset_id") != expected_asset_id:
        raise ValueError("quality report asset_id does not match the plan")
    if report.get("artifact_id") != expected_artifact_id:
        raise ValueError("quality report artifact_id does not match the plan")
    evidence = sd_evidence_from_gate_report(report)
    if evidence.get("mode") != expected_mode:
        raise ValueError("quality report mode does not match the plan")
    # Validate before writing so malformed report fields never become a result.
    normalize_quality_evidence({"sd": evidence})
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect fixed SD quality-gate evidence")
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-asset-id", required=True)
    parser.add_argument("--expected-artifact-id", required=True)
    parser.add_argument("--expected-mode", default="text_to_image")
    parser.add_argument("--result-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _read_report(Path(args.report), args.expected_report_sha256)
        evidence = collect_evidence(
            report,
            expected_asset_id=args.expected_asset_id,
            expected_artifact_id=args.expected_artifact_id,
            expected_mode=args.expected_mode,
        )
        result = {
            "quality_completed": 1,
            "quality_evidence": {"sd": evidence},
        }
        output = Path(args.result_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError, QualityEvidenceError):
        # Do not echo a potentially sensitive report path or report payload.
        print("SD quality evidence collection failed", file=sys.stderr)
        return 2
    print(json.dumps({"status": "collected"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
