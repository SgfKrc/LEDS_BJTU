#!/usr/bin/env python3
"""Summarize three EX-N3 baseline rounds without changing thresholds.

The resulting JSON is a review artifact for the explicit §6.2.5 plan/document
revision.  It never edits a plan, rubric, or source file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _rate(record: Mapping[str, Any], name: str) -> float | None:
    try:
        return float(record["quality"]["llm"][name]["rate"])
    except (KeyError, TypeError, ValueError):
        return None


def _summary(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    variation = max(abs(value - median) for value in values)
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "median": median,
        "variation_margin": variation,
        "suggested_floor": max(0.0, median - variation),
    }


def _earliest_record(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select by recorded timestamp, with input order as the legacy fallback."""
    return min(items, key=lambda record: str(record.get("timestamp", "")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize an EX-N3 three-round calibration")
    parser.add_argument("--records", required=True, help="records.jsonl from one calibration plan")
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--select-earliest-unique", action="store_true",
        help="explicitly select the earliest record for duplicate experiment IDs and audit it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records_path = Path(args.records).expanduser()
    try:
        records = [
            json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read EX-N3 records: {exc}", file=sys.stderr)
        return 2

    selected = [
        record for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("calibration"), Mapping)
        and record["calibration"].get("series_id") == args.series_id
    ]
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected:
        experiment_id = record.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            print("EX-N3 calibration record is missing experiment_id", file=sys.stderr)
            return 1
        by_unit[experiment_id].append(record)
    duplicates = [
        {"experiment_id": experiment_id, "observed_records": len(items)}
        for experiment_id, items in sorted(by_unit.items()) if len(items) > 1
    ]
    if duplicates and not args.select_earliest_unique:
        print(
            "EX-N3 calibration contains duplicate experiment records; rerun after resolving "
            "or use the explicit --select-earliest-unique recovery mode",
            file=sys.stderr,
        )
        return 1
    selected = [
        _earliest_record(items)
        for _experiment_id, items in sorted(by_unit.items())
    ]
    correctness = [value for record in selected if (value := _rate(record, "correctness")) is not None]
    formatting = [value for record in selected if (value := _rate(record, "format")) is not None]
    if len(correctness) != 3 or len(formatting) != 3:
        print("EX-N3 calibration requires exactly three completed LLM evidence rounds", file=sys.stderr)
        return 1
    payload = {
        "schema_version": "qlh.experiment_quality_calibration.v1",
        "series_id": args.series_id,
        "rounds_required": 3,
        "rounds_observed": 3,
        "source_records": sum(len(items) for items in by_unit.values()),
        "duplicate_recovery": {
            "mode": "earliest_per_experiment_id" if duplicates else "none",
            "duplicates": duplicates,
        },
        "correctness": _summary(correctness),
        "format": _summary(formatting),
        "next_action": "Review this artifact, then explicitly revise §6.2.3 and create a new plan threshold_version; no threshold was changed automatically.",
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "series_id": args.series_id,
        "rounds_observed": 3,
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
