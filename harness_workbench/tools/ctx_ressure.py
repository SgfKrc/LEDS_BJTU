"""Deterministic context-pressure runner for ``TOOL-CTX-RESS-01``.

The pressure tool is an orchestration and invariant-checking layer around the
existing context measurement fixture.  It never invokes a model, opens a
network connection, or treats early-fact recall as answer quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..research.context_measure import (
    DEFAULT_BUDGETS,
    STRATEGIES,
    ContextMeasureFixture,
    ContextMeasureReport,
    build_context_measure_fixture,
    run_context_measure,
)


CTX_RESSURE_SCHEMA = "qlh.harness.ctx_ressure.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PressureCheck:
    """One deterministic invariant result for a pressure run."""

    check_id: str
    passed: bool
    observed: Any
    expected: Any
    detail: str

    def __post_init__(self) -> None:
        if not self.check_id or not isinstance(self.detail, str):
            raise ValueError("pressure check identity and detail are required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
            "detail": self.detail,
        }


def _check(
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    detail: str,
) -> PressureCheck:
    return PressureCheck(check_id, bool(passed), observed, expected, detail)


def _evaluate_checks(measurement: ContextMeasureReport) -> tuple[PressureCheck, ...]:
    observations = measurement.observations
    expected_cells = len(measurement.budgets) * len(STRATEGIES)
    checks: list[PressureCheck] = []
    checks.append(
        _check(
            "cells_complete",
            len(observations) == expected_cells,
            len(observations),
            expected_cells,
            "one observation is present for every strategy and budget",
        )
    )
    bounded = all(item.input_tokens <= item.input_budget for item in observations)
    checks.append(
        _check(
            "input_budget_bound",
            bounded,
            max((item.input_tokens - item.input_budget for item in observations), default=0),
            "<= 0",
            "assembled context never exceeds its input budget",
        )
    )
    provenance = all(
        item.fixture_digest == measurement.fixture.digest
        and item.seed == measurement.seed
        and item.runner_kind == "fixture"
        and not item.network_used
        and not item.weights_loaded
        for item in observations
    )
    checks.append(
        _check(
            "fixture_provenance",
            provenance,
            {
                "fixture_digest": measurement.fixture.digest,
                "seed": measurement.seed,
                "runner_kind": "fixture",
                "network_used": False,
                "weights_loaded": False,
            },
            "fixed fixture-only provenance",
            "every cell carries the same model-free provenance envelope",
        )
    )
    monotonic: dict[str, bool] = {}
    for strategy in STRATEGIES:
        values = [
            item.early_facts_recalled
            for item in observations
            if item.strategy == strategy
        ]
        monotonic[strategy] = values == sorted(values)
    checks.append(
        _check(
            "recall_curve_monotonic",
            all(monotonic.values()),
            monotonic,
            {strategy: True for strategy in STRATEGIES},
            "early-fact recall must not regress as the budget increases",
        )
    )
    memory_rows = [item for item in observations if item.strategy == "memory"]
    non_memory_clean = all(
        item.memory_entries_written == 0 and item.memory_entries_recalled == 0
        for item in observations
        if item.strategy != "memory"
    )
    memory_bounded = all(
        item.memory_entries_recalled <= item.memory_entries_written
        for item in memory_rows
    )
    checks.append(
        _check(
            "memory_extract_recall_bound",
            non_memory_clean and memory_bounded,
            {
                "non_memory_clean": non_memory_clean,
                "recall_not_above_written": memory_bounded,
            },
            {"non_memory_clean": True, "recall_not_above_written": True},
            "memory extraction and recall stay scoped to the memory strategy",
        )
    )
    checks.append(
        _check(
            "folding_observed",
            any(item.omitted_messages > 0 for item in observations),
            sum(item.omitted_messages for item in observations),
            "> 0",
            "the 30-round fixture exercises bounded context folding",
        )
    )
    return tuple(checks)


def _curve_summary(measurement: ContextMeasureReport) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        rows = [item for item in measurement.observations if item.strategy == strategy]
        full_recall = next(
            (item.input_budget for item in rows if item.early_facts_recalled == item.early_facts_total),
            None,
        )
        summary[strategy] = {
            "budget_count": len(rows),
            "min_recall": min((item.early_fact_recall_rate for item in rows), default=0.0),
            "max_recall": max((item.early_fact_recall_rate for item in rows), default=0.0),
            "first_full_recall_budget": full_recall,
            "max_omitted_messages": max((item.omitted_messages for item in rows), default=0),
        }
    return summary


@dataclass(frozen=True, slots=True)
class ContextPressureReport:
    """Pressure-run output with curves and invariant checks."""

    measurement: ContextMeasureReport
    checks: tuple[PressureCheck, ...]
    schema: str = CTX_RESSURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CTX_RESSURE_SCHEMA:
            raise ValueError("unsupported context pressure schema")
        if self.measurement.fixture.rounds != 30:
            raise ValueError("context pressure requires the 30-round fixture")
        if not self.checks:
            raise ValueError("context pressure requires invariant checks")

    @property
    def valid(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "fixture": self.measurement.fixture.as_dict(),
            "budgets": list(self.measurement.budgets),
            "strategies": list(STRATEGIES),
            "curve_summary": _curve_summary(self.measurement),
            "checks": [check.as_dict() for check in self.checks],
            "measurement": self.measurement.as_dict(),
            "runner_kind": "fixture",
            "network_used": False,
            "weights_loaded": False,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-CTX-RESS-01 context pressure report",
            "",
            f"- Valid: `{str(self.valid).lower()}`; fixture: `{self.measurement.fixture.id}`; rounds: `30`",
            f"- Budgets: `{', '.join(str(value) for value in self.measurement.budgets)}`; seed: `{self.measurement.seed}`",
            "- Runner: `fixture`; weights loaded: `false`; network used: `false`.",
            "- Early-fact recall is an assembled-context metric, not model answer quality.",
            "",
            "## Recall curve",
            "",
            "| strategy | budget | recalled | recall rate | input tokens | omitted messages | memory written | memory recalled |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in self.measurement.observations:
            lines.append(
                f"| `{item.strategy}` | {item.input_budget} | {item.early_facts_recalled}/{item.early_facts_total} | "
                f"{item.early_fact_recall_rate:.3f} | {item.input_tokens} | {item.omitted_messages} | "
                f"{item.memory_entries_written} | {item.memory_entries_recalled} |"
            )
        lines.extend(("", "## Invariant checks", "", "| check | passed | observed | detail |", "| --- | --- | --- | --- |"))
        for check in self.checks:
            observed = json.dumps(check.observed, ensure_ascii=False, sort_keys=True)
            lines.append(f"| `{check.check_id}` | {str(check.passed).lower()} | `{observed}` | {check.detail} |")
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def run_context_pressure(
    fixture: ContextMeasureFixture | None = None,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 17,
) -> ContextPressureReport:
    """Run the 30-round context pressure matrix without model or network access."""

    fixture = fixture or build_context_measure_fixture(seed=seed)
    if fixture.rounds != 30:
        raise ValueError("context pressure requires the 30-round fixture")
    measurement = run_context_measure(fixture, budgets=budgets, seed=seed)
    return ContextPressureReport(measurement, _evaluate_checks(measurement))


def build_context_pressure_report(
    fixture: ContextMeasureFixture | None = None,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 17,
) -> ContextPressureReport:
    """Report-oriented alias for :func:`run_context_pressure`."""

    return run_context_pressure(fixture, budgets=budgets, seed=seed)


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budgets must be comma-separated integers") from exc
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    if tuple(sorted(set(budgets))) != budgets or any(item <= 0 for item in budgets):
        raise argparse.ArgumentTypeError("budgets must be sorted, unique, and positive")
    return budgets


def _write_text(path_value: str, content: str) -> None:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixture-only context pressure matrix")
    parser.add_argument("--budgets", type=_parse_budgets, default=DEFAULT_BUDGETS, help="sorted input budgets, e.g. 64,96,128")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be non-negative")
    report = run_context_pressure(budgets=args.budgets, seed=args.seed)
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    outputs = 0
    if args.json_path:
        if args.json_path == "-":
            print(json_text, end="")
        else:
            _write_text(args.json_path, json_text)
        outputs += 1
    if args.markdown_path:
        if args.markdown_path == "-":
            print(markdown_text, end="")
        else:
            _write_text(args.markdown_path, markdown_text)
        outputs += 1
    if not outputs:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "CTX_RESSURE_SCHEMA",
    "ContextPressureReport",
    "PressureCheck",
    "build_context_pressure_report",
    "build_parser",
    "main",
    "run_context_pressure",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
