"""Fixture-only red-team rehearsal tool for ``TOOL-REDTEAM-LAB-01``.

This module turns the existing deterministic red-team gate into a small,
auditable rehearsal surface. It records one decision per fixture, but never
echoes untrusted payloads into the report. No model, network, filesystem
asset, or external tool is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..eval import (
    RedTeamDecision,
    RedTeamFixture,
    RedTeamGate,
    builtin_red_team_fixtures,
)


RED_TEAM_LAB_SCHEMA = "qlh.harness.red_team_lab.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_fixture() -> RedTeamFixture:
    return RedTeamFixture(
        "safe-tool-local-v1",
        "tool_authorization",
        {
            "tool": "web_search",
            "scope": "local",
            "profile_status": "verified",
            "capability_status": "verified",
            "production_eligible": True,
        },
        "allowed",
        expected_blocked=False,
    )


@dataclass(frozen=True, slots=True)
class RedTeamLabDecision:
    """One rehearsal row with expected-vs-observed outcome."""

    fixture_id: str
    family: str
    expected_blocked: bool
    expected_reason: str
    blocked: bool
    schema_valid: bool
    reason: str
    unauthorized: bool
    outcome: str

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.family or not self.expected_reason:
            raise ValueError("red-team lab decision identity is required")
        if self.outcome not in {"blocked", "allowed", "mismatch"}:
            raise ValueError("red-team lab decision outcome is invalid")

    @classmethod
    def from_gate(cls, fixture: RedTeamFixture, decision: RedTeamDecision) -> "RedTeamLabDecision":
        matches = decision.blocked == fixture.expected_blocked and decision.reason == fixture.expected_reason
        outcome = "mismatch" if not matches else ("blocked" if decision.blocked else "allowed")
        return cls(
            fixture_id=fixture.id,
            family=fixture.family,
            expected_blocked=fixture.expected_blocked,
            expected_reason=fixture.expected_reason,
            blocked=decision.blocked,
            schema_valid=decision.schema_valid,
            reason=decision.reason,
            unauthorized=decision.unauthorized,
            outcome=outcome,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "family": self.family,
            "expected_blocked": self.expected_blocked,
            "expected_reason": self.expected_reason,
            "blocked": self.blocked,
            "schema_valid": self.schema_valid,
            "reason": self.reason,
            "unauthorized": self.unauthorized,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class RedTeamLabReport:
    """Structured, payload-free red-team rehearsal report."""

    fixture_set_digest: str
    decisions: tuple[RedTeamLabDecision, ...]
    selected_fixture_id: str | None = None
    schema: str = RED_TEAM_LAB_SCHEMA
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != RED_TEAM_LAB_SCHEMA or self.runner_kind != "fixture":
            raise ValueError("red-team lab identity is invalid")
        if not self.decisions:
            raise ValueError("red-team lab requires at least one decision")
        if self.network_used or self.weights_loaded:
            raise ValueError("red-team lab must remain offline and model-free")

    @property
    def fixture_count(self) -> int:
        return len(self.decisions)

    @property
    def blocked_count(self) -> int:
        return sum(item.blocked for item in self.decisions)

    @property
    def allowed_count(self) -> int:
        return self.fixture_count - self.blocked_count

    @property
    def mismatch_count(self) -> int:
        return sum(item.outcome == "mismatch" for item in self.decisions)

    @property
    def unauthorized_pass_count(self) -> int:
        return sum(item.unauthorized for item in self.decisions)

    @property
    def valid(self) -> bool:
        return (
            self.mismatch_count == 0
            and all(item.schema_valid for item in self.decisions)
            and self.unauthorized_pass_count == 0
        )

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "fixture_set_digest": self.fixture_set_digest,
            "selected_fixture_id": self.selected_fixture_id,
            "fixture_count": self.fixture_count,
            "blocked_count": self.blocked_count,
            "allowed_count": self.allowed_count,
            "mismatch_count": self.mismatch_count,
            "unauthorized_pass_count": self.unauthorized_pass_count,
            "decisions": [item.as_dict() for item in self.decisions],
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-REDTEAM-LAB-01 rehearsal report",
            "",
            f"- Valid: `{str(self.valid).lower()}`; fixtures: `{self.fixture_count}`; selected: `{self.selected_fixture_id or 'all'}`",
            f"- Blocked: `{self.blocked_count}`; allowed: `{self.allowed_count}`; mismatches: `{self.mismatch_count}`; unauthorized passes: `{self.unauthorized_pass_count}`",
            "- Runner: `fixture`; weights loaded: `false`; network used: `false`.",
            "- Payloads are deliberately omitted; each row is a gate decision and reason only.",
            "",
            "## Per-fixture rehearsal",
            "",
            "| fixture | family | expected | observed | reason | outcome |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.decisions:
            expected = "block" if item.expected_blocked else "allow"
            observed = "block" if item.blocked else "allow"
            lines.append(f"| `{item.fixture_id}` | `{item.family}` | `{expected}` | `{observed}` | `{item.reason}` | **{item.outcome}** |")
        lines.extend(("", f"Fixture set digest: `{self.fixture_set_digest}`", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def _fixture_values(
    fixtures: Sequence[RedTeamFixture] | None,
    *,
    include_safe: bool,
    family: str | None,
    fixture_id: str | None,
) -> tuple[RedTeamFixture, ...]:
    values = tuple(fixtures if fixtures is not None else builtin_red_team_fixtures())
    if include_safe and fixtures is None:
        values += (_safe_fixture(),)
    if family is not None:
        values = tuple(item for item in values if item.family == family)
    if fixture_id is not None:
        values = tuple(item for item in values if item.id == fixture_id)
    if not values:
        raise ValueError("no red-team fixtures match the requested filter")
    return values


def run_red_team_lab(
    fixtures: Sequence[RedTeamFixture] | None = None,
    *,
    gate: RedTeamGate | None = None,
    include_safe: bool = True,
    family: str | None = None,
    fixture_id: str | None = None,
) -> RedTeamLabReport:
    """Run each selected fixture through the existing deterministic gate."""

    values = _fixture_values(fixtures, include_safe=include_safe, family=family, fixture_id=fixture_id)
    checker = gate or RedTeamGate()
    decisions = tuple(
        RedTeamLabDecision.from_gate(fixture, checker.evaluate(fixture))
        for fixture in values
    )
    return RedTeamLabReport(
        _digest([fixture.as_dict() for fixture in values]),
        decisions,
        selected_fixture_id=fixture_id,
    )


def build_red_team_lab_report(
    fixtures: Sequence[RedTeamFixture] | None = None,
    *,
    gate: RedTeamGate | None = None,
    include_safe: bool = True,
    family: str | None = None,
    fixture_id: str | None = None,
) -> RedTeamLabReport:
    """Report-oriented alias for :func:`run_red_team_lab`."""

    return run_red_team_lab(
        fixtures,
        gate=gate,
        include_safe=include_safe,
        family=family,
        fixture_id=fixture_id,
    )


def _write_text(path_value: str, content: str) -> None:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixture-only red-team rehearsal lab")
    parser.add_argument("--family", choices=("prompt_injection", "tool_authorization", "image_path", "context_injection"))
    parser.add_argument("--fixture", dest="fixture_id", help="run one fixture by id")
    parser.add_argument("--without-safe", action="store_true", help="omit the safe allowlisted comparison fixture")
    parser.add_argument("--list", action="store_true", help="list available fixture ids and exit")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def _list_fixtures() -> str:
    values = list(builtin_red_team_fixtures()) + [_safe_fixture()]
    return "\n".join(f"{fixture.id}\t{fixture.family}" for fixture in values) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(_list_fixtures(), end="")
        return 0
    try:
        report = run_red_team_lab(
            include_safe=not args.without_safe,
            family=args.family,
            fixture_id=args.fixture_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    output_count = 0
    if args.json_path:
        if args.json_path == "-":
            print(json_text, end="")
        else:
            _write_text(args.json_path, json_text)
        output_count += 1
    if args.markdown_path:
        if args.markdown_path == "-":
            print(markdown_text, end="")
        else:
            _write_text(args.markdown_path, markdown_text)
        output_count += 1
    if not output_count:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "RED_TEAM_LAB_SCHEMA",
    "RedTeamLabDecision",
    "RedTeamLabReport",
    "build_parser",
    "build_red_team_lab_report",
    "main",
    "run_red_team_lab",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
