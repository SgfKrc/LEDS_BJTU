import json

import pytest

from harness_workbench.tools import (
    CTX_RESSURE_SCHEMA,
    build_context_pressure_report,
    run_context_pressure,
)
from harness_workbench.research import build_context_measure_fixture
from harness_workbench.tools.ctx_ressure import build_parser, main


def test_context_pressure_runs_complete_fixture_matrix_and_checks_invariants():
    report = run_context_pressure(budgets=(64, 96, 128, 192))

    assert report.schema == CTX_RESSURE_SCHEMA
    assert report.valid is True
    assert len(report.measurement.observations) == 12
    assert {check.check_id for check in report.checks} == {
        "cells_complete",
        "input_budget_bound",
        "fixture_provenance",
        "recall_curve_monotonic",
        "memory_extract_recall_bound",
        "folding_observed",
    }
    assert all(check.passed for check in report.checks)
    assert report.as_dict()["runner_kind"] == "fixture"
    assert report.as_dict()["network_used"] is False
    assert report.as_dict()["weights_loaded"] is False


def test_context_pressure_is_digest_stable_and_markdown_has_curve_and_checks():
    first = build_context_pressure_report(budgets=(64, 96, 128))
    second = build_context_pressure_report(budgets=(64, 96, 128))

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "## Recall curve" in markdown
    assert "`recall_curve_monotonic`" in markdown
    assert "not model answer quality" in markdown


def test_context_pressure_requires_exact_thirty_round_fixture():
    fixture = build_context_measure_fixture()

    class WrongRoundFixture:
        rounds = 29

    with pytest.raises(ValueError, match="30-round"):
        run_context_pressure(WrongRoundFixture())


def test_context_pressure_parser_rejects_unordered_or_invalid_budgets():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--budgets", "96,64"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--budgets", "64,0"])


def test_context_pressure_cli_writes_json_and_markdown(tmp_path):
    json_path = tmp_path / "pressure.json"
    markdown_path = tmp_path / "pressure.md"

    assert main([
        "--budgets", "64,96",
        "--json", str(json_path),
        "--markdown", str(markdown_path),
    ]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == CTX_RESSURE_SCHEMA
    assert payload["valid"] is True
    assert "Recall curve" in markdown_path.read_text(encoding="utf-8")


def test_context_pressure_cli_returns_failure_for_non_monotonic_custom_report(monkeypatch, capsys):
    from harness_workbench.tools import ctx_ressure

    class BrokenReport:
        valid = False

        def as_dict(self):
            return {"valid": False}

        def to_markdown(self):
            return "broken\n"

    monkeypatch.setattr(ctx_ressure, "run_context_pressure", lambda **_: BrokenReport())
    assert ctx_ressure.main([]) == 1
    assert capsys.readouterr().out == "broken\n"
