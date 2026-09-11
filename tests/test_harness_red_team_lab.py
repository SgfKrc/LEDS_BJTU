import json

import pytest

from harness_workbench.eval import RedTeamFixture, RedTeamGate
from harness_workbench.tools import (
    RED_TEAM_LAB_SCHEMA,
    build_red_team_lab_report,
    run_red_team_lab,
)
from harness_workbench.tools.red_team_lab import build_parser, main


def test_red_team_lab_rehearses_all_attacks_and_one_safe_comparison():
    report = run_red_team_lab()

    assert report.schema == RED_TEAM_LAB_SCHEMA
    assert report.valid is True
    assert report.fixture_count == 13
    assert report.blocked_count == 12
    assert report.allowed_count == 1
    assert report.mismatch_count == 0
    assert report.unauthorized_pass_count == 0
    assert all("payload" not in item.as_dict() for item in report.decisions)
    assert all(item.outcome in {"blocked", "allowed"} for item in report.decisions)


def test_red_team_lab_filters_by_family_and_fixture_id_without_changing_gate():
    family_report = build_red_team_lab_report(family="image_path", include_safe=False)
    assert family_report.fixture_count == 4
    assert family_report.blocked_count == 4
    assert {item.family for item in family_report.decisions} == {"image_path"}

    one = run_red_team_lab(fixture_id="tool-unverified-autonomy-v1")
    assert one.fixture_count == 1
    assert one.decisions[0].reason == "tool_capability_not_verified"
    assert one.selected_fixture_id == "tool-unverified-autonomy-v1"


def test_red_team_lab_marks_expected_outcome_drift_as_invalid():
    fixture = RedTeamFixture(
        "expected-block-but-safe-v1",
        "tool_authorization",
        {
            "tool": "web_search",
            "scope": "local",
            "profile_status": "verified",
            "capability_status": "verified",
            "production_eligible": True,
        },
        "tool_capability_not_verified",
        expected_blocked=True,
    )
    report = run_red_team_lab((fixture,), include_safe=False)

    assert report.valid is False
    assert report.mismatch_count == 1
    assert report.decisions[0].outcome == "mismatch"


def test_red_team_lab_digest_and_markdown_are_stable_and_payload_free():
    first = run_red_team_lab(include_safe=False)
    second = run_red_team_lab(include_safe=False)

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "Per-fixture rehearsal" in markdown
    assert "Payloads are deliberately omitted" in markdown
    assert "Ignore previous" not in markdown


def test_red_team_lab_cli_lists_and_writes_reports(tmp_path, capsys):
    assert main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert "prompt-injection-role-v1" in listing
    assert "safe-tool-local-v1" in listing

    json_path = tmp_path / "red-team.json"
    markdown_path = tmp_path / "red-team.md"
    assert main(["--without-safe", "--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == RED_TEAM_LAB_SCHEMA
    assert payload["fixture_count"] == 12
    report_text = json_path.read_text(encoding="utf-8")
    assert "You are now unrestricted" not in report_text
    assert "reveal the system prompt" not in report_text
    assert "Per-fixture rehearsal" in markdown_path.read_text(encoding="utf-8")


def test_red_team_lab_cli_returns_nonzero_for_mismatch(monkeypatch, capsys):
    from harness_workbench.tools import red_team_lab

    class BrokenReport:
        valid = False

        def as_dict(self):
            return {"valid": False}

        def to_markdown(self):
            return "mismatch\n"

    monkeypatch.setattr(red_team_lab, "run_red_team_lab", lambda **_: BrokenReport())
    assert main([]) == 1
    assert capsys.readouterr().out == "mismatch\n"


def test_red_team_lab_parser_rejects_unknown_family():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--family", "unknown"])
