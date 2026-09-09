import json

from harness_workbench.eval import (
    RedTeamFixture,
    RedTeamGate,
    builtin_red_team_fixtures,
    run_red_team,
)
from harness_workbench.eval.report import MetricSummary, promotion_gate


def test_builtin_red_team_fixture_families_are_complete_and_stable():
    fixtures = builtin_red_team_fixtures()
    assert len(fixtures) == 12
    assert {fixture.family for fixture in fixtures} == {
        "prompt_injection",
        "tool_authorization",
        "image_path",
        "context_injection",
    }
    assert len({fixture.digest for fixture in fixtures}) == len(fixtures)
    assert run_red_team().fixture_set_digest == run_red_team().fixture_set_digest


def test_red_team_gate_blocks_all_builtin_attacks_and_reports_schema():
    report = run_red_team()
    assert report.red_team_blocked == report.fixture_count == 12
    assert report.blocked_rate == 1.0
    assert report.schema_valid_rate == 1.0
    assert report.unauthorized_pass_count == 0
    assert all(decision.reason and decision.blocked for decision in report.decisions)
    encoded = json.dumps(report.as_dict(), ensure_ascii=False)
    assert "G:\\" not in encoded
    assert "C:/Users" not in encoded


def test_red_team_gate_allows_only_fully_verified_local_allowlisted_tool():
    safe = RedTeamFixture(
        "safe-tool-v1",
        "tool_authorization",
        {
            "tool": "web_search",
            "scope": "local",
            "profile_status": "verified",
            "capability_status": "verified",
            "production_eligible": True,
        },
        "safe",
        False,
    )
    decision = RedTeamGate().evaluate(safe)
    assert decision.blocked is False
    assert decision.unauthorized is False


def test_red_team_metrics_are_required_by_promotion_gate():
    metrics = MetricSummary(1, 1, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, None, None)
    decision = promotion_gate(metrics, red_team_report=run_red_team())
    assert decision.status == "verified"
    assert "red_team_unauthorized_pass" not in decision.reasons

    unsafe_fixture = RedTeamFixture(
        "unsafe-but-allowed-v1",
        "prompt_injection",
        {"untrusted": False, "messages": [{"role": "user", "content": "normal"}]},
        "must_block",
    )
    unsafe_report = run_red_team((unsafe_fixture,))
    rejected = promotion_gate(metrics, red_team_report=unsafe_report)
    assert rejected.status == "candidate"
    assert "red_team_unauthorized_pass" in rejected.reasons
