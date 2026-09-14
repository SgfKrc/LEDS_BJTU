import json
import subprocess
import sys
from pathlib import Path

from harness_workbench.research import ROLE_ASYMMETRY_SCHEMA, build_role_asymmetry_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "role_asymmetry_report.py"


def test_role_report_is_deterministic_and_fixture_only():
    first = build_role_asymmetry_report()
    second = build_role_asymmetry_report()

    assert first.digest == second.digest
    payload = first.as_dict()
    assert payload["schema"] == ROLE_ASYMMETRY_SCHEMA
    assert payload["runner_kind"] == "fixture"
    assert payload["weights_loaded"] is False
    assert payload["network_used"] is False
    assert payload["report_digest"] == first.digest


def test_report_separates_reference_from_experimental_and_planned_roles():
    report = build_role_asymmetry_report()
    comparisons = {item["id"]: item for item in report.as_dict()["comparisons"]}

    assert comparisons["deepseek-reference"]["status"] == "reference_only"
    assert comparisons["draft-verify"]["status"] == "experimental"
    assert comparisons["sub1b-specialization"]["status"] == "planned"
    assert any("does not reproduce" in item for item in comparisons["deepseek-reference"]["non_equivalences"])
    assert "DS3-0324-7B or an explicitly permitted external endpoint" in comparisons["draft-verify"]["mapping"]["verify"]


def test_hypotheses_have_controls_metrics_and_falsifiers():
    report = build_role_asymmetry_report()
    hypotheses = {item["id"]: item for item in report.as_dict()["hypotheses"]}

    assert set(hypotheses) == {"ASYM-01", "ASYM-02", "ASYM-03"}
    assert "tokens_per_round" in hypotheses["ASYM-02"]["primary_metrics"]
    assert "tokens_per_round <= 1.5" in hypotheses["ASYM-02"]["falsifier"]
    assert "shared tokenizer identity" in hypotheses["ASYM-01"]["required_evidence"]
    assert all(item["control"] and item["falsifier"] for item in hypotheses.values())


def test_report_cli_emits_machine_readable_json_without_model_access():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "-"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == ROLE_ASYMMETRY_SCHEMA
    assert payload["runner_kind"] == "fixture"
    assert "weights_loaded" not in result.stderr


def test_markdown_makes_production_boundary_visible():
    markdown = build_role_asymmetry_report().to_markdown()

    assert "QLH does not reproduce" in markdown
    assert "architecture reference" in markdown
    assert "Exact distribution" in markdown
    assert "production" in markdown.lower()
