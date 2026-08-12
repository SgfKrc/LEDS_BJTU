"""EX-N3-SD-N1: fixed SD quality-report collection without model execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core import cli
from experiment_core.collector import build_record
from experiment_core.plan import load_plan
from experiment_core.runner import UnitOutcome
from experiment_sd15_quality_unit import collect_evidence


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "quality_reports" / "sd15-text-to-image-passed-v1.json"
PLAN = ROOT / "fixtures" / "experiment-plans" / "plan-quality-sd15-bridge-fixture-v1.json"
ADAPTER = ROOT / "scripts" / "experiment_sd15_quality_unit.py"
FIXTURE_SHA256 = "87a7bb286d21b2b50b1c721504b606367d5e15cd43f644fb2c6da7a018f9ad1f"


def _fixture_report() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _run_adapter(output: Path, *, digest: str = FIXTURE_SHA256) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(ADAPTER),
        "--report", str(FIXTURE),
        "--expected-report-sha256", digest,
        "--expected-asset-id", "sd15_90s_retrovers_v1",
        "--expected-artifact-id", "sd15_90s_retrovers_v1",
        "--expected-mode", "text_to_image",
        "--result-file", str(output),
    ], cwd=ROOT, capture_output=True, text=True, check=False)


def test_sd_report_fixture_hash_is_frozen():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_adapter_collects_only_redacted_sd_evidence(tmp_path):
    output = tmp_path / "result.json"
    completed = _run_adapter(output)
    assert completed.returncode == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result == {
        "quality_completed": 1,
        "quality_evidence": {
            "sd": {
                "mode": "text_to_image",
                "asset_id": "sd15_90s_retrovers_v1",
                "artifact_id": "sd15_90s_retrovers_v1",
                "source_schema_version": 1,
                "automatic_gate": {
                    "passed": True, "output_count": 10, "unique_output_count": 10,
                },
                "manual_review": {"status": "passed", "required_reviewers": 2},
            }
        },
    }
    serialized = json.dumps(result)
    assert "fixture prompt" not in serialized
    assert "C:/fixture" not in serialized
    assert "fixture-reviewer" not in serialized


def test_adapter_rejects_mismatched_report_hash_without_writing_result(tmp_path):
    output = tmp_path / "result.json"
    completed = _run_adapter(output, digest="0" * 64)
    assert completed.returncode == 2
    assert not output.exists()
    assert "SD quality evidence collection failed" in completed.stderr
    assert str(FIXTURE) not in completed.stderr


def test_collector_rejects_sd_evidence_for_an_undeclared_asset(tmp_path):
    plan = load_plan(PLAN)
    raw = collect_evidence(
        _fixture_report(),
        expected_asset_id="sd15_90s_retrovers_v1",
        expected_artifact_id="sd15_90s_retrovers_v1",
        expected_mode="text_to_image",
    )
    raw["asset_id"] = "sd15_other_v1"
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.01,
        metrics={"quality_completed": 1, "quality_evidence": {"sd": raw}},
    )
    record = build_record(
        plan, plan.units[0], outcome, prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    assert record["performance_gate"]["status"] == "passed"
    assert record["quality_gate"]["status"] == "failed"
    assert record["gate"]["status"] == "passed"  # Fixture plan keeps quality advisory.


def test_fixture_plan_runs_through_cli_without_cuda_or_report_leakage(tmp_path):
    output = tmp_path / "experiment"
    assert cli.main(["--plan", str(PLAN), "--out", str(output)]) == 0
    records = (output / "records.jsonl").read_text(encoding="utf-8")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "\"status\": \"passed\"" in records
    assert "fixture prompt" not in records + report
    assert "C:/fixture" not in records + report
    assert "fixture-reviewer" not in records + report
