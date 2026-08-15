"""EX-N3-GEMMA-S1 counter-only evidence contract tests (no model execution)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core import cli
from experiment_core.collector import build_record
from experiment_core.plan import PlanError, load_plan
from experiment_core.quality import QualityEvidenceError, normalize_quality_evidence
from experiment_core.runner import UnitOutcome
from experiment_gemma_quality_unit import collect_evidence


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "fixtures" / "experiment-plans" / "plan-quality-gemma-bridge-fixture-v1.json"
CONTRACT = ROOT / "fixtures" / "quality_rubrics" / "gemma-judge-counts-v1.json"
EVIDENCE = ROOT / "fixtures" / "quality_reports" / "gemma-judge-counts-fixture-v1.json"
ADAPTER = ROOT / "scripts" / "experiment_gemma_quality_unit.py"
CONTRACT_SHA256 = "593c93446cc04553a2c058f36b988aee53b3c994ff92dde9d1539bcf81144b88"
EVIDENCE_SHA256 = "a0a33752672169585e0bf07eb39ae56879e460ebd64add0262f44311a0a2136f"
GEMMA_SPEC = {
    "model": "gemma4:12b",
    "judge_contract_id": "gemma-judge-counts-v1",
    "judge_contract_sha256": CONTRACT_SHA256,
}


def _raw_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _run_adapter(
    output: Path, *, evidence_digest: str = EVIDENCE_SHA256,
    evidence_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    evidence = evidence_path or EVIDENCE
    return subprocess.run([
        sys.executable, str(ADAPTER),
        "--evidence", str(evidence), "--expected-evidence-sha256", evidence_digest,
        "--judge-contract", str(CONTRACT), "--expected-judge-contract-sha256", CONTRACT_SHA256,
        "--expected-model", "gemma4:12b", "--expected-contract-id", "gemma-judge-counts-v1",
        "--result-file", str(output),
    ], cwd=ROOT, capture_output=True, text=True, check=False)


def _quality_payload() -> dict:
    return {
        "gemma_judge": {
            "model": "gemma4:12b",
            "judge_contract_id": "gemma-judge-counts-v1",
            "judge_contract_sha256": CONTRACT_SHA256,
            "topic_hit": {"evaluated_count": 10, "passed_count": 8},
            "key_element_coverage": {"evaluated_count": 10, "passed_count": 6},
        }
    }


def test_gemma_contract_and_fixture_are_hash_pinned():
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == EVIDENCE_SHA256


def test_counter_only_adapter_collects_v2_evidence_without_runtime_or_text(tmp_path):
    output = tmp_path / "result.json"
    completed = _run_adapter(output)
    assert completed.returncode == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["quality_completed"] == 1
    judge = result["quality_evidence"]["gemma_judge"]
    assert judge == _quality_payload()["gemma_judge"]
    serialized = json.dumps(result)
    assert "prompt" not in serialized
    assert "image" not in serialized
    assert "reasoning" not in serialized


def test_adapter_rejects_hash_drift_and_extra_evidence_fields_without_result(tmp_path):
    output = tmp_path / "result.json"
    assert _run_adapter(output, evidence_digest="0" * 64).returncode == 2
    assert not output.exists()
    raw = _raw_evidence()
    raw["completion"] = "do not store this"
    with pytest.raises(ValueError, match="unsupported"):
        collect_evidence(
            raw, _contract(), expected_model="gemma4:12b",
            expected_contract_id="gemma-judge-counts-v1",
            expected_contract_sha256=CONTRACT_SHA256,
        )


def test_normalizer_binds_model_contract_and_refuses_completion_content():
    normalized = normalize_quality_evidence(
        _quality_payload(), expected_gemma_judge=GEMMA_SPEC,
    )
    assert normalized["schema_version"] == "qlh.experiment_quality.v2"
    assert normalized["gemma_judge"]["topic_hit"]["rate"] == 0.8
    payload = _quality_payload()
    payload["gemma_judge"]["reasoning"] = "must not persist"
    with pytest.raises(QualityEvidenceError, match="unsupported"):
        normalize_quality_evidence(payload, expected_gemma_judge=GEMMA_SPEC)


def test_gemma_v2_record_schema_and_advisory_gate(tmp_path):
    plan = load_plan(PLAN)
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.01,
        metrics={"quality_completed": 1, "quality_evidence": _quality_payload()},
    )
    record = build_record(
        plan, plan.units[0], outcome, prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    assert record["quality_gate"]["status"] == "passed"
    assert record["gate"]["status"] == "passed"
    schema = json.loads((ROOT / "schemas" / "experiment-record.schema.json").read_text(encoding="utf-8"))
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(record)) == []


def test_gemma_required_plan_allowed_with_approved_baselines(tmp_path):
    # Decision 2026-08-13: gemma_judge is eligible for quality.required after
    # the real calibration + dual review; the fixture's 0.70/0.50 floors are at
    # or above the approved 0.70/0.40, so required=true now loads fine.
    raw = json.loads(PLAN.read_text(encoding="utf-8"))
    raw["quality"]["required"] = True
    path = tmp_path / "required-gemma.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    plan = load_plan(path)
    assert plan.quality.required is True


def test_adapter_accepts_real_evidence_with_contract_sha(tmp_path):
    # Real judge output carries judge_contract_sha256; the adapter must accept
    # it (and reject a mismatching SHA) instead of treating it as unsupported.
    output = tmp_path / "result.json"
    raw = _raw_evidence()
    raw["judge_contract_sha256"] = CONTRACT_SHA256
    ev_path = tmp_path / "real-evidence.json"
    ev_path.write_text(json.dumps(raw), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(ev_path.read_bytes()).hexdigest()
    assert _run_adapter(output, evidence_digest=digest, evidence_path=ev_path).returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["quality_completed"] == 1

    raw["judge_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        collect_evidence(
            raw, _contract(), expected_model="gemma4:12b",
            expected_contract_id="gemma-judge-counts-v1",
            expected_contract_sha256=CONTRACT_SHA256,
        )


def test_fixture_plan_runs_through_cli_without_gemma(tmp_path):
    out = tmp_path / "out"
    assert cli.main(["--plan", str(PLAN), "--out", str(out)]) == 0
    record = (out / "records.jsonl").read_text(encoding="utf-8")
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "qlh.experiment_quality.v2" in record
    assert "0.8" in report and "0.6" in report
    assert "completion" not in record + report
    assert "reasoning" not in record + report
