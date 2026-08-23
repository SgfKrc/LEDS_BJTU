"""EX-N3-PROD tests: completed evidence becomes a fail-closed receipt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core.collector import evaluate_quality_gate
from experiment_core.plan import load_plan
from experiment_core.production_quality import audit_production_quality
from experiment_core.quality import normalize_quality_evidence
from experiment_quality_production_gate import main as production_gate_main


PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "experiment-plans"
    / "plan-quality-total-gate-v1.json"
)


def _quality_for(plan, check: str) -> dict:
    if check == "llm":
        raw = {
            "llm": {
                "prompt_set_id": plan.prompt_set["id"],
                "prompt_set_sha256": plan.prompt_set["sha256"],
                "correctness": {"evaluated_count": 4, "passed_count": 1},
                "format": {"evaluated_count": 11, "passed_count": 0},
            },
        }
    elif check == "sd":
        raw = {
            "sd": {
                "mode": "text_to_image",
                "asset_id": "sd15_90s_retrovers_v1",
                "artifact_id": "sd15_90s_retrovers_v1",
                "source_schema_version": 1,
                "automatic_gate": {
                    "passed": True,
                    "output_count": 10,
                    "unique_output_count": 10,
                },
                "manual_review": {"status": "passed", "required_reviewers": 2},
            },
        }
    else:
        raw = {
            "gemma_judge": {
                "model": "gemma4:12b",
                "judge_contract_id": "gemma-judge-counts-v1",
                "judge_contract_sha256": (
                    "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374"
                ),
                "topic_hit": {"evaluated_count": 10, "passed_count": 9},
                "key_element_coverage": {"evaluated_count": 20, "passed_count": 12},
                "manual_review": {"status": "passed", "required_reviewers": 2},
            },
        }
    return normalize_quality_evidence(
        raw,
        expected_prompt_set=plan.prompt_set,
        expected_gemma_judge=plan.quality.gemma_judge,
    )


def _valid_records() -> list[dict]:
    plan = load_plan(PLAN_PATH)
    plan.verify_prompt_set()
    records = []
    for unit in plan.units:
        check = unit.quality_checks[0]
        quality = _quality_for(plan, check)
        quality_gate = evaluate_quality_gate(
            quality, plan.quality, unit.quality_checks, baseline_record=None,
        )
        records.append({
            "experiment_id": unit.experiment_id,
            "plan_id": plan.plan_id,
            "prompt_set": dict(plan.prompt_set),
            "model": dict(unit.model),
            "params": dict(unit.params),
            "runs": unit.runs,
            "calibration": dict(plan.quality.calibration),
            "metrics": {"quality_completed": 1},
            "quality": quality,
            "quality_gate": quality_gate,
            "performance_gate": {"status": "passed"},
            "gate": {"status": "passed"},
        })
    return records


def _write_records(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_production_gate_promotes_complete_three_side_evidence(tmp_path):
    decision = audit_production_quality(PLAN_PATH, _write_records(tmp_path, _valid_records()))

    assert decision["status"] == "passed"
    assert [entry["status"] for entry in decision["coverage"]] == ["passed"] * 3
    assert decision["policy"]["quality_required"] is True
    assert decision["records"]["count"] == 3
    assert decision["reasons"] == []


def test_production_gate_rejects_missing_required_unit(tmp_path):
    decision = audit_production_quality(
        PLAN_PATH,
        _write_records(tmp_path, _valid_records()[:-1]),
    )

    assert decision["status"] == "failed"
    assert decision["reasons"] == [{"code": "record_missing", "experiment_id": "exp-0003"}]
    assert decision["coverage"][-1]["status"] == "failed"


def test_production_gate_rejects_noncanonical_counter_rate(tmp_path):
    records = _valid_records()
    records[0]["quality"]["llm"]["correctness"]["rate"] = 0.99

    decision = audit_production_quality(PLAN_PATH, _write_records(tmp_path, records))

    assert decision["status"] == "invalid"
    assert {reason["code"] for reason in decision["reasons"]} == {
        "quality_evidence_not_canonical",
    }


def test_production_gate_recalculates_the_plan_performance_threshold(tmp_path):
    records = _valid_records()
    records[0]["metrics"]["quality_completed"] = 0

    decision = audit_production_quality(PLAN_PATH, _write_records(tmp_path, records))

    assert decision["status"] == "failed"
    assert decision["reasons"] == [
        {"code": "performance_gate_not_passed", "experiment_id": "exp-0001"},
    ]


def test_production_gate_treats_missing_manual_review_as_quality_failure(tmp_path):
    records = _valid_records()
    plan = load_plan(PLAN_PATH)
    records[2]["quality"] = normalize_quality_evidence(
        {
            "gemma_judge": {
                "model": "gemma4:12b",
                "judge_contract_id": "gemma-judge-counts-v1",
                "judge_contract_sha256": (
                    "be7bcea3e736e0009c2ff3e110f54309263a960e2b6ae892e3c5de7a302e4374"
                ),
                "topic_hit": {"evaluated_count": 10, "passed_count": 9},
                "key_element_coverage": {"evaluated_count": 20, "passed_count": 12},
            },
        },
        expected_prompt_set=plan.prompt_set,
        expected_gemma_judge=plan.quality.gemma_judge,
    )
    records[2]["quality_gate"] = {
        "status": "passed", "required": True, "checks": ["gemma_judge"],
    }

    decision = audit_production_quality(PLAN_PATH, _write_records(tmp_path, records))

    assert decision["status"] == "failed"
    assert decision["reasons"] == [
        {"code": "quality_gate_not_passed", "experiment_id": "exp-0003"},
    ]


def test_production_gate_rejects_plan_that_does_not_require_quality(tmp_path):
    plan_raw = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan_raw["quality"]["required"] = False
    plan_path = tmp_path / "plan-not-required.json"
    plan_path.write_text(json.dumps(plan_raw), encoding="utf-8")

    decision = audit_production_quality(plan_path, _write_records(tmp_path, _valid_records()))

    assert decision["status"] == "invalid"
    assert decision["reasons"] == [{"code": "plan_quality_not_required"}]


def test_production_gate_does_not_reflect_untrusted_evidence(tmp_path):
    secret = "private prompt and reviewer identity must not appear"
    records = _valid_records()
    records[0]["quality"]["llm"]["private_output"] = secret

    decision = audit_production_quality(PLAN_PATH, _write_records(tmp_path, records))

    assert decision["status"] == "invalid"
    assert decision["reasons"] == [
        {"code": "quality_evidence_invalid", "experiment_id": "exp-0001"},
    ]
    assert secret not in json.dumps(decision)


def test_production_gate_cli_writes_only_redacted_receipt(tmp_path, capsys):
    records = _write_records(tmp_path, _valid_records())
    receipt = tmp_path / "production-quality.json"

    assert production_gate_main([
        "--plan", str(PLAN_PATH), "--records", str(records), "--output", str(receipt),
    ]) == 0

    printed = capsys.readouterr().out
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert "raw_log" not in printed
    assert "private" not in printed
