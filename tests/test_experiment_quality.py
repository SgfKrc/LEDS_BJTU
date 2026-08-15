"""EX-N3-S0 quality-evidence contract tests (no model execution)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_core.collector import build_record
from experiment_core.plan import load_plan
from experiment_core.quality import (
    QualityEvidenceError,
    normalize_quality_evidence,
    sd_evidence_from_gate_report,
)
from experiment_core.report import build_report
from experiment_core.runner import UnitOutcome


PROMPT_SET = {
    "id": "ps-v1-zh-en-code",
    "sha256": "8cc555f57fa23d45c16820f77cc90b507da04578a1e90eebf46e19d4eb2568a3",
}


def _llm_evidence() -> dict:
    return {
        "llm": {
            "prompt_set_id": PROMPT_SET["id"],
            "prompt_set_sha256": PROMPT_SET["sha256"],
            "correctness": {"evaluated_count": 20, "passed_count": 18},
            "format": {"evaluated_count": 20, "passed_count": 19},
        }
    }


def _plan(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "plan_id": "plan-quality-v1",
        "prompt_set": PROMPT_SET,
        "units": [{
            "experiment_id": "exp-0001",
            "name": "quality unit",
            "command": [sys.executable, "-c", "pass"],
            "resources": {}, "params": {}, "model": {}, "runs": 1,
            "timeout_s": 10, "gate": {"metric": "decode_tok_s", "threshold": 1},
        }],
    }), encoding="utf-8")
    return load_plan(path)


def test_normalize_llm_quality_derives_rates_and_binds_prompt_set():
    quality = normalize_quality_evidence(_llm_evidence(), expected_prompt_set=PROMPT_SET)
    assert quality["status"] == "collected"
    assert quality["correct_rate"] == 0.9
    assert quality["format_rate"] == 0.95
    assert quality["llm"]["correctness"]["passed_count"] == 18


def test_quality_rejects_unknown_or_mismatched_llm_evidence():
    payload = _llm_evidence()
    payload["llm"]["output_text"] = "do not persist model output"
    with pytest.raises(QualityEvidenceError, match="unsupported"):
        normalize_quality_evidence(payload, expected_prompt_set=PROMPT_SET)

    payload = _llm_evidence()
    payload["llm"]["prompt_set_id"] = "another-fixed-set"
    with pytest.raises(QualityEvidenceError, match="does not match"):
        normalize_quality_evidence(payload, expected_prompt_set=PROMPT_SET)


def test_sd_gate_adapter_counts_lists_without_copying_sensitive_report_fields():
    raw_report = {
        "schema_version": 1,
        "asset_id": "sd15-original-v1",
        "artifact_id": "sha256:" + "a" * 64,
        "prompt": "this prompt must never enter experiment records",
        "images": ["C:/private/image-a.png", "C:/private/image-b.png"],
        "automatic_gate": {"passed": True, "unique_images": ["hash-a", "hash-b"]},
        "manual_gate": {"passed": False, "required_reviewers": 2, "reviews": []},
        "status": "pending_manual_review",
    }
    payload = {"sd": sd_evidence_from_gate_report(raw_report)}
    quality = normalize_quality_evidence(payload, expected_prompt_set=PROMPT_SET)
    assert quality["sd"]["automatic_gate"] == {
        "status": "passed", "output_count": 2, "unique_output_count": 2,
    }
    assert quality["sd"]["manual_review"]["status"] == "pending"
    serialized = json.dumps(quality)
    assert "private" not in serialized
    assert "prompt" not in serialized
    assert "hash-a" not in serialized


def test_collector_redacts_malformed_evidence_and_keeps_performance_gate(tmp_path):
    plan = _plan(tmp_path)
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.1,
        metrics={
            "decode_tok_s": 2.0,
            "quality_evidence": {"llm": {"output_text": "private completion"}},
        },
    )
    record = build_record(
        plan, plan.units[0], outcome,
        prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    assert record["gate"]["status"] == "passed"
    assert record["quality"]["status"] == "invalid"
    assert "quality_evidence" not in record["metrics"]
    assert "private completion" not in json.dumps(record)


def test_report_consumes_structured_quality_without_changing_gate(tmp_path):
    plan = _plan(tmp_path)
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.1,
        metrics={"decode_tok_s": 2.0, "quality_evidence": _llm_evidence()},
    )
    record = build_record(
        plan, plan.units[0], outcome,
        prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    report_path, summary_path = build_report(
        tmp_path / "out", [record],
        {"plan_id": plan.plan_id, "title": "quality", "prompt_set": PROMPT_SET},
    )
    text = report_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "质量证据（EX-N3-S0）" in text
    assert "0.9" in text and "0.95" in text
    assert record["gate"]["status"] == "passed"
    assert summary["quality_evidence"] == {
        "collected": 1, "not_collected": 0, "invalid": 0,
    }


def test_record_schema_accepts_v1_evidence_and_pre_evidence_records(tmp_path):
    plan = _plan(tmp_path)
    outcome = UnitOutcome(
        experiment_id="exp-0001", status="passed", exit_code=0, duration_s=0.1,
        metrics={"decode_tok_s": 2.0, "quality_evidence": _llm_evidence()},
    )
    record = build_record(
        plan, plan.units[0], outcome,
        prompt_set_dir=plan.verify_prompt_set(), records={},
    )
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "experiment-record.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(record)) == []

    legacy = dict(record)
    legacy["quality"] = {"correct_rate": None, "format_rate": None}
    assert list(validator.iter_errors(legacy)) == []


# ---- P5 loose_contains 判题口径（2026-08-16） ----

def _loose_check(accepted, output):
    from experiment_core.objective_rubric import _evaluate_check
    return _evaluate_check({"kind": "loose_contains", "accepted": accepted}, output)


def test_loose_contains_matches_time_expression_with_chinese_units():
    # E4 人工复核场景：推理风格输出 "13时54分" 在 v1 normalized_contains 下
    # 判错（不包含 "13:54"）；宽松归一化应命中。
    assert _loose_check(["13:54", "13：54"], "火车 8:00 出发，约 13时54分 到达")
    assert _loose_check(["13:54"], "13点54分")


def test_loose_contains_matches_answer_marker_tail():
    assert _loose_check(["160"], "先计算速度……因此答案是 160。")
    assert _loose_check(["3"], "经过推导，答案：3")


def test_loose_contains_matches_isolated_chinese_digits():
    assert _loose_check(["3"], "所以结果是三")
    assert _loose_check(["2"], "答案为两")


def test_loose_contains_keeps_false_positive_floor():
    # 无关文本与错误答案不应命中
    assert not _loose_check(["13:54"], "火车晚点，未能在 8:00 出发")
    assert not _loose_check(["160"], "答案是 16")
    assert not _loose_check(["3"], "三个苹果放在桌上，答案是 4")


def test_normalized_contains_unchanged_by_p5():
    # v1 语义不回退："13时54分" 仍不包含 "13:54"
    from experiment_core.objective_rubric import _evaluate_check
    assert not _evaluate_check(
        {"kind": "normalized_contains", "accepted": ["13:54"]},
        "13时54分",
    )
    assert _evaluate_check(
        {"kind": "normalized_contains", "accepted": ["13:54"]},
        "答案 13:54",
    )
