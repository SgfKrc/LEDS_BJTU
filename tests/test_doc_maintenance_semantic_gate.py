"""DOCAGENT-M2-GATE fixture integrity and agreement checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_semantic_gate import (  # noqa: E402
    SemanticBaselineError,
    evaluate_semantic_baseline,
    load_semantic_baseline,
    prepare_semantic_baseline_audit,
)


def _baseline(tmp_path: Path) -> dict:
    samples = []
    for index in range(4):
        samples.append({
            "id": f"sample-{index}", "doc": f"docs/{index}.md",
            "sha256": chr(97 + index) * 64, "rules": ["R1"],
            "expected_judgement": "accurate" if index == 3 else "needs_review",
            "human_rationale": "Reviewed by a maintainer.",
        })
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "schema_version": "qlh.docagent.semantic_baseline.v1",
        "baseline_id": "m2-test-v1", "minimum_matches": 3, "samples": samples,
    }), encoding="utf-8")
    return load_semantic_baseline(path)


def _audit(baseline: dict) -> dict:
    return {"docs": [{
        "doc": sample["doc"], "sha256": sample["sha256"],
        "status_line": "> status", "findings": [{"rule": "R1", "level": "warn"}],
    } for sample in baseline["samples"]]}


def _report(baseline: dict, observed: list[str], *, source="llm") -> dict:
    return {"judgements": [{
        "doc": sample["doc"], "judgement": value, "source": source,
    } for sample, value in zip(baseline["samples"], observed, strict=True)]}


def test_freezes_four_samples_and_selects_only_matching_m1_records(tmp_path):
    baseline = _baseline(tmp_path)
    selected = prepare_semantic_baseline_audit(baseline, _audit(baseline))

    assert len(selected["docs"]) == 4
    assert [record["doc"] for record in selected["docs"]] == [
        sample["doc"] for sample in baseline["samples"]
    ]


def test_changed_document_hash_or_m1_rule_invalidates_baseline(tmp_path):
    baseline = _baseline(tmp_path)
    audit = _audit(baseline)
    audit["docs"][0]["sha256"] = "f" * 64
    with pytest.raises(SemanticBaselineError, match="sample_hash_mismatch:sample-0"):
        prepare_semantic_baseline_audit(baseline, audit)

    audit = _audit(baseline)
    audit["docs"][0]["findings"] = [{"rule": "R5", "level": "info"}]
    with pytest.raises(SemanticBaselineError, match="sample_rules_mismatch:sample-0"):
        prepare_semantic_baseline_audit(baseline, audit)


def test_gate_requires_three_provider_backed_human_label_matches(tmp_path):
    baseline = _baseline(tmp_path)
    passing = evaluate_semantic_baseline(
        baseline, _report(baseline, ["needs_review", "needs_review", "needs_review", "stale"]),
    )
    assert passing["status"] == "passed"
    assert passing["matches"] == 3
    assert passing["required_matches"] == 3

    failed = evaluate_semantic_baseline(
        baseline, _report(baseline, ["accurate", "needs_review", "stale", "accurate"]),
    )
    assert failed["status"] == "failed"
    assert failed["matches"] == 2


def test_fallback_or_missing_provider_judgement_cannot_pass(tmp_path):
    baseline = _baseline(tmp_path)
    result = evaluate_semantic_baseline(
        baseline,
        _report(baseline, ["needs_review", "needs_review", "needs_review", "accurate"], source="none"),
    )

    assert result["status"] == "invalid"
    assert result["matches"] == 0
    assert all(reason.startswith("judgement_not_provider_backed") for reason in result["reasons"])


def test_baseline_rejects_wrong_sample_count(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": "qlh.docagent.semantic_baseline.v1",
        "baseline_id": "m2-test-v1", "minimum_matches": 3, "samples": [],
    }), encoding="utf-8")
    with pytest.raises(SemanticBaselineError, match="exactly four"):
        load_semantic_baseline(path)
