from pathlib import Path
import json

import pytest

from harness_workbench.tools import (
    V1_POLICY,
    V2_POLICY,
    builtin_judge_policy_fixture,
    load_judge_rubric,
    run_judge_policy_diff,
)


def test_builtin_judge_fixture_demonstrates_v2_rescues_without_model_execution():
    v1, v2, outputs = builtin_judge_policy_fixture()
    report = run_judge_policy_diff(v1, v2, outputs)
    metrics = report.metrics()

    assert report.v1_policy == V1_POLICY
    assert report.v2_policy == V2_POLICY
    assert metrics["v1_passed_count"] == 2
    assert metrics["v2_passed_count"] == 4
    assert metrics["rescued_count"] == 2
    assert metrics["regressed_count"] == 0
    assert metrics["prompt_set_match"] is True
    assert report.runner_kind == "fixture"
    assert report.network_used is False
    assert report.weights_loaded is False
    assert "13时54分" not in json.dumps(report.as_dict(), ensure_ascii=False)


def test_judge_policy_diff_is_digest_stable_and_markdown_is_chartable():
    v1, v2, outputs = builtin_judge_policy_fixture()
    first = run_judge_policy_diff(v1, v2, outputs)
    second = run_judge_policy_diff(v1, v2, outputs)

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "v1_192_normalized_contains" in markdown
    assert "v2_512_loose_contains" in markdown
    assert "**rescued**" in markdown


def test_judge_policy_reports_missing_truncated_and_skipped_cases():
    v1, v2, outputs = builtin_judge_policy_fixture()
    v2 = type(v2)(v2.rubric_id, v2.prompt_set_id, v2.prompt_set_sha256, v2.entries[:-1], v2.source_digest)
    outputs = {"math-time": outputs["math-time"], "math-number": outputs["math-number"]}
    report = run_judge_policy_diff(v1, v2, outputs, truncated_prompt_ids={"math-number"})
    by_id = {item.prompt_id: item for item in report.diffs}

    assert by_id["math-time"].outcome == "rescued"
    assert by_id["math-number"].outcome == "invalid"
    assert by_id["math-number"].v1.status == "truncated"
    assert "status" in report.skipped_prompt_ids
    assert by_id["math-chinese-number"].outcome == "invalid"
    assert report.metrics()["invalid_count"] == 2


def test_judge_policy_loads_locked_project_rubrics_and_compares_common_entries():
    root = Path(__file__).resolve().parents[1]
    v1 = load_judge_rubric(
        root / "fixtures/quality_rubrics/llm-objective-ps-v1-v1.json",
        expected_sha256="25f42a642e78d540f7c60265556f18bd7cd3f7dc342273a0a8a6019c4c38e2c1",
    )
    v2 = load_judge_rubric(root / "fixtures/quality_rubrics/llm-objective-ps-v1-v2.json")
    outputs = {
        "math-001": "推导后答案为 13时54分。",
        "math-002": "答案是 160。",
        "math-004": "答案是三。",
        "math-005": "答案是 1/2。",
    }
    report = run_judge_policy_diff(v1, v2, outputs)

    assert len(report.diffs) == 4
    assert report.metrics()["v1_passed_count"] == 2
    assert report.metrics()["v2_passed_count"] == 4
    assert report.metrics()["prompt_set_match"] is False


def test_judge_policy_rejects_policy_kind_drift_and_hash_mismatch(tmp_path):
    v1, v2, outputs = builtin_judge_policy_fixture()
    bad_v1 = type(v1)(
        v1.rubric_id,
        v1.prompt_set_id,
        v1.prompt_set_sha256,
        tuple(type(entry)(entry.prompt_id, entry.accepted, "loose_contains") for entry in v1.entries),
        v1.source_digest,
    )
    with pytest.raises(ValueError, match="v1 rubric check kind"):
        run_judge_policy_diff(bad_v1, v2, outputs)

    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"rubric_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_judge_rubric(path, expected_sha256="0" * 64)
