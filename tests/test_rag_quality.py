from __future__ import annotations

import json

import pytest

from src.rag_quality import (
    RagQualityError,
    evaluate_rag_quality,
    load_quality_cases,
)


def _cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [
        {
            "case_id": f"case-{index:02d}",
            "query": f"project term {index}",
            "relevant_source_ids": [f"source-{index}"],
        }
        for index in range(30)
    ]}), encoding="utf-8")
    return load_quality_cases(path)


def test_quality_gate_redacts_queries_and_measures_hit_and_citation(tmp_path):
    cases = _cases(tmp_path)

    def search(query, limit):
        index = int(query.rsplit(" ", 1)[-1])
        return [{
            "chunk_id": f"chunk-{index}",
            "source_id": f"source-{index}",
            "relative_ref": f"docs/{index}.md",
        }]

    result = evaluate_rag_quality(cases, search)
    assert result["status"] == "passed"
    assert result["case_count"] == 30
    assert result["hit_at_k"] == 1.0
    assert result["citation_rate"] == 1.0
    assert "project term" not in json.dumps(result)
    assert all("query_sha256" in item for item in result["details"])


def test_quality_gate_reports_failed_recall_without_leaking_query(tmp_path):
    cases = _cases(tmp_path)
    result = evaluate_rag_quality(cases, lambda query, limit: [], min_hit_rate=0.8)
    assert result["status"] == "failed"
    assert result["hit_at_k"] == 0.0
    assert result["citation_rate"] == 0.0


def test_quality_case_count_is_frozen_to_thirty(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "case_id": "only", "query": "one", "relevant_source_ids": ["source"],
    }]), encoding="utf-8")
    with pytest.raises(RagQualityError) as exc:
        load_quality_cases(path)
    assert exc.value.code == "case_count_mismatch"
