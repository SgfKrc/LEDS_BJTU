"""DOCAGENT-M3-QA baseline integrity and redacted metric tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_retrieval_gate import (  # noqa: E402
    RetrievalBaselineError,
    evaluate_retrieval_baseline,
    load_retrieval_baseline,
    prepare_retrieval_baseline,
)


def _baseline(tmp_path: Path) -> dict:
    samples = []
    for index in range(30):
        doc = f"docs/case-{index:02d}.md"
        samples.append({
            "id": f"case-{index:02d}",
            "query": f"private benchmark query {index}",
            "target_docs": [doc],
            "target_sha256": {doc: f"{index:02x}" * 32},
            "human_rationale": "Maintainer reviewed the target document.",
        })
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({
        "schema_version": "qlh.docagent.retrieval_baseline.v1",
        "baseline_id": "m3-test-v1",
        "embedding_model": "nomic-embed-text:latest",
        "minimum_hit_at_5": 0.6,
        "samples": samples,
    }), encoding="utf-8")
    return load_retrieval_baseline(path)


def _audit(baseline: dict) -> dict:
    return {"docs": [
        {"doc": doc, "sha256": digest, "findings": []}
        for sample in baseline["samples"]
        for doc, digest in sample["target_sha256"].items()
    ]}


def test_baseline_requires_thirty_sha_pinned_samples(tmp_path):
    baseline = _baseline(tmp_path)
    prepare_retrieval_baseline(baseline, _audit(baseline))
    assert len(baseline["samples"]) == 30


def test_changed_target_hash_invalidates_retrieval_baseline(tmp_path):
    baseline = _baseline(tmp_path)
    audit = _audit(baseline)
    audit["docs"][0]["sha256"] = "f" * 64
    with pytest.raises(RetrievalBaselineError, match="target_hash_mismatch:case-00"):
        prepare_retrieval_baseline(baseline, audit)


def test_quality_gate_measures_top_five_and_redacts_queries(tmp_path):
    baseline = _baseline(tmp_path)

    def search(query: str, limit: int):
        index = int(query.rsplit(" ", 1)[-1])
        if index < 18:
            return [{"doc_id": f"docs/case-{index:02d}.md"}]
        return [{"doc_id": "docs/other.md"}]

    result = evaluate_retrieval_baseline(baseline, search, mode="semantic")
    encoded = json.dumps(result)
    assert result["status"] == "passed"
    assert result["hit_at_5"] == 0.6
    assert result["mean_reciprocal_rank"] == 0.6
    assert "private benchmark query" not in encoded


def test_quality_gate_rejects_invalid_document_references(tmp_path):
    baseline = _baseline(tmp_path)
    with pytest.raises(RetrievalBaselineError, match="invalid document reference"):
        evaluate_retrieval_baseline(baseline, lambda query, limit: [{"doc_id": None}], mode="fts")
