import json

import pytest

from harness_workbench.tools import (
    RAG_BASELINE_SCHEMA,
    RagBaselineError,
    builtin_rag_baseline_cases,
    builtin_rag_baseline_documents,
    run_rag_baseline,
)
from harness_workbench.tools.rag_baseline import (
    RAG_BASELINE_INPUT_SCHEMA,
    load_rag_baseline_input,
    main,
)


def test_dual_side_baseline_is_frozen_and_model_free():
    first = run_rag_baseline()
    second = run_rag_baseline()

    assert first.schema == RAG_BASELINE_SCHEMA
    assert first.valid is True
    assert first.as_dict() == second.as_dict()
    assert first.document_count == 6
    assert first.case_count == 30
    assert first.sides["main_project"]["hit_at_k"] == 1.0
    assert first.sides["harness"]["hit_at_k"] == 1.0
    assert first.sides["main_project"]["mean_reciprocal_rank"] == 1.0
    assert first.sides["harness"]["mean_reciprocal_rank"] == 1.0
    assert first.model_used is False
    assert first.network_used is False


def test_baseline_report_redacts_queries_and_documents():
    report = run_rag_baseline()
    encoded = json.dumps(report.as_dict(), ensure_ascii=False)
    assert "ragbaseanchoralpha" not in encoded
    assert "Offline retrieval fixture" not in encoded
    assert "query_sha256" in encoded
    assert all("query" not in entry for entry in report.sides["main_project"]["details"])


def test_baseline_markdown_contains_both_sides_and_digest_only():
    report = run_rag_baseline()
    markdown = report.to_markdown()
    assert "hit@k" in markdown
    assert "MRR" in markdown
    assert "main_project" in markdown
    assert "harness" in markdown
    assert "ragbaseanchoralpha" not in markdown


def test_baseline_input_loader_rejects_wrong_count_and_absolute_refs(tmp_path):
    payload = {
        "schema": RAG_BASELINE_INPUT_SCHEMA,
        "documents": [
            {"document_id": "doc", "source_ref": "C:/secret.md", "title": "x", "text": "marker"}
        ],
        "cases": [],
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RagBaselineError, match="repository-relative"):
        load_rag_baseline_input(path)


def test_baseline_input_loader_accepts_frozen_fixture_shape(tmp_path):
    documents = [doc.as_dict() for doc in builtin_rag_baseline_documents()]
    cases = [case.as_dict() for case in builtin_rag_baseline_cases()]
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"schema": RAG_BASELINE_INPUT_SCHEMA, "documents": documents, "cases": cases}), encoding="utf-8")
    loaded_documents, loaded_cases = load_rag_baseline_input(path)
    assert loaded_documents == builtin_rag_baseline_documents()
    assert loaded_cases == builtin_rag_baseline_cases()


def test_baseline_top_k_and_case_boundaries_fail_closed():
    with pytest.raises(RagBaselineError, match="between 1 and 100"):
        run_rag_baseline(top_k=0)
    with pytest.raises(RagBaselineError, match="exactly 30"):
        run_rag_baseline(cases=builtin_rag_baseline_cases()[:-1])


def test_baseline_cli_writes_redacted_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "rag-baseline.json"
    markdown_path = tmp_path / "rag-baseline.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == RAG_BASELINE_SCHEMA
    assert payload["valid"] is True
    assert "ragbaseanchoralpha" not in json_path.read_text(encoding="utf-8")
    assert "MRR" in markdown_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""


def test_baseline_package_exports_are_lazy():
    report = run_rag_baseline(top_k=1)
    assert report.valid is True
    assert report.top_k == 1
