import json

import pytest

from harness_workbench.tools import BENCHMARK_LEDGER_SCHEMA, run_benchmark_ledger
from harness_workbench.tools.benchmark_ledger import (
    BenchmarkLedgerReport,
    BenchmarkRecord,
    build_parser,
    load_benchmark_payload,
    main,
)


def test_benchmark_ledger_default_preserves_fixture_and_not_run_boundaries():
    report = run_benchmark_ledger()

    assert report.schema == BENCHMARK_LEDGER_SCHEMA
    assert report.record_count == 4
    assert report.valid is True
    assert all(report.checks.values())
    assert {record.claim_class for record in report.records} == {"single_host", "not_run"}
    assert report.records[-1].eligible_for_claim is False
    assert any(record.record_id == "physical-dual-host" for record in report.records)
    assert any(group.dimension == "single_host" for group in report.groups)


def test_benchmark_ledger_default_digest_markdown_is_stable_and_explicit():
    first = run_benchmark_ledger()
    second = run_benchmark_ledger()

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "Citation table" in markdown
    assert "NOT RUN" in markdown
    assert "tokens/s" not in markdown
    assert "127.0.0.1" not in markdown


def test_benchmark_ledger_aggregates_multiple_models_from_records_file(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            {
                "schema": "qlh.benchmark_ledger.v1",
                "records": [
                    {
                        "record_id": "qwen-a",
                        "model_id": "QW1.8B",
                        "topology": "single_host_single_process",
                        "host_count": 1,
                        "sample_count": 3,
                        "status": "passed",
                        "metrics": {"median_ms": 2.0},
                        "claim_scope": "fixture only",
                    },
                    {
                        "record_id": "gemma-b",
                        "model_id": "Gemma-small",
                        "topology": "single_host_dual_process",
                        "host_count": 1,
                        "sample_count": 3,
                        "status": "passed",
                        "metrics": {"median_ms": 3.0},
                        "claim_scope": "fixture only",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = run_benchmark_ledger([path])
    assert report.record_count == 2
    assert report.model_ids == ("Gemma-small", "QW1.8B")
    assert any(group.dimension == "multi_model" for group in report.groups)
    assert report.valid is True


def test_benchmark_ledger_rejects_unsafe_and_nonfinite_inputs(tmp_path):
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        json.dumps(
            {
                "schema": "qlh.experiment_record.v1",
                "record_id": "unsafe",
                "status": "passed",
                "source": "127.0.0.1",
                "metrics": {"median_ms": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="address"):
        load_benchmark_payload(unsafe)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        '{"schema":"qlh.experiment_record.v1","record_id":"bad","status":"passed","metrics":{"median_ms":NaN}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        load_benchmark_payload(nonfinite)


def test_benchmark_ledger_rejects_duplicate_records_and_unknown_schema(tmp_path):
    first = tmp_path / "first.json"
    first.write_text(
        json.dumps({"schema": "qlh.experiment_record.v1", "record_id": "same", "status": "passed", "claim_scope": "fixture"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collide"):
        run_benchmark_ledger([first, first])

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"schema": "unknown.v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_benchmark_payload(unknown)


def test_benchmark_record_constructor_rejects_absolute_source_and_invalid_status():
    with pytest.raises(ValueError, match="repository-relative"):
        BenchmarkRecord("x", "C:\\secret.json", "a" * 64, "model", "single_host", 1, 1, 1, "passed", "single_host", {}, False, "fixture")
    with pytest.raises(ValueError, match="status"):
        BenchmarkRecord("x", "record.json", "a" * 64, "model", "single_host", 1, 1, 1, "unknown", "single_host", {}, False, "fixture")


def test_benchmark_ledger_root_mode_ignores_unsupported_files(tmp_path):
    supported = tmp_path / "supported.json"
    supported.write_text(
        json.dumps({"schema": "qlh.experiment_record.v1", "record_id": "ok", "status": "passed", "claim_scope": "fixture", "metrics": {"x": 1}}),
        encoding="utf-8",
    )
    unsupported = tmp_path / "ignored.json"
    unsupported.write_text(json.dumps({"schema": "unknown.v1"}), encoding="utf-8")
    report = run_benchmark_ledger(root=tmp_path)
    assert report.record_count == 1
    assert report.ignored_sources == ("ignored.json",)
    assert report.valid is True


def test_benchmark_ledger_keeps_relative_source_paths(tmp_path):
    path = tmp_path / "nested" / "record.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps({"schema": "qlh.experiment_record.v1", "record_id": "relative", "status": "passed", "claim_scope": "fixture", "metrics": {"x": 1}}),
        encoding="utf-8",
    )
    record = load_benchmark_payload(path)[0]
    assert record.source_ref == "record.json"
    assert not record.source_ref.startswith(("/", "\\"))


def test_benchmark_ledger_cli_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "ledger.json"
    markdown_path = tmp_path / "ledger.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == BENCHMARK_LEDGER_SCHEMA
    assert payload["record_count"] == 4
    assert "Citation table" in markdown_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""


def test_benchmark_ledger_parser_supports_repeated_input_and_root():
    args = build_parser().parse_args(["--input", "a.json", "--input", "b.json", "--strict"])
    assert args.inputs == ["a.json", "b.json"]
    assert args.strict is True
