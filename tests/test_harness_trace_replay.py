import json

import pytest

from harness_workbench.tools import TRACE_REPLAY_SCHEMA, run_trace_replay
from harness_workbench.tools.trace_replay import TraceReplayReport, TraceScenario, build_parser, load_trace_events, main


def test_trace_replay_default_has_documented_scenarios_and_passes_checks():
    report = run_trace_replay()

    assert report.schema == TRACE_REPLAY_SCHEMA
    assert report.scenario_count == 4
    assert report.event_count == 13
    assert report.valid is True
    assert all(report.checks.values())
    assert all(event.workflow_ref and event.workflow_ref.startswith("wf-") for event in report.events)


def test_trace_replay_contains_recovery_and_ipv6_transitions():
    report = run_trace_replay()
    kinds = {event.kind for event in report.events}

    assert {"worker_disconnected", "attempt_expired", "stage_reassigned"} <= kinds
    assert {"ipv6_connected", "remote_stage_completed"} <= kinds
    assert "layer_pipeline_started" in kinds
    fallback = run_trace_replay(scenario_id="worker-disconnect-reassignment-2026-08-21")
    assert fallback.events[-1].summary.endswith("reassignment count 1")


def test_trace_replay_filters_scenario_kind_and_date():
    report = run_trace_replay(kind="stage_reassigned")
    assert report.scenario_count == 1
    assert report.event_count == 1
    assert report.events[0].kind == "stage_reassigned"
    assert run_trace_replay(date="2026-08-21").scenario_count == 3
    with pytest.raises(ValueError, match="no trace scenarios"):
        run_trace_replay(scenario_id="missing")


def test_trace_replay_digest_markdown_and_report_are_stable_and_redacted():
    first = run_trace_replay()
    second = run_trace_replay()

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "## Timeline" in markdown
    assert "fd7a:" not in markdown
    assert "127.0.0.1" not in markdown
    assert "C:\\" not in markdown


def test_trace_replay_loads_normalized_input_and_hashes_workflow_id(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "schema": "qlh.trace_events.v1",
                "events": [
                    {
                        "timestamp": "2026-09-11T12:00:00+08:00",
                        "kind": "workflow_completed",
                        "actor": "coordinator",
                        "status": "completed",
                        "summary": "custom completion",
                        "workflow_id": "wf-secret-source-id",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    events = load_trace_events(path)
    report = run_trace_replay(input_path=path)
    assert events[0].workflow_ref.startswith("wf-")
    assert report.runner_kind == "input"
    assert report.valid is True


def test_trace_replay_rejects_raw_addresses_paths_and_credentials(tmp_path):
    path = tmp_path / "unsafe.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "timestamp": "2026-09-11T12:00:00+08:00",
                        "kind": "workflow_completed",
                        "actor": "coordinator",
                        "status": "completed",
                        "summary": "endpoint 127.0.0.1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw address"):
        load_trace_events(path)


def test_trace_replay_rejects_unknown_input_schema(tmp_path):
    path = tmp_path / "wrong-schema.json"
    path.write_text(
        json.dumps(
            {
                "schema": "other.schema.v1",
                "events": [
                    {
                        "timestamp": "2026-09-11T12:00:00+08:00",
                        "kind": "workflow_completed",
                        "summary": "done",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported trace input schema"):
        load_trace_events(path)


def test_trace_replay_marks_out_of_order_timestamps_invalid():
    report = run_trace_replay()
    scenario = report.scenarios[0]
    reversed_events = tuple(reversed(scenario.events))
    broken = TraceReplayReport((TraceScenario(
        scenario.scenario_id,
        scenario.title,
        scenario.date,
        reversed_events,
        scenario.expected_outcome,
        scenario.source_ref,
    ),))
    assert broken.valid is False
    assert broken.checks["events_ordered"] is False


def test_trace_replay_cli_lists_and_writes_reports(tmp_path, capsys):
    assert main(["--list"]) == 0
    assert "worker-disconnect-reassignment-2026-08-21" in capsys.readouterr().out
    json_path = tmp_path / "trace.json"
    markdown_path = tmp_path / "trace.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == TRACE_REPLAY_SCHEMA
    assert payload["event_count"] == 13
    assert "Timeline" in markdown_path.read_text(encoding="utf-8")


def test_trace_replay_parser_rejects_unknown_kind():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--kind", "unknown"])
