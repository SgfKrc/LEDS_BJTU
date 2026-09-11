import json

import pytest

from harness_workbench.tools import API_WORKBENCH_SCHEMA, APIWorkbenchError, builtin_api_cases, run_api_workbench
from harness_workbench.tools.api_workbench import (
    APIResponse,
    API_WORKBENCH_INPUT_SCHEMA,
    MemoryAPITransport,
    ProbeCase,
    UrllibAPITransport,
    _fixture_transports,
    main,
)


def test_api_workbench_default_is_offline_and_contract_complete():
    report = run_api_workbench()

    assert report.schema == API_WORKBENCH_SCHEMA
    assert report.valid is True
    assert report.runner_kind == "fixture"
    assert report.network_used is False
    assert report.weights_loaded is False
    assert len(report.cases) == 5
    assert report.drift_count == 1
    assert report.checks == {
        "cases_complete": True,
        "status_captured": True,
        "request_body_redacted": True,
        "endpoint_paths_relative": True,
        "latency_nonnegative": True,
        "drift_explicit": True,
        "offline_boundary": True,
    }


def test_api_workbench_records_status_error_code_and_latency_without_bodies():
    harness, main_transport = _fixture_transports()
    report = run_api_workbench(harness_transport=harness, main_transport=main_transport)
    invalid = next(item for item in report.cases if item.case_id == "chat-invalid")

    assert (invalid.harness_status, invalid.main_status) == (400, 422)
    assert invalid.mismatches == ("status_code", "error_code")
    assert invalid.status == "drifted"
    assert invalid.error_code == "invalid_messages->validation_error"
    assert invalid.harness_latency_ms == 0.4
    assert invalid.main_latency_ms == 0.5
    serialized = json.dumps(report.as_dict(), ensure_ascii=False)
    assert "fixture prompt" not in serialized
    assert "fixture answer" not in serialized
    assert "response body" not in serialized


def test_api_workbench_digest_and_markdown_are_deterministic():
    first = run_api_workbench(harness_transport=_fixture_transports()[0], main_transport=_fixture_transports()[1])
    second = run_api_workbench(harness_transport=_fixture_transports()[0], main_transport=_fixture_transports()[1])

    assert first.digest == second.digest
    assert first.as_dict() == second.as_dict()
    markdown = first.to_markdown()
    assert "TOOL-API-WB-01" in markdown
    assert "chat-invalid" in markdown
    assert "status_code; error_code" in markdown
    assert "fixture answer" not in markdown


def test_api_workbench_exposes_explicit_endpoint_mapping():
    cases = {item.case_id: item for item in builtin_api_cases()}

    assert (cases["health"].harness_path, cases["health"].main_path) == ("/healthz", "/api/health")
    assert (cases["models"].harness_path, cases["models"].main_path) == ("/v1/models", "/api/models")
    assert (cases["chat-stream"].harness_path, cases["chat-stream"].main_path) == ("/v1/chat/completions", "/api/chat/stream")
    assert cases["chat-completion"].request_digest != cases["chat-stream"].request_digest


def test_api_workbench_can_run_a_custom_case_with_injected_transports():
    case = ProbeCase("custom-health", "health", "GET", "/healthz", None, "GET", "/api/health", None)
    harness = MemoryAPITransport({"GET /healthz": APIResponse(200, {"status": "ok"}, latency_ms=0.1)})
    main_transport = MemoryAPITransport({"GET /api/health": APIResponse(200, {"status": "ok"}, latency_ms=0.2)})
    report = run_api_workbench([case], harness_transport=harness, main_transport=main_transport)

    assert report.valid is True
    assert report.cases[0].status == "matched"
    assert harness.calls[0]["path"] == "/healthz"
    assert main_transport.calls[0]["path"] == "/api/health"


def test_api_workbench_rejects_unsafe_cases_and_credentials():
    with pytest.raises(APIWorkbenchError, match="relative API paths"):
        ProbeCase("bad", "health", "GET", "https://example.invalid/health", None, "GET", "/api/health", None)
    with pytest.raises(APIWorkbenchError, match="credential field"):
        ProbeCase("bad", "chat", "POST", "/v1/chat/completions", {"api_key": "secret"}, "POST", "/api/chat", {"message": "x"})


def test_api_workbench_network_transport_is_fail_closed_by_default():
    transport = UrllibAPITransport("http://127.0.0.1:9")

    with pytest.raises(APIWorkbenchError, match="disabled"):
        transport.request("GET", "/healthz", payload=None, headers={})
    assert transport.network_used is False


def test_api_workbench_cli_writes_fixture_reports(tmp_path):
    json_path = tmp_path / "api.json"
    markdown_path = tmp_path / "api.md"

    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == API_WORKBENCH_SCHEMA
    assert payload["network_used"] is False
    assert "chat-stream" in markdown_path.read_text(encoding="utf-8")


def test_api_workbench_input_schema_is_stable(tmp_path):
    input_path = tmp_path / "cases.json"
    input_path.write_text(
        json.dumps(
            {
                "schema": API_WORKBENCH_INPUT_SCHEMA,
                "cases": [
                    {
                        "case_id": "health",
                        "operation": "health",
                        "harness_method": "GET",
                        "harness_path": "/healthz",
                        "main_method": "GET",
                        "main_path": "/api/health",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--input", str(input_path)]) == 0
