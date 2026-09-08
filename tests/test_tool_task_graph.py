"""G4 TaskGraph routing, cancellation, and audit contract tests."""

from __future__ import annotations

import threading
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.task_graph import TaskGraphCoordinator, WorkflowCancelled
from src.task_journal import SQLiteTaskJournal
from src.tool_gateway import TOOL_REQUEST_SCHEMA, TOOL_RESULT_SCHEMA, ToolGatewayPolicy, error_result, ToolGatewayError
from src.tool_gateway_adapters import FakeToolProvider, ToolGatewayExecutor, ToolProviderError
from src.tool_task_graph import (
    HOST_ROUTER_PROVIDER_ID,
    ToolRoutePolicy,
    ToolRouter,
    ToolTaskGraphAdapter,
)


def _request(request_id: str = "req_tasktool_01") -> dict:
    return {
        "schema": TOOL_REQUEST_SCHEMA,
        "request_id": request_id,
        "tool_name": "web_search",
        "arguments": {"query": "QLH", "top_k": 1},
        "user_scope": "local_user",
        "network_scope": "explicit_opt_in",
        "deadline_ms": 5000,
    }


def _ok_result(request_id: str) -> dict:
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "ok",
        "items": [{"title": "QLH", "url": "https://example.com/qlh", "snippet": "summary"}],
        "citations": [{"url": "https://example.com/qlh", "sha256": "a" * 64}],
        "truncated": False,
        "policy": {"redirects": 0, "bytes": 64, "content_type": "text/plain"},
    }


def _executor(provider_id: str, handler) -> ToolGatewayExecutor:
    policy = ToolGatewayPolicy(data_scope="allow_all")
    return ToolGatewayExecutor(
        {provider_id: FakeToolProvider("web_search", provider_id, handler)},
        policy=policy,
    )


def test_unverified_sidecar_is_skipped_and_host_result_runs_in_task_graph(tmp_path):
    calls: list[str] = []

    def host_handler(request):
        calls.append("host")
        return _ok_result(request["request_id"])

    def sidecar_handler(request):
        calls.append("sidecar")
        return _ok_result(request["request_id"])

    host = _executor("host_provider", host_handler)
    sidecar = _executor("sidecar_provider", sidecar_handler)
    router = ToolRouter(
        host_executor=host,
        sidecar_executor=sidecar,
        policy=ToolGatewayPolicy(data_scope="allow_all"),
        route_policy=ToolRoutePolicy(sidecar_capability="unknown"),
    )
    journal = SQLiteTaskJournal(str(tmp_path / "tool-journal.db"), acquire_instance_lock=False)
    coordinator = TaskGraphCoordinator(journal=journal)
    adapter = ToolTaskGraphAdapter(coordinator, router)

    output, snapshot = adapter.run(
        _request(),
        allow_external=True,
        workflow_id="wf_toolroute01",
    )

    assert calls == ["host"]
    assert output["tool_result"]["status"] == "ok"
    assert output["tool_context"]["schema"] == "qlh.tool_context.v1"
    assert output["tool_context"]["role"] == "tool"
    assert output["tool_context"]["name"] == "web_search"
    assert output["route"]["route"] == HOST_ROUTER_PROVIDER_ID
    assert output["route"]["reason_code"] == "sidecar_capability_not_verified"
    assert snapshot["state"] == "result_ready"
    assert snapshot["stages"][0]["stage_type"] == "tool_request"

    audit = adapter.audit("wf_toolroute01", journal=journal)
    assert audit["mode"] == "read_only"
    assert audit["runtime_actions_enabled"] is False
    assert audit["summary"]["attempt_count"] == 1
    assert coordinator.commit_result("wf_toolroute01")["state"] == "completed"
    coordinator.close()
    journal.close()


def test_verified_sidecar_retryable_failure_falls_back_to_host():
    calls: list[str] = []

    def sidecar_handler(_request):
        calls.append("sidecar")
        raise ToolProviderError("timeout", "sidecar timeout", retryable=True)

    def host_handler(request):
        calls.append("host")
        return _ok_result(request["request_id"])

    router = ToolRouter(
        host_executor=_executor("host_provider", host_handler),
        sidecar_executor=_executor("sidecar_provider", sidecar_handler),
        policy=ToolGatewayPolicy(data_scope="allow_all"),
        route_policy=ToolRoutePolicy(sidecar_capability="verified"),
    )

    report = router.execute(_request("req_sidecar_01"), allow_external=True)

    assert calls == ["sidecar", "host"]
    assert report.result["status"] == "ok"
    assert report.route == HOST_ROUTER_PROVIDER_ID
    assert report.fallback_used is True
    assert report.reason_code == "sidecar_retryable:timeout"
    assert [item["status"] for item in report.attempts].count("failed") == 1


def test_sidecar_non_retryable_error_does_not_fallback():
    calls: list[str] = []

    def sidecar_handler(_request):
        calls.append("sidecar")
        raise ToolProviderError("bad_request", "invalid", retryable=False)

    def host_handler(request):
        calls.append("host")
        return _ok_result(request["request_id"])

    router = ToolRouter(
        host_executor=_executor("host_provider", host_handler),
        sidecar_executor=_executor("sidecar_provider", sidecar_handler),
        policy=ToolGatewayPolicy(data_scope="allow_all"),
        route_policy=ToolRoutePolicy(sidecar_capability="verified"),
    )

    report = router.execute(_request("req_sidecar_02"), allow_external=True)

    assert calls == ["sidecar"]
    assert report.result["status"] == "error"
    assert report.result["error"]["code"] == "bad_request"
    assert report.fallback_used is False


def test_scope_denial_happens_before_task_graph_registration():
    host = _executor("host_provider", lambda request: _ok_result(request["request_id"]))
    router = ToolRouter(
        host_executor=host,
        policy=ToolGatewayPolicy(data_scope="opt_in"),
    )
    coordinator = TaskGraphCoordinator()
    adapter = ToolTaskGraphAdapter(coordinator, router)

    with pytest.raises(Exception) as exc:
        adapter.run(_request("req_scope_01"), allow_external=False, workflow_id="wf_scope01")
    assert getattr(exc.value, "code", "") == "scope_denied"
    with pytest.raises(Exception):
        coordinator.get("wf_scope01")
    coordinator.close()


def test_cancel_fences_pre_registered_workflow_and_late_start():
    calls: list[str] = []
    release = threading.Event()
    started = threading.Event()

    def host_handler(request):
        calls.append("host")
        started.set()
        release.wait(3.0)
        return _ok_result(request["request_id"])

    router = ToolRouter(
        host_executor=_executor("host_provider", host_handler),
        policy=ToolGatewayPolicy(data_scope="allow_all"),
    )
    coordinator = TaskGraphCoordinator()
    adapter = ToolTaskGraphAdapter(coordinator, router)
    workflow_id = "wf_canceltool1"
    outcome: dict[str, object] = {}

    def run_workflow():
        try:
            adapter.run(_request("req_cancel_01"), allow_external=True, workflow_id=workflow_id)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run_workflow)
    thread.start()
    assert started.wait(2.0)
    cancel_snapshot = adapter.cancel(workflow_id)
    assert cancel_snapshot is not None
    release.set()
    thread.join(4.0)

    assert isinstance(outcome.get("error"), WorkflowCancelled)
    assert coordinator.get(workflow_id)["state"] == "cancelled"
    assert calls == ["host"]
    coordinator.close()

    pre_cancelled = TaskGraphCoordinator()
    pre_adapter = ToolTaskGraphAdapter(pre_cancelled, router)
    assert pre_adapter.cancel("wf_pending01") is None
    with pytest.raises(WorkflowCancelled):
        pre_adapter.run(_request("req_pending_01"), allow_external=True, workflow_id="wf_pending01")
    assert calls == ["host"]
    pre_cancelled.close()
