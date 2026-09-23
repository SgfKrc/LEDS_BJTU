"""Compatibility gates for the scheduler/API large-file split."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import api_server  # noqa: E402
import scheduler  # noqa: E402
from scheduler_sidecars import SchedulerSidecarMixin  # noqa: E402
from scheduler import (  # noqa: E402
    NodeInfo,
    NodeRole,
    NodeState,
    PipelineQueue,
    Scheduler,
    _bootstrap_api_port,
    _node_supports_forward_layers,
)


SCHEDULER_FACADE_SYMBOLS = {
    "Scheduler",
    "PipelineQueue",
    "NodeInfo",
    "NodeState",
    "NodeRole",
    "_node_supports_forward_layers",
    "_bootstrap_api_port",
}


def _scheduler_sources() -> list[Path]:
    sources = [ROOT / "src" / "scheduler.py"]
    sources.extend(sorted((ROOT / "src").glob("scheduler_*.py")))
    return sources


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    return set()


def _module_names_from_head() -> set[str]:
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/scheduler.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    tree = ast.parse(source)
    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in statement.names if alias.name != "*")
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.update(_target_names(target))
        elif isinstance(statement, ast.AnnAssign):
            names.update(_target_names(statement.target))
    return names


def _openapi_snapshot() -> dict:
    path = ROOT / "tests" / "fixtures" / "refactor_largefile_openapi_baseline.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_openapi_paths() -> dict[str, list[str]]:
    return {
        path: sorted(operation_map.keys())
        for path, operation_map in sorted(api_server.app.openapi()["paths"].items())
    }


def _wait_for(event: threading.Event, timeout: float = 3.0) -> None:
    assert event.wait(timeout), "异步转发测试未在限定时间内完成"


def test_scheduler_facade_exports_required_symbols() -> None:
    assert SCHEDULER_FACADE_SYMBOLS <= set(scheduler.__all__)
    assert SCHEDULER_FACADE_SYMBOLS <= set(dir(scheduler))


def test_scheduler_facade_preserves_head_module_names() -> None:
    # The split must not silently remove names that existed at the start of
    # this ticket. TYPE_CHECKING-only names are absent from runtime dir().
    runtime_names = set(dir(scheduler))
    missing = (_module_names_from_head() - runtime_names) - {
        "InferenceHost",
        "SchedulerCallbacks",
    }
    assert not missing, f"scheduler facade lost HEAD symbols: {sorted(missing)}"


def test_scheduler_source_scan_covers_current_and_split_modules() -> None:
    sources = _scheduler_sources()
    assert sources
    assert all(path.exists() and path.read_text(encoding="utf-8") for path in sources)


def test_scheduler_sidecars_are_mixin_methods_with_facade_factory() -> None:
    assert issubclass(Scheduler, SchedulerSidecarMixin)
    assert Scheduler.configure_gemma4_pipeline_sidecar is (
        SchedulerSidecarMixin.configure_gemma4_pipeline_sidecar
    )
    instance = Scheduler()
    assert instance._qwen3_multisidecar_factory() is scheduler.Qwen3PipelineMultiSidecar


def test_effective_role_reads_scheduler_runtime_global(monkeypatch) -> None:
    import config

    instance = Scheduler()
    instance._role_override = None
    instance._auto_role_controller = None
    monkeypatch.setattr(config, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(scheduler, "NODE_ROLE", "client")

    assert instance._effective_role() == "client"


def test_scheduler_lock_identity_is_instance_owned() -> None:
    instance = Scheduler()
    locks = [
        getattr(instance, name)
        for name in ("_inference_lock", "_layer_config_lock", "_layer_execution_lock")
    ]

    assert all(lock is getattr(instance, name) for lock, name in zip(
        locks,
        ("_inference_lock", "_layer_config_lock", "_layer_execution_lock"),
    ))
    assert len({id(lock) for lock in locks}) == 3


def test_handle_infer_forward_runs_pipeline_and_returns_result(monkeypatch) -> None:
    import local_store

    instance = Scheduler()
    sent: list[dict] = []
    completed: list[tuple[str, str, dict]] = []
    started = threading.Event()
    monkeypatch.setattr(local_store, "save_local_conversation_turn", lambda **_: None)
    monkeypatch.setattr(instance, "start_infer_task", lambda prompt, request_id=None: "task-forward")
    monkeypatch.setattr(
        instance,
        "complete_infer_task",
        lambda task_id, content, metrics: completed.append((task_id, content, metrics)),
    )
    monkeypatch.setattr(instance, "record_task_complete", lambda **_: None)
    monkeypatch.setattr(
        instance,
        "run_pipeline_safe",
        lambda **kwargs: started.set() or {
            "response": "forwarded",
            "thinking": "",
            "metrics": {"distributed_used": True, "engine": "test"},
        },
    )
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda client_id, task_id, content, metrics=None, **kwargs: sent.append({
            "client_id": client_id,
            "task_id": task_id,
            "content": content,
            "metrics": metrics,
            **kwargs,
        }),
    )

    instance.handle_infer_forward("worker-1", {
        "data": {
            "prompt": "hello",
            "forward_request_id": "forward-1",
            "request_id": "request-1",
            "routing_preference": "distributed_required",
        },
    })

    _wait_for(started)
    deadline = time.monotonic() + 3.0
    while (not sent or instance._forward_cancel_events) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sent == [{
        "client_id": "worker-1",
        "task_id": "task-forward",
        "content": "forwarded",
        "metrics": {"distributed_used": True, "engine": "test"},
        "thinking_content": "",
        "forward_request_id": "forward-1",
    }]
    assert completed == [
        ("task-forward", "forwarded", {"distributed_used": True, "engine": "test"}),
    ]
    assert not instance._forward_cancel_events


def test_handle_infer_forward_rejects_full_concurrency_slot(monkeypatch) -> None:
    instance = Scheduler()
    sent: list[dict] = []

    class FullSlots:
        def acquire(self, blocking=False):
            return False

        def release(self):
            raise AssertionError("a rejected request must not release a slot")

    instance._forward_infer_slots = FullSlots()
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda *args, **kwargs: sent.append({"args": args, "kwargs": kwargs}),
    )
    instance.handle_infer_forward("worker-1", {
        "data": {"forward_request_id": "full-1", "prompt": "hello"},
    })

    assert len(sent) == 1
    assert sent[0]["kwargs"]["status"] == "error"
    assert "并发上限" in sent[0]["kwargs"]["error"]


def test_handle_infer_forward_cancellation_reaches_pipeline(monkeypatch) -> None:
    instance = Scheduler()
    pipeline_started = threading.Event()
    sent: list[dict] = []
    observed_cancel: list[threading.Event] = []
    monkeypatch.setattr(instance, "start_infer_task", lambda prompt, request_id=None: "task-cancel")
    monkeypatch.setattr(instance, "fail_infer_task", lambda task_id, error: None)
    monkeypatch.setattr(
        instance,
        "run_pipeline_safe",
        lambda **kwargs: (
            observed_cancel.append(kwargs["_cancel_event"]),
            pipeline_started.set(),
            kwargs["_cancel_event"].wait(2.0),
            {"response": "", "metrics": {}, "error": "cancelled"},
        )[-1],
    )
    monkeypatch.setattr(
        instance,
        "_send_infer_result",
        lambda *args, **kwargs: sent.append({"args": args, "kwargs": kwargs}),
    )

    instance.handle_infer_forward("worker-1", {
        "data": {"forward_request_id": "cancel-1", "prompt": "hello"},
    })
    _wait_for(pipeline_started)
    instance._on_tcp_message("worker-1", {
        "type": "infer_cancel",
        "data": {"forward_request_id": "cancel-1"},
    })

    deadline = time.monotonic() + 3.0
    while (not sent or instance._forward_cancel_events) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert observed_cancel and observed_cancel[0].is_set()
    assert sent[0]["kwargs"]["status"] == "error"
    assert sent[0]["kwargs"]["error"] == "cancelled"
    assert not instance._forward_cancel_events


def test_api_openapi_path_method_snapshot_is_stable() -> None:
    snapshot = _openapi_snapshot()
    actual = _canonical_openapi_paths()
    encoded = json.dumps(actual, separators=(",", ":"), ensure_ascii=True)

    assert len(actual) == snapshot["path_count"]
    assert sum(len(methods) for methods in actual.values()) == snapshot["operation_count"]
    assert hashlib.sha256(encoded.encode()).hexdigest() == snapshot["canonical_path_method_sha256"]


def test_api_log_route_order_keeps_literal_routes_before_path_parameter() -> None:
    routes = list(api_server.app.routes)
    recent = [
        index for index, route in enumerate(routes)
        if route.path == "/api/logs/recent" and "GET" in (route.methods or set())
    ]
    wildcard = [
        index for index, route in enumerate(routes)
        if route.path == "/api/logs/{filename:path}"
    ]

    assert recent and wildcard
    assert max(recent) < min(wildcard)


@pytest.mark.parametrize(
    "name",
    (
        "app",
        "model_manager",
        "scheduler",
        "ChatGenerationCancelled",
        "_request_id_ctx",
        "_run_log_retention_cleanup",
        "_client_supports_forward_layers",
        "run_api_servers",
    ),
)
def test_api_facade_preserves_imported_symbols(name: str) -> None:
    assert hasattr(api_server, name), name
