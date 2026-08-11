import os
import sys
import types
import asyncio
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
import model_host as model_host_module
from model_host import model_host


def test_lazy_model_manager_defers_construction(monkeypatch):
    calls: list[str] = []

    fake_module = types.ModuleType("model_module")

    class FakeModelManager:
        def __init__(self):
            calls.append("init")
            self.is_loaded = True

    fake_module.ModelManager = FakeModelManager
    monkeypatch.setitem(sys.modules, "model_module", fake_module)

    proxy = model_host_module._LazyModelManager()

    assert calls == []
    assert repr(proxy) == "<_LazyModelManager unloaded>"

    assert proxy.is_loaded is True
    assert calls == ["init"]


def test_available_models_does_not_touch_model_manager_when_unloaded(monkeypatch):
    class BombModelManager:
        def __getattr__(self, name):
            raise AssertionError(f"model_manager should stay lazy, accessed {name}")

    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", BombModelManager())
    # The endpoint serializes model_config availability itself.  Patch the
    # direct config source so this lazy-manager test cannot pass by relying on
    # whatever model files happen to exist on the developer machine.
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [])

    result = asyncio.run(api_server.list_available_models())

    assert result["current"] is None
    assert result["current_engine"] is None


def test_frontend_bootstrap_endpoints_keep_model_manager_lazy(monkeypatch):
    fake_module = types.ModuleType("model_module")

    def fail_on_model_manager_access(name):
        raise AssertionError(f"model_module should not be imported for bootstrap endpoints: {name}")

    fake_module.__getattr__ = fail_on_model_manager_access
    monkeypatch.setitem(sys.modules, "model_module", fake_module)
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", model_host_module._LazyModelManager())

    class FakeScheduler:
        _max_nodes = 3

        def _effective_role(self):
            return "master"

        def get_effective_node_id(self):
            return "test-master"

    monkeypatch.setattr(api_server, "scheduler", FakeScheduler())
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])

    client = TestClient(api_server.app)

    for path in (
        "/api/status",
        "/api/models/current",
        "/api/models/available",
        "/api/models",
    ):
        response = client.get(path)
        assert response.status_code == 200, path


def test_reserved_pipeline_worker_does_not_auto_load_full_model(monkeypatch):
    class UnloadedManager:
        is_loaded = False

    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", UnloadedManager())
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: True,
    )
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: True,
    )
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    auto_load_calls = []
    monkeypatch.setattr(
        api_server, "_auto_load_default_model", lambda: auto_load_calls.append(True),
    )

    api_server._ensure_chat_model_or_forwarding()

    assert auto_load_calls == []


def test_reserved_worker_rejects_local_model_even_when_already_loaded(monkeypatch):
    class SegmentManager:
        is_loaded = True

    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "model_manager", SegmentManager())
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: False,
    )
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: True,
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._ensure_chat_model_or_forwarding()

    assert exc_info.value.status_code == 503
