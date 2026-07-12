import os
import sys
import types
import asyncio
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


def test_lazy_model_manager_defers_construction(monkeypatch):
    calls: list[str] = []

    fake_module = types.ModuleType("model_module")

    class FakeModelManager:
        def __init__(self):
            calls.append("init")
            self.is_loaded = True

    fake_module.ModelManager = FakeModelManager
    monkeypatch.setitem(sys.modules, "model_module", fake_module)

    proxy = api_server._LazyModelManager()

    assert calls == []
    assert repr(proxy) == "<_LazyModelManager unloaded>"

    assert proxy.is_loaded is True
    assert calls == ["init"]


def test_available_models_does_not_touch_model_manager_when_unloaded(monkeypatch):
    class BombModelManager:
        def __getattr__(self, name):
            raise AssertionError(f"model_manager should stay lazy, accessed {name}")

    import model_downloader

    monkeypatch.setattr(model_downloader, "gguf_model_exists", lambda: False)
    monkeypatch.setattr(model_downloader, "safetensors_model_exists", lambda: False)
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", BombModelManager())

    result = asyncio.run(api_server.list_available_models())

    assert result["current"] is None
    assert result["current_engine"] is None


def test_frontend_bootstrap_endpoints_keep_model_manager_lazy(monkeypatch):
    fake_module = types.ModuleType("model_module")

    def fail_on_model_manager_access(name):
        raise AssertionError(f"model_module should not be imported for bootstrap endpoints: {name}")

    fake_module.__getattr__ = fail_on_model_manager_access
    monkeypatch.setitem(sys.modules, "model_module", fake_module)
    monkeypatch.setattr(api_server, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", api_server._LazyModelManager())

    class FakeScheduler:
        _max_nodes = 3

        def _effective_role(self):
            return "master"

        def get_effective_node_id(self):
            return "test-master"

    monkeypatch.setattr(api_server, "scheduler", FakeScheduler())
    monkeypatch.setattr(api_server, "_get_db_experimental_models", lambda: [])

    client = TestClient(api_server.app)

    for path in (
        "/api/status",
        "/api/models/current",
        "/api/models/available",
        "/api/models",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
