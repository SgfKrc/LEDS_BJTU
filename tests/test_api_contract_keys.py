"""阶段 0 复核补充测试：API 响应 key 契约（防误替换回归）

阶段 0.2 迁移 model_loaded/current_quant 等运行时状态到 model_host 时，
响应字典的 key 曾被误替换为字面量 "model_host.current_quant"（破坏前端
MetricsPanel/ChatPanel 读取）。本测试锁定关键端点响应 key 不被迁移破坏。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

import api_server as api_server_mod
from api_server import app
from model_host import model_host


@pytest.fixture
def client():
    return TestClient(app)


class TestApiResponseKeys:
    """响应 JSON 的字段名必须保持 api_server 既有契约（前端/TUI 依赖）。"""

    def test_presets_returns_current_quant_key(self, client):
        res = client.get("/api/presets")
        assert res.status_code == 200
        body = res.json()
        assert "current_quant" in body, "key 被误替换为 model_host.current_quant"
        assert "presets" in body
        assert "max_new_tokens" in body

    def test_status_returns_current_quant_key(self, client, monkeypatch):
        monkeypatch.setattr(model_host, "model_loaded", True)
        res = client.get("/api/status")
        assert res.status_code == 200
        body = res.json()
        assert "current_quant" in body, "key 被误替换为 model_host.current_quant"
        assert "model_loaded" in body
        # 迁移后值来自 host（与 API 层一致）
        assert body["model_loaded"] is model_host.model_loaded

    def test_status_quant_value_follows_host(self, client, monkeypatch):
        monkeypatch.setattr(model_host, "model_loaded", True)
        monkeypatch.setattr(model_host, "current_quant", "int8")
        res = client.get("/api/status")
        assert res.json()["current_quant"] == "int8"

    def test_health_shape(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"status", "timestamp"}

    def test_auth_capability_is_explicit_when_running_direct_api(self, client):
        """The standalone API must not make Account fail with a 404 probe."""
        res = client.get("/api/auth/capability")
        assert res.status_code == 200
        body = res.json()
        assert body["required"] is False
        assert body["enforced"] is False
        assert body["available"] is False
        assert body["service"] == "api_server"
        assert body["reason_code"] == "auth_control_plane_unavailable"

    def test_storage_health_is_local_first_and_retired_remote(self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod._local_store,
            "local_store_health",
            lambda: {"status": "ok", "backend": "sqlite", "writable": True},
        )
        res = client.get("/api/storage/health")
        assert res.status_code == 200
        body = res.json()
        assert body["local"]["backend"] == "sqlite"
        assert body["effective_mode"] == "local_only"
        assert body["remote"] == {"status": "retired", "backend": "postgresql", "mode": "retired"}
        assert body["projection"]["pending_events"] == 0

    def test_speculative_capability_is_zero_network_and_fail_closed(self, client, monkeypatch):
        res = client.get("/api/experimental/speculative/capability")
        assert res.status_code == 200
        body = res.json()
        assert body["execution_mode"] == "speculative_assisted"
        assert body["available"] is False
        assert body["reason_code"] in {"disabled_by_config", "verify_endpoint_missing"}

    @staticmethod
    def _cluster_status_payload(network_path=None):
        payload = {
            "run_mode": "single",
            "nodes_ready": True,
            "nodes": {
                "master": {
                    "node_id": "master",
                    "role": "master",
                    "state": "online",
                },
            },
            "current_task": None,
            "tcp_server": None,
            "pipeline": None,
            "pipeline_queue": None,
        }
        if network_path is not None:
            payload["network_path"] = network_path
            payload["nodes"]["master"]["network_path"] = network_path
        return payload

    def test_cluster_status_omits_optional_network_path_for_old_snapshot(
            self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_status",
            lambda: self._cluster_status_payload(),
        )

        response = client.get("/api/cluster/status")

        assert response.status_code == 200
        body = response.json()
        assert "network_path" not in body
        assert "network_path" not in body["nodes"]["master"]
        assert set(body) == {
            "run_mode", "nodes_ready", "nodes", "current_task",
            "tcp_server", "pipeline", "pipeline_queue",
        }

    def test_cluster_status_projects_network_path_without_changing_old_keys(
            self, client, monkeypatch):
        network_path = {
            "schema_version": 1,
            "path_kind": "derp",
            "availability": "available",
            "endpoint": {"role": "master", "host_scope": "tailscale_ipv4", "port": 8888},
            "tailscale": None,
            "tcp_probe": {"state": "available", "reason": "existing_connection", "elapsed_ms": None},
            "quality": {"schema_version": 1, "rtt_ms_p95": 30.0},
        }
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_status",
            lambda: self._cluster_status_payload(network_path),
        )

        response = client.get("/api/cluster/status")

        assert response.status_code == 200
        body = response.json()
        assert body["network_path"] == network_path
        assert body["nodes"]["master"]["network_path"] == network_path
        assert body["nodes_ready"] is True
