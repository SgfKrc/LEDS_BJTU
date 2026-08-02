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
