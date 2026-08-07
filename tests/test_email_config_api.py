"""
API 测试 — 管理员收件邮箱配置端点（GET/POST /api/cluster/email-config）
====================================================================
覆盖：查询返回配置且不泄露 SMTP 凭据；保存调用 set_admin_email 并回传生效值；
非法邮箱映射为 400；scheduler-svc 微服务版端点与单体版行为一致。
"""

import json
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
import email_notifier
import scheduler_svc_http


class TestEmailConfigApi:
    """单体 api_server 的 /api/cluster/email-config 端点。"""

    @pytest.fixture(autouse=True)
    def _master_role(self, monkeypatch):
        """默认以主节点角色调用（管理员邮箱为集群级配置，仅主节点可读写）。"""
        monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")

    def test_get_returns_config_without_secrets(self, monkeypatch):
        monkeypatch.setattr(
            email_notifier,
            "admin_email_config",
            lambda: {"recipient": "a@b.com", "source": "node_config"},
        )
        monkeypatch.setattr(email_notifier, "SMTP_SENDER", "s@qq.com")
        monkeypatch.setattr(email_notifier, "SMTP_PASSWORD", "super-secret")
        with TestClient(api_server.app) as client:
            res = client.get("/api/cluster/email-config")
        assert res.status_code == 200
        body = res.json()
        assert body["recipient"] == "a@b.com"
        assert body["source"] == "node_config"
        assert body["smtp_configured"] is True
        assert "super-secret" not in json.dumps(body)

    def test_post_saves_recipient_and_returns_effective(self, monkeypatch):
        captured = {}

        def fake_set(email):
            captured["email"] = email
            return email.lower()

        monkeypatch.setattr(email_notifier, "set_admin_email", fake_set)
        with TestClient(api_server.app) as client:
            res = client.post(
                "/api/cluster/email-config",
                json={"recipient": "Ops@Example.com"},
            )
        assert res.status_code == 200
        assert captured["email"] == "Ops@Example.com"
        assert res.json() == {"status": "ok", "recipient": "ops@example.com"}

    def test_post_rejects_invalid_email(self, monkeypatch):
        def fake_set(email):
            raise ValueError("非法邮箱地址: 'bad'")

        monkeypatch.setattr(email_notifier, "set_admin_email", fake_set)
        with TestClient(api_server.app) as client:
            res = client.post(
                "/api/cluster/email-config",
                json={"recipient": "bad"},
            )
        assert res.status_code == 400
        assert "非法邮箱地址" in res.json()["detail"]

    def test_post_empty_clears_override(self, monkeypatch):
        captured = {}

        def fake_set(email):
            captured["email"] = email
            return "env@example.com"

        monkeypatch.setattr(email_notifier, "set_admin_email", fake_set)
        with TestClient(api_server.app) as client:
            res = client.post("/api/cluster/email-config", json={"recipient": ""})
        assert res.status_code == 200
        assert captured["email"] == ""
        assert res.json() == {"status": "ok", "recipient": "env@example.com"}

    # ---- 问题 #1 修复：仅主节点可配置管理员邮箱 ----

    def test_slave_get_email_config_forbidden(self, monkeypatch):
        monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
        with TestClient(api_server.app) as client:
            res = client.get("/api/cluster/email-config")
        assert res.status_code == 403
        assert "仅主节点" in res.json()["detail"]

    def test_slave_post_email_config_forbidden(self, monkeypatch):
        monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
        with TestClient(api_server.app) as client:
            res = client.post(
                "/api/cluster/email-config",
                json={"recipient": "x@example.com"},
            )
        assert res.status_code == 403

    def test_slave_email_test_forbidden(self, monkeypatch):
        monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
        with TestClient(api_server.app) as client:
            res = client.post("/api/cluster/email-test")
        assert res.status_code == 403


class TestSchedulerSvcEmailConfigApi:
    """scheduler-svc 微服务版 /cluster/email-config 端点。"""

    @pytest.fixture
    def svc_client(self, monkeypatch):
        app = FastAPI()
        app.include_router(scheduler_svc_http.router)
        class _FakeScheduler:
            def _effective_role(self):
                return "master"
        scheduler_svc_http.set_scheduler(_FakeScheduler())
        with TestClient(app) as client:
            yield client
        scheduler_svc_http.reset_scheduler()

    def test_get_and_post_agree_with_monolith(self, svc_client, monkeypatch):
        monkeypatch.setattr(
            email_notifier,
            "admin_email_config",
            lambda: {"recipient": "svc@example.com", "source": "node_config"},
        )
        monkeypatch.setattr(email_notifier, "SMTP_SENDER", "s@qq.com")
        monkeypatch.setattr(email_notifier, "SMTP_PASSWORD", "p")

        res = svc_client.get("/cluster/email-config")
        assert res.status_code == 200
        assert res.json()["recipient"] == "svc@example.com"

        def fake_set(email):
            return email.lower()

        monkeypatch.setattr(email_notifier, "set_admin_email", fake_set)
        res = svc_client.post(
            "/cluster/email-config",
            json={"recipient": "svc-ops@example.com"},
        )
        assert res.status_code == 200
        assert res.json()["recipient"] == "svc-ops@example.com"

    # ---- 问题 #1 修复：仅主节点可配置管理员邮箱 ----

    def test_slave_get_forbidden(self, svc_client, monkeypatch):
        class _SlaveScheduler:
            def _effective_role(self):
                return "client"
        scheduler_svc_http.set_scheduler(_SlaveScheduler())
        res = svc_client.get("/cluster/email-config")
        assert res.status_code == 403
        assert "仅主节点" in res.json()["detail"]

    def test_slave_post_forbidden(self, svc_client, monkeypatch):
        class _SlaveScheduler:
            def _effective_role(self):
                return "client"
        scheduler_svc_http.set_scheduler(_SlaveScheduler())
        res = svc_client.post(
            "/cluster/email-config",
            json={"recipient": "x@example.com"},
        )
        assert res.status_code == 403
