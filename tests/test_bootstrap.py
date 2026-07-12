"""
单元测试 — 首次连接自动部署
===========================
纯逻辑测试，不启动后端、不访问网络。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_tailscale_cidr_is_trusted():
    from bootstrap import is_trusted_bootstrap_source

    assert is_trusted_bootstrap_source("100.64.0.1")
    assert is_trusted_bootstrap_source("100.127.255.254")
    assert is_trusted_bootstrap_source("127.0.0.1")
    assert not is_trusted_bootstrap_source("8.8.8.8")
    assert not is_trusted_bootstrap_source("192.168.1.10")


def test_normalize_node_id_rejects_master():
    from bootstrap import normalize_node_id

    assert normalize_node_id("master", "pc").startswith("client_")
    assert normalize_node_id("android phone/1", "android") == "android_phone_1"


def test_persist_bootstrap_response_writes_node_config(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    monkeypatch.delenv("QLH_CLUSTER_SECRET", raising=False)
    monkeypatch.delenv("QLH_MASTER_HOST", raising=False)
    monkeypatch.delenv("QLH_MASTER_PORT", raising=False)

    from node_config import load_node_config, persist_bootstrap_response

    response = {
        "status": "ok",
        "cluster": {
            "cluster_id": "test-cluster",
            "master_api_host": "100.64.0.10",
            "master_api_port": 8000,
            "master_tcp_host": "100.64.0.10",
            "master_tcp_port": 8888,
            "cluster_secret": "secret-123",
        },
        "node": {
            "node_id": "client-test",
            "role": "client",
            "node_type": "pc",
            "pipeline_worker": True,
        },
    }

    path = persist_bootstrap_response(response)
    assert path.is_file()
    data = load_node_config()
    assert data["bootstrapped"] is True
    assert data["cluster"]["cluster_secret"] == "secret-123"
    assert data["cluster"]["master_tcp_host"] == "100.64.0.10"
    assert data["node"]["node_id"] == "client-test"
    assert os.environ["QLH_CLUSTER_SECRET"] == "secret-123"
    assert os.environ["QLH_MASTER_HOST"] == "100.64.0.10"
    assert os.environ["QLH_MASTER_PORT"] == "8888"


def test_bootstrap_api_port_ignores_local_api_port(monkeypatch):
    monkeypatch.delenv("QLH_BOOTSTRAP_API_PORT", raising=False)
    monkeypatch.delenv("QLH_MASTER_API_PORT", raising=False)
    monkeypatch.setenv("QLH_API_PORT", "8001")

    from scheduler import _bootstrap_api_port

    assert _bootstrap_api_port() == 8000

    monkeypatch.setenv("QLH_MASTER_API_PORT", "18000")
    assert _bootstrap_api_port() == 18000

    monkeypatch.setenv("QLH_BOOTSTRAP_API_PORT", "18001")
    assert _bootstrap_api_port() == 18001


def test_apply_runtime_config_syncs_loaded_scheduler(monkeypatch):
    import config as cfg
    import scheduler as scheduler_mod
    from node_config import apply_runtime_config

    monkeypatch.setattr(cfg, "NODE_ID", "old-client", raising=False)
    monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ID", "stale-client", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)

    apply_runtime_config({
        "cluster": {
            "cluster_secret": "secret-456",
            "master_tcp_host": "100.64.0.20",
            "master_tcp_port": 8889,
        },
        "node": {
            "node_id": "client-runtime",
            "role": "client",
        },
    })

    assert cfg.NODE_ID == "client-runtime"
    assert cfg.NODE_ROLE == "client"
    assert scheduler_mod.NODE_ID == "client-runtime"
    assert scheduler_mod.NODE_ROLE == "client"
