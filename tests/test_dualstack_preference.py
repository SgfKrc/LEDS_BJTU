"""Regression coverage for explicit dual-stack bootstrap preferences."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_explicit_ipv6_preference_survives_restart_and_keeps_bootstrap_fallback(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    from node_config import (
        apply_node_config_to_env,
        get_bootstrap_master_endpoint,
        get_preferred_master_endpoint,
        load_node_config,
        persist_preferred_master_endpoint,
        write_node_config,
    )

    write_node_config({
        "cluster": {
            "cluster_id": "cluster-a",
            "master_tcp_host": "100.90.76.108",
            "master_tcp_port": 8888,
        },
        "node": {"role": "client"},
    })
    endpoint = persist_preferred_master_endpoint("[fd7a:115c:a1e0::8d01:4cc5]", 8888)

    assert endpoint == {
        "host": "fd7a:115c:a1e0::8d01:4cc5",
        "port": 8888,
        "address_family": "ipv6",
    }
    data = load_node_config()
    assert get_preferred_master_endpoint(data) == endpoint
    assert get_bootstrap_master_endpoint(data) == {
        "host": "100.90.76.108",
        "port": 8888,
        "address_family": "ipv4",
    }

    monkeypatch.setenv("QLH_CLIENT_MASTER_HOST", "100.90.76.108")
    monkeypatch.setenv("QLH_CLIENT_MASTER_PORT", "8888")
    apply_node_config_to_env(data)
    assert os.environ["QLH_CLIENT_MASTER_HOST"] == endpoint["host"]
    assert os.environ["QLH_CLIENT_MASTER_PORT"] == "8888"


def test_bootstrap_preserves_preference_only_for_same_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
    from node_config import build_bootstrap_config, write_node_config

    write_node_config({
        "cluster": {
            "cluster_id": "cluster-a",
            "master_tcp_host": "100.90.76.108",
            "master_tcp_port": 8888,
            "preferred_master_endpoint": {
                "host": "fd7a:115c:a1e0::8d01:4cc5",
                "port": 8888,
            },
        },
    })
    same_cluster = build_bootstrap_config({
        "cluster": {"cluster_id": "cluster-a", "master_tcp_host": "100.90.76.108"},
        "node": {"role": "client"},
    })
    other_cluster = build_bootstrap_config({
        "cluster": {"cluster_id": "cluster-b", "master_tcp_host": "100.90.76.109"},
        "node": {"role": "client"},
    })

    assert same_cluster["cluster"]["preferred_master_endpoint"]["host"] == "fd7a:115c:a1e0::8d01:4cc5"
    assert "preferred_master_endpoint" not in other_cluster["cluster"]


def test_fallback_order_is_preference_then_bootstrap_then_tailnet(monkeypatch):
    from scheduler import Scheduler
    import node_config

    scheduler = Scheduler()
    monkeypatch.setattr(
        node_config,
        "get_bootstrap_master_endpoint",
        lambda: {"host": "100.90.76.108", "port": 8888, "address_family": "ipv4"},
    )
    monkeypatch.setattr(
        scheduler,
        "discover_master",
        lambda *, skip_config=False: {
            "found": True,
            "master_host": "master.tailnet.ts.net",
            "master_port": 8888,
            "stale": False,
            "source": "tailnet",
        },
    )

    candidates = scheduler._discover_master_fallbacks(
        "fd7a:115c:a1e0::8d01:4cc5", 8888,
    )

    assert [(item["master_host"], item["source"]) for item in candidates] == [
        ("100.90.76.108", "bootstrap_config"),
        ("master.tailnet.ts.net", "tailnet"),
    ]


def test_explicit_connect_persists_the_canonical_endpoint_after_registration(
    monkeypatch,
):
    import config as cfg
    import node_config
    import scheduler as scheduler_mod
    from scheduler import Scheduler

    monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
    monkeypatch.setattr(cfg, "NODE_ID", "client-preference", raising=False)
    monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-preference", raising=False)
    persisted = []
    monkeypatch.setattr(
        node_config,
        "persist_preferred_master_endpoint",
        lambda host, port: persisted.append((host, port)) or {
            "host": host,
            "port": port,
            "address_family": "ipv6",
        },
    )

    class FakeTCPClient:
        def __init__(self, server_host, server_port, client_id, role, **_kwargs):
            self.server_host = server_host
            self.server_port = server_port
            self.client_id = client_id
            self.role = role
            self.is_registered = False
            self._running = False
            self.sock = object()
            self.last_register_error = ""

        def connect(self, on_message=None):
            self.is_registered = True
            self._running = True
            return True

        def send_data(self, _data, _message_type):
            return None

    monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)
    scheduler = Scheduler()
    scheduler._role_override = "client"

    result = scheduler.connect_to_master(
        "[fd7a:115c:a1e0::8d01:4cc5]", 8888, persist_preference=True,
    )

    assert result["endpoint_preference"] == {
        "persisted": True,
        "address_family": "ipv6",
    }
    assert persisted == [("fd7a:115c:a1e0::8d01:4cc5", 8888)]
