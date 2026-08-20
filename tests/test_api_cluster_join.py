from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "src")


@pytest.fixture
def join_api(monkeypatch, tmp_path):
    import api_server

    monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "control.sqlite3"))
    monkeypatch.setenv("QLH_CLUSTER_ID", "cluster-test")
    monkeypatch.setenv("QLH_CLUSTER_JOIN_BOOLEAN_APPROVAL", "true")
    monkeypatch.setattr(api_server, "_join_ledger_instance", None)
    role = {"value": "client"}
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: role["value"])
    monkeypatch.setattr(api_server.scheduler, "get_effective_node_id", lambda: "client-test")
    monkeypatch.setattr(api_server.scheduler, "can_join_existing_master", lambda: True)
    monkeypatch.setattr(
        api_server.scheduler,
        "connect_to_master",
        lambda host, port, **kwargs: {
            "status": "connected", "master_host": host, "master_port": port,
        },
    )
    with TestClient(api_server.app) as client:
        yield client, role
    monkeypatch.setattr(api_server, "_join_ledger_instance", None)


def test_join_request_issue_consume_and_replay_rejected(join_api):
    client, role = join_api
    created = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "[fd7a:115c::1]:8888", "target_node_id": "client-test"},
    )
    assert created.status_code == 200, created.text
    request_code = created.json()["request_code"]
    assert request_code.startswith("qlhjoinreq1.")

    role["value"] = "master"
    denied = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code, "auth_verified": False},
    )
    assert denied.status_code == 403

    issued = client.post(
        "/api/cluster/join/grant",
        json={"request_code": request_code, "auth_verified": True},
    )
    assert issued.status_code == 200, issued.text
    grant_code = issued.json()["grant_code"]
    assert grant_code.startswith("qlhjoin1.")
    assert issued.json()["qr_payload"] == grant_code

    role["value"] = "client"
    consumed = client.post("/api/cluster/join/consume", json={"grant_code": grant_code})
    assert consumed.status_code == 200, consumed.text
    assert consumed.json()["role"] == "client"

    replay = client.post("/api/cluster/join/consume", json={"grant_code": grant_code})
    assert replay.status_code == 409
    assert replay.json()["detail"]["code"] == "request_not_found"


def test_join_grant_requires_master_role(join_api):
    client, role = join_api
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "auth_verified": True},
    )
    assert response.status_code == 403
    role["value"] = "master"
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "auth_verified": True},
    )
    assert response.status_code == 400


def test_boolean_approval_is_disabled_by_default(join_api, monkeypatch):
    client, role = join_api
    role["value"] = "master"
    monkeypatch.delenv("QLH_CLUSTER_JOIN_BOOLEAN_APPROVAL", raising=False)
    response = client.post(
        "/api/cluster/join/grant",
        json={"request_code": "qlhjoinreq1.invalid", "auth_verified": True},
    )
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "auth_control_plane_unavailable"


def test_provisional_master_request_uses_future_client_id(join_api, monkeypatch):
    client, role = join_api
    monkeypatch.setattr("socket.gethostname", lambda: "provisional-box")
    import api_server
    monkeypatch.setattr(api_server.scheduler, "get_effective_node_id", lambda: "master")
    # The fixture's role is master for the API decision, while the scheduler
    # reports the pre-switch identity as master.
    role["value"] = "master"
    response = client.post(
        "/api/cluster/join/request",
        json={"master_endpoint": "100.64.0.10:8888"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["target_node_id"] == "client_provisional-box"
