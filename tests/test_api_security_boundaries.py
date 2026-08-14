"""Security rejection contracts for legacy single-process API routes."""

import asyncio
import os
import sys

import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


def _request(host: str, *, headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/security-test",
        "raw_path": b"/security-test",
        "query_string": b"",
        "headers": encoded_headers,
        "client": (host, 32123),
        "server": ("testserver", 80),
    })


def test_system_shutdown_rejects_remote_missing_and_wrong_tokens(monkeypatch):
    remote = _request("192.0.2.10")
    request = api_server.SystemShutdownRequest(reason="test")

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "")
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_server.system_shutdown(request, remote))
    assert missing.value.status_code == 403

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "expected-token")
    with pytest.raises(HTTPException) as wrong:
        asyncio.run(api_server.system_shutdown(
            request,
            _request("192.0.2.10", headers={
                "X-QLH-Shutdown-Token": "wrong-token",
            }),
        ))
    assert wrong.value.status_code == 403


def test_system_shutdown_accepts_only_the_configured_remote_token(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, *, target, daemon, name):
            self._target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            started.append(self._target)

    monkeypatch.setattr(api_server, "_SHUTDOWN_TOKEN", "expected-token")
    monkeypatch.setattr(api_server.threading, "Thread", FakeThread)

    result = asyncio.run(api_server.system_shutdown(
        api_server.SystemShutdownRequest(reason="authorized-test"),
        _request("192.0.2.10", headers={
            "X-QLH-Shutdown-Token": "expected-token",
        }),
    ))

    assert result["ok"] is True
    assert started == [api_server._graceful_exit]


@pytest.mark.parametrize("filename", [
    "../model.gguf",
    "nested/model.gguf",
    "..\\model.gguf",
    "model.bin",
])
def test_model_download_rejects_paths_and_non_gguf_filenames(filename):
    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.download_model_file(filename))

    assert rejected.value.status_code == 400


def test_downloadable_model_rejects_untrusted_peer(monkeypatch):
    import bootstrap

    monkeypatch.setattr(
        bootstrap, "is_trusted_bootstrap_source", lambda host: False,
    )

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.downloadable_pytorch_model(
            _request("192.0.2.10"),
        ))

    assert rejected.value.status_code == 403


def test_log_download_rejects_remote_peer_without_admin_token(monkeypatch):
    monkeypatch.setattr(
        api_server, "_get_request_client", lambda request: "192.0.2.10",
    )
    monkeypatch.delenv("QLH_LOG_ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(api_server.download_log_file(
            _request("192.0.2.10"), "example.log",
        ))

    assert rejected.value.status_code == 403


def test_cluster_mutation_endpoints_map_scheduler_denials_to_forbidden(
    monkeypatch,
):
    class DeniedScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def reset_master_identity():
            return {"status": "denied", "reason": "client denied"}

        @staticmethod
        def connect_to_master(host, port, *, force_bootstrap=False):
            del host, port, force_bootstrap
            return {"status": "denied", "reason": "client denied"}

        @staticmethod
        def transfer_master_role(target_node_id):
            del target_node_id
            return {"status": "denied", "reason": "client denied"}

    monkeypatch.setattr(api_server, "scheduler", DeniedScheduler())

    with pytest.raises(HTTPException) as reset:
        asyncio.run(api_server.reset_master_identity(
            api_server.ResetIdentityRequest(confirm="reset"),
        ))
    with pytest.raises(HTTPException) as connect:
        asyncio.run(api_server.connect_to_master(
            api_server.ConnectToMasterRequest(master_host="100.64.0.10"),
        ))
    with pytest.raises(HTTPException) as transfer:
        asyncio.run(api_server.transfer_master_role(
            api_server.TransferMasterRequest(target_node_id="worker-next"),
        ))

    assert {
        reset.value.status_code,
        connect.value.status_code,
        transfer.value.status_code,
    } == {403}


def test_review_create_and_vote_keep_their_role_and_eligibility_gates(
    monkeypatch,
):
    class UntrustedReviewScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def get_effective_node_id():
            return "worker-no-gpu"

        @staticmethod
        def can_node_vote(node_id):
            assert node_id == "worker-no-gpu"
            return False, "not eligible"

    monkeypatch.setattr(api_server, "scheduler", UntrustedReviewScheduler())

    with pytest.raises(HTTPException) as create:
        asyncio.run(api_server.create_review_ticket(
            api_server.CreateReviewRequest(target_node_id="worker-next"),
        ))
    with pytest.raises(HTTPException) as vote:
        asyncio.run(api_server.cast_review_vote(
            api_server.CastVoteRequest(ticket_id="review-test", vote=1),
        ))

    assert create.value.status_code == 403
    assert vote.value.status_code == 403


def test_review_side_effect_endpoints_require_master_role(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())

    endpoints = (
        lambda: api_server.trigger_expire_check(),
        lambda: api_server.delete_review_ticket("review-test"),
        api_server.delete_resolved_review_tickets,
        api_server.trigger_mail_poll,
    )
    for endpoint in endpoints:
        with pytest.raises(HTTPException) as rejected:
            asyncio.run(endpoint())
        assert rejected.value.status_code == 403
        assert rejected.value.detail == "仅主节点可执行此操作"


def test_review_side_effect_endpoints_execute_for_master(monkeypatch):
    import email_notifier
    import review

    class MasterScheduler:
        @staticmethod
        def _effective_role():
            return "master"

    class FakeReviewManager:
        def resolve_expired(self):
            return ["review-expired"]

        def delete_ticket(self, ticket_id):
            assert ticket_id == "review-test"
            return True

        def delete_resolved(self):
            return 2

    monkeypatch.setattr(api_server, "scheduler", MasterScheduler())
    monkeypatch.setattr(review, "ReviewManager", FakeReviewManager)
    monkeypatch.setattr(
        email_notifier, "poll_mail_once", lambda: {"polled": 1},
    )

    assert asyncio.run(api_server.trigger_expire_check()) == {
        "expired": ["review-expired"],
        "count": 1,
    }
    assert asyncio.run(api_server.delete_review_ticket("review-test")) == {
        "status": "deleted",
        "ticket_id": "review-test",
    }
    assert asyncio.run(api_server.delete_resolved_review_tickets()) == {
        "status": "deleted",
        "count": 2,
    }
    assert asyncio.run(api_server.trigger_mail_poll()) == {"polled": 1}


# ================================================================
# T-3 读端点开放契约（§4.2.1 矩阵 3/4/5）：网络信任域内读开放，
# 不因部署角色变化而 403（写端点的 master 门在
# test_review_side_effect_endpoints_* 已覆盖）
# ================================================================

def test_review_read_endpoints_stay_open_for_non_master(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

        @staticmethod
        def get_effective_node_id():
            return "node-client"

        @staticmethod
        def can_node_vote(node_id):
            return True, ""

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())
    # 列表：非 master 角色必须可读（不 403）
    result = asyncio.run(api_server.list_review_tickets())
    assert "tickets" in result and "count" in result
    # can-vote 自查询：开放
    result = asyncio.run(api_server.check_can_vote())
    assert "can_vote" in result


def test_review_ticket_detail_missing_is_404_not_403_for_non_master(monkeypatch):
    class ClientScheduler:
        @staticmethod
        def _effective_role():
            return "client"

    monkeypatch.setattr(api_server, "scheduler", ClientScheduler())
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api_server.get_review_ticket("ticket-does-not-exist"))
    # 非 master 读不存在工单 → 404（资源不存在优先于角色拒绝，契约 4）
    assert missing.value.status_code == 404
