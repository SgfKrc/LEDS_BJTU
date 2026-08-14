import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen3_pipeline_data_plane import (  # noqa: E402
    Qwen3ArtifactTransferRuntime,
    router,
)
from qwen3_pipeline_transfer import (  # noqa: E402
    MAX_TRANSFER_CHUNK_BYTES,
    QWEN3_TRANSFER_PREFIX,
    Qwen3ArtifactTransferClient,
    Qwen3TransferError,
    Qwen3TransferTicketSigner,
    TransferResponse,
)


SECRET = "qwen3-transfer-test-secret-value!!"
CHAIN_ID = "a" * 64
PEER_ID = "node-b"


def _environment(tmp_path, *, now=None):
    clock = (lambda: now[0]) if now is not None else None
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / "state",
        cluster_secret=SECRET,
        clock=clock,
    )
    peer = {"id": PEER_ID}
    app = FastAPI()

    @app.middleware("http")
    async def inject_authenticated_peer(request, call_next):
        if peer["id"] is not None:
            request.scope["qlh_authenticated_peer_id"] = peer["id"]
        return await call_next(request)

    app.state.qwen3_artifact_transfer = runtime
    app.include_router(router)
    return TestClient(app), runtime, peer


def _requester(client, *, calls=None):
    def request(method, url, headers, body):
        parsed = urlsplit(url)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        if calls is not None:
            calls.append({
                "method": method,
                "url": target,
                "headers": dict(headers),
                "body_bytes": 0 if body is None else len(body),
            })
        response = client.request(
            method,
            target,
            headers=dict(headers),
            content=body,
        )
        return TransferResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    return request


def _begin(
    runtime,
    data,
    *,
    phase="prefill",
    from_segment=0,
    generation=1,
    peer_node_id=PEER_ID,
    ttl=60,
):
    return runtime.begin_receive(
        base_url="http://127.0.0.1:9876",
        peer_node_id=peer_node_id,
        chain_id=CHAIN_ID,
        generation=generation,
        phase=phase,
        from_segment=from_segment,
        to_segment=from_segment + 1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        ttl_seconds=ttl,
    )


def _url(plan):
    return f"{QWEN3_TRANSFER_PREFIX}/{plan['transfer_id']}"


def _headers(plan):
    return {"Authorization": f"Bearer {plan['ticket']}"}


def test_ticket_is_bound_to_every_handoff_dimension_and_authenticated_peer():
    now = [1000.0]
    signer = Qwen3TransferTicketSigner(SECRET, clock=lambda: now[0])
    ticket = signer.issue(
        transfer_id="qtx_" + "1" * 32,
        peer_node_id=PEER_ID,
        chain_id=CHAIN_ID,
        generation=7,
        phase="decode",
        from_segment=1,
        to_segment=2,
        size_bytes=123,
        sha256="b" * 64,
        ttl_seconds=5,
        nonce="c" * 32,
    )

    payload = signer.verify(ticket, authenticated_peer_id=PEER_ID)
    assert payload == {
        "schema_version": 1,
        "transfer_id": "qtx_" + "1" * 32,
        "peer_node_id": PEER_ID,
        "chain_id": CHAIN_ID,
        "generation": 7,
        "phase": "decode",
        "from_segment": 1,
        "to_segment": 2,
        "size_bytes": 123,
        "sha256": "b" * 64,
        "expires_at": 1005,
        "nonce": "c" * 32,
    }
    with pytest.raises(Qwen3TransferError, match="another peer") as wrong_peer:
        signer.verify(ticket, authenticated_peer_id="node-c")
    assert wrong_peer.value.reason_code == "qwen3_transfer_peer_mismatch"

    encoded, signature = ticket.split(".")
    changed = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(Qwen3TransferError) as tampered:
        signer.verify(f"{encoded}.{changed}", authenticated_peer_id=PEER_ID)
    assert tampered.value.reason_code == "qwen3_transfer_ticket_signature"
    with pytest.raises(Qwen3TransferError) as non_ascii:
        signer.verify("票据.无效", authenticated_peer_id=PEER_ID)
    assert non_ascii.value.reason_code == "qwen3_transfer_ticket_invalid"

    now[0] = 1005.0
    with pytest.raises(Qwen3TransferError) as expired:
        signer.verify(ticket, authenticated_peer_id=PEER_ID)
    assert expired.value.reason_code == "qwen3_transfer_ticket_expired"


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("segment_count", [2, 3])
def test_loopback_streams_two_and_three_segment_handoffs(
    tmp_path,
    phase,
    segment_count,
):
    http, runtime, _ = _environment(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls = []
    transfer = Qwen3ArtifactTransferClient(
        source_root,
        _requester(http, calls=calls),
        chunk_bytes=19,
    )

    for source_segment in range(segment_count - 1):
        data = (
            f"{phase}:{segment_count}:{source_segment}:".encode("ascii")
            + bytes(range(97))
        )
        source = source_root / f"{phase}-{source_segment}.pt"
        source.write_bytes(data)
        plan = _begin(
            runtime,
            data,
            phase=phase,
            from_segment=source_segment,
            generation=11,
        )

        receipt = transfer.upload(source=source, plan=plan)

        assert runtime.receiver.artifact_path(plan["transfer_id"]).read_bytes() == data
        assert receipt["status"] == "committed"
        assert receipt["from_segment"] == source_segment
        assert receipt["to_segment"] == source_segment + 1
        assert receipt["full_model_materialized"] is False
        assert "ticket" not in receipt
        assert "path" not in json.dumps(receipt).lower()

    patches = [call for call in calls if call["method"] == "PATCH"]
    assert patches
    assert all(0 < call["body_bytes"] <= 19 for call in patches)
    assert sum(call["body_bytes"] for call in patches) > 0


def test_disconnect_after_ack_resumes_from_persisted_offset(tmp_path):
    http, runtime, _ = _environment(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    data = bytes(range(128))
    source = source_root / "resume.pt"
    source.write_bytes(data)
    plan = _begin(runtime, data)
    delegate = _requester(http)
    state = {"disconnect": True}

    def disconnect_once(method, url, headers, body):
        response = delegate(method, url, headers, body)
        if method == "PATCH" and state["disconnect"]:
            state["disconnect"] = False
            raise OSError("simulated connection loss after server acknowledgement")
        return response

    interrupted = Qwen3ArtifactTransferClient(
        source_root,
        disconnect_once,
        chunk_bytes=32,
    )
    with pytest.raises(Qwen3TransferError) as failure:
        interrupted.upload(source=source, plan=plan)
    assert failure.value.reason_code == "qwen3_transfer_connection_failed"

    status = http.get(_url(plan), headers=_headers(plan))
    assert status.status_code == 200
    assert status.json()["received_bytes"] == 32

    resumed = Qwen3ArtifactTransferClient(
        source_root,
        delegate,
        chunk_bytes=32,
    ).upload(source=source, plan=plan)
    assert resumed["received_bytes"] == len(data)
    assert runtime.receiver.artifact_path(plan["transfer_id"]).read_bytes() == data


def test_exact_replay_is_idempotent_but_changed_replay_fails_and_cleans(tmp_path):
    http, runtime, _ = _environment(tmp_path)
    data = b"replay-contract-data"
    plan = _begin(runtime, data)
    headers = {
        **_headers(plan),
        "Upload-Offset": "0",
        "Content-Type": "application/octet-stream",
    }
    first = http.patch(_url(plan), headers=headers, content=data[:8])
    replay = http.patch(_url(plan), headers=headers, content=data[:8])
    changed = http.patch(_url(plan), headers=headers, content=b"X" * 8)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["received_bytes"] == 8
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "qwen3_transfer_replay_mismatch"
    status = http.get(_url(plan), headers=_headers(plan))
    assert status.json()["status"] == "failed"
    with pytest.raises(Qwen3TransferError):
        runtime.receiver.artifact_path(plan["transfer_id"])
    assert not list(runtime.receiver.root.glob("*.part"))


def test_gap_offset_fails_closed_and_removes_partial_file(tmp_path):
    http, runtime, _ = _environment(tmp_path)
    data = b"out-of-order"
    plan = _begin(runtime, data)
    response = http.patch(
        _url(plan),
        headers={
            **_headers(plan),
            "Upload-Offset": "2",
            "Content-Type": "application/octet-stream",
        },
        content=data[:3],
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "qwen3_transfer_offset_mismatch"
    assert not list(runtime.receiver.root.glob("*.part"))


def test_wrong_peer_cannot_mutate_or_destroy_legitimate_receive_session(tmp_path):
    http, _, peer = _environment(tmp_path)
    data = b"peer-bound-transfer"
    plan = _begin(http.app.state.qwen3_artifact_transfer, data)
    peer["id"] = "node-c"
    denied = http.patch(
        _url(plan),
        headers={
            **_headers(plan),
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=data,
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "qwen3_transfer_peer_mismatch"

    peer["id"] = PEER_ID
    status = http.get(_url(plan), headers=_headers(plan))
    assert status.status_code == 200
    assert status.json()["status"] == "receiving"
    assert status.json()["received_bytes"] == 0


def test_truncated_commit_is_resumable_but_digest_mismatch_is_cleaned(tmp_path):
    http, runtime, _ = _environment(tmp_path)
    data = b"declared-artifact"
    plan = _begin(runtime, data)
    chunk_headers = {
        **_headers(plan),
        "Upload-Offset": "0",
        "Content-Type": "application/octet-stream",
    }
    partial = http.patch(_url(plan), headers=chunk_headers, content=data[:5])
    incomplete = http.post(f"{_url(plan)}/commit", headers=_headers(plan))
    assert partial.status_code == 200
    assert incomplete.status_code == 409
    assert incomplete.json()["detail"]["code"] == "qwen3_transfer_incomplete"
    assert runtime.receiver.root.joinpath(f".{plan['transfer_id']}.part").is_file()

    second = _begin(runtime, data, phase="decode")
    altered = b"X" + data[1:]
    upload = http.patch(
        _url(second),
        headers={
            **_headers(second),
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=altered,
    )
    mismatch = http.post(f"{_url(second)}/commit", headers=_headers(second))
    assert upload.status_code == 200
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["code"] == "qwen3_transfer_digest_mismatch"
    assert not runtime.receiver.root.joinpath(f".{second['transfer_id']}.part").exists()


def test_cancel_and_expiry_cleanup_remove_staging_artifacts(tmp_path):
    now = [2000.0]
    http, runtime, _ = _environment(tmp_path, now=now)
    data = b"cleanup-artifact"
    cancelled = _begin(runtime, data, ttl=5)
    http.patch(
        _url(cancelled),
        headers={
            **_headers(cancelled),
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=data[:3],
    )
    response = http.delete(_url(cancelled), headers=_headers(cancelled))
    assert response.status_code == 200
    assert response.json()["cleanup_complete"] is True
    assert response.json()["status"] == "cancelled"

    expiring = _begin(runtime, data, ttl=5)
    now[0] = 2005.0
    cleanup = runtime.cleanup_expired()
    assert cleanup == {
        "expired_sessions": 1,
        "removed_artifacts": 1,
        "cleanup_failures": 0,
        "cleanup_complete": True,
    }
    expired = http.get(_url(expiring), headers=_headers(expiring))
    assert expired.status_code == 401
    assert expired.json()["detail"]["code"] == "qwen3_transfer_ticket_expired"


def test_missing_trusted_peer_malformed_ticket_and_scope_mismatch_fail_closed(tmp_path):
    http, runtime, peer = _environment(tmp_path)
    data = b"authorization-contract"
    plan = _begin(runtime, data)
    peer["id"] = None
    missing_peer = http.get(_url(plan), headers=_headers(plan))
    assert missing_peer.status_code == 401
    assert missing_peer.json()["detail"]["code"] == "qwen3_transfer_peer_auth_missing"

    peer["id"] = PEER_ID
    malformed = http.get(
        _url(plan),
        headers={"Authorization": "Bearer malformed"},
    )
    assert malformed.status_code == 401
    assert malformed.json()["detail"]["code"] == "qwen3_transfer_ticket_invalid"
    wrong_target = http.get(
        f"{QWEN3_TRANSFER_PREFIX}/qtx_{'f' * 32}",
        headers=_headers(plan),
    )
    assert wrong_target.status_code == 403
    assert wrong_target.json()["detail"]["code"] == "qwen3_transfer_scope_mismatch"


def test_chunk_limit_plan_metadata_and_source_root_are_enforced(tmp_path):
    http, runtime, _ = _environment(tmp_path)
    data = b"bounded"
    plan = _begin(runtime, data)
    assert "path" not in json.dumps(plan).lower()
    assert str(tmp_path) not in json.dumps(plan)

    oversize = http.patch(
        _url(plan),
        headers={
            **_headers(plan),
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=b"x" * (MAX_TRANSFER_CHUNK_BYTES + 1),
    )
    assert oversize.status_code == 413
    assert oversize.json()["detail"]["code"] == "qwen3_transfer_chunk_oversize"

    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(data)
    transfer = Qwen3ArtifactTransferClient(source_root, _requester(http))
    with pytest.raises(Qwen3TransferError) as escaped:
        transfer.upload(source=outside, plan=plan)
    assert escaped.value.reason_code == "qwen3_transfer_source_scope"

    unsafe_plan = dict(plan)
    unsafe_plan["base_url"] = "http://example.com:9876"
    inside = source_root / "inside.pt"
    inside.write_bytes(data)
    with pytest.raises(Qwen3TransferError) as unsafe_url:
        transfer.upload(source=inside, plan=unsafe_plan)
    assert unsafe_url.value.reason_code == "qwen3_transfer_plan_invalid"


def test_disabled_status_does_not_expose_internal_state():
    app = FastAPI()
    app.state.qwen3_artifact_transfer_reason = "cluster_secret_unavailable"
    app.include_router(router)
    response = TestClient(app).get(f"{QWEN3_TRANSFER_PREFIX}/status")
    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "reason": "cluster_secret_unavailable",
        "prefix": QWEN3_TRANSFER_PREFIX,
    }
