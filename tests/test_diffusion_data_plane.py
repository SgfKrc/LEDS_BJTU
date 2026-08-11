import hashlib
import io
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.data_plane import (  # noqa: E402
    DATA_PLANE_PREFIX,
    MAX_TRANSFER_CHUNK_BYTES,
    DiffusionDataPlaneRuntime,
    router,
)
from diffusion.distributed import (  # noqa: E402
    BlobAuthorizationError,
    BlobConflict,
    BlobNotFound,
    BlobValidationError,
)
import diffusion.data_plane as data_plane_module  # noqa: E402


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 40, 80)).save(output, format="PNG")
    return output.getvalue()


def _client(tmp_path, *, now=None):
    clock = (lambda: now[0]) if now is not None else None
    runtime = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="s" * 32,
        clock=clock,
    )
    app = FastAPI()
    app.state.diffusion_data_plane = runtime
    app.include_router(router)
    return TestClient(app), runtime


def _begin(runtime, data, *, attempt_id="att_data_plane_12345678", ttl=60):
    return runtime.begin_upload(
        attempt_id=attempt_id,
        grant_ttl_seconds=ttl,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        expected_size=len(data),
        content_type="image/png",
        purpose="output",
        owner_scope="workflow_12345678",
        width=16,
        height=12,
    )


def _upload_url(attempt_id, upload_id):
    return f"{DATA_PLANE_PREFIX}/attempts/{attempt_id}/uploads/{upload_id}"


def test_chunk_upload_status_replay_and_commit_are_idempotent(tmp_path):
    client, runtime = _client(tmp_path)
    data = _png()
    attempt_id = "att_data_plane_12345678"
    created = _begin(runtime, data, attempt_id=attempt_id)
    upload = created["upload"]
    url = _upload_url(attempt_id, upload["upload_id"])
    headers = {"Authorization": f"Bearer {created['grant']}"}
    split = len(data) // 2

    first = client.patch(
        url,
        headers={**headers, "Upload-Offset": "0", "Content-Type": "application/octet-stream"},
        content=data[:split],
    )
    assert first.status_code == 200
    assert first.headers["Upload-Offset"] == str(split)

    replay = client.patch(
        url,
        headers={**headers, "Upload-Offset": "0", "Content-Type": "application/octet-stream"},
        content=data[:split],
    )
    assert replay.status_code == 200
    assert replay.json()["received_bytes"] == split

    mismatch = client.patch(
        url,
        headers={**headers, "Upload-Offset": "0", "Content-Type": "application/octet-stream"},
        content=b"x" * split,
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "upload_replay_mismatch"

    second = client.patch(
        url,
        headers={
            **headers,
            "Upload-Offset": str(split),
            "Content-Type": "application/octet-stream",
        },
        content=data[split:],
    )
    assert second.status_code == 200
    assert second.json()["received_bytes"] == len(data)

    status = client.get(url, headers=headers)
    assert status.status_code == 200
    assert status.headers["Upload-Offset"] == str(len(data))

    committed = client.post(f"{url}/commit", headers=headers)
    repeated = client.post(f"{url}/commit", headers=headers)
    assert committed.status_code == 200
    assert repeated.status_code == 200
    assert repeated.json() == committed.json()
    runtime.close()


def test_download_is_attempt_blob_lease_and_range_scoped(tmp_path):
    now = [100.0]
    client, runtime = _client(tmp_path, now=now)
    data = _png()
    attempt_id = "att_download_12345678"
    created = _begin(runtime, data, attempt_id=attempt_id, ttl=5)
    upload_url = _upload_url(attempt_id, created["upload"]["upload_id"])
    upload_headers = {"Authorization": f"Bearer {created['grant']}"}
    client.patch(
        upload_url,
        headers={
            **upload_headers,
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=data,
    )
    descriptor = client.post(f"{upload_url}/commit", headers=upload_headers).json()

    download = runtime.grant_download(
        descriptor["blob_id"],
        attempt_id=attempt_id,
        ttl_seconds=5,
    )
    lease_id = download["lease"]["lease_id"]
    url = (
        f"{DATA_PLANE_PREFIX}/attempts/{attempt_id}/blobs/{descriptor['blob_id']}"
    )
    headers = {
        "Authorization": f"Bearer {download['grant']}",
        "X-Blob-Lease": lease_id,
    }
    first = client.get(f"{url}?offset=0&length=20", headers=headers)
    second = client.get(f"{url}?offset=20&length={len(data)}", headers=headers)
    assert first.status_code == 206
    assert second.status_code == 206
    assert first.content + second.content == data
    assert first.headers["X-Blob-SHA256"] == hashlib.sha256(data).hexdigest()

    wrong_attempt = client.get(
        url.replace(attempt_id, "att_wrong_12345678"),
        headers=headers,
    )
    assert wrong_attempt.status_code == 401
    assert wrong_attempt.json()["detail"]["code"] == "transfer_scope_mismatch"

    wrong_blob = client.get(
        url.replace(descriptor["blob_id"], "img_abcdef1234567890"),
        headers=headers,
    )
    assert wrong_blob.status_code == 403
    assert wrong_blob.json()["detail"]["code"] == "transfer_scope_mismatch"

    now[0] = 106.0
    expired = client.get(url, headers=headers)
    assert expired.status_code == 401
    assert expired.json()["detail"]["code"] == "transfer_grant_expired"
    runtime.close()


def test_data_plane_rejects_unbounded_body_and_reports_disabled_state(tmp_path):
    client, runtime = _client(tmp_path)
    data = _png()
    created = _begin(runtime, data)
    url = _upload_url("att_data_plane_12345678", created["upload"]["upload_id"])
    rejected = client.patch(
        url,
        headers={
            "Authorization": f"Bearer {created['grant']}",
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=b"x" * (MAX_TRANSFER_CHUNK_BYTES + 1),
    )
    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "transfer_chunk_too_large"
    runtime.close()

    app = FastAPI()
    app.state.diffusion_data_plane_reason = "cluster_secret_missing"
    app.include_router(router)
    disabled = TestClient(app).get(f"{DATA_PLANE_PREFIX}/status")
    assert disabled.status_code == 200
    assert disabled.json() == {
        "enabled": False,
        "reason": "cluster_secret_missing",
        "prefix": DATA_PLANE_PREFIX,
    }


def test_upload_endpoints_require_matching_grant_attempt_and_upload_scope(tmp_path):
    client, runtime = _client(tmp_path)
    data = _png()
    attempt_id = "att_data_plane_12345678"
    created = _begin(runtime, data, attempt_id=attempt_id)
    upload_id = created["upload"]["upload_id"]
    url = _upload_url(attempt_id, upload_id)
    headers = {"Authorization": f"Bearer {created['grant']}"}

    missing = client.get(url)
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "missing_transfer_grant"
    assert missing.headers["WWW-Authenticate"] == "Bearer"

    wrong_attempt = client.get(
        _upload_url("att_other_scope_12345678", upload_id), headers=headers,
    )
    assert wrong_attempt.status_code == 401
    assert wrong_attempt.json()["detail"]["code"] == "transfer_scope_mismatch"

    wrong_url = _upload_url(attempt_id, "up_wrong_scope_12345678")
    scope_responses = [
        client.get(wrong_url, headers=headers),
        client.patch(
            wrong_url,
            headers={
                **headers,
                "Upload-Offset": "0",
                "Content-Type": "application/octet-stream",
            },
            content=data,
        ),
        client.post(f"{wrong_url}/commit", headers=headers),
    ]
    assert [response.status_code for response in scope_responses] == [403, 403, 403]
    assert all(
        response.json()["detail"]["code"] == "transfer_scope_mismatch"
        for response in scope_responses
    )
    runtime.close()


def test_upload_chunk_rejects_invalid_request_shapes(tmp_path):
    client, runtime = _client(tmp_path)
    data = _png()
    created = _begin(runtime, data)
    url = _upload_url("att_data_plane_12345678", created["upload"]["upload_id"])
    headers = {"Authorization": f"Bearer {created['grant']}"}

    invalid_offset = client.patch(
        url,
        headers={
            **headers,
            "Upload-Offset": "not-an-integer",
            "Content-Type": "application/octet-stream",
        },
        content=data,
    )
    unsupported_type = client.patch(
        url,
        headers={**headers, "Upload-Offset": "0", "Content-Type": "text/plain"},
        content=data,
    )
    empty_chunk = client.patch(
        url,
        headers={
            **headers,
            "Upload-Offset": "0",
            "Content-Type": "application/octet-stream",
        },
        content=b"",
    )
    commit_body = client.post(f"{url}/commit", headers=headers, content=b"unexpected")

    assert (invalid_offset.status_code, invalid_offset.json()["detail"]["code"]) == (
        400, "invalid_upload_offset",
    )
    assert (unsupported_type.status_code, unsupported_type.json()["detail"]["code"]) == (
        415, "unsupported_transfer_content_type",
    )
    assert (empty_chunk.status_code, empty_chunk.json()["detail"]["code"]) == (
        400, "empty_transfer_chunk",
    )
    assert (commit_body.status_code, commit_body.json()["detail"]["code"]) == (
        400, "commit_body_not_allowed",
    )
    runtime.close()


@pytest.mark.parametrize(
    "error,status",
    [
        (BlobAuthorizationError("denied", code="denied"), 403),
        (BlobNotFound("missing", code="missing"), 404),
        (BlobConflict("conflict", code="conflict"), 409),
        (BlobValidationError("invalid", code="invalid"), 422),
    ],
)
def test_data_plane_http_error_mapping_is_stable(error, status):
    response = data_plane_module._http_error(error)

    assert response.status_code == status
    assert response.detail == {"code": error.code, "message": str(error)}
