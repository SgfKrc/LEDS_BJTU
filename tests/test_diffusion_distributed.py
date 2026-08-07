import hashlib
import io
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.distributed import (  # noqa: E402
    BlobAuthorizationError,
    BlobConflict,
    BlobNotFound,
    BlobTransferTokenSigner,
    BlobValidationError,
    DiffusionArtifactComponent,
    DiffusionArtifactManifest,
    PersistentImageBlobStore,
)
import diffusion.distributed as distributed_module  # noqa: E402


def _png(*, size=(16, 12), color=(20, 40, 80)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _put(store, data, *, owner="session-a", purpose="input_image", **kwargs):
    return store.put_bytes(
        data,
        content_type="image/png",
        purpose=purpose,
        owner_scope=owner,
        width=16,
        height=12,
        **kwargs,
    )


def test_chunked_upload_is_persistent_and_never_exposes_a_path(tmp_path):
    data = _png()
    store = PersistentImageBlobStore(tmp_path / "blobs")
    session = store.begin_upload(
        expected_sha256=hashlib.sha256(data).hexdigest(),
        expected_size=len(data),
        content_type="image/png",
        purpose="input_image",
        owner_scope="session-a",
        width=16,
        height=12,
    )
    split = len(data) // 2
    first = store.write_upload(session.upload_id, offset=0, data=data[:split])
    second = store.write_upload(session.upload_id, offset=split, data=data[split:])
    descriptor = store.commit_upload(session.upload_id)

    assert first.received_bytes == split
    assert second.received_bytes == len(data)
    assert descriptor.sha256 == hashlib.sha256(data).hexdigest()
    assert descriptor.snapshot().keys() == {
        "blob_id",
        "sha256",
        "size_bytes",
        "content_type",
        "width",
        "height",
        "purpose",
        "created_at",
        "expires_at",
        "lease_count",
    }
    store.close()

    reopened = PersistentImageBlobStore(tmp_path / "blobs")
    try:
        assert reopened.descriptor(descriptor.blob_id).snapshot() == descriptor.snapshot()
        assert reopened.snapshot()["objects"] == 1
    finally:
        reopened.close()


def test_upload_rejects_gap_overflow_hash_mime_and_dimensions(tmp_path):
    data = _png()
    store = PersistentImageBlobStore(tmp_path / "blobs")
    try:
        session = store.begin_upload(
            expected_sha256="0" * 64,
            expected_size=len(data),
            content_type="image/png",
            purpose="input_image",
            owner_scope="session-a",
            width=16,
            height=12,
        )
        with pytest.raises(BlobConflict, match="offset") as gap:
            store.write_upload(session.upload_id, offset=1, data=data[:1])
        assert gap.value.code == "upload_offset_mismatch"
        store.write_upload(session.upload_id, offset=0, data=data)
        with pytest.raises(BlobValidationError, match="SHA-256") as digest:
            store.commit_upload(session.upload_id)
        assert digest.value.code == "sha256_mismatch"

        wrong_mime = store.begin_upload(
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size=len(data),
            content_type="image/jpeg",
            purpose="input_image",
            owner_scope="session-a",
            width=16,
            height=12,
        )
        store.write_upload(wrong_mime.upload_id, offset=0, data=data)
        with pytest.raises(BlobValidationError) as mime:
            store.commit_upload(wrong_mime.upload_id)
        assert mime.value.code == "content_type_mismatch"

        wrong_size = store.begin_upload(
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size=len(data),
            content_type="image/png",
            purpose="input_image",
            owner_scope="session-a",
            width=12,
            height=16,
        )
        store.write_upload(wrong_size.upload_id, offset=0, data=data)
        with pytest.raises(BlobValidationError) as dimensions:
            store.commit_upload(wrong_size.upload_id)
        assert dimensions.value.code == "dimension_mismatch"
    finally:
        store.close()


def test_identical_uploads_deduplicate_objects_without_crossing_owner_scope(tmp_path):
    data = _png()
    store_a = PersistentImageBlobStore(tmp_path / "blobs")
    store_b = PersistentImageBlobStore(tmp_path / "blobs")
    try:
        first = _put(store_a, data, owner="session-a")
        duplicate = _put(store_b, data, owner="session-a")
        other_owner = _put(store_b, data, owner="session-b")

        assert duplicate.blob_id == first.blob_id
        assert other_owner.blob_id != first.blob_id
        assert store_a.snapshot()["blobs"] == 2
        assert store_a.snapshot()["objects"] == 1
        assert store_a.snapshot()["total_bytes"] == len(data)
    finally:
        store_b.close()
        store_a.close()


def test_attempt_lease_guards_reads_deletes_and_expiry(tmp_path):
    now = [100.0]
    store = PersistentImageBlobStore(
        tmp_path / "blobs",
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    data = _png()
    try:
        descriptor = _put(store, data)
        lease = store.acquire_lease(
            descriptor.blob_id,
            attempt_id="att_test_12345678",
            ttl_seconds=5,
        )
        assert store.read_all(
            descriptor.blob_id,
            lease_id=lease.lease_id,
            attempt_id="att_test_12345678",
        ) == data
        with pytest.raises(BlobAuthorizationError):
            store.read_all(
                descriptor.blob_id,
                lease_id=lease.lease_id,
                attempt_id="att_other_12345678",
            )
        with pytest.raises(BlobConflict) as active:
            store.delete(descriptor.blob_id, owner_scope="session-a")
        assert active.value.code == "blob_in_use"

        now[0] = 106.0
        with pytest.raises(BlobAuthorizationError):
            store.read_all(
                descriptor.blob_id,
                lease_id=lease.lease_id,
                attempt_id="att_test_12345678",
            )
        assert store.descriptor(descriptor.blob_id).lease_count == 0
        assert store.delete(descriptor.blob_id, owner_scope="session-a") is True
    finally:
        store.close()


def test_parent_reference_and_owner_checks_fail_closed(tmp_path):
    store = PersistentImageBlobStore(tmp_path / "blobs")
    try:
        parent = _put(store, _png(color=(10, 20, 30)))
        child = _put(
            store,
            _png(color=(40, 50, 60)),
            purpose="output",
            parent_blob_ids=(parent.blob_id,),
            deduplicate=False,
        )
        with pytest.raises(BlobAuthorizationError) as owner:
            store.delete(child.blob_id, owner_scope="session-b")
        assert owner.value.code == "blob_owner_mismatch"
        with pytest.raises(BlobConflict) as referenced:
            store.delete(parent.blob_id, owner_scope="session-a")
        assert referenced.value.code == "blob_referenced"
        assert store.delete(child.blob_id, owner_scope="session-a") is True
        assert store.delete(parent.blob_id, owner_scope="session-a") is True
        assert store.snapshot()["objects"] == 0
    finally:
        store.close()


def test_capacity_eviction_preserves_parents_of_the_incoming_blob(tmp_path):
    parent_data = _png(color=(1, 2, 3))
    unrelated_data = _png(color=(4, 5, 6))
    child_data = _png(color=(7, 8, 9))
    total_limit = len(parent_data) + len(unrelated_data) + len(child_data) - 1
    store = PersistentImageBlobStore(
        tmp_path / "blobs",
        max_blob_bytes=max(len(parent_data), len(unrelated_data), len(child_data)),
        max_total_bytes=total_limit,
    )
    try:
        parent = _put(store, parent_data)
        unrelated = _put(store, unrelated_data)
        child = _put(
            store,
            child_data,
            purpose="output",
            parent_blob_ids=(parent.blob_id,),
            deduplicate=False,
        )

        assert store.descriptor(parent.blob_id).blob_id == parent.blob_id
        assert store.descriptor(child.blob_id).blob_id == child.blob_id
        with pytest.raises(BlobNotFound):
            store.descriptor(unrelated.blob_id)
    finally:
        store.close()


def test_expiry_cleanup_reaches_parent_child_fixed_point(tmp_path):
    now = [100.0]
    store = PersistentImageBlobStore(
        tmp_path / "blobs",
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    try:
        parent = _put(store, _png(color=(10, 11, 12)))
        _put(
            store,
            _png(color=(13, 14, 15)),
            purpose="output",
            parent_blob_ids=(parent.blob_id,),
            deduplicate=False,
        )
        now[0] = 106.0

        assert store.cleanup()["blobs_removed"] == 2
        assert store.snapshot()["objects"] == 0
    finally:
        store.close()


def test_object_file_is_not_deleted_before_metadata_transaction_commits(tmp_path):
    store = PersistentImageBlobStore(tmp_path / "blobs")
    data = _png(color=(31, 41, 59))
    try:
        descriptor = _put(store, data)
        lease = store.acquire_lease(
            descriptor.blob_id,
            attempt_id="att_rollback_12345678",
        )
        store.release_lease(lease.lease_id)

        with store._lock:
            store._begin()
            store._delete_blob_locked(descriptor.blob_id)
            store._rollback()

        lease = store.acquire_lease(
            descriptor.blob_id,
            attempt_id="att_after_rollback_12345678",
        )
        assert store.read_all(
            descriptor.blob_id,
            lease_id=lease.lease_id,
            attempt_id="att_after_rollback_12345678",
        ) == data
    finally:
        store.close()


def test_failed_blob_metadata_commit_keeps_upload_retryable(tmp_path, monkeypatch):
    store = PersistentImageBlobStore(tmp_path / "blobs")
    first_data = _png(color=(21, 34, 55))
    retry_data = _png(color=(89, 144, 233))
    try:
        first = _put(store, first_data)
        session = store.begin_upload(
            expected_sha256=hashlib.sha256(retry_data).hexdigest(),
            expected_size=len(retry_data),
            content_type="image/png",
            purpose="input_image",
            owner_scope="session-a",
            width=16,
            height=12,
        )
        store.write_upload(session.upload_id, offset=0, data=retry_data)
        monkeypatch.setattr(
            distributed_module.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex=first.blob_id.removeprefix("img_")),
        )

        with pytest.raises(sqlite3.IntegrityError):
            store.commit_upload(session.upload_id)

        monkeypatch.undo()
        retried = store.commit_upload(session.upload_id)
        lease = store.acquire_lease(
            retried.blob_id,
            attempt_id="att_commit_retry_12345678",
        )
        assert store.read_all(
            retried.blob_id,
            lease_id=lease.lease_id,
            attempt_id="att_commit_retry_12345678",
        ) == retry_data
    finally:
        store.close()


def test_expired_partial_upload_is_removed(tmp_path):
    now = [100.0]
    store = PersistentImageBlobStore(
        tmp_path / "blobs",
        upload_ttl_seconds=5,
        clock=lambda: now[0],
    )
    data = _png()
    try:
        session = store.begin_upload(
            expected_sha256=hashlib.sha256(data).hexdigest(),
            expected_size=len(data),
            content_type="image/png",
            purpose="input_image",
            owner_scope="session-a",
            width=16,
            height=12,
        )
        store.write_upload(session.upload_id, offset=0, data=data[:10])
        now[0] = 106.0
        assert store.cleanup()["uploads_removed"] == 1
        with pytest.raises(BlobNotFound):
            store.write_upload(session.upload_id, offset=10, data=data[10:20])
    finally:
        store.close()


def test_transfer_grant_is_attempt_direction_and_expiry_scoped():
    now = [100.0]
    signer = BlobTransferTokenSigner("x" * 32, clock=lambda: now[0])
    token = signer.issue(
        blob_id="img_1234567890abcdef",
        lease_id="bls_1234567890abcdef",
        attempt_id="att_test_12345678",
        direction="download",
        ttl_seconds=5,
    )

    payload = signer.verify(
        token,
        direction="download",
        attempt_id="att_test_12345678",
    )
    assert payload["blob_id"] == "img_1234567890abcdef"
    with pytest.raises(BlobAuthorizationError) as wrong_direction:
        signer.verify(
            token,
            direction="upload",
            attempt_id="att_test_12345678",
        )
    assert wrong_direction.value.code == "transfer_scope_mismatch"
    with pytest.raises(BlobAuthorizationError) as tampered:
        signer.verify(
            token[:-1] + ("A" if token[-1] != "A" else "B"),
            direction="download",
            attempt_id="att_test_12345678",
        )
    assert tampered.value.code == "invalid_transfer_signature"
    now[0] = 106.0
    with pytest.raises(BlobAuthorizationError) as expired:
        signer.verify(
            token,
            direction="download",
            attempt_id="att_test_12345678",
        )
    assert expired.value.code == "transfer_grant_expired"


def test_upload_transfer_grant_binds_an_upload_session_not_a_blob_lease():
    signer = BlobTransferTokenSigner("x" * 32, clock=lambda: 100.0)
    token = signer.issue(
        upload_id="upl_1234567890abcdef",
        attempt_id="att_upload_12345678",
        direction="upload",
        ttl_seconds=5,
    )

    payload = signer.verify(
        token,
        direction="upload",
        attempt_id="att_upload_12345678",
    )
    assert payload["upload_id"] == "upl_1234567890abcdef"
    assert payload["blob_id"] is None
    assert payload["lease_id"] is None

    with pytest.raises(BlobValidationError) as mixed_scope:
        signer.issue(
            upload_id="upl_1234567890abcdef",
            blob_id="img_1234567890abcdef",
            lease_id="bls_1234567890abcdef",
            attempt_id="att_upload_12345678",
            direction="upload",
            ttl_seconds=5,
        )
    assert mixed_scope.value.code == "invalid_transfer_identity"


def test_artifact_manifest_is_ordered_and_content_addressed():
    manifest = DiffusionArtifactManifest.build(
        artifact_id="sd15_instruction_bundle",
        pipeline_kind="sd15_instruction_pipeline",
        revision="revision_20260807",
        components=(
            DiffusionArtifactComponent(
                artifact_id="instruction",
                artifact_kind="sd15_instruction_pipeline",
                sha256="b" * 64,
            ),
            DiffusionArtifactComponent(
                artifact_id="base",
                artifact_kind="sd15_pipeline",
                sha256="a" * 64,
            ),
        ),
    )
    snapshot = manifest.snapshot()

    assert [item["artifact_id"] for item in snapshot["components"]] == [
        "base",
        "instruction",
    ]
    body = {key: value for key, value in snapshot.items() if key != "sha256"}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert snapshot["sha256"] == hashlib.sha256(encoded).hexdigest()
