"""Immutable local payload references for TaskGraph optimization candidates.

This module is deliberately not wired into ``TaskGraphCoordinator``.  It binds
one shadow optimizer fan-out plan to a content digest and data scope, stores the
payload below an explicit local root, and materializes short-lived local files
for authorized consumers.  Public references never contain payload bodies or
filesystem paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PAYLOAD_REF_SCHEMA_VERSION = "qlh.task_graph_payload_ref.v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]{0,63}/[a-z0-9][a-z0-9.+-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_REF_KEYS = frozenset({
    "schema_version",
    "payload_id",
    "content_sha256",
    "size_bytes",
    "media_type",
    "data_scope",
    "source_stage_id",
    "consumer_stage_ids",
    "reference_sha256",
})
_PLAN_KEYS = frozenset({
    "payload_ref",
    "source_stage_id",
    "target_stage_ids",
    "reason_code",
})


class TaskPayloadError(RuntimeError):
    """Base error for the local shadow payload store."""


class TaskPayloadValidationError(TaskPayloadError):
    """Raised when a reference, plan, or payload fails validation."""


class TaskPayloadAuthorizationError(TaskPayloadError):
    """Raised when scope or consumer identity does not match the reference."""


class TaskPayloadNotFound(TaskPayloadError):
    """Raised when a referenced local payload no longer exists."""


class TaskPayloadConflict(TaskPayloadError):
    """Raised when immutable stored content does not match its reference."""


class TaskPayloadInUse(TaskPayloadError):
    """Raised when release is attempted while a local materialization is live."""


def validate_payload_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach a public payload reference."""

    if not isinstance(reference, Mapping) or set(reference) != _PAYLOAD_REF_KEYS:
        raise TaskPayloadValidationError("payload reference fields are invalid")
    if reference.get("schema_version") != PAYLOAD_REF_SCHEMA_VERSION:
        raise TaskPayloadValidationError("unsupported payload reference schema")
    content_sha256 = reference.get("content_sha256")
    reference_sha256 = reference.get("reference_sha256")
    if not isinstance(content_sha256, str) or not _SHA256_RE.fullmatch(content_sha256):
        raise TaskPayloadValidationError("content_sha256 is invalid")
    if not isinstance(reference_sha256, str) or not _SHA256_RE.fullmatch(
        reference_sha256,
    ):
        raise TaskPayloadValidationError("reference_sha256 is invalid")
    if reference.get("payload_id") != f"payload:{reference_sha256}":
        raise TaskPayloadValidationError("payload_id does not match reference digest")
    size_bytes = reference.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise TaskPayloadValidationError("size_bytes must be a positive integer")
    media_type = reference.get("media_type")
    if not isinstance(media_type, str) or not _MEDIA_TYPE_RE.fullmatch(media_type):
        raise TaskPayloadValidationError("media_type is invalid")
    data_scope = _identifier(reference.get("data_scope"), "data_scope")
    source_stage_id = _identifier(
        reference.get("source_stage_id"), "source_stage_id",
    )
    consumer_stage_ids = _identifiers(
        reference.get("consumer_stage_ids"), "consumer_stage_ids",
    )
    if not consumer_stage_ids:
        raise TaskPayloadValidationError("consumer_stage_ids cannot be empty")
    if list(consumer_stage_ids) != sorted(set(consumer_stage_ids)):
        raise TaskPayloadValidationError(
            "consumer_stage_ids must be sorted and unique",
        )
    base = {
        "schema_version": PAYLOAD_REF_SCHEMA_VERSION,
        "content_sha256": content_sha256,
        "size_bytes": size_bytes,
        "media_type": media_type,
        "data_scope": data_scope,
        "source_stage_id": source_stage_id,
        "consumer_stage_ids": list(consumer_stage_ids),
    }
    if _digest(base) != reference_sha256:
        raise TaskPayloadValidationError(
            "payload reference digest does not match content contract",
        )
    return json.loads(json.dumps(reference, ensure_ascii=True, sort_keys=True))


def bind_payload_plan(
    store: "TaskPayloadStore",
    plan: Mapping[str, Any],
    payload: bytes | bytearray | memoryview,
    *,
    data_scope: str,
    media_type: str = "application/octet-stream",
) -> dict[str, Any]:
    """Bind one G1/G2 fan-out plan entry to immutable local content."""

    if not isinstance(store, TaskPayloadStore):
        raise TaskPayloadValidationError("store must be a TaskPayloadStore")
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_KEYS:
        raise TaskPayloadValidationError("payload plan entry fields are invalid")
    source_stage_id = _identifier(plan.get("source_stage_id"), "source_stage_id")
    target_stage_ids = _identifiers(
        plan.get("target_stage_ids"), "target_stage_ids",
    )
    if len(target_stage_ids) < 2 or list(target_stage_ids) != sorted(
        set(target_stage_ids),
    ):
        raise TaskPayloadValidationError(
            "payload plan requires at least two sorted unique consumers",
        )
    if plan.get("payload_ref") != f"payload:{source_stage_id}":
        raise TaskPayloadValidationError("payload plan slot does not match source")
    if plan.get("reason_code") != "immutable_fanout_source":
        raise TaskPayloadValidationError("payload plan is not immutable fan-out")
    return store.put(
        payload,
        data_scope=data_scope,
        source_stage_id=source_stage_id,
        consumer_stage_ids=target_stage_ids,
        media_type=media_type,
    )


class TaskPayloadStore:
    """Small content-addressed store for shadow TaskGraph payload contracts."""

    def __init__(self, root: str | os.PathLike[str], *, max_payload_bytes: int = 16 << 20):
        if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int):
            raise TaskPayloadValidationError("max_payload_bytes must be an integer")
        if max_payload_bytes <= 0:
            raise TaskPayloadValidationError("max_payload_bytes must be positive")
        self.root = Path(root).resolve()
        self.objects_dir = self.root / "objects"
        self.manifests_dir = self.root / "manifests"
        self.materialized_dir = self.root / "materialized"
        self.max_payload_bytes = max_payload_bytes
        self._lock = threading.RLock()
        self._active: dict[str, int] = {}
        for directory in (
            self.objects_dir, self.manifests_dir, self.materialized_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                raise TaskPayloadValidationError(
                    "payload store directory cannot be a symlink",
                )
        self._cleanup_stale_materializations()

    def put(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        data_scope: str,
        source_stage_id: str,
        consumer_stage_ids: Sequence[str],
        media_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TaskPayloadValidationError("payload must be bytes-like")
        body = bytes(payload)
        if not body:
            raise TaskPayloadValidationError("payload cannot be empty")
        if len(body) > self.max_payload_bytes:
            raise TaskPayloadValidationError("payload exceeds configured size limit")
        scope = _identifier(data_scope, "data_scope")
        source = _identifier(source_stage_id, "source_stage_id")
        consumers = _identifiers(consumer_stage_ids, "consumer_stage_ids")
        if not consumers or list(consumers) != sorted(set(consumers)):
            raise TaskPayloadValidationError(
                "consumer_stage_ids must be sorted and unique",
            )
        if not isinstance(media_type, str) or not _MEDIA_TYPE_RE.fullmatch(media_type):
            raise TaskPayloadValidationError("media_type is invalid")
        content_sha256 = hashlib.sha256(body).hexdigest()
        base = {
            "schema_version": PAYLOAD_REF_SCHEMA_VERSION,
            "content_sha256": content_sha256,
            "size_bytes": len(body),
            "media_type": media_type,
            "data_scope": scope,
            "source_stage_id": source,
            "consumer_stage_ids": list(consumers),
        }
        reference_sha256 = _digest(base)
        reference = {
            **base,
            "payload_id": f"payload:{reference_sha256}",
            "reference_sha256": reference_sha256,
        }
        checked = validate_payload_reference(reference)
        object_path = self._object_path(reference_sha256)
        manifest_path = self._manifest_path(reference_sha256)
        with self._lock:
            if object_path.exists() or manifest_path.exists():
                self._verify_stored(checked)
                return checked
            self._atomic_write(object_path, body)
            try:
                self._atomic_write(
                    manifest_path,
                    json.dumps(
                        checked,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8"),
                )
            except Exception:
                object_path.unlink(missing_ok=True)
                raise
        return checked

    @contextmanager
    def materialize(
        self,
        reference: Mapping[str, Any],
        *,
        consumer_stage_id: str,
        data_scope: str,
    ) -> Iterator[Path]:
        checked = validate_payload_reference(reference)
        consumer = _identifier(consumer_stage_id, "consumer_stage_id")
        scope = _identifier(data_scope, "data_scope")
        self._authorize(checked, consumer=consumer, scope=scope)
        reference_sha256 = checked["reference_sha256"]
        target = self.materialized_dir / (
            f"{reference_sha256}-{uuid.uuid4().hex}.payload"
        )
        with self._lock:
            self._verify_stored(checked)
            source = self._object_path(reference_sha256)
            try:
                shutil.copyfile(source, target)
                self._verify_file(target, checked)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            self._active[reference_sha256] = self._active.get(reference_sha256, 0) + 1
        try:
            yield target
        finally:
            with self._lock:
                target.unlink(missing_ok=True)
                remaining = self._active.get(reference_sha256, 1) - 1
                if remaining > 0:
                    self._active[reference_sha256] = remaining
                else:
                    self._active.pop(reference_sha256, None)

    def release(self, reference: Mapping[str, Any]) -> None:
        checked = validate_payload_reference(reference)
        reference_sha256 = checked["reference_sha256"]
        with self._lock:
            if self._active.get(reference_sha256, 0):
                raise TaskPayloadInUse("payload has active materializations")
            object_path = self._object_path(reference_sha256)
            manifest_path = self._manifest_path(reference_sha256)
            if not object_path.exists() and not manifest_path.exists():
                raise TaskPayloadNotFound("payload reference is not stored")
            object_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            objects = [
                path for path in self.objects_dir.iterdir()
                if path.is_file() and not path.is_symlink()
            ]
            return {
                "schema_version": PAYLOAD_REF_SCHEMA_VERSION,
                "object_count": len(objects),
                "manifest_count": sum(
                    1 for path in self.manifests_dir.iterdir()
                    if path.is_file() and not path.is_symlink()
                ),
                "active_materialization_count": sum(self._active.values()),
                "stored_bytes": sum(path.stat().st_size for path in objects),
            }

    def _authorize(
        self,
        reference: Mapping[str, Any],
        *,
        consumer: str,
        scope: str,
    ) -> None:
        if reference["data_scope"] != scope:
            raise TaskPayloadAuthorizationError("payload data scope does not match")
        if consumer not in reference["consumer_stage_ids"]:
            raise TaskPayloadAuthorizationError("stage is not a payload consumer")

    def _verify_stored(self, reference: Mapping[str, Any]) -> None:
        object_path = self._object_path(reference["reference_sha256"])
        manifest_path = self._manifest_path(reference["reference_sha256"])
        if not object_path.is_file() or not manifest_path.is_file():
            raise TaskPayloadNotFound("payload object or manifest is missing")
        if object_path.is_symlink() or manifest_path.is_symlink():
            raise TaskPayloadConflict("payload store entry cannot be a symlink")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TaskPayloadConflict("payload manifest is corrupt") from exc
        try:
            stored_reference = validate_payload_reference(manifest)
        except TaskPayloadValidationError as exc:
            raise TaskPayloadConflict("payload manifest is invalid") from exc
        if stored_reference != reference:
            raise TaskPayloadConflict("payload manifest does not match reference")
        self._verify_file(object_path, reference)

    @staticmethod
    def _verify_file(path: Path, reference: Mapping[str, Any]) -> None:
        if path.stat().st_size != reference["size_bytes"]:
            raise TaskPayloadConflict("payload size does not match reference")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != reference["content_sha256"]:
            raise TaskPayloadConflict("payload content digest does not match reference")

    def _object_path(self, reference_sha256: str) -> Path:
        if not _SHA256_RE.fullmatch(reference_sha256):
            raise TaskPayloadValidationError("reference digest is invalid")
        return self.objects_dir / f"{reference_sha256}.payload"

    def _manifest_path(self, reference_sha256: str) -> Path:
        if not _SHA256_RE.fullmatch(reference_sha256):
            raise TaskPayloadValidationError("reference digest is invalid")
        return self.manifests_dir / f"{reference_sha256}.json"

    @staticmethod
    def _atomic_write(path: Path, body: bytes) -> None:
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=path.parent, prefix=".payload-",
            ) as handle:
                temporary_name = handle.name
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)

    def _cleanup_stale_materializations(self) -> None:
        for path in self.materialized_dir.iterdir():
            if path.is_file() and path.name.endswith(".payload"):
                path.unlink(missing_ok=True)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskPayloadValidationError(f"{field_name} must be a safe identifier")
    return value


def _identifiers(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TaskPayloadValidationError(f"{field_name} must be a sequence")
    return tuple(_identifier(item, field_name) for item in value)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PAYLOAD_REF_SCHEMA_VERSION",
    "TaskPayloadAuthorizationError",
    "TaskPayloadConflict",
    "TaskPayloadError",
    "TaskPayloadInUse",
    "TaskPayloadNotFound",
    "TaskPayloadStore",
    "TaskPayloadValidationError",
    "bind_payload_plan",
    "validate_payload_reference",
]
