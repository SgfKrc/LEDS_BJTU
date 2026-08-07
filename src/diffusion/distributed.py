"""Persistent image blobs and transfer grants for distributed diffusion.

The task-worker control plane carries descriptors only. Blob bytes live in a
content-addressed data directory and are transferred under attempt-scoped
leases. This module deliberately contains no HTTP server so it can be tested
before the RemoteDiffusionProvider and data-plane routes are enabled.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


DEFAULT_MAX_BLOB_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_PIXELS = 2048 * 2048
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
DEFAULT_TTL_SECONDS = 30 * 60
DEFAULT_UPLOAD_TTL_SECONDS = 10 * 60
DEFAULT_LEASE_TTL_SECONDS = 2 * 60
MAX_TRANSFER_GRANT_SECONDS = 5 * 60

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_BLOB_ID = re.compile(r"^img_[A-Za-z0-9_-]{16,96}$")
_UPLOAD_ID = re.compile(r"^upl_[A-Za-z0-9_-]{16,96}$")
_LEASE_ID = re.compile(r"^bls_[A-Za-z0-9_-]{16,96}$")
_CONTENT_TYPES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}
_PIPELINE_KINDS = {
    "sd15_pipeline",
    "sd15_inpaint_pipeline",
    "sd15_instruction_pipeline",
    "sd15_ip_adapter",
}


class DistributedBlobError(RuntimeError):
    """Base data-plane failure with a stable machine-readable code."""

    def __init__(self, message: str, *, code: str):
        self.code = code
        super().__init__(message)


class BlobNotFound(DistributedBlobError):
    pass


class BlobConflict(DistributedBlobError):
    pass


class BlobValidationError(DistributedBlobError):
    pass


class BlobAuthorizationError(DistributedBlobError):
    pass


@dataclass(frozen=True)
class PersistentBlobDescriptor:
    blob_id: str
    sha256: str
    size_bytes: int
    content_type: str
    width: int
    height: int
    purpose: str
    created_at: float
    expires_at: float
    lease_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "width": self.width,
            "height": self.height,
            "purpose": self.purpose,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "lease_count": self.lease_count,
        }


@dataclass(frozen=True)
class BlobUploadSession:
    upload_id: str
    expected_sha256: str
    expected_size: int
    received_bytes: int
    expires_at: float

    def snapshot(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "expected_sha256": self.expected_sha256,
            "expected_size": self.expected_size,
            "received_bytes": self.received_bytes,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class BlobLease:
    lease_id: str
    blob_id: str
    attempt_id: str
    expires_at: float

    def snapshot(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "blob_id": self.blob_id,
            "attempt_id": self.attempt_id,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class DiffusionArtifactComponent:
    artifact_id: str
    artifact_kind: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            _SAFE_ID.fullmatch(self.artifact_id) is None
            or _SAFE_ID.fullmatch(self.artifact_kind) is None
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("diffusion artifact component identity is invalid")

    def snapshot(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DiffusionArtifactManifest:
    artifact_id: str
    pipeline_kind: str
    revision: str
    sha256: str
    components: tuple[DiffusionArtifactComponent, ...]

    @classmethod
    def build(
        cls,
        *,
        artifact_id: str,
        pipeline_kind: str,
        revision: str,
        components: tuple[DiffusionArtifactComponent, ...],
    ) -> "DiffusionArtifactManifest":
        if _SAFE_ID.fullmatch(artifact_id) is None:
            raise ValueError("diffusion manifest artifact_id is invalid")
        if pipeline_kind not in _PIPELINE_KINDS:
            raise ValueError("diffusion manifest pipeline_kind is unsupported")
        if _SAFE_ID.fullmatch(revision) is None:
            raise ValueError("diffusion manifest revision is invalid")
        ordered = tuple(sorted(components, key=lambda item: item.artifact_id))
        if not ordered or len(ordered) > 16:
            raise ValueError("diffusion manifest components must be a non-empty bounded tuple")
        if len({item.artifact_id for item in ordered}) != len(ordered):
            raise ValueError("diffusion manifest component IDs must be unique")
        body = {
            "artifact_id": artifact_id,
            "pipeline_kind": pipeline_kind,
            "revision": revision,
            "components": [item.snapshot() for item in ordered],
        }
        digest = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            artifact_id=artifact_id,
            pipeline_kind=pipeline_kind,
            revision=revision,
            sha256=digest,
            components=ordered,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "pipeline_kind": self.pipeline_kind,
            "revision": self.revision,
            "sha256": self.sha256,
            "components": [item.snapshot() for item in self.components],
        }


class PersistentImageBlobStore:
    """Cross-process blob metadata with content-addressed immutable objects."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        upload_ttl_seconds: float = DEFAULT_UPLOAD_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if min(max_blob_bytes, max_pixels, max_total_bytes) <= 0:
            raise ValueError("blob limits must be positive")
        if max_blob_bytes > max_total_bytes:
            raise ValueError("max_blob_bytes must not exceed max_total_bytes")
        if ttl_seconds <= 0 or upload_ttl_seconds <= 0:
            raise ValueError("blob TTL values must be positive")
        self.root = Path(root).expanduser().resolve()
        self.objects_dir = self.root / "objects"
        self.uploads_dir = self.root / "uploads"
        self.db_path = self.root / "blobs.sqlite3"
        self.max_blob_bytes = int(max_blob_bytes)
        self.max_pixels = int(max_pixels)
        self.max_total_bytes = int(max_total_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self.upload_ttl_seconds = float(upload_ttl_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()
        self.cleanup()

    def _initialize_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS objects (
                sha256 TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                relative_path TEXT NOT NULL,
                ref_count INTEGER NOT NULL CHECK(ref_count >= 0)
            );
            CREATE TABLE IF NOT EXISTS object_gc (
                sha256 TEXT PRIMARY KEY,
                relative_path TEXT NOT NULL,
                queued_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blobs (
                blob_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL REFERENCES objects(sha256),
                content_type TEXT NOT NULL,
                purpose TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS blobs_expiry_idx ON blobs(expires_at);
            CREATE INDEX IF NOT EXISTS blobs_dedupe_idx
                ON blobs(sha256, purpose, owner_scope, width, height);
            CREATE TABLE IF NOT EXISTS blob_parents (
                blob_id TEXT NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
                parent_blob_id TEXT NOT NULL REFERENCES blobs(blob_id) ON DELETE RESTRICT,
                PRIMARY KEY(blob_id, parent_blob_id)
            );
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                expected_sha256 TEXT NOT NULL,
                expected_size INTEGER NOT NULL,
                received_bytes INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                purpose TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                temp_name TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL,
                deduplicate INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS uploads_expiry_idx ON uploads(expires_at);
            CREATE TABLE IF NOT EXISTS completed_uploads (
                upload_id TEXT PRIMARY KEY,
                blob_id TEXT NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
                completed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blob_leases (
                lease_id TEXT PRIMARY KEY,
                blob_id TEXT NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
                attempt_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS blob_leases_blob_idx ON blob_leases(blob_id);
            CREATE INDEX IF NOT EXISTS blob_leases_expiry_idx ON blob_leases(expires_at);
            """
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise BlobConflict("blob store is closed", code="store_closed")

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise BlobValidationError(
                "blob metadata must contain strict JSON",
                code="invalid_metadata",
            ) from exc

    @staticmethod
    def _safe_value(value: str, field: str) -> str:
        normalized = str(value or "").strip()
        if _SAFE_ID.fullmatch(normalized) is None:
            raise BlobValidationError(
                f"{field} is invalid",
                code=f"invalid_{field}",
            )
        return normalized

    def _object_path(self, sha256: str) -> Path:
        return self.objects_dir / sha256[:2] / sha256

    def _begin(self) -> None:
        self._db.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._db.in_transaction:
            self._db.execute("ROLLBACK")

    def _purge_expired_locked(self, now: float) -> None:
        self._db.execute("DELETE FROM blob_leases WHERE expires_at <= ?", (now,))
        expired_uploads = list(
            self._db.execute(
                "SELECT upload_id, temp_name FROM uploads WHERE expires_at <= ?",
                (now,),
            )
        )
        for row in expired_uploads:
            (self.uploads_dir / row["temp_name"]).unlink(missing_ok=True)
        self._db.execute("DELETE FROM uploads WHERE expires_at <= ?", (now,))
        while True:
            expired_blobs = list(
                self._db.execute(
                    """
                    SELECT b.blob_id
                    FROM blobs b
                    WHERE b.expires_at <= ?
                      AND NOT EXISTS (
                        SELECT 1 FROM blob_leases l WHERE l.blob_id = b.blob_id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM blob_parents p WHERE p.parent_blob_id = b.blob_id
                      )
                    """,
                    (now,),
                )
            )
            if not expired_blobs:
                break
            for row in expired_blobs:
                self._delete_blob_locked(row["blob_id"])

    def _delete_blob_locked(self, blob_id: str) -> None:
        row = self._db.execute(
            "SELECT sha256 FROM blobs WHERE blob_id = ?",
            (blob_id,),
        ).fetchone()
        if row is None:
            return
        sha256 = row["sha256"]
        self._db.execute("DELETE FROM blobs WHERE blob_id = ?", (blob_id,))
        self._db.execute(
            "UPDATE objects SET ref_count = ref_count - 1 WHERE sha256 = ?",
            (sha256,),
        )
        object_row = self._db.execute(
            "SELECT relative_path, ref_count FROM objects WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if object_row is not None and object_row["ref_count"] == 0:
            self._db.execute(
                """
                INSERT INTO object_gc(sha256, relative_path, queued_at)
                VALUES (?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    relative_path = excluded.relative_path,
                    queued_at = excluded.queued_at
                """,
                (sha256, object_row["relative_path"], self._clock()),
            )
            self._db.execute("DELETE FROM objects WHERE sha256 = ?", (sha256,))

    def _drain_object_gc(self) -> None:
        """Delete unreferenced object files only after their metadata transaction commits."""
        with self._lock:
            self._ensure_open()
            while True:
                try:
                    self._begin()
                    row = self._db.execute(
                        """
                        SELECT g.sha256, g.relative_path
                        FROM object_gc g
                        LEFT JOIN objects o ON o.sha256 = g.sha256
                        WHERE o.sha256 IS NULL
                        ORDER BY g.queued_at ASC
                        LIMIT 1
                        """
                    ).fetchone()
                    if row is None:
                        self._db.execute("COMMIT")
                        return
                    path = (self.root / row["relative_path"]).resolve()
                    try:
                        path.relative_to(self.objects_dir.resolve())
                    except ValueError as exc:
                        raise BlobAuthorizationError(
                            "blob GC path escaped the data directory",
                            code="invalid_object_path",
                        ) from exc
                    path.unlink(missing_ok=True)
                    self._db.execute(
                        "DELETE FROM object_gc WHERE sha256 = ?",
                        (row["sha256"],),
                    )
                    self._db.execute("COMMIT")
                except OSError:
                    # A transient Windows file lock must not turn a committed
                    # logical delete into a reported failure. A later cleanup
                    # retries the durable queue entry.
                    self._rollback()
                    return
                except Exception:
                    self._rollback()
                    raise

    def _enforce_capacity_locked(
        self,
        incoming_size: int,
        now: float,
        *,
        exclude_blob_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._purge_expired_locked(now)
        current = int(
            self._db.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM objects"
            ).fetchone()["total"]
        )
        if current + incoming_size <= self.max_total_bytes:
            return
        candidates = list(
            self._db.execute(
                """
                SELECT b.blob_id
                FROM blobs b
                WHERE NOT EXISTS (
                    SELECT 1 FROM blob_leases l WHERE l.blob_id = b.blob_id
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM blob_parents p WHERE p.parent_blob_id = b.blob_id
                  )
                ORDER BY b.created_at ASC, b.blob_id ASC
                """
            )
        )
        for row in candidates:
            if row["blob_id"] in exclude_blob_ids:
                continue
            self._delete_blob_locked(row["blob_id"])
            current = int(
                self._db.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM objects"
                ).fetchone()["total"]
            )
            if current + incoming_size <= self.max_total_bytes:
                return
        raise BlobConflict(
            "blob store capacity is held by active leases or references",
            code="blob_store_full",
        )

    def begin_upload(
        self,
        *,
        expected_sha256: str,
        expected_size: int,
        content_type: str,
        purpose: str,
        owner_scope: str,
        width: int,
        height: int,
        metadata: Optional[dict[str, Any]] = None,
        parent_blob_ids: tuple[str, ...] = (),
        deduplicate: bool = True,
    ) -> BlobUploadSession:
        sha256 = str(expected_sha256 or "").lower()
        if _SHA256.fullmatch(sha256) is None:
            raise BlobValidationError(
                "expected_sha256 is invalid",
                code="invalid_sha256",
            )
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 < expected_size <= self.max_blob_bytes
        ):
            raise BlobValidationError(
                "expected_size exceeds the blob limit",
                code="invalid_size",
            )
        normalized_type = str(content_type or "").lower()
        if normalized_type not in _CONTENT_TYPES:
            raise BlobValidationError(
                "content_type is not an allowed image type",
                code="invalid_content_type",
            )
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or width * height > self.max_pixels
        ):
            raise BlobValidationError(
                "image dimensions exceed the pixel limit",
                code="invalid_dimensions",
            )
        normalized_purpose = self._safe_value(purpose, "purpose")
        normalized_owner = self._safe_value(owner_scope, "owner_scope")
        normalized_parents = tuple(dict.fromkeys(parent_blob_ids))
        if any(_BLOB_ID.fullmatch(value) is None for value in normalized_parents):
            raise BlobValidationError(
                "parent_blob_ids contains an invalid identifier",
                code="invalid_parent_blob_id",
            )
        if normalized_parents and deduplicate:
            raise BlobValidationError(
                "derived blobs with parents must disable descriptor deduplication",
                code="invalid_deduplication_mode",
            )
        now = self._clock()
        upload_id = f"upl_{uuid.uuid4().hex}"
        temp_name = f"{upload_id}.part"
        temp_path = self.uploads_dir / temp_name
        with self._lock:
            self._ensure_open()
            temp_path.touch(exist_ok=False)
            try:
                self._begin()
                self._purge_expired_locked(now)
                for parent_id in normalized_parents:
                    if self._db.execute(
                        "SELECT 1 FROM blobs WHERE blob_id = ?",
                        (parent_id,),
                    ).fetchone() is None:
                        raise BlobNotFound(
                            f"parent blob not found: {parent_id}",
                            code="parent_blob_not_found",
                        )
                self._db.execute(
                    """
                    INSERT INTO uploads(
                        upload_id, expected_sha256, expected_size, received_bytes,
                        content_type, purpose, owner_scope, width, height,
                        created_at, expires_at, temp_name, metadata_json,
                        parent_ids_json, deduplicate
                    ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        upload_id,
                        sha256,
                        expected_size,
                        normalized_type,
                        normalized_purpose,
                        normalized_owner,
                        width,
                        height,
                        now,
                        now + self.upload_ttl_seconds,
                        temp_name,
                        self._json(metadata or {}),
                        self._json(list(normalized_parents)),
                        1 if deduplicate else 0,
                    ),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                temp_path.unlink(missing_ok=True)
                raise
        return BlobUploadSession(
            upload_id=upload_id,
            expected_sha256=sha256,
            expected_size=expected_size,
            received_bytes=0,
            expires_at=now + self.upload_ttl_seconds,
        )

    def write_upload(self, upload_id: str, *, offset: int, data: bytes) -> BlobUploadSession:
        if _UPLOAD_ID.fullmatch(str(upload_id or "")) is None:
            raise BlobValidationError("upload_id is invalid", code="invalid_upload_id")
        chunk = bytes(data)
        if not chunk:
            raise BlobValidationError("upload chunk is empty", code="empty_chunk")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise BlobValidationError("upload offset is invalid", code="invalid_offset")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                row = self._db.execute(
                    "SELECT * FROM uploads WHERE upload_id = ?",
                    (upload_id,),
                ).fetchone()
                if row is None:
                    raise BlobNotFound("upload session not found", code="upload_not_found")
                received_bytes = row["received_bytes"]
                temp_path = self.uploads_dir / row["temp_name"]
                if (
                    not temp_path.is_file()
                    or temp_path.stat().st_size != received_bytes
                ):
                    raise BlobConflict(
                        "upload temp file does not match the committed prefix",
                        code="upload_state_mismatch",
                    )
                if offset < received_bytes:
                    if offset + len(chunk) > received_bytes:
                        raise BlobConflict(
                            "replayed upload chunk overlaps uncommitted bytes",
                            code="upload_overlap_mismatch",
                        )
                    with temp_path.open("rb") as handle:
                        handle.seek(offset)
                        existing = handle.read(len(chunk))
                    if not hmac.compare_digest(existing, chunk):
                        raise BlobConflict(
                            "replayed upload chunk differs from committed bytes",
                            code="upload_replay_mismatch",
                        )
                    self._db.execute("COMMIT")
                    return BlobUploadSession(
                        upload_id=upload_id,
                        expected_sha256=row["expected_sha256"],
                        expected_size=row["expected_size"],
                        received_bytes=received_bytes,
                        expires_at=row["expires_at"],
                    )
                if offset > received_bytes:
                    raise BlobConflict(
                        "upload offset does not match the committed prefix",
                        code="upload_offset_mismatch",
                    )
                received = offset + len(chunk)
                if received > row["expected_size"]:
                    raise BlobValidationError(
                        "upload exceeds the declared size",
                        code="upload_too_large",
                    )
                with temp_path.open("ab") as handle:
                    handle.write(chunk)
                    handle.flush()
                self._db.execute(
                    "UPDATE uploads SET received_bytes = ? WHERE upload_id = ?",
                    (received, upload_id),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
        return BlobUploadSession(
            upload_id=upload_id,
            expected_sha256=row["expected_sha256"],
            expected_size=row["expected_size"],
            received_bytes=received,
            expires_at=row["expires_at"],
        )

    def upload_session(self, upload_id: str) -> BlobUploadSession:
        if _UPLOAD_ID.fullmatch(str(upload_id or "")) is None:
            raise BlobValidationError("upload_id is invalid", code="invalid_upload_id")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                row = self._db.execute(
                    "SELECT * FROM uploads WHERE upload_id = ?",
                    (upload_id,),
                ).fetchone()
                if row is None:
                    raise BlobNotFound(
                        "upload session not found",
                        code="upload_not_found",
                    )
                session = BlobUploadSession(
                    upload_id=upload_id,
                    expected_sha256=row["expected_sha256"],
                    expected_size=row["expected_size"],
                    received_bytes=row["received_bytes"],
                    expires_at=row["expires_at"],
                )
                self._db.execute("COMMIT")
                return session
            except Exception:
                self._rollback()
                raise

    def _validate_image_file(
        self,
        path: Path,
        *,
        content_type: str,
        width: int,
        height: int,
    ) -> None:
        try:
            from PIL import Image

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as image:
                    if (image.format or "").upper() != _CONTENT_TYPES[content_type]:
                        raise BlobValidationError(
                            "image magic bytes do not match content_type",
                            code="content_type_mismatch",
                        )
                    if getattr(image, "n_frames", 1) != 1:
                        raise BlobValidationError(
                            "animated or multi-page images are not accepted",
                            code="multi_frame_image",
                        )
                    if image.size != (width, height):
                        raise BlobValidationError(
                            "decoded dimensions do not match the descriptor",
                            code="dimension_mismatch",
                        )
                    if width * height > self.max_pixels:
                        raise BlobValidationError(
                            "decoded image exceeds the pixel limit",
                            code="pixel_limit_exceeded",
                        )
                    image.load()
        except BlobValidationError:
            raise
        except Exception as exc:
            raise BlobValidationError(
                "uploaded bytes are not a valid supported image",
                code="invalid_image",
            ) from exc

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _descriptor_locked(self, blob_id: str, now: float) -> PersistentBlobDescriptor:
        row = self._db.execute(
            """
            SELECT b.*, o.size_bytes,
                   (SELECT COUNT(*) FROM blob_leases l
                    WHERE l.blob_id = b.blob_id AND l.expires_at > ?) AS lease_count
            FROM blobs b
            JOIN objects o ON o.sha256 = b.sha256
            WHERE b.blob_id = ?
            """,
            (now, blob_id),
        ).fetchone()
        if row is None:
            raise BlobNotFound(f"image blob not found: {blob_id}", code="blob_not_found")
        return PersistentBlobDescriptor(
            blob_id=row["blob_id"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            content_type=row["content_type"],
            width=row["width"],
            height=row["height"],
            purpose=row["purpose"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            lease_count=row["lease_count"],
        )

    def _completed_upload_locked(
        self,
        upload_id: str,
        now: float,
    ) -> PersistentBlobDescriptor | None:
        row = self._db.execute(
            "SELECT blob_id FROM completed_uploads WHERE upload_id = ?",
            (upload_id,),
        ).fetchone()
        if row is None:
            return None
        return self._descriptor_locked(row["blob_id"], now)

    def commit_upload(self, upload_id: str) -> PersistentBlobDescriptor:
        if _UPLOAD_ID.fullmatch(str(upload_id or "")) is None:
            raise BlobValidationError("upload_id is invalid", code="invalid_upload_id")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            completed = self._completed_upload_locked(upload_id, now)
            if completed is not None:
                return completed
            row = self._db.execute(
                "SELECT * FROM uploads WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if row is None or row["expires_at"] <= now:
                completed = self._completed_upload_locked(upload_id, now)
                if completed is not None:
                    return completed
                self.cleanup()
                raise BlobNotFound("upload session not found", code="upload_not_found")
            if row["received_bytes"] != row["expected_size"]:
                raise BlobConflict(
                    "upload is incomplete",
                    code="upload_incomplete",
                )
            temp_path = self.uploads_dir / row["temp_name"]
            if not temp_path.is_file() or temp_path.stat().st_size != row["expected_size"]:
                completed = self._completed_upload_locked(upload_id, now)
                if completed is not None:
                    return completed
                raise BlobConflict(
                    "upload temp file does not match committed bytes",
                    code="upload_state_mismatch",
                )
            actual_sha256 = self._file_sha256(temp_path)
            if actual_sha256 != row["expected_sha256"]:
                raise BlobValidationError(
                    "upload SHA-256 does not match the descriptor",
                    code="sha256_mismatch",
                )
            self._validate_image_file(
                temp_path,
                content_type=row["content_type"],
                width=row["width"],
                height=row["height"],
            )
            parent_ids = tuple(json.loads(row["parent_ids_json"]))
            try:
                self._begin()
                self._purge_expired_locked(now)
                completed = self._completed_upload_locked(upload_id, now)
                if completed is not None:
                    self._db.execute("COMMIT")
                    return completed
                if row["deduplicate"]:
                    duplicate = self._db.execute(
                        """
                        SELECT blob_id FROM blobs
                        WHERE sha256 = ? AND purpose = ? AND owner_scope = ?
                          AND width = ? AND height = ? AND expires_at > ?
                        ORDER BY created_at ASC LIMIT 1
                        """,
                        (
                            actual_sha256,
                            row["purpose"],
                            row["owner_scope"],
                            row["width"],
                            row["height"],
                            now,
                        ),
                    ).fetchone()
                    if duplicate is not None:
                        self._db.execute(
                            "INSERT INTO completed_uploads VALUES (?, ?, ?)",
                            (upload_id, duplicate["blob_id"], now),
                        )
                        self._db.execute(
                            "DELETE FROM uploads WHERE upload_id = ?",
                            (upload_id,),
                        )
                        self._db.execute("COMMIT")
                        try:
                            temp_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        return self.descriptor(duplicate["blob_id"])
                object_row = self._db.execute(
                    "SELECT ref_count, size_bytes, relative_path FROM objects WHERE sha256 = ?",
                    (actual_sha256,),
                ).fetchone()
                incoming_size = 0 if object_row is not None else row["expected_size"]
                self._enforce_capacity_locked(
                    incoming_size,
                    now,
                    exclude_blob_ids=frozenset(parent_ids),
                )
                for parent_id in parent_ids:
                    if self._db.execute(
                        "SELECT 1 FROM blobs WHERE blob_id = ?",
                        (parent_id,),
                    ).fetchone() is None:
                        raise BlobNotFound(
                            f"parent blob not found: {parent_id}",
                            code="parent_blob_not_found",
                        )
                object_path = self._object_path(actual_sha256)
                relative_path = object_path.relative_to(self.root).as_posix()
                if object_row is None:
                    object_path.parent.mkdir(parents=True, exist_ok=True)
                    if object_path.exists():
                        if (
                            not object_path.is_file()
                            or object_path.stat().st_size != row["expected_size"]
                            or self._file_sha256(object_path) != actual_sha256
                        ):
                            raise BlobConflict(
                                "content-addressed object file is corrupt",
                                code="object_corrupt",
                            )
                    else:
                        staged_path = object_path.with_name(
                            f".{actual_sha256}.{upload_id}.staged"
                        )
                        try:
                            shutil.copyfile(temp_path, staged_path)
                            os.replace(staged_path, object_path)
                        finally:
                            staged_path.unlink(missing_ok=True)
                    self._db.execute(
                        "DELETE FROM object_gc WHERE sha256 = ?",
                        (actual_sha256,),
                    )
                    self._db.execute(
                        "INSERT INTO objects VALUES (?, ?, ?, 1)",
                        (actual_sha256, row["expected_size"], relative_path),
                    )
                else:
                    existing_path = (self.root / object_row["relative_path"]).resolve()
                    if (
                        not existing_path.is_file()
                        or object_row["size_bytes"] != row["expected_size"]
                        or existing_path.stat().st_size != row["expected_size"]
                        or self._file_sha256(existing_path) != actual_sha256
                    ):
                        raise BlobConflict(
                            "content-addressed object file is corrupt",
                            code="object_corrupt",
                        )
                    self._db.execute(
                        "UPDATE objects SET ref_count = ref_count + 1 WHERE sha256 = ?",
                        (actual_sha256,),
                    )
                blob_id = f"img_{uuid.uuid4().hex}"
                self._db.execute(
                    """
                    INSERT INTO blobs(
                        blob_id, sha256, content_type, purpose, owner_scope,
                        width, height, created_at, expires_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        blob_id,
                        actual_sha256,
                        row["content_type"],
                        row["purpose"],
                        row["owner_scope"],
                        row["width"],
                        row["height"],
                        now,
                        now + self.ttl_seconds,
                        row["metadata_json"],
                    ),
                )
                self._db.executemany(
                    "INSERT INTO blob_parents(blob_id, parent_blob_id) VALUES (?, ?)",
                    ((blob_id, parent_id) for parent_id in parent_ids),
                )
                self._db.execute(
                    "INSERT INTO completed_uploads VALUES (?, ?, ?)",
                    (upload_id, blob_id, now),
                )
                self._db.execute(
                    "DELETE FROM uploads WHERE upload_id = ?",
                    (upload_id,),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._drain_object_gc()
            return self._descriptor_locked(blob_id, now)

    def put_bytes(
        self,
        data: bytes,
        *,
        content_type: str,
        purpose: str,
        owner_scope: str,
        width: int,
        height: int,
        metadata: Optional[dict[str, Any]] = None,
        parent_blob_ids: tuple[str, ...] = (),
        deduplicate: bool = True,
    ) -> PersistentBlobDescriptor:
        payload = bytes(data)
        session = self.begin_upload(
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            content_type=content_type,
            purpose=purpose,
            owner_scope=owner_scope,
            width=width,
            height=height,
            metadata=metadata,
            parent_blob_ids=parent_blob_ids,
            deduplicate=deduplicate,
        )
        self.write_upload(session.upload_id, offset=0, data=payload)
        return self.commit_upload(session.upload_id)

    def descriptor(self, blob_id: str) -> PersistentBlobDescriptor:
        if _BLOB_ID.fullmatch(str(blob_id or "")) is None:
            raise BlobValidationError("blob_id is invalid", code="invalid_blob_id")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                descriptor = self._descriptor_locked(blob_id, now)
                self._db.execute("COMMIT")
                return descriptor
            except Exception:
                self._rollback()
                raise

    def descriptor_for_owner(
        self, blob_id: str, *, owner_scope: str,
    ) -> PersistentBlobDescriptor:
        """Return a descriptor only when its persisted owner matches exactly."""
        normalized_owner = self._safe_value(owner_scope, "owner_scope")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                row = self._db.execute(
                    "SELECT owner_scope FROM blobs WHERE blob_id = ?",
                    (blob_id,),
                ).fetchone()
                if row is None:
                    raise BlobNotFound(
                        f"image blob not found: {blob_id}",
                        code="blob_not_found",
                    )
                if not hmac.compare_digest(row["owner_scope"], normalized_owner):
                    raise BlobAuthorizationError(
                        "blob owner scope does not match",
                        code="blob_owner_mismatch",
                    )
                descriptor = self._descriptor_locked(blob_id, now)
                self._db.execute("COMMIT")
                return descriptor
            except Exception:
                self._rollback()
                raise

    def acquire_lease(
        self,
        blob_id: str,
        *,
        attempt_id: str,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    ) -> BlobLease:
        normalized_attempt = self._safe_value(attempt_id, "attempt_id")
        if ttl_seconds <= 0 or ttl_seconds > MAX_TRANSFER_GRANT_SECONDS:
            raise BlobValidationError(
                "lease TTL is outside the allowed range",
                code="invalid_lease_ttl",
            )
        now = self._clock()
        lease = BlobLease(
            lease_id=f"bls_{uuid.uuid4().hex}",
            blob_id=blob_id,
            attempt_id=normalized_attempt,
            expires_at=now + float(ttl_seconds),
        )
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                self._descriptor_locked(blob_id, now)
                self._db.execute(
                    "INSERT INTO blob_leases VALUES (?, ?, ?, ?, ?)",
                    (lease.lease_id, blob_id, normalized_attempt, now, lease.expires_at),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
        return lease

    def renew_lease(self, lease_id: str, *, ttl_seconds: float) -> BlobLease:
        if _LEASE_ID.fullmatch(str(lease_id or "")) is None:
            raise BlobValidationError("lease_id is invalid", code="invalid_lease_id")
        if ttl_seconds <= 0 or ttl_seconds > MAX_TRANSFER_GRANT_SECONDS:
            raise BlobValidationError(
                "lease TTL is outside the allowed range",
                code="invalid_lease_ttl",
            )
        now = self._clock()
        expires_at = now + float(ttl_seconds)
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                row = self._db.execute(
                    "SELECT * FROM blob_leases WHERE lease_id = ?",
                    (lease_id,),
                ).fetchone()
                if row is None:
                    raise BlobNotFound("blob lease not found", code="lease_not_found")
                self._db.execute(
                    "UPDATE blob_leases SET expires_at = ? WHERE lease_id = ?",
                    (expires_at, lease_id),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
        return BlobLease(
            lease_id=lease_id,
            blob_id=row["blob_id"],
            attempt_id=row["attempt_id"],
            expires_at=expires_at,
        )

    def release_lease(self, lease_id: str) -> bool:
        if _LEASE_ID.fullmatch(str(lease_id or "")) is None:
            raise BlobValidationError("lease_id is invalid", code="invalid_lease_id")
        with self._lock:
            self._ensure_open()
            cursor = self._db.execute(
                "DELETE FROM blob_leases WHERE lease_id = ?",
                (lease_id,),
            )
            return cursor.rowcount > 0

    def _leased_object_path(
        self,
        blob_id: str,
        *,
        lease_id: str,
        attempt_id: str,
    ) -> Path:
        now = self._clock()
        with self._lock:
            self._ensure_open()
            row = self._db.execute(
                """
                SELECT o.relative_path
                FROM blob_leases l
                JOIN blobs b ON b.blob_id = l.blob_id
                JOIN objects o ON o.sha256 = b.sha256
                WHERE l.lease_id = ? AND l.blob_id = ? AND l.attempt_id = ?
                  AND l.expires_at > ?
                """,
                (lease_id, blob_id, attempt_id, now),
            ).fetchone()
            if row is None:
                raise BlobAuthorizationError(
                    "active attempt-scoped blob lease is required",
                    code="invalid_blob_lease",
                )
            path = (self.root / row["relative_path"]).resolve()
            try:
                path.relative_to(self.objects_dir.resolve())
            except ValueError as exc:
                raise BlobAuthorizationError(
                    "blob object path escaped the data directory",
                    code="invalid_object_path",
                ) from exc
            if not path.is_file():
                raise BlobNotFound("blob object file is missing", code="object_missing")
            return path

    def iter_chunks(
        self,
        blob_id: str,
        *,
        lease_id: str,
        attempt_id: str,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        if chunk_size <= 0 or chunk_size > 4 * 1024 * 1024:
            raise BlobValidationError(
                "chunk_size is outside the allowed range",
                code="invalid_chunk_size",
            )
        path = self._leased_object_path(
            blob_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
        )
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk

    def read_chunk(
        self,
        blob_id: str,
        *,
        lease_id: str,
        attempt_id: str,
        offset: int,
        length: int,
    ) -> bytes:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise BlobValidationError("blob offset is invalid", code="invalid_offset")
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or length > 4 * 1024 * 1024
        ):
            raise BlobValidationError(
                "blob chunk length is outside the allowed range",
                code="invalid_chunk_size",
            )
        path = self._leased_object_path(
            blob_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
        )
        size = path.stat().st_size
        if offset > size:
            raise BlobValidationError(
                "blob offset exceeds the object size",
                code="invalid_offset",
            )
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)

    def read_all(self, blob_id: str, *, lease_id: str, attempt_id: str) -> bytes:
        return b"".join(
            self.iter_chunks(
                blob_id,
                lease_id=lease_id,
                attempt_id=attempt_id,
            )
        )

    def delete(self, blob_id: str, *, owner_scope: str) -> bool:
        normalized_owner = self._safe_value(owner_scope, "owner_scope")
        now = self._clock()
        with self._lock:
            self._ensure_open()
            try:
                self._begin()
                self._purge_expired_locked(now)
                row = self._db.execute(
                    "SELECT owner_scope FROM blobs WHERE blob_id = ?",
                    (blob_id,),
                ).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return False
                if not hmac.compare_digest(row["owner_scope"], normalized_owner):
                    raise BlobAuthorizationError(
                        "blob owner scope does not match",
                        code="blob_owner_mismatch",
                    )
                if self._db.execute(
                    "SELECT 1 FROM blob_leases WHERE blob_id = ? AND expires_at > ?",
                    (blob_id, now),
                ).fetchone() is not None:
                    raise BlobConflict("blob has an active lease", code="blob_in_use")
                if self._db.execute(
                    "SELECT 1 FROM blob_parents WHERE parent_blob_id = ?",
                    (blob_id,),
                ).fetchone() is not None:
                    raise BlobConflict(
                        "blob is referenced by another result",
                        code="blob_referenced",
                    )
                self._delete_blob_locked(blob_id)
                self._db.execute("COMMIT")
                self._drain_object_gc()
                return True
            except Exception:
                self._rollback()
                raise

    def cleanup(self) -> dict[str, int]:
        now = self._clock()
        with self._lock:
            self._ensure_open()
            before_blobs = int(self._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0])
            before_uploads = int(self._db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0])
            before_leases = int(self._db.execute("SELECT COUNT(*) FROM blob_leases").fetchone()[0])
            try:
                self._begin()
                self._purge_expired_locked(now)
                self._db.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
            self._drain_object_gc()
            return {
                "blobs_removed": before_blobs - int(
                    self._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
                ),
                "uploads_removed": before_uploads - int(
                    self._db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
                ),
                "leases_removed": before_leases - int(
                    self._db.execute("SELECT COUNT(*) FROM blob_leases").fetchone()[0]
                ),
            }

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            self._ensure_open()
            self.cleanup()
            return {
                "backend": "sqlite_content_addressed",
                "blobs": int(self._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]),
                "objects": int(self._db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]),
                "pending_object_gc": int(
                    self._db.execute("SELECT COUNT(*) FROM object_gc").fetchone()[0]
                ),
                "uploads": int(self._db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]),
                "active_leases": int(
                    self._db.execute(
                        "SELECT COUNT(*) FROM blob_leases WHERE expires_at > ?",
                        (now,),
                    ).fetchone()[0]
                ),
                "total_bytes": int(
                    self._db.execute(
                        "SELECT COALESCE(SUM(size_bytes), 0) FROM objects"
                    ).fetchone()[0]
                ),
                "max_blob_bytes": self.max_blob_bytes,
                "max_pixels": self.max_pixels,
                "max_total_bytes": self.max_total_bytes,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def __enter__(self) -> "PersistentImageBlobStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class BlobTransferTokenSigner:
    """Issue and validate short attempt-scoped HMAC transfer grants."""

    def __init__(self, secret: bytes | str, *, clock: Callable[[], float] = time.time):
        key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(key) < 32:
            raise ValueError("blob transfer secret must contain at least 32 bytes")
        self._key = key
        self._clock = clock

    @staticmethod
    def _encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        try:
            return base64.urlsafe_b64decode(value + padding)
        except (ValueError, TypeError) as exc:
            raise BlobAuthorizationError(
                "blob transfer grant is malformed",
                code="invalid_transfer_grant",
            ) from exc

    def issue(
        self,
        *,
        attempt_id: str,
        direction: str,
        ttl_seconds: float,
        blob_id: str | None = None,
        lease_id: str | None = None,
        upload_id: str | None = None,
    ) -> str:
        normalized_attempt = PersistentImageBlobStore._safe_value(
            attempt_id,
            "attempt_id",
        )
        if direction not in {"download", "upload"}:
            raise BlobValidationError(
                "transfer direction is invalid",
                code="invalid_transfer_direction",
            )
        if direction == "download":
            if (
                not isinstance(blob_id, str)
                or _BLOB_ID.fullmatch(blob_id) is None
                or not isinstance(lease_id, str)
                or _LEASE_ID.fullmatch(lease_id) is None
                or upload_id is not None
            ):
                raise BlobValidationError(
                    "download grant requires only a blob and lease identifier",
                    code="invalid_transfer_identity",
                )
        elif (
            not isinstance(upload_id, str)
            or _UPLOAD_ID.fullmatch(upload_id) is None
            or blob_id is not None
            or lease_id is not None
        ):
            raise BlobValidationError(
                "upload grant requires only an upload session identifier",
                code="invalid_transfer_identity",
            )
        if ttl_seconds <= 0 or ttl_seconds > MAX_TRANSFER_GRANT_SECONDS:
            raise BlobValidationError(
                "transfer grant TTL is outside the allowed range",
                code="invalid_transfer_ttl",
            )
        payload = {
            "attempt_id": normalized_attempt,
            "blob_id": blob_id,
            "direction": direction,
            "expires_at": int(self._clock() + ttl_seconds),
            "lease_id": lease_id,
            "nonce": uuid.uuid4().hex,
            "upload_id": upload_id,
            "version": 1,
        }
        encoded = self._encode(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        signature = self._encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}"

    def verify(
        self,
        token: str,
        *,
        direction: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 4096:
            raise BlobAuthorizationError(
                "blob transfer grant is malformed",
                code="invalid_transfer_grant",
            )
        try:
            encoded, signature = token.split(".", 1)
        except ValueError as exc:
            raise BlobAuthorizationError(
                "blob transfer grant is malformed",
                code="invalid_transfer_grant",
            ) from exc
        expected = self._encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise BlobAuthorizationError(
                "blob transfer grant signature is invalid",
                code="invalid_transfer_signature",
            )
        try:
            payload = json.loads(self._decode(encoded))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BlobAuthorizationError(
                "blob transfer grant payload is invalid",
                code="invalid_transfer_grant",
            ) from exc
        expected_fields = {
            "attempt_id",
            "blob_id",
            "direction",
            "expires_at",
            "lease_id",
            "nonce",
            "upload_id",
            "version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise BlobAuthorizationError(
                "blob transfer grant fields are invalid",
                code="invalid_transfer_grant",
            )
        if payload["version"] != 1 or payload["direction"] != direction:
            raise BlobAuthorizationError(
                "blob transfer grant scope does not match",
                code="transfer_scope_mismatch",
            )
        if not hmac.compare_digest(str(payload["attempt_id"]), str(attempt_id)):
            raise BlobAuthorizationError(
                "blob transfer grant attempt does not match",
                code="transfer_scope_mismatch",
            )
        expires_at = payload["expires_at"]
        if isinstance(expires_at, bool) or not isinstance(expires_at, int):
            raise BlobAuthorizationError(
                "blob transfer grant expiry is invalid",
                code="invalid_transfer_grant",
            )
        if expires_at <= int(self._clock()):
            raise BlobAuthorizationError(
                "blob transfer grant has expired",
                code="transfer_grant_expired",
            )
        if direction == "download" and (
            not isinstance(payload["blob_id"], str)
            or _BLOB_ID.fullmatch(payload["blob_id"]) is None
            or not isinstance(payload["lease_id"], str)
            or _LEASE_ID.fullmatch(payload["lease_id"]) is None
            or payload["upload_id"] is not None
        ):
            raise BlobAuthorizationError(
                "blob transfer grant identity is invalid",
                code="invalid_transfer_grant",
            )
        if direction == "upload" and (
            not isinstance(payload["upload_id"], str)
            or _UPLOAD_ID.fullmatch(payload["upload_id"]) is None
            or payload["blob_id"] is not None
            or payload["lease_id"] is not None
        ):
            raise BlobAuthorizationError(
                "upload transfer grant identity is invalid",
                code="invalid_transfer_grant",
            )
        return payload


__all__ = [
    "BlobAuthorizationError",
    "BlobConflict",
    "BlobLease",
    "BlobNotFound",
    "BlobTransferTokenSigner",
    "BlobUploadSession",
    "BlobValidationError",
    "DistributedBlobError",
    "DiffusionArtifactComponent",
    "DiffusionArtifactManifest",
    "PersistentBlobDescriptor",
    "PersistentImageBlobStore",
]
