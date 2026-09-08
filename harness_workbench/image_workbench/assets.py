"""User-owned storage for generated image blobs and session metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import GeneratedImage, ImageAdapterError


@dataclass(frozen=True, slots=True)
class ImageAssetRecord:
    asset_id: str
    sha256: str
    mime_type: str
    size_bytes: int
    owner_scope: str
    prompt: str = ""
    width: int | None = None
    height: int | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "owner_scope": self.owner_scope,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
        }


class ImageAssetStore:
    """Store blobs under an explicit user-owned root with atomic writes."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.blob_root = self.root / "blobs"
        self.metadata_root = self.root / "metadata"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        image: GeneratedImage,
        *,
        prompt: str = "",
        owner_scope: str = "local",
    ) -> ImageAssetRecord:
        digest = hashlib.sha256(image.data).hexdigest()
        asset_id = f"img_{digest[:24]}"
        suffix = _suffix(image.mime_type)
        blob_path = self.blob_root / f"{asset_id}{suffix}"
        _atomic_write(blob_path, image.data)
        record = ImageAssetRecord(
            asset_id=asset_id,
            sha256=digest,
            mime_type=image.mime_type,
            size_bytes=len(image.data),
            owner_scope=_owner_scope(owner_scope),
            prompt=prompt[:4000],
            width=image.width,
            height=image.height,
            seed=image.seed,
        )
        _atomic_write(
            self.metadata_root / f"{asset_id}.json",
            (json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        )
        return record

    def get_record(self, asset_id: str) -> ImageAssetRecord:
        if not _asset_id(asset_id):
            raise ImageAdapterError("invalid image asset id", code="invalid_asset_id", status_code=400)
        path = self.metadata_root / f"{asset_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageAdapterError("image asset not found", code="asset_not_found", status_code=404) from exc
        if not isinstance(payload, Mapping):
            raise ImageAdapterError("image asset metadata is invalid", code="asset_invalid", status_code=500)
        try:
            return ImageAssetRecord(
                asset_id=str(payload["asset_id"]),
                sha256=str(payload["sha256"]),
                mime_type=str(payload["mime_type"]),
                size_bytes=int(payload["size_bytes"]),
                owner_scope=str(payload["owner_scope"]),
                prompt=str(payload.get("prompt", "")),
                width=_optional_int(payload.get("width")),
                height=_optional_int(payload.get("height")),
                seed=_optional_int(payload.get("seed")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageAdapterError("image asset metadata is invalid", code="asset_invalid", status_code=500) from exc

    def read(self, asset_id: str) -> tuple[ImageAssetRecord, bytes]:
        record = self.get_record(asset_id)
        path = self.blob_root / f"{record.asset_id}{_suffix(record.mime_type)}"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ImageAdapterError("image asset blob is unavailable", code="asset_blob_missing", status_code=404) from exc
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ImageAdapterError("image asset integrity check failed", code="asset_integrity_error", status_code=500)
        return record, data


def _asset_id(value: str) -> bool:
    return isinstance(value, str) and value.startswith("img_") and value[4:].isalnum() and len(value) <= 64


def _owner_scope(value: str) -> str:
    cleaned = str(value or "local").strip()
    if not cleaned or len(cleaned) > 128 or any(char in cleaned for char in "\\/\r\n"):
        raise ValueError("owner_scope is invalid")
    return cleaned


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _suffix(mime_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime_type, ".png")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["ImageAssetRecord", "ImageAssetStore"]
