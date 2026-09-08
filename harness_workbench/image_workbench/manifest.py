"""Verification of the SD asset manifest produced by QLH asset tooling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MANIFEST_NAME = ".qlh-sd-asset.json"


@dataclass(frozen=True, slots=True)
class AssetManifestFile:
    path: str
    size_bytes: int
    sha256: str = ""


@dataclass(frozen=True, slots=True)
class AssetManifest:
    asset_id: str
    artifact_id: str
    files: tuple[AssetManifestFile, ...]
    schema_version: int = 0


@dataclass(frozen=True, slots=True)
class AssetManifestReport:
    valid: bool
    asset_id: str | None = None
    artifact_id: str | None = None
    errors: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    size_mismatches: tuple[Mapping[str, Any], ...] = ()
    hash_mismatches: tuple[Mapping[str, Any], ...] = ()
    manifest: AssetManifest | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "asset_id": self.asset_id,
            "artifact_id": self.artifact_id,
            "errors": list(self.errors),
            "missing": list(self.missing),
            "size_mismatches": [dict(item) for item in self.size_mismatches],
            "hash_mismatches": [dict(item) for item in self.hash_mismatches],
            "file_count": len(self.manifest.files) if self.manifest else 0,
        }


def validate_asset_manifest(
    root: Path | str,
    *,
    expected_asset_id: str | None = None,
    full_hash: bool = False,
) -> AssetManifestReport:
    """Validate a QLH offline SD manifest without importing the main runtime.

    The validator deliberately checks only the portable manifest contract.  A
    CUDA/diffusers executor remains responsible for validating model loading.
    """

    root_path = Path(root).expanduser()
    manifest_path = root_path / MANIFEST_NAME
    if not root_path.is_dir():
        return AssetManifestReport(False, errors=("asset_directory_missing",))
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AssetManifestReport(False, errors=("manifest_missing",))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return AssetManifestReport(False, errors=("manifest_invalid_json",))
    if not isinstance(payload, Mapping):
        return AssetManifestReport(False, errors=("manifest_not_object",))
    asset = payload.get("asset")
    if not isinstance(asset, Mapping):
        return AssetManifestReport(False, errors=("manifest_asset_missing",))
    asset_id = _text(asset.get("asset_id"))
    artifact_id = _text(asset.get("artifact_id"))
    errors: list[str] = []
    if not asset_id:
        errors.append("asset_id_missing")
    if expected_asset_id and asset_id != expected_asset_id:
        errors.append("asset_id_mismatch")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        errors.append("manifest_files_missing")
        return AssetManifestReport(False, asset_id, artifact_id, tuple(errors))
    files: list[AssetManifestFile] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping):
            errors.append("manifest_file_invalid")
            continue
        name = _text(item.get("path"))
        if not name or not _safe_relative(name):
            errors.append("manifest_file_path_invalid")
            continue
        if name in seen:
            errors.append("manifest_file_duplicate")
            continue
        seen.add(name)
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append("manifest_file_size_invalid")
            continue
        digest = _text(item.get("sha256"))
        if digest and (len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest)):
            errors.append("manifest_file_hash_invalid")
            continue
        files.append(AssetManifestFile(name, size, digest.lower()))

    manifest = AssetManifest(asset_id, artifact_id, tuple(files), _as_int(payload.get("schema_version")))
    missing: list[str] = []
    size_mismatches: list[Mapping[str, Any]] = []
    hash_mismatches: list[Mapping[str, Any]] = []
    for item in files:
        candidate = root_path / item.path
        if not candidate.is_file():
            missing.append(item.path)
            continue
        actual_size = candidate.stat().st_size
        if actual_size != item.size_bytes:
            size_mismatches.append({"path": item.path, "expected": item.size_bytes, "actual": actual_size})
        if full_hash and item.sha256:
            actual_hash = _sha256(candidate)
            if actual_hash != item.sha256:
                hash_mismatches.append({"path": item.path, "expected": item.sha256, "actual": actual_hash})
    valid = not errors and not missing and not size_mismatches and not hash_mismatches
    return AssetManifestReport(
        valid,
        asset_id,
        artifact_id,
        tuple(errors),
        tuple(missing),
        tuple(size_mismatches),
        tuple(hash_mismatches),
        manifest,
    )


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    first = path.parts[0] if path.parts else ""
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and ":" not in first
        and value not in {"", "."}
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["AssetManifest", "AssetManifestFile", "AssetManifestReport", "MANIFEST_NAME", "validate_asset_manifest"]
