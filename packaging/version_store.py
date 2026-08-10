"""Crash-safe application version store for Launcher UP-N3."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from install_manifest import InstallManifestError, verify_manifest_file_if_present
from signing import default_trusted_keys_dir
from update_core import UpdateError, default_state_dir, version_key

try:
    import tarfile
except ImportError:  # pragma: no cover
    tarfile = None  # type: ignore[assignment]


class VersionStoreError(UpdateError):
    """Expected version-store failure."""


_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_POINTER_SCHEMA = 1


def default_version_store_root() -> Path:
    override = os.environ.get("QLH_VERSION_STORE", "").strip()
    if override:
        return Path(override).expanduser()
    app_home = os.environ.get("QLH_APP_HOME", "").strip()
    if app_home:
        return Path(app_home).expanduser() / "versions-store"
    return default_state_dir() / "app"


def _safe_version(version: str, variant: str) -> tuple[str, str, str]:
    version = str(version).strip()
    variant = str(variant).strip().lower()
    try:
        version_key(version)
    except UpdateError as exc:
        raise VersionStoreError(f"invalid version: {version!r}") from exc
    if not _VARIANT_RE.fullmatch(variant):
        raise VersionStoreError(f"invalid version variant: {variant!r}")
    return version, variant, f"{version}-{variant}"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_pointer(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != _POINTER_SCHEMA:
        return None
    try:
        version = str(value["version"])
        variant = str(value["variant"])
        expected = _safe_version(version, variant)[2]
    except (KeyError, VersionStoreError):
        return None
    if value.get("directory") != expected:
        return None
    return value


def _reject_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VersionStoreError(f"bundle contains unsupported symlink: {path.name}")


def _verify_install_manifest(
    root: Path, *, expected_version: str, expected_variant: str,
) -> None:
    trusted = root / "pubkeys"
    keys_dir = trusted if trusted.is_dir() else default_trusted_keys_dir()
    try:
        mapping = verify_manifest_file_if_present(root, trusted_keys_dir=keys_dir)
    except InstallManifestError as exc:
        raise VersionStoreError(f"version install manifest rejected: {exc}") from exc
    if mapping is None:
        return
    if mapping["app_id"] != "qlh-edge-inference" or mapping["package_kind"] != "application":
        raise VersionStoreError("version install manifest has the wrong package identity")
    if mapping["version"] != expected_version or mapping["variant"] != expected_variant:
        raise VersionStoreError(
            "version install manifest identity mismatch: "
            f"{mapping['version']}/{mapping['variant']} != {expected_version}/{expected_variant}"
        )


def _safe_member_path(root: Path, name: str) -> Path:
    candidate = (root / name.replace("\\", "/")).resolve()
    base = root.resolve()
    if candidate != base and base not in candidate.parents:
        raise VersionStoreError(f"bundle member escapes destination: {name!r}")
    return candidate


@dataclass(frozen=True)
class VersionPointer:
    version: str
    variant: str
    directory: str
    activated_at: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VersionPointer":
        version = str(value["version"])
        variant = str(value["variant"])
        directory = _safe_version(version, variant)[2]
        return cls(version, variant, directory, str(value.get("activated_at", "")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _POINTER_SCHEMA,
            "version": self.version,
            "variant": self.variant,
            "directory": self.directory,
            "activated_at": self.activated_at,
        }


HealthCheck = Callable[[Path], bool]


class VersionStore:
    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root).expanduser() if root else default_version_store_root()
        self.versions_dir = self.root / "versions"
        self.staging_dir = self.root / "staging"
        self.current_file = self.root / "current.json"
        self.previous_file = self.root / "previous.json"

    def _pointer(self, path: Path) -> VersionPointer | None:
        value = _load_pointer(path)
        return VersionPointer.from_mapping(value) if value else None

    def _payload_path(self, pointer: VersionPointer | None) -> Path | None:
        if pointer is None:
            return None
        path = (self.versions_dir / pointer.directory).resolve()
        base = self.versions_dir.resolve()
        if path != base and base not in path.parents:
            raise VersionStoreError("version pointer escapes versions directory")
        return path

    def current(self) -> VersionPointer | None:
        return self._pointer(self.current_file)

    def previous(self) -> VersionPointer | None:
        return self._pointer(self.previous_file)

    def active_path(self) -> Path | None:
        path = self._payload_path(self.current())
        return path if path and path.is_dir() else None

    def status(self) -> dict[str, Any]:
        current = self.current()
        previous = self.previous()
        current_path = self._payload_path(current)
        previous_path = self._payload_path(previous)
        versions = []
        if self.versions_dir.is_dir():
            versions = sorted(path.name for path in self.versions_dir.iterdir() if path.is_dir())
        return {
            "root": str(self.root),
            "current": current.as_dict() if current else None,
            "current_exists": bool(current_path and current_path.is_dir()),
            "previous": previous.as_dict() if previous else None,
            "previous_exists": bool(previous_path and previous_path.is_dir()),
            "versions": versions,
        }

    def stage_directory(self, source: str | os.PathLike[str], version: str, variant: str) -> Path:
        version, variant, directory = _safe_version(version, variant)
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise VersionStoreError(f"bundle source is not a directory: {source}")
        _reject_symlinks(source_path)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        target = self.versions_dir / directory
        if target.exists():
            raise VersionStoreError(f"version already staged: {directory}")
        temporary = self.staging_dir / f".{directory}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copytree(source_path, temporary, symlinks=False)
            _reject_symlinks(temporary)
            _verify_install_manifest(
                temporary, expected_version=version, expected_variant=variant,
            )
            os.replace(temporary, target)
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(exc, VersionStoreError):
                raise
            raise VersionStoreError(f"stage directory failed: {exc}") from exc
        return target

    def stage_archive(self, archive: str | os.PathLike[str], version: str, variant: str) -> Path:
        _, _, directory = _safe_version(version, variant)
        archive_path = Path(archive).expanduser().resolve()
        if not archive_path.is_file():
            raise VersionStoreError(f"bundle archive not found: {archive}")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.staging_dir / f".{directory}.{uuid.uuid4().hex}.extract"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as bundle:
                    for member in bundle.infolist():
                        destination = _safe_member_path(temporary, member.filename)
                        if member.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                            continue
                        unix_mode = (member.external_attr >> 16) & 0o170000
                        if unix_mode == 0o120000:
                            raise VersionStoreError(f"bundle contains unsupported symlink: {member.filename}")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(member) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
            elif tarfile is not None and tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as bundle:
                    for member in bundle.getmembers():
                        if member.issym() or member.islnk():
                            raise VersionStoreError(f"bundle contains unsupported link: {member.name}")
                        if not (member.isdir() or member.isfile()):
                            raise VersionStoreError(f"bundle contains unsupported entry: {member.name}")
                        _safe_member_path(temporary, member.name)
                    bundle.extractall(temporary)
            else:
                raise VersionStoreError(f"unsupported bundle archive: {archive_path.name}")
            _reject_symlinks(temporary)
            result = self.stage_directory(temporary, version, variant)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return result

    def activate(self, version: str, variant: str, *, health_check: HealthCheck | None = None) -> VersionPointer:
        version, variant, directory = _safe_version(version, variant)
        candidate = self.versions_dir / directory
        if not candidate.is_dir():
            raise VersionStoreError(f"version is not staged: {directory}")
        _reject_symlinks(candidate)
        _verify_install_manifest(
            candidate, expected_version=version, expected_variant=variant,
        )
        if health_check is not None:
            try:
                healthy = bool(health_check(candidate))
            except Exception as exc:
                raise VersionStoreError(f"health check raised: {exc}") from exc
            if not healthy:
                raise VersionStoreError(f"health check failed: {directory}")
        old = self.current()
        pointer = VersionPointer(version, variant, directory, datetime.now(timezone.utc).isoformat())
        if old and old.directory == directory:
            _atomic_write_json(self.current_file, pointer.as_dict())
            return pointer
        if old is not None and self._payload_path(old) and self._payload_path(old).is_dir():
            _atomic_write_json(self.previous_file, old.as_dict())
        _atomic_write_json(self.current_file, pointer.as_dict())
        return pointer

    def rollback(self, *, health_check: HealthCheck | None = None) -> VersionPointer:
        current = self.current()
        target = self.previous()
        target_path = self._payload_path(target)
        if target is None or target_path is None or not target_path.is_dir():
            raise VersionStoreError("no healthy previous version is available")
        _reject_symlinks(target_path)
        if health_check is not None and not health_check(target_path):
            raise VersionStoreError(f"rollback health check failed: {target.directory}")
        if current is not None:
            _atomic_write_json(self.previous_file, current.as_dict())
        _atomic_write_json(self.current_file, target.as_dict())
        return target

    def recover(self) -> VersionPointer | None:
        current = self.current()
        current_path = self._payload_path(current)
        if current is not None and current_path and current_path.is_dir():
            return current
        previous = self.previous()
        previous_path = self._payload_path(previous)
        if previous is None or previous_path is None or not previous_path.is_dir():
            return None
        _atomic_write_json(self.current_file, previous.as_dict())
        return previous


def marker_health_check(path: Path) -> bool:
    marker = path / "health.ok"
    return marker.is_file() or any(path.iterdir())
