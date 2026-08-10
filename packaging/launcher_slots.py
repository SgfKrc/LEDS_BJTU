"""A/B self-update slots and diagnostics for Launcher UP-N4.

The installed Launcher remains the stable entrypoint.  New PyInstaller
bundles are staged into the inactive slot, validated, and activated by an
atomic JSON pointer.  The stable entrypoint can delegate normal UI/app
commands to the active slot, while maintenance commands always stay on the
stable copy so a broken slot cannot hide repair operations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from install_manifest import InstallManifestError, verify_manifest_file_if_present
from signing import default_trusted_keys_dir
from update_core import UpdateError, default_state_dir


class LauncherSlotError(UpdateError):
    """Expected A/B slot or diagnostic failure."""


_SCHEMA = 1
_SLOTS = ("a", "b")
_MAINTENANCE_COMMANDS = {
    "check", "download", "install", "launcher-status", "launcher-stage",
    "launcher-activate", "launcher-rollback", "launcher-recover", "diagnostics", "verify", "diagnose", "repair",
    "version-status", "version-stage", "version-activate", "version-rollback",
    "version-recover",
}


def default_launcher_slot_root() -> Path:
    override = os.environ.get("QLH_LAUNCHER_SLOTS", "").strip()
    if override:
        return Path(override).expanduser()
    return default_state_dir() / "launcher-slots"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_slot(value: Any) -> str:
    slot = str(value).lower()
    if slot not in _SLOTS:
        raise LauncherSlotError(f"invalid launcher slot: {value!r}")
    return slot


def _reject_links(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LauncherSlotError(f"launcher bundle contains symlink: {path.name}")


def _verify_install_manifest(
    root: Path, *, required: bool, expected_version: str | None = None,
) -> None:
    trusted = root / "pubkeys"
    keys_dir = trusted if trusted.is_dir() else default_trusted_keys_dir()
    try:
        mapping = verify_manifest_file_if_present(
            root, trusted_keys_dir=keys_dir, required=required,
        )
    except InstallManifestError as exc:
        raise LauncherSlotError(f"launcher install manifest rejected: {exc}") from exc
    if mapping is None:
        return
    if mapping["app_id"] != "qlh-launcher" or mapping["package_kind"] != "launcher":
        raise LauncherSlotError("launcher install manifest has the wrong package identity")
    if mapping["variant"] != "any":
        raise LauncherSlotError("launcher install manifest variant must be any")
    if expected_version is not None and mapping["version"] != str(expected_version):
        raise LauncherSlotError(
            f"launcher install manifest version mismatch: {mapping['version']} != {expected_version}"
        )


def _safe_member(root: Path, name: str) -> Path:
    candidate = (root / name.replace("\\", "/")).resolve()
    base = root.resolve()
    if candidate != base and base not in candidate.parents:
        raise LauncherSlotError(f"launcher bundle member escapes destination: {name!r}")
    return candidate


@dataclass(frozen=True)
class LauncherPointer:
    slot: str
    version: str
    entrypoint: str
    activated_at: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LauncherPointer":
        return cls(
            slot=_safe_slot(value["slot"]),
            version=str(value["version"]),
            entrypoint=str(value["entrypoint"]),
            activated_at=str(value.get("activated_at", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA,
            "slot": self.slot,
            "version": self.version,
            "entrypoint": self.entrypoint,
            "activated_at": self.activated_at,
        }


HealthCheck = Callable[[Path], bool]


class LauncherSlotStore:
    def __init__(self, root: str | os.PathLike[str] | None = None):
        self.root = Path(root).expanduser() if root else default_launcher_slot_root()
        self.slots_dir = self.root / "slots"
        self.staging_dir = self.root / "staging"
        self.current_file = self.root / "current.json"
        self.previous_file = self.root / "previous.json"

    def _pointer(self, path: Path) -> LauncherPointer | None:
        value = _read_json(path)
        if not value or value.get("schema_version") != _SCHEMA:
            return None
        try:
            return LauncherPointer.from_mapping(value)
        except (KeyError, LauncherSlotError):
            return None

    def current(self) -> LauncherPointer | None:
        return self._pointer(self.current_file)

    def previous(self) -> LauncherPointer | None:
        return self._pointer(self.previous_file)

    def slot_path(self, slot: str) -> Path:
        slot = _safe_slot(slot)
        return self.slots_dir / slot

    def _entrypoint_path(self, pointer: LauncherPointer | None) -> Path | None:
        if pointer is None:
            return None
        root = self.slot_path(pointer.slot).resolve()
        candidate = (root / pointer.entrypoint).resolve()
        if candidate != root and root not in candidate.parents:
            raise LauncherSlotError("launcher pointer entrypoint escapes slot")
        return candidate

    def active_path(self) -> Path | None:
        pointer = self.current()
        path = self._entrypoint_path(pointer)
        return path if path and path.is_file() else None

    def _choose_inactive(self) -> str:
        current = self.current()
        return "b" if current and current.slot == "a" else "a"

    def _entrypoint_for(self, root: Path) -> str:
        candidates = (
            "QLH-Launcher.exe",
            "qlh-launcher.exe",
            "qlh-launcher",
            "qlh_launcher.py",
        )
        for name in candidates:
            if (root / name).is_file():
                return name
        raise LauncherSlotError("launcher bundle has no recognized entrypoint")

    def _validate(
        self,
        root: Path,
        *,
        require_install_manifest: bool = False,
        expected_version: str | None = None,
    ) -> str:
        if not root.is_dir():
            raise LauncherSlotError(f"launcher slot is not a directory: {root}")
        _reject_links(root)
        _verify_install_manifest(
            root, required=require_install_manifest, expected_version=expected_version,
        )
        entrypoint = self._entrypoint_for(root)
        if (root / "health.ok").is_file():
            return entrypoint
        if entrypoint.endswith(".py"):
            try:
                compile((root / entrypoint).read_text(encoding="utf-8"), entrypoint, "exec")
            except (OSError, SyntaxError) as exc:
                raise LauncherSlotError(f"launcher health check failed: {exc}") from exc
        elif (root / entrypoint).stat().st_size <= 0:
            raise LauncherSlotError("launcher executable is empty")
        return entrypoint

    def status(self) -> dict[str, Any]:
        current = self.current()
        previous = self.previous()
        return {
            "root": str(self.root),
            "current": current.as_dict() if current else None,
            "current_exists": bool(self._entrypoint_path(current) and self._entrypoint_path(current).is_file()),
            "previous": previous.as_dict() if previous else None,
            "previous_exists": bool(self._entrypoint_path(previous) and self._entrypoint_path(previous).is_file()),
            "slots": {
                slot: self.slot_path(slot).is_dir() for slot in _SLOTS
            },
        }

    def stage_directory(
        self,
        source: str | os.PathLike[str],
        version: str,
        *,
        require_install_manifest: bool = False,
    ) -> LauncherPointer:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise LauncherSlotError(f"launcher bundle source is not a directory: {source}")
        _reject_links(source_path)
        slot = self._choose_inactive()
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.staging_dir / f".{slot}.{uuid.uuid4().hex}.tmp"
        target = self.slot_path(slot)
        try:
            shutil.copytree(source_path, temporary, symlinks=False)
            entrypoint = self._validate(
                temporary,
                require_install_manifest=require_install_manifest,
                expected_version=str(version),
            )
            shutil.rmtree(target, ignore_errors=True)
            os.replace(temporary, target)
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(exc, LauncherSlotError):
                raise
            raise LauncherSlotError(f"launcher stage failed: {exc}") from exc
        return LauncherPointer(slot, str(version), entrypoint, "")

    def stage_archive(
        self,
        archive: str | os.PathLike[str],
        version: str,
        *,
        require_install_manifest: bool = False,
    ) -> LauncherPointer:
        archive_path = Path(archive).expanduser().resolve()
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            raise LauncherSlotError("launcher self-update currently accepts ZIP bundles only")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.staging_dir / f".extract.{uuid.uuid4().hex}"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(archive_path) as bundle:
                for member in bundle.infolist():
                    destination = _safe_member(temporary, member.filename)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    unix_mode = (member.external_attr >> 16) & 0o170000
                    if unix_mode == 0o120000:
                        raise LauncherSlotError(f"launcher ZIP contains symlink: {member.filename}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
            return self.stage_directory(
                temporary, version, require_install_manifest=require_install_manifest,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def activate(self, version: str, *, health_check: HealthCheck | None = None) -> LauncherPointer:
        slot = self._choose_inactive()
        candidate = self.slot_path(slot)
        entrypoint = self._validate(
            candidate,
            require_install_manifest=(candidate / "manifest" / "install-manifest.json").is_file(),
            expected_version=str(version),
        )
        if health_check is not None and not health_check(candidate):
            raise LauncherSlotError(f"launcher health check failed: {slot}")
        old = self.current()
        pointer = LauncherPointer(slot, str(version), entrypoint, datetime.now(timezone.utc).isoformat())
        if old is not None:
            _atomic_json(self.previous_file, old.as_dict())
        _atomic_json(self.current_file, pointer.as_dict())
        return pointer

    def rollback(self, *, health_check: HealthCheck | None = None) -> LauncherPointer:
        target = self.previous()
        path = self._entrypoint_path(target)
        if target is None or path is None or not path.is_file():
            raise LauncherSlotError("no previous healthy Launcher slot is available")
        target_root = self.slot_path(target.slot)
        self._validate(
            target_root,
            require_install_manifest=(target_root / "manifest" / "install-manifest.json").is_file(),
            expected_version=target.version,
        )
        if health_check is not None and not health_check(self.slot_path(target.slot)):
            raise LauncherSlotError("previous Launcher slot health check failed")
        old = self.current()
        if old is not None:
            _atomic_json(self.previous_file, old.as_dict())
        _atomic_json(self.current_file, target.as_dict())
        return target

    def recover(self) -> LauncherPointer | None:
        current = self.current()
        if current and self._entrypoint_path(current) and self._entrypoint_path(current).is_file():
            try:
                current_root = self.slot_path(current.slot)
                self._validate(
                    current_root,
                    require_install_manifest=(current_root / "manifest" / "install-manifest.json").is_file(),
                    expected_version=current.version,
                )
                return current
            except LauncherSlotError:
                pass
        previous = self.previous()
        if previous and self._entrypoint_path(previous) and self._entrypoint_path(previous).is_file():
            previous_root = self.slot_path(previous.slot)
            self._validate(
                previous_root,
                require_install_manifest=(previous_root / "manifest" / "install-manifest.json").is_file(),
                expected_version=previous.version,
            )
            _atomic_json(self.current_file, previous.as_dict())
            return previous
        return None

    def diagnostics(self, output: str | os.PathLike[str], *, extra_paths: list[Path] | None = None) -> Path:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        files: list[tuple[str, Path]] = []
        for name, path in (("current.json", self.current_file), ("previous.json", self.previous_file)):
            if path.is_file():
                files.append((f"launcher/{name}", path))
        state_dir = default_state_dir()
        for path in (state_dir / "launcher.json",):
            if path.is_file():
                files.append((f"state/{path.name}", path))
        for path in extra_paths or []:
            if path.is_file() and path.name.lower().endswith((".log", ".txt", ".json")):
                prefix = "diagnosis" if path.name.startswith("qlh-diagnose-") else "logs"
                files.append((f"{prefix}/{path.name}", path))
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "slot_status": self.status(),
            "platform": os.name,
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("diagnostic.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            for name, path in files:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                archive.writestr(name, _redact(text))
        return destination


def _redact(value: str) -> str:
    redacted = value
    for key in ("password", "secret", "token", "private_key", "authorization"):
        redacted = __import__("re").sub(
            rf'("?{key}"?\s*[:=]\s*)("[^"\n]*"|[^,\s}}]+)',
            rf'\1"<redacted>"', redacted, flags=__import__("re").IGNORECASE,
        )
    return redacted


def should_delegate(command: str | None) -> bool:
    return command not in _MAINTENANCE_COMMANDS
