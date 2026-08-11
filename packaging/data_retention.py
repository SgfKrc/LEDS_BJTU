"""User-data retention and re-association transactions for UP-N6.4/6.4W.

The installer owns the application tree, while these directories belong to the
user and must survive an uninstall/reinstall cycle. Same-filesystem handoffs
use atomic directory renames. Cross-filesystem handoffs use a recoverable,
hash-verified copy transaction and never delete a source directory before all
destination directories have been committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


APP_ID = "qlh-edge-inference"
RETENTION_SCHEMA_VERSION = 1
RETENTION_MARKER = ".qlh-retention.json"
TRANSACTION_SCHEMA_VERSION = 1
RETENTION_JOURNAL = ".qlh-retention-transaction.json"
USER_DATA_DIRS = ("models", "chat_history", "logs", "config", "local_docs")
COPY_BUFFER_SIZE = 4 * 1024 * 1024
MIN_FREE_RESERVE = 64 * 1024 * 1024
MAX_FREE_RESERVE = 1024 * 1024 * 1024
_TRANSACTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRANSACTION_PHASES = {"copying", "prepared", "committed", "deleting_source"}


class DataRetentionError(RuntimeError):
    """Expected fail-closed retention or re-association failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DataRetentionError(f"cannot inspect path: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        return True
    if os.name == "nt":
        # FILE_ATTRIBUTE_REPARSE_POINT.  lstat does not expose this bit on all
        # supported Python versions, so check it explicitly where available.
        attributes = getattr(metadata, "st_file_attributes", 0)
        if attributes & 0x400:
            return True
    return False


def _safe_existing_directory(value: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if _is_link_or_reparse(candidate):
        raise DataRetentionError(f"{label} is a link or reparse point: {candidate}")
    if not candidate.is_dir():
        raise DataRetentionError(f"{label} is not a directory: {candidate}")
    try:
        return candidate.resolve()
    except OSError as exc:
        raise DataRetentionError(f"cannot resolve {label}: {candidate}") from exc


def _safe_data_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists() and _is_link_or_reparse(candidate):
        raise DataRetentionError(f"data root is a link or reparse point: {candidate}")
    try:
        return candidate.resolve()
    except OSError as exc:
        raise DataRetentionError(f"cannot resolve data root: {candidate}") from exc


def default_data_root() -> Path:
    """Return the external, user-owned data root for the current platform."""
    override = os.environ.get("QLH_DATA_DIR", "").strip()
    if override:
        return _safe_data_root(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if not base:
            base = str(Path.home() / "AppData" / "Local")
        return _safe_data_root(Path(base) / "QLH-Edge-Inference" / "data")
    base = os.environ.get("XDG_DATA_HOME", "").strip()
    if not base:
        base = str(Path.home() / ".local" / "share")
    return _safe_data_root(Path(base) / "qlh-edge-inference" / "data")


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_roots(install_root: Path, data_root: Path) -> None:
    if install_root == data_root or _inside(data_root, install_root) or _inside(install_root, data_root):
        raise DataRetentionError("install root and data root must be separate directories")


def _entry_path(root: Path, name: str) -> Path:
    if name not in USER_DATA_DIRS:
        raise DataRetentionError(f"unsupported retained directory: {name}")
    return root / name


def _metadata_path(data_root: Path) -> Path:
    return data_root / RETENTION_MARKER


def _journal_path(data_root: Path) -> Path:
    return data_root / RETENTION_JOURNAL


def _write_json_atomic(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(destination.parent):
        raise DataRetentionError(f"metadata parent is a link or reparse point: {destination.parent}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".part", dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise DataRetentionError(f"cannot write transaction metadata: {destination}") from exc


def _write_marker(data_root: Path, *, state: str, present: Mapping[str, bool]) -> None:
    payload = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "app_id": APP_ID,
        "state": state,
        "updated_at": _utc_now(),
        "directories": [
            {"name": name, "present": bool(present.get(name, False))}
            for name in USER_DATA_DIRS
        ],
    }
    _write_json_atomic(_metadata_path(data_root), payload)


def _read_marker(data_root: Path) -> dict[str, Any]:
    marker = _metadata_path(data_root)
    if not marker.is_file() or _is_link_or_reparse(marker):
        raise DataRetentionError(f"retention marker is missing or unsafe: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataRetentionError("retention marker is unreadable") from exc
    if not isinstance(payload, dict):
        raise DataRetentionError("retention marker must be an object")
    if payload.get("schema_version") != RETENTION_SCHEMA_VERSION or payload.get("app_id") != APP_ID:
        raise DataRetentionError("retention marker schema or application id is not trusted")
    if payload.get("state") not in {"retained", "reassociated"}:
        raise DataRetentionError("retention marker has an unknown state")
    directories = payload.get("directories")
    if not isinstance(directories, list):
        raise DataRetentionError("retention marker directories are invalid")
    names = []
    for item in directories:
        # Inno Setup writes a compact string-only marker; the Python marker
        # includes a ``present`` bit.  Both forms are intentionally accepted.
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            name = None
        if name not in USER_DATA_DIRS:
            raise DataRetentionError("retention marker contains an unsupported directory")
        names.append(name)
    if len(names) != len(set(names)):
        raise DataRetentionError("retention marker contains duplicate directories")
    return payload


def _same_filesystem(first: Path, second: Path) -> bool:
    try:
        return first.stat().st_dev == second.stat().st_dev
    except OSError as exc:
        raise DataRetentionError("cannot determine retention filesystem boundary") from exc


def _path_identity(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _hash_regular_file(path: Path) -> tuple[int, str]:
    if _is_link_or_reparse(path):
        raise DataRetentionError(f"refusing linked or reparse-point file: {path}")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise DataRetentionError(f"unsupported non-regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
    except DataRetentionError:
        raise
    except OSError as exc:
        raise DataRetentionError(f"cannot read user-data file: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != before.st_size:
        raise DataRetentionError(f"user-data file changed while hashing: {path}")
    return size, digest.hexdigest()


def _relative_inventory_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    parsed = PurePosixPath(relative)
    if (
        not relative
        or relative == "."
        or "\\" in relative
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise DataRetentionError(f"unsafe user-data relative path: {relative!r}")
    return relative


def _inventory_directory(name: str, directory: Path) -> dict[str, Any]:
    if name not in USER_DATA_DIRS:
        raise DataRetentionError(f"unsupported retained directory: {name}")
    if not directory.is_dir() or _is_link_or_reparse(directory):
        raise DataRetentionError(f"user-data directory is missing or unsafe: {directory}")

    entries: list[dict[str, Any]] = []
    identities: set[str] = set()

    def add_identity(relative: str) -> None:
        identity = relative.casefold()
        if identity in identities:
            raise DataRetentionError(f"case-colliding user-data path: {relative}")
        identities.add(identity)

    def walk_error(exc: OSError) -> None:
        raise DataRetentionError(f"cannot enumerate user-data directory: {directory}") from exc

    for current_text, directory_names, file_names in os.walk(
        directory, topdown=True, followlinks=False, onerror=walk_error,
    ):
        current = Path(current_text)
        if _is_link_or_reparse(current):
            raise DataRetentionError(f"nested user-data directory is unsafe: {current}")
        directory_names.sort()
        file_names.sort()
        for child_name in directory_names:
            child = current / child_name
            if _is_link_or_reparse(child):
                raise DataRetentionError(f"nested user-data directory is unsafe: {child}")
            relative = _relative_inventory_path(child, directory)
            add_identity(relative)
            entries.append({"path": relative, "type": "directory"})
        for child_name in file_names:
            child = current / child_name
            relative = _relative_inventory_path(child, directory)
            add_identity(relative)
            size, digest = _hash_regular_file(child)
            entries.append({"path": relative, "type": "file", "size": size, "sha256": digest})

    entries.sort(key=lambda item: (item["path"].casefold(), item["path"], item["type"]))
    return {
        "name": name,
        "total_bytes": sum(item.get("size", 0) for item in entries),
        "entries": entries,
    }


def _validate_inventory_payload(directories: Any) -> list[dict[str, Any]]:
    if not isinstance(directories, list):
        raise DataRetentionError("retention journal directories are invalid")
    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for directory in directories:
        if not isinstance(directory, dict) or set(directory) != {"name", "total_bytes", "entries"}:
            raise DataRetentionError("retention journal directory entry is invalid")
        name = directory.get("name")
        total_bytes = directory.get("total_bytes")
        entries = directory.get("entries")
        if name not in USER_DATA_DIRS or name in names:
            raise DataRetentionError("retention journal contains an unsupported or duplicate directory")
        if not isinstance(total_bytes, int) or isinstance(total_bytes, bool) or total_bytes < 0:
            raise DataRetentionError("retention journal byte count is invalid")
        if not isinstance(entries, list):
            raise DataRetentionError("retention journal file inventory is invalid")
        names.add(name)
        paths: set[str] = set()
        calculated_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise DataRetentionError("retention journal inventory entry is invalid")
            entry_type = entry.get("type")
            expected_keys = {"path", "type"} if entry_type == "directory" else {
                "path", "type", "size", "sha256",
            }
            if set(entry) != expected_keys or entry_type not in {"directory", "file"}:
                raise DataRetentionError("retention journal inventory entry schema is invalid")
            relative = entry.get("path")
            if not isinstance(relative, str):
                raise DataRetentionError("retention journal inventory path is invalid")
            parsed = PurePosixPath(relative)
            if (
                not relative
                or relative == "."
                or "\\" in relative
                or parsed.is_absolute()
                or any(part in {"", ".", ".."} for part in parsed.parts)
            ):
                raise DataRetentionError("retention journal inventory path is unsafe")
            identity = relative.casefold()
            if identity in paths:
                raise DataRetentionError("retention journal contains colliding paths")
            paths.add(identity)
            if entry_type == "file":
                size = entry.get("size")
                digest = entry.get("sha256")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise DataRetentionError("retention journal file size is invalid")
                if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                    raise DataRetentionError("retention journal file digest is invalid")
                calculated_bytes += size
        if calculated_bytes != total_bytes:
            raise DataRetentionError("retention journal byte count does not match inventory")
        validated.append(directory)
    return validated


def _write_journal(data_root: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _utc_now()
    _write_json_atomic(_journal_path(data_root), payload)


def _read_journal(
    data_root: Path,
    install_root: Path,
    *,
    expected_operation: str | None = None,
) -> dict[str, Any]:
    journal = _journal_path(data_root)
    if not journal.is_file() or _is_link_or_reparse(journal):
        raise DataRetentionError(f"retention journal is missing or unsafe: {journal}")
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataRetentionError("retention journal is unreadable") from exc
    required = {
        "schema_version", "app_id", "transaction_id", "operation", "phase",
        "install_root", "data_root", "marker_present", "directories", "updated_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DataRetentionError("retention journal schema is invalid")
    if payload.get("schema_version") != TRANSACTION_SCHEMA_VERSION or payload.get("app_id") != APP_ID:
        raise DataRetentionError("retention journal identity is not trusted")
    transaction_id = payload.get("transaction_id")
    operation = payload.get("operation")
    phase = payload.get("phase")
    if not isinstance(transaction_id, str) or not _TRANSACTION_ID_RE.fullmatch(transaction_id):
        raise DataRetentionError("retention journal transaction id is invalid")
    if operation not in {"retain", "reassociate"} or (
        expected_operation is not None and operation != expected_operation
    ):
        raise DataRetentionError("retention journal operation does not match the requested action")
    if phase not in _TRANSACTION_PHASES:
        raise DataRetentionError("retention journal phase is invalid")
    journal_install = payload.get("install_root")
    journal_data = payload.get("data_root")
    if not isinstance(journal_install, str) or not isinstance(journal_data, str):
        raise DataRetentionError("retention journal roots are invalid")
    if not Path(journal_install).is_absolute() or not Path(journal_data).is_absolute():
        raise DataRetentionError("retention journal roots must be absolute")
    if (
        _path_identity(Path(journal_install)) != _path_identity(install_root)
        or _path_identity(Path(journal_data)) != _path_identity(data_root)
    ):
        raise DataRetentionError("retention journal roots do not match the current installation")
    marker_present = payload.get("marker_present")
    if (
        not isinstance(marker_present, list)
        or any(not isinstance(name, str) for name in marker_present)
        or any(name not in USER_DATA_DIRS for name in marker_present)
        or len(marker_present) != len(set(marker_present))
    ):
        raise DataRetentionError("retention journal marker directory list is invalid")
    if not isinstance(payload.get("updated_at"), str) or not payload["updated_at"]:
        raise DataRetentionError("retention journal timestamp is invalid")
    _validate_inventory_payload(payload.get("directories"))
    return payload


def _transaction_roots(payload: Mapping[str, Any], install_root: Path, data_root: Path) -> tuple[Path, Path]:
    if payload["operation"] == "retain":
        return install_root, data_root
    return data_root, install_root


def _transaction_staging_root(destination_root: Path, transaction_id: str) -> Path:
    return destination_root / f".qlh-retention-staging-{transaction_id}"


def _transaction_delete_root(source_root: Path, transaction_id: str) -> Path:
    return source_root / f".qlh-retention-delete-{transaction_id}"


def _verify_directory_inventory(directory: Path, expected: Mapping[str, Any]) -> None:
    actual = _inventory_directory(str(expected["name"]), directory)
    if actual["total_bytes"] != expected["total_bytes"] or actual["entries"] != expected["entries"]:
        raise DataRetentionError(f"copied user-data inventory does not match: {expected['name']}")


def _copy_file_verified(
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    transaction_id: str,
) -> None:
    if destination.exists() or destination.is_symlink():
        size, digest = _hash_regular_file(destination)
        if size == expected["size"] and digest == expected["sha256"]:
            return
        if destination.is_dir():
            raise DataRetentionError(f"staging file path is a directory: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(destination.parent):
        raise DataRetentionError(f"staging parent is unsafe: {destination.parent}")
    temporary = destination.with_name(f".{destination.name}.{transaction_id}.part")
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_dir() or _is_link_or_reparse(temporary):
            raise DataRetentionError(f"staging temporary path is unsafe: {temporary}")
        temporary.unlink()

    digest = hashlib.sha256()
    copied = 0
    try:
        before = source.stat()
        if _is_link_or_reparse(source) or not stat.S_ISREG(before.st_mode):
            raise DataRetentionError(f"source file is unsafe: {source}")
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while True:
                chunk = reader.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise DataRetentionError(f"source file changed while copying: {source}")
        if copied != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise DataRetentionError(f"source file no longer matches transaction inventory: {source}")
        os.replace(temporary, destination)
    except DataRetentionError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise DataRetentionError(f"could not copy user-data file: {source}") from exc


def _copy_transaction_directories(
    payload: Mapping[str, Any],
    source_root: Path,
    destination_root: Path,
) -> None:
    transaction_id = str(payload["transaction_id"])
    staging_root = _transaction_staging_root(destination_root, transaction_id)
    staging_root.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(staging_root):
        raise DataRetentionError(f"transaction staging root is unsafe: {staging_root}")
    for inventory in payload["directories"]:
        name = str(inventory["name"])
        source = _entry_path(source_root, name)
        staged = _entry_path(staging_root, name)
        if staged.exists() and (not staged.is_dir() or _is_link_or_reparse(staged)):
            raise DataRetentionError(f"transaction staging directory is unsafe: {staged}")
        staged.mkdir(parents=True, exist_ok=True)
        for entry in inventory["entries"]:
            relative = Path(*PurePosixPath(entry["path"]).parts)
            staged_path = staged / relative
            if entry["type"] == "directory":
                if staged_path.exists() and (not staged_path.is_dir() or _is_link_or_reparse(staged_path)):
                    raise DataRetentionError(f"transaction staging path is unsafe: {staged_path}")
                staged_path.mkdir(parents=True, exist_ok=True)
            else:
                _copy_file_verified(source / relative, staged_path, entry, transaction_id)
        _verify_directory_inventory(staged, inventory)


def _preflight_cross_volume_space(destination_root: Path, directories: list[dict[str, Any]]) -> None:
    total_bytes = sum(directory["total_bytes"] for directory in directories)
    reserve = max(MIN_FREE_RESERVE, min(MAX_FREE_RESERVE, total_bytes // 100))
    try:
        free_bytes = shutil.disk_usage(destination_root).free
    except OSError as exc:
        raise DataRetentionError("cannot determine destination free space") from exc
    required = total_bytes + reserve
    if free_bytes < required:
        raise DataRetentionError(
            f"insufficient destination space for verified retention: required={required} free={free_bytes}"
        )


def _commit_staged_directories(
    payload: Mapping[str, Any],
    source_root: Path,
    destination_root: Path,
) -> None:
    del source_root  # Root identity is already bound by the validated journal.
    staging_root = _transaction_staging_root(destination_root, str(payload["transaction_id"]))
    for inventory in payload["directories"]:
        name = str(inventory["name"])
        staged = _entry_path(staging_root, name)
        destination = _entry_path(destination_root, name)
        staged_exists = staged.exists() or staged.is_symlink()
        destination_exists = destination.exists() or destination.is_symlink()
        if staged_exists and destination_exists:
            if payload["operation"] == "reassociate" and destination.is_dir() and not _is_link_or_reparse(destination):
                try:
                    next(destination.iterdir())
                except StopIteration:
                    destination.rmdir()
                    destination_exists = False
            if destination_exists:
                raise DataRetentionError(f"destination appeared during transaction commit: {destination}")
        if staged_exists:
            if not staged.is_dir() or _is_link_or_reparse(staged):
                raise DataRetentionError(f"transaction staging directory is unsafe: {staged}")
            _verify_directory_inventory(staged, inventory)
            os.replace(staged, destination)
        elif not destination_exists:
            raise DataRetentionError(f"transaction directory is missing from staging and destination: {name}")
        _verify_directory_inventory(destination, inventory)
    if staging_root.exists():
        try:
            staging_root.rmdir()
        except OSError as exc:
            raise DataRetentionError(f"transaction staging root is not empty: {staging_root}") from exc


def _delete_committed_sources(
    payload: Mapping[str, Any],
    source_root: Path,
    destination_root: Path,
) -> None:
    transaction_id = str(payload["transaction_id"])
    delete_root = _transaction_delete_root(source_root, transaction_id)
    delete_root.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(delete_root):
        raise DataRetentionError(f"transaction delete root is unsafe: {delete_root}")
    for inventory in payload["directories"]:
        name = str(inventory["name"])
        source = _entry_path(source_root, name)
        destination = _entry_path(destination_root, name)
        quarantined = _entry_path(delete_root, name)
        _verify_directory_inventory(destination, inventory)
        source_exists = source.exists() or source.is_symlink()
        quarantined_exists = quarantined.exists() or quarantined.is_symlink()
        if source_exists and quarantined_exists:
            raise DataRetentionError(f"source and delete quarantine both exist: {name}")
        if source_exists:
            _verify_directory_inventory(source, inventory)
            os.replace(source, quarantined)
            quarantined_exists = True
        if quarantined_exists:
            _verify_directory_inventory(quarantined, inventory)
            try:
                shutil.rmtree(quarantined)
            except OSError as exc:
                raise DataRetentionError(f"could not remove verified source directory: {name}") from exc
    try:
        delete_root.rmdir()
    except OSError as exc:
        raise DataRetentionError(f"transaction delete root is not empty: {delete_root}") from exc


def _resume_cross_volume_transaction(
    install_root: Path,
    data_root: Path,
    *,
    expected_operation: str,
    recovered: bool = True,
) -> dict[str, Any]:
    payload = _read_journal(data_root, install_root, expected_operation=expected_operation)
    source_root, destination_root = _transaction_roots(payload, install_root, data_root)
    phase = payload["phase"]
    try:
        if phase == "copying":
            _copy_transaction_directories(payload, source_root, destination_root)
            payload["phase"] = "prepared"
            _write_journal(data_root, payload)
            phase = "prepared"
        if phase == "prepared":
            _commit_staged_directories(payload, source_root, destination_root)
            payload["phase"] = "committed"
            _write_journal(data_root, payload)
            phase = "committed"
        if phase == "committed":
            marker_present = {name: name in payload["marker_present"] for name in USER_DATA_DIRS}
            marker_state = "retained" if payload["operation"] == "retain" else "reassociated"
            _write_marker(data_root, state=marker_state, present=marker_present)
            payload["phase"] = "deleting_source"
            _write_journal(data_root, payload)
            phase = "deleting_source"
        if phase == "deleting_source":
            _delete_committed_sources(payload, source_root, destination_root)
            _journal_path(data_root).unlink()
    except DataRetentionError:
        raise
    except OSError as exc:
        raise DataRetentionError(f"cross-volume retention failed during phase: {phase}") from exc

    action = "retained" if payload["operation"] == "retain" else "reassociated"
    result_directories = payload["marker_present"] if action == "retained" else [
        directory["name"] for directory in payload["directories"]
    ]
    return {
        "ok": True,
        "action": action,
        "data_root": str(data_root),
        "directories": result_directories,
        "transfer_mode": "verified-copy",
        "recovered": recovered,
    }


def _start_cross_volume_transaction(
    install_root: Path,
    data_root: Path,
    *,
    operation: str,
    source_directories: list[str],
    marker_present: list[str],
) -> dict[str, Any]:
    source_root = install_root if operation == "retain" else data_root
    destination_root = data_root if operation == "retain" else install_root
    directories = [_inventory_directory(name, _entry_path(source_root, name)) for name in source_directories]
    _preflight_cross_volume_space(destination_root, directories)
    payload: dict[str, Any] = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "app_id": APP_ID,
        "transaction_id": uuid.uuid4().hex,
        "operation": operation,
        "phase": "copying",
        "install_root": str(install_root),
        "data_root": str(data_root),
        "marker_present": marker_present,
        "directories": directories,
        "updated_at": _utc_now(),
    }
    _write_journal(data_root, payload)
    return _resume_cross_volume_transaction(
        install_root, data_root, expected_operation=operation, recovered=False,
    )


def retention_status(
    install_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read only status; this never reads user-data contents."""
    install = _safe_existing_directory(install_root, "install root")
    data = _safe_data_root(data_root or default_data_root())
    _validate_roots(install, data)
    marker = _metadata_path(data)
    result: dict[str, Any] = {
        "ok": True,
        "install_root": str(install),
        "data_root": str(data),
        "marker": marker.is_file(),
        "state": "none",
        "directories": {},
        "transaction": None,
    }
    journal = _journal_path(data)
    if journal.exists() or journal.is_symlink():
        transaction = _read_journal(data, install)
        result["transaction"] = {
            "operation": transaction["operation"],
            "phase": transaction["phase"],
            "directories": [directory["name"] for directory in transaction["directories"]],
        }
    if marker.is_file():
        payload = _read_marker(data)
        result["state"] = payload["state"]
    for name in USER_DATA_DIRS:
        source = _entry_path(install, name)
        retained = _entry_path(data, name)
        result["directories"][name] = {
            "install": source.is_dir() if not _is_link_or_reparse(source) else False,
            "retained": retained.is_dir() if not _is_link_or_reparse(retained) else False,
        }
    return result


def retain_user_data(
    install_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
    *,
    confirm: bool = False,
) -> dict[str, Any]:
    """Move user directories out of the install tree before uninstall."""
    if not confirm:
        raise DataRetentionError("retaining user data requires explicit confirmation")
    install = _safe_existing_directory(install_root, "install root")
    data = _safe_data_root(data_root or default_data_root())
    _validate_roots(install, data)
    data.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(data):
        raise DataRetentionError("data root is a link or reparse point")
    if _journal_path(data).exists() or _journal_path(data).is_symlink():
        return _resume_cross_volume_transaction(
            install, data, expected_operation="retain",
        )

    moves: list[tuple[Path, Path]] = []
    present: dict[str, bool] = {}
    for name in USER_DATA_DIRS:
        source = _entry_path(install, name)
        destination = _entry_path(data, name)
        source_exists = source.exists() or source.is_symlink()
        destination_exists = destination.exists() or destination.is_symlink()
        if source_exists and _is_link_or_reparse(source):
            raise DataRetentionError(f"refusing to move linked user directory: {source}")
        if destination_exists and _is_link_or_reparse(destination):
            raise DataRetentionError(f"retained directory is unsafe: {destination}")
        if source_exists and not source.is_dir():
            raise DataRetentionError(f"user data entry is not a directory: {source}")
        if destination_exists and not destination.is_dir():
            raise DataRetentionError(f"retained entry is not a directory: {destination}")
        if source_exists and destination_exists:
            raise DataRetentionError(f"retained directory already exists; refusing merge: {name}")
        present[name] = bool(source_exists or destination_exists)
        if source_exists:
            moves.append((source, destination))

    if moves and not _same_filesystem(install, data):
        return _start_cross_volume_transaction(
            install,
            data,
            operation="retain",
            source_directories=[source.name for source, _ in moves],
            marker_present=[name for name in USER_DATA_DIRS if present.get(name)],
        )

    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination in moves:
            os.replace(source, destination)
            moved.append((source, destination))
        _write_marker(data, state="retained", present=present)
    except (OSError, DataRetentionError) as exc:
        for source, destination in reversed(moved):
            try:
                if destination.exists() and not source.exists():
                    os.replace(destination, source)
            except OSError:
                pass
        if isinstance(exc, DataRetentionError):
            raise
        raise DataRetentionError("could not retain user data atomically") from exc

    return {
        "ok": True,
        "action": "retained",
        "data_root": str(data),
        "directories": [name for name in USER_DATA_DIRS if present.get(name)],
    }


def reassociate_user_data(
    install_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
    *,
    confirm: bool = False,
) -> dict[str, Any]:
    """Move retained directories into a freshly installed application tree."""
    if not confirm:
        raise DataRetentionError("re-associating user data requires explicit confirmation")
    install = _safe_existing_directory(install_root, "install root")
    data = _safe_data_root(data_root or default_data_root())
    _validate_roots(install, data)
    if _journal_path(data).exists() or _journal_path(data).is_symlink():
        return _resume_cross_volume_transaction(
            install, data, expected_operation="reassociate",
        )
    marker = _metadata_path(data)
    if not marker.exists() and not marker.is_symlink():
        return {"ok": True, "action": "none", "data_root": str(data), "directories": []}
    payload = _read_marker(data)
    if payload["state"] != "retained":
        return {"ok": True, "action": "already-reassociated", "data_root": str(data), "directories": []}

    moves: list[tuple[Path, Path]] = []
    empty_targets: list[Path] = []
    names = {
        item if isinstance(item, str) else item["name"]
        for item in payload["directories"]
        if isinstance(item, str) or item.get("present")
    }
    for name in USER_DATA_DIRS:
        if name not in names:
            continue
        source = _entry_path(data, name)
        destination = _entry_path(install, name)
        if not source.is_dir() or _is_link_or_reparse(source):
            raise DataRetentionError(f"retained directory is missing or unsafe: {source}")
        if destination.exists() or destination.is_symlink():
            if _is_link_or_reparse(destination):
                raise DataRetentionError(f"install user directory is unsafe: {destination}")
            if not destination.is_dir():
                raise DataRetentionError(f"install user entry is not a directory: {destination}")
            try:
                next(destination.iterdir())
            except StopIteration:
                empty_targets.append(destination)
            else:
                raise DataRetentionError(f"fresh install contains non-empty user directory: {name}")
        moves.append((source, destination))

    if moves and not _same_filesystem(install, data):
        return _start_cross_volume_transaction(
            install,
            data,
            operation="reassociate",
            source_directories=[source.name for source, _ in moves],
            marker_present=[],
        )

    moved: list[tuple[Path, Path]] = []
    try:
        for target in empty_targets:
            target.rmdir()
        for source, destination in moves:
            os.replace(source, destination)
            moved.append((source, destination))
        _write_marker(data, state="reassociated", present={name: False for name in USER_DATA_DIRS})
    except (OSError, DataRetentionError) as exc:
        for source, destination in reversed(moved):
            try:
                if destination.exists() and not source.exists():
                    os.replace(destination, source)
            except OSError:
                pass
        for target in empty_targets:
            try:
                if not target.exists():
                    target.mkdir()
            except OSError:
                pass
        if isinstance(exc, DataRetentionError):
            raise
        raise DataRetentionError("could not re-associate user data atomically") from exc
    return {
        "ok": True,
        "action": "reassociated",
        "data_root": str(data),
        "directories": [str(destination.name) for _, destination in moved],
    }


def auto_reassociate_user_data(
    install_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Complete a previously confirmed reinstall handoff, if one exists."""
    try:
        data = _safe_data_root(data_root or default_data_root())
    except RuntimeError:
        # Platform-simulation tests and restricted service accounts can lack a
        # resolvable home directory.  Without a data root there is no marker
        # to consume, so this must not block a normal application launch.
        return {"ok": True, "action": "none", "directories": []}
    marker = _metadata_path(data)
    if not marker.exists() and not marker.is_symlink():
        return {"ok": True, "action": "none", "directories": []}
    payload = _read_marker(data)
    if payload["state"] != "retained":
        return {"ok": True, "action": "none", "directories": []}
    return reassociate_user_data(install_root, data, confirm=True)


def _print(value: Mapping[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlh-data-retention")
    parser.add_argument("command", choices=("status", "retain", "reassociate"))
    parser.add_argument("--root", required=True, help="application install root")
    parser.add_argument("--data-root", help="external user-data root")
    parser.add_argument("--yes", action="store_true", help="confirm moving user data")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = retention_status(args.root, args.data_root)
        elif args.command == "retain":
            result = retain_user_data(args.root, args.data_root, confirm=args.yes)
        else:
            result = reassociate_user_data(args.root, args.data_root, confirm=args.yes)
    except DataRetentionError as exc:
        result = {"ok": False, "action": "failed", "error": str(exc)}
    _print(result, args.as_json)
    return 0 if result.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
