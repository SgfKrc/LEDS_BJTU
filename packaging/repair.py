"""Fail-closed single-file application repair for UP-N6.3.

Repair is deliberately separate from diagnosis and normal startup.  It only
writes paths listed by the locally verified install manifest, and only after a
signed update manifest has pinned a same-version repair index.  User data is
therefore outside both enumeration and mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urljoin, urlparse

from install_manifest import (
    MANIFEST_RELATIVE_PATH,
    InstallManifestError,
    normalize_relative_path,
    resolve_install_manifest_path,
    verify_install_manifest,
    verify_install_tree,
    load_install_manifest,
)
from signing import default_trusted_keys_dir
from update_core import (
    UpdateAsset,
    UpdateError,
    UpdateManifest,
    default_state_dir,
    download_asset,
    fetch_latest,
    select_asset,
)


REPAIR_SCHEMA_VERSION = 1
REPAIR_INDEX_SCHEMA_VERSION = 1
REPAIR_INDEX_TYPE = "qlh_repair_index"
MAX_REPAIR_FILES = 10
MAX_REPAIR_BYTES = 64 * 1024 * 1024
MAX_REPAIR_INDEX_BYTES = 64 * 1024 * 1024
_SHA256_LENGTH = 64
_REPAIRABLE_CATEGORIES = frozenset({"missing", "size", "hash"})


class RepairError(UpdateError):
    """Expected, non-destructive repair failure."""


@dataclass(frozen=True)
class RepairFile:
    path: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True)
class _PreparedFile:
    repair_file: RepairFile
    target: Path
    staged: Path


@dataclass(frozen=True)
class _AppliedFile:
    prepared: _PreparedFile
    backup: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trusted_keys_dir(root: Path, supplied: str | os.PathLike[str] | None) -> str | os.PathLike[str]:
    if supplied is not None:
        return supplied
    bundled = root / "pubkeys"
    if bundled.is_dir() and not bundled.is_symlink():
        return bundled
    return default_trusted_keys_dir()


def _read_verified_manifest(
    root: Path, trusted_keys_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    manifest_path = root / Path(MANIFEST_RELATIVE_PATH)
    try:
        return verify_install_manifest(
            load_install_manifest(manifest_path),
            trusted_keys_dir=_trusted_keys_dir(root, trusted_keys_dir),
        )
    except InstallManifestError as exc:
        raise RepairError(f"local install manifest is not trusted: {exc}") from exc


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: str(manifest[key])
        for key in ("app_id", "version", "platform", "variant", "package_kind", "key_id")
    }


def _summary(report: Mapping[str, Any]) -> dict[str, int]:
    value = report.get("summary")
    return dict(value) if isinstance(value, Mapping) else {}


def _report(
    root: Path,
    *,
    action: str,
    ok: bool,
    manifest: Mapping[str, Any] | None = None,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    candidates: Iterable[Mapping[str, Any]] = (),
    repaired: Iterable[str] = (),
    advice: Iterable[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "command": "repair",
        "root": str(root),
        "action": action,
        "ok": ok,
        "manifest": _manifest_identity(manifest) if manifest else None,
        "verification_before": _summary(before or {}),
        "verification_after": _summary(after or {}),
        "candidates": [dict(item) for item in candidates],
        "repaired": list(repaired),
        "advice": list(advice),
    }
    if error:
        result["error"] = error
    return result


def _is_safe_origin_path(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        return False
    if "\\" in value or any(ord(character) < 32 for character in value):
        return False
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return False
    return all(part not in {"", ".", ".."} for part in parsed.path.split("/")[1:])


def _load_repair_index(path: Path, *, index_url: str, manifest: Mapping[str, Any]) -> list[RepairFile]:
    try:
        if path.stat().st_size > MAX_REPAIR_INDEX_BYTES:
            raise RepairError("repair index exceeds the size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"cannot read repair index: {exc}") from exc
    if not isinstance(value, dict):
        raise RepairError("repair index root must be an object")
    required = {"schema_version", "manifest_type", "app_id", "version", "platform", "variant", "files"}
    if set(value) != required:
        raise RepairError("repair index fields are invalid")
    if value["schema_version"] != REPAIR_INDEX_SCHEMA_VERSION or value["manifest_type"] != REPAIR_INDEX_TYPE:
        raise RepairError("unsupported repair index schema")
    for key in ("app_id", "version", "platform", "variant"):
        if value.get(key) != manifest.get(key):
            raise RepairError(f"repair index {key} does not match the installed release")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RepairError("repair index files must be a non-empty list")
    baseline = {str(entry["path"]): entry for entry in manifest["files"]}
    files: list[RepairFile] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256", "url"}:
            raise RepairError("repair index file entry has invalid fields")
        try:
            relative_path = normalize_relative_path(item["path"])
            size = item["size"]
            sha256 = item["sha256"]
        except InstallManifestError as exc:
            raise RepairError(f"invalid repair index path: {exc}") from exc
        if (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not isinstance(sha256, str) or len(sha256) != _SHA256_LENGTH
            or any(char not in "0123456789abcdef" for char in sha256)
        ):
            raise RepairError(f"invalid repair index metadata: {relative_path}")
        if relative_path.casefold() in seen:
            raise RepairError(f"duplicate repair index path: {relative_path}")
        seen.add(relative_path.casefold())
        if relative_path not in baseline:
            raise RepairError(f"repair index contains a path outside the local baseline: {relative_path}")
        expected = baseline[relative_path]
        if expected["size"] != size or expected["sha256"] != sha256:
            raise RepairError(f"repair index content does not match the local baseline: {relative_path}")
        url = item["url"]
        if not _is_safe_origin_path(url):
            raise RepairError(f"repair index URL is unsafe: {relative_path}")
        resolved = urljoin(index_url, url)
        resolved_parts = urlparse(resolved)
        index_parts = urlparse(index_url)
        if (
            resolved_parts.scheme != index_parts.scheme
            or resolved_parts.netloc != index_parts.netloc
        ):
            raise RepairError(f"repair payload leaves the trusted source: {relative_path}")
        files.append(RepairFile(relative_path, size, sha256, resolved))
    return files


def _select_current_repair_index(
    sources: list[str],
    *,
    profile: Mapping[str, str],
    manifest: Mapping[str, Any],
    timeout: float,
    trusted_keys_dir: str | os.PathLike[str] | None,
    fetcher: Callable[..., UpdateManifest] | None,
) -> tuple[UpdateManifest, tuple[str, ...], UpdateAsset]:
    from update_core import fetch_manifest

    fetch = fetcher or (
        lambda url, timeout: fetch_manifest(
            url, timeout=timeout, trusted_keys_dir=trusted_keys_dir or default_trusted_keys_dir(),
        )
    )
    update_manifest, failures = fetch_latest(sources, timeout=timeout, fetcher=fetch)
    if not update_manifest.signature_verified:
        reason = update_manifest.signature_error or "signature is missing or untrusted"
        raise RepairError(f"repair update manifest is not trusted: {reason}")
    if update_manifest.tag != manifest["version"]:
        raise RepairError(
            "repair assets only support the installed version; use full update or rollback first"
        )
    try:
        index_asset = select_asset(
            update_manifest,
            platform=str(profile["platform"]),
            variant=str(profile["variant"]),
            arch=str(profile["arch"]),
            kind="repair-index",
        )
    except UpdateError as exc:
        raise RepairError("no signed repair index is published for this installation") from exc
    return update_manifest, failures, index_asset


def _hash_matches(path: Path, *, size: int, sha256: str) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == sha256
    except OSError:
        return False


def _copy_with_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _copy_verified_to_target_parent(source: Path, target: Path, item: RepairFile) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.qlh-repair-{uuid.uuid4().hex}.tmp"
    try:
        _copy_with_fsync(source, temporary)
        if not _hash_matches(temporary, size=item.size, sha256=item.sha256):
            raise RepairError(f"staged repair file verification failed: {item.path}")
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _backup_path(root: Path, counter: int) -> Path:
    return root / f"{counter:03d}-{uuid.uuid4().hex}.bak"


def _backup_existing(target: Path, backup_root: Path, counter: int) -> Path | None:
    try:
        metadata = target.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RepairError(f"cannot stat existing repair target: {target.name}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RepairError(f"repair target is not a regular file: {target.name}")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(backup_root, counter)
    try:
        _copy_with_fsync(target, backup)
    except OSError as exc:
        backup.unlink(missing_ok=True)
        raise RepairError(f"cannot create repair backup: {target.name}: {exc}") from exc
    return backup


def _restore_backup(target: Path, backup: Path) -> None:
    temporary = target.parent / f".{target.name}.qlh-rollback-{uuid.uuid4().hex}.tmp"
    try:
        _copy_with_fsync(backup, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _rollback(install_root: Path, applied: list[_AppliedFile]) -> None:
    for item in reversed(applied):
        try:
            target = resolve_install_manifest_path(
                install_root, item.prepared.repair_file.path,
            )
            if item.backup is not None:
                _restore_backup(target, item.backup)
            elif _hash_matches(
                target,
                size=item.prepared.repair_file.size,
                sha256=item.prepared.repair_file.sha256,
            ):
                target.unlink()
        except OSError:
            # Keep the original repair error as the actionable failure.  A
            # retained .bak snapshot is safer than an over-broad cleanup.
            continue


def _candidate_failures(
    before: Mapping[str, Any], manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    entries = {str(entry["path"]): entry for entry in manifest["files"]}
    candidates: list[dict[str, Any]] = []
    blockers: list[str] = []
    failures = before.get("failed")
    if not isinstance(failures, list):
        return candidates, ["verification report is malformed"]
    seen: set[str] = set()
    for failure in failures:
        if not isinstance(failure, Mapping):
            blockers.append("verification report contains an invalid failure")
            continue
        category = str(failure.get("category", ""))
        path = str(failure.get("path", ""))
        if category not in _REPAIRABLE_CATEGORIES or path not in entries:
            blockers.append(f"{path or '.'}:{category or 'unknown'}")
            continue
        if path in seen:
            continue
        seen.add(path)
        entry = entries[path]
        candidates.append({"path": path, "size": entry["size"], "sha256": entry["sha256"]})
    return candidates, blockers


def repair_install(
    root: str | os.PathLike[str],
    *,
    sources: list[str],
    profile: Mapping[str, str],
    trusted_keys_dir: str | os.PathLike[str] | None = None,
    timeout: float = 8.0,
    download_dir: str | os.PathLike[str] | None = None,
    backup_dir: str | os.PathLike[str] | None = None,
    fetcher: Callable[..., UpdateManifest] | None = None,
    downloader: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    """Repair signed application files from same-version signed assets.

    The function always performs a deep preflight because quick/full checks do
    not enumerate every modification.  It downloads every required payload
    before replacing the first file, then deep-verifies the result.
    """
    install_root = Path(root).expanduser()
    if not install_root.is_dir() or install_root.is_symlink():
        raise RepairError("install root is not a regular directory")
    install_root = install_root.resolve()
    manifest = _read_verified_manifest(install_root, trusted_keys_dir)
    if manifest["app_id"] != "qlh-edge-inference" or manifest["package_kind"] != "application":
        raise RepairError("repair only supports signed QLH application installations")
    # The installed signed baseline, rather than GPU auto-detection or a UI
    # preference, selects repair assets.  A CPU package on a CUDA host must
    # never receive CUDA payloads.
    profile = {
        **dict(profile),
        "platform": str(manifest["platform"]),
        "variant": str(manifest["variant"]),
    }
    if not profile.get("arch"):
        raise RepairError("repair profile is missing an architecture")
    before = verify_install_tree(install_root, level="deep", trusted_keys_dir=trusted_keys_dir)
    if not before["ok"] and any(
        item.get("category") == "signature"
        for item in before.get("failed", []) if isinstance(item, Mapping)
    ):
        return _report(
            install_root, action="blocked", ok=False, manifest=manifest, before=before,
            advice=("The local install manifest is not trusted; use a signed full installer.",),
        )
    if before["ok"]:
        return _report(
            install_root, action="none", ok=True, manifest=manifest, before=before,
            advice=("No signed application file needs repair.",),
        )
    candidates, blockers = _candidate_failures(before, manifest)
    if blockers or not candidates:
        return _report(
            install_root, action="escalate", ok=False, manifest=manifest, before=before,
            candidates=candidates,
            advice=("Repair only handles missing, size, or hash failures in signed program files. Use a signed full installer.",),
            error="unsupported integrity failures: " + ", ".join(blockers or ["none"]),
        )
    total_bytes = sum(int(item["size"]) for item in candidates)
    if len(candidates) > MAX_REPAIR_FILES or total_bytes > MAX_REPAIR_BYTES:
        return _report(
            install_root, action="escalate", ok=False, manifest=manifest, before=before,
            candidates=candidates,
            advice=("The repair set exceeds the single-file repair threshold. Use a signed full installer.",),
            error=f"repair threshold exceeded: files={len(candidates)}, bytes={total_bytes}",
        )
    if not sources:
        raise RepairError("no repair update source is configured")
    _, _, index_asset = _select_current_repair_index(
        sources, profile=profile, manifest=manifest, timeout=timeout,
        trusted_keys_dir=trusted_keys_dir, fetcher=fetcher,
    )
    work_root = Path(download_dir).expanduser() if download_dir else default_state_dir() / "repair-downloads"
    work_root.mkdir(parents=True, exist_ok=True)
    fetch_file = downloader or download_asset
    try:
        index_path = fetch_file(index_asset, work_root, timeout=max(30.0, timeout))
    except UpdateError as exc:
        raise RepairError(f"cannot download the signed repair index: {exc}") from exc
    index_files = _load_repair_index(index_path, index_url=index_asset.url, manifest=manifest)
    indexed = {item.path: item for item in index_files}
    prepared: list[_PreparedFile] = []
    for candidate in candidates:
        relative_path = str(candidate["path"])
        item = indexed.get(relative_path)
        if item is None:
            raise RepairError(f"repair index has no payload for: {relative_path}")
        asset = UpdateAsset(
            name=f"repair-{item.sha256}", url=item.url, size=item.size,
            sha256=item.sha256, platform=str(profile["platform"]),
            variant=str(profile["variant"]), arch=str(profile["arch"]), kind="repair-file",
        )
        try:
            staged = fetch_file(asset, work_root, timeout=max(30.0, timeout))
        except UpdateError as exc:
            raise RepairError(f"cannot download repair payload for {relative_path}: {exc}") from exc
        if not _hash_matches(staged, size=item.size, sha256=item.sha256):
            raise RepairError(f"repair payload verification failed: {relative_path}")
        target = resolve_install_manifest_path(install_root, relative_path)
        prepared.append(_PreparedFile(item, target, staged))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = Path(backup_dir).expanduser() if backup_dir else default_state_dir() / "repair-backups" / stamp
    applied: list[_AppliedFile] = []
    try:
        for counter, item in enumerate(prepared, start=1):
            # Resolve again after downloads so a path changed into a link is not written.
            target = resolve_install_manifest_path(install_root, item.repair_file.path)
            backup = _backup_existing(target, backups, counter)
            temporary = _copy_verified_to_target_parent(item.staged, target, item.repair_file)
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            if not _hash_matches(target, size=item.repair_file.size, sha256=item.repair_file.sha256):
                raise RepairError(f"atomic replacement verification failed: {item.repair_file.path}")
            applied.append(_AppliedFile(_PreparedFile(item.repair_file, target, item.staged), backup))
        after = verify_install_tree(install_root, level="deep", trusted_keys_dir=trusted_keys_dir)
        if not after["ok"]:
            _rollback(install_root, applied)
            return _report(
                install_root, action="failed", ok=False, manifest=manifest, before=before, after=after,
                candidates=candidates,
                advice=("Post-repair verification failed; original files were restored from .bak snapshots. Use a signed full installer.",),
            )
    except Exception as exc:
        _rollback(install_root, applied)
        if isinstance(exc, RepairError):
            raise
        raise RepairError(f"repair replacement failed: {exc}") from exc
    return _report(
        install_root, action="repaired", ok=True, manifest=manifest, before=before, after=after,
        candidates=candidates, repaired=(item.repair_file.path for item in prepared),
        advice=("Repair completed and deep verification passed.",),
    )


def build_repair_index(
    root: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str],
    payload_dir: str | os.PathLike[str],
    url_prefix: str,
    trusted_keys_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Create publisher-side payload copies and a same-release repair index."""
    source_root = Path(root).expanduser().resolve()
    manifest = _read_verified_manifest(source_root, trusted_keys_dir)
    if manifest["app_id"] != "qlh-edge-inference" or manifest["package_kind"] != "application":
        raise RepairError("repair index source must be a signed QLH application tree")
    verified = verify_install_tree(source_root, level="deep", trusted_keys_dir=trusted_keys_dir)
    if not verified["ok"]:
        raise RepairError("repair index source fails deep verification")
    if not _is_safe_origin_path(url_prefix.rstrip("/")):
        raise RepairError("repair index URL prefix must be an absolute safe path")
    prefix = url_prefix.rstrip("/")
    payload_root = Path(payload_dir).expanduser()
    payload_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        relative_path = str(entry["path"])
        source = resolve_install_manifest_path(source_root, relative_path)
        digest_name = str(entry["sha256"])
        destination = payload_root / digest_name
        if not _hash_matches(destination, size=entry["size"], sha256=digest_name):
            destination.unlink(missing_ok=True)
            _copy_with_fsync(source, destination)
        if not _hash_matches(destination, size=entry["size"], sha256=digest_name):
            raise RepairError(f"cannot verify generated repair payload: {relative_path}")
        files.append({
            "path": relative_path,
            "size": entry["size"],
            "sha256": digest_name,
            "url": f"{prefix}/{digest_name}",
        })
    index = {
        "schema_version": REPAIR_INDEX_SCHEMA_VERSION,
        "manifest_type": REPAIR_INDEX_TYPE,
        "app_id": manifest["app_id"],
        "version": manifest["version"],
        "platform": manifest["platform"],
        "variant": manifest["variant"],
        "files": files,
    }
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_raw = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def format_repair(report: Mapping[str, Any]) -> str:
    action = str(report.get("action", "failed"))
    if action == "repaired":
        return f"Repair completed: {len(report.get('repaired', []))} file(s); deep verification passed."
    if action == "none":
        return "No signed application file needs repair."
    if action == "escalate":
        return "Single-file repair was not run; use a signed full installer."
    if action == "blocked":
        return "Repair is blocked because the local signed baseline is not trusted."
    return str(report.get("error") or "Repair failed.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlh-repair")
    subcommands = parser.add_subparsers(dest="command", required=True)
    repair = subcommands.add_parser("repair")
    repair.add_argument("--root", required=True)
    repair.add_argument("--source", action="append", required=True)
    repair.add_argument("--platform", required=True)
    repair.add_argument("--variant", required=True)
    repair.add_argument("--arch", required=True)
    repair.add_argument("--trusted-keys-dir")
    repair.add_argument("--timeout", type=float, default=8.0)
    repair.add_argument("--download-dir")
    repair.add_argument("--backup-dir")
    repair.add_argument("--json", action="store_true")
    index = subcommands.add_parser("build-index")
    index.add_argument("--root", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--payload-dir", required=True)
    index.add_argument("--url-prefix", required=True)
    index.add_argument("--trusted-keys-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-index":
            output = build_repair_index(
                args.root, output=args.output, payload_dir=args.payload_dir,
                url_prefix=args.url_prefix, trusted_keys_dir=args.trusted_keys_dir,
            )
            print(json.dumps({"repair_index": str(output)}, ensure_ascii=False, sort_keys=True))
            return 0
        report = repair_install(
            args.root, sources=args.source,
            profile={"platform": args.platform, "variant": args.variant, "arch": args.arch},
            trusted_keys_dir=args.trusted_keys_dir, timeout=args.timeout,
            download_dir=args.download_dir, backup_dir=args.backup_dir,
        )
    except RepairError as exc:
        report = {"schema_version": REPAIR_SCHEMA_VERSION, "command": "repair", "ok": False, "action": "failed", "error": str(exc)}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(format_repair(report))
    return 0 if report.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
