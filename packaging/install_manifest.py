"""Signed installed-file baseline and runtime verifier for Launcher UP-N6.

UP-N6.0 established the signed baseline.  UP-N6.1 adds a read-only
quick/full/deep verifier which only visits paths explicitly listed in that
baseline, never application user-data roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_TYPE = "qlh_install"
MANIFEST_RELATIVE_PATH = "manifest/install-manifest.json"
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_FILE_COUNT = 250_000
VERIFY_LEVELS = frozenset({"quick", "full", "deep"})
LARGE_FILE_BYTES = 64 * 1024 * 1024
SAMPLE_WINDOW_BYTES = 1024 * 1024
RESERVED_USER_DATA_ROOTS = frozenset(
    {"models", "chat_history", "logs", "config", "local_docs"}
)
ALLOWED_PLATFORMS = frozenset({"windows", "linux"})
ALLOWED_PACKAGE_KINDS = frozenset({"application", "launcher"})
ALLOWED_FILE_KINDS = frozenset(
    {"application", "documentation", "frontend", "health", "metadata", "runtime", "tool"}
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InstallManifestError(RuntimeError):
    """Expected install-manifest build or validation failure."""


@dataclass(frozen=True)
class IncludeSource:
    source: Path
    destination: str


def _report_failure(
    path: str,
    category: str,
    *,
    expected: Any = None,
    actual: Any = None,
    detail: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "category": category}
    if expected is not None:
        result["expected"] = expected
    if actual is not None:
        result["actual"] = actual
    if detail:
        result["detail"] = detail
    return result


def _advice_for_failures(failures: Sequence[Mapping[str, Any]]) -> list[str]:
    categories = {str(item.get("category", "")) for item in failures}
    advice: list[str] = []
    if "signature" in categories:
        advice.append("安装基线签名无效；请停止使用该目录，并从可信更新源覆盖安装。")
    if {"missing", "size", "hash"} & categories:
        advice.append("程序文件缺失或损坏；后续 repair 可修复前请先从可信更新源重新安装。")
    if {"unsafe", "io"} & categories:
        advice.append("安装路径包含链接、重解析点或不可读文件；请检查杀毒软件、磁盘和安装目录权限。")
    if "version" in categories:
        advice.append("安装目录版本与签名清单不一致；请使用更新或回滚恢复匹配版本。")
    if not advice:
        advice.append("未发现签名清单中程序文件的完整性问题。")
    return advice


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InstallManifestError(f"{field} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InstallManifestError(f"{field} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise InstallManifestError(f"{field} must include a timezone")
    return value


def _validate_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise InstallManifestError(f"invalid {field}: {value!r}")
    return value


def _validate_version(value: Any) -> str:
    if not isinstance(value, str):
        raise InstallManifestError("version must be a string")
    version = value.strip()
    if not version or len(version) > 128 or any(ord(ch) < 32 for ch in version):
        raise InstallManifestError(f"invalid version: {value!r}")
    if any(ch in version for ch in "/\\"):
        raise InstallManifestError(f"invalid version: {value!r}")
    return version


def normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise InstallManifestError(f"invalid install path: {value!r}")
    if value != unicodedata.normalize("NFC", value):
        raise InstallManifestError(f"install path is not NFC-normalized: {value!r}")
    if "\\" in value or value.startswith("/") or any(ord(ch) < 32 for ch in value):
        raise InstallManifestError(f"unsafe install path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise InstallManifestError(f"unsafe install path: {value!r}")
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise InstallManifestError(f"unsafe install path: {value!r}")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized != value:
        raise InstallManifestError(f"non-canonical install path: {value!r}")
    return normalized


def _is_reserved_user_data_path(relative_path: str) -> bool:
    first = relative_path.split("/", 1)[0].casefold()
    return first in RESERVED_USER_DATA_ROOTS


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _file_kind(relative_path: str) -> str:
    path = relative_path.casefold()
    first = path.split("/", 1)[0]
    name = path.rsplit("/", 1)[-1]
    if name == "health.ok":
        return "health"
    if name == "version.txt" or first == "manifest":
        return "metadata"
    if first == "docs":
        return "documentation"
    if first == "tools" or first == "model-tools":
        return "tool"
    if first == "frontend":
        return "frontend"
    if first in {"_internal", "venv", "bin", "src", "pubkeys"}:
        return "runtime"
    return "application"


def _hash_file(path: Path) -> tuple[int, str]:
    if _is_link_or_reparse(path):
        raise InstallManifestError(f"install tree contains a link or reparse point: {path.name}")
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise InstallManifestError(f"cannot stat install file: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise InstallManifestError(f"install tree contains a non-regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise InstallManifestError(f"cannot hash install file: {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise InstallManifestError(f"install file changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _iter_tree_files(root: Path, destination: str = "") -> Iterable[tuple[Path, str]]:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise InstallManifestError(f"install source is not a regular directory: {root}")
    for current_raw, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        relative_dir = current.relative_to(root)
        prefix_parts = [] if str(relative_dir) == "." else list(relative_dir.parts)
        kept_directories: list[str] = []
        for name in sorted(directories, key=lambda item: (item.casefold(), item)):
            path = current / name
            target_parts = [part for part in (destination, *prefix_parts, name) if part]
            target = PurePosixPath(*target_parts).as_posix()
            if _is_reserved_user_data_path(target):
                continue
            if _is_link_or_reparse(path):
                raise InstallManifestError(
                    f"install tree contains a directory link or reparse point: {target}"
                )
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(filenames, key=lambda item: (item.casefold(), item)):
            target_parts = [part for part in (destination, *prefix_parts, name) if part]
            target = normalize_relative_path(PurePosixPath(*target_parts).as_posix())
            if target == MANIFEST_RELATIVE_PATH:
                continue
            if _is_reserved_user_data_path(target):
                continue
            yield current / name, target


def collect_install_files(
    root: str | os.PathLike[str], *, includes: Sequence[IncludeSource] = (),
) -> list[dict[str, Any]]:
    root_path = Path(root).expanduser().resolve()
    sources: list[tuple[Path, str]] = list(_iter_tree_files(root_path))
    for include in includes:
        source = include.source.expanduser().resolve()
        destination = normalize_relative_path(include.destination)
        if _is_reserved_user_data_path(destination):
            raise InstallManifestError(f"explicit include targets user data: {destination}")
        if source.is_dir():
            sources.extend(_iter_tree_files(source, destination))
        elif source.is_file():
            sources.append((source, destination))
        else:
            raise InstallManifestError(f"install include does not exist: {include.source}")

    by_path: dict[str, tuple[Path, str]] = {}
    for source, relative_path in sources:
        folded = relative_path.casefold()
        if folded in by_path:
            other = by_path[folded][1]
            raise InstallManifestError(
                f"duplicate or case-colliding install path: {other!r} / {relative_path!r}"
            )
        by_path[folded] = (source, relative_path)
    if not by_path:
        raise InstallManifestError("install tree contains no program files")
    if len(by_path) > MAX_FILE_COUNT:
        raise InstallManifestError(f"install tree exceeds {MAX_FILE_COUNT} files")

    files: list[dict[str, Any]] = []
    for source, relative_path in sorted(
        by_path.values(), key=lambda item: (item[1].casefold(), item[1]),
    ):
        size, sha256 = _hash_file(source)
        files.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": sha256,
                "kind": _file_kind(relative_path),
            }
        )
    return files


def build_install_manifest(
    root: str | os.PathLike[str],
    *,
    app_id: str,
    version: str,
    platform: str,
    variant: str,
    package_kind: str,
    includes: Sequence[IncludeSource] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "app_id": _validate_identifier(app_id, field="app_id"),
        "version": _validate_version(version),
        "platform": str(platform),
        "variant": _validate_identifier(variant, field="variant"),
        "package_kind": str(package_kind),
        "scope": "application_files_only",
        "generated_at": generated_at or _utc_now(),
        "files": collect_install_files(root, includes=includes),
    }
    validate_install_manifest(mapping, require_signature=False)
    return mapping


def validate_install_manifest(
    mapping: Mapping[str, Any], *, require_signature: bool = True,
) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise InstallManifestError("install manifest root must be an object")
    required = {
        "schema_version", "manifest_type", "app_id", "version", "platform",
        "variant", "package_kind", "scope", "generated_at", "files",
    }
    signature_fields = {"key_id", "signed_at", "signature"}
    keys = set(mapping)
    missing = required - keys
    if missing:
        raise InstallManifestError(f"install manifest is missing fields: {sorted(missing)}")
    unknown = keys - required - signature_fields
    if unknown:
        raise InstallManifestError(f"install manifest has unknown fields: {sorted(unknown)}")
    present_signature = keys & signature_fields
    if require_signature and present_signature != signature_fields:
        raise InstallManifestError("install manifest must include key_id/signed_at/signature")
    if present_signature and present_signature != signature_fields:
        raise InstallManifestError("install manifest signature fields are incomplete")
    if mapping["schema_version"] != SCHEMA_VERSION or isinstance(mapping["schema_version"], bool):
        raise InstallManifestError("unsupported install manifest schema")
    if mapping["manifest_type"] != MANIFEST_TYPE:
        raise InstallManifestError("invalid install manifest type")
    _validate_identifier(mapping["app_id"], field="app_id")
    _validate_version(mapping["version"])
    if not isinstance(mapping["platform"], str) or mapping["platform"] not in ALLOWED_PLATFORMS:
        raise InstallManifestError(f"unsupported platform: {mapping['platform']!r}")
    _validate_identifier(mapping["variant"], field="variant")
    if (
        not isinstance(mapping["package_kind"], str)
        or mapping["package_kind"] not in ALLOWED_PACKAGE_KINDS
    ):
        raise InstallManifestError(f"unsupported package_kind: {mapping['package_kind']!r}")
    if mapping["scope"] != "application_files_only":
        raise InstallManifestError("install manifest scope must be application_files_only")
    _validate_timestamp(mapping["generated_at"], field="generated_at")
    if present_signature:
        _validate_identifier(mapping["key_id"], field="key_id")
        _validate_timestamp(mapping["signed_at"], field="signed_at")
        if not isinstance(mapping["signature"], str) or not mapping["signature"]:
            raise InstallManifestError("install manifest signature must be non-empty")

    files = mapping["files"]
    if not isinstance(files, list) or not files or len(files) > MAX_FILE_COUNT:
        raise InstallManifestError("install manifest files must be a non-empty bounded list")
    seen: set[str] = set()
    sort_keys: list[tuple[str, str]] = []
    for entry in files:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256", "kind"}:
            raise InstallManifestError("install manifest file entry has invalid fields")
        path = normalize_relative_path(entry["path"])
        if path == MANIFEST_RELATIVE_PATH:
            raise InstallManifestError("install manifest cannot hash itself")
        if _is_reserved_user_data_path(path):
            raise InstallManifestError(f"install manifest includes user data path: {path}")
        folded = path.casefold()
        if folded in seen:
            raise InstallManifestError(f"duplicate or case-colliding install path: {path}")
        seen.add(folded)
        sort_keys.append((folded, path))
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise InstallManifestError(f"invalid install file size: {path}")
        if not isinstance(entry["sha256"], str) or not _SHA256_RE.fullmatch(entry["sha256"]):
            raise InstallManifestError(f"invalid install file sha256: {path}")
        if (
            not isinstance(entry["kind"], str)
            or entry["kind"] not in ALLOWED_FILE_KINDS
            or entry["kind"] != _file_kind(path)
        ):
            raise InstallManifestError(f"invalid install file kind: {path}")
    if sort_keys != sorted(sort_keys):
        raise InstallManifestError("install manifest files are not in canonical order")
    return dict(mapping)


def load_install_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise InstallManifestError("install manifest exceeds the size limit")
        value = json.loads(source.read_text(encoding="utf-8"))
    except InstallManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InstallManifestError(f"cannot read install manifest: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise InstallManifestError("install manifest root must be an object")
    return value


def verify_install_manifest(
    mapping: Mapping[str, Any], *, trusted_keys_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    validated = validate_install_manifest(mapping, require_signature=True)
    from signing import verify_manifest_signature

    verified, reason = verify_manifest_signature(
        validated, trusted_keys_dir=trusted_keys_dir,
    )
    if not verified:
        raise InstallManifestError(f"install manifest signature rejected: {reason}")
    return validated


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InstallManifestError(f"cannot create install manifest directory: {path.parent}: {exc}") from exc
    if _is_link_or_reparse(path.parent):
        raise InstallManifestError("install manifest directory is a link or reparse point")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise InstallManifestError(f"cannot persist install manifest: {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def persist_verified_install_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    install_root: str | os.PathLike[str],
    trusted_keys_dir: str | os.PathLike[str] | None,
) -> Path:
    mapping = load_install_manifest(manifest) if isinstance(manifest, (str, os.PathLike)) else dict(manifest)
    verified = verify_install_manifest(mapping, trusted_keys_dir=trusted_keys_dir)
    root = Path(install_root).expanduser().resolve()
    if not root.is_dir() or _is_link_or_reparse(root):
        raise InstallManifestError(f"install root is not a regular directory: {root}")
    destination = root / Path(MANIFEST_RELATIVE_PATH)
    _atomic_write_json(destination, verified)
    return destination


def write_signed_install_manifest(
    root: str | os.PathLike[str],
    *,
    app_id: str,
    version: str,
    platform: str,
    variant: str,
    package_kind: str,
    private_key_path: str | os.PathLike[str],
    trusted_keys_dir: str | os.PathLike[str] | None,
    includes: Sequence[IncludeSource] = (),
    generated_at: str | None = None,
    signed_at: str | None = None,
) -> Path:
    from signing import sign_manifest

    unsigned = build_install_manifest(
        root,
        app_id=app_id,
        version=version,
        platform=platform,
        variant=variant,
        package_kind=package_kind,
        includes=includes,
        generated_at=generated_at,
    )
    signed = sign_manifest(
        unsigned, private_key_path=private_key_path, signed_at=signed_at,
    )
    return persist_verified_install_manifest(
        signed, install_root=root, trusted_keys_dir=trusted_keys_dir,
    )


def verify_manifest_file_if_present(
    root: str | os.PathLike[str],
    *,
    trusted_keys_dir: str | os.PathLike[str] | None,
    required: bool = False,
) -> dict[str, Any] | None:
    path = Path(root) / Path(MANIFEST_RELATIVE_PATH)
    if not path.is_file():
        if required:
            raise InstallManifestError("bundle has no signed install-manifest.json")
        return None
    return verify_install_manifest(
        load_install_manifest(path), trusted_keys_dir=trusted_keys_dir,
    )


def _runtime_trusted_keys_dir(root: Path, supplied: str | os.PathLike[str] | None) -> str | os.PathLike[str] | None:
    if supplied is not None:
        return supplied
    bundled = root / "pubkeys"
    if bundled.is_dir() and not _is_link_or_reparse(bundled):
        return bundled
    from signing import default_trusted_keys_dir

    return default_trusted_keys_dir()


def _safe_install_root(root: str | os.PathLike[str]) -> Path:
    candidate = Path(root).expanduser()
    if not candidate.is_dir():
        raise InstallManifestError(f"install root is not a regular directory: {candidate}")
    if _is_link_or_reparse(candidate):
        raise InstallManifestError("install root is a link or reparse point")
    try:
        return candidate.resolve()
    except OSError as exc:
        raise InstallManifestError(f"cannot resolve install root: {candidate}: {exc}") from exc


def _manifest_file_path(root: Path, relative_path: str) -> Path:
    """Resolve one signed path while refusing link/reparse traversal."""
    target = root
    for part in PurePosixPath(relative_path).parts:
        target = target / part
        if _is_link_or_reparse(target):
            raise InstallManifestError(
                f"install path traverses a link or reparse point: {relative_path}"
            )
    return target


def resolve_install_manifest_path(
    root: str | os.PathLike[str], relative_path: str,
) -> Path:
    """Resolve one signed application path without following links.

    This is intentionally narrower than a general path utility: callers must
    pass a path already listed in a verified install manifest.  It is exposed
    for maintenance operations such as UP-N6.3 repair so they share the same
    link/reparse-point boundary as runtime verification.
    """
    install_root = _safe_install_root(root)
    return _manifest_file_path(
        install_root, normalize_relative_path(relative_path),
    )


def _stat_manifest_file(root: Path, relative_path: str) -> tuple[Path, os.stat_result]:
    path = _manifest_file_path(root, relative_path)
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise InstallManifestError(f"install file is missing: {relative_path}") from exc
    except OSError as exc:
        raise InstallManifestError(f"cannot stat install file: {relative_path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise InstallManifestError(f"install path is not a regular file: {relative_path}")
    return path, metadata


def _sample_file(path: Path, size: int) -> None:
    """Read fixed windows from a large file to catch unreadable disk sectors.

    Schema v1 stores only whole-file SHA-256 values.  The sample has no
    signed comparison digest and therefore proves readability, not integrity;
    callers must report that distinction explicitly.
    """
    window = min(size, SAMPLE_WINDOW_BYTES)
    offsets = sorted({0, max(0, (size - window) // 2), max(0, size - window)})
    try:
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                remaining = window
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise InstallManifestError(f"sample read ended early: {path.name}")
                    remaining -= len(chunk)
    except InstallManifestError:
        raise
    except OSError as exc:
        raise InstallManifestError(f"cannot sample install file: {path.name}: {exc}") from exc


def _launcher_entrypoint(manifest: Mapping[str, Any]) -> str:
    if manifest["platform"] == "windows":
        return "QLH-Launcher.exe" if manifest["package_kind"] == "launcher" else "QLH-Edge-Inference.exe"
    return "bin/qlh-launcher" if manifest["package_kind"] == "launcher" else "bin/qlh-app"


def _is_critical_runtime_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].casefold()
    return (
        name in {"base_library.zip", "python", "python.exe", "python3", "python3.exe"}
        or name.startswith("libpython")
        or (name.startswith("python") and name.endswith((".dll", ".pyd")))
    )


def _quick_paths(manifest: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    entries = {str(entry["path"]): entry for entry in manifest["files"]}
    required = [_launcher_entrypoint(manifest), "version.txt"]
    if manifest["package_kind"] == "launcher":
        required.append("health.ok")
    missing = [
        _report_failure(
            path,
            "missing",
            expected="listed in signed install manifest",
            actual="not listed",
            detail="关键启动文件没有可信基线",
        )
        for path in required
        if path not in entries
    ]
    paths = {path for path in required if path in entries}
    for entry in manifest["files"]:
        path = str(entry["path"])
        if entry["kind"] == "health" or _is_critical_runtime_path(path):
            paths.add(path)
        if path in {"tools/QLH-Install-Manifest.exe", "bin/install_manifest.py"}:
            paths.add(path)
    return sorted(paths, key=lambda item: (item.casefold(), item)), missing


def _initial_verify_report(root: str | os.PathLike[str], level: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": "verify",
        "root": str(Path(root).expanduser()),
        "level": level,
        "ok": False,
        "manifest": None,
        "passed": [],
        "failed": [],
        "summary": {"checked": 0, "passed": 0, "failed": 0, "hash_verified": 0, "sampled": 0},
        "advice": [],
    }


def _finish_verify_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    summary["passed"] = len(report["passed"])
    summary["failed"] = len(report["failed"])
    summary["checked"] = summary["passed"] + summary["failed"]
    report["ok"] = not report["failed"]
    report["advice"] = _advice_for_failures(report["failed"])
    return report


def verify_install_tree(
    root: str | os.PathLike[str],
    *,
    level: str = "quick",
    trusted_keys_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify one installed program tree against its signed baseline.

    ``quick`` hashes only launch-critical files, ``full`` hashes files up to
    64 MiB and samples larger files for readability, and ``deep`` hashes every
    listed file.  No mode walks the installation tree: only signed program
    paths are visited, which keeps user data outside this operation.
    """
    if level not in VERIFY_LEVELS:
        raise InstallManifestError(f"unsupported verify level: {level!r}")
    report = _initial_verify_report(root, level)
    try:
        install_root = _safe_install_root(root)
    except InstallManifestError as exc:
        report["failed"].append(_report_failure(".", "unsafe", detail=str(exc)))
        return _finish_verify_report(report)
    report["root"] = str(install_root)

    try:
        manifest = verify_manifest_file_if_present(
            install_root,
            trusted_keys_dir=_runtime_trusted_keys_dir(install_root, trusted_keys_dir),
            required=True,
        )
        assert manifest is not None
    except InstallManifestError as exc:
        report["failed"].append(
            _report_failure(
                MANIFEST_RELATIVE_PATH,
                "signature",
                expected="valid Ed25519 signature from trusted QLH release key",
                actual="rejected",
                detail=str(exc),
            )
        )
        return _finish_verify_report(report)

    report["manifest"] = {
        key: manifest[key]
        for key in ("app_id", "version", "platform", "variant", "package_kind", "key_id")
    }
    report["passed"].append({"path": MANIFEST_RELATIVE_PATH, "check": "signature"})
    entries = {str(entry["path"]): entry for entry in manifest["files"]}
    if level == "quick":
        paths, baseline_failures = _quick_paths(manifest)
    else:
        _, baseline_failures = _quick_paths(manifest)
        paths = [str(entry["path"]) for entry in manifest["files"]]
    report["failed"].extend(baseline_failures)

    for relative_path in paths:
        entry = entries[relative_path]
        try:
            path, metadata = _stat_manifest_file(install_root, relative_path)
        except InstallManifestError as exc:
            text = str(exc)
            category = "missing" if "is missing" in text else "unsafe" if "link or reparse" in text or "not a regular" in text else "io"
            report["failed"].append(
                _report_failure(relative_path, category, expected=entry["size"], detail=text)
            )
            continue
        if metadata.st_size != entry["size"]:
            report["failed"].append(
                _report_failure(relative_path, "size", expected=entry["size"], actual=metadata.st_size)
            )
            continue

        sampled = level == "full" and metadata.st_size > LARGE_FILE_BYTES
        try:
            if sampled:
                _sample_file(path, metadata.st_size)
            else:
                _, digest = _hash_file(path)
                if digest != entry["sha256"]:
                    report["failed"].append(
                        _report_failure(relative_path, "hash", expected=entry["sha256"], actual=digest)
                    )
                    continue
        except InstallManifestError as exc:
            report["failed"].append(_report_failure(relative_path, "io", detail=str(exc)))
            continue

        if relative_path == "version.txt":
            try:
                installed_version = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as exc:
                report["failed"].append(_report_failure(relative_path, "io", detail=str(exc)))
                continue
            if installed_version != manifest["version"]:
                report["failed"].append(
                    _report_failure(
                        relative_path,
                        "version",
                        expected=manifest["version"],
                        actual=installed_version,
                    )
                )
                continue

        report["passed"].append(
            {"path": relative_path, "check": "size+sample" if sampled else "sha256"}
        )
        if sampled:
            report["summary"]["sampled"] += 1
        else:
            report["summary"]["hash_verified"] += 1
    return _finish_verify_report(report)


def _parse_include(value: str) -> IncludeSource:
    if "=" not in value:
        raise InstallManifestError("--include must use SOURCE=DESTINATION")
    source, destination = value.split("=", 1)
    if not source or not destination:
        raise InstallManifestError("--include must use SOURCE=DESTINATION")
    return IncludeSource(Path(source), destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlh-install-manifest",
        description="Build, validate and verify signed QLH install manifests (UP-N6).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="scan, sign, verify and atomically persist a manifest")
    build.add_argument("--root", required=True)
    build.add_argument("--app-id", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--platform", required=True, choices=sorted(ALLOWED_PLATFORMS))
    build.add_argument("--variant", required=True)
    build.add_argument("--package-kind", required=True, choices=sorted(ALLOWED_PACKAGE_KINDS))
    build.add_argument("--key", default=os.environ.get("QLH_SIGNING_KEY", ""))
    build.add_argument("--trusted-keys-dir")
    build.add_argument("--include", action="append", default=[], metavar="SOURCE=DESTINATION")

    validate = sub.add_parser("validate", help="validate contract and Ed25519 signature")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--trusted-keys-dir")

    persist = sub.add_parser("persist", help="verify a supplied manifest before atomic install")
    persist.add_argument("--manifest", required=True)
    persist.add_argument("--install-root", required=True)
    persist.add_argument("--trusted-keys-dir")

    verify = sub.add_parser("verify", help="read-only quick/full/deep verification of an installed tree")
    verify.add_argument("--root", required=True)
    verify.add_argument("--level", choices=sorted(VERIFY_LEVELS), default="quick")
    verify.add_argument("--trusted-keys-dir")
    verify.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        from signing import default_trusted_keys_dir

        trusted = options.trusted_keys_dir or default_trusted_keys_dir()
        if options.command == "build":
            if not options.key:
                raise InstallManifestError("release build requires --key or QLH_SIGNING_KEY")
            path = write_signed_install_manifest(
                options.root,
                app_id=options.app_id,
                version=options.version,
                platform=options.platform,
                variant=options.variant,
                package_kind=options.package_kind,
                private_key_path=options.key,
                trusted_keys_dir=trusted,
                includes=[_parse_include(value) for value in options.include],
            )
            mapping = load_install_manifest(path)
            print(json.dumps({
                "manifest": str(path),
                "file_count": len(mapping["files"]),
                "key_id": mapping["key_id"],
            }, ensure_ascii=False))
            return 0
        if options.command == "validate":
            mapping = verify_install_manifest(
                load_install_manifest(options.manifest), trusted_keys_dir=trusted,
            )
            print(json.dumps({
                "verified": True,
                "file_count": len(mapping["files"]),
                "key_id": mapping["key_id"],
            }, ensure_ascii=False))
            return 0
        if options.command == "verify":
            report = verify_install_tree(
                options.root,
                level=options.level,
                trusted_keys_dir=options.trusted_keys_dir,
            )
            if options.as_json:
                print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            else:
                summary = report["summary"]
                state = "passed" if report["ok"] else "failed"
                print(
                    f"verify {options.level}: {state}; "
                    f"checked={summary['checked']} passed={summary['passed']} "
                    f"failed={summary['failed']} sampled={summary['sampled']}"
                )
                for item in report["failed"]:
                    print(
                        f"  [{item['category']}] {item['path']}: "
                        f"{item.get('detail') or item.get('actual', 'mismatch')}",
                        file=sys.stderr,
                    )
                for advice in report["advice"]:
                    print(f"  建议: {advice}", file=sys.stderr)
            return 0 if report["ok"] else 3
        path = persist_verified_install_manifest(
            options.manifest,
            install_root=options.install_root,
            trusted_keys_dir=trusted,
        )
        print(json.dumps({"verified": True, "manifest": str(path)}, ensure_ascii=False))
        return 0
    except InstallManifestError as exc:
        print(f"install-manifest: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
