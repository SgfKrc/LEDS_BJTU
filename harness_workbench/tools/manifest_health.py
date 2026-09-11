"""Read-only model manifest and gitignore health reporting.

``TOOL-MANIFEST-HLTH-01`` deliberately does not load a model or download an
asset.  The default scan reads file metadata, sidecar declarations and small
JSON manifests.  Full content hashing is an explicit opt-in because model
files can be multi-gigabyte files on the development machine.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_HEALTH_SCHEMA = "qlh.harness.manifest_health.v1"
MODEL_SUFFIXES = frozenset((".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_reparse(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if callable(checker):
        try:
            if checker():
                return True
        except OSError:
            pass
    try:
        return path.is_symlink()
    except OSError:
        return False


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path != root else "."


def _safe_relative(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _iter_files(root: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    files: list[Path] = []
    skipped: list[str] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            candidate = current_path / name
            if _is_reparse(candidate):
                skipped.append(_relative(candidate, root))
            else:
                kept.append(name)
        directories[:] = kept
        for name in names:
            candidate = current_path / name
            if _is_reparse(candidate):
                skipped.append(_relative(candidate, root))
            elif candidate.is_file():
                files.append(candidate)
    return tuple(sorted(files)), tuple(sorted(set(skipped)))


@dataclass(frozen=True, slots=True)
class IgnoreRule:
    pattern: str
    negated: bool = False
    directory_only: bool = False


def _load_ignore_rules(path: Path | None) -> tuple[IgnoreRule, ...]:
    if path is None or not path.is_file():
        return ()
    rules: list[IgnoreRule] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ()
    for raw in lines:
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        negated = value.startswith("!")
        if negated:
            value = value[1:]
        directory_only = value.endswith("/")
        value = value.rstrip("/").replace("\\", "/")
        if value:
            rules.append(IgnoreRule(value.lstrip("/"), negated, directory_only))
    return tuple(rules)


def _rule_matches(rule: IgnoreRule, relative: str, *, is_directory: bool) -> bool:
    if rule.directory_only and not is_directory:
        # A directory rule also applies to files below that directory.
        return relative.startswith(rule.pattern.rstrip("/") + "/")
    pattern = rule.pattern
    name = PurePosixPath(relative).name
    if pattern.endswith("/"):
        pattern = pattern.rstrip("/")
    if "/" not in pattern:
        return fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(relative, pattern)
    if fnmatch.fnmatchcase(relative, pattern):
        return True
    if "**" in pattern:
        compact = pattern.replace("**/", "")
        return fnmatch.fnmatchcase(relative, compact)
    return False


def _ignored(relative: str, rules: Sequence[IgnoreRule]) -> tuple[bool, str | None]:
    ignored = False
    matched: str | None = None
    for rule in rules:
        if _rule_matches(rule, relative, is_directory=False):
            ignored = not rule.negated
            matched = ("!" if rule.negated else "") + rule.pattern
    return ignored, matched


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(r"(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _sidecar_for(path: Path) -> tuple[Path | None, str | None, str]:
    exact = path.with_name(path.name + ".sha256")
    if exact.is_file():
        return exact, _read_sidecar(exact), "sidecar"
    aggregate = path.parent / "model.sha256"
    if aggregate.is_file():
        return aggregate, _read_sidecar(aggregate), "aggregate_sidecar"
    return None, None, "missing"


def _hash_status(path: Path, expected: str | None, *, verify_hash: bool, declaration: str) -> tuple[str, str | None]:
    if declaration == "missing":
        return "missing", None
    if not expected or not _SHA256.fullmatch(expected):
        return "invalid_declaration", expected
    if declaration == "aggregate_sidecar":
        return "aggregate_declared", expected
    if not verify_hash:
        return "declared", expected
    try:
        actual = _sha256_file(path)
    except OSError:
        return "unreadable", expected
    return ("verified" if actual == expected else "mismatch"), expected


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix[1:] if suffix else "file"


def _artifact_record(
    path: Path,
    root: Path,
    *,
    ignored_rules: Sequence[IgnoreRule],
    ignore_base: Path,
    verify_hash: bool,
) -> dict[str, Any]:
    relative = _relative(path, root)
    try:
        ignore_relative = path.relative_to(ignore_base).as_posix()
    except ValueError:
        ignore_relative = relative
    ignored, ignore_rule = _ignored(ignore_relative, ignored_rules)
    sidecar, expected, declaration = _sidecar_for(path)
    sha_status, expected = _hash_status(path, expected, verify_hash=verify_hash, declaration=declaration)
    try:
        size = int(path.stat().st_size)
    except OSError:
        size = 0
        sha_status = "unreadable"
    return {
        "path": relative,
        "kind": _artifact_kind(path),
        "size_bytes": size,
        "ignored": ignored,
        "ignore_rule": ignore_rule,
        "sidecar_path": _relative(sidecar, root) if sidecar else None,
        "expected_sha256": expected,
        "sha256_status": sha_status,
    }


def _read_json(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None, "too_large"
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid"
    return (value, None) if isinstance(value, Mapping) else (None, "object_required")


def _check_manifest(path: Path, root: Path, *, verify_hash: bool) -> dict[str, Any]:
    relative = _relative(path, root)
    payload, error = _read_json(path)
    if payload is None:
        return {"path": relative, "kind": "json", "valid": False, "errors": [error or "invalid"]}
    errors: list[str] = []
    checked = 0
    hash_checked = 0

    def check_entry(entry: Mapping[str, Any], label: str) -> None:
        nonlocal checked, hash_checked
        relative_value = _safe_relative(entry.get("path", entry.get("filename")))
        if relative_value is None:
            errors.append(f"{label}:unsafe_path")
            return
        target = path.parent / Path(*relative_value.parts)
        if not target.is_file():
            errors.append(f"{label}:missing")
            return
        checked += 1
        expected_size = entry.get("size_bytes", entry.get("size"))
        if isinstance(expected_size, int) and target.stat().st_size != expected_size:
            errors.append(f"{label}:size_mismatch")
        expected_hash = entry.get("sha256")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
                errors.append(f"{label}:sha256_invalid")
            elif verify_hash:
                hash_checked += 1
                if _sha256_file(target) != expected_hash.lower():
                    errors.append(f"{label}:sha256_mismatch")

    entries = payload.get("files")
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                errors.append(f"files[{index}]:entry_invalid")
            else:
                check_entry(entry, f"files[{index}]")
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for name, entry in artifacts.items():
            if not isinstance(entry, Mapping):
                errors.append(f"artifacts.{name}:entry_invalid")
            else:
                check_entry(entry, f"artifacts.{name}")
    weight_map = payload.get("weight_map")
    if isinstance(weight_map, Mapping):
        for name, filename in sorted(weight_map.items()):
            check_entry({"filename": filename}, f"weight_map.{name}")
    if not any(key in payload for key in ("files", "artifacts", "weight_map")):
        errors.append("manifest_entries_missing")
    return {
        "path": relative,
        "kind": "manifest",
        "valid": not errors,
        "errors": errors,
        "entries_checked": checked,
        "hashes_checked": hash_checked,
        "schema_version": payload.get("schema_version"),
    }


@dataclass(frozen=True, slots=True)
class ManifestHealthReport:
    root_label: str
    artifacts: tuple[Mapping[str, Any], ...]
    manifests: tuple[Mapping[str, Any], ...]
    scanned_file_count: int
    ignored_file_count: int
    unignored_model_paths: tuple[str, ...]
    skipped_reparse_points: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    verify_hash: bool
    gitignore_path: str | None
    schema: str = MANIFEST_HEALTH_SCHEMA
    read_only: bool = True
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_HEALTH_SCHEMA or not self.root_label:
            raise ValueError("manifest health report identity is invalid")
        if self.scanned_file_count < 0 or self.ignored_file_count < 0 or self.ignored_file_count > self.scanned_file_count:
            raise ValueError("manifest health file counts are invalid")
        if not self.read_only or self.network_used or self.weights_loaded:
            raise ValueError("manifest health must be read-only and model-free")

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "root_label": self.root_label,
            "valid": self.valid,
            "artifacts": [dict(item) for item in self.artifacts],
            "manifests": [dict(item) for item in self.manifests],
            "scanned_file_count": self.scanned_file_count,
            "ignored_file_count": self.ignored_file_count,
            "unignored_model_paths": list(self.unignored_model_paths),
            "skipped_reparse_points": list(self.skipped_reparse_points),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "verify_hash": self.verify_hash,
            "gitignore_path": self.gitignore_path,
            "read_only": self.read_only,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# Model manifest health",
            "",
            f"- Root: `{self.root_label}`; valid: `{self.valid}`; hash mode: `{'full' if self.verify_hash else 'sidecar-declaration'}`",
            f"- Files scanned: `{self.scanned_file_count}`; ignored: `{self.ignored_file_count}`; read-only: `true`; weights loaded: `false`; network used: `false`",
            "",
            "| path | kind | size | ignored | SHA status | sidecar |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
        for item in self.artifacts:
            lines.append(
                f"| `{item['path']}` | `{item['kind']}` | {item['size_bytes']} | {str(item['ignored']).lower()} | "
                f"`{item['sha256_status']}` | `{item['sidecar_path'] or ''}` |"
            )
        if self.manifests:
            lines.extend(("", "## Manifests", "", "| path | valid | checked | errors |", "| --- | --- | ---: | --- |"))
            for item in self.manifests:
                lines.append(f"| `{item['path']}` | {str(item['valid']).lower()} | {item.get('entries_checked', 0)} | {', '.join(item.get('errors', [])) or '-'} |")
        if self.warnings:
            lines.extend(("", "Warnings: " + "; ".join(self.warnings)))
        if self.errors:
            lines.extend(("", "Errors: " + "; ".join(self.errors)))
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def scan_manifest_health(
    root: str | Path,
    *,
    gitignore: str | Path | None = None,
    verify_hash: bool = False,
) -> ManifestHealthReport:
    """Scan a model root without loading weights or changing any file."""

    target = Path(root).expanduser().absolute()
    root_label = target.name or "models"
    if not target.exists():
        return ManifestHealthReport(root_label, (), (), 0, 0, (), (), (), ("model_root_missing",), verify_hash, None)
    if not target.is_dir():
        return ManifestHealthReport(root_label, (), (), 0, 0, (), (), (), ("model_root_not_directory",), verify_hash, None)
    ignore_path = Path(gitignore).expanduser().absolute() if gitignore else target.parent / ".gitignore"
    rules = _load_ignore_rules(ignore_path)
    files, skipped = _iter_files(target)
    artifacts = tuple(
        _artifact_record(item, target, ignored_rules=rules, ignore_base=ignore_path.parent, verify_hash=verify_hash)
        for item in files
        if item.suffix.lower() in MODEL_SUFFIXES
    )
    manifest_files = tuple(
        item for item in files
        if item.name.endswith((".manifest.json", ".lock.json")) or item.name == "model.safetensors.index.json"
    )
    manifests = tuple(_check_manifest(item, target, verify_hash=verify_hash) for item in manifest_files)
    ignored_count = 0
    for item in files:
        relative_to_ignore = item.relative_to(ignore_path.parent).as_posix() if ignore_path.parent in item.parents or item == ignore_path.parent else _relative(item, target)
        if _ignored(relative_to_ignore, rules)[0]:
            ignored_count += 1
    unignored = tuple(item["path"] for item in artifacts if not item["ignored"])
    warnings: list[str] = []
    errors: list[str] = []
    if not rules:
        warnings.append("gitignore_not_found_or_empty")
    if unignored:
        warnings.append(f"model_files_not_ignored:{len(unignored)}")
    if skipped:
        warnings.extend(f"reparse_point_skipped:{item}" for item in skipped)
    for item in artifacts:
        if item["sha256_status"] in {"missing"}:
            warnings.append(f"{item['sha256_status']}:{item['path']}")
        elif item["sha256_status"] in {"invalid_declaration", "mismatch", "unreadable"}:
            errors.append(f"{item['sha256_status']}:{item['path']}")
    for item in manifests:
        if not item["valid"]:
            errors.extend(f"{item['path']}:{error}" for error in item.get("errors", ()))
    return ManifestHealthReport(
        root_label=root_label,
        artifacts=artifacts,
        manifests=manifests,
        scanned_file_count=len(files),
        ignored_file_count=ignored_count,
        unignored_model_paths=unignored,
        skipped_reparse_points=skipped,
        warnings=tuple(sorted(set(warnings))),
        errors=tuple(sorted(set(errors))),
        verify_hash=verify_hash,
        gitignore_path=ignore_path.name if ignore_path.is_file() else None,
    )


def build_manifest_health_report(
    root: str | Path,
    *,
    gitignore: str | Path | None = None,
    verify_hash: bool = False,
) -> ManifestHealthReport:
    """Report-oriented alias for :func:`scan_manifest_health`."""

    return scan_manifest_health(root, gitignore=gitignore, verify_hash=verify_hash)


__all__ = [
    "MANIFEST_HEALTH_SCHEMA",
    "MODEL_SUFFIXES",
    "ManifestHealthReport",
    "build_manifest_health_report",
    "scan_manifest_health",
]
