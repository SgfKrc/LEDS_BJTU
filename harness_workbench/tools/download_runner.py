"""Offline-first, resumable model artifact download runner.

The runner owns only the artifact transfer contract. It validates a pinned
manifest, writes into a private ``.part`` staging file, resumes with HTTP
Range when available, verifies the complete SHA-256, writes a sidecar and
atomically publishes the target. Real HTTP is disabled by default; tests and
offline demonstrations use the injectable in-memory transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .manifest_health import scan_manifest_health


DOWNLOAD_RUNNER_SCHEMA = "qlh.harness.download_runner.v1"
DOWNLOAD_MANIFEST_SCHEMA = "qlh.download_manifest.v1"
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_REVISION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,127}$")
_FLOATING_REVISIONS = frozenset({"main", "master", "latest", "default", "head"})
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class DownloadError(ValueError):
    """Stable, redacted transfer failure."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DownloadError("invalid_manifest", f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if _ABSOLUTE_PATH.search(normalized) or path.is_absolute() or ".." in path.parts or "://" in normalized:
        raise DownloadError("unsafe_path", f"{label} must stay inside the target root")
    return path.as_posix()


def _validate_url(value: Any, *, label: str = "url") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise DownloadError("invalid_source", f"{label} is invalid")
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise DownloadError("invalid_source", f"{label} syntax is invalid") from exc
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or (port is not None and not 1 <= port <= 65535):
        raise DownloadError("unsafe_source", f"{label} must be HTTPS without credentials")
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class DownloadFilePin:
    path: str
    url: str
    sha256: str
    size_bytes: int
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative(self.path, label="file.path"))
        object.__setattr__(self, "url", _validate_url(self.url))
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise DownloadError("invalid_manifest", "file.sha256 must be a SHA-256 digest")
        object.__setattr__(self, "sha256", self.sha256.lower())
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise DownloadError("invalid_manifest", "file.size_bytes must be a non-negative integer")
        if not isinstance(self.required, bool):
            raise DownloadError("invalid_manifest", "file.required must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "url": self.url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class DownloadManifest:
    model_id: str
    revision: str
    target_root: str
    files: tuple[DownloadFilePin, ...]
    source_ref: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = DOWNLOAD_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DOWNLOAD_MANIFEST_SCHEMA or not isinstance(self.model_id, str) or not self.model_id.strip():
            raise DownloadError("invalid_manifest", "manifest identity is invalid")
        if not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision) or self.revision.lower() in _FLOATING_REVISIONS:
            raise DownloadError("unpinned_revision", "manifest revision must be a non-floating pinned value")
        object.__setattr__(self, "target_root", _safe_relative(self.target_root, label="target_root"))
        if not self.files:
            raise DownloadError("invalid_manifest", "manifest requires at least one file")
        if len({item.path for item in self.files}) != len(self.files):
            raise DownloadError("duplicate_file", "manifest file paths must be unique")
        if self.source_ref and (_ABSOLUTE_PATH.search(self.source_ref) or "://" in self.source_ref):
            raise DownloadError("unsafe_metadata", "manifest source_ref must be a local relative reference")
        if not _path_free(self.metadata):
            raise DownloadError("unsafe_metadata", "manifest metadata contains an unsafe path")

    @property
    def manifest_id(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    @property
    def pinned_sha_revision(self) -> bool:
        return bool(re.fullmatch(r"[0-9a-f]{40}", self.revision.lower()))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "model_id": self.model_id,
            "revision": self.revision,
            "target_root": self.target_root,
            "source_ref": self.source_ref,
            "files": [item.as_dict() for item in self.files],
            "metadata": dict(self.metadata),
        }
        if include_digest:
            value["manifest_id"] = self.manifest_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DownloadManifest":
        if not isinstance(value, Mapping) or value.get("schema") != DOWNLOAD_MANIFEST_SCHEMA:
            raise DownloadError("invalid_schema", "unsupported or missing download manifest schema")
        files_value = value.get("files")
        if not isinstance(files_value, list):
            raise DownloadError("invalid_manifest", "manifest files must be an array")
        base = value.get("source_base_url")
        if base is not None:
            base = _validate_url(base, label="source_base_url").rstrip("/") + "/"
        files: list[DownloadFilePin] = []
        for index, raw in enumerate(files_value):
            if not isinstance(raw, Mapping):
                raise DownloadError("invalid_manifest", f"files[{index}] must be an object")
            path = raw.get("path")
            url = raw.get("url")
            if url is None and base:
                url = urllib.parse.urljoin(base, _safe_relative(path, label=f"files[{index}].path"))
            files.append(
                DownloadFilePin(
                    path=path,
                    url=url,
                    sha256=raw.get("sha256"),
                    size_bytes=raw.get("size_bytes"),
                    required=raw.get("required", True),
                )
            )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise DownloadError("invalid_manifest", "manifest metadata must be an object")
        return cls(
            model_id=value.get("model_id", ""),
            revision=value.get("revision", ""),
            target_root=value.get("target_root", ""),
            files=tuple(files),
            source_ref=str(value.get("source_ref", "")),
            metadata=dict(metadata),
        )


def _path_free(value: Any) -> bool:
    if isinstance(value, str):
        return not _ABSOLUTE_PATH.search(value)
    if isinstance(value, Mapping):
        return all(_path_free(key) and _path_free(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return all(_path_free(child) for child in value)
    return True


@dataclass(frozen=True, slots=True)
class DownloadResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class DownloadTransport(Protocol):
    network_used: bool

    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float, max_bytes: int) -> DownloadResponse:
        ...


class UrllibDownloadTransport:
    """Real HTTPS transport, opt-in only and never enabled by the CLI default."""

    network_used = True

    def __init__(self, *, enabled: bool = False, timeout_seconds: float = 5.0) -> None:
        self.enabled = bool(enabled)
        self.network_used = self.enabled
        self.timeout_seconds = float(timeout_seconds)

    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float, max_bytes: int) -> DownloadResponse:
        if not self.enabled:
            raise DownloadError("network_disabled", "real download transport is disabled", retryable=False)
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_seconds, timeout_seconds)) as response:
                body = response.read(max_bytes + 1)
                return DownloadResponse(int(response.status), dict(response.headers.items()), body)
        except urllib.error.HTTPError as exc:
            body = exc.read(max_bytes + 1)
            return DownloadResponse(int(exc.code), dict(exc.headers.items()), body)
        except (OSError, TimeoutError) as exc:
            raise DownloadError("network_error", "download transport failed", retryable=True) from exc


class MemoryDownloadTransport:
    """Deterministic Range-aware transport for offline tests and demonstrations."""

    network_used = False

    def __init__(self, payloads: Mapping[str, bytes], *, failures: Mapping[str, Sequence[str]] | None = None) -> None:
        self.payloads = {str(key): bytes(value) for key, value in payloads.items()}
        self.failures = {str(key): list(value) for key, value in (failures or {}).items()}
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(self, url: str, *, headers: Mapping[str, str], timeout_seconds: float, max_bytes: int) -> DownloadResponse:
        del timeout_seconds
        self.calls.append((url, dict(headers)))
        failures = self.failures.get(url, [])
        if failures:
            failure = failures.pop(0)
            if failure == "timeout":
                raise DownloadError("timeout", "fixture transport timeout", retryable=True)
            status = int(failure)
            return DownloadResponse(status, {}, b"")
        if url not in self.payloads:
            return DownloadResponse(404, {}, b"")
        payload = self.payloads[url]
        range_value = str(headers.get("Range", ""))
        start = 0
        end: int | None = None
        if range_value.startswith("bytes="):
            try:
                start_text, end_text = range_value[6:].split("-", 1)
                start = int(start_text)
                end = int(end_text) if end_text else None
            except ValueError:
                return DownloadResponse(416, {}, b"")
        if start > len(payload):
            return DownloadResponse(416, {"Content-Range": f"bytes */{len(payload)}"}, b"")
        body = payload[start : end + 1 if end is not None else None]
        if len(body) > max_bytes:
            return DownloadResponse(413, {}, b"")
        if start or end is not None:
            headers_out = {
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(payload)}",
            }
            return DownloadResponse(206, headers_out, body)
        return DownloadResponse(200, {"Accept-Ranges": "bytes", "Content-Length": str(len(body))}, body)


@dataclass(frozen=True, slots=True)
class DownloadFileResult:
    path: str
    status: str
    attempts: int
    downloaded_bytes: int
    resumed: bool
    expected_sha256: str
    actual_sha256: str | None = None
    sidecar_path: str | None = None
    error_code: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "attempts": self.attempts,
            "downloaded_bytes": self.downloaded_bytes,
            "resumed": self.resumed,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "sidecar_path": self.sidecar_path,
            "error_code": self.error_code,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DownloadRunReport:
    manifest_id: str
    model_id: str
    revision: str
    target_root: str
    status: str
    files: tuple[DownloadFileResult, ...]
    health_summary: Mapping[str, Any]
    limitations: tuple[str, ...]
    runner_kind: str = "plan"
    network_used: bool = False
    weights_loaded: bool = False
    schema: str = DOWNLOAD_RUNNER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DOWNLOAD_RUNNER_SCHEMA or self.status not in {"planned", "ready", "already_ready", "blocked", "failed"}:
            raise DownloadError("invalid_report", "download report identity is invalid")
        if self.network_used and self.runner_kind == "plan":
            raise DownloadError("invalid_report", "plan report cannot use network")
        if self.weights_loaded:
            raise DownloadError("invalid_report", "download runner cannot load weights")

    @property
    def checks(self) -> dict[str, bool]:
        required = [item for item in self.files if item.status != "skipped_optional"]
        return {
            "manifest_id_present": bool(self.manifest_id),
            "target_paths_relative": all(not _ABSOLUTE_PATH.search(item.path) and ".." not in PurePosixPath(item.path).parts for item in self.files),
            "sha256_verified": self.status == "planned" or all(item.status in {"ready", "already_ready"} and item.actual_sha256 == item.expected_sha256 for item in required),
            "no_failed_files": all(item.status not in {"failed", "blocked"} for item in self.files),
            "sidecar_recorded": self.status == "planned" or all(bool(item.sidecar_path) for item in required if item.status in {"ready", "already_ready"}),
            "post_download_health": self.status == "planned" or bool(self.health_summary.get("valid", False)),
            "weights_not_loaded": not self.weights_loaded,
        }

    @property
    def valid(self) -> bool:
        return self.status in {"planned", "ready", "already_ready"} and all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "manifest_id": self.manifest_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "target_root": self.target_root,
            "status": self.status,
            "runner_kind": self.runner_kind,
            "files": [item.as_dict() for item in self.files],
            "health_summary": dict(self.health_summary),
            "limitations": list(self.limitations),
            "checks": self.checks,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-DL-RUNNER-01 download report",
            "",
            f"- Model: `{self.model_id}`; revision: `{self.revision}`; status: `{self.status}`; valid: `{str(self.valid).lower()}`",
            f"- Runner: `{self.runner_kind}`; network used: `{str(self.network_used).lower()}`; weights loaded: `false`; report digest: `{self.digest}`",
            "",
            "## Files",
            "",
            "| path | status | attempts | bytes | resumed | SHA-256 | error |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        for item in self.files:
            lines.append(f"| `{item.path}` | `{item.status}` | {item.attempts} | {item.downloaded_bytes} | {str(item.resumed).lower()} | `{item.actual_sha256 or 'NOT VERIFIED'}` | `{item.error_code or ''}` |")
        lines.extend(("", "## Post-download health", "", f"- Valid: `{str(self.health_summary.get('valid', False)).lower()}`; artifacts: `{self.health_summary.get('artifact_count', 0)}`; errors: `{'; '.join(self.health_summary.get('errors', [])) or '-'}`", "", "## Limitations", ""))
        lines.extend(f"- {item}" for item in self.limitations)
        lines.extend(("", "## Checks", ""))
        lines.extend(f"- `{name}`: **{'passed' if passed else 'failed'}**" for name, passed in self.checks.items())
        lines.append("")
        return "\n".join(lines)


class DownloadRunner:
    """Perform a pinned transfer into a local root with no implicit network."""

    def __init__(self, root: str | Path, *, transport: DownloadTransport | None = None, max_attempts: int = 3, timeout_seconds: float = 5.0, max_response_bytes: int = _MAX_RESPONSE_BYTES, chunk_bytes: int = 16 * 1024 * 1024) -> None:
        self.root = Path(root).expanduser().absolute()
        self.transport = transport
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 8:
            raise DownloadError("invalid_arguments", "max_attempts must be between 1 and 8")
        if not 0.1 <= float(timeout_seconds) <= 30.0:
            raise DownloadError("invalid_arguments", "timeout_seconds must be between 0.1 and 30")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or not 1024 <= max_response_bytes <= _MAX_RESPONSE_BYTES:
            raise DownloadError("invalid_arguments", "max_response_bytes is outside the policy")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or not 1024 <= chunk_bytes <= max_response_bytes:
            raise DownloadError("invalid_arguments", "chunk_bytes is outside the response policy")
        self.max_attempts = max_attempts
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = max_response_bytes
        self.chunk_bytes = chunk_bytes

    def run(self, manifest: DownloadManifest, *, allow_transport: bool = False, dry_run: bool = False) -> DownloadRunReport:
        if not isinstance(manifest, DownloadManifest):
            raise DownloadError("invalid_manifest", "runner requires a DownloadManifest")
        if dry_run:
            planned = tuple(DownloadFileResult(item.path, "planned", 0, 0, False, item.sha256) for item in manifest.files)
            return self._report(manifest, "planned", planned, runner_kind="plan", network_used=False)
        results: list[DownloadFileResult] = []
        for item in manifest.files:
            result = self._run_file(manifest, item, allow_transport=allow_transport)
            results.append(result)
            if result.status in {"failed", "blocked"} and item.required:
                break
        if any(item.status == "failed" for item in results):
            status = "failed"
        elif any(item.status == "blocked" for item in results):
            status = "blocked"
        elif all(item.status == "already_ready" for item in results):
            status = "already_ready"
        else:
            status = "ready"
        network_used = bool(allow_transport and self.transport is not None and getattr(self.transport, "network_used", True))
        runner_kind = "plan" if self.transport is None else ("transport" if network_used else "fixture")
        return self._report(manifest, status, tuple(results), runner_kind=runner_kind, network_used=network_used)

    def _target(self, manifest: DownloadManifest, relative: str) -> Path:
        if self.root.is_symlink() or getattr(self.root, "is_junction", lambda: False)():
            raise DownloadError("reparse_point", "download root cannot be a symlink or junction")
        root = self.root.resolve(strict=False)
        target = root / Path(*PurePosixPath(manifest.target_root, relative).parts)
        try:
            target.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise DownloadError("unsafe_path", "download target escapes root") from exc
        current = root
        for part in PurePosixPath(manifest.target_root, relative).parts[:-1]:
            current = current / part
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                raise DownloadError("reparse_point", "download target crosses a symlink or junction")
        return target

    def _run_file(self, manifest: DownloadManifest, item: DownloadFilePin, *, allow_transport: bool) -> DownloadFileResult:
        try:
            target = self._target(manifest, item.path)
        except DownloadError as exc:
            return DownloadFileResult(item.path, "failed", 0, 0, False, item.sha256, error_code=exc.code, error=str(exc))
        if target.exists():
            if not target.is_file() or target.is_symlink():
                return DownloadFileResult(item.path, "failed", 0, 0, False, item.sha256, error_code="target_not_file", error="target is not a regular file")
            try:
                actual = _sha256_file(target)
            except OSError as exc:
                return DownloadFileResult(item.path, "failed", 0, 0, False, item.sha256, error_code="target_unreadable", error=type(exc).__name__)
            if target.stat().st_size == item.size_bytes and actual == item.sha256:
                sidecar = target.with_name(target.name + ".sha256")
                _atomic_text(sidecar, f"{item.sha256}  {target.name}\n")
                return DownloadFileResult(item.path, "already_ready", 0, item.size_bytes, False, item.sha256, actual, sidecar.relative_to(self.root).as_posix())
            return DownloadFileResult(item.path, "failed", 0, int(target.stat().st_size), False, item.sha256, actual, error_code="target_hash_mismatch", error="existing target does not match pinned SHA-256")
        if not allow_transport or self.transport is None:
            return DownloadFileResult(item.path, "blocked", 0, 0, False, item.sha256, error_code="network_disabled", error="transport is not enabled")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            for parent in [target.parent, *target.parent.parents]:
                if parent == self.root.parent:
                    break
                if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
                    raise DownloadError("reparse_point", "download target crosses a symlink or junction")
        except OSError as exc:
            return DownloadFileResult(item.path, "failed", 0, 0, False, item.sha256, error_code="target_unwritable", error=type(exc).__name__)
        part = target.with_name(target.name + ".part")
        state = part.with_name(part.name + ".json")
        try:
            resumed = self._prepare_part(part, state, manifest, item)
        except DownloadError as exc:
            return DownloadFileResult(item.path, "failed", 0, 0, False, item.sha256, error_code=exc.code, error=str(exc))
        attempts = 0
        consecutive_failures = 0
        last_error: DownloadError | None = None
        while True:
            attempts += 1
            try:
                current = part.stat().st_size if part.exists() else 0
                resumed = resumed or bool(current)
                headers = {"Accept": "application/octet-stream"}
                if item.size_bytes:
                    end = min(item.size_bytes - 1, current + self.chunk_bytes - 1)
                    headers["Range"] = f"bytes={current}-{end}"
                response = self.transport.get(item.url, headers=headers, timeout_seconds=self.timeout_seconds, max_bytes=self.max_response_bytes)
                if response.status_code in _RETRY_STATUSES:
                    raise DownloadError(f"http_{response.status_code}", "remote transfer is retryable", retryable=True)
                if response.status_code == 416 and current:
                    part.write_bytes(b"")
                    resumed = False
                    continue
                if response.status_code not in {200, 206}:
                    raise DownloadError(f"http_{response.status_code}", "remote transfer was rejected")
                if current and response.status_code == 200:
                    part.write_bytes(b"")
                    current = 0
                    resumed = False
                elif current and response.status_code != 206:
                    raise DownloadError("range_unsupported", "server did not honor the resume range")
                if len(response.body) > self.max_response_bytes:
                    raise DownloadError("response_too_large", "response exceeds transfer limit")
                with part.open("ab") as handle:
                    handle.write(response.body)
                downloaded = part.stat().st_size
                if downloaded > item.size_bytes:
                    raise DownloadError("size_exceeded", "download exceeded pinned size")
                _atomic_json(state, {"schema": DOWNLOAD_RUNNER_SCHEMA, "manifest_id": manifest.manifest_id, "path": item.path, "url": item.url, "expected_sha256": item.sha256, "expected_size": item.size_bytes, "downloaded_bytes": downloaded})
                if response.body:
                    consecutive_failures = 0
                if downloaded < item.size_bytes:
                    if not response.body:
                        raise DownloadError("partial_response", "transfer ended before the pinned size", retryable=True)
                    continue
                actual = _sha256_file(part)
                if actual != item.sha256:
                    raise DownloadError("sha256_mismatch", "downloaded bytes do not match the pinned SHA-256")
                os.replace(part, target)
                sidecar = target.with_name(target.name + ".sha256")
                _atomic_text(sidecar, f"{item.sha256}  {target.name}\n")
                try:
                    state.unlink()
                except FileNotFoundError:
                    pass
                return DownloadFileResult(item.path, "ready", attempts, item.size_bytes, resumed, item.sha256, actual, sidecar.relative_to(self.root).as_posix())
            except DownloadError as exc:
                last_error = exc
                if not exc.retryable:
                    break
                consecutive_failures += 1
                if consecutive_failures >= self.max_attempts:
                    break
            except OSError as exc:
                last_error = DownloadError("io_error", "staging file operation failed", retryable=True)
                consecutive_failures += 1
                if consecutive_failures >= self.max_attempts:
                    break
        downloaded = part.stat().st_size if part.exists() else 0
        error = last_error or DownloadError("download_failed", "download failed")
        return DownloadFileResult(item.path, "failed", attempts, downloaded, resumed, item.sha256, error_code=error.code, error=str(error))

    def _prepare_part(self, part: Path, state: Path, manifest: DownloadManifest, item: DownloadFilePin) -> bool:
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise DownloadError("reparse_point", "staging file cannot be a symlink or junction")
        if not part.exists():
            return False
        try:
            payload = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        matching = payload.get("manifest_id") == manifest.manifest_id and payload.get("path") == item.path and payload.get("expected_sha256") == item.sha256 and payload.get("expected_size") == item.size_bytes
        if not matching or part.stat().st_size > item.size_bytes:
            part.write_bytes(b"")
            try:
                state.unlink()
            except FileNotFoundError:
                pass
            return False
        return part.stat().st_size > 0

    def _report(self, manifest: DownloadManifest, status: str, files: tuple[DownloadFileResult, ...], *, runner_kind: str, network_used: bool) -> DownloadRunReport:
        health: dict[str, Any] = {"valid": False, "artifact_count": 0, "scanned_file_count": 0, "errors": [], "warnings": []}
        if self.root.is_dir() and status != "planned":
            report = scan_manifest_health(self.root)
            health = {"valid": report.valid, "artifact_count": len(report.artifacts), "scanned_file_count": report.scanned_file_count, "manifest_count": len(report.manifests), "errors": list(report.errors), "warnings": list(report.warnings), "report_digest": report.digest}
        limitations = [
            "This runner verifies pinned bytes and local staging only; it does not load a model or run a quality smoke test.",
            "Real HTTPS transport is disabled unless an explicit transport and allow_transport=True are supplied.",
        ]
        if not manifest.pinned_sha_revision:
            limitations.append("Revision is syntactically pinned but is not a 40-character commit SHA; release policy may require a full revision pin.")
        if health.get("errors"):
            limitations.append("Post-download manifest health contains errors and must be repaired before registration.")
        return DownloadRunReport(manifest.manifest_id, manifest.model_id, manifest.revision, manifest.target_root, status, files, health, tuple(limitations), runner_kind=runner_kind, network_used=network_used)


def load_download_manifest(path: str | Path) -> DownloadManifest:
    source = Path(path)
    if source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise DownloadError("manifest_too_large", "download manifest exceeds the size limit")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError("invalid_manifest", "download manifest is unreadable or invalid") from exc
    return DownloadManifest.from_dict(payload)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or execute a pinned, resumable model artifact transfer")
    parser.add_argument("--manifest", required=True, metavar="PATH", help="qlh.download_manifest.v1 JSON")
    parser.add_argument("--root", default="models", metavar="PATH", help="target model root")
    parser.add_argument("--execute", action="store_true", help="attempt the configured transport; default is metadata-only plan")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_download_manifest(args.manifest)
        runner = DownloadRunner(args.root)
        report = runner.run(manifest, allow_transport=False, dry_run=not args.execute)
    except (OSError, ValueError, DownloadError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    outputs = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        outputs += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        outputs += 1
    if not outputs:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "DOWNLOAD_MANIFEST_SCHEMA",
    "DOWNLOAD_RUNNER_SCHEMA",
    "DownloadError",
    "DownloadFilePin",
    "DownloadFileResult",
    "DownloadManifest",
    "DownloadResponse",
    "DownloadRunReport",
    "DownloadRunner",
    "MemoryDownloadTransport",
    "UrllibDownloadTransport",
    "build_parser",
    "load_download_manifest",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
