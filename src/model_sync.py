"""Tailnet model synchronization for PyTorch pipeline workers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import model_config as mc


_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_ARTIFACT_SUFFIXES = (
    ".safetensors", ".bin", ".json", ".py", ".tiktoken",
    ".model", ".txt", ".jinja", ".spm", ".vocab",
)
_HASH_META_NAME = "model.sha256.meta.json"
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def compute_file_sha256(path: str | Path) -> str:
    """Hash one file and cache only while its size and mtime stay unchanged."""
    value_path = Path(path)
    stat = value_path.stat()
    key = (str(value_path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with value_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def compute_model_sha256(model_path: str, *, use_cache: bool = True) -> str:
    """Hash every executable PyTorch model artifact in stable path order."""
    root = Path(model_path)
    if not root.is_dir():
        return ""

    cache_path = root / "model.sha256"
    metadata_path = root / _HASH_META_NAME
    artifact_files = sorted(
        path for path in root.rglob("*")
        if (
            path.is_file()
            and path.name not in {"model.sha256", _HASH_META_NAME}
            and not path.name.endswith(".part")
            and path.name.lower().endswith(_ARTIFACT_SUFFIXES)
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )
    if not artifact_files:
        return ""

    fingerprint = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in artifact_files
    ]
    if use_cache and cache_path.is_file() and metadata_path.is_file():
        try:
            cached = cache_path.read_text(encoding="utf-8").strip().split()[0]
            cached_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if len(cached) == 64 and cached_meta == fingerprint:
                return cached
        except (OSError, IndexError, json.JSONDecodeError):
            pass

    digest = hashlib.sha256()
    for path in artifact_files:
        relative = path.relative_to(root).as_posix()
        file_sha = compute_file_sha256(path)
        digest.update(
            f"{relative}\0{path.stat().st_size}\0{file_sha}\n".encode("utf-8")
        )
    value = digest.hexdigest()
    try:
        cache_path.write_text(f"{value}  {root.name}\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(fingerprint, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass
    return value


def resolve_worker_model_path(model_id: str) -> str:
    """Resolve a registry model path, or a safe local external-model directory."""
    if not _MODEL_ID_RE.fullmatch(model_id or ""):
        raise ValueError(f"invalid model_id: {model_id!r}")
    config = mc.get_model_config(model_id)
    if config and config.model_path:
        return os.path.abspath(mc.resolve_model_path(config.model_path))
    return os.path.abspath(mc.resolve_model_path(os.path.join("models", model_id)))


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model manifest HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"model manifest request failed: {exc.reason}") from exc


def _safe_destination(root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"unsafe model file path: {relative_path!r}")
    destination = (root / normalized).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"model file escapes destination: {relative_path!r}") from exc
    return destination


def _file_sha256(path: Path) -> str:
    # 同步路径必须验证当前内容；compute_file_sha256 的 stat key 会让已替换文件失效。
    return compute_file_sha256(path)


def _download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if 0 < offset < expected_size else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = offset > 0 and getattr(response, "status", None) == 206
            with partial.open("ab" if append else "wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"model file download failed: {exc.reason}") from exc

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {destination.name}: "
            f"actual={actual_size}, expected={expected_size}"
        )
    os.replace(partial, destination)


def ensure_model_available(
    master_host: str,
    master_api_port: int,
    model_id: str,
    expected_sha256: str,
    *,
    timeout: float = 10.0,
) -> str:
    """Ensure the worker owns the master's exact active PyTorch model."""
    encoded_model_id = urllib.parse.quote(model_id, safe="")
    base_url = f"http://{master_host}:{int(master_api_port)}"
    manifest = _read_json(
        f"{base_url}/api/models/downloadable?model_id={encoded_model_id}",
        timeout,
    )
    if manifest.get("model_id") != model_id:
        raise RuntimeError("master returned a different model manifest")
    manifest_sha256 = str(manifest.get("sha256", ""))
    if expected_sha256 and manifest_sha256 != expected_sha256:
        raise RuntimeError("master model manifest changed while configuring pipeline")

    destination_root = Path(resolve_worker_model_path(model_id))
    destination_root.mkdir(parents=True, exist_ok=True)
    files = manifest.get("files") or []
    if not files:
        raise RuntimeError("master model manifest contains no files")

    def _sync_entry(entry: dict[str, Any], force: bool = False) -> None:
        relative_path = str(entry.get("path", ""))
        expected_size = int(entry.get("size_bytes", -1))
        expected_file_sha = str(entry.get("sha256", ""))
        if expected_size < 0:
            raise RuntimeError(f"invalid model file size: {relative_path}")
        destination = _safe_destination(destination_root, relative_path)
        valid = destination.is_file() and destination.stat().st_size == expected_size
        if valid and expected_file_sha:
            valid = _file_sha256(destination) == expected_file_sha
        if force or not valid:
            encoded_path = urllib.parse.quote(relative_path.replace("\\", "/"), safe="/")
            _download_file(
                f"{base_url}/api/models/files/{encoded_model_id}/{encoded_path}",
                destination,
                expected_size,
            )
            if expected_file_sha and _file_sha256(destination) != expected_file_sha:
                raise RuntimeError(f"SHA256 mismatch after download: {relative_path}")

    for entry in files:
        _sync_entry(entry)

    # The worker directory is an exact mirror of the active model. Old shards
    # left by a previous revision participate in the aggregate digest and can
    # otherwise make synchronization fail forever.
    manifest_artifacts = {
        str(entry.get("path", "")).replace("\\", "/").lstrip("/")
        for entry in files
        if str(entry.get("path", "")).lower().endswith(_ARTIFACT_SUFFIXES)
    }
    for local_artifact in destination_root.rglob("*"):
        relative_path = local_artifact.relative_to(destination_root)
        if (not local_artifact.is_file()
                or local_artifact.name in {"model.sha256", _HASH_META_NAME}
                or not local_artifact.name.lower().endswith(_ARTIFACT_SUFFIXES)
                or any(part.startswith(".") for part in relative_path.parts)):
            continue
        relative = relative_path.as_posix()
        if relative not in manifest_artifacts:
            local_artifact.unlink()

    expected = expected_sha256 or manifest_sha256
    actual_sha256 = compute_model_sha256(str(destination_root), use_cache=False)
    if expected and actual_sha256 != expected:
        # A file may have changed after the first validation pass; force an exact mirror.
        for entry in files:
            if str(entry.get("path", "")).lower().endswith(_ARTIFACT_SUFFIXES):
                _sync_entry(entry, force=True)
        actual_sha256 = compute_model_sha256(str(destination_root), use_cache=False)
    if expected and actual_sha256 != expected:
        raise RuntimeError(
            f"model SHA256 mismatch after synchronization: "
            f"local={actual_sha256[:16]}..., master={expected[:16]}..."
        )
    return str(destination_root)
