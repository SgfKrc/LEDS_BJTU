"""Read-only model inventory generation and comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

INVENTORY_KIND = "qlh_model_inventory"
SCHEMA_VERSION = 1
MODEL_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
WEIGHT_SUFFIXES = MODEL_SUFFIXES - {".gguf", ".onnx"}
IGNORED_DIR_NAMES = {".cache", "models_old_backup", "__pycache__"}
IGNORED_FILE_NAMES = {
    ".ds_store",
    "thumbs.db",
    "model.sha256",
    "model.sha256.meta.json",
}
IGNORED_FILE_SUFFIXES = {".part", ".tmp", ".sha256", ".pyc"}
ASSET_KINDS = {"gguf", "model_file", "diffusion", "pytorch", "model_directory"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_reparse(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is not None:
        try:
            if checker():
                return True
        except OSError:
            pass
    try:
        return path.is_symlink()
    except OSError:
        return False


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return "<outside-root>"


def _ignore_file(path: Path) -> bool:
    name = path.name.lower()
    if name == ".qlh-sd-asset.json":
        return False
    return (
        name.startswith(".")
        or name in IGNORED_FILE_NAMES
        or any(name.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES)
    )


def _iter_asset_files(asset: Path) -> tuple[list[Path], list[str], list[str]]:
    if asset.is_file():
        return [asset], [], []
    files: list[Path] = []
    skipped: list[str] = []
    errors: list[str] = []

    def on_error(error: OSError) -> None:
        errors.append(f"directory scan failed (errno={error.errno})")

    for current, directories, names in os.walk(asset, followlinks=False, onerror=on_error):
        current_path = Path(current)
        kept: list[str] = []
        for name in directories:
            candidate = current_path / name
            if name.startswith(".") or name.lower() in IGNORED_DIR_NAMES:
                continue
            if _is_reparse(candidate):
                skipped.append(_safe_relative(candidate, asset))
            else:
                kept.append(name)
        directories[:] = kept
        for name in names:
            candidate = current_path / name
            if _ignore_file(candidate):
                continue
            if _is_reparse(candidate):
                skipped.append(_safe_relative(candidate, asset))
                continue
            try:
                if candidate.is_file():
                    files.append(candidate)
            except OSError as exc:
                errors.append(f"file inspection failed for {_safe_relative(candidate, asset)} (errno={exc.errno})")
    return sorted(files, key=lambda item: _safe_relative(item, asset)), sorted(skipped), errors


def _feed_field(digest: Any, value: str | int) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_kind(asset: Path, files: list[Path]) -> str:
    if asset.is_file():
        return "gguf" if asset.suffix.lower() == ".gguf" else "model_file"
    relative_names = {_safe_relative(item, asset) for item in files}
    suffixes = {item.suffix.lower() for item in files}
    if ".qlh-sd-asset.json" in relative_names or "model_index.json" in relative_names:
        return "diffusion"
    if suffixes & WEIGHT_SUFFIXES:
        return "pytorch"
    return "model_directory"


def _build_asset(asset: Path, *, full_hash: bool) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    files, skipped, errors = _iter_asset_files(asset)
    if not files:
        return None, skipped, errors
    structure = hashlib.sha256()
    content = hashlib.sha256() if full_hash else None
    logical_size = 0
    base = asset if asset.is_dir() else asset.parent
    for item in files:
        relative = _safe_relative(item, base) if asset.is_dir() else "."
        try:
            size = int(item.stat().st_size)
        except OSError as exc:
            errors.append(f"stat failed for {relative} (errno={exc.errno})")
            continue
        logical_size += size
        _feed_field(structure, relative)
        _feed_field(structure, size)
        if content is not None:
            try:
                file_digest = _content_sha256(item)
            except OSError as exc:
                errors.append(f"hash failed for {relative} (errno={exc.errno})")
                continue
            _feed_field(content, relative)
            _feed_field(content, size)
            _feed_field(content, file_digest)
    if errors:
        return None, skipped, errors
    return {
        "asset_id": asset.name,
        "kind": _asset_kind(asset, files),
        "logical_size_bytes": logical_size,
        "file_count": len(files),
        "structure_digest": structure.hexdigest(),
        "content_digest": content.hexdigest() if content is not None else None,
    }, skipped, []


def build_inventory(root: str | Path, *, full_hash: bool = False) -> dict[str, Any]:
    """Build an inventory without writing to or following links inside the model root."""
    target = Path(root).expanduser().absolute()
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inventory_kind": INVENTORY_KIND,
        "tool": "models_sync_status",
        "operation": "inventory",
        "hash_mode": "sha256" if full_hash else "structure",
        "read_only": True,
        "assets": [],
        "warnings": [],
        "errors": [],
    }
    if not target.exists():
        return {**base, "valid": False, "errors": ["model root does not exist"]}
    if not target.is_dir():
        return {**base, "valid": False, "errors": ["model root is not a directory"]}
    try:
        entries = sorted(target.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return {**base, "valid": False, "errors": [f"model root cannot be listed (errno={exc.errno})"]}
    for entry in entries:
        if entry.name.startswith(".") or entry.name.lower() in IGNORED_DIR_NAMES:
            continue
        if _is_reparse(entry):
            base["warnings"].append(f"junction/reparse point not traversed: {entry.name}")
            continue
        try:
            is_file = entry.is_file()
            is_dir = entry.is_dir()
        except OSError as exc:
            base["errors"].append(f"entry inspection failed for {entry.name} (errno={exc.errno})")
            continue
        if is_file and (entry.suffix.lower() not in MODEL_SUFFIXES or _ignore_file(entry)):
            continue
        if not is_file and not is_dir:
            continue
        asset, skipped, errors = _build_asset(entry, full_hash=full_hash)
        base["warnings"].extend(
            f"junction/reparse point not traversed: {entry.name}/{item}" for item in skipped
        )
        base["errors"].extend(f"{entry.name}: {error}" for error in errors)
        if asset is not None:
            base["assets"].append(asset)
    base["assets"].sort(key=lambda item: item["asset_id"])
    base["warnings"] = sorted(set(base["warnings"]))
    base["valid"] = not base["errors"]
    return base


def _valid_asset_id(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
    ):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and len(path.parts) == 1 and path.parts[0] not in {".", ".."}


def validate_inventory(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and normalize an untrusted inventory before comparison."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["inventory must be a JSON object"]
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version; expected {SCHEMA_VERSION}")
    if value.get("inventory_kind") != INVENTORY_KIND:
        errors.append(f"inventory_kind must be {INVENTORY_KIND}")
    hash_mode = value.get("hash_mode")
    if hash_mode not in {"structure", "sha256"}:
        errors.append("hash_mode must be structure or sha256")
    raw_assets = value.get("assets")
    if not isinstance(raw_assets, list):
        errors.append("assets must be an array")
        raw_assets = []
    elif len(raw_assets) > 10000:
        errors.append("assets exceeds the 10000 item limit")
        raw_assets = []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_assets):
        prefix = f"assets[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{prefix} must be an object")
            continue
        asset_id = raw.get("asset_id")
        if isinstance(asset_id, str) and asset_id in seen:
            errors.append(f"duplicate asset_id: {asset_id}")
        if isinstance(asset_id, str):
            seen.add(asset_id)
        if not _valid_asset_id(asset_id):
            errors.append(f"{prefix}.asset_id must be one relative path component")
            continue
        kind = raw.get("kind")
        size = raw.get("logical_size_bytes")
        file_count = raw.get("file_count")
        structure_digest = raw.get("structure_digest")
        content_digest = raw.get("content_digest")
        if kind not in ASSET_KINDS:
            errors.append(f"{prefix}.kind is unsupported")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"{prefix}.logical_size_bytes must be a non-negative integer")
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
            errors.append(f"{prefix}.file_count must be a positive integer")
        if not isinstance(structure_digest, str) or HEX64.fullmatch(structure_digest) is None:
            errors.append(f"{prefix}.structure_digest must be lowercase SHA-256")
        if hash_mode == "sha256":
            if not isinstance(content_digest, str) or HEX64.fullmatch(content_digest) is None:
                errors.append(f"{prefix}.content_digest must be lowercase SHA-256 in sha256 mode")
        elif content_digest is not None:
            errors.append(f"{prefix}.content_digest must be null in structure mode")
        normalized.append({
            "asset_id": asset_id,
            "kind": kind,
            "logical_size_bytes": size,
            "file_count": file_count,
            "structure_digest": structure_digest,
            "content_digest": content_digest,
        })
    if errors:
        return None, errors
    return {
        "schema_version": SCHEMA_VERSION,
        "inventory_kind": INVENTORY_KIND,
        "hash_mode": hash_mode,
        "assets": sorted(normalized, key=lambda item: item["asset_id"]),
    }, []


def load_inventory(path: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as exc:
        return None, [f"inventory file cannot be read (errno={exc.errno})"]
    except (UnicodeError, json.JSONDecodeError):
        return None, ["inventory file is not valid UTF-8 JSON"]
    return validate_inventory(value)


def compare_inventories(local: Any, peer: Any) -> dict[str, Any]:
    """Compare two validated or untrusted inventories without exposing source paths."""
    local_inventory, local_errors = validate_inventory(local)
    peer_inventory, peer_errors = validate_inventory(peer)
    errors = [f"local: {error}" for error in local_errors] + [f"peer: {error}" for error in peer_errors]
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "models_sync_status",
        "operation": "compare",
        "read_only": True,
        "valid": False,
        "in_sync": False,
        "missing_on_peer": [],
        "extra_on_peer": [],
        "mismatched": [],
        "matched_count": 0,
        "errors": errors,
    }
    if errors or local_inventory is None or peer_inventory is None:
        return base
    if local_inventory["hash_mode"] != peer_inventory["hash_mode"]:
        base["errors"] = ["local and peer hash_mode must match"]
        return base
    local_assets = {item["asset_id"]: item for item in local_inventory["assets"]}
    peer_assets = {item["asset_id"]: item for item in peer_inventory["assets"]}
    local_ids = set(local_assets)
    peer_ids = set(peer_assets)
    missing = sorted(local_ids - peer_ids)
    extra = sorted(peer_ids - local_ids)
    mismatched: list[dict[str, Any]] = []
    matched_count = 0
    fields = ["kind", "logical_size_bytes", "file_count", "structure_digest"]
    if local_inventory["hash_mode"] == "sha256":
        fields.append("content_digest")
    for asset_id in sorted(local_ids & peer_ids):
        changed = [field for field in fields if local_assets[asset_id][field] != peer_assets[asset_id][field]]
        if changed:
            mismatched.append({"asset_id": asset_id, "changed_fields": changed})
        else:
            matched_count += 1
    in_sync = not missing and not extra and not mismatched
    base.update({
        "valid": True,
        "in_sync": in_sync,
        "hash_mode": local_inventory["hash_mode"],
        "local_asset_count": len(local_assets),
        "peer_asset_count": len(peer_assets),
        "missing_on_peer": missing,
        "extra_on_peer": extra,
        "mismatched": mismatched,
        "matched_count": matched_count,
        "errors": [],
    })
    return base


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write an explicitly requested JSON result outside model assets."""
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "build_inventory",
    "compare_inventories",
    "load_inventory",
    "validate_inventory",
    "write_json",
]
