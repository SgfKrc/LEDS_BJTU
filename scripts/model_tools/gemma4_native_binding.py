"""Frozen identity contract for the managed Gemma 4 native binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
MARKER_FILENAME = "qlh-gemma4-native-binding.json"
LOCK_PATH = Path(__file__).with_name("gemma4_native_binding.lock.json")
ROOT = Path(__file__).resolve().parents[2]
LLAMA_LOCK_PATH = Path(__file__).with_name("llama_quantize.lock.json")


def _read_lock() -> dict[str, Any]:
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemma 4 native binding lock is unavailable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Gemma 4 native binding lock schema is invalid")
    return value


def expected_binding_marker() -> dict[str, Any]:
    """Return the canonical marker expected inside the managed package."""
    lock = _read_lock()
    required = {"marker", "package", "upstream", "patch", "abi"}
    if set(lock) != {"schema_version", *required}:
        raise RuntimeError("Gemma 4 native binding lock fields are invalid")
    package = lock["package"]
    upstream = lock["upstream"]
    patch = lock["patch"]
    abi = lock["abi"]
    if not all(isinstance(item, Mapping) for item in (package, upstream, patch, abi)):
        raise RuntimeError("Gemma 4 native binding lock sections are invalid")
    symbols = abi.get("mtmd_python_symbols")
    if not isinstance(symbols, list) or not symbols or not all(isinstance(item, str) for item in symbols):
        raise RuntimeError("Gemma 4 native binding ABI symbol list is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "marker": str(lock["marker"]),
        "package": dict(package),
        "upstream": dict(upstream),
        "patch": dict(patch),
        "abi": {
            "mtmd_python_symbols": list(symbols),
            "digest": str(abi.get("digest", "")),
        },
    }


def verify_binding_sources() -> None:
    """Verify that the frozen marker inputs still match repository sources."""
    marker = expected_binding_marker()
    patch_relative = Path(marker["patch"]["path"])
    if patch_relative.is_absolute() or ".." in patch_relative.parts:
        raise RuntimeError("Gemma 4 native binding patch path is invalid")
    patch_path = (ROOT / patch_relative).resolve(strict=False)
    try:
        patch_path.relative_to(ROOT.resolve(strict=False))
        patch_digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise RuntimeError("Gemma 4 native binding patch is unavailable") from exc
    if patch_digest != marker["patch"]["sha256"]:
        raise RuntimeError("Gemma 4 native binding patch digest does not match the lock")

    try:
        llama_lock = json.loads(LLAMA_LOCK_PATH.read_text(encoding="utf-8"))
        revision = llama_lock["upstream"]["revision"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("llama.cpp revision lock is unavailable") from exc
    if revision != marker["upstream"]["revision"]:
        raise RuntimeError("Gemma 4 native binding revision does not match the llama.cpp lock")

    symbols = marker["abi"]["mtmd_python_symbols"]
    abi_digest = hashlib.sha256(
        json.dumps(symbols, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if abi_digest != marker["abi"]["digest"]:
        raise RuntimeError("Gemma 4 native binding ABI digest does not match the lock")


def marker_path(site_packages: str | Path) -> Path:
    """Return the marker path without exposing the managed absolute path."""
    return Path(site_packages).expanduser().resolve(strict=False) / "llama_cpp" / MARKER_FILENAME


def write_binding_marker(site_packages: str | Path) -> Path:
    """Write the canonical marker after a managed binding build succeeds."""
    verify_binding_sources()
    package_root = Path(site_packages).expanduser().resolve(strict=False) / "llama_cpp"
    if not package_root.is_dir():
        raise RuntimeError("managed llama_cpp package directory is missing")
    destination = package_root / MARKER_FILENAME
    destination.write_text(
        json.dumps(expected_binding_marker(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def validate_binding_marker(site_packages: str | Path) -> None:
    """Fail closed unless the package carries the exact project marker."""
    marker = marker_path(site_packages)
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("gemma4-native binding marker is missing or invalid") from exc
    if actual != expected_binding_marker():
        raise RuntimeError("gemma4-native binding marker does not match the frozen manifest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write-marker", action="store_true")
    actions.add_argument("--check", action="store_true")
    parser.add_argument("--site-packages", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.write_marker:
        write_binding_marker(args.site_packages)
    else:
        validate_binding_marker(args.site_packages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
