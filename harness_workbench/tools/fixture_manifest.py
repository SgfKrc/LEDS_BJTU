"""Versioned fixture manifest for cross-side contracts.

Builds (or verifies) stable sha256 fingerprints for the fixtures shared between
the main project and Koakumix: prompt-set payloads (``prompts.jsonl``, the very
file the EX-N3 quality runner hashes), objective rubrics and experiment plans.
The manifest carries no timestamps, so ``--check`` is deterministic and CI-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

FIXTURE_MANIFEST_SCHEMA = "qlh.fixture_manifest.v1"
MANIFEST_NAME = "MANIFEST.json"
_PATTERNS = (
    "prompt_sets/*/prompts.jsonl",
    "quality_rubrics/*.json",
    "experiment-plans/*.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(fixture_root: Path) -> dict[str, Any]:
    """Fingerprint every tracked fixture file under ``fixture_root`` (sorted, stable)."""
    entries: list[dict[str, Any]] = []
    for pattern in _PATTERNS:
        for path in sorted(fixture_root.glob(pattern)):
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": path.relative_to(fixture_root).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    entries.sort(key=lambda item: str(item["path"]))
    return {
        "schema_version": FIXTURE_MANIFEST_SCHEMA,
        "fixture_root": fixture_root.as_posix(),
        "entry_count": len(entries),
        "entries": entries,
    }


def verify_manifest(fixture_root: Path, manifest_path: Path | None = None) -> tuple[bool, str]:
    """Return (ok, message) by comparing the stored manifest to the current files."""
    target = manifest_path or (fixture_root / MANIFEST_NAME)
    if not target.is_file():
        return False, f"manifest not found: {target}"
    stored = json.loads(target.read_text(encoding="utf-8"))
    current = build_manifest(fixture_root)
    if stored.get("entries") != current["entries"]:
        return False, "manifest entries differ from current fixture fingerprints"
    if stored.get("schema_version") != FIXTURE_MANIFEST_SCHEMA:
        return False, "manifest schema_version mismatch"
    return True, f"manifest verified ({current['entry_count']} entries)"


def _resolve_fixture_root(given: Path) -> Path:
    candidate = given.expanduser().resolve()
    nested = candidate / "fixtures"
    if nested.is_dir():
        return nested
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the versioned fixture manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root or fixtures directory")
    parser.add_argument("--check", action="store_true", help="verify the existing MANIFEST.json")
    parser.add_argument("--json", type=Path, help="manifest output path (default: <fixtures>/MANIFEST.json)")
    args = parser.parse_args(argv)
    fixture_root = _resolve_fixture_root(args.root)
    if args.check:
        ok, message = verify_manifest(fixture_root, args.json)
        print(("OK " if ok else "FAIL ") + message)
        return 0 if ok else 1
    manifest = build_manifest(fixture_root)
    out = args.json or (fixture_root / MANIFEST_NAME)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {out} ({manifest['entry_count']} entries)")
    return 0


__all__ = ["FIXTURE_MANIFEST_SCHEMA", "MANIFEST_NAME", "build_manifest", "verify_manifest", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
