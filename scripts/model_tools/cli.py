"""Command line dispatcher for the P0 model tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gguf import GGUFError, inspect_gguf, verify_gguf
from .sweep import sweep_models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH read-only model asset tools")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", aliases=["gguf_inspect"], help="inspect a GGUF header")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--json", action="store_true", dest="as_json")
    verify = commands.add_parser("verify", aliases=["gguf_verify"], help="verify GGUF structure and sidecar hash")
    verify.add_argument("path", type=Path)
    verify.add_argument("--full-hash", action="store_true")
    verify.add_argument("--json", action="store_true", dest="as_json")
    sweep = commands.add_parser("sweep", aliases=["models_sweep"], help="sweep a model directory")
    sweep.add_argument("root", type=Path, nargs="?", default=Path("models"))
    sweep.add_argument("--full-hash", action="store_true")
    sweep.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _human(command: str, report: dict[str, Any]) -> None:
    if command in {"inspect", "gguf_inspect"}:
        derived = report.get("derived", {})
        print(f"GGUF: {report.get('path')}")
        print(f"version={report.get('version')} tensors={report.get('tensor_count')} metadata={report.get('metadata_count')}")
        print(f"architecture={derived.get('architecture')} name={derived.get('name')} context={derived.get('context_length')}")
        print(f"tensor_types={derived.get('tensor_types')}")
        if report.get("errors"):
            print("errors:")
            for error in report["errors"]:
                print(f"  - {error}")
        return
    if command in {"verify", "gguf_verify"}:
        print(f"{'OK' if report.get('valid') else 'FAIL'}: {report.get('path')}")
        print(f"structure={report.get('structure_valid')} sha256_checked={report.get('sha256_checked')} sidecar={report.get('sidecar')}")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    print(f"{'OK' if report.get('valid') else 'WARN/FAIL'}: {report.get('root')}")
    print(f"gguf={len(report.get('gguf', []))} diffusion={len(report.get('diffusion_assets', []))} junctions={len(report.get('junctions', []))} orphan_candidates={len(report.get('orphan_files', []))}")
    for warning in report.get("warnings", []):
        print(f"  - {warning}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"inspect", "gguf_inspect"}:
            report = inspect_gguf(args.path)
        elif args.command in {"verify", "gguf_verify"}:
            report = verify_gguf(args.path, full_hash=args.full_hash)
        else:
            report = sweep_models(args.root, full_hash=args.full_hash)
    except (OSError, GGUFError, ValueError) as exc:
        report = {"valid": False, "errors": [str(exc)]}
    if getattr(args, "as_json", False):
        # ASCII escaping keeps JSON usable on Windows consoles still using GBK.
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        _human(args.command, report)
    return 0 if report.get("valid", False) else 1
