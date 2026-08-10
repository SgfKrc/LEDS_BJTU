"""Command line dispatcher for the MODEL-TOOLS CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gguf import GGUFError, inspect_gguf, verify_gguf
from .gguf_convert import execute_conversion, plan_conversion
from .llm_smoke_matrix import run_smoke_matrix
from .maintenance import clean_models, model_disk_usage
from .sd15_batch import run_prompt_batch, run_sampler_matrix
from .sweep import sweep_models
from .sync_status import build_inventory, compare_inventories, load_inventory, write_json

ROOT = Path(__file__).resolve().parents[2]


def _ensure_output_outside_roots(output: Path | None, roots: list[Path]) -> None:
    if output is None:
        return
    target = output.expanduser().absolute().resolve(strict=False)
    for root in roots:
        base = root.expanduser().absolute().resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError:
            continue
        raise ValueError("--output must be outside model roots")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH model asset tools")
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
    usage = commands.add_parser("disk-usage", aliases=["model_disk_usage"], help="report model disk usage")
    usage.add_argument("root", type=Path, nargs="?", default=Path("models"))
    usage.add_argument("--json", action="store_true", dest="as_json")
    clean = commands.add_parser("clean", aliases=["models_clean"], help="list or remove stale model files")
    clean.add_argument("root", type=Path, nargs="?", default=Path("models"))
    clean.add_argument("--apply", action="store_true", help="perform deletion after explicit confirmation")
    clean.add_argument("--confirm", default=None, help="must be CLEAN when --apply is used")
    clean.add_argument("--min-age-hours", type=float, default=24.0)
    clean.add_argument("--include-duplicates", action="store_true", help="hash model files and include duplicate candidates")
    clean.add_argument("--include-caches", action="store_true", help="allow stale .cache trees to be removed")
    clean.add_argument("--include-old-backups", action="store_true", help="allow stale models_old_backup trees to be removed")
    clean.add_argument("--json", action="store_true", dest="as_json")
    prompt_batch = commands.add_parser("sd15-prompt-batch", aliases=["sd15_prompt_batch"], help="run a bounded SD15 prompt matrix")
    prompt_batch.add_argument("--asset-id", default="sd15_90s_retrovers_v1")
    prompt_batch.add_argument("--model-path", default="")
    prompt_batch.add_argument("--output-dir", default="")
    prompt_batch.add_argument("--preset", default="sd15_retrovers_space_courier_v1")
    prompt_batch.add_argument("--prompt", action="append", default=[])
    prompt_batch.add_argument("--prompt-file", default="", help="UTF-8 file with one prompt per line")
    prompt_batch.add_argument("--seed", action="append", type=int, default=[])
    prompt_batch.add_argument("--steps", type=int, default=0)
    prompt_batch.add_argument("--json", action="store_true", dest="as_json")
    sampler = commands.add_parser("sd15-sampler-matrix", aliases=["sd15_sampler_matrix"], help="run a bounded SD15 sampler/step matrix")
    sampler.add_argument("--asset-id", default="sd15_90s_retrovers_v1")
    sampler.add_argument("--model-path", default="")
    sampler.add_argument("--output-dir", default="")
    sampler.add_argument("--preset", default="sd15_retrovers_space_courier_v1")
    sampler.add_argument("--prompt", default="")
    sampler.add_argument("--scheduler", action="append", dest="schedulers", default=[])
    sampler.add_argument("--steps", action="append", type=int, dest="steps_list", default=[])
    sampler.add_argument("--seed", type=int, default=19950101)
    sampler.add_argument("--json", action="store_true", dest="as_json")
    sync = commands.add_parser("sync-status", aliases=["models_sync_status"], help="compare read-only model inventories")
    sync_operations = sync.add_subparsers(dest="sync_operation", required=True)
    inventory = sync_operations.add_parser("inventory", help="generate a model inventory")
    inventory.add_argument("root", type=Path, nargs="?", default=Path("models"))
    inventory.add_argument("--full-hash", action="store_true")
    inventory.add_argument("--output", type=Path, default=None)
    inventory.add_argument("--json", action="store_true", dest="as_json")
    compare = sync_operations.add_parser("compare", help="compare roots and/or saved inventories")
    compare.add_argument("--local-root", type=Path, default=None)
    compare.add_argument("--local-inventory", type=Path, default=None)
    compare.add_argument("--peer-root", type=Path, default=None)
    compare.add_argument("--peer-inventory", type=Path, default=None)
    compare.add_argument("--full-hash", action="store_true")
    compare.add_argument("--output", type=Path, default=None)
    compare.add_argument("--json", action="store_true", dest="as_json")
    llm = commands.add_parser("llm-smoke-matrix", aliases=["llm_smoke_matrix"], help="run a bounded LLM smoke matrix")
    llm.add_argument("--model-id", action="append", default=[])
    llm.add_argument("--format", action="append", dest="formats", choices=["gguf", "safetensors"], default=[])
    llm.add_argument("--max-models", type=int, default=32)
    llm.add_argument("--max-new-tokens", type=int, default=32)
    llm.add_argument("--timeout-seconds", type=float, default=180.0)
    llm.add_argument("--quant", choices=["fp16", "int8", "int4"], default="int4")
    llm.add_argument("--allow-cpu", action="store_true")
    llm.add_argument("--require-complete", action="store_true", help="fail when any selected unit is skipped")
    llm.add_argument("--output", type=Path, default=None)
    llm.add_argument("--json", action="store_true", dest="as_json")
    convert = commands.add_parser("gguf-convert", aliases=["gguf_convert"], help="preflight or explicitly execute an HF to GGUF conversion")
    source_group = convert.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--model-id", default=None, help="registered Safetensors model ID")
    source_group.add_argument("--source", type=Path, default=None, help="Safetensors model directory")
    convert.add_argument("--target", type=Path, default=None, help="future or executed GGUF output path")
    convert.add_argument("--outtype", default="Q4_K_M", help="GGUF output type, e.g. Q4_K_M, Q8_0, F16")
    convert.add_argument("--converter", type=Path, default=None, help="explicit convert_hf_to_gguf.py")
    convert.add_argument("--quantizer", type=Path, default=None, help="explicit llama-quantize executable")
    convert.add_argument("--apply", action="store_true", help="execute through same-volume staging")
    convert.add_argument("--confirm", default=None, help="must be CONVERT when --apply is used")
    convert.add_argument("--timeout-seconds", type=float, default=3600.0, help="per-stage execution timeout")
    convert.add_argument("--output", type=Path, default=None, help="write result JSON outside model roots")
    convert.add_argument("--json", action="store_true", dest="as_json")
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
    if command in {"disk-usage", "model_disk_usage"}:
        totals = report.get("totals", {})
        print(f"Usage: {report.get('root')}")
        print(f"files={totals.get('file_count', 0)} logical={totals.get('logical_size_bytes', 0)} allocated={totals.get('allocated_size_bytes', 0)} unique_allocated={totals.get('unique_allocated_size_bytes', 0)}")
        for entry in report.get("entries", []):
            print(f"  {entry['path']}: files={entry['file_count']} logical={entry['logical_size_bytes']} allocated={entry['allocated_size_bytes']}")
        for warning in report.get("warnings", []):
            print(f"  - {warning}")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    if command in {"clean", "models_clean"}:
        mode = "APPLIED" if report.get("applied") else "DRY-RUN"
        print(f"{mode}: {report.get('root')}")
        print(f"candidates={len(report.get('candidates', []))} deleted={len(report.get('deleted', []))}")
        for item in report.get("candidates", []):
            marker = "safe" if item.get("safe_to_delete") else "review"
            print(f"  [{marker}] {item.get('kind')}: {item.get('path')} ({item.get('size_bytes')} bytes)")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    if command in {"sd15-prompt-batch", "sd15_prompt_batch", "sd15-sampler-matrix", "sd15_sampler_matrix"}:
        print(f"{'PASS' if report.get('automatic_gate', {}).get('passed') else 'FAIL'}: {report.get('tool')}")
        print(f"outputs={report.get('automatic_gate', {}).get('outputs', 0)} unique={report.get('automatic_gate', {}).get('unique_images', 0)} contact_sheet={report.get('contact_sheet')}")
        for item in report.get("jobs", []):
            print(f"  {item.get('label')}: scheduler={item.get('scheduler')} steps={item.get('steps')} seed={item.get('seed')} elapsed={item.get('elapsed_seconds'):.3f}s")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    if command in {"sync-status", "models_sync_status"}:
        if report.get("operation") == "inventory":
            print(f"{'OK' if report.get('valid') else 'FAIL'} inventory: mode={report.get('hash_mode')} assets={len(report.get('assets', []))}")
        else:
            print(f"{'SYNCED' if report.get('in_sync') else 'DIFF/FAIL'}: mode={report.get('hash_mode', 'unknown')}")
            print(
                f"matched={report.get('matched_count', 0)} "
                f"missing={len(report.get('missing_on_peer', []))} "
                f"extra={len(report.get('extra_on_peer', []))} "
                f"mismatched={len(report.get('mismatched', []))}"
            )
            for asset_id in report.get("missing_on_peer", []):
                print(f"  [missing_on_peer] {asset_id}")
            for asset_id in report.get("extra_on_peer", []):
                print(f"  [extra_on_peer] {asset_id}")
            for item in report.get("mismatched", []):
                print(f"  [mismatched] {item.get('asset_id')}: {','.join(item.get('changed_fields', []))}")
        for warning in report.get("warnings", []):
            print(f"  - {warning}")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    if command in {"llm-smoke-matrix", "llm_smoke_matrix"}:
        summary = report.get("summary", {})
        state = "PASS" if summary.get("gate_passed") else "FAIL"
        if summary.get("gate_passed") and not summary.get("coverage_complete"):
            state = "PASS (partial coverage)"
        print(f"{state}: {report.get('tool')}")
        print(f"units={summary.get('units_total', 0)} executed={summary.get('units_executed', 0)} passed={summary.get('units_passed', 0)} failed={summary.get('units_failed', 0)} skipped={summary.get('units_skipped', 0)} jobs={summary.get('jobs_passed', 0)}/{summary.get('jobs_failed', 0)}")
        for item in report.get("models", []):
            print(f"  {item.get('model_id')} [{item.get('format')}]: {item.get('status')} jobs={len(item.get('jobs', []))}")
            if item.get("error"):
                print(f"    - {item['error'].get('code')}: {item['error'].get('message')}")
        for error in report.get("errors", []):
            print(f"  - {error}")
        return
    if command in {"gguf-convert", "gguf_convert"}:
        state = "PUBLISHED" if report.get("execution", {}).get("published") else ("READY" if report.get("valid") else "BLOCKED")
        print(f"{state}: {report.get('tool')}")
        print(f"source={report.get('source', {}).get('label')} output_type={report.get('output_type')} target={report.get('target', {}).get('label')}")
        toolchain = report.get("toolchain", {})
        quantizer = toolchain.get("quantizer", {})
        print(f"converter={toolchain.get('converter', {}).get('status')} quantizer={quantizer.get('status')} quantizer_verification={quantizer.get('verification')} architecture_supported={toolchain.get('architecture_supported')}")
        space = report.get("space", {})
        print(f"estimated_output_bytes={space.get('estimated_output_bytes', 0)} required_free_bytes={space.get('required_free_bytes', 0)}")
        print(f"read_only={report.get('read_only')} writes_performed={report.get('writes_performed')}")
        if report.get("execution"):
            print(f"execution_started={report['execution'].get('started')} published={report['execution'].get('published')}")
            for stage in report["execution"].get("stages", []):
                print(f"  [{stage.get('status')}] {stage.get('stage')}")
            if report["execution"].get("error"):
                error = report["execution"]["error"]
                print(f"  - {error.get('code')}: {error.get('message')}")
        for error in report.get("errors", []):
            print(f"  - {error.get('code')}: {error.get('message')}")
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
        elif args.command in {"disk-usage", "model_disk_usage"}:
            report = model_disk_usage(args.root)
        elif args.command in {"clean", "models_clean"}:
            report = clean_models(
                args.root,
                apply=args.apply,
                confirmation=args.confirm,
                min_age_hours=args.min_age_hours,
                include_duplicates=args.include_duplicates,
                include_caches=args.include_caches,
                include_old_backups=args.include_old_backups,
            )
        elif args.command in {"sd15-prompt-batch", "sd15_prompt_batch"}:
            from diffusion import get_asset_spec, get_preset

            prompts = list(args.prompt)
            if args.prompt_file:
                prompts.extend(
                    line.strip()
                    for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
            if not prompts:
                raise ValueError("provide --prompt or --prompt-file")
            spec = get_asset_spec(args.asset_id)
            report = run_prompt_batch(
                asset_id=args.asset_id,
                model_path=args.model_path or spec.target_path(ROOT),
                output_dir=args.output_dir or ROOT / "build" / "model-tools" / "sd15-prompt-batch",
                preset=get_preset(args.preset),
                prompts=prompts,
                seeds=args.seed or [19950101],
                steps=args.steps or get_preset(args.preset).steps,
            )
        elif args.command in {"sd15-sampler-matrix", "sd15_sampler_matrix"}:
            from diffusion import get_asset_spec, get_preset

            preset = get_preset(args.preset)
            spec = get_asset_spec(args.asset_id)
            report = run_sampler_matrix(
                asset_id=args.asset_id,
                model_path=args.model_path or spec.target_path(ROOT),
                output_dir=args.output_dir or ROOT / "build" / "model-tools" / "sd15-sampler-matrix",
                preset=preset,
                prompt=args.prompt or preset.prompt,
                schedulers=args.schedulers or ["EulerDiscreteScheduler", "DDIMScheduler", "DPMSolverMultistepScheduler"],
                steps_list=args.steps_list or [20, 28],
                seed=args.seed,
            )
        elif args.command in {"sync-status", "models_sync_status"}:
            if args.sync_operation == "inventory":
                _ensure_output_outside_roots(args.output, [args.root])
                report = build_inventory(args.root, full_hash=args.full_hash)
            else:
                if (args.local_root is None) == (args.local_inventory is None):
                    raise ValueError("provide exactly one of --local-root or --local-inventory")
                if (args.peer_root is None) == (args.peer_inventory is None):
                    raise ValueError("provide exactly one of --peer-root or --peer-inventory")
                _ensure_output_outside_roots(args.output, [root for root in (args.local_root, args.peer_root) if root is not None])
                if args.local_root is not None:
                    local = build_inventory(args.local_root, full_hash=args.full_hash)
                    local_errors = [] if local.get("valid") else local.get("errors", ["inventory generation failed"])
                else:
                    local, local_errors = load_inventory(args.local_inventory)
                if args.peer_root is not None:
                    peer = build_inventory(args.peer_root, full_hash=args.full_hash)
                    peer_errors = [] if peer.get("valid") else peer.get("errors", ["inventory generation failed"])
                else:
                    peer, peer_errors = load_inventory(args.peer_inventory)
                if local_errors or peer_errors or local is None or peer is None:
                    report = {
                        "schema_version": 1,
                        "tool": "models_sync_status",
                        "operation": "compare",
                        "read_only": True,
                        "valid": False,
                        "in_sync": False,
                        "hash_mode": None,
                        "local_asset_count": 0,
                        "peer_asset_count": 0,
                        "matched_count": 0,
                        "missing_on_peer": [],
                        "extra_on_peer": [],
                        "mismatched": [],
                        "errors": [f"local: {error}" for error in local_errors]
                        + [f"peer: {error}" for error in peer_errors],
                    }
                else:
                    report = compare_inventories(local, peer)
            if args.output is not None:
                write_json(args.output, report)
        elif args.command in {"llm-smoke-matrix", "llm_smoke_matrix"}:
            _ensure_output_outside_roots(args.output, [ROOT / "models"])
            report = run_smoke_matrix(
                model_ids=args.model_id or None,
                formats=args.formats or None,
                max_models=args.max_models,
                max_new_tokens=args.max_new_tokens,
                timeout_seconds=args.timeout_seconds,
                quant=args.quant,
                allow_cpu=args.allow_cpu,
                require_complete=args.require_complete,
            )
            if args.output is not None:
                write_json(args.output, report)
        elif args.command in {"gguf-convert", "gguf_convert"}:
            _ensure_output_outside_roots(args.output, [ROOT / "models"])
            if args.output is not None and args.target is not None:
                if args.output.expanduser().absolute().resolve(strict=False) == args.target.expanduser().absolute().resolve(strict=False):
                    raise ValueError("--output report path must differ from --target model path")
            if args.confirm is not None and not args.apply:
                raise ValueError("--confirm requires --apply")
            operation = execute_conversion if args.apply else plan_conversion
            kwargs = {
                "model_id": args.model_id,
                "source": args.source,
                "target": args.target,
                "outtype": args.outtype,
                "converter": args.converter,
                "quantizer": args.quantizer,
            }
            if args.apply:
                kwargs.update({"timeout_seconds": args.timeout_seconds, "confirmation": args.confirm})
            report = operation(**kwargs)
            if args.output is not None:
                write_json(args.output, report)
        else:
            report = sweep_models(args.root, full_hash=args.full_hash)
    except (OSError, GGUFError, ValueError) as exc:
        if args.command in {"sync-status", "models_sync_status", "llm-smoke-matrix", "llm_smoke_matrix", "gguf-convert", "gguf_convert"}:
            message = str(exc) if isinstance(exc, ValueError) else f"operation failed (errno={getattr(exc, 'errno', None)})"
            tool = "gguf_convert" if args.command in {"gguf-convert", "gguf_convert"} else ("llm_smoke_matrix" if args.command in {"llm-smoke-matrix", "llm_smoke_matrix"} else "models_sync_status")
            report = {
                "tool": tool,
                "operation": ("execute" if getattr(args, "apply", False) else "dry_run") if tool == "gguf_convert" else ("matrix" if tool == "llm_smoke_matrix" else "compare"),
                "request_valid": False if tool == "gguf_convert" else None,
                "valid": False,
                "in_sync": False,
                "errors": [message],
            }
        else:
            report = {"valid": False, "errors": [str(exc)]}
    if getattr(args, "as_json", False):
        # ASCII escaping keeps JSON usable on Windows consoles still using GBK.
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        _human(args.command, report)
    if args.command in {"sync-status", "models_sync_status"}:
        if not report.get("valid", False):
            return 2
        if report.get("operation") == "compare" and not report.get("in_sync", False):
            return 1
        return 0
    if args.command in {"llm-smoke-matrix", "llm_smoke_matrix"}:
        if not report.get("valid", False):
            return 2
        return 0 if report.get("summary", {}).get("gate_passed", False) else 1
    if args.command in {"gguf-convert", "gguf_convert"}:
        if not report.get("request_valid", True):
            return 2
        return 0 if report.get("valid", False) else 1
    return 0 if report.get("valid", False) else 1
