#!/usr/bin/env python
"""Compare exact PyTorch layer-range inference with compressed activations.

This is an opt-in, single-host experiment. It loads the full QLH PyTorch path
as a reference, then runs two QLH layer-range managers with an explicit
serialization boundary. It never changes the production transport.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
DEFAULT_REPORT_DIR = ROOT / "local_docs" / "evidence" / "torch-activation-compression"
MODES = ("none", "f16", "int8_block128", "int4_block128")
REPORT_SCHEMA = "qlh.torch_activation_compression_report.v1"
DEFAULT_PROMPT = (
    "A distributed inference system can divide a neural network into consecutive layer ranges. "
    "Each worker loads only its assigned weights, computes hidden activations, and passes them "
    "to the next worker while keeping its own key and value cache local. The coordinator must "
    "preserve token positions, model identity, data type, and cache ownership across every "
    "request. Compressing an activation can reduce network traffic, but the decoded tensor may "
    "change logits near a decision boundary. Therefore an experiment should compare every "
    "prefill position and every autoregressively generated token against an uncompressed "
    "reference. It should also record the actual tensor bytes, framing overhead, codec time, "
    "link bandwidth, and latency variation. A smaller payload alone does not prove faster "
    "inference: serialization, device copies, scheduling, and network contention can dominate. "
    "The safest deployment policy keeps compression disabled until the exact model, workload, "
    "worker placement, and transport have passed repeatable correctness and performance gates."
)
_LEGACY_SIZE_CACHE: dict[tuple[tuple[int, ...], str], int] = {}


def roundtrip_activation(hidden: Any, mode: str, torch_module: Any) -> dict[str, Any]:
    """Serialize one activation boundary and return a CPU tensor plus metrics."""
    if mode not in MODES:
        raise ValueError(f"unsupported activation mode: {mode}")
    if not isinstance(hidden, torch_module.Tensor) or hidden.ndim < 2:
        raise ValueError("hidden activation must be a rank-2-or-higher tensor")
    if not torch_module.isfinite(hidden).all().item():
        raise ValueError("hidden activation contains non-finite values")

    from src.tcp_comm import deserialize_tensor_fast, serialize_tensor_fast

    shape = tuple(int(dim) for dim in hidden.shape)
    source_tensor_bytes = int(hidden.numel() * hidden.element_size())
    cache_key = (shape, str(hidden.dtype))
    if mode == "none":
        started = time.perf_counter()
        payload = serialize_tensor_fast(hidden)
        restored = deserialize_tensor_fast(payload)
        baseline_bytes = len(payload)
        _LEGACY_SIZE_CACHE[cache_key] = baseline_bytes
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    else:
        import numpy as np

        from src.relay_hidden_quant import decode_hidden, encode_hidden

        baseline_bytes = _LEGACY_SIZE_CACHE.get(cache_key)
        if baseline_bytes is None:
            raise ValueError("run the none-mode baseline before measuring compressed modes")
        started = time.perf_counter()
        source_cpu = hidden.detach().to(device="cpu").contiguous()
        n_embd = shape[-1]
        n_tokens = math.prod(shape[:-1])
        source_f32 = source_cpu.to(dtype=torch_module.float32).numpy()
        payload = encode_hidden(source_f32.reshape(n_tokens, n_embd), mode, n_tokens, n_embd)
        decoded = decode_hidden(payload, mode, n_tokens, n_embd)
        restored_array = np.frombuffer(decoded, dtype="<f4").copy().reshape(shape)
        restored = torch_module.from_numpy(restored_array)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    restored = restored.contiguous()
    if tuple(restored.shape) != shape:
        raise RuntimeError("activation round-trip changed tensor shape")
    if not torch_module.isfinite(restored).all().item():
        raise RuntimeError("activation round-trip produced non-finite values")
    source_cpu = hidden.detach().to(device="cpu").contiguous()
    error = (restored.to(dtype=torch_module.float32) - source_cpu.to(
        dtype=torch_module.float32
    )).abs()
    max_abs_error = float(error.max().item()) if error.numel() else 0.0
    mean_abs_error = float(error.mean().item()) if error.numel() else 0.0
    return {
        "tensor": restored,
        "payload_bytes": len(payload),
        "legacy_serialized_bytes": baseline_bytes,
        "source_tensor_bytes": source_tensor_bytes,
        "payload_ratio_vs_legacy_serialization": len(payload) / max(1, baseline_bytes),
        "payload_ratio_vs_raw_activation": len(payload) / max(1, source_tensor_bytes),
        "codec_roundtrip_ms": elapsed_ms,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "shape": list(shape),
        "source_dtype": str(hidden.dtype).removeprefix("torch."),
    }


def _cache_from_output(output: dict[str, Any]) -> Any:
    cache = output.get("past_key_values")
    if cache is None:
        cache = output.get("cache")
    if cache is None:
        raise RuntimeError("forward_layers did not return a cache")
    return cache


def _argmax_rows(logits: Any) -> list[list[int]]:
    return logits.argmax(dim=-1).detach().to(device="cpu").tolist()


def _generated_token_ids(logits: Any) -> list[int]:
    return [int(value) for value in logits[:, -1, :].argmax(dim=-1).detach().cpu().tolist()]


def _sync(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def _run_reference(manager: Any, prompt_ids: Any, decode_steps: int) -> dict[str, Any]:
    output = manager.forward_layers(input_ids=prompt_ids, use_cache=True)
    cache = _cache_from_output(output)
    prefill_argmax = _argmax_rows(output["logits"])
    generated = _generated_token_ids(output["logits"])
    for _ in range(1, decode_steps):
        token_ids = prompt_ids.new_tensor([[generated[-1]]])
        output = manager.forward_layers(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = _cache_from_output(output)
        generated.extend(_generated_token_ids(output["logits"]))
    return {"prefill_argmax": prefill_argmax, "generated": generated[:decode_steps]}


def _run_split(
    first: Any,
    second: Any,
    prompt_ids: Any,
    decode_steps: int,
    mode: str,
    torch_module: Any,
) -> dict[str, Any]:
    _sync(torch_module, first.get_device())
    started = time.perf_counter()
    first_output = first.forward_layers(
        input_ids=prompt_ids, use_cache=True, apply_lm_head=False,
    )
    first_cache = _cache_from_output(first_output)
    boundary = roundtrip_activation(first_output["hidden_states"], mode, torch_module)
    transfer_rows = [{key: value for key, value in boundary.items() if key != "tensor"}]
    second_output = second.forward_layers(
        hidden_states=boundary["tensor"], use_cache=True, apply_lm_head=True,
    )
    second_cache = _cache_from_output(second_output)
    prefill_argmax = _argmax_rows(second_output["logits"])
    generated = _generated_token_ids(second_output["logits"])

    for _ in range(1, decode_steps):
        token_ids = torch_module.tensor([[generated[-1]]], dtype=torch_module.long)
        first_output = first.forward_layers(
            input_ids=token_ids,
            past_key_values=first_cache,
            use_cache=True,
            apply_lm_head=False,
        )
        first_cache = _cache_from_output(first_output)
        boundary = roundtrip_activation(first_output["hidden_states"], mode, torch_module)
        transfer_rows.append({key: value for key, value in boundary.items() if key != "tensor"})
        second_output = second.forward_layers(
            hidden_states=boundary["tensor"],
            past_key_values=second_cache,
            use_cache=True,
            apply_lm_head=True,
        )
        second_cache = _cache_from_output(second_output)
        generated.extend(_generated_token_ids(second_output["logits"]))

    _sync(torch_module, first.get_device())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prefill_transfer = transfer_rows[0]
    decode_transfers = transfer_rows[1:]
    return {
        "prefill_argmax": prefill_argmax,
        "generated": generated[:decode_steps],
        "wall_ms": elapsed_ms,
        "transfer": {
            "prefill_payload_bytes": prefill_transfer["payload_bytes"],
            "prefill_legacy_serialized_bytes": prefill_transfer["legacy_serialized_bytes"],
            "prefill_source_tensor_bytes": prefill_transfer["source_tensor_bytes"],
            "decode_payload_bytes": sum(row["payload_bytes"] for row in decode_transfers),
            "decode_legacy_serialized_bytes": sum(
                row["legacy_serialized_bytes"] for row in decode_transfers
            ),
            "decode_source_tensor_bytes": sum(
                row["source_tensor_bytes"] for row in decode_transfers
            ),
            "decode_boundary_count": len(decode_transfers),
            "codec_roundtrip_ms": sum(row["codec_roundtrip_ms"] for row in transfer_rows),
            "max_abs_error": max(row["max_abs_error"] for row in transfer_rows),
            "mean_abs_error": statistics.fmean(row["mean_abs_error"] for row in transfer_rows),
            "source_dtype": prefill_transfer["source_dtype"],
            "prefill_shape": prefill_transfer["shape"],
            "decode_shape": decode_transfers[0]["shape"] if decode_transfers else None,
        },
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_prefill = reference["prefill_argmax"]
    candidate_prefill = candidate["prefill_argmax"]
    prefill_total = sum(len(row) for row in reference_prefill)
    prefill_shape_matches = (
        len(reference_prefill) == len(candidate_prefill)
        and all(len(left) == len(right) for left, right in zip(
            reference_prefill, candidate_prefill,
        ))
    )
    first_prefill_mismatch = None
    for batch_index, (left_row, right_row) in enumerate(zip(
        reference_prefill, candidate_prefill,
    )):
        first_position = next(
            (index for index, pair in enumerate(zip(left_row, right_row))
             if pair[0] != pair[1]),
            None,
        )
        if first_position is not None:
            first_prefill_mismatch = {
                "batch_index": batch_index,
                "position": first_position,
            }
            break
    prefill_matches = sum(
        left == right
        for left_row, right_row in zip(reference_prefill, candidate_prefill)
        for left, right in zip(left_row, right_row)
    ) if len(reference_prefill) == len(candidate_prefill) else 0
    reference_tokens = reference["generated"]
    candidate_tokens = candidate["generated"]
    first_mismatch = next(
        (index for index, pair in enumerate(zip(reference_tokens, candidate_tokens))
         if pair[0] != pair[1]),
        None,
    )
    generated_matches = sum(
        left == right for left, right in zip(reference_tokens, candidate_tokens)
    ) if len(reference_tokens) == len(candidate_tokens) else 0
    return {
        "prefill_argmax_matches": prefill_matches,
        "prefill_argmax_total": prefill_total,
        "first_prefill_argmax_mismatch": first_prefill_mismatch,
        "generated_token_matches": generated_matches,
        "generated_token_total": len(reference_tokens),
        "first_generated_mismatch": first_mismatch,
        "exact": (
            prefill_total > 0
            and prefill_shape_matches
            and prefill_matches == prefill_total
            and len(reference_tokens) == len(candidate_tokens)
            and generated_matches == len(reference_tokens)
        ),
    }


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    cv = statistics.pstdev(samples) / median if median else math.inf
    return {
        "samples_ms": [round(sample, 4) for sample in samples],
        "median_ms": round(median, 4),
        "runtime_cv": round(cv, 6),
    }


def _candidate_for_cross_device_link_test(
    *,
    mode: str,
    exact: bool,
    device_type: str,
    repeats: int,
    prefill_tokens: int,
    decode_steps: int,
    candidate_cv: float,
    baseline_cv: float | None,
    max_runtime_cv: float | None,
) -> bool:
    return bool(
        mode != "none"
        and exact
        and device_type == "cuda"
        and repeats >= 3
        and prefill_tokens >= 128
        and decode_steps >= 32
        and max_runtime_cv is not None
        and baseline_cv is not None
        and candidate_cv <= max_runtime_cv
        and baseline_cv <= max_runtime_cv
    )


def _manifest_identity(model_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"name": model_dir.name}
    for filename, key_map in (
        ("model.manifest.json", ("manifest_sha256", "model_type", "source")),
        ("config.json", ("model_type", "num_hidden_layers", "hidden_size")),
    ):
        path = model_dir / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result[f"{filename}_error"] = "unreadable_or_invalid_json"
            continue
        for key in key_map:
            if key in data:
                result[key] = data[key]
    return result


def _git_provenance() -> dict[str, Any]:
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": dirty}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-layer", type=int, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["none", "f16", "int8_block128", "int4_block128"])
    parser.add_argument("--max-runtime-cv", type=float, default=None,
                        help="Explicit stability threshold; omission never admits latency evidence")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU method validation; CPU evidence cannot satisfy CUDA admission")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.prefill_tokens < 1 or args.decode_steps < 1 or args.warmup < 0 or args.repeats < 1
        or not args.model_dir.is_dir()
        or args.max_runtime_cv is not None
        and (not math.isfinite(args.max_runtime_cv) or args.max_runtime_cv < 0 or args.max_runtime_cv > 1)
    ):
        print("FAIL: invalid workload, policy, or model directory", file=sys.stderr)
        return 2
    if args.modes[0] != "none" or len(args.modes) != len(set(args.modes)):
        print("FAIL: modes must be unique and start with none", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import config as qlh_config
    import model_module
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("FAIL: CUDA unavailable; pass --allow-cpu for method validation only", file=sys.stderr)
        return 2
    qlh_config.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    model_module.USE_COMPILE = False

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=False, local_files_only=True,
    )
    prompt_ids = tokenizer(
        args.prompt, return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    if prompt_ids.numel() == 0:
        print("FAIL: prompt tokenized to an empty sequence", file=sys.stderr)
        return 2
    original_prompt_tokens = int(prompt_ids.shape[1])
    if original_prompt_tokens < args.prefill_tokens:
        repeats_needed = (args.prefill_tokens + original_prompt_tokens - 1) // original_prompt_tokens
        prompt_ids = prompt_ids.repeat(1, repeats_needed)
    prompt_ids = prompt_ids[:, :args.prefill_tokens].contiguous()

    model_config = model_module.AutoConfig.from_pretrained(
        str(args.model_dir), trust_remote_code=False, local_files_only=True,
    )
    text_config = getattr(model_config, "text_config", None) or model_config
    total_layers = int(text_config.num_hidden_layers)
    split_layer = args.split_layer or total_layers // 2
    if split_layer <= 0 or split_layer >= total_layers:
        print("FAIL: split-layer must divide the model into two non-empty ranges", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    full = model_module.ModelManager()
    first = second = None
    try:
        full.load_layer_range(
            0, total_layers, has_embedding=True, has_lm_head=True,
            model_path=str(args.model_dir), quant_type="fp16", total_layers=total_layers,
        )
        device = full.get_device()
        if device.type == "cuda":
            device_info = {
                "type": "cuda",
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            }
        else:
            device_info = {
                "type": str(device.type),
                "name": platform.processor() or platform.machine(),
                "logical_cpu_count": __import__("os").cpu_count(),
            }
        parameter_dtype = str(next(full.model.parameters()).dtype).removeprefix("torch.")
        prompt_ids = prompt_ids.to(device=device)
        _sync(torch, device)
        reference = _run_reference(full, prompt_ids, args.decode_steps)
        _sync(torch, device)
        full.unload_model()
        del full
        full = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        first = model_module.ModelManager()
        second = model_module.ModelManager()
        first.load_layer_range(
            0, split_layer, has_embedding=True, has_lm_head=False,
            model_path=str(args.model_dir), quant_type="fp16", total_layers=total_layers,
        )
        second.load_layer_range(
            split_layer, total_layers, has_embedding=False, has_lm_head=True,
            model_path=str(args.model_dir), quant_type="fp16", total_layers=total_layers,
        )
        if str(first.get_device()) != str(second.get_device()):
            raise RuntimeError("experiment requires both layer ranges on one device")

        rows: dict[str, Any] = {}
        all_exact = True
        none_median_ms = None
        none_runtime_cv = None
        for mode in args.modes:
            for _ in range(args.warmup):
                _run_split(first, second, prompt_ids, args.decode_steps, mode, torch)
                _sync(torch, device)
            outcomes = []
            for _ in range(args.repeats):
                _sync(torch, device)
                outcome = _run_split(
                    first, second, prompt_ids, args.decode_steps, mode, torch,
                )
                _sync(torch, device)
                outcomes.append(outcome)
            comparisons = [_comparison(reference, item) for item in outcomes]
            exact = all(item["exact"] for item in comparisons)
            all_exact = all_exact and exact
            transfer = outcomes[0]["transfer"]
            wire_bytes = (
                transfer["prefill_payload_bytes"] + transfer["decode_payload_bytes"]
            )
            legacy_bytes = (
                transfer["prefill_legacy_serialized_bytes"]
                + transfer["decode_legacy_serialized_bytes"]
            )
            raw_activation_bytes = (
                transfer["prefill_source_tensor_bytes"]
                + transfer["decode_source_tensor_bytes"]
            )
            latency = _latency_summary([item["wall_ms"] for item in outcomes])
            if mode == "none":
                none_median_ms = latency["median_ms"]
                none_runtime_cv = latency["runtime_cv"]
            rows[mode] = {
                "exact_vs_full_mainrepo_pytorch": exact,
                "comparisons": comparisons,
                "latency": latency,
                "transfer": {
                    **transfer,
                    "total_payload_bytes": wire_bytes,
                    "total_legacy_serialized_bytes": legacy_bytes,
                    "total_source_tensor_bytes": raw_activation_bytes,
                    "payload_ratio_vs_legacy_serialization": wire_bytes / max(1, legacy_bytes),
                    "payload_ratio_vs_raw_activation": wire_bytes / max(1, raw_activation_bytes),
                    "codec_envelope_bytes_included": False,
                },
                "candidate_for_cross_device_link_test": _candidate_for_cross_device_link_test(
                    mode=mode,
                    exact=exact,
                    device_type=device.type,
                    repeats=args.repeats,
                    prefill_tokens=args.prefill_tokens,
                    decode_steps=args.decode_steps,
                    candidate_cv=latency["runtime_cv"],
                    baseline_cv=none_runtime_cv,
                    max_runtime_cv=args.max_runtime_cv,
                ),
            }
            rows[mode]["wall_ratio_vs_none"] = (
                round(none_median_ms / latency["median_ms"], 6)
                if none_median_ms is not None else None
            )
        none_median_ms = rows.get("none", {}).get("latency", {}).get("median_ms")
        for row in rows.values():
            row["wall_ratio_vs_none"] = (
                round(none_median_ms / row["latency"]["median_ms"], 6)
                if none_median_ms is not None else None
            )
    finally:
        if full is not None:
            full.unload_model()
        if first is not None:
            first.unload_model()
        if second is not None:
            second.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticket": "TORCH-ACT-COMPRESS-01",
        "experiment_kind": "mainrepo_pytorch_full_vs_two_layer_ranges",
        "model": _manifest_identity(args.model_dir),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": device_info,
            "effective_dtype": parameter_dtype,
            "compile_enabled": False,
        },
        "workload": {
            "layer_count": total_layers,
            "split_layer": split_layer,
            "prefill_tokens": int(prompt_ids.shape[1]),
            "decode_steps": args.decode_steps,
            "warmup_runs_per_mode": args.warmup,
            "repeats": args.repeats,
            "prompt_shape_adjustment": (
                "repeated_prompt_tokens" if original_prompt_tokens < args.prefill_tokens
                else "truncated_prompt_tokens" if original_prompt_tokens > args.prefill_tokens
                else "none"
            ),
            "prompt_text_recorded": False,
            "effective_prompt_token_sha256": hashlib.sha256(
                prompt_ids.detach().to(device="cpu").contiguous().numpy().tobytes()
            ).hexdigest(),
            "original_prompt_tokens": original_prompt_tokens,
        },
        "policy": {
            "max_runtime_cv": args.max_runtime_cv,
            "token_gate": "all prefill-position argmax and generated token IDs must match the full QLH PyTorch reference",
        },
        "results": rows,
        "admission": {
            "software_exactness_passed": all_exact,
            "candidate_for_cross_device_link_test": any(
                row["candidate_for_cross_device_link_test"] for row in rows.values()
            ),
            "cross_device_hardware_admitted": False,
            "production_runtime_enabled": False,
            "reason": (
                "single-host experiment does not measure cross-device transfer or link contention; "
                "codec payload excludes envelope metadata and production tensor transport remains unchanged"
            ),
        },
        "measurement_scope": {
            "none_mode_bytes": "actual QLH serialize_tensor_fast payload bytes",
            "compressed_mode_bytes": "activation codec payload only; excludes outer transport framing and codec mode metadata",
            "latency": "single-host full-model-reference and two-layer-range wall time; not a network throughput claim",
        },
        "provenance": _git_provenance(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = args.json_out or DEFAULT_REPORT_DIR / f"torch-activation-compress-{stamp}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[report] {json_out}")
    print(f"[device] {device_info}")
    for mode, row in rows.items():
        transfer = row["transfer"]
        comparison = row["comparisons"][0]
        print(
            f"[{mode}] exact={row['exact_vs_full_mainrepo_pytorch']} "
            f"tokens={comparison['generated_token_matches']}/{comparison['generated_token_total']} "
            f"payload={transfer['total_payload_bytes']}/{transfer['total_legacy_serialized_bytes']} "
            f"({transfer['payload_ratio_vs_legacy_serialization']:.3f}x legacy; "
            f"{transfer['payload_ratio_vs_raw_activation']:.3f}x raw activation)"
        )
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
