#!/usr/bin/env python
"""Profile QLH's PyTorch layer-range path by operator signature and phase.

The dispatch mode adds metadata scopes only for attribution. Decision-grade
latency comes from separate, uninstrumented wall-clock samples in the report.
This is an opt-in research tool and is not imported by runtime or Edge code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
REPORT_SCHEMA = "qlh.torch_operator_profile.v1"
SCOPE_PREFIX = "qlh.operator."


def _is_tensor(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("shape", "dtype", "device", "layout"))


def _tensor_spec(value: Any) -> dict[str, Any]:
    try:
        shape = [int(dimension) for dimension in value.shape]
    except (TypeError, ValueError):
        shape = [str(dimension) for dimension in value.shape]
    try:
        stride = [int(step) for step in value.stride()]
    except (AttributeError, RuntimeError, TypeError, ValueError):
        stride = None
    return {
        "shape": shape,
        "stride": stride,
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "layout": str(value.layout).removeprefix("torch."),
    }


def _describe_tree(value: Any) -> Any:
    if _is_tensor(value):
        return {"tensor": _tensor_spec(value)}
    if isinstance(value, tuple):
        return {"tuple": [_describe_tree(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_describe_tree(item) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": {
                str(key): _describe_tree(value[key])
                for key in sorted(value, key=lambda item: str(item))
            }
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__qualname__}


def _operator_name(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", None)
    return str(name or func).split(".")[0]


def make_dispatch_capture_mode(phase: str):
    """Create a TorchDispatchMode lazily so importing this module never imports torch."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.profiler import record_function
    except ImportError as exc:  # pragma: no cover - depends on optional research runtime
        raise RuntimeError("This PyTorch build does not provide TorchDispatchMode") from exc

    class DispatchCaptureMode(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.phase = phase
            self.records: dict[str, dict[str, Any]] = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            operator = _operator_name(func)
            inputs = {"args": _describe_tree(args), "kwargs": _describe_tree(kwargs)}
            canonical_inputs = json.dumps(
                inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            )
            identity = json.dumps(
                {"operator": operator, "inputs": canonical_inputs},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            )
            label = SCOPE_PREFIX + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            record = self.records.setdefault(label, {
                "operator": operator,
                "inputs": inputs,
                "calls": 0,
                "outputs": Counter(),
            })
            with record_function(label):
                result = func(*args, **kwargs)
            record["calls"] += 1
            record["outputs"][json.dumps(
                _describe_tree(result), sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            )] += 1
            return result

    return DispatchCaptureMode()


def _event_device_time_us(event: Any) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def collect_operator_rows(profile: Any, capture_mode: Any, *, phase: str,
                          invocations_per_sample: int) -> list[dict[str, Any]]:
    """Join metadata scopes to their direct aten events without inclusive double counting."""
    observed: dict[str, list[Any]] = defaultdict(list)
    for event in profile.events():
        if str(getattr(event, "name", "")).startswith(SCOPE_PREFIX):
            device_type = getattr(event, "device_type", None)
            if getattr(device_type, "name", None) not in (None, "CPU"):
                continue
            children = [
                child for child in (getattr(event, "cpu_children", None) or ())
                if str(getattr(child, "name", "")).startswith("aten::")
            ]
            observed[str(event.name)].append(children)

    rows: list[dict[str, Any]] = []
    for label, record in sorted(capture_mode.records.items()):
        scope_events = observed.get(label, [])
        if len(scope_events) != record["calls"]:
            raise RuntimeError(
                f"Profiler/capture event mismatch for {record['operator']}: "
                f"dispatch={record['calls']} profile={len(scope_events)}"
            )
        op_events = [event for group in scope_events for event in group]
        unmatched_dispatch_calls = sum(not group for group in scope_events)
        rows.append({
            "phase": phase,
            "operator": record["operator"],
            "inputs": record["inputs"],
            "outputs": [
                {"signature": json.loads(signature), "calls": count}
                for signature, count in sorted(record["outputs"].items())
            ],
            "calls": record["calls"],
            "calls_per_sample": round(record["calls"] / max(1, invocations_per_sample), 6),
            "profiler_operator_events": len(op_events),
            "unmatched_dispatch_calls": unmatched_dispatch_calls,
            "profiler_event_names": sorted({str(event.name) for event in op_events}),
            "self_cpu_us_total": round(sum(float(event.self_cpu_time_total) for event in op_events), 3),
            "self_cpu_us_per_sample": round(
                sum(float(event.self_cpu_time_total) for event in op_events)
                / max(1, invocations_per_sample), 3,
            ),
            "self_device_us_total": round(sum(_event_device_time_us(event) for event in op_events), 3),
            "self_device_us_per_sample": round(
                sum(_event_device_time_us(event) for event in op_events)
                / max(1, invocations_per_sample), 3,
            ),
        })
    return rows


def summarize_samples(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in samples)
    if not ordered:
        raise ValueError("at least one measurement is required")
    median = statistics.median(ordered)
    return {
        "samples_ms": [round(value, 6) for value in ordered],
        "count": len(ordered),
        "median_ms": round(median, 6),
        "mean_ms": round(statistics.fmean(ordered), 6),
        "min_ms": round(ordered[0], 6),
        "max_ms": round(ordered[-1], 6),
        "population_stddev_ms": round(statistics.pstdev(ordered), 6),
    }


def _synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", "") == "cuda":
        torch_module.cuda.synchronize(device)


def _cache_from_output(output: dict[str, Any]) -> Any:
    cache = output.get("past_key_values")
    if cache is None:
        cache = output.get("cache")
    if cache is None:
        raise RuntimeError("forward_layers did not return a cache for decode profiling")
    return cache


def _profile_phase(torch_module: Any, manager: Any, device: Any, prompt_ids: Any,
                   *, phase: str, decode_steps: int, warmup: int, warmup_seconds: float,
                   repeats: int,
                   trace_path: Path | None = None) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    def forward_prefill():
        with torch_module.no_grad():
            return manager.forward_layers(input_ids=prompt_ids, use_cache=True)

    def forward_decode_step(cache: Any, token_ids: Any):
        with torch_module.no_grad():
            return manager.forward_layers(
                input_ids=token_ids, past_key_values=cache, use_cache=True,
            )

    token_ids = prompt_ids[:, -1:].contiguous()

    def timed_sample() -> float:
        output = None
        cache = None
        if phase == "decode":
            output = forward_prefill()
            cache = _cache_from_output(output)
            _synchronize(torch_module, device)
        start = time.perf_counter()
        if phase == "prefill":
            output = forward_prefill()
        else:
            for _ in range(decode_steps):
                output = forward_decode_step(cache, token_ids)
                cache = _cache_from_output(output)
        _synchronize(torch_module, device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        divisor = decode_steps if phase == "decode" else 1
        return elapsed_ms / divisor

    def warmup_once() -> None:
        if phase == "prefill":
            forward_prefill()
        else:
            cache = _cache_from_output(forward_prefill())
            for _ in range(decode_steps):
                cache = _cache_from_output(forward_decode_step(cache, token_ids))

    # Warm both actual phase shapes and hold load long enough for laptop GPU
    # clocks to settle, matching the repeated-run discipline of relay profiling.
    warmup_start = time.perf_counter()
    warmup_iterations = 0
    for _ in range(warmup):
        warmup_once()
        warmup_iterations += 1
    while time.perf_counter() - warmup_start < warmup_seconds:
        warmup_once()
        warmup_iterations += 1
    _synchronize(torch_module, device)
    warmup_elapsed_ms = (time.perf_counter() - warmup_start) * 1000.0

    wall_samples = [timed_sample() for _ in range(repeats)]
    output = forward_prefill() if phase == "decode" else None
    cache = _cache_from_output(output) if output is not None else None
    activities = [ProfilerActivity.CPU]
    if getattr(device, "type", "") == "cuda":
        activities.append(ProfilerActivity.CUDA)
    capture = make_dispatch_capture_mode(phase)
    _synchronize(torch_module, device)
    profile_start = time.perf_counter()
    with profile(activities=activities, record_shapes=False, profile_memory=False) as profiler:
        with capture:
            if phase == "prefill":
                forward_prefill()
            else:
                for _ in range(decode_steps):
                    output = forward_decode_step(cache, token_ids)
                    cache = _cache_from_output(output)
    _synchronize(torch_module, device)
    profiled_wall_ms = (time.perf_counter() - profile_start) * 1000.0
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    invocations = decode_steps if phase == "decode" else 1
    rows = collect_operator_rows(
        profiler, capture, phase=phase, invocations_per_sample=invocations,
    )
    baseline = summarize_samples(wall_samples)
    return {
        "phase": phase,
        "measured_unit": "ms_per_forward_call",
        "warmup": {
            "minimum_iterations": warmup,
            "steady_seconds_target": warmup_seconds,
            "iterations_completed": warmup_iterations,
            "elapsed_ms": round(warmup_elapsed_ms, 3),
        },
        "decode_steps_per_sample": decode_steps if phase == "decode" else None,
        "uninstrumented_wall": baseline,
        "profiled_wall_ms": round(profiled_wall_ms, 6),
        "profiler_to_uninstrumented_median_ratio": round(
            profiled_wall_ms / max(1e-9, baseline["median_ms"]
                                   * (decode_steps if phase == "decode" else 1)),
            4,
        ),
        "operator_time_semantics": (
            "sum of self times from direct aten child events under each metadata scope; "
            "exclusive operator-event time, not inclusive scope time"
        ),
        "operators": rows,
        "operator_count": len(rows),
        "dispatch_call_count": sum(row["calls"] for row in rows),
        "unmatched_dispatch_calls": sum(row["unmatched_dispatch_calls"] for row in rows),
        "chrome_trace": str(trace_path) if trace_path is not None else None,
    }


def _git_provenance() -> dict[str, Any]:
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


def _artifact_identity(model_dir: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"name": model_dir.name, "manifest_sha256": None}
    manifest_path = model_dir / "model.manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity["manifest_sha256"] = manifest.get("manifest_sha256")
            identity["model_type"] = manifest.get("model_type")
            identity["source"] = manifest.get("source")
        except (OSError, json.JSONDecodeError):
            identity["manifest_error"] = "unreadable_or_invalid_json"
    config_path = model_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            identity["architecture"] = config.get("model_type")
            identity["num_hidden_layers"] = config.get("num_hidden_layers")
            identity["hidden_size"] = config.get("hidden_size")
        except (OSError, json.JSONDecodeError):
            identity["config_error"] = "unreadable_or_invalid_json"
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layers", type=int, default=12,
                        help="Profile the first N layers through the QLH layer-range engine")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--warmup-seconds", type=float, default=3.0,
                        help="Keep the measured phase shape active for at least this long before sampling")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Permit CPU-only profiling; CPU results are not CUDA evidence")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--chrome-trace-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.layers < 1 or args.prefill_tokens < 1 or args.decode_steps < 1:
        print("FAIL: layers, prefill-tokens and decode-steps must be positive", file=sys.stderr)
        return 2
    if args.warmup < 0 or args.warmup_seconds < 0 or args.repeats < 1:
        print("FAIL: warmup values must be non-negative and repeats must be positive", file=sys.stderr)
        return 2
    if not args.model_dir.is_dir():
        print(f"FAIL: model directory not found: {args.model_dir}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import config as qlh_config
    import model_module

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("FAIL: CUDA is unavailable; pass --allow-cpu only for CPU method validation", file=sys.stderr)
        return 2
    if not hasattr(torch.profiler, "profile"):
        print("FAIL: installed PyTorch has no torch.profiler.profile", file=sys.stderr)
        return 2

    qlh_config.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    model_module.USE_COMPILE = False
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=False, local_files_only=True,
    )
    token_ids = tokenizer(
        args.prompt, return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    if token_ids.numel() == 0:
        print("FAIL: prompt tokenized to an empty sequence", file=sys.stderr)
        return 2
    original_prompt_tokens = int(token_ids.shape[1])
    if token_ids.shape[1] < args.prefill_tokens:
        repeats_needed = (args.prefill_tokens + token_ids.shape[1] - 1) // token_ids.shape[1]
        token_ids = token_ids.repeat(1, repeats_needed)
    token_ids = token_ids[:, :args.prefill_tokens].contiguous()

    manager = model_module.ModelManager()
    manager.load_layer_range(
        0, args.layers, has_embedding=True, has_lm_head=False,
        model_path=str(args.model_dir), quant_type="fp16",
    )
    device = manager.get_device()
    token_ids = token_ids.to(device=device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_properties = torch.cuda.get_device_properties(device)
        device_info = {
            "type": "cuda",
            "name": device_name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(device_properties.total_memory),
        }
    else:
        device_info = {
            "type": str(device.type),
            "name": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        }
    effective_quant_type = manager.quant_type
    parameter_dtype = str(next(manager.model.parameters()).dtype).removeprefix("torch.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = args.json_out or (
        ROOT / "local_docs" / "evidence" / "torch-op-profile"
        / f"torch-operator-profile-{stamp}.json"
    )
    trace_dir = args.chrome_trace_dir
    phases = {}
    try:
        for phase in ("prefill", "decode"):
            trace_path = trace_dir / f"{phase}.trace.json" if trace_dir else None
            phases[phase] = _profile_phase(
                torch, manager, device, token_ids,
                phase=phase,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                warmup_seconds=args.warmup_seconds,
                repeats=args.repeats,
                trace_path=trace_path,
            )
    finally:
        manager.unload_model()

    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticket": "TORCH-OP-PROFILE-01",
        "experiment_kind": "mainrepo_pytorch_layer_range",
        "model": _artifact_identity(args.model_dir),
        "layer_range": [0, args.layers],
        "has_embedding": True,
        "has_lm_head": False,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": getattr(__import__("transformers"), "__version__", None),
            "torch_cuda_runtime": torch.version.cuda,
            "device": device_info,
            "effective_quant_type": effective_quant_type,
            "model_parameter_dtype": parameter_dtype,
            "compile_enabled": False,
            "fixed_decode_input_token_id": int(token_ids[0, -1].item()),
            "prompt_text_recorded": False,
        },
        "workload": {
            "phase_separation": ["prefill", "decode"],
            "prefill_tokens": int(token_ids.shape[1]),
            "prompt_token_count_before_shape_adjustment": original_prompt_tokens,
            "prompt_shape_adjustment": (
                "repeated_prompt_tokens" if original_prompt_tokens < args.prefill_tokens
                else "truncated_prompt_tokens" if original_prompt_tokens > args.prefill_tokens
                else "none"
            ),
            "decode_steps": args.decode_steps,
            "decode_token_policy": "repeat_final_prompt_token; no sampling or LM head",
            "warmup_samples": args.warmup,
            "warmup_seconds_target": args.warmup_seconds,
            "latency_repeats": args.repeats,
            "sampling_or_lm_head_included": False,
        },
        "method": {
            "timing": "uninstrumented perf_counter wall clock with CUDA synchronization when applicable",
            "operator_attribution": (
                "TorchDispatchMode metadata scopes joined to direct aten child profiler events"
            ),
            "scope_overhead_warning": (
                "operator timing is diagnostic; use uninstrumented wall samples for latency comparisons"
            ),
            "no_parameter_or_tensor_values_recorded": True,
        },
        "provenance": _git_provenance(),
        "phases": phases,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[profile] {json_out}")
    print(f"[device] {device_info}")
    for phase, data in phases.items():
        print(
            f"[{phase}] median={data['uninstrumented_wall']['median_ms']:.3f} ms/call | "
            f"operators={data['operator_count']} | unmatched={data['unmatched_dispatch_calls']} | "
            f"profiler ratio={data['profiler_to_uninstrumented_median_ratio']:.2f}x"
        )
    if device.type != "cuda":
        print("[notice] CPU-only method validation; this report is not CUDA cost evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
