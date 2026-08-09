#!/usr/bin/env python3
"""MF-MEM-N1 CUDA memory transport harness.

This is an isolated synthetic transport benchmark. It does not modify or load the
production inference runtime. Fixed tile sizes are intentional: a resource failure
must be reported instead of silently shrinking the workload.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


FIXED_TILE_MIB = (64, 256, 1024)
MODE_SLOTS = {
    "pageable": 1,
    "pinned-single": 1,
    "pinned-double": 2,
    "pinned-triple": 3,
}


class ResourceRejected(RuntimeError):
    """The fixed benchmark case cannot be allocated on this machine."""


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 100.0:
        raise ValueError("quantile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("sample_stats requires at least one value")
    numeric = [float(value) for value in values]
    return {
        "p50_ms": percentile(numeric, 50.0),
        "p95_ms": percentile(numeric, 95.0),
        "max_ms": max(numeric),
        "mean_ms": sum(numeric) / len(numeric),
    }


def timeline_metrics(
    copy_intervals: Sequence[tuple[float, float]],
    compute_intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Summarize two CUDA-stream timelines expressed in milliseconds."""
    if not copy_intervals or not compute_intervals:
        raise ValueError("copy and compute timelines must be non-empty")
    for start, end in (*copy_intervals, *compute_intervals):
        if start < 0.0 or end < start:
            raise ValueError("timeline intervals must be ordered and non-negative")

    copy_total = sum(end - start for start, end in copy_intervals)
    compute_total = sum(end - start for start, end in compute_intervals)
    overlap = 0.0
    for copy_start, copy_end in copy_intervals:
        for compute_start, compute_end in compute_intervals:
            overlap += max(
                0.0,
                min(copy_end, compute_end) - max(copy_start, compute_start),
            )

    stalls: list[float] = []
    compute_ready = 0.0
    for compute_start, compute_end in compute_intervals:
        stalls.append(max(0.0, compute_start - compute_ready))
        compute_ready = compute_end
    steady_stalls = stalls[1:] or stalls
    stall_stats = sample_stats(steady_stalls)
    timeline_start = min(copy_intervals[0][0], compute_intervals[0][0])
    timeline_end = max(copy_intervals[-1][1], compute_intervals[-1][1])
    makespan = max(0.0, timeline_end - timeline_start)
    overlap_capacity = min(copy_total, compute_total)

    return {
        "copy_total_ms": copy_total,
        "compute_total_ms": compute_total,
        "makespan_ms": makespan,
        "overlap_ms": overlap,
        "effective_overlap_ratio": (
            min(1.0, overlap / overlap_capacity) if overlap_capacity > 0.0 else 0.0
        ),
        "hidden_copy_ratio": min(1.0, overlap / copy_total) if copy_total > 0.0 else 0.0,
        "initial_fill_stall_ms": stalls[0],
        "stall_p50_ms": stall_stats["p50_ms"],
        "stall_p95_ms": stall_stats["p95_ms"],
        "stall_max_ms": stall_stats["max_ms"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MF-MEM-N1 fixed CUDA memory transport harness",
    )
    parser.add_argument("--tile-mib", type=int, required=True, choices=FIXED_TILE_MIB)
    parser.add_argument("--mode", required=True, choices=tuple(MODE_SLOTS))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compute-ms", type=float, default=20.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--result-file", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 3:
        raise ValueError("iterations must be >= 3 for percentile metrics")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.compute_ms < 0.0:
        raise ValueError("compute-ms must be >= 0")


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise ResourceRejected("PyTorch is not installed") from exc
    return torch


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gbps(byte_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return 0.0
    return byte_count / (elapsed_ms / 1000.0) / 1_000_000_000.0


def _calibrate_sleep(torch, target_ms: float) -> tuple[int, float]:
    if target_ms <= 0.0:
        return 0, 0.0
    if not hasattr(torch.cuda, "_sleep"):
        raise ResourceRejected("torch.cuda._sleep is unavailable")
    cycles = 1_000_000
    measured = 0.0
    for _ in range(4):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        end.synchronize()
        measured = max(float(start.elapsed_time(end)), 0.001)
        cycles = max(1, int(cycles * target_ms / measured))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    end.synchronize()
    return cycles, float(start.elapsed_time(end))


def _allocate_buffers(torch, tile_bytes: int, slots: int, pinned: bool, device):
    host_slots = []
    device_slots = []
    try:
        for slot in range(slots):
            host = torch.empty(tile_bytes, dtype=torch.uint8, pin_memory=pinned)
            host.fill_(((slot + 1) * 17) % 251)
            host_slots.append(host)
            device_slots.append(
                torch.empty(tile_bytes, dtype=torch.uint8, device=device)
            )
    except (RuntimeError, MemoryError) as exc:
        host_slots.clear()
        device_slots.clear()
        gc.collect()
        torch.cuda.empty_cache()
        kind = "pinned" if pinned else "pageable"
        raise ResourceRejected(
            f"cannot allocate fixed {tile_bytes // (1024 ** 2)} MiB "
            f"{kind} tile with {slots} slot(s): {exc}"
        ) from exc
    return host_slots, device_slots


def _measure_direction(
    torch,
    host_slots,
    device_slots,
    *,
    direction: str,
    iterations: int,
    warmup: int,
    non_blocking: bool,
) -> dict[str, Any]:
    if direction not in {"h2d", "d2h"}:
        raise ValueError(f"unsupported direction: {direction}")
    stream = torch.cuda.Stream()

    def enqueue(index: int, timed: bool):
        start = torch.cuda.Event(enable_timing=True) if timed else None
        end = torch.cuda.Event(enable_timing=True) if timed else None
        slot = index % len(host_slots)
        with torch.cuda.stream(stream):
            if start is not None:
                start.record(stream)
            if direction == "h2d":
                device_slots[slot].copy_(host_slots[slot], non_blocking=non_blocking)
            else:
                host_slots[slot].copy_(device_slots[slot], non_blocking=non_blocking)
            if end is not None:
                end.record(stream)
        return start, end

    for index in range(warmup):
        enqueue(index, timed=False)
    stream.synchronize()

    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for index in range(iterations):
        wall_started = time.perf_counter()
        start, end = enqueue(index, timed=True)
        assert start is not None and end is not None
        end.synchronize()
        wall_samples.append((time.perf_counter() - wall_started) * 1000.0)
        gpu_samples.append(float(start.elapsed_time(end)))

    tile_bytes = int(host_slots[0].numel() * host_slots[0].element_size())
    gpu_total = sum(gpu_samples)
    wall_total = sum(wall_samples)
    return {
        "wire_gbps": _gbps(tile_bytes * iterations, gpu_total),
        "effective_gbps": _gbps(tile_bytes * iterations, wall_total),
        "gpu": sample_stats(gpu_samples),
        "wall": sample_stats(wall_samples),
        "samples": iterations,
    }


def _run_pipeline(
    torch,
    host_slots,
    device_slots,
    *,
    iterations: int,
    non_blocking: bool,
    sleep_cycles: int,
) -> tuple[dict[str, Any], int]:
    torch.cuda.synchronize()

    # The current layer is resident before steady-state scheduling begins. Measure
    # that cold fill separately, then overlap compute(i) with copy(i + 1).
    fill_start = torch.cuda.Event(enable_timing=True)
    fill_end = torch.cuda.Event(enable_timing=True)
    fill_start.record()
    device_slots[0].copy_(host_slots[0], non_blocking=False)
    fill_end.record()
    fill_end.synchronize()
    initial_fill_ms = float(fill_start.elapsed_time(fill_end))

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    origin.synchronize()

    copy_events = []
    compute_events = []
    slot_free = [None for _ in device_slots]

    def enqueue_compute(index: int, wait_for=None):
        slot = index % len(device_slots)
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(compute_stream):
            if wait_for is not None:
                compute_stream.wait_event(wait_for)
            compute_start.record(compute_stream)
            if sleep_cycles > 0:
                torch.cuda._sleep(sleep_cycles)
            compute_end.record(compute_stream)
        slot_free[slot] = compute_end
        compute_events.append((compute_start, compute_end))

    enqueue_compute(0)
    for index in range(1, iterations):
        slot = index % len(device_slots)
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            if slot_free[slot] is not None:
                copy_stream.wait_event(slot_free[slot])
            copy_start.record(copy_stream)
            device_slots[slot].copy_(host_slots[slot], non_blocking=non_blocking)
            copy_end.record(copy_stream)
        copy_events.append((copy_start, copy_end))
        enqueue_compute(index, wait_for=copy_end)

    torch.cuda.synchronize()
    copy_intervals = [
        (float(origin.elapsed_time(start)), float(origin.elapsed_time(end)))
        for start, end in copy_events
    ]
    compute_intervals = [
        (float(origin.elapsed_time(start)), float(origin.elapsed_time(end)))
        for start, end in compute_events
    ]
    metrics = timeline_metrics(copy_intervals, compute_intervals)
    metrics["initial_fill_stall_ms"] = initial_fill_ms
    metrics["copy_intervals_ms"] = [list(interval) for interval in copy_intervals]
    metrics["compute_intervals_ms"] = [list(interval) for interval in compute_intervals]
    tile_bytes = int(host_slots[0].numel() * host_slots[0].element_size())
    metrics["pipeline_effective_gbps"] = _gbps(
        tile_bytes * (iterations - 1),
        metrics["makespan_ms"],
    )
    # Validate copied data after timing so allocator activity cannot perturb streams.
    checksum = sum(
        int(device_slot[::65536].sum(dtype=torch.int64).item())
        for device_slot in device_slots
    )
    samples_per_slot = (tile_bytes + 65535) // 65536
    expected_checksum = sum(
        (((slot + 1) * 17) % 251) * samples_per_slot
        for slot in range(len(device_slots))
    )
    if checksum != expected_checksum:
        raise RuntimeError(
            f"transport checksum mismatch: expected {expected_checksum}, got {checksum}"
        )
    metrics["checksum_expected"] = expected_checksum
    return metrics, checksum


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise ResourceRejected("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ResourceRejected(f"CUDA device {args.device} is unavailable")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_before, total_vram = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))

    tile_bytes = args.tile_mib * 1024 * 1024
    slots = MODE_SLOTS[args.mode]
    pinned = args.mode != "pageable"
    host_slots = []
    device_slots = []
    started = time.perf_counter()
    try:
        host_slots, device_slots = _allocate_buffers(
            torch, tile_bytes, slots, pinned, device,
        )
        sleep_cycles, calibrated_compute_ms = _calibrate_sleep(
            torch, args.compute_ms,
        )
        h2d = _measure_direction(
            torch,
            host_slots,
            device_slots,
            direction="h2d",
            iterations=args.iterations,
            warmup=args.warmup,
            non_blocking=pinned,
        )
        d2h = _measure_direction(
            torch,
            host_slots,
            device_slots,
            direction="d2h",
            iterations=args.iterations,
            warmup=args.warmup,
            non_blocking=pinned,
        )
        if args.warmup:
            _run_pipeline(
                torch,
                host_slots,
                device_slots,
                iterations=args.warmup,
                non_blocking=pinned,
                sleep_cycles=sleep_cycles,
            )
        pipeline, checksum = _run_pipeline(
            torch,
            host_slots,
            device_slots,
            iterations=args.iterations,
            non_blocking=pinned,
            sleep_cycles=sleep_cycles,
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_after, _ = torch.cuda.mem_get_info(device)
        properties = torch.cuda.get_device_properties(device)
        compute_capability = torch.cuda.get_device_capability(device)
        elapsed_s = time.perf_counter() - started
        explicit_pinned_bytes = tile_bytes * slots if pinned else 0
        explicit_pageable_bytes = tile_bytes * slots if not pinned else 0
        return {
            "schema_version": 1,
            "status": "passed",
            "completed": 1,
            "ticket": "MF-MEM-N1",
            "mode": args.mode,
            "tile_mib": args.tile_mib,
            "slots": slots,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "compute_target_ms": args.compute_ms,
            "compute_calibrated_ms": calibrated_compute_ms,
            "h2d_wire_gbps": h2d["wire_gbps"],
            "h2d_effective_gbps": h2d["effective_gbps"],
            "d2h_wire_gbps": d2h["wire_gbps"],
            "d2h_effective_gbps": d2h["effective_gbps"],
            "pipeline_effective_gbps": pipeline["pipeline_effective_gbps"],
            "effective_overlap_ratio": pipeline["effective_overlap_ratio"],
            "hidden_copy_ratio": pipeline["hidden_copy_ratio"],
            "initial_fill_stall_ms": pipeline["initial_fill_stall_ms"],
            "stall_p50_ms": pipeline["stall_p50_ms"],
            "stall_p95_ms": pipeline["stall_p95_ms"],
            "stall_max_ms": pipeline["stall_max_ms"],
            "peak_pinned_bytes": explicit_pinned_bytes,
            "peak_pageable_bytes": explicit_pageable_bytes,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "device_slot_bytes": tile_bytes * slots,
            "elapsed_s": elapsed_s,
            "checksum": checksum,
            "h2d": h2d,
            "d2h": d2h,
            "pipeline": pipeline,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "explicit_pinned_bytes": explicit_pinned_bytes,
                "explicit_pageable_bytes": explicit_pageable_bytes,
                "device_slot_bytes": tile_bytes * slots,
                "free_vram_before_bytes": int(free_before),
                "free_vram_after_bytes": int(free_after),
                "total_vram_bytes": int(total_vram),
            },
            "environment": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "driver": _driver_version(),
                "gpu": properties.name,
                "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
                "device": args.device,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        host_slots.clear()
        device_slots.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result_path = Path(args.result_file).expanduser().resolve()
    try:
        _validate_args(args)
        result = run_case(args)
    except ResourceRejected as exc:
        result = {
            "schema_version": 1,
            "status": "resource_rejected",
            "completed": 0,
            "ticket": "MF-MEM-N1",
            "mode": args.mode,
            "tile_mib": args.tile_mib,
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(result_path, result)
        print(f"MF-MEM-N1 resource rejected: {exc}", file=sys.stderr)
        return 4
    except (RuntimeError, ValueError) as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "completed": 0,
            "ticket": "MF-MEM-N1",
            "mode": args.mode,
            "tile_mib": args.tile_mib,
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(result_path, result)
        print(f"MF-MEM-N1 failed: {exc}", file=sys.stderr)
        return 1
    _write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
