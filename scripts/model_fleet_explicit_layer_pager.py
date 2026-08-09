#!/usr/bin/env python3
"""MF-MEM-N2 isolated PyTorch explicit layer pager.

This module is a synthetic sidecar harness. It is deliberately not imported by
the production inference service. A fixed 64 MiB tile is staged through exactly
two pinned host/device slots and every stream dependency is explicit. The
harness reports resource rejection instead of silently shrinking the case.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import gc
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Sequence


FIXED_TILE_MIB = 64
FIXED_SLOTS = 2
SAMPLE_STRIDE = 65536


class ResourceRejected(RuntimeError):
    """The fixed N2 allocation cannot be made on the selected device."""


def _digest(values: Sequence[float]) -> str:
    payload = json.dumps([float(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _timeline_metrics(
    copy_intervals: Sequence[tuple[float, float]],
    compute_intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    if not copy_intervals or not compute_intervals:
        return {
            "copy_total_ms": 0.0,
            "compute_total_ms": 0.0,
            "makespan_ms": 0.0,
            "overlap_ms": 0.0,
            "effective_overlap_ratio": 0.0,
            "initial_fill_stall_ms": 0.0,
            "stall_p95_ms": 0.0,
        }
    copy_total = sum(end - start for start, end in copy_intervals)
    compute_total = sum(end - start for start, end in compute_intervals)
    overlap = sum(
        max(0.0, min(copy_end, compute_end) - max(copy_start, compute_start))
        for copy_start, copy_end in copy_intervals
        for compute_start, compute_end in compute_intervals
    )
    ready = 0.0
    stalls: list[float] = []
    for start, end in compute_intervals:
        stalls.append(max(0.0, start - ready))
        ready = end
    steady = sorted(stalls[1:] or stalls)
    p95_index = min(len(steady) - 1, max(0, int(round((len(steady) - 1) * 0.95))))
    timeline_start = min(copy_intervals[0][0], compute_intervals[0][0])
    timeline_end = max(copy_intervals[-1][1], compute_intervals[-1][1])
    capacity = min(copy_total, compute_total)
    return {
        "copy_total_ms": copy_total,
        "compute_total_ms": compute_total,
        "makespan_ms": max(0.0, timeline_end - timeline_start),
        "overlap_ms": overlap,
        "effective_overlap_ratio": min(1.0, overlap / capacity) if capacity > 0 else 0.0,
        "initial_fill_stall_ms": stalls[0],
        "stall_p95_ms": steady[p95_index] if steady else 0.0,
    }


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ResourceRejected("PyTorch is not installed") from exc
    return torch


@dataclass(frozen=True)
class PagerConfig:
    tile_mib: int = FIXED_TILE_MIB
    slots: int = FIXED_SLOTS
    backend: str = "auto"
    device_index: int = 0


class ExplicitLayerPager:
    """Two-slot explicit pager with a CPU fallback for deterministic tests."""

    def __init__(self, config: PagerConfig):
        if config.tile_mib != FIXED_TILE_MIB:
            raise ValueError(f"N2 tile_mib must remain fixed at {FIXED_TILE_MIB}")
        if config.slots != FIXED_SLOTS:
            raise ValueError(f"N2 slots must remain fixed at {FIXED_SLOTS}")
        if config.backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be auto, cpu, or cuda")
        self.torch = _load_torch()
        cuda_available = bool(self.torch.cuda.is_available())
        if config.backend == "cuda" and not cuda_available:
            raise ResourceRejected("CUDA is unavailable")
        use_cuda = config.backend == "cuda" or (config.backend == "auto" and cuda_available)
        if use_cuda and config.device_index >= self.torch.cuda.device_count():
            raise ResourceRejected(f"CUDA device {config.device_index} is unavailable")
        self.config = config
        self.use_cuda = use_cuda
        self.device = self.torch.device(
            f"cuda:{config.device_index}" if use_cuda else "cpu"
        )
        if use_cuda:
            self.torch.cuda.set_device(self.device)
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.tile_bytes = config.tile_mib * 1024 * 1024
        if self.tile_bytes % 4:
            raise ValueError("fixed float32 tile must be divisible by four")
        self.tile_elements = self.tile_bytes // 4
        self.host_slots: list[Any] = []
        self.device_slots: list[Any] = []
        self.copy_stream = None
        self.compute_stream = None
        self._closed = False
        self._allocate()

    def _allocate(self) -> None:
        pinned = self.use_cuda
        try:
            for _ in range(FIXED_SLOTS):
                self.host_slots.append(
                    self.torch.empty(
                        self.tile_elements,
                        dtype=self.torch.float32,
                        device="cpu",
                        pin_memory=pinned,
                    )
                )
                self.device_slots.append(
                    self.torch.empty(
                        self.tile_elements,
                        dtype=self.torch.float32,
                        device=self.device,
                    )
                )
            if self.use_cuda:
                self.copy_stream = self.torch.cuda.Stream(device=self.device)
                self.compute_stream = self.torch.cuda.Stream(device=self.device)
        except (MemoryError, RuntimeError) as exc:
            self.close()
            raise ResourceRejected(
                "cannot allocate fixed 64 MiB x 2 host/device pager slots: " + str(exc)
            ) from exc

    def _validate_layers(self, layers: Sequence[Any]) -> None:
        if not layers:
            raise ValueError("at least one synthetic layer is required")
        for index, layer in enumerate(layers):
            if layer.device.type != "cpu":
                raise ValueError(f"layer {index} must be a CPU source tensor")
            if layer.dtype != self.torch.float32 or layer.numel() != self.tile_elements:
                raise ValueError(
                    f"layer {index} must be float32 with exactly {self.tile_elements} elements"
                )

    def _compute(self, slot: Any, batch_size: int) -> Any:
        sample = slot[::SAMPLE_STRIDE]
        return sample.to(dtype=self.torch.float64).sum() * batch_size

    def run(
        self,
        layers: Sequence[Any],
        *,
        batch_size: int = 1,
        cancellation: Event | None = None,
        cancel_after_layer: int | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("pager is closed")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if cancel_after_layer is not None and cancel_after_layer < 1:
            raise ValueError("cancel_after_layer must be >= 1")
        self._validate_layers(layers)
        cancellation = cancellation or Event()
        copy_intervals: list[list[float]] = []
        compute_intervals: list[list[float]] = []
        outputs: list[Any] = []
        slot_free: list[Any] = [None, None]
        stage_pool: ThreadPoolExecutor | None = None
        started = time.perf_counter()
        completed = 0
        cancelled = False

        if self.use_cuda:
            origin = self.torch.cuda.Event(enable_timing=True)
            origin.record()
            origin.synchronize()
            copy_events: list[tuple[Any, Any]] = []
            compute_events: list[tuple[Any, Any]] = []
        else:
            origin = None
            copy_events = []
            compute_events = []

        try:
            # Staging is an explicit bounded queue. The worker waits on the
            # previous compute event before reusing a host slot, so a CPU write
            # can overlap the current GPU layer without racing the DMA engine.
            if self.use_cuda:
                stage_pool = ThreadPoolExecutor(max_workers=FIXED_SLOTS)

                def stage(index: int) -> None:
                    slot = index % FIXED_SLOTS
                    if slot_free[slot] is not None:
                        slot_free[slot].synchronize()
                    self.host_slots[slot].copy_(layers[index])

                staged: dict[int, Future[None]] = {
                    index: stage_pool.submit(stage, index)
                    for index in range(min(FIXED_SLOTS, len(layers)))
                }
            else:
                stage_pool = None
                staged = {}

            for index, source in enumerate(layers):
                if cancellation.is_set():
                    cancelled = True
                    break
                slot_index = index % FIXED_SLOTS
                if self.use_cuda:
                    staged[index].result()
                    copy_start = self.torch.cuda.Event(enable_timing=True)
                    copy_end = self.torch.cuda.Event(enable_timing=True)
                    with self.torch.cuda.stream(self.copy_stream):
                        copy_start.record(self.copy_stream)
                        self.device_slots[slot_index].copy_(
                            self.host_slots[slot_index], non_blocking=True
                        )
                        copy_end.record(self.copy_stream)
                    compute_start = self.torch.cuda.Event(enable_timing=True)
                    compute_end = self.torch.cuda.Event(enable_timing=True)
                    with self.torch.cuda.stream(self.compute_stream):
                        self.compute_stream.wait_event(copy_end)
                        compute_start.record(self.compute_stream)
                        outputs.append(self._compute(self.device_slots[slot_index], batch_size))
                        compute_end.record(self.compute_stream)
                    copy_events.append((copy_start, copy_end))
                    compute_events.append((compute_start, compute_end))
                    slot_free[slot_index] = compute_end
                    next_index = index + FIXED_SLOTS
                    if next_index < len(layers) and not cancellation.is_set():
                        staged[next_index] = stage_pool.submit(stage, next_index)
                else:
                    self.host_slots[slot_index].copy_(source)
                    self.device_slots[slot_index].copy_(self.host_slots[slot_index])
                    outputs.append(self._compute(self.device_slots[slot_index], batch_size))
                completed += 1
                if cancel_after_layer is not None and completed >= cancel_after_layer:
                    cancellation.set()
        finally:
            if self.use_cuda:
                self.copy_stream.synchronize()
                self.compute_stream.synchronize()
                if stage_pool is not None:
                    stage_pool.shutdown(wait=True)

        values = [float(value.detach().cpu().item()) for value in outputs]
        cancelled = cancelled or (completed < len(layers) and cancellation.is_set())
        output_digest = _digest(values)
        if self.use_cuda and origin is not None:
            copy_intervals = [
                [float(origin.elapsed_time(start)), float(origin.elapsed_time(end))]
                for start, end in copy_events
            ]
            compute_intervals = [
                [float(origin.elapsed_time(start)), float(origin.elapsed_time(end))]
                for start, end in compute_events
            ]
        timeline = _timeline_metrics(
            [tuple(interval) for interval in copy_intervals],
            [tuple(interval) for interval in compute_intervals],
        )
        elapsed_s = time.perf_counter() - started
        peak_allocated = 0
        peak_reserved = 0
        if self.use_cuda:
            peak_allocated = int(self.torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(self.torch.cuda.max_memory_reserved(self.device))
        return {
            "schema_version": 1,
            "ticket": "MF-MEM-N2",
            "status": "cancelled" if cancelled else "completed",
            "completed": int(not cancelled and completed == len(layers)),
            "cancelled": int(cancelled),
            "layers": len(layers),
            "completed_layers": completed,
            "batch_size": batch_size,
            "tile_mib": FIXED_TILE_MIB,
            "slots": FIXED_SLOTS,
            "backend": "cuda" if self.use_cuda else "cpu",
            "output_values": values,
            "output_digest": output_digest,
            "peak_pinned_bytes": self.tile_bytes * FIXED_SLOTS if self.use_cuda else 0,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "copy_intervals_ms": copy_intervals,
            "compute_intervals_ms": compute_intervals,
            "timeline": timeline,
            "elapsed_s": elapsed_s,
        }

    def close(self) -> dict[str, int]:
        if self._closed:
            return {"released_vram_allocated_bytes": 0, "released_vram_reserved_bytes": 0}
        if self.use_cuda:
            self.torch.cuda.synchronize(self.device)
        self.host_slots.clear()
        self.device_slots.clear()
        self.copy_stream = None
        self.compute_stream = None
        gc.collect()
        released_allocated = 0
        released_reserved = 0
        if self.use_cuda:
            self.torch.cuda.empty_cache()
            released_allocated = int(self.torch.cuda.memory_allocated(self.device))
            released_reserved = int(self.torch.cuda.memory_reserved(self.device))
        self._closed = True
        return {
            "released_vram_allocated_bytes": released_allocated,
            "released_vram_reserved_bytes": released_reserved,
        }

    def __enter__(self) -> "ExplicitLayerPager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def make_synthetic_layers(torch: Any, count: int, elements: int) -> list[Any]:
    if count < 1:
        raise ValueError("layers must be >= 1")
    return [
        torch.full((elements,), float(index + 1), dtype=torch.float32, device="cpu")
        for index in range(count)
    ]


def resident_reference(layers: Sequence[Any], batch_size: int, torch: Any) -> dict[str, Any]:
    values = [
        float(layer[::SAMPLE_STRIDE].to(dtype=torch.float64).sum().item() * batch_size)
        for layer in layers
    ]
    return {"values": values, "digest": _digest(values)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MF-MEM-N2 explicit 64 MiB two-slot layer pager")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cancel-after", type=int, default=0)
    parser.add_argument("--result-file", required=True)
    return parser


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    torch = _load_torch()
    layers = make_synthetic_layers(torch, args.layers, FIXED_TILE_MIB * 1024 * 1024 // 4)
    reference_started = time.perf_counter()
    reference = resident_reference(layers, args.batch_size, torch)
    reference_ms = (time.perf_counter() - reference_started) * 1000.0
    pager = ExplicitLayerPager(
        PagerConfig(backend=args.backend, device_index=args.device)
    )
    try:
        result = pager.run(
            layers,
            batch_size=args.batch_size,
            cancel_after_layer=args.cancel_after or None,
        )
        result["resident_reference_ms"] = reference_ms
        result["resident_reference_digest"] = reference["digest"]
        result["output_equal"] = int(
            not result["cancelled"] and result["output_digest"] == reference["digest"]
        )
        result["environment"] = {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        return result
    finally:
        release = pager.close()
        if "result" in locals():
            result.update(release)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = Path(args.result_file).expanduser().resolve()
    try:
        if args.layers < 1 or args.batch_size < 1:
            raise ValueError("layers and batch-size must be >= 1")
        if args.cancel_after < 0 or args.cancel_after > args.layers:
            raise ValueError("cancel-after must be between 0 and layers")
        result = run_case(args)
        _write_json(result_path, result)
        return 0
    except ResourceRejected as exc:
        _write_json(result_path, {
            "schema_version": 1,
            "ticket": "MF-MEM-N2",
            "status": "resource_rejected",
            "completed": 0,
            "cancelled": 0,
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return 4
    except (RuntimeError, ValueError) as exc:
        _write_json(result_path, {
            "schema_version": 1,
            "ticket": "MF-MEM-N2",
            "status": "failed",
            "completed": 0,
            "cancelled": 0,
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
