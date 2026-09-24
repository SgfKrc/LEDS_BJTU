#!/usr/bin/env python
"""Measure QLH PyTorch layer costs and same-host CPU/CUDA split inference.

This is an opt-in research probe. It does not enable runtime placement. Reports
contain model/input identity, uninstrumented phase timings, actual cache layout,
and (on CUDA hosts) both CPU<->CUDA 12/12 split directions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
DEFAULT_REPORT_DIR = ROOT / "local_docs" / "evidence" / "torch-hardware-admit"
REPORT_SCHEMA = "qlh.torch_hardware_admit.v1"
DEFAULT_PROMPT = (
    "A distributed inference system assigns consecutive transformer layers to "
    "independent devices. Each stage must preserve token positions, activation "
    "dtype, model identity, and its own attention cache. The coordinator measures "
    "prefill and autoregressive decode separately, accounts for transfer latency, "
    "and falls back only by restarting the complete request. Hardware placement "
    "must be based on repeatable measurements rather than average per-layer cost."
)


def _parse_nvidia_smi_csv(output: str) -> dict[str, Any]:
    """Parse one ``nvidia-smi --format=csv`` row without trusting locale text."""
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if not rows:
        raise ValueError("nvidia-smi returned no rows")
    values = [item.strip() for item in rows[-1]]
    if len(values) != 11:
        raise ValueError(f"nvidia-smi returned {len(values)} columns, expected 11")
    fields = (
        "index", "name", "clock_sm_mhz", "clock_max_sm_mhz", "temperature_c",
        "power_w", "power_limit_w", "utilization_gpu_pct", "memory_used_mib",
        "event_reasons_active", "throttle_reasons_active",
    )
    result: dict[str, Any] = {"source": "nvidia-smi"}
    for field, value in zip(fields, values):
        if field in {"name", "event_reasons_active", "throttle_reasons_active"}:
            result[field] = value
        else:
            result[field] = None if value in {"[N/A]", "N/A", ""} else float(value)
    return result


def _parse_typeperf_csv(output: str) -> dict[str, float]:
    """Parse the last PDH CSV sample, retaining numeric counter values only."""
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    header_index = next(
        (index for index, row in enumerate(rows) if any(cell.startswith("\\") for cell in row)),
        None,
    )
    if header_index is None or header_index + 1 >= len(rows):
        raise ValueError("typeperf returned no counter row")
    headers = rows[header_index]
    result: dict[str, float] = {}
    for values in rows[header_index + 1:]:
        if len(headers) != len(values):
            continue
        candidate: dict[str, float] = {}
        for header, value in zip(headers, values):
            if not header.startswith("\\"):
                continue
            try:
                candidate[header] = float(value)
            except ValueError:
                continue
        if candidate:
            result = candidate
            break
    if not result:
        raise ValueError("typeperf returned no numeric counters")
    return result


def _collect_nvidia_smi() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu,"
            "power.draw,power.limit,utilization.gpu,memory.used,clocks_event_reasons.active,"
            "clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=5, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or "nvidia-smi failed").strip()[:300])
    return _parse_nvidia_smi_csv(completed.stdout)


def _collect_typeperf() -> dict[str, float]:
    counters = [
        r"\Processor(_Total)\% Processor Time",
        r"\Processor Information(_Total)\% Processor Utility",
        r"\Processor Information(_Total)\Processor Frequency",
        r"\Thermal Zone Information(*)\Temperature",
    ]
    completed = subprocess.run(
        ["typeperf", *counters, "-sc", "1", "-si", "1", "-y"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=8, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or "typeperf failed").strip()[:300])
    return _parse_typeperf_csv(completed.stdout)


class _HardwareTelemetrySampler:
    """Persistent sidecar samplers; they never participate in the timing path."""

    def __init__(self, source: str, interval_s: float = 1.0) -> None:
        self.source = source
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.stream_samples: dict[str, list[dict[str, Any]]] = {"gpu": [], "system": []}
        self.errors: list[dict[str, str]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._processes: list[subprocess.Popen[str]] = []

    def _append(self, source: str, payload: dict[str, Any]) -> None:
        sample = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), source: payload}
        with self._lock:
            self.samples.append(sample)
            self.stream_samples[source].append(sample)

    def _record_error(self, source: str, error: BaseException) -> None:
        with self._lock:
            self.errors.append({"source": source, "error": str(error)[:300]})

    def _spawn(self, command: list[str]) -> subprocess.Popen[str]:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.DEVNULL,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(command, **kwargs)
        self._processes.append(process)
        return process

    def _read_gpu(self) -> None:
        command = [
            "nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.max.sm,temperature.gpu,"
            "power.draw,power.limit,utilization.gpu,memory.used,clocks_event_reasons.active,"
            "clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits", "-l", str(max(1, int(self.interval_s))),
        ]
        try:
            process = self._spawn(command)
            assert process.stdout is not None
            for line in process.stdout:
                if self._stop.is_set():
                    break
                if not line.strip():
                    continue
                try:
                    self._append("gpu", _parse_nvidia_smi_csv(line))
                except ValueError as exc:
                    self._record_error("gpu", exc)
        except (OSError, subprocess.SubprocessError) as exc:
            self._record_error("gpu", exc)

    def _read_system(self) -> None:
        if os.name != "nt":
            self._record_error("system", RuntimeError("typeperf is only available on Windows"))
            return
        counters = [
            r"\Processor(_Total)\% Processor Time",
            r"\Processor Information(_Total)\% Processor Utility",
            r"\Processor Information(_Total)\Processor Frequency",
            r"\Thermal Zone Information(*)\Temperature",
        ]
        try:
            process = self._spawn(["typeperf", *counters, "-si", str(max(1, int(self.interval_s))), "-y"])
            assert process.stdout is not None
            headers: list[str] | None = None
            for line in process.stdout:
                if self._stop.is_set():
                    break
                rows = list(csv.reader([line.rstrip("\r\n")]))
                if not rows:
                    continue
                row = rows[0]
                if any(cell.startswith("\\") for cell in row):
                    headers = row
                    continue
                if headers is None or len(headers) != len(row):
                    continue
                values: dict[str, float] = {}
                for header, value in zip(headers, row):
                    if not header.startswith("\\"):
                        continue
                    try:
                        values[header] = float(value)
                    except ValueError:
                        continue
                if values:
                    self._append("system", values)
        except (OSError, subprocess.SubprocessError) as exc:
            self._record_error("system", exc)

    def start(self) -> None:
        if self.source == "none":
            return
        targets = [("gpu", self._read_gpu)] if self.source == "cuda" else [
            ("gpu", self._read_gpu), ("system", self._read_system),
        ]
        for name, target in targets:
            thread = threading.Thread(target=target, name=f"torch-hw-telemetry-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        for process in self._processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for thread in self._threads:
            thread.join(timeout=10)
        return {
            "source": self.source,
            "interval_s": self.interval_s,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "streams": self.stream_samples,
            "errors": self.errors,
            "sampling_complete": all(not thread.is_alive() for thread in self._threads),
        }


def summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms or any(not math.isfinite(value) or value <= 0 for value in samples_ms):
        raise ValueError("timing samples must be finite positive values")
    values = [float(value) for value in samples_ms]
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "population_stddev_ms": deviation,
        "cv": deviation / mean if mean else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_nonnegative_samples(samples_ms: list[float]) -> dict[str, Any]:
    """Summarize coarse diagnostic clocks that may quantize short work to zero."""
    if not samples_ms or any(not math.isfinite(value) or value < 0 for value in samples_ms):
        raise ValueError("diagnostic samples must be finite non-negative values")
    values = [float(value) for value in samples_ms]
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": mean,
        "population_stddev_ms": deviation,
        "cv": deviation / mean if mean else None,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _parse_cpu_affinity(value: str) -> list[int]:
    """Parse a stable logical-CPU list used for reproducible local probes."""
    try:
        cpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cpu affinity must be comma-separated integers") from exc
    if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("cpu affinity must contain unique non-negative CPUs")
    return cpus


def _apply_cpu_affinity(cpus: list[int] | None) -> dict[str, Any]:
    """Apply process affinity before Torch/model initialization and attest the result."""
    if cpus is None:
        return {"requested_logical_cpus": None, "mask": None, "applied": False, "method": "not_requested"}
    logical_count = os.cpu_count() or 1
    if any(cpu >= logical_count for cpu in cpus):
        raise ValueError(f"cpu affinity contains CPU outside logical range 0..{logical_count - 1}")
    mask = sum(1 << cpu for cpu in cpus)
    method = "windows_SetProcessAffinityMask"
    applied = False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        applied = bool(kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.c_size_t(mask),
        ))
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        applied = True
        method = "sched_setaffinity"
    else:
        method = "unsupported"
    if not applied:
        raise RuntimeError(f"failed to apply CPU affinity mask {mask:#x}")
    return {
        "requested_logical_cpus": cpus,
        "mask": mask,
        "applied": True,
        "method": method,
    }


def _phase_measurement_order(repeats: int) -> list[str]:
    """Balance phase order while keeping each prefill/decode pair adjacent."""
    order: list[str] = []
    for index in range(repeats):
        order.extend(("prefill", "decode") if index % 2 == 0 else ("decode", "prefill"))
    return order


def _fingerprint(value: Any) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def summarize_cache(cache: Any, torch_module: Any) -> dict[str, Any]:
    """Describe materialized cache tensors without retaining or serializing values."""
    rows: list[dict[str, Any]] = []
    visited: set[int] = set()
    tensor_type = torch_module.Tensor

    def visit(value: Any, path: str, depth: int = 0) -> None:
        if isinstance(value, tensor_type):
            rows.append({
                "path": path,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype).removeprefix("torch."),
                "device": str(value.device),
                "layout": str(value.layout).removeprefix("torch."),
                "bytes": int(value.numel() * value.element_size()),
            })
            return
        if value is None or depth >= 6:
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}", depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
        elif hasattr(value, "__dict__"):
            for key, item in sorted(vars(value).items()):
                if not key.startswith("_"):
                    visit(item, f"{path}.{key}", depth + 1)

    visit(cache, type(cache).__name__)
    rows.sort(key=lambda row: row["path"])
    structural_identity = [
        {key: row[key] for key in ("path", "shape", "dtype", "layout")}
        for row in rows
    ]
    placement_identity = [
        {key: row[key] for key in ("path", "shape", "dtype", "device", "layout")}
        for row in rows
    ]
    return {
        "cache_type": type(cache).__name__,
        "tensor_count": len(rows),
        "tensor_bytes": sum(row["bytes"] for row in rows),
        "structural_layout_fingerprint": _fingerprint(structural_identity),
        "placement_fingerprint": _fingerprint(placement_identity),
        "tensors": rows,
    }


def compare_reports(cpu_report: dict[str, Any], cuda_report: dict[str, Any], *, max_cv: float) -> dict[str, Any]:
    """Fail-closed comparison of a matched CPU/CUDA pair of probe reports."""
    reasons: list[str] = []
    if cpu_report.get("schema_version") != REPORT_SCHEMA or cuda_report.get("schema_version") != REPORT_SCHEMA:
        reasons.append("report_schema_mismatch")
    if cpu_report.get("model_identity") != cuda_report.get("model_identity"):
        reasons.append("model_artifact_mismatch")
    if cpu_report.get("tokenizer_sha256") != cuda_report.get("tokenizer_sha256"):
        reasons.append("tokenizer_mismatch")
    cpu_runtime = cpu_report.get("runtime", {})
    cuda_runtime = cuda_report.get("runtime", {})
    if cpu_runtime.get("device", {}).get("type") != "cpu":
        reasons.append("cpu_report_device_mismatch")
    if cuda_runtime.get("device", {}).get("type") != "cuda":
        reasons.append("cuda_report_device_mismatch")
    if cpu_runtime.get("host_identity") != cuda_runtime.get("host_identity"):
        reasons.append("host_identity_mismatch")
    if cpu_runtime.get("threads") != cuda_runtime.get("threads"):
        reasons.append("thread_count_mismatch")
    cpu_affinity = cpu_runtime.get("cpu_affinity")
    cuda_affinity = cuda_runtime.get("cpu_affinity")
    if cpu_affinity is not None and cuda_affinity is not None and cpu_affinity != cuda_affinity:
        reasons.append("cpu_affinity_mismatch")
    if (
        cpu_runtime.get("interop_threads") is not None
        and cuda_runtime.get("interop_threads") is not None
        and cpu_runtime.get("interop_threads") != cuda_runtime.get("interop_threads")
    ):
        reasons.append("interop_thread_count_mismatch")
    if cpu_runtime.get("python") != cuda_runtime.get("python"):
        reasons.append("python_runtime_mismatch")
    if cpu_runtime.get("transformers") != cuda_runtime.get("transformers"):
        reasons.append("transformers_runtime_mismatch")
    cpu_torch_base = str(cpu_runtime.get("torch", "")).split("+", 1)[0]
    cuda_torch_base = str(cuda_runtime.get("torch", "")).split("+", 1)[0]
    if not cpu_torch_base or cpu_torch_base != cuda_torch_base:
        reasons.append("torch_runtime_version_mismatch")
    if cpu_report.get("precision_mode") != "fp32" or cuda_report.get("precision_mode") != "fp32":
        reasons.append("matched_fp32_control_missing")
    cpu_workloads = cpu_report.get("workloads", {})
    cuda_workloads = cuda_report.get("workloads", {})
    if set(cpu_workloads) != set(cuda_workloads):
        reasons.append("workload_set_mismatch")
    else:
        for key in sorted(cpu_workloads):
            if (
                cpu_workloads[key].get("prefill_tokens") != cuda_workloads[key].get("prefill_tokens")
                or cpu_workloads[key].get("decode_steps") != cuda_workloads[key].get("decode_steps")
            ):
                reasons.append(f"workload_shape_mismatch:{key}")
            if cpu_workloads[key].get("prompt_token_sha256") != cuda_workloads[key].get("prompt_token_sha256"):
                reasons.append(f"prompt_identity_mismatch:{key}")
            if cpu_workloads[key].get("reference", {}).get("generated_token_ids") != cuda_workloads[key].get("reference", {}).get("generated_token_ids"):
                reasons.append(f"full_model_generated_tokens_differ:{key}")

    cpu_range_keys = set(cpu_report.get("range_profiles", {}))
    cuda_range_keys = set(cuda_report.get("range_profiles", {}))
    cost_checks: list[dict[str, Any]] = []
    for report_name, report in (("cpu", cpu_report), ("cuda", cuda_report)):
        for key, row in report.get("range_profiles", {}).items():
            for phase in ("prefill", "decode"):
                samples = row.get(phase, {}).get("uninstrumented_wall", {}).get("samples_ms", [])
                cv = row.get(phase, {}).get("uninstrumented_wall", {}).get("cv")
                passed = len(samples) >= 3 and isinstance(cv, (int, float)) and cv <= max_cv
                cost_checks.append({"report": report_name, "workload": key, "phase": phase, "samples": len(samples), "cv": cv, "stable": passed})
                if not passed:
                    reasons.append(f"unstable_or_insufficient_samples:{report_name}:{key}:{phase}")
    kv_layout_checks: list[dict[str, Any]] = []
    for key in sorted(cpu_range_keys):
        cpu_row = cpu_report.get("range_profiles", {}).get(key, {})
        cuda_row = cuda_report.get("range_profiles", {}).get(key, {})
        for phase in ("prefill", "decode"):
            cpu_kv = cpu_row.get("actual_kv_cache", {}).get(phase, {})
            cuda_kv = cuda_row.get("actual_kv_cache", {}).get(phase, {})
            cpu_layout = cpu_kv.get("structural_layout_fingerprint")
            cuda_layout = cuda_kv.get("structural_layout_fingerprint")
            matched = bool(cpu_layout) and cpu_layout == cuda_layout
            kv_layout_checks.append({"workload": key, "phase": phase, "matched": matched})
            if not matched:
                reasons.append(f"kv_structural_layout_mismatch:{key}:{phase}")
    if not cpu_range_keys or cpu_range_keys != cuda_range_keys:
        reasons.append("range_matrix_mismatch")
    for workload_key in cpu_workloads:
        observed_ends = {
            int(key.split(":", 1)[1])
            for key in cpu_range_keys
            if key.startswith(f"{workload_key}:") and ":" in key
        }
        if observed_ends != {4, 12, 24}:
            reasons.append(f"range_matrix_incomplete:{workload_key}")

    pair_rows: list[dict[str, Any]] = []
    for key, row in cuda_report.get("heterogeneous_pairs", {}).items():
        directions = row.get("directions", {})
        for direction in ("cpu_to_cuda", "cuda_to_cpu"):
            if direction not in directions:
                reasons.append(f"heterogeneous_direction_missing:{key}:{direction}")
                continue
            values = directions.get(direction, {})
            exact = values.get("prefill_argmax_exact") is True and values.get("generated_tokens_exact") is True
            stable = all(
                len(values.get(phase, {}).get("samples_ms", [])) >= 3
                and values.get(phase, {}).get("cv", math.inf) <= max_cv
                for phase in ("prefill", "decode")
            )
            pair_rows.append({"workload": key, "direction": direction, "exact": exact, "stable": stable})
            if not exact:
                reasons.append(f"heterogeneous_correctness_failed:{key}:{direction}")
            if not stable:
                reasons.append(f"heterogeneous_timing_unstable:{key}:{direction}")
    for key, row in cuda_report.get("heterogeneous_pairs", {}).items():
        directions = row.get("directions", {})
        cpu_to_cuda = directions.get("cpu_to_cuda", {}).get("kv_cache_after_decode", {})
        cuda_to_cpu = directions.get("cuda_to_cpu", {}).get("kv_cache_after_decode", {})
        for stage in ("first_stage_cache", "second_stage_cache"):
            left = cpu_to_cuda.get(stage, {}).get("structural_layout_fingerprint")
            right = cuda_to_cpu.get(stage, {}).get("structural_layout_fingerprint")
            if not left or left != right:
                reasons.append(f"heterogeneous_kv_layout_mismatch:{key}:{stage}")
    expected_directions = {
        f"{key}:{direction}"
        for key in cpu_workloads
        for direction in ("cpu_to_cuda", "cuda_to_cpu")
    }
    observed_directions = {f"{item['workload']}:{item['direction']}" for item in pair_rows}
    if observed_directions != expected_directions:
        reasons.append("heterogeneous_direction_matrix_incomplete")

    pair_identity_ok = not any(reason in reasons for reason in (
        "report_schema_mismatch", "model_artifact_mismatch", "tokenizer_mismatch",
        "cpu_report_device_mismatch", "cuda_report_device_mismatch",
        "host_identity_mismatch", "thread_count_mismatch", "cpu_affinity_mismatch",
        "interop_thread_count_mismatch",
        "python_runtime_mismatch", "transformers_runtime_mismatch",
        "torch_runtime_version_mismatch",
        "matched_fp32_control_missing", "workload_set_mismatch",
    )) and not any(reason.startswith((
        "prompt_identity_mismatch:", "workload_shape_mismatch:", "full_model_generated_tokens_differ:",
    )) for reason in reasons)
    split_checks_ok = bool(pair_rows) and all(item["exact"] and item["stable"] for item in pair_rows)
    expected_pair_count = len(cpu_workloads) * 2
    split_checks_ok = split_checks_ok and len(pair_rows) == expected_pair_count
    phase_matrix_ok = (
        cpu_range_keys == cuda_range_keys
        and bool(cpu_range_keys)
        and not any(reason.startswith((
            "range_matrix_incomplete:", "unstable_or_insufficient_samples:",
            "kv_structural_layout_mismatch:",
        )) or reason == "range_matrix_mismatch" for reason in reasons)
    )

    return {
        "schema_version": "qlh.torch_hardware_admit_comparison.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_runtime_cv": max_cv,
        "matched_fp32_cpu_cuda_pair": pair_identity_ok,
        "phase_cost_matrix_admitted": pair_identity_ok and phase_matrix_ok,
        "same_host_cpu_cuda_split_admitted": (
            pair_identity_ok and split_checks_ok
            and phase_matrix_ok
            and not any(reason.startswith("heterogeneous_kv_layout_mismatch:") for reason in reasons)
            and "heterogeneous_direction_matrix_incomplete" not in reasons
        ),
        "cross_host_admitted": False,
        "production_runtime_enabled": False,
        "cross_host_reason": "no_production_equivalent_cross_host_inference_measurement",
        "reasons": sorted(set(reasons)),
        "cost_checks": cost_checks,
        "heterogeneous_pair_checks": pair_rows,
        "kv_layout_checks": kv_layout_checks,
        "source_reports": {
            "cpu_commit": cpu_report.get("provenance", {}).get("commit"),
            "cuda_commit": cuda_report.get("provenance", {}).get("commit"),
            "cpu_report_dirty": cpu_report.get("provenance", {}).get("working_tree_dirty"),
            "cuda_report_dirty": cuda_report.get("provenance", {}).get("working_tree_dirty"),
        },
        "runtime_builds": {
            "cpu": cpu_runtime.get("torch"),
            "cuda": cuda_runtime.get("torch"),
            "same_torch_version_base": cpu_torch_base == cuda_torch_base,
            "cuda_runtime": cuda_runtime.get("torch_cuda_runtime"),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": dirty}


def _artifact_identity(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "model.manifest.json"
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if not weight_path.is_file():
        raise FileNotFoundError(f"expected a single local safetensors file: {weight_path}")
    return {
        "name": model_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "weights_sha256": _sha256_file(weight_path),
        "weights_bytes": weight_path.stat().st_size,
        "model_type": config.get("model_type"),
        "num_hidden_layers": int(config.get("num_hidden_layers", 0)),
        "hidden_size": int(config.get("hidden_size", 0)),
    }


def _sync(torch_module: Any, *devices: Any) -> None:
    for device in devices:
        if getattr(device, "type", None) == "cuda":
            torch_module.cuda.synchronize(device)


def _cache_from_output(output: dict[str, Any]) -> Any:
    cache = output.get("cache")
    if cache is None:
        cache = output.get("past_key_values")
    if cache is None:
        raise RuntimeError("QLH forward_layers returned no cache object")
    return cache


def _argmax(logits: Any) -> list[list[int]]:
    return logits.argmax(dim=-1).detach().cpu().tolist()


def _generated(logits: Any) -> int:
    return int(logits[:, -1, :].argmax(dim=-1).detach().cpu().item())


def _run_reference(manager: Any, prompt_ids: Any, decode_steps: int, torch_module: Any) -> dict[str, Any]:
    with torch_module.no_grad():
        output = manager.forward_layers(input_ids=prompt_ids, use_cache=True)
        prefill_argmax = _argmax(output["logits"])
        cache = _cache_from_output(output)
        generated = [_generated(output["logits"])]
        for _ in range(decode_steps):
            token = prompt_ids.new_tensor([[generated[-1]]])
            output = manager.forward_layers(input_ids=token, past_key_values=cache, use_cache=True)
            cache = _cache_from_output(output)
            generated.append(_generated(output["logits"]))
    return {"prefill_argmax": prefill_argmax, "generated_token_ids": generated}


def _prepare_manager(manager_module: Any, model_dir: Path, start: int, end: int, total: int,
                     *, dtype_mode: str, device: Any = None) -> Any:
    manager = manager_module.ModelManager()
    manager.load_layer_range(
        start, end, has_embedding=(start == 0), has_lm_head=(end == total),
        model_path=str(model_dir), quant_type="fp16", total_layers=total,
    )
    if device is not None:
        manager.model.to(device=device)
    if dtype_mode == "fp32" or (
        dtype_mode == "deployment_default" and manager.get_device().type == "cpu"
    ):
        manager.model.float()
    manager.model.eval()
    return manager


def _run_segment_prefill(manager: Any, token_ids: Any, *, full_range: bool) -> dict[str, Any]:
    return manager.forward_layers(
        input_ids=token_ids, use_cache=True, apply_lm_head=full_range,
    )


def _measure_segment(torch_module: Any, manager: Any, device: Any, token_ids: Any,
                     reference: dict[str, Any], *, layer_end: int, total_layers: int,
                     warmup: int, repeats: int, decode_steps: int) -> tuple[dict[str, Any], Any]:
    full_range = layer_end == total_layers
    generated = reference["generated_token_ids"]

    def prefill() -> dict[str, Any]:
        with torch_module.no_grad():
            return _run_segment_prefill(manager, token_ids, full_range=full_range)

    def decode(cache: Any) -> tuple[Any, list[int]]:
        outputs = None
        observed: list[int] = []
        with torch_module.no_grad():
            for token_id in generated[:decode_steps]:
                token = token_ids.new_tensor([[token_id]])
                outputs = manager.forward_layers(
                    input_ids=token, past_key_values=cache, use_cache=True,
                    apply_lm_head=full_range,
                )
                cache = _cache_from_output(outputs)
                if full_range:
                    observed.append(_generated(outputs["logits"]))
                else:
                    hidden = outputs["hidden_states"]
                    if not torch_module.isfinite(hidden).all().item():
                        raise RuntimeError("layer-range decode produced non-finite activations")
        return outputs, observed

    for _ in range(warmup):
        output = prefill()
        cache = _cache_from_output(output)
        decode(cache)

    _sync(torch_module, device)
    prefill_samples: list[float] = []
    prefill_process_cpu_samples: list[float] = []
    gc_before = [int(item["collections"]) for item in gc.get_stats()]
    last_prefill = None
    prefill_exact = True
    decode_samples: list[float] = []
    decode_process_cpu_samples: list[float] = []
    decode_exact = True
    last_decode = None
    for phase in _phase_measurement_order(repeats):
        if phase == "prefill":
            _sync(torch_module, device)
            process_started = time.process_time()
            started = time.perf_counter()
            output = prefill()
            _sync(torch_module, device)
            prefill_samples.append((time.perf_counter() - started) * 1000.0)
            prefill_process_cpu_samples.append((time.process_time() - process_started) * 1000.0)
            last_prefill = output
            hidden = output.get("hidden_states")
            if hidden is not None and not torch_module.isfinite(hidden).all().item():
                raise RuntimeError("layer-range prefill produced non-finite activations")
            if full_range:
                prefill_exact = prefill_exact and _argmax(output["logits"]) == reference["prefill_argmax"]
            continue

        output = prefill()
        cache = _cache_from_output(output)
        _sync(torch_module, device)
        process_started = time.process_time()
        started = time.perf_counter()
        last_decode, observed = decode(cache)
        _sync(torch_module, device)
        decode_samples.append((time.perf_counter() - started) * 1000.0 / decode_steps)
        decode_process_cpu_samples.append(
            (time.process_time() - process_started) * 1000.0 / decode_steps
        )
        if full_range:
            decode_exact = decode_exact and observed == generated[1:decode_steps + 1]

    cache_summaries = {
        "prefill": summarize_cache(_cache_from_output(last_prefill), torch_module),
        "decode": summarize_cache(_cache_from_output(last_decode), torch_module),
    }
    gc_after = [int(item["collections"]) for item in gc.get_stats()]
    row = {
        "layer_range": [0, layer_end],
        "has_embedding": True,
        "has_lm_head": full_range,
        "prefill": {
            "measured_unit": "ms_per_forward_call",
            "uninstrumented_wall": summarize_samples(prefill_samples),
            "process_cpu_time_diagnostic_ms_per_forward_call": summarize_nonnegative_samples(prefill_process_cpu_samples),
            "argmax_exact_vs_same_device_full_reference": prefill_exact if full_range else None,
            "finite_hidden_verified": True,
        },
        "decode": {
            "measured_unit": "ms_per_forward_call",
            "decode_steps_per_sample": decode_steps,
            "input_token_policy": "teacher_forced_from_same_device_full_model_greedy_reference",
            "uninstrumented_wall": summarize_samples(decode_samples),
            "process_cpu_time_diagnostic_ms_per_forward_call": summarize_nonnegative_samples(decode_process_cpu_samples),
            "generated_tokens_exact_vs_same_device_full_reference": decode_exact if full_range else None,
        },
        "measurement_order": _phase_measurement_order(repeats),
        "gc_collections_during_measurement": [after - before for before, after in zip(gc_before, gc_after)],
        "actual_kv_cache": cache_summaries,
    }
    return row, last_prefill.get("hidden_states")


def _pair_forward(torch_module: Any, first: Any, second: Any, prompt: Any,
                  *, cache_first: Any = None, cache_second: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    with torch_module.no_grad():
        if cache_first is None:
            first_out = first.forward_layers(input_ids=prompt, use_cache=True, apply_lm_head=False)
        else:
            first_out = first.forward_layers(
                input_ids=prompt, past_key_values=cache_first, use_cache=True, apply_lm_head=False,
            )
        first_cache = _cache_from_output(first_out)
        second_out = second.forward_layers(
            hidden_states=first_out["hidden_states"], past_key_values=cache_second,
            use_cache=True, apply_lm_head=True,
        )
        second_cache = _cache_from_output(second_out)
    return (first_out | {"cache": first_cache}, second_out | {"cache": second_cache})


def _run_split_pair(torch_module: Any, first: Any, second: Any, prompt: Any,
                    reference: dict[str, Any], *, decode_steps: int) -> dict[str, Any]:
    first_out, second_out = _pair_forward(torch_module, first, second, prompt)
    prefill_argmax = _argmax(second_out["logits"])
    generated = [_generated(second_out["logits"])]
    first_cache = _cache_from_output(first_out)
    second_cache = _cache_from_output(second_out)
    for _ in range(decode_steps):
        token = prompt.new_tensor([[generated[-1]]])
        first_out, second_out = _pair_forward(
            torch_module, first, second, token,
            cache_first=first_cache, cache_second=second_cache,
        )
        first_cache = _cache_from_output(first_out)
        second_cache = _cache_from_output(second_out)
        generated.append(_generated(second_out["logits"]))
    return {
        "prefill_argmax": prefill_argmax,
        "generated_token_ids": generated,
        "first_stage_cache": summarize_cache(first_cache, torch_module),
        "second_stage_cache": summarize_cache(second_cache, torch_module),
    }


def _time_split_direction(torch_module: Any, first: Any, second: Any, prompt: Any,
                          reference: dict[str, Any], *, decode_steps: int,
                          warmup: int, repeats: int, max_cv: float) -> dict[str, Any]:
    devices = (first.get_device(), second.get_device())
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    exact_prefill = True
    exact_decode = True
    last_result = None
    boundary_copy_samples: list[float] = []
    boundary_copy_identity: dict[str, Any] | None = None

    def prefill_once() -> tuple[dict[str, Any], dict[str, Any]]:
        return _pair_forward(torch_module, first, second, prompt)

    def decode_from_prefill(first_out: dict[str, Any], second_out: dict[str, Any]) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
        first_cache = _cache_from_output(first_out)
        second_cache = _cache_from_output(second_out)
        generated = [_generated(second_out["logits"])]
        for _ in range(decode_steps):
            token = prompt.new_tensor([[generated[-1]]])
            first_out, second_out = _pair_forward(
                torch_module, first, second, token,
                cache_first=first_cache, cache_second=second_cache,
            )
            first_cache = _cache_from_output(first_out)
            second_cache = _cache_from_output(second_out)
            generated.append(_generated(second_out["logits"]))
        return generated, first_out, second_out

    for _ in range(warmup):
        warm_first, warm_second = prefill_once()
        decode_from_prefill(warm_first, warm_second)
    for _ in range(repeats):
        _sync(torch_module, *devices)
        started = time.perf_counter()
        _first_out, second_out = prefill_once()
        _sync(torch_module, *devices)
        prefill_samples.append((time.perf_counter() - started) * 1000.0)
        exact_prefill = exact_prefill and _argmax(second_out["logits"]) == reference["prefill_argmax"]

        source_hidden = _first_out.get("hidden_states")
        if source_hidden is None:
            raise RuntimeError("first split stage returned no boundary activation")
        copied = None
        destination_dtype = next(second.model.parameters()).dtype
        for _copy_index in range(3):
            _sync(torch_module, devices[0], devices[1])
            copy_started = time.perf_counter()
            copied = source_hidden.to(
                device=devices[1], dtype=destination_dtype, non_blocking=False,
            )
            _sync(torch_module, devices[0], devices[1])
            boundary_copy_samples.append((time.perf_counter() - copy_started) * 1000.0)
        difference = (
            source_hidden.detach().to(device="cpu", dtype=torch_module.float32)
            - copied.detach().to(device="cpu", dtype=torch_module.float32)
        ).abs()
        boundary_copy_identity = {
            "shape": [int(size) for size in source_hidden.shape],
            "dtype": str(source_hidden.dtype).removeprefix("torch."),
            "source_device": str(source_hidden.device),
            "destination_device": str(devices[1]),
            "destination_dtype": str(destination_dtype).removeprefix("torch."),
            "bytes": int(source_hidden.numel() * source_hidden.element_size()),
            "destination_bytes": int(copied.numel() * copied.element_size()),
            "numerically_exact": not difference.numel() or float(difference.max().item()) == 0.0,
            "max_abs_conversion_error": float(difference.max().item()) if difference.numel() else 0.0,
        }

        _sync(torch_module, *devices)
        decode_first, decode_second = prefill_once()
        _sync(torch_module, *devices)
        started = time.perf_counter()
        generated, first_out, second_out = decode_from_prefill(decode_first, decode_second)
        _sync(torch_module, *devices)
        decode_samples.append((time.perf_counter() - started) * 1000.0 / decode_steps)
        exact_decode = exact_decode and generated == reference["generated_token_ids"]
        last_result = {
            "first_stage_cache": summarize_cache(_cache_from_output(first_out), torch_module),
            "second_stage_cache": summarize_cache(_cache_from_output(second_out), torch_module),
        }

    return {
        "prefill_argmax_exact": exact_prefill,
        "generated_tokens_exact": exact_decode,
        "prefill": summarize_samples(prefill_samples),
        "decode": summarize_samples(decode_samples),
        "boundary_copy": {
            **(boundary_copy_identity or {}),
            "samples_ms": boundary_copy_samples,
            "median_ms": statistics.median(boundary_copy_samples),
        },
        "kv_cache_after_decode": last_result,
        "max_runtime_cv": max_cv,
    }


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("loopback peer closed mid-frame")
        data.extend(chunk)
    return bytes(data)


def _loopback_tensor_transport(tensor: Any, torch_module: Any, *, repeats: int) -> dict[str, Any]:
    from src.tcp_comm import deserialize_tensor_fast, serialize_tensor_fast

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    errors: list[str] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                while True:
                    header = connection.recv(4)
                    if not header:
                        return
                    header += _recv_exact(connection, 4 - len(header))
                    size = struct.unpack(">I", header)[0]
                    body = _recv_exact(connection, size)
                    connection.sendall(header + body)
        except OSError as exc:
            errors.append(type(exc).__name__)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, name="torch-hw-loopback", daemon=True)
    thread.start()
    codec_samples: list[float] = []
    wire_samples: list[float] = []
    decode_samples: list[float] = []
    total_samples: list[float] = []
    tensor_cpu = tensor.detach().cpu().contiguous()
    with socket.create_connection(("127.0.0.1", port), timeout=10.0) as client:
        for _ in range(repeats):
            started_total = time.perf_counter()
            started = time.perf_counter()
            payload = serialize_tensor_fast(tensor)
            codec_samples.append((time.perf_counter() - started) * 1000.0)
            frame = struct.pack(">I", len(payload)) + payload
            started = time.perf_counter()
            client.sendall(frame)
            header = _recv_exact(client, 4)
            size = struct.unpack(">I", header)[0]
            echoed = _recv_exact(client, size)
            wire_samples.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            restored = deserialize_tensor_fast(echoed)
            decode_samples.append((time.perf_counter() - started) * 1000.0)
            if not torch_module.equal(tensor_cpu, restored):
                raise RuntimeError("QLH tensor serializer loopback changed activation values")
            total_samples.append((time.perf_counter() - started_total) * 1000.0)
    thread.join(timeout=10.0)
    if thread.is_alive() or errors:
        raise RuntimeError(f"loopback transport did not close cleanly: {errors}")
    return {
        "scope": "same_host_loopback_tcp_echo; QLH serialize_tensor_fast payload; no auth/control envelope",
        "tensor_shape": [int(size) for size in tensor.shape],
        "tensor_dtype": str(tensor.dtype).removeprefix("torch."),
        "tensor_bytes": int(tensor.numel() * tensor.element_size()),
        "payload_bytes": len(payload),
        "exact_roundtrip": True,
        "serialize_ms": summarize_samples(codec_samples),
        "tcp_roundtrip_ms": summarize_samples(wire_samples),
        "deserialize_ms": summarize_samples(decode_samples),
        "end_to_end_ms": summarize_samples(total_samples),
    }


def _device_info(torch_module: Any) -> dict[str, Any]:
    if torch_module.cuda.is_available():
        device = torch_module.device("cuda:0")
        props = torch_module.cuda.get_device_properties(device)
        return {
            "type": "cuda", "name": torch_module.cuda.get_device_name(device),
            "capability": list(torch_module.cuda.get_device_capability(device)),
            "total_memory_bytes": int(props.total_memory),
        }
    return {"type": "cpu", "name": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count()}


def _compare_cli(cpu_path: Path, cuda_path: Path, out_path: Path, max_cv: float) -> int:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    result = compare_reports(cpu, cuda, max_cv=max_cv)
    result["source_paths"] = {"cpu": str(cpu_path), "cuda": str(cuda_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[comparison] {out_path}")
    print(f"[same-host CPU/CUDA split admitted] {result['same_host_cpu_cuda_split_admitted']}")
    print("[cross-host inference admitted] False (no production-equivalent cross-host inference measurement)")
    return 0 if result["same_host_cpu_cuda_split_admitted"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CPU_REPORT", "CUDA_REPORT"))
    parser.add_argument("--max-cv", type=float, default=0.10)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill-tokens", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--ranges", nargs="+", type=int, default=[4, 12, 24])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260923,
                        help="fixed seed used to randomize workload and layer-range measurement order")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--cpu-affinity", type=_parse_cpu_affinity, default=None, metavar="CPU[,CPU...]",
        help="pin the probe process to logical CPUs before importing Torch (opt-in)",
    )
    parser.add_argument(
        "--interop-threads", type=int, default=None, metavar="N",
        help="set Torch inter-op threads for controlled local probes (opt-in)",
    )
    parser.add_argument(
        "--telemetry", choices=("none", "cuda", "all"), default="none",
        help="collect opt-in sidecar hardware telemetry without changing timing calls",
    )
    parser.add_argument("--precision", choices=("fp32", "deployment_default"), default="fp32",
                        help="fp32 controls device comparison; deployment_default observes CPU FP32/CUDA FP16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.max_cv) or not 0 <= args.max_cv <= 1:
        print("FAIL: max-cv must be in [0, 1]", file=sys.stderr)
        return 2
    if args.compare:
        out = args.json_out or DEFAULT_REPORT_DIR / "cpu-cuda-comparison.json"
        return _compare_cli(args.compare[0], args.compare[1], out, args.max_cv)
    if (args.repeats < 3 or args.warmup < 0 or args.decode_steps < 1 or args.threads < 1
            or (args.interop_threads is not None and args.interop_threads < 1)
            or not args.prefill_tokens or any(value < 1 for value in args.prefill_tokens)
            or any(value < 1 for value in args.ranges) or not args.model_dir.is_dir()):
        print("FAIL: invalid workload or model directory", file=sys.stderr)
        return 2

    try:
        cpu_affinity = _apply_cpu_affinity(args.cpu_affinity)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import config as qlh_config
    import model_module
    from transformers import AutoTokenizer

    qlh_config.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    model_module.USE_COMPILE = False
    if args.interop_threads is not None:
        torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device_info = _device_info(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=False, local_files_only=True,
    )
    vocab_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if vocab_ids.numel() == 0:
        raise RuntimeError("prompt tokenized to an empty input")
    model_config = model_module.AutoConfig.from_pretrained(
        str(args.model_dir), trust_remote_code=False, local_files_only=True,
    )
    total_layers = int(model_config.num_hidden_layers)
    ranges = sorted(set(args.ranges))
    if ranges[-1] != total_layers or any(end > total_layers for end in ranges):
        print(f"FAIL: ranges must include total layer count {total_layers} and stay in bounds", file=sys.stderr)
        return 2
    workload_sizes = sorted(set(args.prefill_tokens))
    tokenizer_path = args.model_dir / "tokenizer.json"
    tokenizer_sha256 = _sha256_file(tokenizer_path) if tokenizer_path.is_file() else None
    model_identity = _artifact_identity(args.model_dir)
    runtime_dtype = "float32" if args.precision == "fp32" else "device_default"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "ticket": "TORCH-HW-ADMIT-01",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": (
            "same_model_matched_fp32_cpu_cuda_layer_and_split_matrix"
            if args.precision == "fp32" else "same_model_device_default_precision_observation"
        ),
        "model_identity": model_identity,
        "tokenizer_sha256": tokenizer_sha256,
        "precision_mode": args.precision,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "torch_cuda_runtime": torch.version.cuda,
            "device": device_info,
            "host_identity": platform.node(),
            "operating_system": platform.platform(),
            "threads": args.threads,
            "interop_threads": int(torch.get_num_interop_threads()),
            "interop_threads_requested": args.interop_threads,
            "cpu_affinity": cpu_affinity,
            "compile_enabled": False,
            "measurement_dtype": runtime_dtype,
        },
        "policy": {
            "warmup_iterations": args.warmup,
            "repeats_requested": args.repeats,
            "repeats_minimum": 3,
            "max_runtime_cv": args.max_cv,
            "decode_steps": args.decode_steps,
            "phase_order": "alternating_prefill_decode_pairs",
            "timing_diagnostic": "process_cpu_time_is_observational_and_not_an_admission_metric",
            "prompt_length_policy": "tile exact local prompt token sequence, then truncate to requested length",
        },
        "measurement_execution_order": {"seed": args.order_seed, "workload_tokens": [], "ranges_by_workload": {}},
        "workloads": {},
        "range_profiles": {},
        "heterogeneous_pairs": {},
        "cross_host": {
            "admitted": False,
            "reason": "same_host_runner_does_not_measure_cross_host_inference; see separate link evidence",
            "production_transport_tested": False,
        },
        "provenance": _git_provenance(),
        "telemetry": {
            "source": args.telemetry,
            "status": "pending",
            "window_scope": "range_profiles_and_heterogeneous_pairs",
            "collection_errors_fail_closed": True,
        },
    }

    full = _prepare_manager(
        model_module, args.model_dir, 0, total_layers, total_layers,
        dtype_mode=args.precision,
    )
    device = full.get_device()
    runtime_dtype = str(next(full.model.parameters()).dtype).removeprefix("torch.")
    report["runtime"]["device"] = device_info
    report["runtime"]["measurement_dtype"] = runtime_dtype
    references: dict[str, dict[str, Any]] = {}
    prompt_tensors: dict[str, Any] = {}
    try:
        for token_count in workload_sizes:
            key = str(token_count)
            repeat_count = (token_count + vocab_ids.shape[1] - 1) // vocab_ids.shape[1]
            prompt_ids = vocab_ids.repeat(1, repeat_count)[:, :token_count].contiguous().to(device)
            reference = _run_reference(full, prompt_ids, args.decode_steps, torch)
            references[key] = reference
            prompt_tensors[key] = prompt_ids
            token_hash = hashlib.sha256(prompt_ids.detach().cpu().contiguous().numpy().astype("<i8").tobytes()).hexdigest()
            report["workloads"][key] = {
                "prefill_tokens": token_count,
                "decode_steps": args.decode_steps,
                "prompt_token_sha256": token_hash,
                "reference": reference,
                "reference_fingerprint": _fingerprint(reference),
            }
        full.unload_model()
    finally:
        full.unload_model()
    del full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    telemetry = _HardwareTelemetrySampler(args.telemetry)
    telemetry.start()
    loopback_rows: dict[str, Any] = {}
    workload_measurement_order = list(workload_sizes)
    order_rng = random.Random(args.order_seed)
    order_rng.shuffle(workload_measurement_order)
    report["measurement_execution_order"]["workload_tokens"] = workload_measurement_order
    for token_count in workload_sizes:
        range_order = list(ranges)
        order_rng.shuffle(range_order)
        report["measurement_execution_order"]["ranges_by_workload"][str(token_count)] = range_order

    for token_count in workload_measurement_order:
        prompt_key = str(token_count)
        prompt = prompt_tensors[prompt_key]
        reference = references[prompt_key]
        for layer_end in report["measurement_execution_order"]["ranges_by_workload"][prompt_key]:
            manager = _prepare_manager(
                model_module, args.model_dir, 0, layer_end, total_layers, dtype_mode=args.precision,
            )
            if manager.get_device().type != device.type:
                raise RuntimeError("layer range changed execution device")
            key = f"{token_count}:{layer_end}"
            window_started = datetime.now(timezone.utc).isoformat()
            try:
                row, link_hidden = _measure_segment(
                    torch, manager, device, prompt, reference,
                    layer_end=layer_end, total_layers=total_layers,
                    warmup=args.warmup, repeats=args.repeats, decode_steps=args.decode_steps,
                )
                row["telemetry_window"] = {
                    "started_at_utc": window_started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                row["loader_metrics"] = getattr(manager, "_layer_load_metrics", None)
                report["range_profiles"][key] = row
                if layer_end == 12 and link_hidden is not None:
                    loopback_rows[prompt_key] = _loopback_tensor_transport(
                        link_hidden, torch, repeats=max(3, args.repeats),
                    )
            finally:
                manager.unload_model()
                del manager
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    report["loopback_transport"] = loopback_rows

    if torch.cuda.is_available():
        for token_count in workload_sizes:
            prompt_key = str(token_count)
            prompt = prompt_tensors[prompt_key]
            reference = references[prompt_key]
            row = {"split_layer": total_layers // 2, "directions": {}}
            for direction in ("cpu_to_cuda", "cuda_to_cpu"):
                first_device, second_device = (
                    ("cpu", "cuda:0") if direction == "cpu_to_cuda" else ("cuda:0", "cpu")
                )
                first = second = None
                window_started = datetime.now(timezone.utc).isoformat()
                try:
                    first = _prepare_manager(
                        model_module, args.model_dir, 0, total_layers // 2, total_layers,
                        dtype_mode=args.precision, device=first_device,
                    )
                    second = _prepare_manager(
                        model_module, args.model_dir, total_layers // 2, total_layers, total_layers,
                        dtype_mode=args.precision, device=second_device,
                    )
                    result = _time_split_direction(
                        torch, first, second, prompt, reference,
                        decode_steps=args.decode_steps, warmup=args.warmup,
                        repeats=args.repeats, max_cv=args.max_cv,
                    )
                    result["telemetry_window"] = {
                        "started_at_utc": window_started,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    row["directions"][direction] = {
                        "prefill_argmax_exact": result["prefill_argmax_exact"],
                        "generated_tokens_exact": result["generated_tokens_exact"],
                        "prefill": result["prefill"],
                        "decode": result["decode"],
                        "boundary_copy": result["boundary_copy"],
                        "kv_cache_after_decode": result["kv_cache_after_decode"],
                        "telemetry_window": result["telemetry_window"],
                    }
                finally:
                    for manager in (first, second):
                        if manager is not None:
                            manager.unload_model()
                    del first, second
                    torch.cuda.empty_cache()
            report["heterogeneous_pairs"][prompt_key] = row

    report["telemetry"] = telemetry.stop()

    report["local_same_host_admission"] = {
        "same_device_phase_costs": "requires paired CPU and CUDA reports for matching artifact/input identity",
        "cpu_cuda_layer_split": "measured here" if torch.cuda.is_available() else "not_available_on_cpu_only_host",
        "cross_host": False,
        "production_runtime_enabled": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.json_out or DEFAULT_REPORT_DIR / f"torch-hardware-admit-{device.type}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hardware profile] {out}")
    print(f"[device] {device_info}; host={platform.node()}; precision={args.precision}/{runtime_dtype}; model_sha256={model_identity['weights_sha256']}")
    print(f"[range profiles] {len(report['range_profiles'])}; [loopback TCP] {len(loopback_rows)}")
    if not torch.cuda.is_available():
        print("[notice] CPU profile only; CPU/CUDA pair admission remains pending")
    else:
        print(f"[same-host mixed placements] {len(report['heterogeneous_pairs'])} workloads x 2 directions")
    print("[cross-host] not evaluated by this runner; consult separate cross-host evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
