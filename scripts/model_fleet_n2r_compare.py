#!/usr/bin/env python3
"""MF-MEM-N2R isolated real-model comparison harness.

This harness loads the fixed local DeepSeek-R1-Distill-Qwen-7B Safetensors
artifact only for explicit experiments. It compares resident NF4 and
device_map=auto placement. CPU-only and real module-level explicit paging are
resource/adapter gates in this ticket and fail closed instead of silently
falling back to another model or a smaller workload.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import platform
from threading import Lock
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-r1-distill-qwen-7b"
MODEL_INDEX = MODEL_PATH / "model.safetensors.index.json"
TARGET_TICKET = "MF-MEM-N2R"


class ResourceRejected(RuntimeError):
    """The requested N2R comparison mode cannot run safely."""


class PagerCancelled(RuntimeError):
    """Explicit layer pager cancellation reached a layer boundary."""


class TokenLatencyProcessor:
    """Capture generation timing without changing greedy decoding decisions."""

    def __init__(self, started: float):
        self.started = started
        self.ttft_ms = None
        self.inter_token_ms = []
        self._last = None

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        now = time.perf_counter()
        if self.ttft_ms is None:
            self.ttft_ms = (now - self.started) * 1000.0
        elif self._last is not None:
            self.inter_token_ms.append((now - self._last) * 1000.0)
        self._last = now
        return scores


def _load_torch_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ResourceRejected(f"required runtime dependency unavailable: {exc}") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def fixed_weight_bytes(model_index: Path = MODEL_INDEX) -> int:
    if not model_index.is_file():
        raise ResourceRejected(f"fixed Safetensors index is missing: {model_index}")
    raw = json.loads(model_index.read_text(encoding="utf-8"))
    total = int(raw.get("metadata", {}).get("total_size", 0))
    if total <= 0:
        raise ResourceRejected("fixed Safetensors index has no positive total_size")
    return total


def _available_ram_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return 0


def _digest_tensor(torch: Any, tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_input_batch(torch: Any, tokenizer: Any, *, context: int, batch_size: int) -> dict[str, Any]:
    if context not in {128, 2048, 8192}:
        raise ValueError("context must be one of 128, 2048, 8192")
    if batch_size not in {1, 4}:
        raise ValueError("batch_size must be one of 1, 4")
    seed_ids = tokenizer(
        "Explain explicit prefetch and why it can reduce demand paging latency.",
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    repeats = (context + int(seed_ids.numel()) - 1) // int(seed_ids.numel())
    ids = seed_ids.repeat(repeats)[:context].unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": attention_mask}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class FixedTileBridge:
    """Two fixed 64 MiB host/device slots used as an explicit transfer bridge."""

    TILE_BYTES = 64 * 1024 * 1024
    SLOTS = 2

    def __init__(self, torch: Any, device: Any):
        self.torch = torch
        self.device = device
        try:
            self.host_slots = [
                torch.empty(self.TILE_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
                for _ in range(self.SLOTS)
            ]
            self.device_slots = [
                torch.empty(self.TILE_BYTES, dtype=torch.uint8, device=device)
                for _ in range(self.SLOTS)
            ]
            self.stream = torch.cuda.Stream(device=device)
        except (MemoryError, RuntimeError) as exc:
            self.close()
            raise ResourceRejected(f"cannot allocate fixed 64 MiB x 2 tile bridge: {exc}") from exc
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.chunk_count = 0
        self.peak_module_bytes = 0
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_future = None
        self.prefetched = None
        self.prefetch_intervals = []
        self.prefetch_stage_bytes = 0
        self.prefetch_stage_chunks = 0
        self.prefetch_lock = Lock()

    @staticmethod
    def _parameter_bytes(torch: Any, parameter: Any) -> Any:
        value = parameter.detach().contiguous()
        try:
            return value.view(torch.uint8).reshape(-1)
        except RuntimeError as exc:
            raise ResourceRejected(
                f"parameter storage cannot be represented as explicit byte tiles: {exc}"
            ) from exc

    def page_in_module(self, module: Any) -> int:
        if self.prefetch_future is not None:
            self.prefetch_future.result()
        with self.prefetch_lock:
            staged = self.prefetched if self.prefetched and self.prefetched["module_id"] == id(module) else None
            self.prefetched = None
            self.prefetch_future = None
        if staged is not None:
            module_bytes = int(staged["module_bytes"])
            for slot_index, size in staged["chunks"]:
                host = self.host_slots[slot_index]
                device = self.device_slots[slot_index]
                with self.torch.cuda.stream(self.stream):
                    device[:size].copy_(host[:size], non_blocking=True)
                self.stream.synchronize()
                self.h2d_bytes += size
                self.chunk_count += 1
            self.peak_module_bytes = max(self.peak_module_bytes, module_bytes)
            return module_bytes
        module_bytes = 0
        slot_index = 0
        for parameter in module.parameters():
            raw = self._parameter_bytes(self.torch, parameter)
            module_bytes += int(raw.numel())
            for offset in range(0, int(raw.numel()), self.TILE_BYTES):
                end = min(int(raw.numel()), offset + self.TILE_BYTES)
                size = end - offset
                host = self.host_slots[slot_index]
                device = self.device_slots[slot_index]
                host[:size].copy_(raw[offset:end])
                with self.torch.cuda.stream(self.stream):
                    device[:size].copy_(host[:size], non_blocking=True)
                self.stream.synchronize()
                self.h2d_bytes += size
                self.chunk_count += 1
                slot_index = (slot_index + 1) % self.SLOTS
        self.peak_module_bytes = max(self.peak_module_bytes, module_bytes)
        return module_bytes

    def prefetch_module(self, module: Any) -> None:
        if self.prefetch_future is not None:
            raise RuntimeError("only one prefetched layer may be in flight")

        def stage() -> None:
            started = time.perf_counter()
            module_bytes = 0
            chunks = []
            slot_index = 0
            for parameter in module.parameters():
                raw = self._parameter_bytes(self.torch, parameter)
                module_bytes += int(raw.numel())
                for offset in range(0, int(raw.numel()), self.TILE_BYTES):
                    end = min(int(raw.numel()), offset + self.TILE_BYTES)
                    size = end - offset
                    self.host_slots[slot_index][:size].copy_(raw[offset:end])
                    chunks.append((slot_index, size))
                    slot_index = (slot_index + 1) % self.SLOTS
            finished = time.perf_counter()
            with self.prefetch_lock:
                self.prefetched = {
                    "module_id": id(module),
                    "module_bytes": module_bytes,
                    "chunks": chunks,
                }
                self.prefetch_intervals.append((started, finished))
                self.prefetch_stage_bytes += module_bytes
                self.prefetch_stage_chunks += len(chunks)

        self.prefetch_future = self.prefetch_executor.submit(stage)

    def close(self) -> None:
        if hasattr(self, "prefetch_executor"):
            self.prefetch_executor.shutdown(wait=True)
        if hasattr(self, "stream"):
            self.stream.synchronize()
        if hasattr(self, "host_slots"):
            self.host_slots.clear()
        if hasattr(self, "device_slots"):
            self.device_slots.clear()
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()

    def record_page_out(self, byte_count: int) -> None:
        """Record the corresponding layer page-out accounted by the adapter."""
        self.d2h_bytes += int(byte_count)


def install_explicit_layer_hooks(
    torch: Any,
    model: Any,
    *,
    bridge: FixedTileBridge | None = None,
    cancel_after_layer: int | None = None,
    prefetch_distance: int = 0,
) -> dict[str, Any]:
    """Install explicit page-in/page-out hooks around real Qwen2 decoder layers."""
    if prefetch_distance not in {0, 1}:
        raise ValueError("prefetch_distance must be 0 or 1")
    layers = list(model.model.layers)
    if not layers:
        raise ResourceRejected("Qwen2 decoder layers are unavailable")
    device = torch.device("cuda:0")
    layer_bytes = []
    for layer in layers:
        layer_bytes.append(sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in layer.parameters()
        ))
    for layer in layers:
        layer.to("cpu")
    state = {
        "page_in": 0,
        "page_out": 0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "cancelled": 0,
        "compute_intervals": [],
        "compute_started": {},
    }
    handles = []

    def page_in(module, _inputs):
        index = int(getattr(module, "_qlh_layer_index"))
        if cancel_after_layer is not None and state["page_in"] >= cancel_after_layer:
            state["cancelled"] = 1
            raise PagerCancelled(f"cancelled before layer {index}")
        torch.cuda.synchronize(device)
        if bridge is not None:
            bridge.page_in_module(module)
        module.to(device)
        state["page_in"] += 1
        state["h2d_bytes"] += layer_bytes[index]
        state["compute_started"][id(module)] = time.perf_counter()
        if bridge is not None and prefetch_distance == 1 and index + 1 < len(layers):
            bridge.prefetch_module(layers[index + 1])

    def page_out(module, _inputs, _output):
        index = int(getattr(module, "_qlh_layer_index"))
        torch.cuda.synchronize(device)
        forward_finished = time.perf_counter()
        module.to("cpu")
        state["page_out"] += 1
        state["d2h_bytes"] += layer_bytes[index]
        started = state["compute_started"].pop(id(module), None)
        if started is not None:
            state["compute_intervals"].append((started, forward_finished))
        if bridge is not None:
            bridge.record_page_out(layer_bytes[index])

    for index, layer in enumerate(layers):
        layer._qlh_layer_index = index
        handles.append(layer.register_forward_pre_hook(page_in))
        handles.append(layer.register_forward_hook(page_out))
    return {
        "handles": handles,
        "state": state,
        "layer_count": len(layers),
        "layer_bytes_total": sum(layer_bytes),
        "adapter": "module_move_hook",
        "bridge": bridge,
        "cancelled": state["cancelled"],
        "prefetch_distance": prefetch_distance,
    }


def _rejected(mode: str, reason: str, *, result_path: Path, ticket: str) -> int:
    _write_json(result_path, {
        "schema_version": 1,
        "ticket": ticket,
        "status": "resource_rejected",
        "completed": 1,
        "resource_rejected": 1,
        "mode": mode,
        "reason": reason,
        "completed_at": time.time(),
    })
    return 0


def _interval_overlap_stats(
    source_intervals: Sequence[tuple[float, float]],
    target_intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    source_total = sum(max(0.0, end - start) for start, end in source_intervals)
    target_total = sum(max(0.0, end - start) for start, end in target_intervals)
    overlap = sum(
        max(0.0, min(source_end, target_end) - max(source_start, target_start))
        for source_start, source_end in source_intervals
        for target_start, target_end in target_intervals
    )
    capacity = min(source_total, target_total)
    return {
        "source_total_ms": source_total * 1000.0,
        "target_total_ms": target_total * 1000.0,
        "overlap_ms": overlap * 1000.0,
        "effective_overlap_ratio": overlap / capacity if capacity > 0.0 else 0.0,
    }


def _latency_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    active_ticket = getattr(args, "ticket", TARGET_TICKET)
    if args.mode not in {"resident-nf4", "device-map-auto", "cpu-only", "explicit-layer-pager", "explicit-layer-pager-64"}:
        raise ValueError("unsupported mode")
    weight_bytes = fixed_weight_bytes()
    available_ram = _available_ram_bytes()
    if args.mode == "cpu-only":
        required = int(weight_bytes * 1.20) + 2 * 1024**3
        if available_ram <= 0 or available_ram < required:
            raise ResourceRejected(
                f"CPU-only requires at least {required} bytes available RAM; "
                f"current available RAM is {available_ram}"
            )
        raise ResourceRejected("CPU-only is intentionally not enabled for the 8 GB CUDA N2R gate")
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _load_torch_transformers()
    if not torch.cuda.is_available():
        raise ResourceRejected("CUDA is unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    placement: Any = "auto"
    started = time.perf_counter()
    model = None
    hook_runtime = None
    bridge = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map=placement,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=False,
        )
        load_s = time.perf_counter() - started
        if args.mode in {"explicit-layer-pager", "explicit-layer-pager-64"}:
            bridge = FixedTileBridge(torch, torch.device("cuda:0")) if args.mode == "explicit-layer-pager-64" else None
            hook_runtime = install_explicit_layer_hooks(
                torch,
                model,
                bridge=bridge,
                cancel_after_layer=args.cancel_after_layer or None,
                prefetch_distance=args.prefetch_distance if bridge is not None else 0,
            )
        inputs = build_input_batch(
            torch,
            tokenizer,
            context=args.context,
            batch_size=args.batch_size,
        )
        first_device = next(model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generate_started = time.perf_counter()
        latency_processor = TokenLatencyProcessor(generate_started) if args.measure_latency else None
        logits_processor = None
        if latency_processor is not None:
            from transformers import LogitsProcessorList

            logits_processor = LogitsProcessorList([latency_processor])
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "use_cache": args.mode not in {"explicit-layer-pager", "explicit-layer-pager-64"},
                "pad_token_id": tokenizer.eos_token_id,
            }
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = logits_processor
            output = model.generate(**inputs, **generate_kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - generate_started) * 1000.0
        generated_tokens = max(0, int(output.shape[1] - inputs["input_ids"].shape[1]))
        total_tokens = generated_tokens * args.batch_size
        result = {
            "schema_version": 1,
            "ticket": active_ticket,
            "status": "completed",
            "completed": 1,
            "resource_rejected": 0,
            "mode": args.mode,
            "model_id": "deepseek-r1-distill-qwen-7b",
            "artifact_path": str(MODEL_PATH),
            "artifact_weight_bytes": weight_bytes,
            "quantization": "nf4",
            "pager_adapter": hook_runtime["adapter"] if hook_runtime else None,
            "fixed_tile_mib": 64 if args.mode == "explicit-layer-pager-64" else None,
            "tile_slots": 2 if args.mode == "explicit-layer-pager-64" else 0,
            "prefetch_distance": hook_runtime["prefetch_distance"] if hook_runtime else 0,
            "context": args.context,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "load_s": load_s,
            "e2e_ms": elapsed_ms,
            "decode_tok_s": total_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            "output_digest": _digest_tensor(torch, output),
            "first_parameter_device": str(first_device),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "h2d_bytes": hook_runtime["bridge"].h2d_bytes if hook_runtime and hook_runtime["bridge"] else (hook_runtime["state"]["h2d_bytes"] if hook_runtime else 0),
            "d2h_bytes": hook_runtime["state"]["d2h_bytes"] if hook_runtime else 0,
            "page_in_count": hook_runtime["state"]["page_in"] if hook_runtime else 0,
            "page_out_count": hook_runtime["state"]["page_out"] if hook_runtime else 0,
            "paged_layer_count": hook_runtime["layer_count"] if hook_runtime else 0,
            "paged_layer_bytes_total": hook_runtime["layer_bytes_total"] if hook_runtime else 0,
            "bridge_chunk_count": hook_runtime["bridge"].chunk_count if hook_runtime and hook_runtime["bridge"] else 0,
            "bridge_peak_module_bytes": hook_runtime["bridge"].peak_module_bytes if hook_runtime and hook_runtime["bridge"] else 0,
            "prefetch_stage_bytes": hook_runtime["bridge"].prefetch_stage_bytes if hook_runtime and hook_runtime["bridge"] else 0,
            "prefetch_stage_chunks": hook_runtime["bridge"].prefetch_stage_chunks if hook_runtime and hook_runtime["bridge"] else 0,
            "prefetch_host_overlap": _interval_overlap_stats(
                hook_runtime["bridge"].prefetch_intervals,
                hook_runtime["state"]["compute_intervals"],
            ) if hook_runtime and hook_runtime["bridge"] else None,
            "cuda_stream_overlap_ratio": 0.0 if hook_runtime and hook_runtime["bridge"] else None,
            "ttft_ms": latency_processor.ttft_ms if latency_processor else None,
            "inter_token_latency": _latency_stats(latency_processor.inter_token_ms) if latency_processor else None,
            "page_fault_groups": None,
            "available_ram_before_bytes": available_ram,
            "environment": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
        }
        return result
    except PagerCancelled as exc:
        bridge = hook_runtime["bridge"] if hook_runtime else None
        return {
            "schema_version": 1,
            "ticket": active_ticket,
            "status": "cancelled",
            "completed": 0,
            "cancelled": 1,
            "resource_rejected": 0,
            "mode": args.mode,
            "context": args.context,
            "batch_size": args.batch_size,
            "cancel_after_layer": args.cancel_after_layer,
            "reason": str(exc),
            "page_in_count": hook_runtime["state"]["page_in"] if hook_runtime else 0,
            "page_out_count": hook_runtime["state"]["page_out"] if hook_runtime else 0,
            "h2d_bytes": bridge.h2d_bytes if bridge else 0,
            "d2h_bytes": bridge.d2h_bytes if bridge else 0,
            "bridge_chunk_count": bridge.chunk_count if bridge else 0,
            "bridge_peak_module_bytes": bridge.peak_module_bytes if bridge else 0,
            "prefetch_distance": hook_runtime["prefetch_distance"] if hook_runtime else 0,
            "prefetch_stage_bytes": bridge.prefetch_stage_bytes if bridge else 0,
            "prefetch_stage_chunks": bridge.prefetch_stage_chunks if bridge else 0,
        }
    finally:
        if bridge is not None:
            bridge.close()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MF-MEM-N2R fixed 7B NF4 isolated comparison")
    parser.add_argument("--mode", choices=("resident-nf4", "device-map-auto", "cpu-only", "explicit-layer-pager", "explicit-layer-pager-64"), required=True)
    parser.add_argument("--context", type=int, choices=(128, 2048, 8192), default=128)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--max-new-tokens", type=int, choices=(1, 4, 8, 32, 128), default=8)
    parser.add_argument("--cancel-after-layer", type=int, default=0)
    parser.add_argument("--prefetch-distance", type=int, choices=(0, 1), default=0)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--ticket", default=TARGET_TICKET)
    parser.add_argument("--result-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = args.result_file.resolve()
    try:
        result = run_case(args)
    except ResourceRejected as exc:
        return _rejected(args.mode, str(exc), result_path=result_path, ticket=args.ticket)
    except (RuntimeError, ValueError) as exc:
        _write_json(result_path, {
            "schema_version": 1,
            "ticket": args.ticket,
            "status": "failed",
            "completed": 0,
            "resource_rejected": 0,
            "mode": args.mode,
            "error": str(exc)[:4096],
        })
        return 2
    _write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
