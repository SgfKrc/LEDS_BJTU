"""Isolated Qwen3 pipeline resource gate and assignment smoke worker.

The default operation is metadata-only.  It reads the local Safetensors
index and tensor headers, computes the exact selected tensor budget, and
checks the sidecar runtime and node-local memory before any weight is
materialized.  A real assignment load is only allowed with ``execute=true``
and still goes through the filtered-index loader.
"""

from __future__ import annotations

import importlib
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

try:
    from .qwen3_pipeline_adapter import (
        QWEN3_ADAPTER_SCHEMA_VERSION,
        Qwen3AdapterError,
        render_without_thinking,
        select_qwen3_assignment_keys,
        validate_qwen3_assignment,
    )
    from .qwen3_pipeline_chain import (
        build_hidden_handoff,
        build_kv_contract,
        validate_kv_contract,
        validate_segment_plan,
    )
except ImportError:  # direct sidecar script execution
    from qwen3_pipeline_adapter import (  # type: ignore
        QWEN3_ADAPTER_SCHEMA_VERSION,
        Qwen3AdapterError,
        render_without_thinking,
        select_qwen3_assignment_keys,
        validate_qwen3_assignment,
    )
    from qwen3_pipeline_chain import (  # type: ignore
        build_hidden_handoff,
        build_kv_contract,
        validate_kv_contract,
        validate_segment_plan,
    )


TOOL = "qwen3_pipeline_smoke"
OPERATION = "qwen3_pipeline_smoke"
MAX_INPUT_BYTES = 256 * 1024
MIN_TRANSFORMERS = (4, 51, 0)
DEFAULT_SAFETY_MARGIN = 1.2
DEFAULT_RESERVE_BYTES = 512 * 1024**2
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "f16": 2,
    "int32": 4,
    "float32": 4,
    "f32": 4,
    "float64": 8,
    "int64": 8,
}


def _version_tuple(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(value).split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        values.append(int(digits))
    return tuple(values or [0])


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3AdapterError(f"invalid {path.name}") from exc
    if not isinstance(value, dict):
        raise Qwen3AdapterError(f"{path.name} must contain a JSON object")
    return value


def _available_ram_bytes() -> int:
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except Exception:
        if os.name == "nt":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                            ("available_extended", ctypes.c_ulonglong)]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, int(status.available))
        try:
            return max(0, int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")))
        except (AttributeError, OSError, ValueError):
            return 0


def _gpu_memory(torch_module: Any) -> tuple[int, int]:
    try:
        cuda = getattr(torch_module, "cuda")
        if not bool(cuda.is_available()):
            return 0, 0
        free, total = cuda.mem_get_info()
        return max(0, int(free)), max(0, int(total))
    except Exception:
        return 0, 0


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return max(0, int(psutil.Process(os.getpid()).memory_info().rss))
    except Exception:
        return None


def _cache_sequence_length(cache: Any, expected: int | None = None) -> int | None:
    if cache is None:
        return None
    try:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
    except Exception:
        pass
    candidates: list[int] = []

    def visit(value: Any) -> None:
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                candidates.extend(int(item) for item in shape if 0 < int(item) <= 40960)
            except (TypeError, ValueError):
                pass
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(cache)
    if expected is not None:
        exact = [value for value in candidates if value == int(expected)]
        if exact:
            return int(expected)
        nearby = [value for value in candidates if value >= int(expected)]
        if nearby:
            return min(nearby)
    return max(candidates) if candidates else None


def _tensor_bytes(dtype: str, shape: Any) -> int:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise Qwen3AdapterError("Safetensors tensor shape is invalid")
    try:
        count = math.prod(int(value) for value in shape)
    except (TypeError, ValueError):
        raise Qwen3AdapterError("Safetensors tensor shape is invalid") from None
    if count <= 0:
        raise Qwen3AdapterError("Safetensors tensor shape is empty")
    unit = _DTYPE_BYTES.get(str(dtype).lower())
    if unit is None:
        raise Qwen3AdapterError(f"unsupported Safetensors dtype: {dtype}")
    return count * unit


def _assignment_budget(root: Path, selected_keys: list[str], weight_map: dict[str, str]) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise Qwen3AdapterError("safetensors is required for Qwen3 assignment inspection") from exc
    by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        filename = str(weight_map.get(key, "")).replace("\\", "/")
        if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise Qwen3AdapterError("Qwen3 assignment shard path is unsafe")
        by_shard.setdefault(filename, []).append(key)
    selected_bytes = 0
    tensor_count = 0
    shard_bytes = 0
    for filename, keys in by_shard.items():
        shard = (root / filename).resolve(strict=False)
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise Qwen3AdapterError("Qwen3 assignment shard escapes model root") from exc
        if not shard.is_file():
            raise Qwen3AdapterError(f"Qwen3 assignment shard is missing: {filename}")
        shard_bytes += int(shard.stat().st_size)
        # ``framework=np`` keeps the header-only budget gate usable before
        # Torch is installed in the isolated execution environment.
        with safe_open(str(shard), framework="np", device="cpu") as handle:
            for key in keys:
                if key not in handle.keys():
                    raise Qwen3AdapterError(f"Qwen3 assignment shard is missing key: {key}")
                selected_bytes += _tensor_bytes(handle.get_slice(key).get_dtype(), handle.get_slice(key).get_shape())
                tensor_count += 1
    return {
        "selected_tensor_count": tensor_count,
        "selected_tensor_bytes": selected_bytes,
        "assignment_shard_bytes": shard_bytes,
        "shard_count": len(by_shard),
    }


def _prepare_filtered_assignment(
    root: Path,
    selected_keys: list[str],
    weight_map: dict[str, str],
) -> Path:
    """Create a temporary C3-shaped assignment without touching the source.

    Shards are hard-linked when possible, so a smoke does not duplicate
    multi-gigabyte files.  A cross-volume link falls back to a copy and is
    removed with the staging directory after the worker exits.
    """
    stage = Path(tempfile.mkdtemp(prefix=".qlh-qwen3-assignment-", dir=str(root.parent)))
    try:
        _populate_filtered_assignment(stage, root, selected_keys, weight_map)
        return stage
    except BaseException:
        # 中途异常（如 shard 缺失/越界）也必须清理 staging——历史上泄漏的
        # 残留（2026-08-17 清理 3.9G）正是这里未兜底造成的。
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _populate_filtered_assignment(
    stage: Path,
    root: Path,
    selected_keys: list[str],
    weight_map: dict[str, str],
) -> None:
    """填充 staging 内容（从 _prepare_filtered_assignment 拆出，保证自清理兜底）。"""
    try:
        shutil.copy2(root / "config.json", stage / "config.json")
        # Tokenizer/chat-template support is small and must remain available
        # inside the filtered assignment; weight shards stay hard-link-first.
        for support_name in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "chat_template.jinja", "vocab.json",
            "merges.txt", "spiece.model", "tokenizer.model",
        ):
            support = root / support_name
            if support.is_file():
                shutil.copy2(support, stage / support_name)
        filtered_map: dict[str, str] = {}
        for key in selected_keys:
            filename = str(weight_map.get(key, "")).replace("\\", "/")
            relative = Path(filename)
            if not filename or relative.is_absolute() or ".." in relative.parts:
                raise Qwen3AdapterError("Qwen3 assignment shard path is unsafe")
            source = (root / relative).resolve(strict=False)
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise Qwen3AdapterError("Qwen3 assignment shard escapes model root") from exc
            if not source.is_file():
                raise Qwen3AdapterError(f"Qwen3 assignment shard is missing: {filename}")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(str(source), str(target))
                except OSError:
                    shutil.copy2(source, target)
            filtered_map[key] = relative.as_posix()
        (stage / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": filtered_map, "metadata": {"total_size": 0}}, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _cleanup_stale_assignments(root: Path) -> None:
    """清理模型根下历史遗留的 assignment staging（异常退出残留）。

    worker 单实例串行（stdin 同步协议），启动清理不会误删正在使用的目录；
    只删 `qlh-qwen3-assignment-` 前缀目录（worker 自己的 mkdtemp 命名）。
    """
    parent = root.parent
    if not parent.is_dir():
        return
    removed = 0
    for entry in parent.iterdir():
        if (entry.name.startswith(".qlh-qwen3-assignment-")
                and entry.is_dir()):
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    if removed:
        print(f"qwen3 pipeline smoke: 清理 {removed} 个遗留 assignment staging",
              file=sys.stderr)


def _base_result(request: dict[str, Any]) -> dict[str, Any]:
    return {        "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
        "tool": TOOL,
        "operation": OPERATION,
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "preflight_failed",
        "runtime": {
            "isolated": False,
            "transformers_version": None,
            "torch_version": None,
            "accelerate_available": False,
            "safetensors_available": False,
            "minimum_transformers": ".".join(map(str, MIN_TRANSFORMERS)),
        },
        "assignment": {
            "layer_range": request.get("layer_range"),
            "selected_tensor_count": 0,
            "selected_tensor_bytes": 0,
            "assignment_shard_bytes": 0,
            "shard_count": 0,
        },
        "resources": {
            "device": request.get("device", "auto"),
            "available_ram_bytes": None,
            "available_vram_bytes": None,
            "required_ram_bytes": None,
            "required_device_bytes": None,
            "safety_margin": request.get("safety_margin", DEFAULT_SAFETY_MARGIN),
            "reserve_bytes": request.get("reserve_bytes", DEFAULT_RESERVE_BYTES),
            "passed": False,
        },
        "tokenizer": {
            "checked": False,
            "thinking_disabled": False,
            "rendered_chars": None,
            "input_token_count": None,
        },
        "execution": {"attempted": False, "synthetic_forward": False, "error": None},
        "errors": [],
    }


def _error(result: dict[str, Any], code: str, message: str, *, status: str | None = None) -> dict[str, Any]:
    result["errors"].append({"code": code, "message": str(message)[:2048]})
    if status:
        result["status"] = status
    return result


def _resource_values(request: dict[str, Any], key: str, fallback: int) -> int:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise Qwen3AdapterError(f"{key} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3AdapterError(f"{key} must be a non-negative integer") from exc
    if parsed < 0:
        raise Qwen3AdapterError(f"{key} must be a non-negative integer")
    return parsed


def _target_multiplier(device: str, dtype: Any) -> float:
    normalized = str(dtype or "").lower()
    if normalized in {"fp16", "float16", "bf16", "bfloat16"}:
        return 1.0
    return 1.0 if str(device).startswith("cuda") else 2.0


def _chain_result(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
        "tool": "qwen3_pipeline_chain_smoke",
        "operation": "qwen3_pipeline_chain_smoke",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "preflight_failed",
        "chain_id": str(request.get("chain_id", "qwen3-local-smoke")),
        "segments": [],
        "tokenizer": {"checked": False, "thinking_disabled": False, "input_token_count": None},
        "execution": {
            "attempted": False,
            "prefill_complete": False,
            "decode_complete": False,
            "hidden_handoffs": [],
            "kv_contracts": {"prefill": [], "decode": []},
            "loads": [],
            "full_model_materialized": False,
            "cleanup_complete": False,
            "error": None,
        },
        "errors": [],
    }


def _chain_error(
    result: dict[str, Any], code: str, message: str, *, status: str,
) -> dict[str, Any]:
    result["gate_passed"] = False
    result["status"] = status
    result["errors"].append({"code": code, "message": str(message)[:2048]})
    return result


def execute_chain_request(
    request: dict[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Run a bounded two/three-segment Qwen3 chain in one sidecar process.

    Segments are loaded serially for prefill, released, and loaded serially
    again for decode.  Hidden states and segment-owned KV are the only live
    execution state across a load boundary, so the worker never needs to
    materialize the complete model or all assignments at once.
    """
    result = _chain_result(request)
    active_stage: Path | None = None
    adapter: Any = None
    try:
        if request.get("schema_version") != QWEN3_ADAPTER_SCHEMA_VERSION:
            return _chain_error(result, "invalid_request", "unsupported Qwen3 chain protocol", status="invalid_request")
        if request.get("read_only") is not True or request.get("network_access") != "disabled":
            return _chain_error(result, "invalid_request", "Qwen3 chain smoke must be read-only and network-disabled", status="invalid_request")
        root = Path(str(request.get("model_path", ""))).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            return _chain_error(result, "model_path_invalid", "Qwen3 model directory is missing", status="artifact_rejected")
        config = _load_object(root / "config.json")
        total_layers = int(config.get("num_hidden_layers", 0) or 0)
        segments = validate_segment_plan(request.get("segments", []), total_layers=total_layers)
        devices = {str(item.get("device", "auto")) for item in segments}
        dtypes = {str(item.get("dtype") or "") for item in segments}
        if len(devices) != 1 or len(dtypes) != 1:
            raise Qwen3AdapterError("local Qwen3 chain requires one device and dtype across hidden handoffs")

        # Preflight every segment before materializing any weight.  This gives
        # the chain all-or-nothing admission while retaining per-segment
        # tensor and capacity evidence.
        for segment in segments:
            segment_request = {
                "schema_version": QWEN3_ADAPTER_SCHEMA_VERSION,
                "operation": OPERATION,
                "read_only": True,
                "network_access": "disabled",
                "model_path": str(root),
                "layer_range": segment["layer_range"],
                "has_embedding": segment["has_embedding"],
                "has_lm_head": segment["has_lm_head"],
                "execute": False,
                "device": segment["device"],
                "dtype": segment["dtype"],
                "safety_margin": request.get("safety_margin", DEFAULT_SAFETY_MARGIN),
                "reserve_bytes": request.get("reserve_bytes", DEFAULT_RESERVE_BYTES),
                "controller_python": request.get("controller_python"),
            }
            for key in ("available_ram_bytes", "available_vram_bytes"):
                if key in request:
                    segment_request[key] = request[key]
            preflight = execute_request(segment_request, module_loader=module_loader)
            result["segments"].append({
                "segment_index": segment["segment_index"],
                "layer_range": segment["layer_range"],
                "has_embedding": segment["has_embedding"],
                "has_lm_head": segment["has_lm_head"],
                "device": preflight.get("resources", {}).get("device"),
                "dtype": segment["dtype"],
                "status": preflight.get("status"),
                "gate_passed": preflight.get("gate_passed", False),
                "assignment": preflight.get("assignment", {}),
                "resources": preflight.get("resources", {}),
            })
            if not preflight.get("gate_passed", False):
                first = (preflight.get("errors") or [{"code": "segment_rejected", "message": "Qwen3 segment preflight failed"}])[0]
                return _chain_error(
                    result,
                    str(first.get("code", "segment_rejected")),
                    str(first.get("message", "Qwen3 segment preflight failed")),
                    status=str(preflight.get("status", "segment_rejected")),
                )
            # Freeze the resource decision before execution.  Passing
            # ``auto`` into the architecture loader would make device choice
            # implicit and could invalidate the preflight budget.
            segment["device"] = str(preflight.get("resources", {}).get("device", segment["device"]))
        result["gate_passed"] = True
        result["status"] = "ready_for_qwen3_pipeline_chain_smoke"
        if not bool(request.get("execute", False)):
            return result

        result["execution"]["attempted"] = True
        from transformers import AutoTokenizer
        import torch
        try:
            from .qwen3_pipeline_adapter import load_qwen3_layer_assignment
        except ImportError:  # direct sidecar script execution
            from qwen3_pipeline_adapter import load_qwen3_layer_assignment  # type: ignore

        index = _load_object(root / "model.safetensors.index.json")
        weight_map = {str(key): str(value) for key, value in index.get("weight_map", {}).items()}
        tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        rendered = render_without_thinking(tokenizer, [{"role": "user", "content": "Reply with the word OK."}])
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2 or int(input_ids.shape[1]) <= 0:
            raise Qwen3AdapterError("Qwen3 tokenizer produced invalid chain input")
        result["tokenizer"].update({
            "checked": True,
            "thinking_disabled": True,
            "input_token_count": int(input_ids.shape[1]),
        })
        chain_id = result["chain_id"]
        segment_caches: list[Any] = []

        def release_loaded() -> None:
            nonlocal adapter, active_stage
            adapter = None
            if active_stage is not None:
                shutil.rmtree(active_stage, ignore_errors=True)
                active_stage = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def load_segment(segment: dict[str, Any], phase: str) -> tuple[Any, dict[str, Any]]:
            nonlocal adapter, active_stage
            selected_keys = select_qwen3_assignment_keys(
                weight_map.keys(),
                start_layer=segment["layer_range"][0],
                end_layer=segment["layer_range"][1],
                has_embedding=segment["has_embedding"],
                has_lm_head=segment["has_lm_head"],
                tie_word_embeddings=tie_word_embeddings,
            )
            active_stage = _prepare_filtered_assignment(root, selected_keys, weight_map)
            rss_before = _process_rss_bytes()
            if str(segment["device"]).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            adapter, metrics = load_qwen3_layer_assignment(
                active_stage,
                start_layer=segment["layer_range"][0],
                end_layer=segment["layer_range"][1],
                has_embedding=segment["has_embedding"],
                has_lm_head=segment["has_lm_head"],
                device=segment["device"],
                dtype=segment["dtype"],
            )
            metrics.update({
                "segment_index": segment["segment_index"],
                "phase": phase,
                "rss_before_bytes": rss_before,
                "full_model_materialized": False,
            })
            return adapter, metrics

        hidden = None
        logits = None
        for index, segment in enumerate(segments):
            current, metrics = load_segment(segment, "prefill")
            if index == 0:
                forward = current.forward(input_ids=input_ids, use_cache=True)
            else:
                if hidden is None:
                    raise Qwen3AdapterError("Qwen3 prefill hidden handoff is missing")
                expected_device, _ = current._device_dtype(current.model)
                if expected_device is not None and hidden.device != expected_device:
                    raise Qwen3AdapterError("Qwen3 hidden handoff device does not match next segment")
                forward = current.forward(hidden_states=hidden, use_cache=True)
            hidden = forward.get("hidden_states")
            logits = forward.get("logits")
            cache = forward.get("past_key_values")
            expected_length = int(input_ids.shape[1])
            if cache is None or _cache_sequence_length(cache, expected=expected_length) != expected_length:
                raise Qwen3AdapterError("Qwen3 prefill KV cache length mismatch")
            segment_caches.append(cache)
            output = hidden if hidden is not None else logits
            result["execution"]["kv_contracts"]["prefill"].append(build_kv_contract(
                chain_id=chain_id,
                segment_index=index,
                layer_range=segment["layer_range"],
                sequence_length=expected_length,
                batch_size=int(input_ids.shape[0]),
                dtype=str(output.dtype),
                device=str(output.device),
                phase="prefill",
                generation=0,
            ))
            if index < len(segments) - 1:
                handoff = build_hidden_handoff(
                    hidden,
                    chain_id=chain_id,
                    from_segment=index,
                    to_segment=index + 1,
                )
                result["execution"]["hidden_handoffs"].append(handoff)
            metrics["rss_after_forward_bytes"] = _process_rss_bytes()
            if str(segment["device"]).startswith("cuda"):
                metrics["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                metrics["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            result["execution"]["loads"].append(metrics)
            release_loaded()
        if logits is None:
            raise Qwen3AdapterError("final Qwen3 segment did not return logits")
        result["execution"]["prefill_complete"] = True
        result["execution"]["prefill_logits_shape"] = list(logits.shape)

        hidden = None
        logits = None
        decode_ids = input_ids[:, -1:]
        for index, (segment, cache) in enumerate(zip(segments, segment_caches)):
            prefill_contract = result["execution"]["kv_contracts"]["prefill"][index]
            validate_kv_contract(
                prefill_contract,
                chain_id=chain_id,
                segment_index=index,
                layer_range=segment["layer_range"],
                sequence_length=int(input_ids.shape[1]),
                batch_size=int(input_ids.shape[0]),
                dtype=prefill_contract["dtype"],
                device=prefill_contract["device"],
                phase="prefill",
                generation=0,
            )
            current, metrics = load_segment(segment, "decode")
            if index == 0:
                forward = current.forward(input_ids=decode_ids, past_key_values=cache, use_cache=True)
            else:
                if hidden is None:
                    raise Qwen3AdapterError("Qwen3 decode hidden handoff is missing")
                expected_device, _ = current._device_dtype(current.model)
                if expected_device is not None and hidden.device != expected_device:
                    raise Qwen3AdapterError("Qwen3 decode handoff device does not match next segment")
                forward = current.forward(hidden_states=hidden, past_key_values=cache, use_cache=True)
            hidden = forward.get("hidden_states")
            logits = forward.get("logits")
            cache = forward.get("past_key_values")
            expected_length = int(input_ids.shape[1]) + 1
            if cache is None or _cache_sequence_length(cache, expected=expected_length) != expected_length:
                raise Qwen3AdapterError("Qwen3 decode KV cache length mismatch")
            output = hidden if hidden is not None else logits
            result["execution"]["kv_contracts"]["decode"].append(build_kv_contract(
                chain_id=chain_id,
                segment_index=index,
                layer_range=segment["layer_range"],
                sequence_length=expected_length,
                batch_size=int(decode_ids.shape[0]),
                dtype=str(output.dtype),
                device=str(output.device),
                phase="decode",
                generation=1,
            ))
            metrics["rss_after_forward_bytes"] = _process_rss_bytes()
            if str(segment["device"]).startswith("cuda"):
                metrics["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                metrics["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            result["execution"]["loads"].append(metrics)
            release_loaded()
        if logits is None:
            raise Qwen3AdapterError("final Qwen3 decode segment did not return logits")
        result["execution"]["decode_complete"] = True
        result["execution"]["decode_logits_shape"] = list(logits.shape)
        result["execution"]["cleanup_complete"] = True
        result["status"] = "qwen3_pipeline_chain_smoke_passed"
        return result
    except (ImportError, ModuleNotFoundError) as exc:
        result["execution"]["error"] = exc.__class__.__name__
        return _chain_error(result, "sidecar_dependency_missing", "Qwen3 sidecar dependency is unavailable", status="runtime_unavailable")
    except (TypeError, ValueError, OSError, Qwen3AdapterError) as exc:
        result["execution"]["error"] = exc.__class__.__name__
        return _chain_error(result, "chain_execution_failed" if result["execution"]["attempted"] else "chain_preflight_failed", str(exc), status="execution_failed" if result["execution"]["attempted"] else "preflight_failed")
    except Exception as exc:
        result["execution"]["error"] = exc.__class__.__name__
        return _chain_error(result, "chain_execution_failed", exc.__class__.__name__, status="execution_failed")
    finally:
        adapter = None
        if active_stage is not None:
            shutil.rmtree(active_stage, ignore_errors=True)
        gc.collect()


def execute_request(
    request: dict[str, Any],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    result = _base_result(request)
    try:
        if request.get("operation") == "qwen3_pipeline_chain_smoke":
            return execute_chain_request(request, module_loader=module_loader)
        if request.get("schema_version") != QWEN3_ADAPTER_SCHEMA_VERSION or request.get("operation") != OPERATION:
            return _error(result, "invalid_request", "unsupported Qwen3 pipeline smoke protocol", status="invalid_request")
        if request.get("read_only") is not True or request.get("network_access") != "disabled":
            return _error(result, "invalid_request", "Qwen3 pipeline smoke must be read-only and network-disabled", status="invalid_request")
        root = Path(str(request.get("model_path", ""))).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            return _error(result, "model_path_invalid", "Qwen3 assignment directory is missing", status="artifact_rejected")
        # 启动时清理历史遗留 staging（worker 单实例串行；防止异常退出残留累积）
        _cleanup_stale_assignments(root)
        config = _load_object(root / "config.json")
        model_type = str(config.get("model_type", "") or "").lower()
        total_layers = int(config.get("num_hidden_layers", 0) or 0)
        layer_range = request.get("layer_range")
        if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
            return _error(result, "layer_range_invalid", "layer_range must contain [start, end]", status="invalid_request")
        start_layer, end_layer = int(layer_range[0]), int(layer_range[1])
        has_embedding = bool(request.get("has_embedding", False))
        has_lm_head = bool(request.get("has_lm_head", False))
        tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        index = _load_object(root / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise Qwen3AdapterError("Qwen3 Safetensors index has no weight_map")
        selected_keys = select_qwen3_assignment_keys(
            weight_map.keys(), start_layer=start_layer, end_layer=end_layer,
            has_embedding=has_embedding, has_lm_head=has_lm_head,
            tie_word_embeddings=tie_word_embeddings,
        )
        assignment = validate_qwen3_assignment(
            model_type=model_type, total_layers=total_layers,
            start_layer=start_layer, end_layer=end_layer,
            has_embedding=has_embedding, has_lm_head=has_lm_head,
            keys=selected_keys, tie_word_embeddings=tie_word_embeddings,
        )
        result["assignment"].update(_assignment_budget(root, selected_keys, {str(k): str(v) for k, v in weight_map.items()}))
        result["assignment"].update({"layer_range": [start_layer, end_layer], "selected_key_count": len(assignment["selected_keys"]), "tie_word_embeddings": tie_word_embeddings})

        # A CPU assignment can be rejected from metadata and host memory
        # alone.  This keeps a clearly impossible request from importing any
        # execution dependency, and is important on small sidecar hosts.
        requested_device = str(request.get("device", "auto") or "auto").lower()
        if requested_device == "cpu":
            early_ram = _resource_values(request, "available_ram_bytes", _available_ram_bytes())
            early_margin = float(request.get("safety_margin", DEFAULT_SAFETY_MARGIN))
            early_reserve = _resource_values(request, "reserve_bytes", DEFAULT_RESERVE_BYTES)
            if not math.isfinite(early_margin) or early_margin < 1.0:
                raise Qwen3AdapterError("safety_margin must be at least 1.0")
            early_required = math.ceil(int(result["assignment"]["selected_tensor_bytes"]) * 2.0 * early_margin) + early_reserve
            result["resources"].update({
                "device": "cpu", "available_ram_bytes": early_ram,
                "available_vram_bytes": 0, "required_ram_bytes": early_required,
                "required_device_bytes": early_required, "safety_margin": early_margin,
                "reserve_bytes": early_reserve,
            })
            if early_ram < early_required:
                return _error(result, "insufficient_assignment_capacity", "selected Qwen3 assignment exceeds node-local free memory", status="resource_rejected")

        missing_dependencies: list[str] = []
        try:
            transformers = module_loader("transformers")
            result["runtime"]["transformers_version"] = str(getattr(transformers, "__version__", "0.0.0"))
        except (ImportError, ModuleNotFoundError):
            transformers = None
            missing_dependencies.append("transformers")
        try:
            torch_module = module_loader("torch")
            result["runtime"]["torch_version"] = str(getattr(torch_module, "__version__", "unknown"))
        except (ImportError, ModuleNotFoundError):
            torch_module = None
            missing_dependencies.append("torch")
        try:
            module_loader("accelerate")
            result["runtime"]["accelerate_available"] = True
        except (ImportError, ModuleNotFoundError):
            missing_dependencies.append("accelerate")
        result["runtime"]["safetensors_available"] = True
        result["runtime"]["isolated"] = bool(request.get("controller_python")) and Path(sys.executable).resolve() != Path(str(request["controller_python"])).expanduser().absolute().resolve(strict=False)
        if not result["runtime"]["isolated"]:
            return _error(result, "runtime_not_isolated", "Qwen3 worker must use a dedicated Python environment", status="runtime_rejected")
        if transformers is not None and _version_tuple(result["runtime"]["transformers_version"]) < MIN_TRANSFORMERS:
            return _error(result, "transformers_too_old", "isolated sidecar requires transformers >= 4.51.0", status="runtime_rejected")

        free_vram, _total_vram = _gpu_memory(torch_module) if torch_module is not None else (0, 0)
        available_ram = _resource_values(request, "available_ram_bytes", _available_ram_bytes())
        available_vram = _resource_values(request, "available_vram_bytes", free_vram)
        margin = float(request.get("safety_margin", DEFAULT_SAFETY_MARGIN))
        reserve = _resource_values(request, "reserve_bytes", DEFAULT_RESERVE_BYTES)
        if not math.isfinite(margin) or margin < 1.0:
            raise Qwen3AdapterError("safety_margin must be at least 1.0")
        requested_device = str(request.get("device", "auto") or "auto").lower()
        cuda_available = bool(torch_module is not None and getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
        device = "cuda" if requested_device == "auto" and cuda_available else ("cpu" if requested_device == "auto" else requested_device)
        if device.startswith("cuda") and not cuda_available:
            return _error(result, "cuda_unavailable", "CUDA assignment requested but sidecar CUDA is unavailable", status="resource_rejected")
        source_bytes = int(result["assignment"]["selected_tensor_bytes"])
        target_multiplier = 1.0 if device.startswith("cuda") else 2.0
        required_device = math.ceil(source_bytes * target_multiplier * margin) + reserve
        required_ram = math.ceil(source_bytes * (1.0 if device.startswith("cuda") else 2.0) * margin) + reserve
        result["resources"].update({
            "device": device, "available_ram_bytes": available_ram, "available_vram_bytes": available_vram,
            "required_ram_bytes": required_ram, "required_device_bytes": required_device,
            "safety_margin": margin, "reserve_bytes": reserve,
        })
        available_device = available_vram if device.startswith("cuda") else available_ram
        if available_ram < required_ram or available_device < required_device:
            return _error(result, "insufficient_assignment_capacity", "selected Qwen3 assignment exceeds node-local free memory", status="resource_rejected")
        result["resources"]["passed"] = True
        if missing_dependencies:
            return _error(result, "sidecar_dependency_missing", "Qwen3 sidecar dependencies are unavailable: " + ", ".join(missing_dependencies), status="runtime_unavailable")
        result["gate_passed"] = True
        result["status"] = "ready_for_qwen3_pipeline_smoke"
        if not bool(request.get("execute", False)):
            return result

        result["execution"]["attempted"] = True
        stage: Path | None = None
        try:
            from .qwen3_pipeline_adapter import load_qwen3_layer_assignment
        except ImportError:  # direct sidecar script execution
            from qwen3_pipeline_adapter import load_qwen3_layer_assignment  # type: ignore
        # Exercise the actual local Qwen3 chat template before loading the
        # segment.  Only aggregate shape information is returned.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
        )
        rendered = render_without_thinking(
            tokenizer,
            [{"role": "user", "content": "Reply with the word OK."}],
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2 or int(input_ids.shape[1]) <= 0:
            raise Qwen3AdapterError("Qwen3 tokenizer produced invalid smoke input")
        result["tokenizer"].update({
            "checked": True,
            "thinking_disabled": True,
            "rendered_chars": len(rendered),
            "input_token_count": int(input_ids.shape[1]),
        })
        rss_before = _process_rss_bytes()
        if device.startswith("cuda"):
            try:
                torch_module.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        stage = _prepare_filtered_assignment(root, selected_keys, {str(k): str(v) for k, v in weight_map.items()})
        adapter, metrics = load_qwen3_layer_assignment(
            stage, start_layer=start_layer, end_layer=end_layer,
            has_embedding=has_embedding, has_lm_head=has_lm_head,
            device=device, dtype=request.get("dtype"),
        )
        kv_smoke = bool(request.get("kv_smoke", True)) and bool(has_embedding)
        cache_report: dict[str, Any] = {
            "enabled": kv_smoke,
            "prefill_token_count": int(input_ids.shape[1]) if has_embedding else None,
            "decode_token_count": 0,
            "cache_present": False,
            "cache_sequence_length": None,
        }
        if has_embedding:
            import torch

            if kv_smoke:
                prefill = adapter.forward(input_ids=input_ids.to(device), use_cache=True)
                past_key_values = prefill.get("past_key_values")
                forward = adapter.forward(
                    input_ids=input_ids[:, -1:].to(device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                decoded_cache = forward.get("past_key_values")
                cache_report.update({
                    "decode_token_count": 1,
                    "cache_present": decoded_cache is not None,
                    "cache_sequence_length": _cache_sequence_length(
                        decoded_cache,
                        expected=int(input_ids.shape[1]) + 1,
                    ),
                })
            else:
                forward = adapter.forward(input_ids=input_ids.to(device), use_cache=False)
        else:
            import torch

            hidden_size = int(config.get("hidden_size", 1) or 1)
            forward = adapter.forward(hidden_states=torch.zeros((1, 3, hidden_size), device=device), use_cache=False)
        rss_after = _process_rss_bytes()
        metrics.update({
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": (rss_after - rss_before) if rss_before is not None and rss_after is not None else None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        })
        if device.startswith("cuda"):
            try:
                metrics["cuda_peak_allocated_bytes"] = int(torch_module.cuda.max_memory_allocated())
                metrics["cuda_peak_reserved_bytes"] = int(torch_module.cuda.max_memory_reserved())
            except Exception:
                pass
        result["execution"].update({"synthetic_forward": True, "metrics": metrics, "kv_cache": cache_report, "output_kind": "logits" if "logits" in forward else "hidden_states"})
    except (ImportError, ModuleNotFoundError) as exc:
        return _error(result, "sidecar_dependency_missing", f"Qwen3 sidecar dependency is unavailable: {exc.name or exc.__class__.__name__}", status="runtime_unavailable")
    except (TypeError, ValueError, OSError, Qwen3AdapterError) as exc:
        if result.get("execution", {}).get("attempted"):
            result["gate_passed"] = False
            result["execution"]["error"] = exc.__class__.__name__
            return _error(result, "execution_failed", str(exc), status="execution_failed")
        return _error(result, "preflight_failed", str(exc), status="preflight_failed")
    except Exception as exc:
        if result.get("execution", {}).get("attempted"):
            result["gate_passed"] = False
            result["execution"]["error"] = exc.__class__.__name__
            return _error(result, "execution_failed", exc.__class__.__name__, status="execution_failed")
        return _error(result, "preflight_failed", exc.__class__.__name__, status="preflight_failed")
    finally:
        if 'stage' in locals() and stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    return result


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Qwen3 pipeline smoke request exceeds protocol limit")
    try:
        request = json.loads(raw.decode("utf-8"))
        result = execute_request(request)
    except Exception as exc:
        result = _base_result({})
        result["valid"] = False
        result["status"] = "invalid_request"
        result["errors"] = [{"code": "invalid_request", "message": exc.__class__.__name__}]
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result.get("valid", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
