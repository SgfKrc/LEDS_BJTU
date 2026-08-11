"""Isolated model trial-loader used by the MODEL-FLEET control plane.

The process accepts one JSON request on stdin and emits one JSON result line on
stdout. Model libraries may write diagnostics to stderr. Network access is not
used: all loaders are configured for local files and trust_remote_code=False.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


GIB = 1024**3
GGUF_FILE_TYPES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 7: "q8_0", 8: "q5_0",
    9: "q5_1", 10: "q2_k", 11: "q3_k_s", 12: "q3_k_m", 13: "q3_k_l",
    14: "q4_k_s", 15: "q4_k_m", 16: "q5_k_s", 17: "q5_k_m", 18: "q6_k",
    19: "iq2_xxs", 20: "iq2_xs", 21: "q2_k_s", 22: "iq3_xs",
    23: "iq3_xxs", 24: "iq1_s", 25: "iq4_nl", 26: "iq3_s", 27: "iq3_m",
    28: "iq2_s", 29: "iq2_m", 30: "iq4_xs", 31: "iq1_m", 32: "bf16",
    36: "tq1_0", 37: "tq2_0", 38: "mxfp4_moe", 39: "nvfp4", 40: "q1_0",
}


def _available_ram_bytes() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return 0
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return page_size * available_pages


def _gpu_memory() -> tuple[int, int, bool]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, 0, False
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes), int(total_bytes), True
    except Exception:
        return 0, 0, False


def _result(
    request: dict[str, Any],
    status: str,
    *,
    loader_version: str | None = None,
    load_ms: int | None = None,
    details: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_id": str(request.get("request_id", "unknown")),
        "artifact_id": str(request.get("artifact_id", "unknown")),
        "status": status,
        "engine": str(request.get("engine", "unknown")),
        "runtime_profile": str(request.get("runtime_profile", "unknown")),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loader_version": loader_version,
        "load_ms": load_ms,
        "details": details or {},
        "error": error,
    }


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("schema_version") != 1 or request.get("operation") != "trial_load":
        raise ValueError("unsupported sidecar protocol")
    if request.get("trust_remote_code") is not False:
        raise ValueError("trust_remote_code must be false")
    model_path = Path(str(request.get("model_path", "")))
    if not model_path.is_absolute() or not model_path.exists():
        raise ValueError("model_path must be an existing absolute path")
    if request.get("format") not in {"gguf", "safetensors"}:
        raise ValueError("unsupported model format")
    if not isinstance(request.get("files"), list) or not request["files"]:
        raise ValueError("files are required")


def _verify_file(file_path: Path, expected_size: int, expected_digest: str) -> None:
    if not file_path.is_file() or file_path.stat().st_size != expected_size:
        raise ValueError(f"artifact size mismatch: {file_path.name}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise ValueError(f"artifact digest mismatch: {file_path.name}")


def _verify_artifact(request: dict[str, Any]) -> int:
    started = time.perf_counter()
    model_path = Path(request["model_path"]).resolve()
    for item in request["files"]:
        rel_path = Path(str(item["path"]))
        if request["format"] == "gguf":
            target = model_path
        else:
            target = (model_path / rel_path).resolve()
            try:
                target.relative_to(model_path)
            except ValueError as exc:
                raise ValueError("artifact path escapes model directory") from exc
        _verify_file(target, int(item["size"]), str(item["sha256"]))
    return int((time.perf_counter() - started) * 1000)


def _load_gguf(request: dict[str, Any]) -> dict[str, Any]:
    import llama_cpp
    from llama_cpp import Llama

    verify_ms = _verify_artifact(request)
    options = request.get("options") or {}
    started = time.perf_counter()
    model = Llama(
        model_path=request["model_path"],
        n_ctx=int(options.get("n_ctx", 128)),
        n_batch=min(64, int(options.get("n_ctx", 128))),
        n_threads=int(options.get("n_threads", 4)),
        n_gpu_layers=int(options.get("n_gpu_layers", 0)),
        use_mmap=True,
        verbose=False,
    )
    load_ms = int((time.perf_counter() - started) * 1000)
    try:
        metadata = model.metadata or {}
        tokens = model.tokenize(b"runtime sidecar", add_bos=False)
        file_type = int(metadata.get("general.file_type", -1))
        details = {
            "format": "gguf",
            "architecture": metadata.get("general.architecture"),
            "model_name": metadata.get("general.name"),
            "file_type": file_type,
            "quantization": GGUF_FILE_TYPES.get(file_type),
            "context_length": metadata.get("qwen2.context_length")
            or metadata.get("llama.context_length"),
            "tokenizer_model": metadata.get("tokenizer.ggml.model"),
            "token_probe_count": len(tokens),
            "verify_ms": verify_ms,
            "n_ctx": int(options.get("n_ctx", 128)),
            "n_gpu_layers": int(options.get("n_gpu_layers", 0)),
        }
        return _result(
            request,
            "ready",
            loader_version=f"llama-cpp-python/{llama_cpp.__version__}",
            load_ms=load_ms,
            details=details,
        )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
        del model
        gc.collect()


def _load_safetensors(request: dict[str, Any]) -> dict[str, Any]:
    weight_bytes = sum(
        int(item["size"])
        for item in request["files"]
        if str(item["path"]).lower().endswith(".safetensors")
    )
    ram_free = _available_ram_bytes()
    gpu_free, gpu_total, cuda_available = _gpu_memory()
    required_bytes = int(weight_bytes * 1.12) + 512 * 1024**2
    usable_bytes = max(0, ram_free - GIB) + max(0, gpu_free - GIB)
    resource_details = {
        "format": "safetensors",
        "weight_bytes": weight_bytes,
        "required_bytes": required_bytes,
        "available_ram_bytes": ram_free,
        "available_vram_bytes": gpu_free,
        "total_vram_bytes": gpu_total,
        "cuda_available": cuda_available,
        "usable_combined_bytes": usable_bytes,
        "trust_remote_code": False,
    }
    if usable_bytes < required_bytes:
        return _result(
            request,
            "resource_rejected",
            details=resource_details,
            error={
                "code": "insufficient_memory",
                "message": "insufficient RAM/VRAM headroom for isolated Safetensors trial load",
            },
        )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    verify_ms = _verify_artifact(request)
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": "auto",
    }
    if cuda_available and importlib.util.find_spec("accelerate") is not None:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {
            0: max(1, int((gpu_free - GIB) / GIB)) * GIB,
            "cpu": max(1, int((ram_free - GIB) / GIB)) * GIB,
        }
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(request["model_path"], **kwargs)
    load_ms = int((time.perf_counter() - started) * 1000)
    try:
        first_parameter = next(model.parameters())
        details = {
            **resource_details,
            "architecture": getattr(model.config, "model_type", None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "first_parameter_device": str(first_parameter.device),
            "first_parameter_dtype": str(first_parameter.dtype),
            "verify_ms": verify_ms,
        }
        return _result(
            request,
            "ready",
            loader_version=f"transformers/{transformers.__version__};torch/{torch.__version__}",
            load_ms=load_ms,
            details=details,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("sidecar request exceeds 1 MiB")
    request = json.loads(raw.decode("utf-8"))
    _validate_request(request)
    try:
        if request["format"] == "gguf":
            result = _load_gguf(request)
        else:
            result = _load_safetensors(request)
    except MemoryError as error:
        result = _result(
            request,
            "resource_rejected",
            error={"code": "memory_error", "message": str(error) or "model load ran out of memory"},
        )
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = _result(
            request,
            "load_failed",
            error={"code": "loader_error", "message": str(error)[:4096]},
        )
    sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(2) from exc
