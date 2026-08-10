"""Bounded, read-only LLM smoke matrix orchestration."""

from __future__ import annotations

import json
import os
import psutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

PROMPT_SET_ID = "qlh-llm-smoke-v1"
MAX_PROMPTS = 4
MAX_MODELS = 32
MAX_NEW_TOKENS = 64
DEFAULT_TIMEOUT_SECONDS = 180.0


def fixed_prompts() -> list[dict[str, Any]]:
    context = " ".join(f"item-{index:02d}=ignore" for index in range(1, 41))
    return [
        {"id": "zh_basic", "text": "请用中文回答：北京交通大学的办学特色是什么？只回答一句话。"},
        {"id": "en_basic", "text": "Answer in one short English sentence: what is a model smoke test?"},
        {"id": "json_object", "text": "Your entire response must be exactly this JSON object, with no Markdown or explanation: {\"ok\":true,\"kind\":\"smoke\"}"},
        {"id": "context_marker", "text": f"Read this bounded context: {context}. The final marker is QLH-SMOKE-MARKER-7F3A. Return only the final marker."},
    ]


def _formats_for_model(config: Any) -> list[str]:
    if config.model_type == "both":
        return ["gguf", "safetensors"]
    return [config.model_type]


def discover_units(model_ids: list[str] | None = None, formats: list[str] | None = None) -> list[dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import model_config as mc

    requested = set(model_ids or [])
    requested_formats = set(formats or [])
    units: list[dict[str, Any]] = []
    for config in mc.get_builtin_models():
        if requested and config.model_id not in requested:
            continue
        for model_format in _formats_for_model(config):
            if requested_formats and model_format not in requested_formats:
                continue
            if model_format == "gguf":
                path = Path(mc.resolve_model_path(config.gguf_path))
                engine = "llama_cpp"
            else:
                path = Path(mc.resolve_model_path(config.model_path))
                engine = "pytorch"
            available = path.is_file() if model_format == "gguf" else (path.is_dir() and mc.has_safetensors_files(config))
            units.append({
                "model_id": config.model_id,
                "name": config.name,
                "format": model_format,
                "engine": engine,
                "path": str(path),
                "available": available,
                "recommended_vram_gb": config.recommended_vram_gb,
                "asset_size_bytes": path.stat().st_size if available and model_format == "gguf" else (
                    sum(item.stat().st_size for item in path.iterdir() if item.suffix.lower() in {".safetensors", ".bin"}) if available else 0
                ),
            })
    if requested:
        known = {config.model_id for config in mc.get_builtin_models()}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown model_id: {', '.join(unknown)}")
    return units


def _worker_command() -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("llm_smoke_worker.py"))]


def _run_worker(unit: dict[str, Any], prompts: list[dict[str, Any]], *, quant: str, max_new_tokens: int, timeout_seconds: float, allow_cpu: bool) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "operation": "llm_smoke_worker",
        "model_id": unit["model_id"],
        "format": unit["format"],
        "engine": unit["engine"],
        "model_path": unit["path"],
        "quant": quant,
        "max_new_tokens": max_new_tokens,
        "cuda_available": True,
        "prompts": prompts,
    }
    with tempfile.TemporaryDirectory(prefix="qlh-llm-smoke-") as cache_dir:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HOME"] = cache_dir
        env["HF_MODULES_CACHE"] = str(Path(cache_dir) / "modules")
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(cache_dir) / "hub")
        try:
            completed = subprocess.run(
                _worker_command(),
                input=json.dumps(request, ensure_ascii=True),
                text=True,
                capture_output=True,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": {"code": "timeout", "message": "worker timed out"}, "jobs": []}
    parsed: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("operation") == "llm_smoke_worker":
            parsed = candidate
            break
    if parsed is None:
        return {"status": "failed", "error": {"code": "invalid_worker_output", "message": "worker did not return a valid result"}, "jobs": []}
    return parsed


def _resource_rejection(unit: dict[str, Any], *, allow_cpu: bool) -> str | None:
    asset_size = int(unit.get("asset_size_bytes", 0))
    available_ram = int(psutil.virtual_memory().available)
    if unit["format"] == "gguf":
        required_ram = int(asset_size * 1.2) + 512 * 1024**2
        return "insufficient_ram" if available_ram < required_ram else None
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            free_vram, total_vram = torch.cuda.mem_get_info()
            recommended = int(float(unit.get("recommended_vram_gb", 0)) * 1024**3)
            required_free = min(recommended, int(asset_size * 0.55) + 512 * 1024**2)
            if int(total_vram) < int(recommended * 0.9) or int(free_vram) < required_free:
                return "insufficient_vram"
            return None
    except Exception:
        cuda_available = False
    if not cuda_available and not allow_cpu:
        return "cuda_required"
    required_ram = int(asset_size * 2.2) + 1024**3
    return "insufficient_ram" if available_ram < required_ram else None


def run_smoke_matrix(
    *,
    model_ids: list[str] | None = None,
    formats: list[str] | None = None,
    max_models: int = MAX_MODELS,
    max_new_tokens: int = 32,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    quant: str = "int4",
    allow_cpu: bool = False,
    require_complete: bool = False,
    worker_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if max_models < 1 or max_models > MAX_MODELS:
        raise ValueError(f"max_models must be between 1 and {MAX_MODELS}")
    if max_new_tokens < 1 or max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 0 and 3600")
    if quant not in {"fp16", "int8", "int4"}:
        raise ValueError("quant must be fp16, int8 or int4")
    prompts = fixed_prompts()
    units = discover_units(model_ids, formats)
    discovered_count = len(units)
    if len(units) > max_models:
        units = units[:max_models]
    results: list[dict[str, Any]] = []
    for unit in units:
        result_base = {key: unit[key] for key in ("model_id", "name", "format", "engine", "recommended_vram_gb")}
        if not unit["available"]:
            results.append({**result_base, "status": "skipped", "reason": "asset_missing", "jobs": [], "error": None})
            continue
        resource_rejection = _resource_rejection(unit, allow_cpu=allow_cpu)
        if resource_rejection:
            results.append({**result_base, "status": "skipped", "reason": resource_rejection, "jobs": [], "error": None})
            continue
        started = time.perf_counter()
        runner = worker_runner or _run_worker
        try:
            worker_result = runner(unit, prompts, quant=quant, max_new_tokens=max_new_tokens, timeout_seconds=timeout_seconds, allow_cpu=allow_cpu)
        except Exception as exc:
            message = str(exc).replace(str(unit["path"]), "<model-path>")
            worker_result = {"status": "failed", "jobs": [], "error": {"code": "runner_error", "message": message[:2048]}}
        results.append({
            **result_base,
            "status": "passed" if worker_result.get("status") == "passed" else "failed",
            "jobs": worker_result.get("jobs", []),
            "load_ms": worker_result.get("load_ms"),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "error": worker_result.get("error"),
        })
    executed = [item for item in results if item["status"] in {"passed", "failed"}]
    passed_jobs = sum(1 for item in executed for job in item.get("jobs", []) if job.get("status") == "passed")
    failed_jobs = sum(1 for item in executed for job in item.get("jobs", []) if job.get("status") == "failed")
    execution_gate_passed = bool(executed) and not failed_jobs and all(item["status"] == "passed" for item in executed)
    skipped_count = sum(item["status"] == "skipped" for item in results)
    coverage_complete = bool(results) and not skipped_count and discovered_count == len(results)
    gate_passed = execution_gate_passed and (coverage_complete or not require_complete)
    return {
        "schema_version": 1,
        "tool": "llm_smoke_matrix",
        "operation": "matrix",
        "valid": True,
        "read_only": True,
        "prompt_set_id": PROMPT_SET_ID,
        "prompts": [{"id": item["id"], "text": item["text"]} for item in prompts],
        "limits": {"max_models": max_models, "max_new_tokens": max_new_tokens, "timeout_seconds": timeout_seconds, "quant": quant, "allow_cpu": allow_cpu, "require_complete": require_complete},
        "models": results,
        "summary": {
            "units_discovered": discovered_count,
            "units_total": len(results),
            "selection_truncated": discovered_count > len(results),
            "units_executed": len(executed),
            "units_passed": sum(item["status"] == "passed" for item in executed),
            "units_failed": sum(item["status"] == "failed" for item in executed),
            "units_skipped": skipped_count,
            "jobs_passed": passed_jobs,
            "jobs_failed": failed_jobs,
            "execution_gate_passed": execution_gate_passed,
            "coverage_complete": coverage_complete,
            "gate_passed": gate_passed,
        },
        "errors": (
            ["coverage is incomplete"] if require_complete and execution_gate_passed and not coverage_complete
            else ([] if execution_gate_passed else (["no runnable model units"] if not executed else []))
        ),
    }


__all__ = ["fixed_prompts", "discover_units", "run_smoke_matrix"]
