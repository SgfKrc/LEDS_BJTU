"""Isolated worker for one MODEL-TOOLS LLM smoke unit."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
WORKER_OPERATION = "llm_smoke_worker"
PROMPT_IDS = {"zh_basic", "en_basic", "json_object", "context_marker"}


def _redact_absolute_paths(message: str) -> str:
    redacted = re.sub(r"(?i)[A-Z]:[\\/][^\r\n\t\"']+", "<path>", str(message))
    return re.sub(r"(?<![A-Za-z0-9.])/(?:[^/\s\"']+/)+[^\s\"']+", "<path>", redacted)


def _error(message: str, code: str = "worker_error") -> dict[str, str]:
    return {"code": code, "message": _redact_absolute_paths(message)[:2048]}


def _safe_error(exc: Exception, model_path: Path, code: str) -> dict[str, str]:
    message = str(exc)
    for value in {str(model_path), str(model_path.resolve(strict=False))}:
        if value:
            message = message.replace(value, "<model-path>")
    return _error(message, code)


def validate_output(prompt_id: str, content: str) -> dict[str, Any]:
    """Apply deterministic gates to generated text without storing the text."""
    text = str(content or "").strip()
    result: dict[str, Any] = {
        "non_empty": bool(text),
        "language_valid": None,
        "json_valid": None,
        "context_marker_found": None,
        "passed": bool(text),
    }
    if prompt_id == "zh_basic":
        result["language_valid"] = any("\u4e00" <= char <= "\u9fff" for char in text)
        result["passed"] = bool(text) and result["language_valid"] is True
    elif prompt_id == "en_basic":
        ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
        result["language_valid"] = ascii_letters >= 8
        result["passed"] = bool(text) and result["language_valid"] is True
    elif prompt_id == "json_object":
        try:
            parsed = json.loads(text)
            result["json_valid"] = parsed == {"ok": True, "kind": "smoke"}
        except (TypeError, ValueError, json.JSONDecodeError):
            result["json_valid"] = False
        result["passed"] = bool(text) and result["json_valid"] is True
    elif prompt_id == "context_marker":
        result["context_marker_found"] = "QLH-SMOKE-MARKER-7F3A" in text
        result["passed"] = bool(text) and result["context_marker_found"] is True
    return result


def _job_result(prompt: dict[str, Any], *, content: str | None = None, error: dict[str, str] | None = None, elapsed_ms: int | None = None) -> dict[str, Any]:
    output = str(content or "")
    validation = validate_output(str(prompt.get("id", "unknown")), output) if error is None else {
        "non_empty": False,
        "json_valid": None,
        "context_marker_found": None,
        "passed": False,
    }
    return {
        "prompt_id": str(prompt.get("id", "unknown")),
        "status": "passed" if error is None and validation["passed"] else "failed",
        "elapsed_ms": elapsed_ms,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if output else None,
        "validation": validation,
        "error": error,
    }


def execute_request(request: dict[str, Any], manager_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Execute one model unit; prompt failures are isolated from one another."""
    if request.get("schema_version") != SCHEMA_VERSION or request.get("operation") != WORKER_OPERATION:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("unsupported worker protocol", "invalid_request")}
    model_path = Path(str(request.get("model_path", "")))
    if not model_path.is_absolute() or not model_path.exists():
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("model_path must be an existing absolute path", "invalid_request")}
    model_id = request.get("model_id")
    if not isinstance(model_id, str) or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model_id) is None:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("invalid model_id", "invalid_request")}
    prompts = request.get("prompts")
    if not isinstance(prompts, list) or not prompts or len(prompts) > 8:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("prompts must contain 1-8 items", "invalid_request")}
    model_format = request.get("format")
    engine = request.get("engine")
    if model_format not in {"gguf", "safetensors"} or engine not in {"llama_cpp", "pytorch"}:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("unsupported model format or engine", "invalid_request")}
    if (model_format == "gguf") != (engine == "llama_cpp"):
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("model format and engine do not match", "invalid_request")}
    max_new_tokens = request.get("max_new_tokens", 16)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("max_new_tokens must be between 1 and 64", "invalid_request")}
    if request.get("quant", "int4") not in {"fp16", "int8", "int4"}:
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("unsupported quant", "invalid_request")}
    prompt_ids: list[str] = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or prompt.get("id") not in PROMPT_IDS or not isinstance(prompt.get("text"), str) or not 1 <= len(prompt["text"]) <= 4096:
            return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("invalid prompt item", "invalid_request")}
        prompt_ids.append(prompt["id"])
    if len(set(prompt_ids)) != len(prompt_ids):
        return {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("duplicate prompt_id", "invalid_request")}
    manager = None
    jobs: list[dict[str, Any]] = []
    loaded = False
    load_started = time.perf_counter()
    load_ms: int | None = None
    try:
        manager = manager_factory() if manager_factory is not None else None
        if manager is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
            from model_module import ModelManager
            manager = ModelManager()
        manager.load_model(
            model_path=str(model_path),
            model_id=str(request.get("model_id", "")),
            engine=str(request.get("engine", "llama_cpp")),
            quant_type=str(request.get("quant", "int4")),
            profile={"tier": "laptop", "gpu": {"cuda_available": bool(request.get("cuda_available", True))}},
        )
        loaded = True
        load_ms = int((time.perf_counter() - load_started) * 1000)
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "operation": WORKER_OPERATION,
            "valid": True,
            "status": "failed",
            "model_id": str(request.get("model_id", "unknown")),
            "format": str(request.get("format", "unknown")),
            "engine": str(request.get("engine", "unknown")),
            "load_ms": load_ms,
            "jobs": [],
            "error": _safe_error(exc, model_path, "load_failed"),
        }
    try:
        for prompt in prompts:
            started = time.perf_counter()
            try:
                response = manager.chat(
                    [
                        {"role": "system", "content": "Follow the requested language and output format exactly."},
                        {"role": "user", "content": str(prompt.get("text", ""))},
                    ],
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                content = response.get("content", "") if isinstance(response, dict) else str(response)
                jobs.append(_job_result(prompt, content=content, elapsed_ms=int((time.perf_counter() - started) * 1000)))
            except Exception as exc:
                jobs.append(_job_result(prompt, error=_safe_error(exc, model_path, "generation_failed"), elapsed_ms=int((time.perf_counter() - started) * 1000)))
    finally:
        if loaded:
            try:
                manager.unload_model()
            except Exception:
                pass
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": WORKER_OPERATION,
        "valid": True,
        "status": "passed" if jobs and all(job["status"] == "passed" for job in jobs) else "failed",
        "model_id": str(request.get("model_id", "unknown")),
        "format": str(request.get("format", "unknown")),
        "engine": str(request.get("engine", "unknown")),
        "load_ms": load_ms,
        "jobs": jobs,
        "error": None,
    }


def main() -> int:
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        result = {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "invalid_request", "error": _error("request exceeds 1 MiB", "invalid_request")}
    else:
        try:
            result = execute_request(json.loads(raw.decode("utf-8")))
        except Exception as exc:
            result = {"schema_version": SCHEMA_VERSION, "operation": WORKER_OPERATION, "valid": False, "status": "worker_error", "error": _error(str(exc))}
    sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
