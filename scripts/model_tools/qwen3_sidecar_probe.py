"""Controller for the isolated Qwen3 Transformers preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TOOL = "qwen3_sidecar_probe"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 600.0


def _request_error(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_sidecar_preflight",
        "valid": False,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "invalid_request",
        "errors": [{"code": code, "message": message}],
    }


def _worker_failure(code: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_sidecar_preflight",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "worker_failed",
        "errors": [{"code": code, "message": "isolated Qwen3 sidecar did not return a valid result"}],
    }


def _sidecar_python() -> Path | None:
    override = os.environ.get("QLH_QWEN3_SIDECAR_PYTHON", "").strip()
    if override:
        candidate = Path(override).expanduser().absolute().resolve(strict=False)
    elif os.name == "nt":
        candidate = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / ".venv-qwen3-sidecar" / "bin" / "python"
    return candidate if candidate.is_file() else None


def _worker_command() -> list[str] | None:
    python = _sidecar_python()
    if python is None:
        return None
    return [str(python), str(Path(__file__).with_name("qwen3_sidecar_probe_worker.py"))]


def _run_worker(request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    command = _worker_command()
    if command is None:
        return _worker_failure("sidecar_runtime_missing")
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "NO_PROXY": "*",
    })
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    try:
        completed = subprocess.run(
            command, input=json.dumps(request, ensure_ascii=True, separators=(",", ":")),
            text=True, capture_output=True, cwd=str(ROOT), env=env,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return _worker_failure("timeout")
    except OSError:
        return _worker_failure("worker_start_failed")
    for line in reversed(completed.stdout.splitlines()):
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and report.get("schema_version") == SCHEMA_VERSION and report.get("tool") == TOOL:
            return report
    return _worker_failure("invalid_worker_output")


def run_qwen3_sidecar_probe(
    *,
    model: Path | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model is None:
        return _request_error("model_path_required", "Qwen3 sidecar preflight requires --model")
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return _request_error("timeout_invalid", f"timeout_seconds must be between 0 and {MAX_TIMEOUT_SECONDS:g}")
    model_path = model.expanduser().absolute().resolve(strict=False)
    request = {
        "schema_version": SCHEMA_VERSION,
        "operation": "qwen3_sidecar_preflight",
        "tool": TOOL,
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model_path),
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
    }
    try:
        report = (worker_runner or _run_worker)(request, float(timeout_seconds))
    except Exception:
        return _worker_failure("worker_runner_failed")
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _worker_failure("invalid_worker_output")
    return report


__all__ = ["run_qwen3_sidecar_probe"]
