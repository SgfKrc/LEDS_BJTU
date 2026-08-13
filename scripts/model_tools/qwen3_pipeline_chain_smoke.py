"""Controller for bounded two/three-segment Qwen3 chain smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TOOL = "qwen3_pipeline_chain_smoke"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 1800.0


def _base(valid: bool, status: str, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": TOOL,
        "valid": valid,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": status,
        "errors": errors or [],
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


def _run_worker(request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    python = _sidecar_python()
    if python is None:
        return _base(True, "worker_failed", [{"code": "sidecar_runtime_missing", "message": "isolated Qwen3 sidecar Python is missing"}])
    command = [str(python), str(Path(__file__).with_name("qwen3_pipeline_smoke_worker.py"))]
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "NO_PROXY": "*"})
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _base(True, "worker_failed", [{"code": "timeout", "message": "isolated Qwen3 chain worker timed out"}])
    except OSError:
        return _base(True, "worker_failed", [{"code": "worker_start_failed", "message": "isolated Qwen3 chain worker could not start"}])
    for line in reversed(completed.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("tool") == TOOL and value.get("schema_version") == SCHEMA_VERSION:
            return value
    return _base(True, "worker_failed", [{"code": "invalid_worker_output", "message": "isolated Qwen3 chain worker returned no valid result"}])


def run_qwen3_pipeline_chain_smoke(
    *,
    model: Path | None,
    segments: list[dict[str, Any]],
    chain_id: str = "qwen3-local-smoke",
    execute: bool = False,
    safety_margin: float = 1.2,
    reserve_bytes: int = 512 * 1024**2,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if model is None:
        return _base(False, "invalid_request", [{"code": "model_path_required", "message": "Qwen3 chain smoke requires --model"}])
    if not isinstance(segments, list) or len(segments) not in {2, 3}:
        return _base(False, "invalid_request", [{"code": "segments_invalid", "message": "Qwen3 chain requires two or three segments"}])
    if any(not isinstance(segment, dict) for segment in segments):
        return _base(False, "invalid_request", [{"code": "segments_invalid", "message": "Qwen3 chain segments must be objects"}])
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return _base(False, "invalid_request", [{"code": "timeout_invalid", "message": "timeout_seconds is outside the allowed range"}])
    root = model.expanduser().absolute().resolve(strict=False)
    request = {
        "schema_version": SCHEMA_VERSION,
        "operation": TOOL,
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(root),
        "segments": segments,
        "chain_id": str(chain_id),
        "execute": bool(execute),
        "safety_margin": float(safety_margin),
        "reserve_bytes": int(reserve_bytes),
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
    }
    try:
        report = (worker_runner or _run_worker)(request, float(timeout_seconds))
    except Exception:
        return _base(True, "worker_failed", [{"code": "worker_runner_failed", "message": "Qwen3 chain worker runner failed"}])
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _base(True, "worker_failed", [{"code": "invalid_worker_output", "message": "Qwen3 chain worker result is invalid"}])
    return report


__all__ = ["run_qwen3_pipeline_chain_smoke"]
