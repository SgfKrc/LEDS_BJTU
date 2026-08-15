"""Controller for the isolated Qwen3 multimodal AutoProcessor smoke."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen3_multimodal_preflight import (  # noqa: E402
    inspect_mm1_processor_assets,
    validate_mm1_visual_worker_request,
)


TOOL = "qwen3_multimodal_processor_probe"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 180.0
MAX_TIMEOUT_SECONDS = 600.0
MAX_REQUEST_BYTES = 256 * 1024


def _base_result(*, status: str, gate_passed: bool, errors: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_worker_processor_smoke",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": gate_passed,
        "status": status,
        "errors": errors or [],
    }


def _request_error(code: str, message: str) -> dict[str, Any]:
    return _base_result(
        status="invalid_request",
        gate_passed=False,
        errors=[{"code": code, "message": message}],
    )


def _worker_failure(code: str) -> dict[str, Any]:
    return _base_result(
        status="worker_failed",
        gate_passed=False,
        errors=[{
            "code": code,
            "message": "isolated Qwen3 multimodal processor worker did not return a valid result",
        }],
    )


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
    return [str(python), str(Path(__file__).with_name("qwen3_multimodal_processor_probe_worker.py"))]


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
    encoded = json.dumps(request, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_REQUEST_BYTES:
        return _worker_failure("request_too_large")
    try:
        completed = subprocess.run(
            command,
            input=encoded,
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=env,
            timeout=timeout_seconds,
            check=False,
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


def run_qwen3_multimodal_processor_probe(
    *,
    model: Path | None,
    manifest: Mapping[str, Any] | None,
    visual_request: Mapping[str, Any] | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
    media_smoke: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct AutoProcessor in the isolated sidecar after MM1.5 validation."""
    if model is None:
        return _request_error("model_path_required", "MM1.6 processor smoke requires --model")
    if not isinstance(manifest, Mapping):
        return _request_error("manifest_required", "MM1.6 processor smoke requires an MM1 manifest")
    if not isinstance(visual_request, Mapping):
        return _request_error("visual_request_required", "MM1.6 processor smoke requires a validated visual request")
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return _request_error("timeout_invalid", f"timeout_seconds must be between 0 and {MAX_TIMEOUT_SECONDS:g}")
    model_path = model.expanduser().absolute().resolve(strict=False)
    try:
        inspection = inspect_mm1_processor_assets(model_path, manifest)
        safe_request = validate_mm1_visual_worker_request(
            visual_request, manifest=manifest, inspection=inspection,
        )
    except Exception as exc:
        return _request_error("mm1_preflight_rejected", exc.__class__.__name__)
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_worker_processor_smoke",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model_path),
        "manifest": dict(manifest),
        "visual_request": safe_request,
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
        "media_smoke": dict(media_smoke) if media_smoke else None,
    }
    try:
        report = (worker_runner or _run_worker)(request, float(timeout_seconds))
    except Exception:
        return _worker_failure("worker_runner_failed")
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _worker_failure("invalid_worker_output")
    return report


__all__ = ["run_qwen3_multimodal_processor_probe"]
