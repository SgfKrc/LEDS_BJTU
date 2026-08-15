"""Controller for the isolated Qwen3-VL vision tower weight probe (MM1.15)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TOOL = "qwen3_multimodal_vision_tower_probe"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_TIMEOUT_SECONDS = 900.0


def _request_error(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_tower_weight_smoke",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "invalid_request",
        "errors": [{"code": code, "message": message}],
    }


def _sidecar_python() -> Path | None:
    candidate = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else None


def run_qwen3_multimodal_vision_tower_probe(
    *,
    model: Path | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Load the real vision tower weights in the isolated sidecar (MM1.15)."""
    if model is None:
        return _request_error("model_path_required", "MM1.15 vision tower probe requires --model")
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return _request_error("timeout_invalid", f"timeout_seconds must be between 0 and {MAX_TIMEOUT_SECONDS:g}")
    sidecar = _sidecar_python()
    if sidecar is None:
        return _request_error("sidecar_missing", "MM1.15 requires the isolated Qwen3 pipeline sidecar")
    model_path = model.expanduser().absolute().resolve(strict=False)
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_visual_tower_weight_smoke",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model_path),
    }
    worker = Path(__file__).with_name("qwen3_multimodal_vision_tower_probe_worker.py")
    try:
        proc = subprocess.run(
            [str(sidecar), str(worker)],
            input=json.dumps(request, separators=(",", ":")),
            capture_output=True, text=True, encoding="utf-8",
            timeout=float(timeout_seconds), cwd=str(ROOT),
        )
    except Exception:
        return _request_error("worker_failed", "vision tower worker did not return")
    try:
        report = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return _request_error("worker_output_invalid", "vision tower worker output is invalid")
    if not isinstance(report, dict) or report.get("tool") != TOOL or report.get("schema_version") != SCHEMA_VERSION:
        return _request_error("worker_report_invalid", "vision tower worker report is invalid")
    encoded = json.dumps(report, ensure_ascii=True, separators=(",", ":")).lower()
    for path_value in (str(model_path), str(ROOT)):
        if path_value.lower() in encoded:
            return _request_error("worker_report_sensitive", "vision tower worker report contains a path")
    if report.get("status") == "vision_tower_weights_loaded":
        response = report.get("response")
        if (
            not isinstance(response, dict)
            or response.get("weight_materialized") is not True
            or response.get("text_weights_loaded") is not False
            or response.get("full_model_materialized") is not False
        ):
            return _request_error("worker_report_inconsistent", "vision tower worker report is inconsistent")
    return report


__all__ = ["run_qwen3_multimodal_vision_tower_probe"]
