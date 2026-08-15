"""Controller for the isolated Qwen3-VL real vision semantics smoke (MM1.18)."""

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


TOOL = "qwen3_multimodal_vision_text_smoke"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_TIMEOUT_SECONDS = 1800.0


def _request_error(code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_vision_text_real_semantics",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "invalid_request",
        "errors": [{"code": code, "message": message}],
    }


def run_qwen3_multimodal_vision_text_smoke(
    *,
    model: Path | None,
    image: Path | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    text_chain_id: str = "a" * 64,
    generation: int = 4,
) -> dict[str, Any]:
    """Load the real model (4-bit) and run a fixed-image semantics smoke."""
    if model is None or image is None:
        return _request_error("request_incomplete", "MM1.18 requires --model and --image")
    sidecar = ROOT / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    if not sidecar.is_file():
        return _request_error("sidecar_missing", "MM1.18 requires the isolated Qwen3 pipeline sidecar")
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_vision_text_real_semantics",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model.expanduser().absolute().resolve(strict=False)),
        "image_path": str(image.expanduser().absolute().resolve(strict=False)),
        "text_chain_id": text_chain_id,
        "generation": generation,
    }
    worker = Path(__file__).with_name("qwen3_multimodal_vision_text_smoke_worker.py")
    try:
        proc = subprocess.run(
            [str(sidecar), str(worker)],
            input=json.dumps(request, separators=(",", ":")),
            capture_output=True, text=True, encoding="utf-8",
            timeout=float(timeout_seconds), cwd=str(ROOT),
        )
    except Exception:
        return _request_error("worker_failed", "vision text smoke worker did not return")
    try:
        report = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return _request_error("worker_output_invalid", "vision text smoke worker output is invalid")
    return report


__all__ = ["run_qwen3_multimodal_vision_text_smoke"]
