"""Read-only, fail-closed preflight for native Gemma 4 multimodal experiments."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TOOL = "gemma4_native_probe"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 180.0
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_N_CTX = 512
MAX_N_CTX = 4096
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIB = 1024**3


def _pinned_llama_cpp_revision() -> str | None:
    try:
        from .llama_quantize_toolchain import load_lock

        return str(load_lock()["upstream"]["revision"])
    except Exception:
        return None


def _request_error(message: str, *, code: str = "invalid_request") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "native_multimodal_preflight",
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
        "operation": "native_multimodal_preflight",
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": False,
        "status": "worker_failed",
        "errors": [{"code": code, "message": "isolated native probe did not return a valid result"}],
    }


def _worker_command() -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("gemma4_native_probe_worker.py"))]


def _run_worker(request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            _worker_command(),
            input=json.dumps(request, ensure_ascii=True, separators=(",", ":")),
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _worker_failure("timeout")
    except OSError:
        return _worker_failure("worker_start_failed")

    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("schema_version") == SCHEMA_VERSION
            and candidate.get("tool") == TOOL
        ):
            return candidate
    return _worker_failure("invalid_worker_output")


def _normalize_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.expanduser().absolute().resolve(strict=False)


def _validate_sha256(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")


def _required_free_ram_bytes(model: Path | None, mmproj: Path | None) -> int:
    if model is None or mmproj is None or not model.is_file() or not mmproj.is_file():
        return 0
    # mmap does not remove the need for working-set and OS headroom on Windows.
    paired_bytes = model.stat().st_size + mmproj.stat().st_size
    return max(2 * GIB, int(paired_bytes * 1.10) + GIB)


def run_native_probe(
    *,
    model: Path | None = None,
    mmproj: Path | None = None,
    model_sha256: str = "",
    mmproj_sha256: str = "",
    n_ctx: int = DEFAULT_N_CTX,
    require_audio: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a no-network binding check or an explicitly requested artifact probe.

    Supplying any artifact-related option selects the artifact mode. In that
    mode both local files and their expected SHA-256 digests are mandatory, so
    an arbitrary GGUF/projector pair cannot become a product capability merely
    because it happened to initialize on one machine.
    """
    if not isinstance(n_ctx, int) or not 128 <= n_ctx <= MAX_N_CTX:
        return _request_error(
            f"n_ctx must be between 128 and {MAX_N_CTX}",
            code="n_ctx_invalid",
        )
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        return _request_error(
            f"timeout_seconds must be between 0 and {MAX_TIMEOUT_SECONDS:g}",
            code="timeout_invalid",
        )

    model = _normalize_path(model)
    mmproj = _normalize_path(mmproj)
    artifact_requested = bool(model or mmproj or model_sha256 or mmproj_sha256)
    if artifact_requested:
        if model is None or mmproj is None or not model_sha256 or not mmproj_sha256:
            return _request_error(
                "artifact probe requires --model, --mmproj, --model-sha256 and --mmproj-sha256",
                code="artifact_identity_incomplete",
            )
        try:
            _validate_sha256("model_sha256", model_sha256)
            _validate_sha256("mmproj_sha256", mmproj_sha256)
        except ValueError as exc:
            return _request_error(str(exc), code="artifact_sha256_invalid")

    request = {
        "schema_version": SCHEMA_VERSION,
        "operation": "native_multimodal_preflight",
        "tool": TOOL,
        "read_only": True,
        "network_access": "disabled",
        "project_llama_cpp_revision": _pinned_llama_cpp_revision(),
        "artifact_requested": artifact_requested,
        "model_path": str(model) if model is not None else "",
        "mmproj_path": str(mmproj) if mmproj is not None else "",
        "model_sha256": model_sha256.lower(),
        "mmproj_sha256": mmproj_sha256.lower(),
        "required_free_ram_bytes": _required_free_ram_bytes(model, mmproj) if artifact_requested else 0,
        "n_ctx": n_ctx,
        "require_audio": bool(require_audio),
    }
    runner = worker_runner or _run_worker
    try:
        report = runner(request, float(timeout_seconds))
    except Exception:
        return _worker_failure("worker_runner_failed")
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _worker_failure("invalid_worker_output")
    return report


__all__ = ["DEFAULT_N_CTX", "run_native_probe"]
