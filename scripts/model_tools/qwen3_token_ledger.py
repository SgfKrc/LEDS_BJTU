"""Controller for isolated local Qwen3 token-ledger decoding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
TOOL = "qwen3_token_ledger"
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 600.0
MAX_TEXT_BYTES = 64 * 1024


def _base_result(
    status: str, *, gate_passed: bool = False, code: str | None = None,
    operation: str = "qwen3_token_ledger_decode",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": operation,
        "valid": True,
        "read_only": True,
        "network_access": "disabled",
        "gate_passed": gate_passed,
        "status": status,
        "errors": ([{"code": code, "message": "isolated tokenizer worker did not complete"}] if code else []),
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


def _worker_failure(
    code: str, *, operation: str = "qwen3_token_ledger_decode",
) -> dict[str, Any]:
    return _base_result("worker_failed", code=code, operation=operation)


def _run_worker(request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    operation = str(request.get("operation") or "qwen3_token_ledger_decode")
    python = _sidecar_python()
    if python is None:
        return _worker_failure("sidecar_runtime_missing", operation=operation)
    worker = Path(__file__).with_name("qwen3_token_ledger_worker.py")
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
    if len(encoded.encode("utf-8")) > 256 * 1024:
        return _worker_failure("request_too_large", operation=operation)
    try:
        completed = subprocess.run(
            [str(python), str(worker)],
            input=encoded,
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=env,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _worker_failure("timeout", operation=operation)
    except OSError:
        return _worker_failure("worker_start_failed", operation=operation)
    for line in reversed(completed.stdout.splitlines()):
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and report.get("schema_version") == SCHEMA_VERSION and report.get("tool") == TOOL:
            return report
    return _worker_failure("invalid_worker_output", operation=operation)


def _metadata(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    ledger_id = str(value.get("ledger_id") or "")
    digest = str(value.get("sha256") or "").lower()
    size = value.get("size_bytes")
    token_count = value.get("token_count")
    sampling_sha256 = value.get("sampling_sha256")
    draw_count = value.get("draw_count")
    quality_sha256 = value.get("quality_sha256")
    policy_snapshot_sha256 = value.get("policy_snapshot_sha256")
    policy_id = value.get("policy_id")
    policy_version = value.get("policy_version")
    if (
        not ledger_id
        or len(ledger_id) > 128
        or any(char in ledger_id for char in "/\\\x00")
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or value.get("status") != "committed"
        or value.get("content_kind") != "generated_token_ledger"
        or isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or not 1 <= token_count <= 4096
        or value.get("stop_reason") not in {"eos", "max_new_tokens"}
        or (
            sampling_sha256 is not None
            and (
                not isinstance(sampling_sha256, str)
                or len(sampling_sha256) != 64
                or any(char not in "0123456789abcdef" for char in sampling_sha256)
            )
        )
        or (
            draw_count is not None
            and (
                isinstance(draw_count, bool)
                or not isinstance(draw_count, int)
                or draw_count != token_count
            )
        )
        or ((sampling_sha256 is None) != (draw_count is None))
        or (
            quality_sha256 is not None
            and (
                not isinstance(quality_sha256, str)
                or len(quality_sha256) != 64
                or any(char not in "0123456789abcdef" for char in quality_sha256)
            )
        )
        or ((quality_sha256 is not None) and sampling_sha256 is None)
        or (
            policy_snapshot_sha256 is not None
            and (
                not isinstance(policy_snapshot_sha256, str)
                or len(policy_snapshot_sha256) != 64
                or any(char not in "0123456789abcdef" for char in policy_snapshot_sha256)
            )
        )
        or ((policy_snapshot_sha256 is not None) != (policy_id is not None))
        or ((policy_snapshot_sha256 is not None) != (policy_version is not None))
        or (
            policy_id is not None
            and (
                not isinstance(policy_id, str)
                or not policy_id
                or len(policy_id) > 128
                or any(char in policy_id for char in "/\\\x00")
            )
        )
        or (
            policy_version is not None
            and (
                not isinstance(policy_version, str)
                or not policy_version
                or len(policy_version) > 128
                or any(char in policy_version for char in "/\\\x00")
            )
        )
    ):
        return None
    result = {
        "ledger_id": ledger_id,
        "size_bytes": size,
        "sha256": digest,
        "status": "committed",
        "content_kind": "generated_token_ledger",
        "token_count": token_count,
        "stop_reason": value["stop_reason"],
    }
    if sampling_sha256 is not None:
        result["sampling_sha256"] = sampling_sha256
        result["draw_count"] = draw_count
    if quality_sha256 is not None:
        result["quality_sha256"] = quality_sha256
    if policy_snapshot_sha256 is not None:
        result["policy_snapshot_sha256"] = policy_snapshot_sha256
        result["policy_id"] = policy_id
        result["policy_version"] = policy_version
    return result


def run_qwen3_token_ledger_decode(
    *,
    model: Path | None,
    ledger: Path | None,
    ledger_metadata: Mapping[str, Any] | None,
    text_max_bytes: int = MAX_TEXT_BYTES,
    expected_chain_id: str = "",
    expected_generation: int = 0,
    expected_first_sequence: int = 0,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata = _metadata(ledger_metadata or {})
    if model is None or ledger is None:
        return _base_result("invalid_request", code="model_and_ledger_required")
    if metadata is None:
        return _base_result("invalid_request", code="ledger_metadata_invalid")
    if (
        isinstance(text_max_bytes, bool)
        or not isinstance(text_max_bytes, int)
        or not 0 < text_max_bytes <= MAX_TEXT_BYTES
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS
        or isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation < 0
        or isinstance(expected_first_sequence, bool)
        or not isinstance(expected_first_sequence, int)
        or expected_first_sequence <= 0
    ):
        return _base_result("invalid_request", code="decode_limits_invalid")
    model_path = model.expanduser().absolute().resolve(strict=False)
    ledger_path = ledger.expanduser().absolute().resolve(strict=False)
    if not model_path.is_dir() or not ledger_path.is_file():
        return _base_result("invalid_request", code="model_or_ledger_missing")
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_token_ledger_decode",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(model_path),
        "ledger_path": str(ledger_path),
        "ledger_metadata": metadata,
        "text_max_bytes": text_max_bytes,
        "expected_chain_id": str(expected_chain_id),
        "expected_generation": expected_generation,
        "expected_first_sequence": expected_first_sequence,
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
    }
    try:
        report = (worker_runner or _run_worker)(request, float(timeout_seconds))
    except Exception:
        return _worker_failure("worker_runner_failed")
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _worker_failure("invalid_worker_output")
    return report


def run_qwen3_token_ledger_replay(
    *,
    ledger: Path | None,
    ledger_metadata: Mapping[str, Any] | None,
    expected_chain_id: str = "",
    expected_generation: int = 0,
    expected_first_sequence: int = 0,
    expected_sampling_sha256: str = "",
    expected_quality_sha256: str = "",
    expected_policy_snapshot_sha256: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    operation = "qwen3_token_ledger_replay"
    metadata = _metadata(ledger_metadata or {})
    if ledger is None:
        return _base_result("invalid_request", code="ledger_required", operation=operation)
    if not isinstance(expected_sampling_sha256, str):
        return _base_result("invalid_request", code="sampling_digest_invalid", operation=operation)
    if expected_sampling_sha256 and (
        len(expected_sampling_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sampling_sha256)
    ):
        return _base_result("invalid_request", code="sampling_digest_invalid", operation=operation)
    if not isinstance(expected_quality_sha256, str):
        return _base_result("invalid_request", code="quality_digest_invalid", operation=operation)
    if expected_quality_sha256 and (
        len(expected_quality_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_quality_sha256)
    ):
        return _base_result("invalid_request", code="quality_digest_invalid", operation=operation)
    if not isinstance(expected_policy_snapshot_sha256, str):
        return _base_result("invalid_request", code="policy_digest_invalid", operation=operation)
    if expected_policy_snapshot_sha256 and (
        len(expected_policy_snapshot_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_policy_snapshot_sha256)
    ):
        return _base_result("invalid_request", code="policy_digest_invalid", operation=operation)
    if metadata is None:
        return _base_result("invalid_request", code="ledger_metadata_invalid", operation=operation)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS
        or isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation < 0
        or isinstance(expected_first_sequence, bool)
        or not isinstance(expected_first_sequence, int)
        or expected_first_sequence <= 0
    ):
        return _base_result("invalid_request", code="replay_limits_invalid", operation=operation)
    ledger_path = ledger.expanduser().absolute().resolve(strict=False)
    if not ledger_path.is_file():
        return _base_result("invalid_request", code="ledger_missing", operation=operation)
    request = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "qwen3_token_ledger_replay",
        "read_only": True,
        "network_access": "disabled",
        "ledger_path": str(ledger_path),
        "ledger_metadata": metadata,
        "expected_chain_id": str(expected_chain_id),
        "expected_generation": expected_generation,
        "expected_first_sequence": expected_first_sequence,
        "expected_sampling_sha256": expected_sampling_sha256,
        "expected_quality_sha256": expected_quality_sha256,
        "expected_policy_snapshot_sha256": expected_policy_snapshot_sha256,
        "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
    }
    try:
        report = (worker_runner or _run_worker)(request, float(timeout_seconds))
    except Exception:
        return _worker_failure("worker_runner_failed", operation=operation)
    if not isinstance(report, dict) or report.get("tool") != TOOL:
        return _worker_failure("invalid_worker_output", operation=operation)
    return report


__all__ = ["run_qwen3_token_ledger_decode", "run_qwen3_token_ledger_replay"]
