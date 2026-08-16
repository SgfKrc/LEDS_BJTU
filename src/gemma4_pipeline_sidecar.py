"""Controller for an isolated Gemma 4 Unified pipeline sidecar."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable


SCHEMA_VERSION = 1
OPERATION = "gemma4_pipeline_sidecar"
MAX_FRAME_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 180.0
MAX_TIMEOUT_SECONDS = 1800.0
EXECUTION_PHASES = {"prefill", "decode"}


class Gemma4SidecarError(RuntimeError):
    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)
        self.reason = str(reason)
        super().__init__(self.reason)


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Gemma4SidecarError(
            "gemma4_sidecar_protocol_invalid",
            "sidecar frame is not JSON serializable",
        ) from exc
    if len(encoded) > MAX_FRAME_BYTES:
        raise Gemma4SidecarError(
            "gemma4_sidecar_frame_oversize",
            "sidecar control frame exceeds 256 KiB",
        )
    return encoded


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_sidecar_python() -> Path:
    root = _project_root()
    if os.name == "nt":
        return root / ".venv-gemma4-pipeline" / "Scripts" / "python.exe"
    return root / ".venv-gemma4-pipeline" / "bin" / "python"


def _native_sidecar_python() -> Path:
    root = _project_root()
    if os.name == "nt":
        return root / ".venv-gemma4-native" / "Scripts" / "python.exe"
    return root / ".venv-gemma4-native" / "bin" / "python"


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class Gemma4PipelineSidecarSession:
    """Bound one filtered Gemma 4 text assignment to an isolated process."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_id: str,
        model_sha256: str,
        config_id: str,
        plan_id: str,
        node_id: str,
        layer_range: tuple[int, int] | list[int],
        total_layers: int,
        has_embedding: bool,
        has_lm_head: bool,
        required_shared_kv_types: list[str] | tuple[str, ...] = (),
        produced_shared_kv_types: list[str] | tuple[str, ...] = (),
        execution_device: str = "cpu",
        dtype: str = "float32",
        generation: int = 0,
        assignment_manifest_sha256: str = "",
        safety_margin: float = 1.2,
        reserve_bytes: int = 512 * 1024**2,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sidecar_python: str | Path | None = None,
        worker_runner: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        root = Path(model_path).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise Gemma4SidecarError(
                "gemma4_sidecar_model_missing", "model assignment directory is unavailable",
            )
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
            total = int(total_layers)
            margin = float(safety_margin)
            reserve = int(reserve_bytes)
            timeout = float(timeout_seconds)
        except (IndexError, TypeError, ValueError) as exc:
            raise Gemma4SidecarError(
                "gemma4_sidecar_contract_invalid", "sidecar dimensions are invalid",
            ) from exc
        if start < 0 or end <= start or end > total or total <= 0:
            raise Gemma4SidecarError(
                "gemma4_sidecar_contract_invalid", "sidecar layer range is invalid",
            )
        if not 1.0 <= margin <= 100.0 or reserve < 0 or not 0 < timeout <= MAX_TIMEOUT_SECONDS:
            raise Gemma4SidecarError(
                "gemma4_sidecar_contract_invalid", "sidecar resource or timeout bound is invalid",
            )
        allowed_types = {"full_attention", "sliding_attention"}
        required = sorted({str(value) for value in required_shared_kv_types})
        produced = sorted({str(value) for value in produced_shared_kv_types})
        if any(value not in allowed_types for value in required + produced):
            raise Gemma4SidecarError(
                "gemma4_sidecar_contract_invalid", "sidecar shared-KV types are invalid",
            )
        self.model_path = root
        self._identity = {
            "model_id": str(model_id),
            "model_sha256": str(model_sha256),
            "config_id": str(config_id),
            "plan_id": str(plan_id),
            "node_id": str(node_id),
            "layer_range": [start, end],
            "total_layers": total,
            "has_embedding": bool(has_embedding),
            "has_lm_head": bool(has_lm_head),
            "required_shared_kv_types": required,
            "produced_shared_kv_types": produced,
            "execution_device": str(execution_device),
            "dtype": str(dtype),
            "generation": int(generation),
            "assignment_manifest_sha256": str(assignment_manifest_sha256),
        }
        self.safety_margin = margin
        self.reserve_bytes = reserve
        self.timeout_seconds = timeout
        self._runner = worker_runner
        self._sidecar_python = (
            Path(sidecar_python).expanduser().absolute().resolve(strict=False)
            if sidecar_python is not None else _default_sidecar_python()
        )
        if self._sidecar_python == _native_sidecar_python().resolve(strict=False):
            raise Gemma4SidecarError(
                "gemma4_sidecar_native_venv_forbidden",
                ".venv-gemma4-native is reserved for llama.cpp/MTMD",
            )
        self._process: subprocess.Popen[str] | None = None
        self.phase = "idle"
        self._last_report: dict[str, Any] = {}

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def _request(self, phase: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "operation": OPERATION,
            "phase": phase,
            "read_only": True,
            "network_access": "disabled",
            "model_path": str(self.model_path),
            **self._identity,
            "safety_margin": self.safety_margin,
            "reserve_bytes": self.reserve_bytes,
            "controller_python": str(Path(sys.executable).resolve(strict=False)),
        }

    def _start(self) -> None:
        if self._runner is not None:
            return
        if self._process is not None and self._process.poll() is None:
            return
        if not self._sidecar_python.is_file():
            raise Gemma4SidecarError(
                "gemma4_sidecar_runtime_missing",
                "isolated .venv-gemma4-pipeline Python is missing",
            )
        if self._sidecar_python == Path(sys.executable).resolve(strict=False):
            raise Gemma4SidecarError(
                "gemma4_sidecar_not_isolated", "sidecar must use a different Python environment",
            )
        worker = _project_root() / "scripts" / "model_tools" / "gemma4_pipeline_runtime_worker.py"
        env = dict(os.environ)
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NO_PROXY": "*",
        })
        for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            env.pop(name, None)
        try:
            self._process = subprocess.Popen(
                [str(self._sidecar_python), str(worker)],
                cwd=str(_project_root()),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise Gemma4SidecarError(
                "gemma4_sidecar_start_failed", "isolated Gemma 4 sidecar could not start",
            ) from exc

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        encoded = _json_bytes(request)
        if self._runner is not None:
            try:
                report = self._runner(dict(request), self.timeout_seconds)
            except Exception as exc:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_worker_failed", "sidecar runner failed",
                ) from exc
        else:
            self._start()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_worker_failed", "sidecar pipes are unavailable",
                )
            try:
                process.stdin.write(encoded.decode("utf-8") + "\n")
                process.stdin.flush()
            except OSError as exc:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_worker_failed", "sidecar control write failed",
                ) from exc
            lines: list[str] = []
            errors: list[BaseException] = []

            def _readline() -> None:
                try:
                    lines.append(process.stdout.readline())
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            reader = threading.Thread(target=_readline, daemon=True)
            reader.start()
            reader.join(timeout=self.timeout_seconds)
            if reader.is_alive():
                self._terminate_process()
                raise Gemma4SidecarError(
                    "gemma4_sidecar_timeout", "sidecar lifecycle step timed out",
                )
            if errors:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_worker_failed", "sidecar control read failed",
                ) from errors[0]
            line = lines[0] if lines else ""
            if not line:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_worker_crashed", "sidecar exited without a report",
                )
            if len(line.encode("utf-8")) > MAX_FRAME_BYTES:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_frame_oversize", "sidecar response exceeds 256 KiB",
                )
            try:
                report = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_protocol_invalid", "sidecar returned invalid JSON",
                ) from exc
        if not isinstance(report, dict):
            raise Gemma4SidecarError(
                "gemma4_sidecar_protocol_invalid", "sidecar report is not an object",
            )
        _json_bytes(report)
        if report.get("schema_version") != SCHEMA_VERSION or report.get("operation") != OPERATION:
            raise Gemma4SidecarError(
                "gemma4_sidecar_protocol_invalid", "sidecar report identity is invalid",
            )
        self._last_report = dict(report)
        return dict(report)

    def _require(self, phase: str, status: str) -> dict[str, Any]:
        report = self._exchange(self._request(phase))
        if report.get("status") != status or report.get("gate_passed") is not True:
            errors = report.get("errors") or [{}]
            first = errors[0] if isinstance(errors, list) else errors
            code = str(first.get("code", "gemma4_sidecar_rejected")) if isinstance(first, dict) else "gemma4_sidecar_rejected"
            reason = str(first.get("message", "sidecar rejected lifecycle step")) if isinstance(first, dict) else str(first)
            raise Gemma4SidecarError(code, reason)
        return report

    def prepare(self) -> dict[str, Any]:
        if self.phase != "idle":
            raise Gemma4SidecarError(
                "gemma4_sidecar_phase_invalid", "sidecar session is not idle",
            )
        report = self._require("prepare", "prepared")
        self.phase = "prepared"
        return self.snapshot(report)

    def commit(self) -> dict[str, Any]:
        if self.phase != "prepared":
            raise Gemma4SidecarError(
                "gemma4_sidecar_phase_invalid", "sidecar commit requires prepared state",
            )
        report = self._require("commit", "committed")
        self.phase = "committed"
        return self.snapshot(report)

    def execute(
        self,
        *,
        phase: str,
        artifact_root: str | Path,
        input_ref: str | Path,
        output_ref: str | Path,
        kv_ref: str | Path | None,
        chain_id: str,
        segment_index: int,
        sequence_length: int,
        batch_size: int,
        has_next_segment: bool,
        generation: int,
    ) -> dict[str, Any]:
        if self.phase != "committed" or phase not in EXECUTION_PHASES:
            raise Gemma4SidecarError(
                "gemma4_sidecar_phase_invalid", "sidecar execution phase is invalid",
            )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise Gemma4SidecarError(
                "gemma4_sidecar_artifact_root_missing", "artifact root is unavailable",
            )

        def _ref(value: str | Path, *, must_exist: bool) -> Path:
            path = Path(value).expanduser().absolute().resolve(strict=False)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise Gemma4SidecarError(
                    "gemma4_sidecar_artifact_scope", "artifact escapes local root",
                ) from exc
            if must_exist and not path.is_file():
                raise Gemma4SidecarError(
                    "gemma4_sidecar_artifact_missing", "input artifact is unavailable",
                )
            return path

        input_path = _ref(input_ref, must_exist=True)
        output_path = _ref(output_ref, must_exist=False)
        kv_path = _ref(kv_ref, must_exist=True) if kv_ref is not None else None
        input_evidence = _file_evidence(input_path)
        kv_evidence = _file_evidence(kv_path) if kv_path is not None else None
        request = {
            **self._request(phase),
            "data_plane": "local_artifact",
            "artifact_root": str(root),
            "input_ref": str(input_path),
            "input_bytes": input_evidence[0],
            "input_sha256": input_evidence[1],
            "output_ref": str(output_path),
            "kv_ref": str(kv_path) if kv_path is not None else None,
            "kv_bytes": kv_evidence[0] if kv_evidence is not None else None,
            "kv_sha256": kv_evidence[1] if kv_evidence is not None else None,
            "chain_id": str(chain_id),
            "segment_index": int(segment_index),
            "sequence_length": int(sequence_length),
            "batch_size": int(batch_size),
            "has_next_segment": bool(has_next_segment),
            "generation": int(generation),
        }
        report = self._exchange(request)
        if report.get("status") != "executed" or report.get("gate_passed") is not True:
            errors = report.get("errors") or [{}]
            first = errors[0] if isinstance(errors, list) else errors
            code = str(first.get("code", "gemma4_sidecar_execution_rejected")) if isinstance(first, dict) else "gemma4_sidecar_execution_rejected"
            reason = str(first.get("message", "sidecar execution rejected")) if isinstance(first, dict) else str(first)
            raise Gemma4SidecarError(code, reason)
        execution = report.get("execution")
        if not isinstance(execution, dict) or not output_path.is_file():
            raise Gemma4SidecarError(
                "gemma4_sidecar_artifact_missing", "sidecar returned no output artifact",
            )
        output_evidence = _file_evidence(output_path)
        if output_evidence != (
            int(execution.get("artifact_bytes", -1)),
            str(execution.get("artifact_sha256", "")),
        ):
            output_path.unlink(missing_ok=True)
            raise Gemma4SidecarError(
                "gemma4_sidecar_artifact_mismatch", "output artifact evidence differs",
            )
        return report

    def release(self) -> dict[str, Any]:
        if self.phase == "released":
            return self.snapshot()
        if self.phase not in {"prepared", "committed", "aborted"}:
            raise Gemma4SidecarError(
                "gemma4_sidecar_phase_invalid", "sidecar release is unavailable",
            )
        report = self._require("release", "released")
        self.phase = "released"
        self._terminate_process()
        return self.snapshot(report)

    def abort(self) -> dict[str, Any]:
        if self.phase in {"idle", "released"}:
            self._terminate_process()
            if self.phase == "idle":
                self.phase = "aborted"
            return self.snapshot()
        try:
            report = self._require("abort", "aborted")
        finally:
            self._terminate_process()
        self.phase = "aborted"
        return self.snapshot(report)

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def snapshot(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        source = report if report is not None else self._last_report
        return {
            "schema_version": SCHEMA_VERSION,
            "operation": OPERATION,
            "phase": self.phase,
            "status": source.get("status", self.phase),
            "gate_passed": source.get("gate_passed", self.phase in {"prepared", "committed", "released"}),
            "full_model_materialized": False,
            "multimodal_materialized": False,
            "segment_materialized": self.phase == "committed",
            "cleanup_complete": bool(source.get("cleanup_complete", self.phase in {"released", "aborted"})),
            "assignment": source.get("assignment", {}),
            "resources": source.get("resources", {}),
            "runtime": source.get("runtime", {}),
            "execution": source.get("execution", {}),
            "errors": source.get("errors", []),
        }

    def __enter__(self) -> "Gemma4PipelineSidecarSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.phase in {"prepared", "committed"}:
            try:
                self.release()
            except Gemma4SidecarError:
                self.abort()
        else:
            self._terminate_process()


__all__ = [
    "Gemma4PipelineSidecarSession",
    "Gemma4SidecarError",
    "MAX_FRAME_BYTES",
    "OPERATION",
]
