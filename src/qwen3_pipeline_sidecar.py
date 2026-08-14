"""Controller for the isolated Qwen3 node-local execution sidecar.

The controller owns only a bounded JSONL control channel.  Model weights,
tokenizers and adapters stay inside the dedicated sidecar process; the main
runtime receives aggregate resource and lifecycle evidence only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
OPERATION = "qwen3_pipeline_sidecar"
MAX_FRAME_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 180.0
MAX_TIMEOUT_SECONDS = 1800.0
EXECUTION_PHASES = {"prefill", "decode"}


class Qwen3SidecarError(RuntimeError):
    """The isolated sidecar rejected or could not complete a lifecycle step."""

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
        raise Qwen3SidecarError(
            "qwen3_sidecar_protocol_invalid", "sidecar frame is not JSON serializable",
        ) from exc
    if len(encoded) > MAX_FRAME_BYTES:
        raise Qwen3SidecarError(
            "qwen3_sidecar_frame_oversize", "sidecar control frame exceeds 256 KiB",
        )
    return encoded


def _default_sidecar_python() -> Path:
    root = Path(__file__).resolve().parents[1]
    if os.name == "nt":
        return root / ".venv-qwen3-sidecar" / "Scripts" / "python.exe"
    return root / ".venv-qwen3-sidecar" / "bin" / "python"


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class Qwen3PipelineSidecarSession:
    """A bounded prepare/commit/release session for one local segment."""

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
            raise Qwen3SidecarError("qwen3_sidecar_model_missing", "model directory is unavailable")
        try:
            start, end = int(layer_range[0]), int(layer_range[1])
            total = int(total_layers)
            margin = float(safety_margin)
            reserve = int(reserve_bytes)
            timeout = float(timeout_seconds)
        except (IndexError, TypeError, ValueError) as exc:
            raise Qwen3SidecarError("qwen3_sidecar_contract_invalid", "sidecar dimensions are invalid") from exc
        if start < 0 or end <= start or end > total or total <= 0:
            raise Qwen3SidecarError("qwen3_sidecar_contract_invalid", "sidecar layer range is invalid")
        if not 1.0 <= margin <= 100.0 or reserve < 0 or not 0 < timeout <= MAX_TIMEOUT_SECONDS:
            raise Qwen3SidecarError("qwen3_sidecar_contract_invalid", "sidecar resource or timeout bound is invalid")
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
            "controller_python": str(Path(sys.executable).absolute().resolve(strict=False)),
        }

    def _start(self) -> None:
        if self._runner is not None:
            return
        if not self._sidecar_python.is_file():
            raise Qwen3SidecarError(
                "qwen3_sidecar_runtime_missing", "isolated Qwen3 sidecar Python is missing",
            )
        if self._sidecar_python == Path(sys.executable).absolute().resolve(strict=False):
            raise Qwen3SidecarError(
                "qwen3_sidecar_not_isolated", "sidecar must use a different Python environment",
            )
        worker = Path(__file__).resolve().parents[1] / "scripts" / "model_tools" / "qwen3_pipeline_runtime_worker.py"
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
            self._process = subprocess.Popen(
                [str(self._sidecar_python), str(worker)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise Qwen3SidecarError(
                "qwen3_sidecar_start_failed", "isolated Qwen3 sidecar could not start",
            ) from exc

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        encoded = _json_bytes(request)
        if self._runner is not None:
            try:
                report = self._runner(dict(request), self.timeout_seconds)
            except Exception as exc:
                raise Qwen3SidecarError(
                    "qwen3_sidecar_worker_failed", "sidecar runner failed",
                ) from exc
        else:
            self._start()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise Qwen3SidecarError("qwen3_sidecar_worker_failed", "sidecar pipes are unavailable")
            try:
                process.stdin.write(encoded.decode("utf-8") + "\n")
                process.stdin.flush()
            except OSError as exc:
                raise Qwen3SidecarError("qwen3_sidecar_worker_failed", "sidecar control write failed") from exc
            line_holder: list[str] = []
            read_error: list[BaseException] = []

            def _readline() -> None:
                try:
                    line_holder.append(process.stdout.readline())
                except BaseException as exc:  # pragma: no cover - OS-specific pipe failure
                    read_error.append(exc)

            reader = threading.Thread(target=_readline, daemon=True)
            reader.start()
            reader.join(timeout=self.timeout_seconds)
            if reader.is_alive():
                self._terminate_process()
                raise Qwen3SidecarError(
                    "qwen3_sidecar_timeout", "sidecar lifecycle step timed out",
                )
            if read_error:
                raise Qwen3SidecarError(
                    "qwen3_sidecar_worker_failed", "sidecar control read failed",
                ) from read_error[0]
            line = line_holder[0] if line_holder else ""
            if not line:
                raise Qwen3SidecarError("qwen3_sidecar_worker_crashed", "sidecar exited before returning a report")
            if len(line.encode("utf-8")) > MAX_FRAME_BYTES:
                raise Qwen3SidecarError("qwen3_sidecar_frame_oversize", "sidecar response exceeds 256 KiB")
            try:
                report = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Qwen3SidecarError("qwen3_sidecar_protocol_invalid", "sidecar returned invalid JSON") from exc
        if not isinstance(report, dict):
            raise Qwen3SidecarError("qwen3_sidecar_protocol_invalid", "sidecar report is not an object")
        _json_bytes(report)
        if report.get("schema_version") != SCHEMA_VERSION or report.get("operation") != OPERATION:
            raise Qwen3SidecarError("qwen3_sidecar_protocol_invalid", "sidecar report identity is invalid")
        self._last_report = dict(report)
        return dict(report)

    def _require_phase(self, phase: str, expected_status: str) -> dict[str, Any]:
        report = self._exchange(self._request(phase))
        status = str(report.get("status", ""))
        if status != expected_status or report.get("gate_passed") is not True:
            errors = report.get("errors") or [{"message": "sidecar rejected lifecycle step"}]
            message = str(errors[0].get("message", "sidecar rejected lifecycle step")) if isinstance(errors[0], dict) else str(errors[0])
            raise Qwen3SidecarError(
                str(errors[0].get("code", "qwen3_sidecar_rejected")) if isinstance(errors[0], dict) else "qwen3_sidecar_rejected",
                message,
            )
        return report

    def prepare(self) -> dict[str, Any]:
        if self.phase != "idle":
            raise Qwen3SidecarError("qwen3_sidecar_phase_invalid", "sidecar session is not idle")
        report = self._require_phase("prepare", "prepared")
        self.phase = "prepared"
        return self.snapshot(report)

    def commit(self) -> dict[str, Any]:
        if self.phase != "prepared":
            raise Qwen3SidecarError("qwen3_sidecar_phase_invalid", "sidecar commit requires prepared state")
        report = self._require_phase("commit", "committed")
        self.phase = "committed"
        return self.snapshot(report)

    def execute(
        self,
        *,
        phase: str,
        artifact_root: str | Path,
        input_ref: str | Path,
        output_ref: str | Path,
        kv_ref: str | Path | None = None,
        chain_id: str,
        segment_index: int,
        sequence_length: int,
        batch_size: int,
        has_next_segment: bool,
        generation: int,
        dtype: str,
        device: str,
    ) -> dict[str, Any]:
        """Execute one local data-plane step without putting tensors in JSONL.

        The references point to controller-owned local artifacts.  The sidecar
        reads/writes those artifacts and returns only bounded metadata.
        """
        if self.phase != "committed":
            raise Qwen3SidecarError(
                "qwen3_sidecar_phase_invalid", "sidecar execution requires committed state",
            )
        if phase not in EXECUTION_PHASES:
            raise Qwen3SidecarError(
                "qwen3_sidecar_phase_invalid", "sidecar execution phase is invalid",
            )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise Qwen3SidecarError(
                "qwen3_sidecar_artifact_root_missing", "sidecar artifact root is unavailable",
            )

        def _ref(value: str | Path, *, must_exist: bool) -> Path:
            candidate = Path(value).expanduser().absolute().resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise Qwen3SidecarError(
                    "qwen3_sidecar_artifact_scope", "sidecar artifact escapes local root",
                ) from exc
            if must_exist and not candidate.is_file():
                raise Qwen3SidecarError(
                    "qwen3_sidecar_artifact_missing", "sidecar input artifact is unavailable",
                )
            return candidate

        input_path = _ref(input_ref, must_exist=True)
        output_path = _ref(output_ref, must_exist=False)
        kv_path = _ref(kv_ref, must_exist=True) if kv_ref is not None else None
        input_bytes, input_sha256 = _file_evidence(input_path)
        kv_evidence = _file_evidence(kv_path) if kv_path is not None else None

        request = {
            **self._request(phase),
            "data_plane": "local_artifact",
            "artifact_root": str(root),
            "input_ref": str(input_path),
            "input_bytes": input_bytes,
            "input_sha256": input_sha256,
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
            "dtype": str(dtype),
            "device": str(device),
        }
        report = self._exchange(request)
        if report.get("status") != "executed" or report.get("gate_passed") is not True:
            errors = report.get("errors") or [{"message": "sidecar execution rejected"}]
            first = errors[0] if isinstance(errors, list) else errors
            if isinstance(first, dict):
                code = str(first.get("code", "qwen3_sidecar_execution_rejected"))
                message = str(first.get("message", "sidecar execution rejected"))
            else:
                code = "qwen3_sidecar_execution_rejected"
                message = str(first)
            raise Qwen3SidecarError(code, message)
        execution = report.get("execution")
        if not isinstance(execution, dict) or not output_path.is_file():
            raise Qwen3SidecarError(
                "qwen3_sidecar_artifact_missing", "sidecar returned no output artifact",
            )
        output_bytes, output_sha256 = _file_evidence(output_path)
        if (
            execution.get("artifact_bytes") != output_bytes
            or execution.get("artifact_sha256") != output_sha256
        ):
            output_path.unlink(missing_ok=True)
            raise Qwen3SidecarError(
                "qwen3_sidecar_artifact_mismatch", "sidecar output artifact evidence does not match",
            )
        return dict(report)

    def release(self) -> dict[str, Any]:
        if self.phase not in {"prepared", "committed", "aborted"}:
            if self.phase == "released":
                return self.snapshot(self._last_report)
            raise Qwen3SidecarError("qwen3_sidecar_phase_invalid", "sidecar release is not available")
        report = self._require_phase("release", "released")
        self.phase = "released"
        self._terminate_process()
        return self.snapshot(report)

    def abort(self) -> dict[str, Any]:
        if self.phase in {"idle", "released"}:
            self._terminate_process()
            self.phase = "aborted" if self.phase == "idle" else self.phase
            return self.snapshot(self._last_report)
        try:
            report = self._require_phase("abort", "aborted")
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
            "segment_materialized": self.phase == "committed",
            "cleanup_complete": bool(source.get("cleanup_complete", self.phase in {"released", "aborted"})),
            "assignment": source.get("assignment", {}),
            "resources": source.get("resources", {}),
            "execution": source.get("execution", {}),
            "errors": source.get("errors", []),
        }

    def __enter__(self) -> "Qwen3PipelineSidecarSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.phase in {"prepared", "committed"}:
            try:
                self.release()
            except Qwen3SidecarError:
                self.abort()
        else:
            self._terminate_process()


class Qwen3NetworkSidecarExecutor:
    """Adapt one target sidecar session to the network consume boundary.

    The callback receives the target-local input path from the coordinator and
    returns only sidecar metadata plus an internal ``output_path`` marker.  The
    coordinator consumes that marker locally and converts it into a path-free
    output reference before replying to the upstream peer.
    """

    def __init__(
        self,
        session: Qwen3PipelineSidecarSession,
        *,
        artifact_root: str | Path,
    ) -> None:
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Qwen3SidecarError(
                "qwen3_sidecar_artifact_root_missing", "network sidecar artifact root is unavailable",
            )
        self.session = session
        self.artifact_root = root
        self._outputs: dict[tuple[str, int, str], Path] = {}
        self._prefill_outputs: dict[tuple[str, int], Path] = {}

    @staticmethod
    def _safe_token(value: Any) -> str:
        token = str(value or "")
        if not token or len(token) > 128 or any(char in token for char in ("/", "\\", "\x00")):
            raise Qwen3SidecarError(
                "qwen3_sidecar_contract_invalid", "network sidecar identity is invalid",
            )
        return token

    def __call__(self, input_path: Path, request: Mapping[str, Any]) -> Mapping[str, Any]:
        chain_id = self._safe_token(request.get("chain_id"))
        transfer_id = self._safe_token(request.get("transfer_id"))
        phase = str(request.get("phase", ""))
        generation = int(request.get("generation", -1))
        segment_index = int(request.get("segment_index", -1))
        if phase not in EXECUTION_PHASES or generation < 0 or segment_index < 0:
            raise Qwen3SidecarError(
                "qwen3_sidecar_contract_invalid", "network sidecar execution contract is invalid",
            )
        key = (chain_id, segment_index, phase)
        output = self.artifact_root / (
            f"qwen3-consume-{transfer_id}-{phase}-{generation}-{segment_index}.pt"
        )
        kv_ref = None
        if phase == "decode":
            kv_ref = self._prefill_outputs.get((chain_id, segment_index))
            if kv_ref is None or not kv_ref.is_file():
                raise Qwen3SidecarError(
                    "qwen3_sidecar_kv_missing", "target sidecar has no prefill KV artifact",
                )
        report = self.session.execute(
            phase=phase,
            artifact_root=self.artifact_root,
            input_ref=input_path,
            output_ref=output,
            kv_ref=kv_ref,
            chain_id=chain_id,
            segment_index=segment_index,
            sequence_length=int(request["sequence_length"]),
            batch_size=int(request["batch_size"]),
            has_next_segment=bool(request["has_next_segment"]),
            generation=generation,
            dtype=str(request["dtype"]),
            device=str(request["device"]),
        )
        self._outputs[key] = output
        if phase == "prefill":
            self._prefill_outputs[(chain_id, segment_index)] = output
        return {**dict(report), "output_path": str(output)}

    def cleanup(self, request: Mapping[str, Any], reason_code: str = "cleanup") -> None:
        chain_id = str(request.get("chain_id", ""))
        segment_index = int(request.get("segment_index", -1) or -1)
        for key, path in list(self._outputs.items()):
            if key[0] == chain_id and (segment_index < 0 or key[1] == segment_index):
                path.unlink(missing_ok=True)
                self._outputs.pop(key, None)
        for key, path in list(self._prefill_outputs.items()):
            if key[0] == chain_id and (segment_index < 0 or key[1] == segment_index):
                path.unlink(missing_ok=True)
                self._prefill_outputs.pop(key, None)
        if self.session.phase in {"prepared", "committed"}:
            try:
                self.session.abort()
            except Qwen3SidecarError:
                self.session._terminate_process()


__all__ = [
    "MAX_FRAME_BYTES", "OPERATION", "Qwen3PipelineSidecarSession",
    "Qwen3NetworkSidecarExecutor", "Qwen3SidecarError",
]
