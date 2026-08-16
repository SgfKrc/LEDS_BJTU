"""Single-host orchestration for two/three Gemma 4 sidecars."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from gemma4_pipeline_contract import validate_gemma4_pipeline_contract


class Gemma4MultiSidecarError(RuntimeError):
    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)
        self.reason = str(reason)
        super().__init__(self.reason)


def _chain_token(chain_id: str) -> str:
    return hashlib.sha256(str(chain_id).encode("utf-8")).hexdigest()[:20]


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class Gemma4PipelineMultiSidecar:
    """Run local prefill/decode through committed Gemma segments."""

    def __init__(
        self,
        *,
        sessions: Sequence[Any],
        segments: Sequence[dict[str, Any]],
        artifact_root: str | Path,
        chain_id: str,
        generation: int,
    ) -> None:
        if not str(chain_id).strip() or len(sessions) != len(segments):
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_contract_invalid",
                "chain identity or session count is invalid",
            )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_artifact_root_missing",
                "artifact root is unavailable",
            )
        self.sessions = list(sessions)
        self.segments = [dict(segment) for segment in segments]
        self.artifact_root = root
        self.chain_id = str(chain_id)
        self._token = _chain_token(self.chain_id)
        self.generation = int(generation)
        self.phase = "idle"
        self._created: list[Path] = []
        self._prefill_outputs: list[Path] = []
        self._outputs: dict[str, list[Path]] = {}
        self._reports: dict[str, list[dict[str, Any]]] = {}
        self._cleanup_complete = False
        self._sequence_length = 0

    @classmethod
    def from_contract(
        cls,
        *,
        contract: dict[str, Any],
        artifact_root: str | Path,
        session_factory: Callable[[dict[str, Any]], Any],
    ) -> "Gemma4PipelineMultiSidecar":
        canonical = validate_gemma4_pipeline_contract(contract)
        sessions: list[Any] = []
        try:
            for segment in canonical["segments"]:
                sessions.append(session_factory({
                    "model_id": canonical["model_id"],
                    "model_sha256": canonical["model_sha256"],
                    "config_id": canonical["config_id"],
                    "plan_id": canonical["plan_id"],
                    "generation": canonical["generation"],
                    "total_layers": canonical["total_layers"],
                    **segment,
                }))
        except Exception as exc:
            for session in reversed(sessions):
                try:
                    session.abort()
                except Exception:
                    pass
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_session_failed",
                "Gemma 4 sidecar session construction failed",
            ) from exc
        return cls(
            sessions=sessions,
            segments=canonical["segments"],
            artifact_root=artifact_root,
            chain_id=canonical["contract_sha256"],
            generation=int(canonical["generation"]),
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "generation": self.generation,
            "phase": self.phase,
            "segment_count": len(self.sessions),
            "full_model_materialized": False,
            "multimodal_materialized": False,
            "created_artifact_count": len(self._created),
            "cleanup_complete": self._cleanup_complete,
            "executed_phases": sorted(self._reports),
            "sequence_length": self._sequence_length,
            "last_reports": {
                phase: [dict(report) for report in reports]
                for phase, reports in self._reports.items()
            },
        }

    def _artifact(self, phase: str, segment_index: int) -> Path:
        path = self.artifact_root / (
            f"gemma4-{self._token}-{self.generation}-{phase}-"
            f"{segment_index}-{uuid4().hex}.pt"
        )
        self._created.append(path)
        self._cleanup_complete = False
        return path

    def _cleanup_artifacts(self) -> None:
        failures = 0
        for path in list(self._created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failures += 1
        self._created.clear()
        self._prefill_outputs.clear()
        self._outputs.clear()
        self._reports.clear()
        self._cleanup_complete = failures == 0

    def _abort_all(self, reason: str) -> None:
        for session in reversed(self.sessions):
            try:
                session.abort()
            except Exception:
                pass
        self._cleanup_artifacts()
        self.phase = "aborted"

    def prepare(self) -> dict[str, Any]:
        if self.phase != "idle":
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "chain is not idle",
            )
        try:
            reports = [session.prepare() for session in self.sessions]
            self.phase = "prepared"
            return {"status": "prepared", "reports": reports, "snapshot": self.snapshot}
        except Exception as exc:
            self._abort_all("prepare_failed")
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_prepare_failed", "one Gemma 4 segment rejected prepare",
            ) from exc

    def commit(self) -> dict[str, Any]:
        if self.phase != "prepared":
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "commit requires prepared chain",
            )
        try:
            reports = [session.commit() for session in self.sessions]
            self.phase = "committed"
            return {"status": "committed", "reports": reports, "snapshot": self.snapshot}
        except Exception as exc:
            self._abort_all("commit_failed")
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_commit_failed", "one Gemma 4 segment rejected commit",
            ) from exc

    def _execute_phase(
        self,
        phase: str,
        *,
        input_ref: str | Path,
        batch_size: int,
        sequence_length: int,
    ) -> dict[str, Any]:
        if self.phase not in {"committed", "prefilled", "decoded"}:
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "chain is not executable",
            )
        if phase == "prefill" and self._reports.get("prefill"):
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "prefill was already executed",
            )
        if phase == "decode" and not self._prefill_outputs:
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "decode requires prefill KV artifacts",
            )
        current_input = Path(input_ref).expanduser().absolute().resolve(strict=False)
        try:
            current_input.relative_to(self.artifact_root)
        except ValueError as exc:
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_artifact_scope", "input artifact escapes chain root",
            ) from exc
        if not current_input.is_file():
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_artifact_missing", "input artifact is unavailable",
            )
        reports: list[dict[str, Any]] = []
        outputs: list[Path] = []
        try:
            for index, (session, segment) in enumerate(zip(self.sessions, self.segments)):
                output = self._artifact(phase, index)
                kv_ref = self._prefill_outputs[index] if phase == "decode" else None
                report = session.execute(
                    phase=phase,
                    artifact_root=self.artifact_root,
                    input_ref=current_input,
                    output_ref=output,
                    kv_ref=kv_ref,
                    chain_id=self.chain_id,
                    segment_index=index,
                    sequence_length=int(sequence_length),
                    batch_size=int(batch_size),
                    has_next_segment=index < len(self.sessions) - 1,
                    generation=self.generation,
                )
                if report.get("status") != "executed" or report.get("gate_passed") is not True:
                    raise Gemma4MultiSidecarError(
                        "gemma4_multisidecar_execution_failed",
                        "sidecar did not return an executed report",
                    )
                execution = report.get("execution")
                if (
                    not isinstance(execution, dict)
                    or execution.get("full_model_materialized") is not False
                    or execution.get("multimodal_materialized") is not False
                    or execution.get("segment_materialized") is not True
                    or not output.is_file()
                    or _file_evidence(output) != (
                        int(execution.get("artifact_bytes", -1)),
                        str(execution.get("artifact_sha256", "")),
                    )
                ):
                    raise Gemma4MultiSidecarError(
                        "gemma4_multisidecar_materialization_invalid",
                        "sidecar execution evidence is unsafe",
                    )
                if index < len(self.sessions) - 1:
                    handoff = report.get("hidden_handoff")
                    if not isinstance(handoff, dict) or handoff.get("from_segment") != index:
                        raise Gemma4MultiSidecarError(
                            "gemma4_multisidecar_handoff_missing",
                            "non-final segment returned no hidden handoff",
                        )
                reports.append(dict(report))
                outputs.append(output)
                current_input = output
            self._outputs[phase] = outputs
            self._reports[phase] = reports
            if phase == "prefill":
                self._prefill_outputs = list(outputs)
                self.phase = "prefilled"
            else:
                self.phase = "decoded"
            self._sequence_length = int(sequence_length)
            return {"status": f"{phase}d", "reports": reports, "snapshot": self.snapshot}
        except Exception:
            self._abort_all(f"{phase}_failed")
            raise

    def prefill(self, *, input_ref: str | Path, batch_size: int, sequence_length: int) -> dict[str, Any]:
        return self._execute_phase(
            "prefill", input_ref=input_ref,
            batch_size=batch_size, sequence_length=sequence_length,
        )

    def decode(self, *, input_ref: str | Path, batch_size: int, sequence_length: int) -> dict[str, Any]:
        return self._execute_phase(
            "decode", input_ref=input_ref,
            batch_size=batch_size, sequence_length=sequence_length,
        )

    def final_output_ref(self, phase: str) -> Path:
        outputs = self._outputs.get(str(phase), [])
        if not outputs:
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_output_missing", "chain has no output for phase",
            )
        return outputs[-1]

    def release(self) -> dict[str, Any]:
        if self.phase == "released":
            return {"status": "released", "snapshot": self.snapshot}
        if self.phase not in {"prepared", "committed", "prefilled", "decoded"}:
            raise Gemma4MultiSidecarError(
                "gemma4_multisidecar_phase_invalid", "release is unavailable",
            )
        try:
            reports = [session.release() for session in reversed(self.sessions)]
        finally:
            self._cleanup_artifacts()
        self.phase = "released"
        return {"status": "released", "reports": reports, "snapshot": self.snapshot}

    def abort(self) -> dict[str, Any]:
        if self.phase == "aborted":
            return {"status": "aborted", "snapshot": self.snapshot}
        self._abort_all("cancelled")
        return {"status": "aborted", "snapshot": self.snapshot}


__all__ = ["Gemma4MultiSidecarError", "Gemma4PipelineMultiSidecar"]
