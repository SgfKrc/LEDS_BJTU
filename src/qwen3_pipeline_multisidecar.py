"""Single-host orchestration for two/three isolated Qwen3 sidecars.

Only bounded metadata crosses the sidecar JSONL control channel.  Hidden
states and per-segment KV caches are written by sidecars to a controller-owned
local artifact directory; the controller passes references and validates the
returned handoff/KV contracts before advancing to the next segment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from qwen3_pipeline_contract import (
    QWEN3_HANDOFF_SCHEMA_VERSION,
    build_kv_contract,
    validate_segment_plan,
)
from qwen3_pipeline_network import (
    Qwen3ResolvedArtifact,
    build_local_artifact_reference,
    validate_qwen3_artifact_reference,
)


class Qwen3MultiSidecarError(RuntimeError):
    """A multi-sidecar chain failed closed and was cleaned up."""

    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)
        self.reason = str(reason)
        super().__init__(self.reason)


def _dtype_alias(value: Any) -> str:
    raw = str(value or "").lower()
    return {
        "float16": "torch.float16",
        "fp16": "torch.float16",
        "torch.float16": "torch.float16",
        "bfloat16": "torch.bfloat16",
        "bf16": "torch.bfloat16",
        "torch.bfloat16": "torch.bfloat16",
        "float32": "torch.float32",
        "fp32": "torch.float32",
        "torch.float32": "torch.float32",
    }.get(raw, raw)


def _device_alias(value: Any) -> str:
    raw = str(value or "").lower()
    if raw == "cuda":
        return "cuda"
    if raw.startswith("cuda:"):
        return "cuda"
    return raw


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


def cleanup_qwen3_local_artifacts(
    artifact_root: str | Path, chain_id: str,
) -> dict[str, Any]:
    """Remove only artifacts belonging to one persisted local chain."""
    root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
    if not root.exists():
        return {"cleanup_complete": True, "removed_artifact_count": 0}
    if not root.is_dir():
        raise Qwen3MultiSidecarError(
            "qwen3_multisidecar_artifact_root_missing", "artifact root is unavailable",
        )
    removed = 0
    failed = 0
    for path in root.glob(f"qwen3-{_chain_token(chain_id)}-*.pt"):
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            failed += 1
    return {
        "cleanup_complete": failed == 0,
        "removed_artifact_count": removed,
        "failed_artifact_count": failed,
    }


class Qwen3PipelineMultiSidecar:
    """Coordinate a local 2/3-segment sidecar chain."""

    def __init__(
        self,
        *,
        sessions: Sequence[Any],
        segments: Sequence[dict[str, Any]],
        artifact_root: str | Path,
        chain_id: str,
        generation: int = 0,
        node_ids: Sequence[str] | None = None,
        handoff_transport: Any | None = None,
        hidden_size: int | None = None,
    ) -> None:
        if not str(chain_id).strip():
            raise Qwen3MultiSidecarError("qwen3_multisidecar_chain_invalid", "chain_id is required")
        if len(sessions) != len(segments):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_chain_invalid", "session and segment counts do not match",
            )
        try:
            normalized = validate_segment_plan(segments, total_layers=sum(
                int(segment["layer_range"][1]) - int(segment["layer_range"][0])
                for segment in segments
            ))
        except Exception as exc:
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_chain_invalid", str(exc),
            ) from exc
        if any(
            not str(segment.get("device", "")).strip()
            or str(segment.get("device")) == "auto"
            or not str(segment.get("dtype", "")).strip()
            for segment in normalized
        ):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid",
                "multi-sidecar execution requires explicit device and dtype per segment",
            )
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_artifact_root_missing", "artifact root is unavailable",
            )
        try:
            generation = int(generation)
        except (TypeError, ValueError) as exc:
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid", "generation is invalid",
            ) from exc
        if generation < 0:
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid", "generation is invalid",
            )
        if hidden_size is not None:
            try:
                hidden_size = int(hidden_size)
            except (TypeError, ValueError) as exc:
                raise Qwen3MultiSidecarError(
                    "qwen3_multisidecar_contract_invalid", "hidden size is invalid",
                ) from exc
            if hidden_size <= 0:
                raise Qwen3MultiSidecarError(
                    "qwen3_multisidecar_contract_invalid", "hidden size is invalid",
                )
        self.sessions = list(sessions)
        self.segments = normalized
        resolved_node_ids = list(node_ids or [f"segment-{index}" for index in range(len(normalized))])
        if (
            len(resolved_node_ids) != len(normalized)
            or any(not isinstance(node_id, str) or not node_id for node_id in resolved_node_ids)
            or len(set(resolved_node_ids)) != len(resolved_node_ids)
        ):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid", "sidecar node identities are invalid",
            )
        self.node_ids = resolved_node_ids
        self.artifact_root = root
        self.chain_id = str(chain_id)
        self._chain_token = _chain_token(self.chain_id)
        self.generation = generation
        self.hidden_size = hidden_size
        self.phase = "idle"
        self._created: list[Path] = []
        self._prefill_outputs: list[Path] = []
        self._outputs_by_phase: dict[str, list[Path]] = {}
        self._reports_by_phase: dict[str, list[dict[str, Any]]] = {}
        self._last_report: dict[str, Any] = {}
        self._cleanup_complete = False
        self._handoff_transport = handoff_transport
        self._handoff_references: dict[str, list[dict[str, Any]]] = {}
        self._decode_step_count = 0
        self._kv_sequence_length: int | None = None
        self._decode_history: list[dict[str, int]] = []

    @classmethod
    def from_contract(
        cls,
        *,
        contract: dict[str, Any],
        artifact_root: str | Path,
        session_factory: Callable[[dict[str, Any]], Any],
        handoff_transport: Any | None = None,
    ) -> "Qwen3PipelineMultiSidecar":
        """Bind a canonical QW3.6 transaction to a single-host chain."""
        from qwen3_pipeline_transaction import validate_qwen3_dry_run_contract

        canonical = validate_qwen3_dry_run_contract(contract)
        if canonical.get("execution_mode") != "node_local_sidecar":
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid",
                "multi-sidecar execution requires node_local_sidecar mode",
            )
        frames: list[dict[str, Any]] = []
        chain_segments: list[dict[str, Any]] = []
        for segment in canonical["segments"]:
            frames.append({
                "model_id": canonical["model_id"],
                "model_sha256": canonical["model_sha256"],
                "config_id": canonical["config_id"],
                "plan_id": canonical["plan_id"],
                "generation": canonical["generation"],
                "total_layers": canonical["total_layers"],
                **segment,
            })
            chain_segments.append({
                "layer_range": segment["layer_range"],
                "has_embedding": segment["has_embedding"],
                "has_lm_head": segment["has_lm_head"],
                "device": segment["execution_device"],
                "dtype": segment["dtype"],
            })
        sessions: list[Any] = []
        try:
            for frame in frames:
                sessions.append(session_factory(frame))
        except Exception as exc:
            for session in reversed(sessions):
                try:
                    session.abort()
                except Exception:
                    continue
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_session_failed", "sidecar session construction failed",
            ) from exc
        chain = cls(
            sessions=sessions,
            segments=chain_segments,
            artifact_root=artifact_root,
            chain_id=canonical["contract_sha256"],
            generation=int(canonical["generation"]),
            node_ids=[segment["node_id"] for segment in canonical["segments"]],
            handoff_transport=handoff_transport,
            hidden_size=int(canonical["hidden_size"]),
        )
        if handoff_transport is not None:
            try:
                handoff_transport.activate(canonical)
            except Exception as exc:
                chain._abort_all("network_activate_failed")
                raise Qwen3MultiSidecarError(
                    "qwen3_multisidecar_network_failed",
                    "network handoff transport could not activate",
                ) from exc
        return chain

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "generation": self.generation,
            "phase": self.phase,
            "segment_count": len(self.sessions),
            "full_model_materialized": False,
            "created_artifact_count": len(self._created),
            "cleanup_complete": self._cleanup_complete,
            "executed_phases": sorted(self._reports_by_phase),
            "handoff_modes": sorted({
                reference["mode"]
                for references in self._handoff_references.values()
                for reference in references
            }),
            "handoff_reference_count": sum(
                len(references) for references in self._handoff_references.values()
            ),
            "decode_step_count": self._decode_step_count,
            "kv_sequence_length": self._kv_sequence_length,
            "decode_history": [dict(value) for value in self._decode_history],
            "last_report": dict(self._last_report),
        }

    def _abort_all(self, reason_code: str = "aborted") -> None:
        sidecar_results: list[dict[str, Any]] = []
        for session in reversed(self.sessions):
            try:
                session.abort()
                sidecar_results.append({"aborted": True})
            except Exception as exc:
                sidecar_results.append({"aborted": False, "error": exc.__class__.__name__})
        removed = self._cleanup_artifacts()
        self.phase = "aborted"
        self._last_report = {
            "abort": {
                "reason_code": str(reason_code),
                "cleanup_complete": self._cleanup_complete,
                "removed_artifact_count": removed,
                "sidecars": sidecar_results,
            },
        }

    def _cleanup_artifacts(self) -> int:
        paths = set(self._created)
        paths.update(self.artifact_root.glob(f"qwen3-{self._chain_token}-*.pt"))
        removed = 0
        failed = 0
        for path in reversed(list(paths)):
            try:
                existed = path.exists()
                path.unlink(missing_ok=True)
                removed += int(existed)
            except OSError:
                failed += 1
                continue
        self._created.clear()
        self._prefill_outputs.clear()
        self._outputs_by_phase.clear()
        self._reports_by_phase.clear()
        self._handoff_references.clear()
        self._decode_step_count = 0
        self._kv_sequence_length = None
        self._decode_history.clear()
        transport_cleanup = {"cleanup_complete": True, "cleanup_failures": 0}
        if self._handoff_transport is not None:
            try:
                transport_cleanup = dict(self._handoff_transport.cleanup())
            except Exception:
                transport_cleanup = {"cleanup_complete": False, "cleanup_failures": 1}
        failed += int(transport_cleanup.get("cleanup_failures", 0) or 0)
        removed += int(transport_cleanup.get("removed_artifacts", 0) or 0)
        if transport_cleanup.get("cleanup_complete") is not True:
            failed += int(not transport_cleanup.get("cleanup_failures"))
        self._cleanup_complete = failed == 0
        return removed

    def _artifact(
        self, phase: str, segment_index: int, *, execution_generation: int | None = None,
    ) -> Path:
        generation = self.generation if execution_generation is None else int(execution_generation)
        path = self.artifact_root / (
            f"qwen3-{self._chain_token}-{generation}-{phase}-"
            f"{segment_index}-{uuid4().hex}.pt"
        )
        self._created.append(path)
        self._cleanup_complete = False
        return path

    @staticmethod
    def _validate_report(
        report: dict[str, Any],
        *,
        phase: str,
        segment: dict[str, Any],
        chain_id: str,
        generation: int,
        batch_size: int,
        sequence_length: int,
        has_next: bool,
        hidden_size: int | None = None,
        handoff_sequence_length: int | None = None,
    ) -> None:
        if not isinstance(report, dict) or report.get("status") != "executed" or report.get("gate_passed") is not True:
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_execution_failed", "sidecar did not return an executed report",
            )
        execution = report.get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("full_model_materialized") is not False
            or execution.get("segment_materialized") is not True
            or int(execution.get("artifact_bytes", 0) or 0) <= 0
            or not isinstance(execution.get("artifact_sha256"), str)
            or len(execution["artifact_sha256"]) != 64
        ):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_materialization_invalid", "sidecar execution evidence is unsafe",
            )
        kv = report.get("kv_contract")
        expected_kv = build_kv_contract(
            chain_id=chain_id,
            segment_index=int(segment["segment_index"]),
            layer_range=segment["layer_range"],
            sequence_length=sequence_length,
            batch_size=batch_size,
            dtype=_dtype_alias(segment["dtype"]),
            device=_device_alias(segment["device"]),
            phase=phase,
            generation=generation,
        )
        if not isinstance(kv, dict) or any(
            _dtype_alias(kv.get(key)) != _dtype_alias(value)
            if key == "dtype" else _device_alias(kv.get(key)) != _device_alias(value)
            if key == "device" else kv.get(key) != value
            for key, value in expected_kv.items()
        ):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_kv_mismatch", "sidecar KV contract does not match chain contract",
            )
        handoff = report.get("hidden_handoff")
        if has_next:
            if not isinstance(handoff, dict) or handoff.get("schema_version") != QWEN3_HANDOFF_SCHEMA_VERSION:
                raise Qwen3MultiSidecarError(
                    "qwen3_multisidecar_handoff_missing", "non-final segment returned no hidden handoff",
                )
            expected_handoff_sequence = (
                int(handoff_sequence_length)
                if handoff_sequence_length is not None else int(sequence_length)
            )
            expected_shape_prefix = [batch_size, expected_handoff_sequence]
            if (
                handoff.get("chain_id") != chain_id
                or handoff.get("from_segment") != segment["segment_index"]
                or handoff.get("to_segment") != segment["segment_index"] + 1
                or handoff.get("shape", [])[:2] != expected_shape_prefix
                or handoff.get("sequence_length") != expected_handoff_sequence
                or _dtype_alias(handoff.get("dtype")) != _dtype_alias(segment["dtype"])
                or _device_alias(handoff.get("device")) != _device_alias(segment["device"])
                or (
                    hidden_size is not None
                    and (
                        handoff.get("shape", [0, 0, 0])[2] != hidden_size
                        or handoff.get("hidden_size") != hidden_size
                    )
                )
            ):
                raise Qwen3MultiSidecarError(
                    "qwen3_multisidecar_handoff_mismatch", "hidden handoff does not match segment boundary",
                )
        elif handoff is not None:
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_handoff_unexpected", "final segment returned a hidden handoff",
            )

    def prepare(self) -> dict[str, Any]:
        if self.phase != "idle":
            raise Qwen3MultiSidecarError("qwen3_multisidecar_phase_invalid", "chain is not idle")
        try:
            reports = [session.prepare() for session in self.sessions]
        except Exception as exc:
            self._abort_all("prepare_failed")
            raise Qwen3MultiSidecarError("qwen3_multisidecar_prepare_failed", str(exc)) from exc
        self.phase = "prepared"
        self._last_report = {"prepare": reports}
        return self.snapshot

    def commit(self) -> dict[str, Any]:
        if self.phase != "prepared":
            raise Qwen3MultiSidecarError("qwen3_multisidecar_phase_invalid", "chain commit requires prepared state")
        try:
            reports = [session.commit() for session in self.sessions]
            for report in reports:
                if report.get("full_model_materialized") is True:
                    raise Qwen3MultiSidecarError(
                        "qwen3_multisidecar_materialization_invalid", "a sidecar reported full-model materialization",
                    )
        except Exception as exc:
            self._abort_all("commit_failed")
            if isinstance(exc, Qwen3MultiSidecarError):
                raise
            raise Qwen3MultiSidecarError("qwen3_multisidecar_commit_failed", str(exc)) from exc
        self.phase = "committed"
        self._last_report = {"commit": reports}
        return self.snapshot

    def _execute(
        self,
        *,
        phase: str,
        input_ref: str | Path,
        batch_size: int,
        sequence_length: int,
        generation: int,
        kv_refs: Sequence[Path | None] | None,
        input_sequence_length: int | None = None,
    ) -> dict[str, Any]:
        if (
            (phase == "prefill" and self.phase != "committed")
            or (phase == "decode" and self.phase not in {"prefilled", "decoded"})
        ):
            raise Qwen3MultiSidecarError("qwen3_multisidecar_phase_invalid", "chain is not ready for execution")
        if int(batch_size) <= 0 or int(sequence_length) <= 0 or int(generation) < 0:
            raise Qwen3MultiSidecarError("qwen3_multisidecar_contract_invalid", "execution dimensions are invalid")
        if input_sequence_length is None:
            input_sequence_length = int(sequence_length)
        if int(input_sequence_length) <= 0 or int(input_sequence_length) > int(sequence_length):
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid", "handoff sequence length is invalid",
            )
        resolved_kv_refs = list(kv_refs or [])
        if phase == "decode" and (
            len(resolved_kv_refs) != len(self.sessions)
            or any(path is None or not Path(path).is_file() for path in resolved_kv_refs)
        ):
            raise Qwen3MultiSidecarError("qwen3_multisidecar_kv_missing", "decode has no per-segment prefill KV artifacts")
        current_input = Path(input_ref).expanduser().absolute().resolve(strict=False)
        reports: list[dict[str, Any]] = []
        outputs: list[Path] = []
        handoff_references: list[dict[str, Any]] = []
        try:
            if self._handoff_transport is not None:
                self._handoff_transport.begin_phase(phase, int(generation))
            for index, (session, segment) in enumerate(zip(self.sessions, self.segments)):
                output = self._artifact(
                    phase, index, execution_generation=int(generation),
                )
                kv_ref = resolved_kv_refs[index] if phase == "decode" else None
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
                    generation=int(generation),
                    dtype=str(segment["dtype"]),
                    device=str(segment["device"]),
                )
                self._validate_report(
                    report,
                    phase=phase,
                    segment=segment,
                    chain_id=self.chain_id,
                    generation=int(generation),
                    batch_size=int(batch_size),
                    sequence_length=int(sequence_length),
                    has_next=index < len(self.sessions) - 1,
                    hidden_size=self.hidden_size,
                    handoff_sequence_length=input_sequence_length,
                )
                if index < len(self.sessions) - 1:
                    handoff = report.get("hidden_handoff") or {}
                    next_segment = self.segments[index + 1]
                    if (
                        _dtype_alias(handoff.get("dtype")) != _dtype_alias(next_segment["dtype"])
                        or _device_alias(handoff.get("device")) != _device_alias(next_segment["device"])
                    ):
                        raise Qwen3MultiSidecarError(
                            "qwen3_multisidecar_boundary_mismatch",
                            "hidden handoff dtype/device differs from next segment",
                        )
                    if self._handoff_transport is None:
                        if len(self.chain_id) == 64:
                            reference = build_local_artifact_reference(
                                output,
                                artifact_root=self.artifact_root,
                                source_node_id=self.node_ids[index],
                                target_node_id=self.node_ids[index + 1],
                                chain_id=self.chain_id,
                                generation=int(generation),
                                phase=phase,
                                from_segment=index,
                                to_segment=index + 1,
                            )
                            handoff_references.append(reference)
                        next_input = output
                    else:
                        actual_size, actual_sha256 = _file_evidence(output)
                        execution = report.get("execution") or {}
                        if (
                            execution.get("artifact_bytes") != actual_size
                            or execution.get("artifact_sha256") != actual_sha256
                        ):
                            raise Qwen3MultiSidecarError(
                                "qwen3_multisidecar_artifact_mismatch",
                                "network handoff source evidence changed",
                            )
                        resolved = self._handoff_transport.transfer(
                            source_path=output,
                            chain_id=self.chain_id,
                            generation=int(generation),
                            phase=phase,
                            from_segment=index,
                            to_segment=index + 1,
                            source_node_id=self.node_ids[index],
                            target_node_id=self.node_ids[index + 1],
                        )
                        if not isinstance(resolved, Qwen3ResolvedArtifact):
                            raise Qwen3MultiSidecarError(
                                "qwen3_multisidecar_network_failed",
                                "network handoff returned no resolved artifact",
                            )
                        reference = validate_qwen3_artifact_reference(resolved.reference)
                        next_input = Path(resolved.path).expanduser().absolute().resolve(strict=False)
                        try:
                            next_input.relative_to(self.artifact_root)
                        except ValueError as exc:
                            raise Qwen3MultiSidecarError(
                                "qwen3_multisidecar_artifact_scope",
                                "resolved network artifact escapes chain root",
                            ) from exc
                        if (
                            not next_input.is_file()
                            or _file_evidence(next_input)
                            != (reference["size_bytes"], reference["sha256"])
                            or reference["chain_id"] != self.chain_id
                            or reference["generation"] != int(generation)
                            or reference["phase"] != phase
                            or reference["from_segment"] != index
                            or reference["to_segment"] != index + 1
                            or reference["source_node_id"] != self.node_ids[index]
                            or reference["target_node_id"] != self.node_ids[index + 1]
                        ):
                            raise Qwen3MultiSidecarError(
                                "qwen3_multisidecar_network_mismatch",
                                "resolved network artifact does not match chain boundary",
                            )
                        handoff_references.append(reference)
                else:
                    next_input = output
                reports.append({
                    "segment_index": index,
                    "execution": report.get("execution", {}),
                    "hidden_handoff": report.get("hidden_handoff"),
                    "kv_contract": report.get("kv_contract"),
                    "artifact_reference": (
                        handoff_references[-1]
                        if index < len(self.sessions) - 1 and handoff_references
                        else None
                    ),
                })
                outputs.append(output)
                current_input = next_input
            if self._handoff_transport is not None:
                self._handoff_transport.finish_phase(phase, int(generation))
            if phase == "decode":
                for previous in resolved_kv_refs:
                    assert previous is not None
                    if Path(previous) not in outputs:
                        Path(previous).unlink(missing_ok=True)
        except Exception as exc:
            self._abort_all(f"{phase}_failed")
            if isinstance(exc, Qwen3MultiSidecarError):
                raise
            raise Qwen3MultiSidecarError("qwen3_multisidecar_execution_failed", str(exc)) from exc
        if phase == "prefill":
            self._prefill_outputs = list(outputs)
            self._kv_sequence_length = int(sequence_length)
            self.phase = "prefilled"
        elif phase == "decode":
            self._prefill_outputs = list(outputs)
            self._kv_sequence_length = int(sequence_length)
            self._decode_step_count += 1
            self._decode_history.append({
                "step_index": self._decode_step_count,
                "generation": int(generation),
                "sequence_length": int(sequence_length),
                "input_sequence_length": int(input_sequence_length),
            })
            self.phase = "decoded"
        self._outputs_by_phase[phase] = list(outputs)
        self._reports_by_phase[phase] = list(reports)
        self._handoff_references[phase] = list(handoff_references)
        self._last_report = {phase: reports}
        return self.snapshot

    def final_output_ref(self, phase: str) -> Path:
        """Return an internal local artifact reference for the parity gate."""
        outputs = self._outputs_by_phase.get(str(phase), [])
        if not outputs or not outputs[-1].is_file():
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_artifact_missing", "final execution artifact is unavailable",
            )
        return outputs[-1]

    def execution_reports(self, phase: str) -> list[dict[str, Any]]:
        return [dict(report) for report in self._reports_by_phase.get(str(phase), [])]

    def artifact_refs(self, phase: str) -> list[Path]:
        return list(self._outputs_by_phase.get(str(phase), []))

    def handoff_references(self, phase: str) -> list[dict[str, Any]]:
        return [dict(value) for value in self._handoff_references.get(str(phase), [])]

    def prefill(self, *, input_ref: str | Path, batch_size: int, sequence_length: int) -> dict[str, Any]:
        return self._execute(
            phase="prefill", input_ref=input_ref, batch_size=batch_size,
            sequence_length=sequence_length, generation=self.generation,
            kv_refs=None,
        )

    def decode(
        self,
        *,
        input_ref: str | Path,
        batch_size: int,
        sequence_length: int,
        input_sequence_length: int | None = None,
    ) -> dict[str, Any]:
        try:
            requested_sequence = int(sequence_length)
            requested_input = (
                int(input_sequence_length) if input_sequence_length is not None else None
            )
        except (TypeError, ValueError) as exc:
            self._abort_all("decode_sequence_invalid")
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_contract_invalid", "decode dimensions are invalid",
            ) from exc
        previous_sequence = self._kv_sequence_length
        if (
            previous_sequence is None
            or requested_sequence <= previous_sequence
            or (
                requested_input is not None
                and requested_sequence != previous_sequence + requested_input
            )
            or (self.phase == "decoded" and requested_input is None)
        ):
            self._abort_all("decode_sequence_invalid")
            raise Qwen3MultiSidecarError(
                "qwen3_multisidecar_sequence_mismatch",
                "decode sequence does not monotonically extend the current KV cache",
            )
        return self._execute(
            phase="decode", input_ref=input_ref, batch_size=batch_size,
            sequence_length=requested_sequence,
            generation=self.generation + self._decode_step_count + 1,
            kv_refs=self._prefill_outputs,
            input_sequence_length=requested_input,
        )

    def release(self) -> dict[str, Any]:
        if self.phase not in {"committed", "prefilled", "decoded", "parity_passed"}:
            raise Qwen3MultiSidecarError("qwen3_multisidecar_phase_invalid", "chain release is not available")
        errors: list[str] = []
        for session in reversed(self.sessions):
            try:
                session.release()
            except Exception as exc:
                errors.append(str(exc))
        removed = self._cleanup_artifacts()
        self.phase = "released" if not errors else "aborted"
        if errors:
            self._abort_all("release_failed")
            raise Qwen3MultiSidecarError("qwen3_multisidecar_release_failed", errors[0])
        self._last_report = {
            "release": {
                "cleanup_complete": self._cleanup_complete,
                "removed_artifact_count": removed,
            },
        }
        return self.snapshot

    def abort(self) -> dict[str, Any]:
        self._abort_all("aborted")
        return self.snapshot

    def cancel(self) -> dict[str, Any]:
        self._abort_all("cancelled")
        return self.snapshot

    def recover_after_restart(self) -> dict[str, Any]:
        """Clean stale local artifacts and abort child sessions after restart."""
        self._abort_all("restart_recovery")
        return self.snapshot


__all__ = [
    "Qwen3MultiSidecarError",
    "Qwen3PipelineMultiSidecar",
    "cleanup_qwen3_local_artifacts",
]
