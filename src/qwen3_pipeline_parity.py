"""CPU-only parity gate for the local Qwen3 sidecar chain.

The gate consumes local tensor artifacts and emits metadata-only evidence. It
does not admit production traffic, select devices, or fall back to a full
model when parity fails.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


class Qwen3ParityError(ValueError):
    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)
        self.reason = str(reason)
        super().__init__(self.reason)


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _scoped(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser().absolute().resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Qwen3ParityError(
            "qwen3_parity_artifact_scope", "parity artifact escapes local root",
        ) from exc
    if not path.is_file():
        raise Qwen3ParityError("qwen3_parity_artifact_missing", "parity artifact is unavailable")
    return path


def _load_logits(path: Path) -> Any:
    try:
        import torch

        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Qwen3ParityError("qwen3_parity_artifact_invalid", "parity artifact could not be read") from exc
    if not isinstance(payload, dict):
        raise Qwen3ParityError("qwen3_parity_artifact_invalid", "parity artifact is not an object")
    logits = payload.get("logits")
    if logits is None or not hasattr(logits, "shape"):
        raise Qwen3ParityError("qwen3_parity_logits_missing", "parity artifact has no logits tensor")
    if getattr(logits, "device", None) is not None and str(logits.device) != "cpu":
        raise Qwen3ParityError("qwen3_parity_device_invalid", "CPU parity artifact is not CPU-resident")
    if not bool(torch.isfinite(logits).all().item()):
        raise Qwen3ParityError("qwen3_parity_nonfinite", "parity logits contain non-finite values")
    return logits


def _compare(reference: Any, candidate: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        raise Qwen3ParityError("qwen3_parity_shape_mismatch", "reference and candidate logits shapes differ")
    reference = reference.detach().to(dtype=torch.float32)
    candidate = candidate.detach().to(dtype=torch.float32)
    delta = (candidate - reference).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    denominator = reference.abs().clamp_min(1e-12)
    max_relative = float((delta / denominator).max().item()) if delta.numel() else 0.0
    passed = bool(torch.allclose(candidate, reference, rtol=float(rtol), atol=float(atol)))
    return {
        "passed": passed,
        "shape": [int(item) for item in reference.shape],
        "max_abs_error": max_abs,
        "max_relative_error": max_relative,
        "rtol": float(rtol),
        "atol": float(atol),
    }


def _validate_execution_evidence(
    *,
    artifact_root: Path,
    artifact_refs: Sequence[str | Path],
    reports: Sequence[dict[str, Any]],
    phase: str,
    segment_count: int,
    generation: int,
) -> dict[str, Any]:
    if len(artifact_refs) != segment_count or len(reports) != segment_count:
        raise Qwen3ParityError("qwen3_parity_segment_count", "parity execution segment count is incomplete")
    for index, (raw_path, report) in enumerate(zip(artifact_refs, reports)):
        path = _scoped(artifact_root, raw_path)
        size, digest = _file_evidence(path)
        execution = report.get("execution") if isinstance(report, dict) else None
        if (
            not isinstance(execution, dict)
            or execution.get("artifact_bytes") != size
            or execution.get("artifact_sha256") != digest
            or execution.get("full_model_materialized") is not False
            or execution.get("segment_materialized") is not True
        ):
            raise Qwen3ParityError(
                "qwen3_parity_evidence_mismatch", f"segment {index} artifact evidence mismatch",
            )
        kv = report.get("kv_contract")
        if (
            not isinstance(kv, dict)
            or kv.get("segment_index") != index
            or kv.get("phase") != phase
            or kv.get("generation") != generation
            or int(kv.get("sequence_length", 0) or 0) <= 0
        ):
            raise Qwen3ParityError(
                "qwen3_parity_kv_mismatch", f"segment {index} KV evidence mismatch",
            )
        if index < segment_count - 1:
            handoff = report.get("hidden_handoff")
            shape = handoff.get("shape") if isinstance(handoff, dict) else None
            if (
                not isinstance(handoff, dict)
                or handoff.get("from_segment") != index
                or handoff.get("to_segment") != index + 1
                or not isinstance(shape, list)
                or len(shape) != 3
                or any(int(item) <= 0 for item in shape)
            ):
                raise Qwen3ParityError(
                    "qwen3_parity_handoff_mismatch", f"segment {index} hidden handoff evidence mismatch",
                )
        elif report.get("hidden_handoff") is not None:
            raise Qwen3ParityError("qwen3_parity_handoff_unexpected", "final segment returned a hidden handoff")
    return {
        "phase": phase,
        "segment_count": segment_count,
        "generation": generation,
        "artifact_count": len(artifact_refs),
    }


def evaluate_qwen3_cpu_parity(
    *,
    artifact_root: str | Path,
    reference_prefill: str | Path,
    candidate_prefill: str | Path,
    reference_decode: str | Path,
    candidate_decode: str | Path,
    prefill_artifacts: Sequence[str | Path],
    prefill_reports: Sequence[dict[str, Any]],
    decode_artifacts: Sequence[str | Path],
    decode_reports: Sequence[dict[str, Any]],
    segment_count: int,
    generation: int,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Return metadata-only CPU parity evidence or a structured rejection."""
    try:
        root = Path(artifact_root).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise Qwen3ParityError("qwen3_parity_artifact_root_missing", "parity artifact root is unavailable")
        segment_count = int(segment_count)
        generation = int(generation)
        if segment_count not in {2, 3} or generation < 0:
            raise Qwen3ParityError("qwen3_parity_contract_invalid", "parity dimensions are invalid")
        if not 0 <= float(rtol) <= 1 or not 0 <= float(atol) <= 1:
            raise Qwen3ParityError("qwen3_parity_tolerance_invalid", "parity tolerance is invalid")
        candidate_prefill_path = _scoped(root, candidate_prefill)
        candidate_decode_path = _scoped(root, candidate_decode)
        reference_prefill_path = _scoped(root, reference_prefill)
        reference_decode_path = _scoped(root, reference_decode)
        prefill_evidence = _validate_execution_evidence(
            artifact_root=root, artifact_refs=prefill_artifacts,
            reports=prefill_reports, phase="prefill",
            segment_count=segment_count, generation=generation,
        )
        decode_evidence = _validate_execution_evidence(
            artifact_root=root, artifact_refs=decode_artifacts,
            reports=decode_reports, phase="decode",
            segment_count=segment_count, generation=generation + 1,
        )
        prefill = _compare(
            _load_logits(reference_prefill_path), _load_logits(candidate_prefill_path),
            rtol=float(rtol), atol=float(atol),
        )
        decode = _compare(
            _load_logits(reference_decode_path), _load_logits(candidate_decode_path),
            rtol=float(rtol), atol=float(atol),
        )
        passed = bool(prefill["passed"] and decode["passed"])
        return {
            "schema_version": 1,
            "gate": "qwen3_cpu_parity",
            "status": "passed" if passed else "rejected",
            "gate_passed": passed,
            "full_model_fallback": False,
            "full_model_materialized": False,
            "prefill": prefill,
            "decode": decode,
            "execution": {"prefill": prefill_evidence, "decode": decode_evidence},
            "errors": [] if passed else [{"code": "qwen3_parity_logits_mismatch", "message": "CPU logits parity failed"}],
        }
    except Qwen3ParityError as exc:
        return {
            "schema_version": 1,
            "gate": "qwen3_cpu_parity",
            "status": "rejected",
            "gate_passed": False,
            "full_model_fallback": False,
            "full_model_materialized": False,
            "errors": [{"code": exc.reason_code, "message": exc.reason}],
        }
    except (TypeError, ValueError, OSError, OverflowError) as exc:
        return {
            "schema_version": 1,
            "gate": "qwen3_cpu_parity",
            "status": "rejected",
            "gate_passed": False,
            "full_model_fallback": False,
            "full_model_materialized": False,
            "errors": [{"code": "qwen3_parity_invalid", "message": str(exc)[:2048]}],
        }


__all__ = ["Qwen3ParityError", "evaluate_qwen3_cpu_parity"]
