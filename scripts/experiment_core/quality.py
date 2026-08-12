"""Quality-evidence v1 normalization for experiment records.

This module deliberately records only structured, reviewable evidence.  It does
not execute a model, inspect prompts or outputs, or turn human image review
into an automatic score.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


QUALITY_SCHEMA_VERSION = "qlh.experiment_quality.v1"
QUALITY_SCHEMA_VERSION_V2 = "qlh.experiment_quality.v2"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NOT_COLLECTED_REASONS = {
    "not_provided",
    "not_applicable",
    "resource_unavailable",
    "source_unavailable",
    "pending_evaluation",
}
_SD_MODES = {"text_to_image", "img2img", "ip_adapter", "inpaint", "instruction"}
_MANUAL_REVIEW_STATUSES = {"passed", "failed", "pending", "not_required"}
_MAX_COUNT = 1_000_000_000


class QualityEvidenceError(ValueError):
    """Raised when a result-file quality_evidence payload is not v1 evidence."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualityEvidenceError(f"{label} must be an object")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise QualityEvidenceError(f"{label} contains unsupported fields")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise QualityEvidenceError(f"{label} must be a bounded identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise QualityEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNT:
        raise QualityEvidenceError(f"{label} must be an integer in range")
    return value


def _report_count(value: Any, label: str) -> int:
    """Read a count or count a report list without retaining its contents."""
    if isinstance(value, list):
        return _count(len(value), label)
    return _count(value, label)


def _rate(raw: Any, label: str) -> dict[str, Any]:
    value = _mapping(raw, label)
    _only_keys(value, {"evaluated_count", "passed_count"}, label)
    if "evaluated_count" not in value or "passed_count" not in value:
        raise QualityEvidenceError(f"{label} requires evaluated_count and passed_count")
    evaluated = _count(value["evaluated_count"], f"{label}.evaluated_count")
    passed = _count(value["passed_count"], f"{label}.passed_count")
    if evaluated == 0 or passed > evaluated:
        raise QualityEvidenceError(f"{label} has invalid counts")
    return {
        "evaluated_count": evaluated,
        "passed_count": passed,
        "rate": passed / evaluated,
    }


def _normalize_llm(raw: Any, expected_prompt_set: Mapping[str, Any] | None) -> dict[str, Any]:
    value = _mapping(raw, "llm")
    _only_keys(
        value,
        {"prompt_set_id", "prompt_set_sha256", "correctness", "format"},
        "llm",
    )
    required = {"prompt_set_id", "prompt_set_sha256", "correctness", "format"}
    if not required.issubset(value):
        raise QualityEvidenceError("llm requires a fixed prompt set and both rate counters")
    prompt_set_id = _identifier(value["prompt_set_id"], "llm.prompt_set_id")
    prompt_set_sha256 = _sha256(value["prompt_set_sha256"], "llm.prompt_set_sha256")
    if expected_prompt_set:
        expected_id = expected_prompt_set.get("id")
        expected_sha256 = expected_prompt_set.get("sha256")
        if expected_id and prompt_set_id != expected_id:
            raise QualityEvidenceError("llm prompt_set_id does not match the experiment plan")
        if expected_sha256 and prompt_set_sha256 != expected_sha256:
            raise QualityEvidenceError("llm prompt_set_sha256 does not match the experiment plan")
    return {
        "prompt_set_id": prompt_set_id,
        "prompt_set_sha256": prompt_set_sha256,
        "correctness": _rate(value["correctness"], "llm.correctness"),
        "format": _rate(value["format"], "llm.format"),
    }


def _normalize_sd(raw: Any) -> dict[str, Any]:
    value = _mapping(raw, "sd")
    _only_keys(
        value,
        {
            "mode",
            "asset_id",
            "artifact_id",
            "source_schema_version",
            "automatic_gate",
            "manual_review",
        },
        "sd",
    )
    required = {
        "mode", "asset_id", "artifact_id", "source_schema_version",
        "automatic_gate", "manual_review",
    }
    if not required.issubset(value):
        raise QualityEvidenceError("sd requires asset, gate, and manual-review evidence")
    mode = value["mode"]
    if mode not in _SD_MODES:
        raise QualityEvidenceError("sd.mode is unsupported")
    source_schema_version = _count(value["source_schema_version"], "sd.source_schema_version")
    if source_schema_version != 1:
        raise QualityEvidenceError("sd.source_schema_version must be 1")
    automatic = _mapping(value["automatic_gate"], "sd.automatic_gate")
    _only_keys(automatic, {"passed", "output_count", "unique_output_count"}, "sd.automatic_gate")
    if not {"passed", "output_count", "unique_output_count"}.issubset(automatic):
        raise QualityEvidenceError("sd.automatic_gate is incomplete")
    if not isinstance(automatic["passed"], bool):
        raise QualityEvidenceError("sd.automatic_gate.passed must be boolean")
    output_count = _count(automatic["output_count"], "sd.automatic_gate.output_count")
    unique_output_count = _count(
        automatic["unique_output_count"], "sd.automatic_gate.unique_output_count",
    )
    if unique_output_count > output_count:
        raise QualityEvidenceError("sd automatic unique_output_count exceeds output_count")
    manual = _mapping(value["manual_review"], "sd.manual_review")
    _only_keys(manual, {"status", "required_reviewers"}, "sd.manual_review")
    if not {"status", "required_reviewers"}.issubset(manual):
        raise QualityEvidenceError("sd.manual_review is incomplete")
    if manual["status"] not in _MANUAL_REVIEW_STATUSES:
        raise QualityEvidenceError("sd.manual_review.status is unsupported")
    required_reviewers = _count(
        manual["required_reviewers"], "sd.manual_review.required_reviewers",
    )
    return {
        "mode": mode,
        "asset_id": _identifier(value["asset_id"], "sd.asset_id"),
        "artifact_id": _identifier(value["artifact_id"], "sd.artifact_id"),
        "source_schema_version": source_schema_version,
        "automatic_gate": {
            "status": "passed" if automatic["passed"] else "failed",
            "output_count": output_count,
            "unique_output_count": unique_output_count,
        },
        "manual_review": {
            "status": manual["status"],
            "required_reviewers": required_reviewers,
        },
    }


def _normalize_gemma_judge(
    raw: Any,
    expected_gemma_judge: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep only frozen Gemma judge identity and counters-derived rates.

    Judge completions, prompts, images, and explanations are intentionally not
    part of this contract.  They must never reach an experiment result file.
    """
    value = _mapping(raw, "gemma_judge")
    _only_keys(
        value,
        {
            "model", "judge_contract_id", "judge_contract_sha256",
            "topic_hit", "key_element_coverage",
        },
        "gemma_judge",
    )
    required = {
        "model", "judge_contract_id", "judge_contract_sha256",
        "topic_hit", "key_element_coverage",
    }
    if not required.issubset(value):
        raise QualityEvidenceError("gemma_judge requires model and both rate counters")
    model = _identifier(value["model"], "gemma_judge.model")
    contract_id = _identifier(
        value["judge_contract_id"], "gemma_judge.judge_contract_id",
    )
    contract_sha256 = _sha256(
        value["judge_contract_sha256"], "gemma_judge.judge_contract_sha256",
    )
    if expected_gemma_judge:
        if model != expected_gemma_judge.get("model"):
            raise QualityEvidenceError("gemma_judge.model does not match the experiment plan")
        if contract_id != expected_gemma_judge.get("judge_contract_id"):
            raise QualityEvidenceError(
                "gemma_judge.judge_contract_id does not match the experiment plan",
            )
        if contract_sha256 != expected_gemma_judge.get("judge_contract_sha256"):
            raise QualityEvidenceError(
                "gemma_judge.judge_contract_sha256 does not match the experiment plan",
            )
    return {
        "model": model,
        "judge_contract_id": contract_id,
        "judge_contract_sha256": contract_sha256,
        "topic_hit": _rate(value["topic_hit"], "gemma_judge.topic_hit"),
        "key_element_coverage": _rate(
            value["key_element_coverage"],
            "gemma_judge.key_element_coverage",
        ),
    }


def not_collected_quality(reason: str = "not_provided") -> dict[str, Any]:
    """Return the explicit v1 representation for absent quality evidence."""
    if reason not in _NOT_COLLECTED_REASONS:
        raise QualityEvidenceError("unsupported not_collected_reason")
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "not_collected",
        "not_collected_reason": reason,
        "correct_rate": None,
        "format_rate": None,
        "llm": None,
        "sd": None,
    }


def invalid_quality_evidence() -> dict[str, Any]:
    """Return a redacted marker; parser errors are never copied into records."""
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "invalid",
        "not_collected_reason": "invalid_evidence",
        "correct_rate": None,
        "format_rate": None,
        "llm": None,
        "sd": None,
    }


def normalize_quality_evidence(
    raw: Any,
    *,
    expected_prompt_set: Mapping[str, Any] | None = None,
    expected_gemma_judge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a result-file ``quality_evidence`` payload into v1.

    The accepted input is intentionally smaller than the record representation.
    It rejects prose, prompt/output content, paths, reviewer identities, and
    unknown fields rather than persisting them opportunistically.
    """
    if raw is None:
        return not_collected_quality()
    value = _mapping(raw, "quality_evidence")
    _only_keys(
        value,
        {"llm", "sd", "gemma_judge", "not_collected_reason"},
        "quality_evidence",
    )
    llm_raw = value.get("llm")
    sd_raw = value.get("sd")
    gemma_raw = value.get("gemma_judge")
    if llm_raw is None and sd_raw is None and gemma_raw is None:
        return not_collected_quality(str(value.get("not_collected_reason", "not_provided")))
    if value.get("not_collected_reason") is not None:
        raise QualityEvidenceError("collected evidence cannot carry not_collected_reason")
    llm = _normalize_llm(llm_raw, expected_prompt_set) if llm_raw is not None else None
    sd = _normalize_sd(sd_raw) if sd_raw is not None else None
    gemma_judge = (
        _normalize_gemma_judge(gemma_raw, expected_gemma_judge)
        if gemma_raw is not None else None
    )
    normalized = {
        "schema_version": (
            QUALITY_SCHEMA_VERSION_V2 if gemma_judge is not None else QUALITY_SCHEMA_VERSION
        ),
        "status": "collected",
        "not_collected_reason": None,
        "correct_rate": llm["correctness"]["rate"] if llm else None,
        "format_rate": llm["format"]["rate"] if llm else None,
        "llm": llm,
        "sd": sd,
    }
    if gemma_judge is not None:
        normalized["gemma_judge"] = gemma_judge
    return normalized


def sd_evidence_from_gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Make a normalized SD input from an existing SD quality-gate report.

    Callers must explicitly choose to attach this returned value as result-file
    quality evidence.  The adapter reads only fixed, non-prose report fields.
    """
    value = _mapping(report, "sd quality-gate report")
    if value.get("schema_version") != 1:
        raise QualityEvidenceError("sd quality-gate report schema_version must be 1")
    automatic = _mapping(value.get("automatic_gate"), "sd quality-gate automatic_gate")
    manual = _mapping(value.get("manual_gate"), "sd quality-gate manual_gate")
    raw_status = value.get("status")
    if raw_status in {"pending_manual_review", "partial_pass"}:
        manual_status = "pending"
    elif raw_status == "failed":
        manual_status = "failed"
    elif manual.get("passed") is True:
        manual_status = "passed"
    elif manual.get("passed") is False:
        manual_status = "pending"
    else:
        manual_status = "not_required"
    output_value = automatic.get("output_count", automatic.get("outputs", value.get("images", 0)))
    unique_value = automatic.get(
        "unique_output_count",
        automatic.get("unique_images", automatic.get("unique_results", 0)),
    )
    return {
        "mode": value.get("mode", "text_to_image"),
        "asset_id": value.get("asset_id"),
        "artifact_id": value.get("artifact_id"),
        "source_schema_version": 1,
        "automatic_gate": {
            "passed": automatic.get("passed"),
            "output_count": _report_count(output_value, "sd quality-gate output_count"),
            "unique_output_count": _report_count(unique_value, "sd quality-gate unique_output_count"),
        },
        "manual_review": {
            "status": manual_status,
            "required_reviewers": manual.get("required_reviewers", 0),
        },
    }
