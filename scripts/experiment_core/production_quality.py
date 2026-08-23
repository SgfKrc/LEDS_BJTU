"""Fail-closed, read-only production audit for EX-N3 quality evidence.

This module promotes a completed experiment run into a production-quality
decision without executing a model.  It deliberately emits only identifiers,
hashes, counts, and status codes.  Raw logs, paths, prompts, completions, and
reviewer identities never cross this boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .collector import evaluate_gate, evaluate_quality_gate
from .plan import PlanError, PlanManifest, load_plan
from .quality import QualityEvidenceError, normalize_quality_evidence


PRODUCTION_QUALITY_SCHEMA_VERSION = "qlh.experiment_quality.production_gate.v1"
_QUALITY_CHECKS = {"llm", "sd", "gemma_judge"}


def _reason(
    code: str,
    *,
    experiment_id: str | None = None,
    check: str | None = None,
) -> dict[str, str]:
    """Return a bounded reason object with no untrusted input content."""
    result = {"code": code}
    if experiment_id is not None:
        result["experiment_id"] = experiment_id
    if check is not None:
        result["check"] = check
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _status_from_reasons(
    invalid: list[dict[str, str]],
    failed: list[dict[str, str]],
) -> str:
    if invalid:
        return "invalid"
    if failed:
        return "failed"
    return "passed"


def _expected_checks(plan: PlanManifest) -> list[str]:
    return sorted({check for unit in plan.units for check in unit.quality_checks})


def _policy_summary(plan: PlanManifest) -> dict[str, Any]:
    quality = plan.quality
    assert quality is not None
    result: dict[str, Any] = {
        "quality_required": quality.required,
        "checks": _expected_checks(plan),
    }
    if quality.calibration is not None:
        result["calibration"] = dict(quality.calibration)
    return result


def _decision(
    *,
    plan_id: str | None,
    plan_sha256: str | None,
    prompt_set: Mapping[str, Any] | None,
    records_sha256: str | None,
    records_count: int,
    policy: Mapping[str, Any] | None,
    coverage: list[dict[str, Any]],
    invalid: list[dict[str, str]],
    failed: list[dict[str, str]],
) -> dict[str, Any]:
    reasons = [*invalid, *failed]
    result: dict[str, Any] = {
        "schema_version": PRODUCTION_QUALITY_SCHEMA_VERSION,
        "status": _status_from_reasons(invalid, failed),
        "records": {
            "sha256": records_sha256,
            "count": records_count,
        },
        "coverage": coverage,
        "reasons": reasons,
    }
    if plan_id is not None and plan_sha256 is not None and prompt_set is not None:
        result["plan"] = {
            "id": plan_id,
            "sha256": plan_sha256,
            "prompt_set_id": prompt_set.get("id"),
            "prompt_set_sha256": prompt_set.get("sha256"),
        }
    if policy is not None:
        result["policy"] = dict(policy)
    return result


def _load_records(source: Path) -> tuple[bytes | None, list[Mapping[str, Any]], list[dict[str, str]]]:
    """Read JSONL without reflecting malformed content into audit output."""
    try:
        raw = source.read_bytes()
    except OSError:
        return None, [], [_reason("records_unreadable")]
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return raw, [], [_reason("records_not_utf8")]

    records: list[Mapping[str, Any]] = []
    reasons: list[dict[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(_reason("record_json_invalid"))
            continue
        if not isinstance(value, Mapping):
            reasons.append(_reason("record_not_object"))
            continue
        records.append(value)
    return raw, records, reasons


def _strict_mapping(value: Any, keys: set[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != keys:
        return None
    return value


def _raw_rate(value: Any) -> dict[str, Any] | None:
    item = _strict_mapping(value, {"evaluated_count", "passed_count", "rate"})
    if item is None:
        return None
    return {
        "evaluated_count": item["evaluated_count"],
        "passed_count": item["passed_count"],
    }


def _raw_manual_review(value: Any) -> dict[str, Any] | None:
    item = _strict_mapping(value, {"status", "required_reviewers"})
    if item is None:
        return None
    reviewers = item["required_reviewers"]
    if isinstance(reviewers, bool) or not isinstance(reviewers, int):
        return None
    return {"status": item["status"], "required_reviewers": reviewers}


def _stored_quality_to_raw(
    quality: Any,
    *,
    checks: tuple[str, ...],
) -> dict[str, Any] | None:
    """Reverse only the public normalized shape so it can be revalidated.

    This is intentionally not a general deserializer.  It accepts exactly the
    record shape emitted by ``normalize_quality_evidence`` and reconstructs the
    smaller raw contract expected by that function.
    """
    if not isinstance(quality, Mapping):
        return None
    schema = quality.get("schema_version")
    required_keys = {
        "schema_version", "status", "not_collected_reason", "correct_rate",
        "format_rate", "llm", "sd",
    }
    if schema == "qlh.experiment_quality.v2":
        required_keys.add("gemma_judge")
    elif schema != "qlh.experiment_quality.v1":
        return None
    if set(quality) != required_keys or quality.get("status") != "collected":
        return None
    if quality.get("not_collected_reason") is not None:
        return None

    raw: dict[str, Any] = {}
    seen_checks: set[str] = set()
    llm = quality.get("llm")
    if llm is not None:
        item = _strict_mapping(
            llm,
            {"prompt_set_id", "prompt_set_sha256", "correctness", "format"},
        )
        correctness = _raw_rate(item.get("correctness")) if item else None
        formatting = _raw_rate(item.get("format")) if item else None
        if item is None or correctness is None or formatting is None:
            return None
        raw["llm"] = {
            "prompt_set_id": item["prompt_set_id"],
            "prompt_set_sha256": item["prompt_set_sha256"],
            "correctness": correctness,
            "format": formatting,
        }
        seen_checks.add("llm")

    sd = quality.get("sd")
    if sd is not None:
        item = _strict_mapping(
            sd,
            {
                "mode", "asset_id", "artifact_id", "source_schema_version",
                "automatic_gate", "manual_review",
            },
        )
        automatic = (
            _strict_mapping(
                item.get("automatic_gate"),
                {"status", "output_count", "unique_output_count"},
            )
            if item else None
        )
        manual = _raw_manual_review(item.get("manual_review")) if item else None
        if item is None or automatic is None or manual is None:
            return None
        if automatic["status"] not in {"passed", "failed"}:
            return None
        raw["sd"] = {
            "mode": item["mode"],
            "asset_id": item["asset_id"],
            "artifact_id": item["artifact_id"],
            "source_schema_version": item["source_schema_version"],
            "automatic_gate": {
                "passed": automatic["status"] == "passed",
                "output_count": automatic["output_count"],
                "unique_output_count": automatic["unique_output_count"],
            },
            "manual_review": manual,
        }
        seen_checks.add("sd")

    gemma = quality.get("gemma_judge")
    if gemma is not None:
        item = _strict_mapping(
            gemma,
            {
                "model", "judge_contract_id", "judge_contract_sha256",
                "topic_hit", "key_element_coverage", "manual_review",
            },
        )
        topic_hit = _raw_rate(item.get("topic_hit")) if item else None
        coverage = _raw_rate(item.get("key_element_coverage")) if item else None
        manual_value = item.get("manual_review") if item else None
        manual = _raw_manual_review(manual_value) if manual_value is not None else None
        if item is None or topic_hit is None or coverage is None:
            return None
        gemma_raw = {
            "model": item["model"],
            "judge_contract_id": item["judge_contract_id"],
            "judge_contract_sha256": item["judge_contract_sha256"],
            "topic_hit": topic_hit,
            "key_element_coverage": coverage,
        }
        if manual_value is not None:
            if manual is None:
                return None
            gemma_raw["manual_review"] = manual
        raw["gemma_judge"] = gemma_raw
        seen_checks.add("gemma_judge")

    if seen_checks != set(checks):
        return None
    return raw


def _record_identity_matches(record: Mapping[str, Any], unit: Any, plan: PlanManifest) -> bool:
    prompt_set = record.get("prompt_set")
    return (
        record.get("plan_id") == plan.plan_id
        and isinstance(prompt_set, Mapping)
        and prompt_set.get("id") == plan.prompt_set.get("id")
        and prompt_set.get("sha256") == plan.prompt_set.get("sha256")
        and record.get("model") == dict(unit.model)
        and record.get("params") == dict(unit.params)
        and record.get("runs") == unit.runs
    )


def _record_calibration_matches(record: Mapping[str, Any], plan: PlanManifest) -> bool:
    expected = dict(plan.quality.calibration) if plan.quality and plan.quality.calibration else None
    return record.get("calibration") == expected


def _record_gate_passed(record: Mapping[str, Any], key: str, checks: tuple[str, ...] | None = None) -> bool:
    gate = record.get(key)
    if not isinstance(gate, Mapping) or gate.get("status") != "passed":
        return False
    if key == "quality_gate":
        return (
            gate.get("required") is True
            and gate.get("checks") == list(checks or ())
        )
    return True


def _recalculate_performance_gate(
    record: Mapping[str, Any],
    unit: Any,
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the plan-derived performance status, never trusting stored text."""
    if unit.gate is None:
        return "invalid"
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return "invalid"
    baseline_metrics = None
    if unit.baseline_experiment_id:
        baseline = records_by_id.get(unit.baseline_experiment_id)
        if not isinstance(baseline, Mapping):
            return "invalid"
        candidate = baseline.get("metrics")
        if not isinstance(candidate, Mapping):
            return "invalid"
        baseline_metrics = candidate
    try:
        status, _ = evaluate_gate(
            unit.gate,
            metrics,
            baseline_metrics=baseline_metrics,
        )
    except (TypeError, ValueError):
        return "invalid"
    return status


def audit_production_quality(
    plan_path: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    """Audit a completed EX-N3 run without changing it or running any model."""
    plan_source = Path(plan_path).expanduser()
    records_source = Path(records_path).expanduser()
    try:
        plan_raw = plan_source.read_bytes()
        plan = load_plan(plan_source)
        plan.verify_prompt_set()
    except (OSError, PlanError):
        return _decision(
            plan_id=None,
            plan_sha256=None,
            prompt_set=None,
            records_sha256=None,
            records_count=0,
            policy=None,
            coverage=[],
            invalid=[_reason("plan_contract_invalid")],
            failed=[],
        )

    records_raw, raw_records, invalid = _load_records(records_source)
    records_sha256 = _sha256(records_raw) if records_raw is not None else None
    if plan.quality is None or not plan.quality.required:
        invalid.append(_reason("plan_quality_not_required"))
        return _decision(
            plan_id=plan.plan_id,
            plan_sha256=_sha256(plan_raw),
            prompt_set=plan.prompt_set,
            records_sha256=records_sha256,
            records_count=len(raw_records),
            policy=None,
            coverage=[],
            invalid=invalid,
            failed=[],
        )

    units = {unit.experiment_id: unit for unit in plan.units}
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in raw_records:
        record_id = record.get("experiment_id")
        if not isinstance(record_id, str) or record_id not in units:
            invalid.append(_reason("record_not_declared"))
            continue
        if record_id in records_by_id:
            invalid.append(_reason("record_duplicate", experiment_id=record_id))
            continue
        records_by_id[record_id] = record

    failed: list[dict[str, str]] = []
    coverage: list[dict[str, Any]] = []
    for unit in plan.units:
        checks = unit.quality_checks
        entry: dict[str, Any] = {
            "experiment_id": unit.experiment_id,
            "checks": list(checks),
            "status": "failed",
        }
        record = records_by_id.get(unit.experiment_id)
        if record is None:
            failed.append(_reason("record_missing", experiment_id=unit.experiment_id))
            coverage.append(entry)
            continue
        if not _record_identity_matches(record, unit, plan):
            invalid.append(_reason("record_identity_mismatch", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        if not _record_calibration_matches(record, plan):
            invalid.append(_reason("record_calibration_mismatch", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        performance_status = _recalculate_performance_gate(record, unit, records_by_id)
        if performance_status == "invalid":
            invalid.append(_reason("performance_gate_invalid", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        if (
            performance_status != "passed"
            or not _record_gate_passed(record, "gate")
            or not _record_gate_passed(record, "performance_gate")
        ):
            failed.append(_reason("performance_gate_not_passed", experiment_id=unit.experiment_id))
            coverage.append(entry)
            continue
        if not _record_gate_passed(record, "quality_gate", checks):
            failed.append(_reason("recorded_quality_gate_not_passed", experiment_id=unit.experiment_id))
            coverage.append(entry)
            continue
        raw_quality = _stored_quality_to_raw(record.get("quality"), checks=checks)
        if raw_quality is None:
            invalid.append(_reason("quality_evidence_invalid", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        try:
            normalized = normalize_quality_evidence(
                raw_quality,
                expected_prompt_set=plan.prompt_set,
                expected_gemma_judge=plan.quality.gemma_judge,
            )
        except QualityEvidenceError:
            invalid.append(_reason("quality_evidence_invalid", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        if normalized != record.get("quality"):
            invalid.append(_reason("quality_evidence_not_canonical", experiment_id=unit.experiment_id))
            entry["status"] = "invalid"
            coverage.append(entry)
            continue
        quality_gate = evaluate_quality_gate(
            normalized,
            plan.quality,
            checks,
            baseline_record=(
                records_by_id.get(unit.baseline_experiment_id)
                if unit.baseline_experiment_id else None
            ),
        )
        if quality_gate.get("status") != "passed" or quality_gate.get("required") is not True:
            failed.append(_reason("quality_gate_not_passed", experiment_id=unit.experiment_id))
            coverage.append(entry)
            continue
        entry["status"] = "passed"
        coverage.append(entry)

    return _decision(
        plan_id=plan.plan_id,
        plan_sha256=_sha256(plan_raw),
        prompt_set=plan.prompt_set,
        records_sha256=records_sha256,
        records_count=len(raw_records),
        policy=_policy_summary(plan),
        coverage=coverage,
        invalid=invalid,
        failed=failed,
    )
