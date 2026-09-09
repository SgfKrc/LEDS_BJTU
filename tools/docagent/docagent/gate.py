"""Integrity contracts for reports, rules, evolution records, and CI gates."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .baseline import BaselineError, ensure_matches, validate_baseline
from .evolution import validate_evolution
from .profile import ProfileConfigError, apply_profile, load_profile, profile_metadata
from .rules import RulesConfigError, load_rules, rules_fingerprint


GATE_SCHEMA_VERSION = "qlh.docagent.gate.v1"
VERIFY_SCHEMA_VERSION = "qlh.docagent.gate-verify.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_FIELDS = {"run_ts", "rules", "ruleset_version", "rules_fingerprint", "profile", "docs"}


class GateError(ValueError):
    """Raised when an integrity artifact cannot be safely used."""


class GateMismatch(GateError):
    """Raised when an artifact is valid but does not match current inputs."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_doc_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise GateError(f"{label} must be a repository-relative path")
    return path.as_posix()


def _validate_profile(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"name", "schema_version", "profile_version", "docs_dir"}:
        raise GateError("report.profile has an invalid metadata shape")
    version = value["profile_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise GateError("report.profile.profile_version must be a positive integer")
    for field in ("name", "schema_version"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise GateError(f"report.profile.{field} must be a non-empty string")
    return {
        "name": value["name"],
        "schema_version": value["schema_version"],
        "profile_version": version,
        "docs_dir": _safe_doc_path(value["docs_dir"], "report.profile.docs_dir"),
    }


def _validate_report_documents(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise GateError("report.docs must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"doc", "status_line", "updated_at", "sha256", "findings"}:
            raise GateError(f"report.docs[{index}] has an invalid document shape")
        doc = _safe_doc_path(item["doc"], f"report.docs[{index}].doc")
        if doc in seen:
            raise GateError(f"report contains duplicate document: {doc}")
        seen.add(doc)
        if not isinstance(item["status_line"], str):
            raise GateError(f"report.docs[{index}].status_line must be a string")
        if item["updated_at"] is not None and not isinstance(item["updated_at"], str):
            raise GateError(f"report.docs[{index}].updated_at must be a string or null")
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise GateError(f"report.docs[{index}].sha256 must be a lowercase SHA-256 digest")
        if not isinstance(item["findings"], list):
            raise GateError(f"report.docs[{index}].findings must be a list")
        for finding_index, finding in enumerate(item["findings"]):
            if not isinstance(finding, Mapping) or set(finding) != {"rule", "level", "message"}:
                raise GateError(
                    f"report.docs[{index}].findings[{finding_index}] has an invalid shape"
                )
            if any(not isinstance(finding[field], str) or not finding[field].strip() for field in finding):
                raise GateError(f"report.docs[{index}].findings[{finding_index}] has invalid values")
        record = copy.deepcopy(dict(item))
        record["doc"] = doc
        result.append(record)
    return result


def _report_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(report))
    identity.pop("report_fingerprint", None)
    return identity


def report_fingerprint(report: Mapping[str, Any]) -> str:
    """Return the self-integrity digest of a scanner report."""
    return _digest(_report_identity(report))


def validate_report(payload: Mapping[str, Any], *, require_fingerprint: bool = False) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    allowed = _REPORT_FIELDS | {"report_fingerprint"}
    missing = sorted(_REPORT_FIELDS - set(data))
    unknown = sorted(set(data) - allowed)
    if missing:
        raise GateError(f"report missing fields: {', '.join(missing)}")
    if unknown:
        raise GateError(f"report has unknown fields: {', '.join(unknown)}")
    if require_fingerprint and "report_fingerprint" not in data:
        raise GateError("report.report_fingerprint is required")
    if not isinstance(data["run_ts"], str) or not data["run_ts"].strip():
        raise GateError("report.run_ts must be a non-empty string")
    if not isinstance(data["rules"], Mapping) or not data["rules"]:
        raise GateError("report.rules must be a non-empty mapping")
    if isinstance(data["ruleset_version"], bool) or not isinstance(data["ruleset_version"], int) or data["ruleset_version"] < 1:
        raise GateError("report.ruleset_version must be a positive integer")
    if not isinstance(data["rules_fingerprint"], str) or not _SHA256.fullmatch(data["rules_fingerprint"]):
        raise GateError("report.rules_fingerprint must be a lowercase SHA-256 digest")
    data["profile"] = _validate_profile(data["profile"])
    data["docs"] = _validate_report_documents(data["docs"])
    if "report_fingerprint" in data:
        if not isinstance(data["report_fingerprint"], str) or not _SHA256.fullmatch(data["report_fingerprint"]):
            raise GateError("report.report_fingerprint must be a lowercase SHA-256 digest")
        if data["report_fingerprint"] != report_fingerprint(data):
            raise GateMismatch("report fingerprint mismatch: report content was modified")
    return data


def _gate_identity(gate: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(gate))
    identity.pop("gate_fingerprint", None)
    return identity


def validate_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    required = {
        "schema_version", "gate_version", "state", "created_at", "report", "rules",
        "profile", "baseline", "evolution", "gate_fingerprint",
    }
    if set(data) != required:
        raise GateError("gate record has an invalid field set")
    if data["schema_version"] != GATE_SCHEMA_VERSION or data["gate_version"] != 1:
        raise GateError("unsupported gate schema")
    if data["state"] != "passed":
        raise GateError("gate.state must be passed")
    if not isinstance(data["created_at"], str) or not data["created_at"].strip():
        raise GateError("gate.created_at must be a non-empty string")
    for section in ("report", "rules"):
        item = data[section]
        if not isinstance(item, Mapping):
            raise GateError(f"gate.{section} must be a mapping")
        expected = {"fingerprint", "run_ts"} if section == "report" else {"fingerprint", "ruleset_version"}
        if set(item) != expected:
            raise GateError(f"gate.{section} has an invalid shape")
        if not isinstance(item["fingerprint"], str) or not _SHA256.fullmatch(item["fingerprint"]):
            raise GateError(f"gate.{section}.fingerprint must be a lowercase SHA-256 digest")
    if not isinstance(data["report"]["run_ts"], str) or not data["report"]["run_ts"].strip():
        raise GateError("gate.report.run_ts must be a non-empty string")
    if isinstance(data["rules"]["ruleset_version"], bool) or not isinstance(data["rules"]["ruleset_version"], int) or data["rules"]["ruleset_version"] < 1:
        raise GateError("gate.rules.ruleset_version must be a positive integer")
    data["profile"] = _validate_profile(data["profile"])
    for field in ("baseline", "evolution"):
        if data[field] is not None and not isinstance(data[field], Mapping):
            raise GateError(f"gate.{field} must be a mapping or null")
    if data["baseline"] is not None and set(data["baseline"]) != {"created_at", "rules_fingerprint"}:
        raise GateError("gate.baseline has an invalid shape")
    if data["evolution"] is not None and set(data["evolution"]) != {
        "evolution_fingerprint", "new_rules_fingerprint", "state",
    }:
        raise GateError("gate.evolution has an invalid shape")
    for value, label in (
        (data["gate_fingerprint"], "gate.gate_fingerprint"),
        (data["baseline"]["rules_fingerprint"] if data["baseline"] else None, "gate.baseline.rules_fingerprint"),
        (data["evolution"]["evolution_fingerprint"] if data["evolution"] else None, "gate.evolution.evolution_fingerprint"),
        (data["evolution"]["new_rules_fingerprint"] if data["evolution"] else None, "gate.evolution.new_rules_fingerprint"),
    ):
        if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
            raise GateError(f"{label} must be a lowercase SHA-256 digest")
    if data["evolution"] is not None and data["evolution"]["state"] not in {"approved", "released"}:
        raise GateError("gate.evolution.state must be approved or released")
    if data["gate_fingerprint"] != _digest(_gate_identity(data)):
        raise GateMismatch("gate fingerprint mismatch: gate artifact was modified")
    return data


def build_gate_record(
    report: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
    evolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_report = validate_report(report, require_fingerprint=True)
    selected_baseline = validate_baseline(baseline) if baseline is not None else None
    if selected_baseline is not None:
        try:
            ensure_matches(selected_report, selected_baseline)
        except BaselineError as exc:
            raise GateError(str(exc)) from exc
    selected_evolution = validate_evolution(evolution) if evolution is not None else None
    if selected_evolution is not None and selected_evolution["new"]["fingerprint"] != selected_report["rules_fingerprint"]:
        raise GateError("evolution new rules fingerprint does not match report rules fingerprint")
    payload: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_version": 1,
        "state": "passed",
        "created_at": _now(),
        "report": {
            "run_ts": selected_report["run_ts"],
            "fingerprint": selected_report["report_fingerprint"],
        },
        "rules": {
            "ruleset_version": selected_report["ruleset_version"],
            "fingerprint": selected_report["rules_fingerprint"],
        },
        "profile": copy.deepcopy(selected_report["profile"]),
        "baseline": (
            {
                "created_at": selected_baseline["created_at"],
                "rules_fingerprint": selected_baseline["rules_fingerprint"],
            }
            if selected_baseline is not None else None
        ),
        "evolution": (
            {
                "evolution_fingerprint": selected_evolution["evolution_fingerprint"],
                "new_rules_fingerprint": selected_evolution["new"]["fingerprint"],
                "state": selected_evolution["state"],
            }
            if selected_evolution is not None else None
        ),
    }
    payload["gate_fingerprint"] = _digest(payload)
    return validate_gate(payload)


def write_gate(path: str | Path, gate: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    payload = validate_gate(gate)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"cannot read report file: {source}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"report file {source} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise GateError(f"report file {source} must contain a mapping")
    try:
        return validate_report(payload, require_fingerprint=True)
    except GateMismatch:
        raise
    except GateError as exc:
        raise GateError(f"report file {source}: {exc}") from exc


def load_gate(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"cannot read gate file: {source}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"gate file {source} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise GateError(f"gate file {source} must contain a mapping")
    try:
        return validate_gate(payload)
    except GateMismatch:
        raise
    except GateError as exc:
        raise GateError(f"gate file {source}: {exc}") from exc


def _current_rules(report: Mapping[str, Any], rules_path: str | Path, profile: str | Path | Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw_rules = load_rules(rules_path)
        selected_profile = load_profile(profile if profile is not None else str(report["profile"]["name"]))
        configured = apply_profile(raw_rules, selected_profile)
    except (RulesConfigError, ProfileConfigError) as exc:
        raise GateError(str(exc)) from exc
    return configured, selected_profile


def verify_gate(
    report_path: str | Path,
    rules_path: str | Path,
    *,
    profile: str | Path | Mapping[str, Any] | None = None,
    gate_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    evolution_path: str | Path | None = None,
) -> dict[str, Any]:
    report = load_report(report_path)
    configured, selected_profile = _current_rules(report, rules_path, profile)
    actual_rules_fingerprint = rules_fingerprint(configured)
    actual_profile = profile_metadata(selected_profile)
    if actual_rules_fingerprint != report["rules_fingerprint"]:
        raise GateMismatch(
            "rules fingerprint mismatch: "
            f"report={report['rules_fingerprint']} current={actual_rules_fingerprint}"
        )
    if configured["ruleset_version"] != report["ruleset_version"]:
        raise GateMismatch("ruleset version mismatch between report and rules file")
    if actual_profile != report["profile"]:
        raise GateMismatch("profile metadata mismatch between report and selected profile")
    selected_gate = load_gate(gate_path) if gate_path is not None else None
    baseline = load_baseline(baseline_path) if baseline_path is not None else None
    evolution = load_evolution(evolution_path) if evolution_path is not None else None
    if baseline is not None:
        try:
            ensure_matches(report, baseline)
        except BaselineError as exc:
            raise GateMismatch(str(exc)) from exc
    if evolution is not None and evolution["new"]["fingerprint"] != report["rules_fingerprint"]:
        raise GateMismatch("evolution new rules fingerprint does not match report rules fingerprint")
    if selected_gate is not None:
        if selected_gate["report"]["run_ts"] != report["run_ts"]:
            raise GateMismatch("gate/report run timestamp mismatch")
        if selected_gate["report"]["fingerprint"] != report["report_fingerprint"]:
            raise GateMismatch("gate/report fingerprint mismatch")
        if selected_gate["rules"]["fingerprint"] != actual_rules_fingerprint:
            raise GateMismatch("gate/rules fingerprint mismatch")
        if selected_gate["rules"]["ruleset_version"] != report["ruleset_version"]:
            raise GateMismatch("gate/ruleset version mismatch")
        if selected_gate["profile"] != actual_profile:
            raise GateMismatch("gate/profile metadata mismatch")
        if selected_gate["baseline"] is not None and baseline is None:
            raise GateMismatch("gate artifact requires --baseline")
        if selected_gate["evolution"] is not None and evolution is None:
            raise GateMismatch("gate artifact requires --evolution")
        if baseline is not None:
            if selected_gate["baseline"] is None or selected_gate["baseline"]["rules_fingerprint"] != baseline["rules_fingerprint"]:
                raise GateMismatch("gate/baseline fingerprint mismatch")
        if evolution is not None:
            if selected_gate["evolution"] is None or selected_gate["evolution"]["evolution_fingerprint"] != evolution["evolution_fingerprint"]:
                raise GateMismatch("gate/evolution fingerprint mismatch")
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "passed",
        "report_fingerprint": report["report_fingerprint"],
        "rules_fingerprint": actual_rules_fingerprint,
        "ruleset_version": report["ruleset_version"],
        "profile": actual_profile,
        "checks": {
            "report": True,
            "rules": True,
            "profile": True,
            "baseline": baseline is not None,
            "evolution": evolution is not None,
            "gate": selected_gate is not None,
        },
    }


__all__ = [
    "GATE_SCHEMA_VERSION",
    "VERIFY_SCHEMA_VERSION",
    "GateError",
    "GateMismatch",
    "build_gate_record",
    "load_gate",
    "load_report",
    "report_fingerprint",
    "validate_gate",
    "validate_report",
    "verify_gate",
    "write_gate",
]
