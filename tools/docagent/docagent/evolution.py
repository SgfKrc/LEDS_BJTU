"""Fail-closed rule evolution and approval gates for docagent."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .rules import RulesConfigError
from .rules_diff import diff_rules


EVOLUTION_SCHEMA_VERSION = "qlh.docagent.evolution.v1"
APPROVAL_SCHEMA_VERSION = "qlh.docagent.approval.v1"
EVOLUTION_STATES = ("proposed", "preflight", "approved", "rejected", "released")
_UNSET = object()


class EvolutionError(ValueError):
    """Raised when a rule evolution record or transition is invalid."""


class ApprovalRequired(EvolutionError):
    """Raised when a high-risk transition has no matching human approval."""

    def __init__(self, record: Mapping[str, Any]):
        super().__init__("human approval is required for this ruleset change")
        self.record = copy.deepcopy(dict(record))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvolutionError(f"{label} must be a non-empty string")
    return value.strip()


def _rule_id(path: object) -> str | None:
    if not isinstance(path, str):
        return None
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] == "rules" and parts[1]:
        return parts[1]
    return None


def _rule_level(value: object) -> str | None:
    if isinstance(value, Mapping) and value.get("level") in {"info", "warn", "error"}:
        return str(value["level"])
    return None


def _classify(diff: Mapping[str, Any], old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    high = False
    old_by_id = {rule["id"]: rule for rule in old["rules"]}
    new_by_id = {rule["id"]: rule for rule in new["rules"]}
    for item in diff["added"]:
        path = item["path"]
        if path.count(".") == 1 and _rule_level(item.get("value")) in {"warn", "error"}:
            high = True
            reasons.append(f"added_{item['value']['level']}_rule:{_rule_id(path)}")
    for item in diff["removed"]:
        path = item["path"]
        if path.count(".") == 1 and _rule_level(item.get("value")) in {"warn", "error"}:
            high = True
            reasons.append(f"removed_{item['value']['level']}_rule:{_rule_id(path)}")
    for item in diff["changed"]:
        path = item["path"]
        rule_id = _rule_id(path)
        if path.endswith(".level"):
            high = True
            reasons.append(f"severity_change:{rule_id}")
        elif path.endswith(".enabled") and rule_id:
            old_level = _rule_level(old_by_id.get(rule_id))
            new_level = _rule_level(new_by_id.get(rule_id))
            if old_level in {"warn", "error"} or new_level in {"warn", "error"}:
                high = True
                reasons.append(f"severity_rule_enabled_change:{rule_id}")
        elif path.count(".") == 1:
            high = True
            reasons.append(f"rule_shape_change:{rule_id}")
    return ("high" if high else "low", high, sorted(set(reasons)))


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EvolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_approval(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    required = {"schema_version", "decision", "actor", "timestamp", "note", "evolution_fingerprint"}
    if set(data) != required:
        raise EvolutionError("approval has an invalid field set")
    if data["schema_version"] != APPROVAL_SCHEMA_VERSION:
        raise EvolutionError(f"unsupported approval schema: {data['schema_version']!r}")
    if data["decision"] not in {"approved", "rejected"}:
        raise EvolutionError("approval.decision must be approved or rejected")
    for field in ("actor", "timestamp", "note"):
        data[field] = _non_empty_string(data[field], f"approval.{field}")
    data["evolution_fingerprint"] = _validate_digest(
        data["evolution_fingerprint"], "approval.evolution_fingerprint",
    )
    return data


def load_approval(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvolutionError(f"cannot read approval file: {source}") from exc
    except json.JSONDecodeError as exc:
        raise EvolutionError(f"approval file {source} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvolutionError(f"approval file {source} must contain a mapping")
    try:
        return validate_approval(payload)
    except EvolutionError as exc:
        raise EvolutionError(f"approval file {source}: {exc}") from exc


def _identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "change_note": record["change_note"],
        "old": record["old"],
        "new": record["new"],
        "diff": record["diff"],
        "risk": record["risk"],
        "requires_approval": record["requires_approval"],
        "risk_reasons": record["risk_reasons"],
    }


def _evolution_fingerprint(record: Mapping[str, Any]) -> str:
    return _fingerprint(_identity(record))


def create_evolution(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    change_note: str,
) -> dict[str, Any]:
    note = _non_empty_string(change_note, "change_note")
    if len(note) > 2000:
        raise EvolutionError("change_note must be at most 2000 characters")
    try:
        diff = diff_rules(old, new, old_source="old", new_source="new")
    except RulesConfigError as exc:
        raise EvolutionError(str(exc)) from exc
    risk, requires_approval, reasons = _classify(diff, old, new)
    record: dict[str, Any] = {
        "schema_version": EVOLUTION_SCHEMA_VERSION,
        "evolution_version": 1,
        "state": "proposed",
        "previous_state": None,
        "created_at": _now(),
        "updated_at": _now(),
        "change_note": note,
        "risk": risk,
        "requires_approval": requires_approval,
        "risk_reasons": reasons,
        "old": copy.deepcopy(diff["old"]),
        "new": copy.deepcopy(diff["new"]),
        "diff": {
            "changed": copy.deepcopy(diff["changed"]),
            "added": copy.deepcopy(diff["added"]),
            "removed": copy.deepcopy(diff["removed"]),
            "summary": copy.deepcopy(diff["summary"]),
        },
        "approval": None,
        "gate": "pending",
    }
    record["evolution_fingerprint"] = _evolution_fingerprint(record)
    return validate_evolution(record)


def validate_evolution(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    required = {
        "schema_version", "evolution_version", "state", "previous_state", "created_at", "updated_at",
        "change_note", "risk", "requires_approval", "risk_reasons", "old", "new", "diff",
        "approval", "gate", "evolution_fingerprint",
    }
    if set(data) != required:
        raise EvolutionError("evolution record has an invalid field set")
    if data["schema_version"] != EVOLUTION_SCHEMA_VERSION:
        raise EvolutionError(f"unsupported evolution schema: {data['schema_version']!r}")
    if data["evolution_version"] != 1:
        raise EvolutionError("unsupported evolution_version")
    if data["state"] not in EVOLUTION_STATES:
        raise EvolutionError(f"invalid evolution state: {data['state']!r}")
    if data["previous_state"] is not None and data["previous_state"] not in EVOLUTION_STATES:
        raise EvolutionError("evolution.previous_state is invalid")
    for field in ("created_at", "updated_at", "change_note"):
        data[field] = _non_empty_string(data[field], f"evolution.{field}")
    if data["risk"] not in {"low", "high"}:
        raise EvolutionError("evolution.risk must be low or high")
    if not isinstance(data["requires_approval"], bool):
        raise EvolutionError("evolution.requires_approval must be boolean")
    if data["requires_approval"] != (data["risk"] == "high"):
        raise EvolutionError("evolution.risk and requires_approval disagree")
    if not isinstance(data["risk_reasons"], list) or any(not isinstance(item, str) or not item for item in data["risk_reasons"]):
        raise EvolutionError("evolution.risk_reasons must be a string list")
    if data["risk"] == "high" and not data["risk_reasons"]:
        raise EvolutionError("high-risk evolution must include risk_reasons")
    for field in ("old", "new"):
        item = data[field]
        if not isinstance(item, Mapping):
            raise EvolutionError(f"evolution.{field} must be a mapping")
        if set(item) != {"source", "ruleset_version", "fingerprint"}:
            raise EvolutionError(f"evolution.{field} has an invalid metadata shape")
        for subfield in ("source", "ruleset_version"):
            if subfield not in item:
                raise EvolutionError(f"evolution.{field}.{subfield} is required")
        _non_empty_string(item["source"], f"evolution.{field}.source")
        if item["source"] != field:
            raise EvolutionError(f"evolution.{field}.source must be {field!r}")
        if not isinstance(item["ruleset_version"], int) or isinstance(item["ruleset_version"], bool) or item["ruleset_version"] < 1:
            raise EvolutionError(f"evolution.{field}.ruleset_version must be a positive integer")
        _validate_digest(item.get("fingerprint"), f"evolution.{field}.fingerprint")
    if not isinstance(data["diff"], Mapping) or set(data["diff"]) != {"changed", "added", "removed", "summary"}:
        raise EvolutionError("evolution.diff has an invalid shape")
    for field in ("changed", "added", "removed"):
        if not isinstance(data["diff"][field], list):
            raise EvolutionError(f"evolution.diff.{field} must be a list")
    if not isinstance(data["diff"]["summary"], Mapping) or set(data["diff"]["summary"]) != {
        "changed", "added", "removed", "identical",
    }:
        raise EvolutionError("evolution.diff.summary must be a mapping")
    summary = data["diff"]["summary"]
    if data["approval"] is not None:
        data["approval"] = validate_approval(data["approval"])
    if data["gate"] not in {"pending", "pending_approval", "passed", "rejected"}:
        raise EvolutionError("evolution.gate is invalid")
    for field in ("changed", "added", "removed"):
        if summary[field] != len(data["diff"][field]):
            raise EvolutionError(f"evolution.diff.summary.{field} does not match the diff")
    if not isinstance(summary["identical"], bool) or summary["identical"] != (
        not any(data["diff"][field] for field in ("changed", "added", "removed"))
    ):
        raise EvolutionError("evolution.diff.summary.identical is inconsistent")
    expected_gates = {
        "proposed": {"pending"},
        "preflight": {"pending", "pending_approval"},
        "approved": {"passed"},
        "rejected": {"rejected"},
        "released": {"passed"},
    }
    if data["gate"] not in expected_gates[data["state"]]:
        raise EvolutionError("evolution.gate is inconsistent with state")
    expected = _evolution_fingerprint(data)
    if data["evolution_fingerprint"] != expected:
        raise EvolutionError("evolution_fingerprint does not match the rule change")
    data["evolution_fingerprint"] = _validate_digest(
        data["evolution_fingerprint"], "evolution.evolution_fingerprint",
    )
    if data["approval"] is not None and data["approval"]["evolution_fingerprint"] != data["evolution_fingerprint"]:
        raise EvolutionError("approval does not match evolution_fingerprint")
    return data


def _with_state(
    record: Mapping[str, Any],
    state: str,
    *,
    gate: str,
    approval: Mapping[str, Any] | None | object = _UNSET,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result["previous_state"] = result["state"]
    result["state"] = state
    result["updated_at"] = _now()
    result["gate"] = gate
    if approval is not _UNSET:
        if approval is not None:
            result["approval"] = copy.deepcopy(dict(approval))
        else:
            result["approval"] = None
    return validate_evolution(result)


def transition_evolution(
    record: Mapping[str, Any],
    target_state: str,
    approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = validate_evolution(record)
    if target_state not in EVOLUTION_STATES:
        raise EvolutionError(f"invalid target state: {target_state!r}")
    if target_state == current["state"]:
        return current
    allowed = {
        "proposed": {"preflight"},
        "preflight": {"approved", "rejected"},
        "approved": {"released", "rejected"},
        "rejected": {"proposed"},
        "released": set(),
    }
    if target_state not in allowed[current["state"]]:
        raise EvolutionError(f"invalid evolution transition: {current['state']} -> {target_state}")
    if target_state == "approved":
        if current["requires_approval"]:
            if approval is None:
                pending = copy.deepcopy(current)
                pending["gate"] = "pending_approval"
                raise ApprovalRequired(validate_evolution(pending))
            selected = validate_approval(approval)
            if selected["decision"] != "approved":
                raise EvolutionError("an approved transition requires approval.decision=approved")
        else:
            selected = validate_approval(approval) if approval is not None else {
                "schema_version": APPROVAL_SCHEMA_VERSION,
                "decision": "approved",
                "actor": "docagent:auto",
                "timestamp": _now(),
                "note": "low-risk ruleset change auto-approved",
                "evolution_fingerprint": current["evolution_fingerprint"],
            }
        return _with_state(current, "approved", gate="passed", approval=selected)
    if target_state == "rejected":
        if approval is None:
            raise ApprovalRequired(current)
        selected = validate_approval(approval)
        if selected["decision"] != "rejected":
            raise EvolutionError("a rejected transition requires approval.decision=rejected")
        return _with_state(current, "rejected", gate="rejected", approval=selected)
    if target_state == "released":
        if current["approval"] is None or current["approval"]["decision"] != "approved":
            raise EvolutionError("released state requires a prior approved transition")
        return _with_state(current, "released", gate="passed")
    if target_state == "proposed":
        return _with_state(current, "proposed", gate="pending", approval=None)
    return _with_state(current, "preflight", gate="pending")


def write_evolution(path: str | Path, record: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    payload = validate_evolution(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def load_evolution(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvolutionError(f"cannot read evolution record: {source}") from exc
    except json.JSONDecodeError as exc:
        raise EvolutionError(f"evolution record {source} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvolutionError(f"evolution record {source} must contain a mapping")
    try:
        return validate_evolution(payload)
    except EvolutionError as exc:
        raise EvolutionError(f"evolution record {source}: {exc}") from exc


__all__ = [
    "APPROVAL_SCHEMA_VERSION",
    "ApprovalRequired",
    "EVOLUTION_SCHEMA_VERSION",
    "EVOLUTION_STATES",
    "EvolutionError",
    "create_evolution",
    "load_approval",
    "load_evolution",
    "transition_evolution",
    "validate_approval",
    "validate_evolution",
    "write_evolution",
]
