"""Stdlib-first loader and validator for the standalone rules contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RULES_SCHEMA_VERSION = "qlh.docagent.rules.v1"
DEFAULT_RULES_PATH = Path(__file__).with_name("data") / "rules.yaml"
RULE_IDS = frozenset({"R1", "R2", "R3", "R4", "R5"})
RULE_LEVELS = frozenset({"info", "warn", "error"})
REQUIRED_RULE_FIELDS = frozenset({"id", "name", "level", "enabled", "description", "parameters"})


class RulesConfigError(ValueError):
    """Raised when rules cannot be safely used by the scanner."""


def _parse_payload(text: str, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RulesConfigError(
                f"rules file {source} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        try:
            value = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            raise RulesConfigError(f"rules file {source} is not valid YAML") from exc
        if value is None:
            raise RulesConfigError(f"rules file {source} is empty") from json_error
    if not isinstance(value, dict):
        raise RulesConfigError("rules document must be a mapping")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RulesConfigError(f"{label} must be a mapping")
    return value


def _string_list(value: object, label: str) -> None:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RulesConfigError(f"{label} must be a non-empty string list")


def _validate_parameters(rule_id: str, params: Mapping[str, Any]) -> None:
    if rule_id == "R1":
        _string_list(params.get("stale_status_hints"), "R1.stale_status_hints")
        _string_list(params.get("done_markers"), "R1.done_markers")
        for field in ("ignore_status_done_markers", "status_stem_parentheses"):
            if not isinstance(params.get(field), bool):
                raise RulesConfigError(f"R1.{field} must be boolean")
    elif rule_id == "R2":
        if not isinstance(params.get("git_scope"), str) or not params["git_scope"].strip():
            raise RulesConfigError("R2.git_scope must be a non-empty string")
        if not isinstance(params.get("only_current_document"), bool):
            raise RulesConfigError("R2.only_current_document must be boolean")
    elif rule_id == "R3":
        for field in ("source_root", "match_mode"):
            if not isinstance(params.get(field), str) or not params[field].strip():
                raise RulesConfigError(f"R3.{field} must be a non-empty string")
        if not isinstance(params.get("ignore_commit_subject"), bool):
            raise RulesConfigError("R3.ignore_commit_subject must be boolean")
        _string_list(params.get("topic_stop_words"), "R3.topic_stop_words")
    elif rule_id == "R4":
        if not isinstance(params.get("relative_root"), str) or not params["relative_root"].strip():
            raise RulesConfigError("R4.relative_root must be a non-empty string")
        _string_list(params.get("ignored_schemes"), "R4.ignored_schemes")
        if not isinstance(params.get("decode_url_path"), bool):
            raise RulesConfigError("R4.decode_url_path must be boolean")
    elif rule_id == "R5":
        for field in ("status_pattern", "lifecycle_pattern"):
            if not isinstance(params.get(field), str) or not params[field].strip():
                raise RulesConfigError(f"R5.{field} must be a non-empty string")
        if not isinstance(params.get("ignore_lifecycle_heading"), bool):
            raise RulesConfigError("R5.ignore_lifecycle_heading must be boolean")


def validate_rules(
    payload: Mapping[str, Any],
    *,
    allow_rule_set_changes: bool = False,
) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    if data.get("schema_version") != RULES_SCHEMA_VERSION:
        raise RulesConfigError(
            f"unsupported rules schema: {data.get('schema_version')!r}; expected {RULES_SCHEMA_VERSION}"
        )
    version = data.get("ruleset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise RulesConfigError("ruleset_version must be a positive integer")

    defaults = _mapping(data.get("defaults"), "defaults")
    status_window = defaults.get("status_window_lines")
    body_window = defaults.get("body_window_lines")
    levels = defaults.get("levels")
    if isinstance(status_window, bool) or not isinstance(status_window, int) or status_window < 1:
        raise RulesConfigError("defaults.status_window_lines must be a positive integer")
    if isinstance(body_window, bool) or not isinstance(body_window, int) or body_window < status_window:
        raise RulesConfigError("defaults.body_window_lines must be >= status_window_lines")
    if not isinstance(levels, list) or not levels or any(level not in RULE_LEVELS for level in levels):
        raise RulesConfigError("defaults.levels must be a non-empty list of valid levels")
    _mapping(data.get("exemptions"), "exemptions")

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise RulesConfigError("rules must be a list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _mapping(raw_rule, f"rules[{index}]")
        missing = sorted(REQUIRED_RULE_FIELDS - set(rule))
        if missing:
            raise RulesConfigError(f"rules[{index}] missing fields: {', '.join(missing)}")
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise RulesConfigError(f"rules[{index}] has unknown rule id: {rule_id!r}")
        if not allow_rule_set_changes and rule_id not in RULE_IDS:
            raise RulesConfigError(f"rules[{index}] has unknown rule id: {rule_id!r}")
        if rule_id in seen:
            raise RulesConfigError(f"duplicate rule id: {rule_id}")
        seen.add(rule_id)
        if rule.get("level") not in RULE_LEVELS:
            raise RulesConfigError(f"rule {rule_id} has invalid level: {rule.get('level')!r}")
        if not isinstance(rule.get("enabled"), bool):
            raise RulesConfigError(f"rule {rule_id}.enabled must be boolean")
        if not isinstance(rule.get("name"), str) or not rule["name"].strip():
            raise RulesConfigError(f"rule {rule_id}.name must be non-empty")
        if not isinstance(rule.get("description"), str) or not rule["description"].strip():
            raise RulesConfigError(f"rule {rule_id}.description must be non-empty")
        params = _mapping(rule.get("parameters"), f"rule {rule_id}.parameters")
        if rule_id in RULE_IDS:
            _validate_parameters(rule_id, params)
        normalized.append(dict(rule))
    if not normalized:
        raise RulesConfigError("rules must contain at least one rule")
    if not allow_rule_set_changes and seen != RULE_IDS:
        raise RulesConfigError(f"rules missing required IDs: {', '.join(sorted(RULE_IDS - seen))}")
    data["rules"] = sorted(normalized, key=lambda item: item["id"])
    return data


def load_rules(
    path: str | Path | None = None,
    *,
    allow_rule_set_changes: bool = False,
) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_RULES_PATH
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RulesConfigError(f"cannot read rules file: {source}") from exc
    try:
        return validate_rules(
            _parse_payload(text, source),
            allow_rule_set_changes=allow_rule_set_changes,
        )
    except RulesConfigError as exc:
        raise RulesConfigError(f"rules file {source}: {exc}") from exc


def rules_fingerprint(
    rules: Mapping[str, Any] | None = None,
    *,
    allow_rule_set_changes: bool = False,
) -> str:
    normalized = validate_rules(
        rules if rules is not None else load_rules(),
        allow_rule_set_changes=allow_rule_set_changes,
    )
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_RULES_PATH",
    "RULES_SCHEMA_VERSION",
    "RulesConfigError",
    "load_rules",
    "rules_fingerprint",
    "validate_rules",
]
