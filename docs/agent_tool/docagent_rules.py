"""Load and validate the versioned docagent rules contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RULES_SCHEMA_VERSION = "qlh.docagent.rules.v1"
DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")
RULE_IDS = frozenset({"R1", "R2", "R3", "R4", "R5"})
RULE_LEVELS = frozenset({"info", "warn", "error"})
REQUIRED_RULE_FIELDS = frozenset({"id", "name", "level", "enabled", "description", "parameters"})


class RulesConfigError(ValueError):
    """Raised when a rules file cannot be safely used."""


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


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RulesConfigError(f"{label} must be a mapping")
    return value


def _require_string_list(value: object, label: str) -> None:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RulesConfigError(f"{label} must be a non-empty string list")


def _validate_rule_parameters(rule_id: str, parameters: Mapping[str, Any]) -> None:
    if rule_id == "R1":
        _require_string_list(parameters.get("stale_status_hints"), "R1.stale_status_hints")
        _require_string_list(parameters.get("done_markers"), "R1.done_markers")
        for field in ("ignore_status_done_markers", "status_stem_parentheses"):
            if not isinstance(parameters.get(field), bool):
                raise RulesConfigError(f"R1.{field} must be boolean")
    elif rule_id == "R2":
        if not isinstance(parameters.get("git_scope"), str) or not parameters["git_scope"].strip():
            raise RulesConfigError("R2.git_scope must be a non-empty string")
        if not isinstance(parameters.get("only_current_document"), bool):
            raise RulesConfigError("R2.only_current_document must be boolean")
    elif rule_id == "R3":
        for field in ("source_root", "match_mode"):
            if not isinstance(parameters.get(field), str) or not parameters[field].strip():
                raise RulesConfigError(f"R3.{field} must be a non-empty string")
        if not isinstance(parameters.get("ignore_commit_subject"), bool):
            raise RulesConfigError("R3.ignore_commit_subject must be boolean")
        _require_string_list(parameters.get("topic_stop_words"), "R3.topic_stop_words")
    elif rule_id == "R4":
        if not isinstance(parameters.get("relative_root"), str) or not parameters["relative_root"].strip():
            raise RulesConfigError("R4.relative_root must be a non-empty string")
        _require_string_list(parameters.get("ignored_schemes"), "R4.ignored_schemes")
        if not isinstance(parameters.get("decode_url_path"), bool):
            raise RulesConfigError("R4.decode_url_path must be boolean")
    elif rule_id == "R5":
        for field in ("status_pattern", "lifecycle_pattern"):
            if not isinstance(parameters.get(field), str) or not parameters[field].strip():
                raise RulesConfigError(f"R5.{field} must be a non-empty string")
        if not isinstance(parameters.get("ignore_lifecycle_heading"), bool):
            raise RulesConfigError("R5.ignore_lifecycle_heading must be boolean")


def validate_rules(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a v1 rules payload without mutating its input."""
    data = copy.deepcopy(dict(payload))
    if data.get("schema_version") != RULES_SCHEMA_VERSION:
        raise RulesConfigError(
            f"unsupported rules schema: {data.get('schema_version')!r}; expected {RULES_SCHEMA_VERSION}"
        )
    version = data.get("ruleset_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise RulesConfigError("ruleset_version must be a positive integer")

    defaults = _require_mapping(data.get("defaults"), "defaults")
    status_window = defaults.get("status_window_lines")
    body_window = defaults.get("body_window_lines")
    levels = defaults.get("levels")
    if isinstance(status_window, bool) or not isinstance(status_window, int) or status_window < 1:
        raise RulesConfigError("defaults.status_window_lines must be a positive integer")
    if isinstance(body_window, bool) or not isinstance(body_window, int) or body_window < status_window:
        raise RulesConfigError("defaults.body_window_lines must be >= status_window_lines")
    if not isinstance(levels, list) or not levels or any(level not in RULE_LEVELS for level in levels):
        raise RulesConfigError("defaults.levels must be a non-empty list of valid levels")
    _require_mapping(data.get("exemptions"), "exemptions")

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise RulesConfigError("rules must be a list")
    seen: set[str] = set()
    normalized_rules = []
    for index, raw_rule in enumerate(raw_rules):
        rule = _require_mapping(raw_rule, f"rules[{index}]")
        missing = sorted(REQUIRED_RULE_FIELDS - set(rule))
        if missing:
            raise RulesConfigError(f"rules[{index}] missing fields: {', '.join(missing)}")
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or rule_id not in RULE_IDS:
            raise RulesConfigError(f"rules[{index}] has unknown rule id: {rule_id!r}")
        if rule_id in seen:
            raise RulesConfigError(f"duplicate rule id: {rule_id}")
        seen.add(rule_id)
        if rule.get("level") not in RULE_LEVELS:
            raise RulesConfigError(f"rule {rule_id} has invalid level: {rule.get('level')!r}")
        if not isinstance(rule.get("enabled"), bool):
            raise RulesConfigError(f"rule {rule_id}.enabled must be boolean")
        if not isinstance(rule.get("name"), str) or not str(rule.get("name")).strip():
            raise RulesConfigError(f"rule {rule_id}.name must be non-empty")
        if not isinstance(rule.get("description"), str) or not str(rule.get("description")).strip():
            raise RulesConfigError(f"rule {rule_id}.description must be non-empty")
        parameters = _require_mapping(rule.get("parameters"), f"rule {rule_id}.parameters")
        _validate_rule_parameters(rule_id, parameters)
        normalized_rules.append(dict(rule))

    if seen != RULE_IDS:
        missing = sorted(RULE_IDS - seen)
        raise RulesConfigError(f"rules missing required IDs: {', '.join(missing)}")
    data["rules"] = sorted(normalized_rules, key=lambda item: item["id"])
    return data


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load the validated rules contract from the default or explicit path."""
    source = Path(path) if path is not None else DEFAULT_RULES_PATH
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RulesConfigError(f"cannot read rules file: {source}") from exc
    return validate_rules(_parse_payload(text, source))


def rules_fingerprint(rules: Mapping[str, Any] | None = None) -> str:
    """Return a stable SHA-256 fingerprint for a validated rules payload."""
    normalized = validate_rules(rules if rules is not None else load_rules())
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_rule(rule_id: str, rules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return one validated rule by ID, or fail closed for unknown IDs."""
    requested = str(rule_id or "").strip()
    payload = load_rules() if rules is None else validate_rules(rules)
    for rule in payload["rules"]:
        if rule["id"] == requested:
            return copy.deepcopy(rule)
    raise RulesConfigError(f"unknown rule id: {requested!r}")


__all__ = [
    "DEFAULT_RULES_PATH",
    "RULE_IDS",
    "RULE_LEVELS",
    "RULES_SCHEMA_VERSION",
    "RulesConfigError",
    "get_rule",
    "load_rules",
    "rules_fingerprint",
    "validate_rules",
]
