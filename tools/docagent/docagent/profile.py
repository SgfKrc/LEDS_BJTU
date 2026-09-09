"""Profile contract for adapting docagent to different repositories."""

from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from .rules import validate_rules


PROFILE_SCHEMA_VERSION = "qlh.docagent.profile.v1"
DEFAULT_PROFILE_NAME = "qlh"
DEFAULT_PROFILES_DIR = Path(__file__).with_name("data") / "profiles"
PROFILE_FIELDS = frozenset({
    "schema_version", "profile_version", "name", "docs_dir", "exclude", "vocabulary", "git",
})
VOCABULARY_FIELDS = frozenset({
    "status_exemptions", "stale_status_hints", "done_markers", "topic_stop_words",
})


class ProfileConfigError(ValueError):
    """Raised when a profile cannot be safely used by the scanner."""


def _parse_payload(text: str, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ProfileConfigError(
                f"profile file {source} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        try:
            value = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            raise ProfileConfigError(f"profile file {source} is not valid YAML") from exc
        if value is None:
            raise ProfileConfigError(f"profile file {source} is empty") from json_error
    if not isinstance(value, dict):
        raise ProfileConfigError("profile document must be a mapping")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileConfigError(f"{label} must be a mapping")
    return value


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileConfigError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/").strip().strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ProfileConfigError(f"{label} must stay within the repository")
    return path.as_posix()


def _string_list(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ProfileConfigError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} string list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ProfileConfigError(f"{label} must contain non-empty strings")
    return [str(item) for item in value]


def validate_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    missing = sorted(PROFILE_FIELDS - set(data))
    unknown = sorted(set(data) - PROFILE_FIELDS)
    if missing:
        raise ProfileConfigError(f"profile missing fields: {', '.join(missing)}")
    if unknown:
        raise ProfileConfigError(f"profile has unknown fields: {', '.join(unknown)}")
    if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ProfileConfigError(
            f"unsupported profile schema: {data.get('schema_version')!r}; expected {PROFILE_SCHEMA_VERSION}"
        )
    version = data.get("profile_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProfileConfigError("profile_version must be a positive integer")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ProfileConfigError("profile.name must be a non-empty string")

    data["docs_dir"] = _safe_relative(data["docs_dir"], "profile.docs_dir")
    data["exclude"] = [
        str(item).replace("\\", "/").strip().strip("/")
        for item in _string_list(data["exclude"], "profile.exclude", allow_empty=True)
    ]
    for index, pattern in enumerate(data["exclude"]):
        if (
            not pattern
            or PurePosixPath(pattern).is_absolute()
            or PureWindowsPath(pattern).is_absolute()
            or ".." in PurePosixPath(pattern).parts
        ):
            raise ProfileConfigError(f"profile.exclude[{index}] must stay within the repository")

    vocabulary = _mapping(data["vocabulary"], "profile.vocabulary")
    if set(vocabulary) != VOCABULARY_FIELDS:
        raise ProfileConfigError("profile.vocabulary must contain exactly the documented vocabulary fields")
    data["vocabulary"] = {
        field: _string_list(vocabulary[field], f"profile.vocabulary.{field}")
        for field in sorted(VOCABULARY_FIELDS)
    }

    git = _mapping(data["git"], "profile.git")
    if set(git) != {"enabled", "status_scope", "source_root"}:
        raise ProfileConfigError("profile.git must contain enabled, status_scope, and source_root")
    if not isinstance(git["enabled"], bool):
        raise ProfileConfigError("profile.git.enabled must be boolean")
    data["git"] = {
        "enabled": git["enabled"],
        "status_scope": _safe_relative(git["status_scope"], "profile.git.status_scope"),
        "source_root": _safe_relative(git["source_root"], "profile.git.source_root"),
    }
    return data


def _profile_path(value: str | Path | None) -> Path:
    if value is None:
        return DEFAULT_PROFILES_DIR / f"{DEFAULT_PROFILE_NAME}.yaml"
    candidate = Path(value)
    if candidate.is_file() or candidate.suffix in {".yaml", ".yml", ".json"} or candidate.parent != Path("."):
        return candidate
    return DEFAULT_PROFILES_DIR / f"{candidate.name}.yaml"


def load_profile(profile: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        return validate_profile(profile)
    source = _profile_path(profile)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        label = str(profile) if profile is not None else DEFAULT_PROFILE_NAME
        raise ProfileConfigError(f"cannot read profile {label}: {source}") from exc
    try:
        return validate_profile(_parse_payload(text, source))
    except ProfileConfigError as exc:
        raise ProfileConfigError(f"profile file {source}: {exc}") from exc


def apply_profile(rules: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    configured = validate_rules(rules)
    selected = validate_profile(profile)
    vocabulary = selected["vocabulary"]
    configured["exemptions"]["status_contains"] = vocabulary["status_exemptions"]
    for rule_id, field in (("R1", "stale_status_hints"), ("R1", "done_markers"), ("R3", "topic_stop_words")):
        _rule = next(rule for rule in configured["rules"] if rule["id"] == rule_id)
        _rule["parameters"][field] = vocabulary[field]
    git = selected["git"]
    for rule_id in ("R2", "R3"):
        next(rule for rule in configured["rules"] if rule["id"] == rule_id)["enabled"] = git["enabled"]
    next(rule for rule in configured["rules"] if rule["id"] == "R2")["parameters"]["git_scope"] = git["status_scope"]
    next(rule for rule in configured["rules"] if rule["id"] == "R3")["parameters"]["source_root"] = git["source_root"]
    next(rule for rule in configured["rules"] if rule["id"] == "R4")["parameters"]["relative_root"] = selected["docs_dir"]
    return validate_rules(configured)


def profile_metadata(profile: Mapping[str, Any]) -> dict[str, Any]:
    selected = validate_profile(profile)
    return {
        "name": selected["name"],
        "schema_version": selected["schema_version"],
        "profile_version": selected["profile_version"],
        "docs_dir": selected["docs_dir"],
    }


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "DEFAULT_PROFILES_DIR",
    "PROFILE_SCHEMA_VERSION",
    "ProfileConfigError",
    "apply_profile",
    "load_profile",
    "profile_metadata",
    "validate_profile",
]
