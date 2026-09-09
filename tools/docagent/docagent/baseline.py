"""Locked, path-free baseline snapshots for docagent reports."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


BASELINE_SCHEMA_VERSION = "qlh.docagent.baseline.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BaselineError(ValueError):
    """Raised when a baseline is missing, malformed, or cannot be compared."""


def _safe_doc_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BaselineError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/").strip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise BaselineError(f"{label} must be a repository-relative path")
    return path.as_posix()


def _profile_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineError("baseline.profile must be a mapping")
    required = {"name", "schema_version", "profile_version", "docs_dir"}
    if set(value) != required:
        raise BaselineError("baseline.profile has an invalid metadata shape")
    version = value["profile_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BaselineError("baseline.profile.profile_version must be a positive integer")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise BaselineError("baseline.profile.name must be a non-empty string")
    if not isinstance(value["schema_version"], str) or not value["schema_version"].strip():
        raise BaselineError("baseline.profile.schema_version must be a non-empty string")
    return {
        "name": value["name"],
        "schema_version": value["schema_version"],
        "profile_version": version,
        "docs_dir": _safe_doc_path(value["docs_dir"], "baseline.profile.docs_dir"),
    }


def _validate_documents(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BaselineError("baseline.docs must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise BaselineError(f"baseline.docs[{index}] must be a mapping")
        required = {"doc", "status_line", "updated_at", "sha256", "findings"}
        if set(item) != required:
            raise BaselineError(f"baseline.docs[{index}] has an invalid document shape")
        doc = _safe_doc_path(item["doc"], f"baseline.docs[{index}].doc")
        if doc in seen:
            raise BaselineError(f"baseline contains duplicate document: {doc}")
        seen.add(doc)
        digest = item["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise BaselineError(f"baseline.docs[{index}].sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(item["status_line"], str)
            or (item["updated_at"] is not None and not isinstance(item["updated_at"], str))
        ):
            raise BaselineError(f"baseline.docs[{index}] has invalid status metadata")
        if not isinstance(item["findings"], list):
            raise BaselineError(f"baseline.docs[{index}].findings must be a list")
        records.append(copy.deepcopy(dict(item)))
        records[-1]["doc"] = doc
    return records


def validate_baseline(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(payload))
    required = {
        "schema_version", "baseline_version", "created_at", "rules", "ruleset_version",
        "rules_fingerprint", "profile", "docs",
    }
    if set(data) != required:
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required)
        if missing:
            raise BaselineError(f"baseline missing fields: {', '.join(missing)}")
        raise BaselineError(f"baseline has unknown fields: {', '.join(unknown)}")
    if data["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"unsupported baseline schema: {data['schema_version']!r}; expected {BASELINE_SCHEMA_VERSION}"
        )
    version = data["baseline_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BaselineError("baseline_version must be a positive integer")
    if not isinstance(data["created_at"], str) or not data["created_at"].strip():
        raise BaselineError("baseline.created_at must be a non-empty string")
    if not isinstance(data["rules"], Mapping) or not data["rules"]:
        raise BaselineError("baseline.rules must be a non-empty mapping")
    ruleset_version = data["ruleset_version"]
    if isinstance(ruleset_version, bool) or not isinstance(ruleset_version, int) or ruleset_version < 1:
        raise BaselineError("baseline.ruleset_version must be a positive integer")
    if not isinstance(data["rules_fingerprint"], str) or not _SHA256.fullmatch(data["rules_fingerprint"]):
        raise BaselineError("baseline.rules_fingerprint must be a lowercase SHA-256 digest")
    data["profile"] = _profile_metadata(data["profile"])
    data["docs"] = _validate_documents(data["docs"])
    return data


def baseline_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_version": 1,
        "created_at": report["run_ts"],
        "rules": copy.deepcopy(report["rules"]),
        "ruleset_version": report["ruleset_version"],
        "rules_fingerprint": report["rules_fingerprint"],
        "profile": copy.deepcopy(report["profile"]),
        "docs": copy.deepcopy(report["docs"]),
    }
    return validate_baseline(payload)


def write_baseline(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    payload = baseline_from_report(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def load_baseline(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(f"cannot read baseline file: {source}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"baseline file {source} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise BaselineError(f"baseline file {source} must contain a mapping")
    try:
        return validate_baseline(payload)
    except BaselineError as exc:
        raise BaselineError(f"baseline file {source}: {exc}") from exc


def ensure_matches(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    selected = validate_baseline(baseline)
    if selected["ruleset_version"] != report["ruleset_version"]:
        raise BaselineError(
            "baseline ruleset version mismatch: "
            f"baseline={selected['ruleset_version']} current={report['ruleset_version']}"
        )
    if selected["rules_fingerprint"] != report["rules_fingerprint"]:
        raise BaselineError(
            "baseline rules fingerprint mismatch: "
            f"baseline={selected['rules_fingerprint']} current={report['rules_fingerprint']}"
        )
    if selected["profile"] != report["profile"]:
        raise BaselineError("baseline profile metadata mismatch")


__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselineError",
    "baseline_from_report",
    "ensure_matches",
    "load_baseline",
    "validate_baseline",
    "write_baseline",
]
