"""Deterministic, non-executing rubric checks for EX-N3 LLM quality runs.

The rubric fixture is versioned and hash-pinned by the experiment plan.  Model
completions stay in memory: callers receive counters and statuses only.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class RubricError(ValueError):
    """The local rubric is malformed or does not match its declared identity."""


_ROOT_KEYS = {"schema_version", "rubric_id", "prompt_set", "entries"}
_ENTRY_KEYS = {"prompt_id", "correctness", "format"}
_CHECK_KINDS = {
    "normalized_contains",
    "json_object_fields",
    "python_dict_exact",
    "python_function",
    "numbered_list",
    "csv_header_rows",
    "paragraph_terms",
}


@dataclass(frozen=True)
class ObjectiveRubric:
    rubric_id: str
    prompt_set: Mapping[str, str]
    entries: tuple[Mapping[str, Any], ...]


def _only_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    if set(value) - allowed:
        raise RubricError(f"{label} contains unsupported fields")


def _check_spec(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RubricError(f"{label} must be an object")
    kind = raw.get("kind")
    if kind not in _CHECK_KINDS:
        raise RubricError(f"{label} has an unsupported check kind")
    if kind == "normalized_contains":
        _only_keys(raw, {"kind", "accepted"}, label)
        accepted = raw.get("accepted")
        if not isinstance(accepted, list) or not accepted or not all(isinstance(v, str) for v in accepted):
            raise RubricError(f"{label}.accepted must be a non-empty string list")
    elif kind == "json_object_fields":
        _only_keys(raw, {"kind", "fields"}, label)
        fields = raw.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            raise RubricError(f"{label}.fields must be a non-empty object")
        if not all(isinstance(k, str) and v in {"string", "number", "string_array"} for k, v in fields.items()):
            raise RubricError(f"{label}.fields contains an unsupported type")
    elif kind == "python_dict_exact":
        _only_keys(raw, {"kind", "expected"}, label)
        expected = raw.get("expected")
        if not isinstance(expected, Mapping) or not expected:
            raise RubricError(f"{label}.expected must be a non-empty object")
    elif kind == "python_function":
        _only_keys(raw, {"kind", "minimum_functions"}, label)
        if raw.get("minimum_functions") != 1:
            raise RubricError(f"{label}.minimum_functions must be 1")
    elif kind == "numbered_list":
        _only_keys(raw, {"kind", "items"}, label)
        if raw.get("items") != 3:
            raise RubricError(f"{label}.items must be 3")
    elif kind == "csv_header_rows":
        _only_keys(raw, {"kind", "header", "data_rows"}, label)
        header = raw.get("header")
        if header != ["id", "name", "score"] or raw.get("data_rows") != 2:
            raise RubricError(f"{label} must use the frozen CSV contract")
    else:  # paragraph_terms
        _only_keys(raw, {"kind", "required_terms"}, label)
        terms = raw.get("required_terms")
        if not isinstance(terms, list) or not terms or not all(isinstance(v, str) for v in terms):
            raise RubricError(f"{label}.required_terms must be a non-empty string list")
    return dict(raw)


def load_objective_rubric(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_prompt_set: Mapping[str, str] | None = None,
) -> ObjectiveRubric:
    """Read and validate a pinned rubric without exposing its contents to records."""
    candidate = Path(path)
    try:
        raw_bytes = candidate.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RubricError("objective rubric is unavailable") from exc
    if expected_sha256 and hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise RubricError("objective rubric SHA-256 mismatch")
    if not isinstance(raw, Mapping):
        raise RubricError("objective rubric must be an object")
    _only_keys(raw, _ROOT_KEYS, "objective rubric")
    if raw.get("schema_version") != 1:
        raise RubricError("objective rubric schema_version must be 1")
    rubric_id = raw.get("rubric_id")
    if not isinstance(rubric_id, str) or not rubric_id:
        raise RubricError("objective rubric requires rubric_id")
    prompt_set = raw.get("prompt_set")
    if not isinstance(prompt_set, Mapping) or not isinstance(prompt_set.get("id"), str) or not isinstance(prompt_set.get("sha256"), str):
        raise RubricError("objective rubric requires a prompt_set identity")
    if expected_prompt_set and dict(prompt_set) != dict(expected_prompt_set):
        raise RubricError("objective rubric prompt set does not match plan")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RubricError("objective rubric entries must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping):
            raise RubricError("objective rubric entry must be an object")
        _only_keys(item, _ENTRY_KEYS, "objective rubric entry")
        prompt_id = item.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise RubricError("objective rubric prompt_id must be unique")
        checks: dict[str, Any] = {"prompt_id": prompt_id}
        for dimension in ("correctness", "format"):
            if dimension in item:
                checks[dimension] = _check_spec(item[dimension], f"{prompt_id}.{dimension}")
        if len(checks) == 1:
            raise RubricError("objective rubric entry needs correctness or format")
        seen.add(prompt_id)
        normalized.append(checks)
    return ObjectiveRubric(rubric_id=rubric_id, prompt_set=dict(prompt_set), entries=tuple(normalized))


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _evaluate_check(check: Mapping[str, Any], output: str) -> bool:
    kind = check["kind"]
    if kind == "normalized_contains":
        normalized = _normalise(output)
        return any(_normalise(value) in normalized for value in check["accepted"])
    if kind == "json_object_fields":
        try:
            value = json.loads(_code(output))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(value, Mapping):
            return False
        for key, expected in check["fields"].items():
            item = value.get(key)
            if expected == "string" and not isinstance(item, str):
                return False
            if expected == "number" and (isinstance(item, bool) or not isinstance(item, (int, float))):
                return False
            if expected == "string_array" and (
                not isinstance(item, list) or not all(isinstance(part, str) for part in item)
            ):
                return False
        return True
    if kind == "python_dict_exact":
        try:
            value = ast.literal_eval(_code(output))
        except (SyntaxError, ValueError, TypeError):
            return False
        return isinstance(value, dict) and value == check["expected"]
    if kind == "python_function":
        try:
            tree = ast.parse(_code(output), mode="exec")
        except SyntaxError:
            return False
        return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body) >= 1
    if kind == "numbered_list":
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return len(lines) == check["items"] and all(
            re.match(r"^\d+[.)]\s+\S", line) is not None for line in lines
        )
    if kind == "csv_header_rows":
        try:
            rows = list(csv.reader(io.StringIO(output.strip())))
        except csv.Error:
            return False
        return (
            len(rows) == check["data_rows"] + 1
            and rows[0] == check["header"]
            and all(len(row) == len(check["header"]) for row in rows[1:])
        )
    if kind == "paragraph_terms":
        stripped = output.strip()
        return "\n\n" not in stripped and all(
            _normalise(term) in _normalise(stripped) for term in check["required_terms"]
        )
    raise RubricError("unsupported objective rubric check")


def score_objective_outputs(
    rubric: ObjectiveRubric,
    outputs: Mapping[str, str | None],
    *,
    truncated_prompt_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Return derived counters only; completion contents never leave this call."""
    truncated = truncated_prompt_ids or set()
    counters = {
        "correctness": {"evaluated_count": 0, "passed_count": 0, "invalid_count": 0},
        "format": {"evaluated_count": 0, "passed_count": 0, "invalid_count": 0},
    }
    for entry in rubric.entries:
        prompt_id = entry["prompt_id"]
        output = outputs.get(prompt_id)
        for dimension in ("correctness", "format"):
            if dimension not in entry:
                continue
            if not isinstance(output, str):
                counters[dimension]["invalid_count"] += 1
                continue
            counters[dimension]["evaluated_count"] += 1
            if prompt_id not in truncated and _evaluate_check(entry[dimension], output):
                counters[dimension]["passed_count"] += 1
    return counters
