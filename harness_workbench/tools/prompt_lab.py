"""Offline A/B rendering lab for ``TOOL-PROMPT-LAB-01``.

The lab compares PromptProfile policy, not model quality.  It renders the
same normalized message cases through two or more profiles and reports only
stable digests and length metrics; message and system-prompt text never enter
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..adaptation import PromptProfile, render_prompt_messages
from ..adaptation.builtin import builtin_adaptation_profiles
from ..context_engine import ContextMessage
from ..context_engine.tokenizer import HeuristicTokenizer, count_message


PROMPT_LAB_SCHEMA = "qlh.harness.prompt_lab.v1"
PROMPT_LAB_INPUT_SCHEMA = "qlh.prompt_lab.v1"
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _path_free(value: Any) -> None:
    if isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise ValueError("prompt lab input cannot contain absolute paths")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _path_free(key)
            _path_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _path_free(child)


@dataclass(frozen=True, slots=True)
class PromptLabCase:
    """One normalized conversation used by every profile."""

    case_id: str
    messages: tuple[ContextMessage, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.case_id or not self.messages:
            raise ValueError("prompt lab case id and messages are required")
        if any(not isinstance(message, ContextMessage) for message in self.messages):
            raise TypeError("prompt lab cases require ContextMessage values")
        _path_free(self.description)

    @property
    def messages_digest(self) -> str:
        return _digest([message.as_dict() for message in self.messages])

    def as_dict(self) -> dict[str, Any]:
        # Deliberately omit message content; the digest binds the exact case.
        return {
            "case_id": self.case_id,
            "message_count": len(self.messages),
            "roles": [message.role for message in self.messages],
            "messages_digest": self.messages_digest,
            "description_chars": len(self.description),
            "description_digest": _digest(self.description),
        }


def builtin_prompt_lab_cases() -> tuple[PromptLabCase, ...]:
    """Return a small, stable set of safe prompts for template comparison."""

    return (
        PromptLabCase(
            "factual-short-v1",
            (ContextMessage("user", "Name the primary color in this sentence: red."),),
            "short factual answer",
        ),
        PromptLabCase(
            "structured-json-v1",
            (ContextMessage("user", "Return JSON with keys answer and confidence."),),
            "structured output request",
        ),
        PromptLabCase(
            "context-recall-v1",
            (
                ContextMessage("user", "The project codename is SILVER-FOX."),
                ContextMessage("assistant", "Recorded."),
                ContextMessage("user", "Summarize the plan and include the codename."),
            ),
            "short multi-turn recall",
        ),
    )


def _case_from_mapping(value: Mapping[str, Any], index: int) -> PromptLabCase:
    _path_free(value)
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, (list, tuple)):
        raise ValueError("prompt lab case messages must be an array")
    messages = tuple(ContextMessage.from_value(message) for message in raw_messages)
    return PromptLabCase(str(value.get("case_id", f"case-{index + 1}")), messages, str(value.get("description", "")))


def load_prompt_cases(path: str | Path) -> tuple[PromptLabCase, ...]:
    """Load cases from a JSON list or ``PROMPT_LAB_INPUT_SCHEMA`` object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("cases"), list):
        schema = payload.get("schema")
        if schema is not None and schema != PROMPT_LAB_INPUT_SCHEMA:
            raise ValueError(f"unsupported prompt lab input schema: {schema}")
        raw_cases = payload["cases"]
    else:
        raise ValueError("prompt lab input must be a JSON case list or object with cases")
    if not raw_cases or not all(isinstance(item, Mapping) for item in raw_cases):
        raise ValueError("prompt lab input requires at least one case object")
    cases = tuple(_case_from_mapping(item, index) for index, item in enumerate(raw_cases))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("prompt lab case ids must be unique")
    return cases


def _profile_from_mapping(value: Mapping[str, Any]) -> PromptProfile:
    _path_free(value)
    return PromptProfile(
        id=str(value.get("id", "")),
        family=str(value.get("family", "")),
        system_prompt=str(value.get("system_prompt", "")),
        stop=tuple(value.get("stop", ())),
        thinking=str(value.get("thinking", "unknown")),
        tool_mode=str(value.get("tool_mode", "host_router")),
        structured_output=str(value.get("structured_output", "json_repair")),
        version=str(value.get("version", "v1")),
    )


def load_prompt_profiles(path: str | Path) -> tuple[PromptProfile, ...]:
    """Load two or more PromptProfile objects from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_profiles = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("profiles"), list):
        schema = payload.get("schema")
        if schema is not None and schema != PROMPT_LAB_INPUT_SCHEMA:
            raise ValueError(f"unsupported prompt lab input schema: {schema}")
        raw_profiles = payload["profiles"]
    else:
        raise ValueError("prompt lab profile input must be a JSON profile list or object with profiles")
    if not raw_profiles or not all(isinstance(item, Mapping) for item in raw_profiles):
        raise ValueError("prompt lab profile input requires at least one profile object")
    profiles = tuple(_profile_from_mapping(item) for item in raw_profiles)
    if len({profile.id for profile in profiles}) != len(profiles):
        raise ValueError("prompt lab profile ids must be unique")
    return profiles


def _profile_digest(profile: PromptProfile) -> str:
    return _digest(profile.as_dict())


def _profile_snapshot(profile: PromptProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.id,
        "family": profile.family,
        "version": profile.version,
        "profile_digest": _profile_digest(profile),
        "system_prompt_chars": len(profile.system_prompt),
        "system_prompt_digest": _digest(profile.system_prompt),
        "stop_count": len(profile.stop),
        "thinking": profile.thinking,
        "tool_mode": profile.tool_mode,
        "structured_output": profile.structured_output,
    }


def _profile_field_summary(field: str, value: Any) -> Any:
    if field == "system_prompt":
        return {"chars": len(value), "digest": _digest(value)}
    if field == "stop":
        return {"count": len(value), "digest": _digest(list(value))}
    return value


@dataclass(frozen=True, slots=True)
class PromptRenderResult:
    profile_id: str
    case_id: str
    rendered_digest: str
    rendered_message_count: int
    preserved_message_count: int
    system_message_count: int
    system_prompt_chars: int
    rendered_chars: int
    estimated_tokens: int

    def __post_init__(self) -> None:
        if not self.profile_id or not self.case_id or not self.rendered_digest:
            raise ValueError("prompt render identity is required")
        for name in (
            "rendered_message_count",
            "preserved_message_count",
            "system_message_count",
            "system_prompt_chars",
            "rendered_chars",
            "estimated_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError("prompt render metrics must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "case_id": self.case_id,
            "rendered_digest": self.rendered_digest,
            "rendered_message_count": self.rendered_message_count,
            "preserved_message_count": self.preserved_message_count,
            "system_message_count": self.system_message_count,
            "system_prompt_chars": self.system_prompt_chars,
            "rendered_chars": self.rendered_chars,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True, slots=True)
class PromptCaseDelta:
    case_id: str
    profile_a: str
    profile_b: str
    chars_delta: int
    estimated_tokens_delta: int
    system_chars_delta: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "profile_a": self.profile_a,
            "profile_b": self.profile_b,
            "chars_delta": self.chars_delta,
            "estimated_tokens_delta": self.estimated_tokens_delta,
            "system_chars_delta": self.system_chars_delta,
        }


@dataclass(frozen=True, slots=True)
class PromptProfileDiff:
    profile_a: str
    profile_b: str
    changed_fields: tuple[str, ...]
    fields: Mapping[str, Mapping[str, Any]]
    case_deltas: tuple[PromptCaseDelta, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_a": self.profile_a,
            "profile_b": self.profile_b,
            "changed_fields": list(self.changed_fields),
            "fields": {key: dict(value) for key, value in self.fields.items()},
            "case_deltas": [item.as_dict() for item in self.case_deltas],
        }


@dataclass(frozen=True, slots=True)
class PromptLabReport:
    model_family: str
    profiles: tuple[PromptProfile, ...]
    cases: tuple[PromptLabCase, ...]
    renders: tuple[PromptRenderResult, ...]
    comparisons: tuple[PromptProfileDiff, ...]
    selected_profile_ids: tuple[str, ...] = ()
    selected_case_ids: tuple[str, ...] = ()
    schema: str = PROMPT_LAB_SCHEMA
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != PROMPT_LAB_SCHEMA or self.runner_kind not in {"fixture", "input"}:
            raise ValueError("prompt lab identity is invalid")
        if len(self.profiles) < 2:
            raise ValueError("prompt lab requires at least two profiles for A/B comparison")
        if not self.cases or not self.renders:
            raise ValueError("prompt lab requires cases and renders")
        if self.network_used or self.weights_loaded:
            raise ValueError("prompt lab must remain offline and model-free")

    @property
    def profile_count(self) -> int:
        return len(self.profiles)

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def render_count(self) -> int:
        return len(self.renders)

    @property
    def checks(self) -> dict[str, bool]:
        expected = self.profile_count * self.case_count
        matrix_complete = self.render_count == expected and len(
            {(item.profile_id, item.case_id) for item in self.renders}
        ) == expected
        profile_ids = {profile.id for profile in self.profiles}
        cases_by_id = {case.case_id: case for case in self.cases}
        case_ids = set(cases_by_id)
        identities_valid = all(item.profile_id in profile_ids and item.case_id in case_ids for item in self.renders)
        injection_valid = all(
            item.system_message_count == 1
            and item.case_id in cases_by_id
            and item.preserved_message_count == len(cases_by_id[item.case_id].messages)
            and item.rendered_message_count == item.preserved_message_count + 1
            for item in self.renders
        )
        profiles_distinct = len({_profile_digest(profile) for profile in self.profiles}) == self.profile_count
        diff_present = bool(self.comparisons) and all(item.changed_fields for item in self.comparisons)
        redacted = all(
            "content" not in case.as_dict()
            for case in self.cases
        ) and all(
            field != "system_prompt"
            or all(set(change[side]) == {"chars", "digest"} for side in ("a", "b"))
            for comparison in self.comparisons
            for field, change in comparison.fields.items()
        )
        return {
            "render_matrix_complete": matrix_complete,
            "render_identities_valid": identities_valid,
            "system_injected_once": injection_valid,
            "profiles_distinct": profiles_distinct,
            "profile_diffs_present": diff_present,
            "payloads_omitted": redacted,
            "offline_boundary": not self.network_used and not self.weights_loaded,
        }

    @property
    def valid(self) -> bool:
        return all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "model_family": self.model_family,
            "profile_count": self.profile_count,
            "case_count": self.case_count,
            "render_count": self.render_count,
            "selected_profile_ids": list(self.selected_profile_ids),
            "selected_case_ids": list(self.selected_case_ids),
            "profiles": [_profile_snapshot(profile) for profile in self.profiles],
            "cases": [case.as_dict() for case in self.cases],
            "renders": [item.as_dict() for item in self.renders],
            "comparisons": [item.as_dict() for item in self.comparisons],
            "checks": self.checks,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-PROMPT-LAB-01 A/B prompt render report",
            "",
            f"- Valid: `{str(self.valid).lower()}`; model family: `{self.model_family}`; profiles: `{self.profile_count}`; cases: `{self.case_count}`",
            f"- Renders: `{self.render_count}`; runner: `{self.runner_kind}`; weights loaded: `false`; network used: `false`.",
            "- Message and system-prompt text are omitted; digests bind the exact inputs while lengths remain comparable.",
            "",
            "## Profiles",
            "",
            "| profile | family | system chars | estimated policy | stop count | profile digest |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
        for profile in self.profiles:
            lines.append(
                f"| `{profile.id}` | `{profile.family}` | {len(profile.system_prompt)} | `{profile.thinking}/{profile.tool_mode}/{profile.structured_output}` | {len(profile.stop)} | `{_profile_digest(profile)}` |"
            )
        lines.extend(("", "## Profile differences", "", "| A | B | changed fields |", "| --- | --- | --- |"))
        for comparison in self.comparisons:
            lines.append(f"| `{comparison.profile_a}` | `{comparison.profile_b}` | `{', '.join(comparison.changed_fields)}` |")
        lines.extend(("", "## Length comparison", "", "| case | A | B | token delta (B-A) | char delta (B-A) |", "| --- | --- | --- | ---: | ---: |"))
        for comparison in self.comparisons:
            for delta in comparison.case_deltas:
                lines.append(
                    f"| `{delta.case_id}` | `{delta.profile_a}` | `{delta.profile_b}` | {delta.estimated_tokens_delta} | {delta.chars_delta} |"
                )
        lines.extend(("", "## Checks", ""))
        for name, passed in self.checks.items():
            lines.append(f"- `{name}`: **{'passed' if passed else 'failed'}**")
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def _select_profiles(profiles: Sequence[PromptProfile], selected: Sequence[str] | None) -> tuple[PromptProfile, ...]:
    values = tuple(profiles)
    if selected:
        wanted = tuple(selected)
        if len(set(wanted)) != len(wanted):
            raise ValueError("selected prompt profile ids must be unique")
        values = tuple(profile for profile in values if profile.id in wanted)
        if set(wanted) != {profile.id for profile in values}:
            raise ValueError("unknown prompt profile id")
        values = tuple(next(profile for profile in values if profile.id == profile_id) for profile_id in wanted)
    if len(values) < 2:
        raise ValueError("prompt lab requires at least two selected profiles")
    return values


def _select_cases(cases: Sequence[PromptLabCase], selected: Sequence[str] | None) -> tuple[PromptLabCase, ...]:
    values = tuple(cases)
    if selected:
        wanted = tuple(selected)
        by_id = {case.case_id: case for case in values}
        if any(case_id not in by_id for case_id in wanted):
            raise ValueError("unknown prompt lab case id")
        values = tuple(by_id[case_id] for case_id in wanted)
    if not values:
        raise ValueError("prompt lab requires at least one selected case")
    return values


def _render(profile: PromptProfile, case: PromptLabCase) -> PromptRenderResult:
    rendered = render_prompt_messages(profile, case.messages, include_metadata=False)
    normalized = tuple(ContextMessage.from_value(message) for message in rendered)
    tokenizer = HeuristicTokenizer()
    rendered_chars = sum(len(message.content) for message in normalized)
    estimated_tokens = sum(count_message(tokenizer, message) for message in normalized)
    return PromptRenderResult(
        profile.id,
        case.case_id,
        _digest(rendered),
        len(normalized),
        sum(1 for message in case.messages if message.role != "system"),
        sum(1 for message in normalized if message.role == "system"),
        len(profile.system_prompt),
        rendered_chars,
        estimated_tokens,
    )


def _compare(a: PromptProfile, b: PromptProfile, cases: Sequence[PromptLabCase], renders: Sequence[PromptRenderResult]) -> PromptProfileDiff:
    fields: dict[str, Mapping[str, Any]] = {}
    for field in ("family", "version", "system_prompt", "stop", "thinking", "tool_mode", "structured_output"):
        left = getattr(a, field)
        right = getattr(b, field)
        if left != right:
            fields[field] = {
                "a": _profile_field_summary(field, left),
                "b": _profile_field_summary(field, right),
            }
    rendered_by_key = {(item.profile_id, item.case_id): item for item in renders}
    deltas = tuple(
        PromptCaseDelta(
            case.case_id,
            a.id,
            b.id,
            rendered_by_key[(b.id, case.case_id)].rendered_chars - rendered_by_key[(a.id, case.case_id)].rendered_chars,
            rendered_by_key[(b.id, case.case_id)].estimated_tokens - rendered_by_key[(a.id, case.case_id)].estimated_tokens,
            len(b.system_prompt) - len(a.system_prompt),
        )
        for case in cases
    )
    return PromptProfileDiff(a.id, b.id, tuple(fields), fields, deltas)


def run_prompt_lab(
    profiles: Sequence[PromptProfile] | None = None,
    cases: Sequence[PromptLabCase] | None = None,
    *,
    model_family: str = "QW1.8B",
    profile_file: str | Path | None = None,
    case_file: str | Path | None = None,
    selected_profiles: Sequence[str] | None = None,
    selected_cases: Sequence[str] | None = None,
) -> PromptLabReport:
    if profiles is not None and profile_file is not None:
        raise ValueError("provide profiles or profile_file, not both")
    if cases is not None and case_file is not None:
        raise ValueError("provide cases or case_file, not both")
    profile_values = tuple(profiles) if profiles is not None else (
        load_prompt_profiles(profile_file) if profile_file is not None else builtin_adaptation_profiles(model_family)[0]
    )
    case_values = tuple(cases) if cases is not None else (
        load_prompt_cases(case_file) if case_file is not None else builtin_prompt_lab_cases()
    )
    selected_profile_values = _select_profiles(profile_values, selected_profiles)
    selected_case_values = _select_cases(case_values, selected_cases)
    renders = tuple(_render(profile, case) for profile in selected_profile_values for case in selected_case_values)
    comparisons = tuple(
        _compare(selected_profile_values[index], selected_profile_values[index + 1], selected_case_values, renders)
        for index in range(len(selected_profile_values) - 1)
    )
    return PromptLabReport(
        model_family,
        selected_profile_values,
        selected_case_values,
        renders,
        comparisons,
        tuple(profile.id for profile in selected_profile_values),
        tuple(case.case_id for case in selected_case_values),
        runner_kind="input" if profile_file or case_file else "fixture",
    )


def build_prompt_lab_report(*args: Any, **kwargs: Any) -> PromptLabReport:
    return run_prompt_lab(*args, **kwargs)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PromptProfile A/B renders offline")
    parser.add_argument("--model-family", default="QW1.8B", help="built-in adaptation family (default: QW1.8B)")
    parser.add_argument("--profile-file", metavar="PATH", help="JSON PromptProfile list/object")
    parser.add_argument("--case-file", metavar="PATH", help="JSON normalized case list/object")
    parser.add_argument("--profile", dest="profile_ids", action="append", help="select a profile id; repeat for A/B")
    parser.add_argument("--case", dest="case_ids", action="append", help="select a case id; repeat to include more")
    parser.add_argument("--list", action="store_true", help="list built-in profiles and cases")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def _list_values(model_family: str) -> str:
    profiles = builtin_adaptation_profiles(model_family)[0]
    cases = builtin_prompt_lab_cases()
    lines = ["[profiles]"] + [f"{profile.id}\t{profile.family}" for profile in profiles]
    lines += ["[cases]"] + [f"{case.case_id}\t{case.description}" for case in cases]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(_list_values(args.model_family), end="")
        return 0
    try:
        report = run_prompt_lab(
            model_family=args.model_family,
            profile_file=args.profile_file,
            case_file=args.case_file,
            selected_profiles=args.profile_ids,
            selected_cases=args.case_ids,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    output_count = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        output_count += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        output_count += 1
    if not output_count:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "PROMPT_LAB_INPUT_SCHEMA",
    "PROMPT_LAB_SCHEMA",
    "PromptCaseDelta",
    "PromptLabCase",
    "PromptLabReport",
    "PromptProfileDiff",
    "PromptRenderResult",
    "build_parser",
    "build_prompt_lab_report",
    "builtin_prompt_lab_cases",
    "load_prompt_cases",
    "load_prompt_profiles",
    "main",
    "run_prompt_lab",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
