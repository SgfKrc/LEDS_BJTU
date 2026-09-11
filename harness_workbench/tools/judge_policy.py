"""Offline v1/v2 judging-policy diff for ``TOOL-JUDGE-POLICY-01``.

The tool answers one narrow question: how does the same completion score under
the historical ``normalized_contains`` policy versus the v2
``loose_contains`` policy?  Completion text is consumed in memory and never
appears in a report; reports contain statuses, counts, hashes, and reasons.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


JUDGE_POLICY_SCHEMA = "qlh.harness.judge_policy_diff.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECK_KINDS = frozenset(("normalized_contains", "loose_contains"))
_CHINESE_DIGITS = {
    "零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
    "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _loose_normalise(text: str) -> str:
    value = _normalise(text)
    value = re.sub(r"(\d+)\s*(?:时|点)\s*(\d+)\s*分?", r"\1:\2", value)
    value = re.sub(r"(\d+)\s*小时\s*(\d+)\s*分钟?", r"\1:\2", value)
    for chinese, arabic in _CHINESE_DIGITS.items():
        value = re.sub(rf"(?<![零一两二三四五六七八九]){chinese}(?![零一两二三四五六七八九])", arabic, value)
    return value


def _answer_candidates(text: str) -> tuple[str, ...]:
    for marker in ("答案是", "答案：", "答案为", "答案:", "因此", "所以"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            tail = re.split(r"[。！？\n]", tail, 1)[0].strip()
            return (tail,) if tail else (text,)
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    return (text, lines[-1]) if lines else (text,)


@dataclass(frozen=True, slots=True)
class JudgePolicy:
    id: str
    match: str
    max_new_tokens: int
    description: str

    def __post_init__(self) -> None:
        if self.match not in _CHECK_KINDS:
            raise ValueError("judge policy match must be normalized_contains or loose_contains")
        if not self.id or self.max_new_tokens <= 0 or not self.description.strip():
            raise ValueError("judge policy fields are invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "match": self.match,
            "max_new_tokens": self.max_new_tokens,
            "description": self.description,
        }


V1_POLICY = JudgePolicy(
    "v1_192_normalized_contains",
    "normalized_contains",
    192,
    "NFKC/casefold/whitespace normalization over the whole completion.",
)
V2_POLICY = JudgePolicy(
    "v2_512_loose_contains",
    "loose_contains",
    512,
    "Answer-candidate extraction plus time and standalone Chinese-number normalization.",
)


@dataclass(frozen=True, slots=True)
class JudgeRubricEntry:
    prompt_id: str
    accepted: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        if not self.prompt_id or self.kind not in _CHECK_KINDS or not self.accepted or any(not value for value in self.accepted):
            raise ValueError("judge rubric entry is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "accepted": list(self.accepted), "kind": self.kind}


@dataclass(frozen=True, slots=True)
class JudgeRubric:
    rubric_id: str
    prompt_set_id: str
    prompt_set_sha256: str
    entries: tuple[JudgeRubricEntry, ...]
    source_digest: str

    def __post_init__(self) -> None:
        if not self.rubric_id or not self.prompt_set_id or not _SHA256.fullmatch(self.prompt_set_sha256):
            raise ValueError("judge rubric identity is invalid")
        if not self.entries or len({entry.prompt_id for entry in self.entries}) != len(self.entries):
            raise ValueError("judge rubric entries must be non-empty and unique")
        if not _SHA256.fullmatch(self.source_digest):
            raise ValueError("judge rubric source digest is invalid")

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(entry.prompt_id for entry in self.entries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "prompt_set_id": self.prompt_set_id,
            "prompt_set_sha256": self.prompt_set_sha256,
            "entries": [entry.as_dict() for entry in self.entries],
            "source_digest": self.source_digest,
        }


def _rubric_from_mapping(raw: Mapping[str, Any], *, dimension: str = "correctness", source_digest: str | None = None) -> JudgeRubric:
    if not isinstance(raw, Mapping):
        raise ValueError("judge rubric must be an object")
    rubric_id = raw.get("rubric_id")
    prompt_set = raw.get("prompt_set")
    entries = raw.get("entries")
    if not isinstance(rubric_id, str) or not rubric_id or not isinstance(prompt_set, Mapping):
        raise ValueError("judge rubric identity is required")
    prompt_set_id = prompt_set.get("id")
    prompt_set_sha256 = prompt_set.get("sha256")
    if not isinstance(prompt_set_id, str) or not isinstance(prompt_set_sha256, str) or not _SHA256.fullmatch(prompt_set_sha256):
        raise ValueError("judge rubric prompt set identity is invalid")
    if not isinstance(entries, list):
        raise ValueError("judge rubric entries must be a list")
    values: list[JudgeRubricEntry] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("judge rubric entry must be an object")
        prompt_id = item.get("prompt_id")
        check = item.get(dimension)
        if check is None:
            continue
        if not isinstance(prompt_id, str) or not isinstance(check, Mapping):
            raise ValueError("judge rubric entry is malformed")
        kind = check.get("kind")
        accepted = check.get("accepted")
        if kind not in _CHECK_KINDS or not isinstance(accepted, list) or not accepted or not all(isinstance(value, str) and value for value in accepted):
            raise ValueError("judge rubric supports only non-empty normalized/loose accepted checks")
        values.append(JudgeRubricEntry(prompt_id, tuple(accepted), str(kind)))
    if not values:
        raise ValueError(f"judge rubric has no {dimension} entries")
    return JudgeRubric(
        str(rubric_id),
        prompt_set_id,
        prompt_set_sha256,
        tuple(values),
        source_digest or _digest(raw),
    )


def load_judge_rubric(
    source: Mapping[str, Any] | str | Path,
    *,
    dimension: str = "correctness",
    expected_sha256: str | None = None,
) -> JudgeRubric:
    """Load a v1/v2 rubric from a mapping or JSON file with optional SHA pin."""

    if isinstance(source, Mapping):
        return _rubric_from_mapping(source, dimension=dimension)
    path = Path(source)
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("judge rubric is unavailable or invalid JSON") from exc
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("judge rubric SHA-256 mismatch")
    return _rubric_from_mapping(raw, dimension=dimension, source_digest=digest)


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    status: str
    passed: bool | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "missing", "truncated"}:
            raise ValueError("unsupported judge decision status")
        if self.status in {"passed", "failed"} and self.passed is not True and self.passed is not False:
            raise ValueError("evaluated decisions need a boolean result")
        if self.status in {"missing", "truncated"} and self.passed is not None:
            raise ValueError("invalid decisions cannot have a boolean result")

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "passed": self.passed, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class JudgePolicyDiff:
    prompt_id: str
    v1: JudgeDecision
    v2: JudgeDecision
    outcome: str

    def __post_init__(self) -> None:
        if self.outcome not in {"rescued", "regressed", "unchanged", "invalid"}:
            raise ValueError("unsupported policy diff outcome")

    def as_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "v1": self.v1.as_dict(), "v2": self.v2.as_dict(), "outcome": self.outcome}


def _evaluate(policy: JudgePolicy, accepted: Sequence[str], output: str | None, *, truncated: bool) -> JudgeDecision:
    if not isinstance(output, str):
        return JudgeDecision("missing", None, "completion_missing")
    if truncated:
        return JudgeDecision("truncated", None, "completion_truncated")
    if policy.match == "normalized_contains":
        normalized = _normalise(output)
        passed = any(_normalise(value) in normalized for value in accepted)
    else:
        candidates = tuple(_loose_normalise(value) for value in _answer_candidates(output))
        passed = any(_loose_normalise(value) in candidate for value in accepted for candidate in candidates)
    return JudgeDecision("passed" if passed else "failed", passed, "accepted_match" if passed else "accepted_match_missing")


def _outcome(v1: JudgeDecision, v2: JudgeDecision) -> str:
    if v1.passed is None or v2.passed is None:
        return "invalid"
    if not v1.passed and v2.passed:
        return "rescued"
    if v1.passed and not v2.passed:
        return "regressed"
    return "unchanged"


@dataclass(frozen=True, slots=True)
class JudgePolicyReport:
    v1_policy: JudgePolicy
    v2_policy: JudgePolicy
    v1_rubric: JudgeRubric
    v2_rubric: JudgeRubric
    diffs: tuple[JudgePolicyDiff, ...]
    skipped_prompt_ids: tuple[str, ...]
    outputs_digest: str
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False
    schema: str = JUDGE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != JUDGE_POLICY_SCHEMA or self.runner_kind != "fixture" or self.network_used or self.weights_loaded:
            raise ValueError("judge policy report is fixture-only")
        if not _SHA256.fullmatch(self.outputs_digest):
            raise ValueError("outputs digest is invalid")
        if not self.diffs:
            raise ValueError("judge policy report needs at least one diff")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def metrics(self) -> dict[str, Any]:
        def counts(policy: str) -> tuple[int, int]:
            decisions = [getattr(item, policy) for item in self.diffs]
            evaluated = sum(decision.passed is not None for decision in decisions)
            passed = sum(decision.passed is True for decision in decisions)
            return evaluated, passed

        v1_evaluated, v1_passed = counts("v1")
        v2_evaluated, v2_passed = counts("v2")
        return {
            "v1_evaluated_count": v1_evaluated,
            "v1_passed_count": v1_passed,
            "v1_rate": v1_passed / v1_evaluated if v1_evaluated else 0.0,
            "v2_evaluated_count": v2_evaluated,
            "v2_passed_count": v2_passed,
            "v2_rate": v2_passed / v2_evaluated if v2_evaluated else 0.0,
            "rescued_count": sum(item.outcome == "rescued" for item in self.diffs),
            "regressed_count": sum(item.outcome == "regressed" for item in self.diffs),
            "unchanged_count": sum(item.outcome == "unchanged" for item in self.diffs),
            "invalid_count": sum(item.outcome == "invalid" for item in self.diffs),
            "prompt_set_match": self.v1_rubric.prompt_set_sha256 == self.v2_rubric.prompt_set_sha256,
        }

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "v1_policy": self.v1_policy.as_dict(),
            "v2_policy": self.v2_policy.as_dict(),
            "v1_rubric": self.v1_rubric.as_dict(),
            "v2_rubric": self.v2_rubric.as_dict(),
            "metrics": self.metrics(),
            "diffs": [item.as_dict() for item in self.diffs],
            "skipped_prompt_ids": list(self.skipped_prompt_ids),
            "outputs_digest": self.outputs_digest,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        metrics = self.metrics()
        lines = [
            "# Judge policy diff: v1 vs v2",
            "",
            f"- v1: `{self.v1_policy.id}` (`{self.v1_policy.match}`, max new tokens `{self.v1_policy.max_new_tokens}`)",
            f"- v2: `{self.v2_policy.id}` (`{self.v2_policy.match}`, max new tokens `{self.v2_policy.max_new_tokens}`)",
            f"- runner: `fixture`; weights loaded: `false`; network used: `false`; prompt-set SHA match: `{metrics['prompt_set_match']}`",
            "- Completion text is intentionally omitted; statuses and hashes are the audit surface.",
            "",
            "| prompt | v1 | v2 | outcome | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in self.diffs:
            lines.append(f"| `{item.prompt_id}` | {item.v1.status} | {item.v2.status} | **{item.outcome}** | {item.v1.reason} / {item.v2.reason} |")
        lines.extend(
            (
                "",
                f"v1: {metrics['v1_passed_count']}/{metrics['v1_evaluated_count']} ({metrics['v1_rate']:.3f}); "
                f"v2: {metrics['v2_passed_count']}/{metrics['v2_evaluated_count']} ({metrics['v2_rate']:.3f}); "
                f"rescued: {metrics['rescued_count']}; regressed: {metrics['regressed_count']}.",
                "",
                f"Report digest: `{self.digest}`",
            )
        )
        return "\n".join(lines) + "\n"


def run_judge_policy_diff(
    v1_rubric: JudgeRubric,
    v2_rubric: JudgeRubric,
    outputs: Mapping[str, str | None],
    *,
    truncated_prompt_ids: set[str] | None = None,
    v1_policy: JudgePolicy = V1_POLICY,
    v2_policy: JudgePolicy = V2_POLICY,
) -> JudgePolicyReport:
    """Compare the same in-memory outputs under the two fixed policies."""

    v1_entries = {entry.prompt_id: entry for entry in v1_rubric.entries}
    v2_entries = {entry.prompt_id: entry for entry in v2_rubric.entries}
    common = tuple(prompt_id for prompt_id in v1_rubric.prompt_ids if prompt_id in v2_entries)
    if not common:
        raise ValueError("v1 and v2 rubrics have no common prompt ids")
    if any(v1_entries[prompt_id].kind != v1_policy.match for prompt_id in common):
        raise ValueError("v1 rubric check kind does not match v1 policy")
    if any(v2_entries[prompt_id].kind != v2_policy.match for prompt_id in common):
        raise ValueError("v2 rubric check kind does not match v2 policy")
    truncated = truncated_prompt_ids or set()
    diffs = []
    for prompt_id in common:
        v1 = _evaluate(v1_policy, v1_entries[prompt_id].accepted, outputs.get(prompt_id), truncated=prompt_id in truncated)
        v2 = _evaluate(v2_policy, v2_entries[prompt_id].accepted, outputs.get(prompt_id), truncated=prompt_id in truncated)
        diffs.append(JudgePolicyDiff(prompt_id, v1, v2, _outcome(v1, v2)))
    skipped = tuple(sorted((set(v1_entries) | set(v2_entries)) - set(common)))
    output_hashes = {
        prompt_id: hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else None
        for prompt_id, value in sorted(outputs.items())
    }
    return JudgePolicyReport(
        v1_policy=v1_policy,
        v2_policy=v2_policy,
        v1_rubric=v1_rubric,
        v2_rubric=v2_rubric,
        diffs=tuple(diffs),
        skipped_prompt_ids=skipped,
        outputs_digest=_digest(output_hashes),
    )


def builtin_judge_policy_fixture() -> tuple[JudgeRubric, JudgeRubric, Mapping[str, str]]:
    """Return a small calibration pair that demonstrates two v2 rescues."""

    entries_v1 = (
        JudgeRubricEntry("math-time", ("13:54", "13：54"), "normalized_contains"),
        JudgeRubricEntry("math-number", ("160",), "normalized_contains"),
        JudgeRubricEntry("math-chinese-number", ("3",), "normalized_contains"),
        JudgeRubricEntry("status", ("STATUS=READY",), "normalized_contains"),
    )
    entries_v2 = tuple(
        JudgeRubricEntry(entry.prompt_id, entry.accepted, "loose_contains") for entry in entries_v1
    )
    prompt_set = "fixture-prompt-set-v1"
    prompt_set_sha = _digest(prompt_set)
    v1 = JudgeRubric("fixture-rubric-v1", prompt_set, prompt_set_sha, entries_v1, _digest([entry.as_dict() for entry in entries_v1]))
    v2 = JudgeRubric("fixture-rubric-v2", prompt_set, prompt_set_sha, entries_v2, _digest([entry.as_dict() for entry in entries_v2]))
    outputs = {
        "math-time": "列车预计 13时54分 到达。",
        "math-number": "答案是 160。",
        "math-chinese-number": "答案是三。",
        "status": "STATUS=READY",
    }
    return v1, v2, outputs


__all__ = [
    "JUDGE_POLICY_SCHEMA",
    "JudgeDecision",
    "JudgePolicy",
    "JudgePolicyDiff",
    "JudgePolicyReport",
    "JudgeRubric",
    "JudgeRubricEntry",
    "V1_POLICY",
    "V2_POLICY",
    "builtin_judge_policy_fixture",
    "load_judge_rubric",
    "run_judge_policy_diff",
]
