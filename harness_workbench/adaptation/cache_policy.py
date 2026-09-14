"""Static prompt checks for provider prefix-cache friendly system prompts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Pattern


@dataclass(frozen=True, slots=True)
class PromptCacheViolation:
    line: int
    rule: str


_RULES: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "timestamp",
        re.compile(
            r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}"
            r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
        ),
    ),
    ("windows_absolute_path", re.compile(r"\b[A-Za-z]:[\\/][^\s`\"']+")),
    ("unc_path", re.compile(r"(?<!\w)\\\\[A-Za-z0-9._-]+[\\/][^\s`\"']+")),
    (
        "posix_absolute_path",
        re.compile(r"(?<![A-Za-z0-9:/])/(?:home|Users|tmp|var|mnt|workspace|root|opt|srv)/[^\s`\"']+"),
    ),
    (
        "uuid",
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
    ),
    ("runtime_interpolation", re.compile(r"\$\{[^}\r\n]+\}|\{\{[^}\r\n]+\}\}")),
    (
        "runtime_id_value",
        re.compile(
            r"\b(?:job|request|session|trace|checkpoint)[_-]?id\s*[:=]\s*['\"]?[A-Za-z0-9][A-Za-z0-9._-]*",
            re.IGNORECASE,
        ),
    ),
)


def find_prompt_cache_violations(prompt: str) -> tuple[PromptCacheViolation, ...]:
    """Return line/rule pairs for values that make a system prompt dynamic."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    violations: list[PromptCacheViolation] = []
    for line_number, line in enumerate(prompt.splitlines(), 1):
        for rule, pattern in _RULES:
            if pattern.search(line):
                violations.append(PromptCacheViolation(line_number, rule))
    return tuple(violations)
