"""Structured STATE summaries and their validation rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from .types import ContextMessage


STATE_FIELDS = ("what", "decisions", "artifacts", "open", "next")


class StateValidationError(ValueError):
    """Raised when a model-produced STATE or patch is unsafe to apply."""


def empty_state() -> dict[str, list[str]]:
    return {field: [] for field in STATE_FIELDS}


def _validate_field(field: str, value: Any) -> list[str]:
    if field not in STATE_FIELDS:
        raise StateValidationError(f"unknown STATE field: {field}")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise StateValidationError(f"STATE field {field!r} must be a list of strings")
    return list(value)


def validate_state(value: Mapping[str, Any], *, allow_partial: bool = False) -> dict[str, list[str]]:
    """Validate and copy a bounded STATE document.

    Partial values are useful for patches, while stored documents are always
    normalized to all five fields.  Unknown keys are rejected to prevent a
    small model from smuggling arbitrary control data into the session.
    """

    if not isinstance(value, Mapping):
        raise StateValidationError("STATE must be an object")
    result = empty_state()
    for key, item in value.items():
        if key == "delete":
            continue
        result[key] = _validate_field(key, item)
    if allow_partial:
        return {key: result[key] for key in value if key in STATE_FIELDS}
    return result


def apply_state_patch(
    current: Mapping[str, Any] | None,
    patch: Mapping[str, Any],
    *,
    allow_delete: bool = False,
) -> dict[str, list[str]]:
    """Apply a schema-checked patch; deletion requires explicit consent."""

    base = validate_state(current or {})
    if not isinstance(patch, Mapping):
        raise StateValidationError("STATE patch must be an object")
    updates = validate_state(patch, allow_partial=True)
    for key, value in updates.items():
        base[key] = value

    deleted = patch.get("delete", [])
    if deleted:
        if not allow_delete:
            raise StateValidationError("STATE deletion requires explicit confirmation")
        if not isinstance(deleted, list) or any(item not in STATE_FIELDS for item in deleted):
            raise StateValidationError("STATE delete must list known fields")
        for key in deleted:
            base[key] = []
    return base


@dataclass(frozen=True, slots=True)
class SummaryResult:
    """A model-independent summary result accepted by the policy layer."""

    state: Mapping[str, Any]
    text: str | None = None
    source_message_ids: tuple[str, ...] = ()

    def validated_state(self) -> dict[str, list[str]]:
        return validate_state(self.state)

    def render(self, *, max_characters: int | None = None) -> str:
        state = self.validated_state()
        rendered = self.text or json.dumps(state, ensure_ascii=False, sort_keys=True)
        if max_characters is not None and max_characters >= 0:
            rendered = rendered[:max_characters]
        return rendered


class SummaryProvider(Protocol):
    def summarize(self, messages: Iterable[ContextMessage]) -> SummaryResult:
        """Return a bounded structured summary for omitted messages."""


@dataclass(frozen=True, slots=True)
class RuleBasedSummarizer:
    """Deterministic fallback used before a dedicated summary model exists."""

    max_items_per_field: int = 4
    max_item_characters: int = 160

    def summarize(self, messages: Iterable[ContextMessage]) -> SummaryResult:
        state = empty_state()
        source_ids: list[str] = []
        for message in messages:
            if message.message_id:
                source_ids.append(message.message_id)
            text = " ".join(message.content.split())[: self.max_item_characters]
            if not text:
                continue
            if message.role == "user":
                field = "what"
            elif message.is_output:
                field = "artifacts"
            elif message.role == "assistant":
                field = "decisions"
            else:
                field = "open"
            if len(state[field]) < self.max_items_per_field:
                state[field].append(text)
        if source_ids:
            state["next"].append(f"review omitted messages: {len(source_ids)}")
        return SummaryResult(state=state, source_message_ids=tuple(source_ids))
