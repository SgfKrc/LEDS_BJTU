"""Value objects used by the context engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ContextMessage:
    """A normalized chat message with optional harness metadata.

    ``turn_id`` is the preferred round boundary.  If it is absent, the policy
    derives turns from user messages.  ``pinned`` content is never dropped;
    if it cannot fit, the build fails explicitly.
    """

    role: str
    content: str
    message_id: str | None = None
    turn_id: str | int | None = None
    pinned: bool = False
    kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("message role is required")
        if not isinstance(self.content, str):
            raise TypeError("context message content must be text")
        if self.kind is not None and not isinstance(self.kind, str):
            raise TypeError("message kind must be text when provided")

    @property
    def is_output(self) -> bool:
        return self.role == "tool" or self.kind in {"tool", "image"}

    @classmethod
    def from_value(cls, value: "ContextMessage | Mapping[str, Any]") -> "ContextMessage":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("messages must be ContextMessage or mapping values")
        content = value.get("content", "")
        if isinstance(content, (list, tuple, dict)):
            # Multimodal adapters can preserve their own structure later; the
            # S1 context engine uses a stable textual representation for cost.
            import json

            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        return cls(
            role=str(value.get("role", "")),
            content=content,
            message_id=value.get("message_id", value.get("id")),
            turn_id=value.get("turn_id"),
            pinned=bool(value.get("pinned", False)),
            kind=value.get("kind"),
            metadata=dict(value.get("metadata", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.message_id is not None:
            value["message_id"] = self.message_id
        if self.turn_id is not None:
            value["turn_id"] = self.turn_id
        if self.pinned:
            value["pinned"] = True
        if self.kind is not None:
            value["kind"] = self.kind
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True, slots=True)
class ContextLedgerEntry:
    """Per-message accounting retained in a context snapshot."""

    message_id: str | None
    role: str
    turn_id: str | int | None
    tokens: int
    retained: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "turn_id": self.turn_id,
            "tokens": self.tokens,
            "retained": self.retained,
            "reason": self.reason,
        }
