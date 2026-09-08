"""User-visible context management notices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ContextNotice:
    """A stable, serializable explanation for a context transformation."""

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("notice code and message are required")
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("notice severity must be info, warning, or error")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
            "severity": self.severity,
        }
