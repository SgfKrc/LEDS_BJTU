"""Context budget calculations shared by policy and adapters."""

from __future__ import annotations

from dataclasses import dataclass


class ContextBudgetError(ValueError):
    """Raised when a context budget cannot reserve generation capacity."""


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """A conservative input budget for one generation request.

    ``overhead`` covers the system prompt, tool definitions, asset cards and
    a small protocol margin.  The engine never silently borrows from
    ``max_new_tokens`` when calculating the input budget.
    """

    n_ctx: int
    max_new_tokens: int
    overhead: int = 512
    alignment: int = 1

    def __post_init__(self) -> None:
        for name in ("n_ctx", "max_new_tokens", "overhead"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContextBudgetError(f"{name} must be a non-negative integer")
        if self.n_ctx <= 0:
            raise ContextBudgetError("n_ctx must be greater than zero")
        if self.max_new_tokens <= 0:
            raise ContextBudgetError("max_new_tokens must be greater than zero")
        if not isinstance(self.alignment, int) or self.alignment <= 0:
            raise ContextBudgetError("alignment must be a positive integer")

    @property
    def raw_input_budget(self) -> int:
        """Return the unaligned budget, or raise before an invalid request."""

        remaining = self.n_ctx - self.max_new_tokens - self.overhead
        if remaining <= 0:
            raise ContextBudgetError(
                "max_new_tokens and overhead leave no input budget "
                f"(n_ctx={self.n_ctx}, max_new_tokens={self.max_new_tokens}, "
                f"overhead={self.overhead})"
            )
        return remaining

    @property
    def input_budget(self) -> int:
        """Return an aligned input budget safe for the backend."""

        raw = self.raw_input_budget
        aligned = raw - (raw % self.alignment)
        if aligned <= 0:
            raise ContextBudgetError(
                f"input budget {raw} is smaller than alignment {self.alignment}"
            )
        return aligned

    def as_dict(self) -> dict[str, int]:
        return {
            "n_ctx": self.n_ctx,
            "max_new_tokens": self.max_new_tokens,
            "overhead": self.overhead,
            "alignment": self.alignment,
            "input_budget": self.input_budget,
        }
