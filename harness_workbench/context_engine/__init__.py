"""Small-model friendly, backend-agnostic context management."""

from .budget import ContextBudget, ContextBudgetError
from .compression import (
    COMPRESSION_STRATEGIES,
    STATE_VARIANTS,
    CompressionStep,
    compact_verbatim,
    render_state,
)
from .notices import ContextNotice
from .policy import (
    ContextBuildError,
    ContextPolicy,
    ContextPolicyConfig,
    ContextSnapshot,
    PinnedContentOverflow,
)
from .summarize import (
    STATE_FIELDS,
    StateValidationError,
    SummaryResult,
    apply_state_patch,
    validate_state,
)
from .types import ContextMessage, ContextLedgerEntry

__all__ = [
    "ContextBudget",
    "ContextBudgetError",
    "COMPRESSION_STRATEGIES",
    "STATE_VARIANTS",
    "CompressionStep",
    "compact_verbatim",
    "render_state",
    "ContextBuildError",
    "ContextLedgerEntry",
    "ContextMessage",
    "ContextNotice",
    "ContextPolicy",
    "ContextPolicyConfig",
    "ContextSnapshot",
    "PinnedContentOverflow",
    "STATE_FIELDS",
    "StateValidationError",
    "SummaryResult",
    "apply_state_patch",
    "validate_state",
]
