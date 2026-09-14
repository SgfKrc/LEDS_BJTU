"""Model-specific prompt, context and resource adaptation plans."""

from .profiles import (
    AdaptationPlan,
    AdaptationVariant,
    ContextStrategy,
    PromptProfile,
    ResourceProfile,
    build_variant_matrix,
    render_prompt_messages,
)
from .cache_policy import PromptCacheViolation, find_prompt_cache_violations

__all__ = [
    "AdaptationPlan",
    "AdaptationVariant",
    "ContextStrategy",
    "PromptProfile",
    "ResourceProfile",
    "build_variant_matrix",
    "render_prompt_messages",
    "PromptCacheViolation",
    "find_prompt_cache_violations",
]
