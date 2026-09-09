"""Small, intentionally opinionated adaptation presets."""

from __future__ import annotations

from .profiles import ContextStrategy, PromptProfile, ResourceProfile
from ..context_engine import ContextBudget, ContextPolicyConfig


def builtin_adaptation_profiles(model_family: str) -> tuple[tuple[PromptProfile, ...], tuple[ContextStrategy, ...], tuple[ResourceProfile, ...]]:
    family = model_family.lower()
    if "gemma" in family:
        prompt_family = "gemma_chat_v1"
        stop = ("<end_of_turn>",)
    elif "qwen" in family or family.startswith("qw"):
        prompt_family = "qwen_chat_v1"
        stop = ("<|im_end|>",)
    else:
        prompt_family = "generic_chat_v1"
        stop = ()
    prompts = (
        PromptProfile(
            id=f"{prompt_family}-minimal",
            family=prompt_family,
            system_prompt="Answer clearly and briefly. Preserve facts from the user.",
            stop=stop,
            thinking="disabled",
            tool_mode="host_router",
        ),
        PromptProfile(
            id=f"{prompt_family}-structured",
            family=prompt_family,
            system_prompt="Answer clearly. Use the supplied context and return the requested structure exactly.",
            stop=stop,
            thinking="disabled",
            tool_mode="host_router",
            structured_output="grammar_first",
        ),
    )
    contexts = (
        ContextStrategy(
            id="recent-window",
            config=ContextPolicyConfig(recent_turns=6, summary_trigger_ratio=1.0, recent_turn_ratio=0.55),
            description="Keep recent rounds; do not invoke a model summary.",
        ),
        ContextStrategy(
            id="state-summary",
            config=ContextPolicyConfig(recent_turns=4, summary_trigger_ratio=0.70, recent_turn_ratio=0.35),
            description="Fold old rounds into schema-validated STATE.",
        ),
    )
    resources = (
        ResourceProfile(
            id="8gb-safe",
            n_ctx=2048,
            max_new_tokens=256,
            kv_cache="q8_0",
            gpu_layers="auto",
            max_batch=1,
        ),
        ResourceProfile(
            id="8gb-balanced",
            n_ctx=4096,
            max_new_tokens=512,
            kv_cache="q8_0",
            gpu_layers="auto",
            max_batch=2,
        ),
    )
    return prompts, contexts, resources
