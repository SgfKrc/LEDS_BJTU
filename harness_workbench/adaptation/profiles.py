"""Immutable, path-free adaptation profiles for small models."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..context_engine import ContextBudget, ContextPolicyConfig
from ..context_engine.types import ContextMessage
from ..model_profiles.schema import ModelProfile


class AdaptationValidationError(ValueError):
    """Raised when an adaptation profile is not reproducible or unsafe."""


_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|(?:^|[\\s(])[\\/]{1,2})")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _path_free(value: Any) -> None:
    if isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise AdaptationValidationError("adaptation profiles cannot contain absolute paths")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _path_free(key)
            _path_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _path_free(child)


@dataclass(frozen=True, slots=True)
class PromptProfile:
    id: str
    family: str
    system_prompt: str
    stop: tuple[str, ...] = ()
    thinking: str = "unknown"
    tool_mode: str = "host_router"
    structured_output: str = "json_repair"
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.id or not self.family or not self.version:
            raise AdaptationValidationError("prompt id, family and version are required")
        if self.thinking not in {"unknown", "enabled", "disabled"}:
            raise AdaptationValidationError("thinking must be unknown, enabled, or disabled")
        if self.tool_mode not in {"host_router", "sidecar_candidate", "autonomous_tools", "disabled"}:
            raise AdaptationValidationError("unsupported tool mode")
        if any(not isinstance(item, str) for item in self.stop):
            raise AdaptationValidationError("stop values must be strings")
        _path_free(self.system_prompt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "system_prompt": self.system_prompt,
            "stop": list(self.stop),
            "thinking": self.thinking,
            "tool_mode": self.tool_mode,
            "structured_output": self.structured_output,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ContextStrategy:
    id: str
    config: ContextPolicyConfig = field(default_factory=ContextPolicyConfig)
    budget: ContextBudget | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise AdaptationValidationError("context strategy id is required")
        if self.description:
            _path_free(self.description)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "config": {
                "recent_turns": self.config.recent_turns,
                "summary_trigger_ratio": self.config.summary_trigger_ratio,
                "recent_turn_ratio": self.config.recent_turn_ratio,
                "max_tool_outputs": self.config.max_tool_outputs,
                "compression_strategy": self.config.compression_strategy,
                "state_variant": self.config.state_variant,
                "verbatim_max_characters": self.config.verbatim_max_characters,
                "memory_recall_ratio": self.config.memory_recall_ratio,
                "memory_recall_limit": self.config.memory_recall_limit,
            },
            "budget": self.budget.as_dict() if self.budget else None,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    id: str
    n_ctx: int
    max_new_tokens: int
    kv_cache: str = "unknown"
    gpu_layers: str | int = "auto"
    max_batch: int = 1
    expected_rss_bytes: int | None = None
    expected_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise AdaptationValidationError("resource profile id is required")
        if self.n_ctx <= 0 or self.max_new_tokens <= 0 or self.max_batch <= 0:
            raise AdaptationValidationError("resource values must be positive")
        for name in ("expected_rss_bytes", "expected_vram_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise AdaptationValidationError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "n_ctx": self.n_ctx,
            "max_new_tokens": self.max_new_tokens,
            "kv_cache": self.kv_cache,
            "gpu_layers": self.gpu_layers,
            "max_batch": self.max_batch,
            "expected_rss_bytes": self.expected_rss_bytes,
            "expected_vram_bytes": self.expected_vram_bytes,
        }


@dataclass(frozen=True, slots=True)
class AdaptationVariant:
    model_profile_digest: str
    prompt: PromptProfile
    context: ContextStrategy
    resource: ResourceProfile
    variant_id: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.model_profile_digest):
            raise AdaptationValidationError("model_profile_digest must be a SHA-256 digest")
        computed = self.compute_id()
        if self.variant_id and self.variant_id != computed:
            raise AdaptationValidationError("variant_id does not match adaptation inputs")
        object.__setattr__(self, "variant_id", computed)

    def compute_id(self) -> str:
        value = {
            "model_profile_digest": self.model_profile_digest,
            "prompt": self.prompt.as_dict(),
            "context": self.context.as_dict(),
            "resource": self.resource.as_dict(),
        }
        return hashlib.sha256(_canonical(value)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "model_profile_digest": self.model_profile_digest,
            "prompt": self.prompt.as_dict(),
            "context": self.context.as_dict(),
            "resource": self.resource.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class AdaptationPlan:
    model_profile: ModelProfile
    prompts: tuple[PromptProfile, ...]
    contexts: tuple[ContextStrategy, ...]
    resources: tuple[ResourceProfile, ...]
    holdout_fixture_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompts or not self.contexts or not self.resources:
            raise AdaptationValidationError("an adaptation plan needs prompt, context, and resource profiles")

    def variants(self) -> tuple[AdaptationVariant, ...]:
        return build_variant_matrix(
            self.model_profile,
            self.prompts,
            self.contexts,
            self.resources,
        )


def build_variant_matrix(
    model_profile: ModelProfile,
    prompts: Iterable[PromptProfile],
    contexts: Iterable[ContextStrategy],
    resources: Iterable[ResourceProfile],
) -> tuple[AdaptationVariant, ...]:
    prompt_values = tuple(prompts)
    context_values = tuple(contexts)
    resource_values = tuple(resources)
    if not prompt_values or not context_values or not resource_values:
        raise AdaptationValidationError("variant matrix dimensions cannot be empty")
    return tuple(
        AdaptationVariant(model_profile.digest, prompt, context, resource)
        for prompt in prompt_values
        for context in context_values
        for resource in resource_values
    )


def render_prompt_messages(
    prompt: PromptProfile,
    messages: Sequence[ContextMessage | Mapping[str, Any]],
    *,
    include_metadata: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Apply a profile's system policy without claiming tokenizer fidelity."""

    normalized = [ContextMessage.from_value(message) for message in messages]
    system = {"role": "system", "content": prompt.system_prompt}
    if include_metadata:
        system["metadata"] = {
            "prompt_profile_id": prompt.id,
            "prompt_family": prompt.family,
            "thinking": prompt.thinking,
            "tool_mode": prompt.tool_mode,
            "structured_output": prompt.structured_output,
        }
    output: list[dict[str, Any]] = [system]
    output.extend(message.as_dict() for message in normalized if message.role != "system")
    return tuple(output)
