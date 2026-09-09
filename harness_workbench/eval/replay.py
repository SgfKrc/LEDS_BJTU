"""Deterministic adaptation replay with an injectable model runner."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from ..adaptation.profiles import AdaptationVariant, render_prompt_messages
from ..context_engine import ContextBudget, ContextPolicy
from .fixtures import EvalFixture, fixture_digest


class ReplayRunner(Protocol):
    def run(
        self,
        *,
        variant: AdaptationVariant,
        fixture: EvalFixture,
        messages: tuple[Mapping[str, Any], ...],
        seed: int,
    ) -> "ReplayObservation":
        """Execute one fixture; real and fake backends use this contract."""


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    output: str
    latency_ms: float
    first_token_ms: float | None = None
    rss_peak_bytes: int | None = None
    vram_peak_bytes: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False
    truncated: bool = False
    error_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        for name in ("rss_peak_bytes", "vram_peak_bytes", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "latency_ms": self.latency_ms,
            "first_token_ms": self.first_token_ms,
            "rss_peak_bytes": self.rss_peak_bytes,
            "vram_peak_bytes": self.vram_peak_bytes,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallback_used": self.fallback_used,
            "truncated": self.truncated,
            "error_code": self.error_code,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReplayItem:
    fixture_id: str
    fixture_digest: str
    observation: ReplayObservation
    messages_digest: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_digest": self.fixture_digest,
            "observation": self.observation.as_dict(),
            "messages_digest": self.messages_digest,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ReplayReport:
    variant: AdaptationVariant
    fixture_set_digest: str
    items: tuple[ReplayItem, ...]
    started_at: float
    runner_kind: str = "unknown"

    @property
    def replay_digest(self) -> str:
        value = {
            "variant_id": self.variant.variant_id,
            "fixture_set_digest": self.fixture_set_digest,
            "items": [item.as_dict() for item in self.items],
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self, *, include_outputs: bool = False) -> dict[str, Any]:
        items = []
        for item in self.items:
            value = item.as_dict()
            if not include_outputs:
                value["observation"] = {
                    key: data
                    for key, data in value["observation"].items()
                    if key != "output"
                }
            items.append(value)
        return {
            "variant": self.variant.as_dict(),
            "fixture_set_digest": self.fixture_set_digest,
            "items": items,
            "started_at": self.started_at,
            "runner_kind": self.runner_kind,
            "replay_digest": self.replay_digest,
            "network_used": False,
            "weights_loaded": False,
        }


def run_replay(
    variant: AdaptationVariant,
    fixtures: Sequence[EvalFixture],
    runner: ReplayRunner,
    *,
    context_policy: ContextPolicy | None = None,
    seed: int = 17,
    runner_kind: str = "injected",
) -> ReplayReport:
    if not fixtures:
        raise ValueError("replay requires at least one fixture")
    started_at = time.time()
    fixture_values = tuple(fixtures)
    items: list[ReplayItem] = []
    for index, fixture in enumerate(fixture_values):
        messages = tuple(dict(item) for item in fixture.messages)
        if context_policy is not None:
            budget = variant.context.budget or ContextBudget(
                n_ctx=variant.resource.n_ctx,
                max_new_tokens=variant.resource.max_new_tokens,
                overhead=512,
            )
            snapshot = context_policy.build(messages, budget)
            messages = tuple(message.as_dict() for message in snapshot.messages)
        rendered = render_prompt_messages(variant.prompt, messages, include_metadata=True)
        encoded = json.dumps(rendered, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        messages_digest = hashlib.sha256(encoded).hexdigest()
        observation = runner.run(
            variant=variant,
            fixture=fixture,
            messages=rendered,
            seed=seed + index,
        )
        if not isinstance(observation, ReplayObservation):
            raise TypeError("replay runner must return ReplayObservation")
        items.append(
            ReplayItem(
                fixture_id=fixture.id,
                fixture_digest=fixture.digest,
                observation=observation,
                messages_digest=messages_digest,
                seed=seed + index,
            )
        )
    return ReplayReport(
        variant=variant,
        fixture_set_digest=fixture_digest(fixture_values),
        items=tuple(items),
        started_at=started_at,
        runner_kind=runner_kind,
    )
