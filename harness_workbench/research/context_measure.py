"""Deterministic context-strategy measurement for ``EX-CTX-MEAS-01``.

This module measures context retention, not model answer quality.  It uses a
fixed thirty-round conversation, the existing context policy, an isolated
SQLite memory store, and the rule-based STATE fallback.  No model weights,
network calls, or user paths enter the report.
"""

from __future__ import annotations

import hashlib
import json
import re
import gc
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Sequence

from ..context_engine import ContextBudget, ContextMessage, ContextPolicy, ContextPolicyConfig
from ..context_engine.tokenizer import HeuristicTokenizer, TokenCounter, count_message
from ..memory import MemoryStore


SCHEMA = "qlh.harness.context_measure.v1"
DEFAULT_BUDGETS = (64, 96, 128, 192, 256, 384)
STRATEGIES = ("window", "state", "memory")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextMeasureFixture:
    """A path-free, deterministic conversation with early facts and a query."""

    id: str
    rounds: int
    messages: tuple[ContextMessage, ...]
    early_facts: tuple[str, ...]
    query: str
    seed: int = 17
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.id or not _IDENTIFIER.fullmatch(self.id):
            raise ValueError("fixture id is invalid")
        if self.rounds != 30:
            raise ValueError("context measurement fixture must contain exactly 30 rounds")
        if len(self.messages) != self.rounds * 2:
            raise ValueError("fixture must contain a user and assistant message per round")
        if not self.early_facts or any(not item.strip() for item in self.early_facts):
            raise ValueError("fixture needs at least one early fact")
        if not self.query.strip() or self.seed < 0 or not self.version:
            raise ValueError("fixture query, seed, and version are required")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "rounds": self.rounds,
            "messages": [message.as_dict() for message in self.messages],
            "early_facts": list(self.early_facts),
            "query": self.query,
            "seed": self.seed,
            "version": self.version,
        }
        if include_digest:
            value["fixture_digest"] = self.digest
        return value


def build_context_measure_fixture(*, seed: int = 17) -> ContextMeasureFixture:
    """Build the shared 30-round fixture used by all strategy cells."""

    facts = (
        "Fact: early fact 00 says the project codename is SILVER-FOX.",
        "Fact: early fact 01 says the audit owner is TEAM-NORTH.",
        "Fact: early fact 02 says the release gate is REDACTED-FIXTURE.",
    )
    messages: list[ContextMessage] = []
    for round_id in range(30):
        if round_id < len(facts):
            content = facts[round_id]
        elif round_id == 29:
            content = "Which early facts were recorded? Answer from the supplied conversation."
        else:
            content = f"Round {round_id:02d} follow-up: discuss implementation detail {round_id}."
        messages.append(ContextMessage("user", content, f"ctx-u-{round_id:02d}", round_id))
        messages.append(
            ContextMessage(
                "assistant",
                f"Round {round_id:02d} acknowledged the current turn.",
                f"ctx-a-{round_id:02d}",
                round_id,
            )
        )
    return ContextMeasureFixture(
        id="context-30-round-early-facts",
        rounds=30,
        messages=tuple(messages),
        early_facts=facts,
        query="early fact",
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class ContextMeasureObservation:
    """One strategy/budget cell with only auditable, non-model metrics."""

    strategy: str
    input_budget: int
    input_tokens: int
    early_facts_total: int
    early_facts_recalled: int
    state_tokens: int
    memory_tokens: int
    recent_tokens: int
    omitted_messages: int
    memory_entries_written: int
    memory_entries_recalled: int
    compression_strategy: str
    fixture_digest: str
    seed: int
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False
    notices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError("unsupported context measurement strategy")
        if self.input_budget <= 0 or self.input_tokens < 0:
            raise ValueError("budget and token counts must be non-negative")
        if self.early_facts_total <= 0 or not 0 <= self.early_facts_recalled <= self.early_facts_total:
            raise ValueError("early fact counts are invalid")
        if any(value < 0 for value in (self.state_tokens, self.memory_tokens, self.recent_tokens, self.omitted_messages, self.memory_entries_written, self.memory_entries_recalled)):
            raise ValueError("measurement counts must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fixture_digest) or self.seed < 0:
            raise ValueError("fixture digest and seed are invalid")
        if self.runner_kind != "fixture" or self.network_used or self.weights_loaded:
            raise ValueError("context measurement is fixture-only and model-free")

    @property
    def early_fact_recall_rate(self) -> float:
        return self.early_facts_recalled / self.early_facts_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "input_budget": self.input_budget,
            "input_tokens": self.input_tokens,
            "early_facts_total": self.early_facts_total,
            "early_facts_recalled": self.early_facts_recalled,
            "early_fact_recall_rate": self.early_fact_recall_rate,
            "state_tokens": self.state_tokens,
            "memory_tokens": self.memory_tokens,
            "recent_tokens": self.recent_tokens,
            "omitted_messages": self.omitted_messages,
            "memory_entries_written": self.memory_entries_written,
            "memory_entries_recalled": self.memory_entries_recalled,
            "compression_strategy": self.compression_strategy,
            "fixture_digest": self.fixture_digest,
            "seed": self.seed,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
            "notices": list(self.notices),
        }


@dataclass(frozen=True, slots=True)
class ContextMeasureReport:
    """Chart-ready context curves and their reproducibility envelope."""

    fixture: ContextMeasureFixture
    budgets: tuple[int, ...]
    observations: tuple[ContextMeasureObservation, ...]
    seed: int = 17
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported context measurement schema")
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("budgets must be a sorted tuple of positive values")
        expected = len(self.budgets) * len(STRATEGIES)
        if len(self.observations) != expected:
            raise ValueError("one observation is required for every strategy/budget cell")
        if {item.strategy for item in self.observations} != set(STRATEGIES):
            raise ValueError("all context strategies must be measured")
        pairs = {(item.strategy, item.input_budget) for item in self.observations}
        if len(pairs) != expected or any((strategy, budget) not in pairs for strategy in STRATEGIES for budget in self.budgets):
            raise ValueError("observations must contain one unique cell per strategy and budget")
        if self.fixture.seed != self.seed:
            raise ValueError("fixture seed must match report seed")
        if any(item.fixture_digest != self.fixture.digest or item.seed != self.seed for item in self.observations):
            raise ValueError("observation provenance does not match report")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def series(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            strategy: tuple(
                {
                    "input_budget": item.input_budget,
                    "early_fact_recall_rate": item.early_fact_recall_rate,
                    "early_facts_recalled": item.early_facts_recalled,
                    "input_tokens": item.input_tokens,
                    "state_tokens": item.state_tokens,
                    "memory_tokens": item.memory_tokens,
                    "omitted_messages": item.omitted_messages,
                }
                for item in self.observations
                if item.strategy == strategy
            )
            for strategy in STRATEGIES
        }

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "fixture": self.fixture.as_dict(),
            "budgets": list(self.budgets),
            "observations": [item.as_dict() for item in self.observations],
            "series": {key: list(items) for key, items in self.series().items()},
            "seed": self.seed,
            "runner_kind": "fixture",
            "network_used": False,
            "weights_loaded": False,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        """Render a compact paper/defense-ready table without claiming quality."""

        lines = [
            "# EX-CTX-MEAS-01 context strategy measurement",
            "",
            f"- Fixture: `{self.fixture.id}` ({self.fixture.rounds} rounds, `{self.fixture.digest[:12]}`)",
            f"- Seed: `{self.seed}`; runner: `fixture`; weights loaded: `false`; network used: `false`",
            "- Metric: early-fact recall from the assembled context; this is not model answer quality.",
            "",
            "| strategy | input budget | recalled | recall rate | input tokens | STATE tokens | memory tokens | omitted messages |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in self.observations:
            lines.append(
                f"| `{item.strategy}` | {item.input_budget} | {item.early_facts_recalled}/{item.early_facts_total} | "
                f"{item.early_fact_recall_rate:.3f} | {item.input_tokens} | {item.state_tokens} | "
                f"{item.memory_tokens} | {item.omitted_messages} |"
            )
        lines.extend(
            (
                "",
                "## Chart data",
                "",
                "The `series` object in the JSON report is ready for a line chart with `input_budget` on the x-axis and `early_fact_recall_rate` on the y-axis.",
                "",
                f"Report digest: `{self.digest}`",
            )
        )
        return "\n".join(lines) + "\n"


def _budget(value: int) -> ContextBudget:
    return ContextBudget(n_ctx=value + 128, max_new_tokens=64, overhead=64)


def _strategy_policy(strategy: str, tokenizer: TokenCounter) -> ContextPolicy:
    if strategy == "window":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=1.0,
            recent_turn_ratio=0.60,
            compression_strategy="mask",
            memory_recall_ratio=0.0,
            memory_recall_limit=0,
        )
    elif strategy == "state":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=0.70,
            recent_turn_ratio=0.35,
            compression_strategy="state",
            state_variant="compact",
            memory_recall_ratio=0.0,
            memory_recall_limit=0,
        )
    elif strategy == "memory":
        config = ContextPolicyConfig(
            recent_turns=4,
            summary_trigger_ratio=1.0,
            recent_turn_ratio=0.45,
            compression_strategy="mask",
            memory_recall_ratio=0.30,
            memory_recall_limit=8,
        )
    else:
        raise ValueError("unsupported context measurement strategy")
    return ContextPolicy(config=config, tokenizer=tokenizer)


def _count_kind(messages: Iterable[ContextMessage], kind: str, tokenizer: TokenCounter) -> int:
    return sum(count_message(tokenizer, message) for message in messages if message.kind == kind)


def _measure_cell(
    fixture: ContextMeasureFixture,
    strategy: str,
    input_budget: int,
    *,
    tokenizer: TokenCounter,
) -> ContextMeasureObservation:
    policy = _strategy_policy(strategy, tokenizer)
    observation: ContextMeasureObservation
    temporary = TemporaryDirectory(prefix="qlh-context-measure-")
    directory = temporary.name
    store: MemoryStore | None = None
    snapshot = None
    try:
        store = MemoryStore(f"{directory}/memory.sqlite3") if strategy == "memory" else None
        snapshot = policy.build(
            fixture.messages,
            _budget(input_budget),
            memory_store=store,
            memory_owner_scope="context-measure",
            memory_source_session_id=fixture.id,
            memory_query=fixture.query if store is not None else None,
        )
        assembled = "\n".join(message.content for message in snapshot.messages)
        recalled = sum(1 for fact in fixture.early_facts if fact in assembled)
        omitted = sum(1 for entry in snapshot.ledger if not entry.retained)
        recent_tokens = sum(
            count_message(tokenizer, message)
            for message in snapshot.messages
            if message.kind not in {"summary", "memory", "verbatim"}
        )
        observation = ContextMeasureObservation(
            strategy=strategy,
            input_budget=input_budget,
            input_tokens=snapshot.input_tokens,
            early_facts_total=len(fixture.early_facts),
            early_facts_recalled=recalled,
            state_tokens=_count_kind(snapshot.messages, "summary", tokenizer),
            memory_tokens=_count_kind(snapshot.messages, "memory", tokenizer),
            recent_tokens=recent_tokens,
            omitted_messages=omitted,
            memory_entries_written=len(snapshot.memory_entry_ids),
            memory_entries_recalled=len(snapshot.memory_recall_ids),
            compression_strategy=snapshot.compression_strategy,
            fixture_digest=fixture.digest,
            seed=fixture.seed,
            notices=tuple(notice.code for notice in snapshot.notices),
        )
    finally:
        # sqlite3.Connection.__exit__ commits but does not close the object;
        # release the store before TemporaryDirectory removes the database on
        # Windows, where an open WAL handle prevents unlinking.
        del snapshot
        del store
        del policy
        gc.collect()
        temporary.cleanup()
    return observation


def run_context_measure(
    fixture: ContextMeasureFixture | None = None,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 17,
    tokenizer: TokenCounter | None = None,
) -> ContextMeasureReport:
    """Run all strategy/budget cells with no model or network access."""

    fixture = fixture or build_context_measure_fixture(seed=seed)
    if fixture.seed != seed:
        raise ValueError("fixture seed must match measurement seed")
    budget_values = tuple(int(value) for value in budgets)
    if not budget_values or tuple(sorted(set(budget_values))) != budget_values or any(value <= 0 for value in budget_values):
        raise ValueError("budgets must be sorted, unique, and positive")
    counter = tokenizer or HeuristicTokenizer()
    observations = tuple(
        _measure_cell(fixture, strategy, input_budget, tokenizer=counter)
        for strategy in STRATEGIES
        for input_budget in budget_values
    )
    return ContextMeasureReport(fixture, budget_values, observations, seed=seed)


def build_context_measure_report(
    fixture: ContextMeasureFixture | None = None,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    seed: int = 17,
    tokenizer: TokenCounter | None = None,
) -> ContextMeasureReport:
    """Named builder alias for callers that prefer report-oriented APIs."""

    return run_context_measure(fixture, budgets=budgets, seed=seed, tokenizer=tokenizer)


__all__ = [
    "CONTEXT_MEASURE_SCHEMA",
    "DEFAULT_BUDGETS",
    "STRATEGIES",
    "ContextMeasureFixture",
    "ContextMeasureObservation",
    "ContextMeasureReport",
    "build_context_measure_fixture",
    "build_context_measure_report",
    "run_context_measure",
]


CONTEXT_MEASURE_SCHEMA = SCHEMA
