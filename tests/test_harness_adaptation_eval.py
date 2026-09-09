"""S2.5 deterministic adaptation and evaluation workbench tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from harness_workbench.adaptation import (
    AdaptationPlan,
    ContextStrategy,
    PromptProfile,
    ResourceProfile,
    build_variant_matrix,
    render_prompt_messages,
)
from harness_workbench.adaptation.builtin import builtin_adaptation_profiles
from harness_workbench.context_engine import ContextBudget, ContextPolicy, ContextPolicyConfig
from harness_workbench.eval import (
    EvalFixture,
    ReplayObservation,
    build_evaluation_report,
    builtin_fixtures,
    pareto_frontier,
    run_replay,
    run_red_team,
)
from harness_workbench.eval.report import promotion_gate, summarize_replay
from tests.test_harness_model_profiles import _profile


class FixtureRunner:
    def __init__(self, outputs: dict[str, str] | None = None, *, rss: int = 100, vram: int = 50) -> None:
        self.outputs = outputs or {}
        self.rss = rss
        self.vram = vram
        self.calls = []

    def run(self, *, variant, fixture, messages, seed):
        self.calls.append((variant.variant_id, fixture.id, seed, messages))
        output = self.outputs.get(fixture.id)
        if output is None:
            expected = fixture.expected
            if expected.get("must_refuse"):
                output = "Refuse: request is not allowed."
            elif expected.get("json_keys"):
                output = json.dumps({key: "ok" for key in expected["json_keys"]})
            else:
                output = " ".join(expected.get("contains", ())) or "ok"
        return ReplayObservation(
            output=output,
            latency_ms=float(10 + seed),
            first_token_ms=2.0,
            rss_peak_bytes=self.rss,
            vram_peak_bytes=self.vram,
            input_tokens=len(messages),
            output_tokens=len(output.split()),
        )


def test_builtin_adaptation_matrix_has_two_dimensions_and_stable_ids() -> None:
    prompts, contexts, resources = builtin_adaptation_profiles("QW1.8B")
    profile = _profile()
    variants = build_variant_matrix(profile, prompts, contexts, resources)
    assert len(variants) == 8
    assert len({variant.variant_id for variant in variants}) == 8
    assert variants == build_variant_matrix(profile, prompts, contexts, resources)
    assert all("\\" not in json.dumps(variant.as_dict(), ensure_ascii=False) for variant in variants)


def test_adaptation_plan_and_rendering_are_model_specific() -> None:
    profile = _profile()
    prompts, contexts, resources = builtin_adaptation_profiles("Gemma-small")
    plan = AdaptationPlan(profile, prompts, contexts, resources, holdout_fixture_ids=("format-holdout-v1",))
    assert len(plan.variants()) == 8
    rendered = render_prompt_messages(
        prompts[0],
        ({"role": "user", "content": "hello"},),
        include_metadata=True,
    )
    assert rendered[0]["role"] == "system"
    assert rendered[0]["metadata"]["prompt_profile_id"] == prompts[0].id
    assert rendered[1]["content"] == "hello"


def test_replay_uses_context_policy_and_does_not_need_model_weights() -> None:
    profile = _profile()
    prompt = PromptProfile("fixture-prompt", "generic_chat_v1", "Answer briefly.")
    context = ContextStrategy(
        "fixture-context",
        config=ContextPolicyConfig(recent_turns=1, summary_trigger_ratio=0.7),
        budget=ContextBudget(n_ctx=256, max_new_tokens=64, overhead=32),
    )
    resource = ResourceProfile("fixture-resource", n_ctx=256, max_new_tokens=64)
    variant = build_variant_matrix(profile, (prompt,), (context,), (resource,))[0]
    fixtures = builtin_fixtures()[:2]
    runner = FixtureRunner()
    report = run_replay(variant, fixtures, runner, context_policy=ContextPolicy(config=context.config), seed=41)
    assert len(report.items) == 2
    assert report.items[0].seed == 41
    assert report.items[1].seed == 42
    assert report.replay_digest == run_replay(
        variant, fixtures, FixtureRunner(), context_policy=ContextPolicy(config=context.config), seed=41
    ).replay_digest
    assert all("SILVER" not in json.dumps(item.as_dict()) for item in report.items)
    assert report.as_dict()["weights_loaded"] is False
    assert report.as_dict()["network_used"] is False


def test_report_and_promotion_gate_require_holdout_quality() -> None:
    profile = _profile()
    prompt = PromptProfile("fixture-prompt", "generic_chat_v1", "Answer briefly.")
    context = ContextStrategy("fixture-context")
    resource = ResourceProfile("fixture-resource", n_ctx=2048, max_new_tokens=128)
    variant = build_variant_matrix(profile, (prompt,), (context,), (resource,))[0]
    fixtures = builtin_fixtures()
    runner = FixtureRunner(outputs={"format-holdout-v1": "wrong", "citation-holdout-v1": "wrong"})
    report = run_replay(variant, fixtures, runner)
    metrics = summarize_replay(report, fixtures)
    holdout = tuple(fixture for fixture in fixtures if fixture.holdout)
    holdout_report = run_replay(variant, holdout, runner)
    result = build_evaluation_report(
        report,
        fixtures,
        holdout_report=holdout_report,
        holdout_fixtures=holdout,
        red_team_report=run_red_team(),
    )
    assert result["promotion"]["status"] == "candidate"
    assert result["promotion"]["production_eligible"] is False
    assert "holdout_quality_below_threshold" in result["promotion"]["reasons"]
    assert result["metrics"]["red_team_blocked"] == 12
    assert result["metrics"]["red_team_block_rate"] == 1.0
    assert result["metrics"]["schema_valid_rate"] == 1.0
    assert result["metrics"]["unauthorized_pass_count"] == 0
    assert result["red_team"]["schema"] == "qlh.harness.red_team.v1"
    assert "wrong" not in json.dumps(result)
    assert metrics.quality_rate < 1.0


def test_promotion_gate_rejects_truncation_even_when_text_is_correct() -> None:
    profile = _profile()
    prompt = PromptProfile("fixture-prompt", "generic_chat_v1", "Answer briefly.")
    variant = build_variant_matrix(
        profile,
        (prompt,),
        (ContextStrategy("fixture-context"),),
        (ResourceProfile("fixture-resource", n_ctx=2048, max_new_tokens=128),),
    )[0]
    fixture = builtin_fixtures()[0]

    class TruncatingRunner(FixtureRunner):
        def run(self, **kwargs):
            observation = super().run(**kwargs)
            return replace(observation, truncated=True)

    report = run_replay(variant, (fixture,), TruncatingRunner())
    metrics = summarize_replay(report, (fixture,))
    decision = promotion_gate(metrics)
    assert decision.status == "candidate"
    assert "truncation_detected" in decision.reasons


def test_pareto_marks_lower_quality_and_higher_resource_variant_dominated() -> None:
    profile = _profile()
    prompt = PromptProfile("fixture-prompt", "generic_chat_v1", "Answer briefly.")
    contexts = (ContextStrategy("a"), ContextStrategy("b"))
    resources = (ResourceProfile("a", n_ctx=2048, max_new_tokens=128), ResourceProfile("b", n_ctx=4096, max_new_tokens=256))
    variants = build_variant_matrix(profile, (prompt,), contexts, resources)
    fixtures = (builtin_fixtures()[0],)
    reports = []
    summaries = []
    for variant, rss, output in ((variants[0], 100, "red"), (variants[-1], 200, "wrong")):
        report = run_replay(variant, fixtures, FixtureRunner(outputs={fixtures[0].id: output}, rss=rss, vram=rss))
        reports.append(report)
        summaries.append(summarize_replay(report, fixtures))
    points = pareto_frontier(zip(reports, summaries))
    dominated = {point.variant_id: point.dominated for point in points}
    assert dominated[variants[0].variant_id] is False
    assert dominated[variants[-1].variant_id] is True


def test_invalid_adaptation_profile_rejects_path() -> None:
    with pytest.raises(ValueError):
        PromptProfile("bad", "generic", "Read C:\\secret\\prompt.txt")
