from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from task_graph_speculation import (
    TaskGraphSpeculationError,
    build_speculation_profile,
    recommend_speculative_execution,
    validate_speculation_profile,
)


def _stage(**overrides):
    value = {
        "stage_id": "answer",
        "state": "running",
        "pure": True,
        "cancellable": True,
        "result_arbitration": "single_winner",
        "primary_provider": "worker-a",
        "candidate_provider": "worker-b",
        "latency": {"p95_ms": 900, "elapsed_ms": 950, "deadline_ms": 2000},
        "resources": {
            "extra_cost_ratio": 0.25,
            "idle_compatible_workers": 1,
            "provider_admitted": True,
            "failure_domain_changed": True,
        },
    }
    value.update(overrides)
    return value


def test_profile_digest_and_shadow_candidate_are_deterministic():
    profile = build_speculation_profile()
    report = recommend_speculative_execution([_stage()], profile=profile)
    assert report["status"] == "candidate"
    assert report["runtime_actions_enabled"] is False
    assert report["summary"]["winner_policy"] == "single_atomic_commit"
    again = recommend_speculative_execution([_stage()], profile=profile)
    assert report["digest"] == again["digest"]
    assert report["candidates"][0]["candidate_digest"] == again["candidates"][0]["candidate_digest"]


@pytest.mark.parametrize(("field", "reason"), [
    ("pure", "stage_not_pure"),
    ("cancellable", "stage_not_cancellable"),
    ("result_arbitration", "winner_policy_missing"),
])
def test_speculation_requires_safe_stage_semantics(field, reason):
    stage = _stage(**{field: False if field != "result_arbitration" else "best_effort"})
    report = recommend_speculative_execution([stage])
    assert report["status"] == "no_candidate"
    assert report["rejections"] == [{"stage_id": "answer", "reason_code": reason}]


@pytest.mark.parametrize(("change", "reason"), [
    ({"resources": {"extra_cost_ratio": 0.9, "idle_compatible_workers": 1, "provider_admitted": True, "failure_domain_changed": True}}, "extra_cost_budget_exceeded"),
    ({"resources": {"extra_cost_ratio": 0.1, "idle_compatible_workers": 0, "provider_admitted": True, "failure_domain_changed": True}}, "no_idle_compatible_worker"),
    ({"resources": {"extra_cost_ratio": 0.1, "idle_compatible_workers": 1, "provider_admitted": False, "failure_domain_changed": True}}, "candidate_provider_not_admitted"),
    ({"resources": {"extra_cost_ratio": 0.1, "idle_compatible_workers": 1, "provider_admitted": True, "failure_domain_changed": False}}, "failure_domain_not_changed"),
    ({"latency": {"p95_ms": 100, "elapsed_ms": 200, "deadline_ms": 2000}}, "tail_latency_below_threshold"),
])
def test_speculation_rejects_resource_and_latency_gates(change, reason):
    report = recommend_speculative_execution([_stage(**change)])
    assert report["rejections"] == [{"stage_id": "answer", "reason_code": reason}]


def test_profile_tampering_and_forbidden_evidence_fail_closed():
    profile = build_speculation_profile()
    tampered = copy.deepcopy(profile)
    tampered["max_extra_cost_ratio"] = 1.0
    with pytest.raises(TaskGraphSpeculationError):
        validate_speculation_profile(tampered)
    with pytest.raises(TaskGraphSpeculationError):
        recommend_speculative_execution([_stage(prompt="secret")])
