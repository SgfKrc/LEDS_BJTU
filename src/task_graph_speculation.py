"""Shadow-only speculative execution admission for TG-OPT-G5.3-P0.

The module decides whether a second attempt would be worth considering from
bounded latency/resource summaries.  It never starts a Worker, mutates a
TaskGraph, creates an attempt, or commits a winner.  Real dual-Worker
execution and atomic winner arbitration remain an external gate.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


SPECULATION_PROFILE_SCHEMA_VERSION = "qlh.task_graph_speculation_profile.v1"
SPECULATION_RECOMMENDATION_SCHEMA_VERSION = "qlh.task_graph_speculation_recommendation.v1"
SPECULATION_RECOMMENDER_VERSION = "task-speculation-recommendation-v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_KEYS = frozenset({
    "schema_version", "profile_id", "min_tail_latency_ms", "min_elapsed_ratio",
    "max_extra_cost_ratio", "max_candidates", "require_dual_worker", "contract_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version", "recommender_version", "mode", "status", "runtime_actions_enabled",
    "profile", "candidates", "rejections", "summary", "digest",
})
_CANDIDATE_KEYS = frozenset({
    "stage_id", "primary_provider", "candidate_provider", "reason_code",
    "tail_latency_ms", "elapsed_ms", "deadline_ms", "extra_cost_ratio",
    "failure_domain_changed", "dual_worker_required", "candidate_digest",
})
_REJECTION_KEYS = frozenset({"stage_id", "reason_code"})
_FORBIDDEN = frozenset({
    "body", "content", "error", "history", "output", "path", "prompt", "raw",
    "request_id", "root_input", "secret", "token", "url", "tensor",
})


class TaskGraphSpeculationError(ValueError):
    pass


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TaskGraphSpeculationError(f"{field} is invalid")
    return value


def _bounded(value: object, minimum: float, maximum: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskGraphSpeculationError(f"{field} is invalid")
    number = float(value)
    if number != number or number == float("inf") or number == float("-inf") or not minimum <= number <= maximum:
        raise TaskGraphSpeculationError(f"{field} is outside the allowed range")
    return number


def _assert_safe(mapping: Mapping[str, Any]) -> None:
    if any(str(key).lower() in _FORBIDDEN for key in mapping):
        raise TaskGraphSpeculationError("speculation evidence contains forbidden fields")


def build_speculation_profile(
    *,
    profile_id: str = "g5-speculation-default.v1",
    min_tail_latency_ms: int = 250,
    min_elapsed_ratio: float = 1.0,
    max_extra_cost_ratio: float = 0.50,
    max_candidates: int = 8,
    require_dual_worker: bool = True,
) -> dict[str, Any]:
    profile = {
        "schema_version": SPECULATION_PROFILE_SCHEMA_VERSION,
        "profile_id": _identifier(profile_id, "profile_id"),
        "min_tail_latency_ms": int(_bounded(min_tail_latency_ms, 1, 7 * 24 * 60 * 60 * 1000, "min_tail_latency_ms")),
        "min_elapsed_ratio": _bounded(min_elapsed_ratio, 0.1, 100.0, "min_elapsed_ratio"),
        "max_extra_cost_ratio": _bounded(max_extra_cost_ratio, 0.0, 10.0, "max_extra_cost_ratio"),
        "max_candidates": int(_bounded(max_candidates, 1, 256, "max_candidates")),
        "require_dual_worker": bool(require_dual_worker),
    }
    profile["contract_digest"] = _digest(profile)
    return validate_speculation_profile(profile)


def validate_speculation_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, Mapping) or set(profile) != _PROFILE_KEYS:
        raise TaskGraphSpeculationError("speculation profile fields are invalid")
    _assert_safe(profile)
    if profile.get("schema_version") != SPECULATION_PROFILE_SCHEMA_VERSION:
        raise TaskGraphSpeculationError("unsupported speculation profile schema")
    _identifier(profile.get("profile_id"), "profile_id")
    min_tail = int(_bounded(profile.get("min_tail_latency_ms"), 1, 7 * 24 * 60 * 60 * 1000, "min_tail_latency_ms"))
    ratio = _bounded(profile.get("min_elapsed_ratio"), 0.1, 100.0, "min_elapsed_ratio")
    cost = _bounded(profile.get("max_extra_cost_ratio"), 0.0, 10.0, "max_extra_cost_ratio")
    max_candidates = int(_bounded(profile.get("max_candidates"), 1, 256, "max_candidates"))
    if not isinstance(profile.get("require_dual_worker"), bool):
        raise TaskGraphSpeculationError("require_dual_worker is invalid")
    unsigned = {
        "schema_version": profile["schema_version"], "profile_id": profile["profile_id"],
        "min_tail_latency_ms": min_tail, "min_elapsed_ratio": ratio,
        "max_extra_cost_ratio": cost, "max_candidates": max_candidates,
        "require_dual_worker": profile["require_dual_worker"],
    }
    supplied = profile.get("contract_digest")
    if not isinstance(supplied, str) or not _SHA256.fullmatch(supplied) or _digest(unsigned) != supplied:
        raise TaskGraphSpeculationError("speculation profile digest mismatch")
    return dict(unsigned, contract_digest=supplied)


def recommend_speculative_execution(
    stages: list[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return bounded candidates/rejections from summary-only evidence."""
    checked_profile = validate_speculation_profile(profile or build_speculation_profile())
    if not isinstance(stages, list) or len(stages) > 256:
        raise TaskGraphSpeculationError("stages are invalid")
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise TaskGraphSpeculationError("stage evidence is invalid")
        _assert_safe(stage)
        stage_id = _identifier(stage.get("stage_id"), "stage_id")
        reason = _rejection_reason(stage, checked_profile)
        if reason is not None:
            rejections.append({"stage_id": stage_id, "reason_code": reason})
            continue
        latency = stage["latency"]
        resources = stage["resources"]
        candidate = {
            "stage_id": stage_id,
            "primary_provider": _identifier(stage["primary_provider"], "primary_provider"),
            "candidate_provider": _identifier(stage["candidate_provider"], "candidate_provider"),
            "reason_code": "tail_latency_risk_within_cost_budget",
            "tail_latency_ms": int(latency["p95_ms"]),
            "elapsed_ms": int(latency["elapsed_ms"]),
            "deadline_ms": int(latency["deadline_ms"]),
            "extra_cost_ratio": float(resources["extra_cost_ratio"]),
            "failure_domain_changed": True,
            "dual_worker_required": bool(checked_profile["require_dual_worker"]),
        }
        candidate["candidate_digest"] = _digest(candidate)
        if len(candidates) < int(checked_profile["max_candidates"]):
            candidates.append(candidate)
        else:
            rejections.append({"stage_id": stage_id, "reason_code": "candidate_limit_reached"})
    status = "candidate" if candidates else "no_candidate"
    report = {
        "schema_version": SPECULATION_RECOMMENDATION_SCHEMA_VERSION,
        "recommender_version": SPECULATION_RECOMMENDER_VERSION,
        "mode": "shadow",
        "status": status,
        "runtime_actions_enabled": False,
        "profile": checked_profile,
        "candidates": candidates,
        "rejections": rejections,
        "summary": {
            "stage_count": len(stages),
            "candidate_count": len(candidates),
            "rejection_count": len(rejections),
            "dual_worker_required": bool(checked_profile["require_dual_worker"]),
            "winner_policy": "single_atomic_commit",
        },
    }
    report["digest"] = _digest(report)
    return report


def _rejection_reason(stage: Mapping[str, Any], profile: Mapping[str, Any]) -> str | None:
    required = ("stage_id", "primary_provider", "candidate_provider", "latency", "resources")
    if any(key not in stage for key in required):
        return "evidence_incomplete"
    if stage.get("state") not in {"running", None}:
        return "stage_not_running"
    if stage.get("pure") is not True:
        return "stage_not_pure"
    if stage.get("cancellable") is not True:
        return "stage_not_cancellable"
    if stage.get("result_arbitration") != "single_winner":
        return "winner_policy_missing"
    if stage.get("primary_provider") == stage.get("candidate_provider"):
        return "provider_not_independent"
    latency = stage["latency"]
    resources = stage["resources"]
    if not isinstance(latency, Mapping) or not isinstance(resources, Mapping):
        return "evidence_incomplete"
    try:
        p95 = _bounded(latency.get("p95_ms"), 1, 7 * 24 * 60 * 60 * 1000, "p95_ms")
        elapsed = _bounded(latency.get("elapsed_ms"), 0, 7 * 24 * 60 * 60 * 1000, "elapsed_ms")
        deadline = _bounded(latency.get("deadline_ms"), 1, 7 * 24 * 60 * 60 * 1000, "deadline_ms")
        extra = _bounded(resources.get("extra_cost_ratio"), 0.0, 10.0, "extra_cost_ratio")
    except TaskGraphSpeculationError:
        return "evidence_invalid"
    if p95 < float(profile["min_tail_latency_ms"]):
        return "tail_latency_below_threshold"
    if elapsed < p95 * float(profile["min_elapsed_ratio"]) and elapsed < deadline:
        return "tail_latency_not_reached"
    if extra > float(profile["max_extra_cost_ratio"]):
        return "extra_cost_budget_exceeded"
    if int(resources.get("idle_compatible_workers", 0)) < 1:
        return "no_idle_compatible_worker"
    if resources.get("provider_admitted") is not True:
        return "candidate_provider_not_admitted"
    if resources.get("failure_domain_changed") is not True:
        return "failure_domain_not_changed"
    return None


__all__ = [
    "SPECULATION_PROFILE_SCHEMA_VERSION", "SPECULATION_RECOMMENDATION_SCHEMA_VERSION",
    "TaskGraphSpeculationError", "build_speculation_profile", "validate_speculation_profile",
    "recommend_speculative_execution",
]
