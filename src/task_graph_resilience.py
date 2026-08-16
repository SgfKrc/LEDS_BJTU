"""Shadow-only checkpoint and cascade-cancellation recommendations.

G4.3 consumes matching G4.1 structure analysis and G4.2 ordering evidence.
It emits bounded recommendations for future review, but never creates a
checkpoint, cancels a Stage, mutates a TaskGraph, or selects a runtime policy.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Mapping
from typing import Any

from task_graph_analysis import validate_structure_analysis
from task_graph_optimization import require_graph_kind
from task_graph_ordering import validate_ordering_simulation


RESILIENCE_PROFILE_SCHEMA_VERSION = "qlh.task_graph_resilience_profile.v1"
RESILIENCE_RECOMMENDATION_SCHEMA_VERSION = (
    "qlh.task_graph_resilience_recommendation.v1"
)
RESILIENCE_RECOMMENDER_VERSION = "task-resilience-recommendation-v1"

_MAX_DURATION_MS = 7 * 24 * 60 * 60 * 1000
_MAX_SCOPE_SIZE = 4096
_MAX_RECOMMENDATIONS = 256
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_KEYS = frozenset({
    "schema_version",
    "profile_id",
    "min_checkpoint_recompute_ms",
    "min_checkpoint_scope_size",
    "max_checkpoint_candidates",
    "min_cancel_scope_size",
    "max_cancel_scopes",
    "contract_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version",
    "recommender_version",
    "mode",
    "status",
    "selected_graph_kind",
    "runtime_actions_enabled",
    "graph_digest",
    "analysis_digest",
    "simulation_digest",
    "recommendation_profile",
    "fallback",
    "checkpoint_candidates",
    "checkpoint_rejections",
    "cancellation_scopes",
    "summary",
    "digest",
})
_CHECKPOINT_KEYS = frozenset({
    "stage_id",
    "priority_rank",
    "reason_code",
    "dominated_stage_ids",
    "dominated_stage_count",
    "upstream_recompute_expected_ms",
    "affected_work_expected_ms",
    "benefit_score",
    "critical_path_member",
    "post_dominator_boundary_node_id",
    "candidate_digest",
})
_CHECKPOINT_REJECTION_KEYS = frozenset({"stage_id", "reason_code"})
_CANCELLATION_KEYS = frozenset({
    "trigger_stage_id",
    "priority_rank",
    "reason_code",
    "applies_to_state",
    "affected_stage_ids",
    "affected_stage_count",
    "affected_work_expected_ms",
    "contains_side_effect_stage",
    "post_dominator_boundary_node_id",
    "scope_digest",
})
_SUMMARY_KEYS = frozenset({
    "stage_count",
    "edge_count",
    "checkpoint_candidate_count",
    "checkpoint_rejection_count",
    "checkpoint_affected_stage_count",
    "checkpoint_total_benefit_score",
    "cancellation_scope_count",
    "cancellation_affected_stage_count",
    "cancellation_total_affected_work_ms",
    "runtime_action_count",
})
_CHECKPOINT_REASONS = frozenset({
    "critical_dominator_recompute_boundary",
    "dominator_recompute_boundary",
})
_CHECKPOINT_REJECTION_REASONS = frozenset({
    "exit_stage",
    "stage_not_pure",
    "dominated_scope_too_small",
    "recompute_cost_below_threshold",
    "candidate_limit_reached",
})
_CANCELLATION_REASONS = frozenset({
    "dependency_success_threshold_exhausted",
})
_FALLBACK_REASONS = frozenset({
    "evidence_graph_mismatch",
    "analysis_unavailable",
    "simulation_unavailable",
    "evidence_digest_mismatch",
})
_FORBIDDEN_KEYS = frozenset({
    "body",
    "config",
    "content",
    "error",
    "history",
    "output",
    "path",
    "prompt",
    "raw",
    "root_input",
    "secret",
    "token",
    "url",
})


class TaskGraphResilienceError(ValueError):
    """Raised when a resilience profile or recommendation is malformed."""


def build_resilience_recommendation_profile(
    *,
    profile_id: str = "g4-resilience-default.v1",
    min_checkpoint_recompute_ms: int = 1,
    min_checkpoint_scope_size: int = 1,
    max_checkpoint_candidates: int = 8,
    min_cancel_scope_size: int = 1,
    max_cancel_scopes: int = 16,
) -> dict[str, Any]:
    """Build one bounded G4.3 recommendation profile."""

    profile = {
        "schema_version": RESILIENCE_PROFILE_SCHEMA_VERSION,
        "profile_id": _identifier(profile_id, "profile_id"),
        "min_checkpoint_recompute_ms": _bounded_int(
            min_checkpoint_recompute_ms,
            1,
            _MAX_DURATION_MS,
            "min_checkpoint_recompute_ms",
        ),
        "min_checkpoint_scope_size": _bounded_int(
            min_checkpoint_scope_size,
            1,
            _MAX_SCOPE_SIZE,
            "min_checkpoint_scope_size",
        ),
        "max_checkpoint_candidates": _bounded_int(
            max_checkpoint_candidates,
            1,
            _MAX_RECOMMENDATIONS,
            "max_checkpoint_candidates",
        ),
        "min_cancel_scope_size": _bounded_int(
            min_cancel_scope_size,
            1,
            _MAX_SCOPE_SIZE,
            "min_cancel_scope_size",
        ),
        "max_cancel_scopes": _bounded_int(
            max_cancel_scopes,
            1,
            _MAX_RECOMMENDATIONS,
            "max_cancel_scopes",
        ),
    }
    profile["contract_digest"] = _digest(profile)
    return validate_resilience_recommendation_profile(profile)


def validate_resilience_recommendation_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach a resilience recommendation profile."""

    if not isinstance(profile, Mapping) or set(profile) != _PROFILE_KEYS:
        raise TaskGraphResilienceError("resilience profile fields are invalid")
    _assert_no_forbidden_fields(profile)
    if profile.get("schema_version") != RESILIENCE_PROFILE_SCHEMA_VERSION:
        raise TaskGraphResilienceError("unsupported resilience profile schema")
    _identifier(profile.get("profile_id"), "profile_id")
    _bounded_int(
        profile.get("min_checkpoint_recompute_ms"),
        1,
        _MAX_DURATION_MS,
        "min_checkpoint_recompute_ms",
    )
    _bounded_int(
        profile.get("min_checkpoint_scope_size"),
        1,
        _MAX_SCOPE_SIZE,
        "min_checkpoint_scope_size",
    )
    _bounded_int(
        profile.get("max_checkpoint_candidates"),
        1,
        _MAX_RECOMMENDATIONS,
        "max_checkpoint_candidates",
    )
    _bounded_int(
        profile.get("min_cancel_scope_size"),
        1,
        _MAX_SCOPE_SIZE,
        "min_cancel_scope_size",
    )
    _bounded_int(
        profile.get("max_cancel_scopes"),
        1,
        _MAX_RECOMMENDATIONS,
        "max_cancel_scopes",
    )
    supplied_digest = _sha256(profile.get("contract_digest"), "contract_digest")
    unsigned = {
        key: value for key, value in profile.items() if key != "contract_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphResilienceError("resilience profile digest mismatch")
    return _detached(profile)


def recommend_task_graph_resilience(
    projection: Mapping[str, Any],
    analysis: Mapping[str, Any],
    simulation: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate checkpoint and not-started cancellation recommendations."""

    logical = require_graph_kind(projection, "logical_dag")
    checked_analysis = validate_structure_analysis(analysis)
    checked_simulation = validate_ordering_simulation(simulation)
    checked_profile = validate_resilience_recommendation_profile(profile)

    fallback_reason = _evidence_fallback_reason(
        logical, checked_analysis, checked_simulation,
    )
    if fallback_reason:
        status = "fallback"
        checkpoint_candidates: list[dict[str, Any]] = []
        checkpoint_rejections: list[dict[str, str]] = []
        cancellation_scopes: list[dict[str, Any]] = []
    else:
        status = "evaluated"
        checkpoint_candidates, checkpoint_rejections = (
            _checkpoint_recommendations(
                logical, checked_analysis, checked_profile,
            )
        )
        cancellation_scopes = _cancellation_recommendations(
            logical, checked_analysis, checked_profile,
        )

    report = {
        "schema_version": RESILIENCE_RECOMMENDATION_SCHEMA_VERSION,
        "recommender_version": RESILIENCE_RECOMMENDER_VERSION,
        "mode": "shadow",
        "status": status,
        "selected_graph_kind": "logical_dag",
        "runtime_actions_enabled": False,
        "graph_digest": logical["digest"],
        "analysis_digest": checked_analysis["digest"],
        "simulation_digest": checked_simulation["digest"],
        "recommendation_profile": checked_profile,
        "fallback": {
            "used": status == "fallback",
            "reason_code": fallback_reason or "recommendations_ready",
        },
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_rejections": checkpoint_rejections,
        "cancellation_scopes": cancellation_scopes,
        "summary": {
            "stage_count": logical["summary"]["stage_count"],
            "edge_count": logical["summary"]["edge_count"],
            "checkpoint_candidate_count": len(checkpoint_candidates),
            "checkpoint_rejection_count": len(checkpoint_rejections),
            "checkpoint_affected_stage_count": sum(
                item["dominated_stage_count"]
                for item in checkpoint_candidates
            ),
            "checkpoint_total_benefit_score": sum(
                item["benefit_score"] for item in checkpoint_candidates
            ),
            "cancellation_scope_count": len(cancellation_scopes),
            "cancellation_affected_stage_count": sum(
                item["affected_stage_count"] for item in cancellation_scopes
            ),
            "cancellation_total_affected_work_ms": sum(
                item["affected_work_expected_ms"]
                for item in cancellation_scopes
            ),
            "runtime_action_count": 0,
        },
    }
    report["digest"] = _digest(report)
    return validate_resilience_recommendation(report)


def validate_resilience_recommendation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted or operator-visible G4.3 recommendation."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphResilienceError("resilience recommendation fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != RESILIENCE_RECOMMENDATION_SCHEMA_VERSION:
        raise TaskGraphResilienceError("unsupported resilience recommendation schema")
    if report.get("recommender_version") != RESILIENCE_RECOMMENDER_VERSION:
        raise TaskGraphResilienceError("unsupported resilience recommender version")
    if report.get("mode") != "shadow":
        raise TaskGraphResilienceError("resilience recommendations must be shadow-only")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphResilienceError("recommendations cannot select a derived graph")
    if report.get("runtime_actions_enabled") is not False:
        raise TaskGraphResilienceError("runtime resilience actions must remain disabled")
    _sha256(report.get("graph_digest"), "graph_digest")
    _sha256(report.get("analysis_digest"), "analysis_digest")
    _sha256(report.get("simulation_digest"), "simulation_digest")
    profile = validate_resilience_recommendation_profile(
        report.get("recommendation_profile"),
    )

    status = report.get("status")
    if status not in {"evaluated", "fallback"}:
        raise TaskGraphResilienceError("resilience recommendation status is invalid")
    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphResilienceError("resilience fallback is invalid")
    if status == "evaluated":
        if fallback != {"used": False, "reason_code": "recommendations_ready"}:
            raise TaskGraphResilienceError("evaluated resilience fallback is invalid")
    elif (
        fallback.get("used") is not True
        or fallback.get("reason_code") not in _FALLBACK_REASONS
    ):
        raise TaskGraphResilienceError("resilience fallback reason is invalid")

    checkpoints = _validate_checkpoint_candidates(
        report.get("checkpoint_candidates"), profile,
    )
    rejections = _validate_checkpoint_rejections(
        report.get("checkpoint_rejections"),
    )
    cancellations = _validate_cancellation_scopes(
        report.get("cancellation_scopes"), profile,
    )
    checkpoint_stage_ids = {item["stage_id"] for item in checkpoints}
    rejection_stage_ids = {item["stage_id"] for item in rejections}
    if checkpoint_stage_ids.intersection(rejection_stage_ids):
        raise TaskGraphResilienceError(
            "checkpoint candidates and rejections overlap",
        )
    if status == "fallback" and (checkpoints or rejections or cancellations):
        raise TaskGraphResilienceError(
            "fallback recommendation cannot contain candidates",
        )

    summary = report.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphResilienceError("resilience summary is invalid")
    expected_summary = {
        "stage_count": summary.get("stage_count"),
        "edge_count": summary.get("edge_count"),
        "checkpoint_candidate_count": len(checkpoints),
        "checkpoint_rejection_count": len(rejections),
        "checkpoint_affected_stage_count": sum(
            item["dominated_stage_count"] for item in checkpoints
        ),
        "checkpoint_total_benefit_score": sum(
            item["benefit_score"] for item in checkpoints
        ),
        "cancellation_scope_count": len(cancellations),
        "cancellation_affected_stage_count": sum(
            item["affected_stage_count"] for item in cancellations
        ),
        "cancellation_total_affected_work_ms": sum(
            item["affected_work_expected_ms"] for item in cancellations
        ),
        "runtime_action_count": 0,
    }
    for field_name, expected in expected_summary.items():
        value = summary.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphResilienceError(
                f"summary {field_name} must be non-negative",
            )
        if value != expected:
            raise TaskGraphResilienceError(
                f"summary {field_name} does not match recommendations",
            )
    if status == "evaluated" and (
        summary["stage_count"]
        != summary["checkpoint_candidate_count"]
        + summary["checkpoint_rejection_count"]
    ):
        raise TaskGraphResilienceError(
            "every Stage requires one checkpoint decision",
        )
    if status == "fallback" and any(
        summary[field_name]
        for field_name in (
            "checkpoint_candidate_count",
            "checkpoint_rejection_count",
            "checkpoint_affected_stage_count",
            "checkpoint_total_benefit_score",
            "cancellation_scope_count",
            "cancellation_affected_stage_count",
            "cancellation_total_affected_work_ms",
            "runtime_action_count",
        )
    ):
        raise TaskGraphResilienceError("fallback summary contains runtime advice")

    supplied_digest = _sha256(report.get("digest"), "recommendation digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphResilienceError("resilience recommendation digest mismatch")
    return _detached(report)


def _evidence_fallback_reason(
    logical: Mapping[str, Any],
    analysis: Mapping[str, Any],
    simulation: Mapping[str, Any],
) -> str:
    if (
        analysis["graph_digest"] != logical["digest"]
        or simulation["graph_digest"] != logical["digest"]
    ):
        return "evidence_graph_mismatch"
    if analysis["status"] != "evaluated":
        return "analysis_unavailable"
    if simulation["status"] != "evaluated":
        return "simulation_unavailable"
    if simulation["analysis_digest"] != analysis["digest"]:
        return "evidence_digest_mismatch"
    return ""


def _checkpoint_recommendations(
    logical: Mapping[str, Any],
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    stage_analysis = {
        item["stage_id"]: item for item in analysis["stage_analysis"]
    }
    timings = {item["stage_id"]: item for item in analysis["stage_timings"]}
    exits = set(analysis["virtual_nodes"]["exit_stage_ids"])
    critical = set(analysis["critical_path"]["stage_ids"])
    admitted: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    for stage_id in sorted(nodes):
        node_id = f"stage:{stage_id}"
        dominated = sorted(
            other_stage_id
            for other_stage_id, item in stage_analysis.items()
            if other_stage_id != stage_id
            and node_id in item["dominator_node_ids"]
        )
        if stage_id in exits:
            rejections.append({"stage_id": stage_id, "reason_code": "exit_stage"})
            continue
        if not nodes[stage_id].get("provider_constraints", {}).get("pure", False):
            rejections.append({
                "stage_id": stage_id,
                "reason_code": "stage_not_pure",
            })
            continue
        if len(dominated) < profile["min_checkpoint_scope_size"]:
            rejections.append({
                "stage_id": stage_id,
                "reason_code": "dominated_scope_too_small",
            })
            continue
        upstream = timings[stage_id]["earliest_finish_expected_ms"]
        if upstream < profile["min_checkpoint_recompute_ms"]:
            rejections.append({
                "stage_id": stage_id,
                "reason_code": "recompute_cost_below_threshold",
            })
            continue
        affected_work = sum(
            timings[other_stage_id]["duration_expected_ms"]
            for other_stage_id in dominated
        )
        benefit_score = upstream * (len(dominated) + 1)
        critical_member = stage_id in critical
        base = {
            "stage_id": stage_id,
            "priority_rank": 0,
            "reason_code": (
                "critical_dominator_recompute_boundary"
                if critical_member else "dominator_recompute_boundary"
            ),
            "dominated_stage_ids": dominated,
            "dominated_stage_count": len(dominated),
            "upstream_recompute_expected_ms": upstream,
            "affected_work_expected_ms": affected_work,
            "benefit_score": benefit_score,
            "critical_path_member": critical_member,
            "post_dominator_boundary_node_id": stage_analysis[stage_id][
                "immediate_post_dominator_node_id"
            ],
        }
        admitted.append(base)

    admitted.sort(key=lambda item: (-item["benefit_score"], item["stage_id"]))
    selected = admitted[:profile["max_checkpoint_candidates"]]
    for item in admitted[profile["max_checkpoint_candidates"]:]:
        rejections.append({
            "stage_id": item["stage_id"],
            "reason_code": "candidate_limit_reached",
        })
    result = []
    for rank, item in enumerate(selected, start=1):
        base = {**item, "priority_rank": rank}
        result.append({**base, "candidate_digest": _digest(base)})
    rejections.sort(key=lambda item: item["stage_id"])
    return result, rejections


def _cancellation_recommendations(
    logical: Mapping[str, Any],
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    nodes = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    timings = {item["stage_id"]: item for item in analysis["stage_timings"]}
    stage_analysis = {
        item["stage_id"]: item for item in analysis["stage_analysis"]
    }
    predecessors = {stage_id: set() for stage_id in nodes}
    successors = {stage_id: set() for stage_id in nodes}
    for edge in logical["edges"]:
        source = _stage_from_node_id(edge["source_node_id"])
        target = _stage_from_node_id(edge["target_node_id"])
        predecessors[target].add(source)
        successors[source].add(target)
    topological = _topological_order(predecessors, successors)
    candidates = []
    for trigger_stage_id in sorted(nodes):
        affected = {trigger_stage_id}
        for stage_id in topological:
            if stage_id == trigger_stage_id or not predecessors[stage_id]:
                continue
            successful_capacity = sum(
                predecessor not in affected
                for predecessor in predecessors[stage_id]
            )
            minimum_successful = nodes[stage_id].get(
                "execution_constraints", {}
            ).get("minimum_successful_dependencies", len(predecessors[stage_id]))
            if successful_capacity < minimum_successful:
                affected.add(stage_id)
        affected.discard(trigger_stage_id)
        affected_stage_ids = sorted(affected)
        if len(affected_stage_ids) < profile["min_cancel_scope_size"]:
            continue
        affected_work = sum(
            timings[stage_id]["duration_expected_ms"]
            for stage_id in affected_stage_ids
        )
        base = {
            "trigger_stage_id": trigger_stage_id,
            "priority_rank": 0,
            "reason_code": "dependency_success_threshold_exhausted",
            "applies_to_state": "not_started",
            "affected_stage_ids": affected_stage_ids,
            "affected_stage_count": len(affected_stage_ids),
            "affected_work_expected_ms": affected_work,
            "contains_side_effect_stage": any(
                not nodes[stage_id].get("provider_constraints", {}).get(
                    "pure", False,
                )
                for stage_id in affected_stage_ids
            ),
            "post_dominator_boundary_node_id": stage_analysis[trigger_stage_id][
                "immediate_post_dominator_node_id"
            ],
        }
        candidates.append(base)
    candidates.sort(key=lambda item: (
        -item["affected_work_expected_ms"],
        -item["affected_stage_count"],
        item["trigger_stage_id"],
    ))
    result = []
    for rank, item in enumerate(
        candidates[:profile["max_cancel_scopes"]], start=1,
    ):
        base = {**item, "priority_rank": rank}
        result.append({**base, "scope_digest": _digest(base)})
    return result


def _validate_checkpoint_candidates(
    value: Any, profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphResilienceError("checkpoint candidates must be a list")
    if len(value) > profile["max_checkpoint_candidates"]:
        raise TaskGraphResilienceError("checkpoint candidate limit was exceeded")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _CHECKPOINT_KEYS:
            raise TaskGraphResilienceError("checkpoint candidate fields are invalid")
        stage_id = _identifier(item.get("stage_id"), "checkpoint stage_id")
        dominated = list(_sorted_stage_ids(
            item.get("dominated_stage_ids"), "dominated_stage_ids",
        ))
        if stage_id in dominated:
            raise TaskGraphResilienceError("checkpoint scope contains its owner")
        dominated_count = _nonnegative_int(
            item.get("dominated_stage_count"), "dominated_stage_count",
        )
        upstream = _nonnegative_int(
            item.get("upstream_recompute_expected_ms"),
            "upstream_recompute_expected_ms",
        )
        affected_work = _nonnegative_int(
            item.get("affected_work_expected_ms"),
            "affected_work_expected_ms",
        )
        benefit_score = _nonnegative_int(
            item.get("benefit_score"), "benefit_score",
        )
        critical = item.get("critical_path_member")
        if not isinstance(critical, bool):
            raise TaskGraphResilienceError("critical_path_member must be boolean")
        reason = item.get("reason_code")
        if reason not in _CHECKPOINT_REASONS or critical != (
            reason == "critical_dominator_recompute_boundary"
        ):
            raise TaskGraphResilienceError("checkpoint reason is inconsistent")
        if dominated_count != len(dominated):
            raise TaskGraphResilienceError("checkpoint scope count is invalid")
        if dominated_count < profile["min_checkpoint_scope_size"]:
            raise TaskGraphResilienceError("checkpoint scope is below profile")
        if upstream < profile["min_checkpoint_recompute_ms"]:
            raise TaskGraphResilienceError("checkpoint cost is below profile")
        if benefit_score != upstream * (dominated_count + 1):
            raise TaskGraphResilienceError("checkpoint benefit score is invalid")
        base = {
            "stage_id": stage_id,
            "priority_rank": _positive_int(
                item.get("priority_rank"), "checkpoint priority_rank",
            ),
            "reason_code": reason,
            "dominated_stage_ids": dominated,
            "dominated_stage_count": dominated_count,
            "upstream_recompute_expected_ms": upstream,
            "affected_work_expected_ms": affected_work,
            "benefit_score": benefit_score,
            "critical_path_member": critical,
            "post_dominator_boundary_node_id": _optional_node_id(
                item.get("post_dominator_boundary_node_id"),
                "checkpoint post_dominator_boundary_node_id",
            ),
        }
        if _sha256(item.get("candidate_digest"), "candidate_digest") != _digest(base):
            raise TaskGraphResilienceError("checkpoint candidate digest mismatch")
        result.append({**base, "candidate_digest": item["candidate_digest"]})
    if [item["priority_rank"] for item in result] != list(
        range(1, len(result) + 1)
    ):
        raise TaskGraphResilienceError("checkpoint priority ranks are invalid")
    if len({item["stage_id"] for item in result}) != len(result):
        raise TaskGraphResilienceError("checkpoint Stage IDs must be unique")
    if result != sorted(
        result, key=lambda item: (-item["benefit_score"], item["stage_id"]),
    ):
        raise TaskGraphResilienceError("checkpoint candidates are not ranked")
    return result


def _validate_checkpoint_rejections(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TaskGraphResilienceError("checkpoint rejections must be a list")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _CHECKPOINT_REJECTION_KEYS:
            raise TaskGraphResilienceError("checkpoint rejection fields are invalid")
        reason = item.get("reason_code")
        if reason not in _CHECKPOINT_REJECTION_REASONS:
            raise TaskGraphResilienceError("checkpoint rejection reason is invalid")
        result.append({
            "stage_id": _identifier(item.get("stage_id"), "rejection stage_id"),
            "reason_code": reason,
        })
    if result != sorted(result, key=lambda item: item["stage_id"]):
        raise TaskGraphResilienceError("checkpoint rejections must be sorted")
    if len({item["stage_id"] for item in result}) != len(result):
        raise TaskGraphResilienceError("checkpoint rejection IDs must be unique")
    return result


def _validate_cancellation_scopes(
    value: Any, profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphResilienceError("cancellation scopes must be a list")
    if len(value) > profile["max_cancel_scopes"]:
        raise TaskGraphResilienceError("cancellation scope limit was exceeded")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _CANCELLATION_KEYS:
            raise TaskGraphResilienceError("cancellation scope fields are invalid")
        trigger = _identifier(item.get("trigger_stage_id"), "trigger_stage_id")
        affected = list(_sorted_stage_ids(
            item.get("affected_stage_ids"), "affected_stage_ids",
        ))
        if trigger in affected:
            raise TaskGraphResilienceError("cancellation scope contains trigger")
        affected_count = _nonnegative_int(
            item.get("affected_stage_count"), "affected_stage_count",
        )
        if affected_count != len(affected):
            raise TaskGraphResilienceError("cancellation scope count is invalid")
        if affected_count < profile["min_cancel_scope_size"]:
            raise TaskGraphResilienceError("cancellation scope is below profile")
        side_effect = item.get("contains_side_effect_stage")
        if not isinstance(side_effect, bool):
            raise TaskGraphResilienceError(
                "contains_side_effect_stage must be boolean",
            )
        if item.get("reason_code") not in _CANCELLATION_REASONS:
            raise TaskGraphResilienceError("cancellation reason is invalid")
        if item.get("applies_to_state") != "not_started":
            raise TaskGraphResilienceError(
                "cancellation suggestions apply only to not-started Stages",
            )
        base = {
            "trigger_stage_id": trigger,
            "priority_rank": _positive_int(
                item.get("priority_rank"), "cancellation priority_rank",
            ),
            "reason_code": item["reason_code"],
            "applies_to_state": "not_started",
            "affected_stage_ids": affected,
            "affected_stage_count": affected_count,
            "affected_work_expected_ms": _nonnegative_int(
                item.get("affected_work_expected_ms"),
                "cancellation affected_work_expected_ms",
            ),
            "contains_side_effect_stage": side_effect,
            "post_dominator_boundary_node_id": _optional_node_id(
                item.get("post_dominator_boundary_node_id"),
                "cancellation post_dominator_boundary_node_id",
            ),
        }
        if _sha256(item.get("scope_digest"), "scope_digest") != _digest(base):
            raise TaskGraphResilienceError("cancellation scope digest mismatch")
        result.append({**base, "scope_digest": item["scope_digest"]})
    if [item["priority_rank"] for item in result] != list(
        range(1, len(result) + 1)
    ):
        raise TaskGraphResilienceError("cancellation priority ranks are invalid")
    if len({item["trigger_stage_id"] for item in result}) != len(result):
        raise TaskGraphResilienceError("cancellation triggers must be unique")
    if result != sorted(result, key=lambda item: (
        -item["affected_work_expected_ms"],
        -item["affected_stage_count"],
        item["trigger_stage_id"],
    )):
        raise TaskGraphResilienceError("cancellation scopes are not ranked")
    return result


def _topological_order(
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
) -> list[str]:
    indegree = {
        stage_id: len(parents) for stage_id, parents in predecessors.items()
    }
    ready = [stage_id for stage_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        stage_id = heapq.heappop(ready)
        result.append(stage_id)
        for child in sorted(successors[stage_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(predecessors):
        raise TaskGraphResilienceError("resilience analysis requires an acyclic graph")
    return result


def _stage_from_node_id(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id.startswith("stage:"):
        raise TaskGraphResilienceError("resilience analysis requires Stage edges")
    return _identifier(node_id.split(":", 1)[1], "edge stage_id")


def _sorted_stage_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskGraphResilienceError(f"{field_name} must be a non-empty list")
    result = tuple(_identifier(item, field_name) for item in value)
    if list(result) != sorted(set(result)):
        raise TaskGraphResilienceError(f"{field_name} must be sorted and unique")
    return result


def _optional_node_id(value: Any, field_name: str) -> str:
    if value == "":
        return ""
    value = _identifier(value, field_name)
    if not value.startswith(("stage:", "virtual:")):
        raise TaskGraphResilienceError(f"{field_name} has an invalid namespace")
    return value


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphResilienceError("recommendation keys must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphResilienceError(
                    f"recommendation contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskGraphResilienceError(f"{field_name} must be non-negative")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TaskGraphResilienceError(f"{field_name} must be positive")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise TaskGraphResilienceError(f"{field_name} is outside the safe range")
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphResilienceError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphResilienceError(f"{field_name} must be a SHA-256 digest")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _detached(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


__all__ = [
    "RESILIENCE_PROFILE_SCHEMA_VERSION",
    "RESILIENCE_RECOMMENDATION_SCHEMA_VERSION",
    "RESILIENCE_RECOMMENDER_VERSION",
    "TaskGraphResilienceError",
    "build_resilience_recommendation_profile",
    "recommend_task_graph_resilience",
    "validate_resilience_recommendation",
    "validate_resilience_recommendation_profile",
]
