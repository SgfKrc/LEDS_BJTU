"""Offline ordering-policy simulation for logical TaskGraphs.

G4.2 compares ordinary topological ordering with a critical-path priority
candidate across deterministic lower/expected/upper duration scenarios.  It is
strictly shadow-only: the selected runtime policy remains topological and no
Scheduler or ready queue is imported or mutated.
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


ORDERING_PROFILE_SCHEMA_VERSION = "qlh.task_graph_ordering_profile.v1"
ORDERING_SIMULATION_SCHEMA_VERSION = "qlh.task_graph_ordering_simulation.v1"
ORDERING_SIMULATOR_VERSION = "task-ordering-simulation-v1"

_MAX_CONCURRENCY = 64
_MAX_READY_BYPASS = 1024
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_KEYS = frozenset({
    "schema_version",
    "profile_id",
    "max_concurrency",
    "max_ready_bypass",
    "contract_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version",
    "simulator_version",
    "mode",
    "status",
    "selected_graph_kind",
    "selected_ordering_policy",
    "graph_digest",
    "analysis_digest",
    "simulation_profile",
    "fallback",
    "policy_results",
    "comparison",
    "summary",
    "digest",
})
_POLICY_KEYS = frozenset({
    "policy",
    "guard_enabled",
    "scenarios",
    "summary",
})
_POLICY_SUMMARY_KEYS = frozenset({
    "scenario_count",
    "makespan_p50_ms",
    "makespan_p95_ms",
    "expected_makespan_ms",
    "expected_stage_completion_p50_ms",
    "expected_stage_completion_p95_ms",
    "expected_ready_wait_p95_ms",
    "expected_ready_wait_max_ms",
    "expected_utilization_ppm",
    "expected_protection_dispatch_count",
    "max_ready_bypass",
})
_SCENARIO_KEYS = frozenset({
    "duration_basis",
    "makespan_ms",
    "stage_completion_p50_ms",
    "stage_completion_p95_ms",
    "ready_wait_p50_ms",
    "ready_wait_p95_ms",
    "ready_wait_max_ms",
    "busy_slot_ms",
    "capacity_slot_ms",
    "utilization_ppm",
    "protection_dispatch_count",
    "max_ready_bypass",
    "dispatch_order",
    "stage_records",
})
_STAGE_RECORD_KEYS = frozenset({
    "stage_id",
    "dispatch_index",
    "ready_at_ms",
    "started_at_ms",
    "completed_at_ms",
    "duration_ms",
    "ready_wait_ms",
    "bypass_count",
    "protection_applied",
})
_COMPARISON_KEYS = frozenset({
    "available",
    "expected_makespan_delta_ms",
    "p95_makespan_delta_ms",
    "expected_improvement_ms",
    "p95_improvement_ms",
    "critical_non_regression",
    "recommendation",
    "comparison_digest",
})
_SUMMARY_KEYS = frozenset({
    "stage_count",
    "edge_count",
    "max_concurrency",
    "max_ready_bypass",
    "policy_count",
    "scenario_count",
    "topological_expected_makespan_ms",
    "critical_expected_makespan_ms",
    "topological_p95_makespan_ms",
    "critical_p95_makespan_ms",
})
_DURATION_BASES = ("lower", "expected", "upper")
_POLICIES = ("topological", "critical_path")
_FALLBACK_REASONS = frozenset({
    "analysis_timing_unavailable",
    "analysis_graph_mismatch",
})
_RECOMMENDATIONS = frozenset({
    "continue_shadow",
    "investigate_regression",
    "unavailable",
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


class TaskGraphOrderingError(ValueError):
    """Raised when an ordering profile or simulation report is malformed."""


def build_ordering_simulation_profile(
    *,
    profile_id: str = "g4-shadow-default.v1",
    max_concurrency: int = 2,
    max_ready_bypass: int = 2,
) -> dict[str, Any]:
    """Build a bounded offline scheduling simulation profile."""

    profile = {
        "schema_version": ORDERING_PROFILE_SCHEMA_VERSION,
        "profile_id": _identifier(profile_id, "profile_id"),
        "max_concurrency": _bounded_int(
            max_concurrency, 1, _MAX_CONCURRENCY, "max_concurrency",
        ),
        "max_ready_bypass": _bounded_int(
            max_ready_bypass, 0, _MAX_READY_BYPASS, "max_ready_bypass",
        ),
    }
    profile["contract_digest"] = _digest(profile)
    return validate_ordering_simulation_profile(profile)


def validate_ordering_simulation_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one G4.2 simulation profile."""

    if not isinstance(profile, Mapping) or set(profile) != _PROFILE_KEYS:
        raise TaskGraphOrderingError("ordering profile fields are invalid")
    _assert_no_forbidden_fields(profile)
    if profile.get("schema_version") != ORDERING_PROFILE_SCHEMA_VERSION:
        raise TaskGraphOrderingError("unsupported ordering profile schema")
    _identifier(profile.get("profile_id"), "profile_id")
    _bounded_int(
        profile.get("max_concurrency"),
        1,
        _MAX_CONCURRENCY,
        "max_concurrency",
    )
    _bounded_int(
        profile.get("max_ready_bypass"),
        0,
        _MAX_READY_BYPASS,
        "max_ready_bypass",
    )
    supplied_digest = _sha256(profile.get("contract_digest"), "contract_digest")
    unsigned = {
        key: value for key, value in profile.items() if key != "contract_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphOrderingError("ordering profile digest mismatch")
    return _detached(profile)


def simulate_task_graph_ordering(
    projection: Mapping[str, Any],
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare topological and critical-path list scheduling offline."""

    logical = require_graph_kind(projection, "logical_dag")
    checked_analysis = validate_structure_analysis(analysis)
    checked_profile = validate_ordering_simulation_profile(profile)

    fallback_reason = ""
    if checked_analysis["graph_digest"] != logical["digest"]:
        fallback_reason = "analysis_graph_mismatch"
    elif (
        checked_analysis["status"] != "evaluated"
        or not checked_analysis["critical_path"]["available"]
    ):
        fallback_reason = "analysis_timing_unavailable"

    if fallback_reason:
        status = "fallback"
        policy_results: list[dict[str, Any]] = []
        comparison = _unavailable_comparison()
    else:
        status = "evaluated"
        policy_results = [
            _simulate_policy(logical, checked_analysis, checked_profile, policy)
            for policy in _POLICIES
        ]
        comparison = _compare_policies(policy_results)

    by_policy = {result["policy"]: result for result in policy_results}
    topological_summary = by_policy.get("topological", {}).get("summary", {})
    critical_summary = by_policy.get("critical_path", {}).get("summary", {})
    report = {
        "schema_version": ORDERING_SIMULATION_SCHEMA_VERSION,
        "simulator_version": ORDERING_SIMULATOR_VERSION,
        "mode": "shadow",
        "status": status,
        "selected_graph_kind": "logical_dag",
        "selected_ordering_policy": "topological",
        "graph_digest": logical["digest"],
        "analysis_digest": checked_analysis["digest"],
        "simulation_profile": checked_profile,
        "fallback": {
            "used": status == "fallback",
            "reason_code": fallback_reason or "simulation_ready",
        },
        "policy_results": policy_results,
        "comparison": comparison,
        "summary": {
            "stage_count": logical["summary"]["stage_count"],
            "edge_count": logical["summary"]["edge_count"],
            "max_concurrency": checked_profile["max_concurrency"],
            "max_ready_bypass": checked_profile["max_ready_bypass"],
            "policy_count": len(policy_results),
            "scenario_count": (
                len(policy_results[0]["scenarios"]) if policy_results else 0
            ),
            "topological_expected_makespan_ms": topological_summary.get(
                "expected_makespan_ms", 0,
            ),
            "critical_expected_makespan_ms": critical_summary.get(
                "expected_makespan_ms", 0,
            ),
            "topological_p95_makespan_ms": topological_summary.get(
                "makespan_p95_ms", 0,
            ),
            "critical_p95_makespan_ms": critical_summary.get(
                "makespan_p95_ms", 0,
            ),
        },
    }
    report["digest"] = _digest(report)
    return validate_ordering_simulation(report)


def validate_ordering_simulation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted or operator-visible ordering simulation."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphOrderingError("ordering simulation fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != ORDERING_SIMULATION_SCHEMA_VERSION:
        raise TaskGraphOrderingError("unsupported ordering simulation schema")
    if report.get("simulator_version") != ORDERING_SIMULATOR_VERSION:
        raise TaskGraphOrderingError("unsupported ordering simulator version")
    if report.get("mode") != "shadow":
        raise TaskGraphOrderingError("ordering simulation must remain shadow-only")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphOrderingError("simulation cannot select a derived graph")
    if report.get("selected_ordering_policy") != "topological":
        raise TaskGraphOrderingError("simulation cannot change Scheduler ordering")
    _sha256(report.get("graph_digest"), "graph_digest")
    _sha256(report.get("analysis_digest"), "analysis_digest")
    profile = validate_ordering_simulation_profile(
        report.get("simulation_profile"),
    )

    status = report.get("status")
    if status not in {"evaluated", "fallback"}:
        raise TaskGraphOrderingError("ordering simulation status is invalid")
    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphOrderingError("ordering simulation fallback is invalid")
    if status == "evaluated":
        if fallback != {"used": False, "reason_code": "simulation_ready"}:
            raise TaskGraphOrderingError("evaluated simulation fallback is invalid")
    elif (
        fallback.get("used") is not True
        or fallback.get("reason_code") not in _FALLBACK_REASONS
    ):
        raise TaskGraphOrderingError("fallback simulation reason is invalid")

    policy_results = _validate_policy_results(
        report.get("policy_results"), profile,
    )
    comparison = _validate_comparison(report.get("comparison"))
    if status == "evaluated":
        if [result["policy"] for result in policy_results] != list(_POLICIES):
            raise TaskGraphOrderingError(
                "evaluated simulation requires both policies",
            )
        _validate_cross_policy_durations(policy_results)
        expected_comparison = _compare_policies(policy_results)
        if comparison != expected_comparison:
            raise TaskGraphOrderingError("policy comparison does not match results")
    elif policy_results or comparison != _unavailable_comparison():
        raise TaskGraphOrderingError(
            "fallback simulation cannot publish policy candidates",
        )

    summary = report.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphOrderingError("ordering simulation summary is invalid")
    by_policy = {result["policy"]: result for result in policy_results}
    topological = by_policy.get("topological", {}).get("summary", {})
    critical = by_policy.get("critical_path", {}).get("summary", {})
    expected_summary = {
        "stage_count": summary.get("stage_count"),
        "edge_count": summary.get("edge_count"),
        "max_concurrency": profile["max_concurrency"],
        "max_ready_bypass": profile["max_ready_bypass"],
        "policy_count": len(policy_results),
        "scenario_count": (
            len(policy_results[0]["scenarios"]) if policy_results else 0
        ),
        "topological_expected_makespan_ms": topological.get(
            "expected_makespan_ms", 0,
        ),
        "critical_expected_makespan_ms": critical.get(
            "expected_makespan_ms", 0,
        ),
        "topological_p95_makespan_ms": topological.get(
            "makespan_p95_ms", 0,
        ),
        "critical_p95_makespan_ms": critical.get("makespan_p95_ms", 0),
    }
    for key, expected in expected_summary.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphOrderingError(f"summary {key} must be non-negative")
        if value != expected:
            raise TaskGraphOrderingError(f"summary {key} does not match results")
    if status == "evaluated":
        scenario_stage_count = len(policy_results[0]["scenarios"][0][
            "stage_records"
        ])
        if summary["stage_count"] != scenario_stage_count:
            raise TaskGraphOrderingError("summary Stage count is inconsistent")
    elif summary["policy_count"] or summary["scenario_count"]:
        raise TaskGraphOrderingError("fallback summary contains simulations")

    supplied_digest = _sha256(report.get("digest"), "simulation digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphOrderingError("ordering simulation digest mismatch")
    return _detached(report)


def _simulate_policy(
    logical: Mapping[str, Any],
    analysis: Mapping[str, Any],
    profile: Mapping[str, Any],
    policy: str,
) -> dict[str, Any]:
    timing_by_stage = {
        item["stage_id"]: item for item in analysis["stage_timings"]
    }
    critical_stage_ids = set(analysis["critical_path"]["stage_ids"])
    scenarios = [
        _simulate_scenario(
            logical,
            timing_by_stage,
            critical_stage_ids,
            profile,
            policy,
            basis,
        )
        for basis in _DURATION_BASES
    ]
    makespans = [scenario["makespan_ms"] for scenario in scenarios]
    expected_scenario = next(
        scenario for scenario in scenarios
        if scenario["duration_basis"] == "expected"
    )
    return {
        "policy": policy,
        "guard_enabled": policy == "critical_path",
        "scenarios": scenarios,
        "summary": {
            "scenario_count": len(scenarios),
            "makespan_p50_ms": _percentile(makespans, 50),
            "makespan_p95_ms": _percentile(makespans, 95),
            "expected_makespan_ms": expected_scenario["makespan_ms"],
            "expected_stage_completion_p50_ms": expected_scenario[
                "stage_completion_p50_ms"
            ],
            "expected_stage_completion_p95_ms": expected_scenario[
                "stage_completion_p95_ms"
            ],
            "expected_ready_wait_p95_ms": expected_scenario[
                "ready_wait_p95_ms"
            ],
            "expected_ready_wait_max_ms": expected_scenario[
                "ready_wait_max_ms"
            ],
            "expected_utilization_ppm": expected_scenario["utilization_ppm"],
            "expected_protection_dispatch_count": expected_scenario[
                "protection_dispatch_count"
            ],
            "max_ready_bypass": max(
                scenario["max_ready_bypass"] for scenario in scenarios
            ),
        },
    }


def _simulate_scenario(
    logical: Mapping[str, Any],
    timing_by_stage: Mapping[str, Mapping[str, Any]],
    critical_stage_ids: set[str],
    profile: Mapping[str, Any],
    policy: str,
    basis: str,
) -> dict[str, Any]:
    stage_ids = sorted(timing_by_stage)
    predecessors = {stage_id: set() for stage_id in stage_ids}
    successors = {stage_id: set() for stage_id in stage_ids}
    for edge in logical["edges"]:
        source = _stage_from_node_id(edge["source_node_id"])
        target = _stage_from_node_id(edge["target_node_id"])
        predecessors[target].add(source)
        successors[source].add(target)
    duration_field = f"duration_{basis}_ms"
    durations = {
        stage_id: timing_by_stage[stage_id][duration_field]
        for stage_id in stage_ids
    }
    remaining = {
        stage_id: timing_by_stage[stage_id]["remaining_expected_ms"]
        for stage_id in stage_ids
    }
    max_concurrency = profile["max_concurrency"]
    max_bypass = profile["max_ready_bypass"]
    guard_enabled = policy == "critical_path"

    ready: dict[str, dict[str, int]] = {
        stage_id: {"ready_at_ms": 0, "bypass_count": 0}
        for stage_id in stage_ids
        if not predecessors[stage_id]
    }
    running: list[tuple[int, str]] = []
    running_ids: set[str] = set()
    completed: set[str] = set()
    completed_at: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    current_time = 0

    while len(completed) < len(stage_ids):
        while ready and len(running) < max_concurrency:
            stage_id, protection_applied = _select_ready_stage(
                ready,
                policy,
                remaining,
                critical_stage_ids,
                max_bypass,
            )
            state = ready.pop(stage_id)
            start = current_time
            finish = start + durations[stage_id]
            records.append({
                "stage_id": stage_id,
                "dispatch_index": len(records),
                "ready_at_ms": state["ready_at_ms"],
                "started_at_ms": start,
                "completed_at_ms": finish,
                "duration_ms": durations[stage_id],
                "ready_wait_ms": start - state["ready_at_ms"],
                "bypass_count": state["bypass_count"],
                "protection_applied": protection_applied,
            })
            if guard_enabled:
                for waiting in ready.values():
                    waiting["bypass_count"] = min(
                        max_bypass,
                        waiting["bypass_count"] + 1,
                    )
            heapq.heappush(running, (finish, stage_id))
            running_ids.add(stage_id)

        if not running:
            raise TaskGraphOrderingError("simulation cannot make progress")
        current_time = running[0][0]
        finished_now = []
        while running and running[0][0] == current_time:
            finish, stage_id = heapq.heappop(running)
            running_ids.remove(stage_id)
            completed.add(stage_id)
            completed_at[stage_id] = finish
            finished_now.append(stage_id)
        for stage_id in sorted(finished_now):
            for child in sorted(successors[stage_id]):
                if (
                    child in completed
                    or child in running_ids
                    or child in ready
                    or not predecessors[child].issubset(completed)
                ):
                    continue
                ready[child] = {
                    "ready_at_ms": max(
                        completed_at[parent] for parent in predecessors[child]
                    ),
                    "bypass_count": 0,
                }

    completions = [record["completed_at_ms"] for record in records]
    waits = [record["ready_wait_ms"] for record in records]
    makespan = max(completions, default=0)
    busy_slot_ms = sum(record["duration_ms"] for record in records)
    capacity_slot_ms = makespan * max_concurrency
    utilization_ppm = (
        busy_slot_ms * 1_000_000 // capacity_slot_ms
        if capacity_slot_ms else 0
    )
    return {
        "duration_basis": basis,
        "makespan_ms": makespan,
        "stage_completion_p50_ms": _percentile(completions, 50),
        "stage_completion_p95_ms": _percentile(completions, 95),
        "ready_wait_p50_ms": _percentile(waits, 50),
        "ready_wait_p95_ms": _percentile(waits, 95),
        "ready_wait_max_ms": max(waits, default=0),
        "busy_slot_ms": busy_slot_ms,
        "capacity_slot_ms": capacity_slot_ms,
        "utilization_ppm": utilization_ppm,
        "protection_dispatch_count": sum(
            record["protection_applied"] for record in records
        ),
        "max_ready_bypass": max(
            (record["bypass_count"] for record in records), default=0,
        ),
        "dispatch_order": [record["stage_id"] for record in records],
        "stage_records": records,
    }


def _select_ready_stage(
    ready: Mapping[str, Mapping[str, int]],
    policy: str,
    remaining: Mapping[str, int],
    critical_stage_ids: set[str],
    max_bypass: int,
) -> tuple[str, bool]:
    if policy == "topological":
        return min(ready), False
    protected = [
        stage_id for stage_id, state in ready.items()
        if state["bypass_count"] >= max_bypass
    ]
    if protected:
        selected = min(
            protected,
            key=lambda stage_id: (
                ready[stage_id]["ready_at_ms"], stage_id,
            ),
        )
        return selected, True
    selected = min(
        ready,
        key=lambda stage_id: (
            0 if stage_id in critical_stage_ids else 1,
            -remaining[stage_id],
            stage_id,
        ),
    )
    return selected, False


def _compare_policies(
    policy_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_policy = {result["policy"]: result["summary"] for result in policy_results}
    topological = by_policy["topological"]
    critical = by_policy["critical_path"]
    expected_delta = (
        critical["expected_makespan_ms"]
        - topological["expected_makespan_ms"]
    )
    p95_delta = critical["makespan_p95_ms"] - topological["makespan_p95_ms"]
    non_regression = p95_delta <= 0
    base = {
        "available": True,
        "expected_makespan_delta_ms": expected_delta,
        "p95_makespan_delta_ms": p95_delta,
        "expected_improvement_ms": -expected_delta,
        "p95_improvement_ms": -p95_delta,
        "critical_non_regression": non_regression,
        "recommendation": (
            "continue_shadow" if non_regression else "investigate_regression"
        ),
    }
    return {**base, "comparison_digest": _digest(base)}


def _unavailable_comparison() -> dict[str, Any]:
    base = {
        "available": False,
        "expected_makespan_delta_ms": 0,
        "p95_makespan_delta_ms": 0,
        "expected_improvement_ms": 0,
        "p95_improvement_ms": 0,
        "critical_non_regression": False,
        "recommendation": "unavailable",
    }
    return {**base, "comparison_digest": _digest(base)}


def _validate_policy_results(
    value: Any, profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphOrderingError("policy results must be a list")
    results = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _POLICY_KEYS:
            raise TaskGraphOrderingError("policy result fields are invalid")
        policy = item.get("policy")
        if policy not in _POLICIES:
            raise TaskGraphOrderingError("ordering policy is unsupported")
        guard_enabled = item.get("guard_enabled")
        if not isinstance(guard_enabled, bool) or guard_enabled != (
            policy == "critical_path"
        ):
            raise TaskGraphOrderingError("policy guard flag is invalid")
        scenarios = _validate_scenarios(
            item.get("scenarios"), profile, guard_enabled,
        )
        summary = _validate_policy_summary(item.get("summary"), scenarios)
        results.append({
            "policy": policy,
            "guard_enabled": guard_enabled,
            "scenarios": scenarios,
            "summary": summary,
        })
    if len({result["policy"] for result in results}) != len(results):
        raise TaskGraphOrderingError("policy result IDs must be unique")
    return results


def _validate_scenarios(
    value: Any,
    profile: Mapping[str, Any],
    guard_enabled: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphOrderingError("policy scenarios must be a list")
    scenarios = [
        _validate_scenario(item, profile, guard_enabled) for item in value
    ]
    if [scenario["duration_basis"] for scenario in scenarios] != list(
        _DURATION_BASES
    ):
        raise TaskGraphOrderingError("duration scenarios are incomplete or unsorted")
    stage_sets = [
        {record["stage_id"] for record in scenario["stage_records"]}
        for scenario in scenarios
    ]
    if stage_sets and any(stage_set != stage_sets[0] for stage_set in stage_sets):
        raise TaskGraphOrderingError("scenario Stage coverage is inconsistent")
    return scenarios


def _validate_scenario(
    value: Any,
    profile: Mapping[str, Any],
    guard_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SCENARIO_KEYS:
        raise TaskGraphOrderingError("scenario fields are invalid")
    basis = value.get("duration_basis")
    if basis not in _DURATION_BASES:
        raise TaskGraphOrderingError("duration basis is unsupported")
    records = _validate_stage_records(
        value.get("stage_records"), profile, guard_enabled,
    )
    dispatch_order = list(_ordered_stage_ids(
        value.get("dispatch_order"), "dispatch_order",
    ))
    if dispatch_order != [record["stage_id"] for record in records]:
        raise TaskGraphOrderingError("dispatch order does not match records")
    completions = [record["completed_at_ms"] for record in records]
    waits = [record["ready_wait_ms"] for record in records]
    makespan = max(completions, default=0)
    busy = sum(record["duration_ms"] for record in records)
    capacity = makespan * profile["max_concurrency"]
    expected_values = {
        "makespan_ms": makespan,
        "stage_completion_p50_ms": _percentile(completions, 50),
        "stage_completion_p95_ms": _percentile(completions, 95),
        "ready_wait_p50_ms": _percentile(waits, 50),
        "ready_wait_p95_ms": _percentile(waits, 95),
        "ready_wait_max_ms": max(waits, default=0),
        "busy_slot_ms": busy,
        "capacity_slot_ms": capacity,
        "utilization_ppm": busy * 1_000_000 // capacity if capacity else 0,
        "protection_dispatch_count": sum(
            record["protection_applied"] for record in records
        ),
        "max_ready_bypass": max(
            (record["bypass_count"] for record in records), default=0,
        ),
    }
    for field_name, expected in expected_values.items():
        supplied = _nonnegative_int(value.get(field_name), field_name)
        if supplied != expected:
            raise TaskGraphOrderingError(
                f"scenario {field_name} does not match Stage records",
            )
    _validate_concurrency(records, profile["max_concurrency"])
    return {
        "duration_basis": basis,
        **expected_values,
        "dispatch_order": dispatch_order,
        "stage_records": records,
    }


def _validate_stage_records(
    value: Any,
    profile: Mapping[str, Any],
    guard_enabled: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TaskGraphOrderingError("Stage records must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _STAGE_RECORD_KEYS:
            raise TaskGraphOrderingError("Stage record fields are invalid")
        ready_at = _nonnegative_int(item.get("ready_at_ms"), "ready_at_ms")
        started = _nonnegative_int(item.get("started_at_ms"), "started_at_ms")
        completed = _nonnegative_int(
            item.get("completed_at_ms"), "completed_at_ms",
        )
        duration = _nonnegative_int(item.get("duration_ms"), "duration_ms")
        ready_wait = _nonnegative_int(item.get("ready_wait_ms"), "ready_wait_ms")
        bypass_count = _nonnegative_int(
            item.get("bypass_count"), "bypass_count",
        )
        protection = item.get("protection_applied")
        if not isinstance(protection, bool):
            raise TaskGraphOrderingError("protection_applied must be boolean")
        if started < ready_at or completed != started + duration:
            raise TaskGraphOrderingError("Stage record timing is invalid")
        if ready_wait != started - ready_at:
            raise TaskGraphOrderingError("Stage ready wait is invalid")
        if guard_enabled:
            if bypass_count > profile["max_ready_bypass"]:
                raise TaskGraphOrderingError("ready bypass bound was exceeded")
            if protection != (
                bypass_count >= profile["max_ready_bypass"]
            ):
                raise TaskGraphOrderingError("protection flag is inconsistent")
        elif bypass_count != 0 or protection:
            raise TaskGraphOrderingError(
                "topological baseline cannot claim starvation protection",
            )
        result.append({
            "stage_id": _identifier(item.get("stage_id"), "record stage_id"),
            "dispatch_index": _nonnegative_int(
                item.get("dispatch_index"), "dispatch_index",
            ),
            "ready_at_ms": ready_at,
            "started_at_ms": started,
            "completed_at_ms": completed,
            "duration_ms": duration,
            "ready_wait_ms": ready_wait,
            "bypass_count": bypass_count,
            "protection_applied": protection,
        })
    if [item["dispatch_index"] for item in result] != list(range(len(result))):
        raise TaskGraphOrderingError("Stage dispatch indexes are invalid")
    if len({item["stage_id"] for item in result}) != len(result):
        raise TaskGraphOrderingError("Stage record IDs must be unique")
    return result


def _validate_policy_summary(
    value: Any, scenarios: list[Mapping[str, Any]],
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_SUMMARY_KEYS:
        raise TaskGraphOrderingError("policy summary fields are invalid")
    expected_scenario = next(
        scenario for scenario in scenarios
        if scenario["duration_basis"] == "expected"
    )
    makespans = [scenario["makespan_ms"] for scenario in scenarios]
    expected = {
        "scenario_count": len(scenarios),
        "makespan_p50_ms": _percentile(makespans, 50),
        "makespan_p95_ms": _percentile(makespans, 95),
        "expected_makespan_ms": expected_scenario["makespan_ms"],
        "expected_stage_completion_p50_ms": expected_scenario[
            "stage_completion_p50_ms"
        ],
        "expected_stage_completion_p95_ms": expected_scenario[
            "stage_completion_p95_ms"
        ],
        "expected_ready_wait_p95_ms": expected_scenario["ready_wait_p95_ms"],
        "expected_ready_wait_max_ms": expected_scenario["ready_wait_max_ms"],
        "expected_utilization_ppm": expected_scenario["utilization_ppm"],
        "expected_protection_dispatch_count": expected_scenario[
            "protection_dispatch_count"
        ],
        "max_ready_bypass": max(
            scenario["max_ready_bypass"] for scenario in scenarios
        ),
    }
    for field_name, expected_value in expected.items():
        if _nonnegative_int(value.get(field_name), field_name) != expected_value:
            raise TaskGraphOrderingError(
                f"policy summary {field_name} does not match scenarios",
            )
    return expected


def _validate_comparison(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _COMPARISON_KEYS:
        raise TaskGraphOrderingError("policy comparison fields are invalid")
    available = value.get("available")
    non_regression = value.get("critical_non_regression")
    if not isinstance(available, bool) or not isinstance(non_regression, bool):
        raise TaskGraphOrderingError("policy comparison flags are invalid")
    recommendation = value.get("recommendation")
    if recommendation not in _RECOMMENDATIONS:
        raise TaskGraphOrderingError("policy recommendation is invalid")
    base = {
        "available": available,
        "expected_makespan_delta_ms": _signed_int(
            value.get("expected_makespan_delta_ms"),
            "expected_makespan_delta_ms",
        ),
        "p95_makespan_delta_ms": _signed_int(
            value.get("p95_makespan_delta_ms"), "p95_makespan_delta_ms",
        ),
        "expected_improvement_ms": _signed_int(
            value.get("expected_improvement_ms"), "expected_improvement_ms",
        ),
        "p95_improvement_ms": _signed_int(
            value.get("p95_improvement_ms"), "p95_improvement_ms",
        ),
        "critical_non_regression": non_regression,
        "recommendation": recommendation,
    }
    if _sha256(value.get("comparison_digest"), "comparison_digest") != _digest(base):
        raise TaskGraphOrderingError("policy comparison digest mismatch")
    return {**base, "comparison_digest": value["comparison_digest"]}


def _validate_cross_policy_durations(
    policy_results: list[Mapping[str, Any]],
) -> None:
    baseline = {
        scenario["duration_basis"]: {
            record["stage_id"]: record["duration_ms"]
            for record in scenario["stage_records"]
        }
        for scenario in policy_results[0]["scenarios"]
    }
    candidate = {
        scenario["duration_basis"]: {
            record["stage_id"]: record["duration_ms"]
            for record in scenario["stage_records"]
        }
        for scenario in policy_results[1]["scenarios"]
    }
    if baseline != candidate:
        raise TaskGraphOrderingError("policies did not use identical durations")


def _validate_concurrency(
    records: list[Mapping[str, Any]], max_concurrency: int,
) -> None:
    events: list[tuple[int, int]] = []
    for record in records:
        if record["duration_ms"] == 0:
            continue
        events.append((record["started_at_ms"], 1))
        events.append((record["completed_at_ms"], -1))
    active = 0
    for _timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0 or active > max_concurrency:
            raise TaskGraphOrderingError("scenario exceeds concurrency capacity")
    if active != 0:
        raise TaskGraphOrderingError("scenario concurrency events are incomplete")


def _stage_from_node_id(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id.startswith("stage:"):
        raise TaskGraphOrderingError("ordering simulation requires Stage edges")
    return _identifier(node_id.split(":", 1)[1], "edge stage_id")


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = (percentile * len(ordered) + 99) // 100 - 1
    return ordered[max(0, index)]


def _ordered_stage_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskGraphOrderingError(f"{field_name} must be a non-empty list")
    result = tuple(_identifier(item, field_name) for item in value)
    if len(set(result)) != len(result):
        raise TaskGraphOrderingError(f"{field_name} must be unique")
    return result


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphOrderingError("simulation keys must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphOrderingError(
                    f"simulation contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskGraphOrderingError(f"{field_name} must be non-negative")
    return value


def _signed_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskGraphOrderingError(f"{field_name} must be an integer")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise TaskGraphOrderingError(f"{field_name} is outside the safe range")
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphOrderingError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphOrderingError(f"{field_name} must be a SHA-256 digest")
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
    "ORDERING_PROFILE_SCHEMA_VERSION",
    "ORDERING_SIMULATION_SCHEMA_VERSION",
    "ORDERING_SIMULATOR_VERSION",
    "TaskGraphOrderingError",
    "build_ordering_simulation_profile",
    "simulate_task_graph_ordering",
    "validate_ordering_simulation",
    "validate_ordering_simulation_profile",
]
