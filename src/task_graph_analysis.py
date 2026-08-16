"""Read-only dominator and critical-path analysis for logical TaskGraphs.

G4.1 derives structural trees and a static critical-path candidate without
rewriting the logical DAG or changing Scheduler ordering.  Duration inputs are
explicit, bounded, digest-only contracts.  Missing or invalid duration data
leaves structural analysis available and falls back to ordinary topological
ordering.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from task_graph_optimization import require_graph_kind


DURATION_ESTIMATE_SCHEMA_VERSION = "qlh.task_graph_duration_estimate.v1"
STRUCTURE_ANALYSIS_SCHEMA_VERSION = "qlh.task_graph_structure_analysis.v1"
STRUCTURE_ANALYZER_VERSION = "task-structure-analysis-v1"

_VIRTUAL_ENTRY = "virtual:entry"
_VIRTUAL_EXIT = "virtual:exit"
_MAX_DURATION_MS = 7 * 24 * 60 * 60 * 1000
_MAX_SAMPLE_COUNT = 1_000_000_000
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DURATION_KEYS = frozenset({
    "schema_version",
    "stage_id",
    "estimate_source",
    "duration_lower_ms",
    "duration_expected_ms",
    "duration_upper_ms",
    "sample_count",
    "profile_version",
    "contract_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version",
    "analyzer_version",
    "mode",
    "status",
    "selected_graph_kind",
    "selected_ordering_policy",
    "graph_digest",
    "duration_contracts_digest",
    "fallback",
    "virtual_nodes",
    "dominator_tree_edges",
    "post_dominator_tree_edges",
    "stage_analysis",
    "stage_timings",
    "critical_path",
    "summary",
    "digest",
})
_VIRTUAL_KEYS = frozenset({
    "entry_node_id",
    "exit_node_id",
    "entry_stage_ids",
    "exit_stage_ids",
    "entry_virtual",
    "exit_virtual",
})
_TREE_EDGE_KEYS = frozenset({"parent_node_id", "child_node_id"})
_STAGE_ANALYSIS_KEYS = frozenset({
    "stage_id",
    "dominator_node_ids",
    "post_dominator_node_ids",
    "immediate_dominator_node_id",
    "immediate_post_dominator_node_id",
})
_TIMING_KEYS = frozenset({
    "stage_id",
    "contract_digest",
    "duration_lower_ms",
    "duration_expected_ms",
    "duration_upper_ms",
    "earliest_start_expected_ms",
    "earliest_finish_expected_ms",
    "remaining_expected_ms",
    "critical_path_member",
})
_CRITICAL_PATH_KEYS = frozenset({
    "available",
    "stage_ids",
    "duration_lower_ms",
    "duration_expected_ms",
    "duration_upper_ms",
    "path_digest",
})
_SUMMARY_KEYS = frozenset({
    "stage_count",
    "edge_count",
    "entry_count",
    "exit_count",
    "entry_virtual_count",
    "exit_virtual_count",
    "dominator_tree_edge_count",
    "post_dominator_tree_edge_count",
    "timing_stage_count",
    "critical_path_stage_count",
    "critical_path_expected_ms",
})
_ESTIMATE_SOURCES = frozenset({
    "stage_type_baseline",
    "historical_interval",
    "operator_profile",
})
_PARTIAL_REASONS = frozenset({
    "duration_contract_invalid",
    "duration_contract_coverage_mismatch",
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


class TaskGraphAnalysisError(ValueError):
    """Raised when an analysis contract or report is malformed."""


def build_stage_duration_estimate(
    stage_id: str,
    *,
    estimate_source: str,
    duration_lower_ms: int,
    duration_expected_ms: int,
    duration_upper_ms: int,
    sample_count: int = 0,
    profile_version: str = "baseline.v1",
) -> dict[str, Any]:
    """Build one bounded, digest-only Stage duration estimate."""

    contract = {
        "schema_version": DURATION_ESTIMATE_SCHEMA_VERSION,
        "stage_id": _identifier(stage_id, "stage_id"),
        "estimate_source": _enum(
            estimate_source, _ESTIMATE_SOURCES, "estimate_source",
        ),
        "duration_lower_ms": _duration(
            duration_lower_ms, "duration_lower_ms", allow_zero=True,
        ),
        "duration_expected_ms": _duration(
            duration_expected_ms, "duration_expected_ms",
        ),
        "duration_upper_ms": _duration(
            duration_upper_ms, "duration_upper_ms",
        ),
        "sample_count": _bounded_int(
            sample_count, 0, _MAX_SAMPLE_COUNT, "sample_count",
        ),
        "profile_version": _identifier(profile_version, "profile_version"),
    }
    contract["contract_digest"] = _digest(contract)
    return validate_stage_duration_estimate(contract)


def validate_stage_duration_estimate(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one Stage duration estimate."""

    if not isinstance(contract, Mapping) or set(contract) != _DURATION_KEYS:
        raise TaskGraphAnalysisError("duration estimate fields are invalid")
    _assert_no_forbidden_fields(contract)
    if contract.get("schema_version") != DURATION_ESTIMATE_SCHEMA_VERSION:
        raise TaskGraphAnalysisError("unsupported duration estimate schema")
    _identifier(contract.get("stage_id"), "stage_id")
    source = _enum(
        contract.get("estimate_source"),
        _ESTIMATE_SOURCES,
        "estimate_source",
    )
    lower = _duration(
        contract.get("duration_lower_ms"),
        "duration_lower_ms",
        allow_zero=True,
    )
    expected = _duration(
        contract.get("duration_expected_ms"), "duration_expected_ms",
    )
    upper = _duration(contract.get("duration_upper_ms"), "duration_upper_ms")
    if not lower <= expected <= upper:
        raise TaskGraphAnalysisError("duration estimate interval is invalid")
    sample_count = _bounded_int(
        contract.get("sample_count"),
        0,
        _MAX_SAMPLE_COUNT,
        "sample_count",
    )
    if source == "historical_interval" and sample_count < 3:
        raise TaskGraphAnalysisError(
            "historical duration estimate requires at least three samples",
        )
    if source != "historical_interval" and sample_count != 0:
        raise TaskGraphAnalysisError(
            "non-historical duration estimate cannot claim samples",
        )
    _identifier(contract.get("profile_version"), "profile_version")
    supplied_digest = _sha256(contract.get("contract_digest"), "contract_digest")
    unsigned = {
        key: value for key, value in contract.items() if key != "contract_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphAnalysisError("duration estimate digest mismatch")
    return _detached(contract)


def analyze_task_graph_structure(
    projection: Mapping[str, Any],
    duration_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive shadow-only structure and critical-path analysis."""

    logical = require_graph_kind(projection, "logical_dag")
    structural = _derive_structural_analysis(logical)
    stage_ids = {
        item["stage_id"] for item in structural["stage_analysis"]
    }
    contracts, partial_reason = _try_duration_contracts(
        duration_contracts, stage_ids,
    )
    if partial_reason:
        status = "partial"
        timings: list[dict[str, Any]] = []
        critical_path = _unavailable_critical_path()
        contracts_digest = _unavailable_contracts_digest(partial_reason)
        fallback = {
            "used": True,
            "reason_code": partial_reason,
        }
    else:
        status = "evaluated"
        timings, critical_path = _derive_critical_path(logical, contracts)
        contracts_digest = _contracts_digest(contracts)
        fallback = {
            "used": False,
            "reason_code": "analysis_ready",
        }

    report = {
        "schema_version": STRUCTURE_ANALYSIS_SCHEMA_VERSION,
        "analyzer_version": STRUCTURE_ANALYZER_VERSION,
        "mode": "shadow",
        "status": status,
        "selected_graph_kind": "logical_dag",
        "selected_ordering_policy": "topological",
        "graph_digest": logical["digest"],
        "duration_contracts_digest": contracts_digest,
        "fallback": fallback,
        "virtual_nodes": structural["virtual_nodes"],
        "dominator_tree_edges": structural["dominator_tree_edges"],
        "post_dominator_tree_edges": structural[
            "post_dominator_tree_edges"
        ],
        "stage_analysis": structural["stage_analysis"],
        "stage_timings": timings,
        "critical_path": critical_path,
        "summary": {
            "stage_count": logical["summary"]["stage_count"],
            "edge_count": logical["summary"]["edge_count"],
            "entry_count": len(structural["virtual_nodes"]["entry_stage_ids"]),
            "exit_count": len(structural["virtual_nodes"]["exit_stage_ids"]),
            "entry_virtual_count": int(
                structural["virtual_nodes"]["entry_virtual"]
            ),
            "exit_virtual_count": int(
                structural["virtual_nodes"]["exit_virtual"]
            ),
            "dominator_tree_edge_count": len(
                structural["dominator_tree_edges"]
            ),
            "post_dominator_tree_edge_count": len(
                structural["post_dominator_tree_edges"]
            ),
            "timing_stage_count": len(timings),
            "critical_path_stage_count": len(critical_path["stage_ids"]),
            "critical_path_expected_ms": critical_path[
                "duration_expected_ms"
            ],
        },
    }
    report["digest"] = _digest(report)
    return validate_structure_analysis(report)


def validate_structure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted or operator-visible G4.1 analysis report."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphAnalysisError("structure analysis fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != STRUCTURE_ANALYSIS_SCHEMA_VERSION:
        raise TaskGraphAnalysisError("unsupported structure analysis schema")
    if report.get("analyzer_version") != STRUCTURE_ANALYZER_VERSION:
        raise TaskGraphAnalysisError("unsupported structure analyzer version")
    if report.get("mode") != "shadow":
        raise TaskGraphAnalysisError("structure analysis must remain shadow-only")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphAnalysisError("analysis cannot select a derived graph")
    if report.get("selected_ordering_policy") != "topological":
        raise TaskGraphAnalysisError("analysis cannot change Scheduler ordering")
    _sha256(report.get("graph_digest"), "graph_digest")
    _sha256(report.get("duration_contracts_digest"), "duration_contracts_digest")

    status = report.get("status")
    if status not in {"evaluated", "partial"}:
        raise TaskGraphAnalysisError("structure analysis status is invalid")
    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphAnalysisError("structure analysis fallback is invalid")
    if status == "evaluated":
        if fallback != {"used": False, "reason_code": "analysis_ready"}:
            raise TaskGraphAnalysisError("evaluated analysis fallback is invalid")
    else:
        if (
            fallback.get("used") is not True
            or fallback.get("reason_code") not in _PARTIAL_REASONS
        ):
            raise TaskGraphAnalysisError("partial analysis fallback is invalid")
        if report["duration_contracts_digest"] != _unavailable_contracts_digest(
            fallback["reason_code"],
        ):
            raise TaskGraphAnalysisError("partial duration digest is invalid")

    stage_analysis = _validate_stage_analysis(report.get("stage_analysis"))
    stage_ids = [item["stage_id"] for item in stage_analysis]
    virtual = _validate_virtual_nodes(report.get("virtual_nodes"), stage_ids)
    _validate_dominator_sets(stage_analysis, virtual)
    dominator_edges = _validate_tree_edges(
        report.get("dominator_tree_edges"), "dominator",
    )
    post_dominator_edges = _validate_tree_edges(
        report.get("post_dominator_tree_edges"), "post-dominator",
    )
    expected_dominator_edges = sorted([
        {
            "parent_node_id": item["immediate_dominator_node_id"],
            "child_node_id": f"stage:{item['stage_id']}",
        }
        for item in stage_analysis
        if item["immediate_dominator_node_id"]
    ], key=_tree_edge_key)
    expected_post_edges = sorted([
        {
            "parent_node_id": item["immediate_post_dominator_node_id"],
            "child_node_id": f"stage:{item['stage_id']}",
        }
        for item in stage_analysis
        if item["immediate_post_dominator_node_id"]
    ], key=_tree_edge_key)
    if dominator_edges != expected_dominator_edges:
        raise TaskGraphAnalysisError("dominator tree does not match Stage analysis")
    if post_dominator_edges != expected_post_edges:
        raise TaskGraphAnalysisError(
            "post-dominator tree does not match Stage analysis",
        )

    timings = _validate_stage_timings(report.get("stage_timings"))
    critical_path = _validate_critical_path(report.get("critical_path"))
    if status == "evaluated":
        if [item["stage_id"] for item in timings] != stage_ids:
            raise TaskGraphAnalysisError(
                "evaluated analysis requires timing for every Stage",
            )
        expected_contracts_digest = _digest({
            "schema_version": 1,
            "contracts": [
                {
                    "stage_id": item["stage_id"],
                    "contract_digest": item["contract_digest"],
                }
                for item in timings
            ],
        })
        if report["duration_contracts_digest"] != expected_contracts_digest:
            raise TaskGraphAnalysisError("duration contracts digest is invalid")
        _validate_available_critical_path(
            critical_path, timings, virtual,
        )
    elif timings or critical_path["available"]:
        raise TaskGraphAnalysisError(
            "partial analysis cannot publish timing candidates",
        )
    elif critical_path != _unavailable_critical_path():
        raise TaskGraphAnalysisError("partial critical path must be unavailable")

    summary = report.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphAnalysisError("structure analysis summary is invalid")
    expected_summary = {
        "stage_count": len(stage_analysis),
        "edge_count": summary.get("edge_count"),
        "entry_count": len(virtual["entry_stage_ids"]),
        "exit_count": len(virtual["exit_stage_ids"]),
        "entry_virtual_count": int(virtual["entry_virtual"]),
        "exit_virtual_count": int(virtual["exit_virtual"]),
        "dominator_tree_edge_count": len(dominator_edges),
        "post_dominator_tree_edge_count": len(post_dominator_edges),
        "timing_stage_count": len(timings),
        "critical_path_stage_count": len(critical_path["stage_ids"]),
        "critical_path_expected_ms": critical_path["duration_expected_ms"],
    }
    for key, expected in expected_summary.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphAnalysisError(f"summary {key} must be non-negative")
        if value != expected:
            raise TaskGraphAnalysisError(f"summary {key} does not match analysis")

    supplied_digest = _sha256(report.get("digest"), "analysis digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphAnalysisError("structure analysis digest mismatch")
    return _detached(report)


def _derive_structural_analysis(logical: Mapping[str, Any]) -> dict[str, Any]:
    stage_nodes = {
        node["node_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    predecessors = {node_id: set() for node_id in stage_nodes}
    successors = {node_id: set() for node_id in stage_nodes}
    for edge in logical["edges"]:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        predecessors[target].add(source)
        successors[source].add(target)
    topological = _topological_order(predecessors, successors)
    entry_nodes = sorted(
        node_id for node_id in stage_nodes if not predecessors[node_id]
    )
    exit_nodes = sorted(
        node_id for node_id in stage_nodes if not successors[node_id]
    )

    dominators, entry_node_id = _dominators(
        topological, predecessors, entry_nodes,
    )
    post_dominators, exit_node_id = _post_dominators(
        topological, successors, exit_nodes,
    )
    immediate_dominators = _immediate_relations(dominators, entry_node_id)
    immediate_post_dominators = _immediate_relations(
        post_dominators, exit_node_id,
    )
    stage_analysis = []
    for node_id in sorted(stage_nodes, key=lambda value: stage_nodes[value]["stage_id"]):
        stage_analysis.append({
            "stage_id": stage_nodes[node_id]["stage_id"],
            "dominator_node_ids": sorted(dominators[node_id]),
            "post_dominator_node_ids": sorted(post_dominators[node_id]),
            "immediate_dominator_node_id": immediate_dominators[node_id],
            "immediate_post_dominator_node_id": (
                immediate_post_dominators[node_id]
            ),
        })
    dominator_edges = sorted([
        {
            "parent_node_id": parent,
            "child_node_id": node_id,
        }
        for node_id, parent in immediate_dominators.items()
        if parent
    ], key=_tree_edge_key)
    post_dominator_edges = sorted([
        {
            "parent_node_id": parent,
            "child_node_id": node_id,
        }
        for node_id, parent in immediate_post_dominators.items()
        if parent
    ], key=_tree_edge_key)
    return {
        "virtual_nodes": {
            "entry_node_id": entry_node_id,
            "exit_node_id": exit_node_id,
            "entry_stage_ids": sorted(
                stage_nodes[node_id]["stage_id"] for node_id in entry_nodes
            ),
            "exit_stage_ids": sorted(
                stage_nodes[node_id]["stage_id"] for node_id in exit_nodes
            ),
            "entry_virtual": entry_node_id == _VIRTUAL_ENTRY,
            "exit_virtual": exit_node_id == _VIRTUAL_EXIT,
        },
        "dominator_tree_edges": dominator_edges,
        "post_dominator_tree_edges": post_dominator_edges,
        "stage_analysis": stage_analysis,
    }


def _dominators(
    topological: Sequence[str],
    predecessors: Mapping[str, set[str]],
    entry_nodes: Sequence[str],
) -> tuple[dict[str, set[str]], str]:
    root = entry_nodes[0] if len(entry_nodes) == 1 else _VIRTUAL_ENTRY
    augmented_predecessors = {
        node_id: set(values) for node_id, values in predecessors.items()
    }
    if root == _VIRTUAL_ENTRY:
        for node_id in entry_nodes:
            augmented_predecessors[node_id].add(root)
    dominators: dict[str, set[str]] = {root: {root}}
    for node_id in topological:
        if node_id == root:
            continue
        parents = augmented_predecessors[node_id]
        common = set.intersection(*(dominators[parent] for parent in parents))
        dominators[node_id] = {node_id, *common}
    return dominators, root


def _post_dominators(
    topological: Sequence[str],
    successors: Mapping[str, set[str]],
    exit_nodes: Sequence[str],
) -> tuple[dict[str, set[str]], str]:
    root = exit_nodes[0] if len(exit_nodes) == 1 else _VIRTUAL_EXIT
    augmented_successors = {
        node_id: set(values) for node_id, values in successors.items()
    }
    if root == _VIRTUAL_EXIT:
        for node_id in exit_nodes:
            augmented_successors[node_id].add(root)
    post_dominators: dict[str, set[str]] = {root: {root}}
    for node_id in reversed(topological):
        if node_id == root:
            continue
        children = augmented_successors[node_id]
        common = set.intersection(*(
            post_dominators[child] for child in children
        ))
        post_dominators[node_id] = {node_id, *common}
    return post_dominators, root


def _immediate_relations(
    relation_sets: Mapping[str, set[str]], root: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for node_id, related in relation_sets.items():
        if node_id == root:
            result[node_id] = ""
            continue
        strict = related - {node_id}
        if not strict:
            raise TaskGraphAnalysisError("analysis relation has no root")
        result[node_id] = sorted(
            strict,
            key=lambda candidate: (-len(relation_sets[candidate]), candidate),
        )[0]
    return result


def _derive_critical_path(
    logical: Mapping[str, Any],
    contracts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = {
        node["node_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    predecessors = {node_id: set() for node_id in nodes}
    successors = {node_id: set() for node_id in nodes}
    for edge in logical["edges"]:
        predecessors[edge["target_node_id"]].add(edge["source_node_id"])
        successors[edge["source_node_id"]].add(edge["target_node_id"])
    topological = _topological_order(predecessors, successors)
    expected = {
        node_id: contracts[node["stage_id"]]["duration_expected_ms"]
        for node_id, node in nodes.items()
    }
    earliest_start: dict[str, int] = {}
    earliest_finish: dict[str, int] = {}
    path_predecessor: dict[str, str] = {}
    for node_id in topological:
        if predecessors[node_id]:
            parent = min(
                predecessors[node_id],
                key=lambda candidate: (-earliest_finish[candidate], candidate),
            )
            earliest_start[node_id] = earliest_finish[parent]
            path_predecessor[node_id] = parent
        else:
            earliest_start[node_id] = 0
            path_predecessor[node_id] = ""
        earliest_finish[node_id] = earliest_start[node_id] + expected[node_id]

    remaining: dict[str, int] = {}
    for node_id in reversed(topological):
        child_remaining = max(
            (remaining[child] for child in successors[node_id]),
            default=0,
        )
        remaining[node_id] = expected[node_id] + child_remaining

    exit_nodes = [node_id for node_id in topological if not successors[node_id]]
    critical_exit = min(
        exit_nodes,
        key=lambda candidate: (-earliest_finish[candidate], candidate),
    )
    critical_nodes = []
    cursor = critical_exit
    while cursor:
        critical_nodes.append(cursor)
        cursor = path_predecessor[cursor]
    critical_nodes.reverse()
    critical_stage_ids = [nodes[node_id]["stage_id"] for node_id in critical_nodes]
    critical_stage_set = set(critical_stage_ids)

    timings = []
    for node_id in sorted(nodes, key=lambda value: nodes[value]["stage_id"]):
        stage_id = nodes[node_id]["stage_id"]
        contract = contracts[stage_id]
        timings.append({
            "stage_id": stage_id,
            "contract_digest": contract["contract_digest"],
            "duration_lower_ms": contract["duration_lower_ms"],
            "duration_expected_ms": contract["duration_expected_ms"],
            "duration_upper_ms": contract["duration_upper_ms"],
            "earliest_start_expected_ms": earliest_start[node_id],
            "earliest_finish_expected_ms": earliest_finish[node_id],
            "remaining_expected_ms": remaining[node_id],
            "critical_path_member": stage_id in critical_stage_set,
        })
    critical_base = {
        "available": True,
        "stage_ids": critical_stage_ids,
        "duration_lower_ms": sum(
            contracts[stage_id]["duration_lower_ms"]
            for stage_id in critical_stage_ids
        ),
        "duration_expected_ms": sum(
            contracts[stage_id]["duration_expected_ms"]
            for stage_id in critical_stage_ids
        ),
        "duration_upper_ms": sum(
            contracts[stage_id]["duration_upper_ms"]
            for stage_id in critical_stage_ids
        ),
    }
    return timings, {
        **critical_base,
        "path_digest": _digest(critical_base),
    }


def _try_duration_contracts(
    contracts: Sequence[Mapping[str, Any]], stage_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], str]:
    if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
        return {}, "duration_contract_invalid"
    result: dict[str, dict[str, Any]] = {}
    try:
        for contract in contracts:
            checked = validate_stage_duration_estimate(contract)
            stage_id = checked["stage_id"]
            if stage_id in result:
                return {}, "duration_contract_invalid"
            result[stage_id] = checked
    except (TaskGraphAnalysisError, TypeError, ValueError):
        return {}, "duration_contract_invalid"
    if set(result) != stage_ids:
        return {}, "duration_contract_coverage_mismatch"
    return result, ""


def _contracts_digest(contracts: Mapping[str, Mapping[str, Any]]) -> str:
    return _digest({
        "schema_version": 1,
        "contracts": [
            {
                "stage_id": stage_id,
                "contract_digest": contracts[stage_id]["contract_digest"],
            }
            for stage_id in sorted(contracts)
        ],
    })


def _unavailable_contracts_digest(reason_code: str) -> str:
    return _digest({
        "schema_version": 1,
        "status": "unavailable",
        "reason_code": reason_code,
    })


def _unavailable_critical_path() -> dict[str, Any]:
    base = {
        "available": False,
        "stage_ids": [],
        "duration_lower_ms": 0,
        "duration_expected_ms": 0,
        "duration_upper_ms": 0,
    }
    return {**base, "path_digest": _digest(base)}


def _topological_order(
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
) -> list[str]:
    indegree = {
        node_id: len(parents) for node_id, parents in predecessors.items()
    }
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        result.append(node_id)
        for child in sorted(successors[node_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(predecessors):
        raise TaskGraphAnalysisError("structure analysis requires an acyclic graph")
    return result


def _validate_stage_analysis(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TaskGraphAnalysisError("Stage analysis must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _STAGE_ANALYSIS_KEYS:
            raise TaskGraphAnalysisError("Stage analysis fields are invalid")
        stage_id = _identifier(item.get("stage_id"), "stage analysis stage_id")
        result.append({
            "stage_id": stage_id,
            "dominator_node_ids": list(_sorted_node_ids(
                item.get("dominator_node_ids"), "dominator_node_ids",
            )),
            "post_dominator_node_ids": list(_sorted_node_ids(
                item.get("post_dominator_node_ids"),
                "post_dominator_node_ids",
            )),
            "immediate_dominator_node_id": _optional_node_id(
                item.get("immediate_dominator_node_id"),
                "immediate_dominator_node_id",
            ),
            "immediate_post_dominator_node_id": _optional_node_id(
                item.get("immediate_post_dominator_node_id"),
                "immediate_post_dominator_node_id",
            ),
        })
    if result != sorted(result, key=lambda item: item["stage_id"]):
        raise TaskGraphAnalysisError("Stage analysis must be sorted")
    stage_ids = [item["stage_id"] for item in result]
    if len(stage_ids) != len(set(stage_ids)):
        raise TaskGraphAnalysisError("Stage analysis IDs must be unique")
    return result


def _validate_virtual_nodes(value: Any, stage_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _VIRTUAL_KEYS:
        raise TaskGraphAnalysisError("virtual node fields are invalid")
    entry_stage_ids = list(_sorted_stage_ids(
        value.get("entry_stage_ids"), "entry_stage_ids",
    ))
    exit_stage_ids = list(_sorted_stage_ids(
        value.get("exit_stage_ids"), "exit_stage_ids",
    ))
    if not set(entry_stage_ids).issubset(stage_ids) or not set(
        exit_stage_ids
    ).issubset(stage_ids):
        raise TaskGraphAnalysisError("virtual node Stage IDs are unknown")
    entry_virtual = value.get("entry_virtual")
    exit_virtual = value.get("exit_virtual")
    if not isinstance(entry_virtual, bool) or not isinstance(exit_virtual, bool):
        raise TaskGraphAnalysisError("virtual node flags must be boolean")
    expected_entry = (
        _VIRTUAL_ENTRY if len(entry_stage_ids) > 1
        else f"stage:{entry_stage_ids[0]}"
    )
    expected_exit = (
        _VIRTUAL_EXIT if len(exit_stage_ids) > 1
        else f"stage:{exit_stage_ids[0]}"
    )
    if entry_virtual != (len(entry_stage_ids) > 1):
        raise TaskGraphAnalysisError("entry virtual flag is invalid")
    if exit_virtual != (len(exit_stage_ids) > 1):
        raise TaskGraphAnalysisError("exit virtual flag is invalid")
    if value.get("entry_node_id") != expected_entry:
        raise TaskGraphAnalysisError("entry analysis node is invalid")
    if value.get("exit_node_id") != expected_exit:
        raise TaskGraphAnalysisError("exit analysis node is invalid")
    return {
        "entry_node_id": expected_entry,
        "exit_node_id": expected_exit,
        "entry_stage_ids": entry_stage_ids,
        "exit_stage_ids": exit_stage_ids,
        "entry_virtual": entry_virtual,
        "exit_virtual": exit_virtual,
    }


def _validate_dominator_sets(
    stage_analysis: Sequence[Mapping[str, Any]],
    virtual: Mapping[str, Any],
) -> None:
    by_node = {
        f"stage:{item['stage_id']}": item for item in stage_analysis
    }
    allowed_dominators = set(by_node)
    allowed_post = set(by_node)
    if virtual["entry_virtual"]:
        allowed_dominators.add(_VIRTUAL_ENTRY)
    if virtual["exit_virtual"]:
        allowed_post.add(_VIRTUAL_EXIT)
    for node_id, item in by_node.items():
        dominators = set(item["dominator_node_ids"])
        post_dominators = set(item["post_dominator_node_ids"])
        if (
            node_id not in dominators
            or virtual["entry_node_id"] not in dominators
            or not dominators.issubset(allowed_dominators)
        ):
            raise TaskGraphAnalysisError("Stage dominator set is invalid")
        if (
            node_id not in post_dominators
            or virtual["exit_node_id"] not in post_dominators
            or not post_dominators.issubset(allowed_post)
        ):
            raise TaskGraphAnalysisError("Stage post-dominator set is invalid")
        expected_idom = _expected_immediate(
            node_id, dominators, by_node, virtual["entry_node_id"], True,
        )
        expected_ipdom = _expected_immediate(
            node_id, post_dominators, by_node, virtual["exit_node_id"], False,
        )
        if item["immediate_dominator_node_id"] != expected_idom:
            raise TaskGraphAnalysisError("immediate dominator is invalid")
        if item["immediate_post_dominator_node_id"] != expected_ipdom:
            raise TaskGraphAnalysisError("immediate post-dominator is invalid")


def _expected_immediate(
    node_id: str,
    related: set[str],
    by_node: Mapping[str, Mapping[str, Any]],
    root: str,
    dominator: bool,
) -> str:
    if node_id == root:
        return ""
    strict = related - {node_id}
    field_name = (
        "dominator_node_ids" if dominator else "post_dominator_node_ids"
    )

    def relation_size(candidate: str) -> int:
        if candidate in {_VIRTUAL_ENTRY, _VIRTUAL_EXIT}:
            return 1
        return len(by_node[candidate][field_name])

    return sorted(
        strict, key=lambda candidate: (-relation_size(candidate), candidate),
    )[0]


def _validate_tree_edges(value: Any, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TaskGraphAnalysisError(f"{field_name} tree edges must be a list")
    result = []
    for edge in value:
        if not isinstance(edge, Mapping) or set(edge) != _TREE_EDGE_KEYS:
            raise TaskGraphAnalysisError(f"{field_name} tree edge fields are invalid")
        result.append({
            "parent_node_id": _node_id(
                edge.get("parent_node_id"), "tree parent_node_id",
            ),
            "child_node_id": _node_id(
                edge.get("child_node_id"), "tree child_node_id",
            ),
        })
    if result != sorted(result, key=_tree_edge_key):
        raise TaskGraphAnalysisError(f"{field_name} tree edges must be sorted")
    if len({(item["parent_node_id"], item["child_node_id"]) for item in result}) != len(result):
        raise TaskGraphAnalysisError(f"{field_name} tree edges must be unique")
    return result


def _validate_stage_timings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphAnalysisError("Stage timings must be a list")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _TIMING_KEYS:
            raise TaskGraphAnalysisError("Stage timing fields are invalid")
        lower = _duration(
            item.get("duration_lower_ms"),
            "timing duration_lower_ms",
            allow_zero=True,
        )
        expected = _duration(
            item.get("duration_expected_ms"), "timing duration_expected_ms",
        )
        upper = _duration(
            item.get("duration_upper_ms"), "timing duration_upper_ms",
        )
        if not lower <= expected <= upper:
            raise TaskGraphAnalysisError("Stage timing interval is invalid")
        start = _nonnegative_int(
            item.get("earliest_start_expected_ms"),
            "earliest_start_expected_ms",
        )
        finish = _nonnegative_int(
            item.get("earliest_finish_expected_ms"),
            "earliest_finish_expected_ms",
        )
        remaining = _nonnegative_int(
            item.get("remaining_expected_ms"), "remaining_expected_ms",
        )
        if finish != start + expected or remaining < expected:
            raise TaskGraphAnalysisError("Stage timing recurrence is invalid")
        critical_member = item.get("critical_path_member")
        if not isinstance(critical_member, bool):
            raise TaskGraphAnalysisError("critical_path_member must be boolean")
        result.append({
            "stage_id": _identifier(item.get("stage_id"), "timing stage_id"),
            "contract_digest": _sha256(
                item.get("contract_digest"), "timing contract_digest",
            ),
            "duration_lower_ms": lower,
            "duration_expected_ms": expected,
            "duration_upper_ms": upper,
            "earliest_start_expected_ms": start,
            "earliest_finish_expected_ms": finish,
            "remaining_expected_ms": remaining,
            "critical_path_member": critical_member,
        })
    if result != sorted(result, key=lambda item: item["stage_id"]):
        raise TaskGraphAnalysisError("Stage timings must be sorted")
    if len({item["stage_id"] for item in result}) != len(result):
        raise TaskGraphAnalysisError("Stage timing IDs must be unique")
    return result


def _validate_critical_path(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CRITICAL_PATH_KEYS:
        raise TaskGraphAnalysisError("critical path fields are invalid")
    available = value.get("available")
    if not isinstance(available, bool):
        raise TaskGraphAnalysisError("critical path available must be boolean")
    stage_ids = list(_ordered_stage_ids(
        value.get("stage_ids"), "critical path stage_ids", allow_empty=True,
    ))
    lower = _nonnegative_int(
        value.get("duration_lower_ms"), "critical duration_lower_ms",
    )
    expected = _nonnegative_int(
        value.get("duration_expected_ms"), "critical duration_expected_ms",
    )
    upper = _nonnegative_int(
        value.get("duration_upper_ms"), "critical duration_upper_ms",
    )
    if not lower <= expected <= upper:
        raise TaskGraphAnalysisError("critical path interval is invalid")
    base = {
        "available": available,
        "stage_ids": stage_ids,
        "duration_lower_ms": lower,
        "duration_expected_ms": expected,
        "duration_upper_ms": upper,
    }
    if _sha256(value.get("path_digest"), "path_digest") != _digest(base):
        raise TaskGraphAnalysisError("critical path digest mismatch")
    return {**base, "path_digest": value["path_digest"]}


def _validate_available_critical_path(
    critical: Mapping[str, Any],
    timings: Sequence[Mapping[str, Any]],
    virtual: Mapping[str, Any],
) -> None:
    if not critical["available"] or not critical["stage_ids"]:
        raise TaskGraphAnalysisError("evaluated critical path must be available")
    by_stage = {item["stage_id"]: item for item in timings}
    path = critical["stage_ids"]
    if any(stage_id not in by_stage for stage_id in path):
        raise TaskGraphAnalysisError("critical path contains unknown Stage")
    if path[0] not in virtual["entry_stage_ids"]:
        raise TaskGraphAnalysisError("critical path must start at an entry Stage")
    if path[-1] not in virtual["exit_stage_ids"]:
        raise TaskGraphAnalysisError("critical path must end at an exit Stage")
    expected_values = {
        "duration_lower_ms": sum(
            by_stage[stage_id]["duration_lower_ms"] for stage_id in path
        ),
        "duration_expected_ms": sum(
            by_stage[stage_id]["duration_expected_ms"] for stage_id in path
        ),
        "duration_upper_ms": sum(
            by_stage[stage_id]["duration_upper_ms"] for stage_id in path
        ),
    }
    if any(critical[key] != value for key, value in expected_values.items()):
        raise TaskGraphAnalysisError("critical path duration does not match timings")
    if critical["duration_expected_ms"] != max(
        by_stage[stage_id]["earliest_finish_expected_ms"]
        for stage_id in virtual["exit_stage_ids"]
    ):
        raise TaskGraphAnalysisError("critical path is not the longest candidate")
    critical_set = set(path)
    if {
        item["stage_id"]
        for item in timings
        if item["critical_path_member"]
    } != critical_set:
        raise TaskGraphAnalysisError("critical path membership is inconsistent")


def _tree_edge_key(edge: Mapping[str, str]) -> tuple[str, str]:
    return edge["parent_node_id"], edge["child_node_id"]


def _sorted_node_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskGraphAnalysisError(f"{field_name} must be a non-empty list")
    result = tuple(_node_id(item, field_name) for item in value)
    if list(result) != sorted(set(result)):
        raise TaskGraphAnalysisError(f"{field_name} must be sorted and unique")
    return result


def _sorted_stage_ids(value: Any, field_name: str) -> tuple[str, ...]:
    result = _ordered_stage_ids(value, field_name)
    if list(result) != sorted(result):
        raise TaskGraphAnalysisError(f"{field_name} must be sorted")
    return result


def _ordered_stage_ids(
    value: Any, field_name: str, *, allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise TaskGraphAnalysisError(f"{field_name} must be a list")
    result = tuple(_identifier(item, field_name) for item in value)
    if len(set(result)) != len(result):
        raise TaskGraphAnalysisError(f"{field_name} must be unique")
    return result


def _optional_node_id(value: Any, field_name: str) -> str:
    if value == "":
        return ""
    return _node_id(value, field_name)


def _node_id(value: Any, field_name: str) -> str:
    value = _identifier(value, field_name)
    if not value.startswith(("stage:", "virtual:")):
        raise TaskGraphAnalysisError(f"{field_name} has an invalid namespace")
    return value


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphAnalysisError("analysis keys must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphAnalysisError(
                    f"analysis contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _duration(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    return _bounded_int(value, minimum, _MAX_DURATION_MS, field_name)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskGraphAnalysisError(f"{field_name} must be non-negative")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise TaskGraphAnalysisError(f"{field_name} is outside the safe range")
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphAnalysisError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphAnalysisError(f"{field_name} must be a SHA-256 digest")
    return value


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise TaskGraphAnalysisError(f"{field_name} is unsupported")
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
    "DURATION_ESTIMATE_SCHEMA_VERSION",
    "STRUCTURE_ANALYSIS_SCHEMA_VERSION",
    "STRUCTURE_ANALYZER_VERSION",
    "TaskGraphAnalysisError",
    "analyze_task_graph_structure",
    "build_stage_duration_estimate",
    "validate_stage_duration_estimate",
    "validate_structure_analysis",
]
