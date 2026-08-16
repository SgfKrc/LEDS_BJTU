"""Shadow-only linear Stage fusion candidates for TaskGraph.

G3.3 fuses only continuous, pure transform chains into an independent
``optimized_dag`` candidate.  The logical DAG remains the selected execution
graph.  Explicit per-Stage boundary mappings preserve failure, cancellation,
logging, and accounting attribution for future runtime work.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from task_graph_optimization import require_graph_kind, validate_projection
from task_graph_sharing import (
    analyze_stage_sharing,
    validate_share_analysis,
    validate_stage_semantics_contract,
)


FUSION_CONTRACT_SCHEMA_VERSION = "qlh.task_graph_fusion_contract.v1"
FUSION_CANDIDATE_SCHEMA_VERSION = "qlh.task_graph_fusion_candidate.v1"
FUSION_CANDIDATE_VERSION = "task-stage-fusion-v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_KEYS = frozenset({
    "schema_version",
    "stage_id",
    "fusion_policy",
    "barrier_class",
    "failure_mapping",
    "cancellation_mapping",
    "logging_mapping",
    "accounting_mapping",
    "contract_digest",
})
_REPORT_KEYS = frozenset({
    "schema_version",
    "fuser_version",
    "mode",
    "status",
    "selected_graph_kind",
    "logical_graph_digest",
    "optimized_graph_digest",
    "analysis_digest",
    "fusion_contracts_digest",
    "fallback",
    "fusion_groups",
    "boundary_map",
    "rejections",
    "summary",
    "candidate_graph",
    "digest",
})
_GROUP_KEYS = frozenset({
    "fused_stage_id",
    "source_stage_ids",
    "chain_digest",
})
_BOUNDARY_KEYS = frozenset({
    "fused_stage_id",
    "source_stage_id",
    "ordinal",
    "failure_boundary_id",
    "cancellation_boundary_id",
    "logging_boundary_id",
    "accounting_boundary_id",
})
_REJECTION_KEYS = frozenset({
    "source_stage_id",
    "target_stage_id",
    "reason_code",
})
_SUMMARY_KEYS = frozenset({
    "logical_stage_count",
    "optimized_stage_count",
    "logical_edge_count",
    "optimized_edge_count",
    "fusion_group_count",
    "fused_source_stage_count",
    "preserved_boundary_count",
    "rejected_boundary_count",
})
_FUSION_POLICIES = frozenset({"allow", "deny"})
_BARRIER_CLASSES = frozenset({"none", "checkpoint", "commit", "billing"})
_MAPPING_POLICIES = frozenset({"preserve_stage", "unavailable"})
_REJECTION_REASONS = frozenset({
    "stage_ineligible",
    "fusion_not_allowed",
    "checkpoint_boundary",
    "commit_boundary",
    "billing_boundary",
    "failure_mapping_unavailable",
    "cancellation_mapping_unavailable",
    "logging_mapping_unavailable",
    "accounting_mapping_unavailable",
    "fan_out_boundary",
    "fan_in_boundary",
    "edge_semantics_boundary",
    "binding_boundary",
    "provider_boundary",
    "model_boundary",
    "data_scope_boundary",
    "schema_boundary",
    "fused_stage_id_collision",
})
_FORBIDDEN_KEYS = frozenset({
    "body",
    "config",
    "content",
    "error",
    "output",
    "path",
    "prompt",
    "raw",
    "root_input",
    "secret",
    "token",
    "url",
})


class TaskGraphFusionError(ValueError):
    """Raised when a Stage fusion contract or candidate is malformed."""


def build_stage_fusion_contract(
    stage_id: str,
    *,
    fusion_policy: str = "allow",
    barrier_class: str = "none",
    failure_mapping: str = "preserve_stage",
    cancellation_mapping: str = "preserve_stage",
    logging_mapping: str = "preserve_stage",
    accounting_mapping: str = "preserve_stage",
) -> dict[str, Any]:
    """Build an explicit, digest-only Stage fusion contract."""

    contract = {
        "schema_version": FUSION_CONTRACT_SCHEMA_VERSION,
        "stage_id": _identifier(stage_id, "stage_id"),
        "fusion_policy": _enum(
            fusion_policy, _FUSION_POLICIES, "fusion_policy",
        ),
        "barrier_class": _enum(
            barrier_class, _BARRIER_CLASSES, "barrier_class",
        ),
        "failure_mapping": _enum(
            failure_mapping, _MAPPING_POLICIES, "failure_mapping",
        ),
        "cancellation_mapping": _enum(
            cancellation_mapping, _MAPPING_POLICIES, "cancellation_mapping",
        ),
        "logging_mapping": _enum(
            logging_mapping, _MAPPING_POLICIES, "logging_mapping",
        ),
        "accounting_mapping": _enum(
            accounting_mapping, _MAPPING_POLICIES, "accounting_mapping",
        ),
    }
    contract["contract_digest"] = _digest(contract)
    return validate_stage_fusion_contract(contract)


def validate_stage_fusion_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one versioned Stage fusion contract."""

    if not isinstance(contract, Mapping) or set(contract) != _CONTRACT_KEYS:
        raise TaskGraphFusionError("stage fusion contract fields are invalid")
    _assert_no_forbidden_fields(contract)
    if contract.get("schema_version") != FUSION_CONTRACT_SCHEMA_VERSION:
        raise TaskGraphFusionError("unsupported stage fusion contract schema")
    _identifier(contract.get("stage_id"), "stage_id")
    _enum(contract.get("fusion_policy"), _FUSION_POLICIES, "fusion_policy")
    _enum(contract.get("barrier_class"), _BARRIER_CLASSES, "barrier_class")
    for field_name in (
        "failure_mapping",
        "cancellation_mapping",
        "logging_mapping",
        "accounting_mapping",
    ):
        _enum(contract.get(field_name), _MAPPING_POLICIES, field_name)
    supplied_digest = _sha256(contract.get("contract_digest"), "contract_digest")
    unsigned = {
        key: value for key, value in contract.items() if key != "contract_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphFusionError("stage fusion contract digest mismatch")
    return _detached(contract)


def fuse_transform_chains(
    projection: Mapping[str, Any],
    semantics_contracts: Sequence[Mapping[str, Any]],
    fusion_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic linear-fusion candidate without changing execution."""

    logical = require_graph_kind(projection, "logical_dag")
    semantics_by_stage = _semantics_contracts_by_stage(semantics_contracts)
    fusion_by_stage = _fusion_contracts_by_stage(fusion_contracts)
    stage_ids = {
        node["stage_id"]
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    if set(semantics_by_stage) != stage_ids:
        raise TaskGraphFusionError(
            "semantics contracts must match projected Stage IDs exactly",
        )
    if set(fusion_by_stage) != stage_ids:
        raise TaskGraphFusionError(
            "fusion contracts must match projected Stage IDs exactly",
        )

    analysis = validate_share_analysis(
        analyze_stage_sharing(logical, semantics_contracts),
    )
    eligibility = {
        result["stage_id"]: result["eligible"]
        for result in analysis["stage_fingerprints"]
    }
    accepted_links, rejections = _classify_transform_links(
        logical,
        semantics_by_stage,
        fusion_by_stage,
        eligibility,
    )
    groups, collision_rejections = _build_fusion_groups(
        accepted_links,
        logical,
        semantics_by_stage,
        fusion_by_stage,
    )
    rejections.extend(collision_rejections)
    rejections.sort(key=lambda item: (
        item["source_stage_id"],
        item["target_stage_id"],
        item["reason_code"],
    ))
    groups.sort(key=lambda group: group["fused_stage_id"])

    status = "evaluated" if groups else "no_op"
    fallback_reason = (
        "fusion_candidate_ready" if groups else "no_fusible_chain"
    )
    candidate = _rebuild_candidate(logical, groups, fallback_reason)
    boundary_map = _build_boundary_map(groups)
    fusion_contracts_digest = _digest({
        "schema_version": 1,
        "contract_digests": [
            fusion_by_stage[stage_id]["contract_digest"]
            for stage_id in sorted(fusion_by_stage)
        ],
    })
    report = {
        "schema_version": FUSION_CANDIDATE_SCHEMA_VERSION,
        "fuser_version": FUSION_CANDIDATE_VERSION,
        "mode": "shadow",
        "status": status,
        "selected_graph_kind": "logical_dag",
        "logical_graph_digest": logical["digest"],
        "optimized_graph_digest": candidate["digest"],
        "analysis_digest": analysis["digest"],
        "fusion_contracts_digest": fusion_contracts_digest,
        "fallback": {
            "used": status != "evaluated",
            "reason_code": fallback_reason,
        },
        "fusion_groups": groups,
        "boundary_map": boundary_map,
        "rejections": rejections,
        "summary": {
            "logical_stage_count": logical["summary"]["stage_count"],
            "optimized_stage_count": candidate["summary"]["stage_count"],
            "logical_edge_count": logical["summary"]["edge_count"],
            "optimized_edge_count": candidate["summary"]["edge_count"],
            "fusion_group_count": len(groups),
            "fused_source_stage_count": sum(
                len(group["source_stage_ids"]) for group in groups
            ),
            "preserved_boundary_count": len(boundary_map),
            "rejected_boundary_count": len(rejections),
        },
        "candidate_graph": candidate,
    }
    report["digest"] = _digest(report)
    return validate_fusion_candidate(report)


def validate_fusion_candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a fusion report and its detached optimized candidate graph."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphFusionError("fusion candidate fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != FUSION_CANDIDATE_SCHEMA_VERSION:
        raise TaskGraphFusionError("unsupported fusion candidate schema")
    if report.get("fuser_version") != FUSION_CANDIDATE_VERSION:
        raise TaskGraphFusionError("unsupported fusion candidate version")
    if report.get("mode") != "shadow":
        raise TaskGraphFusionError("fusion candidate must remain shadow-only")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphFusionError("fusion candidate cannot select optimized graph")
    for field_name in (
        "logical_graph_digest",
        "optimized_graph_digest",
        "analysis_digest",
        "fusion_contracts_digest",
    ):
        _sha256(report.get(field_name), field_name)

    status = report.get("status")
    if status not in {"evaluated", "no_op"}:
        raise TaskGraphFusionError("fusion candidate status is invalid")
    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphFusionError("fusion candidate fallback is invalid")
    if not isinstance(fallback["used"], bool):
        raise TaskGraphFusionError("fusion candidate fallback.used is invalid")
    expected_reason = {
        "evaluated": "fusion_candidate_ready",
        "no_op": "no_fusible_chain",
    }[status]
    if fallback != {
        "used": status != "evaluated",
        "reason_code": expected_reason,
    }:
        raise TaskGraphFusionError("fusion candidate fallback status mismatch")

    checked_candidate = validate_projection(report.get("candidate_graph"))
    if checked_candidate["graph_kind"] != "optimized_dag":
        raise TaskGraphFusionError("candidate graph must be optimized_dag")
    if checked_candidate["digest"] != report["optimized_graph_digest"]:
        raise TaskGraphFusionError("candidate graph digest mismatch")

    groups = _validate_groups(report.get("fusion_groups"))
    boundaries = _validate_boundary_map(report.get("boundary_map"))
    expected_boundaries = _build_boundary_map(groups)
    if boundaries != expected_boundaries:
        raise TaskGraphFusionError("boundary map does not match fusion groups")
    rejections = _validate_rejections(report.get("rejections"))
    if status == "evaluated" and not groups:
        raise TaskGraphFusionError("evaluated fusion candidate requires a group")
    if status == "no_op" and groups:
        raise TaskGraphFusionError("no-op fusion candidate cannot contain groups")

    candidate_nodes = {
        node["stage_id"]: node
        for node in checked_candidate["nodes"]
        if node.get("node_kind") == "stage"
    }
    fused_ids = {group["fused_stage_id"] for group in groups}
    candidate_fused_ids = {
        stage_id for stage_id in candidate_nodes if stage_id.startswith("fused:")
    }
    if candidate_fused_ids != fused_ids:
        raise TaskGraphFusionError("candidate fused Stages do not match groups")
    source_ids = {
        stage_id
        for group in groups
        for stage_id in group["source_stage_ids"]
    }
    if source_ids.intersection(candidate_nodes):
        raise TaskGraphFusionError("source Stage remains in candidate graph")
    for fused_id in fused_ids:
        if candidate_nodes[fused_id].get("stage_type") != "transform_fused":
            raise TaskGraphFusionError("fused Stage type is invalid")
    _assert_sources_absent_from_candidate(checked_candidate, source_ids)

    events = [
        event for event in checked_candidate.get("trace", [])
        if event.get("rule") == "fuse_transform_chain"
    ]
    if status == "evaluated":
        expected_events = [_fusion_trace_event(group) for group in groups]
        if events != expected_events:
            raise TaskGraphFusionError("candidate fusion trace does not match groups")
    elif events != [{
        "rule": "fuse_transform_chain",
        "reason_code": "no_fusible_chain",
        "affected_node_ids": [],
        "accepted": False,
    }]:
        raise TaskGraphFusionError("candidate no-op fusion trace is invalid")

    summary = report.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphFusionError("fusion candidate summary is invalid")
    expected_summary = {
        "logical_stage_count": summary.get("logical_stage_count"),
        "optimized_stage_count": checked_candidate["summary"]["stage_count"],
        "logical_edge_count": summary.get("logical_edge_count"),
        "optimized_edge_count": checked_candidate["summary"]["edge_count"],
        "fusion_group_count": len(groups),
        "fused_source_stage_count": len(source_ids),
        "preserved_boundary_count": len(boundaries),
        "rejected_boundary_count": len(rejections),
    }
    for key, expected in expected_summary.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphFusionError(f"summary {key} must be non-negative")
        if value != expected:
            raise TaskGraphFusionError(f"summary {key} does not match candidate")
    if summary["preserved_boundary_count"] != summary[
        "fused_source_stage_count"
    ]:
        raise TaskGraphFusionError("each fused source requires one boundary map")
    if status == "evaluated" and summary["optimized_stage_count"] != (
        summary["logical_stage_count"]
        - summary["fused_source_stage_count"]
        + summary["fusion_group_count"]
    ):
        raise TaskGraphFusionError("fused Stage count does not match groups")
    if status == "evaluated" and summary["optimized_edge_count"] > summary[
        "logical_edge_count"
    ]:
        raise TaskGraphFusionError("fusion candidate cannot add dependency edges")
    if status == "no_op" and (
        summary["optimized_stage_count"] != summary["logical_stage_count"]
        or summary["optimized_edge_count"] != summary["logical_edge_count"]
    ):
        raise TaskGraphFusionError("no-op candidate changed graph shape")

    supplied_digest = _sha256(report.get("digest"), "fusion candidate digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphFusionError("fusion candidate digest mismatch")
    return _detached(report)


def _classify_transform_links(
    logical: Mapping[str, Any],
    semantics: Mapping[str, Mapping[str, Any]],
    fusion: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, bool],
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    nodes = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    incoming = {stage_id: 0 for stage_id in nodes}
    outgoing = {stage_id: 0 for stage_id in nodes}
    edges: list[tuple[str, str, Mapping[str, Any]]] = []
    for edge in logical["edges"]:
        source = _stage_from_node_id(edge["source_node_id"])
        target = _stage_from_node_id(edge["target_node_id"])
        outgoing[source] += 1
        incoming[target] += 1
        edges.append((source, target, edge))

    accepted: list[tuple[str, str]] = []
    rejections: list[dict[str, str]] = []
    for source, target, edge in sorted(edges, key=lambda item: (item[0], item[1])):
        if (
            nodes[source].get("stage_type") != "transform"
            or nodes[target].get("stage_type") != "transform"
        ):
            continue
        reason = _link_rejection_reason(
            source,
            target,
            edge,
            nodes,
            semantics,
            fusion,
            eligibility,
            incoming,
            outgoing,
        )
        if reason is None:
            accepted.append((source, target))
        else:
            rejections.append({
                "source_stage_id": source,
                "target_stage_id": target,
                "reason_code": reason,
            })
    return accepted, rejections


def _link_rejection_reason(
    source: str,
    target: str,
    edge: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    semantics: Mapping[str, Mapping[str, Any]],
    fusion: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, bool],
    incoming: Mapping[str, int],
    outgoing: Mapping[str, int],
) -> str | None:
    if not eligibility[source] or not eligibility[target]:
        return "stage_ineligible"
    if (
        fusion[source]["fusion_policy"] != "allow"
        or fusion[target]["fusion_policy"] != "allow"
    ):
        return "fusion_not_allowed"
    for stage_id in (source, target):
        barrier = fusion[stage_id]["barrier_class"]
        if barrier != "none":
            return f"{barrier}_boundary"
    for field_name, reason in (
        ("failure_mapping", "failure_mapping_unavailable"),
        ("cancellation_mapping", "cancellation_mapping_unavailable"),
        ("logging_mapping", "logging_mapping_unavailable"),
        ("accounting_mapping", "accounting_mapping_unavailable"),
    ):
        if any(
            fusion[stage_id][field_name] != "preserve_stage"
            for stage_id in (source, target)
        ):
            return reason
    if outgoing[source] != 1:
        return "fan_out_boundary"
    if incoming[target] != 1:
        return "fan_in_boundary"
    edge_contract = edge.get("semantic_contract")
    if not isinstance(edge_contract, Mapping) or not edge_contract.get(
        "pure_dependency", False,
    ):
        if isinstance(edge_contract, Mapping) and edge_contract.get(
            "binding_targets",
        ):
            return "binding_boundary"
        return "edge_semantics_boundary"
    if nodes[target].get("input_bindings"):
        return "binding_boundary"
    if nodes[source].get("provider_constraints") != nodes[target].get(
        "provider_constraints"
    ):
        return "provider_boundary"
    if nodes[source].get("model_identity") != nodes[target].get("model_identity"):
        return "model_boundary"
    if semantics[source]["data_scope"] != semantics[target]["data_scope"]:
        return "data_scope_boundary"
    if semantics[source]["output_schema_version"] != semantics[target][
        "input_schema_version"
    ]:
        return "schema_boundary"
    return None


def _build_fusion_groups(
    accepted_links: Sequence[tuple[str, str]],
    logical: Mapping[str, Any],
    semantics: Mapping[str, Mapping[str, Any]],
    fusion: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    successor = {source: target for source, target in accepted_links}
    predecessor = {target: source for source, target in accepted_links}
    existing_stage_ids = {
        node["stage_id"]
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    groups: list[dict[str, Any]] = []
    collision_rejections: list[dict[str, str]] = []
    for start in sorted(set(successor) - set(predecessor)):
        chain = [start]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chain_digest = _chain_digest(chain, semantics, fusion)
        fused_stage_id = f"fused:{chain_digest}"
        if fused_stage_id in existing_stage_ids:
            collision_rejections.extend({
                "source_stage_id": source,
                "target_stage_id": target,
                "reason_code": "fused_stage_id_collision",
            } for source, target in zip(chain, chain[1:]))
            continue
        groups.append({
            "fused_stage_id": fused_stage_id,
            "source_stage_ids": chain,
            "chain_digest": chain_digest,
        })
    return groups, collision_rejections


def _chain_digest(
    chain: Sequence[str],
    semantics: Mapping[str, Mapping[str, Any]],
    fusion: Mapping[str, Mapping[str, Any]],
) -> str:
    return _digest({
        "schema_version": 1,
        "source_stage_ids": list(chain),
        "semantics_contract_digests": [
            semantics[stage_id]["contract_digest"] for stage_id in chain
        ],
        "fusion_contract_digests": [
            fusion[stage_id]["contract_digest"] for stage_id in chain
        ],
    })


def _build_boundary_map(
    groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for group in groups:
        chain_digest = group["chain_digest"]
        for ordinal, source_stage_id in enumerate(group["source_stage_ids"]):
            boundaries.append({
                "fused_stage_id": group["fused_stage_id"],
                "source_stage_id": source_stage_id,
                "ordinal": ordinal,
                "failure_boundary_id": f"failure:{chain_digest}:{ordinal}",
                "cancellation_boundary_id": (
                    f"cancellation:{chain_digest}:{ordinal}"
                ),
                "logging_boundary_id": f"logging:{chain_digest}:{ordinal}",
                "accounting_boundary_id": f"accounting:{chain_digest}:{ordinal}",
            })
    return boundaries


def _rebuild_candidate(
    logical: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    no_fusion_reason: str,
) -> dict[str, Any]:
    mapping = {
        stage_id: group["fused_stage_id"]
        for group in groups
        for stage_id in group["source_stage_ids"]
    }
    nodes_by_stage = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    group_by_fused = {group["fused_stage_id"]: group for group in groups}
    candidate_nodes: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for stage_id in sorted(nodes_by_stage):
        mapped_stage_id = mapping.get(stage_id, stage_id)
        if mapped_stage_id in emitted:
            continue
        group = group_by_fused.get(mapped_stage_id)
        representative = group["source_stage_ids"][0] if group else stage_id
        node = _detached(nodes_by_stage[representative])
        node["stage_id"] = mapped_stage_id
        node["node_id"] = f"stage:{mapped_stage_id}"
        if group:
            node["stage_type"] = "transform_fused"
        node["depends_on"] = sorted({
            mapping.get(dependency, dependency)
            for dependency in node.get("depends_on", [])
            if mapping.get(dependency, dependency) != mapped_stage_id
        })
        node["input_bindings"] = sorted([
            {
                **binding,
                "dependency_stage_id": mapping.get(
                    binding["dependency_stage_id"],
                    binding["dependency_stage_id"],
                ),
            }
            for binding in node.get("input_bindings", [])
        ], key=lambda binding: binding["target_key"])
        candidate_nodes.append(node)
        emitted.add(mapped_stage_id)

    edge_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in logical["edges"]:
        source = mapping.get(
            _stage_from_node_id(edge["source_node_id"]),
            _stage_from_node_id(edge["source_node_id"]),
        )
        target = mapping.get(
            _stage_from_node_id(edge["target_node_id"]),
            _stage_from_node_id(edge["target_node_id"]),
        )
        if source == target:
            continue
        relation = edge["relation"]
        key = (relation, source, target)
        rebuilt = _detached(edge)
        rebuilt["source_node_id"] = f"stage:{source}"
        rebuilt["target_node_id"] = f"stage:{target}"
        rebuilt["edge_id"] = f"edge:{relation}:{source}:{target}"
        previous = edge_by_key.get(key)
        if previous is not None and previous != rebuilt:
            raise TaskGraphFusionError(
                "fusion candidate edge semantics collide",
            )
        edge_by_key[key] = rebuilt

    candidate = _detached(logical)
    candidate["graph_kind"] = "optimized_dag"
    candidate["nodes"] = sorted(candidate_nodes, key=lambda node: node["node_id"])
    candidate["edges"] = sorted(
        edge_by_key.values(), key=lambda edge: edge["edge_id"],
    )
    candidate["summary"] = _detached(logical["summary"])
    final_stage_id = candidate["summary"]["final_stage_id"]
    candidate["summary"]["final_stage_id"] = mapping.get(
        final_stage_id, final_stage_id,
    )
    candidate["summary"]["stage_count"] = len(candidate_nodes)
    candidate["summary"]["node_count"] = len(candidate_nodes)
    candidate["summary"]["edge_count"] = len(edge_by_key)
    candidate["summary"]["provider_count"] = len({
        provider
        for node in candidate_nodes
        for provider in (
            node.get("provider_constraints", {}).get("requested_provider", ""),
            *node.get("provider_constraints", {}).get("fallback_providers", []),
        )
        if provider
    })
    fusion_events = [_fusion_trace_event(group) for group in groups]
    if not fusion_events:
        fusion_events = [{
            "rule": "fuse_transform_chain",
            "reason_code": no_fusion_reason,
            "affected_node_ids": [],
            "accepted": False,
        }]
    candidate["trace"] = [
        event for event in _detached(logical.get("trace", []))
        if event.get("rule") != "fuse_transform_chain"
    ] + fusion_events
    candidate["digest"] = _digest({
        key: value for key, value in candidate.items() if key != "digest"
    })
    return validate_projection(candidate)


def _fusion_trace_event(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rule": "fuse_transform_chain",
        "reason_code": "linear_transform_candidate",
        "affected_node_ids": sorted([
            f"stage:{group['fused_stage_id']}",
            *(
                f"stage:{stage_id}"
                for stage_id in group["source_stage_ids"]
            ),
        ]),
        "accepted": True,
    }


def _semantics_contracts_by_stage(
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
        raise TaskGraphFusionError("semantics contracts must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        try:
            checked = validate_stage_semantics_contract(contract)
        except ValueError as exc:
            raise TaskGraphFusionError(str(exc)) from exc
        stage_id = checked["stage_id"]
        if stage_id in result:
            raise TaskGraphFusionError(
                "semantics contract Stage IDs must be unique",
            )
        result[stage_id] = checked
    return result


def _fusion_contracts_by_stage(
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
        raise TaskGraphFusionError("fusion contracts must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        checked = validate_stage_fusion_contract(contract)
        stage_id = checked["stage_id"]
        if stage_id in result:
            raise TaskGraphFusionError("fusion contract Stage IDs must be unique")
        result[stage_id] = checked
    return result


def _validate_groups(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphFusionError("fusion groups must be a list")
    result: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_fused: set[str] = set()
    for group in value:
        if not isinstance(group, Mapping) or set(group) != _GROUP_KEYS:
            raise TaskGraphFusionError("fusion group fields are invalid")
        fused_stage_id = _identifier(group.get("fused_stage_id"), "fused_stage_id")
        chain_digest = _sha256(group.get("chain_digest"), "chain_digest")
        if fused_stage_id != f"fused:{chain_digest}":
            raise TaskGraphFusionError("fused Stage ID does not match chain digest")
        source_stage_ids = _ordered_identifiers(
            group.get("source_stage_ids"), "source_stage_ids",
        )
        if len(source_stage_ids) < 2:
            raise TaskGraphFusionError("fusion group requires two source Stages")
        if fused_stage_id in seen_fused or seen_sources.intersection(
            source_stage_ids
        ):
            raise TaskGraphFusionError("fusion groups overlap")
        seen_fused.add(fused_stage_id)
        seen_sources.update(source_stage_ids)
        result.append({
            "fused_stage_id": fused_stage_id,
            "source_stage_ids": list(source_stage_ids),
            "chain_digest": chain_digest,
        })
    if result != sorted(result, key=lambda group: group["fused_stage_id"]):
        raise TaskGraphFusionError("fusion groups must be sorted")
    return result


def _validate_boundary_map(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphFusionError("boundary map must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _BOUNDARY_KEYS:
            raise TaskGraphFusionError("boundary map fields are invalid")
        ordinal = item.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise TaskGraphFusionError("boundary ordinal must be non-negative")
        result.append({
            "fused_stage_id": _identifier(
                item.get("fused_stage_id"), "boundary fused_stage_id",
            ),
            "source_stage_id": _identifier(
                item.get("source_stage_id"), "boundary source_stage_id",
            ),
            "ordinal": ordinal,
            "failure_boundary_id": _identifier(
                item.get("failure_boundary_id"), "failure_boundary_id",
            ),
            "cancellation_boundary_id": _identifier(
                item.get("cancellation_boundary_id"),
                "cancellation_boundary_id",
            ),
            "logging_boundary_id": _identifier(
                item.get("logging_boundary_id"), "logging_boundary_id",
            ),
            "accounting_boundary_id": _identifier(
                item.get("accounting_boundary_id"), "accounting_boundary_id",
            ),
        })
    return result


def _validate_rejections(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TaskGraphFusionError("fusion rejections must be a list")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _REJECTION_KEYS:
            raise TaskGraphFusionError("fusion rejection fields are invalid")
        result.append({
            "source_stage_id": _identifier(
                item.get("source_stage_id"), "rejection source_stage_id",
            ),
            "target_stage_id": _identifier(
                item.get("target_stage_id"), "rejection target_stage_id",
            ),
            "reason_code": _enum(
                item.get("reason_code"),
                _REJECTION_REASONS,
                "rejection reason_code",
            ),
        })
    expected = sorted(result, key=lambda item: (
        item["source_stage_id"],
        item["target_stage_id"],
        item["reason_code"],
    ))
    if result != expected:
        raise TaskGraphFusionError("fusion rejections must be sorted")
    return result


def _assert_sources_absent_from_candidate(
    candidate: Mapping[str, Any], source_ids: set[str],
) -> None:
    for node in candidate["nodes"]:
        if source_ids.intersection(node.get("depends_on", [])):
            raise TaskGraphFusionError("candidate dependency retains source Stage")
        for binding in node.get("input_bindings", []):
            if binding.get("dependency_stage_id") in source_ids:
                raise TaskGraphFusionError("candidate binding retains source Stage")
    source_node_ids = {f"stage:{stage_id}" for stage_id in source_ids}
    for edge in candidate["edges"]:
        if (
            edge.get("source_node_id") in source_node_ids
            or edge.get("target_node_id") in source_node_ids
        ):
            raise TaskGraphFusionError("candidate edge retains source Stage")


def _stage_from_node_id(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id.startswith("stage:"):
        raise TaskGraphFusionError("fusion requires Stage dependency edges")
    return _identifier(node_id.split(":", 1)[1], "edge stage_id")


def _ordered_identifiers(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskGraphFusionError(f"{field_name} must be a non-empty list")
    identifiers = tuple(_identifier(item, field_name) for item in value)
    if len(set(identifiers)) != len(identifiers):
        raise TaskGraphFusionError(f"{field_name} must be unique")
    return identifiers


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphFusionError("fusion candidate keys must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphFusionError(
                    f"fusion candidate contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphFusionError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphFusionError(f"{field_name} must be a SHA-256 digest")
    return value


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise TaskGraphFusionError(f"{field_name} is unsupported")
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
    "FUSION_CANDIDATE_SCHEMA_VERSION",
    "FUSION_CANDIDATE_VERSION",
    "FUSION_CONTRACT_SCHEMA_VERSION",
    "TaskGraphFusionError",
    "build_stage_fusion_contract",
    "fuse_transform_chains",
    "validate_fusion_candidate",
    "validate_stage_fusion_contract",
]
