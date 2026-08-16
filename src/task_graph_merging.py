"""Shadow-only common-subtask merge candidates for TaskGraph.

G3.2 consumes the G3.1 semantic sharing analysis and builds an independent
``optimized_dag`` candidate.  The logical DAG remains the selected execution
graph.  Any merge that could change a downstream join or edge contract is
rejected and returned as a no-op candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from task_graph_optimization import require_graph_kind, validate_projection
from task_graph_sharing import analyze_stage_sharing, validate_share_analysis


MERGE_CANDIDATE_SCHEMA_VERSION = "qlh.task_graph_merge_candidate.v1"
MERGE_CANDIDATE_VERSION = "task-stage-merge-v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_KEYS = frozenset({
    "schema_version",
    "merger_version",
    "mode",
    "status",
    "selected_graph_kind",
    "logical_graph_digest",
    "optimized_graph_digest",
    "analysis_digest",
    "fallback",
    "merge_groups",
    "provenance",
    "rejections",
    "summary",
    "candidate_graph",
    "digest",
})
_GROUP_KEYS = frozenset({
    "merged_stage_id",
    "source_stage_ids",
    "fingerprint_sha256",
})
_PROVENANCE_KEYS = frozenset({"merged_stage_id", "source_stage_ids"})
_REJECTION_KEYS = frozenset({
    "source_stage_ids",
    "fingerprint_sha256",
    "reason_code",
})
_SUMMARY_KEYS = frozenset({
    "logical_stage_count",
    "optimized_stage_count",
    "logical_edge_count",
    "optimized_edge_count",
    "merge_group_count",
    "merged_source_stage_count",
    "rejected_group_count",
})
_REJECTION_REASONS = frozenset({
    "join_arity_would_change",
    "semantic_contract_collision",
    "self_dependency_after_merge",
    "shared_stage_id_collision",
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


class TaskGraphMergeError(ValueError):
    """Raised when a shadow merge candidate is malformed."""


class _MergeSafetyError(RuntimeError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


def merge_shareable_stages(
    projection: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic merge candidate without changing execution."""

    logical = require_graph_kind(projection, "logical_dag")
    analysis = validate_share_analysis(
        analyze_stage_sharing(logical, contracts),
    )
    stage_results = {
        result["stage_id"]: result
        for result in analysis["stage_fingerprints"]
    }
    groups = _share_groups(stage_results)
    merge_records: list[dict[str, Any]] = []
    status = "evaluated"
    fallback_reason = "merge_candidate_ready"
    rejections: list[dict[str, Any]] = []
    if not groups:
        status = "no_op"
        fallback_reason = "no_shareable_pair"
        candidate = _rebuild_candidate(logical, {}, ())
    else:
        try:
            mapping, merge_records = _build_merge_mapping(groups, logical)
            candidate = _rebuild_candidate(logical, mapping, merge_records)
        except _MergeSafetyError as exc:
            status = "fallback"
            fallback_reason = exc.reason_code
            rejections = [
                {
                    "source_stage_ids": group["source_stage_ids"],
                    "fingerprint_sha256": group["fingerprint_sha256"],
                    "reason_code": exc.reason_code,
                }
                for group in groups
            ]
            candidate = _rebuild_candidate(
                logical, {}, (), no_merge_reason=exc.reason_code,
            )

    merge_groups = [
        {
            "merged_stage_id": record["merged_stage_id"],
            "source_stage_ids": record["source_stage_ids"],
            "fingerprint_sha256": record["fingerprint_sha256"],
        }
        for record in merge_records
    ]
    provenance = [
        {
            "merged_stage_id": group["merged_stage_id"],
            "source_stage_ids": list(group["source_stage_ids"]),
        }
        for group in merge_groups
    ]
    report = {
        "schema_version": MERGE_CANDIDATE_SCHEMA_VERSION,
        "merger_version": MERGE_CANDIDATE_VERSION,
        "mode": "shadow",
        "status": status,
        "selected_graph_kind": "logical_dag",
        "logical_graph_digest": logical["digest"],
        "optimized_graph_digest": candidate["digest"],
        "analysis_digest": analysis["digest"],
        "fallback": {
            "used": status != "evaluated",
            "reason_code": fallback_reason,
        },
        "merge_groups": merge_groups,
        "provenance": provenance,
        "rejections": rejections,
        "summary": {
            "logical_stage_count": logical["summary"]["stage_count"],
            "optimized_stage_count": candidate["summary"]["stage_count"],
            "logical_edge_count": logical["summary"]["edge_count"],
            "optimized_edge_count": candidate["summary"]["edge_count"],
            "merge_group_count": len(merge_groups),
            "merged_source_stage_count": sum(
                len(group["source_stage_ids"]) for group in merge_groups
            ),
            "rejected_group_count": len(rejections),
        },
        "candidate_graph": candidate,
    }
    report["digest"] = _digest(report)
    return validate_merge_candidate(report)


def validate_merge_candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a merge report and its detached optimized candidate graph."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphMergeError("merge candidate fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != MERGE_CANDIDATE_SCHEMA_VERSION:
        raise TaskGraphMergeError("unsupported merge candidate schema")
    if report.get("merger_version") != MERGE_CANDIDATE_VERSION:
        raise TaskGraphMergeError("unsupported merge candidate version")
    if report.get("mode") != "shadow":
        raise TaskGraphMergeError("merge candidate must remain shadow-only")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphMergeError("merge candidate cannot select optimized graph")
    _sha256(report.get("logical_graph_digest"), "logical_graph_digest")
    _sha256(report.get("optimized_graph_digest"), "optimized_graph_digest")
    _sha256(report.get("analysis_digest"), "analysis_digest")

    status = report.get("status")
    if status not in {"evaluated", "no_op", "fallback"}:
        raise TaskGraphMergeError("merge candidate status is invalid")
    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphMergeError("merge candidate fallback is invalid")
    if not isinstance(fallback["used"], bool):
        raise TaskGraphMergeError("merge candidate fallback.used is invalid")
    _identifier(fallback["reason_code"], "fallback reason_code")
    if fallback["used"] != (status != "evaluated"):
        raise TaskGraphMergeError("merge candidate fallback status mismatch")
    expected_fallback_reason = {
        "evaluated": "merge_candidate_ready",
        "no_op": "no_shareable_pair",
    }.get(status)
    if expected_fallback_reason is not None and fallback["reason_code"] != (
        expected_fallback_reason
    ):
        raise TaskGraphMergeError("merge candidate fallback reason mismatch")

    candidate = report.get("candidate_graph")
    checked_candidate = validate_projection(candidate)
    if checked_candidate["graph_kind"] != "optimized_dag":
        raise TaskGraphMergeError("candidate graph must be optimized_dag")
    if checked_candidate["digest"] != report["optimized_graph_digest"]:
        raise TaskGraphMergeError("candidate graph digest mismatch")

    merge_groups = _validate_groups(report.get("merge_groups"))
    provenance = _validate_provenance(report.get("provenance"))
    if provenance != [
        {
            "merged_stage_id": group["merged_stage_id"],
            "source_stage_ids": list(group["source_stage_ids"]),
        }
        for group in merge_groups
    ]:
        raise TaskGraphMergeError("provenance does not match merge groups")
    rejections = _validate_rejections(report.get("rejections"))
    if status == "evaluated" and rejections:
        raise TaskGraphMergeError("evaluated merge candidate has rejections")
    if status != "fallback" and rejections:
        raise TaskGraphMergeError("non-fallback merge candidate has rejections")
    if status == "fallback" and merge_groups:
        raise TaskGraphMergeError("fallback merge candidate cannot contain merges")
    if status == "no_op" and merge_groups:
        raise TaskGraphMergeError("no-op merge candidate cannot contain merges")
    if status == "evaluated" and not merge_groups:
        raise TaskGraphMergeError("evaluated merge candidate requires a merge")
    if status == "fallback":
        if not rejections or any(
            rejection["reason_code"] != fallback["reason_code"]
            for rejection in rejections
        ):
            raise TaskGraphMergeError("fallback reason does not match rejections")

    candidate_stage_ids = {
        node["stage_id"]
        for node in checked_candidate["nodes"]
        if node.get("node_kind") == "stage"
    }
    merged_stage_ids = {group["merged_stage_id"] for group in merge_groups}
    candidate_shared_ids = {
        stage_id for stage_id in candidate_stage_ids if stage_id.startswith("shared:")
    }
    if status == "evaluated" and candidate_shared_ids != merged_stage_ids:
        raise TaskGraphMergeError("candidate shared Stages do not match provenance")
    for group in merge_groups:
        if group["merged_stage_id"] not in candidate_stage_ids:
            raise TaskGraphMergeError("merged Stage is missing from candidate graph")
        if candidate_stage_ids.intersection(group["source_stage_ids"]):
            raise TaskGraphMergeError("source Stage remains in candidate graph")

    merge_events = [
        event for event in checked_candidate.get("trace", [])
        if event.get("rule") == "merge_common_subtask"
    ]
    if status == "evaluated":
        if len(merge_events) != len(merge_groups) or any(
            event.get("accepted") is not True
            or event.get("reason_code") != "shared_stage_candidate"
            for event in merge_events
        ):
            raise TaskGraphMergeError("candidate merge trace does not match groups")
    elif len(merge_events) != 1 or merge_events[0] != {
        "rule": "merge_common_subtask",
        "reason_code": fallback["reason_code"],
        "affected_node_ids": [],
        "accepted": False,
    }:
        raise TaskGraphMergeError("candidate no-op trace does not match fallback")

    summary = report.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphMergeError("merge candidate summary is invalid")
    expected_summary = {
        "logical_stage_count": summary["logical_stage_count"],
        "optimized_stage_count": checked_candidate["summary"]["stage_count"],
        "logical_edge_count": summary["logical_edge_count"],
        "optimized_edge_count": checked_candidate["summary"]["edge_count"],
        "merge_group_count": len(merge_groups),
        "merged_source_stage_count": sum(
            len(group["source_stage_ids"]) for group in merge_groups
        ),
        "rejected_group_count": len(rejections),
    }
    for key, expected in expected_summary.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphMergeError(f"summary {key} must be non-negative")
        if value != expected:
            raise TaskGraphMergeError(f"summary {key} does not match candidate")
    if status == "evaluated" and summary["optimized_stage_count"] != (
        summary["logical_stage_count"]
        - summary["merged_source_stage_count"]
        + summary["merge_group_count"]
    ):
        raise TaskGraphMergeError("merged Stage count does not match provenance")
    if status == "evaluated" and summary["optimized_edge_count"] > summary[
        "logical_edge_count"
    ]:
        raise TaskGraphMergeError("merge candidate cannot add dependency edges")
    if status != "evaluated" and summary["optimized_stage_count"] != summary[
        "logical_stage_count"
    ]:
        raise TaskGraphMergeError("no-op candidate changed Stage count")
    if status != "evaluated" and summary["optimized_edge_count"] != summary[
        "logical_edge_count"
    ]:
        raise TaskGraphMergeError("no-op candidate changed edge count")

    supplied_digest = _sha256(report.get("digest"), "merge candidate digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphMergeError("merge candidate digest mismatch")
    return _detached(report)


def _share_groups(
    stage_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_fingerprint: dict[str, list[str]] = {}
    for stage_id, result in stage_results.items():
        if result["eligible"]:
            by_fingerprint.setdefault(result["fingerprint_sha256"], []).append(
                stage_id,
            )
    return [
        {
            "source_stage_ids": sorted(stage_ids),
            "fingerprint_sha256": fingerprint,
        }
        for fingerprint, stage_ids in sorted(by_fingerprint.items())
        if len(stage_ids) >= 2
    ]


def _build_merge_mapping(
    groups: Sequence[Mapping[str, Any]],
    logical: Mapping[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    stage_ids = {
        node["stage_id"]
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    mapping: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for group in groups:
        fingerprint = group["fingerprint_sha256"]
        merged_stage_id = f"shared:{fingerprint}"
        if merged_stage_id in stage_ids or merged_stage_id in mapping.values():
            raise _MergeSafetyError("shared_stage_id_collision")
        source_stage_ids = sorted(group["source_stage_ids"])
        for source_stage_id in source_stage_ids:
            if source_stage_id in mapping:
                raise _MergeSafetyError("shared_stage_id_collision")
            mapping[source_stage_id] = merged_stage_id
        records.append({
            "merged_stage_id": merged_stage_id,
            "source_stage_ids": source_stage_ids,
            "fingerprint_sha256": fingerprint,
        })

    _check_edge_safety(logical, mapping, set(mapping.values()))
    return mapping, records


def _check_edge_safety(
    logical: Mapping[str, Any],
    mapping: Mapping[str, str],
    merged_stage_ids: set[str],
) -> None:
    seen: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for edge in logical["edges"]:
        source = _mapped_node_stage(edge["source_node_id"], mapping)
        target = _mapped_node_stage(edge["target_node_id"], mapping)
        relation = edge["relation"]
        if source == target:
            raise _MergeSafetyError("self_dependency_after_merge")
        key = (relation, source, target)
        previous = seen.get(key)
        if previous is None:
            seen[key] = edge
            continue
        if previous.get("semantic_contract") != edge.get("semantic_contract"):
            raise _MergeSafetyError("semantic_contract_collision")
        if target not in merged_stage_ids:
            raise _MergeSafetyError("join_arity_would_change")

    by_stage = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    for stage_id, node in by_stage.items():
        mapped_stage_id = mapping.get(stage_id, stage_id)
        if mapped_stage_id not in merged_stage_ids:
            continue
        dependencies = {
            mapping.get(dependency, dependency)
            for dependency in node.get("depends_on", [])
        }
        minimum = node.get("execution_constraints", {}).get(
            "minimum_successful_dependencies",
        )
        if minimum is not None and minimum > len(dependencies):
            raise _MergeSafetyError("join_arity_would_change")


def _rebuild_candidate(
    logical: Mapping[str, Any],
    mapping: Mapping[str, str],
    merge_records: Sequence[Mapping[str, Any]],
    *,
    no_merge_reason: str = "no_shareable_pair",
) -> dict[str, Any]:
    nodes_by_stage = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    representative_by_merged = {
        record["merged_stage_id"]: record["source_stage_ids"][0]
        for record in merge_records
    }
    candidate_nodes: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for stage_id in sorted(nodes_by_stage):
        mapped_stage_id = mapping.get(stage_id, stage_id)
        if mapped_stage_id in emitted:
            continue
        representative_stage_id = representative_by_merged.get(
            mapped_stage_id, stage_id,
        )
        node = _detached(nodes_by_stage[representative_stage_id])
        node["stage_id"] = mapped_stage_id
        node["node_id"] = f"stage:{mapped_stage_id}"
        node["depends_on"] = sorted({
            mapping.get(dependency, dependency)
            for dependency in node.get("depends_on", [])
        })
        node["input_bindings"] = sorted(
            [
                {
                    **binding,
                    "dependency_stage_id": mapping.get(
                        binding["dependency_stage_id"],
                        binding["dependency_stage_id"],
                    ),
                }
                for binding in node.get("input_bindings", [])
            ],
            key=lambda binding: binding["target_key"],
        )
        candidate_nodes.append(node)
        emitted.add(mapped_stage_id)

    edge_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in logical["edges"]:
        source = _mapped_node_stage(edge["source_node_id"], mapping)
        target = _mapped_node_stage(edge["target_node_id"], mapping)
        if source == target:
            continue
        relation = edge["relation"]
        key = (relation, source, target)
        if key in edge_by_key:
            continue
        rebuilt_edge = _detached(edge)
        rebuilt_edge["source_node_id"] = f"stage:{source}"
        rebuilt_edge["target_node_id"] = f"stage:{target}"
        rebuilt_edge["edge_id"] = f"edge:{relation}:{source}:{target}"
        edge_by_key[key] = rebuilt_edge

    candidate = _detached(logical)
    candidate["graph_kind"] = "optimized_dag"
    candidate["nodes"] = sorted(candidate_nodes, key=lambda node: node["node_id"])
    candidate["edges"] = sorted(
        edge_by_key.values(), key=lambda edge: edge["edge_id"],
    )
    candidate["summary"] = _detached(logical["summary"])
    candidate["summary"]["final_stage_id"] = mapping.get(
        candidate["summary"]["final_stage_id"],
        candidate["summary"]["final_stage_id"],
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
    merge_events = [
        {
            "rule": "merge_common_subtask",
            "reason_code": "shared_stage_candidate",
            "affected_node_ids": sorted(
                [f"stage:{record['merged_stage_id']}"]
                + [f"stage:{stage_id}" for stage_id in record["source_stage_ids"]],
            ),
            "accepted": True,
        }
        for record in merge_records
    ]
    if not merge_events:
        merge_events = [{
            "rule": "merge_common_subtask",
            "reason_code": no_merge_reason,
            "affected_node_ids": [],
            "accepted": False,
        }]
    candidate["trace"] = [
        event for event in _detached(logical.get("trace", []))
        if event.get("rule") != "merge_common_subtask"
    ] + merge_events
    candidate["digest"] = _digest({
        key: value for key, value in candidate.items() if key != "digest"
    })
    return validate_projection(candidate)


def _mapped_node_stage(node_id: str, mapping: Mapping[str, str]) -> str:
    if not isinstance(node_id, str) or not node_id.startswith("stage:"):
        raise _MergeSafetyError("semantic_contract_collision")
    stage_id = node_id.split(":", 1)[1]
    return mapping.get(stage_id, stage_id)


def _validate_groups(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphMergeError("merge groups must be a list")
    groups: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_merged: set[str] = set()
    for group in value:
        if not isinstance(group, Mapping) or set(group) != _GROUP_KEYS:
            raise TaskGraphMergeError("merge group fields are invalid")
        merged = _identifier(group["merged_stage_id"], "merged_stage_id")
        if not merged.startswith("shared:"):
            raise TaskGraphMergeError("merged stage ID must use shared namespace")
        sources = _sorted_identifiers(group["source_stage_ids"], "source_stage_ids")
        if len(sources) < 2:
            raise TaskGraphMergeError("merge group requires two source stages")
        fingerprint = _sha256(group["fingerprint_sha256"], "group fingerprint")
        if merged != f"shared:{fingerprint}":
            raise TaskGraphMergeError("merged Stage ID does not match fingerprint")
        if merged in seen_merged or seen_sources.intersection(sources):
            raise TaskGraphMergeError("merge groups overlap")
        seen_merged.add(merged)
        seen_sources.update(sources)
        groups.append({
            "merged_stage_id": merged,
            "source_stage_ids": list(sources),
            "fingerprint_sha256": fingerprint,
        })
    if groups != sorted(groups, key=lambda group: group["merged_stage_id"]):
        raise TaskGraphMergeError("merge groups must be sorted")
    return groups


def _validate_provenance(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphMergeError("provenance must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _PROVENANCE_KEYS:
            raise TaskGraphMergeError("provenance fields are invalid")
        result.append({
            "merged_stage_id": _identifier(
                item["merged_stage_id"], "provenance merged_stage_id",
            ),
            "source_stage_ids": list(
                _sorted_identifiers(item["source_stage_ids"], "provenance source_stage_ids"),
            ),
        })
    return result


def _validate_rejections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TaskGraphMergeError("merge rejections must be a list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _REJECTION_KEYS:
            raise TaskGraphMergeError("merge rejection fields are invalid")
        result.append({
            "source_stage_ids": list(_sorted_identifiers(
                item["source_stage_ids"], "rejection source_stage_ids",
            )),
            "fingerprint_sha256": _sha256(
                item["fingerprint_sha256"], "rejection fingerprint",
            ),
            "reason_code": _enum(
                item["reason_code"], _REJECTION_REASONS, "rejection reason_code",
            ),
        })
        if len(result[-1]["source_stage_ids"]) < 2:
            raise TaskGraphMergeError("merge rejection requires two source stages")
    return result


def _sorted_identifiers(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskGraphMergeError(f"{field_name} must be a non-empty list")
    identifiers = tuple(_identifier(item, field_name) for item in value)
    if list(identifiers) != sorted(set(identifiers)):
        raise TaskGraphMergeError(f"{field_name} must be sorted and unique")
    return identifiers


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphMergeError("merge candidate keys must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphMergeError(
                    f"merge candidate contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphMergeError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphMergeError(f"{field_name} must be a SHA-256 digest")
    return value


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise TaskGraphMergeError(f"{field_name} is unsupported")
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
    "MERGE_CANDIDATE_SCHEMA_VERSION",
    "MERGE_CANDIDATE_VERSION",
    "TaskGraphMergeError",
    "merge_shareable_stages",
    "validate_merge_candidate",
]
