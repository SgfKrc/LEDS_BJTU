"""Fixed-template TaskGraph shadow optimization consumer.

The consumer compares a logical projection with a candidate optimized graph,
optionally binds immutable fan-out payload references, and always leaves the
logical DAG selected for execution.  It has no dependency on the Coordinator,
Provider registry, journal, or Worker runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from task_graph_optimization import (
    OPTIMIZER_VERSION,
    optimize_task_graph,
    project_task_graph,
)
from task_graph_payloads import (
    TaskPayloadStore,
    bind_payload_plan,
    validate_payload_reference,
)


SHADOW_REPORT_SCHEMA_VERSION = "qlh.task_graph_shadow_report.v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_KEYS = frozenset({
    "schema_version",
    "template_id",
    "mode",
    "status",
    "selected_graph_kind",
    "logical_graph_digest",
    "optimized_graph_digest",
    "optimizer_version",
    "enabled_rules",
    "fallback",
    "metrics",
    "payload_references",
    "trace",
    "report_digest",
})
_METRIC_KEYS = frozenset({
    "candidate_available",
    "logical_stage_count",
    "optimized_stage_count",
    "logical_edge_count",
    "optimized_edge_count",
    "culled_stage_count",
    "reduced_edge_count",
    "payload_plan_count",
    "payload_bound_count",
    "payload_source_bytes",
    "avoided_inline_bytes",
    "reference_contract_bytes",
})
_FORBIDDEN_KEYS = frozenset({
    "body",
    "error",
    "output",
    "path",
    "root_input",
    "secret",
    "token",
    "url",
})
_TEMPLATE_POLICIES = {
    "dual_candidate": {
        "rules": frozenset(),
        "payload_binding": False,
    },
    "llm_sd15_v1": {
        "rules": frozenset(),
        "payload_binding": False,
    },
    "image_grid_v1": {
        "rules": frozenset(),
        "payload_binding": False,
    },
    "g2_fanout_shadow_v1": {
        "rules": frozenset({"semantic_transitive_reduction"}),
        "payload_binding": True,
    },
}
_TEMPLATE_SHAPES = {
    "dual_candidate": {
        "final_stage_id": "aggregate",
        "stages": {
            "candidate_a": {"stage_type": "full_inference", "depends_on": []},
            "candidate_b": {"stage_type": "full_inference", "depends_on": []},
            "aggregate": {
                "stage_type": "aggregate",
                "depends_on": ["candidate_a", "candidate_b"],
                "minimum_successful_dependencies": 1,
            },
        },
    },
    "llm_sd15_v1": {
        "final_stage_id": "image_generate",
        "stages": {
            "image_prompt": {
                "stage_type": "image_prompt",
                "depends_on": [],
                "pure": True,
                "model_identity": True,
            },
            "image_generate": {
                "stage_type": "image_generate",
                "depends_on": ["image_prompt"],
                "input_bindings": [{
                    "dependency_stage_id": "image_prompt",
                    "output_key": "content",
                    "target_key": "prompt",
                }],
            },
        },
    },
    "image_grid_v1": {
        "final_stage_id": "image_grid",
        "stages": {
            **{
                f"seed_{index}": {
                    "stage_type": "image_generate",
                    "depends_on": [],
                    "pure": True,
                }
                for index in range(4)
            },
            "image_grid": {
                "stage_type": "image_grid",
                "depends_on": [f"seed_{index}" for index in range(4)],
                "minimum_successful_dependencies": 4,
            },
        },
    },
    "g2_fanout_shadow_v1": {
        "final_stage_id": "final",
        "stages": {
            "shared": {
                "stage_type": "transform",
                "depends_on": [],
                "pure": True,
            },
            "left": {"stage_type": "transform", "depends_on": ["shared"]},
            "right": {"stage_type": "transform", "depends_on": ["shared"]},
            "final": {
                "stage_type": "aggregate",
                "depends_on": ["left", "right"],
            },
        },
    },
}


class TaskGraphShadowError(ValueError):
    """Raised when a shadow report is malformed."""


def run_task_graph_shadow(
    template_id: str,
    stages: Sequence[Any] | Mapping[str, Any],
    final_stage_id: str,
    *,
    graph_id: str = "",
    rules: Sequence[str] = (),
    payloads: Mapping[str, bytes | bytearray | memoryview] | None = None,
    payload_store: TaskPayloadStore | None = None,
    data_scope: str = "",
) -> dict[str, Any]:
    """Evaluate one admitted fixed template without changing execution."""

    safe_template = _identifier(template_id, "template_id")
    policy = _TEMPLATE_POLICIES.get(safe_template)
    if policy is None:
        return _fallback_report(safe_template, "template_not_admitted")
    safe_rules = _rules(rules)
    safe_graph_id = (
        _identifier(graph_id, "graph_id")
        if graph_id else f"shadow:{safe_template}"
    )
    logical: dict[str, Any] | None = None
    phase = "projection"
    try:
        logical = project_task_graph(
            stages,
            final_stage_id,
            graph_kind="logical_dag",
            graph_id=safe_graph_id,
        )
        if not _matches_template_shape(safe_template, logical):
            return _fallback_report(
                safe_template, "template_shape_not_admitted", logical=logical,
                enabled_rules=safe_rules,
            )
        if not set(safe_rules).issubset(policy["rules"]):
            return _fallback_report(
                safe_template, "rule_not_admitted", logical=logical,
                enabled_rules=safe_rules,
            )
        phase = "optimization"
        optimized = optimize_task_graph(logical, rules=safe_rules)
        plans = optimized["payload_plan"]
        supplied_payloads = {} if payloads is None else payloads
        if not isinstance(supplied_payloads, Mapping):
            return _fallback_report(
                safe_template, "payload_mapping_invalid", logical=logical,
                enabled_rules=safe_rules,
            )
        planned_sources = {plan["source_stage_id"] for plan in plans}
        if set(supplied_payloads) - planned_sources:
            return _fallback_report(
                safe_template, "payload_source_not_planned", logical=logical,
                enabled_rules=safe_rules,
            )
        if supplied_payloads and not policy["payload_binding"]:
            return _fallback_report(
                safe_template, "payload_binding_not_admitted", logical=logical,
                enabled_rules=safe_rules,
            )
        if supplied_payloads and (
            not isinstance(payload_store, TaskPayloadStore) or not data_scope
        ):
            return _fallback_report(
                safe_template, "payload_store_unavailable", logical=logical,
                enabled_rules=safe_rules,
            )

        phase = "payload_binding"
        references: list[dict[str, Any]] = []
        payload_source_bytes = 0
        avoided_inline_bytes = 0
        reference_contract_bytes = 0
        for plan in plans:
            source_stage_id = plan["source_stage_id"]
            if source_stage_id not in supplied_payloads:
                continue
            body = supplied_payloads[source_stage_id]
            reference = bind_payload_plan(
                payload_store,
                plan,
                body,
                data_scope=data_scope,
            )
            references.append(reference)
            size_bytes = reference["size_bytes"]
            consumer_count = len(reference["consumer_stage_ids"])
            payload_source_bytes += size_bytes
            avoided_inline_bytes += size_bytes * max(consumer_count - 1, 0)
            reference_contract_bytes += len(_canonical_bytes(reference)) * consumer_count

        metrics = {
            "candidate_available": True,
            "logical_stage_count": logical["summary"]["stage_count"],
            "optimized_stage_count": optimized["summary"]["optimized_stage_count"],
            "logical_edge_count": logical["summary"]["edge_count"],
            "optimized_edge_count": optimized["optimized_graph"]["summary"][
                "edge_count"
            ],
            "culled_stage_count": optimized["summary"]["culled_stage_count"],
            "reduced_edge_count": optimized["summary"]["reduced_edge_count"],
            "payload_plan_count": len(plans),
            "payload_bound_count": len(references),
            "payload_source_bytes": payload_source_bytes,
            "avoided_inline_bytes": avoided_inline_bytes,
            "reference_contract_bytes": reference_contract_bytes,
        }
        report = {
            "schema_version": SHADOW_REPORT_SCHEMA_VERSION,
            "template_id": safe_template,
            "mode": "shadow",
            "status": "evaluated",
            "selected_graph_kind": "logical_dag",
            "logical_graph_digest": logical["digest"],
            "optimized_graph_digest": optimized["optimized_graph"]["digest"],
            "optimizer_version": optimized["optimizer_version"],
            "enabled_rules": optimized["enabled_rules"],
            "fallback": {
                "used": False,
                "reason_code": "shadow_execution_unchanged",
            },
            "metrics": metrics,
            "payload_references": sorted(
                references, key=lambda reference: reference["payload_id"],
            ),
            "trace": optimized["trace"],
        }
        report["report_digest"] = _digest(report)
        return validate_shadow_report(report)
    except Exception:
        return _fallback_report(
            safe_template,
            f"{phase}_failed",
            logical=logical,
            enabled_rules=safe_rules,
        )


def validate_shadow_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a report before it is persisted or shown to operators."""

    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphShadowError("shadow report fields are invalid")
    _assert_no_forbidden_fields(report)
    if report.get("schema_version") != SHADOW_REPORT_SCHEMA_VERSION:
        raise TaskGraphShadowError("unsupported shadow report schema")
    _identifier(report.get("template_id"), "template_id")
    if report.get("mode") != "shadow":
        raise TaskGraphShadowError("shadow report mode is invalid")
    status = report.get("status")
    if status not in {"evaluated", "fallback"}:
        raise TaskGraphShadowError("shadow report status is invalid")
    if report.get("selected_graph_kind") != "logical_dag":
        raise TaskGraphShadowError("shadow report cannot select an optimized graph")
    _optional_digest(report.get("logical_graph_digest"), "logical_graph_digest")
    _optional_digest(report.get("optimized_graph_digest"), "optimized_graph_digest")
    if report.get("optimizer_version") != OPTIMIZER_VERSION:
        raise TaskGraphShadowError("shadow optimizer version is invalid")
    enabled_rules = report.get("enabled_rules")
    if not isinstance(enabled_rules, list) or any(
        not isinstance(rule, str) for rule in enabled_rules
    ) or enabled_rules != sorted(set(enabled_rules)):
        raise TaskGraphShadowError("enabled_rules are invalid")

    fallback = report.get("fallback")
    if not isinstance(fallback, Mapping) or set(fallback) != {
        "used", "reason_code",
    }:
        raise TaskGraphShadowError("fallback projection is invalid")
    if not isinstance(fallback.get("used"), bool):
        raise TaskGraphShadowError("fallback.used must be boolean")
    _identifier(fallback.get("reason_code"), "fallback reason_code")
    if fallback["used"] != (status == "fallback"):
        raise TaskGraphShadowError("fallback status does not match report status")

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != _METRIC_KEYS:
        raise TaskGraphShadowError("shadow metrics are invalid")
    if not isinstance(metrics.get("candidate_available"), bool):
        raise TaskGraphShadowError("candidate_available must be boolean")
    for key in _METRIC_KEYS - {"candidate_available"}:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphShadowError(f"metric {key} must be non-negative")
    if metrics["payload_bound_count"] > metrics["payload_plan_count"]:
        raise TaskGraphShadowError("bound payload count exceeds plan count")

    references = report.get("payload_references")
    trace = report.get("trace")
    if not isinstance(references, list) or not isinstance(trace, list):
        raise TaskGraphShadowError("shadow references/trace must be lists")
    checked_references = [validate_payload_reference(item) for item in references]
    if metrics["payload_bound_count"] != len(checked_references):
        raise TaskGraphShadowError("payload bound count does not match references")
    for event in trace:
        _validate_trace_event(event)
    if status == "evaluated":
        if not metrics["candidate_available"]:
            raise TaskGraphShadowError("evaluated report requires a candidate")
        if not report["optimized_graph_digest"]:
            raise TaskGraphShadowError("evaluated report requires an optimized digest")
    elif (
        metrics["candidate_available"]
        or report["optimized_graph_digest"]
        or references
        or trace
    ):
        raise TaskGraphShadowError("fallback report cannot expose candidate state")

    supplied_digest = report.get("report_digest")
    if not isinstance(supplied_digest, str) or not _SHA256_RE.fullmatch(
        supplied_digest,
    ):
        raise TaskGraphShadowError("shadow report digest is invalid")
    unsigned = {key: value for key, value in report.items() if key != "report_digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphShadowError("shadow report digest does not match content")
    return json.loads(json.dumps(report, ensure_ascii=True, sort_keys=True))


def release_shadow_payloads(
    store: TaskPayloadStore,
    report: Mapping[str, Any],
) -> None:
    """Release payload references produced by one validated shadow report."""

    if not isinstance(store, TaskPayloadStore):
        raise TaskGraphShadowError("store must be a TaskPayloadStore")
    checked = validate_shadow_report(report)
    for reference in checked["payload_references"]:
        store.release(reference)


def _fallback_report(
    template_id: str,
    reason_code: str,
    *,
    logical: Mapping[str, Any] | None = None,
    enabled_rules: Sequence[str] = (),
) -> dict[str, Any]:
    logical_stage_count = 0
    logical_edge_count = 0
    logical_digest = ""
    if logical is not None:
        logical_stage_count = int(logical["summary"]["stage_count"])
        logical_edge_count = int(logical["summary"]["edge_count"])
        logical_digest = str(logical["digest"])
    report = {
        "schema_version": SHADOW_REPORT_SCHEMA_VERSION,
        "template_id": template_id,
        "mode": "shadow",
        "status": "fallback",
        "selected_graph_kind": "logical_dag",
        "logical_graph_digest": logical_digest,
        "optimized_graph_digest": "",
        "optimizer_version": OPTIMIZER_VERSION,
        "enabled_rules": sorted(set(enabled_rules)),
        "fallback": {
            "used": True,
            "reason_code": _identifier(reason_code, "fallback reason_code"),
        },
        "metrics": {
            "candidate_available": False,
            "logical_stage_count": logical_stage_count,
            "optimized_stage_count": 0,
            "logical_edge_count": logical_edge_count,
            "optimized_edge_count": 0,
            "culled_stage_count": 0,
            "reduced_edge_count": 0,
            "payload_plan_count": 0,
            "payload_bound_count": 0,
            "payload_source_bytes": 0,
            "avoided_inline_bytes": 0,
            "reference_contract_bytes": 0,
        },
        "payload_references": [],
        "trace": [],
    }
    report["report_digest"] = _digest(report)
    return validate_shadow_report(report)


def _matches_template_shape(
    template_id: str,
    logical: Mapping[str, Any],
) -> bool:
    shape = _TEMPLATE_SHAPES.get(template_id)
    if shape is None or logical.get("summary", {}).get("final_stage_id") != shape[
        "final_stage_id"
    ]:
        return False
    expected = shape["stages"]
    nodes = logical.get("nodes")
    if not isinstance(nodes, list):
        return False
    by_stage = {
        node.get("stage_id"): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("node_kind") == "stage"
    }
    if set(by_stage) != set(expected):
        return False
    for stage_id, contract in expected.items():
        node = by_stage[stage_id]
        if node.get("stage_type") != contract["stage_type"]:
            return False
        if node.get("depends_on") != contract["depends_on"]:
            return False
        constraints = node.get("execution_constraints")
        if not isinstance(constraints, Mapping):
            return False
        if "minimum_successful_dependencies" in contract and constraints.get(
            "minimum_successful_dependencies",
        ) != contract["minimum_successful_dependencies"]:
            return False
        if "pure" in contract:
            provider_constraints = node.get("provider_constraints")
            if not isinstance(provider_constraints, Mapping) or provider_constraints.get(
                "pure",
            ) != contract["pure"]:
                return False
        if contract.get("model_identity") and not isinstance(
            node.get("model_identity"), Mapping,
        ):
            return False
        if "input_bindings" in contract and node.get("input_bindings") != contract[
            "input_bindings"
        ]:
            return False
    return True


def _rules(rules: Sequence[str]) -> list[str]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise TaskGraphShadowError("rules must be a sequence")
    if any(not isinstance(rule, str) for rule in rules):
        raise TaskGraphShadowError("rules must contain text values")
    return sorted(set(rules))


def _validate_trace_event(event: Any) -> None:
    if not isinstance(event, Mapping) or set(event) != {
        "rule", "reason_code", "affected_node_ids", "accepted",
    }:
        raise TaskGraphShadowError("shadow trace event is invalid")
    _identifier(event.get("rule"), "trace rule")
    _identifier(event.get("reason_code"), "trace reason_code")
    affected = event.get("affected_node_ids")
    if not isinstance(affected, list) or len(affected) != len(set(affected)):
        raise TaskGraphShadowError("trace affected_node_ids are invalid")
    for node_id in affected:
        _identifier(node_id, "trace node_id")
    if not isinstance(event.get("accepted"), bool):
        raise TaskGraphShadowError("trace accepted must be boolean")


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphShadowError("shadow report key must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphShadowError(f"shadow report contains forbidden field {key!r}")
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphShadowError(f"{field_name} must be a safe identifier")
    return value


def _optional_digest(value: Any, field_name: str) -> str:
    if value == "":
        return ""
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphShadowError(f"{field_name} is invalid")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


__all__ = [
    "SHADOW_REPORT_SCHEMA_VERSION",
    "TaskGraphShadowError",
    "release_shadow_payloads",
    "run_task_graph_shadow",
    "validate_shadow_report",
]
