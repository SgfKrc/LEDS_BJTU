"""Read-only, privacy-safe TaskGraph projection snapshots.

This module deliberately has no dependency on the TaskGraph executor.  It turns
``StageSpec`` objects or existing workflow snapshots into versioned graph views
for inspection and future shadow planning only.  It must not be used to choose
Providers, mutate a workflow, or replace an execution DAG.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "qlh.task_graph_projection.v1"
GRAPH_KINDS = frozenset({
    "logical_dag",
    "optimized_dag",
    "analysis_view",
    "attempt_graph",
    "provider_topology",
})

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_FORBIDDEN_KEYS = frozenset({
    "blob_id",
    "error",
    "grant",
    "key",
    "lease_id",
    "output",
    "output_body",
    "path",
    "raw_error",
    "request_id",
    "result_metadata",
    "root_input",
    "root_input_overrides",
    "runtime_context",
    "secret",
    "token",
    "url",
})
_TRACE_KEYS = frozenset({
    "accepted",
    "affected_node_ids",
    "reason_code",
    "rule",
})
_MODEL_IDENTITY_KEYS = (
    "engine",
    "format",
    "model_id",
    "revision",
    "sha256",
)


class TaskGraphProjectionError(ValueError):
    """Raised when an input cannot produce a safe projection."""


class GraphTypeError(TaskGraphProjectionError):
    """Raised when a graph view is used as a different graph type."""


def project_task_graph(
    stages: Sequence[Any] | Mapping[str, Any],
    final_stage_id: str,
    *,
    graph_kind: str = "logical_dag",
    graph_id: str = "task_graph",
    trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project static Stage specifications without reading executable payloads.

    ``stages`` may be ``StageSpec`` instances, safe stage mappings, or a
    workflow-shaped mapping containing a ``stages`` sequence.  The caller keeps
    ownership of the source objects; the returned graph is a detached dict.
    """

    rows = _normalise_stages(stages)
    return _build_projection(
        graph_kind,
        rows,
        final_stage_id,
        graph_id=graph_id,
        trace=trace,
    )


def project_workflow_snapshot(
    workflow_snapshot: Mapping[str, Any],
    *,
    graph_kind: str = "attempt_graph",
    trace: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project an existing ``WorkflowRecord.snapshot()`` without altering it."""

    if not isinstance(workflow_snapshot, Mapping):
        raise TaskGraphProjectionError("workflow snapshot must be a mapping")
    stages = workflow_snapshot.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise TaskGraphProjectionError("workflow snapshot must contain stages")
    final_stage_id = _identifier(
        workflow_snapshot.get("final_stage_id"), "final_stage_id",
    )
    graph_id = workflow_snapshot.get("workflow_id", "workflow")
    return _build_projection(
        graph_kind,
        _normalise_stages(stages),
        final_stage_id,
        graph_id=_identifier(graph_id, "workflow_id"),
        trace=trace,
    )


def require_graph_kind(
    projection: Mapping[str, Any], expected_graph_kind: str,
) -> dict[str, Any]:
    """Validate a projection and reject accidental graph-view type mixing."""

    checked = validate_projection(projection)
    _graph_kind(expected_graph_kind)
    if checked["graph_kind"] != expected_graph_kind:
        raise GraphTypeError(
            "projection graph_kind "
            f"{checked['graph_kind']!r} cannot be used as "
            f"{expected_graph_kind!r}",
        )
    return checked


def validate_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, namespaces, privacy boundary, and stable digest."""

    if not isinstance(projection, Mapping):
        raise TaskGraphProjectionError("projection must be a mapping")
    _assert_no_forbidden_content(projection)
    if projection.get("schema_version") != SCHEMA_VERSION:
        raise TaskGraphProjectionError("unsupported projection schema_version")
    graph_kind = _graph_kind(projection.get("graph_kind"))
    _identifier(projection.get("graph_id"), "graph_id")

    nodes = projection.get("nodes")
    edges = projection.get("edges")
    summary = projection.get("summary")
    trace = projection.get("trace")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TaskGraphProjectionError("projection nodes and edges must be lists")
    if not isinstance(summary, Mapping) or not isinstance(trace, list):
        raise TaskGraphProjectionError("projection summary and trace are invalid")

    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise TaskGraphProjectionError("projection node must be a mapping")
        node_id = _identifier(node.get("node_id"), "node_id")
        if node_id in node_ids:
            raise TaskGraphProjectionError(f"duplicate node_id {node_id!r}")
        node_ids.add(node_id)
        _validate_node_namespace(node, graph_kind)

    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise TaskGraphProjectionError("projection edge must be a mapping")
        edge_id = _identifier(edge.get("edge_id"), "edge_id")
        if not edge_id.startswith("edge:"):
            raise TaskGraphProjectionError("edge_id must use the edge: namespace")
        if edge_id in edge_ids:
            raise TaskGraphProjectionError(f"duplicate edge_id {edge_id!r}")
        edge_ids.add(edge_id)
        if edge.get("source_node_id") not in node_ids:
            raise TaskGraphProjectionError("edge source_node_id is unknown")
        if edge.get("target_node_id") not in node_ids:
            raise TaskGraphProjectionError("edge target_node_id is unknown")
        if not isinstance(edge.get("relation"), str):
            raise TaskGraphProjectionError("edge relation must be a string")

    for event in trace:
        _normalise_trace_event(event)

    supplied_digest = projection.get("digest")
    if not isinstance(supplied_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", supplied_digest,
    ):
        raise TaskGraphProjectionError("projection digest is invalid")
    unsigned = {key: value for key, value in projection.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphProjectionError("projection digest does not match content")
    return _detached(projection)


def _build_projection(
    graph_kind: str,
    rows: list[dict[str, Any]],
    final_stage_id: str,
    *,
    graph_id: str,
    trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_kind = _graph_kind(graph_kind)
    graph_id = _identifier(graph_id, "graph_id")
    final_stage_id = _identifier(final_stage_id, "final_stage_id")
    _validate_stage_dag(rows, final_stage_id)
    safe_trace = [_normalise_trace_event(event) for event in trace]

    if graph_kind == "provider_topology":
        nodes, edges = _provider_topology(rows)
    else:
        nodes, edges = _stage_view(rows, graph_kind)

    projection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_kind": graph_kind,
        "graph_id": graph_id,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "stage_count": len(rows),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "final_stage_id": final_stage_id,
            "provider_count": len({
                candidate
                for row in rows
                for candidate in row["provider_candidates"]
            }),
        },
        "trace": safe_trace,
    }
    projection["digest"] = _digest(projection)
    return validate_projection(projection)


def _stage_view(
    rows: list[dict[str, Any]], graph_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for row in rows:
        node: dict[str, Any] = {
            "node_id": f"stage:{row['stage_id']}",
            "node_kind": "stage",
            "stage_id": row["stage_id"],
            "stage_type": row["stage_type"],
            "depends_on": list(row["depends_on"]),
            "input_bindings": row["input_bindings"],
            "provider_constraints": {
                "requested_provider": row["requested_provider"],
                "fallback_providers": list(row["fallback_providers"]),
                "pure": row["pure"],
            },
            "model_identity": row["model_identity"],
        }
        if graph_kind == "attempt_graph":
            node["state"] = row["state"]
            node["lease_epoch"] = row["lease_epoch"]
            node["winner_attempt_id"] = row["winner_attempt_id"]
            node["attempt_count"] = len(row["attempts"])
        nodes.append(node)
        for dependency_id in row["depends_on"]:
            edges.append({
                "edge_id": f"edge:depends:{dependency_id}:{row['stage_id']}",
                "source_node_id": f"stage:{dependency_id}",
                "target_node_id": f"stage:{row['stage_id']}",
                "relation": "depends_on",
            })

    if graph_kind != "attempt_graph":
        return nodes, edges

    for row in rows:
        previous_attempt_node_id = ""
        for attempt in row["attempts"]:
            attempt_node_id = f"attempt:{row['stage_id']}:{attempt['attempt_id']}"
            nodes.append({
                "node_id": attempt_node_id,
                "node_kind": "attempt",
                "stage_id": row["stage_id"],
                "attempt_id": attempt["attempt_id"],
                "provider": attempt["provider"],
                "provider_kind": attempt["provider_kind"],
                "provider_node_id": attempt["provider_node_id"],
                "lease_epoch": attempt["lease_epoch"],
                "state": attempt["state"],
                "reservation_active": attempt["reservation_active"],
                "is_winner": attempt["attempt_id"] == row["winner_attempt_id"],
                "result_digest_present": attempt["result_digest_present"],
            })
            edges.append({
                "edge_id": (
                    f"edge:attempt:{row['stage_id']}:{attempt['attempt_id']}"
                ),
                "source_node_id": f"stage:{row['stage_id']}",
                "target_node_id": attempt_node_id,
                "relation": "attempt_of",
            })
            if previous_attempt_node_id:
                edges.append({
                    "edge_id": (
                        "edge:retry:"
                        f"{previous_attempt_node_id[8:]}:{attempt_node_id[8:]}"
                    ),
                    "source_node_id": previous_attempt_node_id,
                    "target_node_id": attempt_node_id,
                    "relation": "retry_after",
                })
            previous_attempt_node_id = attempt_node_id
    return nodes, edges


def _provider_topology(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    providers: dict[str, set[str]] = {}
    for row in rows:
        for provider in row["provider_candidates"]:
            providers.setdefault(provider, set()).add(row["stage_id"])
        for attempt in row["attempts"]:
            providers.setdefault(attempt["provider"], set()).add(row["stage_id"])
    nodes = [
        {
            "node_id": f"provider:{provider}",
            "node_kind": "provider",
            "provider": provider,
            "requested_by_stage_ids": sorted(stage_ids),
        }
        for provider, stage_ids in sorted(providers.items())
    ]
    # Topology links belong to a future provider/network source, never to stage
    # dependencies.  The empty edge set makes that separation explicit in G0.
    return nodes, []


def _normalise_stages(
    stages: Sequence[Any] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(stages, Mapping):
        stages = stages.get("stages", list(stages.values()))
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise TaskGraphProjectionError("stages must be a sequence")

    rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_id = _identifier(_read(stage, "stage_id"), "stage_id")
        stage_type = _identifier(_read(stage, "stage_type"), "stage_type")
        depends_on = _identifiers(_read(stage, "depends_on", ()), "depends_on")
        requested_provider = _identifier(
            _read(stage, "requested_provider", _read(stage, "provider", "")),
            "provider",
        )
        fallback_providers = _identifiers(
            _read(stage, "fallback_providers", ()), "fallback_providers",
        )
        attempts = _normalise_attempts(_read(stage, "attempts", ()))
        rows.append({
            "stage_id": stage_id,
            "stage_type": stage_type,
            "depends_on": depends_on,
            "requested_provider": requested_provider,
            "fallback_providers": fallback_providers,
            "provider_candidates": (requested_provider, *fallback_providers),
            "pure": bool(_read(stage, "pure", False)),
            "model_identity": _model_identity(_read(stage, "model_identity", None)),
            "input_bindings": _input_bindings(
                _read(stage, "input_bindings", {}),
            ),
            "state": _safe_state(_read(stage, "state", "unknown")),
            "lease_epoch": _nonnegative_int(_read(stage, "lease_epoch", 0), "lease_epoch"),
            "winner_attempt_id": _optional_identifier(
                _read(stage, "winner_attempt_id", ""), "winner_attempt_id",
            ),
            "attempts": attempts,
        })
    return rows


def _normalise_attempts(attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        raise TaskGraphProjectionError("attempts must be a sequence")
    normalised: list[dict[str, Any]] = []
    for attempt in attempts:
        normalised.append({
            "attempt_id": _identifier(_read(attempt, "attempt_id"), "attempt_id"),
            "provider": _identifier(_read(attempt, "provider"), "provider"),
            "provider_kind": _optional_identifier(
                _read(attempt, "provider_kind", ""), "provider_kind",
            ),
            "provider_node_id": _optional_identifier(
                _read(attempt, "provider_node_id", ""), "provider_node_id",
            ),
            "lease_epoch": _nonnegative_int(
                _read(attempt, "lease_epoch", 0), "attempt lease_epoch",
            ),
            "state": _safe_state(_read(attempt, "state", "unknown")),
            "reservation_active": bool(_read(attempt, "reservation_active", False)),
            "result_digest_present": bool(
                _read(attempt, "result_sha256", ""),
            ),
        })
    return normalised


def _input_bindings(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise TaskGraphProjectionError("input_bindings must be a mapping")
    bindings: list[dict[str, str]] = []
    for target, source in value.items():
        target_key = _identifier(target, "input binding target")
        if isinstance(source, Mapping):
            dependency = source.get("dependency_stage_id")
            output_key = source.get("output_key")
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            if len(source) != 2:
                raise TaskGraphProjectionError("input binding source must have two items")
            dependency, output_key = source
        else:
            raise TaskGraphProjectionError("input binding source is invalid")
        bindings.append({
            "target_key": target_key,
            "dependency_stage_id": _identifier(
                dependency, "input binding dependency_stage_id",
            ),
            "output_key": _identifier(output_key, "input binding output_key"),
        })
    return sorted(bindings, key=lambda binding: binding["target_key"])


def _model_identity(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        snapshot = getattr(value, "snapshot", None)
        value = snapshot() if callable(snapshot) else {
            key: getattr(value, key, None) for key in _MODEL_IDENTITY_KEYS
        }
    if not isinstance(value, Mapping):
        raise TaskGraphProjectionError("model_identity must be a mapping")
    identity: dict[str, str] = {}
    for key in _MODEL_IDENTITY_KEYS:
        raw_value = value.get(key)
        if raw_value in (None, ""):
            continue
        identity[key] = _identifier(raw_value, f"model_identity.{key}")
    return identity or None


def _validate_stage_dag(rows: list[dict[str, Any]], final_stage_id: str) -> None:
    by_id = {row["stage_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise TaskGraphProjectionError("stage_id values must be unique")
    if final_stage_id not in by_id:
        raise TaskGraphProjectionError("final_stage_id is not present")
    for row in rows:
        for dependency_id in row["depends_on"]:
            if dependency_id not in by_id:
                raise TaskGraphProjectionError(
                    f"stage {row['stage_id']!r} depends on unknown stage "
                    f"{dependency_id!r}",
                )
            if dependency_id == row["stage_id"]:
                raise TaskGraphProjectionError("a stage cannot depend on itself")
        for binding in row["input_bindings"]:
            if binding["dependency_stage_id"] not in row["depends_on"]:
                raise TaskGraphProjectionError(
                    "input binding dependency must be declared in depends_on",
                )

    pending = {row["stage_id"]: set(row["depends_on"]) for row in rows}
    while pending:
        ready = sorted(stage_id for stage_id, deps in pending.items() if not deps)
        if not ready:
            raise TaskGraphProjectionError("stage graph must be acyclic")
        for stage_id in ready:
            del pending[stage_id]
        ready_set = set(ready)
        for dependencies in pending.values():
            dependencies.difference_update(ready_set)


def _validate_node_namespace(node: Mapping[str, Any], graph_kind: str) -> None:
    node_id = node["node_id"]
    node_kind = node.get("node_kind")
    expected_prefix = {
        "stage": "stage:",
        "attempt": "attempt:",
        "provider": "provider:",
    }.get(node_kind)
    if expected_prefix is None or not node_id.startswith(expected_prefix):
        raise TaskGraphProjectionError("node_id does not match node_kind namespace")
    allowed_kinds = {
        "logical_dag": {"stage"},
        "optimized_dag": {"stage"},
        "analysis_view": {"stage"},
        "attempt_graph": {"stage", "attempt"},
        "provider_topology": {"provider"},
    }[graph_kind]
    if node_kind not in allowed_kinds:
        raise GraphTypeError(
            f"{node_kind!r} nodes are invalid in {graph_kind!r}",
        )


def _normalise_trace_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise TaskGraphProjectionError("trace event must be a mapping")
    if set(event) - _TRACE_KEYS:
        raise TaskGraphProjectionError("trace event contains unsupported fields")
    rule = _identifier(event.get("rule"), "trace rule")
    reason_code = _identifier(event.get("reason_code"), "trace reason_code")
    node_ids = _identifiers(event.get("affected_node_ids", ()), "trace affected_node_ids")
    accepted = event.get("accepted")
    if not isinstance(accepted, bool):
        raise TaskGraphProjectionError("trace accepted must be boolean")
    return {
        "rule": rule,
        "reason_code": reason_code,
        "affected_node_ids": sorted(node_ids),
        "accepted": accepted,
    }


def _assert_no_forbidden_content(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphProjectionError("projection mapping key must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphProjectionError(
                    f"projection contains forbidden field {key!r}",
                )
            _assert_no_forbidden_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_content(nested)


def _read(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _graph_kind(value: Any) -> str:
    if value not in GRAPH_KINDS:
        raise GraphTypeError(f"unsupported graph_kind {value!r}")
    return value


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphProjectionError(f"{field_name} must be a safe identifier")
    return value


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, field_name)


def _identifiers(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TaskGraphProjectionError(f"{field_name} must be a sequence")
    return tuple(_identifier(item, field_name) for item in value)


def _safe_state(value: Any) -> str:
    return _identifier(value, "state") if value not in (None, "") else "unknown"


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskGraphProjectionError(f"{field_name} must be a non-negative integer")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    canonical = _canonical(value)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskGraphProjectionError("projection cannot contain non-finite float")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TaskGraphProjectionError("projection mapping key must be text")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TaskGraphProjectionError(
        f"projection contains unsupported value type {type(value).__name__}",
    )


def _detached(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(_canonical(value), ensure_ascii=True, sort_keys=True))


__all__ = [
    "GRAPH_KINDS",
    "SCHEMA_VERSION",
    "GraphTypeError",
    "TaskGraphProjectionError",
    "project_task_graph",
    "project_workflow_snapshot",
    "require_graph_kind",
    "validate_projection",
]
