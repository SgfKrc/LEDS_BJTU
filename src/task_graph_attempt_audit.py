"""Read-only consistency audit for TaskGraph attempt projections.

G5.1 audits the existing attempt/lease/winner/fallback implementation.  It
projects a workflow snapshot through the privacy-safe attempt graph boundary,
keeps only whitelisted counters and recovery facts, and emits reason codes.
It never retries a Stage, submits a result, changes a lease, or mutates the
source snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from task_graph_optimization import (
    project_workflow_snapshot,
    require_graph_kind,
)


ATTEMPT_AUDIT_SCHEMA_VERSION = "qlh.task_graph_attempt_audit.v1"
ATTEMPT_AUDITOR_VERSION = "task-attempt-audit-v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORKFLOW_STATES = frozenset({
    "created", "running", "result_ready", "completed", "failed", "cancelled",
})
_STAGE_STATES = frozenset({
    "blocked", "ready", "running", "completed", "failed", "skipped", "cancelled",
})
_ATTEMPT_STATES = frozenset({
    "running", "completed", "failed", "expired", "cancelled",
})
_TERMINAL_WORKFLOW_STATES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_STAGE_STATES = frozenset({
    "completed", "failed", "skipped", "cancelled",
})
_TERMINAL_ATTEMPT_STATES = frozenset({
    "completed", "failed", "expired", "cancelled",
})
_RECOVERY_REASONS = frozenset({
    "coordinator_restarted_before_result_commit",
    "coordinator_restarted_during_execution",
})
_FORBIDDEN_KEYS = frozenset({
    "body",
    "content",
    "error",
    "grant",
    "history",
    "lease_id",
    "output",
    "path",
    "prompt",
    "raw",
    "request_id",
    "reservation_id",
    "result_metadata",
    "root_input",
    "runtime_context",
    "secret",
    "token",
    "url",
})

_REPORT_KEYS = frozenset({
    "schema_version",
    "auditor_version",
    "mode",
    "status",
    "runtime_actions_enabled",
    "attempt_graph",
    "audit_evidence",
    "stage_audits",
    "recovery_audit",
    "gaps",
    "summary",
    "digest",
})
_EVIDENCE_KEYS = frozenset({
    "workflow", "stages", "journal", "evidence_digest",
})
_WORKFLOW_EVIDENCE_KEYS = frozenset({
    "workflow_id",
    "state",
    "last_sequence",
    "stage_count",
    "completed_stage_count",
    "failed_stage_count",
    "skipped_stage_count",
    "cancelled_stage_count",
    "attempt_count",
    "retry_count",
    "same_provider_retry_count",
    "result_rejection_count",
    "recovered_after_restart",
    "recovery_reason_code",
    "recovery_pending",
    "runtime_status",
})
_STAGE_EVIDENCE_KEYS = frozenset({
    "stage_id",
    "selected_provider",
    "retry_count",
    "same_provider_retry_count",
    "result_rejection_count",
    "stage_result_available",
    "stage_result_sha256",
    "recovery_reason_code",
    "attempt_results",
})
_ATTEMPT_RESULT_KEYS = frozenset({
    "attempt_id", "result_sha256", "recovery_reason_code",
})
_JOURNAL_EVIDENCE_KEYS = frozenset({
    "provided", "event_count", "sequences", "recovery_events",
})
_RECOVERY_EVENT_KEYS = frozenset({
    "sequence",
    "previous_state",
    "recovery_reason_code",
    "expired_attempts",
    "failed_stages",
    "skipped_stages",
})
_STAGE_AUDIT_KEYS = frozenset({
    "stage_id",
    "status",
    "attempt_count",
    "lease_epoch",
    "winner_count",
    "running_attempt_count",
    "retry_count",
    "same_provider_retry_count",
    "result_rejection_count",
    "gap_reason_codes",
})
_RECOVERY_AUDIT_KEYS = frozenset({
    "status",
    "recovered_after_restart",
    "journal_evidence_provided",
    "event_count",
    "recovery_event_count",
    "reason_code",
    "gap_reason_codes",
})
_GAP_KEYS = frozenset({
    "scope_kind", "stage_id", "attempt_id", "reason_code",
})
_SUMMARY_KEYS = frozenset({
    "stage_count",
    "attempt_count",
    "winner_count",
    "running_attempt_count",
    "retry_count",
    "same_provider_retry_count",
    "result_rejection_count",
    "stage_gap_count",
    "workflow_gap_count",
    "recovery_gap_count",
    "gap_count",
    "runtime_action_count",
})


class TaskGraphAttemptAuditError(ValueError):
    """Raised when an audit input or persisted report is malformed."""


def audit_task_graph_attempts(
    workflow_snapshot: Mapping[str, Any],
    *,
    journal_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit one workflow snapshot without changing TaskGraph runtime state."""

    if not isinstance(workflow_snapshot, Mapping):
        raise TaskGraphAttemptAuditError("workflow snapshot must be a mapping")
    try:
        attempt_graph = project_workflow_snapshot(
            workflow_snapshot,
            graph_kind="attempt_graph",
        )
    except (TypeError, ValueError) as exc:
        raise TaskGraphAttemptAuditError(
            f"workflow snapshot cannot produce a safe attempt graph: {exc}"
        ) from exc
    evidence = _build_evidence(
        workflow_snapshot,
        attempt_graph,
        journal_events=journal_events,
    )
    evaluated = _evaluate(attempt_graph, evidence)
    report = {
        "schema_version": ATTEMPT_AUDIT_SCHEMA_VERSION,
        "auditor_version": ATTEMPT_AUDITOR_VERSION,
        "mode": "read_only",
        "status": evaluated["status"],
        "runtime_actions_enabled": False,
        "attempt_graph": attempt_graph,
        "audit_evidence": evidence,
        "stage_audits": evaluated["stage_audits"],
        "recovery_audit": evaluated["recovery_audit"],
        "gaps": evaluated["gaps"],
        "summary": evaluated["summary"],
    }
    report["digest"] = _digest(report)
    return validate_attempt_graph_audit(report)


def validate_attempt_graph_audit(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and independently recompute a persisted G5.1 audit report."""

    _assert_no_forbidden_fields(report)
    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise TaskGraphAttemptAuditError("attempt audit fields are invalid")
    if report.get("schema_version") != ATTEMPT_AUDIT_SCHEMA_VERSION:
        raise TaskGraphAttemptAuditError("unsupported attempt audit schema")
    if report.get("auditor_version") != ATTEMPT_AUDITOR_VERSION:
        raise TaskGraphAttemptAuditError("unsupported attempt auditor version")
    if report.get("mode") != "read_only":
        raise TaskGraphAttemptAuditError("attempt audit must be read-only")
    if report.get("runtime_actions_enabled") is not False:
        raise TaskGraphAttemptAuditError("attempt audit cannot enable runtime actions")

    try:
        attempt_graph = require_graph_kind(
            report.get("attempt_graph"), "attempt_graph",
        )
    except (TypeError, ValueError) as exc:
        raise TaskGraphAttemptAuditError(f"invalid attempt graph: {exc}") from exc
    evidence = _validate_evidence(report.get("audit_evidence"), attempt_graph)
    expected = _evaluate(attempt_graph, evidence)
    for field_name in (
        "status", "stage_audits", "recovery_audit", "gaps", "summary",
    ):
        if report.get(field_name) != expected[field_name]:
            raise TaskGraphAttemptAuditError(
                f"attempt audit {field_name} does not match evidence"
            )

    supplied_digest = _sha256(report.get("digest"), "audit digest")
    unsigned = {key: value for key, value in report.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphAttemptAuditError("attempt audit digest mismatch")
    return _detached(report)


def _build_evidence(
    snapshot: Mapping[str, Any],
    attempt_graph: Mapping[str, Any],
    *,
    journal_events: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    graph = require_graph_kind(attempt_graph, "attempt_graph")
    workflow_id = _identifier(snapshot.get("workflow_id"), "workflow_id")
    if workflow_id != graph["graph_id"]:
        raise TaskGraphAttemptAuditError("workflow_id does not match attempt graph")
    raw_stages = snapshot.get("stages")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        raise TaskGraphAttemptAuditError("workflow stages must be a sequence")

    stages = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, Mapping):
            raise TaskGraphAttemptAuditError("workflow stage must be a mapping")
        stage_id = _identifier(raw_stage.get("stage_id"), "stage_id")
        raw_attempts = raw_stage.get("attempts", [])
        if not isinstance(raw_attempts, Sequence) or isinstance(
            raw_attempts, (str, bytes),
        ):
            raise TaskGraphAttemptAuditError("stage attempts must be a sequence")
        attempt_results = []
        for raw_attempt in raw_attempts:
            if not isinstance(raw_attempt, Mapping):
                raise TaskGraphAttemptAuditError("attempt must be a mapping")
            attempt_results.append({
                "attempt_id": _identifier(
                    raw_attempt.get("attempt_id"), "attempt_id",
                ),
                "result_sha256": _optional_sha256(
                    raw_attempt.get("result_sha256", ""),
                    "attempt result_sha256",
                ),
                "recovery_reason_code": _optional_identifier(
                    raw_attempt.get("recovery_reason", ""),
                    "attempt recovery_reason",
                ),
            })
        stages.append({
            "stage_id": stage_id,
            "selected_provider": _identifier(
                raw_stage.get("selected_provider", raw_stage.get("provider")),
                "selected_provider",
            ),
            "retry_count": _nonnegative_int(
                raw_stage.get("retry_count", 0), "stage retry_count",
            ),
            "same_provider_retry_count": _nonnegative_int(
                raw_stage.get("same_provider_retry_count", 0),
                "stage same_provider_retry_count",
            ),
            "result_rejection_count": _nonnegative_int(
                raw_stage.get("result_rejection_count", 0),
                "stage result_rejection_count",
            ),
            "stage_result_available": _boolean(
                raw_stage.get("output_available", False),
                "stage output_available",
            ),
            "stage_result_sha256": _optional_sha256(
                raw_stage.get("output_sha256", ""), "stage output_sha256",
            ),
            "recovery_reason_code": _optional_identifier(
                raw_stage.get("recovery_reason", ""),
                "stage recovery_reason",
            ),
            "attempt_results": attempt_results,
        })
    stages.sort(key=lambda item: item["stage_id"])

    workflow = {
        "workflow_id": workflow_id,
        "state": _identifier(snapshot.get("state"), "workflow state"),
        "last_sequence": _nonnegative_int(
            snapshot.get("last_sequence", 0), "last_sequence",
        ),
        "stage_count": _nonnegative_int(
            snapshot.get("stage_count", len(stages)), "stage_count",
        ),
        "completed_stage_count": _nonnegative_int(
            snapshot.get("completed_stage_count", 0), "completed_stage_count",
        ),
        "failed_stage_count": _nonnegative_int(
            snapshot.get("failed_stage_count", 0), "failed_stage_count",
        ),
        "skipped_stage_count": _nonnegative_int(
            snapshot.get("skipped_stage_count", 0), "skipped_stage_count",
        ),
        "cancelled_stage_count": _nonnegative_int(
            snapshot.get("cancelled_stage_count", 0), "cancelled_stage_count",
        ),
        "attempt_count": _nonnegative_int(
            snapshot.get("attempt_count", sum(
                len(stage["attempt_results"]) for stage in stages
            )),
            "attempt_count",
        ),
        "retry_count": _nonnegative_int(
            snapshot.get("retry_count", 0), "workflow retry_count",
        ),
        "same_provider_retry_count": _nonnegative_int(
            snapshot.get("same_provider_retry_count", 0),
            "workflow same_provider_retry_count",
        ),
        "result_rejection_count": _nonnegative_int(
            snapshot.get("result_rejection_count", 0),
            "workflow result_rejection_count",
        ),
        "recovered_after_restart": _boolean(
            snapshot.get("recovered_after_restart", False),
            "recovered_after_restart",
        ),
        "recovery_reason_code": _optional_identifier(
            snapshot.get("recovery_reason", ""), "workflow recovery_reason",
        ),
        "recovery_pending": _boolean(
            snapshot.get("recovery_pending", False), "recovery_pending",
        ),
        "runtime_status": _optional_identifier(
            snapshot.get("runtime_status", ""), "runtime_status",
        ),
    }
    journal = _journal_evidence(journal_events)
    evidence = {"workflow": workflow, "stages": stages, "journal": journal}
    evidence["evidence_digest"] = _digest(evidence)
    return _validate_evidence(evidence, graph)


def _journal_evidence(
    journal_events: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if journal_events is None:
        return {
            "provided": False,
            "event_count": 0,
            "sequences": [],
            "recovery_events": [],
        }
    if not isinstance(journal_events, Sequence) or isinstance(
        journal_events, (str, bytes),
    ):
        raise TaskGraphAttemptAuditError("journal_events must be a sequence")
    sequences = []
    recovery_events = []
    for event in journal_events:
        if not isinstance(event, Mapping):
            raise TaskGraphAttemptAuditError("journal event must be a mapping")
        sequence = _positive_int(event.get("sequence"), "journal sequence")
        sequences.append(sequence)
        if event.get("event_type") != "workflow_recovered_after_restart":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise TaskGraphAttemptAuditError(
                "recovery journal event payload must be a mapping"
            )
        recovery_events.append({
            "sequence": sequence,
            "previous_state": _identifier(
                payload.get("previous_state"), "recovery previous_state",
            ),
            "recovery_reason_code": _identifier(
                payload.get("recovery_reason"), "recovery reason",
            ),
            "expired_attempts": _nonnegative_int(
                payload.get("expired_attempts"), "recovery expired_attempts",
            ),
            "failed_stages": _nonnegative_int(
                payload.get("failed_stages"), "recovery failed_stages",
            ),
            "skipped_stages": _nonnegative_int(
                payload.get("skipped_stages"), "recovery skipped_stages",
            ),
        })
    sequences.sort()
    recovery_events.sort(key=lambda item: item["sequence"])
    return {
        "provided": True,
        "event_count": len(journal_events),
        "sequences": sequences,
        "recovery_events": recovery_events,
    }


def _validate_evidence(
    evidence: Any,
    attempt_graph: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_KEYS:
        raise TaskGraphAttemptAuditError("audit evidence fields are invalid")
    _assert_no_forbidden_fields(evidence)
    workflow = evidence.get("workflow")
    if not isinstance(workflow, Mapping) or set(workflow) != _WORKFLOW_EVIDENCE_KEYS:
        raise TaskGraphAttemptAuditError("workflow evidence fields are invalid")
    workflow_id = _identifier(workflow.get("workflow_id"), "workflow_id")
    if workflow_id != attempt_graph["graph_id"]:
        raise TaskGraphAttemptAuditError("evidence workflow_id mismatch")
    _identifier(workflow.get("state"), "workflow state")
    for key in (
        "last_sequence",
        "stage_count",
        "completed_stage_count",
        "failed_stage_count",
        "skipped_stage_count",
        "cancelled_stage_count",
        "attempt_count",
        "retry_count",
        "same_provider_retry_count",
        "result_rejection_count",
    ):
        _nonnegative_int(workflow.get(key), f"workflow {key}")
    for key in (
        "recovered_after_restart", "recovery_pending",
    ):
        _boolean(workflow.get(key), f"workflow {key}")
    _optional_identifier(
        workflow.get("recovery_reason_code"), "workflow recovery_reason_code",
    )
    _optional_identifier(workflow.get("runtime_status"), "runtime_status")

    graph_stage_nodes = {
        node["stage_id"]: node
        for node in attempt_graph["nodes"]
        if node.get("node_kind") == "stage"
    }
    graph_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in attempt_graph["nodes"]:
        if node.get("node_kind") == "attempt":
            graph_attempts[node["stage_id"]].append(node)

    stages = evidence.get("stages")
    if not isinstance(stages, list):
        raise TaskGraphAttemptAuditError("stage evidence must be a list")
    evidence_stage_ids = []
    for stage in stages:
        if not isinstance(stage, Mapping) or set(stage) != _STAGE_EVIDENCE_KEYS:
            raise TaskGraphAttemptAuditError("stage evidence fields are invalid")
        stage_id = _identifier(stage.get("stage_id"), "stage_id")
        evidence_stage_ids.append(stage_id)
        _identifier(stage.get("selected_provider"), "selected_provider")
        for key in (
            "retry_count", "same_provider_retry_count", "result_rejection_count",
        ):
            _nonnegative_int(stage.get(key), f"stage {key}")
        _boolean(stage.get("stage_result_available"), "stage_result_available")
        _optional_sha256(stage.get("stage_result_sha256"), "stage result_sha256")
        _optional_identifier(
            stage.get("recovery_reason_code"), "stage recovery_reason_code",
        )
        attempt_results = stage.get("attempt_results")
        if not isinstance(attempt_results, list):
            raise TaskGraphAttemptAuditError("attempt result evidence must be a list")
        graph_rows = graph_attempts.get(stage_id, [])
        if len(attempt_results) != len(graph_rows):
            raise TaskGraphAttemptAuditError("attempt result evidence count mismatch")
        for result, graph_row in zip(attempt_results, graph_rows):
            if not isinstance(result, Mapping) or set(result) != _ATTEMPT_RESULT_KEYS:
                raise TaskGraphAttemptAuditError(
                    "attempt result evidence fields are invalid"
                )
            attempt_id = _identifier(result.get("attempt_id"), "attempt_id")
            result_sha = _optional_sha256(
                result.get("result_sha256"), "attempt result_sha256",
            )
            _optional_identifier(
                result.get("recovery_reason_code"),
                "attempt recovery_reason_code",
            )
            if attempt_id != graph_row["attempt_id"]:
                raise TaskGraphAttemptAuditError("attempt result identity mismatch")
            if bool(result_sha) != graph_row["result_digest_present"]:
                raise TaskGraphAttemptAuditError("attempt result digest presence mismatch")
    if evidence_stage_ids != sorted(graph_stage_nodes):
        raise TaskGraphAttemptAuditError("stage evidence identity mismatch")

    journal = evidence.get("journal")
    if not isinstance(journal, Mapping) or set(journal) != _JOURNAL_EVIDENCE_KEYS:
        raise TaskGraphAttemptAuditError("journal evidence fields are invalid")
    provided = _boolean(journal.get("provided"), "journal provided")
    event_count = _nonnegative_int(journal.get("event_count"), "journal event_count")
    sequences = journal.get("sequences")
    recovery_events = journal.get("recovery_events")
    if not isinstance(sequences, list) or not isinstance(recovery_events, list):
        raise TaskGraphAttemptAuditError("journal evidence lists are invalid")
    for sequence in sequences:
        _positive_int(sequence, "journal sequence")
    if sequences != sorted(sequences):
        raise TaskGraphAttemptAuditError("journal sequences must be sorted")
    if event_count != len(sequences):
        raise TaskGraphAttemptAuditError("journal event count mismatch")
    if not provided and (event_count or sequences or recovery_events):
        raise TaskGraphAttemptAuditError("missing journal evidence contains facts")
    for event in recovery_events:
        if not isinstance(event, Mapping) or set(event) != _RECOVERY_EVENT_KEYS:
            raise TaskGraphAttemptAuditError("recovery event fields are invalid")
        sequence = _positive_int(event.get("sequence"), "recovery sequence")
        if sequence not in sequences:
            raise TaskGraphAttemptAuditError("recovery event sequence is unknown")
        _identifier(event.get("previous_state"), "recovery previous_state")
        _identifier(
            event.get("recovery_reason_code"), "recovery reason_code",
        )
        for key in ("expired_attempts", "failed_stages", "skipped_stages"):
            _nonnegative_int(event.get(key), f"recovery {key}")

    supplied_digest = _sha256(
        evidence.get("evidence_digest"), "evidence digest",
    )
    unsigned = {
        key: value for key, value in evidence.items() if key != "evidence_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphAttemptAuditError("audit evidence digest mismatch")
    return _detached(evidence)


def _evaluate(
    attempt_graph: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    graph = require_graph_kind(attempt_graph, "attempt_graph")
    workflow = evidence["workflow"]
    stage_evidence = {item["stage_id"]: item for item in evidence["stages"]}
    stage_nodes = {
        node["stage_id"]: node
        for node in graph["nodes"]
        if node.get("node_kind") == "stage"
    }
    attempts_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph["nodes"]:
        if node.get("node_kind") == "attempt":
            attempts_by_stage[node["stage_id"]].append(node)

    gaps: list[dict[str, str]] = []

    def add_gap(
        scope_kind: str,
        reason_code: str,
        *,
        stage_id: str = "",
        attempt_id: str = "",
    ) -> None:
        gaps.append({
            "scope_kind": scope_kind,
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "reason_code": reason_code,
        })

    all_attempt_ids = [
        attempt["attempt_id"]
        for attempts in attempts_by_stage.values()
        for attempt in attempts
    ]
    for attempt_id, count in sorted(Counter(all_attempt_ids).items()):
        if count > 1:
            add_gap(
                "workflow", "attempt_id_reused_across_stages",
                attempt_id=attempt_id,
            )

    stage_audits = []
    for stage_id in sorted(stage_nodes):
        stage = stage_nodes[stage_id]
        extra = stage_evidence[stage_id]
        attempts = attempts_by_stage.get(stage_id, [])
        attempt_results = {
            item["attempt_id"]: item for item in extra["attempt_results"]
        }
        before = len(gaps)
        stage_state = stage["state"]
        if stage_state not in _STAGE_STATES:
            add_gap("stage", "stage_state_unknown", stage_id=stage_id)

        candidates = [
            stage["provider_constraints"]["requested_provider"],
            *stage["provider_constraints"]["fallback_providers"],
        ]
        selected_provider = extra["selected_provider"]
        selected_index = (
            candidates.index(selected_provider)
            if selected_provider in candidates else -1
        )
        if selected_index < 0:
            add_gap(
                "stage", "selected_provider_not_admitted", stage_id=stage_id,
            )

        provider_indices = []
        epochs = []
        running_attempts = []
        for attempt in attempts:
            attempt_id = attempt["attempt_id"]
            state = attempt["state"]
            epochs.append(attempt["lease_epoch"])
            if state not in _ATTEMPT_STATES:
                add_gap(
                    "attempt", "attempt_state_unknown",
                    stage_id=stage_id, attempt_id=attempt_id,
                )
            if attempt["provider"] not in candidates:
                add_gap(
                    "attempt", "attempt_provider_not_admitted",
                    stage_id=stage_id, attempt_id=attempt_id,
                )
            else:
                provider_indices.append(candidates.index(attempt["provider"]))
            if state == "running":
                running_attempts.append(attempt)
                if not attempt["reservation_active"]:
                    add_gap(
                        "attempt", "running_attempt_reservation_inactive",
                        stage_id=stage_id, attempt_id=attempt_id,
                    )
            elif state in _TERMINAL_ATTEMPT_STATES and attempt[
                "reservation_active"
            ]:
                add_gap(
                    "attempt", "terminal_attempt_reservation_active",
                    stage_id=stage_id, attempt_id=attempt_id,
                )

        if any(
            current <= previous
            for previous, current in zip(epochs, epochs[1:])
        ):
            add_gap(
                "stage", "attempt_epoch_not_strictly_increasing",
                stage_id=stage_id,
            )
        if attempts:
            if stage["lease_epoch"] != max(epochs):
                add_gap("stage", "stage_epoch_mismatch", stage_id=stage_id)
        elif stage["lease_epoch"] != 0:
            add_gap("stage", "stage_epoch_without_attempt", stage_id=stage_id)
        if any(
            current < previous
            for previous, current in zip(provider_indices, provider_indices[1:])
        ):
            add_gap(
                "stage", "fallback_sequence_regressed", stage_id=stage_id,
            )
        expected_retry_count = (
            selected_index + extra["same_provider_retry_count"]
            if selected_index >= 0 else None
        )
        if (
            expected_retry_count is not None
            and extra["retry_count"] != expected_retry_count
        ):
            add_gap("stage", "retry_accounting_mismatch", stage_id=stage_id)
        if extra["same_provider_retry_count"] > extra["retry_count"]:
            add_gap(
                "stage", "same_provider_retry_count_exceeds_total",
                stage_id=stage_id,
            )
        if len(running_attempts) > 1:
            add_gap("stage", "multiple_running_attempts", stage_id=stage_id)
        if stage_state in _TERMINAL_STAGE_STATES and running_attempts:
            add_gap(
                "stage", "terminal_stage_has_running_attempt", stage_id=stage_id,
            )
        if stage_state == "running" and len(running_attempts) != 1:
            add_gap(
                "stage", "running_stage_attempt_cardinality_mismatch",
                stage_id=stage_id,
            )

        winner_id = stage["winner_attempt_id"]
        winner_rows = [
            attempt for attempt in attempts if attempt["attempt_id"] == winner_id
        ] if winner_id else []
        if winner_id:
            if not winner_rows:
                add_gap(
                    "stage", "winner_not_owned_by_stage", stage_id=stage_id,
                    attempt_id=winner_id,
                )
            elif len(winner_rows) > 1:
                add_gap(
                    "stage", "winner_not_unique", stage_id=stage_id,
                    attempt_id=winner_id,
                )
            else:
                winner = winner_rows[0]
                if winner["state"] != "completed":
                    add_gap(
                        "attempt", "winner_not_completed", stage_id=stage_id,
                        attempt_id=winner_id,
                    )
                winner_sha = attempt_results[winner_id]["result_sha256"]
                stage_sha = extra["stage_result_sha256"]
                if (
                    not extra["stage_result_available"]
                    or not winner_sha
                    or not stage_sha
                ):
                    add_gap(
                        "stage", "winner_result_digest_missing",
                        stage_id=stage_id, attempt_id=winner_id,
                    )
                elif winner_sha != stage_sha:
                    add_gap(
                        "stage", "winner_result_digest_mismatch",
                        stage_id=stage_id, attempt_id=winner_id,
                    )
            if stage_state != "completed":
                add_gap(
                    "stage", "winner_on_noncompleted_stage", stage_id=stage_id,
                    attempt_id=winner_id,
                )
        else:
            if stage_state == "completed":
                add_gap(
                    "stage", "completed_stage_without_winner", stage_id=stage_id,
                )
            if extra["stage_result_available"] or extra["stage_result_sha256"]:
                add_gap(
                    "stage", "stage_result_without_winner", stage_id=stage_id,
                )
        for attempt in attempts:
            if attempt["state"] == "completed" and attempt[
                "attempt_id"
            ] != winner_id:
                add_gap(
                    "attempt", "completed_attempt_not_winner",
                    stage_id=stage_id, attempt_id=attempt["attempt_id"],
                )

        stage_gap_codes = sorted({
            gap["reason_code"]
            for gap in gaps[before:]
            if gap["stage_id"] == stage_id
        })
        stage_audits.append({
            "stage_id": stage_id,
            "status": "gaps_found" if stage_gap_codes else "passed",
            "attempt_count": len(attempts),
            "lease_epoch": stage["lease_epoch"],
            "winner_count": len(winner_rows),
            "running_attempt_count": len(running_attempts),
            "retry_count": extra["retry_count"],
            "same_provider_retry_count": extra["same_provider_retry_count"],
            "result_rejection_count": extra["result_rejection_count"],
            "gap_reason_codes": stage_gap_codes,
        })

    if workflow["state"] not in _WORKFLOW_STATES:
        add_gap("workflow", "workflow_state_unknown")
    state_counts = Counter(stage["state"] for stage in stage_nodes.values())
    workflow_expected = {
        "stage_count": len(stage_nodes),
        "completed_stage_count": state_counts["completed"],
        "failed_stage_count": state_counts["failed"],
        "skipped_stage_count": state_counts["skipped"],
        "cancelled_stage_count": state_counts["cancelled"],
        "attempt_count": len(all_attempt_ids),
        "retry_count": sum(item["retry_count"] for item in evidence["stages"]),
        "same_provider_retry_count": sum(
            item["same_provider_retry_count"] for item in evidence["stages"]
        ),
        "result_rejection_count": sum(
            item["result_rejection_count"] for item in evidence["stages"]
        ),
    }
    counter_reasons = {
        "stage_count": "workflow_stage_count_mismatch",
        "completed_stage_count": "workflow_completed_stage_count_mismatch",
        "failed_stage_count": "workflow_failed_stage_count_mismatch",
        "skipped_stage_count": "workflow_skipped_stage_count_mismatch",
        "cancelled_stage_count": "workflow_cancelled_stage_count_mismatch",
        "attempt_count": "workflow_attempt_count_mismatch",
        "retry_count": "workflow_retry_count_mismatch",
        "same_provider_retry_count": (
            "workflow_same_provider_retry_count_mismatch"
        ),
        "result_rejection_count": "workflow_result_rejection_count_mismatch",
    }
    for key, expected in workflow_expected.items():
        if workflow[key] != expected:
            add_gap("workflow", counter_reasons[key])
    final_stage_id = graph["summary"]["final_stage_id"]
    if (
        workflow["state"] in {"result_ready", "completed"}
        and stage_nodes[final_stage_id]["state"] != "completed"
    ):
        add_gap("workflow", "workflow_final_stage_not_completed")

    recovery_start = len(gaps)
    recovered = workflow["recovered_after_restart"]
    journal = evidence["journal"]
    if not recovered:
        if workflow["recovery_reason_code"]:
            add_gap("recovery", "recovery_reason_without_marker")
        recovery_reason = "not_recovered"
    else:
        recovery_reason = "recovery_commit_verified"
        reason_code = workflow["recovery_reason_code"]
        if reason_code not in _RECOVERY_REASONS:
            add_gap("recovery", "recovery_reason_invalid")
        if workflow["state"] != "failed":
            add_gap("recovery", "recovery_workflow_not_failed")
        if workflow["recovery_pending"]:
            add_gap("recovery", "recovery_pending_after_commit")
        if workflow["runtime_status"] not in {"", "terminal"}:
            add_gap("recovery", "recovery_runtime_status_not_terminal")
        for stage_id, attempts in attempts_by_stage.items():
            for attempt in attempts:
                if attempt["state"] not in _TERMINAL_ATTEMPT_STATES:
                    add_gap(
                        "recovery", "recovery_nonterminal_attempt",
                        stage_id=stage_id, attempt_id=attempt["attempt_id"],
                    )
                if attempt["reservation_active"]:
                    add_gap(
                        "recovery", "recovery_active_reservation",
                        stage_id=stage_id, attempt_id=attempt["attempt_id"],
                    )
        if not journal["provided"]:
            add_gap("recovery", "recovery_event_evidence_missing")
        else:
            sequences = journal["sequences"]
            if sequences != list(range(1, len(sequences) + 1)):
                add_gap("recovery", "journal_sequence_not_contiguous")
            if not sequences or sequences[-1] != workflow["last_sequence"]:
                add_gap("recovery", "journal_last_sequence_mismatch")
            recovery_events = journal["recovery_events"]
            if len(recovery_events) != 1:
                add_gap("recovery", "recovery_event_not_unique")
            else:
                event = recovery_events[0]
                if event["sequence"] != workflow["last_sequence"]:
                    add_gap("recovery", "recovery_event_not_last")
                if event["recovery_reason_code"] != reason_code:
                    add_gap("recovery", "recovery_event_reason_mismatch")
                expected_reason = (
                    "coordinator_restarted_before_result_commit"
                    if event["previous_state"] == "result_ready"
                    else "coordinator_restarted_during_execution"
                )
                if event["recovery_reason_code"] != expected_reason:
                    add_gap("recovery", "recovery_previous_state_mismatch")
                marked_attempts = sum(
                    item["recovery_reason_code"] == reason_code
                    for stage in evidence["stages"]
                    for item in stage["attempt_results"]
                )
                marked_failed = sum(
                    stage["recovery_reason_code"] == reason_code
                    and stage_nodes[stage["stage_id"]]["state"] == "failed"
                    for stage in evidence["stages"]
                )
                marked_skipped = sum(
                    stage["recovery_reason_code"] == reason_code
                    and stage_nodes[stage["stage_id"]]["state"] == "skipped"
                    for stage in evidence["stages"]
                )
                if event["expired_attempts"] != marked_attempts:
                    add_gap("recovery", "recovery_expired_attempt_count_mismatch")
                if event["failed_stages"] != marked_failed:
                    add_gap("recovery", "recovery_failed_stage_count_mismatch")
                if event["skipped_stages"] != marked_skipped:
                    add_gap("recovery", "recovery_skipped_stage_count_mismatch")

    gaps.sort(key=lambda item: (
        item["scope_kind"], item["stage_id"], item["attempt_id"],
        item["reason_code"],
    ))
    recovery_gaps = [
        gap for gap in gaps if gap["scope_kind"] == "recovery"
    ]
    recovery_audit = {
        "status": (
            "not_applicable"
            if not recovered and not recovery_gaps
            else ("gaps_found" if recovery_gaps else "passed")
        ),
        "recovered_after_restart": recovered,
        "journal_evidence_provided": journal["provided"],
        "event_count": journal["event_count"],
        "recovery_event_count": len(journal["recovery_events"]),
        "reason_code": (
            "recovery_evidence_incomplete" if recovery_gaps else recovery_reason
        ),
        "gap_reason_codes": sorted({
            gap["reason_code"] for gap in recovery_gaps
        }),
    }
    summary = {
        "stage_count": len(stage_nodes),
        "attempt_count": len(all_attempt_ids),
        "winner_count": sum(
            bool(stage["winner_attempt_id"]) for stage in stage_nodes.values()
        ),
        "running_attempt_count": sum(
            attempt["state"] == "running"
            for attempts in attempts_by_stage.values()
            for attempt in attempts
        ),
        "retry_count": workflow_expected["retry_count"],
        "same_provider_retry_count": workflow_expected[
            "same_provider_retry_count"
        ],
        "result_rejection_count": workflow_expected["result_rejection_count"],
        "stage_gap_count": sum(
            gap["scope_kind"] in {"stage", "attempt"} for gap in gaps
        ),
        "workflow_gap_count": sum(
            gap["scope_kind"] == "workflow" for gap in gaps
        ),
        "recovery_gap_count": len(recovery_gaps),
        "gap_count": len(gaps),
        "runtime_action_count": 0,
    }
    del recovery_start
    return {
        "status": "gaps_found" if gaps else "passed",
        "stage_audits": stage_audits,
        "recovery_audit": recovery_audit,
        "gaps": gaps,
        "summary": summary,
    }


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise TaskGraphAttemptAuditError(
                    f"attempt audit contains forbidden field {key!r}"
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphAttemptAuditError(f"{label} is invalid")
    return value


def _optional_identifier(value: Any, label: str) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value, label)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphAttemptAuditError(f"{label} is invalid")
    return value


def _optional_sha256(value: Any, label: str) -> str:
    if value is None or value == "":
        return ""
    return _sha256(value, label)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TaskGraphAttemptAuditError(f"{label} must be boolean")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskGraphAttemptAuditError(f"{label} must be non-negative")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise TaskGraphAttemptAuditError(f"{label} must be positive")
    return result


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _detached(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
