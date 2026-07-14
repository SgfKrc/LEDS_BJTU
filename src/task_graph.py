"""Task-level DAG execution with explicit workflow and stage state."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, cast

from task_journal import JournalEvent, TaskJournal, TaskJournalError
from task_provider import (
    CallbackExecutionProvider,
    ExecutionProvider,
    PROVIDER_ID_PATTERN,
    ProviderError,
    ProviderRegistry,
    Reservation,
    StageAttempt as ProviderStageAttempt,
    StageRequest,
    StageResult as ProviderStageResult,
    sanitize_result_metadata,
)


TERMINAL_WORKFLOW_STATES = {"completed", "failed", "cancelled"}
TERMINAL_STAGE_STATES = {"completed", "failed", "skipped", "cancelled"}
TERMINAL_ATTEMPT_STATES = {"completed", "failed", "expired", "cancelled"}

WORKFLOW_STATE_TRANSITIONS = {
    "created": {"running", "cancelled"},
    "running": {"result_ready", "failed", "cancelled"},
    "result_ready": {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}
STAGE_STATE_TRANSITIONS = {
    "blocked": {"ready", "skipped", "cancelled"},
    "ready": {"running", "failed", "skipped", "cancelled"},
    "running": {"ready", "completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
    "cancelled": set(),
}
ATTEMPT_STATE_TRANSITIONS = {
    "running": {"completed", "failed", "expired", "cancelled"},
    "completed": set(),
    "failed": set(),
    "expired": set(),
    "cancelled": set(),
}


class TaskGraphError(RuntimeError):
    """Base error for task-graph validation and execution."""


class TaskGraphUnavailable(TaskGraphError):
    """Raised when the task graph cannot preserve required durable state."""


class WorkflowNotFound(TaskGraphError):
    pass


class WorkflowCancelled(TaskGraphError):
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        super().__init__(f"workflow {workflow_id} cancelled")


class WorkflowExecutionError(TaskGraphError):
    def __init__(self, workflow_id: str, stage_id: str, message: str):
        self.workflow_id = workflow_id
        self.stage_id = stage_id
        super().__init__(f"workflow {workflow_id} stage {stage_id} failed: {message}")


class _StageRetryScheduled(TaskGraphError):
    """Internal control flow: a ready Stage moved to its next Provider."""


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    stage_type: str
    depends_on: tuple[str, ...] = ()
    provider: str = "local_full_model"
    fallback_providers: tuple[str, ...] = ()
    pure: bool = False
    lease_timeout_seconds: float = 300.0


@dataclass
class AttemptRecord:
    attempt_id: str
    provider: str
    provider_kind: str = ""
    provider_node_id: str = ""
    reservation_id: str = ""
    lease_id: str = ""
    lease_epoch: int = 0
    lease_expires_at: float = 0.0
    state: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: str = ""
    result_metadata: dict = field(default_factory=dict)
    result_sha256: str = ""

    def snapshot(self) -> dict:
        duration = 0.0
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.started_at)
        return {
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "provider_kind": self.provider_kind,
            "provider_node_id": self.provider_node_id,
            "reservation_id": self.reservation_id,
            "reservation_active": (
                bool(self.reservation_id)
                and self.state not in TERMINAL_ATTEMPT_STATES
            ),
            "lease_id": self.lease_id,
            "lease_epoch": self.lease_epoch,
            "lease_expires_at": self.lease_expires_at,
            "lease_enforced": self.lease_expires_at > 0,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(duration, 6),
            "error": self.error,
            "result_metadata": dict(self.result_metadata),
            "result_sha256": self.result_sha256,
        }


@dataclass
class StageRecord:
    spec: StageSpec
    state: str = "blocked"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    attempts: list[AttemptRecord] = field(default_factory=list)
    output: Optional[dict] = None
    provider_index: int = 0
    lease_epoch: int = 0
    winner_attempt_id: str = ""
    output_digest: str = ""
    retry_count: int = 0
    result_rejection_count: int = 0
    last_result_rejection_reason: str = ""
    last_result_rejected_at: Optional[float] = None

    def provider_candidates(self) -> tuple[str, ...]:
        return (self.spec.provider, *self.spec.fallback_providers)

    def selected_provider(self) -> str:
        candidates = self.provider_candidates()
        return candidates[min(self.provider_index, len(candidates) - 1)]

    def snapshot(self) -> dict:
        output_digest = ""
        output_size = 0
        if self.output is not None:
            encoded = json.dumps(
                self.output, ensure_ascii=False, sort_keys=True, default=str,
            ).encode("utf-8")
            output_digest = hashlib.sha256(encoded).hexdigest()
            output_size = len(encoded)
        duration = 0.0
        if self.started_at is not None and self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.started_at)
        return {
            "stage_id": self.spec.stage_id,
            "stage_type": self.spec.stage_type,
            "depends_on": list(self.spec.depends_on),
            "provider": (
                self.attempts[-1].provider
                if self.attempts else self.selected_provider()
            ),
            "requested_provider": self.spec.provider,
            "fallback_providers": list(self.spec.fallback_providers),
            "selected_provider": self.selected_provider(),
            "pure": self.spec.pure,
            "lease_timeout_seconds": self.spec.lease_timeout_seconds,
            "lease_epoch": self.lease_epoch,
            "winner_attempt_id": self.winner_attempt_id,
            "retry_count": self.retry_count,
            "result_rejection_count": self.result_rejection_count,
            "last_result_rejection_reason": (
                self.last_result_rejection_reason
            ),
            "last_result_rejected_at": self.last_result_rejected_at,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(duration, 6),
            "error": self.error,
            "attempts": [attempt.snapshot() for attempt in self.attempts],
            "output_available": self.output is not None,
            "output_sha256": self.output_digest or output_digest,
            "output_size_bytes": output_size,
        }


@dataclass
class WorkflowRecord:
    workflow_id: str
    request_id: str
    template: str
    final_stage_id: str
    stages: dict[str, StageRecord]
    state: str = "created"
    last_sequence: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    result_ready_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: str = ""
    cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False,
    )
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False,
    )

    def snapshot(self) -> dict:
        with self.lock:
            duration = 0.0
            duration_end = self.finished_at or self.result_ready_at
            if self.started_at is not None and duration_end is not None:
                duration = max(0.0, duration_end - self.started_at)
            stages = [stage.snapshot() for stage in self.stages.values()]
            completed = sum(stage["state"] == "completed" for stage in stages)
            failed = sum(stage["state"] == "failed" for stage in stages)
            cancelled = sum(stage["state"] == "cancelled" for stage in stages)
            attempts = sum(len(stage["attempts"]) for stage in stages)
            retries = sum(int(stage["retry_count"]) for stage in stages)
            rejections = sum(
                int(stage["result_rejection_count"]) for stage in stages
            )
            return {
                "workflow_id": self.workflow_id,
                "request_id": self.request_id,
                "template": self.template,
                "state": self.state,
                "last_sequence": self.last_sequence,
                "final_stage_id": self.final_stage_id,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "result_ready_at": self.result_ready_at,
                "finished_at": self.finished_at,
                "duration_seconds": round(duration, 6),
                "error": self.error,
                "stage_count": len(stages),
                "completed_stage_count": completed,
                "failed_stage_count": failed,
                "cancelled_stage_count": cancelled,
                "attempt_count": attempts,
                "retry_count": retries,
                "result_rejection_count": rejections,
                "cancel_requested": (
                    self.cancel_event.is_set() and self.state != "completed"
                ),
                "stages": stages,
            }


StageExecutor = Callable[
    [StageSpec, dict[str, dict], dict, threading.Event], dict
]

WORKFLOW_ID_PATTERN = re.compile(r"^wf_[A-Za-z0-9_-]{8,96}$")


def dual_candidate_template() -> tuple[list[StageSpec], str]:
    """Return the first fixed workflow: two candidates followed by aggregation."""
    return [
        StageSpec("candidate_a", "full_inference"),
        StageSpec("candidate_b", "full_inference"),
        StageSpec(
            "aggregate",
            "aggregate",
            depends_on=("candidate_a", "candidate_b"),
        ),
    ], "aggregate"


class TaskGraphCoordinator:
    """Run bounded, in-memory workflows without sharing model-internal state."""

    def __init__(
        self,
        max_records: int = 100,
        journal: Optional[TaskJournal] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        max_parallel_stages: int = 4,
        availability_error: str = "",
    ):
        self._max_records = max(1, int(max_records))
        self._journal = journal
        self._provider_registry = provider_registry or ProviderRegistry()
        self._max_parallel_stages = max(1, min(int(max_parallel_stages), 32))
        self._stage_executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.RLock()
        self._active_runs: dict[str, int] = {}
        self._run_condition = threading.Condition(threading.RLock())
        self._closing = False
        self._journal_error = str(availability_error or "")
        self._last_recovery = {
            "recovered_workflows": 0,
            "expired_attempts": 0,
            "failed_stages": 0,
            "skipped_stages": 0,
        }
        self._last_cleanup = {
            "deleted_workflows": 0,
            "deleted_events": 0,
            "deleted_by_age": 0,
            "deleted_by_limit": 0,
            "remaining_terminal": 0,
        }
        self._workflows: dict[str, WorkflowRecord] = {}
        self._pending_cancellations: dict[str, float] = {}
        self._active_provider_attempts: dict[
            str, tuple[ProviderRegistry, str]
        ] = {}
        self._provider_activity_lock = threading.RLock()
        self._lock = threading.RLock()

    @staticmethod
    def _require_transition(
        entity_type: str,
        entity_id: str,
        current_state: str,
        next_state: str,
        transitions: dict[str, set[str]],
    ) -> None:
        if next_state not in transitions.get(current_state, set()):
            raise TaskGraphError(
                f"invalid {entity_type} transition for {entity_id}: "
                f"{current_state} -> {next_state}"
            )

    def _record_event_locked(
        self,
        workflow: WorkflowRecord,
        *,
        entity_type: str,
        entity_id: str,
        event_type: str,
        payload: dict,
        occurred_at: Optional[float] = None,
    ) -> None:
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        previous_sequence = workflow.last_sequence
        sequence = previous_sequence + 1
        event_time = time.time() if occurred_at is None else occurred_at
        workflow.last_sequence = sequence
        if self._journal is None:
            return
        event = JournalEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            workflow_id=workflow.workflow_id,
            sequence=sequence,
            entity_type=entity_type,
            entity_id=entity_id,
            event_type=event_type,
            occurred_at=event_time,
            payload=payload,
        )
        try:
            self._journal.append_event(
                event, self._durable_snapshot_locked(workflow),
            )
        except TaskJournalError as exc:
            workflow.last_sequence = previous_sequence
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    @staticmethod
    def _durable_snapshot_locked(workflow: WorkflowRecord) -> dict:
        snapshot = workflow.snapshot()
        workflow_error = str(snapshot.pop("error", "") or "")
        snapshot["error"] = ""
        snapshot["error_present"] = bool(workflow_error)
        for stage in snapshot.get("stages", []):
            stage_error = str(stage.pop("error", "") or "")
            stage["error"] = ""
            stage["error_present"] = bool(stage_error)
            for attempt in stage.get("attempts", []):
                attempt_error = str(attempt.pop("error", "") or "")
                attempt["error"] = ""
                attempt["error_present"] = bool(attempt_error)
        return snapshot

    def _transition_workflow_locked(
        self,
        workflow: WorkflowRecord,
        next_state: str,
        *,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        previous = (
            workflow.state,
            workflow.started_at,
            workflow.result_ready_at,
            workflow.finished_at,
            workflow.error,
        )
        current_state = workflow.state
        self._require_transition(
            "workflow",
            workflow.workflow_id,
            current_state,
            next_state,
            WORKFLOW_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        workflow.state = next_state
        if next_state == "running":
            workflow.started_at = changed_at
        elif next_state == "result_ready":
            workflow.result_ready_at = changed_at
        elif next_state in TERMINAL_WORKFLOW_STATES:
            workflow.finished_at = changed_at
        if error is not None:
            workflow.error = error
        elif next_state == "completed":
            workflow.error = ""
        try:
            self._record_event_locked(
                workflow,
                entity_type="workflow",
                entity_id=workflow.workflow_id,
                event_type="workflow_state_changed",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                workflow.state,
                workflow.started_at,
                workflow.result_ready_at,
                workflow.finished_at,
                workflow.error,
            ) = previous
            raise

    def _transition_stage_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
        output: Optional[dict] = None,
        set_output: bool = False,
    ) -> None:
        previous = (
            stage.state,
            stage.started_at,
            stage.finished_at,
            stage.error,
            stage.output,
        )
        current_state = stage.state
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            current_state,
            next_state,
            STAGE_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        stage.state = next_state
        if next_state == "running" and stage.started_at is None:
            stage.started_at = changed_at
        if next_state in TERMINAL_STAGE_STATES:
            stage.finished_at = changed_at
        if error is not None:
            stage.error = error
        if set_output:
            stage.output = output
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage",
                entity_id=stage.spec.stage_id,
                event_type="stage_state_changed",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "stage_type": stage.spec.stage_type,
                    "provider": stage.selected_provider(),
                    "lease_epoch": stage.lease_epoch,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                stage.state,
                stage.started_at,
                stage.finished_at,
                stage.error,
                stage.output,
            ) = previous
            raise

    def _transition_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        previous = (attempt.state, attempt.finished_at, attempt.error)
        current_state = attempt.state
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            current_state,
            next_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        attempt.state = next_state
        if next_state in TERMINAL_ATTEMPT_STATES:
            attempt.finished_at = changed_at
        if error is not None:
            attempt.error = error
        try:
            self._record_event_locked(
                workflow,
                entity_type="attempt",
                entity_id=attempt.attempt_id,
                event_type=f"attempt_{next_state}",
                occurred_at=changed_at,
                payload={
                    "from_state": current_state,
                    "to_state": next_state,
                    "stage_id": stage.spec.stage_id,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at": attempt.lease_expires_at,
                    "has_error": bool(error),
                },
            )
        except Exception:
            attempt.state, attempt.finished_at, attempt.error = previous
            raise

    def _append_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
    ) -> None:
        stage.attempts.append(attempt)
        try:
            self._record_event_locked(
                workflow,
                entity_type="attempt",
                entity_id=attempt.attempt_id,
                event_type="attempt_started",
                occurred_at=attempt.started_at,
                payload={
                    "state": attempt.state,
                    "stage_id": stage.spec.stage_id,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "lease_expires_at": attempt.lease_expires_at,
                },
            )
        except Exception:
            stage.attempts.remove(attempt)
            raise

    def _finish_stage_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        next_state: str,
        *,
        now: Optional[float] = None,
        error: Optional[str] = None,
        output: Optional[dict] = None,
        set_output: bool = False,
        result_metadata: Optional[dict] = None,
        result_digest: str = "",
        set_winner: bool = False,
    ) -> None:
        previous_stage = (
            stage.state,
            stage.started_at,
            stage.finished_at,
            stage.error,
            stage.output,
            stage.winner_attempt_id,
            stage.output_digest,
        )
        previous_attempt = (
            attempt.state,
            attempt.finished_at,
            attempt.error,
            dict(attempt.result_metadata),
            attempt.result_sha256,
        )
        stage_from_state = stage.state
        attempt_from_state = attempt.state
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            stage_from_state,
            next_state,
            STAGE_STATE_TRANSITIONS,
        )
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            attempt_from_state,
            next_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        changed_at = time.time() if now is None else now
        stage.state = next_state
        stage.finished_at = changed_at
        attempt.state = next_state
        attempt.finished_at = changed_at
        if error is not None:
            stage.error = error
            attempt.error = error
        if set_output:
            stage.output = output
        if result_metadata is not None:
            attempt.result_metadata = dict(result_metadata)
        if result_digest:
            attempt.result_sha256 = result_digest
        if set_winner:
            stage.winner_attempt_id = attempt.attempt_id
            stage.output_digest = result_digest
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_attempt",
                entity_id=f"{stage.spec.stage_id}:{attempt.attempt_id}",
                event_type=f"stage_attempt_{next_state}",
                occurred_at=changed_at,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "stage_from_state": stage_from_state,
                    "attempt_from_state": attempt_from_state,
                    "to_state": next_state,
                    "provider": attempt.provider,
                    "provider_kind": attempt.provider_kind,
                    "provider_node_id": attempt.provider_node_id,
                    "reservation_id": attempt.reservation_id,
                    "lease_id": attempt.lease_id,
                    "lease_epoch": attempt.lease_epoch,
                    "result_sha256": result_digest,
                    "winner": set_winner,
                    "has_error": bool(error),
                },
            )
        except Exception:
            (
                stage.state,
                stage.started_at,
                stage.finished_at,
                stage.error,
                stage.output,
                stage.winner_attempt_id,
                stage.output_digest,
            ) = previous_stage
            (
                attempt.state,
                attempt.finished_at,
                attempt.error,
                attempt.result_metadata,
                attempt.result_sha256,
            ) = previous_attempt
            raise

    @staticmethod
    def _output_sha256(output: dict) -> str:
        encoded = json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _reject_stage_result_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        result: ProviderStageResult,
        *,
        reason: str,
        result_digest: str,
        now: float,
    ) -> dict:
        previous = (
            stage.result_rejection_count,
            stage.last_result_rejection_reason,
            stage.last_result_rejected_at,
        )
        stage.result_rejection_count += 1
        stage.last_result_rejection_reason = reason
        stage.last_result_rejected_at = now
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_result",
                entity_id=(
                    f"{stage.spec.stage_id}:"
                    f"{result.attempt_id or 'unknown'}"
                ),
                event_type="stage_result_rejected",
                occurred_at=now,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": result.attempt_id,
                    "provider": result.provider_id,
                    "received_lease_epoch": result.lease_epoch,
                    "current_lease_epoch": stage.lease_epoch,
                    "winner_attempt_id": stage.winner_attempt_id,
                    "result_sha256": result_digest,
                    "reason": reason,
                },
            )
        except Exception:
            (
                stage.result_rejection_count,
                stage.last_result_rejection_reason,
                stage.last_result_rejected_at,
            ) = previous
            raise
        return {
            "status": "rejected",
            "reason": reason,
            "stage_id": stage.spec.stage_id,
            "attempt_id": result.attempt_id,
            "lease_epoch": result.lease_epoch,
            "winner_attempt_id": stage.winner_attempt_id,
            "output_sha256": stage.output_digest,
        }

    def _submit_stage_result_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        result: ProviderStageResult,
        *,
        now: Optional[float] = None,
    ) -> dict:
        submitted_at = time.time() if now is None else now
        if not isinstance(result.output, dict):
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="invalid_result_schema",
                result_digest="",
                now=submitted_at,
            )
        try:
            result_digest = self._output_sha256(result.output)
        except (TypeError, ValueError):
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="invalid_result_schema",
                result_digest="",
                now=submitted_at,
            )
        if stage.winner_attempt_id:
            if (
                result.attempt_id == stage.winner_attempt_id
                and result_digest == stage.output_digest
            ):
                return {
                    "status": "idempotent",
                    "reason": "already_committed",
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": result.attempt_id,
                    "lease_epoch": result.lease_epoch,
                    "winner_attempt_id": stage.winner_attempt_id,
                    "output_sha256": stage.output_digest,
                }
            reason = (
                "winner_digest_mismatch"
                if result.attempt_id == stage.winner_attempt_id
                else "winner_already_committed"
            )
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason=reason,
                result_digest=result_digest,
                now=submitted_at,
            )
        if workflow.state in TERMINAL_WORKFLOW_STATES:
            return self._reject_stage_result_locked(
                workflow,
                stage,
                result,
                reason="workflow_terminal",
                result_digest=result_digest,
                now=submitted_at,
            )
        attempt = next(
            (
                item for item in stage.attempts
                if item.attempt_id == result.attempt_id
            ),
            None,
        )
        if attempt is None:
            reason = "attempt_not_owned_by_stage"
        elif result.provider_id != attempt.provider:
            reason = "provider_identity_mismatch"
        elif result.lease_epoch != attempt.lease_epoch:
            reason = "attempt_epoch_mismatch"
        elif attempt.lease_epoch != stage.lease_epoch:
            reason = "stale_lease_epoch"
        elif attempt.state != "running":
            reason = "attempt_not_running"
        elif stage.state != "running":
            reason = "stage_not_running"
        elif (
            attempt.lease_expires_at > 0
            and submitted_at > attempt.lease_expires_at
        ):
            reason = "lease_expired"
        else:
            self._finish_stage_attempt_locked(
                workflow,
                stage,
                attempt,
                "completed",
                now=submitted_at,
                output=result.output,
                set_output=True,
                result_metadata=result.metadata,
                result_digest=result_digest,
                set_winner=True,
            )
            return {
                "status": "committed",
                "reason": "winner_committed",
                "stage_id": stage.spec.stage_id,
                "attempt_id": attempt.attempt_id,
                "lease_epoch": attempt.lease_epoch,
                "winner_attempt_id": attempt.attempt_id,
                "output_sha256": result_digest,
            }
        return self._reject_stage_result_locked(
            workflow,
            stage,
            result,
            reason=reason,
            result_digest=result_digest,
            now=submitted_at,
        )

    @staticmethod
    def _can_retry_stage(stage: StageRecord) -> bool:
        return (
            stage.spec.pure
            and not stage.winner_attempt_id
            and stage.provider_index + 1 < len(stage.provider_candidates())
        )

    def _advance_ready_stage_provider_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        *,
        error_code: str,
    ) -> None:
        if stage.state != "ready" or not self._can_retry_stage(stage):
            raise TaskGraphError("Stage is not eligible for Provider fallback")
        previous_index = stage.provider_index
        previous_retry_count = stage.retry_count
        previous_provider = stage.selected_provider()
        stage.provider_index += 1
        stage.retry_count += 1
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage",
                entity_id=stage.spec.stage_id,
                event_type="stage_reservation_retry_scheduled",
                payload={
                    "stage_id": stage.spec.stage_id,
                    "from_provider": previous_provider,
                    "to_provider": stage.selected_provider(),
                    "error_code": error_code,
                    "pure": stage.spec.pure,
                },
            )
        except Exception:
            stage.provider_index = previous_index
            stage.retry_count = previous_retry_count
            raise

    def _retire_stage_attempt_locked(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        attempt: AttemptRecord,
        *,
        attempt_state: str,
        retry: bool,
        error_code: str,
        error: str,
        now: Optional[float] = None,
    ) -> None:
        stage_next_state = "ready" if retry else "failed"
        self._require_transition(
            "stage",
            stage.spec.stage_id,
            stage.state,
            stage_next_state,
            STAGE_STATE_TRANSITIONS,
        )
        self._require_transition(
            "attempt",
            attempt.attempt_id,
            attempt.state,
            attempt_state,
            ATTEMPT_STATE_TRANSITIONS,
        )
        previous_stage = (
            stage.state,
            stage.finished_at,
            stage.error,
            stage.provider_index,
            stage.retry_count,
        )
        previous_attempt = (
            attempt.state,
            attempt.finished_at,
            attempt.error,
        )
        changed_at = time.time() if now is None else now
        previous_provider = stage.selected_provider()
        stage.state = stage_next_state
        stage.finished_at = None if retry else changed_at
        stage.error = "" if retry else error
        attempt.state = attempt_state
        attempt.finished_at = changed_at
        attempt.error = error
        if retry:
            stage.provider_index += 1
            stage.retry_count += 1
        try:
            self._record_event_locked(
                workflow,
                entity_type="stage_attempt",
                entity_id=f"{stage.spec.stage_id}:{attempt.attempt_id}",
                event_type=(
                    "stage_attempt_retry_scheduled"
                    if retry else "stage_attempt_failed"
                ),
                occurred_at=changed_at,
                payload={
                    "stage_id": stage.spec.stage_id,
                    "attempt_id": attempt.attempt_id,
                    "attempt_state": attempt_state,
                    "stage_state": stage_next_state,
                    "lease_epoch": attempt.lease_epoch,
                    "from_provider": previous_provider,
                    "to_provider": (
                        stage.selected_provider() if retry else ""
                    ),
                    "error_code": error_code,
                    "retry": retry,
                },
            )
        except Exception:
            (
                stage.state,
                stage.finished_at,
                stage.error,
                stage.provider_index,
                stage.retry_count,
            ) = previous_stage
            (
                attempt.state,
                attempt.finished_at,
                attempt.error,
            ) = previous_attempt
            raise

    def _journal_snapshot(self, workflow_id: str) -> Optional[dict]:
        if self._journal is None:
            return None
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            snapshot = self._journal.get_snapshot(workflow_id)
            return self._decorate_persisted_snapshot(snapshot)
        except TaskJournalError as exc:
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    def _journal_snapshots(self, limit: int) -> list[dict]:
        if self._journal is None:
            return []
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            return [
                cast(dict, self._decorate_persisted_snapshot(snapshot))
                for snapshot in self._journal.list_snapshots(limit=limit)
            ]
        except TaskJournalError as exc:
            self._journal_error = f"task journal unavailable: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc

    @staticmethod
    def _decorate_persisted_snapshot(
        snapshot: Optional[dict],
    ) -> Optional[dict]:
        if snapshot is None:
            return None
        decorated = dict(snapshot)
        if decorated.get("state") not in TERMINAL_WORKFLOW_STATES:
            decorated["recovery_pending"] = True
            decorated["runtime_status"] = "persisted_unrecovered"
        else:
            decorated["recovery_pending"] = False
            decorated["runtime_status"] = "terminal"
        return decorated

    def journal_status(self) -> dict:
        if self._journal is None:
            return {
                "enabled": False,
                "available": not bool(self._journal_error),
                "backend": "memory",
                "error": self._journal_error,
                "last_recovery": dict(self._last_recovery),
                "last_cleanup": dict(self._last_cleanup),
            }
        status = self._journal.health()
        if not status.get("available", False) and not self._journal_error:
            self._journal_error = (
                "task journal unavailable: "
                + str(status.get("error") or "health check failed")
            )
        if self._journal_error:
            status["available"] = False
            status["error"] = self._journal_error
        status["last_recovery"] = dict(self._last_recovery)
        status["last_cleanup"] = dict(self._last_cleanup)
        return status

    def register_provider(self, provider: ExecutionProvider) -> None:
        self._provider_registry.register(provider)

    def has_provider(self, provider_id: str) -> bool:
        return self._provider_registry.has_provider(provider_id)

    def provider_ids(self) -> list[str]:
        return self._provider_registry.provider_ids()

    def provider_status(self) -> list[dict]:
        return self._provider_registry.inspect()

    @staticmethod
    def _recovery_event_id(workflow_id: str, last_sequence: int) -> str:
        digest = hashlib.sha256(
            f"{workflow_id}:{last_sequence}:restart_recovery_v1".encode("utf-8")
        ).hexdigest()
        return f"evt_recovery_{digest}"

    @staticmethod
    def _duration_seconds(started_at: Any, finished_at: float) -> float:
        try:
            if started_at is None:
                return 0.0
            return round(max(0.0, finished_at - float(started_at)), 6)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _build_recovered_snapshot(
        cls,
        snapshot: dict,
        recovered_at: float,
    ) -> tuple[dict, dict]:
        workflow_id = str(snapshot.get("workflow_id", ""))
        if not WORKFLOW_ID_PATTERN.fullmatch(workflow_id):
            raise TaskJournalError("invalid workflow_id in recovery snapshot")
        try:
            previous_sequence = int(snapshot["last_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskJournalError(
                f"invalid recovery sequence for {workflow_id}"
            ) from exc
        if previous_sequence <= 0:
            raise TaskJournalError(
                f"invalid recovery sequence for {workflow_id}"
            )
        previous_state = str(snapshot.get("state", ""))
        if previous_state in TERMINAL_WORKFLOW_STATES:
            raise TaskJournalError(
                f"terminal workflow selected for recovery: {workflow_id}"
            )
        recovery_reason = (
            "coordinator_restarted_before_result_commit"
            if previous_state == "result_ready"
            else "coordinator_restarted_during_execution"
        )

        recovered = json.loads(json.dumps(snapshot))
        expired_attempts = 0
        failed_stages = 0
        skipped_stages = 0
        stages = recovered.get("stages", [])
        if not isinstance(stages, list):
            raise TaskJournalError(
                f"invalid stage snapshot for {workflow_id}"
            )
        for stage in stages:
            if not isinstance(stage, dict):
                raise TaskJournalError(
                    f"invalid stage snapshot for {workflow_id}"
                )
            attempts = stage.get("attempts", [])
            if not isinstance(attempts, list):
                raise TaskJournalError(
                    f"invalid attempt snapshot for {workflow_id}"
                )
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    raise TaskJournalError(
                        f"invalid attempt snapshot for {workflow_id}"
                    )
                if attempt.get("state") not in TERMINAL_ATTEMPT_STATES:
                    attempt["state"] = "expired"
                    attempt["reservation_active"] = False
                    attempt["finished_at"] = recovered_at
                    attempt["duration_seconds"] = cls._duration_seconds(
                        attempt.get("started_at"), recovered_at,
                    )
                    attempt["error"] = ""
                    attempt["error_present"] = True
                    attempt["recovery_reason"] = recovery_reason
                    attempt["error_code"] = recovery_reason
                    expired_attempts += 1

            stage_state = str(stage.get("state", ""))
            if stage_state not in TERMINAL_STAGE_STATES:
                if stage_state in {"blocked", "ready", "created"}:
                    stage["state"] = "skipped"
                    skipped_stages += 1
                else:
                    stage["state"] = "failed"
                    failed_stages += 1
                stage["finished_at"] = recovered_at
                stage["duration_seconds"] = cls._duration_seconds(
                    stage.get("started_at"), recovered_at,
                )
                stage["error"] = ""
                stage["error_present"] = True
                stage["recovery_reason"] = recovery_reason
                stage["error_code"] = recovery_reason

        recovered["state"] = "failed"
        recovered["last_sequence"] = previous_sequence + 1
        recovered["finished_at"] = recovered_at
        recovered["duration_seconds"] = cls._duration_seconds(
            recovered.get("started_at"), recovered_at,
        )
        recovered["error"] = ""
        recovered["error_present"] = True
        recovered["recovery_reason"] = recovery_reason
        recovered["error_code"] = recovery_reason
        recovered["recovered_after_restart"] = True
        recovered["recovered_at"] = recovered_at
        recovered["cancel_requested"] = False
        recovered["completed_stage_count"] = sum(
            stage.get("state") == "completed" for stage in stages
        )
        recovered["failed_stage_count"] = sum(
            stage.get("state") == "failed" for stage in stages
        )
        recovered["cancelled_stage_count"] = sum(
            stage.get("state") == "cancelled" for stage in stages
        )
        return recovered, {
            "previous_state": previous_state,
            "recovery_reason": recovery_reason,
            "expired_attempts": expired_attempts,
            "failed_stages": failed_stages,
            "skipped_stages": skipped_stages,
        }

    def recover_persisted_workflows(self, batch_size: int = 100) -> dict:
        if self._journal is None:
            return dict(self._last_recovery)
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        safe_batch_size = max(1, min(int(batch_size), 1000))
        summary = {
            "recovered_workflows": 0,
            "expired_attempts": 0,
            "failed_stages": 0,
            "skipped_stages": 0,
        }
        try:
            while True:
                candidates = self._journal.list_nonterminal_snapshots(
                    limit=safe_batch_size,
                )
                if not candidates:
                    break
                for snapshot in candidates:
                    recovered_at = time.time()
                    recovered, details = self._build_recovered_snapshot(
                        snapshot, recovered_at,
                    )
                    workflow_id = str(recovered["workflow_id"])
                    previous_sequence = int(snapshot["last_sequence"])
                    event = JournalEvent(
                        event_id=self._recovery_event_id(
                            workflow_id, previous_sequence,
                        ),
                        workflow_id=workflow_id,
                        sequence=previous_sequence + 1,
                        entity_type="workflow",
                        entity_id=workflow_id,
                        event_type="workflow_recovered_after_restart",
                        occurred_at=recovered_at,
                        payload=details,
                    )
                    inserted = self._journal.append_event(event, recovered)
                    if inserted:
                        summary["recovered_workflows"] += 1
                        summary["expired_attempts"] += int(
                            details["expired_attempts"]
                        )
                        summary["failed_stages"] += int(
                            details["failed_stages"]
                        )
                        summary["skipped_stages"] += int(
                            details["skipped_stages"]
                        )
        except TaskJournalError as exc:
            self._journal_error = f"task journal recovery failed: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc
        self._last_recovery = summary
        return dict(summary)

    def cleanup_journal(
        self,
        *,
        max_age_days: int = 0,
        max_records: int = 0,
        now: Optional[float] = None,
    ) -> dict:
        if self._journal is None:
            return dict(self._last_cleanup)
        if self._journal_error:
            raise TaskGraphUnavailable(self._journal_error)
        try:
            result = self._journal.cleanup_terminal(
                max_age_seconds=max(0, int(max_age_days)) * 86400.0,
                max_records=max(0, int(max_records)),
                now=now,
            )
        except TaskJournalError as exc:
            self._journal_error = f"task journal cleanup failed: {exc}"
            raise TaskGraphUnavailable(self._journal_error) from exc
        self._last_cleanup = dict(result)
        return dict(result)

    @property
    def availability_error(self) -> str:
        return self._journal_error

    def _finish_active_run(self, workflow_id: str) -> None:
        with self._run_condition:
            self._active_runs.pop(workflow_id, None)
            self._run_condition.notify_all()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            workflows = list(self._workflows.values())
        for workflow in workflows:
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    workflow.cancel_event.set()
                    self._cancel_active_provider_attempts_locked(workflow)
        with self._executor_lock:
            executor = self._stage_executor
            self._stage_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._run_condition:
            while self._active_runs:
                self._run_condition.wait()
        try:
            self._provider_registry.close()
        finally:
            if self._journal is not None:
                self._journal.close()

    def _cancel_active_provider_attempts_locked(
        self,
        workflow: WorkflowRecord,
    ) -> None:
        with self._provider_activity_lock:
            active_attempts = [
                (
                    attempt.attempt_id,
                    self._active_provider_attempts.get(attempt.attempt_id),
                )
                for stage in workflow.stages.values()
                for attempt in stage.attempts
                if attempt.state not in TERMINAL_ATTEMPT_STATES
            ]
        for attempt_id, active in active_attempts:
            if active is None:
                continue
            registry, provider_id = active
            try:
                registry.cancel(provider_id, attempt_id)
            except Exception:
                pass

    def _request_cancel_locked(
        self,
        workflow: WorkflowRecord,
        reason: str = "cancelled",
    ) -> None:
        if workflow.state in TERMINAL_WORKFLOW_STATES:
            return
        was_set = workflow.cancel_event.is_set()
        workflow.cancel_event.set()
        self._cancel_active_provider_attempts_locked(workflow)
        try:
            if workflow.state == "result_ready":
                self._transition_workflow_locked(
                    workflow,
                    "cancelled",
                    error=reason,
                )
            elif was_set:
                return
            else:
                self._record_event_locked(
                    workflow,
                    entity_type="workflow",
                    entity_id=workflow.workflow_id,
                    event_type="workflow_cancel_requested",
                    payload={"state": workflow.state},
                )
        except Exception:
            if not was_set:
                workflow.cancel_event.clear()
            raise

    @staticmethod
    def validate(stages: Iterable[StageSpec], final_stage_id: str) -> list[StageSpec]:
        specs = list(stages)
        if not specs:
            raise TaskGraphError("task graph must contain at least one stage")
        ids = [stage.stage_id for stage in specs]
        if any(not stage_id for stage_id in ids):
            raise TaskGraphError("stage_id must not be empty")
        if any(
            not isinstance(provider_id, str)
            or not PROVIDER_ID_PATTERN.fullmatch(provider_id)
            for spec in specs
            for provider_id in (spec.provider, *spec.fallback_providers)
        ):
            raise TaskGraphError("stage provider_id must be non-empty and safe")
        for spec in specs:
            candidates = (spec.provider, *spec.fallback_providers)
            if len(candidates) != len(set(candidates)):
                raise TaskGraphError(
                    f"stage {spec.stage_id} Provider candidates must be unique"
                )
            if len(spec.fallback_providers) > 4:
                raise TaskGraphError(
                    f"stage {spec.stage_id} has too many fallback Providers"
                )
            if not isinstance(spec.pure, bool):
                raise TaskGraphError(
                    f"stage {spec.stage_id} pure must be a bool"
                )
            try:
                lease_timeout = float(spec.lease_timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise TaskGraphError(
                    f"stage {spec.stage_id} lease timeout must be numeric"
                ) from exc
            if (
                not lease_timeout > 0
                or not lease_timeout < float("inf")
                or lease_timeout > 3600.0
            ):
                raise TaskGraphError(
                    f"stage {spec.stage_id} lease timeout must be in (0, 3600]"
                )
        if len(ids) != len(set(ids)):
            raise TaskGraphError("stage_id must be unique")
        known = set(ids)
        if final_stage_id not in known:
            raise TaskGraphError("final stage does not exist")
        for stage in specs:
            if stage.stage_id in stage.depends_on:
                raise TaskGraphError(f"stage {stage.stage_id} depends on itself")
            missing = set(stage.depends_on) - known
            if missing:
                raise TaskGraphError(
                    f"stage {stage.stage_id} has missing dependencies: "
                    f"{sorted(missing)}"
                )

        indegree = {stage.stage_id: len(stage.depends_on) for stage in specs}
        children: dict[str, list[str]] = {stage_id: [] for stage_id in ids}
        for stage in specs:
            for dependency in stage.depends_on:
                children[dependency].append(stage.stage_id)
        ready = [stage_id for stage_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            stage_id = ready.pop()
            visited += 1
            for child in children[stage_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(specs):
            raise TaskGraphError("task graph must be acyclic")
        return specs

    def _get_stage_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._closing:
                raise TaskGraphUnavailable(
                    "task graph coordinator is closing"
                )
            if self._stage_executor is None:
                self._stage_executor = ThreadPoolExecutor(
                    max_workers=self._max_parallel_stages,
                    thread_name_prefix="task-graph-stage",
                )
            return self._stage_executor

    def _select_ready_batch(
        self,
        ready: list[StageRecord],
        provider_registry: ProviderRegistry,
    ) -> list[StageRecord]:
        statuses = {
            str(item.get("provider_id", "")): item
            for item in provider_registry.inspect()
        }
        remaining = {
            provider_id: max(
                0,
                int(status.get("max_concurrency", 0))
                - int(status.get("active_reservations", 0)),
            )
            for provider_id, status in statuses.items()
        }
        scheduled_by_provider: dict[str, int] = {}
        batch: list[StageRecord] = []
        for stage in ready:
            if len(batch) >= self._max_parallel_stages:
                break
            provider_id = stage.selected_provider()
            status = statuses.get(provider_id)
            invalid_provider = (
                status is None
                or not bool(status.get("healthy", False))
                or stage.spec.stage_type not in status.get(
                    "supported_stage_types", [],
                )
            )
            if invalid_provider:
                batch.append(stage)
                scheduled_by_provider[provider_id] = (
                    scheduled_by_provider.get(provider_id, 0) + 1
                )
                continue
            if remaining.get(provider_id, 0) > 0:
                batch.append(stage)
                remaining[provider_id] -= 1
                scheduled_by_provider[provider_id] = (
                    scheduled_by_provider.get(provider_id, 0) + 1
                )
                continue
            if scheduled_by_provider.get(provider_id, 0) == 0:
                # No in-batch attempt owns the slot. Run one Stage so reserve()
                # can report the external busy/unavailable condition precisely.
                batch.append(stage)
                scheduled_by_provider[provider_id] = 1
        return batch

    def _run_ready_batch(
        self,
        workflow: WorkflowRecord,
        ready: list[StageRecord],
        root_input: dict,
        provider_registry: ProviderRegistry,
    ) -> None:
        prepared: list[tuple[StageRecord, StageRequest, Reservation]] = []
        errors: dict[str, Exception] = {}
        for stage in ready:
            try:
                request, reservation = self._prepare_stage(
                    workflow,
                    stage,
                    root_input,
                    provider_registry,
                )
                prepared.append((stage, request, reservation))
            except _StageRetryScheduled:
                continue
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                if isinstance(
                    exc,
                    (TaskGraphUnavailable, WorkflowCancelled),
                ):
                    workflow.cancel_event.set()
                    break

        if any(
            isinstance(error, (TaskGraphUnavailable, WorkflowCancelled))
            for error in errors.values()
        ):
            for _stage, _request, reservation in prepared:
                provider_registry.release(reservation.reservation_id)
            self._raise_ready_batch_error(ready, errors)

        if len(prepared) == 1:
            stage, request, reservation = prepared[0]
            try:
                self._run_stage(
                    workflow,
                    stage,
                    provider_registry,
                    request,
                    reservation,
                )
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
            self._raise_ready_batch_error(ready, errors)
            return

        try:
            executor = self._get_stage_executor() if prepared else None
        except Exception as exc:
            for _stage, _request, reservation in prepared:
                provider_registry.release(reservation.reservation_id)
            if workflow.cancel_event.is_set():
                raise WorkflowCancelled(workflow.workflow_id) from exc
            raise
        futures: dict[Future[None], StageRecord] = {}
        submitted_reservations: set[str] = set()
        for stage, request, reservation in prepared:
            try:
                future = cast(ThreadPoolExecutor, executor).submit(
                    self._run_stage,
                    workflow,
                    stage,
                    provider_registry,
                    request,
                    reservation,
                )
                futures[future] = stage
                submitted_reservations.add(reservation.reservation_id)
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                break

        for _stage, _request, reservation in prepared:
            if reservation.reservation_id not in submitted_reservations:
                provider_registry.release(reservation.reservation_id)

        for future in as_completed(futures):
            stage = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors[stage.spec.stage_id] = exc
                if isinstance(
                    exc,
                    (TaskGraphUnavailable, WorkflowCancelled),
                ):
                    workflow.cancel_event.set()

        self._raise_ready_batch_error(ready, errors)

    @staticmethod
    def _raise_ready_batch_error(
        ready: list[StageRecord],
        errors: dict[str, Exception],
    ) -> None:
        if not errors:
            return

        unavailable = next(
            (
                error for error in errors.values()
                if isinstance(error, TaskGraphUnavailable)
            ),
            None,
        )
        if unavailable is not None:
            raise unavailable
        cancelled = next(
            (
                error for error in errors.values()
                if isinstance(error, WorkflowCancelled)
            ),
            None,
        )
        if cancelled is not None:
            raise cancelled
        for stage in ready:
            error = errors.get(stage.spec.stage_id)
            if error is not None:
                raise error

    def run_template(
        self,
        template: str,
        root_input: dict,
        execute_stage: Optional[StageExecutor] = None,
        request_id: str = "",
        workflow_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[dict, dict]:
        if template != "dual_candidate":
            raise TaskGraphError(f"unsupported task graph template: {template}")
        stages, final_stage_id = dual_candidate_template()
        return self.run(
            stages=stages,
            final_stage_id=final_stage_id,
            root_input=root_input,
            execute_stage=execute_stage,
            request_id=request_id,
            template=template,
            workflow_id=workflow_id,
            cancel_event=cancel_event,
        )

    def run(
        self,
        stages: Iterable[StageSpec],
        final_stage_id: str,
        root_input: dict,
        execute_stage: Optional[StageExecutor] = None,
        request_id: str = "",
        template: str = "custom",
        workflow_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[dict, dict]:
        specs = self.validate(stages, final_stage_id)
        resolved_workflow_id = workflow_id or f"wf_{uuid.uuid4().hex}"
        if not WORKFLOW_ID_PATTERN.fullmatch(resolved_workflow_id):
            raise TaskGraphError(
                "workflow_id must start with wf_ and contain 8-96 safe characters"
            )
        workflow = WorkflowRecord(
            workflow_id=resolved_workflow_id,
            request_id=request_id,
            template=template,
            final_stage_id=final_stage_id,
            stages={spec.stage_id: StageRecord(spec=spec) for spec in specs},
            cancel_event=cancel_event or threading.Event(),
        )
        with self._lock:
            if self._closing:
                raise TaskGraphUnavailable(
                    "task graph coordinator is closing"
                )
            if self._journal_error:
                raise TaskGraphUnavailable(self._journal_error)
            self._prune_locked()
            if resolved_workflow_id in self._workflows:
                raise TaskGraphError(
                    f"workflow_id already exists: {resolved_workflow_id}"
                )
            if self._journal_snapshot(resolved_workflow_id) is not None:
                raise TaskGraphError(
                    f"workflow_id already exists: {resolved_workflow_id}"
                )
            if self._pending_cancellations.pop(resolved_workflow_id, None) is not None:
                workflow.cancel_event.set()
            self._workflows[resolved_workflow_id] = workflow
            with self._run_condition:
                self._active_runs[resolved_workflow_id] = threading.get_ident()

        try:
            with workflow.lock:
                self._transition_workflow_locked(workflow, "running")
        except TaskGraphUnavailable:
            with self._lock:
                if self._workflows.get(resolved_workflow_id) is workflow:
                    self._workflows.pop(resolved_workflow_id, None)
            self._finish_active_run(resolved_workflow_id)
            raise
        except Exception:
            self._finish_active_run(resolved_workflow_id)
            raise
        execution_registry = self._provider_registry
        owns_execution_registry = False
        if execute_stage is not None:
            execution_registry = ProviderRegistry()
            owns_execution_registry = True
            try:
                specs_by_id = {spec.stage_id: spec for spec in specs}

                def execute_callback(
                    request: StageRequest,
                    provider_cancel_event: threading.Event,
                ) -> dict:
                    return execute_stage(
                        specs_by_id[request.stage_id],
                        request.dependencies,
                        request.root_input,
                        provider_cancel_event,
                    )

                stage_types_by_provider: dict[str, list[str]] = {}
                for spec in specs:
                    for provider_id in (
                        spec.provider, *spec.fallback_providers,
                    ):
                        stage_types_by_provider.setdefault(
                            provider_id, [],
                        ).append(spec.stage_type)
                for provider_id, stage_types in stage_types_by_provider.items():
                    execution_registry.register(CallbackExecutionProvider(
                        provider_id=provider_id,
                        executor=execute_callback,
                        supported_stage_types=tuple(dict.fromkeys(stage_types)),
                    ))
            except Exception as exc:
                try:
                    execution_registry.close()
                except Exception:
                    pass
                try:
                    self._mark_failed(workflow, str(exc))
                finally:
                    self._finish_active_run(resolved_workflow_id)
                raise WorkflowExecutionError(
                    resolved_workflow_id,
                    "provider_setup",
                    str(exc),
                ) from exc
        try:
            while True:
                self._raise_if_cancelled(workflow)
                with workflow.lock:
                    if all(
                        stage.state in TERMINAL_STAGE_STATES
                        for stage in workflow.stages.values()
                    ):
                        break

                    ready: list[StageRecord] = []
                    for stage in workflow.stages.values():
                        if stage.state == "blocked":
                            dependencies = [
                                workflow.stages[dependency]
                                for dependency in stage.spec.depends_on
                            ]
                            if any(
                                dependency.state in {
                                    "failed", "skipped", "cancelled",
                                }
                                for dependency in dependencies
                            ):
                                self._transition_stage_locked(
                                    workflow, stage, "skipped",
                                )
                            elif all(
                                dependency.state == "completed"
                                for dependency in dependencies
                            ):
                                self._transition_stage_locked(
                                    workflow, stage, "ready",
                                )
                        if stage.state == "ready":
                            ready.append(stage)

                if not ready:
                    with workflow.lock:
                        unfinished = [
                            stage for stage in workflow.stages.values()
                            if stage.state not in TERMINAL_STAGE_STATES
                        ]
                    if unfinished:
                        raise TaskGraphError("task graph made no progress")
                    break

                batch = self._select_ready_batch(ready, execution_registry)
                if not batch:
                    raise TaskGraphError(
                        "no ready Stage has schedulable Provider capacity"
                    )
                self._run_ready_batch(
                    workflow,
                    batch,
                    root_input,
                    execution_registry,
                )

            with workflow.lock:
                if workflow.cancel_event.is_set():
                    raise WorkflowCancelled(workflow.workflow_id)
                final_stage = workflow.stages[final_stage_id]
                if final_stage.state != "completed" or final_stage.output is None:
                    raise TaskGraphError("final stage did not complete")
                self._transition_workflow_locked(workflow, "result_ready")
            return final_stage.output, workflow.snapshot()
        except TaskGraphUnavailable:
            raise
        except WorkflowCancelled:
            self._mark_cancelled(workflow)
            raise
        except WorkflowExecutionError as exc:
            self._mark_failed(workflow, str(exc))
            raise
        except Exception as exc:
            if workflow.cancel_event.is_set():
                self._mark_cancelled(workflow)
                raise WorkflowCancelled(workflow.workflow_id) from exc
            self._mark_failed(workflow, str(exc))
            raise WorkflowExecutionError(
                resolved_workflow_id, "graph", str(exc),
            ) from exc
        finally:
            try:
                if owns_execution_registry:
                    execution_registry.close()
            finally:
                self._finish_active_run(resolved_workflow_id)

    def _prepare_stage(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        root_input: dict,
        provider_registry: ProviderRegistry,
    ) -> tuple[StageRequest, Reservation]:
        self._raise_if_cancelled(workflow)
        with workflow.lock:
            dependencies: dict[str, dict] = {}
            for dependency in stage.spec.depends_on:
                dependency_output = workflow.stages[dependency].output
                dependencies[dependency] = cast(dict, dependency_output)
            selected_provider = stage.selected_provider()
        provider_request = StageRequest(
            workflow_id=workflow.workflow_id,
            request_id=workflow.request_id,
            stage_id=stage.spec.stage_id,
            stage_type=stage.spec.stage_type,
            provider_id=selected_provider,
            dependencies=dependencies,
            root_input=root_input,
        )
        try:
            reservation = provider_registry.reserve(provider_request)
        except ProviderError as exc:
            with workflow.lock:
                if workflow.cancel_event.is_set():
                    raise WorkflowCancelled(workflow.workflow_id) from exc
                if exc.retryable and self._can_retry_stage(stage):
                    self._advance_ready_stage_provider_locked(
                        workflow,
                        stage,
                        error_code=exc.code,
                    )
                    raise _StageRetryScheduled(
                        f"{stage.spec.stage_id} retry scheduled"
                    ) from exc
                self._transition_stage_locked(
                    workflow,
                    stage,
                    "failed",
                    error=f"{exc.code}: {exc}",
                )
            raise WorkflowExecutionError(
                workflow.workflow_id,
                stage.spec.stage_id,
                f"{exc.code}: {exc}",
            ) from exc
        with workflow.lock:
            if workflow.cancel_event.is_set():
                try:
                    provider_registry.release(reservation.reservation_id)
                finally:
                    raise WorkflowCancelled(workflow.workflow_id)
        return provider_request, reservation

    def _run_stage(
        self,
        workflow: WorkflowRecord,
        stage: StageRecord,
        provider_registry: ProviderRegistry,
        provider_request: StageRequest,
        reservation: Reservation,
    ) -> None:
        attempt: Optional[AttemptRecord] = None
        provider_attempt: Optional[ProviderStageAttempt] = None
        attempt_started = False
        try:
            with workflow.lock:
                if workflow.cancel_event.is_set():
                    raise WorkflowCancelled(workflow.workflow_id)
                started_at = time.time()
                previous_epoch = stage.lease_epoch
                lease_epoch = previous_epoch + 1
                attempt = AttemptRecord(
                    attempt_id=f"att_{uuid.uuid4().hex}",
                    provider=reservation.provider_id,
                    provider_kind=reservation.provider_kind,
                    provider_node_id=reservation.provider_node_id,
                    reservation_id=reservation.reservation_id,
                    lease_id=f"lease_{uuid.uuid4().hex}",
                    lease_epoch=lease_epoch,
                    lease_expires_at=(
                        started_at + float(stage.spec.lease_timeout_seconds)
                        if reservation.provider_kind
                        not in {"local_full_model", "callback_compatibility"}
                        else 0.0
                    ),
                    started_at=started_at,
                )
                provider_attempt = ProviderStageAttempt(
                    attempt_id=attempt.attempt_id,
                    request=provider_request,
                    provider_id=reservation.provider_id,
                    lease_id=attempt.lease_id,
                    lease_epoch=attempt.lease_epoch,
                    lease_expires_at=attempt.lease_expires_at,
                )
                stage.lease_epoch = lease_epoch
                try:
                    self._transition_stage_locked(
                        workflow, stage, "running", now=started_at,
                    )
                except Exception:
                    stage.lease_epoch = previous_epoch
                    raise
                self._append_attempt_locked(workflow, stage, attempt)
                attempt_started = True
            with self._provider_activity_lock:
                self._active_provider_attempts[attempt.attempt_id] = (
                    provider_registry,
                    reservation.provider_id,
                )
            result = provider_registry.execute(
                cast(ProviderStageAttempt, provider_attempt),
                reservation,
                workflow.cancel_event,
            )
            self._raise_if_cancelled(workflow)
            with workflow.lock:
                submission = self._submit_stage_result_locked(
                    workflow, stage, result,
                )
                if submission["status"] in {"committed", "idempotent"}:
                    return
                if stage.winner_attempt_id:
                    return
                rejection_reason = str(submission["reason"])
                retry = (
                    rejection_reason == "lease_expired"
                    and self._can_retry_stage(stage)
                )
                self._retire_stage_attempt_locked(
                    workflow,
                    stage,
                    attempt,
                    attempt_state=(
                        "expired"
                        if rejection_reason == "lease_expired"
                        else "failed"
                    ),
                    retry=retry,
                    error_code=rejection_reason,
                    error=f"{rejection_reason}: Provider result rejected",
                )
            if retry:
                return
            raise WorkflowExecutionError(
                workflow.workflow_id,
                stage.spec.stage_id,
                f"{rejection_reason}: Provider result rejected",
            )
        except TaskGraphUnavailable:
            raise
        except WorkflowCancelled:
            if attempt_started and attempt is not None:
                with workflow.lock:
                    if (
                        attempt.state == "running"
                        and stage.state == "running"
                    ):
                        self._finish_stage_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "cancelled",
                            now=time.time(),
                        )
            raise
        except WorkflowExecutionError:
            raise
        except Exception as exc:
            if workflow.cancel_event.is_set():
                if attempt_started and attempt is not None:
                    with workflow.lock:
                        if (
                            attempt.state == "running"
                            and stage.state == "running"
                        ):
                            self._finish_stage_attempt_locked(
                                workflow,
                                stage,
                                attempt,
                                "cancelled",
                                now=time.time(),
                            )
                raise WorkflowCancelled(workflow.workflow_id) from exc
            if not attempt_started or attempt is None:
                raise
            with workflow.lock:
                if (
                    attempt.state == "completed"
                    and stage.winner_attempt_id == attempt.attempt_id
                ):
                    return
                if attempt.state != "running" or stage.state != "running":
                    raise
                retry = (
                    isinstance(exc, ProviderError)
                    and exc.retryable
                    and self._can_retry_stage(stage)
                )
                error_code = (
                    exc.code
                    if isinstance(exc, ProviderError)
                    else "stage_execution_failed"
                )
                error = (
                    f"{error_code}: {exc}"
                    if isinstance(exc, ProviderError)
                    else str(exc)
                )
                self._retire_stage_attempt_locked(
                    workflow,
                    stage,
                    attempt,
                    attempt_state="expired" if retry else "failed",
                    retry=retry,
                    error_code=error_code,
                    error=error,
                )
            if retry:
                return
            raise WorkflowExecutionError(
                workflow.workflow_id, stage.spec.stage_id, error,
            ) from exc
        finally:
            if attempt is not None:
                with self._provider_activity_lock:
                    self._active_provider_attempts.pop(
                        attempt.attempt_id, None,
                    )
            active_exception = sys.exc_info()[1]
            try:
                provider_registry.release(reservation.reservation_id)
            except Exception as release_exc:
                if active_exception is None:
                    raise WorkflowExecutionError(
                        workflow.workflow_id,
                        stage.spec.stage_id,
                        f"provider reservation cleanup failed: {release_exc}",
                    ) from release_exc
                active_exception.add_note(
                    "provider reservation cleanup also failed: "
                    f"{release_exc}"
                )

    @staticmethod
    def _raise_if_cancelled(workflow: WorkflowRecord) -> None:
        if workflow.cancel_event.is_set():
            raise WorkflowCancelled(workflow.workflow_id)

    def _mark_cancelled(self, workflow: WorkflowRecord) -> None:
        now = time.time()
        with workflow.lock:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                return
            workflow.cancel_event.set()
            for stage in workflow.stages.values():
                for attempt in stage.attempts:
                    if attempt.state not in TERMINAL_ATTEMPT_STATES:
                        self._transition_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "cancelled",
                            now=now,
                        )
                if stage.state not in TERMINAL_STAGE_STATES:
                    self._transition_stage_locked(
                        workflow, stage, "cancelled", now=now,
                    )
            self._transition_workflow_locked(
                workflow, "cancelled", error="cancelled", now=now,
            )

    def _mark_failed(
        self, workflow: WorkflowRecord, error: str = "",
    ) -> None:
        now = time.time()
        with workflow.lock:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                return
            for stage in workflow.stages.values():
                for attempt in stage.attempts:
                    if attempt.state not in TERMINAL_ATTEMPT_STATES:
                        self._transition_attempt_locked(
                            workflow,
                            stage,
                            attempt,
                            "failed",
                            now=now,
                            error=error,
                        )
                if stage.state not in TERMINAL_STAGE_STATES:
                    next_state = "failed" if stage.state == "running" else "skipped"
                    self._transition_stage_locked(
                        workflow,
                        stage,
                        next_state,
                        now=now,
                        error=error if next_state == "failed" else None,
                    )
            self._transition_workflow_locked(
                workflow, "failed", error=error, now=now,
            )

    def commit_result(self, workflow_id: str) -> dict:
        """Commit a result-ready workflow without allowing terminal reversal."""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state == "completed":
                    return workflow.snapshot()
                if workflow.state == "cancelled":
                    raise WorkflowCancelled(workflow_id)
                if workflow.state != "result_ready":
                    raise TaskGraphError(
                        f"workflow {workflow_id} is not ready to commit: "
                        f"{workflow.state}"
                    )
                if workflow.cancel_event.is_set():
                    self._mark_cancelled(workflow)
                    raise WorkflowCancelled(workflow_id)
                self._transition_workflow_locked(workflow, "completed")
                return workflow.snapshot()

    def submit_stage_result(
        self,
        workflow_id: str,
        stage_id: str,
        result: ProviderStageResult,
    ) -> dict:
        """Submit one Provider result through the lease fencing gate."""
        if not isinstance(result, ProviderStageResult):
            raise TaskGraphError("result must be a StageResult")
        normalized_result = ProviderStageResult(
            output=result.output,
            provider_id=str(result.provider_id or ""),
            metadata=sanitize_result_metadata(
                result.metadata if isinstance(result.metadata, dict) else {}
            ),
            attempt_id=str(result.attempt_id or ""),
            lease_epoch=result.lease_epoch,
        )
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                stage = workflow.stages.get(stage_id)
                if stage is None:
                    raise TaskGraphError(
                        f"workflow {workflow_id} has no stage {stage_id}"
                    )
                return self._submit_stage_result_locked(
                    workflow, stage, normalized_result,
                )

    def discard_result(
        self,
        workflow_id: str,
        reason: str = "cancelled before result commit",
    ) -> dict:
        """Discard an uncommitted final result while preserving terminal states."""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state == "cancelled":
                    return workflow.snapshot()
                if workflow.state != "result_ready":
                    raise TaskGraphError(
                        f"workflow {workflow_id} result cannot be discarded from "
                        f"state {workflow.state}"
                    )
                self._request_cancel_locked(workflow, reason)
                return workflow.snapshot()

    def get(self, workflow_id: str) -> dict:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is not None:
                return workflow.snapshot()
            snapshot = self._journal_snapshot(workflow_id)
            if snapshot is None:
                raise WorkflowNotFound(workflow_id)
            return snapshot

    def list(self, limit: int = 20) -> list[dict]:
        safe_limit = max(1, min(int(limit), self._max_records))
        with self._lock:
            by_id = {
                snapshot["workflow_id"]: snapshot
                for snapshot in (
                    []
                    if self._journal_error
                    else self._journal_snapshots(safe_limit)
                )
            }
            for workflow in self._workflows.values():
                snapshot = workflow.snapshot()
                by_id[snapshot["workflow_id"]] = snapshot
            workflows = sorted(
                by_id.values(),
                key=lambda workflow: float(workflow.get("created_at", 0.0)),
                reverse=True,
            )
            return workflows[:safe_limit]

    def cancel(self, workflow_id: str) -> dict:
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise WorkflowNotFound(workflow_id)
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    self._request_cancel_locked(
                        workflow, "cancelled before result commit",
                    )
            return workflow.snapshot()

    def request_cancel(self, workflow_id: str) -> Optional[dict]:
        """Cancel a workflow or fence a valid ID that has not registered yet."""
        if not WORKFLOW_ID_PATTERN.fullmatch(workflow_id):
            raise TaskGraphError(
                "workflow_id must start with wf_ and contain 8-96 safe characters"
            )
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                snapshot = self._journal_snapshot(workflow_id)
                if snapshot is not None:
                    if snapshot.get("recovery_pending"):
                        raise TaskGraphUnavailable(
                            f"workflow {workflow_id} requires restart recovery"
                        )
                    return snapshot
                self._prune_pending_cancellations_locked()
                if len(self._pending_cancellations) >= self._max_records:
                    oldest = min(
                        self._pending_cancellations,
                        key=lambda item: self._pending_cancellations[item],
                    )
                    self._pending_cancellations.pop(oldest, None)
                self._pending_cancellations[workflow_id] = time.time()
                return None
            with workflow.lock:
                if workflow.state not in TERMINAL_WORKFLOW_STATES:
                    self._request_cancel_locked(
                        workflow, "cancelled before result commit",
                    )
            return workflow.snapshot()

    def _prune_pending_cancellations_locked(self) -> None:
        cutoff = time.time() - 300.0
        expired = [
            workflow_id
            for workflow_id, created_at in self._pending_cancellations.items()
            if created_at < cutoff
        ]
        for workflow_id in expired:
            self._pending_cancellations.pop(workflow_id, None)

    def _prune_locked(self) -> None:
        self._prune_pending_cancellations_locked()
        if len(self._workflows) < self._max_records:
            return
        ordered = sorted(
            self._workflows.values(), key=lambda workflow: workflow.created_at,
        )
        for workflow in ordered:
            if workflow.state in TERMINAL_WORKFLOW_STATES:
                self._workflows.pop(workflow.workflow_id, None)
                if len(self._workflows) < self._max_records:
                    return
        if len(self._workflows) >= self._max_records:
            raise TaskGraphError("task graph registry is full")
