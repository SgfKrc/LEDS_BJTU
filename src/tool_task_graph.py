"""TaskGraph boundary for user-authorized web tools.

G4 deliberately reuses the existing TaskGraph state machine.  This module
only translates a validated tool request into a bounded Stage, selects an
eligible sidecar or the host router, and projects a read-only attempt audit.
It does not start a model process and it never stores raw provider bodies.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:  # Package import for tests; top-level imports match the application's src path.
    from .task_graph import StageSpec, TaskGraphCoordinator
    from .task_graph_attempt_audit import audit_task_graph_attempts
    from .task_journal import TaskJournal
    from .task_provider import ProviderExecutionError
    from .tool_gateway import ToolGateway, ToolGatewayError, ToolGatewayPolicy, error_result
    from .tool_gateway_adapters import ToolExecutionReport, ToolGatewayExecutor
except ImportError:  # pragma: no cover - exercised by the application import layout.
    from task_graph import StageSpec, TaskGraphCoordinator
    from task_graph_attempt_audit import audit_task_graph_attempts
    from task_journal import TaskJournal
    from task_provider import ProviderExecutionError
    from tool_gateway import ToolGateway, ToolGatewayError, ToolGatewayPolicy, error_result
    from tool_gateway_adapters import ToolExecutionReport, ToolGatewayExecutor


TOOL_STAGE_TYPE = "tool_request"
TOOL_WORKFLOW_TEMPLATE = "tool_request"
HOST_ROUTER_PROVIDER_ID = "tool_host_router"
SIDECAR_ROUTER_PROVIDER_ID = "tool_sidecar_router"
_CAPABILITY_STATES = frozenset({"unknown", "declared", "verified", "rejected"})
TOOL_CONTEXT_SCHEMA = "qlh.tool_context.v1"


class ToolTaskGraphError(RuntimeError):
    """Tool-task wiring error that must not be converted into a fake result."""


class ToolTaskCancelled(ToolTaskGraphError):
    """Raised when cancellation wins before or after provider execution."""


@dataclass(frozen=True)
class ToolRoutePolicy:
    """Server-owned route policy; model requests cannot widen it."""

    sidecar_capability: str = "unknown"
    allow_host_fallback: bool = True
    sidecar_provider_id: str = SIDECAR_ROUTER_PROVIDER_ID
    host_provider_id: str = HOST_ROUTER_PROVIDER_ID

    def __post_init__(self) -> None:
        capability = str(self.sidecar_capability or "unknown").strip().lower()
        if capability not in _CAPABILITY_STATES:
            raise ValueError("sidecar_capability must be a known capability state")
        object.__setattr__(self, "sidecar_capability", capability)
        for field_name in ("sidecar_provider_id", "host_provider_id"):
            value = str(getattr(self, field_name) or "")
            if not value or len(value) > 64 or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for char in value):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class ToolRouteReport:
    """Safe routing evidence returned with a tool result."""

    result: dict[str, Any]
    route: str
    fallback_used: bool
    reason_code: str
    attempts: tuple[dict[str, Any], ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "fallback_used": self.fallback_used,
            "reason_code": self.reason_code,
            "attempts": [dict(item) for item in self.attempts],
        }


def _safe_attempts(report: ToolExecutionReport, *, route: str) -> tuple[dict[str, Any], ...]:
    attempts: list[dict[str, Any]] = []
    for item in report.attempts:
        provider = str(item.get("provider", ""))[:64]
        status = str(item.get("status", ""))[:16]
        entry: dict[str, Any] = {"provider": provider, "status": status}
        if "code" in item:
            entry["code"] = str(item.get("code", ""))[:64]
        attempts.append(entry)
    if route:
        attempts.insert(0, {"provider": route[:64], "status": "route_selected"})
    return tuple(attempts)


def _result_error(result: Mapping[str, Any]) -> tuple[str, bool]:
    if result.get("status") != "error":
        return "", False
    error = result.get("error")
    if not isinstance(error, Mapping):
        return "tool_provider_error", False
    return str(error.get("code", "tool_provider_error"))[:64], bool(error.get("retryable", False))


def build_tool_result_context(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a bounded tool-role context for a normal answer model.

    This is the host-router fallback contract: QW1.8B does not need to emit a
    tool call, it only receives the already-normalized result.  Raw response
    bodies, headers, cookies, and provider-specific fields are never copied.
    """

    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise ToolTaskGraphError("tool context inputs must be mappings")
    if result.get("status") != "ok":
        raise ToolTaskGraphError("tool context requires a successful tool result")
    tool_name = str(request.get("tool_name", ""))
    request_id = str(result.get("request_id", ""))
    if tool_name not in {"web_search", "web_fetch"} or not request_id:
        raise ToolTaskGraphError("tool context identity is invalid")
    items = result.get("items")
    citations = result.get("citations")
    if not isinstance(items, list) or not isinstance(citations, list):
        raise ToolTaskGraphError("tool context result is invalid")
    return {
        "schema": TOOL_CONTEXT_SCHEMA,
        "role": "tool",
        "name": tool_name,
        "request_id": request_id,
        "items": [dict(item) for item in items],
        "citations": [dict(item) for item in citations],
        "truncated": bool(result.get("truncated", False)),
    }


class ToolRouter:
    """Route a validated request through a verified sidecar or host gateway."""

    def __init__(
        self,
        *,
        host_executor: ToolGatewayExecutor | None,
        sidecar_executor: ToolGatewayExecutor | None = None,
        policy: ToolGatewayPolicy | None = None,
        route_policy: ToolRoutePolicy | None = None,
    ) -> None:
        self.policy = policy or ToolGatewayPolicy()
        self.route_policy = route_policy or ToolRoutePolicy()
        self.host_executor = host_executor
        self.sidecar_executor = sidecar_executor

    def _execute_executor(
        self,
        executor: ToolGatewayExecutor,
        request: Mapping[str, Any],
        *,
        allow_external: bool,
        route: str,
        fallback_used: bool,
        reason_code: str,
    ) -> ToolRouteReport:
        report = executor.execute(request, allow_external=allow_external)
        code, _retryable = _result_error(report.result)
        attempts = _safe_attempts(report, route=route)
        if report.result.get("status") == "error" and not code:
            code = "tool_provider_error"
        return ToolRouteReport(
            result=report.result,
            route=route,
            fallback_used=fallback_used,
            reason_code=code if code else reason_code,
            attempts=attempts,
        )

    def execute(
        self,
        request: Mapping[str, Any],
        *,
        allow_external: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> ToolRouteReport:
        gateway = ToolGateway(self.policy)
        prepared = gateway.prepare(request, allow_external=allow_external)
        if cancel_event is not None and cancel_event.is_set():
            raise ToolTaskCancelled("tool request cancelled before provider execution")

        sidecar_eligible = (
            self.route_policy.sidecar_capability == "verified"
            and self.sidecar_executor is not None
        )
        if sidecar_eligible:
            sidecar_report = self._execute_executor(
                self.sidecar_executor,
                prepared,
                allow_external=allow_external,
                route=self.route_policy.sidecar_provider_id,
                fallback_used=False,
                reason_code="sidecar_verified",
            )
            if cancel_event is not None and cancel_event.is_set():
                raise ToolTaskCancelled("tool request cancelled after sidecar execution")
            code, retryable = _result_error(sidecar_report.result)
            if sidecar_report.result.get("status") != "error" or not retryable:
                return sidecar_report
            if not self.route_policy.allow_host_fallback or self.host_executor is None:
                return sidecar_report
            host_report = self._execute_executor(
                self.host_executor,
                prepared,
                allow_external=allow_external,
                route=self.route_policy.host_provider_id,
                fallback_used=True,
                reason_code=f"sidecar_retryable:{code}",
            )
            return ToolRouteReport(
                result=host_report.result,
                route=host_report.route,
                fallback_used=True,
                reason_code=host_report.reason_code,
                attempts=sidecar_report.attempts + host_report.attempts,
            )

        reason = (
            "sidecar_capability_rejected"
            if self.route_policy.sidecar_capability == "rejected"
            else "sidecar_capability_not_verified"
        )
        if not self.route_policy.allow_host_fallback or self.host_executor is None:
            return ToolRouteReport(
                result=error_result(
                    prepared["request_id"],
                    ToolGatewayError("provider_unavailable", "no eligible tool route", status_code=503, retryable=True),
                ),
                route="",
                fallback_used=False,
                reason_code=reason,
                attempts=({"provider": "", "status": "rejected"},),
            )
        return self._execute_executor(
            self.host_executor,
            prepared,
            allow_external=allow_external,
            route=self.route_policy.host_provider_id,
            fallback_used=False,
            reason_code=reason,
        )


def make_tool_stage(
    *,
    stage_id: str = "tool_request",
    provider_id: str = HOST_ROUTER_PROVIDER_ID,
    depends_on: Sequence[str] = (),
    lease_timeout_seconds: float = 30.0,
) -> StageSpec:
    """Create the only admitted TaskGraph stage shape for a tool request."""

    if not isinstance(stage_id, str) or not stage_id or len(stage_id) > 96:
        raise ToolTaskGraphError("tool stage_id is invalid")
    if not isinstance(provider_id, str) or not provider_id:
        raise ToolTaskGraphError("tool provider_id is invalid")
    timeout = float(lease_timeout_seconds)
    if not 5.0 <= timeout <= 600.0:
        raise ToolTaskGraphError("tool lease timeout must be between 5 and 600 seconds")
    return StageSpec(
        stage_id=stage_id,
        stage_type=TOOL_STAGE_TYPE,
        provider=provider_id,
        depends_on=tuple(str(value) for value in depends_on),
        pure=False,
        lease_timeout_seconds=timeout,
        retry_safe=False,
    )


class ToolTaskGraphAdapter:
    """Run one Tool Gateway request through the existing TaskGraph contract."""

    def __init__(
        self,
        coordinator: TaskGraphCoordinator,
        router: ToolRouter,
        *,
        policy: ToolGatewayPolicy | None = None,
    ) -> None:
        if not isinstance(coordinator, TaskGraphCoordinator):
            raise TypeError("coordinator must be a TaskGraphCoordinator")
        self.coordinator = coordinator
        self.router = router
        self.policy = policy or router.policy

    def run(
        self,
        request: Mapping[str, Any],
        *,
        allow_external: bool = False,
        request_id: str = "",
        workflow_id: str | None = None,
        session_id: str = "",
        stage_id: str = "tool_request",
        depends_on: Sequence[str] = (),
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Validate before registering the workflow so malformed requests cannot
        # consume a Stage, lease, or journal sequence number.
        prepared = ToolGateway(self.policy).prepare(request, allow_external=allow_external)
        if depends_on:
            raise ToolTaskGraphError(
                "single-stage tool runs cannot declare dependencies; compose a TaskGraph explicitly"
            )
        stage = make_tool_stage(stage_id=stage_id, depends_on=depends_on)
        root_input = {"tool_request": prepared, "allow_external": bool(allow_external)}

        def execute_stage(_stage, _dependencies, root, stage_cancel_event):
            if stage_cancel_event.is_set():
                raise ProviderExecutionError(
                    "tool request cancelled",
                    code="tool_cancelled",
                    provider_id=self.router.route_policy.host_provider_id,
                )
            try:
                report = self.router.execute(
                    root["tool_request"],
                    allow_external=bool(root["allow_external"]),
                    cancel_event=stage_cancel_event,
                )
            except ToolTaskCancelled as exc:
                raise ProviderExecutionError(
                    str(exc), code="tool_cancelled",
                    provider_id=self.router.route_policy.host_provider_id,
                ) from exc
            except ToolGatewayError as exc:
                raise ProviderExecutionError(
                    str(exc), code=exc.code, retryable=exc.retryable,
                    provider_id=self.router.route_policy.host_provider_id,
                ) from exc
            if stage_cancel_event.is_set():
                raise ProviderExecutionError(
                    "tool request cancelled",
                    code="tool_cancelled",
                    provider_id=report.route or self.router.route_policy.host_provider_id,
                )
            if report.result.get("status") != "ok":
                code, retryable = _result_error(report.result)
                raise ProviderExecutionError(
                    f"tool provider returned {code or 'error'}",
                    code=code or "tool_provider_error",
                    retryable=retryable,
                    provider_id=report.route or self.router.route_policy.host_provider_id,
                )
            return {
                "tool_result": report.result,
                "tool_context": build_tool_result_context(root["tool_request"], report.result),
                "route": report.snapshot(),
            }

        return self.coordinator.run(
            [stage],
            stage.stage_id,
            root_input,
            execute_stage=execute_stage,
            request_id=request_id or str(prepared["request_id"]),
            session_id=session_id,
            template=TOOL_WORKFLOW_TEMPLATE,
            workflow_id=workflow_id,
            cancel_event=cancel_event,
        )

    def cancel(self, workflow_id: str) -> dict[str, Any] | None:
        return self.coordinator.request_cancel(workflow_id)

    def audit(self, workflow_id: str, *, journal: TaskJournal | None = None) -> dict[str, Any]:
        snapshot = self.coordinator.get(workflow_id)
        events = journal.list_events(workflow_id) if journal is not None else None
        return audit_task_graph_attempts(snapshot, journal_events=events)


__all__ = [
    "HOST_ROUTER_PROVIDER_ID",
    "SIDECAR_ROUTER_PROVIDER_ID",
    "TOOL_STAGE_TYPE",
    "ToolRoutePolicy",
    "ToolRouteReport",
    "ToolRouter",
    "TOOL_CONTEXT_SCHEMA",
    "ToolTaskCancelled",
    "ToolTaskGraphAdapter",
    "ToolTaskGraphError",
    "build_tool_result_context",
    "make_tool_stage",
]
