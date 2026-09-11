"""Offline, redacted timeline replay for ``TOOL-TRACE-RPL-01``.

The repository does not contain the original dual-machine log stream.  The
built-in scenarios therefore preserve the documented acceptance facts as a
small, auditable fixture.  A normalized JSON event file can be supplied for a
new replay, but raw addresses, paths and credentials are rejected before they
can reach a report.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TRACE_REPLAY_SCHEMA = "qlh.harness.trace_replay.v1"
TRACE_INPUT_SCHEMA = "qlh.trace_events.v1"
TRACE_EVENT_KINDS = frozenset(
    {
        "restart_requested",
        "worker_reregistered",
        "worker_disconnected",
        "attempt_expired",
        "stage_reassigned",
        "layer_pipeline_started",
        "ipv6_connected",
        "remote_stage_completed",
        "workflow_completed",
    }
)
_SECRET_KEYS = frozenset(
    {"token", "secret", "password", "authorization", "credential", "private_key", "api_key"}
)
_IP_RE = re.compile(r"(?<![0-9A-Fa-f:.])(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}|[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){2,})(?![0-9A-Fa-f:.])")
_ABS_PATH_RE = re.compile(r"(?:^[A-Za-z]:[\\/]|^[A-Za-z]:|^/|^\\\\)")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _workflow_ref(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("workflow_id must be a string")
    return f"wf-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]}"


def _contains_raw_secret(value: Any, *, key: str = "") -> bool:
    if key.lower() in _SECRET_KEYS:
        return True
    if isinstance(value, Mapping):
        return any(_contains_raw_secret(item, key=str(name)) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_secret(item, key=key) for item in value)
    if isinstance(value, str):
        if _ABS_PATH_RE.search(value):
            return True
        for match in _IP_RE.findall(value):
            try:
                ipaddress.ip_address(match)
            except ValueError:
                continue
            return True
    return False


def _parse_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trace event timestamp is required")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid trace event timestamp: {value!r}") from exc
    return value.strip()


def _timestamp_key(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A single sanitized event in a replay timeline."""

    event_id: str
    timestamp: str
    kind: str
    actor: str
    status: str
    summary: str
    sequence: int
    workflow_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.actor or not self.status or not self.summary:
            raise ValueError("trace event identity and summary are required")
        if self.kind not in TRACE_EVENT_KINDS:
            raise ValueError(f"unsupported trace event kind: {self.kind}")
        if self.sequence < 0:
            raise ValueError("trace event sequence must be non-negative")
        _parse_timestamp(self.timestamp)
        if _contains_raw_secret(self.as_dict(include_workflow=True)):
            raise ValueError("trace event contains a raw address, path or credential")

    def as_dict(self, *, include_workflow: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "actor": self.actor,
            "status": self.status,
            "summary": self.summary,
        }
        if include_workflow and self.workflow_ref is not None:
            value["workflow_ref"] = self.workflow_ref
        return value


@dataclass(frozen=True, slots=True)
class TraceScenario:
    """A dated, documented timeline and its expected terminal outcome."""

    scenario_id: str
    title: str
    date: str
    events: tuple[TraceEvent, ...]
    expected_outcome: str
    source_ref: str
    evidence_basis: str = "documented acceptance record; normalized fixture"

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.title or not self.expected_outcome:
            raise ValueError("trace scenario identity is required")
        if not self.source_ref or self.source_ref.startswith(("/", "\\")) or "://" in self.source_ref:
            raise ValueError("trace source_ref must be repository-relative")
        if not self.events:
            raise ValueError("trace scenario requires events")
        _parse_timestamp(f"{self.date}T00:00:00")

    @property
    def workflow_refs(self) -> tuple[str, ...]:
        return tuple(sorted({event.workflow_ref for event in self.events if event.workflow_ref}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "date": self.date,
            "expected_outcome": self.expected_outcome,
            "source_ref": self.source_ref,
            "evidence_basis": self.evidence_basis,
            "events": [event.as_dict(include_workflow=True) for event in self.events],
        }


def _event(
    event_id: str,
    timestamp: str,
    kind: str,
    actor: str,
    status: str,
    summary: str,
    sequence: int,
    workflow_id: str | None = None,
) -> TraceEvent:
    return TraceEvent(event_id, timestamp, kind, actor, status, summary, sequence, _workflow_ref(workflow_id))


def builtin_trace_scenarios() -> tuple[TraceScenario, ...]:
    """Return the documented 2026-08-20/21 acceptance timelines."""

    return (
        TraceScenario(
            "layer-pipeline-2026-08-20",
            "QW1.8B dual-machine layer pipeline",
            "2026-08-20",
            (
                _event("layers-01", "2026-08-20T15:00:00+08:00", "layer_pipeline_started", "coordinator", "accepted", "QW1.8B layers 0-21 and 21-24 split selected", 0, "layers-fixture"),
                _event("layers-02", "2026-08-20T15:00:06+08:00", "remote_stage_completed", "worker-surface", "completed", "remote layers 21-24 completed; RTT remained in the 6-12ms bucket", 1, "layers-fixture"),
                _event("layers-03", "2026-08-20T15:00:12+08:00", "workflow_completed", "coordinator", "completed", "three distributed-required runs completed without fallback", 2, "layers-fixture"),
            ),
            "completed over the dual-machine layer pipeline",
            "docs/项目速览-QLH-at-a-Glance.md#双机真机分层推理",
        ),
        TraceScenario(
            "worker-restart-recovery-2026-08-21",
            "Full Worker controlled restart and recovery",
            "2026-08-21",
            (
                _event("restart-01", "2026-08-21T09:10:00+08:00", "restart_requested", "coordinator", "accepted", "controlled worker restart requested", 0, "restart-fixture"),
                _event("restart-02", "2026-08-21T09:10:04+08:00", "worker_reregistered", "worker-surface", "ready", "worker re-registered and acknowledged hello", 1, "restart-fixture"),
                _event("restart-03", "2026-08-21T09:10:11+08:00", "workflow_completed", "coordinator", "completed", "3 stages and 3 attempts completed; distributed path retained", 2, "restart-fixture"),
            ),
            "completed after controlled restart",
            "docs/项目进展与下一步计划.md#2026-08-21",
        ),
        TraceScenario(
            "worker-disconnect-reassignment-2026-08-21",
            "Remote worker disconnect and one reassignment",
            "2026-08-21",
            (
                _event("disconnect-01", "2026-08-21T10:20:00+08:00", "worker_disconnected", "worker-surface", "disconnected", "remote worker connection closed", 0, "disconnect-fixture"),
                _event("disconnect-02", "2026-08-21T10:20:01+08:00", "attempt_expired", "coordinator", "expired", "attempt marked expired with remote disconnect reason", 1, "disconnect-fixture"),
                _event("disconnect-03", "2026-08-21T10:20:02+08:00", "stage_reassigned", "coordinator", "reassigned", "stage reassigned once to master fallback", 2, "disconnect-fixture"),
                _event("disconnect-04", "2026-08-21T10:20:08+08:00", "workflow_completed", "coordinator", "completed", "workflow completed on fallback path; reassignment count 1", 3, "disconnect-fixture"),
            ),
            "completed after one reassignment on master fallback",
            "docs/项目进展与下一步计划.md#2026-08-21",
        ),
        TraceScenario(
            "tailnet-ipv6-completion-2026-08-21",
            "Tailnet IPv6 dual-machine completion",
            "2026-08-21",
            (
                _event("ipv6-01", "2026-08-21T11:30:00+08:00", "ipv6_connected", "worker-surface", "connected", "Tailnet IPv6 endpoint accepted by the coordinator", 0, "ipv6-fixture"),
                _event("ipv6-02", "2026-08-21T11:30:05+08:00", "remote_stage_completed", "worker-surface", "completed", "remote candidate stage completed", 1, "ipv6-fixture"),
                _event("ipv6-03", "2026-08-21T11:30:10+08:00", "workflow_completed", "coordinator", "completed", "distributed workflow completed; fallback and reassignment both zero", 2, "ipv6-fixture"),
            ),
            "completed over Tailnet IPv6 without fallback",
            "docs/验收清单与资源限制登记.md#B3",
        ),
    )


def _event_from_mapping(value: Mapping[str, Any], sequence: int) -> TraceEvent:
    if _contains_raw_secret(value):
        raise ValueError("trace input contains a raw address, path or credential")
    kind = value.get("kind")
    event_id = value.get("event_id", f"event-{sequence + 1}")
    workflow_id = value.get("workflow_id", value.get("workflow_ref"))
    if isinstance(workflow_id, str) and workflow_id.startswith("wf-") and len(workflow_id) == 13:
        workflow_ref = workflow_id
    else:
        workflow_ref = _workflow_ref(workflow_id)
    return TraceEvent(
        str(event_id),
        _parse_timestamp(value.get("timestamp")),
        str(kind),
        str(value.get("actor", "unknown")),
        str(value.get("status", "observed")),
        str(value.get("summary", kind or "event")),
        int(value.get("sequence", sequence)),
        workflow_ref,
    )


def load_trace_events(path: str | Path) -> tuple[TraceEvent, ...]:
    """Load normalized events from a JSON list or ``TRACE_INPUT_SCHEMA`` object."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_events = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("events"), list):
        schema = payload.get("schema")
        if schema is not None and schema != TRACE_INPUT_SCHEMA:
            raise ValueError(f"unsupported trace input schema: {schema}")
        raw_events = payload["events"]
    else:
        raise ValueError("trace input must be a JSON event list or object with events")
    if not raw_events:
        raise ValueError("trace input requires at least one event")
    if not all(isinstance(item, Mapping) for item in raw_events):
        raise ValueError("trace input events must be objects")
    return tuple(_event_from_mapping(item, index) for index, item in enumerate(raw_events))


def _scenario_from_input(path: str | Path) -> TraceScenario:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        events = load_trace_events(path)
        return TraceScenario("custom-trace", "Custom trace replay", events[0].timestamp[:10], events, "custom input replay", "input.json")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("events"), list):
        raise ValueError("trace input must contain events")
    events = load_trace_events(path)
    source_ref = str(payload.get("source_ref", "input.json"))
    if _contains_raw_secret(source_ref):
        raise ValueError("trace source_ref contains a raw address, path or credential")
    return TraceScenario(
        str(payload.get("scenario_id", "custom-trace")),
        str(payload.get("title", "Custom trace replay")),
        str(payload.get("date", events[0].timestamp[:10])),
        events,
        str(payload.get("expected_outcome", "custom replay")),
        source_ref,
        str(payload.get("evidence_basis", "user-supplied normalized event fixture")),
    )


@dataclass(frozen=True, slots=True)
class TraceReplayReport:
    scenarios: tuple[TraceScenario, ...]
    selected_scenario_id: str | None = None
    selected_kind: str | None = None
    schema: str = TRACE_REPLAY_SCHEMA
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False

    def __post_init__(self) -> None:
        if self.schema != TRACE_REPLAY_SCHEMA or self.runner_kind not in {"fixture", "input"}:
            raise ValueError("trace replay identity is invalid")
        if not self.scenarios:
            raise ValueError("trace replay requires at least one scenario")
        if self.network_used or self.weights_loaded:
            raise ValueError("trace replay must remain offline and model-free")

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(event for scenario in self.scenarios for event in scenario.events)

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def checks(self) -> dict[str, bool]:
        # Sequence numbers are scoped to a scenario, so check them separately.
        ordered = all(
            [event.sequence for event in scenario.events] == sorted(event.sequence for event in scenario.events)
            and len({event.sequence for event in scenario.events}) == len(scenario.events)
            and len({event.event_id for event in scenario.events}) == len(scenario.events)
            and [_timestamp_key(event.timestamp) for event in scenario.events]
            == sorted(_timestamp_key(event.timestamp) for event in scenario.events)
            for scenario in self.scenarios
        )
        # A kind-filtered view is intentionally a projection and may omit the
        # terminal event; the unfiltered report still enforces terminal closure.
        terminal = self.selected_kind is not None or all(
            scenario.events[-1].kind == "workflow_completed" for scenario in self.scenarios
        )
        source_refs = all(
            not scenario.source_ref.startswith(("/", "\\"))
            and "://" not in scenario.source_ref
            and not _contains_raw_secret(scenario.source_ref)
            for scenario in self.scenarios
        )
        redacted = not _contains_raw_secret(
            {
                "scenarios": [scenario.as_dict() for scenario in self.scenarios],
                "selected_scenario_id": self.selected_scenario_id,
                "selected_kind": self.selected_kind,
            }
        )
        return {
            "events_ordered": ordered,
            "terminal_completion": terminal,
            "source_refs_relative": source_refs,
            "payloads_redacted": redacted,
            "offline_boundary": not self.network_used and not self.weights_loaded,
        }

    @property
    def valid(self) -> bool:
        return all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "selected_scenario_id": self.selected_scenario_id,
            "selected_kind": self.selected_kind,
            "scenario_count": self.scenario_count,
            "event_count": self.event_count,
            "checks": self.checks,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-TRACE-RPL-01 timeline replay report",
            "",
            f"- Valid: `{str(self.valid).lower()}`; scenarios: `{self.scenario_count}`; events: `{self.event_count}`",
            f"- Selection: scenario `{self.selected_scenario_id or 'all'}`, kind `{self.selected_kind or 'all'}`",
            f"- Runner: `{self.runner_kind}`; weights loaded: `false`; network used: `false`.",
            "- Source logs are represented by documented, normalized fixtures; raw addresses, paths and credentials are omitted.",
            "",
            "## Timeline",
            "",
            "| timestamp | scenario | kind | actor | status | summary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for scenario in self.scenarios:
            for event in scenario.events:
                lines.append(f"| `{event.timestamp}` | `{scenario.scenario_id}` | `{event.kind}` | `{event.actor}` | `{event.status}` | {event.summary} |")
        lines.extend(("", "## Checks", ""))
        for name, passed in self.checks.items():
            lines.append(f"- `{name}`: **{'passed' if passed else 'failed'}**")
        lines.extend(("", f"Report digest: `{self.digest}`", ""))
        return "\n".join(lines)


def _select_scenarios(
    scenarios: Sequence[TraceScenario],
    *,
    scenario_id: str | None,
    kind: str | None,
    date: str | None,
) -> tuple[TraceScenario, ...]:
    selected: list[TraceScenario] = []
    for scenario in scenarios:
        if scenario_id is not None and scenario.scenario_id != scenario_id:
            continue
        if date is not None and scenario.date != date:
            continue
        events = tuple(event for event in scenario.events if kind is None or event.kind == kind)
        if not events:
            continue
        selected.append(
            scenario
            if kind is None
            else TraceScenario(scenario.scenario_id, scenario.title, scenario.date, events, scenario.expected_outcome, scenario.source_ref, scenario.evidence_basis)
        )
    if not selected:
        raise ValueError("no trace scenarios match the requested filter")
    return tuple(selected)


def run_trace_replay(
    scenarios: Sequence[TraceScenario] | None = None,
    *,
    input_path: str | Path | None = None,
    scenario_id: str | None = None,
    kind: str | None = None,
    date: str | None = None,
) -> TraceReplayReport:
    """Build a deterministic replay report from built-ins or normalized JSON."""

    if kind is not None and kind not in TRACE_EVENT_KINDS:
        raise ValueError(f"unsupported trace event kind: {kind}")
    if input_path is not None and scenarios is not None:
        raise ValueError("provide scenarios or input_path, not both")
    values = (_scenario_from_input(input_path),) if input_path is not None else tuple(scenarios or builtin_trace_scenarios())
    selected = _select_scenarios(values, scenario_id=scenario_id, kind=kind, date=date)
    return TraceReplayReport(selected, scenario_id, kind, runner_kind="input" if input_path else "fixture")


def build_trace_replay_report(*args: Any, **kwargs: Any) -> TraceReplayReport:
    return run_trace_replay(*args, **kwargs)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay redacted dual-machine acceptance timelines offline")
    parser.add_argument("--input", metavar="PATH", help="normalized JSON event fixture")
    parser.add_argument("--scenario", dest="scenario_id", help="select one scenario id")
    parser.add_argument("--kind", choices=sorted(TRACE_EVENT_KINDS), help="select one event kind")
    parser.add_argument("--date", help="select one scenario date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="list built-in scenarios and exit")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def _list_scenarios() -> str:
    return "".join(f"{scenario.scenario_id}\t{scenario.title}\n" for scenario in builtin_trace_scenarios())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(_list_scenarios(), end="")
        return 0
    try:
        report = run_trace_replay(input_path=args.input, scenario_id=args.scenario_id, kind=args.kind, date=args.date)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    output_count = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        output_count += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        output_count += 1
    if not output_count:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "TRACE_EVENT_KINDS",
    "TRACE_INPUT_SCHEMA",
    "TRACE_REPLAY_SCHEMA",
    "TraceEvent",
    "TraceReplayReport",
    "TraceScenario",
    "build_parser",
    "build_trace_replay_report",
    "builtin_trace_scenarios",
    "load_trace_events",
    "main",
    "run_trace_replay",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
