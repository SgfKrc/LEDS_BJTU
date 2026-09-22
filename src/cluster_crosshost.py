"""Cross-process HA development gate for P4.5.

This module deliberately stops at a local process boundary.  It is useful for
Windows development because it exercises serialization, worker restart, and
generation fencing without pretending that a loopback pipe is a physical
Surface/y700 link.  Production networking and authentication remain owned by
the transport/TCP layers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import time
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Mapping, Sequence

from cluster_transport import TransportContractError, TransportEnvelope


CROSSHOST_SCHEMA_VERSION = "qlh.cluster.crosshost.v1"
_DEFAULT_TIMEOUT_SECONDS = 5.0


class CrossHostError(RuntimeError):
    """Raised when the local cross-process gate cannot complete safely."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True)
class CrossHostEvidence:
    """Metadata-only result suitable for a local HA development report."""

    scenario: str
    physical_nodes: bool
    process_count: int
    transport: str
    events: tuple[dict[str, Any], ...]
    term_history: tuple[int, ...] = ()
    write_rejections: tuple[dict[str, Any], ...] = ()
    task_outcomes: tuple[dict[str, Any], ...] = ()
    resource_state: Mapping[str, Any] = field(default_factory=dict)
    rto_ms: int = 0
    rpo_last_durable_sequence: int = 0
    rpo_lost_events: int = 0
    violations: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return not self.violations

    def with_control_metadata(
        self,
        *,
        term_history: Sequence[int] = (),
        write_rejections: Sequence[Mapping[str, Any]] = (),
        task_outcomes: Sequence[Mapping[str, Any]] = (),
        rpo_last_durable_sequence: int = 0,
        rpo_lost_events: int = 0,
    ) -> "CrossHostEvidence":
        """Attach metadata from the real quorum/fence/journal components."""

        resolved_terms = tuple(int(value) for value in term_history)
        return replace(
            self,
            term_history=resolved_terms or self.term_history,
            write_rejections=self.write_rejections + tuple(dict(value) for value in write_rejections),
            task_outcomes=self.task_outcomes + tuple(dict(value) for value in task_outcomes),
            rpo_last_durable_sequence=(
                int(rpo_last_durable_sequence)
                if rpo_last_durable_sequence
                else self.rpo_last_durable_sequence
            ),
            rpo_lost_events=(
                int(rpo_lost_events)
                if rpo_lost_events
                else self.rpo_lost_events
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CROSSHOST_SCHEMA_VERSION,
            "scenario": self.scenario,
            "physical_nodes": self.physical_nodes,
            "process_count": self.process_count,
            "transport": self.transport,
            "events": [dict(event) for event in self.events],
            "term_history": list(self.term_history),
            "write_rejections": [dict(item) for item in self.write_rejections],
            "task_outcomes": [dict(item) for item in self.task_outcomes],
            "resource_state": dict(self.resource_state or {}),
            "rto_ms": self.rto_ms,
            "rpo": {
                "last_durable_sequence": self.rpo_last_durable_sequence,
                "lost_events": self.rpo_lost_events,
            },
            "violations": list(self.violations),
            "safe": self.safe,
        }


def _decode_payload(value: Any) -> bytes:
    if not isinstance(value, str):
        raise CrossHostError("payload_invalid", "worker payload must be base64 text")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CrossHostError("payload_invalid", "worker payload is not valid base64") from exc


def _encode_payload(value: bytes) -> str:
    return base64.b64encode(bytes(value)).decode("ascii")


def _worker_command(
    connection: Connection,
    node_id: str,
    state: dict[str, Any],
    command: Mapping[str, Any],
) -> bool:
    """Handle one command and return whether the worker should continue."""

    operation = str(command.get("op", ""))
    if operation == "stop":
        connection.send({"ok": True, "event": "stopped", "node_id": node_id})
        return False
    if operation == "set_generation":
        generation = int(command["generation"])
        attempt_id = str(command["attempt_id"])
        if generation < 0 or not attempt_id:
            raise CrossHostError("generation_invalid", "worker generation state is invalid")
        state["generation"] = generation
        state["attempt_id"] = attempt_id
        state["received"].clear()
        connection.send({
            "ok": True,
            "event": "generation_set",
            "node_id": node_id,
            "generation": generation,
        })
        return True
    generation = int(state["generation"])
    attempt_id = str(state["attempt_id"])
    received: dict[str, int] = state["received"]
    if operation == "build":
        payload = _decode_payload(command.get("payload_b64"))
        envelope = TransportEnvelope.from_payload(
            payload,
            request_id=str(command["request_id"]),
            connection_generation=generation,
            attempt_id=attempt_id,
            channel=str(command["channel"]),
            sequence=int(command["sequence"]),
            deadline_ms=int(command["deadline_ms"]),
        )
        connection.send({
            "ok": True,
            "event": "built",
            "node_id": node_id,
            "envelope": envelope.to_dict(),
            "payload_b64": _encode_payload(payload),
        })
        return True
    if operation == "receive":
        envelope = TransportEnvelope.decode(
            json.dumps(command["envelope"], ensure_ascii=True, sort_keys=True)
        )
        payload = _decode_payload(command.get("payload_b64"))
        now_ms = int(command["now_ms"])
        if envelope.connection_generation != generation:
            raise TransportContractError(
                "generation_stale", "received frame belongs to an old generation"
            )
        if envelope.attempt_id != attempt_id:
            raise TransportContractError(
                "attempt_fenced", "received frame belongs to an old attempt"
            )
        if envelope.is_expired(now_ms=now_ms):
            raise TransportContractError(
                "deadline_exceeded", "received frame deadline has expired"
            )
        if (
            envelope.payload_size != len(payload)
            or envelope.payload_digest != hashlib.sha256(payload).hexdigest()
        ):
            raise TransportContractError(
                "payload_mismatch", "received payload does not match envelope"
            )
        previous = received.get(envelope.channel, -1)
        if envelope.sequence <= previous:
            raise TransportContractError(
                "sequence_duplicate", "received sequence was already delivered"
            )
        if envelope.sequence != previous + 1:
            raise TransportContractError(
                "sequence_out_of_order", "received sequence is not contiguous"
            )
        received[envelope.channel] = envelope.sequence
        connection.send({
            "ok": True,
            "event": "accepted",
            "node_id": node_id,
            "generation": generation,
            "sequence": envelope.sequence,
        })
        return True
    if operation == "snapshot":
        connection.send({
            "ok": True,
            "node_id": node_id,
            "generation": generation,
            "attempt_id": attempt_id,
            "received": dict(received),
        })
        return True
    raise CrossHostError("worker_operation_invalid", "unsupported worker operation")


def _worker_main(connection: Connection, node_id: str) -> None:
    """Run one deliberately small process-boundary transport endpoint."""

    state: dict[str, Any] = {"generation": 0, "attempt_id": "", "received": {}}
    while True:
        try:
            command = connection.recv()
            if not isinstance(command, Mapping):
                raise CrossHostError("worker_command_invalid", "worker command must be an object")
            if not _worker_command(connection, node_id, state, command):
                return
        except EOFError:
            return
        except (TransportContractError, CrossHostError) as exc:
            try:
                connection.send({"ok": False, "code": exc.code})
            except (BrokenPipeError, EOFError, OSError):
                return
        except Exception as exc:  # pragma: no cover - only reports an unexpected worker fault
            try:
                connection.send({"ok": False, "code": getattr(exc, "code", "worker_error")})
            except (BrokenPipeError, EOFError, OSError):
                return


class CrossHostProcessHarness:
    """Run a two-worker restart/reconnect gate with real envelope validation."""

    def __init__(
        self,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        context: str = "spawn",
    ) -> None:
        if timeout_seconds <= 0:
            raise CrossHostError("timeout_invalid", "timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        try:
            self._context = multiprocessing.get_context(context)
        except ValueError as exc:
            raise CrossHostError("process_context_invalid", "unsupported multiprocessing context") from exc
        self._workers: dict[str, tuple[multiprocessing.Process, Connection]] = {}

    def _start_worker(self, node_id: str) -> None:
        if node_id in self._workers:
            raise CrossHostError("worker_exists", "worker is already running")
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(target=_worker_main, args=(child, node_id), daemon=True)
        process.start()
        child.close()
        self._workers[node_id] = (process, parent)

    def _request(self, node_id: str, command: Mapping[str, Any]) -> dict[str, Any]:
        worker = self._workers.get(node_id)
        if worker is None:
            raise CrossHostError("worker_missing", f"worker {node_id} is not running")
        process, connection = worker
        if not process.is_alive():
            raise CrossHostError("worker_exited", f"worker {node_id} exited unexpectedly")
        try:
            connection.send(dict(command))
            if not connection.poll(self.timeout_seconds):
                raise CrossHostError("worker_timeout", f"worker {node_id} did not respond")
            response = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise CrossHostError("worker_unreachable", f"worker {node_id} is unreachable") from exc
        if not isinstance(response, dict):
            raise CrossHostError("worker_response_invalid", "worker response must be an object")
        return response

    def _expect_ok(self, node_id: str, command: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(node_id, command)
        if not response.get("ok"):
            raise CrossHostError(str(response.get("code", "worker_error")), "worker rejected the command")
        return response

    def _stop_worker(self, node_id: str, *, force: bool = False) -> None:
        worker = self._workers.pop(node_id, None)
        if worker is None:
            return
        process, connection = worker
        try:
            if process.is_alive() and not force:
                self._request_existing(connection, {"op": "stop"})
        except (CrossHostError, BrokenPipeError, EOFError, OSError):
            pass
        finally:
            connection.close()
            process.join(timeout=self.timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.timeout_seconds)

    def _request_existing(self, connection: Connection, command: Mapping[str, Any]) -> dict[str, Any]:
        connection.send(dict(command))
        if not connection.poll(self.timeout_seconds):
            raise CrossHostError("worker_timeout", "worker did not stop in time")
        response = connection.recv()
        if not isinstance(response, dict):
            raise CrossHostError("worker_response_invalid", "worker response must be an object")
        return response

    def run(
        self,
        *,
        term_history: Sequence[int] = (),
        write_rejections: Sequence[Mapping[str, Any]] = (),
        task_outcomes: Sequence[Mapping[str, Any]] = (),
        rpo_last_durable_sequence: int = 0,
        rpo_lost_events: int = 0,
    ) -> CrossHostEvidence:
        events: list[dict[str, Any]] = []
        all_rejections = [dict(item) for item in write_rejections]
        now_ms = int(time.time() * 1000)
        payload = b"crosshost-control-metadata"
        try:
            self._start_worker("node-a")
            self._start_worker("node-b")
            for node_id in ("node-a", "node-b"):
                self._expect_ok(
                    node_id,
                    {"op": "set_generation", "generation": 1, "attempt_id": "attempt-1"},
                )
            first = self._expect_ok(
                "node-a",
                {
                    "op": "build",
                    "payload_b64": _encode_payload(payload),
                    "request_id": "crosshost-accepted",
                    "channel": "control",
                    "sequence": 0,
                    "deadline_ms": now_ms + 10_000,
                },
            )
            accepted = self._expect_ok(
                "node-b",
                {
                    "op": "receive",
                    "envelope": first["envelope"],
                    "payload_b64": first["payload_b64"],
                    "now_ms": now_ms,
                },
            )
            events.extend([
                {"event": "workers_started", "process_count": 2},
                {"event": "control_frame_accepted", "generation": accepted["generation"]},
            ])

            old_frame = self._expect_ok(
                "node-a",
                {
                    "op": "build",
                    "payload_b64": _encode_payload(payload),
                    "request_id": "crosshost-stale",
                    "channel": "control",
                    "sequence": 0,
                    "deadline_ms": now_ms + 10_000,
                },
            )
            failure_started = time.perf_counter()
            self._stop_worker("node-b")
            events.append({"event": "receiver_stopped", "reason": "simulated_crash"})
            self._start_worker("node-b")
            self._expect_ok(
                "node-b",
                {"op": "set_generation", "generation": 2, "attempt_id": "attempt-2"},
            )
            self._expect_ok(
                "node-a",
                {"op": "set_generation", "generation": 2, "attempt_id": "attempt-2"},
            )
            stale = self._request(
                "node-b",
                {
                    "op": "receive",
                    "envelope": old_frame["envelope"],
                    "payload_b64": old_frame["payload_b64"],
                    "now_ms": int(time.time() * 1000),
                },
            )
            if stale.get("ok") or stale.get("code") != "generation_stale":
                raise CrossHostError("stale_frame_accepted", "old generation was accepted after restart")
            all_rejections.append({"source": "transport", "code": "generation_stale"})
            events.append({"event": "old_generation_rejected", "code": "generation_stale"})

            second = self._expect_ok(
                "node-a",
                {
                    "op": "build",
                    "payload_b64": _encode_payload(payload),
                    "request_id": "crosshost-reconnected",
                    "channel": "control",
                    "sequence": 0,
                    "deadline_ms": int(time.time() * 1000) + 10_000,
                },
            )
            reconnected = self._expect_ok(
                "node-b",
                {
                    "op": "receive",
                    "envelope": second["envelope"],
                    "payload_b64": second["payload_b64"],
                    "now_ms": int(time.time() * 1000),
                },
            )
            rto_ms = max(0, int((time.perf_counter() - failure_started) * 1000))
            events.extend([
                {"event": "receiver_restarted", "generation": 2},
                {"event": "control_frame_accepted_after_reconnect", "generation": reconnected["generation"]},
                {"event": "rto_measured", "rto_ms": rto_ms},
            ])
            return CrossHostEvidence(
                scenario="local_dual_process_reconnect",
                physical_nodes=False,
                process_count=2,
                transport="multiprocessing_pipe",
                events=tuple(events),
                term_history=tuple(int(value) for value in term_history),
                write_rejections=tuple(all_rejections),
                task_outcomes=tuple(dict(value) for value in task_outcomes),
                resource_state={
                    "worker_processes": 2,
                    "active_workers": 2,
                    "receiver_restart_count": 1,
                },
                rto_ms=rto_ms,
                rpo_last_durable_sequence=int(rpo_last_durable_sequence),
                rpo_lost_events=int(rpo_lost_events),
            )
        finally:
            self._stop_worker("node-a")
            self._stop_worker("node-b")


def write_crosshost_evidence(path: str | Path, evidence: CrossHostEvidence) -> Path:
    """Write one atomic, metadata-only cross-process report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    partial.write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(target)
    return target


__all__ = [
    "CROSSHOST_SCHEMA_VERSION",
    "CrossHostError",
    "CrossHostEvidence",
    "CrossHostProcessHarness",
    "write_crosshost_evidence",
]
