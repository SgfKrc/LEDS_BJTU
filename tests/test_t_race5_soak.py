"""Deterministic randomized soak coverage for the T-RACE-5 local gate.

The production transports are intentionally not exercised here.  This test
uses the in-memory transport contract and a temporary SQLite store so a
failure can be replayed locally from its seed without a second node.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.cluster_transport import (
    STREAM_CHANNEL,
    DeterministicClock,
    FakeTransportLink,
    TransportChunkAck,
    TransportContractError,
    TransportEnvelope,
)
from task_graph import StageSpec, TaskGraphCoordinator
from task_provider import DeterministicFakeProvider, ProviderRegistry, StageResult
from tests.helpers.task_graph_common import assert_no_active_reservations, single_stage


_SOAK_ROUNDS = (
    pytest.param(1, 2026081901, 48, id="round-1"),
    pytest.param(2, 2026082901, 48, id="round-2"),
    pytest.param(3, 2026090801, 48, id="round-3"),
)

_RECEIVE_FAILURES = {
    "attempt_fenced",
    "deadline_exceeded",
    "generation_stale",
    "payload_mismatch",
    "sequence_duplicate",
    "sequence_out_of_order",
}


def _write_failure_evidence(
    tmp_path: Path,
    *,
    scope: str,
    seed: int,
    operations: list[dict],
    state: dict,
    error: BaseException,
) -> None:
    """Persist a compact reproducer when a seed violates an invariant."""
    evidence = {
        "schema": "qlh.t-race-5.failure.v1",
        "scope": scope,
        "seed": seed,
        "operations": operations,
        "state": state,
        "error": f"{type(error).__name__}: {error}",
    }
    path = tmp_path / f"t-race-5-{scope}-seed-{seed}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pytest.fail(
        f"T-RACE-5 {scope} failure is reproducible with seed={seed}; "
        f"evidence={path}; error={type(error).__name__}: {error}",
    )


def _transport_envelope(
    *,
    payload: bytes,
    generation: int,
    attempt_id: str,
    sequence: int,
    deadline_ms: int,
) -> TransportEnvelope:
    return TransportEnvelope.from_payload(
        payload,
        request_id="t-race-5-request",
        connection_generation=generation,
        attempt_id=attempt_id,
        channel=STREAM_CHANNEL,
        sequence=sequence,
        deadline_ms=deadline_ms,
    )


def _receive_or_classify(
    left,
    right,
    *,
    now_ms: int,
    trace: list[dict],
) -> None:
    try:
        frame = right.receive(now_ms=now_ms)
    except TransportContractError as exc:
        assert exc.code in _RECEIVE_FAILURES, exc.code
        trace.append({"action": "receive_rejected", "code": exc.code})
        return

    trace.append(
        {
            "action": "receive_ok",
            "generation": frame.envelope.connection_generation,
            "sequence": frame.envelope.sequence,
        }
    )
    left.acknowledge(
        TransportChunkAck(
            frame.envelope.request_id,
            frame.envelope.connection_generation,
            frame.envelope.attempt_id,
            frame.envelope.sequence,
            len(frame.payload),
        )
    )


def _run_transport_seed(seed: int, trace: list[dict], state: dict) -> None:
    rng = random.Random(seed)
    clock = DeterministicClock(epoch_ms=10_000)
    left, right = FakeTransportLink.pair()
    generation = 1
    attempt_id = f"attempt-{generation}"
    left.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
    right.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
    sequence = 0
    outbound = 0
    inbound = 0

    def update_state() -> None:
        state.update(
            {
                "clock_ms": clock.now_ms,
                "generation": generation,
                "left": left.snapshot(),
                "right": right.snapshot(),
                "outbound": outbound,
                "inbound": inbound,
            }
        )

    update_state()

    for step in range(40):
        actions = ["send", "advance", "reconnect"]
        if outbound:
            actions.extend(["deliver", "drop", "duplicate", "tamper"])
            if outbound >= 2:
                actions.append("reorder")
        if inbound:
            actions.append("receive")
        action = rng.choice(actions)

        if action == "send":
            payload = f"seed={seed};step={step};sequence={sequence}".encode("ascii")
            deadline_ms = clock.now_ms + rng.randint(1, 120)
            left.send(
                _transport_envelope(
                    payload=payload,
                    generation=generation,
                    attempt_id=attempt_id,
                    sequence=sequence,
                    deadline_ms=deadline_ms,
                ),
                payload,
                now_ms=clock.now_ms,
            )
            trace.append(
                {
                    "action": "send",
                    "generation": generation,
                    "sequence": sequence,
                    "deadline_ms": deadline_ms,
                }
            )
            sequence += 1
            outbound += 1
        elif action == "advance":
            milliseconds = rng.randint(0, 80)
            clock.advance(milliseconds)
            trace.append({"action": "advance", "milliseconds": milliseconds})
        elif action == "reconnect":
            generation += 1
            attempt_id = f"attempt-{generation}"
            left.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
            right.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
            sequence = 0
            trace.append({"action": "reconnect", "generation": generation})
        elif action == "deliver":
            left.deliver_next()
            outbound -= 1
            inbound += 1
            trace.append({"action": "deliver"})
        elif action == "drop":
            left.drop_next()
            outbound -= 1
            trace.append({"action": "drop"})
        elif action == "duplicate":
            left.duplicate_next()
            outbound += 1
            trace.append({"action": "duplicate"})
        elif action == "tamper":
            left.tamper_next(b"tampered")
            trace.append({"action": "tamper"})
        elif action == "reorder":
            first = rng.randrange(outbound)
            second = rng.randrange(outbound - 1)
            if second >= first:
                second += 1
            left.reorder(first, second)
            trace.append({"action": "reorder", "first": first, "second": second})
        else:
            _receive_or_classify(
                left,
                right,
                now_ms=clock.now_ms,
                trace=trace,
            )
            inbound -= 1
        update_state()

    while outbound:
        left.deliver_next()
        outbound -= 1
        inbound += 1
        trace.append({"action": "drain_deliver"})
    while inbound:
        _receive_or_classify(left, right, now_ms=clock.now_ms, trace=trace)
        inbound -= 1

    # A queued frame from an older generation must never survive a reconnect.
    payload = b"stale-generation-frame"
    old_generation = generation
    old_attempt = attempt_id
    left.send(
        _transport_envelope(
            payload=payload,
            generation=old_generation,
            attempt_id=old_attempt,
            sequence=sequence,
            deadline_ms=clock.now_ms + 100,
        ),
        payload,
        now_ms=clock.now_ms,
    )
    generation += 1
    attempt_id = f"attempt-{generation}"
    left.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
    right.open(generation=generation, attempt_id=attempt_id, window_bytes=65_536)
    left.deliver_next()
    with pytest.raises(TransportContractError) as stale_error:
        right.receive(now_ms=clock.now_ms)
    assert stale_error.value.code == "generation_stale"
    trace.append({"action": "stale_generation_fenced", "generation": generation})
    update_state()


def _configure_local_store(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    import local_store

    monkeypatch.setattr(local_store, "_legacy_store_dir", lambda: str(path.parent / "legacy"))
    monkeypatch.setattr(local_store, "_sqlite_path", lambda: str(path))
    monkeypatch.setattr(local_store, "_initialized_paths", set())


def _run_store_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    trace: list[dict],
    state: dict,
) -> None:
    import local_store

    store_path = tmp_path / f"t-race-5-{seed}.sqlite3"
    _configure_local_store(monkeypatch, store_path)
    rng = random.Random(seed)
    unique_operations = []
    for index in range(24):
        session_id = f"soak-session-{rng.randrange(4)}"
        unique_operations.append(
            {
                "operation_id": f"soak-{seed}-{index}",
                "session_id": session_id,
                "user": f"question-{seed}-{index}",
                "assistant": f"answer-{seed}-{index}",
            }
        )
    calls = [entry for entry in unique_operations for _ in range(rng.randint(2, 4))]
    rng.shuffle(calls)
    trace.extend(dict(entry) for entry in calls)
    state.update(
        {
            "store_path": str(store_path),
            "calls": len(calls),
            "unique_operations": len(unique_operations),
        }
    )
    workers = 4
    assignments = [calls[offset::workers] for offset in range(workers)]
    start = threading.Barrier(workers + 1)
    observed: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    observed_lock = threading.Lock()

    def writer(entries: list[dict]) -> None:
        try:
            start.wait(timeout=5)
            for entry in entries:
                stored = local_store.save_local_conversation_turn(
                    entry["session_id"],
                    entry["user"],
                    entry["assistant"],
                    {"seed": seed},
                    operation_id=entry["operation_id"],
                )
                local_store.get_local_session(entry["session_id"])
                with observed_lock:
                    observed.append((entry["operation_id"], stored))
        except BaseException as exc:  # Preserve the exact unexpected writer failure.
            with observed_lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(entries,)) for entries in assignments]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "SQLite writer did not terminate"
    assert not errors, [f"{type(error).__name__}: {error}" for error in errors]

    expected_call_counts = Counter(entry["operation_id"] for entry in calls)
    by_operation: dict[str, list[bool]] = defaultdict(list)
    for operation_id, stored in observed:
        by_operation[operation_id].append(stored)
    assert set(by_operation) == set(expected_call_counts)
    for operation_id, call_count in expected_call_counts.items():
        outcomes = by_operation[operation_id]
        assert len(outcomes) == call_count
        assert outcomes.count(True) == 1
        assert outcomes.count(False) == call_count - 1

    turns_by_session = Counter(entry["session_id"] for entry in unique_operations)
    for session_id, turn_count in turns_by_session.items():
        assert len(local_store.load_local_conversation(session_id)) == turn_count * 2
        assert local_store.get_local_session(session_id)["message_count"] == turn_count * 2
    health = local_store.local_store_health()
    assert health["status"] == "ok"
    assert health["journal_mode"] == "wal"
    assert health["synchronous"] == "full"
    state["health"] = health


def _run_task_graph_seed(seed: int, trace: list[dict], state: dict) -> None:
    rng = random.Random(seed)
    for index in range(16):
        primary_fails = rng.choice([False, True])
        state["current_case"] = {
            "case": index,
            "primary_fails": primary_fails,
        }
        registry = ProviderRegistry()
        registry.register(
            DeterministicFakeProvider(
                "primary",
                output_factory=lambda request, cancel_event: {"content": "primary"},
                execution_failures=int(primary_fails),
            )
        )
        registry.register(
            DeterministicFakeProvider(
                "fallback",
                output_factory=lambda request, cancel_event: {"content": "fallback"},
            )
        )
        coordinator = TaskGraphCoordinator(provider_registry=registry)
        workflow_id = f"wf_soak_{seed}_{index}"
        try:
            output, workflow = coordinator.run(
                single_stage(),
                "answer",
                {"message": f"seed={seed};case={index}"},
                workflow_id=workflow_id,
            )
            stage = workflow["stages"][0]
            winner = next(
                attempt
                for attempt in stage["attempts"]
                if attempt["attempt_id"] == stage["winner_attempt_id"]
            )
            duplicate = coordinator.submit_stage_result(
                workflow_id,
                "answer",
                StageResult(
                    output=output,
                    provider_id=winner["provider"],
                    attempt_id=winner["attempt_id"],
                    lease_epoch=winner["lease_epoch"],
                ),
            )
            stale = coordinator.submit_stage_result(
                workflow_id,
                "answer",
                StageResult(
                    output={"content": "stale"},
                    provider_id=winner["provider"],
                    attempt_id="att_stale_t_race5",
                    lease_epoch=winner["lease_epoch"],
                ),
            )
            assert duplicate["status"] == "idempotent"
            assert stale["status"] == "rejected"
            assert stale["reason"] == "winner_already_committed"
            assert len(
                [attempt for attempt in stage["attempts"] if attempt["state"] == "completed"]
            ) == 1
            assert_no_active_reservations(coordinator)
            trace.append(
                {
                    "case": index,
                    "primary_fails": primary_fails,
                    "winner": winner["provider"],
                    "lease_epoch": winner["lease_epoch"],
                }
            )
        finally:
            coordinator.close()
    race_trace = _run_concurrent_task_graph_case(seed)
    trace.append(race_trace)
    state["cases"] = len(trace)


def _run_concurrent_task_graph_case(seed: int) -> dict:
    """Race two same-epoch results through the real winner gate without sleep."""
    provider_started = threading.Barrier(2)
    release_provider = threading.Event()
    registry = ProviderRegistry()
    registry.register(
        DeterministicFakeProvider(
            "primary",
            output_factory=lambda request, cancel_event: {"content": "provider"},
            start_barrier=provider_started,
            block_event=release_provider,
        )
    )
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    workflow_id = f"wf_soak_race_{seed}"
    completed: list[tuple[dict, dict]] = []
    errors: list[BaseException] = []

    def run_workflow() -> None:
        try:
            completed.append(
                coordinator.run(
                    [StageSpec("answer", "full_inference", provider="primary")],
                    "answer",
                    {"message": f"race-seed={seed}"},
                    workflow_id=workflow_id,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    runner = threading.Thread(target=run_workflow)
    runner.start()
    provider_started.wait(timeout=5)
    attempt = coordinator.get(workflow_id)["stages"][0]["attempts"][0]
    submit_started = threading.Barrier(3)
    outcomes: list[dict] = []

    def submit(content: str) -> None:
        submit_started.wait(timeout=5)
        outcomes.append(
            coordinator.submit_stage_result(
                workflow_id,
                "answer",
                StageResult(
                    output={"content": content},
                    provider_id="primary",
                    attempt_id=attempt["attempt_id"],
                    lease_epoch=attempt["lease_epoch"],
                ),
            )
        )

    submitters = [
        threading.Thread(target=submit, args=("winner-a",)),
        threading.Thread(target=submit, args=("winner-b",)),
    ]
    for submitter in submitters:
        submitter.start()
    submit_started.wait(timeout=5)
    for submitter in submitters:
        submitter.join(timeout=5)
    release_provider.set()
    runner.join(timeout=5)
    try:
        assert not any(submitter.is_alive() for submitter in submitters)
        assert not runner.is_alive()
        assert not errors, [f"{type(error).__name__}: {error}" for error in errors]
        assert len(completed) == 1
        assert sorted(outcome["status"] for outcome in outcomes) == ["committed", "rejected"]
        stage = coordinator.get(workflow_id)["stages"][0]
        assert stage["winner_attempt_id"] == attempt["attempt_id"]
        assert len(
            [candidate for candidate in stage["attempts"] if candidate["state"] == "completed"]
        ) == 1
        assert_no_active_reservations(coordinator)
        return {
            "case": "same_epoch_concurrent_winner",
            "winner_attempt_id": stage["winner_attempt_id"],
            "statuses": sorted(outcome["status"] for outcome in outcomes),
        }
    finally:
        coordinator.close()


@pytest.mark.parametrize(("round_number", "base_seed", "case_count"), _SOAK_ROUNDS)
def test_t_race5_randomized_transport_sqlite_and_task_graph_soak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    round_number: int,
    base_seed: int,
    case_count: int,
) -> None:
    """Three fixed randomized rounds must leave only classified terminal states."""
    for offset in range(case_count):
        seed = base_seed + offset
        transport_trace: list[dict] = []
        transport_state: dict = {"seed": seed}
        try:
            _run_transport_seed(seed, transport_trace, transport_state)
        except BaseException as exc:
            _write_failure_evidence(
                tmp_path,
                scope=f"transport-round-{round_number}",
                seed=seed,
                operations=locals().get("transport_trace", []),
                state=locals().get("transport_state", {}),
                error=exc,
            )
        store_trace: list[dict] = []
        store_state: dict = {"seed": seed}
        try:
            _run_store_seed(
                tmp_path,
                monkeypatch,
                seed,
                store_trace,
                store_state,
            )
        except BaseException as exc:
            _write_failure_evidence(
                tmp_path,
                scope=f"sqlite-round-{round_number}",
                seed=seed,
                operations=locals().get("store_trace", []),
                state=locals().get("store_state", {}),
                error=exc,
            )
        task_trace: list[dict] = []
        task_state: dict = {"seed": seed}
        try:
            _run_task_graph_seed(seed, task_trace, task_state)
        except BaseException as exc:
            _write_failure_evidence(
                tmp_path,
                scope=f"task-graph-round-{round_number}",
                seed=seed,
                operations=locals().get("task_trace", []),
                state=locals().get("task_state", {}),
                error=exc,
            )
