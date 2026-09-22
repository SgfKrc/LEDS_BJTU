"""Deterministic P4.5 partition and quorum-race test harness.

The harness drives the real quorum ledger and transport test kit while keeping
network availability, time, and thread ordering under test control.  It is a
test/investigation boundary: it never changes the configured node role and it
does not elect a runtime leader.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Match ``cluster_quorum``'s source-root import contract so the quorum and
# harness modules share the exact same runtime contract classes.
from cluster_control_contract import (
    ControlContractError,
    ControlPlaneAuthority,
    QuorumCertificate,
    VoterSet,
)
from cluster_quorum import QuorumCollector, QuorumOutcome
from cluster_transport import (
    CONTROL_CHANNEL,
    DeterministicClock,
    FakeTransportLink,
    TransportContractError,
    TransportEnvelope,
)


PARTITION_SCHEMA_VERSION = "qlh.cluster.partition.v1"


class PartitionHarnessError(RuntimeError):
    """A deterministic partition scenario violated a safety invariant."""

    def __init__(self, code: str, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        self.code = str(code)
        self.evidence = dict(evidence or {})
        super().__init__(message)


def _certificate_identity(certificate: QuorumCertificate) -> tuple[int, str, str]:
    return certificate.term, certificate.leader_id, certificate.lease_id


def _outcome_view(outcome: QuorumOutcome) -> dict[str, Any]:
    return outcome.to_dict()


@dataclass
class PartitionScenarioResult:
    """Serializable result and invariant record for one scenario."""

    scenario: str
    seed: int
    events: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def accepted_certificates(self) -> list[dict[str, Any]]:
        return [
            outcome["certificate"]
            for outcome in self.outcomes
            if outcome.get("accepted") and isinstance(outcome.get("certificate"), dict)
        ]

    @property
    def safe(self) -> bool:
        return not self.violations

    def assert_safe(self) -> None:
        if self.violations:
            raise PartitionHarnessError(
                "partition_invariant_failed",
                f"{self.scenario} violated: {', '.join(self.violations)}",
                evidence=self.to_dict(),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PARTITION_SCHEMA_VERSION,
            "scenario": self.scenario,
            "seed": self.seed,
            "events": [dict(event) for event in self.events],
            "outcomes": [dict(outcome) for outcome in self.outcomes],
            "accepted_certificates": list(self.accepted_certificates),
            "violations": list(self.violations),
        }


def write_partition_evidence(
    path: str | Path,
    result: PartitionScenarioResult,
    *,
    error: BaseException | None = None,
) -> Path:
    """Atomically retain a redacted, replayable scenario report on failure."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = result.to_dict()
    if error is not None:
        document["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    temporary = target.with_name(f".{target.name}.part")
    temporary.write_text(
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


class PartitionHarness:
    """Drive quorum and delayed-packet scenarios against real test state.

    ``available_voter_ids`` is the controlled network partition.  A caller can
    run the same scenario serially or behind a thread barrier; both paths use
    the same production ``QuorumCollector`` and SQLite voter ledgers.
    """

    def __init__(
        self,
        collector: QuorumCollector,
        *,
        voter_set: VoterSet,
        clock: DeterministicClock | None = None,
        seed: int = 0,
    ) -> None:
        if collector.voter_set != voter_set:
            raise PartitionHarnessError("voter_set_mismatch", "collector and harness voter sets differ")
        if len(voter_set.voters) < 3:
            raise PartitionHarnessError("witness_required", "partition scenarios require at least three voters")
        self.collector = collector
        self.voter_set = voter_set
        self.voter_ids = tuple(sorted(voter_set.voter_map))
        self.clock = clock or DeterministicClock(epoch_ms=1_000)
        self.seed = int(seed)

    def _run_calls(
        self,
        calls: Sequence[Callable[[], QuorumOutcome]],
        *,
        parallel: bool,
        events: list[dict[str, Any]],
    ) -> list[QuorumOutcome]:
        if not parallel:
            events.append({"event": "calls_started", "mode": "serial", "count": len(calls)})
            return [call() for call in calls]

        start = threading.Barrier(len(calls) + 1)
        outcomes: list[QuorumOutcome | None] = [None] * len(calls)
        errors: list[BaseException] = []
        error_lock = threading.Lock()

        def worker(index: int, call: Callable[[], QuorumOutcome]) -> None:
            try:
                start.wait(timeout=5)
                outcomes[index] = call()
            except BaseException as exc:  # Preserve exact worker failures in evidence.
                with error_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(index, call)) for index, call in enumerate(calls)]
        for thread in threads:
            thread.start()
        events.append({"event": "barrier_released", "participants": len(calls)})
        try:
            start.wait(timeout=5)
        except threading.BrokenBarrierError as exc:
            raise PartitionHarnessError("barrier_broken", "partition race start barrier broke") from exc
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            raise PartitionHarnessError("worker_timeout", "partition race worker did not terminate")
        if errors:
            raise PartitionHarnessError(
                "worker_failed",
                f"partition race worker failed: {type(errors[0]).__name__}: {errors[0]}",
            ) from errors[0]
        if any(outcome is None for outcome in outcomes):
            raise PartitionHarnessError("worker_missing_result", "partition race worker returned no outcome")
        return [outcome for outcome in outcomes if outcome is not None]

    @staticmethod
    def _result(
        scenario: str,
        seed: int,
        outcomes: Sequence[QuorumOutcome],
        events: list[dict[str, Any]],
        *,
        max_distinct_writable: int = 1,
        check_certificate_uniqueness: bool = True,
    ) -> PartitionScenarioResult:
        result = PartitionScenarioResult(
            scenario=scenario,
            seed=seed,
            events=events,
            outcomes=[_outcome_view(outcome) for outcome in outcomes],
        )
        identities = {
            _certificate_identity(certificate)
            for certificate in (
                outcome.certificate
                for outcome in outcomes
                if outcome.accepted and outcome.certificate is not None
            )
        }
        if check_certificate_uniqueness and len(identities) > max_distinct_writable:
            result.violations.append("multiple_distinct_writable_certificates")
        return result

    def double_primary_race(
        self,
        *,
        parallel: bool,
        leader_ids: tuple[str, str] | None = None,
    ) -> PartitionScenarioResult:
        """Two candidate leaders race for one term; at most one may write."""

        leaders = leader_ids or (self.voter_ids[0], self.voter_ids[1])
        if any(leader not in self.voter_set.voter_map for leader in leaders):
            raise PartitionHarnessError("leader_invalid", "double-primary leader is not a configured voter")
        now = self.clock.now_ms
        majority_side = self.voter_ids[: self.voter_set.quorum_size]
        minority_side = self.voter_ids[self.voter_set.quorum_size :]
        events = [
            {
                "event": "partition",
                "sides": [list(majority_side), list(minority_side)],
            }
        ]
        calls = [
            lambda leader=leaders[0]: self.collector.acquire(
                leader,
                available_voter_ids=majority_side,
                now_ms=now,
                lease_id=f"double-primary-{leader}-{self.seed}",
            ),
            lambda leader=leaders[1]: self.collector.acquire(
                leader,
                available_voter_ids=minority_side,
                now_ms=now,
                lease_id=f"double-primary-{leader}-{self.seed}",
            ),
        ]
        outcomes = self._run_calls(calls, parallel=parallel, events=events)
        result = self._result("double_primary_race", self.seed, outcomes, events)
        if sum(outcome.accepted for outcome in outcomes) > 1:
            result.violations.append("two_candidates_acquired_write_certificate")
        return result

    def renewal_race(self, certificate: QuorumCertificate, *, parallel: bool) -> PartitionScenarioResult:
        """Race duplicate renewals and ensure they describe one lease identity."""

        now = self.clock.now_ms + 1
        events = [{"event": "renewal_race", "term": certificate.term, "lease_id": certificate.lease_id}]
        calls = [
            lambda: self.collector.renew(
                certificate,
                available_voter_ids=self.voter_ids,
                now_ms=now,
            ),
            lambda: self.collector.renew(
                certificate,
                available_voter_ids=self.voter_ids,
                now_ms=now,
            ),
        ]
        outcomes = self._run_calls(calls, parallel=parallel, events=events)
        result = self._result("renewal_race", self.seed, outcomes, events)
        if any(
            outcome.accepted
            and outcome.certificate is not None
            and _certificate_identity(outcome.certificate) != _certificate_identity(certificate)
            for outcome in outcomes
        ):
            result.violations.append("renewal_changed_lease_identity")
        return result

    def duplicate_term(self) -> PartitionScenarioResult:
        """Reject a second leader while the first term lease remains active."""

        now = self.clock.now_ms
        first = self.collector.acquire(
            self.voter_ids[0],
            available_voter_ids=self.voter_ids,
            now_ms=now,
            lease_id=f"duplicate-term-first-{self.seed}",
        )
        second = self.collector.acquire(
            self.voter_ids[1],
            available_voter_ids=self.voter_ids,
            now_ms=now,
            lease_id=f"duplicate-term-second-{self.seed}",
        )
        events = [{"event": "duplicate_term_attempt", "now_ms": now}]
        result = self._result("duplicate_term", self.seed, (first, second), events)
        if not first.accepted or second.accepted:
            result.violations.append("active_term_was_not_unique")
        return result

    def recovery_merge(self) -> PartitionScenarioResult:
        """Heal a minority partition and reject the old certificate after term advance."""

        initial_now = self.clock.now_ms
        first = self.collector.acquire(
            self.voter_ids[0],
            available_voter_ids=self.voter_ids,
            now_ms=initial_now,
            lease_id=f"recovery-first-{self.seed}",
        )
        if not first.accepted or first.certificate is None:
            raise PartitionHarnessError("initial_certificate_missing", "recovery scenario could not create initial certificate")

        minority = self.collector.acquire(
            self.voter_ids[2],
            available_voter_ids=(self.voter_ids[2],),
            now_ms=initial_now + 1,
            lease_id=f"recovery-minority-{self.seed}",
        )
        recovery_now = first.certificate.expires_at_ms + 1
        self.clock.advance(recovery_now - self.clock.now_ms)
        merged = self.collector.acquire(
            self.voter_ids[1],
            available_voter_ids=self.voter_ids,
            now_ms=recovery_now,
            lease_id=f"recovery-merged-{self.seed}",
        )
        events = [
            {"event": "minority_partition", "available_voters": [self.voter_ids[2]]},
            {"event": "partition_healed", "available_voters": list(self.voter_ids), "now_ms": recovery_now},
        ]
        outcomes = (first, minority, merged)
        # Two certificates are expected across the healed transition, but the
        # old one must be expired before the new one becomes writable.
        result = self._result(
            "recovery_merge", self.seed, outcomes, events,
            check_certificate_uniqueness=False,
        )
        if not merged.accepted or merged.certificate is None:
            result.violations.append("healed_partition_did_not_acquire_certificate")
        elif merged.certificate.term <= first.certificate.term:
            result.violations.append("recovery_term_did_not_advance")
        elif recovery_now < first.certificate.expires_at_ms:
            result.violations.append("writable_certificate_windows_overlap")

        authority = ControlPlaneAuthority(
            cluster_id=self.voter_set.cluster_id,
            voter_set=self.voter_set,
            static_role="master",
        )
        authority.install_certificate(first.certificate, now_ms=initial_now)
        if merged.certificate is not None:
            authority.install_certificate(merged.certificate, now_ms=recovery_now)
            try:
                authority.admit_control_write(first.certificate, now_ms=initial_now)
            except ControlContractError as exc:
                events.append({"event": "old_certificate_rejected", "code": exc.code})
                if exc.code not in {"control_certificate_stale", "control_certificate_not_current"}:
                    result.violations.append("old_certificate_rejection_code_invalid")
            else:
                result.violations.append("old_certificate_remained_writable")
        return result

    def delayed_old_packet(self) -> PartitionScenarioResult:
        """Deliver a pre-recovery packet after reconnect and require fencing."""

        clock = DeterministicClock(epoch_ms=self.clock.now_ms)
        left, right = FakeTransportLink.pair()
        left.open(generation=1, attempt_id="partition-attempt-1")
        right.open(generation=1, attempt_id="partition-attempt-1")
        payload = b"old-control-frame"
        old = TransportEnvelope.from_payload(
            payload,
            request_id=f"partition-{self.seed}",
            connection_generation=1,
            attempt_id="partition-attempt-1",
            channel=CONTROL_CHANNEL,
            sequence=0,
            deadline_ms=clock.now_ms + 10_000,
        )
        left.send(old, payload, now_ms=clock.now_ms)
        left.open(generation=2, attempt_id="partition-attempt-2")
        right.open(generation=2, attempt_id="partition-attempt-2")
        left.deliver_next()
        events = [{"event": "old_packet_delayed", "from_generation": 1, "to_generation": 2}]
        result = PartitionScenarioResult("delayed_old_packet", self.seed, events=events)
        try:
            right.receive(now_ms=clock.now_ms)
        except TransportContractError as exc:
            result.events.append({"event": "old_packet_rejected", "code": exc.code})
            if exc.code != "generation_stale":
                result.violations.append("old_packet_rejection_code_invalid")
        else:
            result.violations.append("old_packet_was_delivered_after_reconnect")
        return result


__all__ = [
    "PARTITION_SCHEMA_VERSION",
    "PartitionHarness",
    "PartitionHarnessError",
    "PartitionScenarioResult",
    "write_partition_evidence",
]
