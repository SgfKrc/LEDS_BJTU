"""Request-scoped lease and fencing contract for llama RPC shards."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Mapping

from cluster_control_contract import ControlContractError, QuorumCertificate, validate_certificate


@dataclass(frozen=True)
class RpcShardLease:
    shard_id: str
    worker_id: str
    model_sha256: str
    epoch: int
    attempt: int
    lease_id: str
    allocation: Mapping[str, Any]
    status: str = "active"
    issued_at: float = 0.0
    lease_expires_at: float = 0.0
    lease_ttl_seconds: float = 30.0
    control_term: int = 0
    certificate_digest: str = ""


@dataclass(frozen=True)
class LeaseDecision:
    accepted: bool
    reason: str
    lease: RpcShardLease | None = None
    result_digest: str = ""


class RpcShardLeaseBook:
    """Small in-memory coordinator used by the RPC supervisor and tests.

    A commit is accepted only for the current lease and epoch. Reassignment
    invalidates the previous lease before a new worker can commit.
    """

    def __init__(self, *, control_fence=None) -> None:
        self._current: dict[str, RpcShardLease] = {}
        self._leases: dict[str, RpcShardLease] = {}
        self._control_fence = control_fence

    def set_control_fence(self, control_fence) -> None:
        self._control_fence = control_fence

    def _admit(self, certificate, *, action: str):
        if self._control_fence is None:
            return None
        if certificate is None:
            return self._control_fence.require_current_permit(action=action)
        return self._control_fence.admit(certificate, action=action, source="lease")

    def assign(
        self,
        shard_id: str,
        worker_id: str,
        model_sha256: str,
        allocation: Mapping[str, Any],
        *,
        lease_seconds: float = 30.0,
        certificate: QuorumCertificate | Mapping[str, Any] | None = None,
    ) -> RpcShardLease:
        permit = self._admit(certificate, action="lease.assign")
        previous = self._current.get(shard_id)
        if previous and previous.status == "active":
            raise ValueError(f"shard {shard_id} already has an active lease")
        epoch = (previous.epoch + 1) if previous else 1
        attempt = (previous.attempt + 1) if previous else 1
        ttl = max(1.0, float(lease_seconds))
        now = time.time()
        lease = RpcShardLease(
            shard_id=shard_id,
            worker_id=worker_id,
            model_sha256=model_sha256,
            epoch=epoch,
            attempt=attempt,
            lease_id=uuid.uuid4().hex,
            allocation=dict(allocation),
            issued_at=now,
            lease_expires_at=now + ttl,
            lease_ttl_seconds=ttl,
            control_term=permit.term if permit else 0,
            certificate_digest=permit.certificate_digest if permit else "",
        )
        self._current[shard_id] = lease
        self._leases[lease.lease_id] = lease
        return lease

    def reassign(
        self,
        shard_id: str,
        worker_id: str,
        model_sha256: str,
        allocation: Mapping[str, Any],
        *,
        reason: str = "worker_lost",
        lease_seconds: float = 30.0,
        certificate: QuorumCertificate | Mapping[str, Any] | None = None,
    ) -> RpcShardLease:
        previous = self._current.get(shard_id)
        if previous and previous.status == "active":
            expired = replace(previous, status=reason)
            self._current[shard_id] = expired
            self._leases[expired.lease_id] = expired
        return self.assign(
            shard_id, worker_id, model_sha256, allocation,
            lease_seconds=lease_seconds,
            certificate=certificate,
        )

    def renew(self, lease_id: str, epoch: int, certificate: QuorumCertificate | Mapping[str, Any] | None = None) -> LeaseDecision:
        try:
            permit = self._admit(certificate, action="lease.renew")
        except ControlContractError as exc:
            return LeaseDecision(False, exc.code)
        lease = self._leases.get(lease_id)
        if lease is None:
            return LeaseDecision(False, "unknown_lease")
        current = self._current.get(lease.shard_id)
        if current is None or current.lease_id != lease_id:
            return LeaseDecision(False, "stale_lease", lease)
        if current.epoch != epoch:
            return LeaseDecision(False, "stale_epoch", current)
        if current.status != "active":
            return LeaseDecision(False, f"lease_{current.status}", current)
        if current.lease_expires_at <= time.time():
            expired = replace(current, status="expired")
            self._current[expired.shard_id] = expired
            self._leases[expired.lease_id] = expired
            return LeaseDecision(False, "lease_expired", expired)
        if permit and (permit.term < current.control_term or permit.certificate_digest != current.certificate_digest):
            return LeaseDecision(False, "control_certificate_stale", current)
        renewed_until = max(
            time.time() + max(1.0, current.lease_ttl_seconds),
            current.lease_expires_at + 0.001,
        )
        renewed = replace(current, lease_expires_at=renewed_until)
        self._current[renewed.shard_id] = renewed
        self._leases[renewed.lease_id] = renewed
        return LeaseDecision(True, "renewed", renewed)

    def check(self, lease_id: str, epoch: int) -> LeaseDecision:
        """Read-only fencing check for a long-lived topology transition."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return LeaseDecision(False, "unknown_lease")
        current = self._current.get(lease.shard_id)
        if current is None or current.lease_id != lease_id:
            return LeaseDecision(False, "stale_lease", current or lease)
        if current.epoch != epoch:
            return LeaseDecision(False, "stale_epoch", current)
        if current.status != "active":
            return LeaseDecision(False, f"lease_{current.status}", current)
        if current.lease_expires_at <= time.time():
            expired = replace(current, status="expired")
            self._current[expired.shard_id] = expired
            self._leases[expired.lease_id] = expired
            return LeaseDecision(False, "lease_expired", expired)
        return LeaseDecision(True, "current", current)

    def commit(
        self,
        lease_id: str,
        epoch: int,
        result: bytes | str,
        certificate: QuorumCertificate | Mapping[str, Any] | None = None,
    ) -> LeaseDecision:
        try:
            permit = self._admit(certificate, action="lease.commit")
        except ControlContractError as exc:
            return LeaseDecision(False, exc.code)
        lease = self._leases.get(lease_id)
        if lease is None:
            return LeaseDecision(False, "unknown_lease")
        current = self._current.get(lease.shard_id)
        if current is None or current.lease_id != lease_id:
            return LeaseDecision(False, "stale_lease", current or lease)
        if current.epoch != epoch:
            return LeaseDecision(False, "stale_epoch", current)
        if current.status != "active":
            return LeaseDecision(False, f"lease_{current.status}", current)
        if current.lease_expires_at <= time.time():
            expired = replace(current, status="expired")
            self._current[expired.shard_id] = expired
            self._leases[expired.lease_id] = expired
            return LeaseDecision(False, "lease_expired", expired)
        if permit and (permit.term < current.control_term or permit.certificate_digest != current.certificate_digest):
            return LeaseDecision(False, "control_certificate_stale", current)
        payload = result.encode("utf-8") if isinstance(result, str) else bytes(result)
        digest = hashlib.sha256(payload).hexdigest()
        committed = replace(current, status="committed")
        self._current[committed.shard_id] = committed
        self._leases[committed.lease_id] = committed
        return LeaseDecision(True, "committed", committed, digest)

    def snapshot(self, shard_id: str) -> RpcShardLease | None:
        return self._current.get(shard_id)
