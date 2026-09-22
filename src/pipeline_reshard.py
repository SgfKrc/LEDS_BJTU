"""Fail-closed recovery planning for a contiguous layer pipeline.

The execution engines own loading and inference.  This module owns only the
control-plane transition: re-solve the surviving topology, prove the required
artifacts are available, and fence the old topology before publishing a new
one.  Keeping that state separate prevents a disconnect from publishing an
unloadable assignment to the data plane.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

try:  # Support both ``src.*`` tests and the scheduler's source-root imports.
    from .llama_rpc_contract import LeaseDecision, RpcShardLease, RpcShardLeaseBook
    from .pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
    from .pipeline_node_contract import (
        PipelineArtifactRef,
        PipelineLayout,
        PipelineNode,
        PipelineNodeContractError,
        pipeline_layout_from_capacity_plan,
        validate_pipeline_nodes,
    )
except ImportError:  # pragma: no cover - exercised by the application entrypoint.
    from llama_rpc_contract import LeaseDecision, RpcShardLease, RpcShardLeaseBook
    from pipeline_capacity import PipelineCapacityError, solve_pipeline_capacity
    from pipeline_node_contract import (
        PipelineArtifactRef,
        PipelineLayout,
        PipelineNode,
        PipelineNodeContractError,
        pipeline_layout_from_capacity_plan,
        validate_pipeline_nodes,
    )


RESHARD_SCHEMA_VERSION = 1


class PipelineReshardError(ValueError):
    """A reshard request cannot be planned or committed safely."""


def _range(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise PipelineReshardError(f"{field} must be a two-item range")
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise PipelineReshardError(f"{field} must contain integers") from exc
    if start < 0 or end <= start:
        raise PipelineReshardError(f"{field} must be a non-empty half-open range")
    return start, end


def _identity(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise PipelineReshardError(f"{field} is required")
    return result


@dataclass(frozen=True)
class PipelineArtifactAvailability:
    """A verified artifact range currently available to one pipeline node."""

    node_id: str
    model_sha256: str
    artifact_kind: str
    layer_range: tuple[int, int]
    artifact_sha256: str = ""
    verified: bool = True
    has_embedding: bool = False
    has_lm_head: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identity(self.node_id, "node_id"))
        object.__setattr__(self, "model_sha256", _identity(
            self.model_sha256, "model_sha256",
        ))
        object.__setattr__(self, "artifact_kind", _identity(
            self.artifact_kind, "artifact_kind",
        ))
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "artifact_sha256", str(self.artifact_sha256 or "").strip())
        object.__setattr__(self, "verified", bool(self.verified))
        object.__setattr__(self, "has_embedding", bool(self.has_embedding))
        object.__setattr__(self, "has_lm_head", bool(self.has_lm_head))

    @classmethod
    def from_node(cls, node: PipelineNode) -> "PipelineArtifactAvailability":
        return cls(
            node_id=node.node_id,
            model_sha256=node.artifact.model_sha256,
            artifact_kind=node.artifact.artifact_kind,
            layer_range=node.artifact.source_layer_range or node.layer_range,
            artifact_sha256=node.artifact.artifact_sha256,
            verified=True,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
        )

    def covers(self, requirement: "PipelineArtifactRequirement") -> bool:
        start, end = self.layer_range
        required_start, required_end = requirement.layer_range
        if not self.verified:
            return False
        return (
            self.node_id == requirement.node_id
            and self.model_sha256.casefold() == requirement.model_sha256.casefold()
            and self.artifact_kind == requirement.artifact_kind
            and start <= required_start
            and end >= required_end
            and (not requirement.has_embedding or self.has_embedding)
            and (not requirement.has_lm_head or self.has_lm_head)
            and (
                not requirement.artifact_sha256
                or (
                    bool(self.artifact_sha256)
                    and self.artifact_sha256.casefold()
                    == requirement.artifact_sha256.casefold()
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_sha256": self.model_sha256,
            "artifact_kind": self.artifact_kind,
            "layer_range": list(self.layer_range),
            "artifact_sha256": self.artifact_sha256,
            "verified": self.verified,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
        }


@dataclass(frozen=True)
class PipelineArtifactRequirement:
    """Artifact coverage a candidate node must prove before activation."""

    node_id: str
    model_sha256: str
    artifact_kind: str
    layer_range: tuple[int, int]
    artifact_sha256: str = ""
    has_embedding: bool = False
    has_lm_head: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identity(self.node_id, "node_id"))
        object.__setattr__(self, "model_sha256", _identity(
            self.model_sha256, "model_sha256",
        ))
        object.__setattr__(self, "artifact_kind", _identity(
            self.artifact_kind, "artifact_kind",
        ))
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "artifact_sha256", str(self.artifact_sha256 or "").strip())
        object.__setattr__(self, "has_embedding", bool(self.has_embedding))
        object.__setattr__(self, "has_lm_head", bool(self.has_lm_head))

    @classmethod
    def from_node(cls, node: PipelineNode) -> "PipelineArtifactRequirement":
        artifact: PipelineArtifactRef = node.artifact
        return cls(
            node_id=node.node_id,
            model_sha256=artifact.model_sha256,
            artifact_kind=artifact.artifact_kind,
            layer_range=node.layer_range,
            artifact_sha256=artifact.artifact_sha256,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "model_sha256": self.model_sha256,
            "artifact_kind": self.artifact_kind,
            "layer_range": list(self.layer_range),
            "artifact_sha256": self.artifact_sha256,
            "has_embedding": self.has_embedding,
            "has_lm_head": self.has_lm_head,
        }


@dataclass(frozen=True)
class PipelineReshardMove:
    """One exact layer interval whose execution owner changes."""

    layer_range: tuple[int, int]
    source_node_id: str
    target_node_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_range", _range(self.layer_range, "layer_range"))
        object.__setattr__(self, "source_node_id", _identity(
            self.source_node_id, "source_node_id",
        ))
        object.__setattr__(self, "target_node_id", _identity(
            self.target_node_id, "target_node_id",
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_range": list(self.layer_range),
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
        }


@dataclass(frozen=True)
class PipelineReshardPlan:
    """A fully validated but not necessarily asset-ready topology change."""

    plan_id: str
    base_epoch: int
    base_contract_sha256: str
    failed_node_ids: tuple[str, ...]
    capacity_plan: Mapping[str, Any]
    candidate_layout: PipelineLayout
    moves: tuple[PipelineReshardMove, ...]
    requirements: tuple[PipelineArtifactRequirement, ...]
    missing_requirements: tuple[PipelineArtifactRequirement, ...]

    @property
    def assets_ready(self) -> bool:
        return not self.missing_requirements

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "base_epoch": self.base_epoch,
            "base_contract_sha256": self.base_contract_sha256,
            "failed_node_ids": list(self.failed_node_ids),
            "assets_ready": self.assets_ready,
            "capacity_plan_id": str(self.capacity_plan.get("plan_id", "") or ""),
            "capacity_reason_code": str(self.capacity_plan.get("reason_code", "") or ""),
            "candidate_layout": self.candidate_layout.to_dict(),
            "moves": [move.to_dict() for move in self.moves],
            "requirements": [item.to_dict() for item in self.requirements],
            "missing_requirements": [
                item.to_dict() for item in self.missing_requirements
            ],
        }


@dataclass(frozen=True)
class PipelineReshardDecision:
    accepted: bool
    reason_code: str
    plan: PipelineReshardPlan | None = None
    reason: str = ""

    @property
    def ready_to_commit(self) -> bool:
        return bool(self.accepted and self.plan is not None and self.plan.assets_ready)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "status": "ready" if self.ready_to_commit else (
                "staged" if self.accepted else "rejected"
            ),
            "accepted": self.accepted,
            "ready_to_commit": self.ready_to_commit,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "plan": self.plan.to_dict() if self.plan else None,
        }


@dataclass(frozen=True)
class PipelineReshardCommitDecision:
    accepted: bool
    reason_code: str
    epoch: int
    layout: PipelineLayout
    lease: RpcShardLease | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "status": "committed" if self.accepted else "rejected",
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "epoch": self.epoch,
            "layout": self.layout.to_dict(),
            "lease": (
                {
                    "lease_id": self.lease.lease_id,
                    "epoch": self.lease.epoch,
                    "attempt": self.lease.attempt,
                    "status": self.lease.status,
                }
                if self.lease is not None else None
            ),
        }


def _node_for_layer(layout: PipelineLayout, layer: int) -> PipelineNode:
    for node in layout.nodes:
        start, end = node.layer_range
        if start <= layer < end:
            return node
    raise PipelineReshardError(f"layout has no owner for layer {layer}")


def _moves(current: PipelineLayout, candidate: PipelineLayout) -> tuple[PipelineReshardMove, ...]:
    boundaries = sorted({
        0,
        current.total_layers,
        *[boundary for node in current.nodes for boundary in node.layer_range],
        *[boundary for node in candidate.nodes for boundary in node.layer_range],
    })
    values: list[PipelineReshardMove] = []
    for start, end in zip(boundaries, boundaries[1:]):
        before = _node_for_layer(current, start).node_id
        after = _node_for_layer(candidate, start).node_id
        if before == after:
            continue
        if (
            values
            and values[-1].source_node_id == before
            and values[-1].target_node_id == after
            and values[-1].layer_range[1] == start
        ):
            previous = values.pop()
            values.append(PipelineReshardMove(
                (previous.layer_range[0], end), before, after,
            ))
        else:
            values.append(PipelineReshardMove((start, end), before, after))
    return tuple(values)


def _candidate_metadata(
    current: PipelineLayout,
    supplied: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result = {str(node_id): dict(value) for node_id, value in (supplied or {}).items()}
    for node in current.nodes:
        metadata = result.setdefault(node.node_id, {})
        metadata.setdefault("is_local", not node.federated and node.location == "local")
        metadata.setdefault("location", node.location)
        metadata.setdefault("kind", node.kind)
        metadata.setdefault("federated", node.federated)
        metadata.setdefault("cross_engine", node.cross_engine)
        metadata.setdefault("engine", node.engine)
        metadata.setdefault("artifact_kind", node.artifact.artifact_kind)
        metadata.setdefault("architecture", node.artifact.architecture or current.architecture)
    return result


def _availability_values(
    current: PipelineLayout,
    failed_node_ids: set[str],
    supplied: Iterable[PipelineArtifactAvailability],
) -> tuple[PipelineArtifactAvailability, ...]:
    values = [
        PipelineArtifactAvailability.from_node(node)
        for node in current.nodes
        if node.node_id not in failed_node_ids
    ]
    for item in supplied:
        if not isinstance(item, PipelineArtifactAvailability):
            raise PipelineReshardError(
                "artifact availability must be PipelineArtifactAvailability",
            )
        if item.node_id not in failed_node_ids:
            values.append(item)
    return tuple(values)


def _missing_requirements(
    requirements: tuple[PipelineArtifactRequirement, ...],
    availability: Iterable[PipelineArtifactAvailability],
) -> tuple[PipelineArtifactRequirement, ...]:
    available = tuple(availability)
    return tuple(
        requirement for requirement in requirements
        if not any(item.covers(requirement) for item in available)
    )


def plan_pipeline_reshard(
    current_layout: PipelineLayout,
    *,
    base_epoch: int,
    failed_node_ids: Iterable[str],
    descriptor: Mapping[str, Any],
    capacity_nodes: Iterable[Mapping[str, Any]],
    node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_availability: Iterable[PipelineArtifactAvailability] = (),
    safety_margin: float = 1.2,
) -> PipelineReshardDecision:
    """Re-solve a failed topology without publishing it to an execution engine.

    Capacity records must describe the survivor's recovery budget, including
    the temporary residency required while a new layer artifact is loaded.
    A distributed layout stays distributed; a two-node loss is deliberately
    rejected here so the pre-existing local fallback remains a distinct path.
    """
    try:
        if not isinstance(current_layout, PipelineLayout):
            raise PipelineReshardError("current_layout must be PipelineLayout")
        validated = validate_pipeline_nodes(
            current_layout.nodes, total_layers=current_layout.total_layers,
        )
        epoch = int(base_epoch)
        if epoch < 1:
            raise PipelineReshardError("base_epoch must be positive")
        failed = tuple(sorted({
            _identity(value, "failed_node_id") for value in failed_node_ids
        }))
        if not failed:
            raise PipelineReshardError("at least one failed node is required")
        known_ids = {node.node_id for node in validated.nodes}
        unknown = sorted(set(failed) - known_ids)
        if unknown:
            raise PipelineReshardError(
                "failed node is not in the active layout: " + ", ".join(unknown),
            )
        if not isinstance(descriptor, Mapping):
            raise PipelineReshardError("descriptor must be a mapping")
        descriptor_sha = str(descriptor.get("model_sha256", "") or "").strip()
        if descriptor_sha.casefold() != validated.model_sha256.casefold():
            return PipelineReshardDecision(
                False,
                "pipeline_reshard_model_identity_mismatch",
                reason="descriptor model_sha256 does not match the active layout",
            )
        survivors = [
            dict(item) for item in capacity_nodes
            if isinstance(item, Mapping)
            and str(item.get("node_id", "") or "").strip() not in set(failed)
        ]
        if len({str(item.get("node_id", "") or "").strip() for item in survivors}) < 2:
            return PipelineReshardDecision(
                False,
                "pipeline_reshard_survivors_unavailable",
                reason="fewer than two survivor capacity records are available",
            )
        capacity_plan = solve_pipeline_capacity(
            dict(descriptor), survivors, safety_margin=safety_margin,
            require_distributed=True,
        )
        if not capacity_plan.get("admitted"):
            return PipelineReshardDecision(
                False,
                str(capacity_plan.get(
                    "reason_code", "pipeline_reshard_capacity_insufficient",
                )),
                reason=str(capacity_plan.get("reason", "")),
            )
        candidate = pipeline_layout_from_capacity_plan(
            capacity_plan,
            node_metadata=_candidate_metadata(validated, node_metadata),
        )
        if set(failed) & {node.node_id for node in candidate.nodes}:
            raise PipelineReshardError("candidate layout retained a failed node")
        requirements = tuple(
            PipelineArtifactRequirement.from_node(node) for node in candidate.nodes
        )
        available = _availability_values(
            validated, set(failed), artifact_availability,
        )
        missing = _missing_requirements(requirements, available)
        moves = _moves(validated, candidate)
        plan_identity = {
            "schema_version": RESHARD_SCHEMA_VERSION,
            "base_epoch": epoch,
            "base_contract_sha256": validated.contract_sha256,
            "failed_node_ids": failed,
            "candidate_contract_sha256": candidate.contract_sha256,
            "requirements": [item.to_dict() for item in requirements],
        }
        plan_id = hashlib.sha256(json.dumps(
            plan_identity, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        plan = PipelineReshardPlan(
            plan_id=plan_id,
            base_epoch=epoch,
            base_contract_sha256=validated.contract_sha256,
            failed_node_ids=failed,
            capacity_plan=capacity_plan,
            candidate_layout=candidate,
            moves=moves,
            requirements=requirements,
            missing_requirements=missing,
        )
        return PipelineReshardDecision(
            True,
            "" if plan.assets_ready else "pipeline_reshard_assets_required",
            plan=plan,
        )
    except (PipelineNodeContractError, PipelineCapacityError, TypeError, ValueError) as exc:
        return PipelineReshardDecision(
            False, "pipeline_reshard_input_invalid", reason=str(exc),
        )


class PipelineReshardCoordinator:
    """Stage and atomically activate recovery layouts using RPC epoch fencing."""

    _TOPOLOGY_SHARD_ID = "pipeline-topology"
    _COORDINATOR_ID = "pipeline-reshard-coordinator"

    def __init__(
        self,
        layout: PipelineLayout,
        *,
        lease_book: RpcShardLeaseBook | None = None,
        artifact_availability: Iterable[PipelineArtifactAvailability] = (),
        control_certificate: Mapping[str, Any] | None = None,
    ) -> None:
        self._layout = validate_pipeline_nodes(layout.nodes, total_layers=layout.total_layers)
        self._lock = threading.RLock()
        self._lease_book = lease_book or RpcShardLeaseBook()
        self._control_certificate = control_certificate
        if self._lease_book._control_fence is not None and control_certificate is None:
            control_certificate = self._lease_book._control_fence.current_certificate()
            self._control_certificate = control_certificate
        self._topology_lease = self._lease_book.assign(
            self._TOPOLOGY_SHARD_ID,
            self._COORDINATOR_ID,
            self._layout.model_sha256,
            {"contract_sha256": self._layout.contract_sha256},
            lease_seconds=86400.0,
            certificate=control_certificate,
        )
        self._availability = list(_availability_values(
            self._layout, set(), artifact_availability,
        ))
        self._staged: dict[str, PipelineReshardPlan] = {}

    @property
    def layout(self) -> PipelineLayout:
        with self._lock:
            return self._layout

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._topology_lease.epoch

    @property
    def topology_lease(self) -> RpcShardLease:
        with self._lock:
            return self._topology_lease

    def fence(self, epoch: int) -> LeaseDecision:
        """Check whether a caller still owns the active topology epoch."""
        with self._lock:
            return self._lease_book.check(self._topology_lease.lease_id, int(epoch))

    def stage_failure(
        self,
        failed_node_ids: Iterable[str],
        *,
        descriptor: Mapping[str, Any],
        capacity_nodes: Iterable[Mapping[str, Any]],
        node_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        artifact_availability: Iterable[PipelineArtifactAvailability] = (),
        safety_margin: float = 1.2,
    ) -> PipelineReshardDecision:
        with self._lock:
            decision = plan_pipeline_reshard(
                self._layout,
                base_epoch=self._topology_lease.epoch,
                failed_node_ids=failed_node_ids,
                descriptor=descriptor,
                capacity_nodes=capacity_nodes,
                node_metadata=node_metadata,
                artifact_availability=tuple(self._availability) + tuple(artifact_availability),
                safety_margin=safety_margin,
            )
            if decision.accepted and decision.plan is not None:
                # Only one recovery transaction may be authoritative for an
                # epoch. A later failure supersedes every earlier candidate.
                self._staged.clear()
                self._staged[decision.plan.plan_id] = decision.plan
            return decision

    def record_artifact(self, availability: PipelineArtifactAvailability) -> None:
        if not isinstance(availability, PipelineArtifactAvailability):
            raise PipelineReshardError(
                "availability must be PipelineArtifactAvailability",
            )
        with self._lock:
            self._availability.append(availability)
            for plan_id, plan in tuple(self._staged.items()):
                self._staged[plan_id] = replace(
                    plan,
                    missing_requirements=_missing_requirements(
                        plan.requirements, self._availability,
                    ),
                )

    def staged_plan(self, plan_id: str) -> PipelineReshardPlan | None:
        """Return one immutable staged plan for execution-plane attestation."""
        with self._lock:
            return self._staged.get(str(plan_id))

    def commit(
        self, plan_id: str, *, expected_epoch: int,
        control_certificate: Mapping[str, Any] | None = None,
    ) -> PipelineReshardCommitDecision:
        with self._lock:
            plan = self._staged.get(str(plan_id))
            current_epoch = self._topology_lease.epoch
            if plan is None:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_unknown_plan", current_epoch, self._layout,
                )
            try:
                requested_epoch = int(expected_epoch)
            except (TypeError, ValueError):
                requested_epoch = -1
            if requested_epoch != current_epoch or plan.base_epoch != current_epoch:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_stale_epoch", current_epoch, self._layout,
                )
            if plan.base_contract_sha256 != self._layout.contract_sha256:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_stale_layout", current_epoch, self._layout,
                )
            if self._lease_book.check(
                self._topology_lease.lease_id, current_epoch,
            ).accepted is False:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_fenced", current_epoch, self._layout,
                )
            missing = _missing_requirements(plan.requirements, self._availability)
            if missing:
                return PipelineReshardCommitDecision(
                    False, "pipeline_reshard_assets_not_ready", current_epoch, self._layout,
                )
            next_lease = self._lease_book.reassign(
                self._TOPOLOGY_SHARD_ID,
                self._COORDINATOR_ID,
                plan.candidate_layout.model_sha256,
                {"contract_sha256": plan.candidate_layout.contract_sha256},
                reason="reshard",
                lease_seconds=86400.0,
                certificate=control_certificate or self._control_certificate,
            )
            self._control_certificate = control_certificate or self._control_certificate
            self._layout = plan.candidate_layout
            self._topology_lease = next_lease
            self._staged.clear()
            return PipelineReshardCommitDecision(
                True, "pipeline_reshard_committed", next_lease.epoch, self._layout,
                next_lease,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            plans = list(self._staged.values())
            staged = [plan.to_dict() for plan in plans]
            return {
                "schema_version": RESHARD_SCHEMA_VERSION,
                "status": (
                    "ready" if plans and all(plan.assets_ready for plan in plans)
                    else "staged" if plans
                    else "active"
                ),
                "epoch": self._topology_lease.epoch,
                "layout": self._layout.to_dict(),
                "staged": staged,
            }
