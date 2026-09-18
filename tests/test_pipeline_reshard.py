"""Automatic layer recovery tests for CORE-RESHARD-01."""

from src.pipeline_node_contract import (
    PipelineArtifactRef,
    PipelineNode,
    PipelineNodeCapacity,
    validate_pipeline_nodes,
)
from src.pipeline_reshard import (
    PipelineArtifactAvailability,
    PipelineReshardCoordinator,
    plan_pipeline_reshard,
)


MODEL_SHA = "a" * 64


def _descriptor():
    return {
        "model_id": "reshard-fixture",
        "model_type": "qwen3",
        "model_sha256": MODEL_SHA,
        "total_layers": 6,
        "layer_weight_bytes": [100] * 6,
        "component_weight_bytes": {
            "embedding": 10,
            "final_norm": 10,
            "lm_head": 10,
            "visual": 0,
            "mtp": 0,
            "multimodal": 0,
            "other": 0,
        },
    }


def _capacity_record(node_id, capacity=360, *, role="client", score=10):
    return {
        "node_id": node_id,
        "role": role,
        "capacity_bytes": capacity,
        "reserve_bytes": 0,
        "runtime_multiplier": 1.0,
        "execution_device": "cpu",
        "capacity_source": "recovery-test",
        "score": score,
    }


def _node(node_id, start, end, *, embedding=False, head=False):
    is_local = node_id == "master"
    return PipelineNode(
        node_id=node_id,
        kind="local" if is_local else "remote_pipeline",
        layer_range=(start, end),
        engine="pytorch",
        location="local" if is_local else f"node:{node_id}",
        capacity=PipelineNodeCapacity(
            capacity_bytes=260,
            required_bytes=220,
            execution_device="cpu",
            capacity_source="initial-test",
        ),
        artifact=PipelineArtifactRef(
            model_sha256=MODEL_SHA,
            artifact_kind="safetensors_assignment",
            architecture="qwen3",
            source_layer_range=(start, end),
        ),
        has_embedding=embedding,
        has_lm_head=head,
        federated=not is_local,
    )


def _layout():
    return validate_pipeline_nodes([
        _node("master", 0, 2, embedding=True),
        _node("worker-a", 2, 4),
        _node("worker-b", 4, 6, head=True),
    ], total_layers=6)


def _survivors(master_capacity=360, worker_capacity=360):
    return [
        _capacity_record(
            "master", master_capacity, role="master", score=20,
        ),
        _capacity_record("worker-b", worker_capacity, score=10),
    ]


def _availability_for(plan):
    return [
        PipelineArtifactAvailability(
            node_id=node.node_id,
            model_sha256=MODEL_SHA,
            artifact_kind="safetensors_assignment",
            layer_range=node.layer_range,
            has_embedding=node.has_embedding,
            has_lm_head=node.has_lm_head,
        )
        for node in plan.candidate_layout.nodes
    ]


def test_failure_reassigns_exact_ranges_only_after_capacity_preflight():
    decision = plan_pipeline_reshard(
        _layout(),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    assert decision.accepted is True
    assert decision.ready_to_commit is False
    assert decision.reason_code == "pipeline_reshard_assets_required"
    assert [node.node_id for node in decision.plan.candidate_layout.nodes] == [
        "master", "worker-b",
    ]
    assert [node.layer_range for node in decision.plan.candidate_layout.nodes] == [
        (0, 3), (3, 6),
    ]
    assert any(
        move.source_node_id == "worker-a" for move in decision.plan.moves
    )
    assert decision.plan.candidate_layout.total_layers == 6


def test_insufficient_survivor_capacity_rejects_without_partial_layout():
    decision = plan_pipeline_reshard(
        _layout(),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(250, 250),
        safety_margin=1.0,
    )

    assert decision.accepted is False
    assert decision.reason_code == "pipeline_distributed_capacity_insufficient"
    assert decision.plan is None


def test_spare_node_can_join_recovery_but_must_prove_its_artifact():
    decision = plan_pipeline_reshard(
        validate_pipeline_nodes([
            _node("master", 0, 3, embedding=True),
            _node("worker-a", 3, 6, head=True),
        ], total_layers=6),
        base_epoch=1,
        failed_node_ids={"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=[
            _capacity_record("master", role="master", score=20),
            _capacity_record("spare", score=5),
        ],
        node_metadata={
            "spare": {
                "location": "node:spare",
                "kind": "remote_pipeline",
                "engine": "pytorch",
                "federated": True,
                "artifact_kind": "safetensors_assignment",
                "architecture": "qwen3",
            },
        },
        safety_margin=1.0,
    )

    assert decision.accepted is True
    assert "spare" in {
        node.node_id for node in decision.plan.candidate_layout.nodes
    }
    assert "spare" in {
        requirement.node_id for requirement in decision.plan.missing_requirements
    }


def test_commit_waits_for_assets_then_atomically_fences_old_epoch():
    coordinator = PipelineReshardCoordinator(_layout())
    old_layout = coordinator.layout
    old_lease = coordinator.topology_lease
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    blocked = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert blocked.accepted is False
    assert blocked.reason_code == "pipeline_reshard_assets_not_ready"
    assert coordinator.layout.contract_sha256 == old_layout.contract_sha256

    for availability in _availability_for(staged.plan):
        coordinator.record_artifact(availability)
    assert coordinator.snapshot()["staged"][0]["assets_ready"] is True

    stale = coordinator.commit(staged.plan.plan_id, expected_epoch=0)
    assert stale.reason_code == "pipeline_reshard_stale_epoch"
    assert coordinator.layout.contract_sha256 == old_layout.contract_sha256

    committed = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert committed.accepted is True
    assert committed.epoch == 2
    assert coordinator.layout.contract_sha256 == staged.plan.candidate_layout.contract_sha256
    assert coordinator.fence(2).accepted is True
    assert coordinator._lease_book.check(old_lease.lease_id, 1).reason == "stale_lease"


def test_wrong_model_or_unverified_artifact_never_opens_commit_gate():
    coordinator = PipelineReshardCoordinator(_layout())
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )
    for requirement in staged.plan.missing_requirements:
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256="b" * 64,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=True,
            has_embedding=requirement.has_embedding,
            has_lm_head=requirement.has_lm_head,
        ))
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256=requirement.model_sha256,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=False,
            has_embedding=requirement.has_embedding,
            has_lm_head=requirement.has_lm_head,
        ))

    result = coordinator.commit(staged.plan.plan_id, expected_epoch=1)
    assert result.accepted is False
    assert result.reason_code == "pipeline_reshard_assets_not_ready"


def test_layer_coverage_without_required_boundary_weights_is_not_ready():
    coordinator = PipelineReshardCoordinator(_layout())
    staged = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )
    for requirement in staged.plan.missing_requirements:
        coordinator.record_artifact(PipelineArtifactAvailability(
            node_id=requirement.node_id,
            model_sha256=requirement.model_sha256,
            artifact_kind=requirement.artifact_kind,
            layer_range=requirement.layer_range,
            verified=True,
            has_embedding=False,
            has_lm_head=False,
        ))

    missing = coordinator.snapshot()["staged"][0]["missing_requirements"]
    assert any(item["has_embedding"] or item["has_lm_head"] for item in missing)
    assert coordinator.commit(
        staged.plan.plan_id, expected_epoch=1,
    ).reason_code == "pipeline_reshard_assets_not_ready"


def test_later_failure_supersedes_earlier_plan_in_the_same_epoch():
    coordinator = PipelineReshardCoordinator(_layout())
    first = coordinator.stage_failure(
        {"worker-a"},
        descriptor=_descriptor(),
        capacity_nodes=_survivors(),
        safety_margin=1.0,
    )

    second = coordinator.stage_failure(
        {"worker-a", "worker-b"},
        descriptor=_descriptor(),
        capacity_nodes=[
            _capacity_record("master", role="master", score=20),
            _capacity_record("spare", score=5),
        ],
        node_metadata={
            "spare": {
                "location": "node:spare",
                "kind": "remote_pipeline",
                "engine": "pytorch",
                "federated": True,
                "artifact_kind": "safetensors_assignment",
                "architecture": "qwen3",
            },
        },
        safety_margin=1.0,
    )

    assert second.accepted is True
    assert second.plan.plan_id != first.plan.plan_id
    assert coordinator.staged_plan(first.plan.plan_id) is None
    assert coordinator.commit(
        first.plan.plan_id, expected_epoch=1,
    ).reason_code == "pipeline_reshard_unknown_plan"
    assert [item["plan_id"] for item in coordinator.snapshot()["staged"]] == [
        second.plan.plan_id,
    ]
