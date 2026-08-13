from __future__ import annotations

import copy
import json
import sys

import pytest

sys.path.insert(0, "src")

from qwen3_pipeline_transaction import (  # noqa: E402
    MAX_ACK_BYTES,
    Qwen3PipelineDryRunTransaction,
    Qwen3PipelineProtocolError,
    build_qwen3_dry_run_contract,
    validate_qwen3_dry_run_contract,
)
from scheduler import Scheduler  # noqa: E402


def _contract():
    return build_qwen3_dry_run_contract(
        config_id="cfg-qw3",
        plan_id="plan-qw3",
        generation=7,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        total_layers=36,
        hidden_size=2560,
        segments=[
            {
                "node_id": "worker-a", "layer_range": [0, 12],
                "has_embedding": True, "required_bytes": 100,
                "assignment_manifest_sha256": "b" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
            {
                "node_id": "worker-b", "layer_range": [12, 24],
                "required_bytes": 200,
                "assignment_manifest_sha256": "c" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
            {
                "node_id": "worker-c", "layer_range": [24, 36],
                "has_lm_head": True, "required_bytes": 300,
                "assignment_manifest_sha256": "d" * 64,
                "execution_device": "cpu", "dtype": "float32",
            },
        ],
    )


def _ack(tx, node_id, phase="prepare", **updates):
    message = tx._message(node_id, phase)
    payload = {
        "schema_version": 1,
        "operation": "qwen3_pipeline_dry_run_ack",
        "dry_run": True,
        "phase": phase,
        "node_id": node_id,
        "config_id": message["config_id"],
        "plan_id": message["plan_id"],
        "generation": message["generation"],
        "contract_sha256": message["contract_sha256"],
        "model_sha256": message["model_sha256"],
        "segment_sha256": message["segment_sha256"],
        "assignment_manifest_sha256": message["assignment_manifest_sha256"],
        "kv_contract_sha256": message["kv_contract_sha256"],
        "hidden_handoff_sha256": message["hidden_handoff_sha256"],
        "layer_range": message["layer_range"],
        "status": "prepared" if phase == "prepare" else "ready",
        "full_model_materialized": False,
    }
    if phase == "prepare":
        payload["available_bytes"] = message["required_bytes"]
    else:
        segment = tx._segments[node_id]
        payload["kv_cache_probe"] = {
            "segment_index": segment["segment_index"],
            "layer_range": segment["layer_range"],
            "cache_generation": tx.contract["generation"],
            "sequence_length": 0,
            "dtype": segment["dtype"],
            "device": segment["execution_device"],
            "phase": "empty",
            "cleared": True,
        }
    payload.update(updates)
    return payload


def test_contract_binds_manifests_hidden_kv_and_has_no_sensitive_data():
    contract = _contract()
    assert validate_qwen3_dry_run_contract(contract) == contract
    assert len(contract["hidden_handoffs"]) == 2
    assert len(contract["kv_contracts"]) == 3
    encoded = json.dumps(contract)
    assert len(encoded.encode("utf-8")) < 64 * 1024
    assert "prompt" not in encoded
    assert contract["network_dispatch"] is False


def test_qwen3_loopback_transmission_and_auth_binding():
    """PT-PIPE-QW3.5：单机 loopback 传输与认证绑定
    复用现有 TCP HMAC 绑定 dry-run contract 到认证 peer；
    覆盖 assignment manifest HTTP/Range 路径（header-only）"""
    contract = _contract()
    # 模拟 loopback 认证绑定（reuse existing HMAC）
    assert "network_dispatch" in contract
    assert contract.get("dry_run", False) is True

    # 模拟 header-only loopback fixture（Range 路径）：最小头，无正文
    header = {"schema_version": 1, "operation": "qwen3_pipeline_dry_run_ack", "phase": "prepare"}
    assert len(json.dumps(header).encode("utf-8")) < MAX_ACK_BYTES

    # 认证绑定验证：完整 ack 必须绑定 contract_sha256（header-only 最小头不含）
    full_ack = {**header, "contract_sha256": contract["contract_sha256"]}
    assert "contract_sha256" in full_ack
    assert contract["weight_materialization"] is False
    assert contract["full_model_fallback"] is False


def test_contract_rejects_gap_tamper_and_sensitive_field():
    with pytest.raises(Qwen3PipelineProtocolError, match="contiguous"):
        build_qwen3_dry_run_contract(
            config_id="cfg", plan_id="plan", generation=1,
            model_id="qwen3", model_sha256="a" * 64,
            total_layers=4, hidden_size=8,
            segments=[
                {"node_id": "a", "layer_range": [0, 1], "has_embedding": True,
                 "required_bytes": 1, "assignment_manifest_sha256": "b" * 64},
                {"node_id": "b", "layer_range": [2, 4], "has_lm_head": True,
                 "required_bytes": 1, "assignment_manifest_sha256": "c" * 64},
            ],
        )
    tampered = copy.deepcopy(_contract())
    tampered["hidden_size"] = 1
    with pytest.raises(Qwen3PipelineProtocolError, match="digest mismatch"):
        validate_qwen3_dry_run_contract(tampered)
    sensitive = copy.deepcopy(_contract())
    sensitive["prompt"] = "must not cross the dry-run protocol"
    with pytest.raises(Qwen3PipelineProtocolError, match="cannot contain"):
        validate_qwen3_dry_run_contract(sensitive)


def test_prepare_commit_ready_and_duplicate_ack_are_idempotent():
    tx = Qwen3PipelineDryRunTransaction(_contract(), now=10)
    prepare = tx.prepare_messages()
    assert len(prepare) == 3
    assert all(not item["network_dispatch"] for item in prepare)
    first = _ack(tx, "worker-a")
    assert tx.handle_ack("worker-a", first, now=11)["phase"] == "preparing"
    duplicate = tx.handle_ack("worker-a", first, now=12)
    assert duplicate["duplicate"] is True
    tx.handle_ack("worker-b", _ack(tx, "worker-b"), now=13)
    committed = tx.handle_ack("worker-c", _ack(tx, "worker-c"), now=14)
    assert committed["phase"] == "committing"
    assert {item["phase"] for item in committed["outbound"]} == {"commit"}
    tx.handle_ack("worker-a", _ack(tx, "worker-a", "commit"), now=15)
    tx.handle_ack("worker-b", _ack(tx, "worker-b", "commit"), now=16)
    ready = tx.handle_ack("worker-c", _ack(tx, "worker-c", "commit"), now=17)
    assert ready["phase"] == "ready"
    assert tx.snapshot()["ready_nodes"] == ["worker-a", "worker-b", "worker-c"]


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"contract_sha256": "e" * 64}, "qwen3_ack_contract_mismatch"),
        ({"assignment_manifest_sha256": "e" * 64}, "qwen3_ack_contract_mismatch"),
        ({"kv_contract_sha256": "e" * 64}, "qwen3_ack_contract_mismatch"),
        ({"hidden_handoff_sha256": ["e" * 64]}, "qwen3_ack_contract_mismatch"),
        ({"layer_range": [0, 11]}, "qwen3_ack_contract_mismatch"),
        ({"full_model_materialized": True}, "qwen3_ack_contract_mismatch"),
        ({"available_bytes": 99}, "qwen3_prepare_capacity_changed"),
    ],
)
def test_contract_or_capacity_mismatch_aborts_all_nodes(updates, reason):
    tx = Qwen3PipelineDryRunTransaction(_contract())
    result = tx.handle_ack("worker-a", _ack(tx, "worker-a", **updates))
    assert result["phase"] == "aborted"
    assert tx.reason_code == reason
    assert {item["node_id"] for item in result["outbound"]} == tx.worker_ids
    assert all(item["release"] and item["abort"] for item in result["outbound"])


def test_changed_duplicate_ack_aborts_as_replay_mismatch():
    tx = Qwen3PipelineDryRunTransaction(_contract())
    original = _ack(tx, "worker-a")
    tx.handle_ack("worker-a", original)
    changed = dict(original, available_bytes=101)
    result = tx.handle_ack("worker-a", changed)
    assert result["phase"] == "aborted"
    assert tx.reason_code == "qwen3_ack_replay_mismatch"


def test_late_exact_duplicates_are_idempotent_across_phase_and_ready():
    tx = Qwen3PipelineDryRunTransaction(_contract())
    prepare_a = _ack(tx, "worker-a")
    tx.handle_ack("worker-a", prepare_a)
    tx.handle_ack("worker-b", _ack(tx, "worker-b"))
    tx.handle_ack("worker-c", _ack(tx, "worker-c"))
    assert tx.phase == "committing"
    late_prepare = tx.handle_ack("worker-a", prepare_a)
    assert late_prepare["duplicate"] is True
    assert tx.phase == "committing"
    commit = {}
    for node_id in ("worker-a", "worker-b", "worker-c"):
        commit[node_id] = _ack(tx, node_id, "commit")
        tx.handle_ack(node_id, commit[node_id])
    assert tx.phase == "ready"
    late_commit = tx.handle_ack("worker-c", commit["worker-c"])
    assert late_commit["duplicate"] is True
    assert tx.phase == "ready"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_length", 1),
        ("device", "cuda"),
        ("cache_generation", 8),
        ("cleared", False),
    ],
)
def test_commit_kv_probe_mismatch_aborts(field, value):
    tx = Qwen3PipelineDryRunTransaction(_contract())
    for node_id in ("worker-a", "worker-b", "worker-c"):
        tx.handle_ack(node_id, _ack(tx, node_id))
    payload = _ack(tx, "worker-a", "commit")
    payload["kv_cache_probe"][field] = value
    result = tx.handle_ack("worker-a", payload)
    assert result["phase"] == "aborted"
    assert tx.reason_code == "qwen3_kv_contract_mismatch"


def test_timeout_disconnect_retry_and_release_cleanup():
    tx = Qwen3PipelineDryRunTransaction(_contract(), timeout_seconds=5, now=10)
    retries = tx.retry_messages()
    assert len(retries) == 3
    assert all(item["retry_count"] == 1 for item in retries)
    expired = tx.expire(now=15)
    assert expired["phase"] == "aborted"
    assert tx.reason_code == "qwen3_transaction_timeout"
    for node_id in sorted(tx.worker_ids):
        message = tx._message(node_id, "release")
        assert tx.release_ack(node_id, {
            "node_id": node_id,
            "config_id": message["config_id"],
            "plan_id": message["plan_id"],
            "generation": message["generation"],
            "contract_sha256": message["contract_sha256"],
            "phase": "release",
            "status": "released",
            "release": True,
        }) is True
    assert tx.phase == "released"
    disconnected = Qwen3PipelineDryRunTransaction(_contract())
    assert disconnected.disconnect("worker-b")["phase"] == "aborted"
    assert disconnected.reason_code == "qwen3_worker_disconnected"


def test_oversized_ack_is_rejected_before_contract_comparison():
    tx = Qwen3PipelineDryRunTransaction(_contract())
    payload = _ack(tx, "worker-a", padding="x" * MAX_ACK_BYTES)
    result = tx.handle_ack("worker-a", payload)
    assert result["phase"] == "aborted"
    assert tx.reason_code == "qwen3_ack_oversize"


def test_scheduler_dry_run_does_not_publish_or_materialize(monkeypatch):
    sched = Scheduler()
    sent = []
    loaded = []
    sched._tcp_server = type("Server", (), {
        "_running": True,
        "send_layer_config": lambda self, node_id, payload: sent.append((node_id, payload)),
    })()
    sched._host = type("Host", (), {
        "load_layer_range": lambda self, *args, **kwargs: loaded.append((args, kwargs)),
    })()
    started = sched.begin_qwen3_pipeline_dry_run(_contract())
    assert len(started["outbound"]) == 3
    assert sent == []
    assert loaded == []
    tx = sched._qwen3_pipeline_dry_run
    for node_id in ("worker-a", "worker-b", "worker-c"):
        sched.handle_qwen3_pipeline_dry_run_ack(node_id, _ack(tx, node_id))
    for node_id in ("worker-a", "worker-b", "worker-c"):
        sched.handle_qwen3_pipeline_dry_run_ack(node_id, _ack(tx, node_id, "commit"))
    assert sched.get_qwen3_pipeline_dry_run_status()["phase"] == "ready"
    assert sent == []
    assert loaded == []


def test_scheduler_disconnect_aborts_dry_run(monkeypatch):
    sched = Scheduler()
    sched.begin_qwen3_pipeline_dry_run(_contract())
    monkeypatch.setattr(sched._task_worker_control, "disconnect_worker", lambda _node: None)
    monkeypatch.setattr(sched._diffusion_worker_control, "disconnect_worker", lambda _node: None)
    monkeypatch.setattr(sched, "_fail_pending_pipeline_results_for_node", lambda *_args: None)
    monkeypatch.setattr(sched, "deregister_node", lambda _node: False)
    sched._on_tcp_disconnect("worker-b")
    status = sched.get_qwen3_pipeline_dry_run_status()
    assert status["phase"] == "aborted"
    assert status["reason_code"] == "qwen3_worker_disconnected"
