from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "src")

import qwen3_pipeline_state as state_module  # noqa: E402


class _Store:
    def __init__(self) -> None:
        self.values = {}

    def get_local_setting(self, key, default):
        return self.values.get(key, default)

    def set_local_setting(self, key, value):
        self.values[key] = value


def test_local_chain_state_round_trip_keeps_only_recovery_metadata(monkeypatch):
    store = _Store()
    monkeypatch.setattr(state_module, "_store", lambda: store)

    saved = state_module.save_qwen3_local_chain_state({
        "contract_sha256": "a" * 64,
        "config_id": "cfg",
        "plan_id": "plan",
        "generation": 4,
        "phase": "parity_passed",
        "segment_count": 2,
        "cleanup_complete": False,
        "artifact_path": "C:/private/model.pt",
        "parity": {
            "schema_version": 1,
            "gate": "qwen3_cpu_parity",
            "status": "passed",
            "gate_passed": True,
            "full_model_fallback": False,
            "full_model_materialized": False,
            "prefill": {"passed": True, "shape": [1, 2], "max_abs_error": 0.0},
            "decode": {"passed": True, "shape": [1, 2], "max_abs_error": 0.0},
            "artifact_path": "C:/private/parity.pt",
            "tensor": object(),
            "errors": [],
        },
    })

    loaded = state_module.load_qwen3_local_chain_state()
    encoded = json.dumps(store.values[state_module.STATE_KEY], sort_keys=True)
    assert loaded == saved
    assert loaded["config_id"] == "cfg"
    assert loaded["parity"]["gate_passed"] is True
    assert "artifact_path" not in encoded
    assert "private" not in encoded
    assert "tensor" not in encoded


def test_invalid_persisted_state_is_fail_closed(monkeypatch):
    store = _Store()
    store.values[state_module.STATE_KEY] = {
        "contract_sha256": "short",
        "phase": "committed",
    }
    monkeypatch.setattr(state_module, "_store", lambda: store)

    loaded = state_module.load_qwen3_local_chain_state()

    assert loaded["phase"] == "recovered_aborted"
    assert loaded["cleanup_complete"] is False


def test_network_ledger_round_trip_is_metadata_only(monkeypatch):
    store = _Store()
    monkeypatch.setattr(state_module, "_store", lambda: store)
    reference = {
        "schema_version": 1,
        "mode": "network",
        "artifact_id": "qout_" + "1" * 32,
        "source_node_id": "node-b",
        "target_node_id": "node-c",
        "chain_id": "a" * 64,
        "generation": 4,
        "phase": "prefill",
        "from_segment": 1,
        "to_segment": 2,
        "size_bytes": 128,
        "sha256": "b" * 64,
        "status": "committed",
        "full_model_materialized": False,
    }
    transfer_id = "qtx_" + "2" * 32
    saved = state_module.save_qwen3_network_ledger({
        "local_node_id": "node-b",
        "last_generation": 4,
        "active_contract": {
            "contract_sha256": "a" * 64,
            "generation": 4,
            "phase": "prefill",
            "segment_count": 3,
            "restart_epoch": 0,
        },
        "transfers": {
            transfer_id: {
                "source_node_id": "node-a",
                "target_node_id": "node-b",
                "chain_id": "a" * 64,
                "generation": 4,
                "phase": "prefill",
                "from_segment": 0,
                "to_segment": 1,
                "size_bytes": 64,
                "sha256": "c" * 64,
                "status": "consumed",
                "received_bytes": 64,
                "input_reference": {
                    **reference,
                    "artifact_id": transfer_id,
                    "source_node_id": "node-a",
                    "target_node_id": "node-b",
                    "from_segment": 0,
                    "to_segment": 1,
                    "size_bytes": 64,
                    "sha256": "c" * 64,
                },
                "output_reference_id": reference["artifact_id"],
                "kv_contract": {"present": True, "shape": [1, 4]},
                "ticket": "secret-ticket",
                "path": "C:/private/input.pt",
            },
        },
        "outputs": {
            reference["artifact_id"]: {
                "reference": reference,
                "parent_transfer_id": transfer_id,
                "status": "transferring",
                "lease_state": "leased",
                "next_transfer_id": "qtx_" + "3" * 32,
                "confirmed_offset": 32,
                "path": "C:/private/output.pt",
            },
        },
    }, "node-b")

    loaded = state_module.load_qwen3_network_ledger("node-b")
    encoded = json.dumps(saved, sort_keys=True)
    assert loaded == saved
    assert loaded["transfers"][transfer_id]["kv_contract"] == {
        "present": True,
        "shape": [1, 4],
        "generation": 4,
        "phase": "prefill",
    }
    assert "secret-ticket" not in encoded
    assert "C:/private" not in encoded
    assert '"path"' not in encoded


def test_network_ledger_rejects_path_shaped_identity_and_non_hex_digest(monkeypatch):
    store = _Store()
    monkeypatch.setattr(state_module, "_store", lambda: store)
    transfer_id = "qtx_" + "1" * 32
    value = {
        "local_node_id": "node-b",
        "last_generation": 1,
        "active_contract": {
            "contract_sha256": "a" * 64,
            "generation": 1,
            "phase": "prefill",
            "segment_count": 2,
            "restart_epoch": 0,
        },
        "transfers": {
            transfer_id: {
                "source_node_id": "C:/private/node-a",
                "target_node_id": "node-b",
                "chain_id": "a" * 64,
                "generation": 1,
                "phase": "prefill",
                "from_segment": 0,
                "to_segment": 1,
                "size_bytes": 8,
                "sha256": "z" * 64,
                "status": "receiving",
                "received_bytes": 0,
            },
        },
        "outputs": {},
    }

    with pytest.raises(ValueError, match="transfer contract"):
        state_module.save_qwen3_network_ledger(value, "node-b")
