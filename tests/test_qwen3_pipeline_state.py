from __future__ import annotations

import json
import sys

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
