"""Regression tests for the product PeerClient relay wire contract."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from inference_service.peer import (  # noqa: E402
    PeerClient,
    RELAY_HIDDEN_WIRE_FORMAT,
)
from scheduler_pipeline import _decode_relay_hidden  # noqa: E402


class _Outcome:
    ok = True
    role = "middle"
    error = ""
    endpoint = "127.0.0.1:50283"

    def __init__(self, hidden: bytes):
        self.hidden = hidden

    def to_metrics(self):
        return {"relay_segment": "middle"}


class _RelayClient:
    received: bytes | None = None

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def forward_hidden(self, hidden: bytes, *, n_tokens: int):
        assert n_tokens == 2
        type(self).received = hidden
        values = np.frombuffer(hidden, dtype=np.float32).copy() + 1.0
        return _Outcome(values.tobytes())


def _peer(sent: list[dict]) -> PeerClient:
    peer = object.__new__(PeerClient)
    peer._node_id = "surface-worker"
    peer._active_layer_config = {
        "relay_segment": {
            "host": "127.0.0.1",
            "port": 50283,
            "role": "middle",
            "n_embd": 3,
            "timeout": 2.0,
        }
    }
    peer._send_layer_result = lambda _task_id, payload, error=None: sent.append(
        {**payload, **({"error": error} if error else {})}
    ) or True
    return peer


def test_peer_relay_roundtrip_uses_raw_f32_contract(monkeypatch):
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _RelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent)

    peer._handle_layer_forward_via_relay(
        {
            "task_id": "t1",
            "step": 0,
            "hidden_states": base64.b64encode(values.tobytes()).decode("ascii"),
            "hidden_shape": list(values.shape),
            "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
        }
    )

    assert _RelayClient.received == values.tobytes()
    assert len(sent) == 1 and "error" not in sent[0]
    assert sent[0]["hidden_wire_format"] == RELAY_HIDDEN_WIRE_FORMAT
    assert sent[0]["hidden_shape"] == [1, 2, 3]
    returned = _decode_relay_hidden(
        sent[0]["hidden_states"], sent[0]["hidden_shape"]
    )
    np.testing.assert_array_equal(returned.numpy(), values + 1.0)


def test_peer_relay_rejects_raw_f32_without_shape(monkeypatch):
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _RelayClient)
    sent: list[dict] = []
    peer = _peer(sent)

    peer._handle_layer_forward_via_relay(
        {
            "task_id": "t2",
            "hidden_states": base64.b64encode(np.zeros(3, dtype=np.float32).tobytes()).decode("ascii"),
            "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
        }
    )

    assert len(sent) == 1
    assert "hidden_shape is required" in sent[0]["error"]
