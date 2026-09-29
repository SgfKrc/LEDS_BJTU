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


def _peer(sent: list[dict], *, role: str = "middle") -> PeerClient:
    peer = object.__new__(PeerClient)
    peer._node_id = "surface-worker"
    peer._active_layer_config = {
        "relay_segment": {
            "host": "127.0.0.1",
            "port": 50283,
            "role": role,
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


# ---- ★ Y-(b)：tail 段（hidden → token）与不支持角色的具名拒绝 -------------


class _TailOutcome:
    ok = True
    role = "tail"
    error = ""
    endpoint = "127.0.0.1:50284"
    hidden = b""

    def __init__(self, token: int):
        self.token = token

    def to_metrics(self):
        return {"relay_segment": "tail@127.0.0.1:50284"}


class _TailRelayClient:
    received: bytes | None = None
    called: str = ""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def forward_hidden(self, hidden: bytes, *, n_tokens: int):  # pragma: no cover
        type(self).called = "forward_hidden"
        raise AssertionError("tail 段不应走 forward_hidden")

    def forward_hidden_to_token(self, hidden: bytes, *, n_tokens: int):
        type(self).called = "forward_hidden_to_token"
        assert n_tokens == 2
        type(self).received = hidden
        return _TailOutcome(4242)


def _payload(values: np.ndarray, task_id: str = "t-tail") -> dict:
    return {
        "task_id": task_id,
        "step": 0,
        "hidden_states": base64.b64encode(values.tobytes()).decode("ascii"),
        "hidden_shape": list(values.shape),
        "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
    }


def test_peer_relay_tail_returns_token_and_never_calls_middle(monkeypatch):
    """★ Y-(b)：`tail` 段必须走 `forward_hidden_to_token`，回 **token**（不带 hidden）。"""
    _TailRelayClient.called = ""
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _TailRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent, role="tail")

    peer._handle_layer_forward_via_relay(_payload(values))

    assert _TailRelayClient.called == "forward_hidden_to_token"
    assert _TailRelayClient.received == values.tobytes()
    assert len(sent) == 1 and "error" not in sent[0]
    assert sent[0]["token"] == 4242
    assert "hidden_states" not in sent[0]
    assert sent[0]["metrics"]["relay_executed"] is True


def test_peer_relay_rejects_role_unsupported_on_the_client_side(monkeypatch):
    """★ `head` 在**从节点这一侧**不支持（head 段要 token 序列，而从节点收到的是 hidden）
    ⇒ 必须**具名拒绝**：绝不"当成 middle 硬算"（那会静默产出错误数值）。"""
    monkeypatch.setattr("relay_segment_client.RelaySegmentClient", _TailRelayClient)
    values = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    sent: list[dict] = []
    peer = _peer(sent, role="head")

    peer._handle_layer_forward_via_relay(_payload(values, task_id="t-head"))

    assert len(sent) == 1
    assert sent[0].get("error") == "relay_role_unsupported:head"
