from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.cluster_transport import (
    FakeTransportLink,
    STREAM_CHANNEL,
    TransportChunkAck,
    TransportContractError,
)
from tcp_comm import MessageType, TCPClient, build_message
from src.transport_runtime import TransportRuntimeBridge


def test_fake_link_roundtrip_uses_runtime_envelope_and_fence() -> None:
    left_endpoint, right_endpoint = FakeTransportLink.pair()
    left = TransportRuntimeBridge(client_id="left", window_bytes=1024)
    right = TransportRuntimeBridge(client_id="right", window_bytes=1024)
    attempt = left.connected(3, attempt_id="attempt-3")
    right.connected(3, attempt_id=attempt)
    left_endpoint.open(generation=3, attempt_id=attempt, window_bytes=1024)
    right_endpoint.open(generation=3, attempt_id=attempt, window_bytes=1024)

    envelope = left.send_v2(
        left_endpoint,
        b"hidden-state",
        request_id="task-1",
        channel=STREAM_CHANNEL,
    )
    left_endpoint.deliver_next()
    frame = right.receive_v2(right_endpoint)
    assert frame.envelope == envelope
    assert frame.payload == b"hidden-state"
    assert left.snapshot()["window"]["inflight_bytes"] == len(frame.payload)

    left_endpoint.acknowledge(
        TransportChunkAck(
            request_id="task-1",
            connection_generation=3,
            attempt_id=attempt,
            sequence=0,
            payload_size=len(frame.payload),
        )
    )
    left.acknowledge_v2(
        TransportChunkAck(
            request_id="task-1",
            connection_generation=3,
            attempt_id=attempt,
            sequence=0,
            payload_size=len(frame.payload),
        )
    )
    assert left.snapshot()["window"]["inflight_bytes"] == 0


def test_runtime_bridge_classifies_failures_and_opens_breaker() -> None:
    endpoint_left, _endpoint_right = FakeTransportLink.pair()
    runtime = TransportRuntimeBridge(
        client_id="left",
        failure_threshold=2,
        cooldown_seconds=60,
    )
    runtime.connected(1, attempt_id="attempt")
    endpoint_left.open(generation=1, attempt_id="attempt", window_bytes=1024)
    endpoint_left.close()
    for _ in range(2):
        with pytest.raises(TransportContractError) as exc:
            runtime.send_v2(endpoint_left, b"x", request_id="task")
        assert exc.value.code == "connection_reset"
    snapshot = runtime.snapshot()
    assert snapshot["breaker"]["state"] == "open"
    assert snapshot["last_failure"]["next_action"] == "reconnect"


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def sendall(self, packet: bytes) -> None:
        self.sent.append(bytes(packet))


def test_tcp_client_bridge_observes_legacy_without_rewriting_packet() -> None:
    runtime = TransportRuntimeBridge(client_id="client-test")
    client = TCPClient(client_id="client-test", transport_runtime=runtime)
    sock = _FakeSocket()
    client.sock = sock
    client._running = True

    client.send_data({"request": "ping"}, MessageType.NODE_LIST_SYNC)

    expected = build_message(MessageType.NODE_LIST_SYNC, {"request": "ping"})
    assert sock.sent == [expected]
    snapshot = client.get_transport_snapshot()
    assert snapshot["counters"]["legacy_outbound"] == 1
    assert snapshot["mode"] == "legacy_tcp"
