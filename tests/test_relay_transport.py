"""Wire and loopback bridge tests for CORE-RELAY-XFRAME-01."""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from src.relay_transport import (
    RELAY_WIRE_MAGIC,
    RelayFrame,
    RelayFrameKind,
    RelayProtocolError,
    RelayTcpClient,
    expected_hidden_bytes,
    open_loopback_listener,
    recv_frame,
    send_frame,
    serve_relay_connection,
)


class _FakeRunner:
    def __init__(self) -> None:
        self.requests: list[tuple[int, bytes]] = []
        self.closed = False

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        self.requests.append((n_tokens, hidden))
        return 1000 + n_tokens

    def close(self) -> None:
        self.closed = True


def test_frame_round_trip_over_partial_socket_reads():
    left, right = socket.socketpair()
    try:
        frame = RelayFrame(RelayFrameKind.HIDDEN, 7, n_tokens=2, payload=b"abcdefgh")
        send_frame(left, frame)
        received = recv_frame(right)
    finally:
        left.close()
        right.close()

    assert received == frame


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (struct.pack("!4sBBHIIQ", b"BAD!", 1, 1, 0, 0, 1, 0), "invalid_magic"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 2, 1, 0, 0, 1, 0), "unsupported_version"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 99, 0, 0, 1, 0), "unknown_frame_kind"),
        (struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 1, 0, 0, 1, 99), "payload_too_large"),
    ],
)
def test_invalid_headers_fail_before_payload_allocation(header: bytes, reason: str):
    left, right = socket.socketpair()
    try:
        left.sendall(header)
        with pytest.raises(RelayProtocolError, match=reason):
            recv_frame(right, max_payload_bytes=16)
    finally:
        left.close()
        right.close()


def test_truncated_payload_is_rejected():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, 1, 1, 0, 0, 1, 8) + b"four")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(RelayProtocolError, match="connection_closed_mid_frame"):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_loopback_client_bridge_preserves_order_and_closes_runner():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    runner = _FakeRunner()
    results = []

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            results.append(serve_relay_connection(connection, runner, n_embd=2, max_tokens=4))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=4) as client:
            assert client.request_token(b"\x00" * expected_hidden_bytes(1, 2), n_tokens=1) == 1001
            assert client.request_token(b"\x01" * expected_hidden_bytes(3, 2), n_tokens=3) == 1003
    finally:
        thread.join(timeout=2)
        listener.close()

    assert not thread.is_alive()
    assert [request[0] for request in runner.requests] == [1, 3]
    assert runner.closed is True
    assert results[0].closed_cleanly is True
    assert results[0].frames == 2
    assert results[0].tokens == 4


def test_bridge_rejects_shape_mismatch_and_returns_error_frame():
    left, right = socket.socketpair()
    runner = _FakeRunner()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(serve_relay_connection(right, runner, n_embd=2)),
        daemon=True,
    )
    thread.start()
    try:
        send_frame(left, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=2, payload=b"too-short"))
        response = recv_frame(left)
    finally:
        thread.join(timeout=2)
        left.close()
        right.close()

    assert response.kind == RelayFrameKind.ERROR
    assert response.payload == b"hidden_payload_size_mismatch"
    assert result[0].closed_cleanly is False
    assert runner.requests == []


def test_non_loopback_endpoints_are_rejected_without_opening_a_socket():
    with pytest.raises(RelayProtocolError, match="non_loopback_bind_rejected"):
        open_loopback_listener("100.100.52.106", 0)
    with pytest.raises(RelayProtocolError, match="non_loopback_endpoint_rejected"):
        RelayTcpClient("100.100.52.106", 12345, n_embd=2)


def test_client_rejects_local_shape_and_limit_errors_before_sending():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    accepted = []
    thread = threading.Thread(target=lambda: accepted.append(listener.accept()), daemon=True)
    thread.start()
    client = RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=2)
    try:
        with pytest.raises(RelayProtocolError, match="hidden_payload_size_mismatch"):
            client.request_token(b"bad", n_tokens=1)
        with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
            client.request_token(b"", n_tokens=3)
    finally:
        client._sock.close()
        listener.close()
        thread.join(timeout=2)
        for connection, _ in accepted:
            connection.close()


def test_tokens_frame_roundtrip_helper():
    """★ P4.5 上游段请求：`TOKENS` 帧的 payload 编解码（紧凑 i32 数组）。"""
    from src.relay_transport import decode_tokens, encode_tokens

    payload = encode_tokens([1, 42, 151936])
    assert len(payload) == 12
    assert decode_tokens(payload, limit=8) == [1, 42, 151936]


def test_tokens_frame_rejects_bad_payload_shape():
    """长度不是 4 的倍数 / 空 payload / 超限都必须 fail-loud（不猜语义）。"""
    from src.relay_transport import decode_tokens, encode_tokens

    with pytest.raises(RelayProtocolError, match="invalid_token_response"):
        decode_tokens(b"", limit=8)
    with pytest.raises(RelayProtocolError, match="invalid_token_response"):
        decode_tokens(b"\x01\x02\x03", limit=8)
    with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
        decode_tokens(encode_tokens([1] * 9), limit=8)
    with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
        encode_tokens([])


def test_tokens_frame_kind_is_distinct():
    """新帧类型不得与既有 HIDDEN / TOKEN / HIDDEN_SEQ 碰撞（协议兼容的硬约束）。"""
    kinds = {RelayFrameKind.HIDDEN, RelayFrameKind.TOKEN, RelayFrameKind.CLOSE,
             RelayFrameKind.ERROR, RelayFrameKind.HIDDEN_SEQ, RelayFrameKind.TOKENS}
    assert len(kinds) == 6
