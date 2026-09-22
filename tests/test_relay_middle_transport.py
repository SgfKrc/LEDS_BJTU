"""跨机中间段往返（HIDDEN → HIDDEN）的传输层守卫。

为什么需要（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §4 P2「多段拓扑」）：
「1 个 torch 上游 + n 个 llama 下游」里，**中间段**要经网络交出 hidden（HIDDEN → HIDDEN），
**末段**才回 token（HIDDEN → TOKEN）。前者是新加的 `RelayTcpClient.request_hidden` /
`serve_relay_middle_connection`，必须与既有帧契约同等地 fail-closed。

这些用例不需要模型：用一个假 runner（把输入 hidden 逐字节加 1）验证往返与全部校验分支。
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_transport import (  # noqa: E402
    RelayProtocolError,
    RelayTcpClient,
    encode_hidden_seq,
    expected_hidden_bytes,
    open_loopback_listener,
    serve_relay_middle_connection,
)


class _FakeMiddleRunner:
    """逐字节 +1：既能验证往返正确，也能暴露"原样返回"这类假通过。"""

    def __init__(self) -> None:
        self.requests: list[tuple[int, bytes]] = []
        self.closed = False
        self.reset_calls = 0

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        self.requests.append((n_tokens, hidden))
        return bytes((value + 1) % 256 for value in hidden)

    def close(self) -> None:
        self.closed = True


class _FakeSeqMiddleRunner(_FakeMiddleRunner):
    def request_hidden_seq(self, hidden: bytes, *, n_tokens: int,
                           meta: dict[str, object]) -> bytes:
        self.requests.append((n_tokens, hidden))
        self.meta = meta
        return bytes((value + 1) % 256 for value in hidden)


def _serve_once(listener: socket.socket, runner, n_embd: int):
    result: dict[str, object] = {}

    def _run() -> None:
        sock, _ = listener.accept()
        try:
            result["bridge"] = serve_relay_middle_connection(sock, runner, n_embd=n_embd,
                                                             max_tokens=64)
        finally:
            sock.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result


def test_middle_round_trip_returns_hidden_and_closes_cleanly():
    n_embd, n_tokens = 8, 3
    runner = _FakeMiddleRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    try:
        thread, result = _serve_once(listener, runner, n_embd)
        payload = bytes(range(expected_hidden_bytes(n_tokens, n_embd)))
        with RelayTcpClient("127.0.0.1", port, n_embd=n_embd) as client:
            produced = client.request_hidden(payload, n_tokens=n_tokens)
        thread.join(timeout=10)
        assert produced == bytes((value + 1) % 256 for value in payload)
        assert runner.requests == [(n_tokens, payload)]
        assert runner.closed is True
        bridge = result["bridge"]
        assert bridge.frames == 1 and bridge.tokens == n_tokens and bridge.closed_cleanly
    finally:
        listener.close()


def test_middle_handles_multiple_rounds_in_one_session():
    n_embd, n_tokens = 4, 2
    runner = _FakeMiddleRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    try:
        thread, result = _serve_once(listener, runner, n_embd)
        with RelayTcpClient("127.0.0.1", port, n_embd=n_embd) as client:
            first = client.request_hidden(b"\x00" * expected_hidden_bytes(n_tokens, n_embd),
                                          n_tokens=n_tokens)
            second = client.request_hidden(b"\x01" * expected_hidden_bytes(n_tokens, n_embd),
                                           n_tokens=n_tokens)
        thread.join(timeout=10)
        assert first == b"\x01" * expected_hidden_bytes(n_tokens, n_embd)
        assert second == b"\x02" * expected_hidden_bytes(n_tokens, n_embd)
        assert result["bridge"].frames == 2
    finally:
        listener.close()


def test_middle_accepts_hidden_seq_metadata_with_frame_overhead():
    n_embd, n_tokens = 8, 4
    runner = _FakeSeqMiddleRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    payload = bytes(range(expected_hidden_bytes(n_tokens, n_embd)))
    meta = {"seq_ids": [0, 0, 1, 1], "positions": [0, 1, 0, 1]}
    try:
        thread, result = _serve_once(listener, runner, n_embd)
        with RelayTcpClient("127.0.0.1", port, n_embd=n_embd) as client:
            produced = client.request_hidden_seq(payload, n_tokens=n_tokens, meta=meta)
        thread.join(timeout=10)
        assert produced == bytes((value + 1) % 256 for value in payload)
        assert runner.meta == meta
        assert result["bridge"].payload_bytes == len(
            encode_hidden_seq(payload, n_tokens=n_tokens, meta=meta))
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        ({"seq_ids": [0]}, "hidden_seq_meta_shape_invalid"),
        ({"positions": [0, -1]}, "hidden_seq_meta_shape_invalid"),
        ({"seq_ids": [0, 1], "extra": [0, 1]}, "hidden_seq_meta_unknown"),
    ],
)
def test_hidden_seq_metadata_is_fail_closed(meta, reason):
    from src.relay_transport import encode_hidden_seq

    with pytest.raises(RelayProtocolError, match=reason):
        encode_hidden_seq(b"\x00" * 16, n_tokens=2, meta=meta)


def test_middle_rejects_outer_hidden_seq_token_count_mismatch():
    from src.relay_transport import RelayFrame, RelayFrameKind, recv_frame, send_frame

    n_embd = 4
    runner = _FakeSeqMiddleRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    try:
        thread, result = _serve_once(listener, runner, n_embd)
        sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        try:
            payload = encode_hidden_seq(b"\x00" * expected_hidden_bytes(2, n_embd),
                                        n_tokens=2, meta={"seq_ids": [0, 0]})
            send_frame(sock, RelayFrame(RelayFrameKind.HIDDEN_SEQ, 0, n_tokens=1,
                                        payload=payload))
            response = recv_frame(sock)
        finally:
            sock.close()
        thread.join(timeout=10)
        assert response.kind == RelayFrameKind.ERROR
        assert response.payload == b"hidden_seq_token_count_mismatch"
        assert result["bridge"].closed_cleanly is False
    finally:
        listener.close()


def test_client_rejects_wrong_payload_size_before_sending():
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    try:
        thread, _result = _serve_once(listener, _FakeMiddleRunner(), 8)
        with RelayTcpClient("127.0.0.1", port, n_embd=8, max_tokens=4,
                            timeout=5.0) as client:
            with pytest.raises(RelayProtocolError, match="hidden_payload_size_mismatch"):
                client.request_hidden(b"\x00" * 4, n_tokens=2)   # 应为 2*8*4 字节
            with pytest.raises(RelayProtocolError, match="token_count_exceeds_limit"):
                client.request_hidden(b"\x00" * 64, n_tokens=5)  # 上限 4
        thread.join(timeout=10)
    finally:
        listener.close()


def test_middle_server_rejects_non_hidden_frames():
    """服务端只接受 HIDDEN 帧；别的帧必须作为协议错误收场（fail-closed）。"""
    import struct  # noqa: PLC0415

    from src.relay_transport import (  # noqa: PLC0415
        RELAY_WIRE_MAGIC,
        RELAY_WIRE_VERSION,
    )

    n_embd = 4
    runner = _FakeMiddleRunner()
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    try:
        thread, result = _serve_once(listener, runner, n_embd)
        sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        try:
            header = struct.pack("!4sBBHIIQ", RELAY_WIRE_MAGIC, RELAY_WIRE_VERSION,
                                 2, 0, 0, 0, 0)  # kind=TOKEN：服务端只收 HIDDEN
            sock.sendall(header)
            response = sock.recv(4096)
        finally:
            sock.close()
        thread.join(timeout=10)
        assert response, "服务端必须回一帧（ERROR）而不是静默断开"
        bridge = result["bridge"]
        assert bridge.closed_cleanly is False
        assert bridge.error  # 具名错误码
    finally:
        listener.close()
