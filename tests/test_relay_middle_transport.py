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
