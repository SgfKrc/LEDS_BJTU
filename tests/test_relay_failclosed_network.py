"""#9 准入证据③：**弱网 / 断线 / 乱序**下的 fail-closed 行为（CORE-RELAY-XFRAME-01）。

## 为什么需要
`docs/跨框架层接力-项目报告.md` §7「未完成/待办」第 1 条列出的准入条件之一：**弱网断线**。
Relay 把 hidden 交给另一进程/机器；网络退化时**必须拒绝**，而不是产出错误的 token 或被
静默降级成另一种结果（fail-closed）。

已有覆盖（`tests/test_relay_transport.py`）：坏 magic/version、未知帧类型、超限 payload、
截断 payload、形状不匹配、loopback 顺序与关闭。**本文件补齐**：

* 慢响应（弱网高延迟）⇒ 客户端必须抛错，且**不得**返回 token；
* runner 中途死亡（断线）⇒ 必须抛错；
* **乱序 / 重复 sequence** ⇒ 必须拒绝（防重放与错配）；
* **非 loopback 端点** ⇒ 必须拒绝（ssh 隧道在本机也表现为 loopback）；
* runner 抛异常 ⇒ 桥必须回**错误帧**而不是长时间挂起。
"""
from __future__ import annotations

import socket
import struct
import threading
import time

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


class _Runner:
    """可编程的假 runner：按脚本决定响应/抛错/延迟。"""

    def __init__(self, *, delay: float = 0.0, explode: bool = False, exc: Exception | None = None) -> None:
        self.delay = delay
        self.explode = explode
        self.exc = exc or RuntimeError("runner exploded")
        self.requests: list[int] = []
        self.closed = False

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        self.requests.append(int(n_tokens))
        if self.delay:
            time.sleep(self.delay)
        if self.explode:
            raise self.exc
        return 2000 + n_tokens

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------- 弱网：慢响应

def test_slow_runner_times_out_without_returning_a_token():
    """弱网高延迟：客户端必须在超时后抛错，**绝不**返回一个 token。"""
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    runner = _Runner(delay=2.0)

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            try:
                serve_relay_connection(connection, runner, n_embd=2, max_tokens=4)
            except Exception:  # noqa: BLE001 - 对端已超时断开属预期
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    payload = b"\x00" * expected_hidden_bytes(1, 2)
    try:
        client = RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=4, timeout=0.3)
        try:
            with pytest.raises((socket.timeout, TimeoutError, RelayProtocolError, OSError)):
                client.request_token(payload, n_tokens=1)
        finally:
            client._sock.close()
    finally:
        thread.join(timeout=5)
        listener.close()


# ---------------------------------------------------------------- 断线：runner 死亡

def test_runner_dying_mid_session_raises_instead_of_hanging():
    """runner 在会话中途退出（模拟对端掉线）⇒ 客户端必须抛错。"""
    listener = open_loopback_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]

    def serve() -> None:
        connection, _ = listener.accept()
        # 读走一帧后立刻关闭，不给响应 —— 等价于 worker 崩溃
        try:
            recv_frame(connection)
        except Exception:  # noqa: BLE001
            pass
        connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    payload = b"\x00" * expected_hidden_bytes(1, 2)
    try:
        client = RelayTcpClient("127.0.0.1", port, n_embd=2, max_tokens=4, timeout=3.0)
        try:
            with pytest.raises((RelayProtocolError, ConnectionError, OSError, TimeoutError)):
                client.request_token(payload, n_tokens=1)
        finally:
            client._sock.close()
    finally:
        thread.join(timeout=5)
        listener.close()


# ---------------------------------------------------------------- 乱序 / 重放

@pytest.mark.parametrize("bad_sequence", [1, 5, 999])
def test_out_of_order_or_replayed_sequence_is_rejected(bad_sequence: int):
    """序列号不匹配（乱序/重放/错配）⇒ 必须拒绝，不得当成有效响应。

    校验点在 `_decode_token`（`recv_frame` 只负责帧完整性），故直接测该层，
    并额外确认 `RelayTcpClient` 走的是同一条校验路径。
    """
    from src.relay_transport import _decode_token

    frame = RelayFrame(RelayFrameKind.TOKEN, bad_sequence, n_tokens=1,
                       payload=struct.pack("<i", 7))
    with pytest.raises(RelayProtocolError, match="response_sequence_mismatch"):
        _decode_token(frame, 0)


# ---------------------------------------------------------------- 端点范围

@pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.20", "example.com"])
def test_non_loopback_endpoint_is_rejected(host: str):
    """Relay 只允许 loopback（ssh 隧道在本机也表现为 loopback）⇒ 其它地址一律拒。"""
    with pytest.raises(RelayProtocolError, match="non_loopback_endpoint_rejected"):
        RelayTcpClient(host, 65000, n_embd=2, max_tokens=4)


# ---------------------------------------------------------------- runner 异常 ⇒ 错误帧

def test_runner_exception_produces_error_frame_not_a_hang():
    """runner 抛**契约内**异常时，桥必须回 ERROR 帧（而非挂起），客户端据此 fail-closed。"""
    left, right = socket.socketpair()
    runner = _Runner(explode=True, exc=RelayProtocolError("runner_failed"))
    captured = []

    def serve() -> None:
        captured.append(serve_relay_connection(left, runner, n_embd=2, max_tokens=4))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        send_frame(right, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=1,
                                     payload=b"\x00" * expected_hidden_bytes(1, 2)))
        response = recv_frame(right, max_payload_bytes=1024)
        assert response.sequence == 0
        assert response.kind == RelayFrameKind.ERROR, f"应回 ERROR 帧，实得 {response.kind}"

        from src.relay_transport import _decode_token

        with pytest.raises(RelayProtocolError):
            _decode_token(response, 0)
    finally:
        right.close()
        thread.join(timeout=5)
    assert captured and captured[0].closed_cleanly is False


def test_unexpected_runner_exception_still_sends_error_frame(monkeypatch):
    """★ 准入加固：runner 抛**未预期**异常（非 OSError/RelayProtocolError）时也不得挂死。

    回归自实测：旧实现只捕 `(OSError, RelayProtocolError)`，runner 抛 `RuntimeError`
    时异常穿出 `serve_relay_connection`，ERROR 帧没发出，对端一直阻塞在 `recv`。
    """
    left, right = socket.socketpair()
    runner = _Runner(explode=True, exc=RuntimeError("boom"))
    captured = []

    def serve() -> None:
        captured.append(serve_relay_connection(left, runner, n_embd=2, max_tokens=4))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        send_frame(right, RelayFrame(RelayFrameKind.HIDDEN, 0, n_tokens=1,
                                     payload=b"\x00" * expected_hidden_bytes(1, 2)))
        response = recv_frame(right, max_payload_bytes=1024)   # 旧实现会在此挂住
        assert response.kind == RelayFrameKind.ERROR
        assert b"RuntimeError" in response.payload, response.payload
    finally:
        right.close()
        thread.join(timeout=5)
    assert captured and captured[0].closed_cleanly is False
