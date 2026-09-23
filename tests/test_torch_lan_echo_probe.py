import socket
import struct
import threading

import pytest

from scripts.torch_lan_echo_probe import recv_exact, serve_connection, summarize


def test_recv_exact_reads_requested_byte_count():
    left, right = socket.socketpair()
    try:
        right.sendall(b"abc")
        right.sendall(b"def")
        assert recv_exact(left, 6) == b"abcdef"
    finally:
        left.close()
        right.close()


def test_echo_server_returns_exact_framed_payload():
    client, server = socket.socketpair()
    worker = threading.Thread(target=serve_connection, args=(server,))
    worker.start()
    payload = bytes(range(256)) * 4
    try:
        client.sendall(struct.pack("!I", len(payload)) + payload)
        assert recv_exact(client, 4 + len(payload)) == struct.pack("!I", len(payload)) + payload
    finally:
        client.close()
        worker.join(timeout=2)
        server.close()
    assert not worker.is_alive()


def test_recv_exact_rejects_oversized_frame():
    left, right = socket.socketpair()
    try:
        with pytest.raises(ValueError, match="invalid frame length"):
            recv_exact(left, 16 * 1024 * 1024 + 5)
    finally:
        left.close()
        right.close()


def test_summarize_retains_raw_samples_and_dispersion():
    report = summarize([1.0, 2.0, 3.0])
    assert report["samples_ms"] == [1.0, 2.0, 3.0]
    assert report["median_ms"] == 2.0
    assert report["count"] == 3
    assert report["cv"] > 0
