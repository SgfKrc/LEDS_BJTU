from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tcp_comm
import transport_port


def test_scheduler_uses_transport_port_boundary() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "scheduler.py").read_text(
        encoding="utf-8",
    )

    assert "from tcp_comm import" not in source
    assert "TCPServer(" not in source
    assert "serialize_tensor_fast" not in source
    assert "deserialize_tensor_fast" not in source


def test_create_server_resolves_current_tcp_server(monkeypatch) -> None:
    calls: list[tuple[str | None, int | None]] = []

    class FakeServer:
        def __init__(self, host, port):
            calls.append((host, port))

    monkeypatch.setattr(tcp_comm, "TCPServer", FakeServer)

    result = transport_port.create_server("127.0.0.1", 43123)

    assert isinstance(result, FakeServer)
    assert calls == [("127.0.0.1", 43123)]


def test_create_client_resolves_current_tcp_client(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(tcp_comm, "TCPClient", FakeClient)

    result = transport_port.create_client(
        server_host="127.0.0.1",
        server_port=43123,
        client_id="node-test",
    )

    assert isinstance(result, FakeClient)
    assert calls == [{
        "server_host": "127.0.0.1",
        "server_port": 43123,
        "client_id": "node-test",
    }]


def test_transport_port_keeps_protocol_values_from_tcp_comm() -> None:
    assert transport_port.MessageType is tcp_comm.MessageType
