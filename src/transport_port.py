"""Control-plane port for the cluster transport implementation.

Scheduler and other control-plane modules depend on this small boundary rather
than importing the legacy TCP implementation directly.  The adapter resolves
the implementation at call time so test doubles and a future transport can be
selected without changing control-plane code.
"""

from __future__ import annotations

from typing import Any


def _implementation() -> Any:
    import tcp_comm

    return tcp_comm


# Message values are protocol data, not a transport lifecycle object.  Keep the
# existing enum identity so persisted/message-level behavior is unchanged.
MessageType = _implementation().MessageType


def create_server(host: str | None = None, port: int | None = None) -> Any:
    return _implementation().TCPServer(host, port)


def create_client(**kwargs: Any) -> Any:
    return _implementation().TCPClient(**kwargs)


def compute_local_model_sha256(
    model_path: str | None = None,
    model_id: str | None = None,
) -> str:
    return str(
        _implementation().TCPClient._compute_local_model_sha256(
            model_path=model_path,
            model_id=model_id,
        )
        or ""
    )


def serialize_tensor(tensor: Any) -> bytes:
    return _implementation().serialize_tensor_fast(tensor)


def deserialize_tensor(data: bytes) -> Any:
    return _implementation().deserialize_tensor_fast(data)


def get_cluster_secret() -> str:
    return str(_implementation()._get_cluster_secret() or "")


def detect_lan_ip() -> str:
    return str(_implementation().detect_lan_ip() or "")


def get_mac_addresses() -> list[str]:
    return list(_implementation().get_mac_addresses() or [])


__all__ = [
    "MessageType",
    "compute_local_model_sha256",
    "create_client",
    "create_server",
    "deserialize_tensor",
    "detect_lan_ip",
    "get_cluster_secret",
    "get_mac_addresses",
    "serialize_tensor",
]
