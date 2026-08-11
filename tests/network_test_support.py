"""Socket helpers that keep test-only port reservations race-free."""

from __future__ import annotations

import socket


_reserved_unreachable_sockets: list[socket.socket] = []


def reserve_unreachable_port() -> int:
    """Reserve a loopback port without listening so connects reliably fail.

    The caller must invoke ``release_reserved_unreachable_ports`` at test
    teardown. Keeping the socket bound removes the close-then-rebind race.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _reserved_unreachable_sockets.append(sock)
    return int(sock.getsockname()[1])


def release_reserved_unreachable_ports() -> None:
    while _reserved_unreachable_sockets:
        _reserved_unreachable_sockets.pop().close()
