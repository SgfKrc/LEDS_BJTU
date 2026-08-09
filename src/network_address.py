"""Canonical host and URL helpers for IPv4/IPv6 endpoints."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable


TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE_IPV6_NETWORK = ipaddress.ip_network("fd7a:115c:a1e0::/48")


def canonical_host(host: str | None) -> str:
    """Return the bare host used in config/storage, without IPv6 brackets."""
    value = (host or "").strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def format_url_host(host: str | None) -> str:
    """Format a bare host for use in a URL authority component."""
    value = canonical_host(host)
    if ":" not in value:
        return value
    # RFC 6874 requires a literal percent in an IPv6 zone id to be URL encoded.
    value = value.replace("%", "%25")
    return f"[{value}]"


def build_url(
    scheme: str,
    host: str | None,
    port: int,
    path: str = "",
) -> str:
    """Build a URL from separate endpoint fields without corrupting IPv6."""
    normalized_scheme = (scheme or "").strip().lower()
    if normalized_scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError(f"unsupported URL scheme: {scheme}")
    authority_host = format_url_host(host)
    if not authority_host:
        raise ValueError("host is required")
    normalized_port = int(port)
    if not 1 <= normalized_port <= 65535:
        raise ValueError(f"invalid port: {port}")
    normalized_path = path or ""
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"{normalized_scheme}://{authority_host}:{normalized_port}{normalized_path}"


def is_tailscale_ip(host: str | None) -> bool:
    """Return whether host is a Tailscale IPv4 or IPv6 address."""
    value = canonical_host(host)
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return address in TAILSCALE_IPV4_NETWORK or address in TAILSCALE_IPV6_NETWORK


def create_listen_sockets(
    hosts: Iterable[str],
    port: int,
    *,
    backlog: int = 2048,
    allow_partial: bool = False,
) -> list[socket.socket]:
    """Create pre-bound TCP sockets with stable cross-platform dual-stack rules."""
    sockets: list[socket.socket] = []
    actual_port = int(port)
    errors: list[OSError] = []
    for host in hosts:
        bind_host = canonical_host(host)
        family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((bind_host, actual_port))
            if actual_port == 0:
                actual_port = int(sock.getsockname()[1])
            sock.listen(backlog)
            sock.set_inheritable(True)
            sockets.append(sock)
        except OSError as exc:
            errors.append(exc)
            sock.close()
            if not allow_partial:
                for opened in sockets:
                    opened.close()
                raise
    if not sockets:
        if errors:
            raise errors[-1]
        raise OSError("no listen hosts configured")
    return sockets
