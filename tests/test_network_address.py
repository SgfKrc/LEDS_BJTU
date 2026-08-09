import socket

import pytest

from src.network_address import (
    build_url,
    canonical_host,
    create_listen_sockets,
    format_url_host,
    is_tailscale_ip,
)


def test_canonical_and_url_hosts():
    assert canonical_host("[::1]") == "::1"
    assert canonical_host("fd7a:115c:a1e0::1") == "fd7a:115c:a1e0::1"
    assert format_url_host("127.0.0.1") == "127.0.0.1"
    assert format_url_host("::1") == "[::1]"
    assert format_url_host("[::1]") == "[::1]"
    assert format_url_host("fe80::1%Ethernet") == "[fe80::1%25Ethernet]"


def test_build_url_supports_ipv4_ipv6_and_dns():
    assert build_url("http", "127.0.0.1", 8000, "/health") == (
        "http://127.0.0.1:8000/health"
    )
    assert build_url("http", "fd7a:115c:a1e0::1", 8000, "health") == (
        "http://[fd7a:115c:a1e0::1]:8000/health"
    )
    assert build_url("https", "master.example.ts.net", 443) == (
        "https://master.example.ts.net:443"
    )


def test_tailscale_ipv4_and_ipv6_ranges():
    assert is_tailscale_ip("100.64.0.1")
    assert is_tailscale_ip("fd7a:115c:a1e0::1")
    assert not is_tailscale_ip("192.168.1.2")
    assert not is_tailscale_ip("2001:db8::1")


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 unavailable")
def test_prebound_dual_stack_sockets_share_port_and_are_v6_only():
    sockets = create_listen_sockets(["0.0.0.0", "::"], 0)
    try:
        assert len(sockets) == 2
        assert len({sock.getsockname()[1] for sock in sockets}) == 1
        ipv6 = next(sock for sock in sockets if sock.family == socket.AF_INET6)
        assert ipv6.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
    finally:
        for sock in sockets:
            sock.close()
