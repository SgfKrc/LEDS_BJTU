"""Contracts for the bounded two-node readiness preflight."""

from __future__ import annotations

import json

from scripts import cluster_node_preflight as preflight
from network_path import TailscaleNetcheckObservation, TailscaleStatusObservation


def _status(**_kwargs):
    return TailscaleStatusObservation("available", _payload={
        "BackendState": "Running",
        "Self": {
            "Online": True,
            "TailscaleIPs": ["100.64.0.10", "fd7a:115c:a1e0::10"],
        },
        "Peer": {},
    })


def _netcheck(**_kwargs):
    return TailscaleNetcheckObservation(
        "available", udp=True, ipv4=True, ipv6=True, nearest_derp_available=True,
    )


def test_client_preflight_probes_only_the_approved_ipv6_endpoint():
    calls = []

    def endpoint_reporter(host, port, **kwargs):
        calls.append((host, port, kwargs["timeout_seconds"]))
        return {
            "endpoint": {"role": "master", "host_scope": "tailscale_ipv6", "port": port},
            "tcp_probe": {"state": "available", "reason": None, "elapsed_ms": 1.0},
            "path": {"path_kind": "tailscale_direct", "availability": "available"},
            "ready": True,
        }

    report = preflight.build_report(
        role="client",
        master_host="fd7a:115c:a1e0::1234",
        require_tailnet=True,
        require_ipv6=True,
        status_collector=_status,
        netcheck_collector=_netcheck,
        endpoint_reporter=endpoint_reporter,
    )

    assert calls == [
        ("fd7a:115c:a1e0::1234", 8000, 2.0),
        ("fd7a:115c:a1e0::1234", 8888, 2.0),
    ]
    assert report["passed"] is True
    assert report["checks"] == {
        "tailscale_cli": True,
        "tailscale_running": True,
        "tailscale_self_online": True,
        "ipv6_socket": True,
        "local_tailnet_ipv6": True,
        "master_api_tcp": True,
        "master_tcp_tcp": True,
        "tailnet_endpoint": True,
        "ipv6_endpoint": True,
    }
    assert "fd7a:115c:a1e0::1234" not in json.dumps(report)


def test_client_preflight_fails_closed_without_master_host():
    report = preflight.build_report(
        role="client",
        status_collector=_status,
        netcheck_collector=_netcheck,
    )

    assert report["passed"] is False
    assert report["checks"]["master_endpoint_provided"] is False
    assert report["endpoints"] == {}


def test_cpu_sidecar_is_ready_without_cuda():
    report = preflight.build_report(
        role="master",
        sidecar="qwen3",
        status_collector=_status,
        netcheck_collector=_netcheck,
        sidecar_checker=lambda kind: {
            "kind": kind,
            "venv_present": True,
            "ready": True,
            "missing_modules": [],
            "torch": {"version": "2.13.0+cpu", "cuda_available": False},
        },
    )

    assert report["passed"] is True
    assert report["checks"]["sidecars"] is True
    assert report["sidecars"][0]["torch"]["cuda_available"] is False


def test_tailscale_cli_response_is_not_treated_as_a_running_daemon():
    report = preflight.build_report(
        role="master",
        status_collector=lambda **_kwargs: TailscaleStatusObservation(
            "available", _payload={"BackendState": "NoState", "Self": {"Online": False}},
        ),
        netcheck_collector=_netcheck,
    )

    assert report["passed"] is False
    assert report["checks"]["tailscale_cli"] is True
    assert report["checks"]["tailscale_running"] is False
    assert report["checks"]["tailscale_self_online"] is False


def test_required_ipv6_rejects_ipv4_master_scope():
    def endpoint_reporter(_host, port, **_kwargs):
        return {
            "endpoint": {"role": "master", "host_scope": "tailscale_ipv4", "port": port},
            "tcp_probe": {"state": "available", "reason": None, "elapsed_ms": 1.0},
            "path": {"path_kind": "tailscale_direct", "availability": "available"},
            "ready": True,
        }

    report = preflight.build_report(
        role="client",
        master_host="100.64.0.1",
        require_ipv6=True,
        status_collector=_status,
        netcheck_collector=_netcheck,
        endpoint_reporter=endpoint_reporter,
    )

    assert report["passed"] is False
    assert report["checks"]["ipv6_endpoint"] is False
