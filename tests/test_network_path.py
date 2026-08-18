from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from src.network_path import (
    HeartbeatQualityWindow,
    TcpProbeObservation,
    TrustedEndpoint,
    build_client_network_path_view,
    classify_trusted_path,
    collect_tailscale_netcheck,
    collect_tailscale_status,
    network_path_diagnostic_json,
    probe_trusted_tcp,
    sanitize_network_path_view,
)


def _ack(
    window: HeartbeatQualityWindow,
    generation: int,
    sent_at: float,
    rtt_ms: float,
) -> float | None:
    assert window.record_send(generation, sent_at) is True
    return window.record_ack(generation, sent_at, sent_at + rtt_ms / 1000.0)


def test_heartbeat_quality_first_sample_and_percentiles():
    window = HeartbeatQualityWindow(max_samples=4, max_rtt_ms=1_000)
    assert window.begin_generation(1) is True

    assert _ack(window, 1, 10.0, 125.0) == pytest.approx(125.0)

    assert window.snapshot() == {
        "schema_version": 1,
        "generation": 1,
        "sample_window_size": 4,
        "sample_count": 1,
        "rtt_ms_p50": 125.0,
        "rtt_ms_p95": 125.0,
        "jitter_ms_p95": None,
        "consecutive_stalls": 0,
        "stalls_in_window": 0,
        "consecutive_reconnects": 0,
        "reconnects_in_window": 0,
        "pending_heartbeat": False,
    }


def test_heartbeat_quality_rolls_bounded_rtt_and_jitter_window():
    window = HeartbeatQualityWindow(max_samples=4)
    window.begin_generation(1)

    for index, rtt_ms in enumerate([10.0, 20.0, 40.0, 70.0, 110.0]):
        _ack(window, 1, 10.0 + index, rtt_ms)

    snapshot = window.snapshot()
    assert snapshot["sample_count"] == 4
    assert snapshot["rtt_ms_p50"] == 40.0
    assert snapshot["rtt_ms_p95"] == 110.0
    assert snapshot["jitter_ms_p95"] == 40.0


def test_heartbeat_quality_rejects_clock_rollback_and_outlier():
    window = HeartbeatQualityWindow(max_samples=4, max_rtt_ms=500)
    window.begin_generation(1)
    window.record_send(1, 10.0)

    assert window.record_ack(1, 10.0, 9.9) is None
    assert window.record_ack(1, 10.0, 11.0) is None
    assert window.snapshot()["pending_heartbeat"] is True

    window.record_send(1, 12.0)
    assert window.record_ack(1, 12.0, 12.1) == pytest.approx(100.0)
    snapshot = window.snapshot()
    assert snapshot["sample_count"] == 1
    assert snapshot["stalls_in_window"] == 1
    assert snapshot["consecutive_stalls"] == 0


@pytest.mark.parametrize("echoed", [-1.0, float("nan"), True, "10.0", "invalid"])
def test_heartbeat_quality_rejects_malformed_echo(echoed):
    window = HeartbeatQualityWindow(max_samples=4)
    window.begin_generation(1)
    window.record_send(1, 10.0)

    assert window.record_ack(1, echoed, 10.1) is None
    assert window.snapshot()["sample_count"] == 0


def test_heartbeat_quality_counts_missing_heartbeats_until_valid_ack():
    window = HeartbeatQualityWindow(max_samples=8)
    window.begin_generation(1)

    window.record_send(1, 10.0)
    window.record_send(1, 11.0)
    window.record_send(1, 12.0)
    pending = window.snapshot()
    assert pending["consecutive_stalls"] == 2
    assert pending["stalls_in_window"] == 2

    assert window.record_ack(1, 12.0, 12.05) == pytest.approx(50.0)
    assert window.snapshot()["consecutive_stalls"] == 0


def test_heartbeat_quality_fences_old_generation_and_late_ack():
    window = HeartbeatQualityWindow(max_samples=8)
    window.begin_generation(1)
    window.record_send(1, 10.0)
    assert window.record_disconnect(1) is True
    assert window.record_disconnect(1) is False
    assert window.begin_generation(2) is True
    window.record_send(2, 20.0)

    assert window.record_ack(1, 10.0, 20.1) is None
    assert window.record_ack(2, 19.0, 20.1) is None
    pending = window.snapshot()
    assert pending["pending_heartbeat"] is True
    assert pending["consecutive_stalls"] == 1
    assert pending["consecutive_reconnects"] == 1
    assert pending["reconnects_in_window"] == 1

    assert window.record_ack(2, 20.0, 20.08) == pytest.approx(80.0)
    recovered = window.snapshot()
    assert recovered["consecutive_stalls"] == 0
    assert recovered["consecutive_reconnects"] == 0


def test_client_path_view_uses_existing_connection_and_redacts_quality():
    class Client:
        is_registered = True
        server_host = "127.0.0.1"
        server_port = 8888

        @staticmethod
        def get_network_quality_snapshot():
            return {
                "schema_version": 1,
                "generation": 2,
                "sample_window_size": 64,
                "sample_count": 2,
                "rtt_ms_p50": 12.0,
                "rtt_ms_p95": 18.0,
                "jitter_ms_p95": 6.0,
                "avg_rtt_ms": 13.0,
                "consecutive_stalls": 0,
                "stalls_in_window": 1,
                "consecutive_reconnects": 0,
                "reconnects_in_window": 1,
                "pending_heartbeat": False,
                "server_host": "must-not-escape",
                "relay_region": "must-not-escape",
            }

    view = build_client_network_path_view(Client())

    assert view["path_kind"] == "lan_direct"
    assert view["availability"] == "available"
    assert view["tcp_probe"]["reason"] == "existing_connection"
    assert view["quality"]["rtt_ms_p95"] == 18.0
    encoded = json.dumps(view)
    assert "127.0.0.1" not in encoded
    assert "must-not-escape" not in encoded


def test_client_path_view_does_not_claim_tailscale_direct_without_path_fact():
    client = SimpleNamespace(
        is_registered=True,
        server_host="100.64.0.10",
        server_port=8888,
    )

    view = build_client_network_path_view(client)

    assert view["path_kind"] == "unknown"
    assert view["availability"] == "degraded"
    assert view["endpoint"]["host_scope"] == "tailscale_ipv4"
    assert view["quality"] is None


def test_client_path_view_invalid_endpoint_degrades_without_raising():
    client = SimpleNamespace(
        is_registered=False,
        server_host="bad host",
        server_port=8888,
    )

    view = build_client_network_path_view(client)

    assert view["path_kind"] == "unknown"
    assert view["availability"] == "unknown"
    assert view["endpoint"] is None
    assert view["tcp_probe"]["reason"] == "invalid_endpoint"


def test_diagnostic_json_and_api_view_share_one_redaction_boundary():
    raw = {
        "schema_version": 999,
        "path_kind": "derp",
        "availability": "available",
        "endpoint": {
            "role": "master",
            "host_scope": "tailscale_ipv4",
            "port": 8888,
            "host": "100.64.0.55",
        },
        "tailscale": {
            "state": "available",
            "reason": "private-region",
            "tailnet": "secret-tailnet",
        },
        "tcp_probe": {
            "state": "available",
            "reason": "existing_connection",
            "elapsed_ms": 2.25,
            "address": "100.64.0.55",
        },
        "quality": {
            "rtt_ms_p95": 30.0,
            "jitter_ms_p95": 5.0,
            "raw_samples": ["secret"],
        },
        "relay_region": "private-region",
    }

    public = sanitize_network_path_view(raw)
    diagnostic = json.loads(network_path_diagnostic_json(raw))

    assert diagnostic == public
    encoded = json.dumps(public)
    for secret in ("100.64.0.55", "secret-tailnet", "private-region", "raw_samples"):
        assert secret not in encoded
    assert public["tailscale"]["reason"] is None


def test_runtime_client_projection_never_invokes_probe_or_tailscale(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("runtime projection must not perform external I/O")

    monkeypatch.setattr("src.network_path.collect_tailscale_status", unexpected)
    monkeypatch.setattr("src.network_path.collect_tailscale_netcheck", unexpected)
    monkeypatch.setattr("src.network_path.probe_trusted_tcp", unexpected)
    client = SimpleNamespace(
        is_registered=True,
        server_host="127.0.0.1",
        server_port=8888,
    )

    view = build_client_network_path_view(client)

    assert view["path_kind"] == "lan_direct"


def _completed(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def test_trusted_endpoint_normalizes_and_redacts_host():
    endpoint = TrustedEndpoint("[fd7a:115c:a1e0::10]", 443)

    assert endpoint.host == "fd7a:115c:a1e0::10"
    assert endpoint.public_descriptor() == {
        "role": "master",
        "host_scope": "tailscale_ipv6",
        "port": 443,
    }
    assert endpoint.host not in json.dumps(endpoint.public_descriptor())


@pytest.mark.parametrize("host", ["", "https://example.test", "127.0.0.1/path", "bad host"])
def test_trusted_endpoint_rejects_untrusted_host_forms(host: str):
    with pytest.raises(ValueError):
        TrustedEndpoint(host)


def test_status_command_uses_argument_vector_and_classifies_direct_tailnet_peer():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(
            json.dumps(
                {
                    "Peer": {
                        "peer-1": {
                            "TailscaleIPs": ["100.100.100.7"],
                            "CurAddr": "198.51.100.5:41641",
                            "Relay": "",
                        }
                    }
                }
            )
        )

    status = collect_tailscale_status(executable="tailscale-test", runner=fake_run)
    snapshot = classify_trusted_path(TrustedEndpoint("100.100.100.7"), tailscale_status=status)

    assert calls[0][0] == ["tailscale-test", "status", "--json"]
    assert "shell" not in calls[0][1]
    assert snapshot.path_kind == "tailscale_direct"
    assert snapshot.availability == "available"
    public = snapshot.public_view()
    assert "100.100.100.7" not in json.dumps(public)
    assert "198.51.100.5" not in json.dumps(public)


def test_status_command_decodes_utf8_bytes_independent_of_windows_locale():
    payload = json.dumps(
        {"Peer": {"peer-1": {"TailscaleIPs": ["100.100.100.7"], "CurAddr": ""}}}
    ).encode("utf-8")

    status = collect_tailscale_status(
        executable="tailscale-test",
        runner=lambda *_args, **_kwargs: _completed(payload),
    )

    assert status.state == "available"
    assert status.reason is None


def test_status_command_rejects_invalid_utf8_bytes_without_raising():
    status = collect_tailscale_status(
        executable="tailscale-test",
        runner=lambda *_args, **_kwargs: _completed(b"\xff\xfe"),
    )

    assert status.public_view() == {"state": "invalid", "reason": "non_text_output"}


def test_relay_path_is_classified_without_exposing_region_or_peer_address():
    status = collect_tailscale_status(
        executable="tailscale-test",
        runner=lambda *_args, **_kwargs: _completed(
            json.dumps(
                {
                    "Peer": {
                        "peer-1": {
                            "TailscaleIPs": ["100.100.100.8"],
                            "CurAddr": "203.0.113.3:41641",
                            "Relay": "derp-hk",
                        }
                    }
                }
            )
        ),
    )

    public = classify_trusted_path(
        TrustedEndpoint("100.100.100.8"), tailscale_status=status
    ).public_view()

    assert public["path_kind"] == "derp"
    serialized = json.dumps(public)
    assert "derp-hk" not in serialized
    assert "203.0.113.3" not in serialized


def test_magic_dns_peer_is_classified_from_status_instead_of_public_dns():
    status = collect_tailscale_status(
        executable="tailscale-test",
        runner=lambda *_args, **_kwargs: _completed(
            json.dumps(
                {
                    "Peer": {
                        "peer-1": {
                            "DNSName": "worker.tail123.ts.net.",
                            "TailscaleIPs": ["100.100.100.9"],
                            "CurAddr": "198.51.100.9:41641",
                            "Relay": "",
                        }
                    }
                }
            )
        ),
    )

    snapshot = classify_trusted_path(
        TrustedEndpoint("worker.tail123.ts.net"), tailscale_status=status
    )

    assert snapshot.path_kind == "tailscale_direct"
    assert snapshot.public_view()["endpoint"]["host_scope"] == "tailnet_dns"


@pytest.mark.parametrize(
    ("runner", "expected_state", "expected_reason"),
    [
        (lambda *_args, **_kwargs: _completed("not json"), "invalid", "invalid_json"),
        (lambda *_args, **_kwargs: _completed("", returncode=1), "command_failed", "nonzero_exit"),
    ],
)
def test_status_failures_are_structured_and_do_not_raise(runner, expected_state, expected_reason):
    observation = collect_tailscale_status(executable="tailscale-test", runner=runner)

    assert observation.public_view() == {"state": expected_state, "reason": expected_reason}


def test_missing_tailscale_binary_is_a_safe_unavailable_observation(monkeypatch):
    monkeypatch.setattr("src.network_path.find_tailscale_executable", lambda: None)

    observation = collect_tailscale_status()

    assert observation.public_view() == {"state": "unavailable", "reason": "executable_not_found"}


def test_tailscale_timeout_is_structured_and_does_not_expose_command_data():
    def timeout_runner(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 2, output="sensitive peer address")

    observation = collect_tailscale_status(executable="tailscale-test", runner=timeout_runner)

    assert observation.public_view() == {"state": "timeout", "reason": "timeout"}
    assert "sensitive" not in json.dumps(observation.public_view())


def test_netcheck_keeps_booleans_but_redacts_derp_name():
    observation = collect_tailscale_netcheck(
        executable="tailscale-test",
        runner=lambda *_args, **_kwargs: _completed(
            "UDP: false\nIPv4: yes, 203.0.113.10:1234\nIPv6: no\nNearest DERP: derp-hk\n"
        ),
    )

    public = observation.public_view()
    assert public == {
        "state": "available",
        "reason": None,
        "udp": False,
        "ipv4": True,
        "ipv6": False,
        "nearest_derp_available": True,
    }
    assert "derp-hk" not in json.dumps(public)


@pytest.mark.parametrize(
    ("endpoint", "probe", "expected_kind"),
    [
        (TrustedEndpoint("192.168.1.20"), TcpProbeObservation("available"), "lan_direct"),
        (TrustedEndpoint("8.8.8.8"), TcpProbeObservation("available"), "public_tcp_direct"),
        (TrustedEndpoint("gateway.example.test", role="gateway"), TcpProbeObservation("available"), "gateway_relay"),
    ],
)
def test_non_tailnet_paths_require_a_successful_exact_tcp_probe(endpoint, probe, expected_kind):
    snapshot = classify_trusted_path(endpoint, tcp_probe=probe)

    assert snapshot.path_kind == expected_kind
    assert snapshot.availability == "available"


def test_failed_or_missing_tcp_probe_never_claims_direct_path():
    endpoint = TrustedEndpoint("192.168.1.20")

    assert classify_trusted_path(endpoint).path_kind == "unknown"
    failed = classify_trusted_path(endpoint, tcp_probe=TcpProbeObservation("unavailable", "connect_failed"))
    assert failed.path_kind == "unknown"
    assert failed.availability == "degraded"


def test_tcp_probe_connects_once_to_exact_trusted_endpoint_and_closes_socket():
    calls = []

    class FakeSocket:
        def close(self):
            calls.append("closed")

    def connector(address, *, timeout):
        calls.append((address, timeout))
        return FakeSocket()

    timestamps = iter([10.0, 10.025])
    observation = probe_trusted_tcp(
        TrustedEndpoint("master.example.test", 8443),
        connector=connector,
        timeout_seconds=1.0,
        clock=lambda: next(timestamps),
    )

    assert calls == [(("master.example.test", 8443), 1.0), "closed"]
    assert observation.public_view() == {"state": "available", "reason": None, "elapsed_ms": 25.0}
