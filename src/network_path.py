"""Trusted, privacy-preserving network path observations for NW1.

This module intentionally does not start background probes or alter routing.
Callers provide an explicit trusted endpoint before a single TCP connect can be
attempted. Tailscale command output is reduced to path facts and never exposed
through the public snapshot helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

try:
    from .network_address import canonical_host, is_tailscale_ip
except ImportError:  # Script/PYTHONPATH=src compatibility.
    from network_address import canonical_host, is_tailscale_ip


PATH_KINDS = frozenset(
    {
        "lan_direct",
        "public_tcp_direct",
        "tailscale_direct",
        "derp",
        "gateway_relay",
        "unknown",
    }
)

_QUALITY_FLOAT_FIELDS = (
    "avg_rtt_ms",
    "rtt_ms_p50",
    "rtt_ms_p95",
    "jitter_ms_p95",
)
_QUALITY_COUNT_FIELDS = (
    "generation",
    "sample_window_size",
    "sample_count",
    "consecutive_stalls",
    "stalls_in_window",
    "consecutive_reconnects",
    "reconnects_in_window",
)
_HOST_SCOPES = frozenset(
    {
        "tailscale_ipv4",
        "tailscale_ipv6",
        "loopback_ipv4",
        "loopback_ipv6",
        "private_ipv4",
        "private_ipv6",
        "public_ipv4",
        "public_ipv6",
        "tailnet_dns",
        "dns",
    }
)
_OBSERVATION_REASONS = frozenset(
    {
        "not_collected",
        "executable_not_found",
        "timeout",
        "command_unavailable",
        "nonzero_exit",
        "non_text_output",
        "output_too_large",
        "invalid_json",
        "invalid_schema",
        "unrecognized_output",
        "existing_connection",
        "disabled",
        "invalid_endpoint",
        "connection_refused",
        "resolution_failed",
        "connect_failed",
    }
)

OBSERVATION_STATES = frozenset(
    {"available", "unavailable", "timeout", "command_failed", "invalid"}
)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 2.0
DEFAULT_TCP_TIMEOUT_SECONDS = 2.0
MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
DEFAULT_HEARTBEAT_SAMPLE_WINDOW = 64
DEFAULT_MAX_HEARTBEAT_RTT_MS = 120_000.0
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)(?:[a-z0-9-]{1,63}\.)*[a-z0-9][a-z0-9-]{0,62}$",
    re.IGNORECASE,
)


def _nearest_rank(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return round(ordered[rank - 1], 3)


class HeartbeatQualityWindow:
    """Thread-safe, generation-fenced heartbeat quality observations."""

    def __init__(
        self,
        *,
        max_samples: int = DEFAULT_HEARTBEAT_SAMPLE_WINDOW,
        max_rtt_ms: float = DEFAULT_MAX_HEARTBEAT_RTT_MS,
    ) -> None:
        if isinstance(max_samples, bool) or not 2 <= int(max_samples) <= 4096:
            raise ValueError("max_samples must be between 2 and 4096")
        if not math.isfinite(float(max_rtt_ms)) or float(max_rtt_ms) <= 0:
            raise ValueError("max_rtt_ms must be finite and positive")
        self._max_samples = int(max_samples)
        self._max_rtt_ms = float(max_rtt_ms)
        self._rtt_samples: deque[float] = deque(maxlen=self._max_samples)
        self._events: deque[str] = deque(maxlen=self._max_samples)
        self._generation: int | None = None
        self._pending_sent_at: float | None = None
        self._disconnected_generation: int | None = None
        self._consecutive_stalls = 0
        self._consecutive_reconnects = 0
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_generation(generation: int) -> int:
        if isinstance(generation, bool):
            raise ValueError("generation must be a non-negative integer")
        value = int(generation)
        if value < 0 or value != generation:
            raise ValueError("generation must be a non-negative integer")
        return value

    @staticmethod
    def _normalize_timestamp(timestamp: float) -> float | None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            return None
        value = float(timestamp)
        return value if math.isfinite(value) and value > 0 else None

    def _mark_stall_locked(self) -> None:
        self._events.append("stall")
        self._consecutive_stalls += 1

    def begin_generation(self, generation: int) -> bool:
        """Start a newer successful connection generation."""
        value = self._normalize_generation(generation)
        with self._lock:
            if self._generation is not None and value <= self._generation:
                return False
            if self._pending_sent_at is not None:
                self._mark_stall_locked()
            if self._generation is not None:
                self._events.append("reconnect")
                self._consecutive_reconnects += 1
            self._generation = value
            self._pending_sent_at = None
            self._disconnected_generation = None
            return True

    def record_send(self, generation: int, sent_at: float) -> bool:
        """Record one send, converting an older pending send into a stall."""
        value = self._normalize_generation(generation)
        normalized_sent_at = self._normalize_timestamp(sent_at)
        if normalized_sent_at is None:
            return False
        with self._lock:
            if self._generation is None:
                self._generation = value
            if value != self._generation:
                return False
            if self._pending_sent_at is not None:
                self._mark_stall_locked()
            self._pending_sent_at = normalized_sent_at
            self._disconnected_generation = None
            return True

    def record_ack(
        self,
        generation: int,
        echoed_sent_at: float,
        received_at: float,
    ) -> float | None:
        """Accept an ACK only for the exact pending send in the current generation."""
        value = self._normalize_generation(generation)
        normalized_sent_at = self._normalize_timestamp(echoed_sent_at)
        normalized_received_at = self._normalize_timestamp(received_at)
        if normalized_sent_at is None or normalized_received_at is None:
            return None
        with self._lock:
            if value != self._generation or self._pending_sent_at is None:
                return None
            if normalized_sent_at != self._pending_sent_at:
                return None
            rtt_ms = (normalized_received_at - normalized_sent_at) * 1000.0
            if not math.isfinite(rtt_ms) or not 0 <= rtt_ms <= self._max_rtt_ms:
                return None
            self._rtt_samples.append(rtt_ms)
            self._events.append("ack")
            self._pending_sent_at = None
            self._consecutive_stalls = 0
            self._consecutive_reconnects = 0
            return rtt_ms

    def record_disconnect(self, generation: int) -> bool:
        """Fence stale disconnects and count one pending heartbeat as stalled."""
        value = self._normalize_generation(generation)
        with self._lock:
            if value != self._generation or self._disconnected_generation == value:
                return False
            if self._pending_sent_at is not None:
                self._mark_stall_locked()
            self._pending_sent_at = None
            self._disconnected_generation = value
            return True

    def snapshot(self) -> dict[str, int | float | bool | None]:
        """Return the stable bounded aggregate without endpoint information."""
        with self._lock:
            samples = list(self._rtt_samples)
            jitter_samples = [
                abs(current - previous)
                for previous, current in zip(samples, samples[1:])
            ]
            events = list(self._events)
            return {
                "schema_version": 1,
                "generation": self._generation or 0,
                "sample_window_size": self._max_samples,
                "sample_count": len(samples),
                "rtt_ms_p50": _nearest_rank(samples, 50),
                "rtt_ms_p95": _nearest_rank(samples, 95),
                "jitter_ms_p95": _nearest_rank(jitter_samples, 95),
                "consecutive_stalls": self._consecutive_stalls,
                "stalls_in_window": events.count("stall"),
                "consecutive_reconnects": self._consecutive_reconnects,
                "reconnects_in_window": events.count("reconnect"),
                "pending_heartbeat": self._pending_sent_at is not None,
            }


def _public_quality_view(snapshot: Any) -> dict[str, Any] | None:
    if not isinstance(snapshot, Mapping):
        return None
    public: dict[str, Any] = {"schema_version": 1}
    for field_name in _QUALITY_COUNT_FIELDS:
        value = snapshot.get(field_name)
        public[field_name] = (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else 0
        )
    for field_name in _QUALITY_FLOAT_FIELDS:
        value = snapshot.get(field_name)
        public[field_name] = (
            round(float(value), 3)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            else None
        )
    public["pending_heartbeat"] = snapshot.get("pending_heartbeat") is True
    return public


def _safe_reason(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _OBSERVATION_REASONS else None


def sanitize_network_path_view(snapshot: Any) -> dict[str, Any] | None:
    """Reduce any path-like mapping to the schema-v1 diagnostic allowlist."""
    if not isinstance(snapshot, Mapping):
        return None

    endpoint = snapshot.get("endpoint")
    safe_endpoint = None
    if isinstance(endpoint, Mapping):
        role = endpoint.get("role")
        host_scope = endpoint.get("host_scope")
        port = endpoint.get("port")
        if (
            role in {"master", "gateway"}
            and host_scope in _HOST_SCOPES
            and isinstance(port, int)
            and not isinstance(port, bool)
            and 1 <= port <= 65535
        ):
            safe_endpoint = {
                "role": role,
                "host_scope": host_scope,
                "port": port,
            }

    tailscale = snapshot.get("tailscale")
    safe_tailscale = None
    if isinstance(tailscale, Mapping):
        state = tailscale.get("state")
        safe_tailscale = {
            "state": state if state in OBSERVATION_STATES else "invalid",
            "reason": _safe_reason(tailscale.get("reason")),
        }

    tcp_probe = snapshot.get("tcp_probe")
    safe_tcp_probe = None
    if isinstance(tcp_probe, Mapping):
        state = tcp_probe.get("state")
        elapsed_ms = tcp_probe.get("elapsed_ms")
        safe_tcp_probe = {
            "state": (
                state
                if state in {"not_run", "available", "unavailable", "timeout"}
                else "not_run"
            ),
            "reason": _safe_reason(tcp_probe.get("reason")),
            "elapsed_ms": (
                round(float(elapsed_ms), 3)
                if isinstance(elapsed_ms, (int, float))
                and not isinstance(elapsed_ms, bool)
                and math.isfinite(float(elapsed_ms))
                and elapsed_ms >= 0
                else None
            ),
        }

    path_kind = snapshot.get("path_kind")
    availability = snapshot.get("availability")
    return {
        "schema_version": 1,
        "path_kind": path_kind if path_kind in PATH_KINDS else "unknown",
        "availability": (
            availability
            if availability in {"available", "degraded", "unknown"}
            else "unknown"
        ),
        "endpoint": safe_endpoint,
        "tailscale": safe_tailscale,
        "tcp_probe": safe_tcp_probe,
        "quality": _public_quality_view(snapshot.get("quality")),
    }


def network_path_diagnostic_json(snapshot: Any) -> str:
    """Serialize the exact same allowlisted view used by API diagnostics."""
    return json.dumps(
        sanitize_network_path_view(snapshot),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_client_network_path_view(tcp_client: Any) -> dict[str, Any] | None:
    """Build an API-safe local view without running probes or CLI commands."""
    if tcp_client is None:
        return None

    quality = None
    get_quality = getattr(tcp_client, "get_network_quality_snapshot", None)
    if callable(get_quality):
        try:
            quality = _public_quality_view(get_quality())
        except Exception:
            quality = None

    connected = bool(getattr(tcp_client, "is_registered", False))
    try:
        endpoint = TrustedEndpoint(
            getattr(tcp_client, "server_host", ""),
            getattr(tcp_client, "server_port", 0),
        )
    except (TypeError, ValueError):
        return sanitize_network_path_view({
            "schema_version": 1,
            "path_kind": "unknown",
            "availability": "degraded" if connected else "unknown",
            "endpoint": None,
            "tailscale": None,
            "tcp_probe": {
                "state": "not_run",
                "reason": "invalid_endpoint",
                "elapsed_ms": None,
            },
            "quality": quality,
        })

    tcp_observation = TcpProbeObservation(
        "available" if connected else "not_run",
        "existing_connection" if connected else "disabled",
    )
    public = classify_trusted_path(
        endpoint,
        tcp_probe=tcp_observation,
    ).public_view()
    return sanitize_network_path_view({
        "schema_version": 1,
        **public,
        "quality": quality,
    })
_NETCHECK_BOOL_RE = re.compile(
    r"^\s*(udp|ipv4|ipv6)\s*:\s*(true|false|yes|no)\b", re.IGNORECASE | re.MULTILINE
)
_NETCHECK_DERP_RE = re.compile(r"^\s*nearest derp\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _validate_timeout(timeout_seconds: float) -> float:
    timeout = float(timeout_seconds)
    if not 0.1 <= timeout <= 10.0:
        raise ValueError("timeout_seconds must be between 0.1 and 10.0")
    return timeout


def _normalize_host(host: str) -> str:
    value = canonical_host(host)
    if not value or any(character.isspace() for character in value):
        raise ValueError("trusted endpoint host is required")
    if any(character in value for character in "/\\@?#"):
        raise ValueError("trusted endpoint host must not contain a URL or path")

    try:
        ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        if not _HOSTNAME_RE.fullmatch(value):
            raise ValueError("trusted endpoint host is invalid")
        return value.lower()
    return value.lower()


def _host_scope(host: str) -> str:
    value = canonical_host(host)
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return "tailnet_dns" if value.lower().endswith(".ts.net") else "dns"

    if is_tailscale_ip(value):
        return "tailscale_ipv6" if address.version == 6 else "tailscale_ipv4"
    if address.is_loopback:
        return "loopback_ipv6" if address.version == 6 else "loopback_ipv4"
    if address.is_private or address.is_link_local:
        return "private_ipv6" if address.version == 6 else "private_ipv4"
    return "public_ipv6" if address.version == 6 else "public_ipv4"


def _is_lan_host(host: str) -> bool:
    value = canonical_host(host)
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return not is_tailscale_ip(value) and (address.is_loopback or address.is_private or address.is_link_local)


def _is_tailnet_host(host: str) -> bool:
    value = canonical_host(host).lower().rstrip(".")
    return is_tailscale_ip(value) or value.endswith(".ts.net")


@dataclass(frozen=True)
class TrustedEndpoint:
    """One exact endpoint approved by the caller for a single TCP probe."""

    host: str
    port: int = 443
    role: str = "master"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _normalize_host(self.host))
        normalized_port = int(self.port)
        if not 1 <= normalized_port <= 65535:
            raise ValueError("trusted endpoint port must be between 1 and 65535")
        object.__setattr__(self, "port", normalized_port)
        if self.role not in {"master", "gateway"}:
            raise ValueError("trusted endpoint role must be master or gateway")

    def public_descriptor(self) -> dict[str, Any]:
        """Return the endpoint category without exposing its host value."""
        return {"role": self.role, "host_scope": _host_scope(self.host), "port": self.port}


@dataclass(frozen=True)
class TailscaleStatusObservation:
    state: str
    reason: str | None = None
    _payload: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.state not in OBSERVATION_STATES:
            raise ValueError("invalid Tailscale observation state")

    @property
    def payload(self) -> Mapping[str, Any] | None:
        """Raw data is for local classification only, never for API projection."""
        return self._payload

    def public_view(self) -> dict[str, str | None]:
        return {"state": self.state, "reason": self.reason}


@dataclass(frozen=True)
class TailscaleNetcheckObservation:
    state: str
    reason: str | None = None
    udp: bool | None = None
    ipv4: bool | None = None
    ipv6: bool | None = None
    nearest_derp_available: bool | None = None

    def __post_init__(self) -> None:
        if self.state not in OBSERVATION_STATES:
            raise ValueError("invalid Tailscale observation state")

    def public_view(self) -> dict[str, str | bool | None]:
        return {
            "state": self.state,
            "reason": self.reason,
            "udp": self.udp,
            "ipv4": self.ipv4,
            "ipv6": self.ipv6,
            "nearest_derp_available": self.nearest_derp_available,
        }


@dataclass(frozen=True)
class TcpProbeObservation:
    state: str
    reason: str | None = None
    elapsed_ms: float | None = None

    def __post_init__(self) -> None:
        if self.state not in {"not_run", "available", "unavailable", "timeout"}:
            raise ValueError("invalid TCP probe observation state")

    def public_view(self) -> dict[str, str | float | None]:
        return {
            "state": self.state,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class PathSnapshot:
    path_kind: str
    availability: str
    endpoint: TrustedEndpoint
    tailscale_status: TailscaleStatusObservation | None = None
    tcp_probe: TcpProbeObservation | None = None

    def __post_init__(self) -> None:
        if self.path_kind not in PATH_KINDS:
            raise ValueError("invalid path kind")
        if self.availability not in {"available", "degraded", "unknown"}:
            raise ValueError("invalid path availability")

    def public_view(self) -> dict[str, Any]:
        """Return the stable API-safe form without raw command or address data."""
        return {
            "path_kind": self.path_kind,
            "availability": self.availability,
            "endpoint": self.endpoint.public_descriptor(),
            "tailscale": self.tailscale_status.public_view() if self.tailscale_status else None,
            "tcp_probe": self.tcp_probe.public_view() if self.tcp_probe else None,
        }


def find_tailscale_executable() -> str | None:
    """Locate the CLI without invoking it or accepting a shell command string."""
    return shutil.which("tailscale") or shutil.which("tailscale.exe")


def _run_tailscale_command(
    arguments: list[str],
    *,
    executable: str | None,
    runner: Callable[..., Any],
    timeout_seconds: float,
) -> tuple[str, str | None, str | None]:
    timeout = _validate_timeout(timeout_seconds)
    command_path = executable or find_tailscale_executable()
    if not command_path:
        return "unavailable", "executable_not_found", None

    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        completed = runner([command_path, *arguments], **kwargs)
    except subprocess.TimeoutExpired:
        return "timeout", "timeout", None
    except FileNotFoundError:
        return "unavailable", "executable_not_found", None
    except OSError:
        return "unavailable", "command_unavailable", None

    if int(getattr(completed, "returncode", 1)) != 0:
        return "command_failed", "nonzero_exit", None
    output = getattr(completed, "stdout", "")
    if not isinstance(output, str):
        return "invalid", "non_text_output", None
    if len(output.encode("utf-8", errors="replace")) > MAX_COMMAND_OUTPUT_BYTES:
        return "invalid", "output_too_large", None
    return "available", None, output


def collect_tailscale_status(
    *,
    executable: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> TailscaleStatusObservation:
    """Read ``tailscale status --json`` with bounded, fail-closed parsing."""
    state, reason, output = _run_tailscale_command(
        ["status", "--json"],
        executable=executable,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    if output is None:
        return TailscaleStatusObservation(state, reason)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return TailscaleStatusObservation("invalid", "invalid_json")
    if not isinstance(payload, Mapping):
        return TailscaleStatusObservation("invalid", "invalid_schema")
    return TailscaleStatusObservation("available", _payload=payload)


def collect_tailscale_netcheck(
    *,
    executable: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> TailscaleNetcheckObservation:
    """Read the stable boolean portions of ``tailscale netcheck`` only."""
    state, reason, output = _run_tailscale_command(
        ["netcheck"],
        executable=executable,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    if output is None:
        return TailscaleNetcheckObservation(state, reason)

    values: dict[str, bool] = {}
    for match in _NETCHECK_BOOL_RE.finditer(output):
        values[match.group(1).lower()] = match.group(2).lower() in {"true", "yes"}
    derp_match = _NETCHECK_DERP_RE.search(output)
    if not values and derp_match is None:
        return TailscaleNetcheckObservation("invalid", "unrecognized_output")
    return TailscaleNetcheckObservation(
        "available",
        udp=values.get("udp"),
        ipv4=values.get("ipv4"),
        ipv6=values.get("ipv6"),
        nearest_derp_available=derp_match is not None and bool(derp_match.group(1).strip()),
    )


def probe_trusted_tcp(
    endpoint: TrustedEndpoint,
    *,
    connector: Callable[..., socket.socket] = socket.create_connection,
    timeout_seconds: float = DEFAULT_TCP_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> TcpProbeObservation:
    """Connect once to the caller-approved endpoint; never enumerate hosts."""
    timeout = _validate_timeout(timeout_seconds)
    started = clock()
    connection: socket.socket | None = None
    try:
        connection = connector((endpoint.host, endpoint.port), timeout=timeout)
    except (socket.timeout, TimeoutError):
        return TcpProbeObservation("timeout", "timeout")
    except ConnectionRefusedError:
        return TcpProbeObservation("unavailable", "connection_refused")
    except socket.gaierror:
        return TcpProbeObservation("unavailable", "resolution_failed")
    except OSError:
        return TcpProbeObservation("unavailable", "connect_failed")
    finally:
        if connection is not None:
            connection.close()
    elapsed_ms = max(0.0, (clock() - started) * 1000.0)
    return TcpProbeObservation("available", elapsed_ms=round(elapsed_ms, 3))


def _tailscale_peer_path(
    endpoint: TrustedEndpoint,
    observation: TailscaleStatusObservation,
) -> str:
    if observation.state != "available" or observation.payload is None:
        return "unknown"
    peers = observation.payload.get("Peer")
    records = peers.values() if isinstance(peers, Mapping) else peers if isinstance(peers, list) else []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        addresses = record.get("TailscaleIPs")
        normalized_addresses = (
            {canonical_host(str(address)).lower() for address in addresses}
            if isinstance(addresses, list)
            else set()
        )
        dns_name = str(record.get("DNSName") or "").lower().rstrip(".")
        target = endpoint.host.lower().rstrip(".")
        if target not in normalized_addresses and target != dns_name:
            continue
        relay = record.get("Relay")
        if relay is True or (isinstance(relay, str) and relay.strip()):
            return "derp"
        if record.get("CurAddr"):
            return "tailscale_direct"
        return "unknown"
    return "unknown"


def classify_trusted_path(
    endpoint: TrustedEndpoint,
    *,
    tailscale_status: TailscaleStatusObservation | None = None,
    tcp_probe: TcpProbeObservation | None = None,
) -> PathSnapshot:
    """Classify one trusted endpoint without using classification for routing."""
    if endpoint.role == "gateway" and tcp_probe and tcp_probe.state == "available":
        return PathSnapshot("gateway_relay", "available", endpoint, tailscale_status, tcp_probe)

    if _is_tailnet_host(endpoint.host):
        path_kind = _tailscale_peer_path(
            endpoint,
            tailscale_status or TailscaleStatusObservation("unavailable", "not_collected"),
        )
        availability = "available" if path_kind != "unknown" else "degraded"
        return PathSnapshot(path_kind, availability, endpoint, tailscale_status, tcp_probe)

    if tcp_probe and tcp_probe.state == "available":
        path_kind = "lan_direct" if _is_lan_host(endpoint.host) else "public_tcp_direct"
        return PathSnapshot(path_kind, "available", endpoint, tailscale_status, tcp_probe)

    availability = "degraded" if tcp_probe and tcp_probe.state in {"timeout", "unavailable"} else "unknown"
    return PathSnapshot("unknown", availability, endpoint, tailscale_status, tcp_probe)
