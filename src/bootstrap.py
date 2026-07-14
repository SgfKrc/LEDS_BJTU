"""First-connect bootstrap for trusted Tailscale nodes."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from node_config import apply_runtime_config, persist_bootstrap_response


DEFAULT_TRUSTED_CIDRS = "100.64.0.0/10,127.0.0.0/8,::1/128"
_WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _cidr_list(raw: str | None = None) -> list[ipaddress._BaseNetwork]:
    value = raw if raw is not None else os.environ.get(
        "QLH_TRUSTED_BOOTSTRAP_CIDRS", DEFAULT_TRUSTED_CIDRS
    )
    networks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue
    return networks


def is_trusted_bootstrap_source(host: str, cidrs: str | None = None) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(ip in network for network in _cidr_list(cidrs))


def normalize_node_type(node_type: str | None) -> str:
    value = (node_type or "pc").strip().lower()
    if value in {"android", "mobile"}:
        return "android"
    return "pc"


def normalize_node_id(node_id: str | None, node_type: str = "pc") -> str:
    raw = (node_id or "").strip()
    if not raw or raw == "master":
        prefix = "android" if node_type == "android" else "client"
        raw = f"{prefix}_{socket.gethostname()}"
    allowed = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    normalized = "".join(allowed).strip("._-")
    if not normalized or normalized == "master":
        normalized = f"client_{socket.gethostname()}"
    return normalized[:64]


def select_advertised_master_host(requested_host: str, lan_ip: str = "") -> str:
    """Keep Tailnet joins on the routable Tailnet address instead of a LAN IP."""
    host = (requested_host or "").strip()
    try:
        address = ipaddress.ip_address(host)
        if address in ipaddress.ip_network("100.64.0.0/10"):
            return host
    except ValueError:
        if host.lower().endswith(".ts.net"):
            return host
    if lan_ip and lan_ip not in {"0.0.0.0", "127.0.0.1", "localhost"}:
        return lan_ip
    return host


def _find_tailscale_executable() -> str | None:
    found = shutil.which("tailscale") or shutil.which("tailscale.exe")
    if found:
        return found
    if sys.platform == "win32":
        candidate = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "Tailscale",
            "tailscale.exe",
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def get_tailnet_peer_ips(timeout: float = 3.0) -> list[str]:
    """Return online Tailnet peer IPv4 addresses without requiring DB access."""
    configured = os.environ.get("QLH_TAILNET_PEERS", "")
    peers = [item.strip() for item in configured.split(",") if item.strip()]

    executable = _find_tailscale_executable()
    if executable:
        try:
            result = subprocess.run(
                [executable, "status", "--json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                creationflags=_WINDOWS_NO_WINDOW,
            )
            if result.returncode == 0 and result.stdout.strip():
                status = json.loads(result.stdout)
                for peer in (status.get("Peer") or {}).values():
                    if peer.get("Online") is False:
                        continue
                    for address in peer.get("TailscaleIPs") or []:
                        try:
                            ip = ipaddress.ip_address(address)
                        except ValueError:
                            continue
                        if ip.version == 4 and is_trusted_bootstrap_source(address):
                            peers.append(address)
        except Exception:
            pass

    return list(dict.fromkeys(peers))


def discover_master_via_tailnet(
    api_port: int = 8000,
    timeout: float = 1.5,
) -> dict[str, Any]:
    """Probe approved Tailnet peers for a confirmed QLH master."""
    peers = get_tailnet_peer_ips()
    if not peers:
        return {"found": False, "source": "tailnet"}

    def _probe(host: str) -> dict[str, Any] | None:
        url = f"http://{host}:{int(api_port)}/api/bootstrap/info"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8") or "{}")
            if not data.get("is_master"):
                return None
            return {
                "found": True,
                "master_host": host,
                "master_port": int(data.get("master_tcp_port") or 8888),
                "master_api_port": int(data.get("master_api_port") or api_port),
                "stale": False,
                "source": "tailnet",
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(12, len(peers))) as executor:
        futures = [executor.submit(_probe, peer) for peer in peers]
        for future in as_completed(futures):
            result = future.result()
            if result:
                for pending in futures:
                    pending.cancel()
                return result
    return {"found": False, "source": "tailnet"}


def first_connect(
    master_api_host: str,
    master_api_port: int = 8000,
    node_id: str | None = None,
    node_type: str = "pc",
    app_variant: str = "",
    capabilities: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    node_type = normalize_node_type(node_type)
    payload = {
        "node_id": normalize_node_id(node_id, node_type),
        "node_type": node_type,
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "app_variant": app_variant,
        "capabilities": capabilities or {},
    }
    url = f"http://{master_api_host}:{int(master_api_port)}/api/bootstrap/first-connect"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"bootstrap HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"bootstrap request failed: {exc.reason}") from exc

    data = json.loads(body or "{}")
    if data.get("status") not in {"ok", "registered", "updated"}:
        raise RuntimeError(f"bootstrap rejected: {data}")

    persist_bootstrap_response(data)
    apply_runtime_config(data)
    return data
