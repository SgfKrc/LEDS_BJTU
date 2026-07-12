"""First-connect bootstrap for trusted Tailscale nodes."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import socket
import urllib.error
import urllib.request
from typing import Any

from node_config import apply_runtime_config, persist_bootstrap_response


DEFAULT_TRUSTED_CIDRS = "100.64.0.0/10,127.0.0.0/8,::1/128"


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
