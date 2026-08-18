#!/usr/bin/env python3
"""Run a bounded, read-only readiness check before connecting a PC worker.

The script checks only caller-provided master endpoints.  It never scans a
subnet, changes node configuration, starts a service, or prints secrets and
raw Tailnet peer records.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_path import (  # noqa: E402
    TrustedEndpoint,
    classify_trusted_path,
    collect_tailscale_netcheck,
    collect_tailscale_status,
    probe_trusted_tcp,
    probe_trusted_tls,
)
from network_address import is_tailscale_ip  # noqa: E402


SCHEMA_VERSION = 1
SIDECAR_MODULES = {
    "qwen3": ("torch", "torchvision", "accelerate", "safetensors", "transformers"),
    "gemma4": ("torch", "accelerate", "safetensors", "transformers"),
}


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _timeout(value: str) -> float:
    timeout = float(value)
    if not 0.1 <= timeout <= 10.0:
        raise argparse.ArgumentTypeError("timeout must be between 0.1 and 10 seconds")
    return timeout


def _venv_python(name: str) -> Path:
    executable = "python.exe" if os.name == "nt" else "python"
    directory = "Scripts" if os.name == "nt" else "bin"
    return ROOT / name / directory / executable


def _sidecar_python(kind: str) -> Path:
    return _venv_python(
        ".venv-qwen3-sidecar" if kind == "qwen3" else ".venv-gemma4-pipeline"
    )


def _local_tailnet_runtime(status: Any) -> dict[str, bool]:
    summary = {
        "backend_running": False,
        "self_online": False,
        "ipv4": False,
        "ipv6": False,
    }
    payload = getattr(status, "payload", None)
    if not isinstance(payload, dict):
        return summary
    summary["backend_running"] = str(payload.get("BackendState", "")).lower() == "running"
    own_record = payload.get("Self")
    if isinstance(own_record, dict):
        summary["self_online"] = own_record.get("Online") is True
    addresses = payload.get("TailscaleIPs", [])
    if (not isinstance(addresses, list) or not addresses) and isinstance(own_record, dict):
        addresses = own_record.get("TailscaleIPs", [])
    if (not isinstance(addresses, list) or not addresses) and isinstance(own_record, dict):
        addresses = own_record.get("Addrs", [])
    for raw in addresses if isinstance(addresses, list) else []:
        value = str(raw).split("/", 1)[0].split("%", 1)[0]
        if not is_tailscale_ip(value):
            continue
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        summary["ipv6" if address.version == 6 else "ipv4"] = True
    return summary


def _check_sidecar(kind: str) -> dict[str, Any]:
    python = _sidecar_python(kind)
    modules = SIDECAR_MODULES[kind]
    result: dict[str, Any] = {
        "kind": kind,
        "venv_present": python.is_file(),
        "ready": False,
        "missing_modules": [],
        "torch": None,
    }
    if not python.is_file():
        result["reason"] = "venv_missing"
        return result

    # Probe the selected interpreter rather than this process so a main-node
    # package cannot make an incomplete sidecar look healthy.
    probe = (
        "import importlib.util, json\n"
        f"modules = {modules!r}\n"
        "missing = [name for name in modules if importlib.util.find_spec(name) is None]\n"
        "payload = {'missing_modules': missing}\n"
        "if not missing:\n"
        " import torch\n"
        " payload['torch'] = {'version': torch.__version__, 'cuda_available': bool(torch.cuda.is_available())}\n"
        "print(json.dumps(payload, ensure_ascii=True))\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["reason"] = f"sidecar_probe_{exc.__class__.__name__.lower()}"
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["reason"] = "sidecar_probe_invalid_output"
        return result
    result["missing_modules"] = list(payload.get("missing_modules", []))
    result["torch"] = payload.get("torch")
    result["ready"] = completed.returncode == 0 and not result["missing_modules"]
    if not result["ready"]:
        result["reason"] = "sidecar_modules_missing"
    return result


def _endpoint_report(
    host: str,
    port: int,
    *,
    timeout_seconds: float,
    tailscale_status: Any,
    probe_tls: bool = False,
) -> dict[str, Any]:
    endpoint = TrustedEndpoint(host, port, role="master")
    tcp_probe = probe_trusted_tcp(endpoint, timeout_seconds=timeout_seconds)
    path = classify_trusted_path(
        endpoint,
        tailscale_status=tailscale_status,
        tcp_probe=tcp_probe,
    )
    report = {
        "endpoint": endpoint.public_descriptor(),
        "tcp_probe": tcp_probe.public_view(),
        "path": path.public_view(),
        "ready": tcp_probe.state == "available",
    }
    if probe_tls:
        tls_probe = probe_trusted_tls(endpoint, timeout_seconds=timeout_seconds)
        report["tls_probe"] = tls_probe.public_view()
        report["ready"] = report["ready"] and tls_probe.state == "available"
    return report


def build_report(
    *,
    role: str,
    master_host: str = "",
    api_port: int = 8000,
    tcp_port: int = 8888,
    tls_port: int = 443,
    timeout_seconds: float = 2.0,
    sidecar: str = "none",
    require_tailnet: bool = False,
    require_ipv6: bool = False,
    require_tls: bool = False,
    status_collector: Callable[..., Any] = collect_tailscale_status,
    netcheck_collector: Callable[..., Any] = collect_tailscale_netcheck,
    endpoint_reporter: Callable[..., dict[str, Any]] = _endpoint_report,
    sidecar_checker: Callable[[str], dict[str, Any]] = _check_sidecar,
) -> dict[str, Any]:
    """Build the public report with injectable collectors for contract tests."""
    tailscale_status = status_collector(timeout_seconds=timeout_seconds)
    netcheck = netcheck_collector(timeout_seconds=timeout_seconds)
    local_tailnet = _local_tailnet_runtime(tailscale_status)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "local": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ipv6_socket_supported": bool(socket.has_ipv6),
        },
        "tailscale": {
            **tailscale_status.public_view(),
            "local_addresses": local_tailnet,
        },
        "netcheck": netcheck.public_view(),
        "endpoints": {},
        "sidecars": [],
        "checks": {},
    }
    checks: dict[str, bool] = {
        "tailscale_cli": tailscale_status.state == "available",
        "tailscale_running": local_tailnet["backend_running"],
        "tailscale_self_online": local_tailnet["self_online"],
        "ipv6_socket": bool(socket.has_ipv6),
    }
    if require_ipv6:
        checks["local_tailnet_ipv6"] = local_tailnet["ipv6"]

    if role == "client":
        if not master_host:
            checks["master_endpoint_provided"] = False
        else:
            api = endpoint_reporter(
                master_host,
                api_port,
                timeout_seconds=timeout_seconds,
                tailscale_status=tailscale_status,
            )
            tcp = endpoint_reporter(
                master_host,
                tcp_port,
                timeout_seconds=timeout_seconds,
                tailscale_status=tailscale_status,
            )
            report["endpoints"] = {"api": api, "tcp": tcp}
            tls = None
            if require_tls:
                tls = endpoint_reporter(
                    master_host,
                    tls_port,
                    timeout_seconds=timeout_seconds,
                    tailscale_status=tailscale_status,
                    probe_tls=True,
                )
                report["endpoints"]["tls"] = tls
            checks["master_api_tcp"] = bool(api["ready"])
            checks["master_tcp_tcp"] = bool(tcp["ready"])
            if require_tls:
                checks["master_tls"] = bool(tls and tls["ready"])
            endpoint_scope = str(api["endpoint"].get("host_scope", ""))
            if require_tailnet:
                checks["tailnet_endpoint"] = endpoint_scope in {
                    "tailscale_ipv4", "tailscale_ipv6", "tailnet_dns",
                }
            if require_ipv6:
                checks["ipv6_endpoint"] = endpoint_scope == "tailscale_ipv6"
    if sidecar != "none":
        sidecar_kinds = tuple(SIDECAR_MODULES) if sidecar == "all" else (sidecar,)
        report["sidecars"] = [sidecar_checker(kind) for kind in sidecar_kinds]
        checks["sidecars"] = all(bool(item.get("ready")) for item in report["sidecars"])

    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("master", "client"), default="client")
    parser.add_argument("--master-host", default="", help="one approved master host; never scanned")
    parser.add_argument("--api-port", type=_port, default=int(os.environ.get("QLH_MASTER_API_PORT", "8000")))
    parser.add_argument("--tcp-port", type=_port, default=int(os.environ.get("QLH_MASTER_PORT", "8888")))
    parser.add_argument("--tls-port", type=_port, default=443)
    parser.add_argument("--timeout", type=_timeout, default=2.0)
    parser.add_argument("--sidecar", choices=("none", "qwen3", "gemma4", "all"), default="none")
    parser.add_argument("--require-tailnet", action="store_true")
    parser.add_argument("--require-ipv6", action="store_true")
    parser.add_argument("--require-tls", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        role=args.role,
        master_host=args.master_host,
        api_port=args.api_port,
        tcp_port=args.tcp_port,
        tls_port=args.tls_port,
        timeout_seconds=args.timeout,
        sidecar=args.sidecar,
        require_tailnet=args.require_tailnet,
        require_ipv6=args.require_ipv6,
        require_tls=args.require_tls,
    )
    encoded = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        destination = args.output.expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
