#!/usr/bin/env python3
"""Verify that local QLH services accept IPv4 and IPv6 connections."""

from __future__ import annotations

import argparse
import json
import locale
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Service:
    name: str
    command: list[str]
    env: dict[str, str]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def decode_log(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), errors="replace")


def wait_for_dual_stack(process: subprocess.Popen[str], port: int, timeout: float) -> dict[str, bool]:
    deadline = time.monotonic() + timeout
    result = {"127.0.0.1": False, "::1": False}

    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        for host in result:
            if not result[host]:
                result[host] = can_connect(host, port)
        if all(result.values()):
            break
        time.sleep(0.1)

    return result


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    process.wait()


def make_services(temp_dir: Path) -> list[tuple[Service, int]]:
    python_port = find_free_port()
    control_port = find_free_port()
    gateway_port = find_free_port()
    python_code = (
        "from api_server import run_api_servers; "
        f"run_api_servers('0.0.0.0', {python_port})"
    )

    common_env = {
        "QLH_DB_ENABLED": "false",
        "QLH_MODEL_STORE": str(temp_dir / "model-store"),
        "QLH_STATE_DIR": str(temp_dir),
    }
    python_env = {
        **common_env,
        "PYTHONPATH": str(ROOT / "src"),
        "QLH_SERVER_PORT": str(python_port),
        "QLH_SQLITE_PATH": str(temp_dir / "python.db"),
    }
    control_env = {
        **common_env,
        "QLH_CONTROL_HOST": "::",
        "QLH_CONTROL_PORT": str(control_port),
        "QLH_SQLITE_PATH": str(temp_dir / "control.db"),
    }
    gateway_env = {
        **common_env,
        "QLH_API_HOST": "::",
        "QLH_API_PORT": str(gateway_port),
    }

    return [
        (
            Service(
                "python-api",
                [sys.executable, "-c", python_code],
                python_env,
            ),
            python_port,
        ),
        (
            Service(
                "control-svc",
                ["node", str(ROOT / "control" / "dist" / "main.js")],
                control_env,
            ),
            control_port,
        ),
        (
            Service(
                "gateway",
                ["node", str(ROOT / "gateway" / "dist" / "main.js")],
                gateway_env,
            ),
            gateway_port,
        ),
    ]


def run_service(service: Service, port: int, timeout: float) -> dict[str, object]:
    env = os.environ.copy()
    env.update(service.env)
    for name in ("DATABASE_URL", "QLH_DB_HOST", "QLH_DB_PORT", "QLH_DB_USER", "QLH_DB_PASSWORD"):
        env.pop(name, None)

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as stdout_log, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_log:
        process = subprocess.Popen(
            service.command,
            cwd=ROOT,
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
            creationflags=creationflags,
        )
        connections = wait_for_dual_stack(process, port, timeout)
        return_code = process.poll()
        stop_process(process)
        stdout_log.seek(0)
        stderr_log.seek(0)
        stdout = decode_log(stdout_log.read())
        stderr = decode_log(stderr_log.read())
    passed = all(connections.values())

    return {
        "service": service.name,
        "port": port,
        "connections": connections,
        "passed": passed,
        "early_exit_code": return_code,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def run_socket_helper() -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from network_address import create_listen_sockets

    sockets = create_listen_sockets(["0.0.0.0", "::"], 0)
    try:
        port = int(sockets[0].getsockname()[1])
        connections = {
            "127.0.0.1": can_connect("127.0.0.1", port),
            "::1": can_connect("::1", port),
        }
        ipv6_only = [
            sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
            for sock in sockets
            if sock.family == socket.AF_INET6
        ]
        passed = all(connections.values()) and ipv6_only == [1]
        return {
            "service": "socket-helper",
            "port": port,
            "connections": connections,
            "ipv6_only": ipv6_only,
            "passed": passed,
        }
    finally:
        for sock in sockets:
            sock.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("all", "socket-helper", "python-api", "control-svc", "gateway"),
        default="all",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    if args.component == "socket-helper":
        results = [run_socket_helper()]
    else:
        with tempfile.TemporaryDirectory(prefix="qlh-ipv6-gate-") as temp_name:
            services = make_services(Path(temp_name))
            if args.component != "all":
                services = [item for item in services if item[0].name == args.component]
            results = [run_service(service, port, args.timeout) for service, port in services]

    print(json.dumps(results, ensure_ascii=True, indent=2))
    return 0 if all(bool(result["passed"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
