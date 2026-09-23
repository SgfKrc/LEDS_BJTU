#!/usr/bin/env python
"""Measure raw framed TCP echo latency for representative activation payload sizes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import socket
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "qlh.torch_lan_echo.v1"
DEFAULT_PAYLOAD_BYTES = (230_957, 919_085)
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


def recv_exact(connection: socket.socket, length: int) -> bytes:
    if length < 0 or length > MAX_PAYLOAD_BYTES + 4:
        raise ValueError(f"invalid frame length: {length}")
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("peer closed before the full frame arrived")
        chunks.extend(chunk)
    return bytes(chunks)


def serve_connection(connection: socket.socket) -> None:
    try:
        while True:
            try:
                header = recv_exact(connection, 4)
            except ConnectionError:
                return
            (length,) = struct.unpack("!I", header)
            if length > MAX_PAYLOAD_BYTES:
                raise ValueError(f"payload exceeds limit: {length}")
            payload = recv_exact(connection, length)
            connection.sendall(header + payload)
    except (ConnectionError, OSError):
        return


def summarize(samples_ms: list[float]) -> dict[str, Any]:
    mean = statistics.mean(samples_ms)
    deviation = statistics.pstdev(samples_ms)
    return {
        "samples_ms": samples_ms,
        "count": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": mean,
        "population_stddev_ms": deviation,
        "cv": deviation / mean if mean else 0.0,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _exchange(connection: socket.socket, payload: bytes) -> float:
    header = struct.pack("!I", len(payload))
    started = time.perf_counter_ns()
    connection.sendall(header + payload)
    echoed_header = recv_exact(connection, 4)
    (echoed_length,) = struct.unpack("!I", echoed_header)
    echoed_payload = recv_exact(connection, echoed_length)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if echoed_length != len(payload) or echoed_payload != payload:
        raise ValueError("echoed payload did not match the sent bytes")
    return elapsed_ms


def run_client(
    host: str,
    port: int,
    *,
    warmup: int = 3,
    repeats: int = 30,
    payload_sizes: tuple[int, ...] = DEFAULT_PAYLOAD_BYTES,
    timeout: float = 10.0,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1 or not payload_sizes:
        raise ValueError("warmup must be non-negative, repeats positive, and sizes non-empty")
    if any(size < 1 or size > MAX_PAYLOAD_BYTES for size in payload_sizes):
        raise ValueError("payload size is outside the supported range")

    payloads = {
        size: random.Random(size).randbytes(size)
        for size in payload_sizes
    }
    samples: dict[int, list[float]] = {size: [] for size in payload_sizes}
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        source_ip, source_port = connection.getsockname()[:2]
        peer_ip, peer_port = connection.getpeername()[:2]
        for _ in range(warmup):
            for size in payload_sizes:
                _exchange(connection, payloads[size])
        for index in range(repeats):
            ordered_sizes = payload_sizes if index % 2 == 0 else tuple(reversed(payload_sizes))
            for size in ordered_sizes:
                samples[size].append(_exchange(connection, payloads[size]))

    return {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "persistent direct TCP framed raw-byte echo; no TLS, control envelope, model runtime, "
            "tensor serialization, inference, or concurrent workload; byte equality checked "
            "after each timed send/receive"
        ),
        "client_host": socket.gethostname(),
        "client_platform": platform.platform(),
        "python": sys.version.split()[0],
        "source_endpoint": {"ip": source_ip, "port": source_port},
        "destination_endpoint": {"ip": peer_ip, "port": peer_port},
        "tcp_nodelay": True,
        "warmup_per_payload": warmup,
        "repeats_per_payload": repeats,
        "payloads": {
            str(size): {
                "bytes": size,
                "sha256": hashlib.sha256(payloads[size]).hexdigest(),
                "echo_exact": True,
                "round_trip": summarize(samples[size]),
            }
            for size in payload_sizes
        },
        "production_transport_tested": False,
        "cross_host_inference_admitted": False,
    }


def _write_report(report: dict[str, Any], output: str | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    server = subparsers.add_parser("server", help="echo one client connection")
    server.add_argument("--bind", required=True)
    server.add_argument("--port", required=True, type=int)
    client = subparsers.add_parser("client", help="measure echo round trips")
    client.add_argument("--host", required=True)
    client.add_argument("--port", required=True, type=int)
    client.add_argument("--warmup", type=int, default=3)
    client.add_argument("--repeats", type=int, default=30)
    client.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_PAYLOAD_BYTES))
    client.add_argument("--timeout", type=float, default=10.0)
    client.add_argument("--output")
    args = parser.parse_args()

    if args.mode == "server":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            listener.settimeout(120.0)
            print(json.dumps({"event": "ready", "endpoint": listener.getsockname()}), flush=True)
            connection, address = listener.accept()
            print(json.dumps({"event": "accepted", "peer": address}), flush=True)
            with connection:
                serve_connection(connection)
    else:
        report = run_client(
            args.host,
            args.port,
            warmup=args.warmup,
            repeats=args.repeats,
            payload_sizes=tuple(args.sizes),
            timeout=args.timeout,
        )
        _write_report(report, args.output)


if __name__ == "__main__":
    main()
