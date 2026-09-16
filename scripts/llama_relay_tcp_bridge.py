#!/usr/bin/env python3
"""Expose one experimental llama Relay stdio runner through loopback TCP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_transport import (  # noqa: E402
    RelayBridgeResult,
    StdioRelayRunner,
    open_loopback_listener,
    serve_relay_connection,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--cut-model", required=True)
    parser.add_argument("--n-embd", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--max-tokens", default=4096, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--max-incomplete-sessions", default=4, type=int)
    parser.add_argument("--report")
    args = parser.parse_args()

    runner_path = Path(args.runner).resolve()
    model_path = Path(args.cut_model).resolve()
    for path in (runner_path, model_path):
        if not path.is_file():
            parser.error(f"missing asset: {path}")

    listener = open_loopback_listener(args.host, args.port)
    listener.settimeout(args.timeout)
    endpoint = listener.getsockname()
    print(json.dumps({"status": "ready", "host": endpoint[0], "port": endpoint[1]}), flush=True)

    runner = StdioRelayRunner(
        [str(runner_path), str(model_path), "--threads", str(args.threads)],
        n_embd=args.n_embd,
        timeout=args.timeout,
    )
    peer = None
    result = None
    incomplete_sessions: list[dict[str, object]] = []
    try:
        while len(incomplete_sessions) <= max(0, args.max_incomplete_sessions):
            connection, peer = listener.accept()
            with connection:
                connection.settimeout(args.timeout)
                result = serve_relay_connection(
                    connection,
                    runner,
                    n_embd=args.n_embd,
                    max_tokens=args.max_tokens,
                )
            if result.closed_cleanly:
                break
            incomplete_sessions.append(
                {"peer": list(peer), "bridge": result.to_dict()}
            )
        if result is None:
            result = RelayBridgeResult(0, 0, 0, False, "no_client_session")
    finally:
        runner.close()
        listener.close()

    report = {
        "status": "evidence_transport_complete" if result.closed_cleanly else "transport_failed",
        "scope": "loopback_or_ssh_tunnel_only",
        "wire_version": 1,
        "dtype": "float32_le",
        "endpoint": {"host": endpoint[0], "port": endpoint[1]},
        "peer": list(peer) if peer else None,
        "n_embd": args.n_embd,
        "max_tokens": args.max_tokens,
        "runner": str(runner_path),
        "cut_model": str(model_path),
        "cut_model_sha256": _sha256(model_path),
        "bridge": result.to_dict(),
        "incomplete_sessions": incomplete_sessions,
        "production_admitted": False,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if result.closed_cleanly else 1


if __name__ == "__main__":
    raise SystemExit(main())
