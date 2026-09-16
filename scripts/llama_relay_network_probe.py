#!/usr/bin/env python3
"""Send a raw float32 hidden-state matrix through the Relay TCP transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.relay_transport import RelayTcpClient, expected_hidden_bytes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--hidden-f32", required=True)
    parser.add_argument("--n-tokens", required=True, type=int)
    parser.add_argument("--n-embd", required=True, type=int)
    parser.add_argument("--expected-token", type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    hidden_path = Path(args.hidden_f32).resolve()
    hidden = hidden_path.read_bytes()
    expected_size = expected_hidden_bytes(args.n_tokens, args.n_embd)
    if len(hidden) != expected_size:
        parser.error(f"hidden size mismatch: got {len(hidden)}, expected {expected_size}")

    with RelayTcpClient(
        args.host,
        args.port,
        n_embd=args.n_embd,
        timeout=args.timeout,
        max_tokens=args.n_tokens,
    ) as client:
        token = client.request_token(hidden, n_tokens=args.n_tokens)

    accepted = args.expected_token is None or token == args.expected_token
    report = {
        "status": "evidence_accepted" if accepted else "evidence_rejected",
        "criterion": "expected_argmax" if args.expected_token is not None else "transport_only",
        "transport": "loopback_tcp",
        "wire_version": 1,
        "hidden_path": str(hidden_path),
        "hidden_bytes": len(hidden),
        "n_tokens": args.n_tokens,
        "n_embd": args.n_embd,
        "token": token,
        "expected_token": args.expected_token,
        "production_admitted": False,
    }
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
