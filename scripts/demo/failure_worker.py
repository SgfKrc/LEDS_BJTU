"""Loopback task worker used by the DEF-P2 failure-injection demo.

The worker completes the real v2 admission handshake, accepts one Stage, and
then exits abruptly with a fixed code.  It never imports or loads a model.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scheduler import Scheduler  # noqa: E402
from tcp_comm import TCPClient  # noqa: E402


INJECTED_EXIT_CODE = 23


def main() -> int:
    parser = argparse.ArgumentParser(description="DEF-P2 injected-failure worker")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()

    capabilities = {
        "stage_types": ["full_inference"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": args.model_id,
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "defense-fixture-v1",
            "sha256": args.model_sha256,
        }],
        "max_concurrency": 1,
    }

    def execute_stage(_request, _cancel_event):
        print("[QLH-FAILURE-WORKER] injected_exit_after_stage_accept", flush=True)
        os._exit(INJECTED_EXIT_CODE)

    scheduler = Scheduler()
    scheduler._host = SimpleNamespace(
        full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute_stage,
    )
    scheduler._role_override = "client"
    scheduler.get_effective_node_id = lambda: args.node_id
    scheduler._task_worker_capabilities = lambda: capabilities
    TCPClient._compute_local_model_sha256 = staticmethod(lambda: "")
    client = TCPClient(
        server_host="127.0.0.1",
        server_port=args.port,
        client_id=args.node_id,
        role="client",
    )
    scheduler._tcp_client = client
    try:
        if not client.connect(
            on_message=lambda outer: scheduler._on_tcp_message("master", outer),
        ):
            return 2
        if not scheduler._send_task_worker_hello(client):
            return 3
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            coordinator = scheduler._task_worker_control.coordinator_snapshot()
            if coordinator.get("manual_stage_dispatch_enabled"):
                break
            time.sleep(0.02)
        else:
            return 4

        # A healthy run never reaches this timeout: execute_stage terminates
        # the process as soon as the first accepted Stage starts.
        while time.monotonic() < deadline:
            time.sleep(0.05)
        return 5
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
