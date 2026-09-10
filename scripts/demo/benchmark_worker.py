"""Model-free loopback task worker for the DEF-P3 control-plane benchmark."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="DEF-P3 fixture benchmark worker")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--task-count", required=True, type=int)
    args = parser.parse_args()
    if args.task_count < 1:
        parser.error("--task-count must be positive")

    capabilities = {
        "stage_types": ["full_inference"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": args.model_id,
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "defense-benchmark-v1",
            "sha256": args.model_sha256,
        }],
        "max_concurrency": 1,
    }
    completed = threading.Event()
    count_lock = threading.Lock()
    completed_count = 0

    def execute_stage(_request, cancel_event):
        nonlocal completed_count
        if cancel_event.is_set():
            raise RuntimeError("fixture benchmark Stage was cancelled")
        with count_lock:
            completed_count += 1
            if completed_count >= args.task_count:
                completed.set()
        return {"content": "fixture-benchmark-result"}

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
        admission_deadline = time.monotonic() + 20.0
        while time.monotonic() < admission_deadline:
            coordinator = scheduler._task_worker_control.coordinator_snapshot()
            if coordinator.get("manual_stage_dispatch_enabled"):
                break
            time.sleep(0.02)
        else:
            return 4
        if not completed.wait(60.0):
            return 5
        active_deadline = time.monotonic() + 5.0
        while scheduler._task_worker_active_attempts and time.monotonic() < active_deadline:
            time.sleep(0.02)
        time.sleep(0.1)
        return 0
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
