"""Independent Scheduler worker process used by the N2.4 integration test."""

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

from scheduler import Scheduler
from tcp_comm import TCPClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()

    capabilities = {
        "stage_types": ["full_inference", "aggregate"],
        "engines": ["pytorch"],
        "models": [{
            "model_id": args.model_id,
            "engine": "pytorch",
            "format": "safetensors",
            "revision": "process-test",
            "sha256": args.model_sha256,
        }],
        "max_concurrency": 1,
    }
    completed = threading.Event()

    def execute_stage(request, cancel_event):
        if cancel_event.is_set():
            raise RuntimeError("process Stage was cancelled before execution")
        completed.set()
        return {
            "content": f"process:{request.stage_id}",
            "worker_pid": os.getpid(),
            "worker_node_id": args.node_id,
        }

    sys.modules["api_server"] = SimpleNamespace(
        _full_chat_execution_lock=threading.RLock(),
        _execute_task_worker_stage=execute_stage,
    )
    scheduler = Scheduler()
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
            on_message=lambda outer: scheduler._on_tcp_message(
                "master", outer,
            ),
        ):
            return 2
        if not scheduler._send_task_worker_hello(client):
            return 3
        admission_deadline = time.time() + 5.0
        while time.time() < admission_deadline:
            coordinator = scheduler._task_worker_control.coordinator_snapshot()
            if coordinator.get("manual_stage_dispatch_enabled"):
                break
            time.sleep(0.02)
        else:
            return 4
        if not completed.wait(20.0):
            return 5
        active_deadline = time.time() + 5.0
        while (
            scheduler._task_worker_active_attempts
            and time.time() < active_deadline
        ):
            time.sleep(0.02)
        time.sleep(0.1)
        return 0
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
