"""Independent v3 SD worker used by the local TCP+HTTP integration test."""

from __future__ import annotations

import argparse
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

import uvicorn
from fastapi import FastAPI
from PIL import Image

import scheduler as scheduler_module
from diffusion.data_plane import DiffusionDataPlaneRuntime, router
from diffusion.worker_runtime import DiffusionWorkerRuntime
from scheduler import Scheduler


class _FakeService:
    def __init__(self, manifest_sha256: str):
        self._manifest_sha256 = manifest_sha256
        self.completed = threading.Event()
        image_buffer = BytesIO()
        Image.new("RGB", (2, 2), (20, 40, 60)).save(image_buffer, format="PNG")
        self._blob = SimpleNamespace(
            data=image_buffer.getvalue(),
            content_type="image/png",
            width=2,
            height=2,
        )

    def snapshot(self):
        return {
            "loaded": True,
            "loaded_artifact": {
                "artifact_id": "sd15_process_v1",
                "artifact": {
                    "artifact_kind": "sd15_pipeline",
                    "sha256": "a" * 64,
                },
            },
            "engine_config": {"dtype": "float16"},
        }

    def submit_generation(self, request, *, owner_scope):
        return {"job_id": "sdjob_process", "state": "queued"}

    def get_job(self, job_id):
        return {
            "job_id": job_id,
            "state": "completed",
            "output_blob_id": "img_worker_process",
            "metrics": {"elapsed_seconds": 0.01, "seed": 1234},
        }

    def get_blob(self, blob_id):
        return self._blob

    def delete_blob(self, blob_id):
        self.completed.set()
        return True

    def cancel_job(self, job_id):
        return {"accepted": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-port", required=True, type=int)
    parser.add_argument("--http-port", required=True, type=int)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--node-id", default="diffusion_process_worker")
    args = parser.parse_args()

    scheduler_module.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED = True
    service = _FakeService("a" * 64)
    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=args.state_dir,
        cluster_secret=os.environ["QLH_CLUSTER_SECRET"],
    )
    http_app = FastAPI()
    http_app.state.diffusion_data_plane = data_plane
    http_app.include_router(router)
    http_server = uvicorn.Server(
        uvicorn.Config(
            http_app,
            host="127.0.0.1",
            port=args.http_port,
            log_level="warning",
        )
    )
    http_thread = threading.Thread(target=http_server.run, daemon=True)
    http_thread.start()

    runtime = DiffusionWorkerRuntime(
        service=service,
        data_plane=data_plane,
        node_id=args.node_id,
        data_plane_base_url=f"http://127.0.0.1:{args.http_port}",
    )
    scheduler = Scheduler()
    scheduler._role_override = "client"
    scheduler.get_effective_node_id = lambda: args.node_id
    scheduler._tcp_client = None
    capabilities = runtime.capabilities()
    if capabilities is None:
        return 2
    if not scheduler.configure_diffusion_worker(
        capabilities=capabilities,
        executor=runtime.execute,
    ) and scheduler._diffusion_worker_adapter is None:
        return 3

    from tcp_comm import TCPClient

    client = TCPClient(
        server_host="127.0.0.1",
        server_port=args.tcp_port,
        client_id=args.node_id,
        role="client",
        node_type="pc",
    )
    scheduler._tcp_client = client
    try:
        if not client.connect(
            on_message=lambda outer: scheduler._on_tcp_message("master", outer),
        ):
            return 4
        if not scheduler._send_diffusion_worker_hello(client):
            return 5
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if scheduler.get_diffusion_worker_protocol_status().get(
                "control_plane_connected",
            ):
                break
            time.sleep(0.02)
        else:
            return 6
        if not service.completed.wait(20.0):
            return 7
        # The local job result has been published; keep HTTP alive while the
        # coordinator consumes its short-lived output transfer grant.
        time.sleep(1.0)
        return 0
    finally:
        client.disconnect()
        scheduler.clear_diffusion_worker()
        http_server.should_exit = True
        http_thread.join(timeout=5.0)
        data_plane.close()


if __name__ == "__main__":
    raise SystemExit(main())
