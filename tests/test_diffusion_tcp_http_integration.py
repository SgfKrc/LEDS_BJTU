import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.coordinator_runtime import DiffusionCoordinatorRuntime  # noqa: E402
from diffusion.data_plane import DiffusionDataPlaneRuntime  # noqa: E402
from scheduler import NodeInfo, NodeRole, Scheduler  # noqa: E402
from task_graph import StageSpec, TaskGraphCoordinator  # noqa: E402
from task_worker_protocol import canonical_sha256  # noqa: E402
from tcp_comm import TCPServer  # noqa: E402


def _manifest():
    body = {
        "artifact_id": "sd15_process_v1",
        "pipeline_kind": "sd15_pipeline",
        "revision": "local-aaaaaaaaaaaa",
        "components": [{
            "artifact_id": "base_pipeline",
            "artifact_kind": "sd15_pipeline",
            "sha256": "a" * 64,
        }],
    }
    return {**body, "sha256": canonical_sha256(body)}


def _wait_for_http(url: str, timeout: float = 10.0):
    from urllib.request import urlopen

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise AssertionError(f"HTTP server did not become ready: {url}")


@pytest.mark.external
def test_sd_image_stage_round_trips_over_real_tcp_and_http(tmp_path, monkeypatch):
    import config as config_module
    import scheduler as scheduler_module

    secret = "diffusion-real-process-secret-32-bytes"
    monkeypatch.setattr(config_module, "CLUSTER_SECRET", secret)
    monkeypatch.setattr(scheduler_module, "RUN_MODE", "distributed")
    monkeypatch.setattr(
        scheduler_module,
        "DIFFUSION_WORKER_EXPERIMENTAL_ENABLED",
        True,
    )
    tcp_port = 0
    http_port = 0
    worker_state = tmp_path / "worker-state"
    coordinator_state = tmp_path / "coordinator-state"
    worker_state.mkdir()
    coordinator_state.mkdir()

    scheduler = Scheduler()
    scheduler._role_override = "master"
    scheduler.get_effective_node_id = lambda: "master_process"
    scheduler.init_nodes()
    scheduler.nodes["worker_process"] = NodeInfo(
        node_id="worker_process",
        role=NodeRole.CLIENT,
        node_type="pc",
    )
    server = TCPServer(host="127.0.0.1", port=tcp_port)
    scheduler._tcp_server = server
    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=coordinator_state,
        cluster_secret=secret,
    )
    coordinator_runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
    assert scheduler.configure_diffusion_coordinator(
        result_ingestor=coordinator_runtime.ingest_result,
        dispatch_enabled=True,
    )
    coordinator = TaskGraphCoordinator(max_records=10)
    process = None
    try:
        server.start(
            on_message=scheduler._on_tcp_message,
            on_disconnect=scheduler._on_tcp_disconnect,
        )
        tcp_port = server.sock.getsockname()[1]
        helper = Path(__file__).parent / "helpers" / "diffusion_worker_process.py"
        env = dict(os.environ)
        env.update({
            "QLH_CLUSTER_SECRET": secret,
            "QLH_DIFFUSION_WORKER_EXPERIMENTAL_ENABLED": "true",
            "PYTHONUTF8": "1",
        })
        process = subprocess.Popen(
            [
                sys.executable,
                str(helper),
                "--tcp-port", str(tcp_port),
                "--http-port", str(http_port),
                "--state-dir", str(worker_state),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_file = worker_state / "http-port"
        deadline = time.time() + 10.0
        while time.time() < deadline and not port_file.exists():
            if process.poll() is not None:
                pytest.fail("diffusion worker exited before publishing its HTTP port")
            time.sleep(0.05)
        assert port_file.exists(), "diffusion worker did not publish its HTTP port"
        http_port = int(port_file.read_text(encoding="ascii"))
        _wait_for_http(
            f"http://127.0.0.1:{http_port}/internal/v1/diffusion/data-plane/status",
        )
        deadline = time.time() + 10.0
        provider = None
        while time.time() < deadline:
            providers = scheduler.remote_diffusion_providers()
            if providers:
                provider = providers[0]
                break
            time.sleep(0.05)
        assert provider is not None, "worker v3 hello did not reach the master"
        coordinator.register_provider(provider)
        manifest = _manifest()
        output, workflow = coordinator.run(
            stages=[StageSpec(
                "image_generate",
                "image_generate",
                provider=provider.provider_id,
                lease_timeout_seconds=30.0,
            )],
            final_stage_id="image_generate",
            root_input={
                "prompt": "a process-isolated test image",
                "negative_prompt": "",
                "seed": 1234,
                "width": 512,
                "height": 512,
                "steps": 1,
                "guidance_scale": 7.5,
                "scheduler": "",
                "artifact_manifest_sha256": manifest["sha256"],
            },
            runtime_context={"diffusion_artifact_manifest": manifest},
            workflow_id="wf_realimage01",
        )
        assert output["image"]["purpose"] == "output"
        assert output["image"]["sha256"]
        assert output["metrics"]["seed"] == 1234
        coordinator.commit_result(workflow["workflow_id"])
        descriptor, data = coordinator_runtime.read_result(
            workflow_id="wf_realimage01",
            blob_id=output["image"]["blob_id"],
        )
        assert descriptor["content_type"] == "image/png"
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        (worker_state / "output-transfer-consumed").write_text(
            "consumed", encoding="ascii",
        )
        assert process.wait(timeout=8) == 0
    finally:
        coordinator.close()
        scheduler.clear_diffusion_coordinator()
        data_plane.close()
        server.stop()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
