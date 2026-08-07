import os
import sys
import threading

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from diffusion.data_plane import DiffusionDataPlaneRuntime
from task_graph import TaskGraphCoordinator
from task_provider import LocalFullModelProvider
from task_worker_protocol import canonical_sha256


def _manifest(artifact_id="sd15_worker_v1", suffix="a"):
    body = {
        "artifact_id": artifact_id,
        "pipeline_kind": "sd15_pipeline",
        "revision": "local-123456789abc",
        "components": [{
            "artifact_id": "base_pipeline",
            "artifact_kind": "sd15_pipeline",
            "sha256": suffix * 64,
        }],
    }
    return {**body, "sha256": canonical_sha256(body)}


class _FakeImageProvider(LocalFullModelProvider):
    def __init__(self, manifest, *, block=False):
        self.node_id = "worker-test"
        self._manifest = manifest
        self.started = threading.Event()
        self._block = block

        def execute(request, cancel_event):
            self.started.set()
            if self._block:
                while not cancel_event.wait(0.01):
                    pass
                raise RuntimeError("cancelled")
            return {
                "image": {
                    "blob_id": "img_distributed_test",
                    "sha256": "b" * 64,
                    "size_bytes": 128,
                    "content_type": "image/png",
                    "width": request.root_input["width"],
                    "height": request.root_input["height"],
                    "purpose": "output",
                },
                "metrics": {"seed": request.root_input["seed"]},
            }

        super().__init__(
            execute,
            provider_id="remote_diffusion_worker_test",
            node_id=self.node_id,
            supported_stage_types=("image_generate",),
            provider_kind="remote_diffusion_worker",
        )

    def artifact_manifests(self):
        return (dict(self._manifest),)


@pytest.fixture
def distributed_api(monkeypatch, tmp_path):
    provider = _FakeImageProvider(_manifest())
    coordinator = TaskGraphCoordinator(max_records=10)
    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    monkeypatch.setattr(api_server, "DIFFUSION_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(api_server, "task_graph_coordinator", coordinator)
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_diffusion_providers",
        lambda: [provider],
    )
    api_server.app.state.diffusion_data_plane = data_plane
    yield provider, coordinator
    data_plane.close()
    api_server.app.state.diffusion_data_plane = None
    coordinator.close()


def test_distributed_image_workflow_returns_completed_single_stage(distributed_api):
    provider, coordinator = distributed_api
    result = api_server._run_distributed_diffusion_generation(
        api_server.DiffusionDistributedGenerateRequest(
            prompt="a test landscape",
            seed=42,
            width=512,
            height=512,
            steps=4,
            workflow_id="wf_imageapi01",
        )
    )

    assert result["status"] == "completed"
    assert result["distributed"] is True
    assert result["provider_id"] == provider.provider_id
    assert result["workflow"]["state"] == "completed"
    assert result["workflow"]["template"] == "image_generate_v1"
    assert result["result"]["image"]["url"].endswith(
        "/wf_imageapi01/blobs/img_distributed_test"
    )
    assert coordinator.get("wf_imageapi01")["state"] == "completed"


def test_distributed_image_workflow_can_be_cancelled(distributed_api):
    provider, _coordinator = distributed_api
    provider._block = True
    request = api_server.DiffusionDistributedGenerateRequest(
        prompt="a cancellable test",
        workflow_id="wf_imagecancel1",
    )
    outcome = []

    thread = threading.Thread(
        target=lambda: outcome.append(_call_workflow(request)),
        daemon=True,
    )
    thread.start()
    assert provider.started.wait(2.0)
    assert api_server.task_graph_coordinator.request_cancel("wf_imagecancel1")
    thread.join(5.0)

    assert not thread.is_alive()
    assert isinstance(outcome[0], HTTPException)
    assert outcome[0].status_code == 409


def _call_workflow(request):
    try:
        return api_server._run_distributed_diffusion_generation(request)
    except HTTPException as exc:
        return exc


def test_distributed_image_workflow_rejects_ambiguous_artifacts(distributed_api):
    provider, _coordinator = distributed_api
    provider._manifest = _manifest("sd15_worker_other_v1", "c")
    other = _FakeImageProvider(_manifest("sd15_worker_third_v1", "d"))
    original = api_server.scheduler.remote_diffusion_providers
    api_server.scheduler.remote_diffusion_providers = lambda: [provider, other]
    try:
        with pytest.raises(HTTPException) as exc_info:
            api_server._run_distributed_diffusion_generation(
                api_server.DiffusionDistributedGenerateRequest(
                    prompt="ambiguous",
                )
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "DIFFUSION_ARTIFACT_SELECTION_REQUIRED"
    finally:
        api_server.scheduler.remote_diffusion_providers = original
        other.close()


def test_distributed_image_workflow_rejects_invalid_workflow_id(distributed_api):
    with pytest.raises(HTTPException) as exc_info:
        api_server._run_distributed_diffusion_generation(
            api_server.DiffusionDistributedGenerateRequest(
                prompt="invalid id",
                workflow_id="not-a-workflow",
            )
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "DIFFUSION_WORKFLOW_INVALID"
