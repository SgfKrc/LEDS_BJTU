import os
import io
import sys
import threading

import pytest
from fastapi import HTTPException
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from diffusion.data_plane import DiffusionDataPlaneRuntime
from task_graph import TaskGraphCoordinator
from task_provider import LocalFullModelProvider, ModelIdentity
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
    def __init__(
        self,
        manifest,
        *,
        block=False,
        provider_id="remote_diffusion_worker_test",
        node_id="worker-test",
    ):
        self.node_id = node_id
        self._manifest = manifest
        self.started = threading.Event()
        self._block = block
        self.calls = []

        def execute(request, cancel_event):
            self.calls.append(request)
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
            provider_id=provider_id,
            node_id=self.node_id,
            supported_stage_types=("image_generate",),
            provider_kind="remote_diffusion_worker",
        )

    def artifact_manifests(self):
        return (dict(self._manifest),)


class _FakeTextProvider(LocalFullModelProvider):
    def __init__(
        self,
        *,
        content="a cinematic mountain cabin at dawn",
        provider_id="remote_text_worker_test",
        node_id="worker-text",
    ):
        self.node_id = node_id
        self.identity = ModelIdentity(
            model_id="qwen-1_8b",
            engine="pytorch",
            format="safetensors",
            revision="local-text-v1",
            sha256="d" * 64,
        )
        self.content = content
        self.requests = []
        self.started = threading.Event()
        self._block = False

        def execute(request, cancel_event):
            self.requests.append(request)
            self.started.set()
            if self._block:
                while not cancel_event.wait(0.01):
                    pass
                raise RuntimeError("cancelled")
            return {"content": self.content, "model": self.identity.model_id}

        super().__init__(
            execute,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=("image_prompt",),
            provider_kind="remote_full_worker",
        )

    def model_identities(self):
        return (self.identity,)

    def supports_model_identity(self, identity, stage_type):
        return identity == self.identity and stage_type == "image_prompt"


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


@pytest.fixture
def mixed_distributed_api(distributed_api, monkeypatch):
    image_provider, coordinator = distributed_api
    text_provider = _FakeTextProvider()
    monkeypatch.setattr(api_server, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr(
        api_server.scheduler,
        "remote_task_worker_providers",
        lambda: [text_provider],
    )
    return text_provider, image_provider, coordinator


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


def test_mixed_workflow_binds_v2_prompt_into_v3_root_input(mixed_distributed_api):
    text_provider, image_provider, coordinator = mixed_distributed_api
    observed = {}

    def execute(request, cancel_event):
        observed["root_input"] = dict(request.root_input)
        observed["dependencies"] = dict(request.dependencies)
        return {
            "image": {
                "blob_id": "img_mixed_test",
                "sha256": "e" * 64,
                "size_bytes": 128,
                "content_type": "image/png",
                "width": request.root_input["width"],
                "height": request.root_input["height"],
                "purpose": "output",
            },
            "metrics": {"seed": request.root_input["seed"]},
        }

    image_provider._executor = execute
    result = api_server._run_distributed_mixed_generation(
        api_server.DiffusionMixedGenerateRequest(
            message="a quiet mountain cabin at dawn",
            seed=42,
            width=512,
            height=512,
            steps=4,
            workflow_id="wf_mixedapi01",
        )
    )

    assert result["status"] == "completed"
    assert result["workflow"]["template"] == "llm_sd15_v1"
    assert result["text_provider_id"] == text_provider.provider_id
    assert result["image_provider_id"] == image_provider.provider_id
    assert result["prompt_contract"]["v3_dependencies"] == "omitted_after_local_binding"
    assert observed["root_input"]["prompt"] == text_provider.content
    assert observed["dependencies"] == {}
    assert text_provider.requests[0].root_input == {
        "message": "a quiet mountain cabin at dawn",
        "task_options": {
            "candidate_max_tokens": 192,
            "temperature": 0.7,
            "top_p": 0.9,
            "show_thinking": False,
        },
    }
    assert coordinator.get("wf_mixedapi01")["state"] == "completed"


def test_mixed_workflow_rejects_text_output_over_contract_limit(mixed_distributed_api):
    text_provider, image_provider, coordinator = mixed_distributed_api
    text_provider.content = "x" * 1001
    with pytest.raises(HTTPException) as exc_info:
        api_server._run_distributed_mixed_generation(
            api_server.DiffusionMixedGenerateRequest(
                message="too long prompt",
                workflow_id="wf_mixedtoolong",
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "MIXED_WORKFLOW_FAILED"
    assert coordinator.get("wf_mixedtoolong")["state"] == "failed"
    assert image_provider.calls == []


def test_mixed_workflow_cancellation_fences_both_channels(mixed_distributed_api):
    text_provider, image_provider, coordinator = mixed_distributed_api
    image_provider._block = True
    request = api_server.DiffusionMixedGenerateRequest(
        message="cancel a mixed workflow",
        workflow_id="wf_mixedcancel01",
    )
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(_call_mixed_workflow(request)),
        daemon=True,
    )
    thread.start()
    assert image_provider.started.wait(2.0)
    assert coordinator.request_cancel("wf_mixedcancel01")
    thread.join(5.0)

    assert not thread.is_alive()
    assert isinstance(outcome[0], HTTPException)
    assert outcome[0].status_code == 409
    assert outcome[0].detail["code"] == "MIXED_WORKFLOW_CANCELLED"
    assert coordinator.get("wf_mixedcancel01")["state"] == "cancelled"
    assert len(text_provider.requests) == 1


def _call_mixed_workflow(request):
    try:
        return api_server._run_distributed_mixed_generation(request)
    except HTTPException as exc:
        return exc


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
    other = _FakeImageProvider(
        _manifest("sd15_worker_third_v1", "d"),
        provider_id="remote_diffusion_worker_other",
        node_id="worker-other",
    )
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


def test_distributed_image_grid_runs_fixed_four_seed_plan(distributed_api):
    provider, coordinator = distributed_api
    data_plane = api_server.app.state.diffusion_data_plane

    def execute(request, cancel_event):
        output = io.BytesIO()
        Image.new(
            "RGB",
            (8, 8),
            (int(request.root_input["seed"]) % 255, 20, 40),
        ).save(output, format="PNG")
        descriptor = data_plane.store.put_bytes(
            output.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope=f"distributed:{request.workflow_id}",
            width=8,
            height=8,
        )
        return {"image": descriptor.snapshot(), "metrics": {"seed": request.root_input["seed"]}}

    provider._executor = execute
    result = api_server._run_distributed_diffusion_grid(
        api_server.DiffusionDistributedGridRequest(
            prompt="a fixed four seed grid",
            seeds=[41, 42, 43, 44],
            width=8,
            height=8,
            steps=1,
            workflow_id="wf_gridapi01",
        )
    )

    assert result["status"] == "completed"
    assert result["distributed"] is False
    assert result["execution_mode"] == "single_node_serial_batch"
    assert result["hardware_validation"] == "pending_two_physical_cuda_pcs"
    assert result["result"]["metrics"]["seeds"] == [41, 42, 43, 44]
    assert len(result["result"]["images"]) == 4
    assert [item["seed"] for item in result["result"]["images"]] == [41, 42, 43, 44]
    assert all(item["node_id"] == "worker-test" for item in result["result"]["images"])
    assert result["workflow"]["template"] == "image_grid_v1"
    assert result["workflow"]["stage_count"] == 5
    assert result["grid_provider_id"] == "diffusion_grid_aggregator"
    assert coordinator.get("wf_gridapi01")["state"] == "completed"


def test_distributed_image_grid_failure_reclaims_completed_seed_blobs(distributed_api):
    provider, coordinator = distributed_api
    data_plane = api_server.app.state.diffusion_data_plane

    def execute(request, cancel_event):
        seed = int(request.root_input["seed"])
        if seed == 52:
            raise RuntimeError("synthetic seed failure")
        output = io.BytesIO()
        Image.new("RGB", (8, 8), (seed % 255, 30, 50)).save(output, format="PNG")
        descriptor = data_plane.store.put_bytes(
            output.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope=f"distributed:{request.workflow_id}",
            width=8,
            height=8,
        )
        return {"image": descriptor.snapshot(), "metrics": {"seed": seed}}

    provider._executor = execute
    with pytest.raises(HTTPException) as exc_info:
        api_server._run_distributed_diffusion_grid(
            api_server.DiffusionDistributedGridRequest(
                prompt="a failing fixed grid",
                seeds=[51, 52, 53, 54],
                width=8,
                height=8,
                steps=1,
                workflow_id="wf_gridfail01",
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "DIFFUSION_GRID_FAILED"
    assert coordinator.get("wf_gridfail01")["state"] == "failed"
    assert data_plane.store.snapshot()["blobs"] == 0


def test_distributed_image_grid_reports_two_actual_workers(distributed_api):
    provider_a, _coordinator = distributed_api
    data_plane = api_server.app.state.diffusion_data_plane
    provider_b = _FakeImageProvider(
        provider_a._manifest,
        provider_id="remote_diffusion_worker_second",
        node_id="worker-second",
    )

    def executor_for(color_offset):
        def execute(request, cancel_event):
            seed = int(request.root_input["seed"])
            output = io.BytesIO()
            Image.new("RGB", (8, 8), ((seed + color_offset) % 255, 50, 70)).save(
                output,
                format="PNG",
            )
            descriptor = data_plane.store.put_bytes(
                output.getvalue(),
                content_type="image/png",
                purpose="output",
                owner_scope=f"distributed:{request.workflow_id}",
                width=8,
                height=8,
            )
            return {"image": descriptor.snapshot(), "metrics": {"seed": seed}}
        return execute

    provider_a._executor = executor_for(0)
    provider_b._executor = executor_for(10)
    original = api_server.scheduler.remote_diffusion_providers
    api_server.scheduler.remote_diffusion_providers = lambda: [provider_b, provider_a]
    try:
        result = api_server._run_distributed_diffusion_grid(
            api_server.DiffusionDistributedGridRequest(
                prompt="a two worker fixed grid",
                seeds=[61, 62, 63, 64],
                width=8,
                height=8,
                steps=1,
                workflow_id="wf_gridnodes01",
            )
        )
    finally:
        api_server.scheduler.remote_diffusion_providers = original
        provider_b.close()

    assert result["distributed"] is True
    assert result["execution_mode"] == "multi_node_batch"
    assert result["node_ids"] == ["worker-second", "worker-test"]
    assert result["provider_ids"] == [
        "remote_diffusion_worker_second",
        "remote_diffusion_worker_test",
    ]
    assert [item["node_id"] for item in result["result"]["images"]] == [
        "worker-second", "worker-test", "worker-second", "worker-test",
    ]
