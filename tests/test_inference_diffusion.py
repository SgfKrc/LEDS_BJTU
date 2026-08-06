import os
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusion.artifacts import DiffusionArtifact
from diffusion.sd15_engine import SD15EngineConfig
from diffusion.service import DiffusionConflictError, DiffusionUnsupportedError
from inference_service.engine_host import EngineHost
from inference_service.routes import router
from model_host import ModelHost


class _DiffusionStub:
    is_loaded = False
    is_busy = False

    def __init__(self):
        self.generated = None
        self.generated_owner_scope = None
        self.loaded = None
        self.asset_downloads = []
        self.uploaded = None
        self.edit_request = None
        self.artifact = DiffusionArtifact(
            path="C:/models/sd15",
            artifact_kind="sd15_pipeline",
            precision="fp16",
            loadable=True,
        )

    def snapshot(self):
        return {"state": "unloaded", "loaded": False}

    def inspect(self, path, *, compute_hash=False):
        return self.artifact

    def register_artifact(self, path, **kwargs):
        return SimpleNamespace(
            snapshot=lambda **_ignored: {
                "artifact_id": kwargs.get("artifact_id") or "sd-local",
                "artifact": self.artifact.to_dict(),
            }
        )

    def list_artifacts(self, *, include_path=False):
        return [{"artifact_id": "sd-local", "artifact": self.artifact.to_dict()}]

    def asset_catalog(self):
        return [{"asset_id": "sd15_original_v1", "installed": True}]

    def asset_status(self, asset_id):
        return {"asset_id": asset_id, "state": "completed", "installed": True}

    def download_asset(self, asset_id, **kwargs):
        self.asset_downloads.append((asset_id, kwargs))
        return {"asset_id": asset_id, "state": "queued"}

    def import_asset(self, asset_id, path, **kwargs):
        return {"asset_id": asset_id, "path": path, "valid": kwargs["license_accepted"]}

    def load(self, artifact_id, config):
        self.loaded = (artifact_id, config)
        return {"state": "loaded", "loaded": True}

    def unload(self):
        return {"state": "unloaded", "loaded": False}

    def submit_generation(self, generation, *, owner_scope='local'):
        self.generated = generation
        self.generated_owner_scope = owner_scope
        return {"job_id": "sdjob_test", "state": "queued"}

    def put_input_blob(self, data, *, purpose, owner_scope):
        self.uploaded = (data, purpose, owner_scope)
        return {
            'blob_id': 'img_input',
            'purpose': purpose,
            'content_type': 'image/png',
            'size_bytes': len(data),
            'width': 16,
            'height': 16,
        }

    def submit_edit(self, edit_request, *, owner_scope):
        self.edit_request = (edit_request, owner_scope)
        if edit_request.mode not in {'img2img', 'reference'}:
            raise DiffusionUnsupportedError('edit executor is not installed')
        return {"job_id": "sdedit_test", "state": "queued", "kind": "edit"}

    def get_job(self, job_id):
        return {"job_id": job_id, "state": "completed", "blob": {"blob_id": "img_test"}}

    def cancel_job(self, job_id):
        return {"accepted": True, "job": {"job_id": job_id}}

    def get_blob(self, blob_id):
        return SimpleNamespace(
            blob_id=blob_id,
            data=b"png-data",
            content_type="image/png",
            sha256="b" * 64,
        )

    def delete_blob(self, blob_id):
        return blob_id == "img_test"


def _client():
    app = FastAPI()
    host = EngineHost()
    host._host = SimpleNamespace(
        model_loaded=False,
        is_loaded=False,
        full_chat_execution_lock=threading.RLock(),
    )
    host._diffusion = _DiffusionStub()
    app.state.engine_host = host
    app.state.node_role = "master"
    app.include_router(router)
    return TestClient(app), host


def test_inference_diffusion_lifecycle_and_generation_contract():
    client, host = _client()
    assert client.get("/v1/diffusion/capabilities").status_code == 200

    registered = client.post(
        "/v1/diffusion/artifacts/register",
        json={"path": "C:/models/sd15", "artifact_id": "sd-local"},
    )
    assert registered.status_code == 200
    assert "path" not in registered.json()["artifact"]

    loaded = client.post(
        "/v1/diffusion/load",
        json={"artifact_id": "sd-local", "profile": "unet_8bit_qkv"},
    )
    assert loaded.status_code == 200
    config = host._diffusion.loaded[1]
    assert config.quantization == "bitsandbytes_8bit_unet"
    assert config.enable_qkv_fusion is True
    assert config.enable_attention_slicing is False

    generated = client.post(
        "/v1/diffusion/generate",
        json={
            "preset_id": "sd15_original_v1",
            "seed": 17,
            "steps": 3,
            "scheduler": "PNDMScheduler",
        },
    )
    assert generated.status_code == 202
    assert host._diffusion.generated.seed == 17
    assert host._diffusion.generated.steps == 3
    assert host._diffusion.generated.scheduler == "PNDMScheduler"
    assert host._diffusion.generated_owner_scope == 'inference-local'

    assert client.get("/v1/diffusion/jobs/sdjob_test").status_code == 200
    assert client.post("/v1/diffusion/jobs/sdjob_test/cancel").status_code == 200
    blob = client.get("/v1/diffusion/blobs/img_test")
    assert blob.status_code == 200
    assert blob.headers["content-type"] == "image/png"
    assert blob.content == b"png-data"
    assert client.delete("/v1/diffusion/blobs/img_test").status_code == 200
    assert client.post("/v1/diffusion/unload").status_code == 200


def test_inference_diffusion_upload_and_edit_contract():
    client, host = _client()
    uploaded = client.post(
        '/v1/diffusion/blobs',
        data={'purpose': 'input_image'},
        files={'file': ('source.webp', b'webp-body', 'image/webp')},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()['purpose'] == 'input_image'
    assert host._diffusion.uploaded == (
        b'webp-body',
        'input_image',
        'inference-local',
    )

    img2img = client.post(
        '/v1/diffusion/edit',
        json={
            'mode': 'img2img',
            'source_blob_id': 'img_input',
            'prompt': 'change the lighting',
            'strength': 0.6,
        },
    )
    assert img2img.status_code == 202
    assert img2img.json()['job_id'] == 'sdedit_test'
    assert host._diffusion.edit_request[1] == 'inference-local'

    reference = client.post(
        '/v1/diffusion/edit',
        json={
            'mode': 'reference',
            'source_blob_id': 'img_input',
            'prompt': 'same person in a city',
            'edit_adapter_id': 'ip-adapter',
            'ip_adapter_scale': 0.65,
        },
    )
    assert reference.status_code == 202
    assert host._diffusion.edit_request[0].mode == 'reference'
    assert host._diffusion.edit_request[0].ip_adapter_scale == 0.65

    edited = client.post(
        '/v1/diffusion/edit',
        json={
            'mode': 'instruction',
            'source_blob_id': 'img_input',
            'instruction': 'turn it into a sketch',
        },
    )
    assert edited.status_code == 501
    assert edited.json()['detail']['code'] == 'DIFFUSION_UNSUPPORTED'
    assert host._diffusion.edit_request[0].mode == 'instruction'


def test_inference_diffusion_asset_contract_uses_pinned_proxy_policy():
    client, host = _client()
    catalog = client.get("/v1/diffusion/assets/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["assets"][0]["installed"] is True

    started = client.post(
        "/v1/diffusion/assets/sd15_original_v1/download",
        json={"license_accepted": True, "use_local_proxy_fallback": True},
    )
    assert started.status_code == 202
    assert host._diffusion.asset_downloads[0][1]["proxy_fallback"] == "http://127.0.0.1:7897"

    imported = client.post(
        "/v1/diffusion/assets/import",
        json={
            "asset_id": "sd15_original_v1",
            "path": "C:/models/sd15",
            "license_accepted": True,
        },
    )
    assert imported.status_code == 200
    assert imported.json()["valid"] is True


def test_inference_diffusion_rejects_client_role():
    app = FastAPI()
    app.state.engine_host = EngineHost()
    app.state.node_role = "client"
    app.include_router(router)
    response = TestClient(app).get("/v1/diffusion/capabilities")
    assert response.status_code == 404


def test_inference_diffusion_inspect_reports_path_only_on_local_endpoint():
    client, _host = _client()
    inspected = client.post(
        "/v1/diffusion/artifacts/inspect",
        json={"path": "C:/models/sd15"},
    )
    assert inspected.status_code == 200
    assert inspected.json()["path"] == "C:/models/sd15"

    listed = client.get("/v1/diffusion/artifacts")
    assert listed.status_code == 200
    assert "path" not in listed.json()["artifacts"][0]["artifact"]


@pytest.mark.parametrize(
    ("model_loaded", "manager_loaded"),
    [(True, False), (False, True), (True, True)],
)
def test_inference_diffusion_load_rejects_either_llm_loaded_flag(
    model_loaded,
    manager_loaded,
):
    client, host = _client()
    host._host.model_loaded = model_loaded
    host._host.is_loaded = manager_loaded

    response = client.post(
        "/v1/diffusion/load",
        json={"artifact_id": "sd-local", "profile": "balanced"},
    )

    assert response.status_code == 409
    assert host._diffusion.loaded is None


def test_internal_auto_load_path_rejects_active_diffusion_engine():
    _client_instance, host = _client()
    host._diffusion.is_loaded = True
    called = []

    with pytest.raises(DiffusionConflictError, match="unload SD"):
        host._run_exclusive_model_change(lambda: called.append(True))

    assert called == []


def test_llm_unload_holds_the_shared_model_lifecycle_lock():
    _client_instance, host = _client()
    lock = host._host.full_chat_execution_lock
    owned_during_unload = []
    host._host.model_loaded = True
    host._host.unload_model = lambda: owned_during_unload.append(lock._is_owned())

    host.unload_model()

    assert owned_during_unload == [True]


def test_sd_load_does_not_materialize_the_lazy_llm_manager():
    host = EngineHost()
    host._host = ModelHost()
    host._diffusion = _DiffusionStub()
    manager = host._host._manager
    assert manager._instance is None

    host.diffusion_load(
        "sd-local",
        SD15EngineConfig(device="cpu", dtype="float32"),
    )

    assert manager._instance is None


def test_empty_llm_unload_does_not_materialize_the_lazy_manager():
    host = EngineHost()
    host._host = ModelHost()
    manager = host._host._manager
    assert manager._instance is None

    host.unload_model()

    assert manager._instance is None
