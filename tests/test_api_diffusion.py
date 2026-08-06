import io
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
from diffusion.artifacts import DiffusionArtifact
from diffusion.service import (
    DiffusionConflictError,
    DiffusionNotFoundError,
    DiffusionService,
    DiffusionUnsupportedError,
)
from model_host import model_host
from model_host import ModelHost


class _ApiService:
    is_loaded = False
    is_busy = False

    def __init__(self):
        self.load_calls = []
        self.generation = None
        self.generation_owner_scope = None
        self.cancelled = []
        self.asset_downloads = []
        self.artifact = DiffusionArtifact(
            path="C:/models/sd15",
            artifact_kind="sd15_pipeline",
            precision="fp16",
            loadable=True,
        )
        self.uploaded = None
        self.edit_request = None

    def snapshot(self):
        return {
            "state": "unloaded",
            "loaded": False,
            "dependencies": {"diffusers": True},
        }

    def inspect(self, path, *, compute_hash=False):
        assert path
        return self.artifact

    def register_artifact(self, path, **kwargs):
        return SimpleNamespace(
            snapshot=lambda **_ignored: {
                "artifact_id": kwargs.get("artifact_id") or "sd-local",
                "name": kwargs.get("name") or "SD local",
                "artifact": self.artifact.to_dict(),
            }
        )

    def list_artifacts(self, *, include_path=False):
        assert include_path is False
        return [{"artifact_id": "sd-local", "artifact": self.artifact.to_dict()}]

    def asset_catalog(self):
        return [{"asset_id": "sd15_original_v1", "installed": True}]

    def asset_status(self, asset_id):
        return {"asset_id": asset_id, "state": "completed", "installed": True}

    def download_asset(self, asset_id, **kwargs):
        if not kwargs["license_accepted"]:
            raise ValueError("license acceptance is required")
        self.asset_downloads.append((asset_id, kwargs["proxy_fallback"]))
        return {"asset_id": asset_id, "state": "queued"}

    def import_asset(self, asset_id, path, **kwargs):
        if not kwargs["license_accepted"]:
            raise ValueError("license acceptance is required")
        return {"asset_id": asset_id, "valid": True, "path": path}

    def load(self, artifact_id, config):
        self.load_calls.append((artifact_id, config))
        return {"state": "loaded", "loaded": True}

    def unload(self):
        return {"state": "unloaded", "loaded": False}

    def submit_generation(self, generation, *, owner_scope='local'):
        self.generation = generation
        self.generation_owner_scope = owner_scope
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

    def submit_edit(self, request, *, owner_scope):
        self.edit_request = (request, owner_scope)
        if request.mode not in {'img2img', 'reference'}:
            raise DiffusionUnsupportedError('edit executor is not installed')
        return {"job_id": "sdedit_test", "state": "queued", "kind": "edit"}

    def get_job(self, job_id):
        if job_id == "missing":
            raise DiffusionNotFoundError("missing job")
        return {"job_id": job_id, "state": "completed", "blob": {"blob_id": "img_test"}}

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        return {"accepted": True, "job": {"job_id": job_id, "state": "running"}}

    def get_blob(self, blob_id):
        if blob_id == "missing":
            raise DiffusionNotFoundError("missing blob")
        return SimpleNamespace(
            blob_id=blob_id,
            data=b"png-data",
            content_type="image/png",
            sha256="a" * 64,
        )

    def delete_blob(self, blob_id):
        return blob_id != "missing"


@pytest.fixture
def diffusion_api(monkeypatch):
    service = _ApiService()
    monkeypatch.setattr(ModelHost, 'has_loaded_model', lambda _self: False)
    monkeypatch.setattr(api_server, "diffusion_service", service)
    monkeypatch.setattr(model_host, "model_loaded", False)
    return TestClient(api_server.app), service


def test_diffusion_api_exposes_capabilities_and_hides_registered_path(diffusion_api):
    client, _service = diffusion_api
    capabilities = client.get("/api/diffusion/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["local_llm_loaded"] is False
    assert {item["preset_id"] for item in capabilities.json()["presets"]} >= {
        "sd15_original_v1",
        "sd15_retrovers_space_courier_v1",
    }

    listed = client.get("/api/diffusion/artifacts")
    assert listed.status_code == 200
    assert "path" not in listed.json()["artifacts"][0]["artifact"]


def test_local_artifact_inspection_and_registration_contract(diffusion_api):
    client, _service = diffusion_api
    inspected = client.post(
        "/api/diffusion/artifacts/inspect",
        json={"path": "C:/models/sd15"},
    )
    assert inspected.status_code == 200
    assert inspected.json()["path"] == "C:/models/sd15"

    registered = client.post(
        "/api/diffusion/artifacts/register",
        json={"path": "C:/models/sd15", "artifact_id": "demo"},
    )
    assert registered.status_code == 200
    assert registered.json()["artifact_id"] == "demo"
    assert "path" not in registered.json()["artifact"]


def test_pinned_asset_download_and_import_require_explicit_license_acceptance(diffusion_api):
    client, service = diffusion_api
    catalog = client.get("/api/diffusion/assets/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["assets"][0]["asset_id"] == "sd15_original_v1"

    rejected = client.post(
        "/api/diffusion/assets/sd15_original_v1/download",
        json={"license_accepted": False},
    )
    assert rejected.status_code == 400

    started = client.post(
        "/api/diffusion/assets/sd15_original_v1/download",
        json={"license_accepted": True, "use_local_proxy_fallback": True},
    )
    assert started.status_code == 202
    assert service.asset_downloads == [
        ("sd15_original_v1", "http://127.0.0.1:7897")
    ]
    imported = client.post(
        "/api/diffusion/assets/import",
        json={
            "asset_id": "sd15_original_v1",
            "path": "C:/models/sd15",
            "license_accepted": True,
        },
    )
    assert imported.status_code == 200
    assert imported.json()["valid"] is True


@pytest.mark.parametrize(
    ("model_loaded", "manager_loaded"),
    [(True, False), (False, True), (True, True)],
)
def test_load_rejects_llm_conflict_before_touching_sd_engine(
    diffusion_api,
    monkeypatch,
    model_loaded,
    manager_loaded,
):
    client, service = diffusion_api
    monkeypatch.setattr(model_host, "model_loaded", model_loaded)
    monkeypatch.setattr(
        api_server,
        "model_manager",
        SimpleNamespace(is_loaded=manager_loaded),
    )

    response = client.post(
        "/api/diffusion/load",
        json={"artifact_id": "sd-local", "profile": "unet_8bit_qkv"},
    )
    assert response.status_code == 409
    assert service.load_calls == []


def test_generate_preset_returns_job_and_blob_contract(diffusion_api):
    client, service = diffusion_api
    generated = client.post(
        "/api/diffusion/generate",
        json={"preset_id": "sd15_original_v1", "seed": 9, "steps": 2},
    )
    assert generated.status_code == 202
    assert generated.json() == {"job_id": "sdjob_test", "state": "queued"}
    assert service.generation.seed == 9
    assert service.generation.steps == 2
    assert service.generation_owner_scope == 'local'

    job = client.get("/api/diffusion/jobs/sdjob_test")
    assert job.status_code == 200
    cancelled = client.post("/api/diffusion/jobs/sdjob_test/cancel")
    assert cancelled.status_code == 200
    assert service.cancelled == ["sdjob_test"]

    blob = client.get("/api/diffusion/blobs/img_test")
    assert blob.status_code == 200
    assert blob.content == b"png-data"
    assert blob.headers["cache-control"] == "private, no-store"


def test_diffusion_api_upload_and_edit_contract(diffusion_api):
    client, service = diffusion_api
    uploaded = client.post(
        '/api/diffusion/blobs',
        data={'purpose': 'input_image'},
        files={'file': ('source.png', b'png-body', 'image/png')},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()['blob_id'] == 'img_input'
    assert service.uploaded == (b'png-body', 'input_image', 'local')

    edit = client.post(
        '/api/diffusion/edit',
        json={
            'mode': 'img2img',
            'source_blob_id': 'img_input',
            'prompt': 'change the lighting',
            'strength': 0.6,
        },
    )
    assert edit.status_code == 202
    assert edit.json()['job_id'] == 'sdedit_test'
    assert service.edit_request[0].strength == 0.6
    assert service.edit_request[1] == 'local'

    reference = client.post(
        '/api/diffusion/edit',
        json={
            'mode': 'reference',
            'source_blob_id': 'img_input',
            'prompt': 'same person in a city',
            'edit_adapter_id': 'ip-adapter',
            'ip_adapter_scale': 0.65,
        },
    )
    assert reference.status_code == 202
    assert service.edit_request[0].mode == 'reference'
    assert service.edit_request[0].edit_adapter_id == 'ip-adapter'
    assert service.edit_request[0].ip_adapter_scale == 0.65

    unsupported = client.post(
        '/api/diffusion/edit',
        json={
            'mode': 'instruction',
            'source_blob_id': 'img_input',
            'instruction': 'turn it into a sketch',
        },
    )
    assert unsupported.status_code == 501
    assert unsupported.json()['detail']['code'] == 'DIFFUSION_UNSUPPORTED'


def test_diffusion_upload_uses_magic_bytes_instead_of_declared_mime(monkeypatch):
    from PIL import Image

    output = io.BytesIO()
    Image.new('RGB', (8, 8), 127).save(output, format='PNG')
    service = DiffusionService(inspector=SimpleNamespace())
    monkeypatch.setattr(api_server, 'diffusion_service', service)
    client = TestClient(api_server.app)
    try:
        accepted = client.post(
            '/api/diffusion/blobs',
            data={'purpose': 'input_image'},
            files={'file': ('spoofed.txt', output.getvalue(), 'text/plain')},
        )
        assert accepted.status_code == 201
        assert accepted.json()['content_type'] == 'image/png'

        rejected = client.post(
            '/api/diffusion/blobs',
            data={'purpose': 'input_image'},
            files={'file': ('fake.png', b'not-a-png', 'image/png')},
        )
        assert rejected.status_code == 400
        assert rejected.json()['detail']['code'] == 'DIFFUSION_INVALID_INPUT'
    finally:
        service.close()


def test_diffusion_api_maps_missing_and_invalid_requests(diffusion_api):
    client, _service = diffusion_api
    missing_job = client.get("/api/diffusion/jobs/missing")
    assert missing_job.status_code == 404
    assert missing_job.json()["detail"]["code"] == "DIFFUSION_NOT_FOUND"
    assert client.get("/api/diffusion/blobs/missing").status_code == 404
    assert client.delete("/api/diffusion/blobs/missing").status_code == 404

    invalid = client.post("/api/diffusion/generate", json={"prompt": ""})
    assert invalid.status_code == 400


def test_remote_client_cannot_probe_server_model_paths():
    request = SimpleNamespace(client=SimpleNamespace(host="100.64.0.2"))
    with pytest.raises(api_server.HTTPException) as exc:
        api_server._require_local_diffusion_path_access(request)
    assert exc.value.status_code == 403


def test_llm_model_change_rejects_loaded_diffusion_service(monkeypatch):
    service = SimpleNamespace(is_loaded=True, is_busy=False)
    monkeypatch.setattr(api_server, "diffusion_service", service)
    with pytest.raises(api_server.HTTPException) as exc:
        api_server._run_exclusive_model_change(lambda: {"success": True})
    assert exc.value.status_code == 409


def test_monolith_sd_load_check_does_not_materialize_lazy_llm_manager(monkeypatch):
    host = ModelHost()
    manager = host._manager
    assert manager._instance is None
    monkeypatch.setattr(api_server, "model_host", host)
    monkeypatch.setattr(api_server, "model_manager", host)

    assert api_server._local_llm_is_loaded() is False
    assert manager._instance is None
