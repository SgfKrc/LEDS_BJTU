import base64
import hashlib
import json

import pytest

from harness_workbench.adapters.base import AdapterCapabilities, AdapterModel, AdapterRequest, AdapterResponse, StreamChunk
from harness_workbench.api_layer import create_app
from harness_workbench.image_workbench import (
    GeneratedImage,
    ImageAdapterCapabilities,
    ImageAssetStore,
    ImageRequest,
    LocalImageEngine,
    LocalImageEngineConfig,
    RemoteQLHImageAdapter,
    validate_asset_manifest,
)
from harness_workbench.image_workbench.remote_qlh import RemoteQLHConfig


def _manifest(root, *, asset_id="sd15-test", content=b"weights"):
    (root / "weights.bin").write_bytes(content)
    (root / ".qlh-sd-asset.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "asset": {"asset_id": asset_id, "artifact_id": "sd-test"},
                "files": [
                    {
                        "path": "weights.bin",
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_manifest_validates_hash_and_rejects_traversal(tmp_path):
    _manifest(tmp_path)
    report = validate_asset_manifest(tmp_path, expected_asset_id="sd15-test", full_hash=True)
    assert report.valid is True
    assert report.as_dict()["file_count"] == 1
    (tmp_path / ".qlh-sd-asset.json").write_text(
        json.dumps({"asset": {"asset_id": "sd15-test"}, "files": [{"path": "../escape", "size_bytes": 0}]}),
        encoding="utf-8",
    )
    invalid = validate_asset_manifest(tmp_path)
    assert invalid.valid is False
    assert "manifest_file_path_invalid" in invalid.errors


class _FakeImageExecutor:
    def __init__(self):
        self.calls = []

    def generate(self, request, manifest):
        self.calls.append((request, manifest))
        return GeneratedImage(b"fake-png", width=request.width, height=request.height, seed=request.seed)

    def close(self):
        return None


def test_local_engine_requires_manifest_and_delegates_after_validation(tmp_path):
    _manifest(tmp_path)
    executor = _FakeImageExecutor()
    engine = LocalImageEngine(LocalImageEngineConfig(tmp_path, model_id="sd15-test"), executor=executor)
    request = ImageRequest.from_mapping({"prompt": "a test", "model": "sd15-test", "size": "512x512", "seed": 4})
    result = engine.generate(request)
    assert result.data == b"fake-png"
    assert executor.calls[0][1].asset_id == "sd15-test"
    assert engine.capabilities().supports_txt2img is True


def test_local_engine_reports_unavailable_executor_without_claiming_images(tmp_path):
    _manifest(tmp_path)
    engine = LocalImageEngine(LocalImageEngineConfig(tmp_path, model_id="sd15-test"))
    capabilities = engine.capabilities()
    assert capabilities.runtime_available is False
    assert capabilities.supports_txt2img is False
    with pytest.raises(Exception) as exc_info:
        engine.generate(ImageRequest.from_mapping({"prompt": "test"}))
    assert getattr(exc_info.value, "code", "") == "local_image_runtime_unavailable"


class _FakeRemoteTransport:
    def __init__(self):
        self.posts = []

    def post_json(self, path, payload):
        self.posts.append((path, payload))
        return {"job_id": "job-1", "state": "queued"}

    def get_json(self, path):
        return {"job_id": "job-1", "state": "completed", "blob": {"blob_id": "img-1"}}

    def get_bytes(self, path):
        return b"remote-png", "image/png"


def test_remote_qlh_maps_job_and_blob_contract():
    transport = _FakeRemoteTransport()
    adapter = RemoteQLHImageAdapter(RemoteQLHConfig("http://master:8000", model_id="sd15_original_v1"), transport)
    result = adapter.generate(ImageRequest.from_mapping({"prompt": "test", "model": "sd15_original_v1", "steps": 2}))
    assert result.data == b"remote-png"
    assert transport.posts[0][0] == "/api/diffusion/generate"
    assert transport.posts[0][1]["preset_id"] == "sd15_original_v1"


def test_remote_qlh_decodes_direct_base64_result():
    class DirectTransport(_FakeRemoteTransport):
        def post_json(self, path, payload):
            return {"b64_json": base64.b64encode(b"direct").decode("ascii"), "mime_type": "image/webp"}

    adapter = RemoteQLHImageAdapter(RemoteQLHConfig("http://master:8000"), DirectTransport())
    result = adapter.generate(ImageRequest.from_mapping({"prompt": "test"}))
    assert result.data == b"direct"
    assert result.mime_type == "image/webp"


def test_image_asset_store_round_trip_and_integrity(tmp_path):
    store = ImageAssetStore(tmp_path / "assets")
    record = store.put(GeneratedImage(b"png-data"), prompt="a prompt", owner_scope="user-1")
    loaded, data = store.read(record.asset_id)
    assert loaded.asset_id == record.asset_id
    assert loaded.owner_scope == "user-1"
    assert data == b"png-data"


class _ChatAdapter:
    def capabilities(self):
        return AdapterCapabilities(backend="fake")

    def models(self):
        return (AdapterModel("fake"),)

    def complete(self, request: AdapterRequest):
        return AdapterResponse("id", request.model, "ok")

    def stream(self, request: AdapterRequest):
        yield StreamChunk("id", request.model, {"content": "ok"}, "stop")

    def close(self):
        return None


class _APIImageAdapter:
    def capabilities(self):
        return ImageAdapterCapabilities("fake-image", supports_txt2img=True, runtime_available=True)

    def generate(self, request):
        return GeneratedImage(b"api-png", width=request.width, height=request.height)

    def close(self):
        return None


def test_image_api_returns_base64_and_user_asset_url(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(_ChatAdapter(), image_adapter=_APIImageAdapter(), image_store=ImageAssetStore(tmp_path / "assets"))
    client = TestClient(app)
    response = client.post("/v1/images/generations", json={"prompt": "test", "response_format": "b64_json"})
    assert response.status_code == 200
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"api-png"
    asset_id = response.json()["data"][0]["asset_id"]
    image = client.get(f"/v1/images/assets/{asset_id}")
    assert image.status_code == 200
    assert image.content == b"api-png"
    url_response = client.post("/v1/images/generations", json={"prompt": "test", "response_format": "url"})
    assert url_response.status_code == 200
    assert url_response.json()["data"][0]["url"].startswith("/v1/images/assets/img_")

