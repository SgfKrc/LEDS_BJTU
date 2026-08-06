import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.assets import (
    ASSET_CATALOG,
    MANIFEST_NAME,
    ORIGINAL_REPO,
    ORIGINAL_REVISION,
    RETRO_REVISION,
    DiffusionAssetFile,
    DiffusionAssetManager,
    DiffusionAssetSpec,
    verify_asset_directory,
)


def _tiny_spec(model_index: bytes, weight: bytes) -> DiffusionAssetSpec:
    revision = "1" * 40
    return DiffusionAssetSpec(
        asset_id="tiny",
        artifact_id="tiny",
        name="Tiny fixture",
        repo_id="test/tiny",
        revision=revision,
        local_dir="models/tiny",
        license_id="test-license",
        model_card_url="https://example.invalid/tiny",
        preset_id="sd15_original_v1",
        files=(
            DiffusionAssetFile("model_index.json", len(model_index), "test/tiny", revision),
            DiffusionAssetFile(
                "unet/weight.safetensors",
                len(weight),
                "test/tiny",
                revision,
                hashlib.sha256(weight).hexdigest(),
            ),
        ),
    )


def _pipeline_index() -> bytes:
    return json.dumps({"_class_name": "StableDiffusionPipeline"}).encode("utf-8")


def _write_fixture(target: Path, model_index: bytes, weight: bytes) -> None:
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        (target / component).mkdir(parents=True, exist_ok=True)
    (target / "model_index.json").write_bytes(model_index)
    (target / "unet" / "weight.safetensors").write_bytes(weight)


def test_retro_catalog_freezes_revision_and_composes_pinned_safety_checker():
    spec = ASSET_CATALOG["sd15_90s_retrovers_v1"]

    assert spec.revision == RETRO_REVISION
    assert spec.composed is True
    safety_files = [item for item in spec.files if item.path.startswith("safety_checker/")]
    assert safety_files
    assert {item.source_repo for item in safety_files} == {ORIGINAL_REPO}
    assert {item.source_revision for item in safety_files} == {ORIGINAL_REVISION}
    assert all(item.sha256 for item in safety_files if item.path.endswith(".safetensors"))


def test_verifier_rejects_same_size_weight_with_wrong_hash(tmp_path, monkeypatch):
    model_index = _pipeline_index()
    spec = _tiny_spec(model_index, b"good")
    monkeypatch.setitem(ASSET_CATALOG, "tiny", spec)
    target = tmp_path / "models" / "tiny"
    _write_fixture(target, model_index, b"evil")

    report = verify_asset_directory(target, "tiny", full_hash=True)

    assert report["valid"] is False
    assert report["hash_mismatches"][0]["path"] == "unet/weight.safetensors"


def test_composed_asset_rejects_an_unbound_safety_checker(tmp_path, monkeypatch):
    model_index = _pipeline_index()
    spec = replace(_tiny_spec(model_index, b"good"), composed=True)
    monkeypatch.setitem(ASSET_CATALOG, "tiny", spec)
    target = tmp_path / "models" / "tiny"
    _write_fixture(target, model_index, b"good")

    report = verify_asset_directory(target, "tiny", full_hash=True)

    assert report["valid"] is False
    assert len(report["composition_errors"]) == 3


def test_background_download_requires_license_writes_manifest_and_calls_ready(
    tmp_path,
    monkeypatch,
):
    model_index = _pipeline_index()
    weight = b"weight"
    spec = _tiny_spec(model_index, weight)
    monkeypatch.setitem(ASSET_CATALOG, "tiny", spec)
    ready = []

    def download(*, spec, target, proxy_url):
        assert spec.asset_id == "tiny"
        assert proxy_url == ""
        _write_fixture(target, model_index, weight)

    manager = DiffusionAssetManager(root=tmp_path, on_ready=lambda spec, path: ready.append((spec, path)), download_fn=download)
    with pytest.raises(ValueError, match="license"):
        manager.start_download("tiny", license_accepted=False)

    manager.start_download("tiny", license_accepted=True)
    deadline = time.time() + 5
    status = manager.status("tiny")
    while status["state"] not in manager.TERMINAL_STATES and time.time() < deadline:
        time.sleep(0.01)
        status = manager.status("tiny")

    assert status["state"] == "completed", status
    assert status["installed"] is True
    assert (tmp_path / "models" / "tiny" / MANIFEST_NAME).is_file()
    assert ready and ready[0][0].artifact_id == "tiny"


def test_discover_installed_notifies_each_asset_path_only_once(tmp_path, monkeypatch):
    model_index = _pipeline_index()
    weight = b"weight"
    spec = _tiny_spec(model_index, weight)
    monkeypatch.setitem(ASSET_CATALOG, "tiny", spec)
    target = spec.target_path(tmp_path)
    _write_fixture(target, model_index, weight)
    report = verify_asset_directory(target, "tiny", full_hash=True)
    assert report["valid"] is True
    from diffusion.assets import _write_manifest

    _write_manifest(target, spec, report)
    ready = []
    manager = DiffusionAssetManager(
        root=tmp_path,
        on_ready=lambda ready_spec, path: ready.append((ready_spec.asset_id, path)),
    )

    manager.discover_installed()
    manager.discover_installed()

    assert ready == [("tiny", target.resolve())]


def test_ready_notification_can_retry_after_registration_failure(tmp_path, monkeypatch):
    model_index = _pipeline_index()
    weight = b"weight"
    spec = _tiny_spec(model_index, weight)
    monkeypatch.setitem(ASSET_CATALOG, "tiny", spec)
    target = spec.target_path(tmp_path)
    _write_fixture(target, model_index, weight)
    report = verify_asset_directory(target, "tiny", full_hash=True)
    assert report["valid"] is True
    from diffusion.assets import _write_manifest

    _write_manifest(target, spec, report)
    attempts = []

    def on_ready(_spec, _path):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary registration failure")

    manager = DiffusionAssetManager(root=tmp_path, on_ready=on_ready)

    with pytest.raises(RuntimeError, match="temporary"):
        manager.discover_installed()
    manager.discover_installed()

    assert len(attempts) == 2
