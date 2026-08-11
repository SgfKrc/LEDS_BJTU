"""Contracts for resolving the pinned Ollama Gemma 4 native candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.model_tools.gemma4_native_artifacts import (
    resolve_ollama_gemma4_12b,
    run_ollama_gemma4_12b_preflight,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    model_payload = b"fixture-model"
    mmproj_payload = b"fixture-mmproj"
    model_digest = _digest(model_payload)
    mmproj_digest = _digest(mmproj_payload)
    root = tmp_path / "private-ollama"
    blob_root = root / "blobs"
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "12b"
    blob_root.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    (blob_root / f"sha256-{model_digest}").write_bytes(model_payload)
    (blob_root / f"sha256-{mmproj_digest}").write_bytes(mmproj_payload)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_digest}",
                "size": len(model_payload),
            },
            {
                "mediaType": "application/vnd.ollama.image.projector",
                "digest": f"sha256:{mmproj_digest}",
                "size": len(mmproj_payload),
                "from": "mmproj-gemma-4-12B-it-bf16.gguf",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lock = {
        "schema_version": 1,
        "tool": "gemma4_native_assets",
        "candidate_id": "fixture",
        "source": {
            "registry": "registry.ollama.ai/library/gemma4",
            "tag": "12b",
            "manifest_sha256": _digest(manifest_path.read_bytes()),
            "model_id": "fixture-model-id",
            "license": "Apache-2.0",
        },
        "artifacts": {
            "model": {
                "media_type": "application/vnd.ollama.image.model",
                "sha256": model_digest,
                "size_bytes": len(model_payload),
                "architecture": "gemma4",
                "base_model": "Gemma 4 12B",
                "context_length": 262144,
            },
            "mmproj": {
                "media_type": "application/vnd.ollama.image.projector",
                "sha256": mmproj_digest,
                "size_bytes": len(mmproj_payload),
                "from": "mmproj-gemma-4-12B-it-bf16.gguf",
                "architecture": "clip",
                "base_model": "Gemma 4 12B",
                "vision_projector_type": "gemma4uv",
            },
        },
    }
    lock_path = tmp_path / "fixture-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return root, lock_path


def _inspect(path: Path) -> dict:
    if path.read_bytes() == b"fixture-model":
        return {"metadata": {
            "general.architecture": "gemma4",
            "general.base_model.0.name": "Gemma 4 12B",
            "gemma4.context_length": 262144,
        }}
    return {"metadata": {
        "general.architecture": "clip",
        "general.type": "mmproj",
        "general.base_model.0.name": "Gemma 4 12B",
        "clip.has_vision_encoder": True,
        "clip.has_audio_encoder": True,
        "clip.vision.projector_type": "gemma4uv",
    }}


def test_resolver_verifies_pinned_manifest_content_and_pair_without_path_leak(tmp_path: Path):
    root, lock_path = _fixture(tmp_path)

    report, pair = resolve_ollama_gemma4_12b(
        models_root=root,
        full_hash=True,
        lock_path=lock_path,
        inspect_fn=_inspect,
    )

    assert report["valid"] is True
    assert report["identity_verified"] is True
    assert report["content_verified"] is True
    assert report["gate_passed"] is True
    assert report["metadata"]["mmproj_has_vision_encoder"] is True
    assert pair is not None and set(pair) == {"model", "mmproj"}
    assert str(tmp_path) not in str(report)


def test_resolver_rejects_manifest_or_content_drift_without_path_leak(tmp_path: Path):
    root, lock_path = _fixture(tmp_path)
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "12b"
    manifest.write_text("{}", encoding="utf-8")

    manifest_report, pair = resolve_ollama_gemma4_12b(
        models_root=root,
        full_hash=True,
        lock_path=lock_path,
        inspect_fn=_inspect,
    )

    assert manifest_report["gate_passed"] is False
    assert manifest_report["errors"][0]["code"] == "manifest_digest_mismatch"
    assert pair is None
    assert str(tmp_path) not in str(manifest_report)

    root, lock_path = _fixture(tmp_path / "content")
    changed = next((root / "blobs").glob("sha256-*"))
    changed.write_bytes(b"changed")
    content_report, pair = resolve_ollama_gemma4_12b(
        models_root=root,
        full_hash=True,
        lock_path=lock_path,
        inspect_fn=_inspect,
    )

    assert content_report["gate_passed"] is False
    assert content_report["errors"][0]["code"] == "artifact_verification_failed"
    assert pair is None
    assert str(tmp_path) not in str(content_report)


def test_combined_preflight_uses_only_verified_pair_and_redacts_paths(tmp_path: Path):
    pair = {"model": tmp_path / "private-model.gguf", "mmproj": tmp_path / "private-mmproj.gguf"}
    assets = {
        "valid": True,
        "gate_passed": True,
        "artifacts": {
            "model": {"digest": "a" * 64},
            "mmproj": {"digest": "b" * 64},
        },
        "errors": [],
    }
    observed = {}

    def resolver(**kwargs):
        observed["resolver"] = kwargs
        return assets, pair

    def runner(**kwargs):
        observed["runner"] = kwargs
        return {
            "status": "resource_rejected",
            "gate_passed": False,
            "errors": [{"code": "insufficient_ram", "message": "fixture"}],
        }

    result = run_ollama_gemma4_12b_preflight(
        models_root=tmp_path,
        n_ctx=128,
        asset_resolver=resolver,
        probe_runner=runner,
    )

    assert observed["resolver"]["full_hash"] is True
    assert observed["runner"]["model_sha256"] == "a" * 64
    assert observed["runner"]["mmproj_sha256"] == "b" * 64
    assert result["status"] == "resource_rejected"
    assert result["gate_passed"] is False
    assert result["errors"][0]["code"] == "insufficient_ram"
    assert str(tmp_path) not in str(result)
