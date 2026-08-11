import hashlib
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference_service import model_runtime_sidecar as sidecar


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trial_request(model_path, *, model_format="safetensors", files=None):
    return {
        "schema_version": 1,
        "operation": "trial_load",
        "request_id": "test-request",
        "artifact_id": "test-artifact",
        "engine": "pytorch",
        "runtime_profile": "test",
        "trust_remote_code": False,
        "model_path": str(model_path.resolve()),
        "format": model_format,
        "files": files or [{"path": "weights.safetensors", "size": 1, "sha256": _digest(b"x")}],
    }


def test_validate_request_accepts_local_trial_load(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    sidecar._validate_request(_trial_request(model_dir))


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation", "unknown"),
        ("trust_remote_code", True),
        ("format", "pickle"),
        ("files", []),
    ],
)
def test_validate_request_rejects_unsafe_or_incomplete_inputs(tmp_path, field, value):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    request = _trial_request(model_dir)
    request[field] = value

    with pytest.raises(ValueError):
        sidecar._validate_request(request)


def test_verify_artifact_accepts_matching_safetensors_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    data = b"model-data"
    (model_dir / "weights.safetensors").write_bytes(data)
    request = _trial_request(
        model_dir,
        files=[{"path": "weights.safetensors", "size": len(data), "sha256": _digest(data)}],
    )

    assert sidecar._verify_artifact(request) >= 0


@pytest.mark.parametrize(
    "size,digest,error",
    [
        (99, _digest(b"model-data"), "artifact size mismatch"),
        (10, _digest(b"other-data"), "artifact digest mismatch"),
    ],
)
def test_verify_file_rejects_size_and_digest_mismatches(tmp_path, size, digest, error):
    artifact = tmp_path / "weights.safetensors"
    artifact.write_bytes(b"model-data")

    with pytest.raises(ValueError, match=error):
        sidecar._verify_file(artifact, size, digest)


@pytest.mark.parametrize("artifact_path", ["../outside.safetensors", "absolute"])
def test_verify_artifact_rejects_paths_outside_safetensors_model_root(tmp_path, artifact_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "outside.safetensors"
    data = b"outside"
    outside.write_bytes(data)
    path = str(outside) if artifact_path == "absolute" else artifact_path
    request = _trial_request(
        model_dir,
        files=[{"path": path, "size": len(data), "sha256": _digest(data)}],
    )

    with pytest.raises(ValueError, match="escapes model directory"):
        sidecar._verify_artifact(request)
