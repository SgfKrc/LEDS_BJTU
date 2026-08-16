from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.model_tools import gemma4_native_binding as binding


def test_binding_sources_match_the_frozen_lock():
    binding.verify_binding_sources()


def test_binding_marker_round_trip(tmp_path):
    site_packages = tmp_path / "site-packages"
    (site_packages / "llama_cpp").mkdir(parents=True)

    marker = binding.write_binding_marker(site_packages)

    assert marker.name == binding.MARKER_FILENAME
    binding.validate_binding_marker(site_packages)


def test_binding_source_verification_rejects_patch_digest_drift(tmp_path, monkeypatch):
    marker = binding.expected_binding_marker()
    patch = tmp_path / marker["patch"]["path"]
    patch.parent.mkdir(parents=True)
    patch.write_bytes(b"changed patch")
    llama_lock = tmp_path / "llama.lock.json"
    llama_lock.write_text(
        json.dumps({"upstream": {"revision": marker["upstream"]["revision"]}}),
        encoding="utf-8",
    )
    lock = tmp_path / "binding.lock.json"
    lock.write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(binding, "ROOT", tmp_path)
    monkeypatch.setattr(binding, "LOCK_PATH", lock)
    monkeypatch.setattr(binding, "LLAMA_LOCK_PATH", llama_lock)

    with pytest.raises(RuntimeError, match="patch digest"):
        binding.verify_binding_sources()


def test_cuda_package_carries_and_verifies_the_binding_manifest():
    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging" / "qlh-cuda.spec").read_text(encoding="utf-8")
    builder = (root / "scripts" / "model_tools" / "build-cuda-llamacpp.bat").read_text(
        encoding="utf-8"
    )

    assert "_validate_gemma4_binding_marker" in spec
    assert "_verify_gemma4_binding_sources" in spec
    assert "gemma4_native_binding.lock.json" in spec
    assert "_GEMMA4_BINDING_MARKER" in spec
    assert "gemma4_native_binding.py" in builder
    assert "--write-marker" in builder
