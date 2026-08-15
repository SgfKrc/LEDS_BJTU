"""Regression tests for the G4.4 independent artifact trust root."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.model_tools import gemma4_native_freeze as freeze


def _configure_fixture(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")

    def spec(path):
        return {
            "source": f"https://example.invalid/{path.name}",
            "license": "apache-2.0",
            "filename": path.name,
            "expected_size": path.stat().st_size,
            "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    artifacts = {"main_gguf": spec(model), "mmproj": spec(mmproj)}
    monkeypatch.setattr(freeze, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(freeze, "LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(freeze, "ARTIFACTS", artifacts)
    return model, mmproj


def test_freeze_rejects_content_outside_the_pinned_identity(monkeypatch, tmp_path):
    model, _mmproj = _configure_fixture(monkeypatch, tmp_path)
    model.write_bytes(b"other")
    with pytest.raises(SystemExit, match="SHA-256"):
        freeze._freeze()
    assert not freeze.LOCK.exists()


def test_check_rejects_a_rewritten_lock(monkeypatch, tmp_path):
    _configure_fixture(monkeypatch, tmp_path)
    record = freeze._freeze()
    record["artifacts"]["main_gguf"]["sha256"] = "0" * 64
    freeze.LOCK.write_text(json.dumps(record), encoding="utf-8")
    assert freeze._check() == 1
