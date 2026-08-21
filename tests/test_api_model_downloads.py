"""API layer tests for model download endpoints (P0A, no network)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server


@pytest.fixture(autouse=True)
def _iso_db(tmp_path, monkeypatch):
    """把 SQLite 指向 tmp，避免污染真实库。"""
    monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "test.sqlite3"))
    import local_store
    local_store._initialized_paths.clear()
    yield
    local_store._initialized_paths.clear()


@pytest.fixture
def _sync_executor(monkeypatch):
    """把 api_server 的下载 executor 换成同步执行（job 立即跑完，不真下载）。"""
    class _SyncPool:
        def submit(self, fn):
            fn()
            return None

    monkeypatch.setattr(api_server, "_download_executor", _SyncPool())
    return


def _mk_gguf(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{\"architectures\": [\"LlamaForCausalLM\"]}", encoding="utf-8")
    (d / "model.Q4_K_M.gguf").write_bytes(b"FAKEGGUF" * 2000)
    return d


def _client():
    return TestClient(api_server.app)


def test_list_presets_returns_structure():
    with _client() as c:
        r = c.get("/api/models/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) >= 1
    for p in presets:
        assert {"id", "display", "kind", "hf_repo", "installable", "blocked_reasons"} <= set(p)


def test_create_download_minimal_validation():
    """缺 source 且缺 preset → 400。"""
    with _client() as c:
        r = c.post("/api/models/downloads", json={})
    assert r.status_code == 400
    assert "SOURCE_REQUIRED" in r.text


def test_create_download_from_local_source_ready(tmp_path, _sync_executor):
    src = _mk_gguf(tmp_path, "src")
    with _client() as c:
        r = c.post("/api/models/downloads", json={
            "source": str(src),
            "target": str(tmp_path / "models" / "fake-gguf"),
            "model_id": "fake-gguf",
            "engine": "llama_cpp",
            "quant": "Q4_K_M",
        })
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["status"] == "ready"
    job_id = job["job_id"]

    # 查单个
    with _client() as c:
        r = c.get(f"/api/models/downloads/{job_id}")
    assert r.status_code == 200
    assert r.json()["job"]["status"] == "ready"

    # 列表
    with _client() as c:
        r = c.get("/api/models/downloads")
    assert r.status_code == 200
    assert any(j["job_id"] == job_id for j in r.json()["jobs"])


def test_create_download_unknown_preset_400():
    with _client() as c:
        r = c.post("/api/models/downloads", json={"preset_id": "no-such-preset"})
    assert r.status_code == 400
    assert "PRESET_NOT_FOUND" in r.text


def test_get_missing_job_404():
    with _client() as c:
        r = c.get("/api/models/downloads/nope")
    assert r.status_code == 404


def test_delete_queued_job_cancels(tmp_path, monkeypatch):
    # 让 executor 不真正执行（job 停在 queued），DELETE 可取消
    class _NoopPool:
        def submit(self, fn):
            return None

    monkeypatch.setattr(api_server, "_download_executor", _NoopPool())
    src = _mk_gguf(tmp_path, "srcq")
    with _client() as c:
        r = c.post("/api/models/downloads", json={
            "source": str(src),
            "target": str(tmp_path / "models" / "q"),
            "model_id": "q",
        })
        job_id = r.json()["job"]["job_id"]
        assert r.json()["job"]["status"] == "queued"
        d = c.delete(f"/api/models/downloads/{job_id}")
    assert d.status_code == 200
    assert d.json()["cancelled"] is True
    assert d.json()["job"]["status"] == "cancelled"
