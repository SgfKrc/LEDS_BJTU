"""Model one-click download job service tests (P0A, no real network)."""

from __future__ import annotations

import sys
import json
import threading
from pathlib import Path

import pytest

sys.path.insert(0, "src")

import model_download_jobs as mj


def _make_gguf_dir(base: Path, *, real_sig: bool = True) -> Path:
    """构造一个含 config.json + 单个 .gguf 的最小合法模型目录并返回其路径。"""
    d = base / "fake-gguf"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{\"architectures\": [\"LlamaForCausalLM\"]}", encoding="utf-8")
    g = d / "model.Q4_K_M.gguf"
    payload = b"FAKEGGUF" * 2000  # > 0 bytes, non-empty
    g.write_bytes(payload)
    return d


def _run_job_sync(**kw):
    """同步 executor：create_job 时立即把 _run_job 调度完（不启后台线程）。"""
    done = threading.Event()

    def executor(fn):
        fn()
        done.set()

    return done


def test_list_presets_has_expected_fields():
    presets = mj.list_presets()
    assert len(presets) == 5
    for p in presets:
        assert {"id", "display", "kind", "hf_repo", "installable"} <= set(p)
        assert "blocked_reasons" in p


def test_create_job_from_local_source_no_network(tmp_path, monkeypatch):
    """本地目录作为 source + sync executor → job 走到 ready，且不触网。"""
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src")
    target = tmp_path / "models" / "fake-gguf"

    # 本地目录 source：_download 跳过真实下载（source 是目录时 _run_job 直接取权重）
    done = threading.Event()

    def executor(fn):
        fn()
        done.set()

    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf",
        engine="llama_cpp", quant="Q4_K_M",
        models_root=str(models_root), allow_cpu=True,
        executor=executor,
    )
    assert job["status"] == mj.STATUS_READY, job
    assert job["model_id"] == "fake-gguf"
    assert job["total_bytes"] > 0
    # 目标已发布 + 注册成功
    assert target.is_dir()
    # job 已持久化（SQLite 表存在）
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_READY


def test_create_job_queued_then_executed(tmp_path, monkeypatch):
    """无 executor → 停在 queued；手动跑 _run 后变 ready。"""
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src2")
    target = tmp_path / "models" / "fake-gguf2"
    # 让 _download 返回 staging 内已造好的权重（不真下载）
    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf2",
        models_root=str(models_root),
    )
    assert job["status"] == mj.STATUS_QUEUED
    # 手动执行 job（模拟线程池调度）
    mj._run_job(
        job["job_id"], source=str(src), target=str(target), model_id="fake-gguf2",
        preset_id="", engine="llama_cpp", quant="Q4_K_M", use_modelscope=False,
        proxy="", expected_sha256="", gguf_path="", models_root=str(models_root),
        allow_cpu=True,
    )
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_READY


def test_sha_mismatch_fails_job(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src3")
    target = tmp_path / "models" / "fake-gguf3"
    job = mj.create_job(
        source=str(src), target=str(target), model_id="fake-gguf3",
        expected_sha256="deadbeef",  # 必然不匹配
        models_root=str(models_root),
    )
    mj._run_job(
        job["job_id"], source=str(src), target=str(target), model_id="fake-gguf3",
        preset_id="", engine="llama_cpp", quant="Q4_K_M", use_modelscope=False,
        proxy="", expected_sha256="deadbeef", gguf_path="",
        models_root=str(models_root), allow_cpu=True,
    )
    reloaded = mj.get_job(job["job_id"])
    assert reloaded["status"] == mj.STATUS_FAILED
    assert reloaded["error_code"] == "SHA256_MISMATCH"
    # staging 已清理，目标未发布
    assert not target.exists()


def test_unknown_preset_raises():
    with pytest.raises(mj.JobError) as exc:
        mj.create_job(preset_id="nope-not-exist", models_root=str(Path(__file__).parent))
    assert exc.value.code == "PRESET_NOT_FOUND"


def test_preset_without_ms_source_blocks_modelscope():
    # qwen-1_8b-gguf-q4 无 ms_path
    with pytest.raises(mj.JobError) as exc:
        mj.create_job(preset_id="qwen-1_8b-gguf-q4", use_modelscope=True,
                      models_root=str(Path(__file__).parent))
    assert exc.value.code == "PRESET_NO_MS_SOURCE"
    assert exc.value.code == "PRESET_NO_MS_SOURCE"


def test_registered_model_written_to_db(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src4")
    target = tmp_path / "models" / "fake-gguf4"

    def executor(fn):
        fn()

    job = mj.create_job(source=str(src), target=str(target), model_id="fake-gguf4",
                        models_root=str(models_root), executor=executor)
    assert job["status"] == mj.STATUS_READY
    import local_store
    registered = local_store.get_local_experimental_models()
    ids = [m.get("model_id") for m in registered if m.get("model_id") == "fake-gguf4"]
    assert ids, "模型应写入 model_registry"


def test_cancel_only_queued(tmp_path):
    models_root = tmp_path / "models"
    src = _make_gguf_dir(tmp_path / "src5")
    target = tmp_path / "models" / "fake-gguf5"
    job = mj.create_job(source=str(src), target=str(target), model_id="fake-gguf5",
                        models_root=str(models_root))
    assert mj.cancel_job(job["job_id"]) is True
    assert mj.get_job(job["job_id"])["status"] == mj.STATUS_CANCELLED
    # 已完成/失败不可取消
    assert mj.cancel_job("no-such-id") is False
