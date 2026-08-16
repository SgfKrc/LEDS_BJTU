from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from scripts.model_tools.cli import main
from scripts.model_tools.qwen3_pipeline_smoke import run_qwen3_pipeline_smoke
from scripts.model_tools import qwen3_pipeline_smoke_worker as worker


def _request(root: Path, **overrides):
    value = {
        "schema_version": 1,
        "operation": "qwen3_pipeline_smoke",
        "read_only": True,
        "network_access": "disabled",
        "model_path": str(root),
        "layer_range": [0, 1],
        "has_embedding": True,
        "has_lm_head": False,
        "controller_python": str(Path("different-python").absolute()),
        "available_ram_bytes": 10**12,
        "available_vram_bytes": 0,
        "device": "cpu",
        "reserve_bytes": 1,
        "safety_margin": 1.0,
    }
    value.update(overrides)
    return value


def _minimal_root(tmp_path: Path) -> Path:
    root = tmp_path / "qwen3"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 2, "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "part.safetensors"}}),
        encoding="utf-8",
    )
    return root


def test_pipeline_worker_reports_budget_before_missing_execution_dependencies(tmp_path, monkeypatch):
    root = _minimal_root(tmp_path)
    monkeypatch.setattr(worker, "select_qwen3_assignment_keys", lambda *args, **kwargs: ["model.layers.0.weight"])
    monkeypatch.setattr(worker, "validate_qwen3_assignment", lambda **kwargs: {"selected_keys": ["model.layers.0.weight"]})
    monkeypatch.setattr(worker, "_assignment_budget", lambda *args, **kwargs: {
        "selected_tensor_count": 1, "selected_tensor_bytes": 1024, "assignment_shard_bytes": 2048, "shard_count": 1,
    })

    def loader(name):
        if name == "transformers":
            return types.SimpleNamespace(__version__="4.57.6")
        raise ModuleNotFoundError(name)

    result = worker.execute_request(_request(root), module_loader=loader)
    assert result["status"] == "runtime_unavailable"
    assert result["assignment"]["selected_tensor_bytes"] == 1024
    assert result["resources"]["passed"] is True
    assert result["errors"][0]["code"] == "sidecar_dependency_missing"


def test_pipeline_worker_rejects_capacity_before_dependency_gate(tmp_path, monkeypatch):
    root = _minimal_root(tmp_path)
    monkeypatch.setattr(worker, "select_qwen3_assignment_keys", lambda *args, **kwargs: ["model.layers.0.weight"])
    monkeypatch.setattr(worker, "validate_qwen3_assignment", lambda **kwargs: {"selected_keys": ["model.layers.0.weight"]})
    monkeypatch.setattr(worker, "_assignment_budget", lambda *args, **kwargs: {
        "selected_tensor_count": 1, "selected_tensor_bytes": 1024, "assignment_shard_bytes": 2048, "shard_count": 1,
    })
    result = worker.execute_request(_request(root, available_ram_bytes=10), module_loader=lambda name: (_ for _ in ()).throw(AssertionError(name)))
    assert result["status"] == "resource_rejected"
    assert result["errors"][0]["code"] == "insufficient_assignment_capacity"


def test_pipeline_controller_forwards_assignment_and_keeps_structured_status(tmp_path):
    report = run_qwen3_pipeline_smoke(
        model=tmp_path,
        layer_range=(2, 4),
        has_embedding=False,
        has_lm_head=True,
        worker_runner=lambda request, timeout: {
            "schema_version": 1, "tool": "qwen3_pipeline_smoke", "operation": "qwen3_pipeline_smoke",
            "valid": True, "gate_passed": False, "status": "resource_rejected", "errors": [],
        },
    )
    assert report["status"] == "resource_rejected"


def test_pipeline_cli_returns_one_for_resource_or_runtime_block(capsys, tmp_path, monkeypatch):
    report = {
        "schema_version": 1, "tool": "qwen3_pipeline_smoke", "operation": "qwen3_pipeline_smoke",
        "valid": True, "gate_passed": False, "status": "runtime_unavailable", "errors": [],
        "assignment": {"layer_range": [0, 1], "selected_tensor_count": 1, "selected_tensor_bytes": 2},
        "resources": {"device": "cpu", "required_device_bytes": 3, "available_ram_bytes": 2},
        "execution": {"attempted": False},
    }
    monkeypatch.setattr("scripts.model_tools.cli.run_qwen3_pipeline_smoke", lambda **kwargs: report)
    assert main(["qwen3-pipeline-smoke", "--model", str(tmp_path), "--start-layer", "0", "--end-layer", "1", "--json"]) == 1
    assert '"runtime_unavailable"' in capsys.readouterr().out


def test_cleanup_stale_assignments_removes_legacy_staging(tmp_path):
    """启动清理：删除模型根下遗留 assignment staging，保留无关目录。"""
    root = tmp_path / "model-root"
    root.mkdir()
    stale = root.parent / ".qlh-qwen3-assignment-stale1"
    stale.mkdir()
    (stale / "config.json").write_text("x", encoding="utf-8")
    keep = root.parent / "unrelated-dir"
    keep.mkdir()

    worker._cleanup_stale_assignments(root)
    assert not stale.exists(), "遗留 staging 应被清理"
    assert keep.exists(), "无关目录不应被误删"


def test_prepare_assignment_cleans_up_on_midway_failure(tmp_path):
    """_prepare_filtered_assignment 中途失败（shard 缺失）时 staging 自清理。"""
    root = tmp_path / "model-root"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    # 请求一个不存在的 shard -> _populate 中途抛错
    before = set(root.parent.glob(".qlh-qwen3-assignment-*"))
    with pytest.raises(Exception):
        worker._prepare_filtered_assignment(
            root, ["missing.key"],
            {"missing.key": "no-such-shard.safetensors"})
    after = set(root.parent.glob(".qlh-qwen3-assignment-*"))
    assert after == before, "中途失败不应留下 staging 残留"


def test_cache_length_prefers_expected_sequence_dimension():
    import torch

    cache = ((torch.zeros(1, 8, 19, 128), torch.zeros(1, 8, 19, 128)),)
    assert worker._cache_sequence_length(cache, expected=19) == 19
