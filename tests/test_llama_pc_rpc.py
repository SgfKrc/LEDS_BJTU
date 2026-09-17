from __future__ import annotations

from pathlib import Path

import pytest

from scripts.llama_pc_rpc import (
    build_plan,
    build_remote_asset_sync_plan,
    plan_report,
    sync_remote_model,
)


def test_pc_rpc_plan_targets_tailscale_worker_without_model_argument(tmp_path: Path):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"gguf")
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    plan = build_plan(model, runtime)
    report = plan_report(plan)

    assert report["remote_target"] == "surface@100.100.52.106"
    assert report["rpc_endpoint"] == "100.100.52.106:50163"
    assert report["loopback"] is False
    assert report["worker_backend"] == "CPU"
    assert report["remote_worker_receives_model"] is False
    assert "--model" not in report["remote_worker_command"]
    assert "--model" in report["host_command"]


def test_pc_rpc_defaults_pin_qwen_18b_identity_path():
    plan = build_plan()

    assert plan.remote_model.endswith(r"models\Qwen-1_8B-Chat.Q4_K_M.gguf")
    assert plan.worker_budget_mib == 512
    assert plan.gpu_layers is None
    assert plan.auto_split is True
    assert plan.remote_threads == 8


def test_pc_rpc_ssh_tunnel_uses_loopback_worker_and_host():
    plan = build_plan(ssh_tunnel=True)
    report = plan_report(plan)

    assert report["ssh_tunnel"] is True
    assert report["loopback"] is True
    assert report["rpc_endpoint"] == "127.0.0.1:50163"
    assert report["remote_worker_command"][2] == "127.0.0.1"
    assert "127.0.0.1:50163" in report["host_command"]
    assert "-L" in report["ssh_tunnel_command"]
    assert "50163:127.0.0.1:50163" in report["ssh_tunnel_command"]


def test_remote_asset_sync_plan_is_sha_verified_and_non_destructive(tmp_path: Path):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"verified-gguf")

    sync_plan = build_remote_asset_sync_plan(
        model,
        "surface@100.100.52.106",
        r"C:\Users\surface\Documents\LEDS_BJTU\models\qwen.gguf",
    )

    assert sync_plan.source_size_bytes == len(b"verified-gguf")
    assert len(sync_plan.source_sha256) == 64
    assert sync_plan.temporary_path.endswith(".part")
    assert sync_plan.backup_dir.endswith(r"_to_delete\remote-models")
    assert "Get-FileHash" in sync_plan.commit_script
    assert "Move-Item" in sync_plan.commit_script
    assert "Remove-Item" not in sync_plan.prepare_script + sync_plan.commit_script
    assert "--model" not in sync_plan.scp_command

    availability = sync_plan.as_pipeline_artifact_availability(
        "rpc-edge", 25, sync_status="already_current",
    )
    assert availability.layer_range == (0, 25)
    assert availability.model_sha256 == sync_plan.source_sha256
    assert availability.artifact_kind == "gguf"


def test_remote_asset_plan_is_not_availability_before_success(tmp_path: Path):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"not-synced")
    sync_plan = build_remote_asset_sync_plan(
        model,
        "surface@100.100.52.106",
        r"C:\Users\surface\Documents\LEDS_BJTU\models\qwen.gguf",
    )

    with pytest.raises(ValueError, match="successful sync"):
        sync_plan.as_pipeline_artifact_availability(
            "rpc-edge", 25, sync_status="dry_run",
        )


def test_remote_asset_sync_plan_rejects_relative_or_traversal_paths(tmp_path: Path):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="absolute Windows path"):
        build_remote_asset_sync_plan(model, "surface@worker", "models\\qwen.gguf")
    with pytest.raises(ValueError, match="unsafe component"):
        build_remote_asset_sync_plan(
            model,
            "surface@worker",
            r"C:\Users\surface\..\qwen.gguf",
        )


def test_remote_asset_sync_apply_requires_prior_plan_confirmation(tmp_path: Path, monkeypatch):
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"fixture")
    plan = build_plan(model=model)
    monkeypatch.setattr(
        "scripts.llama_pc_rpc._remote_asset_state",
        lambda *_args: {"status": "missing"},
    )

    with pytest.raises(ValueError, match="prior dry-run"):
        sync_remote_model(plan, apply=True)
