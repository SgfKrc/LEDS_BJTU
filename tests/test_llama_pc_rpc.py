from __future__ import annotations

from pathlib import Path

from scripts.llama_pc_rpc import build_plan, plan_report


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
    assert plan.gpu_layers == 8
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
