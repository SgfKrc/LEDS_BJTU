from __future__ import annotations

from pathlib import Path

from scripts.llama_rpc_sim import build_plan, parse_backend_evidence, plan_report


def test_backend_evidence_requires_partial_cpu_and_rpc_buffers():
    server_log = """
    offloaded 12/29 layers to GPU
    CPU_Mapped model buffer size = 2501.98 MiB
    CPU_REPACK model buffer size = 1863.42 MiB
    RPC0[127.0.0.1:50153] model buffer size = 1922.05 MiB
    RPC0[127.0.0.1:50153] KV buffer size = 5.50 MiB
    RPC0[127.0.0.1:50153] compute buffer size = 62.63 MiB
    """
    evidence = parse_backend_evidence(server_log, "Starting RPC server\n", worker_budget_mib=2048)

    assert evidence["offloaded_layers"] == 12
    assert evidence["total_layers"] == 29
    assert evidence["model_buffers_mib"]["RPC0"] == 1922.05
    assert evidence["partial_backend_residency_proven"] is True
    assert evidence["capacity_merge_proven"] is True
    assert evidence["physical_single_process_oom_proven"] is False


def test_full_model_worker_log_is_rejected_as_sharding_evidence():
    server_log = "offloaded 29/29 layers to GPU\nRPC0 model buffer size = 4000 MiB\n"
    worker_log = "loading model full.gguf\n"

    evidence = parse_backend_evidence(server_log, worker_log, worker_budget_mib=2048)

    assert evidence["worker_model_path_seen"] is True
    assert evidence["partial_backend_residency_proven"] is False


def test_plan_is_loopback_and_worker_does_not_receive_model_path(tmp_path: Path):
    model = tmp_path / "sample.gguf"
    model.write_bytes(b"gguf")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "ggml-rpc-server.exe").write_bytes(b"worker")
    (runtime / "llama-server.exe").write_bytes(b"host")

    report = plan_report(build_plan(model, runtime))

    assert report["assets_present"] is True
    assert report["torch_imported"] is False
    assert "--model" not in report["worker_command"]
    assert "127.0.0.1" in report["worker_command"]
    assert report["sharding_contract"]["reject_full_model_copy"] is True
