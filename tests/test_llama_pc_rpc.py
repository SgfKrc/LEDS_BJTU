from __future__ import annotations

from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager

import pytest

from scripts.llama_pc_rpc import (
    DEFAULT_LOCAL_MODEL,
    build_plan,
    build_remote_asset_sync_plan,
    default_remote_model,
    plan_report,
    sync_remote_model,
)
from scripts import llama_pc_rpc


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("fixture peer closed")
        data.extend(chunk)
    return bytes(data)


@contextmanager
def _rpc_fixture(*, major: int = 5, minor: int = 0, device_count: int = 1,
                 truncate: bool = False, plain_tcp: bool = False):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                if plain_tcp:
                    conn.sendall(b"not-llama-rpc")
                    return
                command = _recv_exact(conn, 1)[0]
                request_size = struct.unpack("<Q", _recv_exact(conn, 8))[0]
                _recv_exact(conn, request_size)
                assert command == llama_pc_rpc.RPC_CMD_HELLO
                response = struct.pack(
                    "<BBBB24s", major, minor, 0, 0, bytes(24)
                )
                if truncate:
                    conn.sendall(struct.pack("<Q", len(response)))
                    conn.sendall(response[:5])
                    return
                conn.sendall(struct.pack("<Q", len(response)))
                conn.sendall(response)
                command = _recv_exact(conn, 1)[0]
                request_size = struct.unpack("<Q", _recv_exact(conn, 8))[0]
                _recv_exact(conn, request_size)
                assert command == llama_pc_rpc.RPC_CMD_DEVICE_COUNT
                conn.sendall(struct.pack("<Q", 4))
                conn.sendall(struct.pack("<I", device_count))
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        listener.close()
        thread.join(timeout=2)


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


def test_pc_rpc_defaults_derive_remote_model_from_local_name():
    """★ R-R6 复盘（2026-09-25）：远端模型路径**跟随本地文件名**，不再写死已退役的 1.8B 路径。

    旧契约（原名 `...pin_qwen_18b_identity_path`）把 `...\\models\\Qwen-1_8B-Chat.Q4_K_M.gguf`
    钉死，后果：同步任何别的模型都**覆盖那一个文件** ⇒ 出现「文件名说 1.8B、内容是 2b」的名实
    不符，而且**多个模型在远端无法共存**。1.8B 已退役 ⇒ 该契约同步废除。
    """
    plan = build_plan()

    # 缺省 ⇒ 跟随 default_local_model 的文件名
    assert plan.remote_model == default_remote_model(DEFAULT_LOCAL_MODEL)
    assert plan.remote_model.endswith(r"models\qwen3-0.6b-q8_0.gguf")
    # 换本地模型 ⇒ 远端路径跟着变（这才是「多模型可共存」的关键）
    other = build_plan(model="models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf")
    assert other.remote_model.endswith(r"models\qwen35-2b-Q4_K_M.gguf")
    # 显式 --remote-model 仍可覆盖
    pinned = build_plan(remote_model=r"C:\anywhere\pinned.gguf")
    assert pinned.remote_model == r"C:\anywhere\pinned.gguf"

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


def test_rpc_protocol_probe_requires_hello_and_device_count():
    with _rpc_fixture() as port:
        result = llama_pc_rpc._rpc_protocol_probe("127.0.0.1", port)

    assert result == {
        "ok": True,
        "transport": "tcp",
        "protocol_version": "5.0.0",
        "device_count": 1,
    }


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"major": 4}, "protocol mismatch"),
        ({"minor": 1}, "protocol mismatch"),
        ({"device_count": 0}, "zero devices"),
        ({"truncate": True}, "truncated"),
        ({"plain_tcp": True}, "truncated"),
    ],
)
def test_rpc_protocol_probe_fails_closed_for_invalid_ready_responses(fixture_kwargs, message):
    with _rpc_fixture(**fixture_kwargs) as port:
        with pytest.raises(llama_pc_rpc.RpcProtocolError, match=message):
            llama_pc_rpc._rpc_protocol_probe("127.0.0.1", port)


def test_rpc_ready_wait_rejects_process_that_never_listens():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    unused_port = probe.getsockname()[1]
    probe.close()
    try:
        with pytest.raises(RuntimeError, match="did not pass protocol ready"):
            llama_pc_rpc._wait_rpc_ready(
                "127.0.0.1", unused_port, 0.15, process=process, poll_interval=0.01
            )
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_remote_worker_ready_failure_terminates_worker(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = FakeProcess()
    plan = build_plan()
    monkeypatch.setattr(llama_pc_rpc, "_remote_clear_worker_port", lambda _plan: None)
    monkeypatch.setattr(llama_pc_rpc, "_remote_find_worker", lambda _plan: {"pid": 321})
    monkeypatch.setattr(llama_pc_rpc, "_ssh_command", lambda _plan, _script: ["ssh"])
    monkeypatch.setattr(llama_pc_rpc.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        llama_pc_rpc,
        "_wait_rpc_ready",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("RPC not ready")),
    )
    monkeypatch.setattr(llama_pc_rpc, "_ssh", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="RPC not ready"):
        llama_pc_rpc._start_remote_worker(
            plan, "test-run", tmp_path, ready_host="127.0.0.1", ready_timeout=0.1
        )
    assert process.returncode == -15


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
