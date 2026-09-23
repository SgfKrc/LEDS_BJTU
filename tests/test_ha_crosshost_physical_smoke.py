from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import ha_crosshost_physical_smoke as smoke


def test_y700_probe_uses_dynamic_adb_serial_and_arm64_model_gate(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "model=TB321FU\n"
                "sdk=35\n"
                "abi=arm64-v8a,armeabi-v7a\n"
                "nproc=8\n"
                "available_mem_kb=123\n"
                "model_root=/data/data/com.termux/files/home/storage/shared/Download/QLH/models\n"
                "model_root_exists=true\n"
                "gguf_count=6\n"
            ),
            stderr="",
        ),
    )

    result = smoke._run_y700("100.99.211.13:40397", adb="adb")

    assert result["status"] == "passed"
    assert result["transport"] == "adb_wireless_debugging"
    assert result["serial"] == "100.99.211.13:40397"
    assert result["model_gate"] == "ready_for_arm64_model_smoke"
    assert result["observations"]["gguf_count"] == "6"


def test_y700_probe_records_ssh_failure_without_raising(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="connection timed out",
        ),
    )

    result = smoke._run_y700("100.99.211.13:40397", adb="adb")

    assert result["status"] == "failed"
    assert result["error_code"] == "adb_failed"
    assert result["model_gate"] == "blocked_no_gguf"


def test_y700_probe_discovers_online_dynamic_serial(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:] == ["devices", "-l"]:
            return SimpleNamespace(
                returncode=0,
                stdout="List of devices attached\n100.99.211.13:40397\tdevice product:TB321FU\n",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout="model=TB321FU\nabi=arm64-v8a\ngguf_count=1\n",
            stderr="",
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    result = smoke._run_y700(adb="adb", host="100.99.211.13")

    assert result["status"] == "passed"
    assert result["serial"] == "100.99.211.13:40397"
    assert calls[0] == ["adb", "devices", "-l"]
    assert calls[1][0:3] == ["adb", "-s", "100.99.211.13:40397"]


def test_y700_probe_requires_dynamic_serial_when_no_device_is_online(monkeypatch):
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="List of devices attached\n",
            stderr="",
        ),
    )

    result = smoke._run_y700(adb="adb", host="100.99.211.13")

    assert result["status"] == "failed"
    assert result["error_code"] == "dynamic_adb_serial_required"
    assert "dynamic_port" in result["hint"]


def test_quorum_exchange_flag_is_wired_and_off_by_default():
    """★ 跨机 quorum 交换：开关存在、默认关闭、报告始终带 `quorum` 字段（旧行为不变）。"""
    import inspect

    source = inspect.getsource(smoke.main)
    assert "--quorum-exchange" in source
    assert "args.quorum_exchange" in source
    assert '"quorum": quorum' in source


def test_remote_voter_surfaces_stable_error_code():
    """远端 voter 拒绝时必须以 `QuorumError` 带回**稳定错误码**（不吞、不换成通用异常）。"""
    import socket

    left, right = socket.socketpair()
    try:
        voter = smoke._RemoteVoter(left, voter_id="voter-b")
        right.sendall(b'{"ok":false,"code":"quorum_unavailable"}\n')
        with pytest.raises(smoke.QuorumError) as excinfo:
            voter.reserve_term("voter-a")
    finally:
        left.close()
        right.close()
    message = f"{excinfo.value}{getattr(excinfo.value, 'code', '')}"
    assert "quorum_unavailable" in message


def test_weaknet_bridge_stream_adapters_cover_stdio_and_socket():
    """★ 弱网代理的读写适配：socket 走 recv/sendall，stdio 走 read/write（混用会 AttributeError）。"""
    import io
    import socket as _socket

    left, right = _socket.socketpair()
    try:
        right.sendall(b"hello")
        assert smoke._read_stream(left) == b"hello"
        smoke._write_stream(left, b"world")
        assert right.recv(16) == b"world"
    finally:
        left.close()
        right.close()

    buffer = io.BytesIO(b"stdio-payload")
    assert smoke._read_stream(buffer) == b"stdio-payload"
    sink = io.BytesIO()
    smoke._write_stream(sink, b"out")
    assert sink.getvalue() == b"out"


def test_weaknet_bridge_is_routed_before_smoke_flow():
    """`--weaknet-bridge` 必须在冒烟流程**之前**分流（否则 `ProxyCommand` 调用会真去跑冒烟）。"""
    import inspect

    source = inspect.getsource(smoke.main)
    assert "--weaknet-bridge" in source
    assert source.index("_weaknet_bridge") < source.index("_run_surface")
