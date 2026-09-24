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


# ---------------------------------------------------------------- ★ R-R2：丢包下的重试/退避
# §26 诊断出的硬边界：丢包 5% ⇒ 单次 SSH 连接成功率约 1/3 ⇒ `accept()` 干等超时（冒烟 failed）。
# 修法 = **重试 + 指数退避**；下面把三块可测逻辑钉住（退避序列 / 探活重试 / 建连重试）。


def test_smoke_flows_use_connect_with_retry():
    """★ 接线回归：两条冒烟流程都必须**真的**用上可重试建连（否则重试形同虚设）。

    单测重试逻辑不够 —— 若 `_run_surface` / `_run_voter` 仍走裸 `listener.accept()`，
    丢包硬边界（§26）一点没解决。
    """
    import inspect

    surface = inspect.getsource(smoke._run_surface)
    assert "_connect_with_retry" in surface
    assert "listener.accept()" not in surface        # 不许再直接等 accept

    quorum = inspect.getsource(smoke._run_quorum_exchange)
    assert "_connect_with_retry" in quorum
    assert "listener.accept()" not in quorum


def test_retry_defaults_are_bounded():
    """默认重试次数/退避上限必须**有界**（弱网下不能让冒烟无限挂住）。"""
    assert 2 <= smoke.SSH_RETRY_ATTEMPTS <= 5
    assert 0 < smoke.SSH_RETRY_BASE_S <= smoke.SSH_RETRY_MAX_S <= 10.0
    assert max(smoke.retry_delays(smoke.SSH_RETRY_ATTEMPTS)) <= smoke.SSH_RETRY_MAX_S


def test_retry_delays_is_bounded_exponential_backoff():
    """★ 退避**有上限**且**严格不减**：弱网下"多试几次"胜过"一直等"（且不能让冒烟挂太久）。"""
    assert smoke.retry_delays(4, base_s=0.5, max_s=4.0) == [0.5, 1.0, 2.0, 4.0]
    long_plan = smoke.retry_delays(8, base_s=0.5, max_s=4.0)
    assert long_plan[-1] == 4.0
    assert all(delay <= 4.0 for delay in long_plan)
    assert long_plan == sorted(long_plan)
    assert smoke.retry_delays(0) == []
    with pytest.raises(ValueError):
        smoke.retry_delays(-1)


def test_connect_with_retry_recovers_against_real_socket_and_processes():
    """★ 用**真 socket + 真子进程**验证重试：前两个 worker 故意不回连，第三个才连上。

    直接针对 §26 的失败点 —— `accept()` 干等超时：**短超时 + 重试**必须能恢复，
    且失败那两次起的进程要被回收（真进程，不是 mock）。
    """
    import socket as _socket
    import subprocess as _subprocess
    import sys as _sys

    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    spawned: list = []

    def start_worker():
        if len(spawned) < 2:
            # 模拟"SSH 建连失败"：进程起了但从不回连（丢包链路上的常见结局）
            code = "import time; time.sleep(30)"
        else:
            code = (
                "import socket, time\n"
                "time.sleep(0.05)\n"
                f"conn = socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
                "conn.sendall(b'ready\\n')\n"
                "time.sleep(0.5)\n"
            )
        process = _subprocess.Popen([_sys.executable, "-c", code])
        spawned.append(process)
        return process

    try:
        process, connection = smoke._connect_with_retry(
            listener, start_worker, attempts=3, first_frame_timeout_s=0.6)
        with connection:
            assert connection.recv(16) == b"ready\n"
        assert len(spawned) == 3
        assert spawned[0].poll() is not None and spawned[1].poll() is not None   # 失败者已回收
        assert process is spawned[2] and process.poll() is None                  # 成功者仍在
    finally:
        for process in spawned:
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                pass
        listener.close()


def test_connect_with_retry_recovers_after_slow_client():
    """反向情形：首个 worker 回连慢于首超时、第二个才连上 ⇒ 仍成功（重试窗口真的起作用）。"""
    import socket as _socket
    import subprocess as _subprocess
    import sys as _sys

    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = int(listener.getsockname()[1])
    spawned: list = []

    def start_worker():
        delay = 0.45 if not spawned else 0.05      # 第一次刻意慢于 0.3 s 的首超时
        code = (
            "import socket, time\n"
            f"time.sleep({delay})\n"
            f"conn = socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
            "conn.sendall(b'late\\n')\n"
            "time.sleep(0.5)\n"
        )
        process = _subprocess.Popen([_sys.executable, "-c", code])
        spawned.append(process)
        return process

    try:
        _process, connection = smoke._connect_with_retry(
            listener, start_worker, attempts=3, first_frame_timeout_s=0.3)
        with connection:
            assert connection.recv(16) == b"late\n"
        assert len(spawned) == 2                   # 第一个超时被放弃，第二个成功
    finally:
        for process in spawned:
            try:
                process.kill()
            except Exception:  # noqa: BLE001
                pass
        listener.close()


def test_ssh_works_retries_with_backoff_until_success():
    """探活：前两次失败、第三次成功 ⇒ `True`，且**退避确实睡了**（不是空转重试）。"""
    calls: list[object] = []
    slept: list[float] = []

    def runner(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=0 if len(calls) >= 3 else 255, stdout="", stderr="")

    assert smoke._ssh_works("host", attempts=3, runner=runner, sleeper=slept.append) is True
    assert len(calls) == 3
    assert slept == [0.5, 1.0]          # 只在**失败之后**睡；成功即返回


def test_ssh_works_gives_up_without_extra_sleep():
    """全部失败 ⇒ `False`；调用次数 == attempts，且**最后一次失败后不再睡**。"""
    calls: list[int] = []
    slept: list[float] = []

    def runner(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=255, stdout="", stderr="")

    assert smoke._ssh_works("host", attempts=3, runner=runner, sleeper=slept.append) is False
    assert len(calls) == 3
    assert slept == [0.5, 1.0]


def test_connect_with_retry_restarts_worker_and_reaps_failures():
    """★ `accept()` 超时（§26 的失败点）⇒ **重起 worker**（旧进程必须回收）后再试。

    单次 `accept()` 在丢包链路下不够 —— 这正是"重试/退避"要修的正面行为。
    """
    started: list[str] = []
    reaped: list[str] = []

    class _Worker:
        def __init__(self, index: int) -> None:
            self.index = index

        def kill(self) -> None:
            reaped.append(f"worker-{self.index}")

    class _Listener:
        def __init__(self) -> None:
            self.count = 0

        def settimeout(self, _value: float) -> None:   # 真实 listener 必须被设超时
            pass

        def accept(self):
            self.count += 1
            if self.count < 3:
                raise TimeoutError("timed out")
            return ("connection", None)

    def start_worker() -> _Worker:
        worker = _Worker(len(started))
        started.append(f"worker-{worker.index}")
        return worker

    slept: list[float] = []
    process, connection = smoke._connect_with_retry(
        _Listener(), start_worker, attempts=3, sleeper=slept.append)

    assert connection == "connection"
    assert started == ["worker-0", "worker-1", "worker-2"]
    assert reaped == ["worker-0", "worker-1"]        # 只回收失败的，最后一次保留
    assert process.index == 2
    assert slept == [0.5, 1.0]                       # ★ 成功即返回；只在失败后睡


def test_connect_with_retry_reaps_last_worker_when_all_attempts_fail():
    """底线：全部尝试都失败 ⇒ 抛错（带 `process` 供诊断），中间失败的进程**都**被回收。

    ⚠️ **最后一次不回收** —— 调用方要用它的 stderr 定位失败原因（回收责任交给调用方）。
    """
    reaped: list[int] = []

    class _Worker:
        def __init__(self, index: int) -> None:
            self.index = index

        def kill(self) -> None:
            reaped.append(self.index)

    class _Listener:
        def settimeout(self, _value: float) -> None:
            pass

        def accept(self):
            raise TimeoutError("timed out")

    started: list[_Worker] = []

    def start_worker() -> _Worker:
        worker = _Worker(len(started))
        started.append(worker)
        return worker

    with pytest.raises(smoke.PhysicalSmokeError) as excinfo:
        smoke._connect_with_retry(_Listener(), start_worker, attempts=3,
                                  sleeper=lambda _delay: None)

    assert reaped == [0, 1]                       # 前两次被回收
    assert excinfo.value.process is started[-1]   # 最后一次挂到异常上（供诊断）
