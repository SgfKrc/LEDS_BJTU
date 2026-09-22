"""`scripts/relay_health.py` 的守卫测试。

这个脚本存在的理由是**两个真实故障**：
1. 远端服务随 ssh 会话被网络抖动静默带走（隧道端口还在听 ⇒ 连接被 reset）；
2. `CLOSE` 帧误销毁引擎（帧完全正确却报 `rc=-5`）。
两者都只能靠"探活 + 心跳新鲜度"在**开始推理之前**识别出来，所以这两条判据必须有测试钉住。
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("relay_health", ROOT / "scripts" / "relay_health.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tcp_check_accepts_listening_port() -> None:
    module = _load()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        result = module._check_tcp(f"127.0.0.1:{port}", timeout=1.0)
    finally:
        listener.close()
    assert result["ok"] is True
    assert result["reason"] == "listening"


def test_tcp_check_reports_unreachable_with_hint() -> None:
    """★ 案例 1 的守卫：端口不通时必须给出**可执行的排查方向**，而不是一个裸的连接错误。"""
    module = _load()
    result = module._check_tcp("127.0.0.1:50999", timeout=1.0)
    assert result["ok"] is False
    assert result["reason"] == "unreachable"
    assert "已退出" in result["hint"]        # 指向"服务消失"这一真因
    assert "隧道" in result["hint"]


def test_tcp_check_rejects_bad_endpoint_shape() -> None:
    module = _load()
    result = module._check_tcp("127.0.0.1", timeout=1.0)
    assert result["ok"] is False
    assert result["reason"] == "bad_endpoint"


def _fake_ssh(monkeypatch, module, payload: str, returncode: int = 0, stderr: str = ""):
    """把 `subprocess.run` 换成假的 ssh（测试不依赖真实设备）。"""

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=returncode, stdout=payload, stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", fake_run)


def test_ready_check_accepts_fresh_heartbeat(monkeypatch) -> None:
    """★ 心跳新鲜 ⇒ 健康。"""
    module = _load()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    _fake_ssh(monkeypatch, module, json.dumps(
        {"role": "tail", "alive_at": now, "pid": 4242}))
    result = module._check_ready("y700", "/tmp/tail.ready", timeout=1.0, stale_seconds=30.0)
    assert result["ok"] is True
    assert result["reason"] == "heartbeat_ok"
    assert result["pid"] == 4242
    assert result["age_s"] is not None and result["age_s"] < 30.0


def test_ready_check_flags_stale_heartbeat(monkeypatch) -> None:
    """★ 案例 2 的守卫：进程可能还在，但**心跳过期** ⇒ 该段已不再服务。"""
    module = _load()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=600)).isoformat(timespec="seconds").replace("+00:00", "Z")
    _fake_ssh(monkeypatch, module, json.dumps({"role": "middle", "alive_at": old, "pid": 7}))
    result = module._check_ready("surface", "/tmp/mid.ready", timeout=1.0, stale_seconds=30.0)
    assert result["ok"] is False
    assert result["reason"] == "heartbeat_stale"
    assert result["age_s"] and result["age_s"] > 30.0
    assert "重启" in result["hint"]


def test_ready_check_handles_missing_file(monkeypatch) -> None:
    module = _load()
    _fake_ssh(monkeypatch, module, "", returncode=1, stderr="No such file")
    result = module._check_ready("y700", "/tmp/none.ready", timeout=1.0, stale_seconds=30.0)
    assert result["ok"] is False
    assert result["reason"] == "no_ready_file"


def test_ready_check_rejects_remote_shell_metacharacters(monkeypatch) -> None:
    module = _load()
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout="", stderr="missing")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._check_ready("surface", "C:/tmp/a; touch /tmp/pwned `x`", timeout=1.0, stale_seconds=30.0)
    assert calls == []


def test_ready_check_rejects_naive_heartbeat(monkeypatch) -> None:
    module = _load()
    _fake_ssh(monkeypatch, module, json.dumps({"role": "tail", "alive_at": "2026-09-22T12:00:00", "pid": 1}))
    result = module._check_ready("surface", "/tmp/tail.ready", timeout=1.0, stale_seconds=30.0)
    assert result["ok"] is False
    assert result["reason"] == "heartbeat_invalid"
    assert result["age_s"] is None


def test_main_exit_code_and_json(monkeypatch, capsys) -> None:
    """退出码必须能进 CI：全健康 0，有异常 1；`--json` 可被驱动解析。"""
    module = _load()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    try:
        assert module.main(["--check", f"ok=tcp:127.0.0.1:{port}", "--json"]) == 0
        assert json.loads(capsys.readouterr().out.strip())["healthy"] is True
        assert module.main(["--check", "bad=tcp:127.0.0.1:50999", "--json"]) == 1
        assert json.loads(capsys.readouterr().out.strip())["healthy"] is False
    finally:
        listener.close()


def test_main_requires_at_least_one_check() -> None:
    module = _load()
    with pytest.raises(SystemExit):
        module.main([])


def test_output_avoids_non_ascii_arrows() -> None:
    """★ 回归守卫：输出里不得使用 GBK 无法编码的符号（`⇒` 曾把健康检查打成 traceback）。

    一个本该只报"健康 / 异常"的工具，反而比不检查更吵 —— 这类自伤必须钉住。
    """
    source = (ROOT / "scripts" / "relay_health.py").read_text(encoding="utf-8")
    # 注释与 docstring 里出现没问题，只有**输出内容**才必须安全；这里保守地只在 print 行上断言
    for line in source.splitlines():
        if "print(" in line:
            line.encode("gbk")  # 不可编码即抛 UnicodeEncodeError
