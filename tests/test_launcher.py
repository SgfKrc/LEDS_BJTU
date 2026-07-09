"""
单元测试 — 打包版 launcher 的 Tailscale 检测
===========================================
避免打包启动器在 Tailscale 已连接但 CLI status 瞬时失败时误弹未连接提示。
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAUNCHER_PATH = os.path.join(ROOT, "packaging", "launcher.py")


@pytest.fixture()
def launcher_module():
    spec = importlib.util.spec_from_file_location("qlh_packaging_launcher_test", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_result(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_tailscale_status_json_success_after_retry(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "_find_tailscale_exe", lambda: "tailscale.exe")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return _run_result(1, "", "backend not ready")
        return _run_result(
            0,
            '{"Self":{"TailscaleIPs":["100.90.76.108"],"HostName":"pc-master"}}',
            "",
        )

    monkeypatch.setattr(launcher_module._sp, "run", fake_run)
    monkeypatch.setattr(launcher_module.time, "sleep", lambda *_: None)

    status = launcher_module._check_tailscale_status()
    assert status["installed"] is True
    assert status["running"] is True
    assert status["logged_in"] is True
    assert status["tailscale_ip"] == "100.90.76.108"
    assert status["source"] == "status_json"


def test_tailscale_ip_command_fallback(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "_find_tailscale_exe", lambda: "tailscale.exe")

    def fake_run(cmd, **kwargs):
        if cmd[1:3] == ["status", "--json"]:
            return _run_result(1, "", "status unavailable")
        if cmd[1:3] == ["ip", "-4"]:
            return _run_result(0, "100.88.1.2\n", "")
        raise AssertionError(cmd)

    monkeypatch.setattr(launcher_module._sp, "run", fake_run)
    monkeypatch.setattr(launcher_module.time, "sleep", lambda *_: None)

    status = launcher_module._check_tailscale_status()
    assert status["tailscale_ip"] == "100.88.1.2"
    assert status["source"] == "tailscale_ip"
    assert status["logged_in"] is True


def test_tailscale_interface_fallback(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "_find_tailscale_exe", lambda: "tailscale.exe")
    monkeypatch.setattr(
        launcher_module._sp,
        "run",
        lambda *args, **kwargs: _run_result(1, "", "cli failed"),
    )
    monkeypatch.setattr(launcher_module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(launcher_module, "_detect_tailscale_ip_from_interfaces", lambda: "100.77.66.55")

    status = launcher_module._check_tailscale_status()
    assert status["tailscale_ip"] == "100.77.66.55"
    assert status["source"] == "interface"
    assert status["running"] is True


def test_tailscale_not_installed(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "_find_tailscale_exe", lambda: None)
    status = launcher_module._check_tailscale_status()
    assert status["installed"] is False
    assert status["tailscale_ip"] is None
    assert status["source"] == "none"
