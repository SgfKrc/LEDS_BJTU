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


def test_tailscale_status_json_success_uses_single_fast_probe(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "_find_tailscale_exe", lambda: "tailscale.exe")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
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
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 2
    assert all(
        kwargs["creationflags"] == launcher_module._WINDOWS_NO_WINDOW
        for _, kwargs in calls
    )
    assert all(
        kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"
        for _, kwargs in calls
    )


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


def test_existing_qlh_instance_is_recognized(monkeypatch, launcher_module):
    import json
    import urllib.request

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "run_mode": "distributed",
                "node_id": "master",
            }).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert launcher_module._is_existing_qlh_instance(8000) is True


def test_startup_splash_is_safe_when_disabled(launcher_module):
    splash = launcher_module._StartupSplash(enabled=False).start()
    splash.update(150, "ready")
    assert splash._progress == 100
    assert splash._status == "ready"
    splash.close()


def test_windows_dialog_uses_splash_owner(monkeypatch, launcher_module):
    calls = []

    class User32:
        @staticmethod
        def MessageBoxW(owner, message, title, flags):
            calls.append((owner, message, title, flags))
            return 6

    import ctypes
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(user32=User32()), raising=False)
    result = launcher_module._show_windows_messagebox(
        "title", "message", owner_hwnd=1234,
    )
    assert result == 6
    assert calls[0][0] == 1234


def test_webview_runtime_import_failure_falls_back(monkeypatch, launcher_module):
    monkeypatch.setattr(launcher_module, "IS_WINDOWS", True)
    real_import = __import__

    def fail_webview(name, *args, **kwargs):
        if name == "webview":
            raise OSError("missing WebView runtime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_webview)
    assert launcher_module._has_webview() is False


def test_startup_splash_uses_application_icon(launcher_module):
    path = launcher_module._startup_icon_path()
    assert path.endswith("leds.ico")
    assert os.path.isfile(path)


def test_pytorch_tokenizer_runtime_check_loads_local_tokenizer(
    monkeypatch, launcher_module
):
    class FakeTokenizer:
        def encode(self, text):
            assert text
            return [1, 2, 3]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            assert model_path == "local-qwen"
            assert kwargs == {"trust_remote_code": True, "local_files_only": True}
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(sys.modules, "config", SimpleNamespace(MODEL_PATH="local-qwen"))

    name = launcher_module._verify_pytorch_tokenizer_runtime()
    assert name.endswith(".FakeTokenizer")


def test_qwen_runtime_dependency_is_declared_for_packaging():
    requirement_paths = [
        os.path.join(ROOT, "requirements.txt"),
        os.path.join(ROOT, "packaging", "requirements-cpu.txt"),
    ]
    spec_paths = [
        os.path.join(ROOT, "packaging", "qlh-cpu.spec"),
        os.path.join(ROOT, "packaging", "qlh-cuda.spec"),
    ]

    for path in requirement_paths:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "tiktoken" in content
        assert "httpx" in content
    for path in spec_paths:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "'tiktoken'" in content
        assert "'tiktoken._tiktoken'" in content
        assert "'httpx'" in content
        assert "'task_graph'" in content
        assert "'task_journal'" in content
        assert "'task_provider'" in content
        assert "'task_worker_protocol'" in content
        assert "'task_worker_adapter'" in content
        assert "'_sqlite3'" in content
