"""Tests for the dedicated test-environment guard and bootstrapper."""

from pathlib import Path
import sys

from scripts import run_test_channels
from scripts import setup_test_env


def test_test_channel_guard_rejects_system_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "prefix", "C:/Python312")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is False
    assert "setup_test_env.py" in capsys.readouterr().err


def test_test_channel_guard_accepts_virtual_environment(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "G:/qlh/.venv-test")
    monkeypatch.setattr(sys, "base_prefix", "C:/Python312")

    assert run_test_channels._check_python_environment(
        allow_system_python=False,
    ) is True


def test_install_command_keeps_proxy_and_requirements_in_venv(monkeypatch):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(proxy="http://127.0.0.1:7897")

    assert command[:4] == [
        str(test_python), "-m", "pip", "install",
    ]
    assert command[4:6] == ["--proxy", "http://127.0.0.1:7897"]
    assert command[-2:] == ["-r", str(setup_test_env.REQUIREMENTS)]


def test_wheelhouse_disables_network_proxy(monkeypatch, tmp_path):
    test_python = Path("G:/qlh/.venv-test/Scripts/python.exe")
    monkeypatch.setattr(setup_test_env, "_python_path", lambda: test_python)

    command = setup_test_env._install_command(
        proxy="http://127.0.0.1:7897",
        wheelhouse=tmp_path,
    )

    assert "--no-index" in command
    assert f"--find-links={tmp_path}" in command
    assert "--proxy" not in command
