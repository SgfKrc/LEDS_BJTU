"""打包瘦身 runtime_guard 测试：外部运行时依赖守卫（不真正执行 pip / 联网）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "packaging"))

import runtime_guard as guard  # noqa: E402


def _ctx(tmp_path, *, engine="cpu", proxy=""):
    return guard.RuntimeContext(
        root=tmp_path, engine=engine,
        requirements=tmp_path / "requirements-runtime-cpu.txt",
        proxy=proxy,
        runtime_dir=tmp_path / "runtime",
    )


def test_runtime_dir_default_and_override(monkeypatch, tmp_path):
    monkeypatch.delenv(guard.RUNTIME_ENV_VAR, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert guard.runtime_dir() == tmp_path / "appdata" / "QLH-Edge-Inference" / "runtime"
    monkeypatch.setenv(guard.RUNTIME_ENV_VAR, str(tmp_path / "custom"))
    assert guard.runtime_dir() == tmp_path / "custom"


def test_venv_python_platform_path():
    # 本机 Windows（os.name=nt）：返回 Scripts/python.exe；posix 分支由代码分支保证
    path = guard.venv_python(Path("v"))
    assert path.parts[-2:] in (("Scripts", "python.exe"), ("bin", "python"))
    assert path.name == "python.exe" or path.name == "python"


def test_probe_missing_all_when_python_absent(tmp_path):
    missing = guard.probe_missing(tmp_path / "no-python", ("torch", "fastapi"))
    assert set(missing) == {"torch", "fastapi"}


def test_probe_missing_parses(monkeypatch, tmp_path):
    fake = tmp_path / "py.exe"
    fake.write_text("x")
    result = SimpleNamespace(returncode=1, stdout="torch\ntransformers\n")
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: result)
    assert guard.probe_missing(fake, ("torch", "transformers", "fastapi")) == [
        "torch", "transformers",
    ]
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    assert guard.probe_missing(fake, ("torch",)) == []


def test_build_pip_commands_cpu_torch_separate_index(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("")
    ctx = _ctx(tmp_path, engine="cpu", proxy="http://127.0.0.1:7897")
    python = guard.venv_python(tmp_path / "runtime")
    commands = guard.build_pip_commands(python, req, ctx)
    # CPU：torch 单独走 CPU index；其余依赖走默认 PyPI（避免 --index-url 替换主源）
    assert len(commands) == 2
    torch_cmd = commands[0]
    assert guard.PYTORCH_CPU_INDEX in torch_cmd and "torch" in torch_cmd
    assert "--proxy" in torch_cmd and "7897" in " ".join(torch_cmd)
    rest_cmd = commands[1]
    assert "-r" in rest_cmd and str(req) in rest_cmd
    assert "--index-url" not in rest_cmd          # 关键：非 torch 依赖不绑 PyTorch 索引
    assert "--proxy" in rest_cmd


def test_build_pip_commands_cuda_single_requirements(tmp_path):
    req = tmp_path / "req.txt"
    req.write_text("")
    ctx = _ctx(tmp_path, engine="cuda")
    commands = guard.build_pip_commands(guard.venv_python(tmp_path / "runtime"), req, ctx)
    assert len(commands) == 1
    assert "-r" in commands[0] and "--index-url" not in commands[0]


def test_build_pip_commands_cpu_fallback_does_not_pin_rest_to_pytorch(tmp_path):
    ctx = _ctx(tmp_path, engine="cpu")
    commands = guard.build_pip_commands(
        guard.venv_python(tmp_path / "runtime"), tmp_path / "missing.txt", ctx,
    )
    assert len(commands) == 2
    assert "torch" in commands[0] and "--index-url" in commands[0]
    assert "transformers" in commands[1] and "--index-url" not in commands[1]


def _runner_ok(command):
    return 0


def test_ensure_runtime_ok_when_present(monkeypatch, tmp_path):
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("x")
    monkeypatch.setattr(guard, "probe_missing", lambda *a, **k: [])
    report = guard.ensure_runtime(_ctx(tmp_path), runner=_runner_ok)
    assert report["state"] == "ok"
    assert report["missing_before"] == []


def test_ensure_runtime_installs_when_missing(monkeypatch, tmp_path):
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("x")
    calls = []
    monkeypatch.setattr(guard, "probe_missing", lambda *a, **k: ["torch"] if not calls else [])
    monkeypatch.setattr(guard, "venv", SimpleNamespace(
        EnvBuilder=lambda **k: SimpleNamespace(create=lambda p: None),
    ))
    def runner(command):
        calls.append(list(command))
        return 0
    report = guard.ensure_runtime(_ctx(tmp_path), runner=runner)
    assert report["state"] == "installed"
    assert report["missing_before"] == ["torch"]
    assert calls and "pip" in calls[0] and "install" in calls[0]


def test_ensure_runtime_failed_when_install_unsatisfied(monkeypatch, tmp_path):
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("x")
    monkeypatch.setattr(guard, "probe_missing", lambda *a, **k: ["torch"])
    monkeypatch.setattr(guard, "venv", SimpleNamespace(
        EnvBuilder=lambda **k: SimpleNamespace(create=lambda p: None),
    ))
    def runner(command):
        return 1                      # 假装装失败：probe 仍缺
    report = guard.ensure_runtime(_ctx(tmp_path), runner=runner)
    assert report["state"] == "failed"
    assert report["missing_after"] == ["torch"]


def test_ensure_runtime_create_venv_failure(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise OSError("no permission")
    monkeypatch.setattr(guard, "venv", SimpleNamespace(EnvBuilder=lambda **k: SimpleNamespace(create=boom)))
    report = guard.ensure_runtime(_ctx(tmp_path), runner=_runner_ok)
    assert report["state"] == "failed"
    assert "create venv failed" in report["error"]
