"""打包瘦身：qlh_launcher 瘦身包识别、外部 runtime 引导命令与 --runtime-check 入口测试。

monkeypatch 隔离全局状态与 runtime_guard，不触碰真实用户 runtime / 不执行 pip。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "packaging"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runtime_guard as rg  # noqa: E402
import qlh_launcher as ql  # noqa: E402


def _slim_tree(tmp_path) -> Path:
    src = tmp_path / "_internal" / "src"
    src.mkdir(parents=True)
    (src / "api_server.py").write_text("# slim", encoding="utf-8")
    pkg = tmp_path / "packaging"
    pkg.mkdir(exist_ok=True)
    (pkg / "requirements-runtime-cpu.txt").write_text("", encoding="utf-8")
    (pkg / "requirements-runtime-cuda.txt").write_text("", encoding="utf-8")
    return tmp_path


def test_slim_src_root_recognized(tmp_path):
    tree = _slim_tree(tmp_path)
    assert ql._slim_src_root(tree) == tree / "_internal"


def test_slim_src_root_not_when_packaged_exe(tmp_path):
    tree = _slim_tree(tmp_path)
    (tree / "QLH-Edge-Inference.exe").write_text("x")
    assert ql._slim_src_root(tree) is None


def test_slim_src_root_none_without_src(tmp_path):
    assert ql._slim_src_root(tmp_path) is None


def test_runtime_app_command_returns_uvicorn(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(rg, "ensure_runtime", lambda ctx: {"state": "installed"})
    cmd = ql.runtime_app_command(tree, engine="cpu")
    assert cmd is not None
    assert "-m" in cmd and "uvicorn" in cmd and "src.api_server:app" in cmd
    assert "--host" in cmd and "0.0.0.0" in cmd and "--port" in cmd and "8000" in cmd
    # engine=cuda 时 requirements 指向 cuda
    seen = []
    def spy(ctx):
        seen.append(ctx.requirements.name)
        return {"state": "ok"}
    monkeypatch.setattr(rg, "ensure_runtime", spy)
    ql.runtime_app_command(tree, engine="cuda")
    assert seen[-1] == "requirements-runtime-cuda.txt"


def test_runtime_app_command_none_on_failure(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(rg, "ensure_runtime", lambda ctx: {"state": "failed", "error": "install failed"})
    assert ql.runtime_app_command(tree, engine="cpu") is None


def test_app_command_slim_branch(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(ql, "installed_app_root", lambda *a, **k: tree)
    monkeypatch.setattr(ql, "auto_reassociate_user_data", lambda root: None)
    monkeypatch.setattr(rg, "ensure_runtime", lambda ctx: {"state": "ok"})
    cmd = ql.app_command("serve")
    assert "uvicorn" in cmd and "src.api_server:app" in cmd


def test_app_command_slim_raises_on_runtime_missing(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(ql, "installed_app_root", lambda *a, **k: tree)
    monkeypatch.setattr(ql, "auto_reassociate_user_data", lambda root: None)
    monkeypatch.setattr(rg, "ensure_runtime", lambda ctx: {"state": "failed"})
    with pytest.raises(FileNotFoundError):
        ql.app_command("serve")


def test_launch_app_uses_internal_cwd_for_slim(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(ql, "installed_app_root", lambda *a, **k: tree)
    monkeypatch.setattr(rg, "ensure_runtime", lambda ctx: {"state": "ok"})
    captured = {}
    monkeypatch.setattr(ql.subprocess, "Popen",
                        lambda command, cwd=None: captured.update(command=command, cwd=cwd))
    ql.launch_app("serve")
    assert str(captured["cwd"]).endswith(("_internal", "_internal\\", "_internal/"))
    assert "8000" in captured["command"] and "0.0.0.0" in captured["command"]


def test_installed_app_root_recognizes_slim(monkeypatch, tmp_path):
    tree = _slim_tree(tmp_path)
    monkeypatch.setattr(ql, "_candidate_app_roots", lambda *a, **k: [tree])
    assert ql.installed_app_root() == tree
