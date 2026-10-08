"""`#53` 回归：keep-head shim 的定位必须是**两侧共用**的单一事实源。

判据（解析顺序）：显式入参 > 环境变量 `QLH_KEEP_HEAD_SHIM` > 仓库默认构建产物；
都没有时返回空串（由调用方给具名错误）。此前 worker 侧只读环境变量，
会出现"同一台机器 master 能用、worker 报缺环境变量"。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import keep_head_shim  # noqa: E402


def _clear_env(monkeypatch):
    monkeypatch.delenv(keep_head_shim.ENV_VAR, raising=False)


def test_explicit_argument_wins(monkeypatch):
    _clear_env(monkeypatch)
    assert keep_head_shim.resolve_keep_head_shim("C:/explicit/shim.dll") == "C:/explicit/shim.dll"


def test_env_var_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv(keep_head_shim.ENV_VAR, " C:/env/shim.dll ")
    assert keep_head_shim.resolve_keep_head_shim() == "C:/env/shim.dll"
    # 显式入参仍然优先
    assert keep_head_shim.resolve_keep_head_shim("C:/explicit.dll") == "C:/explicit.dll"


def test_repo_default_used_when_env_absent(monkeypatch, tmp_path):
    """没有环境变量时，落到**仓库默认构建产物**（这正是 `#53` 要求对齐的那一档）。"""
    _clear_env(monkeypatch)
    fake = tmp_path / "qlh_keep_head.dll"
    fake.write_bytes(b"stub")
    monkeypatch.setattr(keep_head_shim, "default_shim_path", lambda: fake)
    assert keep_head_shim.resolve_keep_head_shim() == str(fake)


def test_returns_empty_when_nothing_found(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setattr(keep_head_shim, "default_shim_path", lambda: tmp_path / "missing.dll")
    assert keep_head_shim.resolve_keep_head_shim() == ""


def test_missing_hint_names_var_and_default_path(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setattr(keep_head_shim, "default_shim_path", lambda: tmp_path / "missing.dll")
    hint = keep_head_shim.missing_shim_hint()
    # 提示必须**具名**：带环境变量名与已尝试的默认路径（此前只报"需要两个环境变量"）
    assert keep_head_shim.ENV_VAR in hint
    assert "missing.dll" in hint
