"""T3：scripts/model_tools 回归测试（测试修复票排期 P1）。

覆盖 2026-08-16/17 的裸修复：
  - gguf_convert._ensure_converter_patch：自动应用/幂等/apply 失败 fail-closed
    （2fe47a6、8129806：Qwen 老架构 epsilon 键 + 补丁持久化）
  - import_model ModelScope 子进程 GBK 编码（3e26a35：errors=replace）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_tools import gguf_convert as gc  # noqa: E402
from model_tools import import_model as im  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PATCH = (REPO_ROOT / "packaging" / "patches"
              / "llama-cpp-converter-qwen-eps.patch")


def _fake_subprocess(calls: list, returncode: int = 0):
    rc = returncode

    class R:
        returncode = rc
        stdout = ""
        stderr = "apply failed"

    def run(cmd, **kw):
        calls.append(cmd)
        return R()

    return run


def _fake_submodule(tmp_path) -> Path:
    """构造子模块结构（convert_hf_to_gguf.py + conversion/base.py）。"""
    llama = tmp_path / "llama.cpp"
    conv = llama / "conversion"
    conv.mkdir(parents=True)
    (llama / "convert_hf_to_gguf.py").write_text("x", encoding="utf-8")
    return llama


# ---- _ensure_converter_patch ----

def test_patch_idempotent_when_already_applied(tmp_path, monkeypatch):
    """已打补丁（marker 在）-> 跳过，不执行 git apply。"""
    llama = _fake_submodule(tmp_path)
    (llama / "conversion" / "base.py").write_text(
        f"# {gc._PATCH_MARKER}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(gc.subprocess, "run", _fake_subprocess(calls))
    gc._ensure_converter_patch(llama / "convert_hf_to_gguf.py")
    assert calls == [], "已打补丁时不应执行 git apply"


def test_patch_auto_applies_when_missing(tmp_path, monkeypatch):
    """未打补丁 -> 自动 git apply 真实补丁文件（packaging/patches/）。"""
    assert REAL_PATCH.is_file(), "补丁文件必须入库"
    llama = _fake_submodule(tmp_path)
    (llama / "conversion" / "base.py").write_text("unpatched", encoding="utf-8")
    calls = []
    monkeypatch.setattr(gc.subprocess, "run", _fake_subprocess(calls))
    gc._ensure_converter_patch(llama / "convert_hf_to_gguf.py")
    assert calls and "apply" in calls[0], "未打补丁时应自动 git apply"
    assert str(REAL_PATCH) in calls[0], "应 apply 入库的补丁文件"


def test_patch_apply_failure_fails_closed(tmp_path, monkeypatch):
    """git apply 失败 -> fail-closed（GGUFConvertError），不静默产出坏 GGUF。"""
    llama = _fake_submodule(tmp_path)
    (llama / "conversion" / "base.py").write_text("unpatched", encoding="utf-8")
    calls = []
    monkeypatch.setattr(gc.subprocess, "run", _fake_subprocess(calls, returncode=1))
    with pytest.raises(gc.GGUFConvertError, match="failed to apply"):
        gc._ensure_converter_patch(llama / "convert_hf_to_gguf.py")


def test_patch_skips_non_submodule_converter(tmp_path):
    """非子模块 converter（无 conversion/base.py）-> 跳过不报错。"""
    lone = tmp_path / "standalone-convert.py"
    lone.write_text("x", encoding="utf-8")
    gc._ensure_converter_patch(lone)  # 不应抛


# ---- import_model 子进程 GBK ----

def test_import_subprocess_uses_utf8_errors_replace(tmp_path, monkeypatch):
    """3e26a35：ModelScope 下载子进程必须 utf-8 + errors=replace（GBK 崩溃修复）。"""
    captured = {}

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured.update(kw)
        return R()

    monkeypatch.setattr(im.subprocess, "run", fake_run)
    # 走真实 download_model 的 ModelScope 分支（subprocess 已 mock）
    try:
        im.download_model("mock/repo", tmp_path / "stage", use_modelscope=True)
    except Exception:
        pass  # 不依赖真实 modelscope 安装
    assert captured.get("encoding") == "utf-8", "子进程必须显式 utf-8"
    assert captured.get("errors") == "replace", "GBK 字节必须以 replace 容错（修复前崩溃）"
