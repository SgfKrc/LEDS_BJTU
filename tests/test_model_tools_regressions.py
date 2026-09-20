"""T3：scripts/model_tools 回归测试（测试修复票排期 P1）。

覆盖 2026-08-16/17 的裸修复：
  - import_model ModelScope 子进程 GBK 编码（3e26a35：errors=replace）

★ 2026-09-20：删除了 4 项关于 `gguf_convert._ensure_converter_patch` 的用例
（`test_patch_idempotent_when_already_applied` / `_auto_applies_when_missing` /
`_apply_failure_fails_closed` / `_skips_non_submodule_converter`）以及它们的
辅助函数（`_fake_subprocess` / `_fake_submodule` / `REAL_PATCH`）。

**为什么删**：那套逻辑服务的是 `llama-cpp-converter-qwen-eps.patch` —— 一个只为
legacy `QWenLMHeadModel`（即 Qwen-1.8B）补 `layer_norm_epsilon` 候选键的转换器补丁。
Qwen-1.8B 已于 2026-09-19 **退役**（`src/config.py`：`ACTIVE_MODEL_ID = "qwen3-0.6b"`，
默认模型改按设备画像选择），该补丁**已无消费方** ⇒ 连同补丁文件、lock 条目与
`gguf_convert` 中的兜底逻辑一并移除（用户裁定 dec-eb2367bb41d9319f）。

⚠️ 若将来又要转换 legacy Qwen 权重，需要重新打这个补丁（历史实现在
`scripts/model_tools/patches/` 的 git 历史里，commit 2026-08-16 前后）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from model_tools import import_model as im  # noqa: E402


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
