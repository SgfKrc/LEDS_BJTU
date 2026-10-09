"""tests/test_cut_layers_load_selfcheck.py — ⑨ 加载自证（`#67` 最后一条缺口）

背景（`docs/已知问题记录.md` 的 `#67`）：
③（层类型逐位）/ ⑩（按层数组 KV）/ ④（必需张量集合）三条 fail-closed 都只发生在
**生成期**——它们能挡住"我们已知会坏"的源，但**没有任何一处验证"产出的工件真能被
llama.cpp 加载"**。真机加载失败（`missing tensor 'blk.x.<...>'`）此前只能在设备上撞见。

本文件锁住新能力：**用钉死的 llama.cpp 二进制真的加载一次工件**。
判据（取自 llama.cpp 源码的抛出点，见 `llama-model-loader.cpp:560/1098`、`llama.cpp:372`）：
* `exit == 0`，且
* 输出里**不含**失败标志（`missing tensor` / `failed to load model` / `error loading model`
  / `failed to open GGUF file` / `missing tensor info mapping`）。

⚠️ 依赖 `build/cross-framework-layer-poc/llama.cpp/build-cpu/bin/llama-debug.exe`
（该目录 gitignored）⇒ 不可用时**跳过**，不让 CI 因缺本地构建而红。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cut_layers_pipeline import (  # noqa: E402
    LLAMA_LOAD_FAILURE_MARKERS,
    verify_artifact_loads,
)

LLAMA_BIN = (REPO_ROOT / "build" / "cross-framework-layer-poc" / "llama.cpp"
             / "build-cpu" / "bin" / "llama-debug.exe")
#: 真正能加载的 middle 工件（[8,24)，带 token_embd + output_norm）
MIDDLE_ARTIFACT = (REPO_ROOT / "build" / "cross-framework-layer-poc" / "out"
                   / "qwen25-05b-f16-mid8-24.gguf")

_needs_llama = pytest.mark.skipif(
    not LLAMA_BIN.is_file(), reason="本地无 llama.cpp 构建（build/ 下，gitignored）"
)


def test_failure_markers_cover_the_real_throwing_points():
    """失败标志必须覆盖源码里真实会抛的那几条，否则判据会漏判成"加载成功"。"""
    text = " ".join(LLAMA_LOAD_FAILURE_MARKERS).lower()
    for needle in ("missing tensor", "failed to load model", "error loading model",
                   "failed to open gguf file"):
        assert needle in text, f"判据缺少 {needle!r}（源码会抛这条）"


def test_missing_binary_is_reported_as_skipped_not_failed(tmp_path):
    """二进制不可用 ⇒ `ok=None`（跳过语义），不得误判为"加载成功"或"加载失败"。"""
    ok, detail = verify_artifact_loads(tmp_path / "whatever.gguf", tmp_path / "no-such-llama.exe")
    assert ok is None, f"应是跳过语义，实得 ok={ok!r} detail={detail!r}"


def test_nonexistent_artifact_fails_closed(tmp_path):
    """工件不存在 ⇒ 明确失败（`ok=False`），并带上可读原因。"""
    ok, detail = verify_artifact_loads(tmp_path / "nope.gguf", LLAMA_BIN)
    if ok is None:  # 二进制缺失则在无构建环境里跳过
        pytest.skip("无 llama.cpp 构建")
    assert ok is False, (ok, detail)
    assert detail, "失败时必须给出可读原因"


@_needs_llama
def test_real_middle_artifact_passes_load_selfcheck():
    """★ 该绿必须绿：真实 middle 工件能被 llama.cpp 加载（跑一次前向）。"""
    if not MIDDLE_ARTIFACT.is_file():
        pytest.skip(f"缺工件 {MIDDLE_ARTIFACT.name}（可先用 cut_layers_pipeline.py 生成）")
    ok, detail = verify_artifact_loads(MIDDLE_ARTIFACT, LLAMA_BIN, timeout_s=180)
    assert ok is True, f"应能加载：{detail}"
    assert "llama" in detail.lower(), detail


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
