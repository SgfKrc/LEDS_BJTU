"""`#54` 回归：`QLH_PREFER_PYTORCH` 引擎让路的判据。

背景：qwen3.5 的 keep-head 上游挂点在 `output_norm` 之后，llama.cpp 上游被
fail-closed 拒绝（`#35`）；而默认选择里 GGUF 分支的优先级**在 PyTorch 之前**
⇒ 若不让路，本机会落到 llama.cpp 而拿不到该模型。

判据（三项同时满足才让路）：配置位为真 ∧ 画像模型有 safetensors 路径 ∧ 该路径**存在**。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: E402


def test_prefers_pytorch_only_when_all_three_hold(tmp_path):
    real_dir = tmp_path / "qwen3-5-2b"
    real_dir.mkdir()
    assert api_server.should_prefer_pytorch(True, str(real_dir)) is True


def test_flag_off_never_prefers(tmp_path):
    real_dir = tmp_path / "m"
    real_dir.mkdir()
    assert api_server.should_prefer_pytorch(False, str(real_dir)) is False


def test_missing_or_empty_path_never_prefers():
    assert api_server.should_prefer_pytorch(True, "") is False
    assert api_server.should_prefer_pytorch(True, None) is False


def test_nonexistent_path_never_prefers(tmp_path):
    # 路径非空但**不存在** ⇒ 不让路（避免把本机推进一个不存在的模型目录）
    assert api_server.should_prefer_pytorch(True, str(tmp_path / "nope")) is False


def test_non_directory_path_never_prefers(tmp_path):
    f = tmp_path / "file.gguf"
    f.write_bytes(b"x")
    # 是文件不是目录 ⇒ 不让路
    assert api_server.should_prefer_pytorch(True, str(f)) is False
