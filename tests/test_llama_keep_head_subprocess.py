"""在独立进程中运行 keep-head 原生用例，避免 Windows DLL ABI 串库。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_keep_head_native_suite_is_abi_isolated():
    env = os.environ.copy()
    env["QLH_KEEP_HEAD_NATIVE_WORKER"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n0", "tests/test_llama_keep_head.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        "keep-head 原生子进程失败；不能把 ABI 隔离问题降级为跳过:\n"
        f"{output[-8000:]}"
    )


def test_keep_head_switches_to_worker_after_pip_llama_import():
    shim = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
    model = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-head12.gguf"
    if not shim.is_file() or not model.is_file():
        pytest.skip("keep-head native artifacts are unavailable")
    code = """
import numpy as np
import llama_cpp
from llama_keep_head import KeepHeadUpstream
up = KeepHeadUpstream(SHIM, MODEL, n_ctx=512, n_threads=4,
                      extra_dll_dirs=[r'C:\\msys64\\ucrt64\\bin'])
assert up._worker is not None
hidden = up.forward_tokens_to_hidden([100, 200])
assert hidden.shape == (2, up.n_embd)
assert np.isfinite(hidden).all()
up.close()
assert up._worker is None
""".replace("SHIM", repr(str(shim))).replace("MODEL", repr(str(model)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output[-8000:]


def test_worker_path_supports_last_segment_hidden_to_token():
    """★ A14（2026-09-23）：worker 隔离路径下的**末段能力**必须可用。

    回归背景：`forward_hidden_to_token` 的「可选符号检查」原先**没排除 worker 分支** —— worker 模式
    下符号在子进程里（`self._lib is None`），于是必然抛「缺 `qlh_kh_forward_embd_token`」；
    表现为 tail 段 `runner_failed`（本轮接线时才被真实跑到）。这里刻意在**已 import llama_cpp**
    的进程里构造 worker 隔离，再跑一次末段前向。
    """
    shim = ROOT / "build" / "keephead" / "build-cpu" / "bin" / "qlh_keep_head.dll"
    # 末段前向要有 output / output_norm ⇒ 必须用 tail 工件（head 工件没有）
    model = ROOT / "build" / "cross-framework-layer-poc" / "out" / "qwen25-05b-f16-tail8.gguf"
    if not shim.is_file() or not model.is_file():
        pytest.skip("keep-head native artifacts are unavailable")
    code = """
import numpy as np
import llama_cpp                      # ★ 触发 worker 隔离（正是回归场景）
from llama_keep_head import KeepHeadUpstream
up = KeepHeadUpstream(SHIM, MODEL, n_ctx=256, n_threads=4,
                      extra_dll_dirs=[r'C:\\msys64\\ucrt64\\bin'])
assert up._worker is not None, "预期走 worker 隔离路径"
hidden = np.zeros((1, up.n_embd), dtype=np.float32)
token = up.forward_hidden_to_token(hidden)   # ★ 旧实现此处必抛「缺符号」
assert isinstance(token, int) and token >= 0, f"非法 token: {token}"
up.close()
""".replace("SHIM", repr(str(shim))).replace("MODEL", repr(str(model)))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, output[-8000:]
