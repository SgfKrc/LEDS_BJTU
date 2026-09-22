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
