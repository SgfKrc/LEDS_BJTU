"""tests/test_llama_keep_head_decode_error.py — ① 的回归：`llama_decode` 原始 rc 的透出。

背景（2026-09-24）：长 decode 在 `n_ctx / n_seq_max` 用尽时报 `runner_failed`，而 shim 把
`llama_decode` 的非零 rc 一律折成 `-2`、丢了 llama.cpp 的具体错误码（实际是 `rc=1` =
"failed to find a memory slot for batch"）⇒ 无法定位。新增 `qlh_kh_last_error` 透出原始 rc，
**返回码契约不变**；旧 shim 没这个符号，调用方必须按 optional 处理、绝不因此新增失败点。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llama_keep_head import KeepHeadUpstream, ctx_per_seq_for  # noqa: E402


def test_ctx_per_seq_matches_measured_behaviour() -> None:
    """★ 本坑的核心不变式：`n_ctx` 按 `n_seq_max` **均分**。

    三个断言取自实测（不是推算）：`2048/8 = 256`（decode 到第 256 步报 rc=1）、
    `4096/8 = 512`（历史观测失败在 ~481，481+32 prefill ≈ 513）、`8192/8 = 1024`（~1000）。
    """
    assert ctx_per_seq_for(2048, 8) == 256
    assert ctx_per_seq_for(4096, 8) == 512
    assert ctx_per_seq_for(8192, 8) == 1024
    assert ctx_per_seq_for(2048, 1) == 2048
    assert ctx_per_seq_for(552, 0) == 552    # n_seq_max <= 0 视为 1
    assert ctx_per_seq_for(0, 4) == 0


class _LibWithoutLastError:
    """模拟**旧版 shim**：没有 `qlh_kh_last_error` 符号。"""


class _LibWithLastError:
    """模拟新版 shim：透出原始 rc。"""

    def __init__(self, value: int) -> None:
        self._value = value
        self.calls: list[object] = []

    def qlh_kh_last_error(self, handle: object) -> int:
        self.calls.append(handle)
        return self._value


def _bare_upstream(lib: object, handle: object = object()) -> KeepHeadUpstream:
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)   # 不加载 shim / 不起 worker
    upstream._lib = lib
    upstream._handle = handle
    return upstream


def test_missing_symbol_is_silently_zero() -> None:
    """旧 shim ⇒ 返回 0、后缀为空（向后兼容，不新增失败点）。"""
    upstream = _bare_upstream(_LibWithoutLastError())
    assert upstream.last_decode_error() == 0
    assert upstream._decode_error_suffix() == ""


def test_symbol_value_is_surfaced_in_suffix() -> None:
    """新 shim ⇒ 原始 rc 必须原样出现在诊断后缀里。"""
    handle = object()
    lib = _LibWithLastError(1)
    upstream = _bare_upstream(lib, handle)

    assert upstream.last_decode_error() == 1
    assert "rc=1" in upstream._decode_error_suffix()
    assert lib.calls == [handle, handle]   # 后缀会再读一次（shim 不清零，重复读无害）


def test_worker_path_without_local_handle_is_zero() -> None:
    """worker 隔离路径下 `_lib` / `_handle` 为 None ⇒ 不得抛异常，只返回 0。"""
    upstream = _bare_upstream(None, None)
    assert upstream.last_decode_error() == 0
    assert upstream._decode_error_suffix() == ""


def test_broken_symbol_call_does_not_raise() -> None:
    """符号存在但调用异常时，诊断路径必须吞掉（诊断永远不能变成新的失败点）。"""

    class _Broken:
        def qlh_kh_last_error(self, handle: object) -> int:
            raise OSError("boom")

    upstream = _bare_upstream(_Broken())
    assert upstream.last_decode_error() == 0
