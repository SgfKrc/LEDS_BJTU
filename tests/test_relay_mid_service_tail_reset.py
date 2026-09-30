"""末段（pip `llama_cpp` 路径）的会话级 `reset()` 必须真正清 KV。

为什么需要（2026-09-30 三机产品路径验收发现的真缺陷）：
`src/llama_engine.forward_layers_from_hidden` 是"**直接在 KV 里占
`[n_past, n_past+n_tokens)`**"，其方法说明写明"若要在**同一位置**重跑，先
`self._model._ctx.kv_cache_clear()`……同 `n_past` 重跑会得到 `llama_decode rc=-1`"。

`scripts/relay_mid_service.TailRunner`（tail 角色、pip 绑定）原先把 `reset()` 写成只
重置 `self._pos` ⇒ **第二个**产品路径请求从位置 0 重跑时 `llama_decode rc=-1` ⇒
worker 侧 `relay 段委托未成功: … code=runner_failed`（远端日志
`Relay tail runner (seq) failed` / `[session] frames=0 tokens=0 closed_cleanly=False
error=runner_failed`）。

`HeadRunner` / `KeepHeadMiddleRunner` / `ShimTailRunner` 都走 `self._upstream.reset()`
（真清 KV），只有 pip `TailRunner` 漏了 —— 这些用例锁死那个空白。

不需要模型：用一个假 engine 记录 `kv_cache_clear()` 是否被调用。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MOD = None


def _load_module():
    """按路径加载 `scripts/relay_mid_service.py`（它顶部只依赖轻量模块）。"""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "qlh_relay_mid_service", ROOT / "scripts" / "relay_mid_service.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD


class _FakeCtx:
    def __init__(self) -> None:
        self.cleared = 0

    def kv_cache_clear(self) -> None:
        self.cleared += 1


class _FakeLlama:
    def __init__(self) -> None:
        self._ctx = _FakeCtx()


class _FakeEngine:
    def __init__(self) -> None:
        self._model = _FakeLlama()


def _make_runner(engine: object):
    """构造 TailRunner 替身（绕过 __init__，不需要真模型）。"""
    runner = object.__new__(_load_module().TailRunner)
    runner._pos = 12345          # 故意留一个非 0 的旧位置
    runner._engine = engine
    return runner


def test_tail_runner_reset_clears_native_kv() -> None:
    """reset() 必须既归零位置、又调用原生 `kv_cache_clear()`。

    该红必须红：若把 `reset()` 退回成只有 `self._pos = 0`（本次修复前的形态），
    本用例在 `cleared == 1` 上失败 —— 与线上"第二个请求 rc=-1"同型。
    """
    engine = _FakeEngine()
    runner = _make_runner(engine)

    runner.reset()

    assert runner._pos == 0
    assert engine._model._ctx.cleared == 1


def test_tail_runner_reset_tolerates_missing_native_ctx() -> None:
    """引擎未加载 / 结构不匹配时不得抛（reset 在连接收尾路径上一定被调用）。"""
    runner = _make_runner(object())
    runner.reset()
    assert runner._pos == 0


def test_tail_runner_close_also_clears_kv() -> None:
    """close() 等价于 reset()：会话结束时也要清，否则下个会话从位置 0 重跑会 rc=-1。"""
    engine = _FakeEngine()
    runner = _make_runner(engine)
    runner.close()
    assert runner._pos == 0
    assert engine._model._ctx.cleared == 1
