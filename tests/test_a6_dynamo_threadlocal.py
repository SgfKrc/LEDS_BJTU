"""★ A6：`torch._dynamo.config` 的 thread-local 语义与主仓的线程内补设。

背景：`torch.utils._config_module` 明确 "User overrides are thread-local"（torch ≥2.12；本仓
2.13.0 实测「主线程设 64 ⇒ 新线程读回默认 8」）。主仓 `_apply_compile()` 只在**加载线程**设置
`recompile_limit`/`cache_size_limit`，而推理（经 starlette `run_in_threadpool`）发生在 **worker
线程** ⇒ 设置读不到、退回默认 8。

对策：`_ensure_compile_limits_in_current_thread()` 在每个进入模型访问的线程里**幂等**补设。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch._dynamo as dynamo  # noqa: E402

import model_module  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_and_reset_tls(monkeypatch):
    """保存/恢复 dynamo config，并清掉本线程的幂等标记（避免测试间互相影响）。"""
    monkeypatch.setattr(model_module, "USE_COMPILE", True)
    model_module._compile_limit_tls.__dict__.clear()
    saved = (int(dynamo.config.recompile_limit),
             int(getattr(dynamo.config, "cache_size_limit", 0) or 0))
    yield
    dynamo.config.recompile_limit = saved[0]
    dynamo.config.cache_size_limit = saved[1]
    model_module._compile_limit_tls.__dict__.clear()


def _cfg_limit() -> int:
    return int(model_module.COMPILE_RECOMPILE_LIMIT)


def test_module_has_the_helper():
    assert callable(model_module._ensure_compile_limits_in_current_thread)


def test_sets_limit_in_current_thread():
    """在当前线程调用 ⇒ `recompile_limit` 被补设到配置值。"""
    dynamo.config.recompile_limit = 8
    dynamo.config.cache_size_limit = 8
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == _cfg_limit()
    assert dynamo.config.cache_size_limit == _cfg_limit()


def test_new_thread_gets_limit_after_helper():
    """**关键判据**：新线程里裸读拿不到（thread-local），调用 helper 后能拿到配置值。"""
    dynamo.config.recompile_limit = 8
    dynamo.config.cache_size_limit = 8
    box: dict = {}

    def _worker():
        box["naive"] = int(dynamo.config.recompile_limit)   # 裸读：应为默认 8
        model_module._ensure_compile_limits_in_current_thread()
        box["after"] = int(dynamo.config.recompile_limit)   # 补设后：应为配置值

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert box["naive"] == 8, "thread-local 语义应使新线程裸读到默认值"
    assert box["after"] == _cfg_limit()


def test_idempotent_per_thread():
    """同线程二次调用是 no-op（不会把值改回去）。"""
    dynamo.config.recompile_limit = 8
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == _cfg_limit()
    # 人为调大，再调用 helper ⇒ 不应被改小（幂等标记已置）
    dynamo.config.recompile_limit = _cfg_limit() + 100
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == _cfg_limit() + 100


def test_reapplies_after_compile_is_reenabled(monkeypatch):
    """关闭后重新开启 compile，线程内补设不能被旧状态短路。"""
    monkeypatch.setattr(model_module, "USE_COMPILE", False)
    dynamo.config.recompile_limit = 8
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == 8

    monkeypatch.setattr(model_module, "USE_COMPILE", True)
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == _cfg_limit()


def test_does_not_lower_larger_existing_value():
    """当前值**大于**配置值时不写（避免覆盖用户显式调大的值）。"""
    model_module._compile_limit_tls.__dict__.clear()   # 清掉幂等标记，强制走写入判断
    bigger = _cfg_limit() + 50
    dynamo.config.recompile_limit = bigger
    dynamo.config.cache_size_limit = bigger
    model_module._ensure_compile_limits_in_current_thread()
    assert dynamo.config.recompile_limit == bigger
    assert dynamo.config.cache_size_limit == bigger


def test_forward_layers_entry_ensures_limits(monkeypatch):
    """`forward_layers` 被 `@_serialized_model_access` 装饰 ⇒ 其入口会补设（结构断言）。"""
    # 直接验证装饰器 wrapper 会调用 helper：把 helper 换成计数器
    calls = {"n": 0}
    real = model_module._ensure_compile_limits_in_current_thread

    def _spy():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(model_module, "_ensure_compile_limits_in_current_thread", _spy)
    # 重新包一个最小对象：只需 _lock 与一个被装饰的方法
    class _Probe:
        def __init__(self):
            self._lock = threading.RLock()

        @model_module._serialized_model_access
        def touch(self):
            return "ok"

    assert _Probe().touch() == "ok"
    assert calls["n"] == 1, "装饰器 wrapper 必须在每次进入时调用线程内补设"
