"""★ 2026-09-30：`/api/chat` 客户端断开必须触发服务端取消。

此前该同步端点既不接收 `Request` 也不检测断开：实测 `curl -m 3` 切断后服务端仍把
`max_new_tokens` 跑满（decode step 一路到 299/300，`max_new_tokens=300`）—— 纯浪费，
分布式下还持续占用远端 relay 段。修复方式是复用**既有**的取消链路
（`generation_id` → `cancel_event`，与 `cancel_chat_generation` 同一条），
而不是新造一套。

⚠️ 实现细节（有实测依据）：**不能**用 `Request.is_disconnected()`。starlette 1.3 里它用
「立即取消的 CancelScope」做非阻塞探测，只在该 receive 队列里**已经**躺着
`http.disconnect` 时才返回 True；实测本服务（uvicorn 0.49 + starlette 1.3）在
`/api/chat` 上恒返回 False（watcher 日志按 0.25s 轮询 `polls=20/40` 却从不命中）。
因此改为**带超时地真等** `receive()`。

这些用例只测轮询器本身（假 `Request`），不启动真实 ASGI 栈。
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

_MOD = None


class _NullLogger:
    def info(self, *args, **kwargs) -> None: pass
    def warning(self, *args, **kwargs) -> None: pass
    def debug(self, *args, **kwargs) -> None: pass
    def error(self, *args, **kwargs) -> None: pass


def _load_routes_chat():
    """按路径加载模块；`_api_module` 是运行时由 `configure_route_module` 注入的，
    单测里必须补一个**带 logger** 的替身（否则轮询器一写日志就 AttributeError）。"""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "qlh_routes_chat_probe", ROOT / "src" / "api" / "routes_chat.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if getattr(mod, "_api_module", None) is None:
            mod._api_module = types.SimpleNamespace(logger=_NullLogger())
        _MOD = mod
    return _MOD


class _FakeRequest:
    """第 `disconnect_after + 1` 次 `_receive()` 返回 `http.disconnect`，之前"没有消息"。

    "没有消息"用 `await asyncio.sleep(10)` 模拟 —— 真实 `receive()` 就是这么阻塞的，
    轮询器的 `asyncio.wait_for(...)` 会把它超时取消。
    """

    def __init__(self, disconnect_after: int = 0) -> None:
        self.calls = 0
        self._disconnect_after = int(disconnect_after)

    async def _receive(self):
        self.calls += 1
        if self.calls > self._disconnect_after:
            return {"type": "http.disconnect"}
        await asyncio.sleep(10)
        return {"type": "http.request", "body": b"", "more_body": False}


def test_watcher_sets_cancel_event_when_client_disconnects():
    """客户端已断开 ⇒ 置位 `cancel_event`（主节点/worker 据此中止任务）。"""
    mod = _load_routes_chat()
    cancel_event = threading.Event()
    request = _FakeRequest(disconnect_after=0)

    asyncio.run(asyncio.wait_for(
        mod._watch_client_disconnect(request, cancel_event), timeout=3.0))

    assert cancel_event.is_set()


def test_watcher_sets_cancel_event_after_delayed_disconnect():
    """中途断开也要被捕获（收到 disconnect 前一直"没有消息"）。"""
    mod = _load_routes_chat()
    cancel_event = threading.Event()
    request = _FakeRequest(disconnect_after=3)

    asyncio.run(asyncio.wait_for(
        mod._watch_client_disconnect(request, cancel_event), timeout=5.0))

    assert cancel_event.is_set()
    assert request.calls >= 4


def test_watcher_does_not_poll_once_cancelled():
    """已取消（例如 `cancel_chat_generation` 先到）⇒ 轮询器立即退出，不再打扰请求。"""
    mod = _load_routes_chat()
    cancel_event = threading.Event()
    cancel_event.set()
    request = _FakeRequest(disconnect_after=10 ** 9)

    asyncio.run(asyncio.wait_for(
        mod._watch_client_disconnect(request, cancel_event), timeout=3.0))

    assert request.calls == 0


def test_watcher_is_cancellable_by_the_awaited_request():
    """主协程结束（任务完成）时 watcher 被 `cancel()` —— 不得吞掉 CancelledError。"""
    mod = _load_routes_chat()
    cancel_event = threading.Event()
    request = _FakeRequest(disconnect_after=10 ** 9)

    async def _run() -> str:
        task = asyncio.create_task(mod._watch_client_disconnect(request, cancel_event))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "returned"

    assert asyncio.run(_run()) == "cancelled"
    assert not cancel_event.is_set()
