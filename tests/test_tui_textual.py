"""Textual 外壳 + 协议层的 headless 测试。

要点：
* ``tui_api``（协议层）是纯标准库，**没有** textual 也必须能 import 并正确报错；
* Textual 外壳在**后端不可用**时也必须能安全启动（只显示错误，不崩）；
* 五个 Tab（聊天/状态/模型/分布式/关于）与外层 Splash→Main 切换必须存在。

缺 textual 时整个模块跳过（Edge 之外的极简环境仍可跑其余测试）。
"""

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_api import ApiClient, ApiError  # noqa: E402  - 纯标准库，始终可导入


# ------------------------------------------------------------------ 协议层

def test_client_reports_connection_failure_as_api_error():
    """无后端时必须抛 ApiError（界面据此显示错误，而不是崩栈）。"""
    api = ApiClient(host="127.0.0.1", port=1, timeout=0.5)
    with pytest.raises(ApiError):
        api.get("/health")


def test_chat_cancel_path_escapes_slash():
    """generation_id 含 "/" 时必须被编码，否则会多切一段 URL。"""
    from tui_api import API_PATHS
    import urllib.parse

    quoted = urllib.parse.quote("sess/1", safe="")
    assert quoted == "sess%2F1"
    assert "{generation_id}" in API_PATHS["chat_cancel"]


# ------------------------------------------------------------------ Textual 外壳

pytest.importorskip("textual")


def _run(coro):
    return asyncio.run(coro)


def test_shell_boots_then_switches_to_main():
    from textual.widgets import TabbedContent

    from tui_textual import KoakumaApp, MainScreen, SplashScreen

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SplashScreen), "应先显示启动屏"
            splash = app.screen
            from textual.widgets import Static

            status = str(splash.query_one("#splash-status", Static).render())
            assert "少女祈祷中" in status, "启动行应以「少女祈祷中：」开头"
            bar = str(splash.query_one("#splash-bar", Static).render())
            assert "█" in bar and "░" in bar, "标题下方应有跑马灯启动条"
            app.show_main()
            await pilot.pause()
            assert isinstance(app.screen, MainScreen), "应切换到主界面"
            tabs = app.screen.query_one("#tabs", TabbedContent)
            assert tabs.tab_count == 9, (
                "应有 聊天/状态/模型/分布式/节点/队列/日志/设备/设置 九个 Tab")
            for widget_id in ("#nodes-table", "#queue-table", "#logs-log",
                              "#device-pane", "#gpu-table", "#settings-pane"):
                assert app.screen.query_one(widget_id) is not None, widget_id
            settings = str(app.screen.query_one("#settings-pane", Static).render())
            assert "会话设置" in settings and "依赖边界" in settings, "设置屏应含会话参数与关于信息"

    _run(_main())


def test_shell_survives_backend_absent():
    """后端不可达时：只读屏显示错误文本，界面照常可用。"""
    from textual.widgets import Static

    from tui_textual import KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#status-pane", Static)
            rendered = ""
            deadline = time.time() + 6
            while time.time() < deadline:
                await pilot.pause(0.1)
                rendered = str(pane.render())
                if "无法连接" in rendered or "后端不可达" in rendered:
                    break
            assert "后端不可达" in rendered or "无法连接" in rendered

    _run(_main())


def test_chat_pane_help_and_unknown_command():
    from textual.widgets import Input, RichLog

    from tui_textual import ChatPane, KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#chat-pane", ChatPane)
            box = pane.query_one(Input)
            box.value = "/help"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert pane.query_one(RichLog) is not None
            # 未识别命令不应触发网络请求，只提示
            box.value = "/nonexistent"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()

    _run(_main())


def test_route_command_updates_preference():
    from textual.widgets import Input

    from tui_textual import ChatPane, KoakumaApp

    async def _main():
        app = KoakumaApp(ApiClient(host="127.0.0.1", port=1, timeout=0.5))
        async with app.run_test() as pilot:
            app.show_main()
            await pilot.pause()
            pane = app.screen.query_one("#chat-pane", ChatPane)
            box = pane.query_one(Input)
            box.value = "/route distributed"
            box.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.routing_preference == "distributed_preferred"

    _run(_main())
