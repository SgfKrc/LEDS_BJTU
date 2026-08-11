"""
T9 聊天页（Textual PoC）冒烟测试
===============================
headless 运行：fixture 模式重放、输入发送、命令解析、状态栏更新。
真实后端交互由 tests/test_chat_interactive.py 的契约测试覆盖。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import httpx

from tui_chat import ChatInput, TuiChatApp
from tui_sse import SSEDecoder, decode_json_event

FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures" / "chat_interactive_fixture.sse",
)


def _write_fixture(tmp_path, events: list) -> str:
    """把事件列表写成 SSE fixture 文件。"""
    path = tmp_path / "fixture.sse"
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


class _BrokenStreamClient:
    """stream() 直接抛网络错误的假客户端（async context manager）。"""

    def __init__(self, exc: Exception):
        self.exc = exc

    def stream(self, *args, **kwargs):
        class _ContextManager:
            def __init__(self, exc):
                self.exc = exc

            async def __aenter__(self):
                raise self.exc

            async def __aexit__(self, *args):
                return False

        return _ContextManager(self.exc)


class _FakeSessionClient:
    """内存版会话 API 假客户端（GET/POST/PUT/DELETE，async 接口）。"""

    def __init__(self):
        self.calls = []
        self.sessions = {}       # id -> {"id", "title", "messages"}
        self.active_session_id = None

    def _seed(self, session_id, title, messages=None):
        self.sessions[session_id] = {
            "id": session_id,
            "title": title,
            "messages": messages or [],
        }

    def _resp(self, payload, status=200):
        class _Resp:
            def __init__(self, payload, status):
                self._payload = payload
                self.status_code = status

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self):
                return self._payload

        return _Resp(payload, status)

    def _norm(self, path: str) -> str:
        """把完整 URL 规范化为 /api 之后的路径（如 /api/sessions/x → sessions/x）。"""
        if "/api/" in path:
            return path.split("/api/", 1)[1]
        return path.lstrip("/")

    # ---- 路由（async，对齐 httpx.AsyncClient 调用方式）----
    async def get(self, path, params=None):
        self.calls.append(("GET", path))
        path = self._norm(path)
        if path == "sessions":
            sessions = [
                {"id": s["id"], "title": s["title"],
                 "message_count": len(s["messages"])}
                for s in self.sessions.values()
            ]
            return self._resp({"sessions": sessions,
                               "active_session_id": self.active_session_id,
                               "total": len(sessions)})
        if path.startswith("sessions/") and path.endswith("/activate"):
            session_id = path.split("/")[1]
            session = self.sessions.get(session_id)
            if not session:
                return self._resp({"detail": "not found"}, 404)
            self.active_session_id = session_id
            return self._resp({
                "session_id": session_id,
                "title": session["title"],
                "messages": session["messages"],
            })
        return self._resp({})

    async def post(self, path, body=None, json=None):
        self.calls.append(("POST", path))
        path = self._norm(path)
        body = json if json is not None else body
        if path == "sessions":
            session_id = f"sess_{len(self.sessions) + 1}"
            title = (body or {}).get("title") or "新会话"
            self._seed(session_id, title)
            self.active_session_id = session_id
            return self._resp({"id": session_id, "title": title,
                               "message_count": 0, "active": True})
        if path.endswith("/activate"):
            return await self.get(path)
        return self._resp({})

    async def put(self, path, body=None, json=None):
        self.calls.append(("PUT", path))
        path = self._norm(path)
        session_id = path.split("/")[1]
        session = self.sessions.get(session_id)
        if not session:
            return self._resp({"detail": "not found"}, 404)
        payload = json if json is not None else body
        session["title"] = payload["title"]
        return self._resp({"id": session_id, "title": payload["title"]})

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        path = self._norm(path)
        session_id = path.split("/")[1]
        self.sessions.pop(session_id, None)
        if self.active_session_id == session_id:
            self.active_session_id = None
        return self._resp({"status": "deleted", "session_id": session_id})


def _transcript_text(app) -> str:
    """全部 Markdown 内容（含 system 提示消息）。"""
    return "".join(w._markdown for w in app.query("Markdown"))


class TestFixtureParsing:
    """fixture 文件 → 事件流的解析。"""

    def test_fixture_events_parsed(self):
        app = TuiChatApp(fixture=FIXTURE)
        events = app._stream_fixture()
        assert events[0]["start"] is True
        tokens = [e["token"] for e in events if "token" in e]
        assert len(tokens) == 8
        assert events[-1]["cancelled"] is True


class TestSseParserIntegration:
    """tui_chat 与 tui_sse 的集成：真实 SSE 字节流。"""

    def test_chunked_feed_matches_fixture(self):
        raw = Path(FIXTURE).read_bytes()
        decoder = SSEDecoder()
        # 按 7 字节小块喂入，模拟任意 TCP 切分
        payloads = []
        for i in range(0, len(raw), 7):
            for payload in map(decode_json_event, decoder.feed(raw[i:i + 7])):
                if payload is not None:
                    payloads.append(payload)
        for payload in map(decode_json_event, decoder.feed(b"\n\n")):
            if payload is not None:
                payloads.append(payload)
        assert payloads[0]["start"] is True
        assert sum(1 for e in payloads if "token" in e) == 8
        assert payloads[-1]["cancelled"] is True


class TestAppHeadless:
    """Textual headless 冒烟（fixture 模式，不连后端）。"""

    def test_send_message_renders_transcript(self):
        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            async with app.run_test() as pilot:
                assert not app.is_generating
                app.query_one(ChatInput).text = "你好 T9"
                await pilot.press("enter")
                # fixture 流含 8 token × 0.03s sleep + done
                await pilot.pause(1.0)
                assert not app.is_generating
                # user + assistant 两条消息已写入 transcript
                assert len(app._messages) >= 2
                assert "你好 T9" in app._messages[0]._markdown
                assistant_text = "".join(
                    m._markdown for m in app._messages[1:]
                )
                assert "QLH T9 聊天页" in assistant_text
                assert "fixture" in str(app.query_one("#status-bar").render())
        asyncio.run(_run())

    def test_route_command_updates_state(self):
        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/route local_only"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert app.route == "local_only"
                bar = str(app.query_one("#status-bar").render())
                assert "route:local" in bar
        asyncio.run(_run())

    def test_image_command_queues_then_sends_without_persisting_bytes(self, tmp_path):
        image_path = tmp_path / "fixture.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            app.route = "local_only"
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = f"/image {image_path}"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert len(app._pending_images) == 1
                assert app.route == "auto"

                app.query_one(ChatInput).text = ""
                await pilot.press("enter")
                await pilot.pause(1.0)
                assert not app._pending_images
                assert "请描述这些图片" in app._messages[0]._markdown
                assert "data:image" not in "".join(
                    message._markdown for message in app._messages
                )
        asyncio.run(_run())

    def test_unknown_command_notifies(self):
        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/bogus-command"
                await pilot.press("enter")
                await pilot.pause(0.1)
                assert not app.is_generating
        asyncio.run(_run())

    def test_new_session_resets(self):
        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "你好"
                await pilot.press("enter")
                await pilot.pause(1.0)
                app.query_one(ChatInput).text = "/new"
                await pilot.press("enter")
                await pilot.pause(0.2)
                assert app.session_id is None
                assert not app.is_generating
        asyncio.run(_run())


class TestStreamFailureStates:
    """T9.3 错误恢复：空响应/error/断线/取消的确定状态。"""

    def test_empty_stream_shows_termination(self, tmp_path):
        fixture = _write_fixture(tmp_path, [{"start": True, "generation_id": "gen_e"}])

        async def _run():
            app = TuiChatApp(fixture=fixture)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "hi"
                await pilot.press("enter")
                await pilot.pause(0.5)
                assert not app.is_generating
                text = _transcript_text(app)
                assert "流意外结束" in text
        asyncio.run(_run())

    def test_error_event_keeps_partial(self, tmp_path):
        fixture = _write_fixture(tmp_path, [
            {"start": True, "generation_id": "gen_e"},
            {"token": "部分内容"},
            {"error": "引擎故障"},
        ])

        async def _run():
            app = TuiChatApp(fixture=fixture)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "hi"
                await pilot.press("enter")
                await pilot.pause(0.5)
                assert not app.is_generating
                text = _transcript_text(app)
                # 已生成内容保留，错误有确定提示
                assert "部分内容" in text
                assert "引擎故障" in text
        asyncio.run(_run())

    def test_disconnect_shows_state(self):
        async def _run():
            app = TuiChatApp(host="http://127.0.0.1:9")
            app._client = _BrokenStreamClient(httpx.ReadError("connection lost"))
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "hi"
                await pilot.press("enter")
                await pilot.pause(0.5)
                assert not app.is_generating
                text = _transcript_text(app)
                assert "连接中断" in text
        asyncio.run(_run())

    def test_cancel_preserves_partial(self, tmp_path):
        # 20 token 的慢流（0.03s/token），取消窗口足够宽且无 done 事件
        events = [{"start": True, "generation_id": "gen_c"}]
        for i in range(20):
            events.append({"token": f"T{i} "})
        fixture = _write_fixture(tmp_path, events)

        async def _run():
            app = TuiChatApp(fixture=fixture)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "你好"
                await pilot.press("enter")
                await pilot.pause(0.2)  # 部分 token 已到达
                await pilot.press("ctrl+c")
                await pilot.pause(0.5)
                assert not app.is_generating
                text = _transcript_text(app)
                # 取消有确定提示；已生成 partial 保留
                assert "正在停止生成" in text or "已取消" in text
                assert "T0 " in text
                assert "T19" not in text  # 未跑完的部分不出现
        asyncio.run(_run())

    def test_input_cleared_after_send(self):
        async def _run():
            app = TuiChatApp(fixture=FIXTURE)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "你好"
                await pilot.press("enter")
                await pilot.pause(0.3)  # async 提交链（mount→clear）完成
                assert app.query_one(ChatInput).text == ""
        asyncio.run(_run())


class TestSessionManagement:
    """T9.4 会话与历史：new/resume/rename/delete 与启动恢复。"""

    def _make_app(self, fake):
        app = TuiChatApp(host="http://127.0.0.1:9")
        app._client = fake
        return app

    def test_new_creates_session(self):
        async def _run():
            fake = _FakeSessionClient()
            app = self._make_app(fake)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/new"
                await pilot.press("enter")
                await pilot.pause(0.2)
                assert app.session_id == "sess_1"
                assert any(
                    c[0] == "POST" and "/api/sessions" in c[1]
                    for c in fake.calls
                )
        asyncio.run(_run())

    def test_resume_loads_history(self):
        async def _run():
            fake = _FakeSessionClient()
            fake._seed("sess_old", "旧会话", [
                {"role": "user", "content": "历史问题"},
                {"role": "assistant", "content": "历史回答"},
            ])
            app = self._make_app(fake)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/resume sess_old"
                await pilot.press("enter")
                await pilot.pause(0.3)
                assert app.session_id == "sess_old"
                text = _transcript_text(app)
                assert "历史问题" in text
                assert "历史回答" in text
        asyncio.run(_run())

    def test_rename_updates_title(self):
        async def _run():
            fake = _FakeSessionClient()
            fake._seed("sess_r", "旧名")
            app = self._make_app(fake)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/resume sess_r"
                await pilot.press("enter")
                await pilot.pause(0.3)
                app.query_one(ChatInput).text = "/rename 调试流水线"
                await pilot.press("enter")
                await pilot.pause(0.2)
                assert app.session_title == "调试流水线"
                assert any(
                    c[0] == "PUT" and "sess_r" in c[1] for c in fake.calls
                )
        asyncio.run(_run())

    def test_delete_session_resets_state(self):
        async def _run():
            fake = _FakeSessionClient()
            fake._seed("sess_d", "要删除")
            app = self._make_app(fake)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "/resume sess_d"
                await pilot.press("enter")
                await pilot.pause(0.3)
                app.query_one(ChatInput).text = "/delete-session"
                await pilot.press("enter")
                await pilot.pause(0.3)
                assert app.session_id is None
                assert any(
                    c[0] == "DELETE" and "sess_d" in c[1] for c in fake.calls
                )
        asyncio.run(_run())

    def test_startup_restores_active_session(self):
        async def _run():
            fake = _FakeSessionClient()
            fake._seed("sess_active", "最近会话", [
                {"role": "user", "content": "启动恢复"},
            ])
            fake.active_session_id = "sess_active"
            app = self._make_app(fake)
            async with app.run_test() as pilot:
                await pilot.pause(0.4)  # on_mount 恢复 worker 执行
                assert app.session_id == "sess_active"
                text = _transcript_text(app)
                assert "启动恢复" in text
        asyncio.run(_run())

    def test_session_switch_cancels_generation_and_fences_late_tokens(self, tmp_path):
        """生成中 /new：自动取消，迟到 token 不污染新会话。"""
        events = [{"start": True, "generation_id": "gen_race"}]
        for i in range(30):
            events.append({"token": f"L{i} "})
        fixture = _write_fixture(tmp_path, events)

        async def _run():
            app = TuiChatApp(fixture=fixture)
            async with app.run_test() as pilot:
                app.query_one(ChatInput).text = "生成"
                await pilot.press("enter")
                await pilot.pause(0.15)  # 流进行中
                assert app.is_generating
                app.query_one(ChatInput).text = "/new"
                await pilot.press("enter")
                await pilot.pause(0.5)
                assert not app.is_generating
                text = _transcript_text(app)
                # 迟到 token 不得出现在新会话 transcript
                assert "L29" not in text
        asyncio.run(_run())
