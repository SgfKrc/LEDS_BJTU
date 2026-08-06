"""T9 简化聊天页（Textual PoC）— 参考 Claude Code 交互形态的终端对话入口。

模式:
    python -m src.tui_chat --fixture fixtures/chat_interactive_fixture.sse
    python -m src.tui_chat --host http://127.0.0.1:8000

契约:
    POST /api/chat/stream  (streaming_mode=interactive)
      start -> token* -> done | error | cancelled
    POST /api/chat/generations/{id}/cancel
    GET/POST /api/sessions*

设计边界（TUI 适配实施计划 §9）:
    - 双进程入口，不与 tui_admin.py 的手写 ANSI 主循环混用；
    - 无 Textual 时静默提示并退出，不影响管理 TUI / --plain；
    - 模型输出经 Markdown 渲染，不执行内容；ANSI 注入由渲染层过滤；
    - 只以 done 事件 metrics 展示执行模式，不推断分布式。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.parse
import uuid
from typing import List, Optional

try:
    import httpx
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import VerticalScroll
    from textual.message import Message
    from textual.widgets import Header, Markdown, Static, TextArea
except ImportError:  # pragma: no cover - 环境引导路径
    print(
        "T9 聊天页需要可选依赖 Textual + httpx。\n"
        "安装: pip install -r packaging/requirements-tui.txt\n"
        "管理 TUI（start_tui.bat）不受影响。"
    )
    sys.exit(2)

from tui_sse import SSEDecoder, decode_json_event  # noqa: E402
from tui_shared import (  # noqa: E402
    API_PATHS,
    ROUTE_LABELS,
    ROUTING_PREFERENCES,
    build_interactive_request,
    format_metrics,
    help_text,
    parse_session_line,
    resolve_route_arg,
)

APP_VERSION = "t9-poc"


class InputSubmitted(Message):
    """聊天输入提交（Enter 发送）。"""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class CancelRequested(Message):
    """Ctrl+C：请求停止当前生成（TextArea 内部 copy 绑定优先于 App）。"""


class ChatInput(TextArea):
    """多行输入区：Enter 发送，Alt+Enter / Ctrl+J 换行，Ctrl+C 停止。"""

    BINDINGS = []

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.soft_wrap = True
        self.max_length = 4000

    def _on_key(self, event) -> None:
        key = event.key
        if key == "enter":
            event.prevent_default()
            self.post_message(InputSubmitted(self.text))
            return
        if key in ("ctrl+j", "alt+enter", "meta+enter"):
            event.prevent_default()
            self.insert("\n")
            return
        if key == "ctrl+c":
            event.prevent_default()
            self.post_message(CancelRequested())
            return
        super()._on_key(event)


class TuiChatApp(App[None]):
    """T9 聊天页主应用。"""

    CSS = """
    ChatTranscript {
        height: 1fr;
        border-bottom: solid $border;
    }
    ChatInput {
        height: auto;
        max-height: 8;
        border: round $accent;
        margin: 1 2;
    }
    #status-bar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 2;
    }
    #transcript-empty {
        height: 1fr;
        color: $text-muted;
        padding: 1 2;
    }
    .msg-user { margin: 0 2 0 0; }
    .msg-assistant { margin: 0 0 0 2; }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_ignore", "停止"),
        Binding("ctrl+n", "new_session", "新会话"),
        Binding("ctrl+l", "clear_scrollback_hint", "重绘提示"),
        Binding("escape", "escape_hint", "退出提示"),
    ]

    def __init__(self, *, host: str = "", fixture: str = "") -> None:
        super().__init__()
        self.host = host.rstrip("/")
        self.fixture_path = fixture
        self.session_id: Optional[str] = None
        self.session_title: str = "新会话"
        self.route: str = "auto"
        self.thinking: bool = False
        self.is_generating = False
        self.generation_id: Optional[str] = None
        self._epoch = 0
        self._history: List[str] = []
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        self._messages: List[Markdown] = []
        self._status = "未连接"

    # ---- 生命周期 ----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(Static("输入 /help 查看命令；Enter 发送，Alt+Enter 换行。",
                                    id="transcript-empty", markup=False),
                             id="transcript")
        yield ChatInput(placeholder="输入消息…（/help 查看命令）")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self._update_status()
        self.query_one(ChatInput).focus()
        if self.host:
            self._status = f"master@{self.host}"
            self._update_status()
            self.run_worker(self._restore_active_session(), exclusive=False)
            self.run_worker(self._fetch_models_meta(), exclusive=False)

    async def _restore_active_session(self) -> None:
        """启动时恢复最近活跃会话（后端返回 active_session_id）。"""
        if not self.host:
            return
        try:
            res = await self._client.get(f"{self.host}/api{API_PATHS['sessions']}")
            res.raise_for_status()
            data = res.json()
            active = data.get("active_session_id") or ""
            if not active:
                return
            await self._cmd_resume(active)
        except Exception as exc:
            await self._append_system(f"恢复最近会话失败: {exc}")

    # ---- 事件 ----

    async def on_input_submitted(self, message: InputSubmitted) -> None:
        text = message.text.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._run_command(text)
            return
        if self.is_generating:
            self.notify("正在生成，请先 Ctrl+C 停止", severity="warning")
            return
        await self._send_message(text)

    async def on_cancel_requested(self, message: CancelRequested) -> None:
        self.action_cancel_or_ignore()

    # ---- 命令 ----

    async def _run_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("/new",):
            await self._cmd_new_session()
        elif cmd in ("/resume",):
            await self._cmd_resume(arg)
        elif cmd in ("/rename",):
            await self._cmd_rename(arg)
        elif cmd in ("/delete-session",):
            await self._cmd_delete_session()
        elif cmd in ("/route",):
            resolved = resolve_route_arg(arg)
            if resolved:
                self.route = resolved
                self._update_status()
                self.notify(f"路由偏好: {ROUTE_LABELS[resolved]}")
            else:
                self.notify(f"用法: /route {'|'.join(ROUTE_LABELS)}", severity="warning")
        elif cmd in ("/cancel",):
            await self._cancel_generation()
        elif cmd in ("/clear",):
            await self._clear_transcript()
        elif cmd in ("/thinking",):
            self.thinking = arg in ("on", "1", "true")
            self.notify(f"思考内容展示: {'开' if self.thinking else '关'}")
        elif cmd in ("/help",):
            await self._append_system(help_text())
        elif cmd in ("/sessions",):
            await self._cmd_sessions()
        elif cmd in ("/quit", "/exit", "/q"):
            self.exit()
        else:
            self.notify(f"未知命令: {cmd}（/help 查看）", severity="warning")
        self._clear_input()

    async def _auto_cancel_generation(self) -> None:
        """会话/模型切换前自动取消当前生成（计划 §9.6，不拒绝切换）。"""
        if self.is_generating:
            await self._cancel_generation()

    async def _cmd_new_session(self) -> None:
        await self._auto_cancel_generation()
        self._epoch += 1
        if not self.host:
            self.session_id = None
            self.session_title = "新会话（fixture）"
            await self._append_system("已切换到新会话（fixture 模式不持久化）")
            self._update_status()
            return
        try:
            res = await self._client.post(
                f"{self.host}/api{API_PATHS['sessions']}", json={},
            )
            res.raise_for_status()
            data = res.json()
            self.session_id = data.get("id") or data.get("session_id")
            self.session_title = data.get("title") or "新会话"
            await self._append_system(f"已创建会话: {self.session_title}")
            self._update_status()
        except Exception as exc:
            await self._append_system(f"创建会话失败: {exc}")

    async def _cmd_resume(self, session_id: str) -> None:
        if not session_id:
            self.notify("用法: /resume <session_id>", severity="warning")
            return
        await self._auto_cancel_generation()
        self._epoch += 1
        self.session_id = session_id
        if self.host:
            try:
                # activate 端点返回该会话的消息历史（对齐后端契约）
                quoted = urllib.parse.quote(session_id, safe="")
                res = await self._client.post(
                    f"{self.host}/api{API_PATHS['sessions_activate'].format(session_id=quoted)}",
                )
                res.raise_for_status()
                data = res.json()
                self.session_id = data.get("session_id") or session_id
                self.session_title = data.get("title") or "历史会话"
                for item in (data.get("messages") or []):
                    role = item.get("role")
                    content = item.get("content") or ""
                    if role in ("user", "assistant"):
                        await self._append_message(role, content)
                await self._append_system(f"已恢复会话: {self.session_title}")
            except Exception as exc:
                await self._append_system(f"历史加载失败: {exc}")
        else:
            await self._append_system(f"已切换到会话: {session_id}（fixture 模式无历史）")
        self._update_status()

    async def _cmd_rename(self, title: str) -> None:
        if not title:
            self.notify("用法: /rename <title>", severity="warning")
            return
        if not self.session_id or not self.host:
            self.notify("当前无会话可重命名", severity="warning")
            return
        try:
            quoted = urllib.parse.quote(self.session_id, safe="")
            res = await self._client.put(
                f"{self.host}/api{API_PATHS['sessions_detail'].format(session_id=quoted)}",
                json={"title": title},
            )
            res.raise_for_status()
            self.session_title = title
            await self._append_system(f"会话已重命名: {title}")
            self._update_status()
        except Exception as exc:
            await self._append_system(f"重命名失败: {exc}")

    async def _cmd_delete_session(self) -> None:
        if not self.session_id:
            self.notify("当前无会话", severity="warning")
            return
        sid = self.session_id
        await self._auto_cancel_generation()
        self._epoch += 1
        if self.host:
            try:
                quoted = urllib.parse.quote(sid, safe="")
                res = await self._client.delete(
                    f"{self.host}/api{API_PATHS['sessions_detail'].format(session_id=quoted)}",
                )
                res.raise_for_status()
            except Exception as exc:
                await self._append_system(f"删除会话失败: {exc}")
                return
        self.session_id = None
        self.session_title = "新会话"
        await self._clear_transcript()
        await self._append_system(f"会话已删除: {sid}")
        self._update_status()

    async def _cmd_sessions(self) -> None:
        if not self.host:
            await self._append_system("fixture 模式无会话列表")
            return
        try:
            res = await self._client.get(f"{self.host}/api{API_PATHS['sessions']}")
            res.raise_for_status()
            sessions = res.json().get("sessions") or []
            if not sessions:
                await self._append_system("暂无历史会话")
                return
            for s in sessions[-10:]:
                await self._append_system(parse_session_line(s))
        except Exception as exc:
            await self._append_system(f"会话列表失败: {exc}")

    # ---- 发送与流式 ----

    async def _send_message(self, text: str) -> None:
        self._history.append(text)
        await self._append_message("user", text)
        self._epoch += 1
        self._clear_input()
        self.run_worker(
            self._stream_worker(text, self._epoch),
            exclusive=False,
            group="stream",
        )

    def _stream_fixture(self) -> List[dict]:
        """读取 fixture 文件（SSE 格式，逐行喂入保留 chunk 切分语义）。"""
        events: List[dict] = []
        with open(self.fixture_path, "r", encoding="utf-8") as fh:
            decoder = SSEDecoder()
            for line in fh:
                for payload in map(
                    decode_json_event,
                    decoder.feed(line.encode("utf-8")),
                ):
                    if payload is not None:
                        events.append(payload)
            # 文件尾部可能没有事件结束空行：flush 残帧
            for payload in map(decode_json_event, decoder.feed(b"\n\n")):
                if payload is not None:
                    events.append(payload)
        return events

    async def _stream_worker(self, text: str, epoch: int) -> None:
        self.is_generating = True
        self.generation_id = f"gen_{uuid.uuid4().hex}"
        self._update_status()

        assistant = await self._append_message("assistant", "")
        partial = ""
        terminated = False

        def _mark_terminated() -> None:
            nonlocal terminated
            terminated = True

        async def _handle_payload(payload: dict) -> bool:
            """处理一个事件；返回 True 表示流已终止（调用方应退出）。"""
            if payload.get("start"):
                self.generation_id = payload.get("generation_id") or self.generation_id
                if payload.get("session_id"):
                    self.session_id = payload["session_id"]
                self._update_status()
                return False
            if payload.get("done"):
                _mark_terminated()
                await self._show_done(payload, epoch)
                return True
            if payload.get("cancelled"):
                _mark_terminated()
                await self._append_system("生成已取消")
                self._update_status()
                return True
            if payload.get("error"):
                _mark_terminated()
                await self._append_system(f"错误: {payload['error']}")
                return True
            return False

        if self.fixture_path:
            try:
                for payload in self._stream_fixture():
                    if epoch != self._epoch:
                        _mark_terminated()
                        return
                    if payload.get("token"):
                        partial += str(payload["token"])
                        await assistant.update(partial)
                        await asyncio.sleep(0.03)
                        continue
                    if await _handle_payload(payload):
                        return
            except Exception as exc:
                _mark_terminated()
                await self._append_system(f"fixture 读取失败: {exc}")
            finally:
                if not terminated:
                    await self._append_system("流意外结束（未收到 done/cancelled/error 事件）")
                self._finish_generation()
            return

        # ---- 真实后端（interactive 契约）----
        body = build_interactive_request(
            text,
            session_id=self.session_id,
            generation_id=self.generation_id,
            routing_preference=self.route,
            show_thinking=self.thinking,
        )
        try:
            async with self._client.stream(
                "POST", f"{self.host}/api{API_PATHS['chat_stream']}", json=body,
            ) as response:
                response.raise_for_status()
                decoder = SSEDecoder()
                async for chunk in response.aiter_bytes():
                    if epoch != self._epoch:
                        await response.aclose()
                        _mark_terminated()
                        return
                    for payload in map(decode_json_event, decoder.feed(chunk)):
                        if payload is None:
                            continue
                        if payload.get("token"):
                            partial += str(payload["token"])
                            await assistant.update(partial)
                            continue
                        if await _handle_payload(payload):
                            return
        except httpx.HTTPStatusError as exc:
            _mark_terminated()
            await self._append_system(
                f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except (httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError,
                httpx.ConnectTimeout) as exc:
            # 网络断开/超时：已生成内容保留在 assistant 占位中
            _mark_terminated()
            await self._append_system(f"连接中断: {exc}")
        except Exception as exc:
            _mark_terminated()
            await self._append_system(f"流式请求失败: {exc}")
        finally:
            if not terminated:
                # 连接正常关闭但未收到任何终止事件（空响应/服务端提前关闭）
                await self._append_system("连接意外结束（未收到完成事件）；已生成内容保留在上方")
            self._finish_generation()

    async def _show_done(self, payload: dict, epoch: int) -> None:
        if epoch != self._epoch:
            return
        await self._append_system(
            format_metrics(
                payload.get("metrics"),
                history_committed=payload.get("history_committed"),
            ),
        )

    def _finish_generation(self) -> None:
        self.is_generating = False
        self.generation_id = None
        self._update_status()

    async def _cancel_generation(self) -> None:
        if not self.is_generating or not self.generation_id:
            self.notify("当前没有正在生成的请求")
            return
        if self.host:
            try:
                res = await self._client.post(
                    f"{self.host}/api{API_PATHS['chat_cancel'].format(generation_id=self.generation_id)}",
                )
                res.raise_for_status()
            except Exception as exc:
                self.notify(f"取消请求失败: {exc}", severity="warning")
                return
        self._epoch += 1
        await self._append_system("正在停止生成…")
        self._finish_generation()

    async def _fetch_models_meta(self) -> None:
        try:
            res = await self._client.get(f"{self.host}/api{API_PATHS['models_current']}")
            res.raise_for_status()
            data = res.json()
            name = data.get("model") or data.get("name") or data.get("model_id") or ""
            if name:
                self.title = f"QLH Chat — {name}"
        except Exception:
            pass

    # ---- 动作 ----

    def action_cancel_or_ignore(self) -> None:
        if self.is_generating:
            self.run_worker(self._cancel_generation())
        else:
            self.notify("空闲状态，Ctrl+C 不退出；/quit 退出")

    async def action_new_session(self) -> None:
        if self.is_generating:
            self.notify("先停止当前生成", severity="warning")
            return
        self._epoch += 1
        self.session_id = None
        self.session_title = "新会话"
        await self._clear_transcript()
        self._update_status()

    def action_clear_scrollback_hint(self) -> None:
        self.notify("Textual 自动重绘；终端残影可 Ctrl+L 由终端处理")

    def action_escape_hint(self) -> None:
        self.notify("管理 TUI 入口见 start_tui.bat；此处输入 /quit 退出")

    # ---- 工具 ----

    async def _append_message(self, role: str, content: str) -> Markdown:
        transcript = self.query_one("#transcript", VerticalScroll)
        empty = transcript.query("#transcript-empty")
        if empty:
            empty.remove()
        label = "You" if role == "user" else "Assistant"
        md = Markdown(f"**{label}**\n\n{content}")
        md.add_class("msg-user" if role == "user" else "msg-assistant")
        await transcript.mount(md)
        self._messages.append(md)
        transcript.scroll_end(animate=False)
        return md

    async def _append_system(self, text: str) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        md = Markdown(f"`{text}`")
        md.add_class("msg-assistant")
        await transcript.mount(md)
        transcript.scroll_end(animate=False)

    async def _clear_transcript(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children):
            child.remove()
        self._messages = []
        await transcript.mount(Static("会话已清空。", markup=False))

    def _clear_input(self) -> None:
        self.query_one(ChatInput).text = ""

    def _update_status(self) -> None:
        bar = self.query_one("#status-bar", Static)
        mode = "fixture" if self.fixture_path else self.host
        gen = "生成中" if self.is_generating else "空闲"
        bar.update(
            f"{mode} · {self.session_title or self.session_id or '无会话'} · "
            f"{ROUTE_LABELS.get(self.route, self.route)} · {gen} · "
            "Enter 发送 · Alt+Enter 换行 · Ctrl+C 停止 · /help",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tui_chat",
        description="T9 简化聊天页（Textual PoC）",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--host", default="", help="后端地址，如 http://127.0.0.1:8000")
    mode.add_argument(
        "--fixture", default="",
        help="SSE fixture 文件路径（无需后端，重放固定事件流）",
    )
    parser.add_argument("--route", default="auto", help="初始路由偏好")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.route not in ROUTING_PREFERENCES:
        print(f"非法路由偏好: {args.route}")
        return 2
    app = TuiChatApp(host=args.host, fixture=args.fixture)
    app.route = resolve_route_arg(args.route) or args.route
    app.title = "QLH Chat"
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
