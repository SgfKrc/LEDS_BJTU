"""TUI 聊天屏：**零依赖**（标准库）的对话页面，作为统一 TUI 的一个屏幕。

背景（TUI 重写 P2）：
    现状聊天页 `src/tui_chat.py` 依赖 **Textual + httpx**，且与管理 TUI 分属两个进程/窗口
    （用户痛点：「开两个窗口、聊天和其他页面分开」）。
    本模块把聊天能力实现为与 `tui_admin` 各管理屏**平级**的屏幕，**只用标准库** +
    项目内 `tui_shared` / `tui_sse`，从而可在统一 TUI 内切换，且不引入任何第三方依赖。

契约（与 `tui_chat.py` 一致，不放松）::

    POST /api/chat/stream   (streaming_mode=interactive)
    事件体为 JSON（**无 SSE `event:` 字段**），按 key 判别：
        {"start": true, "generation_id": ..., "session_id": ...}   -> 记录上下文
        {"token": "..."}                                           -> 追加增量文本
        {"done": true, "response": ..., "metrics": {...},
         "history_committed": bool}                                -> 结束
        {"cancelled": true, "partial": "..."}                      -> 被取消
        {"error": ...}                                             -> 错误

    POST /api/chat/generations/{id}/cancel
    GET  /api/sessions

复用（避免重复实现）：
    * `tui_shared.API_PATHS` / `build_interactive_request` / `format_metrics` / `parse_session_line`
    * `tui_sse.SSEDecoder` / `decode_json_event`
    * `tui_admin.disp_width`（折行宽度口径）
本模块**新增**的只有：ANSI/控制字符过滤（`tui_admin` 没有，安全必需）与对话渲染。

设计边界（沿用《TUI 适配实施计划》§9）：
    * 模型输出经 **ANSI 过滤**后渲染，**不执行内容**；
    * **只以 `done` 事件的 metrics 展示执行模式，不推断分布式**；
    * 流式读取用 `urllib.request`（替代 httpx）；
    * 本屏**只在交互 TUI 内构造** —— 单命令模式（如 `qlh status`）不拉后端。
"""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, Iterator, List, Optional

from tui_sse import SSEDecoder, decode_json_event
from tui_shared import API_PATHS, build_interactive_request, format_metrics, parse_session_line

# ----------------------------------------------------------------------
# 安全：模型输出不可信，渲染前必须过滤 ANSI 转义与危险控制字符
# ----------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text: str) -> str:
    """过滤 ANSI 转义序列与危险控制字符（保留 \\n 与 \\t）。"""
    if not text:
        return ""
    return _CTRL_RE.sub("", _ANSI_RE.sub("", text))


def wrap_display(text: str, width: int) -> List[str]:
    """按显示宽度折行；宽度口径复用 `tui_admin.disp_width`（中文按 2 列）。"""
    if width <= 0:
        return [text]
    try:
        from tui_admin import disp_width  # 延迟导入：避免模块级循环依赖
    except Exception:  # pragma: no cover - 独立冒烟/单测环境
        disp_width = len  # type: ignore[assignment]
    out: List[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        cur = ""
        for ch in para:
            if cur and disp_width(cur + ch) > width:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    return out


class ChatScreen:
    """聊天屏。

    **不继承** `tui_admin.Screen`，而是实现其**鸭子契约**，以避免 import 期双向依赖
    （`tui_admin` 需要在屏幕注册表里引用本类）。契约与 `Screen` 完全一致：
        name / auto_refresh / fetch() / lines(width) / actions()
        self.app / self.api
    """

    name = "对话"
    auto_refresh = False

    #: 保留的最大消息数（防内存无界增长）
    MAX_MESSAGES = 200
    #: SSE 单次 read 超时（秒）；SSE 是长连接，不设总时长上限
    READ_TIMEOUT = 30

    def __init__(
        self,
        app,
        *,
        session_id: Optional[str] = None,
        routing_preference: str = "auto",
        show_thinking: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        transport: Optional[Callable[[dict], Iterator[dict]]] = None,
    ):
        self.app = app
        self.api = getattr(app, "api", None)
        self.data = None
        self.error = None
        self.last_fetch = 0.0

        self.session_id = session_id
        self.routing_preference = routing_preference
        self.show_thinking = show_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        #: 对话历史 [(role, text)]，role ∈ {"user", "assistant", "system"}
        self.messages: List[tuple] = []
        #: 当前流式生成的 generation id（用于取消）
        self.generation_id: Optional[str] = None
        self.streaming = False
        #: 最近一次 done 的展示行（由 `format_metrics` 生成）与原始 metrics
        self.last_status: Optional[str] = None
        self.last_metrics: Optional[dict] = None
        self.thinking_text = ""
        #: 测试可注入的传输层：接收请求体，产出 payload dict 序列
        self._transport = transport
        self._state_lock = threading.RLock()
        self._send_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Screen 鸭子契约
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False):
        """拉取会话列表（失败不致命，绝不让界面崩溃）。"""
        if self.api is None:
            return self.data
        if not force and self.data is not None:
            return self.data
        try:
            self.data = self.api.get(API_PATHS["sessions"])
            self.error = None
        except Exception as e:
            self.error = "会话列表获取失败: %s" % e
        return self.data

    def fetch(self):
        self.refresh(force=True)
        return self.data

    def lines(self, width: int) -> list:
        with self._state_lock:
            messages = tuple(self.messages[-self.MAX_MESSAGES:])
            session_id = self.session_id
            routing_preference = self.routing_preference
            streaming = self.streaming
            last_status = self.last_status
            error = self.error
            thinking_text = self.thinking_text
        out: list = []
        out.append(("head", "对话   会话: %s   路由: %s" % (
            session_id or "（默认）", routing_preference)))
        out.append(("", ""))
        if not messages:
            out.append(("dim", "  还没有消息。按 i 输入并发送。"))
        for role, text in messages:
            out.append(("title", "你: ") if role == "user" else ("ok", "模型: "))
            for ln in wrap_display(sanitize(text), max(8, width - 4)):
                out.append(("", "  " + ln))
            out.append(("", ""))
        if thinking_text and self.show_thinking:
            out.append(("dim", "  思考:"))
            for ln in wrap_display(sanitize(thinking_text), max(8, width - 6)):
                out.append(("dim", "    " + ln))
        if streaming:
            out.append(("warn", "  …生成中（x 取消）"))
        if last_status:
            out.append(("dim", "  " + last_status))
        if error:
            out.append(("warn", "  ! %s" % error))
        return out

    def actions(self) -> list:
        """与 `Screen.actions()` 同形：[(key, label, handler(ui) -> str|None)]。"""
        return [
            ("i", "输入并发送", self._act_send),
            ("n", "新会话", self._act_new_session),
            ("c", "清空本屏", self._act_clear),
            ("x", "取消生成", self._act_cancel),
        ]

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    def _act_send(self, ui) -> Optional[str]:
        text = ui.prompt("你: ")
        if not text or not text.strip():
            return None
        if not self.start_send(text.strip()):
            return "当前已有生成任务，请先取消"
        return None

    def _act_new_session(self, ui) -> Optional[str]:
        if self.streaming:
            return "当前正在生成，请先取消"
        new_id = None
        if self.api is not None:
            try:
                created = self.api.post(API_PATHS["sessions"], {
                    "title": "新对话",
                }) or {}
                new_id = created.get("id") or created.get("session_id")
            except Exception as exc:
                return "创建会话失败: %s" % exc
        with self._state_lock:
            self.session_id = new_id
            self.messages = []
            self.last_status = None
            self.last_metrics = None
            self.thinking_text = ""
        return "已创建新会话: %s" % new_id if new_id else "已开新会话（下一次发送会创建）"

    def _act_clear(self, ui) -> Optional[str]:
        if self.streaming:
            return "当前正在生成，请先取消"
        with self._state_lock:
            self.messages = []
            self.last_status = None
            self.last_metrics = None
            self.thinking_text = ""
        return "本屏已清空"

    def _act_cancel(self, ui) -> Optional[str]:
        with self._state_lock:
            active = self.streaming
            generation_id = self.generation_id
        if not active or not generation_id:
            return "当前没有进行中的生成"
        self.cancel(generation_id)
        return "已请求取消"

    # ------------------------------------------------------------------
    # 核心：发送与流式接收
    # ------------------------------------------------------------------

    def _begin_send(self, message: str) -> bool:
        with self._state_lock:
            if self.streaming:
                return False
            self.messages.append(("user", message))
            self.messages.append(("assistant", ""))
            self.streaming = True
            self.error = None
            self.thinking_text = ""
            # 客户端预生成 ID，首个 token 前即可取消。
            self.generation_id = "gen_%s" % uuid.uuid4().hex[:12]
            return True

    def start_send(self, message: str) -> bool:
        """在后台线程消费流，保证 ANSI TUI 可以持续重绘增量 token。"""
        if not self._begin_send(message):
            return False
        worker = threading.Thread(
            target=self._consume, args=(message,),
            name="qlh-tui-chat-stream", daemon=True,
        )
        with self._state_lock:
            self._send_thread = worker
        worker.start()
        return True

    def _set_assistant(self, text: str) -> None:
        with self._state_lock:
            if self.messages and self.messages[-1][0] == "assistant":
                self.messages[-1] = ("assistant", text)

    def send(self, message: str) -> None:
        """同步消费一条消息，供脚本和单元测试使用。"""
        if self._begin_send(message):
            self._consume(message)

    def _consume(self, message: str) -> None:
        acc: List[str] = []
        try:
            for payload in self.iter_payloads(message):
                if payload.get("start"):
                    with self._state_lock:
                        if payload.get("generation_id"):
                            self.generation_id = payload["generation_id"]
                        if payload.get("session_id"):
                            self.session_id = payload["session_id"]
                elif "token" in payload:
                    piece = payload.get("token")
                    if isinstance(piece, str) and piece:
                        acc.append(piece)
                        self._set_assistant("".join(acc))
                elif "thinking" in payload:
                    piece = payload.get("thinking")
                    if isinstance(piece, str) and piece:
                        with self._state_lock:
                            self.thinking_text += piece
                elif payload.get("done"):
                    metrics = payload.get("metrics") or None
                    with self._state_lock:
                        self.last_metrics = metrics
                        self.last_status = format_metrics(
                            metrics,
                            history_committed=payload.get("history_committed"),
                        )
                        if payload.get("session_id"):
                            self.session_id = payload["session_id"]
                        thinking = payload.get("thinking_content")
                        if isinstance(thinking, str):
                            self.thinking_text = thinking
                    # done.response 是权威完整文本，用它兜底（避免增量拼接误差）
                    final = payload.get("response")
                    if isinstance(final, str):
                        self._set_assistant(final)
                elif payload.get("cancelled"):
                    partial = payload.get("partial")
                    if isinstance(partial, str) and partial:
                        self._set_assistant(partial)
                    with self._state_lock:
                        if payload.get("session_id"):
                            self.session_id = payload["session_id"]
                        self.error = "已被取消"
                elif payload.get("error"):
                    with self._state_lock:
                        self.error = "后端错误: %s" % payload.get("error")
        except Exception as e:  # 网络/解析异常都不应让 TUI 崩溃
            with self._state_lock:
                self.error = "请求失败: %s" % e
        finally:
            with self._state_lock:
                self.streaming = False
                self.generation_id = None
                self._send_thread = None
                if (not acc and self.messages
                        and self.messages[-1][0] == "assistant"
                        and not self.messages[-1][1]):
                    self.messages.pop()   # 未收到任何内容时清掉占位

    def configure(self, *, routing_preference: Optional[str] = None,
                  show_thinking: Optional[bool] = None) -> None:
        """配置统一 TUI 启动时的请求偏好。"""
        with self._state_lock:
            if routing_preference in {
                "auto", "local_only", "distributed_preferred", "distributed_required",
            }:
                self.routing_preference = routing_preference
            if show_thinking is not None:
                self.show_thinking = bool(show_thinking)

    def resume_session(self, session_id: str) -> str:
        """切换并恢复一个会话，供统一 TUI 命令复用。"""
        if self.streaming:
            return "当前正在生成，请先取消"
        if self.api is None:
            return "聊天屏未连接后端"
        target = str(session_id or "").strip()
        if not target:
            return "会话 ID 不能为空"
        path = API_PATHS["sessions_activate"].format(
            session_id=urllib.parse.quote(target, safe=""),
        )
        try:
            data = self.api.post(path) or {}
            messages = data.get("messages") or []
            restored = []
            for item in messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").lower()
                text = item.get("content", item.get("text", ""))
                if role in {"user", "assistant", "system"} and isinstance(text, str):
                    restored.append((role, text))
            with self._state_lock:
                self.session_id = target
                self.messages = restored
                self.error = None
            return "已恢复会话: %s" % target
        except Exception as exc:
            with self._state_lock:
                self.error = "会话恢复失败: %s" % exc
            return self.error

    def rename_session(self, title: str) -> str:
        """重命名当前会话。"""
        with self._state_lock:
            target = self.session_id
        title = str(title or "").strip()
        if not target:
            return "当前没有可重命名的会话"
        if not title:
            return "标题不能为空"
        if self.api is None:
            return "聊天屏未连接后端"
        try:
            self.api.put(API_PATHS["sessions_detail"].format(
                session_id=urllib.parse.quote(target, safe=""),
            ), {"title": title})
            return "会话已重命名: %s" % title
        except Exception as exc:
            return "重命名会话失败: %s" % exc

    def delete_session(self) -> str:
        """删除当前会话并清理本屏状态。"""
        if self.streaming:
            return "当前正在生成，请先取消"
        with self._state_lock:
            target = self.session_id
        if not target or self.api is None:
            return "当前没有可删除的会话"
        try:
            self.api.delete(API_PATHS["sessions_detail"].format(
                session_id=urllib.parse.quote(target, safe=""),
            ))
            self._act_new_session(None)
            return "已删除会话: %s" % target
        except Exception as exc:
            return "删除会话失败: %s" % exc

    # ------------------------------------------------------------------
    # 兼容旧 API 的流读取实现
    # ------------------------------------------------------------------

    def iter_payloads(self, message: str) -> Iterator[dict]:
        """产出事件的 JSON payload（dict）。默认走 urllib；测试可注入 transport。"""
        body = build_interactive_request(
            message,
            session_id=self.session_id,
            generation_id=self.generation_id,
            routing_preference=self.routing_preference,
            show_thinking=self.show_thinking,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self._transport is not None:
            yield from self._transport(body)
            return

        if self.api is None:
            raise RuntimeError("ChatScreen 未绑定 ApiClient（app.api 缺失）")
        url = self.api.base_url + "/api" + API_PATHS["chat_stream"]
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        decoder = SSEDecoder()
        with urllib.request.urlopen(req, timeout=self.READ_TIMEOUT) as resp:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                for event in decoder.feed(chunk):
                    payload = decode_json_event(event)
                    if payload:
                        yield payload

    def cancel(self, generation_id: str) -> None:
        """请求取消某次生成（失败不致命）。"""
        if self.api is None:
            return
        # 必须用 safe=""：默认 safe="/" 不会编码斜杠，而 generation_id 含 "/" 时
        # 会破坏 URL 路径结构（把路径多切一段）。
        path = API_PATHS["chat_cancel"].format(
            generation_id=urllib.parse.quote(generation_id, safe=""))
        try:
            self.api.post(path)
        except Exception as e:
            self.error = "取消失败: %s" % e

    # ------------------------------------------------------------------
    # 会话辅助（只读，供 P3 的会话页复用）
    # ------------------------------------------------------------------

    def session_lines(self) -> List[str]:
        """把 `/api/sessions` 的返回渲染为行（复用 `parse_session_line`）。"""
        data = self.data or {}
        items = data.get("sessions") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [parse_session_line(s) for s in items if isinstance(s, dict)]


# ======================================================================
# 独立冒烟（不依赖 tui_admin / 不联网）：回放 SSE fixture
# ======================================================================

def smoke_from_fixture(path: str) -> int:
    raw = open(path, "rb").read()
    events = SSEDecoder().feed(raw)
    payloads = [p for p in (decode_json_event(e) for e in events) if p]
    print("  事件数: %d   有效 payload: %d" % (len(events), len(payloads)))

    tokens = [p["token"] for p in payloads if isinstance(p.get("token"), str)]
    text = sanitize("".join(tokens))
    print("  拼接文本长度: %d" % len(text))

    done = next((p for p in payloads if p.get("done")), None)
    if done:
        print("  format_metrics : %s" % format_metrics(
            done.get("metrics"), history_committed=done.get("history_committed")))
        resp = done.get("response") or ""
        print("  done.response 长度: %d" % len(resp))

    cancelled = next((p for p in payloads if p.get("cancelled")), None)
    if cancelled:
        print("  cancelled.partial 长度: %d" % len(cancelled.get("partial") or ""))

    print("  ANSI 残留: %s" % ("有" if _ANSI_RE.search(text) else "无"))
    print("  折行首行: %s" % (wrap_display(text, 40)[:1] or ["(空)"]))
    print("  parse_session_line 可用: %s" % callable(parse_session_line))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="ChatScreen 独立冒烟（离线 fixture 回放）")
    ap.add_argument("--fixture", help="SSE fixture 文件路径")
    args = ap.parse_args(argv)
    if args.fixture:
        return smoke_from_fixture(args.fixture)
    print("用法: python -m src.tui_chat_screen --fixture <file.sse>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
