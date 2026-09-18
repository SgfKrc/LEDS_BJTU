"""Koakuma TUI —— **Textual 外壳**（主仓统一交互入口）。

为什么换掉自绘标准库 TUI（2026-09-17 实测结论）：

* 自绘 ANSI splash 在真实 conhost 下**不可见**——像素字依赖 ``▀``(U+2580) 半块字符 +
  24 位真彩色，默认字体/色深下整块 logo 渲染为空白；且帧序列用 ``\\n`` 换行（VT 模式下
  不回第 0 列）导致逐行错位。这不是一处笔误，而是跨终端自绘的结构性风险。
* Textual 自己处理终端适配（VT 检测、字体无关布局、Rich 渲染、鼠标/滚动/焦点），
  并在 Windows Terminal 与传统 conhost 上都可用。

启动体验（2026-09-17 用户要求）：

* **标题页与启动条合体**：LOGO 下方紧跟一条跑马灯启动条 + 状态行，不再"先纯文本等待、
  再进 TUI"两段式；
* 状态文案统一以 **「少女祈祷中：」** 开头（避免与标题 ``Koakuma`` 重复）；
* 后端冷启动（``BackendSupervisor.ensure_ready``）在启动屏的 worker 线程里执行，
  阶段文本实时反映到启动条下方。

主界面版式（2026-09-17 用户反馈后重做）：

* **导航移到左侧竖栏**（原先顶部 Tab）：形成"左窄右宽"版式，分栏按**黄金比例**
  0.382 : 0.618 分割（``#nav`` = 38%，``#content`` = 62%，即 ``1fr``）；
* 侧栏带 ``min-width``/``max-width`` 兜底，窄终端（80 列）与宽终端都不会失衡；
* **每页统一排版规范**：``.page-title``（页名）→ ``.page-hint``（一句话说明 + 数据源）
  → 内容面板（表格/日志/键值），错误与空态样式一致，不再"裸放一个表"。

边界：

* Textual 是**主仓 TUI 的依赖**（``requirements-tui.txt``；Edge 同样安装，见
  ``requirements-edge.txt``）；协议层 ``src/tui_api.py`` 仍是纯标准库，
  因此**非 UI 路径**（单命令、CI）不需要装 Textual。

用法（由 ``qlh`` 调用）::

    python src/tui_textual.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tui_api import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    ApiClient,
    ApiError,
    activate_session,
    cancel_generation,
    clear_backend_history,
    clear_queue,
    create_session,
    delete_session,
    iter_chat_payloads,
    list_sessions,
    load_model,
    pause_queue,
    rename_session,
    resume_queue,
    set_queue_strategy,
    unload_model,
)
from tui_shared import API_PATHS, COMMAND_SPECS, format_metrics  # noqa: E402

LOGO = (
    "  ██╗  ██╗ ██████╗  █████╗ ██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ \n"
    "  ██║ ██╔╝██╔═══██╗██╔══██╗██║ ██╔╝██║   ██║████╗ ████║██╔══██╗\n"
    "  █████╔╝ ██║   ██║███████║█████╔╝ ██║   ██║██╔████╔██║███████║\n"
    "  ██╔═██╗ ██║   ██║██╔══██║██╔═██╗ ██║   ██║██║╚██╔╝██║██╔══██║\n"
    "  ██║  ██╗╚██████╔╝██║  ██║██║  ██╗╚██████╔╝██║ ╚═╝ ██║██║  ██║\n"
    "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝\n"
)

#: 启动行前缀（用户 2026-09-17 指定：不用 "Koakuma:"，避免与标题重复）
SPLASH_PREFIX = "少女祈祷中："
BAR_WIDTH = 30
BAR_MARQUEE = 9
BAR_BLOCK = "█"
BAR_EMPTY = "░"

#: 侧栏导航页表：(key, 页名, 一句话说明)
PAGES: List[Tuple[str, str, str]] = [
    ("chat", "聊天", "SSE 流式对话 · /help 查看命令"),
    ("status", "状态", "运行概览 · /health /status /models"),
    ("models", "模型", "注册表 · L 加载光标行 · U 卸载当前 · /models"),
    ("cluster", "分布式", "集群资源合计 · /cluster/resources"),
    ("nodes", "节点", "成员与角色 · /cluster/nodes"),
    ("queue", "队列", "MLFQ 三级 · P 暂停/恢复 · S 策略 · C 清空排队"),
    ("logs", "日志", "聚合日志（末尾 200 行）· /cluster/nodes/log-aggregate"),
    ("device", "设备", "本机设备画像与 GPU · /device/profile"),
    ("settings", "设置", "会话参数与依赖边界"),
]

CSS = """
Screen { background: $surface; }

/* ---------------------------------------------------------------- 启动屏 */
#splash-logo { color: #8fa8c4; text-align: center; padding: 1 0 0 0; }
#splash-bar { color: #6b8aa8; text-align: center; padding: 1 0 0 0; }
#splash-status { color: $text; text-align: center; padding: 0 0 1 0; }
#splash-hint { text-align: center; color: $text-disabled; }

/* ------------------------------------------------- 主界面骨架（左窄右宽） */
#topbar { height: 1; padding: 0 1; color: $text-muted; background: $panel; }
#body { height: 1fr; }

/* 黄金分割：38% : 62%（≈0.382 : 0.618），并给窄/宽终端兜底 */
#sidebar {
    width: 38%;          /* 黄金分割窄侧（0.382 : 0.618） */
    min-width: 18;       /* 窄终端兜底：侧栏不被压到不可读 */
    height: 1fr;
    background: $panel;
    border-right: solid $primary 25%;
}
#nav { height: 1fr; padding: 1 0; }
#nav-summary {
    height: auto;
    padding: 0 1 1 1;
    border-top: solid $primary 20%;
    color: $text-muted;
}
#nav ListItem { padding: 0 1; }
#nav ListItem Label { width: 1fr; }
#nav > ListItem.--highlight { background: $primary 35%; text-style: bold; }

#content { width: 1fr; height: 1fr; }

/* ------------------------------------------------- 每页统一排版规范 */
.page { height: 1fr; padding: 1 2; }
.page-title { height: 1; color: $accent; text-style: bold; }
.page-hint { height: 1; color: $text-muted; margin-bottom: 1; }
.panel { height: 1fr; }
.scroll-panel { height: 1fr; }
.empty { color: $text-disabled; }
.error { color: $error; }

/* ---------------------------------------------------------------- 内容 */
#models-table, #resources-table, #nodes-table, #queue-table,
#status-table, #logs-log { height: 1fr; }
#status-pane, #queue-pane, #logs-pane { height: auto; color: $text-muted; }
#device-pane, #settings-pane { height: auto; }
#gpu-table { height: auto; max-height: 14; }

/* ---------------------------------------------------------------- 聊天 */
/* 对话文本用 Static + 缓冲渲染：Textual 8 的 RichLog.write() 不支持 end=，
   逐 token 流式写入会抛 TypeError —— 表现为"聊天得不到回复"。 */
#chat-scroll { height: 1fr; border: round $primary 20%; padding: 0 1; }
#chat-log { height: auto; }
#chat-status { padding: 0 1; height: auto; min-height: 1; color: $text-muted; }
#chat-input { dock: bottom; }
"""


def _fmt_bytes(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def kv(label: str, value: Any) -> str:
    """键值行：标签定宽 12 列，值缺省显示 em dash（各页排版统一）。"""
    text = "—" if value in (None, "") else str(value)
    return f"  {label:<12} {text}"


def _join(value: Any, sep: str = ", ") -> str:
    if isinstance(value, (list, tuple)):
        return sep.join(str(item) for item in value) or "—"
    return "—" if value in (None, "") else str(value)


def _log_lines(source: Any, node_id: str) -> List[str]:
    """把 ``{node_id, logs: [...]}`` 形状的日志段摊平成 ``[node] 行``。"""
    if not isinstance(source, dict):
        return []
    name = source.get("node_id") or node_id
    lines = source.get("logs") or source.get("lines") or []
    if isinstance(lines, str):
        lines = lines.splitlines()
    return [f"[{name}] {line}" for line in lines]


class SplashScreen(Screen):
    """启动屏：LOGO + **启动条** + 状态行（三段一体，见模块头注释）。

    ``wait_for_backend=True`` 时**不自动进入**主界面——由 ``KoakumaApp`` 在后端就绪后
    调用 ``show_main()``；否则短暂展示后自动进入。
    """

    BINDINGS = [Binding("escape,enter,space", "finish", "进入", show=False)]

    def __init__(self, *, status: str = "准备启动", wait_for_backend: bool = False) -> None:
        super().__init__()
        self.splash_status = status
        self.wait_for_backend = bool(wait_for_backend)
        self.bar_pos = 0

    # ------------------------------------------------------------ 渲染

    def compose(self) -> ComposeResult:
        yield Static(LOGO, id="splash-logo")
        yield Static(self.bar_text(), id="splash-bar")
        yield Static(self.status_line(), id="splash-status")
        yield Static("q / ctrl+c 退出 · 任意键进入", id="splash-hint")

    def on_mount(self) -> None:
        self.set_interval(0.09, self.tick_bar)
        if not self.wait_for_backend:
            self.set_timer(0.6, self.action_finish)

    def bar_text(self) -> str:
        cells = [BAR_EMPTY] * BAR_WIDTH
        for offset in range(BAR_MARQUEE):
            cells[(self.bar_pos + offset) % BAR_WIDTH] = BAR_BLOCK
        return "".join(cells)

    def status_line(self) -> str:
        return f"{SPLASH_PREFIX}{self.splash_status}"

    def tick_bar(self) -> None:
        self.bar_pos = (self.bar_pos + 1) % BAR_WIDTH
        try:
            self.query_one("#splash-bar", Static).update(self.bar_text())
        except Exception:  # noqa: BLE001 - 启动屏可能已被替换
            pass

    def set_status(self, text: str) -> None:
        """更新启动条下方的阶段文本（由后端启动 worker 的进度驱动）。"""
        self.splash_status = text
        try:
            self.query_one("#splash-status", Static).update(self.status_line())
        except Exception:  # noqa: BLE001
            pass

    def action_finish(self) -> None:
        self.app.show_main()


class ConfirmScreen(ModalScreen[bool]):
    """写操作前的模态确认。

    项目偏好：**涉及删除/破坏的操作必须先列出将影响的内容**（dry-run 精神），
    所以 ``body`` 必须写清"将要删/改什么"，不能只问一句"确定吗"。
    """

    BINDINGS = [
        Binding("y,enter", "confirm", "确认", show=False),
        Binding("n,escape", "cancel", "取消", show=False),
    ]

    CSS = """
    ConfirmScreen { align: center middle; }
    #confirm-box {
        width: 70; max-width: 92%; height: auto;
        border: thick $warning; background: $surface; padding: 1 2;
    }
    #confirm-title { height: auto; text-style: bold; color: $warning; }
    #confirm-body { height: auto; margin: 1 0; }
    #confirm-keys { height: auto; color: $text-muted; }
    """

    def __init__(self, title: str, body: str, *, confirm_label: str = "确认") -> None:
        super().__init__()
        self.confirm_title = title
        self.confirm_body = body
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self.confirm_title, id="confirm-title")
            yield Static(self.confirm_body, id="confirm-body")
            yield Static(f"[b]y[/] / [b]{self.confirm_label}[/] 执行    [b]n[/] / Esc 取消",
                         id="confirm-keys")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ChatPane(Vertical):
    """聊天页：SSE 流式输出（等价旧 ChatScreen 的事件处理）。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chat_buffer = ""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll"):
            yield Static("", id="chat-log", markup=True)
        yield Static("就绪。输入消息并回车发送；/help 查看命令", id="chat-status")
        yield Input(placeholder="输入消息…（/help 查看命令）", id="chat-input")

    # ------------------------------------------------------------ 缓冲与渲染

    def write_line(self, text: str) -> None:
        """追加一整行（命令输出 / 用户输入）。"""
        self.chat_buffer += ("\n" if self.chat_buffer else "") + text
        self.render_chat()

    def append_chunk(self, piece: str) -> None:
        """流式追加 token（不换行）；整体重渲染由 Static 承担。"""
        self.chat_buffer += piece
        self.render_chat()

    def replace_transcript(self, text: str) -> None:
        self.chat_buffer = text
        self.render_chat()

    def clear_chat(self) -> None:
        self.chat_buffer = ""
        self.render_chat()

    def render_chat(self) -> None:
        self.query_one("#chat-log", Static).update(self.chat_buffer)
        try:
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
        except Exception:  # noqa: BLE001 - 挂载早期可能尚无滚动容器
            pass

    def set_status(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text or "完成")

    # ------------------------------------------------------------ 发送

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        if not text:
            return
        event.input.value = ""
        if text in {"/help", "/?"}:
            self.write_line(self.help_markup())
            return
        if text == "/quit":
            self.app.exit()
            return
        if text == "/clear":
            self.clear_chat()
            return
        if text.startswith("/route "):
            value = text.split(" ", 1)[1].strip()
            mapping = {"auto": "auto", "local": "local_only",
                       "distributed": "distributed_preferred", "required": "distributed_required"}
            if value in mapping:
                self.app.routing_preference = mapping[value]
                self.query_one("#chat-status", Static).update(f"路由偏好 → {mapping[value]}")
            else:
                self.query_one("#chat-status", Static).update(
                    "[red]用法: /route auto|local|distributed|required")
            return
        if text.startswith("/thinking "):
            value = text.split(" ", 1)[1].strip().lower()
            self.app.show_thinking = value in {"on", "1", "true", "yes"}
            self.query_one("#chat-status", Static).update(
                f"thinking → {'on' if self.app.show_thinking else 'off'}")
            return
        if text == "/cancel":
            generation = getattr(self.app, "generation_id", None)
            if generation:
                try:
                    cancel_generation(self.app.api, generation)
                    self.query_one("#chat-status", Static).update("已请求取消")
                except Exception as exc:  # noqa: BLE001
                    self.query_one("#chat-status", Static).update(f"[red]取消失败: {exc}")
            else:
                self.query_one("#chat-status", Static).update("当前没有正在生成的请求")
            return
        if text.startswith("/model"):
            self.cmd_model(text)
            return
        if text.startswith("/queue"):
            self.cmd_queue(text)
            return
        if text == "/sessions":
            self.cmd_sessions()
            return
        if text.startswith("/new"):
            self.cmd_new(text)
            return
        if text.startswith("/resume"):
            self.cmd_resume(text)
            return
        if text.startswith("/rename"):
            self.cmd_rename(text)
            return
        if text == "/delete-session":
            self.cmd_delete_session()
            return
        if text == "/reset":
            self.cmd_reset()
            return
        if text.startswith("/"):
            self.query_one("#chat-status", Static).update(f"[red]未知命令: {text}")
            return

        self.write_line(f"[b $accent]你[/] {text}")
        self.write_line("[dim]assistant[/] ")
        self.stream_reply(text)

    # ------------------------------------------------------------ 命令（模型/队列/会话）

    def help_markup(self) -> str:
        """由 ``COMMAND_SPECS`` 生成帮助（与实现同源，不会"写着却不能用"）。"""
        lines = ["[b]可用命令[/]"]
        for spec in COMMAND_SPECS:
            usage = spec["name"] + (f" {spec['args']}" if spec["args"] else "")
            lines.append(f"  [b]{usage}[/]  {spec['desc']}")
        lines.append("  [dim]写操作都会先弹确认框；模型/队列也可在对应屏用按键操作[/]")
        return "\n".join(lines)

    def status_line(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text)

    def cmd_model(self, text: str) -> None:
        parts = text.split()
        usage = "用法: /model load <model_id> [engine] [quant] | /model unload"
        if len(parts) >= 2 and parts[1].lower() == "unload":
            self.app.confirm(
                "卸载模型",
                "将释放后端当前本地模型：正在生成/排队的请求会失败，"
                "对话上下文与 KV 缓存一并清空。",
                lambda: self.model_call("unload", "", "", ""),
                confirm_label="卸载")
            return
        if len(parts) < 3 or parts[1].lower() != "load":
            self.status_line(usage)
            return
        model_id = parts[2]
        engine = parts[3] if len(parts) > 3 else "llama_cpp"
        quant = parts[4] if len(parts) > 4 else "int4"
        self.app.confirm(
            "加载模型",
            f"将要加载：[b]{model_id}[/]\n引擎 [b]{engine}[/] · 量化 [b]{quant}[/]\n"
            "耗时约 5-20 秒；期间会先卸载当前模型，失败由后端自动回滚。",
            lambda: self.model_call("load", model_id, engine, quant),
            confirm_label="加载")

    @work(thread=True, exclusive=True, group="modelctl")
    def model_call(self, kind: str, model_id: str, engine: str, quant: str) -> None:
        app = self.app
        label = "加载" if kind == "load" else "卸载"
        self.app.call_from_thread(self.status_line, f"正在{label}模型（5-20 秒）…")
        try:
            if kind == "load":
                load_model(app.api, model_id, engine=engine, quant_type=quant)
                text = f"[green]模型已加载[/] {model_id}"
            else:
                unload_model(app.api)
                text = "[green]模型已卸载[/]"
        except ApiError as exc:
            text = f"[red]模型{label}失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text)

    def cmd_queue(self, text: str) -> None:
        parts = text.split()
        usage = "用法: /queue pause | resume | strategy <fifo|mlfq> | clear"
        if len(parts) == 2 and parts[1].lower() in {"pause", "resume"}:
            self.queue_call(parts[1].lower())
            return
        if len(parts) == 3 and parts[1].lower() == "strategy" \
                and parts[2].lower() in {"fifo", "mlfq"}:
            self.queue_call("strategy", parts[2].lower())
            return
        if len(parts) == 2 and parts[1].lower() == "clear":
            self.app.confirm(
                "清空排队任务",
                "将清空后端**排队中**的任务（执行中的不受影响）。此操作不可撤销。",
                lambda: self.queue_call("clear"),
                confirm_label="清空")
            return
        self.status_line(usage)

    @work(thread=True, exclusive=True, group="queuectl")
    def queue_call(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.status_line, f"队列操作 {action} …")
        try:
            if action == "pause":
                pause_queue(app.api)
            elif action == "resume":
                resume_queue(app.api)
            elif action == "strategy":
                set_queue_strategy(app.api, value)
            elif action == "clear":
                clear_queue(app.api)
            else:
                raise ValueError(f"未知队列动作: {action}")
            text = f"[green]队列 {action} 完成[/]" + (f" → {value}" if value else "")
        except (ApiError, ValueError) as exc:
            text = f"[red]队列 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text)

    def cmd_sessions(self) -> None:
        self.session_call("list", "")

    def cmd_new(self, text: str) -> None:
        self.session_call("new", text[len("/new"):].strip())

    def cmd_resume(self, text: str) -> None:
        session_id = text[len("/resume"):].strip()
        if not session_id:
            self.status_line("用法: /resume <session_id>（先 /sessions 查看）")
            return
        self.session_call("resume", session_id)

    def cmd_rename(self, text: str) -> None:
        title = text[len("/rename"):].strip()
        if not title:
            self.status_line("用法: /rename <新标题>")
            return
        self.session_call("rename", title)

    def cmd_delete_session(self) -> None:
        session_id = getattr(self.app, "session_id", None)
        if not session_id:
            self.status_line("当前没有会话（先 /new 或 /resume）")
            return
        self.app.confirm(
            "删除会话",
            f"将删除会话 [b]{session_id}[/] **及其全部对话消息**（数据库 + 内存）。\n"
            "此操作不可撤销。",
            lambda: self.session_call("delete", session_id),
            confirm_label="删除")

    def cmd_reset(self) -> None:
        self.app.confirm(
            "清空后端会话历史",
            "将清空后端当前会话的对话历史与 KV 缓存（本地显示一并清空）。\n"
            "会话本身保留；此操作不可撤销。",
            lambda: self.session_call("reset", ""),
            confirm_label="清空")

    def render_history(self, messages: Any) -> str:
        """把后端会话历史渲染为对话文本（兼容 role/sender 与 content/message 两种命名）。"""
        if not isinstance(messages, list):
            return "[dim]（该会话没有历史消息）[/]"
        lines = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("sender") or "?")
            label = {"user": "[b $accent]你[/]",
                     "assistant": "[dim]assistant[/]"}.get(role, role)
            content = message.get("content") or message.get("message") or ""
            lines.append(f"{label} {content}")
        return "\n".join(lines) or "[dim]（该会话没有历史消息）[/]"

    @work(thread=True, exclusive=True, group="sessions")
    def session_call(self, action: str, value: str) -> None:
        """会话管理（新建/恢复/重命名/删除/清空后端历史）——全部在 worker 线程里跑。"""
        app = self.app
        clear_local = False
        self.app.call_from_thread(self.status_line, f"会话操作 {action} …")
        try:
            if action == "list":
                items = list_sessions(app.api)
                if not items:
                    self.app.call_from_thread(
                        self.status_line, "（后端没有会话；用 /new 创建一个）")
                    return
                lines = ["[b]最近会话[/]（用 /resume <id> 恢复）"]
                for item in items[:10]:
                    sid = item.get("session_id") or item.get("id") or "—"
                    title = item.get("title") or "(未命名)"
                    count = item.get("message_count")
                    lines.append(f"  [b]{sid}[/]  {title}"
                                 + (f" · {count} 条" if count is not None else ""))
                self.app.call_from_thread(self.write_line, "\n".join(lines))
                self.app.call_from_thread(self.status_line, f"共 {len(items)} 个会话")
                return
            if action == "new":
                result = create_session(app.api, value or None)
                session_id = str(result.get("session_id") or result.get("id") or "") or None
                app.session_id = session_id
                text = f"[green]已新建会话[/] {session_id or ''}".strip()
                clear_local = True
            elif action == "resume":
                result = activate_session(app.api, value)
                app.session_id = value
                messages = result.get("messages") or result.get("history") or []
                self.app.call_from_thread(self.replace_transcript, self.render_history(messages))
                text = f"[green]已恢复会话[/] {value}（{len(messages)} 条消息）"
            elif action == "rename":
                if not app.session_id:
                    raise ValueError("当前没有会话")
                rename_session(app.api, app.session_id, value)
                text = f"[green]已重命名[/] → {value}"
            elif action == "delete":
                delete_session(app.api, value)
                if app.session_id == value:
                    app.session_id = None
                text = f"[green]已删除会话[/] {value}"
                clear_local = True
            elif action == "reset":
                clear_backend_history(app.api)
                text = "[green]后端会话历史已清空[/]"
                clear_local = True
            else:
                raise ValueError(f"未知会话动作: {action}")
        except (ApiError, ValueError) as exc:
            text = f"[red]会话 {action} 失败[/]：{exc}"
        if clear_local:
            self.app.call_from_thread(self.clear_chat)
        self.app.call_from_thread(self.write_line, text)
        self.app.call_from_thread(self.status_line, text)

    @work(thread=True, exclusive=True, group="chat")
    def stream_reply(self, message: str) -> None:
        """在后台线程消费 SSE；所有 UI 更新都回到主线程执行。"""
        app = self.app
        acc: List[str] = []
        status = "…"
        error_text = ""
        try:
            for payload in iter_chat_payloads(
                app.api,
                message,
                session_id=getattr(app, "session_id", None),
                generation_id=getattr(app, "generation_id", None),
                routing_preference=app.routing_preference,
                show_thinking=app.show_thinking,
            ):
                if payload.get("start"):
                    if payload.get("generation_id"):
                        app.generation_id = payload["generation_id"]
                    if payload.get("session_id"):
                        app.session_id = payload["session_id"]
                elif isinstance(payload.get("token"), str) and payload["token"]:
                    acc.append(payload["token"])
                    self.app.call_from_thread(self.append_chunk, payload["token"])
                elif isinstance(payload.get("thinking"), str) and payload["thinking"]:
                    if app.show_thinking:
                        self.app.call_from_thread(
                            self.append_chunk, f"[dim italic]{payload['thinking']}[/]")
                elif payload.get("done"):
                    # 后端把失败放在 done.error（例如"本地回退模型加载失败"）；
                    # 早先只看 response/metrics，会把错误吞掉，界面表现为"没回复也没原因"。
                    if payload.get("error"):
                        error_text = str(payload["error"])
                    final = payload.get("response")
                    if isinstance(final, str) and final and "".join(acc) != final:
                        self.app.call_from_thread(self.replace_transcript, final)
                    if payload.get("session_id"):
                        app.session_id = payload["session_id"]
                    if not error_text:
                        status = format_metrics(
                            payload.get("metrics") or None,
                            history_committed=payload.get("history_committed"),
                        )
                elif payload.get("cancelled"):
                    status = "已被取消"
                elif payload.get("error"):
                    error_text = str(payload["error"])
        except Exception as exc:  # noqa: BLE001 - 网络异常不应让界面崩溃
            error_text = f"{type(exc).__name__}: {exc}"
        finally:
            app.generation_id = None
            if error_text:
                first = error_text.strip().splitlines()[0]
                status = f"[red]后端错误: {first}[/]"
                self.app.call_from_thread(self.append_chunk, f"[red]✗ {error_text}[/]")
            self.app.call_from_thread(self.set_status, status)

    # ------------------------------------------------------------ UI 更新（主线程）
    # write_line / append_chunk / replace_transcript / set_status 见上方"缓冲与渲染"段。


class MainScreen(Screen):
    """主界面：左侧导航（黄金比例分栏）+ 右侧内容区（9 屏）。"""

    BINDINGS = [
        Binding("r", "reload", "刷新"),
        Binding("l", "load_model", "加载模型"),
        Binding("u", "unload_model", "卸载模型"),
        Binding("p", "queue_toggle_pause", "暂停/恢复"),
        Binding("s", "queue_cycle_strategy", "调度策略"),
        Binding("c", "queue_clear", "清空排队"),
        Binding("]", "next_page", "下一屏"),
        Binding("[", "prev_page", "上一屏"),
        Binding("q", "quit_app", "退出"),
        Binding("ctrl+c", "quit_app", "退出", show=False),
    ]

    #: 每屏可用键提示（让"这屏能做什么"可见；与 BINDINGS 保持一致）
    PAGE_KEYS = {
        "chat": "/help 看命令",
        "models": "L 加载 · U 卸载",
        "queue": "P 暂停/恢复 · S 策略 · C 清空排队",
    }

    def __init__(self) -> None:
        super().__init__()
        self.page_index = 0
        self.health_text = "…"
        self.model_text = "…"
        self.log_line_count = 0
        #: ``/models`` 原始项（模型屏写操作需要 engine/quant/可用性）
        self.model_rows: Dict[str, Dict[str, Any]] = {}
        #: ``/cluster/queue`` 最近一次结果（队列屏写操作需要 paused/strategy/深度）
        self.queue_state: Dict[str, Any] = {}
        #: 最近一次写操作的结果文本（操作没有回显窗口，故常驻侧栏摘要）
        self.op_status = ""
        self.cuda_available: Optional[bool] = None

    # ------------------------------------------------------------ 版式

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="topbar")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                with ListView(id="nav"):
                    for key, label, _hint in PAGES:
                        yield ListItem(Label(f" {label}"), id=f"nav-{key}")
                yield Static(id="nav-summary")
            with ContentSwitcher(initial=f"page-{PAGES[0][0]}", id="content"):
                with Vertical(id="page-chat", classes="page"):
                    yield Static("聊天", classes="page-title")
                    yield Static(PAGES[0][2], classes="page-hint")
                    yield ChatPane(id="chat-pane")
                with Vertical(id="page-status", classes="page"):
                    yield Static("状态 · 运行概览", classes="page-title")
                    yield Static(PAGES[1][2], classes="page-hint")
                    yield Static("加载中…", id="status-pane")
                    yield DataTable(id="status-table")
                with Vertical(id="page-models", classes="page"):
                    yield Static("模型 · 注册表与当前加载", classes="page-title")
                    yield Static(PAGES[2][2], classes="page-hint")
                    yield DataTable(id="models-table")
                with Vertical(id="page-cluster", classes="page"):
                    yield Static("分布式 · 集群资源", classes="page-title")
                    yield Static(PAGES[3][2], classes="page-hint")
                    yield DataTable(id="resources-table")
                with Vertical(id="page-nodes", classes="page"):
                    yield Static("节点 · 成员与角色", classes="page-title")
                    yield Static(PAGES[4][2], classes="page-hint")
                    yield DataTable(id="nodes-table")
                with Vertical(id="page-queue", classes="page"):
                    yield Static("队列 · MLFQ 三级调度", classes="page-title")
                    yield Static(PAGES[5][2], classes="page-hint")
                    yield Static("加载中…", id="queue-pane")
                    yield DataTable(id="queue-table")
                with Vertical(id="page-logs", classes="page"):
                    yield Static("日志 · 聚合视图", classes="page-title")
                    yield Static(PAGES[6][2], classes="page-hint")
                    yield Static("加载中…", id="logs-pane")
                    yield RichLog(id="logs-log", markup=False, wrap=True)
                with Vertical(id="page-device", classes="page"):
                    yield Static("设备 · 本机画像", classes="page-title")
                    yield Static(PAGES[7][2], classes="page-hint")
                    with VerticalScroll(classes="scroll-panel"):
                        yield Static("加载中…", id="device-pane")
                        yield DataTable(id="gpu-table")
                with Vertical(id="page-settings", classes="page"):
                    yield Static("设置 · 会话与边界", classes="page-title")
                    yield Static(PAGES[8][2], classes="page-hint")
                    with VerticalScroll(classes="scroll-panel"):
                        yield Static("", id="settings-pane")
        yield Footer()

    # ------------------------------------------------------------ 切屏

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """侧栏高亮即切屏（↑↓ 浏览，无需回车）。"""
        if event.item is not None and event.item.id:
            self.switch_page(event.item.id.removeprefix("nav-"))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """鼠标点击/回车同样切屏。"""
        if event.item is not None and event.item.id:
            self.switch_page(event.item.id.removeprefix("nav-"))

    def switch_page(self, key: str) -> None:
        keys = [item[0] for item in PAGES]
        if key not in keys:
            return
        self.page_index = keys.index(key)
        try:
            self.query_one("#content", ContentSwitcher).current = f"page-{key}"
        except Exception:  # noqa: BLE001 - 切屏期间节点可能未挂载
            pass

    def action_next_page(self) -> None:
        self.set_page((self.page_index + 1) % len(PAGES))

    def action_prev_page(self) -> None:
        self.set_page((self.page_index - 1) % len(PAGES))

    def set_page(self, index: int) -> None:
        """按序号切屏，并同步侧栏高亮（键位与鼠标两条路径保持一致）。"""
        key = PAGES[index % len(PAGES)][0]
        self.switch_page(key)
        try:
            self.query_one("#nav", ListView).index = index
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ 文本

    def about_text(self) -> str:
        return (
            "[b $accent]Koakuma[/] · QLH 分布式边缘推理 · Textual 外壳\n"
            + kv("后端地址", self.app.api.base_url)
            + "\n"
            + kv("界面", "9 屏（左栏切换 / [ ] 上下屏 / r 刷新 / q 退出）")
            + "\n"
            + kv("依赖边界", "外壳需 Textual（requirements-tui.txt，Edge 同装）；")
            + "\n"
            + kv("", "协议层 src/tui_api.py 纯标准库，单命令/CI 不依赖 UI")
        )

    def settings_text(self) -> str:
        """设置页：本地会话参数（旧管理 TUI 的「设置」语义，多为启动参数固化）。"""
        api = self.app.api
        return (
            "[b $accent]会话设置[/][dim]（当前会话生效；可用启动参数固化）[/]\n"
            + kv("host", f"{api.host}（支持 Tailscale IP，如 100.x.x.x）") + "\n"
            + kv("port", api.port) + "\n"
            + kv("完整地址", api.base_url) + "\n"
            + kv("请求超时", f"{api.timeout:.0f} 秒") + "\n"
            + kv("自动刷新", f"{self.app.interval:.0f} 秒（状态/模型/分布式/节点/队列/设备）") + "\n"
            + kv("日志 Token", "已设置" if api.log_token else "未设置（聚合日志可能需要 --log-token）") + "\n"
            + kv("路由偏好", self.app.routing_preference) + "\n"
            + kv("thinking", "on" if self.app.show_thinking else "off") + "\n"
            + "\n" + self.about_text()
        )

    # ------------------------------------------------------------ 挂载

    def on_mount(self) -> None:
        self.query_one("#status-table", DataTable).add_columns("项目", "值")
        self.query_one("#models-table", DataTable).add_columns(
            "", "模型 ID", "名称", "格式", "引擎", "状态")
        self.query_one("#resources-table", DataTable).add_columns(
            "节点", "角色 / 状态", "CPU 物理/逻辑", "内存 总/可用", "GPU 数/CUDA", "显存 总/可用")
        self.query_one("#nodes-table", DataTable).add_columns(
            "节点", "角色", "类型", "状态", "地址", "RTT")
        self.query_one("#queue-table", DataTable).add_columns("队列", "深度", "上限(tokens)", "任务")
        self.query_one("#gpu-table", DataTable).add_columns(
            "", "名称", "类型", "CUDA", "显存GB", "驱动")
        self.query_one("#settings-pane", Static).update(self.settings_text())
        self.refresh_topbar()
        self.action_reload()
        self.load_pages()
        self.load_device()
        self.set_interval(self.app.interval, self.action_reload)
        self.set_interval(self.app.interval, self.load_pages)
        self.set_interval(self.app.interval, self.refresh_topbar)

    def refresh_topbar(self) -> None:
        page = PAGES[self.page_index]
        keys = self.PAGE_KEYS.get(page[0])
        hint = f"   [dim]·[/]   [dim]{keys}[/]" if keys else ""
        self.query_one("#topbar", Static).update(
            f"[dim]后端[/] {self.app.api.base_url}"
            f"   [dim]·[/]   [dim]当前[/] {page[1]}{hint}"
            f"   [dim]·[/]   [dim]r 刷新 · [ ] 上下屏 · q 退出[/]")
        self.update_sidebar(page)

    def update_sidebar(self, page: Tuple[str, str, str]) -> None:
        """左栏底部摘要：让黄金分割的窄侧不只是空导航（宽终端尤其明显）。"""
        api = self.app.api
        try:
            self.query_one("#nav-summary", Static).update(
                "[b $accent]后端[/]\n"
                f"{api.host}:{api.port}\n"
                f"[dim]健康[/] {self.health_text}\n"
                f"[dim]模型[/] {self.model_text}\n"
                f"[dim]刷新[/] {self.app.interval:.0f}s   [dim]当前[/] {page[1]}"
                + (f"\n[dim]最近[/] {self.op_status}" if self.op_status else ""))
        except Exception:  # noqa: BLE001 - 挂载期间可能尚未就绪
            pass

    # ------------------------------------------------------------ 只读数据

    @work(thread=True, exclusive=True, group="main")
    def action_reload(self) -> None:
        # 端点/字段以 2026-09-17 实测的后端真实结构为准：
        #   /status → model_name / model_loaded / active_model_id / engine / run_mode / node_*
        #   /models → {models: [...], active_model_id}（19 个内置模型）
        health = self.fetch_json(API_PATHS["health"])
        status = self.fetch_json(API_PATHS["system_status"])
        registry = self.fetch_json(API_PATHS["models_list"])
        resources = self.fetch_json(API_PATHS["cluster_resources"])
        self.app.call_from_thread(self.apply_data, health, status, registry, resources)

    def fetch_json(self, path: str) -> Dict[str, Any]:
        try:
            value = self.app.api.get(path if path.startswith("/") else "/" + path)
            return value if isinstance(value, dict) else {"value": value}
        except ApiError as exc:
            return {"_error": str(exc)}

    def apply_data(self, health: Dict[str, Any], status: Dict[str, Any],
                   registry: Dict[str, Any], resources: Dict[str, Any]) -> None:
        self.fill_status(health, status, registry)
        self.fill_models(registry, status)
        self.fill_resources(resources)
        self.refresh_topbar()

    # ------------------------------------------------------------ 状态页

    def fill_status(self, health: Dict[str, Any], status: Dict[str, Any],
                    registry: Dict[str, Any]) -> None:
        pane = self.query_one("#status-pane", Static)
        table = self.query_one("#status-table", DataTable)
        table.clear()
        if "_error" in health:
            self.health_text = "[red]不可达[/]"
            self.model_text = "—"
            pane.update(f"[red]后端不可达[/]  ·  {health['_error']}")
            table.add_row("后端地址", self.app.api.base_url)
            table.add_row("连接状态", "[red]不可达[/]")
            table.add_row("提示", "确认后端在运行（qlh 会自动拉起本机后端）")
            return

        self.health_text = "[green]ok[/]"
        loaded = bool(status.get("model_loaded"))
        model_name = status.get("model_name") or "—"
        active = status.get("active_model_id") or registry.get("active_model_id")
        self.model_text = str(active) if active else ("[yellow]未加载[/]" if not loaded else "—")
        if loaded:
            pane.update(f"[green]后端可用[/]  ·  模型已加载  ·  {self.app.api.base_url}")
        else:
            pane.update(f"[green]后端可用[/]  ·  [yellow]模型未加载[/]（此时聊天会返回后端错误）"
                        f"  ·  {self.app.api.base_url}")
        table.add_row("后端地址", self.app.api.base_url)
        table.add_row("健康", str(health.get("status") or health.get("ok") or "ok"))
        table.add_row("运行模式", str(status.get("run_mode") or "—"))
        table.add_row("节点角色", f"{status.get('node_role') or '—'} · {status.get('node_id') or '—'}")
        table.add_row("最大节点数", str(status.get("max_nodes", "—")))
        table.add_row("模型", f"{model_name}（{'已加载' if loaded else '未加载'}）")
        table.add_row("当前模型 ID", str(active or "—（未加载）"))
        table.add_row("引擎", str(status.get("engine") or "—（未加载）"))
        table.add_row("量化", str(status.get("current_quant") or "—"))
        table.add_row("流水线", "已准备" if status.get("pipeline_prepared") else "未准备")
        table.add_row("对话轮次", str(status.get("conversation_turns", "—")))
        table.add_row("可用模型数", str(len(registry.get("models") or []))
                      if "_error" not in registry else "—")
        table.add_row("路由偏好", self.app.routing_preference)
        table.add_row("刷新间隔", f"{self.app.interval:.0f} 秒")

    # ------------------------------------------------------------ 模型页

    def fill_models(self, registry: Dict[str, Any], status: Dict[str, Any]) -> None:
        """``/models`` → ``{models: [...], active_model_id}``（19 个内置模型）。

        此前误用 ``/models/current``——它只返回 ``{loaded, quant_type, model_id}``
        且未加载时 ``model_id`` 为 null，于是页面显示"后端未返回模型列表"。
        """
        table = self.query_one("#models-table", DataTable)
        table.clear()
        self.model_rows = {}
        if "_error" in registry:
            table.add_row("", "[red]后端不可用[/]", "", "", "", registry["_error"])
            return
        rows = registry.get("models")
        if not isinstance(rows, list) or not rows:
            table.add_row("", "[dim]（后端未返回模型列表）[/]", "", "", "", "")
            return
        active = registry.get("active_model_id") or status.get("active_model_id")
        for item in rows[:64]:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id") or "—")
            if item.get("is_available") is False:
                state = f"[yellow]不可用[/] {item.get('unavailable_reason') or ''}".strip()
            else:
                state = "[green]可用[/]"
            self.model_rows[model_id] = item
            table.add_row(
                "[green]◆[/]" if model_id == active else "",
                model_id,
                str(item.get("name") or "—"),
                _join(item.get("available_formats")),
                str(item.get("preferred_engine") or "—"),
                state,
                key=model_id,  # 写操作按 row key 取模型，不依赖行序
            )

    # ------------------------------------------------------------ 分布式页

    def fill_resources(self, resources: Dict[str, Any]) -> None:
        """``/cluster/resources`` → ``{available: {local, remote}, totals}``。

        此前按顶层 ``nodes`` 解析——真实返回里没有该键，于是页面显示"无在线节点"。
        """
        table = self.query_one("#resources-table", DataTable)
        table.clear()
        if "_error" in resources:
            table.add_row("[red]不可用[/]", resources["_error"], "", "", "", "")
            return
        available = resources.get("available") or {}
        nodes: List[Dict[str, Any]] = []
        if isinstance(available.get("local"), dict):
            nodes.append(available["local"])
        if isinstance(available.get("remote"), list):
            nodes.extend(item for item in available["remote"] if isinstance(item, dict))
        if not nodes:
            table.add_row("[dim]（无在线节点）[/]", str(resources.get("scope") or "—"),
                          "", "", "", "")
            return
        for node in nodes[:64]:
            cpu = node.get("cpu") or {}
            ram = node.get("ram") or {}
            gpu = node.get("gpu") or {}
            table.add_row(
                str(node.get("node_id") or "—"),
                f"{node.get('role') or '—'} / "
                f"{'[green]在线[/]' if node.get('available') else '[yellow]离线[/]'}",
                f"{cpu.get('physical_cores', '—')} / {cpu.get('logical_cores', '—')}",
                f"{ram.get('total_gb', '—')} / {ram.get('available_gb', '—')} GB",
                f"{gpu.get('count', '—')} / {gpu.get('cuda_count', '—')}",
                f"{gpu.get('vram_total_gb', '—')} / {gpu.get('vram_free_gb', '—')} GB",
            )
        totals = resources.get("totals") or {}
        if totals:
            table.add_row(
                "[b]合计[/]",
                f"{resources.get('available_node_count', '—')}/"
                f"{resources.get('node_count', '—')} 可用",
                f"{totals.get('physical_cores', '—')} / {totals.get('logical_cores', '—')}",
                f"{totals.get('ram_total_gb', '—')} / {totals.get('ram_available_gb', '—')} GB",
                f"{totals.get('gpu_count', '—')} / {totals.get('cuda_gpu_count', '—')}",
                f"{totals.get('vram_total_gb', '—')} / {totals.get('vram_free_gb', '—')} GB",
            )

    # ------------------------------------------------------------ 运维面（节点/队列/日志）

    @work(thread=True, exclusive=True, group="pages")
    def load_pages(self) -> None:
        """拉取节点/队列/日志三屏的只读数据（失败只显示错误，不伪造内容）。"""
        nodes = self.fetch_json(API_PATHS["cluster_nodes"])
        queue = self.fetch_json(API_PATHS["cluster_queue"])
        logs = self.fetch_json(API_PATHS["cluster_log_aggregate"])
        self.app.call_from_thread(self.apply_pages, nodes, queue, logs)

    def apply_pages(self, nodes: Dict[str, Any], queue: Dict[str, Any],
                    logs: Dict[str, Any]) -> None:
        self.fill_nodes(nodes)
        self.fill_queue(queue)
        self.fill_logs(logs)

    def fill_nodes(self, nodes: Dict[str, Any]) -> None:
        table = self.query_one("#nodes-table", DataTable)
        table.clear()
        if "_error" in nodes:
            table.add_row("[red]不可用[/]", nodes["_error"], "", "", "", "")
            return
        items = nodes.get("nodes")
        if isinstance(items, dict):  # {node_id: {...}} 形状
            items = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                     for key, value in items.items()]
        if not items:
            table.add_row("[dim]（无节点数据）[/]", "", "", "", "", "")
            return
        for node in items[:64]:
            if not isinstance(node, dict):
                continue
            state = str(node.get("state") or ("online" if node.get("is_available") else "—"))
            colored = {"online": "[green]在线[/]", "offline": "[red]离线[/]"}.get(state, state)
            rtt = node.get("avg_rtt_ms")
            table.add_row(
                str(node.get("node_id") or "—"),
                str(node.get("role") or "—"),
                str(node.get("node_type") or "—"),
                colored,
                str(node.get("address") or "—"),
                f"{rtt:.0f} ms" if isinstance(rtt, (int, float)) else "—",
            )

    def fill_queue(self, queue: Dict[str, Any]) -> None:
        """``/cluster/queue`` → MLFQ：``q0/q1/q2`` + ``*_depth`` + ``completed_count``。

        此前按 ``tasks`` 解析——真实返回里没有该键（是三级队列），故只能显示"队列空闲"。
        """
        pane = self.query_one("#queue-pane", Static)
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        self.queue_state = queue if "_error" not in queue else {}
        if "_error" in queue:
            pane.update(f"[red]队列不可用[/]  ·  {queue['_error']}")
            return
        if queue.get("paused"):
            state = "[yellow]已暂停[/]"
        elif queue.get("running"):
            state = "[green]运行中[/]"
        else:
            state = "[yellow]未运行[/]"
        pane.update(
            f"{state}  ·  策略 {queue.get('strategy') or '—'}"
            f"  ·  队列 {queue.get('queue_size', '—')}/{queue.get('max_size', '—')}"
            f"  ·  已完成 {queue.get('completed_count', '—')}"
            f"  ·  当前任务 {queue.get('current_task') or '无'}")
        aging = queue.get("aging_params") or {}
        limits = {"q0": aging.get("q0_max_tokens", 128),
                  "q1": aging.get("q1_max_tokens", 512),
                  "q2": None}
        for level in ("q0", "q1", "q2"):
            tasks = queue.get(level) or []
            depth = queue.get(f"{level}_depth", len(tasks))
            preview = ", ".join(
                str(task.get("task_id") or task.get("id") or "task")
                for task in tasks[:3] if isinstance(task, dict)
            ) or "[dim]空[/]"
            if len(tasks) > 3:
                preview += f" … +{len(tasks) - 3}"
            table.add_row(
                f"[b]{level.upper()}[/]",
                str(depth),
                str(limits[level]) if limits[level] else "不限",
                preview,
            )

    def fill_logs(self, logs: Dict[str, Any]) -> None:
        view = self.query_one("#logs-log", RichLog)
        pane = self.query_one("#logs-pane", Static)
        view.clear()
        if "_error" in logs:
            pane.update(f"[red]后端日志不可用[/]  ·  {logs['_error']}"
                        "（聚合日志可能需要 X-QLH-Log-Token，见 qlh --log-token）")
            self.log_line_count = 0
            return
        # 真实结构：{local: {node_id, logs: [...]}, workers: [...], limit, total_workers}
        lines = _log_lines(logs.get("local"), "local")
        workers = logs.get("workers") or []
        if isinstance(workers, dict):
            worker_items = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                            for key, value in workers.items()]
        else:
            worker_items = [item for item in workers if isinstance(item, dict)]
        for item in worker_items:
            lines.extend(_log_lines(item, "worker"))
        shown = lines[-200:]
        self.log_line_count = len(shown)
        pane.update(f"共 {len(lines)} 行（显示末尾 {len(shown)}）"
                    f"  ·  local + {len(worker_items)} worker"
                    f"  ·  total_workers={logs.get('total_workers', 0)}"
                    f"  ·  limit={logs.get('limit', '—')}")
        if not shown:
            view.write("（后端未返回日志行）")
            return
        for line in shown:
            view.write(line)

    # ------------------------------------------------------------ 设备页

    @work(thread=True, exclusive=True, group="device")
    def load_device(self) -> None:
        profile = self.fetch_json(API_PATHS["device_profile"])
        self.app.call_from_thread(self.fill_device, profile)

    def fill_device(self, profile: Dict[str, Any]) -> None:
        pane = self.query_one("#device-pane", Static)
        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if "_error" in profile:
            pane.update(f"[red]设备画像不可用[/]\n{profile['_error']}")
            table.add_row("", "[dim]（画像不可用，无 GPU 数据）[/]", "", "", "", "")
            return
        # 真实结构：platform.{os,os_version,hostname,machine}、cpu.model_name、disk
        platform = profile.get("platform") or {}
        os_text = " ".join(
            str(platform.get(key) or "") for key in ("os", "os_version")).strip()
        cpu = profile.get("cpu") or {}
        ram = profile.get("ram") or profile.get("memory") or {}
        disk = profile.get("disk") or {}
        lines = [
            kv("操作系统", os_text or "—"),
            kv("主机名", platform.get("hostname")),
            kv("架构", f"{platform.get('machine', '—')} · {platform.get('architecture', '—')}"
                       f" · Python {platform.get('python_version', '—')}"),
            kv("CPU", cpu.get("model_name")),
            kv("核心", f"物理 {cpu.get('physical_cores', '—')} / 逻辑 {cpu.get('logical_cores', '—')}"
                       f" · 使用率 {cpu.get('usage_percent', '—')}%"),
            kv("内存", f"总量 {ram.get('total_gb', '—')} GB / 可用 {ram.get('available_gb', '—')} GB"
                       f" · 已用 {ram.get('percent_used', '—')}%"),
        ]
        if disk:
            lines.append(kv("磁盘", f"剩余 {disk.get('free_gb', '—')} GB / "
                                    f"总 {disk.get('total_gb', '—')} GB（{disk.get('path', '—')}）"))
        lines.append(kv("档位评估", f"{profile.get('tier_label', '—')} "
                                    f"({profile.get('tier', '—')}) · 评分 {profile.get('score_total', '—')}"))
        for item in (profile.get("recommendations") or [])[:5]:
            lines.append(f"  [green]建议[/]         {item}")
        for item in (profile.get("warnings") or [])[:5]:
            lines.append(f"  [yellow]警告[/]         {item}")
        pane.update("\n".join(lines))

        gpus = profile.get("gpus") or []
        selected = profile.get("selected_gpu_index", 0)
        self.cuda_available = any(
            bool(gpu.get("cuda_available")) for gpu in gpus if isinstance(gpu, dict))
        if not gpus:
            table.add_row("", "[dim]（未检测到 GPU）[/]", "", "", "", "")
            return
        for index, gpu in enumerate(gpus):
            if not isinstance(gpu, dict):
                continue
            table.add_row(
                "[green]◆[/]" if index == selected else "",
                str(gpu.get("name") or "—"),
                str(gpu.get("gpu_type") or "—"),
                "[green]支持[/]" if gpu.get("cuda_available") else "[yellow]不支持[/]",
                str(gpu.get("vram_total_gb", "—")),
                str(gpu.get("driver_version") or "—"),
            )

    # ------------------------------------------------------------ 写操作

    def write_status(self, text: str) -> None:
        """写操作没有独立回显窗口，结果常驻侧栏摘要（避免"点了没反应"）。"""
        self.op_status = text
        self.refresh_topbar()

    def finish_write_action(self, text: str) -> None:
        """写操作收尾：回显 + 重新拉取受影响的数据。"""
        self.write_status(text)
        self.action_reload()
        self.load_pages()

    def finish_light_action(self, text: str) -> None:
        self.write_status(text)
        self.load_pages()

    def selected_model_id(self, table: DataTable) -> str:
        """取光标行的 row key（``fill_models`` 以 model_id 作 key，不依赖行序）。"""
        if table.row_count == 0:
            return ""
        try:
            row_key, _column_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        except Exception:  # noqa: BLE001 - 空表或坐标越界
            return ""
        return str(row_key.value or "")

    def model_load_args(self, info: Dict[str, Any]) -> Tuple[str, str]:
        """由注册表项推出 (engine, quant_type)——不猜参数，缺省即后端默认。

        量化**优先 int4**（与后端 ``LoadModelRequest`` 默认值一致，显存占用最低）；
        注册表未给 int4 时才退到其列表首项。
        """
        engine = str(info.get("preferred_engine") or "llama_cpp")
        quant_types = info.get("quant_types") or info.get("quantizations") or []
        if isinstance(quant_types, dict):
            quant_types = list(quant_types.keys())
        elif isinstance(quant_types, str):
            quant_types = [quant_types]
        choices = [str(item) for item in quant_types]
        if "int4" in choices:
            quant = "int4"
        else:
            quant = choices[0] if choices else "int4"
        return engine, quant

    def action_load_model(self) -> None:
        """模型屏 L：加载光标行的模型（先确认，再走 worker，避免阻塞 UI）。"""
        if PAGES[self.page_index][0] != "models":
            self.write_status("[yellow]请先切到「模型」屏（侧栏或 [ ] 键）[/]")
            return
        table = self.query_one("#models-table", DataTable)
        model_id = self.selected_model_id(table)
        if not model_id:
            self.write_status("[yellow]模型屏没有可加载的行[/]")
            return
        info = self.model_rows.get(model_id) or {}
        if info.get("is_available") is False:
            reason = info.get("unavailable_reason") or "后端未说明原因"
            self.write_status(f"[yellow]{model_id} 不可用[/]：{reason}")
            return
        engine, quant = self.model_load_args(info)
        name = str(info.get("name") or model_id)
        note = ""
        if engine in {"pytorch", "island"} and self.cuda_available is False:
            note = "\n[yellow]提示[/]：本机 CUDA 不可用，PyTorch/孤岛引擎会走 CPU（很慢）。"
        self.app.confirm(
            "加载模型",
            f"将要加载：[b]{model_id}[/]（{name}）\n"
            f"引擎 [b]{engine}[/] · 量化 [b]{quant}[/]\n"
            "耗时约 5-20 秒；期间会**先卸载当前模型**，失败由后端自动回滚。"
            f"{note}",
            lambda: self.run_model_action("load", model_id, engine, quant),
            confirm_label="加载",
        )

    def action_unload_model(self) -> None:
        """模型屏 U：卸载当前模型（破坏性：会中断正在生成的请求）。"""
        if PAGES[self.page_index][0] != "models":
            self.write_status("[yellow]请先切到「模型」屏（侧栏或 [ ] 键）[/]")
            return
        active = self.model_text.replace("[yellow]", "").replace("[/]", "")
        self.app.confirm(
            "卸载模型",
            f"将**释放**当前本地模型：{active or '（当前未加载）'}\n"
            "影响：正在生成/排队的请求会失败；KV 缓存与对话上下文一并清空。\n"
            "之后需要重新「加载模型」才能对话。",
            lambda: self.run_model_action("unload", "", "", ""),
            confirm_label="卸载",
        )

    @work(thread=True, exclusive=True, group="modelctl")
    def run_model_action(self, kind: str, model_id: str, engine: str, quant: str) -> None:
        """模型控制走独立 worker 组：加载 5-20 秒，不能阻塞 UI 也不与只读刷新互斥。"""
        app = self.app
        if kind == "load":
            self.app.call_from_thread(self.set_busy, f"正在加载 {model_id}（5-20 秒）…")
            try:
                result = load_model(app.api, model_id, engine=engine, quant_type=quant)
                detail = result.get("message") or result.get("detail") or ""
                text = f"[green]模型已加载[/] {model_id} {detail}".strip()
            except ApiError as exc:
                text = f"[red]模型加载失败[/]：{exc}"
        else:
            self.app.call_from_thread(self.set_busy, "正在卸载模型…")
            try:
                unload_model(app.api)
                text = "[green]模型已卸载[/]"
            except ApiError as exc:
                text = f"[red]模型卸载失败[/]：{exc}"
        self.app.call_from_thread(self.finish_write_action, text)

    def set_busy(self, text: str) -> None:
        self.write_status(f"[yellow]…[/] {text}")

    def action_queue_toggle_pause(self) -> None:
        """队列屏 P：暂停/恢复接受新请求（可逆，无需确认）。"""
        if PAGES[self.page_index][0] != "queue":
            self.write_status("[yellow]请先切到「队列」屏[/]")
            return
        if not self.queue_state:
            self.write_status("[yellow]队列数据不可用，无法操作[/]")
            return
        self.run_queue_action("resume" if self.queue_state.get("paused") else "pause")

    def action_queue_cycle_strategy(self) -> None:
        """队列屏 S：mlfq ↔ fifo（影响新请求的排队行为，故先确认）。"""
        if PAGES[self.page_index][0] != "queue":
            self.write_status("[yellow]请先切到「队列」屏[/]")
            return
        if not self.queue_state:
            self.write_status("[yellow]队列数据不可用，无法操作[/]")
            return
        current = str(self.queue_state.get("strategy") or "mlfq").lower()
        target = "fifo" if current == "mlfq" else "mlfq"
        self.app.confirm(
            "切换调度策略",
            f"当前 [b]{current}[/] → [b]{target}[/]\n"
            "影响：新入队请求的优先级与老化（aging）行为；执行中的任务不受影响。",
            lambda: self.run_queue_action("strategy", target),
            confirm_label="切换",
        )

    def action_queue_clear(self) -> None:
        """队列屏 C：清空排队任务（列出将清空的内容后再确认）。"""
        if PAGES[self.page_index][0] != "queue":
            self.write_status("[yellow]请先切到「队列」屏[/]")
            return
        if not self.queue_state:
            self.write_status("[yellow]队列数据不可用，无法操作[/]")
            return
        pending = self.queue_state.get("queue_size", "—")
        listing = []
        for level in ("q0", "q1", "q2"):
            tasks = self.queue_state.get(level) or []
            if not tasks:
                continue
            names = ", ".join(
                str(task.get("task_id") or task.get("id") or "task")
                for task in tasks[:5] if isinstance(task, dict))
            listing.append(f"  {level.upper()}（{len(tasks)}）：{names}")
        self.app.confirm(
            "清空排队任务",
            f"将清空**排队中**的任务（queue_size={pending}），执行中的任务**不受影响**：\n"
            + ("\n".join(listing) if listing else "  （三级队列当前为空）")
            + "\n此操作不可撤销。",
            lambda: self.run_queue_action("clear"),
            confirm_label="清空",
        )

    @work(thread=True, exclusive=True, group="queuectl")
    def run_queue_action(self, action: str, value: str = "") -> None:
        app = self.app
        self.app.call_from_thread(self.set_busy, f"队列操作 {action} …")
        try:
            if action == "pause":
                pause_queue(app.api)
            elif action == "resume":
                resume_queue(app.api)
            elif action == "strategy":
                set_queue_strategy(app.api, value)
            elif action == "clear":
                clear_queue(app.api)
            else:
                raise ValueError(f"未知队列动作: {action}")
            text = f"[green]队列 {action} 完成[/]" + (f" → {value}" if value else "")
        except (ApiError, ValueError) as exc:
            text = f"[red]队列 {action} 失败[/]：{exc}"
        self.app.call_from_thread(self.finish_light_action, text)

    # ------------------------------------------------------------ 动作

    def action_quit_app(self) -> None:
        self.app.exit()


class KoakumaApp(App):
    """统一交互入口的 Textual 实现。"""

    TITLE = "Koakuma"
    SUB_TITLE = "QLH 分布式边缘推理"
    CSS = CSS

    def __init__(self, api: ApiClient, *, interval: float = 5.0,
                 routing_preference: str = "auto", show_thinking: bool = False,
                 supervisor: Any = None) -> None:
        super().__init__()
        self.api = api
        self.interval = float(interval)
        self.routing_preference = routing_preference
        self.show_thinking = show_thinking
        #: 传入 BackendSupervisor 即在启动屏内完成冷启动（LOGO + 启动条同屏反馈）
        self.supervisor = supervisor
        self.session_id: Optional[str] = None
        self.generation_id: Optional[str] = None

    # ------------------------------------------------------------ 启动流程

    def on_mount(self) -> None:
        waiting = self.supervisor is not None
        self.push_screen(SplashScreen(
            status="检查本地后端" if waiting else "连接后端",
            wait_for_backend=waiting,
        ))
        if waiting:
            self.set_interval(0.2, self.refresh_splash_status)
            self.start_backend()

    @work(thread=True, exclusive=True, group="backend")
    def start_backend(self) -> None:
        """在启动屏展示期间把本机后端拉起来（阶段文本实时写回启动屏）。"""
        try:
            self.supervisor.ensure_ready()
        except BaseException as exc:  # noqa: BLE001 - 交回主线程展示
            self.call_from_thread(self.backend_failed, str(exc))
            return
        self.call_from_thread(self.show_main)

    def refresh_splash_status(self) -> None:
        if self.supervisor is None:
            return
        screen = self.screen
        if isinstance(screen, SplashScreen):
            screen.set_status(self.supervisor.status_message)

    def backend_failed(self, message: str) -> None:
        screen = self.screen
        if isinstance(screen, SplashScreen):
            screen.wait_for_backend = False
            screen.set_status(f"[red]后端启动失败[/]：{message}（按任意键进入界面查看状态）")

    def confirm(self, title: str, body: str, on_confirm, *,
                confirm_label: str = "确认") -> None:
        """弹出模态确认；用户确认后才执行 ``on_confirm()``（写操作的统一闸门）。"""
        def _done(approved: Optional[bool]) -> None:
            if approved:
                on_confirm()

        self.push_screen(ConfirmScreen(title, body, confirm_label=confirm_label), _done)

    def show_main(self) -> None:
        """从启动屏切到主界面（重复调用安全）。"""
        if isinstance(self.screen, MainScreen):
            return
        self.switch_screen(MainScreen())


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *, interval: float = 5.0,
        routing_preference: str = "auto", show_thinking: bool = False,
        supervisor: Any = None) -> int:
    """启动 Textual 外壳（供 qlh.py 调用）。

    ``supervisor`` 非空时，本机后端冷启动在**启动屏内**完成（LOGO + 启动条同屏）。
    """
    api = ApiClient(host=host, port=port)
    KoakumaApp(api, interval=interval, routing_preference=routing_preference,
               show_thinking=show_thinking, supervisor=supervisor).run()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Koakuma TUI (Textual shell)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--route", default="auto")
    parser.add_argument("--thinking", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.host, args.port, interval=args.interval,
               routing_preference=args.route, show_thinking=args.thinking)


if __name__ == "__main__":
    raise SystemExit(main())
