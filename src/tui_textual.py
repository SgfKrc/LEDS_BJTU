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
from typing import Any, Dict, List, Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tui_api import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    ApiClient,
    ApiError,
    cancel_generation,
    iter_chat_payloads,
)
from tui_shared import API_PATHS, format_metrics  # noqa: E402

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

CSS = """
Screen { background: $surface; }
#splash-logo { color: #8fa8c4; text-align: center; padding: 1 0 0 0; }
#splash-bar { color: #6b8aa8; text-align: center; padding: 1 0 0 0; }
#splash-status { color: $text; text-align: center; padding: 0 0 1 0; }
#splash-hint { text-align: center; color: $text-disabled; }
#banner { padding: 0 2; color: $text-muted; }
#chat-log { height: 1fr; border: round $primary 30%; padding: 0 1; }
#chat-status { padding: 0 2; height: auto; }
#chat-input { dock: bottom; }
#status-pane { padding: 1 2; }
.about { padding: 1 2; }
DataTable { height: auto; max-height: 100%; }
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


class ChatPane(Vertical):
    """聊天 Tab：SSE 流式输出（等价旧 ChatScreen 的事件处理）。"""

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
        yield Static("就绪。输入消息并回车发送；/help 查看命令", id="chat-status")
        yield Input(placeholder="输入消息…", id="chat-input")

    # ------------------------------------------------------------ 发送

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = (event.value or "").strip()
        if not text:
            return
        event.input.value = ""
        log = self.query_one("#chat-log", RichLog)
        if text in {"/help", "/?"}:
            log.write("[b]可用命令[/] /help /clear /route auto|local|distributed|required "
                      "/thinking on|off /cancel /quit")
            return
        if text == "/quit":
            self.app.exit()
            return
        if text == "/clear":
            log.clear()
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
        if text.startswith("/"):
            self.query_one("#chat-status", Static).update(f"[red]未知命令: {text}")
            return

        log.write(f"[b $accent]你[/] {text}")
        log.write("[dim]assistant[/] ")
        self.stream_reply(text)

    @work(thread=True, exclusive=True)
    def stream_reply(self, message: str) -> None:
        """在后台线程消费 SSE；所有 UI 更新都回到主线程执行。"""
        app = self.app
        acc: List[str] = []
        status = "…"
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
                    final = payload.get("response")
                    if isinstance(final, str) and final and "".join(acc) != final:
                        self.app.call_from_thread(self.replace_transcript, final)
                    if payload.get("session_id"):
                        app.session_id = payload["session_id"]
                    status = format_metrics(
                        payload.get("metrics") or None,
                        history_committed=payload.get("history_committed"),
                    )
                elif payload.get("cancelled"):
                    status = "已被取消"
                elif payload.get("error"):
                    status = f"[red]后端错误: {payload['error']}"
        except Exception as exc:  # noqa: BLE001 - 网络异常不应让界面崩溃
            status = f"[red]请求失败: {exc}"
        finally:
            app.generation_id = None
            self.app.call_from_thread(self.set_status, status)

    # ------------------------------------------------------------ UI 更新（主线程）

    def append_chunk(self, piece: str) -> None:
        self.query_one("#chat-log", RichLog).write(piece, end="")

    def replace_transcript(self, text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("")
        log.write(text)

    def set_status(self, text: str) -> None:
        self.query_one("#chat-status", Static).update(text or "完成")


class MainScreen(Screen):
    """主界面：聊天 + 状态 + 模型 + 分布式 + 关于。"""

    BINDINGS = [
        Binding("r", "reload", "刷新"),
        Binding("q", "quit_app", "退出"),
        Binding("ctrl+c", "quit_app", "退出", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="banner")
        with TabbedContent("聊天", "状态", "模型", "分布式", "关于", id="tabs"):
            with TabPane("聊天", id="tab-chat"):
                yield ChatPane(id="chat-pane")
            with TabPane("状态", id="tab-status"):
                yield VerticalScroll(Static("加载中…", id="status-pane"))
            with TabPane("模型", id="tab-models"):
                yield DataTable(id="models-table")
            with TabPane("分布式", id="tab-distributed"):
                yield DataTable(id="resources-table")
            with TabPane("关于", id="tab-about"):
                yield Static(self.about_text(), id="about-pane", classes="about")
        yield Footer()

    def about_text(self) -> str:
        return (
            f"[b $accent]Koakuma[/] · QLH 分布式边缘推理 · Textual 外壳\n\n"
            f"后端: {self.app.api.base_url}\n"
            f"快捷键: [b]r[/] 刷新 · [b]q[/] 退出 · [b]/help[/] 聊天命令\n\n"
            f"依赖边界: 本 UI 需要 Textual（requirements-tui.txt，Edge 同装）；\n"
            f"协议层 src/tui_api.py 为纯标准库，单命令/CI 路径无需 UI 依赖。"
        )

    def on_mount(self) -> None:
        self.query_one("#banner", Static).update(
            f"[dim]{self.app.api.base_url} · Tab 切换 · r 刷新 · q 退出[/]")
        self.query_one("#models-table", DataTable).add_columns("模型", "格式", "引擎", "状态")
        self.query_one("#resources-table", DataTable).add_columns("节点", "运行模式", "就绪", "任务")
        self.action_reload()
        self.set_interval(self.app.interval, self.action_reload)

    # ------------------------------------------------------------ 只读数据

    @work(thread=True, exclusive=True)
    def action_reload(self) -> None:
        health = self.fetch_json("/health")
        current = self.fetch_json(API_PATHS["models_current"])
        resources = self.fetch_json(API_PATHS["cluster_resources"])
        self.app.call_from_thread(self.apply_data, health, current, resources)

    def fetch_json(self, path: str) -> Dict[str, Any]:
        try:
            value = self.app.api.get(path if path.startswith("/") else "/" + path)
            return value if isinstance(value, dict) else {"value": value}
        except ApiError as exc:
            return {"_error": str(exc)}

    def apply_data(self, health: Dict[str, Any], current: Dict[str, Any],
                   resources: Dict[str, Any]) -> None:
        self.query_one("#status-pane", Static).update(self.status_text(health, current))
        self.fill_models(current)
        self.fill_resources(resources)

    def status_text(self, health: Dict[str, Any], current: Dict[str, Any]) -> str:
        if "_error" in health:
            return f"[red]后端不可达[/]\n{health['_error']}"
        status = health.get("status") or health.get("ok") or "ok"
        lines = [f"后端: [green]{status}[/] · {self.app.api.base_url}"]
        model = current.get("model_id") or current.get("name") or "—"
        lines.append(f"当前模型: [b]{model}[/] · 引擎: {current.get('engine') or '—'}")
        if current.get("format"):
            lines.append(f"格式: {current['format']}")
        return "\n".join(lines)

    def fill_models(self, current: Dict[str, Any]) -> None:
        table = self.query_one("#models-table", DataTable)
        table.clear()
        if "_error" in current:
            table.add_row("[red]不可用[/]", current["_error"], "", "")
            return
        rows = current.get("models") or current.get("items") or []
        if not rows and (current.get("model_id") or current.get("name")):
            rows = [current]
        if not rows:
            table.add_row("—", "—", "—", "后端未返回模型列表")
            return
        for item in rows[:64]:
            if not isinstance(item, dict):
                continue
            table.add_row(
                str(item.get("model_id") or item.get("name") or "—"),
                str(item.get("format") or "—"),
                str(item.get("engine") or "—"),
                str(item.get("status") or item.get("state") or "—"),
            )

    def fill_resources(self, resources: Dict[str, Any]) -> None:
        table = self.query_one("#resources-table", DataTable)
        table.clear()
        if "_error" in resources:
            table.add_row("[red]不可用[/]", resources["_error"], "", "")
            return
        nodes = resources.get("nodes") or []
        if isinstance(nodes, dict):
            nodes = [{"node_id": key, **(value if isinstance(value, dict) else {})}
                     for key, value in nodes.items()]
        if not nodes:
            table.add_row("—", str(resources.get("mode") or "—"), "—", "无节点数据")
            return
        for node in nodes[:64]:
            if not isinstance(node, dict):
                continue
            memory = node.get("ram_available_gb") or node.get("ram_total_gb")
            table.add_row(
                str(node.get("node_id") or node.get("id") or "—"),
                str(node.get("mode") or node.get("role") or "—"),
                "是" if node.get("ready") else "否",
                f"RAM {memory} GiB" if memory else str(node.get("current_task") or "—"),
            )

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

    @work(thread=True, exclusive=True)
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
