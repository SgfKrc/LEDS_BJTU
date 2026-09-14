"""PATCH-02 书架 TUI：三栏（书架列表 → 文档卡 → 预览）。

开发期工具（与 docagent 同级）：允许 Textual 依赖；只读。
键位：↑↓ 选择 · / 检索台 · 1-7 分类过滤 · a 归档开关 · c 编目诊断 · r 刷新 · q 退出。

用法：python -m patchouli.bookshelf --root <repo>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from .catalog import scan
from .config_check import env_status
from .config_editor import ConfigEditor
from .splash import SplashScreen

KIND_ORDER = ["decision", "special-plan", "ticket-plan", "report", "guide", "reference", "other"]
KIND_LABEL = {
    "decision": "决策",
    "special-plan": "计划",
    "ticket-plan": "票计划",
    "report": "报告",
    "guide": "指南",
    "reference": "参考",
    "other": "其他",
}
PREVIEW_LINES = 60


class BookshelfApp(App):
    """三栏书架：列表 → 文档卡 → 预览。"""

    CSS = """
    #shelf { width: 38%; border: solid $accent; }
    #detail { width: 62%; }
    #card { height: auto; max-height: 45%; border: solid $accent; padding: 0 1; }
    #preview { border: solid $accent; padding: 0 1; }
    #query { dock: top; display: none; border: solid $accent; }
    #splash-box { border: solid $accent; padding: 1 2; }
    #editor-box { width: 92%; height: 92%; border: solid $accent; padding: 0 1; }
    #editor-area { height: 1fr; }
    """
    BINDINGS = [
        ("q", "quit", "退出"),
        ("r", "reload", "刷新"),
        ("a", "toggle_archive", "归档"),
        ("c", "diag", "编目诊断"),
        ("h", "history", "流转记录"),
        ("s", "stats", "馆藏统计"),
        ("v", "compare", "参数对照"),
        ("e", "edit_config", "配置"),
        ("/", "focus_search", "检索台"),
        ("escape", "exit_search", "返回书架"),
        ("0", "clear_filter", "全部"),
        *[(str(i + 1), f"filter({i})", KIND_LABEL[kind]) for i, kind in enumerate(KIND_ORDER)],
    ]

    def __init__(self, root: Path, splash: bool | None = None, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self._splash_arg = splash
        self.catalog: dict = {}
        self.filter_kind: str | None = None
        self.show_archive = False
        self.entries: list[dict] = []
        self.preview_text = ""
        self.diag: dict | None = None
        self.diag_text = ""
        self.mode = "shelf"
        self.search_results: list[dict] = []
        self.search_query = ""
        self.current_path: str | None = None
        self.history_text = ""
        self.history_shown = False
        self.stats_text = ""
        self.stats_shown = False
        self.compare_text = ""
        self.compare_shown = False
        self._last_index = 0

    # ---- layout ----
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="检索：文本 / t:票号 / k:类型 / s:缺 / a:含归档 → Enter", id="query", disabled=True)
        with Horizontal():
            yield ListView(id="shelf")
            with Vertical(id="detail"):
                yield Static("选中文档查看信息", id="card", markup=False)
                yield Static("预览", id="preview", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Patchouli 书架"
        use_splash = self._splash_arg if self._splash_arg is not None else not self.is_headless
        if use_splash:
            self.push_screen(SplashScreen("扫描文档树…"))
            self.run_worker(self._load_async, thread=True, name="boot")
        else:
            self.action_reload()
        self._check_env()

    def _load_async(self) -> None:
        self.call_from_thread(self._finish_load, scan(self.root, include_archive=True))

    def _finish_load(self, catalog: dict) -> None:
        self.catalog = catalog
        self._apply_filter()
        if len(self.screen_stack) > 1:  # 关闭 splash（加载完成即关，不额外等待）
            self.pop_screen()

    def _check_env(self) -> None:
        try:
            status = env_status(self.root)
        except Exception:  # noqa: BLE001
            return
        if not status.get("exists"):
            self.notify("未配置 docagent（缺 .env.docagent）：按 e 打开配置编辑（已预填模板）", severity="warning", timeout=12)

    # ---- 配置编辑（写权限仅限 .env.docagent） ----
    def action_edit_config(self) -> None:
        self.push_screen(ConfigEditor(self.root), self._on_editor_closed)

    def _on_editor_closed(self, result: dict | None) -> None:
        if result and result.get("saved", {}).get("ok"):
            self.notify("配置已保存（.env.docagent，已备份 .bak）", timeout=6)
            self._check_env()

    # ---- data ----
    def action_reload(self) -> None:
        self.catalog = scan(self.root, include_archive=True)
        self._apply_filter()

    def _apply_filter(self) -> None:
        entries = [e for e in self.catalog.get("documents", []) if self.show_archive or not e["archived"]]
        if self.filter_kind:
            entries = [e for e in entries if e["kind"] == self.filter_kind]
        entries.sort(key=lambda e: (e["archived"], e["name"]))
        self.entries = entries
        shelf = self.query_one("#shelf", ListView)
        shelf.clear()
        for entry in entries:
            mark = "[档]" if entry["archived"] else "    "
            status = "*" if entry["status"] else "-"
            shelf.append(ListItem(Label(f"{mark} {status} {entry['name'][:36]}")))
        self._show_entry(entries[0] if entries else None)
        self.sub_title = self._filter_desc(len(entries))

    def _filter_desc(self, count: int) -> str:
        kind = KIND_LABEL.get(self.filter_kind, "全部") if self.filter_kind else "全部"
        arch = "含归档" if self.show_archive else "仅现行"
        return f"{kind} · {arch} · {count}/{self.catalog.get('doc_count', 0)}"

    def _show_entry(self, entry: dict | None) -> None:
        card = self.query_one("#card", Static)
        preview = self.query_one("#preview", Static)
        if entry is None:
            card.update("（无匹配文档）")
            preview.update("")
            self.preview_text = ""
            return
        self.current_path = entry["path"]
        rows = [
            f"标题: {entry['title'] or entry['name']}",
            f"路径: {entry['path']}",
            f"类型: {KIND_LABEL.get(entry['kind'], entry['kind'])}",
            f"状态: {entry['status'] or '（缺状态行）'}",
            f"更新: {entry['updated'] or '-'}",
            f"票号: {', '.join(entry['tickets'][:6]) or '-'}",
            f"互链: {entry['link_count']} · 大小: {entry['size_bytes']} B",
        ]
        card.update("\n".join(rows))
        try:
            text = (self.root / entry["path"]).read_text(encoding="utf-8", errors="replace")
            preview.update("\n".join(text.splitlines()[:PREVIEW_LINES]))
            self.preview_text = text
        except OSError as exc:  # noqa: BLE001
            preview.update(f"读取失败: {exc!r}")
            self.preview_text = ""

    # ---- events / actions ----
    def _pick(self, index: int) -> None:
        self._last_index = index
        self.history_shown = False
        self.stats_shown = False
        self.compare_shown = False
        if self.mode == "search":
            self._show_result(index)
        elif 0 <= index < len(self.entries):
            self._show_entry(self.entries[index])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        index = self.query_one("#shelf", ListView).index
        if index is not None:
            self._pick(index)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#shelf", ListView).index
        if index is not None:
            self._pick(index)

    def action_clear_filter(self) -> None:
        self.filter_kind = None
        self._apply_filter()

    def action_filter(self, index: int) -> None:
        kind = KIND_ORDER[index]
        self.filter_kind = None if self.filter_kind == kind else kind
        self._apply_filter()

    def action_toggle_archive(self) -> None:
        self.show_archive = not self.show_archive
        self._apply_filter()

    # ---- PATCH-03 检索台 ----
    def action_focus_search(self) -> None:
        box = self.query_one("#query", Input)
        box.disabled = False
        box.display = True
        box.focus()

    def action_exit_search(self) -> None:
        if self.mode != "search":
            return
        box = self.query_one("#query", Input)
        box.display = False
        box.disabled = True
        box.value = ""
        self.mode = "shelf"
        self.search_results = []
        self._apply_filter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        from .search import search_docs  # 延迟导入（引擎零依赖）

        self.search_query = query
        self.search_results = search_docs(self.root, self.catalog, query)
        self.mode = "search"
        shelf = self.query_one("#shelf", ListView)
        shelf.clear()
        for result in self.search_results:
            loc = f"L{result['line_no']}" if result["line_no"] else "文件名"
            shelf.append(ListItem(Label(f"{loc:>6} {result['line']}")))
        self._show_result(0)
        self.sub_title = f"检索[{query}] · {len(self.search_results)} 条"
        self.query_one("#shelf", ListView).focus()

    def _show_result(self, index: int) -> None:
        card = self.query_one("#card", Static)
        preview = self.query_one("#preview", Static)
        if not (0 <= index < len(self.search_results)):
            card.update("（无结果）")
            preview.update("")
            return
        from .search import context_snippet

        result = self.search_results[index]
        self.current_path = result["path"]
        from .chunks import chunk_for_line, load_chunks

        chunk = chunk_for_line(load_chunks(self.root, result["path"]), result["line_no"] or 1)
        section = f"{chunk['heading']} (L{chunk['start_line']}-{chunk['end_line']})" if chunk else "-"
        loc = f"L{result['line_no']}" if result["line_no"] else "文件名命中"
        card.update("\n".join([
            f"查询: {self.search_query}",
            f"文档: {result['name']}",
            f"路径: {result['path']}",
            f"命中: {loc}",
            f"章节: {section}",
        ]))
        preview.update(context_snippet(self.root, result["path"], result["line_no"]))

    # ---- PATCH-07 参数对照 ----
    def action_compare(self) -> None:
        if self.compare_shown:
            self.compare_shown = False
            self._pick(self._last_index)
            return
        query = self.search_query or ""
        if not query:
            self.sub_title = "先检索（/）再按 v 做参数对照"
            return
        from .compare import compare_search, render_compare

        self.compare_text = render_compare(compare_search(self.root, self.catalog, query))
        self.compare_shown = True
        self.history_shown = False
        self.stats_shown = False
        self.query_one("#preview", Static).update(self.compare_text)
        self.sub_title = f"参数对照[{query}]（v 返回）"

    # ---- PATCH-06 馆藏统计 ----
    def action_stats(self) -> None:
        if self.stats_shown:
            self.stats_shown = False
            self._pick(self._last_index)
            return
        from .stats import compute_stats, render_stats

        self.stats_text = render_stats(compute_stats(self.catalog))
        self.stats_shown = True
        self.history_shown = False
        self.query_one("#preview", Static).update(self.stats_text)
        self.sub_title = "馆藏统计（s 返回）"

    # ---- PATCH-05 流通记录 ----
    def action_history(self) -> None:
        if self.history_shown:
            self.history_shown = False
            self._pick(self._last_index)
            return
        if not self.current_path:
            return
        preview = self.query_one("#preview", Static)
        preview.update("流通记录读取中（git log）…")
        path = self.current_path
        self.run_worker(lambda: self._run_history(path), thread=True, name="history")

    def _run_history(self, path: str) -> None:
        from .history import file_history  # 延迟导入

        self.call_from_thread(self._render_history, path, file_history(self.root, path))

    def _render_history(self, path: str, result: dict) -> None:
        from .history import render_history

        self.history_text = render_history(result, path)
        self.history_shown = True
        self.query_one("#preview", Static).update(self.history_text)
        if result.get("ok"):
            self.sub_title = f"流通记录 · {len(result.get('commits', []))} 次提交"
        else:
            self.sub_title = "流通记录不可用（h 返回）"

    # ---- PATCH-04 编目诊断 ----
    def action_diag(self) -> None:
        if self.diag is not None:  # 再按一次回到文档预览
            self.diag = None
            self._show_entry(self.entries[0] if self.entries else None)
            self.sub_title = self._filter_desc(len(self.entries))
            return
        preview = self.query_one("#preview", Static)
        preview.update("编目诊断运行中（docagent scan）…")
        self.diag_text = ""
        self.run_worker(self._run_diag, thread=True, name="diag")

    def _run_diag(self) -> None:
        from .catalog_diag import run_scan  # 延迟导入：数据层保持零依赖

        self.call_from_thread(self._render_diag, run_scan(self.root))

    def _render_diag(self, result: dict) -> None:
        from .catalog_diag import render_diag

        self.diag = result
        text = render_diag(result)
        self.diag_text = text
        self.query_one("#preview", Static).update(text)
        status = "诊断完成" if result.get("ok") else "诊断失败（c 返回）"
        self.sub_title = f"{status} · 问题 {len(result.get('findings', []))}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patchouli 书架 TUI（只读）")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--no-splash", action="store_true", help="跳过启动动画")
    args = parser.parse_args(argv)
    splash = False if (args.no_splash or os.environ.get("PATCHOULI_NO_SPLASH") == "1") else None
    BookshelfApp(args.root.expanduser().resolve(), splash=splash).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
