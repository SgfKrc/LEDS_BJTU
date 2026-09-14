"""PATCH-02 书架 TUI：三栏（书架列表 → 文档卡 → 预览）。

开发期工具（与 docagent 同级）：允许 Textual 依赖；只读。
键位：↑↓ 选择 · 1-7 分类过滤 · a 归档开关 · r 刷新 · q 退出。

用法：python -m patchouli.bookshelf --root <repo>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from .catalog import scan

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
    """
    BINDINGS = [
        ("q", "quit", "退出"),
        ("r", "reload", "刷新"),
        ("a", "toggle_archive", "归档"),
        ("0", "clear_filter", "全部"),
        *[(str(i + 1), f"filter({i})", KIND_LABEL[kind]) for i, kind in enumerate(KIND_ORDER)],
    ]

    def __init__(self, root: Path, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)
        self.catalog: dict = {}
        self.filter_kind: str | None = None
        self.show_archive = False
        self.entries: list[dict] = []
        self.preview_text = ""

    # ---- layout ----
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield ListView(id="shelf")
            with Vertical(id="detail"):
                yield Static("选中文档查看信息", id="card", markup=False)
                yield Static("预览", id="preview", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Patchouli 书架"
        self.action_reload()

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
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None and 0 <= event.item.index < len(self.entries):
            self._show_entry(self.entries[event.item.index])

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is not None and 0 <= event.item.index < len(self.entries):
            self._show_entry(self.entries[event.item.index])

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patchouli 书架 TUI（只读）")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    BookshelfApp(args.root.expanduser().resolve()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
