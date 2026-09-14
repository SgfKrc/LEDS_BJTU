"""PATCH-02 书架 TUI 测试：逻辑层 + Textual Pilot 冒烟。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import KIND_ORDER, BookshelfApp  # noqa: E402
from textual.widgets import ListView  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> None:
    _write(tmp_path, "docs/甲专项计划.md", "# 甲\n> 状态：现行\n> 更新日期：2026-01-01\n票 `PATCH-02`。\n")
    _write(tmp_path, "docs/乙报告.md", "# 乙\n> 状态：历史参考\n")
    _write(tmp_path, "docs/丙说明.md", "# 丙\n")  # 缺状态行 → guide
    _write(tmp_path, "docs/archive/丁计划.md", "# 丁\n> 状态：历史参考\n")


def test_bookshelf_filters_and_preview(tmp_path: Path) -> None:
    _fixture(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            shelf = app.query_one("#shelf", ListView)
            assert len(app.entries) == 3  # 默认不含归档
            assert all(not e["archived"] for e in app.entries)

            await pilot.press("2")  # special-plan
            assert app.filter_kind == "special-plan"
            assert [e["name"] for e in app.entries] == ["甲专项计划"]
            assert "甲" in app.preview_text  # 预览加载了正文

            await pilot.press("4")  # report
            assert [e["name"] for e in app.entries] == ["乙报告"]

            await pilot.press("0")  # 清除过滤
            assert len(app.entries) == 3

            await pilot.press("a")  # 显示归档
            assert len(app.entries) == 4
            assert any(e["archived"] for e in app.entries)

            assert len(shelf) == 4

    asyncio.run(run())


def test_bookshelf_missing_status_marked(tmp_path: Path) -> None:
    _fixture(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test():
            no_status = [e for e in app.entries if e["status"] is None]
            assert [e["name"] for e in no_status] == ["丙说明"]

    asyncio.run(run())


def test_kind_order_covers_labels() -> None:
    assert "other" in KIND_ORDER and "decision" in KIND_ORDER
