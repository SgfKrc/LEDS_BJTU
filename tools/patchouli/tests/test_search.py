"""PATCH-03 检索台测试：查询解析 / 检索引擎 / Pilot 流程 / 真实冒烟。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.catalog import scan  # noqa: E402
from patchouli.search import context_snippet, parse_query, search_docs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/甲专项计划.md",
        "# 甲\n> 状态：现行\n\n前缀缓存机制设计与实现说明。\n票 `EX-CACHE-01` 收口。\n",
    )
    _write(tmp_path, "docs/乙报告.md", "# 乙\n> 状态：历史参考\n\n缓存实验结论：命中率上升。\n")
    _write(tmp_path, "docs/丙说明.md", "# 丙\n\n无状态行但提到缓存。\n")
    _write(tmp_path, "docs/archive/丁报告.md", "# 丁\n> 状态：历史参考\n\n归档里的缓存记录。\n")


def test_parse_query_mixed() -> None:
    spec = parse_query("缓存 t:EX-CACHE-01 k:report s: a:")
    assert spec["terms"] == ["缓存"]
    assert spec["tickets"] == ["EX-CACHE-01"]
    assert spec["kinds"] == ["report"]
    assert spec["missing_status"] is True and spec["archived"] is True


def test_search_fulltext_line_numbers(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = scan(tmp_path, include_archive=True)
    results = search_docs(tmp_path, catalog, "缓存")
    paths = [r["path"] for r in results]
    assert "docs/甲专项计划.md" in paths
    assert "docs/archive/丁报告.md" not in paths  # 默认不含归档
    dings = [r for r in results if r["path"] == "docs/甲专项计划.md"]
    # 文件名不含"缓存" → 只有行命中（L4）；命中行正确
    assert dings and dings[0]["line_no"] == 4 and "缓存" in dings[0]["line"]


def test_search_filters(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = scan(tmp_path, include_archive=True)
    assert {r["path"] for r in search_docs(tmp_path, catalog, "t:EX-CACHE-01")} == {"docs/甲专项计划.md"}
    assert {r["path"] for r in search_docs(tmp_path, catalog, "缓存 k:report")} == {"docs/乙报告.md"}
    assert {r["path"] for r in search_docs(tmp_path, catalog, "缓存 s:")} == {"docs/丙说明.md"}
    assert {r["path"] for r in search_docs(tmp_path, catalog, "缓存 a:")} >= {"docs/archive/丁报告.md"}


def test_search_no_hit(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = scan(tmp_path, include_archive=True)
    assert search_docs(tmp_path, catalog, "不存在的词zzz") == []


def test_context_snippet_marker(tmp_path: Path) -> None:
    _fixture(tmp_path)
    snippet = context_snippet(tmp_path, "docs/甲专项计划.md", 5, radius=2)
    assert ">>    5 |" in snippet and "缓存" in snippet


def test_real_repo_search_smoke() -> None:
    catalog = scan(REPO_ROOT, include_archive=True)
    results = search_docs(REPO_ROOT, catalog, "缓存", limit=50)
    assert len(results) >= 5
    assert all(r["line_no"] >= 0 for r in results)


def test_bookshelf_search_pilot(tmp_path: Path) -> None:
    _fixture(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            query = app.query_one("#query")
            assert query.display is True
            query.value = "缓存 k:report"
            await pilot.press("enter")
            await pilot.pause()
            assert app.mode == "search"
            assert [r["path"] for r in app.search_results] == ["docs/乙报告.md"]
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert app.mode == "shelf"
            assert len(app.entries) == 3  # 书架恢复（不含归档）

    asyncio.run(run())
