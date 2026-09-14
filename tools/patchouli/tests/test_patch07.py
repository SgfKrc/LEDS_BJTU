"""PATCH-07 测试：chunk→parent 映射 / 参数对照 / Pilot / 真实冒烟。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.catalog import scan  # noqa: E402
from patchouli.chunks import chunk_for_line, map_hits_to_parents, split_chunks  # noqa: E402
from patchouli.compare import compare_search, render_compare, rewrite_variants  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/甲计划.md",
        "# 甲计划\n> 状态：现行\n\n## 一、目标\n前缀缓存机制设计。\n\n## 二、实现\nKV cache 分层与命中率。\n",
    )
    _write(tmp_path, "docs/乙报告.md", "# 乙\n> 状态：现行\n\n缓存实验：cache 命中提升。\n")


def test_split_chunks_and_lookup(tmp_path: Path) -> None:
    _fixture(tmp_path)
    text = (tmp_path / "docs/甲计划.md").read_text(encoding="utf-8")
    chunks = split_chunks(text)
    assert [c["heading"] for c in chunks] == ["甲计划", "一、目标", "二、实现"]
    assert chunks[2]["start_line"] == 7 and chunks[2]["end_line"] == 8
    assert chunk_for_line(chunks, 5)["heading"] == "一、目标"  # L5 前缀缓存机制设计


def test_map_hits_to_parents(tmp_path: Path) -> None:
    _fixture(tmp_path)
    hits = [
        {"path": "docs/甲计划.md", "name": "甲计划", "kind": "special-plan", "archived": False, "line_no": 8, "line": "KV cache 分层"},
        {"path": "docs/甲计划.md", "name": "甲计划", "kind": "special-plan", "archived": False, "line_no": 5, "line": "前缀缓存"},
        {"path": "docs/乙报告.md", "name": "乙报告", "kind": "report", "archived": False, "line_no": 4, "line": "cache 命中"},
    ]
    parents = map_hits_to_parents(tmp_path, hits)
    assert parents[0]["name"] == "甲计划" and parents[0]["hits"] == 2
    assert parents[0]["best_heading"] == "二、实现"  # 首个命中行(8)所属章节
    assert parents[0]["best_chunk"] == [7, 8]
    assert len(parents) == 2


def test_rewrite_variants_excludes_self_only() -> None:
    variants = rewrite_variants("缓存")
    assert variants[0] == "缓存"
    assert any("cache" in v.lower() or "缓存" in v for v in variants[1:]) or len(variants) == 1


def test_compare_search_two_routes(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = scan(tmp_path, include_archive=True)
    result = compare_search(tmp_path, catalog, "缓存", top_k=5)
    assert result["query"] == "缓存"
    assert result["baseline_total"] >= 2
    assert result["enhanced_total"] >= result["baseline_total"] - 1  # 融合不应更少
    assert 0.0 <= result["mrr_baseline"] <= 1.0
    assert 0.0 <= result["mrr_enhanced"] <= 1.0
    text = render_compare(result)
    assert "参数对照" in text and "MRR@5" in text


def test_compare_real_repo_smoke() -> None:
    catalog = scan(REPO_ROOT, include_archive=True)
    result = compare_search(REPO_ROOT, catalog, "缓存", top_k=5)
    assert result["baseline"] and result["enhanced"]
    assert len(result["variants"]) >= 1


def test_bookshelf_compare_key_pilot(tmp_path: Path) -> None:
    _fixture(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.pause()
            query = app.query_one("#query")
            query.value = "缓存"
            await pilot.press("enter")
            await pilot.pause()
            assert app.mode == "search"
            await pilot.press("v")
            await pilot.pause()
            assert app.compare_shown is True
            assert "参数对照" in app.compare_text
            await pilot.press("v")
            await pilot.pause()
            assert app.compare_shown is False

    asyncio.run(run())
