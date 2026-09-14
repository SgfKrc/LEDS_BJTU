"""PATCH-06 馆藏统计测试：聚合 / 渲染 / Pilot / 真实冒烟。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.catalog import scan  # noqa: E402
from patchouli.stats import compute_stats, render_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> None:
    _write(tmp_path, "docs/甲专项计划.md", "# 甲\n> 状态：现行\n> 更新日期：2026-09-01\n票 `EX-A-01` `EX-B-01`。\n[乙报告](乙报告.md)\n")
    _write(tmp_path, "docs/乙报告.md", "# 乙\n> 状态：现行\n> 更新日期：2026-08-01\n票 `EX-A-01`。\n")
    _write(tmp_path, "docs/丙说明.md", "# 丙\n")  # 缺状态行
    _write(tmp_path, "docs/archive/丁报告.md", "# 丁\n> 状态：历史参考\n")


def test_compute_stats(tmp_path: Path) -> None:
    _fixture(tmp_path)
    stats = compute_stats(scan(tmp_path, include_archive=True))
    assert stats["total"] == 4 and stats["archived"] == 1 and stats["active"] == 3
    assert stats["status_with"] == 3 and stats["status_missing"] == 1
    assert stats["by_kind"]["special-plan"] == 1 and stats["by_kind"]["report"] == 2
    assert stats["ticket_count"] == 2
    assert dict(stats["top_tickets"])["EX-A-01"] == 2
    assert stats["link_total"] == 1 and stats["no_links"] == 3


def test_render_stats_contains_fields(tmp_path: Path) -> None:
    _fixture(tmp_path)
    text = render_stats(compute_stats(scan(tmp_path, include_archive=True)))
    for needle in ("馆藏统计", "总量: 4", "75.0%", "EX-A-01(2)", "互链: 总 1"):
        assert needle in text, needle


def test_real_repo_stats_smoke() -> None:
    stats = compute_stats(scan(REPO_ROOT, include_archive=True))
    assert stats["total"] >= 70
    assert stats["status_missing"] >= 1
    assert stats["ticket_count"] >= 20
    assert stats["archived"] >= 8


def test_bookshelf_stats_key_pilot(tmp_path: Path) -> None:
    _fixture(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            assert app.stats_shown is True
            assert "馆藏统计" in app.stats_text
            await pilot.press("s")
            await pilot.pause()
            assert app.stats_shown is False
            # 选择联动重置
            await pilot.press("s")
            await pilot.pause()
            assert app.stats_shown is True
            await pilot.press("down")
            await pilot.pause()
            assert app.stats_shown is False

    asyncio.run(run())
