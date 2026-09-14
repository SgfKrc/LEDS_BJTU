"""PATCH-04 编目诊断测试：摘要解析 / fail-soft / Pilot 集成 / 真实冒烟。"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.catalog_diag import find_docagent, render_diag, run_scan, summarize_scan  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE = {
    "run_ts": "2026-09-14T22:00:00",
    "rules": {"R1": "完成未收口", "R5": "状态行缺失"},
    "docs": [
        {"doc": "docs/a.md", "findings": [{"rule": "R3", "level": "info", "message": "更新日期滞后"}]},
        {"doc": "docs/b.md", "findings": [{"rule": "R5", "level": "warn", "message": "缺状态行"}]},
        {"doc": "docs/c.md", "findings": []},
    ],
}


def test_summarize_scan_counts() -> None:
    summary = summarize_scan(SAMPLE)
    assert summary["ok"] is True
    assert summary["docs"] == 3
    assert summary["by_rule"] == {"R3": 1, "R5": 1}
    assert summary["by_level"] == {"info": 1, "warn": 1}
    assert len(summary["findings"]) == 2


def test_run_scan_missing_runner(tmp_path: Path) -> None:
    result = run_scan(tmp_path)
    assert result["ok"] is False and "docagent not found" in result["error"]


def test_run_scan_parses_subprocess(tmp_path: Path) -> None:
    (tmp_path / "tools" / "docagent").mkdir(parents=True)
    (tmp_path / "tools" / "docagent" / "run.py").write_text("# stub\n", encoding="utf-8")
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(SAMPLE), stderr="")
    with patch("patchouli.catalog_diag.subprocess.run", return_value=fake):
        result = run_scan(tmp_path)
    assert result["ok"] is True and result["docs"] == 3


def test_run_scan_timeout_fail_soft(tmp_path: Path) -> None:
    (tmp_path / "tools" / "docagent").mkdir(parents=True)
    (tmp_path / "tools" / "docagent" / "run.py").write_text("# stub\n", encoding="utf-8")
    with patch("patchouli.catalog_diag.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
        result = run_scan(tmp_path)
    assert result["ok"] is False and "subprocess" in result["error"]


def test_render_diag_branches() -> None:
    ok_text = render_diag(summarize_scan(SAMPLE))
    assert "编目诊断" in ok_text and "R5" in ok_text
    bad_text = render_diag({"ok": False, "error": "boom"})
    assert "失败" in bad_text and "boom" in bad_text


def test_find_docagent_real_repo() -> None:
    assert find_docagent(REPO_ROOT) is not None


def test_real_scan_smoke() -> None:
    """真实 docagent scan（主仓，~0.5s）。"""
    result = run_scan(REPO_ROOT)
    assert result["ok"] is True, result.get("error")
    assert result["docs"] >= 70
    assert isinstance(result["by_rule"], dict)


def test_bookshelf_diag_key_pilot(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "甲专项计划.md").write_text("# 甲\n> 状态：现行\n", encoding="utf-8")

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        with patch("patchouli.catalog_diag.run_scan", return_value=summarize_scan(SAMPLE)):
            async with app.run_test() as pilot:
                await pilot.press("c")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if app.diag is not None:
                        break
                assert app.diag is not None, "worker 未完成"
                assert "编目诊断" in app.diag_text
                await pilot.press("c")  # 回到文档预览
                await pilot.pause()
                assert app.diag is None

    asyncio.run(run())
