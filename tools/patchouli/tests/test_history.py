"""PATCH-05 流通记录测试：git log 引擎 / fail-soft / Pilot / 真实冒烟。"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.history import file_history, render_history  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(tmp_path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=t@t", *args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def _git_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "甲专项计划.md").write_text("# 甲\n> 状态：现行\n第一版\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "初始提交")
    (docs / "甲专项计划.md").write_text("# 甲\n> 状态：现行\n第二版（更新）\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "更新正文")


def test_file_history_two_commits(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    result = file_history(tmp_path, "docs/甲专项计划.md")
    assert result["ok"] is True, result.get("error")
    commits = result["commits"]
    assert len(commits) == 2
    assert commits[0]["subject"] == "更新正文"  # 新→旧
    assert commits[0]["added"] >= 1 and commits[0]["deleted"] >= 1
    assert len(commits[0]["sha"]) == 10


def test_file_history_not_repo(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("x", encoding="utf-8")
    result = file_history(tmp_path, "x.md")
    assert result["ok"] is False


def test_render_history_branches() -> None:
    ok = {"ok": True, "commits": [{"sha": "abc1234567", "date": "2026-09-14", "author": "t", "subject": "提交", "added": 3, "deleted": 1}], "error": None}
    text = render_history(ok, "docs/a.md")
    assert "流通记录" in text and "abc1234567" in text and "+3/-1" in text
    bad = render_history({"ok": False, "commits": [], "error": "boom"}, "docs/a.md")
    assert "不可用" in bad and "boom" in bad


def test_real_repo_history_smoke() -> None:
    result = file_history(REPO_ROOT, "docs/总体下一步计划.md", limit=5)
    assert result["ok"] is True, result.get("error")
    assert len(result["commits"]) >= 1


def test_bookshelf_history_key_pilot(tmp_path: Path) -> None:
    _git_repo(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("h")
            for _ in range(40):
                await pilot.pause(0.05)
                if app.history_shown:
                    break
            assert app.history_shown is True, "历史 worker 未完成"
            assert "流通记录" in app.history_text and "更新正文" in app.history_text
            await pilot.press("h")  # 返回文档预览
            await pilot.pause(0.1)
            assert app.history_shown is False

    asyncio.run(run())
