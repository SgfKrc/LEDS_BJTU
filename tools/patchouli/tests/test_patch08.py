"""PATCH-08 测试：splash 并行加载/跳过 + 配置检测/编辑/持久化。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textual.widgets import TextArea  # noqa: E402

from patchouli.bookshelf import BookshelfApp  # noqa: E402
from patchouli.config_check import env_status, env_template, render_hint, write_env  # noqa: E402
from patchouli.splash import SplashScreen  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _docs(tmp_path: Path) -> None:
    _write(tmp_path, "docs/甲专项计划.md", "# 甲\n> 状态：现行\n")
    _write(tmp_path, "docs/乙报告.md", "# 乙\n> 状态：现行\n")


def test_splash_forced_loads_and_dismisses(tmp_path: Path) -> None:
    _docs(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path, splash=True)
        async with app.run_test() as pilot:
            for _ in range(60):
                await pilot.pause(0.05)
                if len(app.screen_stack) == 1 and app.catalog:
                    break
            assert app.catalog, "并行加载未完成"
            assert len(app.screen_stack) == 1, "splash 未在加载完成后关闭"
            assert len(app.entries) == 2  # 主界面可用

    asyncio.run(run())


def test_splash_any_key_skips(tmp_path: Path) -> None:
    _docs(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path, splash=True)
        async with app.run_test() as pilot:
            assert isinstance(app.screen, SplashScreen)
            await pilot.press("x")  # 任意键跳过
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1

    asyncio.run(run())


def test_headless_default_no_splash(tmp_path: Path) -> None:
    _docs(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)  # splash=None → 非 TTY 自动关闭
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1
            assert not isinstance(app.screen, SplashScreen)

    asyncio.run(run())


def test_env_status_and_template(tmp_path: Path) -> None:
    status = env_status(tmp_path)
    assert status["exists"] is False and status["keys"] == []
    _write(tmp_path, ".env.docagent", "# c\nDOCAGENT_PROVIDER=ollama\nDOCAGENT_DEEPSEEK_API_KEY=sk-x\n")
    status = env_status(tmp_path)
    assert status["exists"] is True
    assert status["keys"] == ["DOCAGENT_PROVIDER", "DOCAGENT_DEEPSEEK_API_KEY"]
    assert "sk-x" not in str(status)  # 脱敏：值绝不出现
    assert "DOCAGENT_PROVIDER" in env_template(tmp_path)  # 无模板时内置默认


def test_write_env_backup_and_escape(tmp_path: Path) -> None:
    _docs(tmp_path)
    first = write_env(tmp_path, "DOCAGENT_PROVIDER=ollama\n")
    assert first["ok"] and first["backup"] is None
    second = write_env(tmp_path, "DOCAGENT_PROVIDER=opencode\n")
    assert second["ok"] and second["backup"] and Path(second["backup"]).is_file()
    assert (tmp_path / ".env.docagent").read_text(encoding="utf-8") == "DOCAGENT_PROVIDER=opencode\n"
    assert (tmp_path / ".env.docagent.bak").read_text(encoding="utf-8") == "DOCAGENT_PROVIDER=ollama\n"


def test_render_hint_missing_and_ready(tmp_path: Path) -> None:
    hint = render_hint(env_status(tmp_path))
    assert "未配置" in hint and "按 e" in hint
    _write(tmp_path, ".env.docagent", "DOCAGENT_PROVIDER=ollama\n")
    ready = render_hint(env_status(tmp_path))
    assert "就绪" in ready and "DOCAGENT_PROVIDER" in ready


def test_config_editor_pilot_saves(tmp_path: Path) -> None:
    _docs(tmp_path)

    async def run() -> None:
        app = BookshelfApp(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("e")
            for _ in range(30):
                await pilot.pause(0.05)
                if len(app.screen_stack) > 1:
                    break
            assert len(app.screen_stack) > 1, "editor 未打开"
            editor_area = app.screen.query_one("#editor-area", TextArea)
            assert "DOCAGENT_PROVIDER" in editor_area.text  # 预填模板
            editor_area.text = "DOCAGENT_PROVIDER=opencode\n"
            await pilot.press("ctrl+s")
            for _ in range(20):
                await pilot.pause(0.1)
                if (tmp_path / ".env.docagent").is_file():
                    break
            assert (tmp_path / ".env.docagent").read_text(encoding="utf-8") == "DOCAGENT_PROVIDER=opencode\n"
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1

    asyncio.run(run())
