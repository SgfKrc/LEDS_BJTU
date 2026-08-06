"""T9 聊天页终端走查（headless 自动化，T9.2/T9.6 布局验收）。

用 Textual pilot 在多个终端尺寸下驱动真实聊天页（fixture 模式），
逐项断言布局、中文/emoji/代码块渲染、流式完成、命令与取消，并导出
SVG 截图到 build/tui-chat/ 作为验收证据。

运行:
    python scripts/tui_chat_walkthrough.py
    python scripts/tui_chat_walkthrough.py --sizes 80x24,120x30

尺寸矩阵：80×24（基准）、120×30（宽屏）、60×18（窄终端下限）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from textual.widgets import Header  # noqa: E402

from tui_chat import ChatInput, TuiChatApp  # noqa: E402

FIXTURE = ROOT / "fixtures" / "chat_interactive_fixture.sse"
OUT_DIR = ROOT / "build" / "tui-chat"

DEFAULT_SIZES = [(80, 24), (120, 30), (60, 18)]


def _write_fixture(path: Path, events: list) -> None:
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _screenshot(app, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(app.export_screenshot(), encoding="utf-8")


async def _run_scenario(size: tuple, results: list, slow_fixture: str) -> None:
    width, height = size
    app = TuiChatApp(fixture=str(FIXTURE))
    checks: list = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    async with app.run_test(size=size) as pilot:
        # ---- 1. 布局 ----
        header = app.query_one(Header)
        transcript = app.query_one("#transcript")
        chat_input = app.query_one(ChatInput)
        status = app.query_one("#status-bar")
        check("Header 占满宽度", header.region.width == width)
        check("Transcript 占主区域",
              transcript.region.height >= height // 2)
        check("输入区高度受限", chat_input.region.height <= 8)
        check("状态栏贴底", status.region.y >= height - 2)

        # ---- 2. 中文 + emoji + 流式渲染 ----
        app.query_one(ChatInput).text = "你好 T9 走查 🚀"
        await pilot.press("enter")
        await pilot.pause(1.5)
        md = "".join(w._markdown for w in app.query("Markdown"))
        check("user 中文+emoji 渲染", "你好 T9 走查 🚀" in md)
        check("assistant 流式全文", "QLH T9 聊天页" in md)
        check("代码块保留", "来自 SSE fixture" in md)
        check("生成完成回到空闲", not app.is_generating)
        bar = str(status.render())
        check("状态栏显示 fixture 与空闲", "fixture" in bar and "空闲" in bar)
        _screenshot(app, OUT_DIR / f"complete-{width}x{height}.svg")

        # ---- 3. 命令 ----
        app.query_one(ChatInput).text = "/route local_only"
        await pilot.press("enter")
        await pilot.pause(0.2)
        check("路由命令生效", app.route == "local_only")
        check("状态栏反映路由", "route:local" in str(status.render()))

        app.query_one(ChatInput).text = "/help"
        await pilot.press("enter")
        await pilot.pause(0.2)
        check("帮助文本出现", "可用命令" in
              "".join(w._markdown for w in app.query("Markdown")))

        app.query_one(ChatInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause(0.3)
        check("新会话重置", app.session_id is None)

        # ---- 4. 取消临界区（慢 fixture）----
        app2 = TuiChatApp(fixture=slow_fixture)
        async with app2.run_test(size=size) as pilot2:
            app2.query_one(ChatInput).text = "长生成"
            await pilot2.press("enter")
            await pilot2.pause(0.2)  # 部分 token 到达
            await pilot2.press("ctrl+c")
            await pilot2.pause(0.6)
            md2 = "".join(w._markdown for w in app2.query("Markdown"))
            check("取消有提示", "正在停止生成" in md2 or "已取消" in md2)
            check("partial 保留", "T0 " in md2)
            check("未生成部分不出现", "T29" not in md2)
            check("取消后空闲", not app2.is_generating)
            _screenshot(app2, OUT_DIR / f"cancelled-{width}x{height}.svg")

        # ---- 5. 退出 ----
        app.query_one(ChatInput).text = "/quit"
        await pilot.press("enter")
        await pilot.pause(0.2)
        check("退出指令触发", not app.is_running)

    results.append((size, checks))


async def _main(sizes: list) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        slow = Path(tmp) / "slow.sse"
        events = [{"start": True, "generation_id": "gen_slow"}]
        for i in range(30):
            events.append({"token": f"T{i} "})
        _write_fixture(slow, events)

        results: list = []
        for size in sizes:
            await _run_scenario(size, results, str(slow))

    total = failed = 0
    for size, checks in results:
        print(f"\n=== 终端 {size[0]}x{size[1]} ===")
        for name, ok, detail in checks:
            total += 1
            if not ok:
                failed += 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
                  + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n走查汇总: {total - failed}/{total} 通过, 失败 {failed}")
    print(f"截图证据: {OUT_DIR}")
    return 1 if failed else 0


def _parse_sizes(raw: str) -> list:
    sizes = []
    for item in raw.split(","):
        w, _, h = item.strip().partition("x")
        sizes.append((int(w), int(h)))
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="T9 聊天页终端走查")
    parser.add_argument("--sizes", default="",
                        help="逗号分隔尺寸，如 80x24,120x30（默认全矩阵）")
    args = parser.parse_args()
    sizes = _parse_sizes(args.sizes) if args.sizes else DEFAULT_SIZES
    return asyncio.run(_main(sizes))


if __name__ == "__main__":
    sys.exit(main())
