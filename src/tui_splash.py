"""Koakuma 启动动画（零依赖 ANSI 实现）——灰蓝配色。

风格仿 `tools/docagent/patchouli/splash.py`（PATCHOULI 启动屏）：
半块像素字（``▀``：fg 占上半、bg 占下半）+ 打字机逐列显现 + 扫描线 + 窄色条 spinner。

与 patchouli 版的差异（按 2026-09-16 需求）：
* 标题 **Koakumix**（原 PATCHOULI）；
* 副标题 **leds-bjtu**（原中古高地德语签名）；
* 配色由紫改为**灰蓝**；
* 状态文案保留 **"少女祈祷中……"**；
* **纯标准库 + ANSI**（主仓 TUI 的零依赖边界；patchouli/Koakumix 用 Textual，不能搬进主仓）。

硬约束（沿用 patchouli 的纪律）：
* 与数据加载**并行**，加载完即关（**不延长启动**）；
* ``--no-splash`` / ``splash=False`` 可关；
* 任意键跳过；
* 非 TTY（管道/重定向）自动不播。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, TypeVar, Union

T = TypeVar("T")

# ---------------------------------------------------------------- 配色（灰蓝）

def _fg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


COLOR_TOP = (17, 23, 34)      # #111722 近黑灰蓝（像素上半）
COLOR_BOTTOM = (61, 79, 102)  # #3d4f66 灰蓝（像素下半 = 主体色）
COLOR_SCAN = (143, 168, 196)  # #8fa8c4 亮灰蓝（扫描线）
COLOR_EDGE = (232, 238, 245)  # #e8eef5 近白（下沿描边）
COLOR_BAR = (61, 79, 102)     # 状态条底色（灰蓝）
COLOR_SIGN = (143, 168, 196)  # 副标题（亮灰蓝）
COLOR_DIM = (110, 126, 148)   # 次要文本

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
UPPER = "\u2580"              # ▀ 上半块：fg 占上、bg 占下

# ------------------------------------------------- 5x3 像素字形（# 实心）

GLYPHS: dict[str, list[str]] = {
    "K": ["#.#", "#.#", "##.", "#.#", "#.#"],
    "O": ["###", "#.#", "#.#", "#.#", "###"],
    "A": ["###", "#.#", "###", "#.#", "#.#"],
    "U": ["#.#", "#.#", "#.#", "#.#", "###"],
    "M": ["#.#", "###", "###", "#.#", "#.#"],
    "I": ["###", ".#.", ".#.", ".#.", "###"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"],
}

WORD = "Koakuma".upper()
SUBTITLE = "leds-bjtu"
STATUS_TEXT = "少女祈祷中……"
GRID_H = 5
SPINNER = ("/", "-", "\\", "|")

TICK_SECONDS = 0.08
TYPING_COLS_PER_TICK = 3.75   # 打字速度（与 patchouli 一致）
SCAN_STEP_TICKS = 2           # 每 2 tick 扫描线下移一行
HOLD_TICKS = 10               # 扫描完毕后的静止 hold
SUB_CHARS_PER_TICK = 2        # 副标题流式速度
SUB_CURSOR = "▌"


def _build_grid(word: str = WORD) -> list[list[str]]:
    width = len(word) * 4 - 1
    grid = [[" "] * width for _ in range(GRID_H)]
    for i, ch in enumerate(word):
        glyph = GLYPHS.get(ch)
        if glyph is None:
            continue
        for r in range(GRID_H):
            for c in range(3):
                if glyph[r][c] == "#":
                    grid[r][i * 4 + c] = "F"
    return grid


GRID = _build_grid()
GRID_W = len(GRID[0])
GRID_ROWS = len(GRID)
LAST_ROW = GRID_ROWS - 1


def _row_colors(row: int, scan_row: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """返回 (上半色, 下半色)。"""
    if row == scan_row:
        return COLOR_SCAN, COLOR_BOTTOM
    if row == LAST_ROW:
        return COLOR_EDGE, COLOR_BOTTOM
    return COLOR_TOP, COLOR_BOTTOM


def render_logo(grid: list[list[str]], scan_row: int = -1, revealed_cols: int = 10**9) -> list[str]:
    """网格 → ANSI 行（RLE 合并同色；打字机 + 扫描线）。"""
    lines: list[str] = []
    for r in range(len(grid)):
        top, bottom = _row_colors(r, scan_row)
        style = _fg(top) + _bg(bottom)
        row = grid[r]
        parts: list[str] = []
        run = 0
        for c in range(len(row)):
            if row[c] == "F" and c < revealed_cols:
                run += 1
            else:
                if run:
                    parts.append(style + UPPER * run + RESET)
                    run = 0
                parts.append(" ")
        if run:
            parts.append(style + UPPER * run + RESET)
        lines.append("".join(parts))
    return lines


def _enable_vt() -> bool:
    """尽力启用 Windows 控制台 VT 处理；失败则返回 False（调用方应跳过动画）。"""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001
        return False


def supported(stream=None) -> bool:
    stream = stream or sys.stdout
    try:
        if not stream.isatty():
            return False
    except Exception:  # noqa: BLE001
        return False
    if os.environ.get("TERM", "").lower() in {"dumb", ""} and os.name != "nt":
        return False
    return _enable_vt()


class TuiSplash:
    """启动动画；`play(loader)` 边播动画边并行执行 `loader`，加载完即收尾关闭。"""

    def __init__(
        self,
        status: Union[str, Callable[[], str]] = STATUS_TEXT,
        *,
        min_show: float = 1.0,
        subtitle: str = SUBTITLE,
        stream=None,
    ) -> None:
        self.status_text = status
        self.subtitle = subtitle
        self.min_show = float(min_show)
        self.stream = stream or sys.stdout
        self._frame = 0
        self._t0: float | None = None
        self._typing_done_frame: int | None = None
        self._sub_done_frame: int | None = None
        self._loaded = False
        self._result: object = None
        self._error: BaseException | None = None
        self._worker: threading.Thread | None = None

    def _status(self) -> str:
        value = self.status_text() if callable(self.status_text) else self.status_text
        return str(value or STATUS_TEXT)

    # ---------------------------------------------------------------- 渲染

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _line_status(self) -> str:
        spin = SPINNER[self._frame % len(SPINNER)]
        bar = _bg(COLOR_BAR) + _fg((255, 255, 255))
        label = f"  {spin} {self._status()}    （任意键跳过动画）"
        return bar + BOLD + label + RESET

    def _anim_done(self) -> bool:
        if self._typing_done_frame is None or self._sub_done_frame is None:
            return False
        title_hold = (self._frame - self._typing_done_frame) >= (GRID_ROWS * SCAN_STEP_TICKS + HOLD_TICKS)
        sub_hold = (self._frame - self._sub_done_frame) >= HOLD_TICKS
        return title_hold and sub_hold

    def _render_frame(self) -> None:
        revealed = min(GRID_W, int(TYPING_COLS_PER_TICK * (self._frame + 2)))
        typing_done = revealed >= GRID_W
        if typing_done and self._typing_done_frame is None:
            self._typing_done_frame = self._frame

        sub_revealed = min(len(self.subtitle), SUB_CHARS_PER_TICK * self._frame)
        if sub_revealed >= len(self.subtitle) and self._sub_done_frame is None:
            self._sub_done_frame = self._frame
        sub_text = self.subtitle[:sub_revealed]
        if sub_revealed < len(self.subtitle):
            sub_text += SUB_CURSOR

        scan_row = -1
        if typing_done and self._typing_done_frame is not None:
            since = self._frame - self._typing_done_frame
            if since < GRID_ROWS * SCAN_STEP_TICKS:
                scan_row = (since // SCAN_STEP_TICKS) % GRID_ROWS

        lines = render_logo(GRID, scan_row, revealed)
        # 光标归位 + 逐行覆盖 + 清到屏底：不再全屏清屏，避免"追加式"终端里出现残影/重播观感。
        out = ["\x1b[H", "\x1b[2K\n"]
        for line in lines:
            out.append("   " + line + "\x1b[2K\n")
        out.append("\n" + "   " + _fg(COLOR_SIGN) + sub_text + RESET + "\x1b[2K\n")
        out.append("\n" + self._line_status() + "\x1b[2K\n")
        out.append("\x1b[J")
        self._write("".join(out))

    # ---------------------------------------------------------------- 主流程

    def play(self, loader: Callable[[], T], *, skip_on_key: bool = True) -> T:
        """播放动画的同时执行 `loader()`（在后台线程），返回其结果。

        加载完成且动画播完（且满足 ``min_show``）后收尾；`skip_on_key` 时任意键可跳过。
        """
        self._t0 = time.monotonic()

        def _run() -> None:
            try:
                self._result = loader()
            except BaseException as exc:  # noqa: BLE001 - 交回主线程
                self._error = exc
            finally:
                self._loaded = True

        worker = threading.Thread(target=_run, name="koakumix-splash-load", daemon=True)
        self._worker = worker
        worker.start()

        # 切到备用屏（alt screen）：退出时终端自动恢复进入前的画面，
        # 避免"清屏 + 残影"被误读为动画反复播放。
        self._write("\x1b[?1049h" + HIDE_CURSOR)
        skip_requested = False
        try:
            while True:
                self._frame += 1
                self._render_frame()
                if self._loaded and (skip_requested or self._anim_done()):
                    if (time.monotonic() - self._t0) >= self.min_show:
                        break
                if skip_on_key and not skip_requested and self._key_pressed():
                    # 只跳过视觉收尾，继续显示 spinner 直到 loader 完成，
                    # 防止用户在冷启动阶段看到无后端的 TUI。
                    skip_requested = True
                time.sleep(TICK_SECONDS)
        finally:
            # 退出备用屏并恢复光标：终端回到进入动画前的画面
            self._write(SHOW_CURSOR + "\x1b[?1049l")

        # 跳过只跳过视觉动画，不能绕过后端就绪条件；否则 TUI 会在
        # API 仍冷启动时进入，产生一屏误导性的连接错误。
        worker.join()
        if self._error is not None:
            raise self._error
        return self._result  # type: ignore[return-value]

    @staticmethod
    def _key_pressed() -> bool:
        """非阻塞探测按键（Windows: msvcrt；POSIX: select）。"""
        try:
            if os.name == "nt":
                import msvcrt

                if msvcrt.kbhit():
                    msvcrt.getch()
                    return True
                return False
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                sys.stdin.read(1)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False


def play_splash(
    loader: Callable[[], T],
    *,
    status: Union[str, Callable[[], str]] = STATUS_TEXT,
    min_show: float = 1.0,
) -> T:
    """便捷入口：环境不支持动画时直接执行 `loader()`。"""
    if not supported():
        return loader()
    splash = TuiSplash(status=status, min_show=min_show)
    try:
        return splash.play(loader)
    except BaseException:  # noqa: BLE001 - 动画失败不应阻断启动
        # loader 的异常必须透传，不能重复执行启动副作用；只有动画本身
        # 失败时等待已启动的 loader，不能重新执行启动副作用。
        if splash._error is not None:
            raise splash._error
        if splash._worker is not None:
            splash._worker.join()
            if splash._error is not None:
                raise splash._error
            return splash._result  # type: ignore[return-value]
        return loader()
