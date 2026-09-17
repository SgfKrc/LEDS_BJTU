"""统一 TUI 启动动画测试：动画不能绕过后端就绪，也不能重复 loader。"""

import io
import time

import pytest

import sys

sys.path.insert(0, "src")

import tui_splash


def test_splash_runs_loader_and_renders_dynamic_status(monkeypatch):
    monkeypatch.setattr(tui_splash, "TICK_SECONDS", 0.001)
    monkeypatch.setattr(
        tui_splash.TuiSplash, "_key_pressed", staticmethod(lambda: False),
    )
    output = io.StringIO()
    splash = tui_splash.TuiSplash(
        status=lambda: "等待 API 健康检查",
        min_show=0,
        stream=output,
    )

    assert splash.play(lambda: "ready", skip_on_key=False) == "ready"
    rendered = output.getvalue()
    assert "等待 API 健康检查" in rendered
    assert "\x1b[?1049h" in rendered
    assert "\x1b[?1049l" in rendered


def test_splash_skip_keeps_waiting_for_loader(monkeypatch):
    monkeypatch.setattr(tui_splash, "TICK_SECONDS", 0.001)
    keys = iter([True, False, False, False])
    monkeypatch.setattr(
        tui_splash.TuiSplash, "_key_pressed",
        staticmethod(lambda: next(keys, False)),
    )

    def slow_loader():
        time.sleep(0.03)
        return "backend-ready"

    splash = tui_splash.TuiSplash(min_show=0, stream=io.StringIO())
    started = time.monotonic()
    assert splash.play(slow_loader) == "backend-ready"
    assert time.monotonic() - started >= 0.02


def test_play_splash_does_not_repeat_loader_error(monkeypatch):
    monkeypatch.setattr(tui_splash, "supported", lambda: True)
    monkeypatch.setattr(tui_splash, "TICK_SECONDS", 0.001)
    monkeypatch.setattr(
        tui_splash.TuiSplash, "_key_pressed", staticmethod(lambda: False),
    )
    calls = []

    def loader():
        calls.append(1)
        raise RuntimeError("boot failed")

    with pytest.raises(RuntimeError, match="boot failed"):
        tui_splash.play_splash(loader, min_show=0)
    assert calls == [1]


def test_animation_write_failure_reuses_started_loader(monkeypatch):
    monkeypatch.setattr(tui_splash, "supported", lambda: True)
    calls = []

    def fail_write(_self, _text):
        raise OSError("terminal closed")

    monkeypatch.setattr(tui_splash.TuiSplash, "_write", fail_write)

    def loader():
        calls.append(1)
        return "backend-ready"

    assert tui_splash.play_splash(loader, min_show=0) == "backend-ready"
    assert calls == [1]
