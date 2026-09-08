"""Textual workbench shell sharing the harness /v1 terminology."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH harness Textual workbench")
    parser.add_argument("--host", default="http://127.0.0.1:8090", help="harness API base URL")
    parser.add_argument("--model", default="harness-default")
    return parser


def _request_json(host: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=8.0) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"harness API unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("harness API returned a non-object response")
    return value


def create_app(*, host: str, model: str) -> Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("Textual is required for the TUI; install the optional harness UI dependency") from exc

    class HarnessApp(App[None]):
        TITLE = "QLH Harness Workbench"
        CSS = """
        Screen { background: #080b12; color: #f4f7fb; }
        Header { background: #0e141e; color: #63e6ff; }
        Footer { background: #0e141e; color: #94a5b7; }
        #layout { height: 1fr; }
        #rail { width: 28; padding: 1 2; background: #0e141e; border: solid #26394b; }
        #main { width: 1fr; padding: 1 3; }
        #status { height: 3; color: #f0bd72; border-bottom: solid #26394b; }
        #transcript { height: 1fr; padding: 1 0; overflow-y: auto; }
        #composer { dock: bottom; height: 5; border: solid #63e6ff; background: #101a27; }
        ListItem { padding: 1; color: #94a5b7; }
        ListItem:hover { background: #142737; color: #63e6ff; }
        .message { padding: 1 0; }
        .muted { color: #647386; }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="layout"):
                with Vertical(id="rail"):
                    yield Label("WORKSPACE", classes="muted")
                    yield ListView(ListItem(Label("▸ 对话")), ListItem(Label("  知识库")), ListItem(Label("  资产")), ListItem(Label("  运行时")))
                    yield Label("SESSION", classes="muted")
                    yield Static("● 默认工作区\n○ fixture 示例", id="sessions")
                with Vertical(id="main"):
                    yield Static("CHECKING · harness API", id="status")
                    yield Static("QLH Harness Workbench\n\n等待一条消息。", id="transcript")
                    yield Input(placeholder="输入消息，回车发送…", id="composer")
            yield Footer()

        def on_mount(self) -> None:
            self._probe()

        def _probe(self) -> None:
            status = self.query_one("#status", Static)
            try:
                health = _request_json(host, "/healthz")
                status.update(f"ONLINE · {health.get('backend', 'harness api')}")
            except RuntimeError:
                status.update("FIXTURE · API 未连接，发送消息只显示离线提示")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return
            event.input.value = ""
            transcript = self.query_one("#transcript", Static)
            current = str(transcript.renderable)
            try:
                result = _request_json(host, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]})
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                transcript.update(current + f"\n\nYOU\n{text}\n\nHARNESS\n{answer}")
            except (RuntimeError, IndexError, AttributeError, TypeError):
                transcript.update(current + f"\n\nYOU\n{text}\n\nFIXTURE\nAPI 未连接，已记录输入但未调用模型。")

    return HarnessApp()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(host=args.host, model=args.model)
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "create_app", "main"]
