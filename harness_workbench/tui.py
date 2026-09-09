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


def _stream_chat(host: str, payload: dict[str, Any]) -> str:
    request = urllib.request.Request(
        host.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event, dict) and event.get("error", {}).get("message"):
                    raise RuntimeError(str(event["error"]["message"]))
                choices = event.get("choices", []) if isinstance(event, dict) else []
                delta = choices[0].get("delta", {}).get("content", "") if choices else ""
                if delta:
                    chunks.append(str(delta))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, IndexError, AttributeError, TypeError) as exc:
        raise RuntimeError(f"harness stream unavailable: {exc}") from exc
    return "".join(chunks)


def create_app(*, host: str, model: str) -> Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static
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
        ListItem:focus { background: #142737; color: #f4f7fb; }
        Input:focus { border: solid #ff5bd7; }
        Button:focus { border: solid #63e6ff; }
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
                    yield Button("+ 新建会话", id="new-session")
                    yield ListView(id="session-list")
                    yield Label("RAG / ASSETS", classes="muted")
                    yield Static("等待能力探测", id="utility-status")
                with Vertical(id="main"):
                    yield Static("CHECKING · harness API", id="status")
                    yield Static("QLH Harness Workbench\n\n等待一条消息。", id="transcript")
                    yield Input(placeholder="输入消息，回车发送…", id="composer")
            yield Footer()

        def on_mount(self) -> None:
            self._session_id: str | None = None
            self._sessions: list[dict[str, Any]] = []
            self._probe()

        def _probe(self) -> None:
            status = self.query_one("#status", Static)
            try:
                health = _request_json(host, "/healthz")
                status.update(f"ONLINE · {health.get('backend', 'harness api')}")
                self._probe_utilities()
                self._load_sessions()
            except RuntimeError:
                status.update("FIXTURE · API 未连接，发送消息只显示离线提示")
                self.query_one("#utility-status", Static).update("RAG / ASSETS · API unavailable")

        def _probe_utilities(self) -> None:
            utility = self.query_one("#utility-status", Static)
            try:
                rag = _request_json(host, "/v1/rag/health")
                rag_status = f"RAG {rag.get('backend', 'unknown')} / {rag.get('chunks', 0)} chunks"
            except RuntimeError:
                rag_status = "RAG unavailable"
            try:
                image = _request_json(host, "/v1/images/capabilities")
                image_status = "TXT2IMG ready" if image.get("runtime_available") and image.get("supports_txt2img") else "TXT2IMG blocked"
            except RuntimeError:
                image_status = "TXT2IMG unavailable"
            utility.update(f"{rag_status}\n{image_status}")

        def _load_sessions(self) -> None:
            try:
                payload = _request_json(host, "/v1/sessions?owner_scope=local&limit=50")
                self._sessions = [item for item in payload.get("sessions", []) if isinstance(item, dict)]
                session_list = self.query_one("#session-list", ListView)
                session_list.clear()
                for item in self._sessions:
                    session_list.append(ListItem(Label(str(item.get("title", "New session"))), name=str(item.get("session_id", ""))))
                if self._sessions:
                    self._select_session(str(self._sessions[0].get("session_id", "")))
            except (RuntimeError, AttributeError, TypeError):
                self._sessions = []

        def _select_session(self, session_id: str) -> None:
            if not session_id:
                return
            try:
                payload = _request_json(host, f"/v1/sessions/{session_id}?owner_scope=local")
                self._session_id = session_id
                lines = []
                for message in payload.get("messages", []):
                    role = str(message.get("role", "system")).upper()
                    lines.append(f"{role}\n{message.get('content', '')}")
                self.query_one("#transcript", Static).update("\n\n".join(lines) or "QLH Harness Workbench\n\n等待一条消息。")
            except (RuntimeError, AttributeError, TypeError):
                self.query_one("#status", Static).update("OFFLINE · 会话加载失败")

        def on_list_view_selected(self, event: Any) -> None:
            if getattr(event.list_view, "id", None) == "session-list":
                self._select_session(str(getattr(event.item, "name", "")))

        def on_button_pressed(self, event: Any) -> None:
            if getattr(event.button, "id", None) != "new-session":
                return
            try:
                created = _request_json(host, "/v1/sessions", {"owner_scope": "local", "title": "New session"})
                self._session_id = str(created.get("session_id", ""))
                self._load_sessions()
                self.query_one("#transcript", Static).update("QLH Harness Workbench\n\n等待一条消息。")
            except RuntimeError:
                self.query_one("#status", Static).update("FIXTURE · API 未连接，无法新建会话")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return
            event.input.value = ""
            transcript = self.query_one("#transcript", Static)
            current = str(transcript.renderable)
            try:
                if self._session_id:
                    _request_json(host, f"/v1/sessions/{self._session_id}/messages", {"owner_scope": "local", "role": "user", "content": text})
                answer = _stream_chat(host, {"model": model, "messages": [{"role": "user", "content": text}]})
                if self._session_id and answer:
                    _request_json(host, f"/v1/sessions/{self._session_id}/messages", {"owner_scope": "local", "role": "assistant", "content": answer})
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
