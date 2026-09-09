from pathlib import Path

import pytest

from harness_workbench.tui import build_parser, create_app


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "harness_workbench" / "ui_react"


def test_tui_parser_and_optional_app_contract():
    args = build_parser().parse_args(["--host", "http://127.0.0.1:8090", "--model", "QW1.8B"])
    assert args.host.endswith(":8090")
    assert args.model == "QW1.8B"
    pytest.importorskip("textual")
    app = create_app(host=args.host, model=args.model)
    assert app.TITLE == "QLH Harness Workbench"


def test_tui_uses_persistent_session_contract():
    tui = (ROOT / "harness_workbench" / "tui.py").read_text(encoding="utf-8")
    assert "/v1/sessions?owner_scope=local" in tui
    assert "/v1/sessions/{self._session_id}/messages" in tui
    assert '"stream": True' in tui
    assert "new-session" in tui


def test_react_ui_is_independent_and_uses_non_green_cyber_accent():
    package = (UI_ROOT / "package.json").read_text(encoding="utf-8")
    styles = (UI_ROOT / "src" / "styles.css").read_text(encoding="utf-8")
    app = (UI_ROOT / "src" / "App.tsx").read_text(encoding="utf-8")
    data = (UI_ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    assert '"react"' in package and '"vite"' in package
    assert "#63e6ff" in styles and "#ff5bd7" in styles
    assert "#c7ff3d" not in styles.lower()
    assert "/v1/chat/completions" in data
    assert "streamChat" in data
    assert "/v1/sessions" in data
    assert "/v1/rag/search" in data
    assert "/v1/images/generations" in data
    assert "/v1/mcp/manifest" in data
    assert "/v1/mcp/call" in data
    assert "citation-context" in app
    assert "MCP 控制面" in app
    assert "mcp-tool-list" in app
    assert "TXT2IMG BLOCKED" in app
    assert "skip-link" in app
    assert "aria-current" in app
    assert "prefers-reduced-motion" in styles
    assert (UI_ROOT / "scripts" / "visual_smoke.mjs").exists()
    assert '"visual:smoke"' in package
    assert "RAG" in app
