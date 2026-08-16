"""文档维护 Agent M2.3：loopback OpenAI 兼容协议、回退与缓存编排。"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_llm import DocAgentConfig, ProviderPlan, ProviderSpec  # noqa: E402
from doc_maintenance_runtime import run_llm_judgements  # noqa: E402


class _Server:
    def __init__(self, handler_factory):
        self.requests: list[tuple[dict, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                headers = dict(self.headers.items())
                outer.requests.append((headers, body))
                status, response = handler_factory(body)
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def _audit(repo: Path) -> dict:
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "x.md").write_text(
        "> 状态：规划\nAPI_KEY=sk-should-not-leak\n"
        "位置 src/private/x.py，链接 https://example.com/private\n",
        encoding="utf-8",
    )
    return {"docs": [{
        "doc": "docs/x.md", "status_line": "> 状态：规划",
        "sha256": "a" * 64,
        "findings": [{"rule": "R1", "level": "warn"}],
    }]}


def _response(body: dict):
    doc_ref = body["messages"][1]["content"]
    doc_ref = json.loads(doc_ref)["input"]["doc"]
    return 200, {
        "choices": [{"message": {"content": json.dumps({
            "doc": doc_ref, "judgement": "accurate", "confidence": 0.9,
            "suggestion": "状态行已足够准确",
        })}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }


def test_loopback_provider_uses_sanitized_payload_and_cache(tmp_path):
    repo = tmp_path / "repo"
    audit = _audit(repo)
    with _Server(_response) as fake:
        config = DocAgentConfig(
            requested_provider="deepseek", provider_explicit=True,
            deepseek_base_url=fake.base_url, deepseek_model="fake",
            deepseek_api_key="sk-secret-not-in-report",
        )
        plan = ProviderPlan((ProviderSpec("deepseek", fake.base_url, "fake", "sk-secret-not-in-report"),))
        first = run_llm_judgements(
            audit, repo, config, tmp_path / "cache.sqlite", plan=plan,
        )
        second = run_llm_judgements(
            audit, repo, config, tmp_path / "cache.sqlite", plan=plan,
        )
    assert first["cost"] == {"cache_hits": 0, "provider_calls": 1, "needs_review": 0}
    assert second["cost"] == {"cache_hits": 1, "provider_calls": 0, "needs_review": 0}
    headers, body = fake.requests[0]
    assert headers["Authorization"] == "Bearer sk-secret-not-in-report"
    assert headers["User-Agent"] == "QLH-DocAgent/0.1"
    sent = body["messages"][1]["content"]
    for forbidden in ("docs/x.md", "src/private", "example.com", "sk-should-not-leak"):
        assert forbidden not in sent
    report = json.dumps(first)
    assert "sk-secret-not-in-report" not in report


def test_provider_failure_falls_back_to_ollama_then_needs_review(tmp_path):
    repo = tmp_path / "repo"
    audit = _audit(repo)
    with _Server(_response) as fake:
        config = DocAgentConfig(requested_provider="deepseek", provider_explicit=True)
        plan = ProviderPlan((
            ProviderSpec("deepseek", "http://127.0.0.1:1/v1", "broken", "sk-secret"),
            ProviderSpec("ollama", fake.base_url, "local"),
        ))
        result = run_llm_judgements(
            audit, repo, config, tmp_path / "cache.sqlite", plan=plan,
        )
    assert result["cost"]["provider_calls"] == 2
    assert result["judgements"][0]["provider"] == "ollama"


def test_no_available_provider_stays_mechanical_and_marks_review(tmp_path):
    repo = tmp_path / "repo"
    result = run_llm_judgements(
        _audit(repo), repo,
        DocAgentConfig(requested_provider="ollama", provider_explicit=True),
        tmp_path / "cache.sqlite", plan=ProviderPlan(()),
    )
    assert result["cost"] == {"cache_hits": 0, "provider_calls": 0, "needs_review": 1}
    assert result["judgements"][0]["judgement"] == "needs_review"
