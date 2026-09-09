from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi.testclient import TestClient

from harness_workbench.adapters.base import AdapterCapabilities, AdapterModel, AdapterResponse, StreamChunk
from harness_workbench.api_layer import create_app
from harness_workbench.rag import RagStore
from harness_workbench.session import SessionStore


class FakeChatAdapter:
    def capabilities(self):
        return AdapterCapabilities(backend="fixture", model_ids=("fixture",))

    def models(self):
        return (AdapterModel("fixture"),)

    def complete(self, request):
        return AdapterResponse("cmpl_fixture", request.model, "MCP chat fixture")

    def stream(self, request) -> Iterable[StreamChunk]:
        yield StreamChunk("stream_fixture", request.model, {"content": "fixture"}, "stop")

    def close(self):
        pass


def _call(client: TestClient, name: str, arguments: Mapping[str, Any]):
    response = client.post("/v1/mcp/call", json={"name": name, "arguments": dict(arguments)})
    assert response.status_code == 200, response.text
    return response.json()


def test_api_bridge_manifest_and_calls_share_injected_harness_stores(tmp_path):
    app = create_app(
        FakeChatAdapter(),
        session_store=SessionStore(tmp_path / "sessions.sqlite3"),
        rag_store=RagStore(tmp_path / "rag.sqlite3"),
    )
    client = TestClient(app)

    manifest = client.get("/v1/mcp/manifest")
    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["schema"] == "qlh.mcp_http_manifest.v1"
    assert {item["name"] for item in payload["tools"]} >= {"chat", "session_create", "rag_search"}
    assert payload["external_mcp"]["configuration_only"] is True
    assert payload["transports"]["stdio"]["command"] == "python -m harness_workbench.mcp_server"

    created = _call(client, "session_create", {"title": "frontend bridge"})
    assert created["result"]["content"][0]["type"] == "text"

    added = _call(client, "rag_add_source", {"source_ref": "fixture.md", "title": "Fixture", "text": "frontend MCP bridge"})
    assert '"chunk_count"' in added["result"]["content"][0]["text"]
    searched = _call(client, "rag_search", {"query": "bridge"})
    assert '"hits"' in searched["result"]["content"][0]["text"]

    chat = _call(client, "chat", {"messages": [{"role": "user", "content": "hello"}]})
    assert "MCP chat fixture" in chat["result"]["content"][0]["text"]


def test_api_bridge_json_rpc_accepts_initialize_and_hides_notification_response():
    client = TestClient(create_app(FakeChatAdapter()))
    initialized = client.post("/v1/mcp/rpc", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialized.status_code == 200
    assert initialized.json()["result"]["capabilities"]["experimental"]["qlh"]["external_mcp"]["configuration_only"] is True
    notification = client.post("/v1/mcp/rpc", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notification.status_code == 204
