from __future__ import annotations

import io
import json
import threading
import urllib.request

import pytest

from harness_workbench.mcp_server import (
    ExternalMCPServerConfig,
    HarnessMCPDependencies,
    MCPContractError,
    MCPServer,
    MCPToolError,
    StdioMCPTransport,
    ToolRegistry,
    create_harness_server,
)
from harness_workbench.mcp_server.transports import SSEMCPTransport
from harness_workbench.rag import RagStore
from harness_workbench.session import SessionStore


def _request(request_id, method, params=None):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}


def _text(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_stdio_initialize_lists_builtin_catalog_and_runs_session_and_rag(tmp_path):
    session_store = SessionStore(tmp_path / "sessions.sqlite3")
    rag_store = RagStore(tmp_path / "rag.sqlite3")
    server = create_harness_server(dependencies=HarnessMCPDependencies(session_store=session_store, rag_store=rag_store))

    listed = server.handle(_request(1, "tools/list"))
    names = {item["name"] for item in listed["result"]["tools"]}
    assert {"chat", "session_create", "rag_search", "web_search", "memory_add", "image_generate"} <= names
    assert listed["result"]["tools"][0]["_meta"]["qlh"]["source"] == "builtin"

    created = server.handle(_request(2, "tools/call", {"name": "session_create", "arguments": {"title": "MCP demo"}}))
    session_id = _text(created)["session_id"]
    assert session_id.startswith("sess_")

    added = server.handle(_request(3, "tools/call", {"name": "rag_add_source", "arguments": {"source_ref": "fixture.md", "title": "Fixture", "text": "MCP registry read path"}}))
    assert _text(added)["chunk_count"] >= 1
    searched = server.handle(_request(4, "tools/call", {"name": "rag_search", "arguments": {"query": "registry"}}))
    assert _text(searched)["hits"][0]["title"] == "Fixture"

    stream_in = io.StringIO("\n".join([
        json.dumps(_request(10, "initialize", {"protocolVersion": "2024-11-05"})),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps(_request(11, "tools/call", {"name": "session_create", "arguments": {}})),
    ]) + "\n")
    stream_out = io.StringIO()
    StdioMCPTransport(server, input_stream=stream_in, output_stream=stream_out).run()
    messages = [json.loads(line) for line in stream_out.getvalue().splitlines()]
    assert [message["id"] for message in messages] == [10, 11]
    assert messages[0]["result"]["serverInfo"]["name"] == "qlh-harness"


def test_unconfigured_builtin_tools_are_listed_with_capability_notice_but_fail_closed():
    server = MCPServer()
    tools = server.handle(_request(1, "tools/list"))["result"]["tools"]
    web = next(item for item in tools if item["name"] == "web_search")
    assert web["_meta"]["qlh"] == {"source": "builtin", "capability": "unconfigured", "configured": False}
    response = server.handle(_request(2, "tools/call", {"name": "web_search", "arguments": {"query": "QLH"}}))
    assert response["result"]["isError"] is True
    assert _text(response)["error"]["code"] == "network_unavailable"


def test_json_rpc_invalid_requests_and_notifications_are_handled():
    server = MCPServer(include_builtin=False)
    assert server.handle({"jsonrpc": "2.0", "method": "ping"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 1, "method": "missing"})["error"]["code"] == -32601
    assert server.handle({"jsonrpc": "1.0", "id": 1, "method": "ping"})["error"]["code"] == -32600
    assert server.handle_json("not json")["error"]["code"] == -32700
    assert server.handle_json("[]")["error"]["code"] == -32600


class FakeExternalMCP:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return {"tools": [{
            "name": "lookup",
            "description": "fixture lookup",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "maxLength": 100}}, "required": ["query"], "additionalProperties": False},
        }]}

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return {"content": [{"type": "text", "text": "fixture result"}]}


def test_external_mcp_declaration_is_path_free_and_fixture_mount_discovers_and_calls():
    config = ExternalMCPServerConfig(server_id="fixture", transport="stdio", command="fake-mcp", args=("--fixture",))
    registry = ToolRegistry()
    registry.declare_external(config)
    assert registry.external_configurations() == (config.as_dict(),)
    fake = FakeExternalMCP()
    definitions = registry.mount_external(config, fake)
    assert [item.name for item in definitions] == ["fixture/lookup"]
    server = MCPServer(registry, include_builtin=False)
    listed = server.handle(_request(1, "tools/list"))["result"]["tools"]
    tool = next(item for item in listed if item["name"] == "fixture/lookup")
    assert tool["_meta"]["qlh"]["capability"] == "declared_external"
    response = server.handle(_request(2, "tools/call", {"name": "fixture/lookup", "arguments": {"query": "hello"}}))
    assert response["result"]["isError"] is False
    assert response["result"]["content"][0]["text"] == "fixture result"
    assert fake.calls == [("lookup", {"query": "hello"})]


def test_external_mcp_error_is_propagated_without_raw_config_or_secret_leakage():
    class Broken(FakeExternalMCP):
        def call_tool(self, name, arguments):
            raise MCPToolError("fixture_timeout", "fixture timed out", retryable=True)

    config = ExternalMCPServerConfig(server_id="fixture", transport="sse", endpoint="https://mcp.example/sse", env_keys=("MCP_MODE",))
    registry = ToolRegistry()
    registry.mount_external(config, Broken())
    server = MCPServer(registry, include_builtin=False)
    response = server.handle(_request(1, "tools/call", {"name": "fixture/lookup", "arguments": {"query": "hello"}}))
    error = _text(response)["error"]
    assert error == {"code": "fixture_timeout", "message": "fixture timed out", "retryable": True}
    encoded = json.dumps(config.as_dict())
    assert "password" not in encoded.lower()
    assert "C:" not in encoded


@pytest.mark.parametrize("kwargs", [
    {"server_id": "bad/id", "transport": "stdio", "command": "fake"},
    {"server_id": "fixture", "transport": "stdio", "command": "C:\\secret\\server.exe"},
    {"server_id": "fixture", "transport": "sse", "endpoint": "https://mcp.example/sse?token=secret"},
    {"server_id": "fixture", "transport": "stdio", "command": "fake", "env_keys": ("API_KEY",)},
])
def test_external_mcp_configuration_rejects_unsafe_values(kwargs):
    with pytest.raises(MCPContractError):
        ExternalMCPServerConfig(**kwargs)


def test_sse_transport_is_loopback_only_and_exposes_endpoint():
    server = MCPServer(include_builtin=False)
    transport = SSEMCPTransport(server, port=0)
    address = transport.start()
    try:
        assert address.startswith("http://127.0.0.1:")
        with pytest.raises(ValueError):
            SSEMCPTransport(server, host="0.0.0.0")
        with urllib.request.urlopen(address, timeout=2) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            first = response.readline().decode()
            second = response.readline().decode()
            assert first == "event: endpoint\n"
            endpoint = json.loads(second.removeprefix("data: "))
            assert endpoint.startswith("/messages?sessionId=mcp_")
    finally:
        transport.close()
