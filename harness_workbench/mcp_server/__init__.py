"""Offline-friendly MCP server and transport contracts for the harness."""

from __future__ import annotations

from .builtin import CHAT_SCHEMA, HarnessMCPDependencies, register_builtin_tools
from .contracts import (
    ExternalMCPServerConfig,
    MCPContractError,
    MCP_EXTERNAL_SCHEMA,
    MCP_JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_SCHEMA,
    MCPToolResult,
    text_result,
    validate_json_schema,
    validate_tool_arguments,
)
from .registry import ExternalMCPClient, ExternalMCPMount, MCPToolError, ToolDefinition, ToolRegistry
from .server import MCPServer, MCPServerInfo
from .transports import SSEMCPTransport, SSE_MCPTransport, StdioMCPTransport


MCPRegistry = ToolRegistry


def create_harness_server(
    *,
    dependencies: HarnessMCPDependencies | None = None,
    registry: ToolRegistry | None = None,
) -> MCPServer:
    """Create a server with the complete built-in harness tool catalog."""

    return MCPServer(registry, dependencies=dependencies)


__all__ = [
    "CHAT_SCHEMA",
    "ExternalMCPClient",
    "ExternalMCPMount",
    "ExternalMCPServerConfig",
    "HarnessMCPDependencies",
    "MCPContractError",
    "MCP_EXTERNAL_SCHEMA",
    "MCP_JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "MCP_SERVER_SCHEMA",
    "MCPRegistry",
    "MCPServer",
    "MCPServerInfo",
    "MCPToolError",
    "MCPToolResult",
    "SSEMCPTransport",
    "SSE_MCPTransport",
    "StdioMCPTransport",
    "ToolDefinition",
    "ToolRegistry",
    "create_harness_server",
    "register_builtin_tools",
    "text_result",
    "validate_json_schema",
    "validate_tool_arguments",
]
