"""JSON-RPC MCP server facade for the harness workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .builtin import HarnessMCPDependencies, register_builtin_tools
from .contracts import (
    MCP_JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    MCP_SERVER_SCHEMA,
    MCPToolResult,
    text_result,
)
from .registry import MCPToolError, ToolRegistry


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    name: str = "qlh-harness"
    version: str = "1.0.0"

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


class MCPServer:
    """Handle MCP lifecycle and tool requests without owning a transport."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        dependencies: HarnessMCPDependencies | None = None,
        include_builtin: bool = True,
        info: MCPServerInfo | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        if include_builtin:
            register_builtin_tools(self.registry, dependencies)
        self.info = info or MCPServerInfo()

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return one JSON-RPC response, or ``None`` for a notification."""

        request_id = request.get("id", _MISSING) if isinstance(request, Mapping) else _MISSING
        if request_id is not _MISSING and (isinstance(request_id, bool) or not isinstance(request_id, (str, int, float))):
            request_id = None
        try:
            method, params, is_notification = self._validate_request(request)
            if method == "notifications/initialized":
                return None
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = self._tools_list(params)
            elif method == "tools/call":
                result = self._tools_call(params)
            else:
                return self._error(request_id, -32601, "method not found") if not is_notification else None
            return None if is_notification else self._result(request_id, result)
        except _MCPRequestFailure as exc:
            return None if request_id is _MISSING else self._error(request_id, exc.rpc_code, exc.message)
        except Exception:
            return None if request_id is _MISSING else self._error(request_id, -32603, "internal MCP server error")

    def handle_json(self, payload: str | bytes) -> dict[str, Any] | None:
        if isinstance(payload, str):
            if len(payload.encode("utf-8")) > 1 * 1024 * 1024:
                return self._error(None, -32600, "MCP message exceeds the size limit")
        elif isinstance(payload, bytes) and len(payload) > 1 * 1024 * 1024:
            return self._error(None, -32600, "MCP message exceeds the size limit")
        try:
            request = json.loads(payload)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return self._error(None, -32700, "invalid JSON")
        if not isinstance(request, Mapping):
            return self._error(None, -32600, "invalid MCP request")
        return self.handle(request)

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise _MCPRequestFailure(-32602, "initialize params must be an object")
        protocol = params.get("protocolVersion")
        if protocol is not None and (not isinstance(protocol, str) or not protocol):
            raise _MCPRequestFailure(-32602, "protocolVersion is invalid")
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "experimental": {
                    "qlh": {
                        "schema": MCP_SERVER_SCHEMA,
                        "builtin_tools": True,
                        "external_mcp": {
                            "supported": True,
                            "real_connections": False,
                            "configuration_only": True,
                        },
                    }
                },
            },
            "serverInfo": self.info.as_dict(),
        }

    def _tools_list(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise _MCPRequestFailure(-32602, "tools/list params must be an object")
        if set(params) - {"cursor"}:
            raise _MCPRequestFailure(-32602, "tools/list contains unknown params")
        if params.get("cursor") not in {None, ""}:
            raise _MCPRequestFailure(-32602, "cursor pagination is not supported")
        return {"tools": [tool.as_dict() for tool in self.registry.list_tools()]}

    def _tools_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping) or set(params) - {"name", "arguments"}:
            raise _MCPRequestFailure(-32602, "tools/call params are invalid")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise _MCPRequestFailure(-32602, "tool name is required")
        try:
            value = self.registry.call(name, params.get("arguments", {}))
        except MCPToolError as exc:
            return text_result(
                {
                    "schema": "qlh.mcp_tool_result.v1",
                    "error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable},
                },
                is_error=True,
            ).as_dict()
        return _coerce_tool_result(value).as_dict()

    @staticmethod
    def _validate_request(request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], bool]:
        if not isinstance(request, Mapping) or request.get("jsonrpc") != MCP_JSONRPC_VERSION:
            raise _MCPRequestFailure(-32600, "invalid MCP request")
        has_id = "id" in request
        request_id = request.get("id", _MISSING)
        if has_id and (isinstance(request_id, bool) or not isinstance(request_id, (str, int, float))):
            raise _MCPRequestFailure(-32600, "MCP request id is invalid")
        method = request.get("method")
        if not isinstance(method, str) or not method or len(method) > 128:
            raise _MCPRequestFailure(-32600, "MCP method is invalid")
        params = request.get("params", {})
        if not isinstance(params, Mapping):
            raise _MCPRequestFailure(-32602, "MCP params must be an object")
        return method, params, not has_id

    @staticmethod
    def _result(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "result": dict(result)}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "error": {"code": code, "message": message}}


class _MCPRequestFailure(Exception):
    def __init__(self, rpc_code: int, message: str) -> None:
        self.rpc_code = rpc_code
        self.message = message
        super().__init__(message)


class _Missing:
    pass


_MISSING = _Missing()


def _coerce_tool_result(value: Any) -> MCPToolResult:
    if isinstance(value, MCPToolResult):
        return value
    if isinstance(value, Mapping) and "content" in value:
        content = value.get("content")
        if isinstance(content, list) and content:
            try:
                return MCPToolResult(
                    tuple(dict(item) for item in content),
                    is_error=bool(value.get("isError", False)),
                    structured_content=value.get("structuredContent") if isinstance(value.get("structuredContent"), Mapping) else None,
                )
            except (TypeError, ValueError):
                pass
    return text_result(value)


__all__ = ["MCPServer", "MCPServerInfo"]
