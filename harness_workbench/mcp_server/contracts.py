"""Small, dependency-free MCP contracts used by the harness server."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


MCP_JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"
MCP_SERVER_SCHEMA = "qlh.mcp_server.v1"
MCP_EXTERNAL_SCHEMA = "qlh.mcp_external_server.v1"
MCP_TOOL_RESULT_SCHEMA = "qlh.mcp_tool_result.v1"
MAX_JSON_BYTES = 1 * 1024 * 1024
MAX_TOOL_NAME = 96
MAX_DESCRIPTION = 4_000
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_COMMAND = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SECRET_WORD = re.compile(r"(?:api[_-]?key|authorization|password|secret|token|private[_-]?key|cookie)", re.I)
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")


class MCPContractError(ValueError):
    """Raised when an MCP contract cannot be admitted."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


def _reject_path(value: Any, *, key: str = "") -> None:
    """Reject path-like configuration values before they reach a client."""

    if isinstance(value, str):
        if _ABSOLUTE_PATH.match(value) or ("\\" in value and key.lower().endswith(("path", "command"))):
            raise MCPContractError("absolute_path_forbidden", "MCP configuration cannot contain absolute paths")
        return
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _reject_path(child_value, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child_value in value:
            _reject_path(child_value, key=key)


def _json_value(value: Any, *, limit: int = MAX_JSON_BYTES) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MCPContractError("json_value_invalid", "MCP value is not JSON serializable") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise MCPContractError("json_value_too_large", "MCP value exceeds the size limit")
    return value


def validate_json_schema(schema: Any) -> dict[str, Any]:
    """Validate the deliberately small JSON Schema subset used by tools."""

    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise MCPContractError("invalid_tool_schema", "MCP tool inputSchema must be an object schema")
    allowed = {"type", "properties", "required", "additionalProperties", "description"}
    if set(schema) - allowed:
        raise MCPContractError("invalid_tool_schema", "MCP tool schema contains unknown fields")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise MCPContractError("invalid_tool_schema", "MCP tool schema properties and required are invalid")
    if schema.get("additionalProperties", False) is not False:
        raise MCPContractError("invalid_tool_schema", "MCP tool schemas must reject unknown arguments")
    if any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in properties):
        raise MCPContractError("invalid_tool_schema", "MCP property names are invalid")
    if any(not isinstance(name, str) or name not in properties for name in required):
        raise MCPContractError("invalid_tool_schema", "MCP required properties are invalid")
    for name, child in properties.items():
        _validate_schema_node(child, field_name=name)
    description = schema.get("description", "")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION:
        raise MCPContractError("invalid_tool_schema", "MCP schema description is too long")
    _reject_path(schema)
    return dict(schema)


def _validate_schema_node(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping) or value.get("type") not in {"string", "integer", "number", "boolean", "array", "object"}:
        raise MCPContractError("invalid_tool_schema", f"MCP property schema is invalid: {field_name}")
    allowed = {"type", "description", "maxLength", "minimum", "maximum", "items", "enum", "pattern", "properties", "required", "additionalProperties"}
    if set(value) - allowed:
        raise MCPContractError("invalid_tool_schema", f"MCP property schema contains unknown fields: {field_name}")
    description = value.get("description", "")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION:
        raise MCPContractError("invalid_tool_schema", f"MCP property description is invalid: {field_name}")
    if "enum" in value and (not isinstance(value["enum"], list) or len(value["enum"]) > 32):
        raise MCPContractError("invalid_tool_schema", f"MCP property enum is invalid: {field_name}")
    if value["type"] == "array":
        _validate_schema_node(value.get("items"), field_name=field_name + "[]")
    if value["type"] == "object":
        child = dict(value)
        child["type"] = "object"
        child.setdefault("properties", {})
        child.setdefault("required", [])
        child.setdefault("additionalProperties", False)
        validate_json_schema(child)


def validate_tool_arguments(schema: Mapping[str, Any], arguments: Any) -> dict[str, Any]:
    """Validate tool arguments without depending on a JSON-schema package."""

    schema = validate_json_schema(schema)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise MCPContractError("invalid_params", "MCP tool arguments must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    unknown = set(arguments) - set(properties)
    if unknown:
        raise MCPContractError("invalid_params", "MCP tool arguments contain unknown fields")
    missing = set(required) - set(arguments)
    if missing:
        raise MCPContractError("invalid_params", "MCP tool arguments are missing required fields")
    for name, value in arguments.items():
        _validate_argument(value, properties[name], name=name)
    return dict(arguments)


def _validate_argument(value: Any, schema: Mapping[str, Any], *, name: str) -> None:
    kind = schema.get("type")
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, Mapping),
    }[kind]
    if not valid:
        raise MCPContractError("invalid_params", f"MCP argument has the wrong type: {name}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        raise MCPContractError("invalid_params", f"MCP argument is too long: {name}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise MCPContractError("invalid_params", f"MCP argument is not finite: {name}")
        if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
            raise MCPContractError("invalid_params", f"MCP argument is outside the allowed range: {name}")
    if isinstance(value, str) and "pattern" in schema:
        try:
            matched = re.fullmatch(str(schema["pattern"]), value)
        except re.error as exc:
            raise MCPContractError("invalid_tool_schema", f"MCP property pattern is invalid: {name}") from exc
        if matched is None:
            raise MCPContractError("invalid_params", f"MCP argument format is invalid: {name}")
    if "enum" in schema and value not in schema["enum"]:
        raise MCPContractError("invalid_params", f"MCP argument is not an allowed value: {name}")
    if isinstance(value, list) and "items" in schema:
        for child in value:
            _validate_argument(child, schema["items"], name=name + "[]")
    if isinstance(value, Mapping) and schema.get("type") == "object":
        child_schema = dict(schema)
        child_schema.setdefault("properties", {})
        child_schema.setdefault("required", [])
        child_schema.setdefault("additionalProperties", False)
        validate_tool_arguments(child_schema, value)


@dataclass(frozen=True, slots=True)
class ExternalMCPServerConfig:
    """Declarative third-party MCP connection point; no connection is opened."""

    server_id: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    endpoint: str | None = None
    env_keys: tuple[str, ...] = ()
    enabled: bool = True
    schema: str = MCP_EXTERNAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MCP_EXTERNAL_SCHEMA:
            raise MCPContractError("invalid_schema", "unsupported external MCP schema")
        if not isinstance(self.server_id, str) or not _IDENTIFIER.fullmatch(self.server_id):
            raise MCPContractError("invalid_server_id", "external MCP server_id is invalid")
        if self.transport not in {"stdio", "sse"}:
            raise MCPContractError("invalid_transport", "external MCP transport must be stdio or sse")
        if self.transport == "stdio":
            if not isinstance(self.command, str) or not _COMMAND.fullmatch(self.command):
                raise MCPContractError("invalid_command", "stdio command must be a bare executable name")
            if self.endpoint is not None:
                raise MCPContractError("invalid_endpoint", "stdio MCP declaration cannot contain an endpoint")
        else:
            if not isinstance(self.endpoint, str) or not self.endpoint.startswith(("http://", "https://")):
                raise MCPContractError("invalid_endpoint", "SSE endpoint must use HTTP(S)")
            from urllib.parse import urlsplit

            parsed = urlsplit(self.endpoint)
            if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise MCPContractError("invalid_endpoint", "SSE endpoint cannot contain credentials, query, or fragment")
            if self.command is not None:
                raise MCPContractError("invalid_command", "SSE MCP declaration cannot contain a command")
        if not isinstance(self.args, (list, tuple)) or len(self.args) > 32 or any(not isinstance(item, str) or not item or len(item) > 512 for item in self.args):
            raise MCPContractError("invalid_args", "external MCP args are invalid")
        if any(_SECRET_WORD.search(item) for item in self.args):
            raise MCPContractError("secret_declaration_forbidden", "secret-bearing external MCP args are not accepted")
        if not isinstance(self.env_keys, (list, tuple)) or len(self.env_keys) > 32 or any(not isinstance(item, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", item) for item in self.env_keys):
            raise MCPContractError("invalid_env_keys", "external MCP env_keys are invalid")
        if any(_SECRET_WORD.search(item) for item in self.env_keys):
            raise MCPContractError("secret_declaration_forbidden", "secret-bearing environment keys are not accepted")
        if not isinstance(self.enabled, bool):
            raise MCPContractError("invalid_enabled", "external MCP enabled must be boolean")
        _reject_path(self.args)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "server_id": self.server_id,
            "transport": self.transport,
            "args": list(self.args),
            "env_keys": list(self.env_keys),
            "enabled": self.enabled,
        }
        if self.command is not None:
            value["command"] = self.command
        if self.endpoint is not None:
            value["endpoint"] = self.endpoint
        return value


@dataclass(frozen=True, slots=True)
class MCPToolResult:
    """MCP tool result with text content and optional structured content."""

    content: tuple[Mapping[str, Any], ...]
    is_error: bool = False
    structured_content: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.content or any(not isinstance(item, Mapping) or item.get("type") != "text" or not isinstance(item.get("text"), str) for item in self.content):
            raise MCPContractError("invalid_tool_result", "MCP tool result content must contain text blocks")
        _json_value([dict(item) for item in self.content])
        if self.structured_content is not None:
            _json_value(self.structured_content)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"content": [dict(item) for item in self.content], "isError": bool(self.is_error)}
        if self.structured_content is not None:
            value["structuredContent"] = dict(self.structured_content)
        return value


def text_result(value: Any, *, is_error: bool = False, code: str | None = None) -> MCPToolResult:
    payload: Any = value
    if code is not None:
        payload = {"schema": MCP_TOOL_RESULT_SCHEMA, "error": {"code": code, "message": str(value)}}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _json_value(encoded)
    return MCPToolResult(({"type": "text", "text": encoded},), is_error=is_error, structured_content=payload if isinstance(payload, Mapping) else None)


__all__ = [
    "ExternalMCPServerConfig",
    "MAX_JSON_BYTES",
    "MCPContractError",
    "MCP_EXTERNAL_SCHEMA",
    "MCP_JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "MCP_SERVER_SCHEMA",
    "MCPToolResult",
    "text_result",
    "validate_json_schema",
    "validate_tool_arguments",
]
