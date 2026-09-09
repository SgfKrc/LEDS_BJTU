"""Built-in and declarative external MCP tool registry."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import (
    ExternalMCPServerConfig,
    MCPContractError,
    MCPToolResult,
    MAX_DESCRIPTION,
    MAX_TOOL_NAME,
    _ABSOLUTE_PATH,
    _SECRET_WORD,
    text_result,
    validate_json_schema,
    validate_tool_arguments,
)


class MCPToolError(RuntimeError):
    """Expected tool failure returned as an MCP ``isError`` result."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


ToolHandler = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler
    read_only: bool = True
    destructive: bool = False
    open_world: bool = False
    source: str = "builtin"
    capability: str = "configured"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not 1 <= len(self.name) <= MAX_TOOL_NAME or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_./-]*", self.name):
            raise MCPContractError("invalid_tool_name", "MCP tool name is invalid")
        if not isinstance(self.description, str) or not self.description.strip() or len(self.description) > MAX_DESCRIPTION:
            raise MCPContractError("invalid_tool_description", "MCP tool description is invalid")
        if not callable(self.handler):
            raise MCPContractError("invalid_tool_handler", "MCP tool handler is not callable")
        if self.source not in {"builtin", "external"}:
            raise MCPContractError("invalid_tool_source", "MCP tool source is invalid")
        if not isinstance(self.read_only, bool) or not isinstance(self.destructive, bool) or not isinstance(self.open_world, bool):
            raise MCPContractError("invalid_tool_annotations", "MCP tool annotations are invalid")
        object.__setattr__(self, "input_schema", validate_json_schema(self.input_schema))
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "openWorldHint": self.open_world,
            },
            "_meta": {
                "qlh": {
                    "source": self.source,
                    "capability": self.capability,
                    **dict(self.metadata),
                }
            },
        }
        return value


class ExternalMCPClient(Protocol):
    def list_tools(self) -> Mapping[str, Any] | Sequence[Mapping[str, Any]]:
        ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class ExternalMCPMount:
    config: ExternalMCPServerConfig
    client: ExternalMCPClient


class ToolRegistry:
    """Own tool names and isolate external MCP tools behind a namespace."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._mounts: dict[str, ExternalMCPMount] = {}
        self._declarations: dict[str, ExternalMCPServerConfig] = {}

    def register(self, definition: ToolDefinition, *, replace: bool = False) -> ToolDefinition:
        if definition.name in self._tools and not replace:
            raise MCPContractError("tool_collision", "MCP tool name is already registered")
        self._tools[definition.name] = definition
        return definition

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def call(self, name: str, arguments: Any = None) -> Any:
        definition = self._tools.get(name)
        if definition is None:
            raise MCPToolError("unknown_tool", "MCP tool is not registered")
        try:
            prepared = validate_tool_arguments(definition.input_schema, arguments)
            return definition.handler(prepared)
        except MCPToolError:
            raise
        except MCPContractError as exc:
            raise MCPToolError(exc.code, str(exc)) from exc
        except KeyError as exc:
            raise MCPToolError("not_found", "requested harness resource was not found") from exc
        except (TypeError, ValueError) as exc:
            raise MCPToolError("invalid_arguments", str(exc)) from exc
        except Exception as exc:
            raise MCPToolError("tool_execution_failed", "MCP tool execution failed") from exc

    def mount_external(self, config: ExternalMCPServerConfig, client: ExternalMCPClient) -> tuple[ToolDefinition, ...]:
        """Discover a fixture/injected MCP server; never creates a process or socket."""

        existing = self._declarations.get(config.server_id)
        if existing is not None and existing != config:
            raise MCPToolError("external_collision", "external MCP server_id is already declared")
        self.declare_external(config, replace=existing is not None)
        if not config.enabled:
            raise MCPToolError("external_disabled", "external MCP declaration is disabled")
        if not hasattr(client, "list_tools") or not hasattr(client, "call_tool"):
            raise MCPToolError("external_client_invalid", "external MCP client does not implement the contract")
        try:
            discovered = client.list_tools()
        except Exception as exc:
            raise MCPToolError("external_discovery_failed", "external MCP tool discovery failed", retryable=True) from exc
        if isinstance(discovered, Mapping):
            discovered = discovered.get("tools")
        if not isinstance(discovered, Sequence) or isinstance(discovered, (str, bytes)) or len(discovered) > 64:
            raise MCPToolError("external_discovery_invalid", "external MCP discovery returned no tools")
        namespaced: list[ToolDefinition] = []
        for raw in discovered:
            if not isinstance(raw, Mapping):
                raise MCPToolError("external_discovery_invalid", "external MCP tool declaration is invalid")
            remote_name = raw.get("name")
            description = raw.get("description", "External MCP tool")
            schema = raw.get("inputSchema")
            if not isinstance(remote_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", remote_name):
                raise MCPToolError("external_discovery_invalid", "external MCP tool name is invalid")
            if not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION:
                raise MCPToolError("external_discovery_invalid", "external MCP tool description is invalid")
            try:
                schema = validate_json_schema(schema)
            except MCPContractError as exc:
                raise MCPToolError("external_discovery_invalid", str(exc)) from exc
            name = f"{config.server_id}/{remote_name}"
            definition = ToolDefinition(
                name=name,
                description=f"External MCP server {config.server_id}: {description}",
                input_schema=schema,
                handler=self._external_handler(config.server_id, remote_name),
                source="external",
                capability="declared_external",
                metadata={"server_id": config.server_id, "remote_name": remote_name, "transport": config.transport},
            )
            namespaced.append(definition)
        if any(definition.name in self._tools for definition in namespaced):
            raise MCPToolError("tool_collision", "external MCP tool name is already registered")
        for definition in namespaced:
            self.register(definition)
        self._mounts[config.server_id] = ExternalMCPMount(config, client)
        return tuple(namespaced)

    def declare_external(self, config: ExternalMCPServerConfig, *, replace: bool = False) -> ExternalMCPServerConfig:
        """Record an external endpoint without opening a process or network connection."""

        if config.server_id in self._declarations and not replace:
            raise MCPContractError("external_collision", "external MCP server_id is already declared")
        self._declarations[config.server_id] = config
        return config

    def external_configurations(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._declarations[name].as_dict() for name in sorted(self._declarations))

    def _external_handler(self, server_id: str, remote_name: str) -> ToolHandler:
        def handler(arguments: Mapping[str, Any]) -> Any:
            mount = self._mounts.get(server_id)
            if mount is None:
                raise MCPToolError("external_unavailable", "external MCP server is not mounted", retryable=True)
            try:
                result = mount.client.call_tool(remote_name, arguments)
            except MCPToolError:
                raise
            except Exception as exc:
                raise MCPToolError("external_call_failed", "external MCP tool call failed", retryable=True) from exc
            return _validate_external_result(result)

        return handler


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MCPContractError("invalid_tool_metadata", "MCP tool metadata must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MCPContractError("invalid_tool_metadata", "MCP tool metadata is not JSON serializable") from exc
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise MCPContractError("invalid_tool_metadata", "MCP tool metadata is too large")
    _check_metadata(value)
    return dict(value)


def _check_metadata(value: Any, *, key: str = "") -> None:
    if _SECRET_WORD.search(key):
        raise MCPContractError("secret_metadata_forbidden", "MCP tool metadata cannot contain secret fields")
    if isinstance(value, str):
        if _ABSOLUTE_PATH.match(value):
            raise MCPContractError("absolute_path_forbidden", "MCP tool metadata cannot contain absolute paths")
        return
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _check_metadata(child_value, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child_value in value:
            _check_metadata(child_value, key=key)


def _validate_external_result(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise MCPToolError("external_result_invalid", "external MCP result must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise MCPToolError("external_result_invalid", "external MCP result is not JSON serializable") from exc
    if len(encoded.encode("utf-8")) > 1 * 1024 * 1024:
        raise MCPToolError("external_result_too_large", "external MCP result is too large")
    if "error" in value:
        error = value["error"]
        if not isinstance(error, Mapping) or not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
            raise MCPToolError("external_result_invalid", "external MCP error object is invalid")
        raise MCPToolError(str(error["code"])[:64], str(error["message"])[:512], retryable=bool(error.get("retryable", False)))
    if "content" in value:
        content = value["content"]
        if not isinstance(content, list) or not content:
            raise MCPToolError("external_result_invalid", "external MCP content is invalid")
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "text" or not isinstance(block.get("text"), str) or len(block["text"]) > 64 * 1024:
                raise MCPToolError("external_result_invalid", "external MCP content block is invalid")
    return dict(value)


__all__ = [
    "ExternalMCPClient",
    "ExternalMCPMount",
    "MCPToolError",
    "ToolDefinition",
    "ToolRegistry",
]
