"""MCP tool definitions backed only by harness-owned components."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..image_workbench.contracts import ImageAdapter, ImageRequest
from ..image_workbench.assets import ImageAssetStore
from ..memory import MemoryStore
from ..rag import HybridRagRetriever, RagStore, build_context
from ..session import SessionStore
from ..tools.network import NetworkClient, NetworkToolError
from .registry import MCPToolError, ToolDefinition, ToolRegistry


CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Chat messages using the harness chat contract.",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "maxLength": 32},
                    "content": {"type": "string", "maxLength": 120000},
                },
                "required": ["role", "content"],
                "additionalProperties": False,
            },
        },
        "model": {"type": "string", "maxLength": 128},
        "stream": {"type": "boolean"},
    },
    "required": ["messages"],
    "additionalProperties": False,
}


def _scope_property() -> dict[str, Any]:
    return {"type": "string", "maxLength": 128, "pattern": "^[^\\/\\r\\n]+$"}


def _schema(properties: Mapping[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class HarnessMCPDependencies:
    """Optional stores and adapters used to populate the built-in tool set."""

    session_store: SessionStore | None = None
    rag_store: RagStore | None = None
    memory_store: MemoryStore | None = None
    network_client: NetworkClient | None = None
    image_adapter: ImageAdapter | None = None
    image_store: ImageAssetStore | None = None
    chat_handler: Callable[[Mapping[str, Any]], Any] | None = None
    rag_retriever: HybridRagRetriever | None = None


def register_builtin_tools(
    registry: ToolRegistry,
    dependencies: HarnessMCPDependencies | None = None,
) -> tuple[ToolDefinition, ...]:
    """Register the stable harness tool names and return their definitions."""

    deps = dependencies or HarnessMCPDependencies()
    definitions = (
        _definition("chat", "Run one chat completion through an injected harness adapter.", CHAT_SCHEMA, _chat(deps), configured=deps.chat_handler is not None, read_only=False),
        _definition("session_create", "Create a user-owned harness session.", _schema({"owner_scope": _scope_property(), "title": {"type": "string", "maxLength": 200}}, required=()), _session_create(deps), configured=deps.session_store is not None, read_only=False),
        _definition("sessions_list", "List sessions in one owner scope.", _schema({"owner_scope": _scope_property(), "limit": {"type": "integer", "minimum": 1, "maximum": 200}}), _sessions_list(deps), configured=deps.session_store is not None),
        _definition("session_get", "Read one session and its messages and assets.", _schema({"session_id": {"type": "string", "maxLength": 128}, "owner_scope": _scope_property()}, required=("session_id",)), _session_get(deps), configured=deps.session_store is not None),
        _definition("rag_search", "Search user-owned RAG chunks and return bounded context and citations.", _schema({"query": {"type": "string", "maxLength": 512}, "owner_scope": _scope_property(), "limit": {"type": "integer", "minimum": 1, "maximum": 50}, "max_chars": {"type": "integer", "minimum": 256, "maximum": 120000}, "max_tokens": {"type": "integer", "minimum": 1, "maximum": 100000}, "source_ids": {"type": "array", "items": {"type": "string", "maxLength": 128}}, "title_prefix": {"type": "string", "maxLength": 200}}, required=("query",)), _rag_search(deps), configured=deps.rag_store is not None or deps.rag_retriever is not None),
        _definition("rag_add_source", "Add a user-owned text source to the local RAG store.", _schema({"source_ref": {"type": "string", "maxLength": 4096}, "title": {"type": "string", "maxLength": 512}, "text": {"type": "string", "maxLength": 1_000_000}, "owner_scope": _scope_property(), "max_chars": {"type": "integer", "minimum": 256, "maximum": 120000}, "overlap_chars": {"type": "integer", "minimum": 0, "maximum": 12000}, "strategy": {"type": "string", "enum": ["fixed", "paragraph", "sentence"]}}, required=("source_ref", "title", "text")), _rag_add_source(deps), configured=deps.rag_store is not None, read_only=False),
        _definition("memory_search", "Search active long-term memory in one owner scope.", _schema({"query": {"type": "string", "maxLength": 512}, "owner_scope": _scope_property(), "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, required=("query",)), _memory_search(deps), configured=deps.memory_store is not None),
        _definition("memory_add", "Add an explicitly supplied fact, preference, or decision.", _schema({"kind": {"type": "string", "enum": ["fact", "preference", "decision"]}, "content": {"type": "string", "maxLength": 32000}, "owner_scope": _scope_property(), "source_session_id": {"type": "string", "maxLength": 128}, "source_message_id": {"type": "string", "maxLength": 128}, "valid_until": {"type": "number"}}, required=("kind", "content")), _memory_add(deps), configured=deps.memory_store is not None, read_only=False),
        _definition("memory_invalidate", "Invalidate a memory entry while retaining its audit record.", _schema({"entry_id": {"type": "string", "maxLength": 128}, "owner_scope": _scope_property(), "reason": {"type": "string", "maxLength": 512}}, required=("entry_id",)), _memory_invalidate(deps), configured=deps.memory_store is not None, read_only=False),
        _definition("memory_delete", "Soft-delete a memory entry with explicit confirmation.", _schema({"entry_id": {"type": "string", "maxLength": 128}, "owner_scope": _scope_property(), "confirm": {"type": "boolean"}, "reason": {"type": "string", "maxLength": 512}}, required=("entry_id", "confirm")), _memory_delete(deps), configured=deps.memory_store is not None, read_only=False, destructive=True),
        _definition("web_search", "Search through the configured, policy-gated web search provider.", _schema({"query": {"type": "string", "maxLength": 512}, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}, "allow_external": {"type": "boolean"}}, required=("query",)), _web_search(deps), configured=deps.network_client is not None, open_world=True),
        _definition("web_fetch", "Fetch one HTTPS resource through the configured SSRF-safe network client.", _schema({"url": {"type": "string", "maxLength": 4096}, "max_chars": {"type": "integer", "minimum": 1, "maximum": 32768}, "allow_external": {"type": "boolean"}}, required=("url",)), _web_fetch(deps), configured=deps.network_client is not None, open_world=True),
        _definition("image_capabilities", "Report the configured image adapter capabilities.", _schema({}), _image_capabilities(deps), configured=deps.image_adapter is not None),
        _definition("image_generate", "Generate one image through the configured image adapter.", _schema({"prompt": {"type": "string", "maxLength": 4000}, "negative_prompt": {"type": "string", "maxLength": 4000}, "model": {"type": "string", "maxLength": 128}, "width": {"type": "integer", "minimum": 64, "maximum": 768}, "height": {"type": "integer", "minimum": 64, "maximum": 768}, "steps": {"type": "integer", "minimum": 1, "maximum": 100}, "guidance_scale": {"type": "number", "minimum": 0, "maximum": 30}, "seed": {"type": "integer"}, "response_format": {"type": "string", "enum": ["b64_json", "url"]}, "user": {"type": "string", "maxLength": 128}}, required=("prompt",)), _image_generate(deps), configured=deps.image_adapter is not None, read_only=False),
    )
    for definition in definitions:
        registry.register(definition)
    return definitions


def _definition(name: str, description: str, schema: Mapping[str, Any], handler: Callable[[Mapping[str, Any]], Any], *, configured: bool, read_only: bool = True, open_world: bool = False, destructive: bool = False) -> ToolDefinition:
    return ToolDefinition(name, description, schema, handler, read_only=read_only, open_world=open_world, destructive=destructive, capability="configured" if configured else "unconfigured", metadata={"configured": configured})


def _require(value: Any, code: str, message: str) -> Any:
    if value is None:
        raise MCPToolError(code, message)
    return value


def _chat(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        callback = _require(deps.chat_handler, "chat_unavailable", "chat adapter is not configured")
        return callback(arguments)

    return handler


def _session_create(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.session_store, "sessions_unavailable", "session store is not configured")
        return store.create(owner_scope=arguments.get("owner_scope", "local"), title=arguments.get("title", "New session")).as_dict()

    return handler


def _sessions_list(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.session_store, "sessions_unavailable", "session store is not configured")
        return {"sessions": [item.as_dict() for item in store.list(owner_scope=arguments.get("owner_scope", "local"), limit=arguments.get("limit", 50))]}

    return handler


def _session_get(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.session_store, "sessions_unavailable", "session store is not configured")
        return store.get(arguments["session_id"], owner_scope=arguments.get("owner_scope"))

    return handler


def _rag_search(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        owner_scope = arguments.get("owner_scope", "local")
        source_ids = tuple(str(item) for item in arguments.get("source_ids", ()))
        if deps.rag_retriever is not None:
            retrieval = deps.rag_retriever.search(arguments["query"], owner_scope=owner_scope, source_ids=source_ids, title_prefix=arguments.get("title_prefix"), limit=arguments.get("limit"))
            hits = list(retrieval.hits)
        else:
            store = _require(deps.rag_store, "rag_unavailable", "RAG store is not configured")
            hits = store.search(arguments["query"], owner_scope=owner_scope, limit=arguments.get("limit", 8), source_ids=source_ids, title_prefix=arguments.get("title_prefix"))
            retrieval = None
        context = build_context([hit.as_dict() for hit in hits], max_chars=arguments.get("max_chars", 8_000), max_tokens=arguments.get("max_tokens"))
        response = {"query": arguments["query"], "hits": [hit.as_dict() for hit in hits], "context": context.as_dict()}
        if retrieval is not None:
            response["retrieval"] = retrieval.as_dict()
        return response

    return handler


def _rag_add_source(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.rag_store, "rag_unavailable", "RAG store is not configured")
        return store.add_document(source_ref=arguments["source_ref"], title=arguments["title"], text=arguments["text"], owner_scope=arguments.get("owner_scope", "local"), max_chars=arguments.get("max_chars", 1200), overlap_chars=arguments.get("overlap_chars", 120), strategy=arguments.get("strategy", "fixed"))

    return handler


def _memory_search(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.memory_store, "memory_unavailable", "memory store is not configured")
        hits = store.search(arguments["query"], owner_scope=arguments.get("owner_scope", "local"), limit=arguments.get("limit", 8))
        return {"query": arguments["query"], "hits": [hit.as_dict() for hit in hits]}

    return handler


def _memory_add(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.memory_store, "memory_unavailable", "memory store is not configured")
        entry = store.add(kind=arguments["kind"], content=arguments["content"], owner_scope=arguments.get("owner_scope", "local"), source_session_id=arguments.get("source_session_id"), source_message_id=arguments.get("source_message_id"), valid_until=arguments.get("valid_until"))
        return entry.as_dict()

    return handler


def _memory_invalidate(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.memory_store, "memory_unavailable", "memory store is not configured")
        return store.invalidate(arguments["entry_id"], owner_scope=arguments.get("owner_scope", "local"), reason=arguments.get("reason")).as_dict()

    return handler


def _memory_delete(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        store = _require(deps.memory_store, "memory_unavailable", "memory store is not configured")
        return store.delete(arguments["entry_id"], owner_scope=arguments.get("owner_scope", "local"), confirm=arguments["confirm"], reason=arguments.get("reason")).as_dict()

    return handler


def _web_search(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        client = _require(deps.network_client, "network_unavailable", "network client is not configured")
        try:
            return client.search(arguments["query"], top_k=arguments.get("top_k", 5), allow_external=arguments.get("allow_external", False)).as_dict()
        except NetworkToolError as exc:
            raise MCPToolError(exc.code, str(exc), retryable=exc.retryable) from exc

    return handler


def _web_fetch(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        client = _require(deps.network_client, "network_unavailable", "network client is not configured")
        try:
            return client.fetch(arguments["url"], max_chars=arguments.get("max_chars"), allow_external=arguments.get("allow_external", False)).as_dict()
        except NetworkToolError as exc:
            raise MCPToolError(exc.code, str(exc), retryable=exc.retryable) from exc

    return handler


def _image_capabilities(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        del arguments
        adapter = _require(deps.image_adapter, "images_unavailable", "image adapter is not configured")
        return adapter.capabilities().as_dict()

    return handler


def _image_generate(deps: HarnessMCPDependencies) -> Callable[[Mapping[str, Any]], Any]:
    def handler(arguments: Mapping[str, Any]) -> Any:
        adapter = _require(deps.image_adapter, "images_unavailable", "image adapter is not configured")
        request = ImageRequest.from_mapping(arguments)
        generated = adapter.generate(request)
        record = deps.image_store.put(generated, prompt=request.prompt, owner_scope=request.user or "local") if deps.image_store is not None else None
        if request.response_format == "url":
            if record is None:
                raise MCPToolError("image_url_unavailable", "image URL response requires an image asset store")
            item: dict[str, Any] = {"url": f"/v1/images/assets/{record.asset_id}", "asset_id": record.asset_id}
        else:
            item = {"b64_json": base64.b64encode(generated.data).decode("ascii")}
            if record is not None:
                item["asset_id"] = record.asset_id
        if record is not None:
            item["metadata"] = record.as_dict()
        return {"created": int(time.time()), "data": [item]}

    return handler


__all__ = ["CHAT_SCHEMA", "HarnessMCPDependencies", "register_builtin_tools"]
