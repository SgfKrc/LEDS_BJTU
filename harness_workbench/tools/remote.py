"""Remote QLH tool adapter with a fail-closed contract boundary.

The adapter speaks the main project's ``qlh.tool_request.v1`` /
``qlh.tool_result.v1`` contract without importing the main project.  A
transport is deliberately injected: constructing this adapter never opens a
socket, which keeps fake-QLH tests and offline harness development safe.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .network import (
    MAX_QUERY_CHARS,
    MAX_SEARCH_ITEMS,
    NetworkPolicy,
    NetworkToolError,
    TOOL_RESULT_SCHEMA as HARNESS_RESULT_SCHEMA,
    _validate_url,
)


TOOL_REQUEST_SCHEMA = "qlh.tool_request.v1"
REMOTE_RESULT_SCHEMA = "qlh.tool_result.v1"
DEFAULT_ENDPOINT = "/api/tool"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class RemoteToolError(NetworkToolError):
    """Stable error from request validation, transport, or remote QLH."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int = 502,
    ) -> None:
        self.status_code = int(status_code)
        super().__init__(code, message, retryable=retryable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "status_code": self.status_code,
        }


class QLHToolTransport(Protocol):
    """Minimal transport needed by the adapter and deterministic fake tests."""

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def close(self) -> None:
        ...


class _UnavailableTransport:
    def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        del path, payload
        raise RemoteToolError(
            "remote_transport_unavailable",
            "remote QLH transport is not configured",
            retryable=True,
            status_code=503,
        )

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class QLHToolAdapterConfig:
    """User-owned QLH endpoint and server-side deadline settings."""

    base_url: str
    endpoint: str = DEFAULT_ENDPOINT
    deadline_ms: int = 5_000
    provider_id: str = "qlh-remote"

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlsplit(str(self.base_url).strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("QLH base_url must use http(s) and include a host")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("QLH base_url cannot contain credentials, query, or fragment")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("QLH base_url port is invalid")
        object.__setattr__(self, "base_url", str(self.base_url).strip().rstrip("/"))

        endpoint = str(self.endpoint or "").strip()
        if not endpoint.startswith("/") or len(endpoint) > 256 or "?" in endpoint or "#" in endpoint:
            raise ValueError("QLH tool endpoint must be an absolute path without query or fragment")
        if any(part in {"", ".", ".."} for part in endpoint.split("/")[1:]):
            raise ValueError("QLH tool endpoint contains an unsafe path segment")
        object.__setattr__(self, "endpoint", endpoint)
        if isinstance(self.deadline_ms, bool) or not isinstance(self.deadline_ms, int) or not 100 <= self.deadline_ms <= 5_000:
            raise ValueError("deadline_ms must be between 100 and 5000")
        provider_id = str(self.provider_id or "").strip()
        if not provider_id or len(provider_id) > 64 or not re.fullmatch(r"[A-Za-z0-9_.-]+", provider_id):
            raise ValueError("provider_id is invalid")
        object.__setattr__(self, "provider_id", provider_id)


class QLHToolAdapter:
    """Map harness tool requests to a remote QLH tool endpoint.

    The adapter accepts the same request mapping used by the main ToolGateway
    provider interface.  It validates the request locally, requires explicit
    ``allow_external`` authorization, and validates every field of the remote
    result before exposing it to the harness.
    """

    def __init__(
        self,
        config: QLHToolAdapterConfig,
        *,
        transport: QLHToolTransport | None = None,
        policy: NetworkPolicy | None = None,
    ) -> None:
        self.config = config
        self.policy = policy or NetworkPolicy()
        self.transport = transport or _UnavailableTransport()
        self.provider_id = config.provider_id

    def execute(self, request: Mapping[str, Any], *, allow_external: bool = False) -> dict[str, Any]:
        prepared = self._prepare_request(request, allow_external=allow_external)
        try:
            raw = self.transport.post_json(self.config.endpoint, prepared)
        except RemoteToolError:
            raise
        except Exception as exc:
            raise RemoteToolError(
                "remote_transport_error",
                "remote QLH tool request failed",
                retryable=True,
                status_code=502,
            ) from exc
        return self._normalize_result(raw, prepared)

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def _prepare_request(self, request: Mapping[str, Any], *, allow_external: bool) -> dict[str, Any]:
        if not allow_external:
            raise RemoteToolError("scope_denied", "remote network tool requires explicit user opt-in", status_code=403)
        if not isinstance(request, Mapping):
            raise RemoteToolError("invalid_request", "tool request must be an object", status_code=400)
        allowed = {"schema", "request_id", "tool_name", "arguments", "user_scope", "network_scope", "deadline_ms"}
        if set(request) - allowed:
            raise RemoteToolError("invalid_request", "tool request contains unknown fields", status_code=400)
        if request.get("schema") != TOOL_REQUEST_SCHEMA:
            raise RemoteToolError("invalid_schema", "unsupported tool request schema", status_code=400)
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise RemoteToolError("invalid_request_id", "request_id format is invalid", status_code=400)
        tool_name = request.get("tool_name")
        if tool_name not in {"web_search", "web_fetch"}:
            raise RemoteToolError("unknown_tool", "tool is not registered", status_code=400)
        if request.get("user_scope", "local_user") != "local_user":
            raise RemoteToolError("invalid_scope", "only local_user scope is supported", status_code=400)
        if request.get("network_scope", "explicit_opt_in") != "explicit_opt_in":
            raise RemoteToolError("scope_override", "request cannot widen network scope", status_code=403)
        deadline_ms = request.get("deadline_ms", self.config.deadline_ms)
        if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or not 100 <= deadline_ms <= self.config.deadline_ms:
            raise RemoteToolError("invalid_deadline", "deadline exceeds the adapter policy", status_code=400)
        arguments = self._prepare_arguments(tool_name, request.get("arguments"))
        return {
            "schema": TOOL_REQUEST_SCHEMA,
            "request_id": request_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "user_scope": "local_user",
            "network_scope": "explicit_opt_in",
            "deadline_ms": deadline_ms,
        }

    def _prepare_arguments(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            raise RemoteToolError("invalid_arguments", "tool arguments must be an object", status_code=400)
        if tool_name == "web_search":
            if set(arguments) - {"query", "top_k"}:
                raise RemoteToolError("invalid_arguments", "web_search arguments contain unknown fields", status_code=400)
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
                raise RemoteToolError("invalid_arguments", "web_search query is invalid", status_code=400)
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_SEARCH_ITEMS:
                raise RemoteToolError("invalid_arguments", "web_search top_k is invalid", status_code=400)
            return {"query": query.strip(), "top_k": top_k}
        if set(arguments) - {"url", "max_chars"}:
            raise RemoteToolError("invalid_arguments", "web_fetch arguments contain unknown fields", status_code=400)
        try:
            url = _validate_url(arguments.get("url"), self.policy)
        except NetworkToolError as exc:
            raise RemoteToolError(exc.code, str(exc), status_code=400) from exc
        max_chars = arguments.get("max_chars", self.policy.max_text_chars)
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= self.policy.max_text_chars:
            raise RemoteToolError("invalid_arguments", "web_fetch max_chars is invalid", status_code=400)
        return {"url": url, "max_chars": max_chars}

    def _normalize_result(self, raw: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise RemoteToolError("invalid_remote_result", "remote QLH returned a non-object result")
        allowed = {"schema", "request_id", "status", "items", "citations", "truncated", "policy", "error"}
        if set(raw) - allowed:
            raise RemoteToolError("invalid_remote_result", "remote result contains unknown fields")
        if raw.get("schema") != REMOTE_RESULT_SCHEMA:
            raise RemoteToolError("invalid_remote_result", "remote result schema is unsupported")
        request_id = request["request_id"]
        if raw.get("request_id") != request_id:
            raise RemoteToolError("invalid_remote_result", "remote result request_id does not match")
        status = raw.get("status")
        if status == "error":
            self._raise_remote_error(raw.get("error"))
        if status != "ok":
            raise RemoteToolError("invalid_remote_result", "remote result status is invalid")
        items = raw.get("items")
        citations = raw.get("citations")
        if not isinstance(items, list) or len(items) > MAX_SEARCH_ITEMS:
            raise RemoteToolError("invalid_remote_result", "remote result items are invalid")
        if not isinstance(citations, list) or len(citations) > MAX_SEARCH_ITEMS:
            raise RemoteToolError("invalid_remote_result", "remote result citations are invalid")
        normalized_items = [self._normalize_item(item) for item in items]
        normalized_citations = [self._normalize_citation(item) for item in citations]
        truncated = raw.get("truncated", False)
        if not isinstance(truncated, bool):
            raise RemoteToolError("invalid_remote_result", "remote result truncated flag is invalid")
        policy = self._normalize_policy(raw.get("policy"))
        return {
            "schema": HARNESS_RESULT_SCHEMA,
            "request_id": request_id,
            "tool_name": request["tool_name"],
            "status": "ok",
            "items": normalized_items,
            "citations": normalized_citations,
            "truncated": truncated,
            "policy": policy,
            "production_network_enabled": False,
        }

    def _normalize_item(self, item: Any) -> dict[str, str]:
        if not isinstance(item, Mapping) or set(item) - {"title", "url", "snippet"}:
            raise RemoteToolError("invalid_remote_result", "remote result item is invalid")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if not isinstance(title, str) or len(title) > 512 or not isinstance(snippet, str) or len(snippet) > self.policy.max_text_chars:
            raise RemoteToolError("invalid_remote_result", "remote result text is outside the policy")
        try:
            url = _validate_url(item.get("url"), self.policy)
        except NetworkToolError as exc:
            raise RemoteToolError(exc.code, str(exc)) from exc
        if not title or not snippet:
            raise RemoteToolError("invalid_remote_result", "remote result item is empty")
        return {"title": title, "url": url, "snippet": snippet}

    def _normalize_citation(self, citation: Any) -> dict[str, str]:
        if not isinstance(citation, Mapping) or set(citation) - {"url", "sha256"}:
            raise RemoteToolError("invalid_remote_result", "remote citation is invalid")
        digest = citation.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RemoteToolError("invalid_remote_result", "remote citation digest is invalid")
        try:
            url = _validate_url(citation.get("url"), self.policy)
        except NetworkToolError as exc:
            raise RemoteToolError(exc.code, str(exc)) from exc
        return {"url": url, "sha256": digest}

    def _normalize_policy(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) - {"redirects", "bytes", "content_type"}:
            raise RemoteToolError("invalid_remote_result", "remote result policy is invalid")
        redirects = value.get("redirects", 0)
        bytes_count = value.get("bytes", 0)
        content_type = value.get("content_type", "text/plain")
        if isinstance(redirects, bool) or not isinstance(redirects, int) or not 0 <= redirects <= self.policy.max_redirects:
            raise RemoteToolError("invalid_remote_result", "remote redirect count is invalid")
        if isinstance(bytes_count, bool) or not isinstance(bytes_count, int) or not 0 <= bytes_count <= self.policy.max_response_bytes:
            raise RemoteToolError("response_too_large", "remote result exceeds the byte limit")
        if not isinstance(content_type, str):
            raise RemoteToolError("invalid_remote_result", "remote content type is invalid")
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in self.policy.allowed_content_types:
            raise RemoteToolError("unsupported_content_type", "remote content type is not allowed")
        return {"redirects": redirects, "bytes": bytes_count, "content_type": normalized_type}

    @staticmethod
    def _raise_remote_error(value: Any) -> None:
        if not isinstance(value, Mapping) or set(value) - {"code", "message", "retryable"}:
            raise RemoteToolError("invalid_remote_error", "remote error object is invalid")
        code = value.get("code")
        message = value.get("message", "")
        retryable = value.get("retryable", False)
        if not isinstance(code, str) or not _ERROR_CODE.fullmatch(code) or not isinstance(message, str) or len(message) > 512 or not isinstance(retryable, bool):
            raise RemoteToolError("invalid_remote_error", "remote error fields are invalid")
        raise RemoteToolError(code, message, retryable=retryable, status_code=502)


__all__ = [
    "DEFAULT_ENDPOINT",
    "QLHToolAdapter",
    "QLHToolAdapterConfig",
    "QLHToolTransport",
    "REMOTE_RESULT_SCHEMA",
    "RemoteToolError",
    "TOOL_REQUEST_SCHEMA",
]
