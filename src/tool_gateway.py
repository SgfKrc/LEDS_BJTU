"""Fail-closed contract and network policy for user-owned web tools.

The gateway is intentionally an execution boundary, not an HTTP client.  G2
validates requests, redirects, resolved addresses, and normalized results.  A
later ticket may attach a provider; providers must call this module before any
socket is opened and after every redirect/DNS resolution.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit

TOOL_REQUEST_SCHEMA = "qlh.tool_request.v1"
TOOL_RESULT_SCHEMA = "qlh.tool_result.v1"
SUPPORTED_TOOLS = frozenset({"web_search", "web_fetch"})
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESULT_BYTES = 1 * 1024 * 1024
MAX_QUERY_CHARS = 512
MAX_URL_CHARS = 4096
MAX_TEXT_CHARS = 32 * 1024
MAX_SEARCH_ITEMS = 10
MAX_REDIRECTS = 3
DEFAULT_DEADLINE_MS = 5000
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
_HOST_LABEL = re.compile(r"^[A-Za-z0-9_\-\u0080-\uffff]+$")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")
_BLOCKED_HOSTS = frozenset({
    "localhost",
    "localhost.localdomain",
    "local",
    "internal",
    "lan",
    "metadata.google.internal",
    "metadata.google.com",
    "instance-data.ec2.internal",
})
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


class ToolGatewayError(ValueError):
    """Stable, provider-independent gateway rejection."""

    def __init__(self, code: str, message: str, *, status_code: int = 400, retryable: bool = False):
        self.code = code
        self.status_code = int(status_code)
        self.retryable = bool(retryable)
        super().__init__(message)


@dataclass(frozen=True)
class ToolGatewayPolicy:
    """Server-owned limits.  Request payloads cannot override these fields."""

    data_scope: str = "opt_in"
    allow_http: bool = False
    max_redirects: int = MAX_REDIRECTS
    max_response_bytes: int = MAX_RESULT_BYTES
    max_text_chars: int = MAX_TEXT_CHARS
    deadline_ms: int = DEFAULT_DEADLINE_MS
    proxy: str = ""
    allowed_content_types: tuple[str, ...] = ("text/html", "text/plain", "application/json", "text/markdown")

    def __post_init__(self) -> None:
        scope = str(self.data_scope or "").strip().lower()
        if scope not in {"deny", "opt_in", "allow_all"}:
            scope = "deny"
        object.__setattr__(self, "data_scope", scope)
        if isinstance(self.max_redirects, bool) or not isinstance(self.max_redirects, int) or not 0 <= self.max_redirects <= MAX_REDIRECTS:
            raise ValueError(f"max_redirects must be between 0 and {MAX_REDIRECTS}")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or not 1024 <= self.max_response_bytes <= MAX_RESULT_BYTES:
            raise ValueError(f"max_response_bytes must be between 1024 and {MAX_RESULT_BYTES}")
        if isinstance(self.max_text_chars, bool) or not isinstance(self.max_text_chars, int) or not 256 <= self.max_text_chars <= MAX_TEXT_CHARS:
            raise ValueError(f"max_text_chars must be between 256 and {MAX_TEXT_CHARS}")
        if isinstance(self.deadline_ms, bool) or not isinstance(self.deadline_ms, int) or not 100 <= self.deadline_ms <= DEFAULT_DEADLINE_MS:
            raise ValueError(f"deadline_ms must be between 100 and {DEFAULT_DEADLINE_MS}")
        if self.proxy:
            validate_proxy_url(self.proxy)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ToolGatewayPolicy":
        values = os.environ if env is None else env
        allow_http = str(values.get("QLH_TOOL_ALLOW_HTTP", "0")).strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            data_scope=values.get("QLH_EXTERNAL_DATA_SCOPE", "opt_in"),
            allow_http=allow_http,
            proxy=str(values.get("QLH_TOOL_PROXY", "") or "").strip(),
        )


def _json_size(value: Any, *, limit: int, code: str) -> None:
    try:
        size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolGatewayError(code, "payload is not JSON serializable") from exc
    if size > limit:
        raise ToolGatewayError(code, "payload exceeds the gateway size limit")


def _require_mapping(value: Any, code: str = "invalid_request") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolGatewayError(code, "payload must be an object")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: set[str], *, code: str = "invalid_request") -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ToolGatewayError(code, "unknown fields are not accepted")


def _normalize_hostname(hostname: str) -> str:
    host = str(hostname or "").strip().rstrip(".").lower()
    if not host or len(host) > 253 or _CONTROL.search(host):
        raise ToolGatewayError("unsafe_url_host", "URL host is invalid")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ToolGatewayError("unsafe_url_host", "URL host is invalid") from exc
    if len(ascii_host) > 253 or ascii_host in _BLOCKED_HOSTS or ascii_host.endswith(_BLOCKED_SUFFIXES):
        raise ToolGatewayError("unsafe_url_host", "URL host is not allowed")
    for label in ascii_host.split("."):
        if not label or len(label) > 63 or not _HOST_LABEL.fullmatch(label) or label.startswith("-") or label.endswith("-"):
            raise ToolGatewayError("unsafe_url_host", "URL host is invalid")
    return ascii_host


def _reject_non_public_ip(value: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    mapped = getattr(value, "ipv4_mapped", None)
    if mapped is not None:
        value = mapped
    if not value.is_global:
        raise ToolGatewayError("unsafe_url_host", "URL resolves to a non-public address")


def _url_parts(url: str, policy: ToolGatewayPolicy) -> tuple[SplitResult, str, int | None]:
    if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_CHARS or _CONTROL.search(url):
        raise ToolGatewayError("invalid_url", "URL is empty, too long, or contains control characters")
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ToolGatewayError("invalid_url", "URL syntax is invalid") from exc
    if parsed.scheme.lower() not in ({"https", "http"} if policy.allow_http else {"https"}):
        raise ToolGatewayError("unsafe_url_scheme", "only HTTPS URLs are allowed")
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise ToolGatewayError("unsafe_url_host", "URL credentials or host are not allowed")
    if parsed.fragment:
        raise ToolGatewayError("invalid_url", "URL fragments are not allowed")
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        normalized_host = str(literal_address)
        _reject_non_public_ip(literal_address)
    else:
        normalized_host = _normalize_hostname(hostname)
    if port is not None and not 1 <= port <= 65535:
        raise ToolGatewayError("unsafe_url_port", "URL port is outside the allowed range")
    return parsed, normalized_host, port


def validate_url(url: str, *, policy: ToolGatewayPolicy | None = None) -> str:
    """Validate a URL without DNS or network I/O and return a canonical URL."""
    resolved_policy = policy or ToolGatewayPolicy()
    parsed, hostname, port = _url_parts(url, resolved_policy)
    hostport = hostname
    if ":" in hostname:
        hostport = f"[{hostname}]"
    if port is not None:
        hostport = f"{hostport}:{port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), hostport, path, parsed.query, ""))


def validate_resolved_addresses(hostname: str, addresses: Iterable[str | ipaddress._BaseAddress]) -> tuple[str, ...]:
    """Apply the post-DNS SSRF gate; callers must invoke this on every hop."""
    normalized = _normalize_hostname(hostname)
    parsed: list[str] = []
    for raw in addresses:
        try:
            address = raw if isinstance(raw, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else ipaddress.ip_address(str(raw))
            _reject_non_public_ip(address)
            parsed.append(str(address))
        except (ValueError, ToolGatewayError) as exc:
            if isinstance(exc, ToolGatewayError):
                raise
            raise ToolGatewayError("unsafe_url_host", "DNS result is not a valid public address") from exc
    if not parsed:
        raise ToolGatewayError("dns_no_public_address", "DNS returned no public address")
    return tuple(sorted(set(parsed)))


def validate_proxy_url(proxy: str) -> str:
    """Validate an explicit user-owned proxy; localhost is allowed only here."""
    if not isinstance(proxy, str) or len(proxy) > 512 or _CONTROL.search(proxy):
        raise ToolGatewayError("invalid_proxy", "proxy URL is invalid")
    try:
        parsed = urlsplit(proxy.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ToolGatewayError("invalid_proxy", "proxy must be an HTTP(S) URL without credentials") from exc
    host = parsed.hostname.lower().rstrip(".")
    if host not in {"localhost", "127.0.0.1", "::1"}:
        try:
            literal_address = ipaddress.ip_address(host)
        except ValueError:
            literal_address = None
        if literal_address is not None:
            _reject_non_public_ip(literal_address)
        else:
            _normalize_hostname(host)
    return proxy.strip()


def validate_redirect_chain(urls: Iterable[str], *, policy: ToolGatewayPolicy | None = None) -> tuple[str, ...]:
    resolved_policy = policy or ToolGatewayPolicy()
    items = list(urls)
    if not items or len(items) > resolved_policy.max_redirects + 1:
        raise ToolGatewayError("redirect_limit_exceeded", "redirect chain exceeds the gateway limit")
    normalized = tuple(validate_url(item, policy=resolved_policy) for item in items)
    schemes = {urlsplit(item).scheme for item in normalized}
    if len(schemes) != 1:
        raise ToolGatewayError("unsafe_redirect", "redirect chain changes URL scheme")
    return normalized


def _scope_authorized(policy: ToolGatewayPolicy, allow_external: bool) -> None:
    if policy.data_scope == "allow_all":
        return
    if policy.data_scope == "opt_in" and allow_external:
        return
    raise ToolGatewayError("scope_denied", "network tool requires explicit user opt-in", status_code=403)


def _validate_arguments(tool_name: str, arguments: Any, policy: ToolGatewayPolicy) -> dict[str, Any]:
    args = _require_mapping(arguments, "invalid_arguments")
    if tool_name == "web_search":
        _strict_keys(args, {"query", "top_k"}, code="invalid_arguments")
        query = args.get("query")
        top_k = args.get("top_k", 5)
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ToolGatewayError("invalid_arguments", "web_search query is invalid")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_SEARCH_ITEMS:
            raise ToolGatewayError("invalid_arguments", "web_search top_k is invalid")
        return {"query": query.strip(), "top_k": top_k}
    if tool_name == "web_fetch":
        _strict_keys(args, {"url", "max_chars"}, code="invalid_arguments")
        url = validate_url(args.get("url"), policy=policy)
        max_chars = args.get("max_chars", policy.max_text_chars)
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or not 1 <= max_chars <= policy.max_text_chars:
            raise ToolGatewayError("invalid_arguments", "web_fetch max_chars is invalid")
        return {"url": url, "max_chars": max_chars}
    raise ToolGatewayError("unknown_tool", "tool is not registered")


def prepare_tool_request(payload: Mapping[str, Any], *, policy: ToolGatewayPolicy | None = None, allow_external: bool = False) -> dict[str, Any]:
    """Validate and normalize a tool request before any provider/network call."""
    resolved_policy = policy or ToolGatewayPolicy()
    _json_size(payload, limit=MAX_REQUEST_BYTES, code="request_too_large")
    request = _require_mapping(payload)
    _strict_keys(request, {"schema", "request_id", "tool_name", "arguments", "user_scope", "network_scope", "deadline_ms"})
    if request.get("schema") != TOOL_REQUEST_SCHEMA:
        raise ToolGatewayError("invalid_schema", "unsupported tool request schema")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ToolGatewayError("invalid_request_id", "request_id format is invalid")
    if request.get("user_scope", "local_user") != "local_user":
        raise ToolGatewayError("invalid_scope", "only local_user scope is supported")
    if request.get("network_scope", "explicit_opt_in") != "explicit_opt_in":
        raise ToolGatewayError("scope_override", "request cannot widen network scope")
    deadline_ms = request.get("deadline_ms", resolved_policy.deadline_ms)
    if not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool) or not 100 <= deadline_ms <= resolved_policy.deadline_ms:
        raise ToolGatewayError("invalid_deadline", "deadline exceeds the gateway policy")
    tool_name = request.get("tool_name")
    if tool_name not in SUPPORTED_TOOLS:
        raise ToolGatewayError("unknown_tool", "tool is not registered")
    _scope_authorized(resolved_policy, allow_external)
    arguments = _validate_arguments(tool_name, request.get("arguments"), resolved_policy)
    return {
        "schema": TOOL_REQUEST_SCHEMA,
        "request_id": request_id,
        "tool_name": tool_name,
        "arguments": arguments,
        "user_scope": "local_user",
        "network_scope": "explicit_opt_in",
        "deadline_ms": deadline_ms,
    }


def _normalize_item(item: Any, policy: ToolGatewayPolicy) -> dict[str, Any]:
    value = _require_mapping(item, "invalid_result")
    _strict_keys(value, {"title", "url", "snippet"}, code="invalid_result")
    title = value.get("title", "")
    snippet = value.get("snippet", "")
    if not isinstance(title, str) or len(title) > 512 or not isinstance(snippet, str) or len(snippet) > policy.max_text_chars:
        raise ToolGatewayError("invalid_result", "tool result text is outside the allowed range")
    return {"title": title, "url": validate_url(value.get("url"), policy=policy), "snippet": snippet}


def normalize_tool_result(payload: Mapping[str, Any], *, request_id: str, policy: ToolGatewayPolicy | None = None) -> dict[str, Any]:
    """Validate a provider result and remove unbounded/raw response fields."""
    resolved_policy = policy or ToolGatewayPolicy()
    _json_size(payload, limit=resolved_policy.max_response_bytes, code="response_too_large")
    result = _require_mapping(payload, "invalid_result")
    _strict_keys(result, {"schema", "request_id", "status", "items", "citations", "truncated", "policy", "error"}, code="invalid_result")
    if result.get("schema") != TOOL_RESULT_SCHEMA or result.get("request_id") != request_id:
        raise ToolGatewayError("invalid_result", "tool result envelope does not match the request")
    status = result.get("status")
    if status not in {"ok", "error"}:
        raise ToolGatewayError("invalid_result", "tool result status is invalid")
    if status == "error":
        error = _require_mapping(result.get("error"), "invalid_result")
        _strict_keys(error, {"code", "message", "retryable"}, code="invalid_result")
        code = error.get("code")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", code):
            raise ToolGatewayError("invalid_result", "tool result error code is invalid")
        message = error.get("message", "")
        if not isinstance(message, str) or len(message) > 512:
            raise ToolGatewayError("invalid_result", "tool result error message is invalid")
        return {"schema": TOOL_RESULT_SCHEMA, "request_id": request_id, "status": "error", "error": {"code": code, "message": message, "retryable": bool(error.get("retryable", False))}}
    items = result.get("items", [])
    if not isinstance(items, list) or len(items) > MAX_SEARCH_ITEMS:
        raise ToolGatewayError("invalid_result", "tool result item count is invalid")
    normalized_items = [_normalize_item(item, resolved_policy) for item in items]
    citations = result.get("citations", [])
    if not isinstance(citations, list) or len(citations) > MAX_SEARCH_ITEMS:
        raise ToolGatewayError("invalid_result", "citation count is invalid")
    normalized_citations: list[dict[str, str]] = []
    for citation in citations:
        value = _require_mapping(citation, "invalid_result")
        _strict_keys(value, {"url", "sha256"}, code="invalid_result")
        digest = value.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ToolGatewayError("invalid_result", "citation digest is invalid")
        normalized_citations.append({"url": validate_url(value.get("url"), policy=resolved_policy), "sha256": digest})
    policy_info = _require_mapping(result.get("policy", {}), "invalid_result")
    _strict_keys(policy_info, {"redirects", "bytes", "content_type"}, code="invalid_result")
    redirects = policy_info.get("redirects", 0)
    bytes_count = policy_info.get("bytes", 0)
    content_type = policy_info.get("content_type", "text/plain")
    if not isinstance(redirects, int) or not 0 <= redirects <= resolved_policy.max_redirects:
        raise ToolGatewayError("invalid_result", "redirect count is invalid")
    if not isinstance(bytes_count, int) or not 0 <= bytes_count <= resolved_policy.max_response_bytes:
        raise ToolGatewayError("response_too_large", "provider response exceeds the gateway limit")
    if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() not in set(resolved_policy.allowed_content_types):
        raise ToolGatewayError("unsupported_content_type", "provider content type is not allowed")
    truncated = result.get("truncated", False)
    if not isinstance(truncated, bool):
        raise ToolGatewayError("invalid_result", "truncated flag is invalid")
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "ok",
        "items": normalized_items,
        "citations": normalized_citations,
        "truncated": truncated,
        "policy": {"redirects": redirects, "bytes": bytes_count, "content_type": content_type.split(";", 1)[0].strip().lower()},
    }


def error_result(request_id: str, error: ToolGatewayError) -> dict[str, Any]:
    """Create a stable, non-sensitive result for a rejected request."""
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        request_id = "invalid-request"
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": request_id,
        "status": "error",
        "error": {"code": error.code, "message": str(error), "retryable": error.retryable},
    }


@dataclass(frozen=True)
class ToolGateway:
    """Small façade for future providers; no network operation is performed."""

    policy: ToolGatewayPolicy = field(default_factory=ToolGatewayPolicy)

    def prepare(self, payload: Mapping[str, Any], *, allow_external: bool = False) -> dict[str, Any]:
        return prepare_tool_request(payload, policy=self.policy, allow_external=allow_external)

    def result(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        return normalize_tool_result(payload, request_id=request_id, policy=self.policy)

    def redirects(self, urls: Iterable[str]) -> tuple[str, ...]:
        return validate_redirect_chain(urls, policy=self.policy)


__all__ = [
    "ToolGateway",
    "ToolGatewayError",
    "ToolGatewayPolicy",
    "TOOL_REQUEST_SCHEMA",
    "TOOL_RESULT_SCHEMA",
    "error_result",
    "normalize_tool_result",
    "prepare_tool_request",
    "validate_proxy_url",
    "validate_redirect_chain",
    "validate_resolved_addresses",
    "validate_url",
]
