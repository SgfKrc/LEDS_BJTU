"""Local web_fetch/web_search with explicit SSRF and response gates.

The module has no dependency on the main QLH runtime.  A transport and DNS
resolver can be injected for deterministic tests; the real urllib transport
is disabled until the server owner explicitly enables production networking.
"""

from __future__ import annotations

import hashlib
import html
import html.parser
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


TOOL_RESULT_SCHEMA = "qlh.harness.tool_result.v1"
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
MAX_TEXT_CHARS = 32 * 1024
MAX_QUERY_CHARS = 512
MAX_SEARCH_ITEMS = 10
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata.google.com",
        "instance-data.ec2.internal",
    }
)
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")
_HOST_LABEL = re.compile(r"^[A-Za-z0-9_\-\u0080-\uffff]+$")


class NetworkToolError(ValueError):
    """Stable, provider-independent network tool rejection."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """Server-owned limits; request arguments cannot widen them."""

    production_network_enabled: bool = False
    allow_http: bool = False
    max_redirects: int = MAX_REDIRECTS
    max_response_bytes: int = MAX_RESPONSE_BYTES
    max_text_chars: int = MAX_TEXT_CHARS
    timeout_seconds: float = 5.0
    proxy: str = ""
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "text/plain",
        "text/markdown",
        "application/json",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.production_network_enabled, bool):
            raise ValueError("production_network_enabled must be boolean")
        if isinstance(self.max_redirects, bool) or not isinstance(self.max_redirects, int) or not 0 <= self.max_redirects <= MAX_REDIRECTS:
            raise ValueError(f"max_redirects must be between 0 and {MAX_REDIRECTS}")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or not 1024 <= self.max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(f"max_response_bytes must be between 1024 and {MAX_RESPONSE_BYTES}")
        if isinstance(self.max_text_chars, bool) or not isinstance(self.max_text_chars, int) or not 256 <= self.max_text_chars <= MAX_TEXT_CHARS:
            raise ValueError(f"max_text_chars must be between 256 and {MAX_TEXT_CHARS}")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not 0.1 <= float(self.timeout_seconds) <= 5.0:
            raise ValueError("timeout_seconds must be between 0.1 and 5.0")
        if self.proxy:
            validate_proxy(self.proxy)
        normalized = tuple(str(item).split(";", 1)[0].strip().lower() for item in self.allowed_content_types if str(item).strip())
        if not normalized:
            raise ValueError("allowed_content_types cannot be empty")
        object.__setattr__(self, "allowed_content_types", normalized)


@dataclass(frozen=True, slots=True)
class NetworkResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes


class NetworkTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        proxy: str,
        timeout_seconds: float,
        max_bytes: int,
        resolved_addresses: Sequence[str],
    ) -> NetworkResponse:
        ...


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    text: str
    content_type: str
    bytes_count: int
    redirects: tuple[str, ...]
    sha256: str
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "text": self.text,
            "content_type": self.content_type,
            "bytes": self.bytes_count,
            "redirects": list(self.redirects),
            "sha256": self.sha256,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    items: tuple[Mapping[str, str], ...]
    citations: tuple[Mapping[str, str], ...]
    redirects: tuple[str, ...]
    bytes_count: int
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "items": [dict(item) for item in self.items],
            "citations": [dict(item) for item in self.citations],
            "redirects": list(self.redirects),
            "bytes": self.bytes_count,
            "truncated": self.truncated,
        }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class UrllibTransport:
    """Explicit HTTP transport; policy and DNS gates run before every hop."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        proxy: str,
        timeout_seconds: float,
        max_bytes: int,
        resolved_addresses: Sequence[str],
    ) -> NetworkResponse:
        del resolved_addresses  # DNS validation is performed by NetworkClient.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}),
            _NoRedirect(),
        )
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = _read_bounded(response, max_bytes)
                return NetworkResponse(
                    int(getattr(response, "status", response.getcode())),
                    {str(key): str(value) for key, value in response.headers.items()},
                    body,
                )
        except urllib.error.HTTPError as exc:
            return NetworkResponse(
                int(exc.code),
                {str(key): str(value) for key, value in exc.headers.items()},
                _read_bounded(exc, max_bytes),
            )
        except (TimeoutError, socket.timeout) as exc:
            raise NetworkToolError("timeout", "network request timed out", retryable=True) from exc
        except urllib.error.URLError as exc:
            raise NetworkToolError("network_error", "network request failed", retryable=True) from exc


class NetworkClient:
    """Policy-first web tools with injectable DNS and transport boundaries."""

    def __init__(
        self,
        *,
        policy: NetworkPolicy | None = None,
        transport: NetworkTransport | None = None,
        resolver: Callable[[str, int], Iterable[str]] | None = None,
        search_endpoint: str | None = None,
    ) -> None:
        self.policy = policy or NetworkPolicy()
        self.transport = transport or UrllibTransport()
        self.resolver = resolver or _resolve
        self.search_endpoint = _validate_url(search_endpoint, self.policy) if search_endpoint else None

    def fetch(self, url: str, *, max_chars: int | None = None, allow_external: bool = False) -> FetchResult:
        self._authorize(allow_external)
        if max_chars is None:
            max_chars = self.policy.max_text_chars
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= self.policy.max_text_chars:
            raise NetworkToolError("invalid_arguments", "max_chars is outside the policy")
        response, chain = self._request(url)
        content_type = _content_type(response.headers)
        self._require_content_type(content_type)
        text = _extract_text(response.body, content_type)
        truncated = len(text) > max_chars
        text = text[:max_chars]
        return FetchResult(
            url=chain[0],
            final_url=chain[-1],
            text=text,
            content_type=content_type,
            bytes_count=len(response.body),
            redirects=tuple(chain),
            sha256=hashlib.sha256(response.body).hexdigest(),
            truncated=truncated,
        )

    def search(self, query: str, *, top_k: int = 5, allow_external: bool = False) -> SearchResult:
        self._authorize(allow_external)
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise NetworkToolError("invalid_arguments", "query is invalid")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_SEARCH_ITEMS:
            raise NetworkToolError("invalid_arguments", "top_k is invalid")
        if not self.search_endpoint:
            raise NetworkToolError("search_unconfigured", "search endpoint is not configured")
        parsed = urllib.parse.urlsplit(self.search_endpoint)
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        params.extend((("q", query.strip()), ("format", "json")))
        endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/search", urllib.parse.urlencode(params), ""))
        response, chain = self._request(endpoint, expected_content_types=("application/json", "text/json"))
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NetworkToolError("invalid_json", "search provider returned invalid JSON") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise NetworkToolError("invalid_payload", "search provider payload has no results list")
        items: list[Mapping[str, str]] = []
        citations: list[Mapping[str, str]] = []
        filtered = 0
        for raw in payload["results"]:
            if len(items) >= top_k:
                break
            if not isinstance(raw, Mapping):
                filtered += 1
                continue
            try:
                safe_url = _validate_url(raw.get("url"), self.policy)
            except NetworkToolError:
                filtered += 1
                continue
            title = _safe_text(raw.get("title"), 512)
            snippet = _safe_text(raw.get("content") or raw.get("snippet"), self.policy.max_text_chars)
            if not title or not snippet:
                filtered += 1
                continue
            item = {"title": title, "url": safe_url, "snippet": snippet}
            items.append(item)
            citations.append({"url": safe_url, "sha256": _digest(item)})
        if payload["results"] and not items:
            raise NetworkToolError("no_safe_results", "search provider returned no safe results")
        return SearchResult(query.strip(), tuple(items), tuple(citations), tuple(chain), len(response.body), bool(filtered or len(payload["results"]) > len(items)))

    def execute(self, tool_name: str, arguments: Mapping[str, Any], *, allow_external: bool = False, request_id: str = "local") -> dict[str, Any]:
        if tool_name == "web_fetch":
            result = self.fetch(arguments.get("url", ""), max_chars=arguments.get("max_chars"), allow_external=allow_external)
            return _ok_envelope(request_id, "web_fetch", [result.as_dict()], [{"url": result.final_url, "sha256": result.sha256}], result.truncated, result.bytes_count, result.content_type, len(result.redirects) - 1, self.policy.production_network_enabled)
        if tool_name == "web_search":
            result = self.search(arguments.get("query", ""), top_k=arguments.get("top_k", 5), allow_external=allow_external)
            return _ok_envelope(request_id, "web_search", list(result.items), list(result.citations), result.truncated, result.bytes_count, "application/json", len(result.redirects) - 1, self.policy.production_network_enabled)
        raise NetworkToolError("unknown_tool", "tool is not registered")

    def _authorize(self, allow_external: bool) -> None:
        if not allow_external:
            raise NetworkToolError("scope_denied", "network tool requires explicit user opt-in")
        if isinstance(self.transport, UrllibTransport) and not self.policy.production_network_enabled:
            raise NetworkToolError("network_disabled", "production network access is disabled")

    def _request(self, url: str, *, expected_content_types: Sequence[str] | None = None) -> tuple[NetworkResponse, list[str]]:
        current = _validate_url(url, self.policy)
        chain = [current]
        while True:
            parsed = urllib.parse.urlsplit(current)
            addresses = _resolve_public(self.resolver, parsed.hostname or "", parsed.port or (443 if parsed.scheme == "https" else 80))
            response = self.transport.get(
                current,
                headers={"Accept": ", ".join(expected_content_types or self.policy.allowed_content_types), "User-Agent": "QLH-Harness-Tool/1"},
                proxy=self.policy.proxy,
                timeout_seconds=float(self.policy.timeout_seconds),
                max_bytes=self.policy.max_response_bytes,
                resolved_addresses=addresses,
            )
            if len(response.body) > self.policy.max_response_bytes:
                raise NetworkToolError("response_too_large", "response exceeds the byte limit")
            location = _header(response.headers, "location")
            if response.status_code in _REDIRECT_STATUSES and location:
                if len(chain) >= self.policy.max_redirects + 1:
                    raise NetworkToolError("redirect_limit_exceeded", "redirect chain exceeds the policy")
                try:
                    current = _validate_url(urllib.parse.urljoin(current, location), self.policy)
                except NetworkToolError as exc:
                    raise NetworkToolError("unsafe_redirect", "redirect target is not allowed") from exc
                chain.append(current)
                continue
            if 300 <= response.status_code < 400:
                raise NetworkToolError("unexpected_redirect", "redirect response has no usable location")
            if response.status_code >= 400:
                retryable = response.status_code in {408, 425, 429} or response.status_code >= 500
                raise NetworkToolError(f"http_{response.status_code}", "provider rejected the request", retryable=retryable)
            if expected_content_types is not None and _content_type(response.headers) not in expected_content_types:
                raise NetworkToolError("unsupported_content_type", "provider content type is not allowed")
            return response, chain

    def _require_content_type(self, content_type: str) -> None:
        if content_type not in self.policy.allowed_content_types:
            raise NetworkToolError("unsupported_content_type", "content type is not allowed")


def validate_proxy(value: str) -> str:
    if not isinstance(value, str) or len(value) > 512 or _CONTROL.search(value):
        raise NetworkToolError("invalid_proxy", "proxy URL is invalid")
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise NetworkToolError("invalid_proxy", "proxy must be HTTP(S) without credentials")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise NetworkToolError("invalid_proxy", "proxy port is invalid")
    return value.strip()


def _validate_url(value: Any, policy: NetworkPolicy) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or _CONTROL.search(value):
        raise NetworkToolError("invalid_url", "URL is empty, too long, or contains control characters")
    try:
        parsed = urllib.parse.urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise NetworkToolError("invalid_url", "URL syntax is invalid") from exc
    allowed_schemes = {"https", "http"} if policy.allow_http else {"https"}
    if parsed.scheme.lower() not in allowed_schemes:
        raise NetworkToolError("unsafe_url_scheme", "only HTTPS URLs are allowed")
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise NetworkToolError("unsafe_url_host", "URL credentials or host are not allowed")
    if parsed.fragment:
        raise NetworkToolError("invalid_url", "URL fragments are not allowed")
    normalized = _normalize_host(hostname)
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        address = None
    if address is not None:
        _reject_non_public(address)
        host = str(address)
    else:
        host = normalized
    if port is not None and not 1 <= port <= 65535:
        raise NetworkToolError("unsafe_url_port", "URL port is invalid")
    hostport = f"[{host}]" if ":" in host else host
    if port is not None:
        hostport += f":{port}"
    return urllib.parse.urlunsplit((parsed.scheme.lower(), hostport, parsed.path or "/", parsed.query, ""))


def _normalize_host(value: str) -> str:
    host = str(value or "").strip().rstrip(".").lower()
    if not host or len(host) > 253 or host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        raise NetworkToolError("unsafe_url_host", "URL host is not allowed")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise NetworkToolError("unsafe_url_host", "URL host is invalid") from exc
    for label in host.split("."):
        if not label or len(label) > 63 or not _HOST_LABEL.fullmatch(label) or label.startswith("-") or label.endswith("-"):
            raise NetworkToolError("unsafe_url_host", "URL host is invalid")
    return host


def _reject_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if not address.is_global:
        raise NetworkToolError("unsafe_url_host", "URL resolves to a non-public address")


def _resolve(resolver: Callable[[str, int], Iterable[str]], host: str, port: int) -> tuple[str, ...]:
    return tuple(info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))


def _resolve_public(resolver: Callable[[str, int], Iterable[str]], host: str, port: int) -> tuple[str, ...]:
    try:
        values = tuple(resolver(host, port))
    except (OSError, socket.gaierror, ValueError) as exc:
        raise NetworkToolError("dns_unavailable", "provider host could not be resolved", retryable=True) from exc
    if not values:
        raise NetworkToolError("dns_no_public_address", "DNS returned no public address")
    addresses: list[str] = []
    for raw in values:
        try:
            address = ipaddress.ip_address(str(raw))
            _reject_non_public(address)
            addresses.append(str(address))
        except NetworkToolError:
            raise
        except ValueError as exc:
            raise NetworkToolError("unsafe_url_host", "DNS result is not an IP address") from exc
    return tuple(sorted(set(addresses)))


def _read_bounded(stream: Any, limit: int) -> bytes:
    try:
        declared = int(_header(getattr(stream, "headers", {}), "content-length", "0") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > limit:
        raise NetworkToolError("response_too_large", "response exceeds the byte limit")
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise NetworkToolError("response_too_large", "response exceeds the byte limit")
    return body


def _header(headers: Mapping[str, Any], name: str, default: str = "") -> str:
    name = name.lower()
    for key, value in headers.items():
        if str(key).lower() == name:
            return str(value or "")
    return default


def _content_type(headers: Mapping[str, Any]) -> str:
    return _header(headers, "content-type", "text/plain").split(";", 1)[0].strip().lower()


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "template"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def _extract_text(body: bytes, content_type: str) -> str:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NetworkToolError("invalid_encoding", "response is not UTF-8 text") from exc
    if content_type == "text/html":
        parser = _TextExtractor()
        parser.feed(text)
        text = " ".join(parser.parts)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def _safe_text(value: Any, limit: int) -> str:
    return html.unescape(str(value or "")).strip()[:limit]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ok_envelope(request_id: str, tool_name: str, items: list[Mapping[str, Any]], citations: list[Mapping[str, Any]], truncated: bool, bytes_count: int, content_type: str, redirects: int, production_network_enabled: bool) -> dict[str, Any]:
    return {
        "schema": TOOL_RESULT_SCHEMA,
        "request_id": str(request_id),
        "tool_name": tool_name,
        "status": "ok",
        "items": items,
        "citations": citations,
        "truncated": bool(truncated),
        "policy": {"redirects": redirects, "bytes": bytes_count, "content_type": content_type},
        "production_network_enabled": bool(production_network_enabled),
    }


__all__ = [
    "FetchResult",
    "NetworkClient",
    "NetworkPolicy",
    "NetworkResponse",
    "NetworkToolError",
    "SearchResult",
    "TOOL_RESULT_SCHEMA",
    "UrllibTransport",
    "validate_proxy",
]
