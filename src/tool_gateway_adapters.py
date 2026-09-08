"""Restricted search/fetch adapters built on top of :mod:`tool_gateway`.

G3 deliberately stops at an injectable provider boundary.  ``UrllibTransport``
is available for an explicit opt-in caller, while tests and future providers can
use ``ToolTransport`` without opening sockets.  The adapters never return raw
response bodies or provider payloads.
"""

from __future__ import annotations

import hashlib
import html
import html.parser
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

try:  # Package import is used by tests; top-level import matches the app's src path.
    from .tool_gateway import (
        ToolGateway,
        ToolGatewayError,
        ToolGatewayPolicy,
        error_result,
        normalize_tool_result,
        validate_proxy_url,
        validate_redirect_chain,
        validate_resolved_addresses,
        validate_url,
    )
except ImportError:  # pragma: no cover - exercised by the application import layout.
    from tool_gateway import (
        ToolGateway,
        ToolGatewayError,
        ToolGatewayPolicy,
        error_result,
        normalize_tool_result,
        validate_proxy_url,
        validate_redirect_chain,
        validate_resolved_addresses,
        validate_url,
    )

DEFAULT_USER_AGENT = "QLH-ToolGateway/1"
DEFAULT_TIMEOUT_SECONDS = 5.0
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
SEARCH_CONTENT_TYPE = "application/json"
FETCH_CONTENT_TYPES = frozenset({"text/html", "text/plain", "text/markdown", "application/json"})


class ToolProviderError(RuntimeError):
    """Provider error with a stable code and fallback classification."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str
    redirects: tuple[str, ...] = ()
    elapsed_ms: int = 0


class ToolTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        proxy: str,
        timeout_seconds: float,
        max_bytes: int,
        policy: ToolGatewayPolicy,
    ) -> TransportResponse:
        ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _header(headers: Mapping[str, str], name: str, default: str = "") -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value or "")
    return default


def _content_type(headers: Mapping[str, str], default: str = "text/plain") -> str:
    return _header(headers, "Content-Type", default).split(";", 1)[0].strip().lower()


def _read_bounded(stream: Any, max_bytes: int) -> bytes:
    try:
        declared = int(_header(getattr(stream, "headers", {}), "Content-Length", "0") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > max_bytes:
        raise ToolProviderError("response_too_large", "provider response exceeds the gateway limit")
    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ToolProviderError("response_too_large", "provider response exceeds the gateway limit")
    return data


def _resolve_public_addresses(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    hostname = parsed.hostname
    if not hostname:
        raise ToolGatewayError("unsafe_url_host", "URL host is missing")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError) as exc:
        raise ToolProviderError("dns_unavailable", "provider host could not be resolved", retryable=True) from exc
    addresses = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            addresses.append(sockaddr[0])
    validate_resolved_addresses(hostname, addresses)


class UrllibTransport:
    """HTTP GET with explicit proxy isolation and per-hop SSRF fencing."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        proxy: str,
        timeout_seconds: float,
        max_bytes: int,
        policy: ToolGatewayPolicy,
    ) -> TransportResponse:
        if proxy:
            proxy = validate_proxy_url(proxy)
        current = validate_url(url, policy=policy)
        chain = [current]
        started = time.perf_counter()
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {}),
            _NoRedirect(),
        )
        while True:
            _resolve_public_addresses(current)
            request = urllib.request.Request(current, headers=dict(headers), method="GET")
            try:
                with opener.open(request, timeout=timeout_seconds) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    response_headers = {str(key): str(value) for key, value in response.headers.items()}
                    body = _read_bounded(response, max_bytes)
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                response_headers = {str(key): str(value) for key, value in exc.headers.items()}
                body = _read_bounded(exc, max_bytes)
            except urllib.error.URLError as exc:
                reason = str(getattr(exc, "reason", exc))
                if "timed out" in reason.lower():
                    raise ToolProviderError("timeout", "provider request timed out", retryable=True) from exc
                raise ToolProviderError("network_error", "provider request failed", retryable=True) from exc
            except (TimeoutError, socket.timeout) as exc:
                raise ToolProviderError("timeout", "provider request timed out", retryable=True) from exc
            except ToolProviderError:
                raise
            except OSError as exc:
                raise ToolProviderError("network_error", "provider request failed", retryable=True) from exc

            location = _header(response_headers, "Location")
            if status in REDIRECT_STATUSES and location:
                if len(chain) >= policy.max_redirects + 1:
                    raise ToolGatewayError("redirect_limit_exceeded", "provider redirect chain exceeds the gateway limit")
                try:
                    next_url = validate_url(urllib.parse.urljoin(current, location), policy=policy)
                except ToolGatewayError:
                    raise ToolGatewayError("unsafe_redirect", "provider redirect target is not allowed") from None
                chain.append(next_url)
                validate_redirect_chain(chain, policy=policy)
                current = next_url
                continue
            return TransportResponse(
                status_code=status,
                headers=response_headers,
                body=body,
                final_url=current,
                redirects=tuple(chain),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )


def _provider_status(response: TransportResponse) -> None:
    if 300 <= response.status_code < 400:
        raise ToolProviderError("unexpected_redirect", "provider returned an unusable redirect")
    if response.status_code in {408, 425, 429} or response.status_code >= 500:
        raise ToolProviderError(f"http_{response.status_code}", "provider temporarily unavailable", retryable=True)
    if response.status_code >= 400:
        raise ToolProviderError(f"http_{response.status_code}", "provider rejected the request")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_text(value: Any, limit: int) -> str:
    return html.unescape(str(value or "")).strip()[:limit]


def _result_envelope(
    *,
    request_id: str,
    items: list[dict[str, Any]],
    citations: list[dict[str, str]],
    response: TransportResponse,
    policy: ToolGatewayPolicy,
    truncated: bool = False,
) -> dict[str, Any]:
    payload = {
        "schema": "qlh.tool_result.v1",
        "request_id": request_id,
        "status": "ok",
        "items": items,
        "citations": citations,
        "truncated": bool(truncated),
        "policy": {
            "redirects": max(0, len(response.redirects) - 1),
            "bytes": len(response.body),
            "content_type": _content_type(response.headers),
        },
    }
    return normalize_tool_result(payload, request_id=request_id, policy=policy)


class SearxSearchAdapter:
    """SearXNG-compatible JSON search adapter with no raw payload passthrough."""

    def __init__(
        self,
        endpoint: str,
        *,
        transport: ToolTransport | None = None,
        policy: ToolGatewayPolicy | None = None,
        provider_id: str = "searxng",
        proxy: str = "",
    ) -> None:
        self.policy = policy or ToolGatewayPolicy()
        self.endpoint = validate_url(endpoint, policy=self.policy)
        self.transport = transport or UrllibTransport()
        self.provider_id = provider_id[:64] or "searxng"
        self.proxy = validate_proxy_url(proxy) if proxy else ""

    def execute(self, request: Mapping[str, Any], *, allow_external: bool = False) -> dict[str, Any]:
        gateway = ToolGateway(self.policy)
        prepared = gateway.prepare(request, allow_external=allow_external)
        arguments = prepared["arguments"]
        parsed = urllib.parse.urlsplit(self.endpoint)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query.extend((("q", arguments["query"]), ("format", "json"), ("categories", "general")))
        query_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/search", urllib.parse.urlencode(query), ""))
        try:
            response = self.transport.get(
                query_url,
                headers={"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT},
                proxy=self.proxy,
                timeout_seconds=self.policy.deadline_ms / 1000,
                max_bytes=self.policy.max_response_bytes,
                policy=self.policy,
            )
            _provider_status(response)
            if _content_type(response.headers, SEARCH_CONTENT_TYPE) not in {"application/json", "text/json"}:
                raise ToolProviderError("unsupported_content_type", "search provider did not return JSON")
            payload = json.loads(response.body.decode("utf-8"))
        except ToolGatewayError:
            raise
        except ToolProviderError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolProviderError("invalid_json", "search provider returned invalid JSON") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise ToolProviderError("invalid_payload", "search provider payload has no results list")
        items: list[dict[str, Any]] = []
        citations: list[dict[str, str]] = []
        filtered = 0
        for raw in payload["results"]:
            if len(items) >= arguments["top_k"]:
                break
            if not isinstance(raw, Mapping):
                filtered += 1
                continue
            raw_url = raw.get("url")
            try:
                safe_url = validate_url(raw_url, policy=self.policy)
            except ToolGatewayError:
                filtered += 1
                continue
            item = {
                "title": _safe_text(raw.get("title"), 512),
                "url": safe_url,
                "snippet": _safe_text(raw.get("content") or raw.get("snippet"), self.policy.max_text_chars),
            }
            if not item["title"] or not item["snippet"]:
                filtered += 1
                continue
            items.append(item)
            citations.append({"url": safe_url, "sha256": _digest(item)})
        if payload["results"] and not items:
            raise ToolProviderError("no_safe_results", "search provider returned no safe results")
        return _result_envelope(
            request_id=prepared["request_id"],
            items=items,
            citations=citations,
            response=response,
            policy=self.policy,
            truncated=filtered > 0 or len(payload["results"]) > len(items),
        )


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1
        elif normalized == "title" and self._skip_depth == 0:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "template"} and self._skip_depth:
            self._skip_depth -= 1
        elif normalized == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = " ".join(data.split())
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
            return
        if sum(len(part) for part in self.text_parts) < self.limit:
            self.text_parts.append(value[: max(0, self.limit - sum(len(part) for part in self.text_parts))])

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()[:512]

    @property
    def text(self) -> str:
        return " ".join(self.text_parts).strip()[: self.limit]


class RestrictedFetchAdapter:
    """Fetch text/HTML/JSON under the Gateway size and citation policy."""

    def __init__(
        self,
        *,
        transport: ToolTransport | None = None,
        policy: ToolGatewayPolicy | None = None,
        provider_id: str = "fetch",
        proxy: str = "",
    ) -> None:
        self.policy = policy or ToolGatewayPolicy()
        self.transport = transport or UrllibTransport()
        self.provider_id = provider_id[:64] or "fetch"
        self.proxy = validate_proxy_url(proxy) if proxy else ""

    def execute(self, request: Mapping[str, Any], *, allow_external: bool = False) -> dict[str, Any]:
        gateway = ToolGateway(self.policy)
        prepared = gateway.prepare(request, allow_external=allow_external)
        arguments = prepared["arguments"]
        url = arguments["url"]
        try:
            response = self.transport.get(
                url,
                headers={"Accept": ", ".join(sorted(FETCH_CONTENT_TYPES)), "User-Agent": DEFAULT_USER_AGENT},
                proxy=self.proxy,
                timeout_seconds=self.policy.deadline_ms / 1000,
                max_bytes=self.policy.max_response_bytes,
                policy=self.policy,
            )
            _provider_status(response)
        except (ToolGatewayError, ToolProviderError):
            raise
        content_type = _content_type(response.headers)
        if content_type not in FETCH_CONTENT_TYPES:
            raise ToolProviderError("unsupported_content_type", "fetch provider content type is not allowed")
        text_body = response.body.decode("utf-8", errors="replace")
        title = ""
        if content_type == "text/html":
            extractor = _TextExtractor(arguments["max_chars"])
            try:
                extractor.feed(text_body)
                extractor.close()
            except (ValueError, AssertionError):
                raise ToolProviderError("invalid_content", "HTML content could not be parsed") from None
            title = extractor.title
            text_body = extractor.text
        else:
            text_body = " ".join(text_body.split())[: arguments["max_chars"]]
        if not text_body:
            raise ToolProviderError("empty_content", "fetch provider returned no text")
        item = {"title": title or url, "url": url, "snippet": text_body}
        citation = {"url": url, "sha256": hashlib.sha256(response.body).hexdigest()}
        return _result_envelope(
            request_id=prepared["request_id"],
            items=[item],
            citations=[citation],
            response=response,
            policy=self.policy,
            truncated=len(text_body) >= arguments["max_chars"] or len(response.body) >= self.policy.max_response_bytes,
        )


@dataclass(frozen=True)
class ToolExecutionReport:
    result: dict[str, Any]
    provider_id: str
    fallback_used: bool
    attempts: tuple[dict[str, Any], ...]


class ToolGatewayExecutor:
    """Provider selection with fallback only for retryable provider failures."""

    def __init__(self, providers: Mapping[str, Any], *, policy: ToolGatewayPolicy | None = None) -> None:
        self.policy = policy or ToolGatewayPolicy()
        self.providers = dict(providers)

    def execute(self, request: Mapping[str, Any], *, allow_external: bool = False) -> ToolExecutionReport:
        gateway = ToolGateway(self.policy)
        prepared = gateway.prepare(request, allow_external=allow_external)
        tool_name = prepared["tool_name"]
        candidates = [provider for provider in self.providers.values() if getattr(provider, "tool_name", tool_name) == tool_name]
        if not candidates:
            error = ToolGatewayError("provider_unavailable", f"no provider is registered for {tool_name}", status_code=503, retryable=True)
            return ToolExecutionReport(error_result(prepared["request_id"], error), "", False, ({"provider": "", "status": "failed", "code": error.code},))
        attempts: list[dict[str, Any]] = []
        for index, provider in enumerate(candidates):
            provider_id = str(getattr(provider, "provider_id", provider.__class__.__name__))[:64]
            try:
                result = provider.execute(prepared, allow_external=allow_external)
                result = normalize_tool_result(result, request_id=prepared["request_id"], policy=self.policy)
                attempts.append({"provider": provider_id, "status": "ok"})
                return ToolExecutionReport(result, provider_id, index > 0, tuple(attempts))
            except ToolGatewayError:
                raise
            except ToolProviderError as exc:
                attempts.append({"provider": provider_id, "status": "failed", "code": exc.code})
                if not exc.retryable or index == len(candidates) - 1:
                    error = ToolGatewayError(exc.code, str(exc), retryable=exc.retryable)
                    return ToolExecutionReport(error_result(prepared["request_id"], error), provider_id, index > 0, tuple(attempts))
        error = ToolGatewayError("provider_unavailable", "no provider completed the request", status_code=503, retryable=True)
        return ToolExecutionReport(error_result(prepared["request_id"], error), "", True, tuple(attempts))


class FakeToolProvider:
    """Deterministic provider fixture for contract and fallback tests."""

    def __init__(self, tool_name: str, provider_id: str, handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | dict[str, Any] | Exception]) -> None:
        self.tool_name = tool_name
        self.provider_id = provider_id
        self.handler = handler

    def execute(self, request: Mapping[str, Any], *, allow_external: bool = False) -> dict[str, Any]:
        value = self.handler(request)
        if isinstance(value, Exception):
            if isinstance(value, ToolProviderError):
                raise value
            raise ToolProviderError("fake_provider_error", "fake provider failed") from value
        return dict(value)


__all__ = [
    "DEFAULT_USER_AGENT",
    "FakeToolProvider",
    "RestrictedFetchAdapter",
    "SearxSearchAdapter",
    "ToolExecutionReport",
    "ToolGatewayExecutor",
    "ToolProviderError",
    "TransportResponse",
    "ToolTransport",
    "UrllibTransport",
]
