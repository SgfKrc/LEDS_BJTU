"""Read-only model repository search with bounded source failover.

Search is deliberately separate from the download job service.  It returns a
small, provider-neutral projection and never forwards provider payloads,
credentials, file lists, or raw transport errors to callers.

For ``source=all``/``source=hf`` the order is:

1. Hugging Face without a proxy;
2. Hugging Face through the configured search proxy (default
   ``http://127.0.0.1:7897``);
3. ModelScope directly as the final provider fallback.

The endpoint is fail-closed: a transport/provider failure is not converted to
an empty successful result.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import proxy_config

DEFAULT_SEARCH_PROXY = "http://127.0.0.1:7897"
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_QUERY_LENGTH = 128
REQUEST_TIMEOUT_SECONDS = 8.0


class ModelSearchError(Exception):
    """A safe, coded search failure suitable for an API response."""

    def __init__(self, code: str, message: str, *, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.attempts = attempts or []


class _ProviderSearchError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _safe_proxy(explicit: str | None) -> str:
    """Resolve the search-only proxy without inheriting generic system proxy."""
    if explicit:
        return proxy_config.resolve_http_proxy(explicit, env={})
    configured = (
        os.environ.get("QLH_MODEL_SEARCH_PROXY", "")
        or os.environ.get("QLH_HTTP_PROXY", "")
        or os.environ.get("QLH_HTTPS_PROXY", "")
    )
    if configured:
        try:
            return proxy_config.resolve_http_proxy(configured, env={})
        except ValueError:
            # An invalid optional environment value must not make direct HF
            # search fail; use the documented local fallback instead.
            pass
    return DEFAULT_SEARCH_PROXY


def _http_json(url: str, *, proxy: str = "", timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
    """Fetch JSON with explicit proxy isolation (empty proxy means direct)."""
    handler = urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}
    )
    opener = urllib.request.build_opener(handler)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "QLH-model-search/1",
        },
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if getattr(response, "status", 200) >= 400:
                raise _ProviderSearchError("http_%s" % response.status)
            return json.loads(response.read().decode("utf-8"))
    except _ProviderSearchError:
        raise
    except urllib.error.HTTPError as exc:
        raise _ProviderSearchError("http_%s" % exc.code) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            code = "timeout"
        else:
            code = "network_error"
        raise _ProviderSearchError(code) from exc
    except (TimeoutError, OSError):
        raise _ProviderSearchError("network_error") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _ProviderSearchError("invalid_json") from None


def _provider_items(payload: Any) -> tuple[list[dict[str, Any]], int | None]:
    """Extract common list/envelope shapes while rejecting malformed data."""
    if isinstance(payload, list):
        items = payload
        total = None
    elif isinstance(payload, dict):
        if payload.get("success") is False:
            raise _ProviderSearchError("provider_error")
        data = payload.get("data", payload)
        total_value = payload.get("total")
        if isinstance(data, dict):
            total_value = data.get("total", data.get("total_count", total_value))
            for key in ("models", "Models", "items", "results", "repos"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
            else:
                items = []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        try:
            total = int(total_value) if total_value is not None else None
        except (TypeError, ValueError):
            total = None
    else:
        raise _ProviderSearchError("invalid_payload")
    return [item for item in items if isinstance(item, dict)], total


def _text(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalise_item(item: dict[str, Any], provider: str) -> dict[str, Any] | None:
    repo_id = _text(
        item.get("id") or item.get("modelId") or item.get("model_id")
        or item.get("name") or item.get("repo_id"),
        256,
    )
    if not repo_id:
        return None
    tasks = item.get("tasks") or item.get("task") or item.get("pipeline_tag")
    if isinstance(tasks, str):
        tasks = [tasks] if tasks else []
    elif not isinstance(tasks, list):
        tasks = []
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    result: dict[str, Any] = {
        "id": repo_id,
        "source": provider,
        "display_name": _text(item.get("display_name") or item.get("name") or repo_id, 256),
        "description": _text(item.get("description"), 500),
        "tasks": [_text(task, 80) for task in tasks[:10] if _text(task, 80)],
        "tags": [_text(tag, 80) for tag in tags[:20] if _text(tag, 80)],
        "downloads": _int_or_none(item.get("downloads") or item.get("download_count")),
        "likes": _int_or_none(item.get("likes") or item.get("like_count")),
        "last_modified": _text(item.get("last_modified") or item.get("lastModified"), 64),
        "license": _text(item.get("license"), 128),
        "private": bool(item.get("private", False)),
        "gated": bool(item.get("gated", False)),
        "url": (
            "https://huggingface.co/" if provider == "hf" else "https://modelscope.cn/models/"
        ) + repo_id,
    }
    size = _int_or_none(
        item.get("file_size") or item.get("size") or item.get("size_bytes")
    )
    if size is not None and size >= 0:
        result["size_bytes"] = size
    return result


def _search_hf(query: str, page: int, limit: int, *, proxy: str) -> tuple[list[dict[str, Any]], int | None]:
    params = urllib.parse.urlencode({
        "search": query,
        "page": page,
        "limit": limit,
        "sort": "downloads",
        "direction": "-1",
    })
    payload = _http_json("https://huggingface.co/api/models?%s" % params, proxy=proxy)
    items, total = _provider_items(payload)
    return [normalised for item in items
            if (normalised := _normalise_item(item, "hf")) is not None], total


def _search_modelscope(query: str, page: int, limit: int, *, proxy: str) -> tuple[list[dict[str, Any]], int | None]:
    params = urllib.parse.urlencode({
        "search": query,
        "page_number": page,
        "page_size": limit,
        "sort": "downloads",
    })
    payload = _http_json("https://modelscope.cn/openapi/v1/models?%s" % params, proxy=proxy)
    items, total = _provider_items(payload)
    return [normalised for item in items
            if (normalised := _normalise_item(item, "ms")) is not None], total


def _success(query: str, source: str, provider: str, results: list[dict[str, Any]],
             total: int | None, attempts: list[dict[str, Any]], fallback_used: bool) -> dict[str, Any]:
    return {
        "query": query,
        "source": source,
        "provider": provider,
        "results": results,
        "total": total if total is not None else len(results),
        "fallback_used": fallback_used,
        "attempts": attempts,
    }


def search_models(query: str, *, source: str = "all", page: int = 1,
                  limit: int = DEFAULT_LIMIT, proxy: str | None = None) -> dict[str, Any]:
    """Search a provider with deterministic HF/proxy/ModelScope failover."""
    query = str(query or "").strip()
    source = str(source or "all").lower().strip()
    if not query:
        raise ModelSearchError("QUERY_REQUIRED", "搜索词不能为空")
    if len(query) > MAX_QUERY_LENGTH:
        raise ModelSearchError("QUERY_TOO_LONG", "搜索词长度超过限制")
    if source not in {"hf", "ms", "all"}:
        raise ModelSearchError("SOURCE_INVALID", "source 必须是 hf、ms 或 all")
    try:
        page = max(1, int(page))
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        raise ModelSearchError("PAGINATION_INVALID", "page/limit 参数无效") from None

    try:
        fallback_proxy = _safe_proxy(proxy)
    except ValueError as exc:
        raise ModelSearchError("PROXY_INVALID", "搜索代理必须是 http:// 或 https:// URL") from exc
    attempts: list[dict[str, Any]] = []

    if source in {"hf", "all"}:
        try:
            results, total = _search_hf(query, page, limit, proxy="")
            attempts.append({"provider": "hf", "transport": "direct", "status": "ok"})
            return _success(query, source, "hf", results, total, attempts, False)
        except _ProviderSearchError as exc:
            attempts.append({"provider": "hf", "transport": "direct", "status": "failed", "code": exc.code})
        try:
            results, total = _search_hf(query, page, limit, proxy=fallback_proxy)
            attempts.append({"provider": "hf", "transport": "proxy", "status": "ok"})
            return _success(query, source, "hf", results, total, attempts, True)
        except _ProviderSearchError as exc:
            attempts.append({"provider": "hf", "transport": "proxy", "status": "failed", "code": exc.code})
        if source == "hf":
            raise ModelSearchError("SEARCH_UNAVAILABLE", "Hugging Face 搜索不可用", attempts=attempts)

    try:
        results, total = _search_modelscope(query, page, limit, proxy="")
        attempts.append({"provider": "ms", "transport": "direct", "status": "ok"})
        return _success(query, source, "ms", results, total, attempts, True)
    except _ProviderSearchError as exc:
        attempts.append({"provider": "ms", "transport": "direct", "status": "failed", "code": exc.code})
        raise ModelSearchError("SEARCH_UNAVAILABLE", "模型源搜索不可用", attempts=attempts) from None


__all__ = ["DEFAULT_SEARCH_PROXY", "ModelSearchError", "search_models"]
