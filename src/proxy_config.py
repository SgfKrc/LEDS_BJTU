"""User-configurable HTTP proxy resolution shared by download workflows."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping
from urllib.parse import urlparse


def _valid_proxy(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("proxy must be an absolute http:// or https:// URL")
    return value


def resolve_http_proxy(explicit: str | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve explicit CLI, QLH, then standard environment proxy settings."""
    values = env if env is not None else os.environ
    if explicit:
        return _valid_proxy(explicit)
    for key in ("QLH_HTTP_PROXY", "QLH_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = values.get(key, "")
        if value:
            return _valid_proxy(value)
    return ""


@contextmanager
def proxy_environment(proxy: str = "", *, environ: dict[str, str] | None = None) -> Iterator[dict[str, str]]:
    """Apply a resolved proxy to a copied environment for a downloader call."""
    target = dict(environ if environ is not None else os.environ)
    if proxy:
        target["HTTP_PROXY"] = proxy
        target["HTTPS_PROXY"] = proxy
        target["http_proxy"] = proxy
        target["https_proxy"] = proxy
    yield target
