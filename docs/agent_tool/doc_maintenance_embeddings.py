#!/usr/bin/env python3
"""文档维护 Agent M3.4：Ollama embedding 适配器（无强制联网）。"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbeddingUnavailable(RuntimeError):
    """不向报告泄露 URL、响应体或本地配置。"""


@dataclass(frozen=True)
class OllamaEmbeddingProvider:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 60:
            raise ValueError("embedding timeout must be within (0, 60]")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        request = Request(
            f"{base}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "QLH-DocAgent/0.1",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise EmbeddingUnavailable(f"http_{exc.code}") from exc
        except (URLError, TimeoutError, socket.timeout):
            raise EmbeddingUnavailable("transport_error") from None
        try:
            embeddings = json.loads(raw)["embeddings"]
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError
            vectors = [[float(value) for value in vector] for vector in embeddings]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise EmbeddingUnavailable("invalid_envelope") from None
        if any(not vector for vector in vectors):
            raise EmbeddingUnavailable("empty_vector")
        return vectors
