"""OpenAI-compatible API surface for the independent harness."""

from .app import APIUnavailable, create_app
from .mapping import APIRequestError, parse_chat_request, response_to_openai, chunk_to_openai

__all__ = [
    "APIRequestError",
    "APIUnavailable",
    "chunk_to_openai",
    "create_app",
    "parse_chat_request",
    "response_to_openai",
]
