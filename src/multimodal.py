"""Bounded multimodal message helpers for OpenAI-compatible providers."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Iterable


MAX_CHAT_IMAGES = 4
MAX_CHAT_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CHAT_IMAGE_TOTAL_BYTES = 16 * 1024 * 1024

_IMAGE_DATA_URL_RE = re.compile(
    r"^data:image/(?P<kind>png|jpeg|webp);base64,(?P<payload>[A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)


def _matches_image_signature(kind: str, data: bytes) -> bool:
    if kind == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if kind == "jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if kind == "webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def validate_image_data_urls(values: Iterable[str] | None) -> list[str]:
    """Validate local image payloads without accepting remote URLs or SVG."""
    image_urls = list(values or [])
    if len(image_urls) > MAX_CHAT_IMAGES:
        raise ValueError(f"最多允许 {MAX_CHAT_IMAGES} 张图片")

    total_bytes = 0
    validated: list[str] = []
    max_payload_chars = ((MAX_CHAT_IMAGE_BYTES + 2) // 3) * 4
    for value in image_urls:
        if not isinstance(value, str):
            raise ValueError("图片必须使用 data URL 字符串")
        match = _IMAGE_DATA_URL_RE.fullmatch(value)
        if match is None:
            raise ValueError("图片仅支持 PNG/JPEG/WebP base64 data URL")
        payload = match.group("payload")
        if len(payload) > max_payload_chars:
            raise ValueError("单张图片不得超过 8 MiB")
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("图片 data URL 的 base64 数据无效") from exc
        if not decoded or len(decoded) > MAX_CHAT_IMAGE_BYTES:
            raise ValueError("单张图片必须为 1 byte 至 8 MiB")
        kind = match.group("kind").lower()
        if not _matches_image_signature(kind, decoded):
            raise ValueError("图片 MIME 类型与文件签名不一致")
        total_bytes += len(decoded)
        if total_bytes > MAX_CHAT_IMAGE_TOTAL_BYTES:
            raise ValueError("图片总大小不得超过 16 MiB")
        validated.append(value)
    return validated


def build_openai_user_content(
    text: str,
    image_data_urls: Iterable[str] | None = None,
) -> str | list[dict[str, Any]]:
    """Build OpenAI structured content; callers validate at the API boundary."""
    images = list(image_data_urls or [])
    if not images:
        return str(text)
    content: list[dict[str, Any]] = [{"type": "text", "text": str(text)}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        }
        for image_url in images
    )
    return content


def normalize_openai_message_content(value: Any) -> str | list[dict[str, Any]]:
    """Allow only text and validated inline-image content parts."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError("消息 content 必须为文本或非空结构化数组")

    normalized: list[dict[str, Any]] = []
    image_urls: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError("结构化消息片段必须为对象")
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            normalized.append({"type": "text", "text": part["text"]})
            continue
        image_url = part.get("image_url")
        if (
            part_type == "image_url"
            and isinstance(image_url, dict)
            and isinstance(image_url.get("url"), str)
        ):
            image_urls.append(image_url["url"])
            normalized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url["url"]},
                }
            )
            continue
        raise ValueError("结构化消息仅支持 text 和 image_url 片段")

    validate_image_data_urls(image_urls)
    return normalized
