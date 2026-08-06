"""T9 SSE 流式解析器 — 健壮的 Server-Sent Events 增量解码。

处理 TUI 聊天页（src/tui_chat.py）对接 `POST /api/chat/stream` 时的全部
字节级问题：

- TCP chunk 任意切分：一个 chunk 可能只有半个事件，也可能包含多个事件；
- UTF-8 多字节字符拆分：中文字符的 3 字节序列可能跨 chunk 边界；
- 事件分隔：`\\n\\n`、`\\r\\n\\r\\n` 与混合换行；
- 多行 data 字段按 SSE 规范用 `\\n` 连接；
- keepalive 注释行（`:` 开头）与空事件静默跳过；
- 尾部无空行的残帧保留到下一个 feed。

用法::

    decoder = SSEDecoder()
    async for chunk in response.aiter_bytes():
        for event in decoder.feed(chunk):
            payload = json.loads(event["data"])
            ...

该模块不依赖 httpx/Textual，可在任意线程或 async 循环中使用；事件顺序
与字节到达顺序一致。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class SSEDecoder:
    """增量 SSE 解码器：`feed(bytes)` 返回该 chunk 内完成的完整事件。"""

    def __init__(self) -> None:
        self._buffer = bytearray()

    # ---- 对外 API ----

    def feed(self, chunk: bytes) -> List[Dict[str, str]]:
        """吞入一段字节，返回其中已完整的事件列表（空事件/注释跳过）。"""
        self._buffer.extend(chunk)
        events: List[Dict[str, str]] = []
        while True:
            end = _find_event_end(self._buffer)
            if end is None:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[:end]
            event = _parse_event_block(raw)
            if event is not None:
                events.append(event)
        return events

    def feed_text(self, text: str) -> List[Dict[str, str]]:
        """便捷入口：直接喂入文本（自动按 UTF-8 编码）。"""
        return self.feed(text.encode("utf-8"))

    def has_pending(self) -> bool:
        """是否还有未完成（尾部无空行）的残帧字节。"""
        return bool(self._buffer)

    def pending_bytes(self) -> bytes:
        """返回当前残帧字节（调试用）。"""
        return bytes(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()


def _find_event_end(buffer: bytearray) -> Optional[int]:
    """找到事件块的结束偏移（含分隔空行）；找不到返回 None。

    分隔符优先匹配 \\r\\n\\r\\n，其次 \\n\\n；也容忍 \\r\\n\\n 等混合形式。
    """
    size = len(buffer)
    for i in range(size - 1):
        if buffer[i] == 10:  # \n
            # \n\n
            if buffer[i + 1] == 10:
                return i + 2
            # \n\r\n
            if (
                buffer[i + 1] == 13
                and i + 2 < size
                and buffer[i + 2] == 10
            ):
                return i + 3
    return None


def _parse_event_block(raw: bytes) -> Optional[Dict[str, str]]:
    """解析一个完整事件块（不含结尾空行）。

    Returns:
        {"event": str, "data": str, "id": str, "retry": str}
        注释块（全部字段行）或空块返回 None（keepalive，跳过）。
    """
    text = raw.decode("utf-8", errors="replace")
    fields: Dict[str, List[str]] = {}
    last_field: Optional[str] = None
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(":"):
            # 注释/keepalive 行
            last_field = None
            continue
        if line.startswith(" "):
            # SSE 规范：以空白开头的行是上一字段的续行（追加，含换行）
            if last_field is not None:
                fields.setdefault(last_field, []).append("\n" + line[1:])
            continue
        if ":" in line:
            name, _, value = line.partition(":")
            name = name.strip()
            value = value[1:] if value.startswith(" ") else value
            last_field = name
            fields.setdefault(name, []).append(value)
        else:
            last_field = line.strip()
            fields.setdefault(last_field, []).append("")

    data_lines = fields.get("data", [])
    if not data_lines and not fields.get("event"):
        return None  # keepalive 或空块
    return {
        "event": (fields.get("event", ["message"])[-1] if fields.get("event") else "message"),
        "data": "\n".join(data_lines),
        "id": fields.get("id", [""])[-1],
        "retry": fields.get("retry", [""])[-1],
    }


def decode_json_event(event: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """把事件的 data 字段解析为 JSON 字典；空 data 或非 JSON 返回 None。"""
    data = (event.get("data") or "").strip()
    if not data:
        return None
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_sse_bytes(chunk: bytes) -> List[Dict[str, Any]]:
    """一次性便捷入口：解析一整段 SSE 文本，返回 JSON 事件列表。"""
    decoder = SSEDecoder()
    events = decoder.feed(chunk)
    result: List[Dict[str, Any]] = []
    for event in events:
        payload = decode_json_event(event)
        if payload is not None:
            result.append(payload)
    return result
