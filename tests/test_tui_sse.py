"""
T9 SSE parser 单元测试
=====================
覆盖计划 §9.9 测试矩阵的 SSE parser 层：任意 chunk 切分、UTF-8 多字节拆分、
一个 chunk 多事件、done/error/cancelled 事件、尾部无空行残帧、keepalive。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

import pytest

from tui_sse import SSEDecoder, decode_json_event, parse_sse_bytes


def _sse_frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class TestChunkSplitting:
    """任意 chunk 切分：一个事件跨多个 chunk。"""

    def test_event_split_at_every_byte_position(self):
        frame = _sse_frame({"token": "你好"})
        for cut in range(1, len(frame)):
            decoder = SSEDecoder()
            events = decoder.feed(frame[:cut]) + decoder.feed(frame[cut:])
            assert len(events) == 1
            payload = decode_json_event(events[0])
            assert payload == {"token": "你好"}

    def test_utf8_multibyte_split(self):
        """中文字符的 3 字节序列跨 chunk 边界不丢失。"""
        frame = _sse_frame({"token": "分布式推理"})
        decoder = SSEDecoder()
        # 在 UTF-8 序列中间切一刀（比如第 7 个字节之后，"分"字可能被拆）
        first = frame[:9]
        rest = frame[9:]
        events = decoder.feed(first) + decoder.feed(rest)
        assert len(events) == 1
        assert decode_json_event(events[0]) == {"token": "分布式推理"}

    def test_one_chunk_many_events(self):
        chunk = (
            _sse_frame({"token": "a"})
            + _sse_frame({"token": "b"})
            + _sse_frame({"done": True, "response": "ab"})
        )
        events = SSEDecoder().feed(chunk)
        assert len(events) == 3
        assert [decode_json_event(e) for e in events] == [
            {"token": "a"},
            {"token": "b"},
            {"done": True, "response": "ab"},
        ]

    def test_mixed_line_endings(self):
        raw = b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\n\n"
        events = SSEDecoder().feed(raw)
        assert len(events) == 2


class TestEventSemantics:
    """done/error/cancelled 事件与 keepalive。"""

    def test_done_error_cancelled_events(self):
        raw = (
            _sse_frame({"token": "x"})
            + _sse_frame({"done": True, "response": "x", "generation_id": "gen_1"})
            + _sse_frame({"error": "boom"})
            + _sse_frame({"cancelled": True, "generation_id": "gen_2"})
        )
        payloads = parse_sse_bytes(raw)
        assert payloads[0] == {"token": "x"}
        assert payloads[1]["done"] is True and payloads[1]["generation_id"] == "gen_1"
        assert payloads[2] == {"error": "boom"}
        assert payloads[3] == {"cancelled": True, "generation_id": "gen_2"}

    def test_keepalive_comment_lines_skipped(self):
        raw = b": keep-alive\n\n" + _sse_frame({"token": "ok"})
        events = SSEDecoder().feed(raw)
        assert len(events) == 1
        assert decode_json_event(events[0]) == {"token": "ok"}

    def test_pending_tail_without_blank_line(self):
        decoder = SSEDecoder()
        frame = _sse_frame({"token": "t"})
        events = decoder.feed(frame[:-1])  # 去掉结尾空行
        assert events == []
        assert decoder.has_pending()
        events = decoder.feed(b"\n")
        assert len(events) == 1
        assert decode_json_event(events[0]) == {"token": "t"}
        assert not decoder.has_pending()

    def test_multiline_data_field_joined_by_newline(self):
        # SSE 规范：data 字段多行时按 \n 连接（同一事件的多个 data 行）
        raw = b"data: {\"token\": \"a\"\ndata: \"b\"}\n\n"
        events = SSEDecoder().feed(raw)
        assert len(events) == 1
        assert events[0]["data"] == '{"token": "a"\n"b"}'

    def test_crlf_only_separator(self):
        raw = b"data: {\"token\": \"x\"}\r\n\r\n"
        events = SSEDecoder().feed(raw)
        assert len(events) == 1


class TestDecodeJsonEvent:
    """data 字段 JSON 解析的容错。"""

    def test_non_json_data_returns_none(self):
        assert decode_json_event({"event": "message", "data": "not json"}) is None

    def test_empty_data_returns_none(self):
        assert decode_json_event({"event": "message", "data": ""}) is None

    def test_json_array_data_returns_none(self):
        assert decode_json_event({"event": "message", "data": "[1,2]"}) is None

    def test_event_type_preserved(self):
        decoder = SSEDecoder()
        events = decoder.feed(b"event: custom\ndata: {\"a\":1}\n\n")
        assert events[0]["event"] == "custom"
