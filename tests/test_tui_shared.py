"""
T9 共享层（src/tui_shared.py）单元测试
====================================
覆盖：interactive 请求体构造、metrics 格式化、路由参数解析、命令注册表。
"""

import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_shared import (
    API_PATHS,
    COMMAND_SPECS,
    build_interactive_request,
    format_metrics,
    help_text,
    load_local_chat_image,
    parse_session_line,
    resolve_route_arg,
)


class TestBuildInteractiveRequest:
    def test_minimal_request(self):
        body = build_interactive_request("你好")
        assert body["message"] == "你好"
        assert body["streaming_mode"] == "interactive"
        assert body["routing_preference"] == "auto"
        assert body["show_thinking"] is False
        assert body["generation_id"] is None

    def test_full_request(self):
        body = build_interactive_request(
            "hi",
            session_id="s1",
            generation_id="gen_x",
            routing_preference="distributed_required",
            show_thinking=True,
        )
        assert body["session_id"] == "s1"
        assert body["generation_id"] == "gen_x"
        assert body["routing_preference"] == "distributed_required"
        assert body["show_thinking"] is True

    def test_invalid_routing_falls_back_to_auto(self):
        body = build_interactive_request("hi", routing_preference="bogus")
        assert body["routing_preference"] == "auto"

    def test_image_request_requires_external_and_preserves_data_url(self):
        image = "data:image/png;base64,iVBORw0KGgo="
        body = build_interactive_request("描述图片", image_data_urls=[image])
        assert body["image_data_urls"] == [image]
        assert body["allow_external"] is True
        assert body["prefer_external"] is True
        assert body["execution_mode"] == "auto"

    def test_image_request_rejects_local_only_route(self):
        with pytest.raises(ValueError, match="local_only"):
            build_interactive_request(
                "描述图片",
                routing_preference="local_only",
                image_data_urls=["data:image/png;base64,iVBORw0KGgo="],
            )


class TestLocalImageLoader:
    def test_loads_png_as_valid_data_url(self, tmp_path):
        image_path = Path(tmp_path) / "tiny.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        image = load_local_chat_image(str(image_path))
        assert image["name"] == "tiny.png"
        assert image["data_url"].startswith("data:image/png;base64,")

    def test_rejects_non_image_file(self, tmp_path):
        image_path = Path(tmp_path) / "not-image.txt"
        image_path.write_text("not an image", encoding="utf-8")
        with pytest.raises(ValueError, match="PNG、JPEG 或 WebP"):
            load_local_chat_image(str(image_path))


class TestFormatMetrics:
    def test_engine_and_tokens(self):
        text = format_metrics(
            {"engine": "llama_cpp", "execution_mode": "local",
             "tokens_generated": 42, "tok_per_sec": 12.34},
        )
        assert "llama_cpp" in text and "local" in text
        assert "42 tokens" in text
        assert "12.3 tok/s" in text

    def test_fallback_reason_shown(self):
        text = format_metrics(
            {"engine": "external_api", "execution_mode": "external_api",
             "fallback": True, "fallback_reason": "timeout"},
        )
        assert "回退" in text and "timeout" in text

    def test_distributed_requested_but_not_used(self):
        text = format_metrics(
            {"engine": "llama_cpp", "execution_mode": "local",
             "distributed_requested": True},
        )
        assert "已请求分布式，实际本地" in text

    def test_history_not_committed(self):
        text = format_metrics({}, history_committed=False)
        assert "历史未提交" in text

    def test_empty_metrics(self):
        assert format_metrics(None) == "unknown · local"


class TestRouteArg:
    def test_full_names(self):
        assert resolve_route_arg("auto") == "auto"
        assert resolve_route_arg("local_only") == "local_only"

    def test_short_aliases(self):
        assert resolve_route_arg("local") == "local_only"
        assert resolve_route_arg("distributed") == "distributed_preferred"
        assert resolve_route_arg("required") == "distributed_required"

    def test_invalid(self):
        assert resolve_route_arg("sideways") is None
        assert resolve_route_arg("") is None


class TestCommandRegistry:
    def test_help_text_mentions_key_commands(self):
        text = help_text()
        for name in ("/new", "/resume", "/route", "/cancel", "/quit"):
            assert name in text

    def test_specs_unique_and_ordered(self):
        names = [spec["name"] for spec in COMMAND_SPECS]
        assert len(names) == len(set(names))
        assert names[0] == "/new" and names[-1] == "/quit"

    def test_session_line(self):
        line = parse_session_line({"session_id": "s1", "title": "调试"})
        assert line == "s1  调试"
        assert parse_session_line({}) == "  (未命名)"


class TestApiPaths:
    def test_cancel_template(self):
        assert API_PATHS["chat_cancel"].format(
            generation_id="gen_x",
        ) == "/chat/generations/gen_x/cancel"
