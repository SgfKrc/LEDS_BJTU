"""ChatScreen 单元测试（TUI 重写 P2 准备工作）

覆盖：
    * Screen 鸭子契约（name / auto_refresh / fetch / lines / actions）
    * 事件分派：start / token / done / cancelled / error
    * 安全：ANSI 与危险控制字符过滤
    * 渲染：中文按 2 列折行
    * 离线 fixture 回放（不联网、不依赖 tui_admin 的终端）
    * 边界：未收到任何内容时清理占位消息

设计说明：
    ChatScreen 通过关键字参数 `transport` 支持注入传输层，因此全部用例**不打网络**。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from tui_chat_screen import ChatScreen, sanitize, wrap_display  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                       "chat_interactive_fixture.sse")


class FakeApi:
    """最小 ApiClient 替身：记录调用，返回可控数据。"""

    base_url = "http://127.0.0.1:8000"

    def __init__(self, sessions=None):
        self.calls = []
        self._sessions = sessions if sessions is not None else {"sessions": []}

    def get(self, path, **kw):
        self.calls.append(("get", path, kw))
        return self._sessions

    def post(self, path, body=None, **kw):
        self.calls.append(("post", path, body))
        return {"ok": True}


class FakeApp:
    def __init__(self, api=None):
        self.api = api if api is not None else FakeApi()


class FakeUI:
    """BaseUI 替身：prompt 返回预设值。"""

    def __init__(self, answer=""):
        self.answer = answer
        self.labels = []

    def prompt(self, label, default=""):
        self.labels.append(label)
        return self.answer

    def confirm(self, label):
        return True


def make_screen(events, **kw):
    """构造一个由给定 payload 序列驱动的 ChatScreen。"""
    payloads = list(events)

    def transport(_body):
        for p in payloads:
            yield p

    return ChatScreen(FakeApp(), transport=transport, **kw), payloads


# ----------------------------------------------------------------------
# 1. Screen 鸭子契约
# ----------------------------------------------------------------------

class TestDuckContract:
    def test_has_screen_attributes(self):
        s = ChatScreen(FakeApp())
        assert s.name == "对话"
        assert s.auto_refresh is False

    def test_has_screen_methods(self):
        s = ChatScreen(FakeApp())
        for meth in ("refresh", "fetch", "lines", "actions"):
            assert callable(getattr(s, meth)), meth

    def test_actions_shape(self):
        """actions() 必须返回 [(key, label, handler(ui) -> str|None)]。"""
        s = ChatScreen(FakeApp())
        acts = s.actions()
        assert acts and all(len(a) == 3 for a in acts)
        keys = [a[0] for a in acts]
        assert keys == ["i", "n", "c", "x"]
        for _k, _label, handler in acts:
            assert callable(handler)

    def test_lines_before_and_after_message(self):
        s = ChatScreen(FakeApp())
        assert any("还没有消息" in t for _st, t in s.lines(80))
        s.messages.append(("user", "你好"))
        s.messages.append(("assistant", "你好，有什么可以帮你？"))
        rendered = "".join(t for _st, t in s.lines(80))
        assert "你:" in rendered and "模型:" in rendered
        assert "你好，有什么可以帮你？" in rendered

    def test_fetch_without_api_is_safe(self):
        """app.api 缺失时不抛异常（防御式）。"""
        s = ChatScreen(FakeApp(api=None))
        s.api = None
        s.refresh(force=True)
        assert s.data is None


# ----------------------------------------------------------------------
# 2. 事件分派
# ----------------------------------------------------------------------

class TestEventDispatch:
    def test_start_token_done(self):
        s, _ = make_screen([
            {"start": True, "generation_id": "g1", "session_id": "s1"},
            {"token": "你"},
            {"token": "好"},
            {"done": True, "response": "你好", "session_id": "s1",
             "history_committed": True,
             "metrics": {"engine": "fixture", "execution_mode": "local",
                         "tokens_generated": 2, "tok_per_sec": 10.0}},
        ])
        s.send("打招呼")
        assert s.session_id == "s1"
        assert s.streaming is False
        assert s.generation_id is None
        assert s.messages[-1] == ("assistant", "你好")
        assert s.last_metrics and s.last_metrics["execution_mode"] == "local"
        assert s.last_status and "local" in s.last_status

    def test_incremental_tokens_visible_during_stream(self):
        """增量 token 应立即反映到 messages（供主循环重绘）。"""
        seen = []

        def transport(_body):
            yield {"token": "A"}
            seen.append(s.messages[-1][1])
            yield {"token": "B"}
            seen.append(s.messages[-1][1])

        s = ChatScreen(FakeApp(), transport=transport)
        s.send("x")
        assert seen == ["A", "AB"]

    def test_cancelled_uses_partial(self):
        s, _ = make_screen([
            {"start": True, "generation_id": "g2"},
            {"token": "取消示例"},
            {"cancelled": True, "generation_id": "g2", "partial": "取消示例：这条流会被中断"},
        ])
        s.send("x")
        assert s.messages[-1] == ("assistant", "取消示例：这条流会被中断")
        assert s.error and "取消" in s.error

    def test_error_event(self):
        s, _ = make_screen([{"error": {"message": "模型未加载"}}])
        s.send("x")
        assert s.error and "模型未加载" in s.error

    def test_transport_exception_does_not_raise(self):
        def transport(_body):
            raise RuntimeError("连接被拒绝")
            yield  # pragma: no cover

        s = ChatScreen(FakeApp(), transport=transport)
        s.send("x")                     # 不应抛出
        assert s.error and "连接被拒绝" in s.error
        assert s.streaming is False

    def test_placeholder_removed_when_no_content(self):
        """只收到 start 没有 token/done 时，不应残留空的 assistant 占位。"""
        s, _ = make_screen([{"start": True, "generation_id": "g3"}])
        s.send("x")
        assert all(not (r == "assistant" and not t) for r, t in s.messages)

    def test_request_body_uses_interactive_mode(self):
        captured = {}

        def transport(body):
            captured.update(body)
            yield {"done": True, "response": "ok"}

        s = ChatScreen(FakeApp(), transport=transport, session_id="sess-1",
                       routing_preference="local_only", max_new_tokens=64)
        s.send("问一句")
        assert captured["message"] == "问一句"
        assert captured["streaming_mode"] == "interactive"
        assert captured["session_id"] == "sess-1"
        assert captured["max_new_tokens"] == 64
        assert captured["routing_preference"] == "local_only"


# ----------------------------------------------------------------------
# 3. 动作
# ----------------------------------------------------------------------

class TestActions:
    def test_send_action_uses_prompt(self):
        s, _ = make_screen([{"done": True, "response": "ok"}])
        ui = FakeUI(answer="  hi  ")
        s.actions()[0][2](ui)                 # ("i", ..., _act_send)
        assert ui.labels == ["你: "]
        assert s.messages[0] == ("user", "hi")

    def test_send_action_ignores_blank(self):
        s, _ = make_screen([{"done": True, "response": "ok"}])
        s.actions()[0][2](FakeUI(answer="   "))
        assert s.messages == []

    def test_new_session_clears_state(self):
        s, _ = make_screen([{"start": True, "session_id": "s1"}, {"token": "x"},
                            {"done": True, "response": "x"}])
        s.send("a")
        s.actions()[1][2](FakeUI())           # ("n", ...)
        assert s.session_id is None and s.messages == [] and s.last_status is None

    def test_cancel_when_idle(self):
        s = ChatScreen(FakeApp())
        msg = s.actions()[3][2](FakeUI())      # ("x", ...)
        assert msg and "没有进行中" in msg

    def test_cancel_sends_request(self):
        api = FakeApi()
        s = ChatScreen(FakeApp(api=api))
        s.streaming = True
        s.generation_id = "g9"
        s.actions()[3][2](FakeUI())
        assert any(c[0] == "post" and "g9" in c[1] for c in api.calls)

    def test_cancel_quotes_generation_id(self):
        api = FakeApi()
        s = ChatScreen(FakeApp(api=api))
        s.cancel("a/b c")
        posted = [c[1] for c in api.calls if c[0] == "post"][0]
        assert "a%2Fb%20c" in posted


# ----------------------------------------------------------------------
# 4. 安全与渲染
# ----------------------------------------------------------------------

class TestSanitizeAndWrap:
    @pytest.mark.parametrize("raw,expected", [
        ("\x1b[31mred\x1b[0m", "red"),
        ("plain text", "plain text"),
        ("a\x07b", "ab"),                  # 危险控制字符
        ("keep\nnewline\tand tab", "keep\nnewline\tand tab"),
        ("", ""),
    ])
    def test_sanitize(self, raw, expected):
        assert sanitize(raw) == expected

    def test_ansi_stripped_from_rendered_lines(self):
        s = ChatScreen(FakeApp())
        s.messages.append(("assistant", "\x1b[31m危险\x1b[0m 正常"))
        rendered = "".join(t for _st, t in s.lines(80))
        assert "\x1b" not in rendered
        assert "危险 正常" in rendered

    def test_wrap_display_cjk_width(self):
        """中文按 2 列：宽度 10 时每行最多 5 个汉字。"""
        lines = wrap_display("一二三四五六七八九十", 10)
        assert lines[0] == "一二三四五"
        assert lines[1] == "六七八九十"

    def test_wrap_display_keeps_blank_lines(self):
        assert wrap_display("a\n\nb", 10) == ["a", "", "b"]


# ----------------------------------------------------------------------
# 5. 离线 fixture 回放
# ----------------------------------------------------------------------

def _fixture_payloads():
    """读取 fixture 的全部 payload。

    注意：`SSEDecoder.feed()` 只返回**已完整**的事件；fixture 最后一条事件之后
    没有尾部空行，残帧会留在 buffer 中，因此须再 `feed(b"\\n\\n")` 冲刷 ——
    这是 `tui_sse` 的既定行为（docstring：「尾部无空行的残帧保留到下一个 feed」）。
    """
    from tui_sse import SSEDecoder, decode_json_event

    dec = SSEDecoder()
    events = dec.feed(open(FIXTURE, "rb").read())
    events = events + dec.feed(b"\n\n")       # 冲刷尾部残帧
    return [p for p in (decode_json_event(e) for e in events) if p]


class TestFixtureReplay:
    def test_fixture_exists(self):
        assert os.path.isfile(FIXTURE), FIXTURE

    def test_replay_two_streams(self):
        """fixture 含 2 条流：正常的（done）与被取消的（cancelled）。"""
        payloads = _fixture_payloads()
        starts = [p for p in payloads if p.get("start")]
        dones = [p for p in payloads if p.get("done")]
        cancelled = [p for p in payloads if p.get("cancelled")]
        assert len(starts) == 2
        assert len(dones) == 1
        assert len(cancelled) == 1
        # metrics 必须来自 done 事件
        assert dones[0]["metrics"]["engine"] == "fixture"

    def test_replay_into_screen(self):
        """把 fixture 第一条流（start..done）喂给屏幕，验证端到端状态。"""
        payloads = _fixture_payloads()
        end = next(i for i, p in enumerate(payloads) if p.get("done")) + 1
        first = payloads[:end]
        s = ChatScreen(FakeApp(), transport=lambda _b: iter(first))
        s.send("你好")
        assert s.messages[-1][0] == "assistant"
        assert s.messages[-1][1]                  # 有内容
        assert s.last_status                      # format_metrics 产出
        assert s.session_id == "sess_fixture"


# ----------------------------------------------------------------------
# 6. 会话辅助
# ----------------------------------------------------------------------

class TestSessionLines:
    def test_session_lines_from_api(self):
        s = ChatScreen(FakeApp(api=FakeApi(sessions={"sessions": [
            {"session_id": "s1", "title": "会话一", "message_count": 3},
        ]})))
        s.fetch()
        lines = s.session_lines()
        assert isinstance(lines, list)

    def test_session_lines_empty_when_missing(self):
        s = ChatScreen(FakeApp())
        s.data = None
        assert s.session_lines() == []
