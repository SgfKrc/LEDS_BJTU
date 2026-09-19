"""★ 深度思考开关（enable_thinking）链路测试（2026-09-19）。

## 背景（用户报告的实际问题）
加载 **Qwen3 0.6B** 时输出会带**超长 `<think>` 部分**。排查发现：

* TUI 只有 `/thinking on|off`，其语义是「思考内容**展示**」（`tui_shared.py` 的 desc），
  即**只控制 UI 显不显示**，**完全不改变模型行为**；
* 引擎层其实**早有真正的开关** —— `llama_engine._set_thinking_mode()` 会设
  `chat_template_kwargs = {"enable_thinking": bool(enabled)}`（对 `qwen3_chat_v1` 模板）；
* **但 `enable_thinking` 在 `api_server` / `scheduler` / `inference_client` 里 0 处**
  ⇒ 没有任何上层能把它传下去；
* 唯一的替代是 `scheduler` 里的**事后剥离**（找 `</think>` 截断）—— 它依赖「模板含 `<think>`」
  与「能找到 `</think>`」两个脆弱判断，任一不成立即失效，**且算力已经浪费**。

本测试锁定新链路：**TUI → api_server → scheduler → 引擎** 的 `enable_thinking` 透传与语义。
"""

from __future__ import annotations

from tui_shared import build_interactive_request


class TestBuildInteractiveRequestCarriesSwitch:
    """`tui_shared.build_interactive_request` 必须把开关带进请求体。"""

    def test_default_is_none(self):
        body = build_interactive_request("hi")
        assert body["show_thinking"] is False
        assert body["enable_thinking"] is None, "默认应为 None（沿用模型模板默认，不干预）"

    def test_off_is_forwarded(self):
        body = build_interactive_request("hi", enable_thinking=False)
        assert body["enable_thinking"] is False

    def test_on_is_forwarded(self):
        body = build_interactive_request("hi", enable_thinking=True)
        assert body["enable_thinking"] is True

    def test_independent_from_show_thinking(self):
        """两者语义独立：可以「不展示」但「仍思考」，也可以「强制不思考」。"""
        body = build_interactive_request("hi", show_thinking=True, enable_thinking=False)
        assert body["show_thinking"] is True
        assert body["enable_thinking"] is False


class TestApiServerRequestModel:
    """`api_server` 的交互请求模型必须接受该字段（且默认 None 不改变旧行为）。"""

    def test_field_exists_and_defaults_none(self):
        import api_server

        model = None
        for name in dir(api_server):
            obj = getattr(api_server, name)
            fields = getattr(obj, "model_fields", None)
            if isinstance(fields, dict) and "show_thinking" in fields and "enable_thinking" in fields:
                model = obj
                break
        assert model is not None, "未找到同时含 show_thinking / enable_thinking 的请求模型"
        f = model.model_fields["enable_thinking"]
        assert f.default is None, "默认必须是 None（保持向后兼容：不干预模型行为）"
        assert "开关" in (f.description or ""), "字段说明应说明它是「开关」而非「展示」"


class TestEngineSwitchSemantics:
    """引擎层 `_set_thinking_mode` 的语义（只有声明 enable_thinking 的模板才生效）。"""

    def _engine(self, chat_template: str):
        from llama_engine import LlamaCppEngine

        eng = LlamaCppEngine()
        eng._chat_template = chat_template
        return eng

    def test_qwen3_template_sets_flag_false(self):
        eng = self._engine("qwen3_chat_v1")
        eng._set_thinking_mode(False)
        assert eng._chat_template_kwargs == {"enable_thinking": False}
        assert eng._thinking_enabled is False

    def test_qwen3_template_sets_flag_true(self):
        eng = self._engine("qwen3_chat_v1")
        eng._set_thinking_mode(True)
        assert eng._chat_template_kwargs == {"enable_thinking": True}

    def test_none_does_not_interfere(self):
        """`None`（auto）⇒ 不干预，保持空 kwargs（沿用模板默认）。"""
        eng = self._engine("qwen3_chat_v1")
        eng._chat_template_kwargs = {}
        eng._set_thinking_mode(None)
        assert eng._chat_template_kwargs == {}, "auto 不应写入任何 template kwargs"

    def test_other_templates_are_ignored(self):
        """非 qwen3 模板（如 qwen2.5/chatml）不声明该开关 ⇒ 调用应被忽略。"""
        eng = self._engine("chatml")
        eng._chat_template_kwargs = {}
        eng._set_thinking_mode(False)
        assert eng._chat_template_kwargs == {}, "不支持的模板不得被写入 enable_thinking"
