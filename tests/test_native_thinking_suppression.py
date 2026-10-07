"""★ 2026-10-07（真机复验根因）：原生思考抑制判据的回归。

真机现象：Route-A（分布式层段）流式请求收不到任何 `token` 事件，非流式请求报
「流水线返回空响应」，而 pipeline 日志显示逐 token 生成正常。

根因：判据把**模板给模型的指令**（prompt 尾部出现 `<think`）当成了「模型已进入思考」。
`qwen3-5-2b` 的 chat template 在 `enable_thinking` 非 true 时注入的是**已闭合**的
`'<think>\\n\\n</think>\\n\\n'` ⇒ 生成段只含正文、永远等不到 `</think>` ⇒
抑制门把正文整段吞掉（流式 0 token），`_format_model_response` 也据此判成空正文。

正确判据：只有「模板注入的思考块**尚未闭合**」时才需要等到 `</think>` 再外露。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scheduler_pipeline import native_thinking_suppression_required  # noqa: E402

# qwen3-5-2b 的 chat template：enable_thinking 非 true 分支注入**已闭合**思考块。
CLOSED_THINK_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n你好<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# enable_thinking=true 分支：注入**未闭合**的思考开始标记（模型还会继续思考）。
OPEN_THINK_PROMPT = (
    "<|im_start|>user\n你好<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)

PLAIN_PROMPT = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"


def test_closed_thinking_block_in_prompt_does_not_suppress():
    """★ 真机根因：模板已闭合思考块 ⇒ 生成文本是正文，绝不能抑制。"""
    assert native_thinking_suppression_required(False, CLOSED_THINK_PROMPT) is False


def test_open_thinking_block_in_prompt_suppresses():
    """模板注入了未闭合的思考开始标记 ⇒ 保留「未完成思考不外露」的语义。"""
    assert native_thinking_suppression_required(False, OPEN_THINK_PROMPT) is True


def test_plain_prompt_never_suppresses():
    assert native_thinking_suppression_required(False, PLAIN_PROMPT) is False


def test_show_thinking_opt_out_disables_suppression():
    """用户显式要求看思考 ⇒ 一律不外露抑制。"""
    assert native_thinking_suppression_required(True, OPEN_THINK_PROMPT) is False


def test_empty_prompt_is_safe():
    assert native_thinking_suppression_required(False, "") is False
    assert native_thinking_suppression_required(False, None) is False


def test_closed_block_after_open_block_counts_as_closed():
    """prompt 尾部既有 `<think` 又有 `</think>` ⇒ 视为已闭合（尾部窗口内判定）。"""
    prompt = "<|im_start|>assistant\n<think>\n想一下\n</think>\n\n"
    assert native_thinking_suppression_required(False, prompt) is False
