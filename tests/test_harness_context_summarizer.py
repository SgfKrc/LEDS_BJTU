"""Offline contract tests for the optional Qwen3 summary role."""

from __future__ import annotations

import time

import pytest

from harness_workbench.adapters import AdapterError, AdapterRequest, AdapterResponse
from harness_workbench.context_engine import (
    ContextBudget,
    ContextMessage,
    ContextPolicy,
    ContextPolicyConfig,
    LLMSummarizer,
)
from harness_workbench.context_engine.tokenizer import FixedTokenCounter


MESSAGES = (
    ContextMessage("user", "The deployment uses SQLite.", "u-1", 1),
    ContextMessage("assistant", "Decision: keep the cache local.", "a-1", 1),
)


class FakeAdapter:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.requests: list[AdapterRequest] = []

    def complete(self, request: AdapterRequest) -> AdapterResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return AdapterResponse("summary-1", request.model, self.content or "")


def _state() -> str:
    return '{"what":["deployment uses SQLite"],"decisions":["keep cache local"],"artifacts":[],"open":[],"next":["verify backup"]}'


def test_llm_summarizer_sends_qwen3_state_contract_and_preserves_sources() -> None:
    adapter = FakeAdapter(_state())
    result = LLMSummarizer(adapter).summarize(MESSAGES)

    assert result.validated_state()["decisions"] == ["keep cache local"]
    assert result.source_message_ids == ("u-1", "a-1")
    assert result.notices == ()
    request = adapter.requests[0]
    assert request.model == "Qwen3-0.6B"
    assert request.temperature == 0.0
    assert request.max_tokens == 256
    assert request.messages[0]["role"] == "system"
    assert "qlh.harness.summary_input.v1" in request.messages[1]["content"]


@pytest.mark.parametrize(
    ("content", "error", "reason"),
    [
        ("not json", None, "invalid_json"),
        (
            '{"what":[],"decisions":[],"artifacts":[],"open":[],"next":[],"delete":[]}',
            None,
            "invalid_state",
        ),
        (None, AdapterError("backend unavailable", code="backend_http_error"), "adapter_error"),
    ],
)
def test_llm_summarizer_fail_closed_to_rule_based_summary(content, error, reason) -> None:
    adapter = FakeAdapter(content, error)
    result = LLMSummarizer(adapter).summarize(MESSAGES)

    assert result.validated_state()["what"] == ["The deployment uses SQLite."]
    assert len(result.notices) == 1
    assert result.notices[0].code == "context.summary_fallback"
    assert result.notices[0].details["reason"] == reason


def test_llm_summarizer_timeout_falls_back_without_leaking_backend_error() -> None:
    class SlowAdapter(FakeAdapter):
        def complete(self, request: AdapterRequest) -> AdapterResponse:
            self.requests.append(request)
            time.sleep(0.15)
            return AdapterResponse("summary-1", request.model, _state())

    started = time.monotonic()
    result = LLMSummarizer(SlowAdapter(), timeout_seconds=0.01).summarize(MESSAGES)

    assert time.monotonic() - started < 0.10
    assert result.notices[0].details == {"model": "Qwen3-0.6B", "reason": "timeout"}


def test_context_policy_exposes_summary_fallback_notice() -> None:
    adapter = FakeAdapter("not json")
    policy = ContextPolicy(
        config=ContextPolicyConfig(recent_turns=1, recent_turn_ratio=0.5),
        tokenizer=FixedTokenCounter(),
        summarizer=LLMSummarizer(adapter),
    )
    messages = [
        ContextMessage("user", "old fact " * 20, "old-user", 0),
        ContextMessage("assistant", "old decision", "old-assistant", 0),
        ContextMessage("user", "new question", "new-user", 1),
        ContextMessage("assistant", "new answer", "new-assistant", 1),
    ]

    snapshot = policy.build(messages, ContextBudget(n_ctx=180, max_new_tokens=40, overhead=40))

    assert any(notice.code == "context.summary_fallback" for notice in snapshot.notices)
    assert snapshot.input_tokens <= snapshot.input_budget
