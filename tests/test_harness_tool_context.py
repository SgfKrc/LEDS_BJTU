import pytest

from harness_workbench.model_profiles import CapabilityState, ModelProfile
from harness_workbench.tools import TOOL_CONTEXT_SCHEMA, ToolContextBuilder, ToolContextError, ToolContextPolicy


def _profile(*, reinjection="verified", tool_call="verified", status="verified", production=True):
    capabilities = {
        "json_output": CapabilityState("verified", ("runtime_fixture_v1",)),
        "tool_call_generation": CapabilityState(tool_call, ("runtime_fixture_v1",)) if tool_call else CapabilityState(),
        "tool_result_reinjection": CapabilityState(reinjection, ("runtime_fixture_v1",)) if reinjection else CapabilityState(),
        "multimodal": CapabilityState("verified", ("runtime_fixture_v1",)),
        "thinking_control": CapabilityState("verified", ("runtime_fixture_v1",)),
    }
    return ModelProfile(
        model_id="tool-answer",
        revision="fixture-v1",
        backend="llama_server",
        artifact_sha256="a" * 64,
        tokenizer_digest="b" * 64,
        chat_template_digest="c" * 64,
        context={"n_ctx": 2048, "input_budget": 1400},
        capabilities=capabilities,
        status=status,
        production_eligible=production,
        evidence={"fixture_set": "tool-context-v1", "artifact_digest_mode": "full_stream"},
    )


def _request(tool_name="web_search", request_id="req_context_01"):
    return {"tool_name": tool_name, "request_id": request_id}


def _result(request_id="req_context_01", schema="qlh.tool_result.v1"):
    return {
        "schema": schema,
        "request_id": request_id,
        "tool_name": "web_search",
        "status": "ok",
        "items": [{"title": "Docs", "url": "https://docs.example/qlh", "snippet": "bounded source"}],
        "citations": [{"url": "https://docs.example/qlh", "sha256": "a" * 64}],
        "truncated": False,
    }


def test_context_builder_requires_verified_reinjection_and_returns_exact_bounded_contract():
    context = ToolContextBuilder().build(_request(), _result(), profile=_profile())

    assert context == {
        "schema": TOOL_CONTEXT_SCHEMA,
        "role": "tool",
        "name": "web_search",
        "request_id": "req_context_01",
        "items": [{"title": "Docs", "url": "https://docs.example/qlh", "snippet": "bounded source"}],
        "citations": [{"url": "https://docs.example/qlh", "sha256": "a" * 64}],
        "truncated": False,
    }


@pytest.mark.parametrize("profile", [_profile(status="candidate"), _profile(reinjection="declared"), _profile(reinjection="rejected")])
def test_unverified_profile_is_fail_closed(profile):
    with pytest.raises(ToolContextError) as exc:
        ToolContextBuilder().build(_request(), _result(), profile=profile)
    assert exc.value.code == "capability_not_verified"


def test_missing_profile_or_decision_is_fail_closed():
    with pytest.raises(ToolContextError, match="verified capability") as exc:
        ToolContextBuilder().build(_request(), _result())
    assert exc.value.code == "capability_not_verified"


def test_autonomous_mode_requires_tool_call_generation_in_addition_to_reinjection():
    profile = _profile(tool_call="declared")
    with pytest.raises(ToolContextError) as exc:
        ToolContextBuilder().build(_request(), _result(), profile=profile, mode="autonomous_tools")
    assert exc.value.code == "autonomous_tools_not_allowed"


def test_fetch_rich_local_item_is_reduced_to_untrusted_bounded_snippet():
    request = _request("web_fetch")
    result = _result()
    result["tool_name"] = "web_fetch"
    result["items"] = [{
        "url": "https://docs.example/qlh",
        "final_url": "https://docs.example/qlh",
        "text": "page text",
        "content_type": "text/html",
        "headers": {"set-cookie": "must not pass"},
    }]
    context = ToolContextBuilder().build(request, result, profile=_profile())
    assert context["items"] == [{"title": "https://docs.example/qlh", "url": "https://docs.example/qlh", "snippet": "page text"}]
    assert "headers" not in context["items"][0]


def test_budget_truncation_is_explicit_and_preserves_item_boundaries():
    result = _result()
    result["items"] = [
        {"title": "A", "url": "https://docs.example/a", "snippet": "a" * 500},
        {"title": "B", "url": "https://docs.example/b", "snippet": "b" * 500},
    ]
    result["citations"] = [
        {"url": "https://docs.example/a", "sha256": "a" * 64},
        {"url": "https://docs.example/b", "sha256": "b" * 64},
    ]
    context = ToolContextBuilder(policy=ToolContextPolicy(max_item_chars=400, max_total_chars=512)).build(
        _request(), result, profile=_profile()
    )
    assert context["truncated"] is True
    assert sum(len(item["title"]) + len(item["snippet"]) for item in context["items"]) <= 512
    assert all(set(item) == {"title", "url", "snippet"} for item in context["items"])


@pytest.mark.parametrize("bad_result,code", [
    ({"status": "error"}, "tool_result_error"),
    ({"schema": "wrong", "status": "ok"}, "invalid_schema"),
    ({"schema": "qlh.tool_result.v1", "status": "ok", "request_id": "other"}, "request_id_mismatch"),
])
def test_result_identity_and_status_are_fail_closed(bad_result, code):
    value = _result()
    value.update(bad_result)
    with pytest.raises(ToolContextError) as exc:
        ToolContextBuilder().build(_request(), value, profile=_profile())
    assert exc.value.code == code


def test_citations_are_scoped_to_retained_items_and_invalid_citation_is_rejected():
    result = _result()
    result["citations"] = [
        {"url": "https://docs.example/other", "sha256": "b" * 64},
        {"url": "https://docs.example/qlh", "sha256": "a" * 64},
    ]
    context = ToolContextBuilder().build(_request(), result, profile=_profile())
    assert context["citations"] == [{"url": "https://docs.example/qlh", "sha256": "a" * 64}]

    result["citations"] = [{"url": "https://docs.example/qlh", "sha256": "not-a-digest"}]
    with pytest.raises(ToolContextError) as exc:
        ToolContextBuilder().build(_request(), result, profile=_profile())
    assert exc.value.code == "invalid_citation"
