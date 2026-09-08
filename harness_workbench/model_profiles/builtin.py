"""Conservative candidate profiles for common small-model roles."""

from __future__ import annotations

from .schema import CapabilityState, ModelProfile


def _unknown_capabilities() -> dict[str, CapabilityState]:
    return {name: CapabilityState() for name in (
        "json_output",
        "tool_call_generation",
        "tool_result_reinjection",
        "multimodal",
        "thinking_control",
    )}


def builtin_profiles() -> tuple[ModelProfile, ...]:
    """Return profiles without binding to a local asset or claiming support."""

    common = {
        "context": {"n_ctx": 4096, "input_budget": 3072, "max_new_tokens": 768},
        "generation": {"temperature": 0.7, "top_p": 0.9, "thinking": "unknown"},
        "resources": {"kv_cache": "unknown", "gpu_layers": "auto"},
        "status": "candidate",
        "production_eligible": False,
        "evidence": {
            "fixture_set": "small-model-core-v1",
            "source": "builtin_candidate",
            "weights_loaded": False,
            "network_used": False,
        },
    }
    return (
        ModelProfile(
            model_id="QW1.8B",
            revision="builtin-qw1-v1",
            backend="llama_server",
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            capabilities=_unknown_capabilities(),
            **common,
        ),
        ModelProfile(
            model_id="Qwen3-0.6B",
            revision="builtin-qwen3-tool-v1",
            backend="transformers_sidecar",
            adaptation={
                "prompt_family": "qwen3_chat_v1",
                "tool_mode": "sidecar_candidate",
                "structured_output": "grammar_first",
                "summary_mode": "state_schema_v1",
            },
            roles=("tool_router", "summarizer"),
            capabilities=_unknown_capabilities(),
            **common,
        ),
        ModelProfile(
            model_id="Gemma-small",
            revision="builtin-gemma-v1",
            backend="llama_server",
            adaptation={
                "prompt_family": "gemma_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            capabilities=_unknown_capabilities(),
            **common,
        ),
    )
