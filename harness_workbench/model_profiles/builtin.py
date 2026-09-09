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
    qwen3_common = {
        **common,
        "generation": {"temperature": 0.6, "top_p": 0.95, "thinking": "declared"},
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
            aliases=("qwen-1_8b",),
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
            aliases=("qwen3-0.6b",),
            capabilities={
                **_unknown_capabilities(),
                "thinking_control": CapabilityState(
                    "declared", ("b1_template_probe_isolated_tokenizer",)
                ),
            },
            **qwen3_common,
        ),
        ModelProfile(
            model_id="Qwen2.5-0.5B",
            revision="builtin-qwen25-0.5b-v1",
            backend="llama_server",
            context={"n_ctx": 32768, "input_budget": 3072, "max_new_tokens": 512},
            generation={"temperature": 0.7, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("qwen2.5-0.5b",),
            capabilities=_unknown_capabilities(),
            evidence={**common["evidence"], "core_model_id": "qwen2.5-0.5b", "probe_ticket": "M-SM-B1"},
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="MiniCPM4-0.5B",
            revision="builtin-minicpm4-0.5b-v1",
            backend="llama_server",
            context={"n_ctx": 32768, "input_budget": 3072, "max_new_tokens": 512},
            generation={"temperature": 0.8, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "minicpm4_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("minicpm4-0.5b",),
            capabilities=_unknown_capabilities(),
            evidence={**common["evidence"], "core_model_id": "minicpm4-0.5b", "probe_ticket": "M-SM-B1"},
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="DistilQwen2.5-DS3-0324-7B",
            revision="builtin-distilqwen-ds3-0324-v1",
            backend="llama_server",
            context={"n_ctx": 32768, "input_budget": 8192, "max_new_tokens": 1024},
            generation={"temperature": 0.7, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("distilqwen25-ds3-0324-7b",),
            capabilities=_unknown_capabilities(),
            evidence={**common["evidence"], "core_model_id": "distilqwen25-ds3-0324-7b", "probe_ticket": "DSW-D1"},
            status="candidate",
            production_eligible=False,
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
