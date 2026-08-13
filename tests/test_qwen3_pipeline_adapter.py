from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, ".")

from scripts.model_tools.qwen3_pipeline_adapter import (  # noqa: E402
    Qwen3AdapterError,
    Qwen3PipelineAdapter,
    render_without_thinking,
    select_qwen3_assignment_keys,
    validate_qwen3_assignment,
)
from scripts.model_tools.qwen3_pipeline_sidecar_worker import execute_request  # noqa: E402


class _Block(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, hidden_states, past_key_value=None, use_cache=False, **_kwargs):
        output = hidden_states * (1 + self.scale)
        present = None
        if use_cache:
            current = hidden_states.unsqueeze(2)
            if past_key_value is not None:
                current = torch.cat((past_key_value[0], current), dim=1)
            present = (current, current)
        return output, present


class _Qwen3Like(nn.Module):
    def __init__(self, layers=4, hidden=4, vocab=8):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([_Block((index + 1) / 10) for index in range(layers)])
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = types.SimpleNamespace(model_type="qwen3", num_hidden_layers=layers)
        self.eval()


class _Tokenizer:
    def __init__(self, rendered="<|im_start|>assistant\n"):
        self.rendered = rendered

    def apply_chat_template(self, messages, **kwargs):
        assert messages
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return self.rendered


def test_thinking_is_hard_disabled_and_unsupported_tokenizer_rejected():
    assert "<think>" not in render_without_thinking(_Tokenizer(), [{"role": "user", "content": "OK"}])
    with pytest.raises(Qwen3AdapterError, match="non-empty thinking"):
        render_without_thinking(_Tokenizer("<think>hidden</think>answer"), [{"role": "user", "content": "OK"}])

    class OldTokenizer:
        def apply_chat_template(self, messages):
            return "answer"

    with pytest.raises(Qwen3AdapterError, match="enable_thinking"):
        render_without_thinking(OldTokenizer(), [{"role": "user", "content": "OK"}])


def test_qwen3_assignment_is_filtered_and_fail_closed():
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.2.input_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    selected = select_qwen3_assignment_keys(
        keys, start_layer=0, end_layer=2, has_embedding=True, has_lm_head=False
    )
    assert selected == [keys[0], keys[1], keys[2], keys[4]]
    report = validate_qwen3_assignment(
        model_type="qwen3", total_layers=3, start_layer=1, end_layer=3,
        has_embedding=False, has_lm_head=True,
        keys=["model.layers.1.weight", "model.layers.2.weight", "model.norm.weight", "lm_head.weight"],
    )
    assert report["layer_range"] == [1, 3]
    with pytest.raises(Qwen3AdapterError, match="unsupported"):
        select_qwen3_assignment_keys(
            ["model.visual.weight"], start_layer=0, end_layer=1,
            has_embedding=True, has_lm_head=False,
        )
    with pytest.raises(Qwen3AdapterError, match="missing layers"):
        validate_qwen3_assignment(
            model_type="qwen3", total_layers=3, start_layer=0, end_layer=2,
            has_embedding=False, has_lm_head=False,
            keys=["model.layers.0.weight", "model.norm.weight"],
        )
    with pytest.raises(Qwen3AdapterError, match="missing embedding"):
        validate_qwen3_assignment(
            model_type="qwen3", total_layers=2, start_layer=0, end_layer=1,
            has_embedding=True, has_lm_head=False,
            keys=["model.layers.0.weight", "model.norm.weight"],
        )
    tied = select_qwen3_assignment_keys(
        ["model.embed_tokens.weight", "model.layers.2.weight", "model.norm.weight"],
        start_layer=2, end_layer=3, has_embedding=False, has_lm_head=True,
        tie_word_embeddings=True,
    )
    assert "model.embed_tokens.weight" in tied


def test_sidecar_worker_returns_structured_gate_result():
    result = execute_request({
        "schema_version": 1,
        "operation": "qwen3_pipeline_adapter_preflight",
        "read_only": True,
        "network_access": "disabled",
        "model_type": "qwen3",
        "total_layers": 2,
        "layer_range": [0, 2],
        "has_embedding": True,
        "has_lm_head": True,
        "keys": [
            "model.embed_tokens.weight",
            "model.layers.0.weight",
            "model.layers.1.weight",
            "model.norm.weight",
            "lm_head.weight",
        ],
    })
    assert result["status"] == "ready_for_synthetic_forward"
    assert result["gate_passed"] is True
    assert result["adapter"]["synthetic_forward_ready"] is True


def test_two_qwen3_segments_match_full_forward_and_keep_local_kv():
    torch.manual_seed(20260813)
    full = _Qwen3Like().eval()
    first_model = full
    last_model = full
    first = Qwen3PipelineAdapter(
        first_model, start_layer=0, end_layer=2,
        has_embedding=True, has_lm_head=False,
    )
    last = Qwen3PipelineAdapter(
        last_model, start_layer=2, end_layer=4,
        has_embedding=False, has_lm_head=True,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = full.lm_head(full.model.norm(full.model.layers[3](
            full.model.layers[2](
                full.model.layers[1](full.model.layers[0](full.model.embed_tokens(input_ids))[0])[0]
            )[0]
        )[0])).detach()
    first_result = first.forward(input_ids=input_ids, use_cache=True)
    last_result = last.forward(
        hidden_states=first_result["hidden_states"],
        use_cache=True,
        past_key_values=None,
    )
    assert torch.allclose(last_result["logits"], expected, atol=1e-6, rtol=1e-6)
    assert len(first_result["past_key_values"]) == 2
    assert len(last_result["past_key_values"]) == 2
    assert last_result["past_key_values"][0][0].shape[1] == 3
