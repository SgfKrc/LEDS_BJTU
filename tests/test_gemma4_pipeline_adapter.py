from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, ".")

from scripts.model_tools.gemma4_pipeline_adapter import (  # noqa: E402
    Gemma4AdapterError,
    Gemma4PipelineAdapter,
    load_gemma4_text_layer_assignment,
    select_gemma4_assignment_keys,
    validate_gemma4_assignment,
)


class _Gemma4Like(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(4, 2)
        self.model.language_model.layers = nn.ModuleList([
            nn.Linear(2, 2, bias=False),
            nn.Linear(2, 2, bias=False),
        ])
        self.model.language_model.norm = nn.RMSNorm(2)
        self.model.embed_vision = nn.Linear(2, 2, bias=False)
        self.model.embed_audio = nn.Linear(2, 2, bias=False)
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.config = config


class _FakeRotary(nn.Module):
    def forward(self, hidden_states, position_ids, layer_type):
        marker = 1.0 if layer_type == "full_attention" else 2.0
        value = torch.full(
            (*hidden_states.shape[:2], 1), marker,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return value, value


class _FakeGemma4Layer(nn.Module):
    def __init__(self, layer_idx, layer_type, *, shared=False, store=False):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.layer_type = str(layer_type)
        self.scale = nn.Parameter(torch.tensor(float(layer_idx + 1)))
        self.self_attn = types.SimpleNamespace(
            layer_idx=int(layer_idx),
            is_kv_shared_layer=bool(shared),
            store_full_length_kv=bool(store),
        )

    def forward(
        self,
        hidden_states,
        *,
        shared_kv_states,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        del position_embeddings, attention_mask, position_ids
        if self.self_attn.is_kv_shared_layer:
            key, _ = shared_kv_states[self.layer_type]
            shared_value = key[:, :, -1:, :].mean(dim=1)
            hidden_states = hidden_states + shared_value
        else:
            key = hidden_states.unsqueeze(1)
            value = (hidden_states * 0.5).unsqueeze(1)
            key, value = past_key_values.update(
                key, value, self.layer_idx,
            )
            if self.self_attn.store_full_length_kv:
                shared_kv_states[self.layer_type] = (key, value)
        return hidden_states + self.scale


class _FakeGemma4TextBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.layers = nn.ModuleList([
            _FakeGemma4Layer(0, "full_attention", store=True),
            _FakeGemma4Layer(1, "sliding_attention", store=True),
            _FakeGemma4Layer(2, "full_attention", shared=True),
            _FakeGemma4Layer(3, "sliding_attention", shared=True),
        ])
        self.norm = nn.Identity()
        self.rotary_emb = _FakeRotary()


class _FakeGemma4Conditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _FakeGemma4TextBody()
        self.lm_head = nn.Linear(3, 8, bias=False)
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_hidden_layers=4,
                layer_types=[
                    "full_attention", "sliding_attention",
                    "full_attention", "sliding_attention",
                ],
            ),
        )


def _config():
    text = types.SimpleNamespace(
        num_hidden_layers=2,
        tie_word_embeddings=False,
    )
    return types.SimpleNamespace(
        model_type="gemma4_unified",
        text_config=text,
    )


def _write_filtered_first_segment(root):
    (root / "config.json").write_text(
        json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {
                "num_hidden_layers": 2,
                "tie_word_embeddings": False,
            },
        }),
        encoding="utf-8",
    )
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.language_model.layers.0.weight": torch.eye(2, dtype=torch.float32),
        "model.language_model.norm.weight": torch.ones(2, dtype=torch.float32),
    }
    save_file(tensors, str(root / "segment.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                key: "segment.safetensors" for key in tensors
            },
            "metadata": {"total_size": 0},
        }),
        encoding="utf-8",
    )


def test_assignment_uses_official_nested_namespace_and_excludes_modalities():
    keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.weight",
        "model.language_model.layers.1.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.embed_vision.proj.weight",
        "model.embed_audio.proj.weight",
    ]
    selected = select_gemma4_assignment_keys(
        keys,
        start_layer=0,
        end_layer=1,
        has_embedding=True,
        has_lm_head=False,
    )
    assert selected == [keys[0], keys[1], keys[3]]
    report = validate_gemma4_assignment(
        model_type="gemma4_unified",
        total_layers=2,
        start_layer=0,
        end_layer=1,
        has_embedding=True,
        has_lm_head=False,
        keys=selected,
    )
    assert report["layer_prefix"] == "model.language_model.layers."
    assert report["multimodal_materialized"] is False
    assert report["full_model_materialized"] is False


def test_assignment_rejects_wrong_identity_and_missing_layers():
    with pytest.raises(Gemma4AdapterError, match="model_type=gemma4_unified"):
        validate_gemma4_assignment(
            model_type="gemma",
            total_layers=2,
            start_layer=0,
            end_layer=1,
            has_embedding=False,
            has_lm_head=False,
            keys=["model.language_model.layers.0.weight"],
        )
    with pytest.raises(Gemma4AdapterError, match="missing layers"):
        validate_gemma4_assignment(
            model_type="gemma4_unified",
            total_layers=2,
            start_layer=0,
            end_layer=2,
            has_embedding=False,
            has_lm_head=False,
            keys=["model.language_model.layers.0.weight"],
        )


def test_filtered_loader_materializes_only_the_assigned_text_segment(
    tmp_path,
    monkeypatch,
):
    import transformers

    _write_filtered_first_segment(tmp_path)
    config = _config()
    auto_config = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: config,
    )
    auto_model = types.SimpleNamespace(
        from_config=lambda *_args, **_kwargs: _Gemma4Like(config),
    )
    monkeypatch.setattr(transformers, "AutoConfig", auto_config)
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        auto_model,
        raising=False,
    )

    model, metrics = load_gemma4_text_layer_assignment(
        tmp_path,
        start_layer=0,
        end_layer=1,
        has_embedding=True,
        has_lm_head=False,
        device="cpu",
        dtype="fp32",
    )

    assert len(model.model.language_model.layers) == 1
    assert model.model.language_model.embed_tokens is not None
    assert model.lm_head is None
    assert model.model.embed_vision is None
    assert model.model.embed_audio is None
    assert not [
        name for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    assert metrics["selected_tensor_count"] == 3
    assert metrics["source_tensor_bytes"] == 56
    assert metrics["full_model_materialized"] is False
    assert metrics["multimodal_materialized"] is False


def test_filtered_loader_keeps_tied_output_weight_without_embedding_module(
    tmp_path,
    monkeypatch,
):
    import transformers

    config = _config()
    config.text_config.tie_word_embeddings = True
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(
            8, dtype=torch.float32,
        ).reshape(4, 2),
        "model.language_model.layers.1.weight": torch.eye(2, dtype=torch.float32),
        "model.language_model.norm.weight": torch.ones(2, dtype=torch.float32),
    }
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "gemma4_unified",
            "text_config": {
                "num_hidden_layers": 2,
                "tie_word_embeddings": True,
            },
        }),
        encoding="utf-8",
    )
    save_file(tensors, str(tmp_path / "segment.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {key: "segment.safetensors" for key in tensors},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        types.SimpleNamespace(from_config=lambda *_args, **_kwargs: _Gemma4Like(config)),
        raising=False,
    )

    model, metrics = load_gemma4_text_layer_assignment(
        tmp_path,
        start_layer=1,
        end_layer=2,
        has_embedding=False,
        has_lm_head=True,
        device="cpu",
        dtype="fp32",
    )

    assert model.model.language_model.embed_tokens is None
    assert model.lm_head is not None
    assert torch.equal(model.lm_head.weight, tensors["model.language_model.embed_tokens.weight"])
    assert metrics["tie_word_embeddings"] is True
    assert metrics["selected_tensor_count"] == 3


def test_filtered_loader_rejects_an_unassigned_layer_before_materialization(
    tmp_path,
    monkeypatch,
):
    import transformers

    _write_filtered_first_segment(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["model.language_model.layers.1.weight"] = "segment.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    config = _config()
    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        types.SimpleNamespace(from_config=lambda *_args, **_kwargs: _Gemma4Like(config)),
        raising=False,
    )

    with pytest.raises(Gemma4AdapterError, match="unassigned keys"):
        load_gemma4_text_layer_assignment(
            tmp_path,
            start_layer=0,
            end_layer=1,
            has_embedding=True,
            has_lm_head=False,
            device="cpu",
        )


def _segment_forward(adapter, *, input_ids=None, hidden_states=None, cache=None, shared=None):
    return adapter.forward(
        input_ids=input_ids,
        hidden_states=hidden_states,
        past_key_values=cache,
        shared_kv_states=shared,
        use_cache=True,
        attention_mask={
            "full_attention": None,
            "sliding_attention": None,
        },
    )


def test_two_segment_prefill_and_decode_preserve_shared_kv_parity():
    torch.manual_seed(7)
    reference_model = _FakeGemma4Conditional()
    segmented_model = _FakeGemma4Conditional()
    segmented_model.load_state_dict(reference_model.state_dict())
    reference = Gemma4PipelineAdapter(
        reference_model,
        start_layer=0,
        end_layer=4,
        has_embedding=True,
        has_lm_head=True,
    )
    first = Gemma4PipelineAdapter(
        segmented_model,
        start_layer=0,
        end_layer=2,
        has_embedding=True,
        has_lm_head=False,
    )
    last = Gemma4PipelineAdapter(
        segmented_model,
        start_layer=2,
        end_layer=4,
        has_embedding=False,
        has_lm_head=True,
    )
    input_ids = torch.tensor([[1, 2, 3]])

    full_prefill = _segment_forward(reference, input_ids=input_ids)
    first_prefill = _segment_forward(first, input_ids=input_ids)
    last_prefill = _segment_forward(
        last,
        hidden_states=first_prefill["hidden_states"],
        shared=first_prefill["shared_kv_states"],
    )
    assert torch.allclose(last_prefill["logits"], full_prefill["logits"])
    assert first_prefill["shared_kv_sequence_lengths"] == {
        "full_attention": 3,
        "sliding_attention": 3,
    }
    assert last_prefill["sequence_length"] == 3

    next_ids = torch.tensor([[4]])
    full_decode = _segment_forward(
        reference,
        input_ids=next_ids,
        cache=full_prefill["past_key_values"],
        shared=full_prefill["shared_kv_states"],
    )
    first_decode = _segment_forward(
        first,
        input_ids=next_ids,
        cache=first_prefill["past_key_values"],
        shared=first_prefill["shared_kv_states"],
    )
    last_decode = _segment_forward(
        last,
        hidden_states=first_decode["hidden_states"],
        cache=last_prefill["past_key_values"],
        shared=first_decode["shared_kv_states"],
    )
    assert torch.allclose(last_decode["logits"], full_decode["logits"])
    assert last_decode["sequence_length"] == 4
    assert last_decode["shared_kv_sequence_lengths"] == {
        "full_attention": 4,
        "sliding_attention": 4,
    }


def test_shared_kv_consumer_fails_before_layer_when_handoff_is_missing():
    adapter = Gemma4PipelineAdapter(
        _FakeGemma4Conditional(),
        start_layer=2,
        end_layer=4,
        has_embedding=False,
        has_lm_head=True,
    )
    with pytest.raises(Gemma4AdapterError, match="shared-KV source is missing"):
        _segment_forward(adapter, hidden_states=torch.zeros(1, 2, 3))
