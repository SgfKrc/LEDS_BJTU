"""Metadata-only Safetensors descriptor tests."""

import json
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "src")

from pipeline_model_descriptor import (  # noqa: E402
    PipelineModelDescriptorError,
    inspect_pipeline_model,
)


def _write_qwen2_fixture(root):
    (root / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "tie_word_embeddings": True,
        }),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.norm.weight": torch.zeros(2, dtype=torch.float16),
        "lm_head.weight": torch.zeros(4, 2, dtype=torch.float16),
    }
    save_file(tensors, str(root / "model.safetensors"))


def _write_gemma4_fixture(root):
    (root / "config.json").write_text(
        json.dumps({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": {
                "num_hidden_layers": 2,
                "hidden_size": 2,
                "tie_word_embeddings": True,
            },
            "vision_config": {"mm_embed_dim": 2},
            "audio_config": {"audio_embed_dim": 2},
        }),
        encoding="utf-8",
    )
    save_file({
        "model.language_model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.language_model.layers.1.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
        "model.language_model.norm.weight": torch.zeros(2, dtype=torch.float16),
        "lm_head.weight": torch.zeros(4, 2, dtype=torch.float16),
        "model.embed_vision.proj.weight": torch.zeros(2, 2, dtype=torch.float16),
        "model.embed_audio.proj.weight": torch.zeros(2, 2, dtype=torch.float16),
    }, str(root / "model.safetensors"))


def test_descriptor_reads_headers_without_materializing_weights(tmp_path):
    _write_qwen2_fixture(tmp_path)
    descriptor = inspect_pipeline_model(tmp_path, model_id="fixture")

    assert descriptor["inspection_mode"] == "safetensors_headers_only"
    assert descriptor["pipeline_runtime_supported"] is True
    assert descriptor["model_id"] == "fixture"
    assert descriptor["total_layers"] == 2
    assert descriptor["indexed_tensor_count"] == 5
    assert descriptor["layer_weight_bytes"] == [4, 4]
    assert descriptor["component_weight_bytes"]["embedding"] == 16
    assert descriptor["component_weight_bytes"]["lm_head"] == 16
    assert descriptor["weight_bytes"] == 44


def test_descriptor_rejects_missing_layer(tmp_path):
    _write_qwen2_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(4, 2, dtype=torch.float16),
            "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float16),
            "model.norm.weight": torch.zeros(2, dtype=torch.float16),
        },
        str(path),
    )
    try:
        inspect_pipeline_model(tmp_path)
    except PipelineModelDescriptorError as exc:
        assert "缺少声明层" in str(exc)
    else:
        raise AssertionError("descriptor accepted an incomplete layer set")


def test_real_qwen3_artifact_is_described_but_not_admitted():
    descriptor = inspect_pipeline_model("models/qwen3-4b", model_id="qwen3-4b")
    assert descriptor["model_type"] == "qwen3"
    assert descriptor["total_layers"] == 36
    assert descriptor["pipeline_runtime_supported"] is False
    assert "adapter" in descriptor["runtime_block_reason"]


def test_gemma4_unified_uses_nested_text_layout_but_requires_sidecar(tmp_path):
    _write_gemma4_fixture(tmp_path)
    descriptor = inspect_pipeline_model(tmp_path, model_id="gemma4-fixture")

    assert descriptor["model_type"] == "gemma4_unified"
    assert descriptor["total_layers"] == 2
    assert descriptor["layer_prefix"] == "model.language_model.layers."
    assert descriptor["layer_weight_bytes"] == [4, 4]
    assert descriptor["component_weight_bytes"]["embedding"] == 16
    assert descriptor["component_weight_bytes"]["multimodal"] == 16
    assert descriptor["pipeline_runtime_supported"] is False
    assert "隔离 Transformers sidecar" in descriptor["runtime_block_reason"]
