import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import model_config as mc


def test_deepseek_r1_distill_qwen_slots_are_registered():
    expected = {
        "deepseek-r1-distill-qwen-1.5b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "models/deepseek-r1-distill-qwen-1.5b",
            4.0,
        ),
        "deepseek-r1-distill-qwen-7b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "models/deepseek-r1-distill-qwen-7b",
            8.0,
        ),
        "deepseek-r1-distill-qwen-14b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "models/deepseek-r1-distill-qwen-14b",
            16.0,
        ),
        "deepseek-r1-distill-qwen-32b": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "models/deepseek-r1-distill-qwen-32b",
            24.0,
        ),
    }

    for model_id, (hf_id, local_suffix, min_vram) in expected.items():
        cfg = mc.get_builtin_model(model_id)

        assert cfg is not None
        assert cfg.name.startswith("DeepSeek-R1-Distill-Qwen")
        assert cfg.model_type == "safetensors"
        assert cfg.is_experimental is True
        assert cfg.location == "external"
        assert cfg.huggingface_id == hf_id
        assert os.path.normpath(cfg.model_path).endswith(os.path.normpath(local_suffix))
        assert cfg.gguf_path == ""
        assert cfg.recommended_vram_gb == min_vram
        assert cfg.max_context == 32768
        assert cfg.quant_types == ["fp16", "int8", "int4"]


def test_experimental_models_are_hidden_without_cuda(monkeypatch):
    monkeypatch.setattr(mc, "is_cuda_available", lambda: False)

    visible_ids = {model.model_id for model in mc.get_visible_models()}

    assert mc.DEFAULT_MODEL_ID in visible_ids
    assert "deepseek-r1-distill-qwen-1.5b" not in visible_ids
    assert "deepseek-r1-distill-qwen-7b" not in visible_ids
    assert "deepseek-r1-distill-qwen-14b" not in visible_ids
    assert "deepseek-r1-distill-qwen-32b" not in visible_ids


def test_deepseek_r1_distill_slots_are_visible_with_cuda(monkeypatch):
    monkeypatch.setattr(mc, "is_cuda_available", lambda: True)

    visible_ids = {model.model_id for model in mc.get_visible_models()}

    assert "deepseek-r1-distill-qwen-1.5b" in visible_ids
    assert "deepseek-r1-distill-qwen-7b" in visible_ids
    assert "deepseek-r1-distill-qwen-14b" in visible_ids
    assert "deepseek-r1-distill-qwen-32b" in visible_ids
