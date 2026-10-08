"""`scripts/gguf_dequant_to_hf.py` 的纯函数回归（不读 GGUF 文件）。

命名映射与"是否转置"是反量化正确性的全部依据（GGUF `[in,out]` vs HF `[out,in]`），
映射错了会让权重静默错位 —— 所以必须被单测钉住。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gguf_dequant_to_hf import (  # noqa: E402
    _BLOCK_SUFFIX_MAP,
    hf_name_for,
    orient,
)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def test_top_level_mapping():
    assert hf_name_for("token_embd.weight") == "model.embed_tokens.weight"
    assert hf_name_for("output_norm.weight") == "model.norm.weight"
    assert hf_name_for("output.weight") == "lm_head.weight"


def test_block_mapping_uses_layer_index():
    assert hf_name_for("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
    assert hf_name_for("blk.23.ffn_down.weight") == "model.layers.23.mlp.down_proj.weight"
    assert hf_name_for("blk.12.attn_k.bias") == "model.layers.12.self_attn.k_proj.bias"


def test_unknown_names_return_none():
    # fail-loud 的前提：未知名必须返回 None，绝不猜测
    assert hf_name_for("blk.0.unknown_thing.weight") is None
    assert hf_name_for("some_random_tensor") is None
    assert hf_name_for("blk.bad") is None


def test_every_block_suffix_in_table_is_mapped():
    # 表内每一项都必须能映射回 HF（防手误写错键名）
    for suffix in _BLOCK_SUFFIX_MAP:
        assert hf_name_for(f"blk.7.{suffix}") == f"model.layers.7.{_BLOCK_SUFFIX_MAP[suffix]}"


def test_orient_keeps_already_hf_layout():
    # 实测：dequantize 已返回 HF 排布（GGUF 声明 [in,out]，数据是 [out,in]）
    arr = np.zeros((4864, 896), dtype=np.float32)
    out = orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")
    assert out.shape == (4864, 896)


def test_orient_transposes_when_given_declared_layout():
    # 若某个 gguf 版本返回"声明排布"，同样要摆正（而不是静默按原样存）
    arr = np.zeros((896, 4864), dtype=np.float32)
    out = orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")
    assert out.shape == (4864, 896)


def test_orient_passes_1d_through():
    v = np.ones(896, dtype=np.float32)
    assert orient(v, gguf_shape=[896], hf_name="model.layers.0.input_layernorm.weight").shape == (896,)


def test_orient_fails_loud_on_unknown_shape():
    arr = np.zeros((3, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="反量化形状"):
        orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")
