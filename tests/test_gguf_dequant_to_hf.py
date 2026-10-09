"""`scripts/gguf_dequant_to_hf.py` 的纯函数回归（不读 GGUF 文件）。

命名映射与"形状怎么摆"是反量化正确性的**全部依据** —— 映射错了会让权重静默错位
（本项目已验证过两次：写死"一律转置"错位 121 个张量；Qwen2 的后缀表套到 Qwen3.5 上
会漏掉 205 个张量）。所以三类东西都必须被单测钉住：
  1. 命名映射（含 **架构隔离**：Qwen2 与 Qwen3.5 的后缀表不能互串）；
  2. MTP 层（`blk.{block_count-1}`）的双前缀落位（`mtp.*` 与 `mtp.layers.0.*`）；
  3. `orient()` 的排布判定与 `conv1d` 补维。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gguf_dequant_to_hf import (  # noqa: E402
    _ARCH,
    _BLOCK_QWEN2,
    _BLOCK_QWEN35,
    apply_hf_inverse,
    arch_supported,
    hf_name_for,
    orient,
)


# ---------------------------------------------------------------- Qwen2 系
def test_qwen2_top_level_mapping():
    assert hf_name_for("token_embd.weight") == "model.embed_tokens.weight"
    assert hf_name_for("output_norm.weight") == "model.norm.weight"
    assert hf_name_for("output.weight") == "lm_head.weight"


def test_qwen2_block_mapping_uses_layer_index():
    assert hf_name_for("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
    assert hf_name_for("blk.23.ffn_down.weight") == "model.layers.23.mlp.down_proj.weight"
    assert hf_name_for("blk.12.attn_k.bias") == "model.layers.12.self_attn.k_proj.bias"


def test_unknown_names_return_none():
    # fail-loud 的前提：未知名必须返回 None，绝不猜测
    assert hf_name_for("blk.0.unknown_thing.weight") is None
    assert hf_name_for("some_random_tensor") is None
    assert hf_name_for("blk.bad") is None


def test_every_qwen2_suffix_in_table_is_mapped():
    for suffix in _BLOCK_QWEN2:
        assert hf_name_for(f"blk.7.{suffix}") == f"model.layers.7.{_BLOCK_QWEN2[suffix]}"


# ---------------------------------------------------------------- Qwen3.5（hybrid）
def test_qwen35_top_level_is_under_language_model():
    assert hf_name_for("token_embd.weight", arch="qwen35") == \
        "model.language_model.embed_tokens.weight"
    assert hf_name_for("output_norm.weight", arch="qwen35") == \
        "model.language_model.norm.weight"


def test_qwen35_full_attention_layer():
    assert hf_name_for("blk.3.attn_q.weight", arch="qwen35") == \
        "model.language_model.layers.3.self_attn.q_proj.weight"
    assert hf_name_for("blk.3.attn_q_norm.weight", arch="qwen35") == \
        "model.language_model.layers.3.self_attn.q_norm.weight"
    # ★ Qwen3.5 的 MLP 前 norm 叫 post_attention_norm（Qwen2 叫 ffn_norm）
    assert hf_name_for("blk.3.post_attention_norm.weight", arch="qwen35") == \
        "model.language_model.layers.3.post_attention_layernorm.weight"


def test_qwen35_linear_attention_layer():
    base = "model.language_model.layers.0.linear_attn."
    assert hf_name_for("blk.0.attn_qkv.weight", arch="qwen35") == base + "in_proj_qkv.weight"
    assert hf_name_for("blk.0.attn_gate.weight", arch="qwen35") == base + "in_proj_z.weight"
    assert hf_name_for("blk.0.ssm_alpha.weight", arch="qwen35") == base + "in_proj_a.weight"
    assert hf_name_for("blk.0.ssm_beta.weight", arch="qwen35") == base + "in_proj_b.weight"
    assert hf_name_for("blk.0.ssm_conv1d.weight", arch="qwen35") == base + "conv1d.weight"
    assert hf_name_for("blk.0.ssm_dt.bias", arch="qwen35") == base + "dt_bias"
    assert hf_name_for("blk.0.ssm_a", arch="qwen35") == base + "A_log"
    assert hf_name_for("blk.0.ssm_norm.weight", arch="qwen35") == base + "norm.weight"
    assert hf_name_for("blk.0.ssm_out.weight", arch="qwen35") == base + "out_proj.weight"


def test_arch_isolation_both_ways():
    """两张后缀表不能互串 —— 串了就是静默错位。"""
    # Qwen2 的 ffn_norm 在 Qwen3.5 上不存在
    assert hf_name_for("blk.0.ffn_norm.weight", arch="qwen35") is None
    # Qwen3.5 的 hybrid 后缀在 Qwen2 上不存在
    for name in ("blk.0.ssm_out.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                 "blk.0.post_attention_norm.weight", "blk.0.attn_q_norm.weight"):
        assert hf_name_for(name, arch="qwen2") is None


def test_every_qwen35_suffix_in_table_is_mapped():
    for suffix in _BLOCK_QWEN35:
        got = hf_name_for(f"blk.7.{suffix}", arch="qwen35")
        assert got == f"model.language_model.layers.7.{_BLOCK_QWEN35[suffix]}"


def test_unsupported_arch_returns_none():
    assert hf_name_for("blk.0.attn_norm.weight", arch="llama") is None
    assert arch_supported("qwen2") and arch_supported("qwen35")
    assert not arch_supported("llama")
    assert "qwen35" in _ARCH


# ---------------------------------------------------------------- MTP 层（双前缀）
def test_mtp_layer_splits_between_mtp_and_mtp_layers():
    # blk.24 = block_count(25) - nextn(1) ⇒ MTP 层
    assert hf_name_for("blk.24.nextn.eh_proj.weight", arch="qwen35", mtp_index=24) == \
        "mtp.fc.weight"
    assert hf_name_for("blk.24.nextn.shared_head_norm.weight", arch="qwen35", mtp_index=24) == \
        "mtp.norm.weight"
    assert hf_name_for("blk.24.attn_q.weight", arch="qwen35", mtp_index=24) == \
        "mtp.layers.0.self_attn.q_proj.weight"
    assert hf_name_for("blk.24.ffn_down.weight", arch="qwen35", mtp_index=24) == \
        "mtp.layers.0.mlp.down_proj.weight"


def test_without_mtp_index_the_extra_layer_is_a_normal_layer():
    # 不给 mtp_index 时不应"猜"它是 MTP 层（宁可落到普通层的路径上）
    assert hf_name_for("blk.24.attn_q.weight", arch="qwen35") == \
        "model.language_model.layers.24.self_attn.q_proj.weight"


# ---------------------------------------------------------------- orient()
def test_orient_keeps_already_hf_layout():
    arr = np.zeros((4864, 896), dtype=np.float32)
    out = orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")
    assert out.shape == (4864, 896)


def test_orient_transposes_when_given_declared_layout():
    arr = np.zeros((896, 4864), dtype=np.float32)
    out = orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")
    assert out.shape == (4864, 896)


def test_orient_passes_1d_through():
    v = np.ones(896, dtype=np.float32)
    assert orient(v, gguf_shape=[896],
                  hf_name="model.layers.0.input_layernorm.weight").shape == (896,)


def test_orient_fails_loud_on_unknown_shape():
    arr = np.zeros((3, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="反量化形状"):
        orient(arr, gguf_shape=[896, 4864], hf_name="model.layers.0.mlp.gate_proj.weight")


def test_orient_conv1d_adds_channel_axis_without_transposing():
    # GGUF 声明 [kernel=4, channels=6144]，反量化给 [channels, kernel]；HF 要 [channels, 1, kernel]
    arr = np.arange(6144 * 4, dtype=np.float32).reshape(6144, 4)
    out = orient(arr, gguf_shape=[4, 6144],
                 hf_name="model.language_model.layers.0.linear_attn.conv1d.weight")
    assert out.shape == (6144, 1, 4)
    # 不转置：第 0 通道的 kernel 值原样保留
    assert np.array_equal(out[0, 0], arr[0])


def test_orient_conv1d_fails_loud_on_bad_shape():
    with pytest.raises(ValueError, match="conv1d"):
        orient(np.zeros((7, 5), dtype=np.float32), gguf_shape=[4, 6144],
               hf_name="model.language_model.layers.0.linear_attn.conv1d.weight")


# ---------------------------------------------------------------- apply_hf_inverse()
def test_inverse_is_noop_for_qwen2():
    v = np.array([0.5, -1.5], dtype=np.float32)
    out = apply_hf_inverse(v, hf_name="model.layers.0.input_layernorm.weight", arch="qwen2")
    assert np.array_equal(out, v)


def test_inverse_subtracts_one_for_qwen35_rmsnorm():
    # GGUF 存的是 1+weight；HF 的 weight 是偏移量 ⇒ 减 1
    v = np.array([1.5, 0.25], dtype=np.float32)
    for name in ("model.language_model.layers.0.input_layernorm.weight",
                 "model.language_model.layers.3.post_attention_layernorm.weight",
                 "model.language_model.layers.3.self_attn.q_norm.weight",
                 "model.language_model.layers.3.self_attn.k_norm.weight",
                 "model.language_model.norm.weight",
                 "mtp.norm.weight",
                 "mtp.pre_fc_norm_embedding.weight",
                 "mtp.pre_fc_norm_hidden.weight"):
        out = apply_hf_inverse(v, hf_name=name, arch="qwen35")
        assert np.allclose(out, v - 1.0), name


def test_inverse_does_not_touch_linear_attn_gated_norm():
    # `linear_attn.norm.weight` 是 Qwen3_5RMSNormGated ⇒ 标准 weight，**不减**
    v = np.array([0.9, 1.1], dtype=np.float32)
    out = apply_hf_inverse(v, hf_name="model.language_model.layers.0.linear_attn.norm.weight",
                           arch="qwen35")
    assert np.array_equal(out, v)


def test_inverse_logs_ssm_a():
    # GGUF 的 ssm_a = -exp(A_log) ⇒ 还原为 log(-x)
    a_log = np.array([-0.5, -2.0, -3.25], dtype=np.float32)
    ssm_a = (-np.exp(a_log)).astype(np.float32)
    out = apply_hf_inverse(ssm_a, hf_name="model.language_model.layers.0.linear_attn.A_log",
                           arch="qwen35")
    assert np.allclose(out, a_log, atol=1e-6)


def test_inverse_leaves_ordinary_weights_alone():
    v = np.arange(6, dtype=np.float32)
    for name in ("model.language_model.layers.0.mlp.down_proj.weight",
                 "model.language_model.layers.0.linear_attn.dt_bias",
                 "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"):
        assert np.array_equal(apply_hf_inverse(v, hf_name=name, arch="qwen35"), v)
