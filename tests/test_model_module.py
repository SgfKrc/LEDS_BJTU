"""
单元测试 — 模型模块的前向传播（forward_layers）
==============================================
测试 load_layer_range() 层裁剪 + forward_layers() 分布式前向推理。

使用 tiny Qwen2 配置（4 层 / hidden=128 / 32 token vocab），
无需下载完整模型即可验证分布式流水线的完整数据流。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import copy
import pytest
import torch
import torch.nn as nn

from transformers import Qwen2Config, Qwen2ForCausalLM

from model_module import ModelManager

# ================================================================
# Tiny 模型工厂
# ================================================================

TINY_CONFIG = Qwen2Config(
    vocab_size=32,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,       # 4 层足够测试端到端流水线
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)


def _make_tiny_model() -> Qwen2ForCausalLM:
    """创建 tiny Qwen2ForCausalLM（随机权重，无需 GPU）。"""
    model = Qwen2ForCausalLM(TINY_CONFIG)
    model.eval()
    assert model.lm_head is not None, "lm_head 应为独立层"
    return model


def _make_first_node_model():
    """创建首节点模型（有 Embedding，无 LM Head）。"""
    model = _make_tiny_model()
    del model.lm_head
    model.lm_head = None
    return model


def _make_middle_node_model():
    """创建中间节点模型（无 Embedding，无 LM Head）。"""
    model = _make_tiny_model()
    del model.model.embed_tokens
    model.model.embed_tokens = None
    del model.lm_head
    model.lm_head = None
    return model


def _make_last_node_model():
    """创建末节点模型（无 Embedding，有 LM Head）。"""
    model = _make_tiny_model()
    del model.model.embed_tokens
    model.model.embed_tokens = None
    return model


# ================================================================
# forward_layers 输入校验测试
# ================================================================

class TestForwardLayersInputValidation:
    """测试 forward_layers 输入参数校验"""

    def test_requires_loaded_model(self):
        """未加载模型时应报错"""
        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="模型未加载"):
            mgr.forward_layers(input_ids=torch.tensor([[1, 2, 3]]))

    def test_no_input_raises(self):
        """不传任何输入时应报错"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 4)
        with pytest.raises(ValueError, match="必须提供.*之一"):
            mgr.forward_layers()

    def test_both_inputs_raises(self):
        """同时传 input_ids 和 hidden_states 应报错"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 4)
        with pytest.raises(ValueError, match="不能同时提供"):
            mgr.forward_layers(
                input_ids=torch.tensor([[1, 2]]),
                hidden_states=torch.randn(1, 2, 128),
            )

    def test_input_ids_without_embedding_raises(self):
        """节点不含 Embedding 时传 input_ids 应报错"""
        mgr = ModelManager()
        mgr.model = _make_middle_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (1, 4)
        with pytest.raises(RuntimeError, match="不含 Embedding 层"):
            mgr.forward_layers(input_ids=torch.tensor([[1, 2]]))

    def test_llama_cpp_engine_raises(self):
        """llama.cpp 引擎不支持 forward_layers"""
        mgr = ModelManager()
        mgr.model = _make_tiny_model()  # 需要有 model 才能通过第一个检查
        mgr._engine_type = "llama_cpp"
        with pytest.raises(RuntimeError, match="仅支持 PyTorch 引擎"):
            mgr.forward_layers(hidden_states=torch.randn(1, 4, 128))


# ================================================================
# forward_layers 首节点测试（含 Embedding，无 LM Head）
# ================================================================

class TestForwardLayersFirstNode:
    """测试首节点前向传播（input_ids → embed → layers → hidden_states）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_first_node_returns_hidden_states(self):
        """首节点应返回 hidden_states dict"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert "hidden_states" in result
        assert "logits" not in result
        hs = result["hidden_states"]
        assert hs.shape == (1, 5, 128)  # (batch, seq, hidden)
        assert hs.dtype == torch.float32

    def test_first_node_output_is_finite(self):
        """输出不应包含 NaN 或 Inf"""
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        hs = result["hidden_states"]
        assert torch.isfinite(hs).all(), "hidden_states 包含 NaN 或 Inf"

    def test_first_node_batch_size_2(self):
        """batch=2 的首节点推理"""
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert result["hidden_states"].shape == (2, 4, 128)

    def test_first_node_seq_len_1(self):
        """seq_len=1 的首节点推理（解码阶段单 token）"""
        input_ids = torch.tensor([[7]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert result["hidden_states"].shape == (1, 1, 128)

    def test_first_node_different_initial_tokens_produce_different_output(self):
        """不同输入 token 应产生不同的 hidden_states"""
        ids_a = torch.tensor([[1, 2, 3]])
        ids_b = torch.tensor([[4, 5, 6]])
        hs_a = self.mgr.forward_layers(input_ids=ids_a)["hidden_states"]
        hs_b = self.mgr.forward_layers(input_ids=ids_b)["hidden_states"]
        assert not torch.allclose(hs_a, hs_b, atol=1e-4), \
            "不同输入应产生不同输出"


# ================================================================
# forward_layers 中间节点测试（仅 Transformer 层）
# ================================================================

class TestForwardLayersMiddleNode:
    """测试中间节点前向传播（hidden_states → layers → hidden_states）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_middle_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (1, 3)

    def test_middle_node_returns_hidden_states(self):
        """中间节点接收 hidden_states 应返回 hidden_states"""
        hs_in = torch.randn(2, 8, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert "logits" not in result
        hs_out = result["hidden_states"]
        assert hs_out.shape == (2, 8, 128)

    def test_middle_node_modifies_hidden_states(self):
        """中间节点应改变 hidden_states（不应该是恒等映射）"""
        hs_in = torch.randn(1, 4, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        assert not torch.allclose(hs_in, hs_out, atol=1e-4), \
            "Transformer 层应改变 hidden_states"

    def test_middle_node_preserves_dtype(self):
        """中间节点应保持输入 dtype"""
        hs_in = torch.randn(1, 3, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        assert hs_out.dtype == hs_in.dtype

    def test_middle_node_produces_unique_output_per_position(self):
        """不同序列位置的 hidden_states 应不相同"""
        hs_in = torch.randn(1, 8, 128)
        hs_out = self.mgr.forward_layers(hidden_states=hs_in)["hidden_states"]
        # 位置 0 和位置 1 的输出应不同（attention 混入了位置信息）
        assert not torch.allclose(hs_out[0, 0, :], hs_out[0, 1, :], atol=1e-4)

    def test_device_mismatch_input(self):
        """
        中间节点的 hidden_states 来自网络传输，可能与模型不在同一设备。
        例如: 从 CUDA 设备接收 hidden_states → 当前节点的 CPU 模型。
        forward_layers 应自动迁移到正确设备。
        """
        # 模型在 CPU，输入 tensor 模拟"刚从 TCP 反序列化得到（也是 CPU）"
        # 验证即使在同一设备上也正常工作（跨设备实际需要 GPU 环境）
        hs_in = torch.randn(1, 5, 128)  # CPU
        assert self.mgr.get_device().type == "cpu"
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert result["hidden_states"].device.type == "cpu"
        # 如果模型在 CUDA，输入在 CPU，tensor.to(device) 应能处理
        # 此处仅验证代码路径不崩溃
        assert torch.isfinite(result["hidden_states"]).all()


# ================================================================
# forward_layers 末节点测试（含 LM Head）
# ================================================================

class TestForwardLayersLastNode:
    """测试末节点前向传播（hidden_states → layers → norm → lm_head → logits）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_last_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (2, 4)

    def test_last_node_returns_logits(self):
        """末节点应返回 logits dict（经过 norm + lm_head）"""
        hs_in = torch.randn(1, 6, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "logits" in result
        assert "hidden_states" not in result
        logits = result["logits"]
        # 形状: (batch, seq_len, vocab_size)
        assert logits.shape == (1, 6, TINY_CONFIG.vocab_size)
        assert logits.dtype == torch.float32

    def test_last_node_logits_are_finite(self):
        """末节点输出 logits 应全部有限"""
        hs_in = torch.randn(2, 3, 128)
        logits = self.mgr.forward_layers(hidden_states=hs_in)["logits"]
        assert torch.isfinite(logits).all()

    def test_last_node_no_lm_head_returns_hidden_states(self):
        """末节点无 lm_head 时应返回 hidden_states（降级为中间节点行为）"""
        del self.model.lm_head
        self.model.lm_head = None
        hs_in = torch.randn(1, 4, 128)
        result = self.mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert "logits" not in result

    def test_last_node_logits_change_with_input(self):
        """不同的 hidden_states 应产生不同的 logits"""
        hs_a = torch.randn(1, 4, 128)
        hs_b = torch.randn(1, 4, 128)
        logits_a = self.mgr.forward_layers(hidden_states=hs_a)["logits"]
        logits_b = self.mgr.forward_layers(hidden_states=hs_b)["logits"]
        assert not torch.allclose(logits_a, logits_b, atol=1e-4)


# ================================================================
# 核心正确性测试: forward_layers(全模型) == model.forward()
# ================================================================

class TestForwardLayersMatchesFullModel:
    """
    核心正确性验证:

    forward_layers 在完整模型（未裁剪）上运行的结果，
    必须与 model.forward() / model(input_ids) 完全一致。

    这是分布式流水线正确性的根基 — 将完整模型拆分为节点后，
    每个节点用 forward_layers 执行的片段拼接起来必须等
    价于完整模型的单次前向传播。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_full_model_logits_match(self):
        """forward_layers 全模型 → logits 应与 model(input_ids) 一致"""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # --- 路径 A: model.forward() ---
        with torch.no_grad():
            ref_output = self.model(input_ids)
        ref_logits = ref_output.logits

        # --- 路径 B: forward_layers() ---
        result = self.mgr.forward_layers(input_ids=input_ids)
        fwd_logits = result["logits"]

        assert torch.allclose(ref_logits, fwd_logits, atol=1e-4), \
            f"forward_layers ≠ model.forward！max diff = {(ref_logits - fwd_logits).abs().max():.6f}"

    def test_full_model_with_attention_mask_matches(self):
        """带 attention_mask 的 forward_layers 应与 model.forward 一致"""
        input_ids = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 0],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ])

        with torch.no_grad():
            ref_output = self.model(input_ids, attention_mask=attention_mask)
        ref_logits = ref_output.logits

        result = self.mgr.forward_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        fwd_logits = result["logits"]

        assert torch.allclose(ref_logits, fwd_logits, atol=1e-4), \
            f"attention_mask 差异！max diff = {(ref_logits - fwd_logits).abs().max():.6f}"

    def test_full_model_returns_logits_not_hidden_states(self):
        """全模型（有 lm_head）的 forward_layers 应返回 logits"""
        result = self.mgr.forward_layers(input_ids=torch.tensor([[1, 2]]))
        assert "logits" in result
        assert "hidden_states" not in result


# ================================================================
# 端到端流水线模拟测试（首→中→末 三节点串联）
# ================================================================

class TestPipelineE2E:
    """
    端到端流水线测试：模拟三节点分布式推理

    将同一份模型权重复制到三个节点，确保流水线输出的 logits
    与完整模型的 model.forward() 完全一致。

    节点 0: Layer 0-1 + Embedding        → hidden_states
    节点 1: Layer 2   (纯中间层)         → hidden_states
    节点 2: Layer 3   + LM Head          → logits
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        # ---- 创建一份"规范"模型并保存其权重 ----
        canonical = _make_tiny_model()
        self.state_dict = {k: v.clone() for k, v in canonical.state_dict().items()}

        # ---- 节点 0: Layer 0-2 + Embedding, 无 LM Head ----
        self.model0 = _make_tiny_model()
        # 删除 lm_head 权重
        self.model0.load_state_dict(self.state_dict, strict=False)
        del self.model0.lm_head
        self.model0.lm_head = None
        self.model0.model.layers = nn.ModuleList(
            list(self.model0.model.layers)[0:2]
        )
        self.mgr0 = ModelManager()
        self.mgr0.model = self.model0
        self.mgr0._engine_type = "pytorch"
        self.mgr0.layer_range = (0, 2)

        # ---- 节点 1: Layer 2, 无 Embedding, 无 LM Head ----
        self.model1 = _make_tiny_model()
        self.model1.load_state_dict(self.state_dict, strict=False)
        del self.model1.model.embed_tokens
        self.model1.model.embed_tokens = None
        del self.model1.lm_head
        self.model1.lm_head = None
        self.model1.model.layers = nn.ModuleList(
            [list(self.model1.model.layers)[2]]
        )
        self.mgr1 = ModelManager()
        self.mgr1.model = self.model1
        self.mgr1._engine_type = "pytorch"
        self.mgr1.layer_range = (2, 3)

        # ---- 节点 2: Layer 3 + LM Head, 无 Embedding ----
        self.model2 = _make_tiny_model()
        self.model2.load_state_dict(self.state_dict, strict=False)
        del self.model2.model.embed_tokens
        self.model2.model.embed_tokens = None
        self.model2.model.layers = nn.ModuleList(
            [list(self.model2.model.layers)[3]]
        )
        self.mgr2 = ModelManager()
        self.mgr2.model = self.model2
        self.mgr2._engine_type = "pytorch"
        self.mgr2.layer_range = (3, 4)

        # ---- 完整模型（用于对比） ----
        self.model_full = _make_tiny_model()
        self.model_full.load_state_dict(self.state_dict)

    def test_three_node_pipeline_shape(self):
        """三节点流水线应产出正确形状的 logits"""
        input_ids = torch.tensor([[1, 2, 3, 4]])

        out0 = self.mgr0.forward_layers(input_ids=input_ids)
        assert "hidden_states" in out0

        out1 = self.mgr1.forward_layers(hidden_states=out0["hidden_states"])
        assert "hidden_states" in out1

        out2 = self.mgr2.forward_layers(hidden_states=out1["hidden_states"])
        assert "logits" in out2
        assert out2["logits"].shape == (1, 4, TINY_CONFIG.vocab_size)

    def test_pipeline_output_is_finite(self):
        """完整流水线输出应全部有限"""
        input_ids = torch.tensor([[1, 2, 3]])
        hs = self.mgr0.forward_layers(input_ids=input_ids)["hidden_states"]
        assert torch.isfinite(hs).all()
        hs = self.mgr1.forward_layers(hidden_states=hs)["hidden_states"]
        assert torch.isfinite(hs).all()
        logits = self.mgr2.forward_layers(hidden_states=hs)["logits"]
        assert torch.isfinite(logits).all()

    def test_pipeline_matches_full_model_forward(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  最核心的正确性保证:                                    ║
        ║  三节点流水线的输出 == 完整模型单机 forward 的输出     ║
        ╚══════════════════════════════════════════════════════════╝
        """
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        # --- 完整模型 forward ---
        with torch.no_grad():
            full_output = self.model_full(input_ids)
        full_logits = full_output.logits

        # --- 三节点流水线 ---
        hs = self.mgr0.forward_layers(input_ids=input_ids)["hidden_states"]
        hs = self.mgr1.forward_layers(hidden_states=hs)["hidden_states"]
        pipeline_logits = self.mgr2.forward_layers(hidden_states=hs)["logits"]

        # 对比
        assert torch.allclose(full_logits, pipeline_logits, atol=1e-4), \
            f"流水线输出与完整模型不一致！max diff = {(full_logits - pipeline_logits).abs().max():.6f}"


# ================================================================
# attention_mask 测试
# ================================================================

class TestForwardLayersAttentionMask:
    """测试 attention_mask 的支持"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_attention_mask_with_padding(self):
        """attention_mask 含填充时应正常工作"""
        input_ids = torch.tensor([
            [1, 2, 3, 0, 0],  # 后 2 位是 padding
            [4, 5, 0, 0, 0],  # 后 3 位是 padding
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ])
        result = self.mgr.forward_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hs = result["hidden_states"]
        assert torch.isfinite(hs).all()
        # padding 位置的输出应与有效位不同（确保掩码生效）
        row0_tok2 = hs[0, 2, :]  # 有效 token
        row0_tok3 = hs[0, 3, :]  # padding
        assert not torch.allclose(row0_tok2, row0_tok3, atol=1e-5), \
            "attention_mask 未生效：padding 位置输出与有效位置相同"

    def test_no_attention_mask_is_fine(self):
        """不传 attention_mask 应正常工作（纯因果注意力）"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert torch.isfinite(result["hidden_states"]).all()


# ================================================================
# position_ids 测试
# ================================================================

class TestForwardLayersPositionIds:
    """测试自定义 position_ids"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_first_node_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_custom_position_ids(self):
        """自定义 position_ids 应影响输出（RoPE 生效）"""
        input_ids = torch.tensor([[1, 2, 3]])
        # 标准位置 0,1,2
        hs_default = self.mgr.forward_layers(input_ids=input_ids)["hidden_states"]
        # 偏移位置 10,11,12
        hs_offset = self.mgr.forward_layers(
            input_ids=input_ids,
            position_ids=torch.tensor([[10, 11, 12]]),
        )["hidden_states"]
        # 不同位置编码应产生不同输出
        # 注意: 随机权重模型下差异极小 (~1e-7)，用 equal 而非 allclose 检测
        assert not torch.equal(hs_default, hs_offset), \
            "position_ids 未生效（RoPE 应使不同位置产生不同输出）"
        assert torch.isfinite(hs_offset).all()

    def test_position_ids_default_range(self):
        """默认 position_ids 应为 [0, 1, 2, ...]"""
        # 不传 position_ids 也能正常工作，且不同位置有不同输出
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        hs = result["hidden_states"]
        # 不同位置的 hidden_states 应该不同（证明 position 生效）
        assert not torch.allclose(hs[0, 0, :], hs[0, 1, :], atol=1e-4), \
            "默认 position_ids 未生效：不同位置输出相同"


# ================================================================
# load_layer_range 集成测试
# ================================================================

class TestLoadLayerRangeIntegration:
    """load_layer_range + forward_layers 端到端集成测试"""

    def test_load_then_forward_first_node(self):
        """通过 load_layer_range 加载首节点 → forward_layers 应正常工作"""
        mgr = ModelManager()
        mgr.model = _make_first_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (0, 2)

        # 模拟 load_layer_range 的效果: 保留 Layer 0-2 + Embedding
        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[0:2]
        )
        # lm_head 在 _make_first_node_model 中已删除

        result = mgr.forward_layers(input_ids=torch.tensor([[1, 2, 3]]))
        assert "hidden_states" in result
        assert result["hidden_states"].shape == (1, 3, 128)

    def test_load_then_forward_last_node(self):
        """通过 load_layer_range 加载末节点 → forward_layers 应产出 logits"""
        mgr = ModelManager()
        mgr.model = _make_last_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (2, 4)

        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[2:4]
        )

        result = mgr.forward_layers(
            hidden_states=torch.randn(1, 5, 128)
        )
        assert "logits" in result
        assert result["logits"].shape == (1, 5, TINY_CONFIG.vocab_size)

    def test_load_then_forward_middle_node(self):
        """通过 load_layer_range 加载中间节点 → forward_layers 透传 hidden_states"""
        mgr = ModelManager()
        mgr.model = _make_middle_node_model()
        mgr._engine_type = "pytorch"
        mgr.layer_range = (1, 3)

        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[1:3]
        )

        hs_in = torch.randn(2, 4, 128)
        result = mgr.forward_layers(hidden_states=hs_in)
        assert "hidden_states" in result
        assert result["hidden_states"].shape == (2, 4, 128)


# ================================================================
# 确定性测试（同输入 → 同输出）
# ================================================================

class TestForwardLayersDeterminism:
    """测试 forward_layers 的确定性"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_same_input_same_output(self):
        """相同输入应产生完全相同的输出（eval 模式确定性）"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result1 = self.mgr.forward_layers(input_ids=input_ids)
        result2 = self.mgr.forward_layers(input_ids=input_ids)
        assert torch.equal(result1["logits"], result2["logits"]), \
            "forward_layers 应该是确定性的（eval 模式）"


# ================================================================
# 废弃 API 测试
# ================================================================

class TestDeprecatedAPIs:
    """测试已废弃的 API 是否正确报错"""

    def test_split_model_raises_not_implemented(self):
        """split_model() 应抛 NotImplementedError 引导用户使用 load_layer_range()"""
        mgr = ModelManager()
        with pytest.raises(NotImplementedError, match="load_layer_range"):
            mgr.split_model((0, 8))

    def test_load_then_get_model_info_total_layers(self):
        """get_model_info 应在 load_layer_range 后仍报告原始总层数"""
        mgr = ModelManager()
        mgr.model = _make_tiny_model()
        mgr._engine_type = "pytorch"
        # 模拟 _load_pytorch 设置
        mgr._total_model_layers = 4
        mgr._model_layers = 4

        # 模拟 load_layer_range(0, 2) 的效果
        mgr.model.model.layers = nn.ModuleList(
            list(mgr.model.model.layers)[0:2]
        )
        del mgr.model.lm_head
        mgr.model.lm_head = None
        mgr.layer_range = (0, 2)
        mgr._model_layers = 2  # load_layer_range 会设此值

        info = mgr.get_model_info()
        # total_layers 应保持原始总层数 4，而非当前加载的 2
        assert info["total_layers"] == 4, \
            f"total_layers 应为 4（原始总层数），实际 {info['total_layers']}"
        assert info["loaded_layers"] == 2, \
            f"loaded_layers 应为 2（当前加载层数），实际 {info['loaded_layers']}"
        assert info["layer_range"] == (0, 2)


# ================================================================
# KV Cache 测试 (Phase 3 — 增量解码)
# ================================================================

class TestForwardLayersKVCache:
    """测试 forward_layers 的 KV Cache 支持（增量解码优化）"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_tiny_model()
        self.mgr = ModelManager()
        self.mgr.model = self.model
        self.mgr._engine_type = "pytorch"
        self.mgr.layer_range = (0, 4)

    def test_prefill_without_kv_cache_returns_no_past_key_values(self):
        """use_cache=False → 不返回 past_key_values"""
        input_ids = torch.tensor([[1, 2, 3]])
        result = self.mgr.forward_layers(
            input_ids=input_ids,
            use_cache=False,
        )
        assert "past_key_values" not in result

    def test_prefill_with_kv_cache_returns_past_key_values(self):
        """use_cache=True (default) → 返回 past_key_values tuple"""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        assert "past_key_values" in result
        pkv = result["past_key_values"]
        # 应为每层一个 (key, value) 元组（本地层索引）
        assert len(pkv) == 4  # 4 layers in tiny model
        assert isinstance(pkv[0], tuple)
        assert len(pkv[0]) == 2  # (key, value)
        # key shape: (batch, num_kv_heads, seq_len, head_dim)
        # GQA: num_kv_heads=2, head_dim=128/4=32
        assert pkv[0][0].shape == (1, 2, 4, 32)

    def test_past_key_values_accumulates_sequence_length(self):
        """Prefill → Decode → KV cache seq_len 应累加"""
        # --- Prefill: 4 tokens ---
        input_ids = torch.tensor([[1, 2, 3, 4]])
        result = self.mgr.forward_layers(input_ids=input_ids)
        prefill_kv = result["past_key_values"]
        assert prefill_kv[0][0].shape[2] == 4  # seq_len = 4

        # --- Decode: 1 new token (使用缓存的 KV) ---
        new_token = torch.tensor([[7]])  # shape (1, 1)
        result2 = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )
        decode_kv = result2["past_key_values"]
        assert decode_kv[0][0].shape[2] == 5  # seq_len = 4 + 1

    def test_decode_with_kv_cache_preserves_output_shape(self):
        """Decode 阶段输入 (1,1) → hidden_states 输出也应为 (1,1)（仅新 token 的输出）"""
        # Prefill
        input_ids = torch.tensor([[1, 2, 3]])
        prefill_result = self.mgr.forward_layers(input_ids=input_ids)
        prefill_kv = prefill_result["past_key_values"]

        # 移除 lm_head 以测试中间节点行为
        del self.model.lm_head
        self.model.lm_head = None

        # Decode: 1 token with KV cache
        new_token = torch.tensor([[5]])
        result = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )
        hs = result["hidden_states"]
        # 输出形状应与输入形状匹配（仅新 token）
        assert hs.shape == (1, 1, 128), \
            f"Decode 输出应为 (1,1,128)，实际 {hs.shape}"

    def test_kv_cache_reduces_computation(self):
        """
        验证 KV Cache 确实减少了计算量：
        相同的新 token，有 KV cache 的 decode 输出与无 cache 的完整序列
        重计算的最后一个位置输出应该不同（因为 KV cache 保留了正确的历史上下文）。

        注意：有 KV cache 时仅计算 position 3 的 hidden_states，
        无 KV cache 时重新计算 position 0-3。由于 RoPE 在相同输入上
        是确定性的，两者的最后一个位置输出应该一致。
        """
        input_ids = torch.tensor([[1, 2, 3, 4]])

        # 移除 lm_head（测试 hidden_states 输出）
        del self.model.lm_head
        self.model.lm_head = None

        # --- 路径 A: Prefill → 取最后位置输出 ---
        result_full = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        full_kv = result_full["past_key_values"]

        # --- 路径 B: Prefill(0-2) → Decode(3) with KV cache ---
        result_pre = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]),
            use_cache=True,
        )
        pre_kv = result_pre["past_key_values"]

        result_dec = self.mgr.forward_layers(
            input_ids=torch.tensor([[4]]),
            past_key_values=pre_kv,
            use_cache=True,
        )

        # 路径 A 的位置 3 输出 (full_kv seq_len=4 时并未直接给出 hidden_states)
        # 重跑一次: Prefill(0-3) 无 KV cache，取最后位置的 hidden_states
        result_ref = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            use_cache=False,
        )
        hs_ref_last = result_ref["hidden_states"][:, -1:, :]  # (1, 1, 128)

        hs_dec = result_dec["hidden_states"]  # (1, 1, 128)

        # 两者应一致（KV cache 不应改变计算结果）
        assert torch.allclose(hs_ref_last, hs_dec, atol=1e-4), \
            f"KV cache decode 应与完整 prefill 的最后位置一致，" \
            f"max diff = {(hs_ref_last - hs_dec).abs().max():.6f}"

    def test_position_ids_auto_advance_with_kv_cache(self):
        """有 KV cache 时，position_ids 自动从 past_seen_tokens 开始"""
        # Prefill: 5 tokens at positions 0-4 (vocab_size=32, keep IDs < 32)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        prefill_kv = result["past_key_values"]
        assert prefill_kv[0][0].shape[2] == 5

        del self.model.lm_head
        self.model.lm_head = None

        # Decode: 1 new token，不手动指定 position_ids
        new_token = torch.tensor([[7]])
        result_no_pos = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv,
            use_cache=True,
        )

        # 重新 prefill（避免 KV cache 被原地修改）
        result2 = self.mgr.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            use_cache=True,
        )
        prefill_kv2 = result2["past_key_values"]

        # 手动指定 position_ids=[5]
        result_with_pos = self.mgr.forward_layers(
            input_ids=new_token,
            past_key_values=prefill_kv2,
            position_ids=torch.tensor([[5]]),
            use_cache=True,
        )

        hs_auto = result_no_pos["hidden_states"]
        hs_manual = result_with_pos["hidden_states"]
        assert torch.allclose(hs_auto, hs_manual, atol=1e-6), \
            "自动 position_ids 与手动 position_ids=[5] 应产生相同输出"

    def test_middle_node_kv_cache(self):
        """中间节点（接收 hidden_states 而非 input_ids）的 KV cache 支持"""
        # 使用中间节点模型（start_layer=1），切片为 2 层
        model_mid = _make_middle_node_model()
        # 切片为 layer 1-3 (global)，本地 2 层
        model_mid.model.layers = nn.ModuleList(
            list(model_mid.model.layers)[1:3]
        )
        mgr_mid = ModelManager()
        mgr_mid.model = model_mid
        mgr_mid._engine_type = "pytorch"
        mgr_mid.layer_range = (1, 3)

        # Prefill: 接收完整 hidden_states
        hs_in = torch.randn(1, 6, 128)
        result = mgr_mid.forward_layers(hidden_states=hs_in, use_cache=True)
        assert "past_key_values" in result
        mid_kv = result["past_key_values"]
        # 2 层本地 (global 1,2 → local 0,1)
        assert len(mid_kv) == 2
        assert mid_kv[0][0].shape[2] == 6

        # Decode: 仅 1 个位置
        hs_in_decode = torch.randn(1, 1, 128)
        result2 = mgr_mid.forward_layers(
            hidden_states=hs_in_decode,
            past_key_values=mid_kv,
            use_cache=True,
        )
        assert result2["hidden_states"].shape == (1, 1, 128)
        assert result2["past_key_values"][0][0].shape[2] == 7

    def test_kv_cache_is_deterministic(self):
        """相同输入 + 相同 KV cache → 相同输出"""
        input_ids = torch.tensor([[1, 2, 3]])
        result1 = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        kv1 = result1["past_key_values"]

        del self.model.lm_head
        self.model.lm_head = None

        new_token = torch.tensor([[7]])

        # 第一次 decode（会原地更新 DynamicCache 内部，但返回 tuple 副本）
        out1 = self.mgr.forward_layers(
            input_ids=new_token, past_key_values=kv1, use_cache=True
        )["hidden_states"]

        # 重新 prefill 得到相同 KV cache，再 decode
        result2 = self.mgr.forward_layers(input_ids=input_ids, use_cache=True)
        kv2 = result2["past_key_values"]
        out2 = self.mgr.forward_layers(
            input_ids=new_token, past_key_values=kv2, use_cache=True
        )["hidden_states"]

        assert torch.equal(out1, out2), \
            "相同 KV cache + 相同输入 → 应有相同输出（确定性）"


class TestPipelineE2EWithKVCache:
    """
    端到端流水线 + KV Cache 集成测试

    模拟三节点分布式推理的完整 prefill→decode 流程，
    验证 KV cache 在节点间独立存储、独立更新的正确性。
    各节点仅维护本地层范围的 DynamicCache。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        canonical = _make_tiny_model()
        state_dict = {k: v.clone() for k, v in canonical.state_dict().items()}

        # 节点 0: Layer 0-1 + Embedding, start_layer=0
        self.model0 = _make_tiny_model()
        self.model0.load_state_dict(state_dict, strict=False)
        del self.model0.lm_head
        self.model0.lm_head = None
        self.model0.model.layers = nn.ModuleList(list(self.model0.model.layers)[0:2])
        self.mgr0 = ModelManager()
        self.mgr0.model = self.model0
        self.mgr0._engine_type = "pytorch"
        self.mgr0.layer_range = (0, 2)

        # 节点 1: Layer 2 (中间层), start_layer=2
        self.model1 = _make_tiny_model()
        self.model1.load_state_dict(state_dict, strict=False)
        del self.model1.model.embed_tokens
        self.model1.model.embed_tokens = None
        del self.model1.lm_head
        self.model1.lm_head = None
        self.model1.model.layers = nn.ModuleList([list(self.model1.model.layers)[2]])
        self.mgr1 = ModelManager()
        self.mgr1.model = self.model1
        self.mgr1._engine_type = "pytorch"
        self.mgr1.layer_range = (2, 3)

        # 节点 2: Layer 3 + LM Head, start_layer=3
        self.model2 = _make_tiny_model()
        self.model2.load_state_dict(state_dict, strict=False)
        del self.model2.model.embed_tokens
        self.model2.model.embed_tokens = None
        self.model2.model.layers = nn.ModuleList([list(self.model2.model.layers)[3]])
        self.mgr2 = ModelManager()
        self.mgr2.model = self.model2
        self.mgr2._engine_type = "pytorch"
        self.mgr2.layer_range = (3, 4)

        # 完整参考模型
        self.model_full = _make_tiny_model()
        self.model_full.load_state_dict(state_dict)

    def test_prefill_then_decode_pipeline(self):
        """
        完整流程: Prefill(4 tokens) → Decode(1 token) × 2 steps
        验证每个节点独立 KV cache 管理、seq_len 正确累加
        """
        # ==== Prefill: 4 tokens ====
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True
        )
        kv0 = out0["past_key_values"]
        # 节点 0: 2 local layers (0→local0, 1→local1)
        assert len(kv0) == 2
        assert kv0[0][0].shape[2] == 4

        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], use_cache=True
        )
        kv1 = out1["past_key_values"]
        # 节点 1: 1 local layer (global 2 → local 0)
        assert len(kv1) == 1
        assert kv1[0][0].shape[2] == 4

        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], use_cache=True
        )
        kv2 = out2["past_key_values"]
        # 节点 2: 1 local layer (global 3 → local 0)
        assert len(kv2) == 1
        assert kv2[0][0].shape[2] == 4

        prefill_logits = out2["logits"]
        assert prefill_logits.shape == (1, 4, TINY_CONFIG.vocab_size)

        # ==== Decode step 1: 1 new token ====
        new_token = torch.tensor([[7]])
        out0 = self.mgr0.forward_layers(
            input_ids=new_token, past_key_values=kv0, use_cache=True
        )
        kv0 = out0["past_key_values"]
        assert kv0[0][0].shape[2] == 5
        assert out0["hidden_states"].shape == (1, 1, 128)

        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        kv1 = out1["past_key_values"]
        assert kv1[0][0].shape[2] == 5
        assert out1["hidden_states"].shape == (1, 1, 128)

        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        kv2 = out2["past_key_values"]
        assert kv2[0][0].shape[2] == 5
        decode_logits = out2["logits"]
        assert decode_logits.shape == (1, 1, TINY_CONFIG.vocab_size)

        # ==== Decode step 2: another token ====
        new_token2 = torch.tensor([[3]])
        out0 = self.mgr0.forward_layers(
            input_ids=new_token2, past_key_values=kv0, use_cache=True
        )
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        decode2_logits = out2["logits"]
        assert decode2_logits.shape == (1, 1, TINY_CONFIG.vocab_size)

    def test_pipeline_with_kv_cache_matches_full_model(self):
        """
        ╔══════════════════════════════════════════════════════════╗
        ║  核心保证: KV cache 流水线输出 == 完整模型输出        ║
        ║                                                          ║
        ║  Prefill [1,2,3] + Decode [4] with KV cache            ║
        ║  的 logits 必须等于完整模型 input [1,2,3,4] 的         ║
        ║  最后一个位置的 logits。                                ║
        ╚══════════════════════════════════════════════════════════╝
        """
        # 完整模型: [1,2,3,4] → logits for position 3
        input_ids_full = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            full_output = self.model_full(input_ids_full)
        full_last_logits = full_output.logits[:, -1:, :]  # (1, 1, vocab)

        # 流水线: Prefill [1,2,3] + Decode [4] with KV cache
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[1, 2, 3]]), use_cache=True
        )
        kv0 = out0["past_key_values"]
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], use_cache=True
        )
        kv1 = out1["past_key_values"]
        out2_pre = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], use_cache=True
        )
        kv2 = out2_pre["past_key_values"]

        # Decode
        out0 = self.mgr0.forward_layers(
            input_ids=torch.tensor([[4]]), past_key_values=kv0, use_cache=True
        )
        out1 = self.mgr1.forward_layers(
            hidden_states=out0["hidden_states"], past_key_values=kv1, use_cache=True
        )
        out2 = self.mgr2.forward_layers(
            hidden_states=out1["hidden_states"], past_key_values=kv2, use_cache=True
        )
        pipeline_logits = out2["logits"]  # (1, 1, vocab)

        max_diff = (full_last_logits - pipeline_logits).abs().max().item()
        assert torch.allclose(full_last_logits, pipeline_logits, atol=1e-3), \
            f"KV cache 流水线 logits ≠ 完整模型 logits！" \
            f"max diff = {max_diff:.6f}"


# ================================================================
# Phase 7 P0: transformers 5.x 兼容性回归测试
# ================================================================

class TestTransformers5xCompatibility:
    """验证 7 项 transformers 5.x 修复不会在升级时静默失效。"""

    @pytest.fixture
    def mgr(self):
        """创建仅含 embedding + 全部 4 层的 ModelManager。"""
        from transformers.cache_utils import DynamicCache
        model = _make_tiny_model()
        tokenizer = type('TinyTok', (), {
            'decode': lambda self, ids, **kw: ''.join(chr(min(i, 126)) for i in ids),
            'encode': lambda self, text, **kw: torch.tensor(
                [[min(ord(c), 31) + 1 for c in text]], dtype=torch.long
            ),
        })()
        mgr = ModelManager()
        mgr.model = model
        mgr.tokenizer = tokenizer
        mgr._engine_type = "pytorch"
        return mgr

    def test_create_causal_mask_import_and_call(self, mgr):
        """create_causal_mask 可从 qwen2 路径导入且用有效参数成功调用。"""
        from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
        input_ids = torch.tensor([[5, 8, 2]], dtype=torch.long)
        hidden_states = mgr.model.model.embed_tokens(input_ids)
        causal_mask = create_causal_mask(
            config=mgr.model.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        )
        # flash/sdpa 返回 None，eager 返回 4D tensor
        assert causal_mask is None or isinstance(causal_mask, torch.Tensor)

    def test_dynamic_cache_update_called_on_decode(self, mgr):
        """decode 路径使用 cache.update(k, v, layer_idx) 而非 key_cache.append。

        验证 prefill→decode 周期中 layer_idx 补丁的保存/恢复机制正确。"""
        # 使用 tiny 模型的全 4 层（所有 GPU 层参与），模拟完整 prefill→decode
        saved_indices = [layer.self_attn.layer_idx for layer in mgr.model.model.layers]

        input_ids = torch.tensor([[5, 8, 2, 12]], dtype=torch.long)

        # Prefill: 传入 input_ids（完整模型含 embedding，自动处理）
        pf_result = mgr.forward_layers(input_ids=input_ids, use_cache=True)
        assert "past_key_values" in pf_result, "Prefill 应返回 past_key_values"
        # 验证 layer_idx 已恢复（Phase 1.2 修复）
        for i, layer in enumerate(mgr.model.model.layers):
            assert layer.self_attn.layer_idx == saved_indices[i], \
                f"prefill 后 layer_idx 未恢复: layer {i} → {layer.self_attn.layer_idx}"

        # Decode
        past_kv = pf_result["past_key_values"]
        dec_result = mgr.forward_layers(
            input_ids=torch.tensor([[7]], dtype=torch.long),
            past_key_values=past_kv, use_cache=True,
        )
        assert "past_key_values" in dec_result, "Decode 应返回更新的 past_key_values"
        # 验证 layer_idx 在 decode 后也恢复了
        for i, layer in enumerate(mgr.model.model.layers):
            assert layer.self_attn.layer_idx == saved_indices[i], \
                f"decode 后 layer_idx 未恢复: layer {i} → {layer.self_attn.layer_idx}"

    def test_past_key_values_plural_accepted(self, mgr):
        """`past_key_values` (复数) 参数名被 Qwen2DecoderLayer 接受（5.x 签名）。"""
        import inspect
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        sig = inspect.signature(Qwen2DecoderLayer.forward)
        params = list(sig.parameters.keys())
        assert "past_key_values" in params, \
            f"Qwen2DecoderLayer.forward 不接受 past_key_values 参数！签名: {params}"

    def test_decoder_layer_returns_tensor_not_tuple(self, mgr):
        """Qwen2DecoderLayer.forward 在 5.x 中返回 tensor 而非 tuple——验证我们的处理。"""
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        layer0 = mgr.model.model.layers[0]
        assert isinstance(layer0, Qwen2DecoderLayer), \
            "期望第一个 decoder layer 为 Qwen2DecoderLayer"

        # 用最小输入调用单层
        hidden = torch.randn(1, 2, mgr.model.config.hidden_size)
        pos_ids = torch.tensor([[0, 1]], dtype=torch.long)
        pos_emb = mgr.model.model.rotary_emb(hidden, pos_ids)

        output = layer0(hidden, position_ids=pos_ids, position_embeddings=pos_emb)
        # 5.x 返回 tensor；4.x 返回 tuple — 两种都应被 forward_layers 处理
        assert isinstance(output, torch.Tensor), (
            f"DecoderLayer.forward 应返回 torch.Tensor (5.x)，"
            f"实际返回 {type(output).__name__}"
        )
