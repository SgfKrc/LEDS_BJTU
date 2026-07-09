# 重模型实验支持计划 — DeepSeek 开源系列集成方案

> **日期**: 2026-07-08  
> **关联**: `src/model_config.py` (模型注册表), `src/model_module.py` (双引擎加载)  
> **状态**: 规划中，待评审  
> **目标**: 在现有 Qwen-1.8B 默认模型基础上，支持 DeepSeek 系列大模型实验

---

## 1. 背景与动机

当前系统默认模型为 Qwen-1.8B-Chat（1.8B 参数），内置实验模型仅 Qwen2.5-7B/14B。对于 CUDA 用户（尤其是 RTX 4090 24GB 等高端 GPU），Qwen 系列的能力天花板偏低。

DeepSeek 系列是当前最强的国产开源大模型家族：
- **DeepSeek-V3**: 671B MoE（37B 活跃参数），旗舰级推理能力
- **DeepSeek-R1**: 强化推理链（CoT），数学/编程/逻辑任务超越 GPT-4o
- **DeepSeek-Coder-V2**: 236B MoE（21B 活跃），代码生成领域 SOTA

**关键问题**: 这些模型大多采用 MoE 架构 + Multi-head Latent Attention (MLA)，与当前系统假设的 Dense Transformer 架构存在差异。需要在 **架构兼容性**、**量化支持**、**VRAM 可行性** 三个维度进行评估。

---

## 2. DeepSeek 模型家族概览

### 2.1 候选模型清单

| 模型 | 总参数 | 活跃参数 | 架构 | 上下文 | 适用场景 |
|------|--------|---------|------|--------|---------|
| **DeepSeek-V2-Lite** | 16B MoE | 2.4B | MLA + DeepSeekMoE | 128K | 轻量实验，16GB VRAM |
| **DeepSeek-Coder-V2-Lite** | 16B MoE | 2.4B | MLA + DeepSeekMoE | 128K | 代码生成入门 |
| **DeepSeek-V2** | 236B MoE | 21B | MLA + DeepSeekMoE | 128K | 综合能力强（旗舰） |
| **DeepSeek-Coder-V2** | 236B MoE | 21B | MLA + DeepSeekMoE | 128K | 代码 SOTA |
| **DeepSeek-V3** | 671B MoE | 37B | MLA + DeepSeekMoE + MTP | 128K | 最新旗舰（2025.3） |
| **DeepSeek-R1** | 671B MoE | 37B | MLA + DeepSeekMoE + CoT | 128K | 推理特化 |
| **DeepSeek-LLM-7B** | 7B Dense | 7B | 标准 LLaMA-like | 4K | 兼容性最佳 |
| **DeepSeek-LLM-67B** | 67B Dense | 67B | 标准 LLaMA-like | 4K | 需多 GPU |

### 2.2 架构特征分析

DeepSeek 模型分为两类架构：

**Dense 架构（兼容现有系统）**:
- DeepSeek-LLM-7B / 67B: 标准 LLaMA-like Transformer，`model.layers` 结构
- 与现有 Qwen 加载路径 **完全兼容**（HuggingFace transformers + bitsandbytes）
- `_count_transformer_layers()` 可直接识别

**MoE + MLA 架构（需要适配）**:
- DeepSeek-V2/V3/Coder 系列: 使用 **Multi-head Latent Attention** 替代标准 MHA
- 层结构仍通过 `model.layers` 暴露（HuggingFace 兼容）
- MLA 的 KV cache 结构不同（compressed latent vector），影响分布式管线 KV cache 传输
- MoE 的专家路由（top-k gating）引入了额外的参数和计算模式

### 2.3 框架支持情况

| 框架 | Dense (7B/67B) | MoE (V2/V3) | 备注 |
|------|:---:|:---:|------|
| **transformers** | ✅ 完全支持 | ✅ 4.40+ 原生支持 | DeepSeek 官方使用 HuggingFace 发布 |
| **bitsandbytes** | ✅ int4/int8 | ⚠️ MoE gate 层不支持量化 | 专家 FFN 可量化，router 保持 fp16 |
| **llama.cpp** | ✅ GGUF 可用 | ⚠️ V2-Lite GGUF 社区支持 | V3/R1 GGUF 需大内存 (>200GB) |
| **vLLM** | ✅ | ✅ 官方推荐 | 生产部署首选，但非本项目方向 |
| **torch.compile** | ✅ +3~8% | ❌ 不兼容 MoE | 动态路由不适用静态图融合 |

---

## 3. VRAM 可行性评估

### 3.1 单 GPU 场景

| 模型 | 精度 | VRAM 需求 | RTX 4060 8G | RTX 4090 24G |
|------|------|----------|:-----------:|:------------:|
| DeepSeek-LLM-7B | int4 | ~5 GB | ✅ | ✅ |
| DeepSeek-LLM-7B | int8 | ~8 GB | ⚠️ 临界 | ✅ |
| DeepSeek-LLM-7B | fp16 | ~14 GB | ❌ | ✅ |
| DeepSeek-V2-Lite (16B) | int4 | ~10 GB | ❌ | ✅ |
| DeepSeek-V2-Lite (16B) | int8 | ~18 GB | ❌ | ✅ |
| DeepSeek-V2 (236B) | int4 | ~60 GB | ❌ | ❌ |
| DeepSeek-V3 (671B) | int4 | ~180 GB | ❌ | ❌ |

### 3.2 多 GPU / 分布式管线场景

本项目核心功能是 **分布式推理**（多节点管线并行），天然适合大模型切分：

| 场景 | 配置 | 可运行模型 |
|------|------|-----------|
| 2× RTX 4090 (48GB) | 管线2段 | DeepSeek-V2-Lite int8 (18GB+18GB) |
| 3× RTX 4060 (24GB) | 管线3段 | DeepSeek-LLM-67B int4 (~24GB) |
| 4× RTX 4090 (96GB) | 管线4段 | DeepSeek-V2 int4 (~60GB) |

### 3.3 推荐优先级

| 优先级 | 模型 | 理由 |
|--------|------|------|
| **P0 立即支持** | DeepSeek-LLM-7B (Dense) | 架构兼容，8GB VRAM 可用，GGUF 可选 |
| **P1 短期支持** | DeepSeek-V2-Lite (16B MoE) | 16GB VRAM 可跑 int8，架构可适配 |
| **P2 中期探索** | DeepSeek-Coder-V2-Lite | 与 V2-Lite 同架构，代码场景有需求 |
| **P3 远期目标** | DeepSeek-V2/V3/R1 (全量) | 需多 GPU 分布式，旗舰级能力 |

---

## 4. 技术适配方案

### 4.1 P0: DeepSeek-LLM-7B（Dense，~2h）

与现有 Qwen 加载路径 100% 兼容，仅需注册模型配置。

**新增 `model_config.py`**:
```python
ModelConfig(
    model_id="deepseek-llm-7b",
    name="DeepSeek-LLM-7B-Chat",
    model_type="both",
    model_path=os.path.join(_APP_ROOT, "models", "deepseek-llm-7b-chat"),
    gguf_path=os.path.join(_APP_ROOT, "models", "deepseek-llm-7b-chat-Q4_K_M.gguf"),
    recommended_vram_gb=8.0,
    max_context=4096,
    is_experimental=True,
    huggingface_id="deepseek-ai/deepseek-llm-7b-chat",
    quant_types=["fp16", "int8", "int4"],
    description="DeepSeek 7B Dense 模型。标准 LLaMA 架构，兼容性好。INT4 ~5GB VRAM。",
    location="external",
),
```

**验证点**:
- `_count_transformer_layers()`: `model.model.layers` → 30 层
- `bitsandbytes` int4/int8 量化正常
- `torch.compile` 可用（+5%）

### 4.2 P1: DeepSeek-V2-Lite（MoE + MLA，~6h）

需要 **3 个关键适配**:

#### 4.2.1 层数检测扩展 (`_count_transformer_layers`)

MoE 模型的层结构也在 `model.model.layers` 下（HuggingFace 兼容），但每层内包含 `mlp.gate`（路由）+ `mlp.experts`（专家列表）+ `self_attn`（MLA）。

```python
def _count_transformer_layers(self) -> int:
    # 现有检测...
    # DeepSeek V2/V3: model.model.layers (HuggingFace 兼容)
    # MoE 检测: 检查 layer.mlp.gate 是否存在
    if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
        layers = self.model.model.layers
        if layers and hasattr(layers[0].mlp, "gate"):
            logger.info("检测到 MoE 架构 (DeepSeekMoE)")
            self._is_moe = True
        return len(layers)
    return 0
```

#### 4.2.2 量化适配 — MoE Gate 保护

bitsandbytes `int4`/`int8` 不能量化 MoE 的 gate/router 层（会导致路由崩溃）。需要在加载时指定 `modules_to_not_convert`:

```python
# 新增 MoE 感知量化配置
def _get_quant_config(self, quant_type: str, is_moe: bool = False):
    if is_moe and quant_type in ("int4", "int8"):
        return BitsAndBytesConfig(
            load_in_4bit=True if quant_type == "int4" else False,
            load_in_8bit=True if quant_type == "int8" else False,
            llm_int8_skip_modules=["gate", "router"],  # MoE gate 保护
            # 或在 model 加载后手动恢复 gate 层为 fp16
        )
    # ... 标准配置
```

**更安全的方案**: 先加载全模型 fp16，检测 MoE gate 层，再用 `prepare_model_for_kbit_training` 方式选择性量化。

#### 4.2.3 KV Cache 分布式传输适配 — MLA 压缩

MLA（Multi-head Latent Attention）使用低秩压缩的 KV cache（latent vector 而非完整 K/V head）。当前分布式管线 `_kv_cache[task_id]` 存储的是 `past_key_values` 标准结构。DeepSeek V2 的 `past_key_values` 结构为：
- `kv_a`: compressed latent (小)
- `k_pe`: position encoding key (标准大小)

**传输量**：MLA 的压缩 KV cache 比标准 MHA 小 ~4-8×（从 `n_heads × head_dim` 降到 `latent_dim`），意外地对分布式传输更友好。

**适配**: `_kv_cache` 存储逻辑 **无需修改** — transformers 4.40+ 已经将 MLA 的 KV 抽象为标准的 `past_key_values` 接口。只需验证序列化/反序列化不破坏 latent 结构。

### 4.3 聊天模板适配

DeepSeek 系列使用与 Qwen 不同的 chat template:

| 模型 | Chat Template |
|------|--------------|
| Qwen-1.8B | `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>` |
| DeepSeek-LLM | `User: ...\n\nAssistant: ...` |
| DeepSeek-V2/V3 | `<｜begin▁of▁sentence｜>User: ...<｜end▁of▁sentence｜>Assistant: ...` |

**适配**: `tokenizer.apply_chat_template()` 自动处理（HuggingFace tokenizer_config.json 包含模板），无需手动干预。但前端预设的 prompt 格式可能需要模型感知的模板切换。

---

## 5. 集成架构设计

### 5.1 整体流程

```
用户选择模型 (前端 ModelSelector / SettingsModal)
        │
        ▼
POST /api/models/switch  { model_id: "deepseek-llm-7b", quant_type: "int4" }
        │
        ▼
ModelManager.switch_model()
  ├── unload_model()          # 卸载当前模型，释放 VRAM
  ├── get_model_config()      # 查模型注册表 → ModelConfig
  ├── select_engine()         # 自动选引擎 (pytorch / llama_cpp)
  ├── load_model()            # 加载新模型
  │     ├── [Dense] → 标准 HuggingFace + bitsandbytes
  │     ├── [MoE]  → MoE 感知量化 (gate 保护) + 架构检测
  │     └── [GGUF] → llama.cpp
  └── _count_transformer_layers() → 写 config 供 Scheduler 分层
        │
        ▼
Scheduler 自动调整分层配置 (get_layer_assignments)
        │
        ▼
前端更新状态: 新模型已就绪
```

### 5.2 模型注册表扩展

```python
# model_config.py BUILTIN_MODELS 新增条目

# === DeepSeek Dense ===
ModelConfig(
    model_id="deepseek-llm-7b",
    name="DeepSeek-LLM-7B-Chat",
    model_type="both",
    model_path=os.path.join(_APP_ROOT, "models", "deepseek-llm-7b-chat"),
    gguf_path=os.path.join(_APP_ROOT, "models", "deepseek-llm-7b-chat-Q4_K_M.gguf"),
    recommended_vram_gb=8.0,
    max_context=4096,
    is_experimental=True,
    huggingface_id="deepseek-ai/deepseek-llm-7b-chat",
    quant_types=["fp16", "int8", "int4"],
    description="DeepSeek 7B Dense 模型。标准 LLaMA 架构，兼容性好。INT4 ~5GB VRAM，GGUF ~4.3GB。",
    location="external",
),

# === DeepSeek MoE Lite ===
ModelConfig(
    model_id="deepseek-v2-lite",
    name="DeepSeek-V2-Lite-Chat (16B MoE)",
    model_type="safetensors",
    model_path=os.path.join(_APP_ROOT, "models", "deepseek-v2-lite-chat"),
    gguf_path="",
    recommended_vram_gb=16.0,
    max_context=131072,        # 128K (实际使用受 VRAM 限制)
    is_experimental=True,
    huggingface_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    quant_types=["int8", "int4"],
    description="DeepSeek-V2-Lite 16B MoE (2.4B 活跃)。MLA + DeepSeekMoE 架构，128K 上下文。需 RTX 4090 或分布式管线。",
    location="external",
),

# === DeepSeek Coder Lite ===
ModelConfig(
    model_id="deepseek-coder-v2-lite",
    name="DeepSeek-Coder-V2-Lite (16B MoE)",
    model_type="safetensors",
    model_path=os.path.join(_APP_ROOT, "models", "deepseek-coder-v2-lite"),
    gguf_path="",
    recommended_vram_gb=16.0,
    max_context=131072,
    is_experimental=True,
    huggingface_id="deepseek-ai/DeepSeek-Coder-V2-Lite",
    quant_types=["int8", "int4"],
    description="DeepSeek-Coder-V2-Lite 16B MoE 代码模型。338 种编程语言，128K 上下文。",
    location="external",
),
```

---

## 6. 实施计划

### 阶段 1: P0 — Dense 模型快速支持（~2h）

| 任务 | 文件 | 工时 |
|------|------|------|
| 注册 DeepSeek-LLM-7B 到内置模型表 | `model_config.py` | 0.3h |
| 添加 DeepSeek chat_template 验证 | `model_module.py` | 0.3h |
| 前端模型列表自动显示新模型 | 无需改动（现有逻辑自动读取） | — |
| 下载测试 + VRAM 基准测试 | 手动 | 1h |
| 单元测试: 加载/推理/卸载 | `tests/test_model_module.py` | 0.4h |

### 阶段 2: P1 — MoE 架构适配（~6h）

| 任务 | 文件 | 工时 |
|------|------|------|
| MoE 架构检测 (`_is_moe` 标志) | `model_module.py` | 0.5h |
| MoE gate 层保护量化 | `model_module.py` | 1.5h |
| KV cache 序列化兼容 MLA | `model_module.py` | 1h |
| `_count_transformer_layers()` 扩展 | `model_module.py` | 0.3h |
| 注册 DeepSeek-V2-Lite + Coder-V2-Lite | `model_config.py` | 0.3h |
| 分布式管线 MoE 适配测试 | `scheduler.py` | 1h |
| 端到端测试: 加载→推理→卸载 | `tests/` | 1.4h |

### 阶段 3: P2/P3 — 大模型 + 分布式优化（~8h，远期）

| 任务 | 说明 | 工时 |
|------|------|------|
| DeepSeek-V2 236B 注册 | 需多 GPU 分布式管线 | 0.5h |
| 自动分层策略适配 MoE | 专家分布感知的层分配 | 3h |
| DeepSeek-R1 CoT 推理支持 | 思维链展示 + 长上下文管理 | 2h |
| GGUF 路径验证 (llama.cpp MoE) | 社区 GGUF 支持验证 | 1.5h |
| 性能基准 + 文档 | 对比报告 | 1h |

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:----:|------|------|
| bitsandbytes 不兼容 MoE gate | 中 | 量化后模型输出乱码 | 自动检测 gate 层 + 保持 fp16 |
| MLA `past_key_values` 序列化失败 | 低 | 分布式 KV cache 传输失败 | 测试 `torch.save/load` roundtrip |
| transformers 版本过低不支持 DeepSeek | 低 | 加载报错 | 要求 `transformers>=4.40.0` |
| 16B 模型 int8 + 128K 上下文 OOM | 高 | 24GB VRAM 不够 | 限制 `max_context` 到实际可用；动态 `max_page_num` |
| MoE 专家路由不均衡 | 中 | 部分节点过载 | 分布式管线中按专家分布分配层 |

---

## 8. 验证计划

### P0 验证
```bash
# 模型注册
python -c "from src.model_config import get_builtin_model; m=get_builtin_model('deepseek-llm-7b'); print(m.name)"

# 加载测试（需模型已下载到 models/deepseek-llm-7b-chat/）
curl -X POST /api/models/switch -d '{"model_id":"deepseek-llm-7b","quant_type":"int4"}'

# GGUF 路径
curl -X POST /api/models/switch -d '{"model_id":"deepseek-llm-7b","quant_type":"Q4_K_M"}'
```

### P1 验证
```bash
# MoE 架构检测
python -c "
from src.model_module import ModelManager
# 模拟加载后检查 _is_moe 标志
"

# KV cache MLA roundtrip
python -c "
# 加载 V2-Lite, 运行 prefill, 保存 past_key_values, 恢复, 继续 decode
# 验证恢复后的 token 一致性
"

# 分布式管线 MoE
# 2 节点各分配一半层，验证 MoE 专家路由在拆分后仍正常
```

---

## 附录 A: 调研补充发现（2026-07-08）

### A.1 DeepSeek-R1 蒸馏模型 — 最佳兼容性方案

DeepSeek-R1 的蒸馏版本基于 **Qwen2.5** 或 **Llama-3** 密集架构，与当前系统 **100% 兼容**:

| 模型 | 基础架构 | Q4_K_M GGUF | 推荐 GPU |
|------|---------|-------------|---------|
| R1-Distill-Qwen-1.5B | Qwen2.5 | ~1.1 GB | 任何 |
| R1-Distill-Qwen-7B | Qwen2.5 | ~4.7 GB | RTX 4060 8GB |
| R1-Distill-Llama-8B | Llama-3.1 | ~5.2 GB | RTX 4060 8GB |
| R1-Distill-Qwen-14B | Qwen2.5 | ~9.0 GB | RTX 4070 12GB+ |
| R1-Distill-Qwen-32B | Qwen2.5 | ~20 GB | RTX 4090 24GB |
| R1-Distill-Llama-70B | Llama-3.3 | ~43 GB | 2× RTX 3090 |

**关键优势**: 零代码更改即可加载 — Qwen/Llama 架构被 transformers 和 llama.cpp 原生支持。Bitsandbytes int4/int8 量化完全兼容。分布式层分割直接可用。

### A.2 DeepSeek-V2-Lite GGUF 路径

截至 2025-01，主流 llama.cpp 已通过 PR #12801 合并 MLA 支持。关键配置:
- `num_key_value_heads=1`（MLA 在 GGUF 中伪装成 MQA）
- 新增元数据: `n_embd_head_k_mla`, `n_embd_head_v_mla`

**推荐集成路径**: llama.cpp GGUF（而非 PyTorch + bitsandbytes），因为:
1. GGUF 量化成熟（Q2_K ~ Q8_0 可用）
2. bitsandbytes 在 MoE 自定义层上有已知问题
3. llama-cpp-python >= 0.3.x 内置 MLA 支持

### A.3 旗舰模型可行性结论

DeepSeek-V3/R1 (671B) 和 DeepSeek-V2/Coder-V2 (236B) **不适合消费级硬件**:
- FP8 原生权重 ~671 GB，Int4 量化仍需要 ~325 GB
- 需要 8× H100 80GB 级别硬件
- 建议通过 vLLM 在专用服务器上部署，不作为 QLH 目标

### A.4 调整后的 P0/P1 优先级

| 优先级 | 模型 | 兼容性 | 工时 | VRAM |
|--------|------|:---:|------|------|
| **P0** | R1-Distill-Qwen-7B/14B | 100% 兼容 | ~2h | 8-16GB |
| **P0** | DeepSeek-LLM-7B-Chat | 100% 兼容 | ~2h | 8GB |
| **P1** | R1-Distill-Qwen-32B | 100% 兼容 (需多GPU) | ~3h | 24GB+ |
| **P2** | DeepSeek-V2-Lite (GGUF) | llama.cpp MLA 支持 | ~6h | 12-16GB |
| **P3** | DeepSeek-Coder-V2-Lite | 同 V2-Lite 架构 | ~4h | 12-16GB |
| 远期 | DeepSeek-V2/V3/R1 (全量) | 需 vLLM + 服务器集群 | — | 160GB+ |
