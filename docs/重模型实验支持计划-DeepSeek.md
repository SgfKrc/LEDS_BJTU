# 重模型实验支持计划 — DeepSeek 开源系列集成方案

> **日期**: 2026-07-08  
> **关联**: `src/model_config.py` (模型注册表), `src/model_module.py` (双引擎加载)  
> **状态**: 部分实施；模型注册槽位已落地，真实加载与分布式兼容仍需逐模型验收
> **目标**: 在现有 Qwen-1.8B 默认模型基础上，支持 DeepSeek 系列大模型实验

---

## 1. 背景与动机

当前系统默认模型为 Qwen-1.8B-Chat（1.8B 参数），模型注册表已经包含 Qwen2.5 与多个 DeepSeek-R1-Distill-Qwen 实验槽位。注册项只表示可被选择和下载，不代表每个模型已经完成真实加载、量化、完整推理和分布式层流水线验收。

DeepSeek 系列是代表性的国产开放权重大模型家族之一：
- **DeepSeek-V3**: 671B MoE（37B 活跃参数），旗舰级推理能力
- **DeepSeek-R1**: 强化推理链（CoT），数学/编程/逻辑任务超越 GPT-4o
- **DeepSeek-Coder-V2**: 236B MoE（21B 活跃），代码生成领域 SOTA

**关键问题**: 这些模型大多采用 MoE 架构 + Multi-head Latent Attention (MLA)，与当前系统假设的 Dense Transformer 架构存在差异。需要在 **架构兼容性**、**量化支持**、**VRAM 可行性** 三个维度进行评估。

---

## 2. DeepSeek 模型家族概览

### 2.1 候选模型清单

| 模型 | 总参数 | 活跃参数 | 架构 | 上下文 | 适用场景 |
|------|--------|---------|------|--------|---------|
| **DeepSeek-V2-Lite** | 16B MoE | 2.4B | MLA + DeepSeekMoE | 32K | 轻量实验，16GB VRAM |
| **DeepSeek-Coder-V2-Lite** | 16B MoE | 2.4B | MLA + DeepSeekMoE | 128K | 代码生成入门 |
| **DeepSeek-V2** | 236B MoE | 21B | MLA + DeepSeekMoE | 128K | 综合能力强（旗舰） |
| **DeepSeek-Coder-V2** | 236B MoE | 21B | MLA + DeepSeekMoE | 128K | 代码 SOTA |
| **DeepSeek-V3** | 671B MoE | 37B | MLA + DeepSeekMoE + MTP | 128K | 旗舰 MoE 基线（2025.3 版本） |
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
    max_context=32768,         # 32K (官方 V2-Lite 上下文；实际使用受 VRAM 限制)
    is_experimental=True,
    huggingface_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    quant_types=["int8", "int4"],
    description="DeepSeek-V2-Lite 16B MoE (2.4B 活跃)。MLA + DeepSeekMoE 架构，32K 上下文。需 RTX 4090 或分布式管线。",
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
| 16B MoE 模型长上下文 OOM | 高 | 24GB VRAM 不够 | V2-Lite 按 32K 上限注册；Coder-V2-Lite 128K 需按显存动态限制 `max_page_num` |
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

## 9. 模型获取与依赖调研落地清单

> 调研日期: 2026-07-10
> 结论: 先支持官方 Safetensors 路径；GGUF 只在选定具体量化仓库和文件名后注册为稳定入口。全量 DeepSeek-R1/V3 不进入本地 P0/P1，优先作为 vLLM 外部服务目标。

### 9.1 官方信息修正

- DeepSeek-R1 全量模型为 671B 总参数、37B 激活参数、128K 上下文；官方卡片仍提示本地运行参考 DeepSeek-V3 仓库，不作为本项目 PyTorch 分层路径的首批目标。
- R1 蒸馏模型是 Qwen2.5 / Llama 系列 Dense 架构，官方说明可按 Qwen 或 Llama 模型使用，是本项目最稳的 DeepSeek 推理能力入口。
- DeepSeek-V2-Lite / V2-Lite-Chat 为 16B 总参数、2.4B 激活参数、32K 上下文；官方 Transformers 示例仍使用 `trust_remote_code=True`，原方案中的 128K 已修正为 32K。
- DeepSeek-Coder-V2-Lite-Instruct 为 16B 总参数、2.4B 激活参数、128K 上下文，代码模型优先级应放在 V2-Lite Chat 跑通之后。
- vLLM 当前支持 `DeepseekForCausalLM`、`DeepseekV2ForCausalLM`、`DeepseekV3ForCausalLM`，适合全量 V2/V3/R1 的外部服务化验证，不直接替代本项目的 PyTorch 层拆分管线。

### 9.2 模型获取矩阵

| 优先级 | HuggingFace 仓库 | 本地目录 / 文件 | 首选引擎 | 磁盘预估 | 许可关注 | 处理结论 |
|--------|------------------|-----------------|----------|----------|----------|----------|
| P0 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `models/deepseek-r1-distill-qwen-7b/` | PyTorch | ~15GB | MIT + Qwen Apache 2.0 派生说明 | 立即注册 Safetensors；GGUF 另选量化仓库后再加 |
| P0 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `models/deepseek-r1-distill-qwen-14b/` | PyTorch | ~30GB | MIT + Qwen Apache 2.0 派生说明 | 高端 GPU P0；8GB 设备不默认展示 |
| P0 | `deepseek-ai/deepseek-llm-7b-chat` | `models/deepseek-llm-7b-chat/` | PyTorch | ~14GB | DeepSeek Model License，支持商用但需保留许可证 | 保留原 P0，作为 DeepSeek Dense 基线 |
| P1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | `models/deepseek-r1-distill-qwen-32b/` | PyTorch / 分布式 | ~65GB | MIT + Qwen Apache 2.0 派生说明 | RTX 4090 或多节点验证，不进默认推荐 |
| P2 | `deepseek-ai/DeepSeek-V2-Lite-Chat` | `models/deepseek-v2-lite-chat/` | PyTorch 实验 / vLLM 对照 | ~32GB | DeepSeek License | 先验证 tokenizer、层数、KV cache；再谈量化 |
| P3 | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `models/deepseek-coder-v2-lite-instruct/` | PyTorch 实验 / vLLM 对照 | ~32GB | DeepSeek License | 复用 V2-Lite 适配结果，补代码场景测试 |
| 远期 | `deepseek-ai/DeepSeek-R1` / `DeepSeek-V3` | 不建议本地落盘 | vLLM / SGLang 服务 | 600GB+ | R1 为 MIT，V3 按官方仓库许可证复核 | 只作为远程 OpenAI-compatible API 接入目标 |

> 磁盘预估按 FP16/BF16 权重约 `参数量 * 2 bytes` 粗算。GGUF 体量取决于具体量化格式和仓库，不在未选仓库前写死到 `ModelConfig`。

### 9.3 下载流程

建议新增显式下载依赖，避免依赖 `transformers` 间接带来的 `huggingface_hub` 版本:

```bash
pip install -U "huggingface_hub>=0.32.0"
```

标准下载采用 `snapshot_download(local_dir=...)`，因为它能保留原仓库结构并支持断点续传。下载前必须先 dry-run 看体量:

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dry-run
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --dry-run
hf download deepseek-ai/deepseek-llm-7b-chat --dry-run
```

落盘到项目 `models/` 的推荐脚本:

```python
from huggingface_hub import snapshot_download

COMMON_PATTERNS = [
    "*.json",
    "*.safetensors",
    "tokenizer*",
    "*.model",
    "*.txt",
    "*.md",
]

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_dir="models/deepseek-r1-distill-qwen-7b",
    allow_patterns=COMMON_PATTERNS,
)
```

Windows PowerShell 建议设置项目内缓存，避免默认缓存落到用户目录导致打包/迁移不可控:

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir models/deepseek-r1-distill-qwen-7b
```

GGUF 获取策略:
- 不把社区 GGUF 仓库直接写入内置注册表，先由用户或维护者选择具体仓库、文件名和量化格式。
- 选定后使用 `hf download <repo> <file.gguf> --local-dir models/`，并将 `gguf_path` 指向单文件。
- 每个 GGUF 文件都记录来源仓库、commit hash、量化格式、文件大小和 SHA256，写入 `models/model_manifest.json`。

### 9.4 依赖版本调研

当前项目依赖与 DeepSeek 实验的关系:

| 依赖 | 当前配置 | 调研结论 | 建议 |
|------|----------|----------|------|
| `torch` | `>=2.2.0` | bitsandbytes 官方当前最低要求为 PyTorch 2.4；CUDA 重模型路径不宜继续按 2.2 验证 | DeepSeek CUDA 实验环境提升到 `torch>=2.4.0`；通用依赖是否提升需跑完整回归 |
| `transformers` | `>=4.45.0,<5.0.0` | P0 Dense/R1-Distill 足够；V2-Lite/Coder 官方示例仍需 `trust_remote_code=True`；全量 R1/V3 仍建议走 vLLM/SGLang 服务 | 维持 `<5.0.0`，避免破坏现有兼容修复；MoE 适配单独建隔离 venv 验证最新版 4.x |
| `accelerate` | `>=1.0.0` | PyTorch 大模型加载和 `device_map` 仍需要 | 保持，P1+ 验证 `device_map="auto"` 与本项目分层加载是否冲突 |
| `bitsandbytes` | `>=0.45.0` | 官方支持 NVIDIA、CPU、Intel XPU/Gaudi，Windows x86-64 CUDA wheel 覆盖 CUDA 11.8-12.9；但 MoE gate/router 量化仍需项目侧保护 | P0 可以继续；P2 MoE 只允许实验开关，不默认 int4 |
| `llama-cpp-python` | 仅 `packaging/requirements-cpu.txt` 有 `>=0.3.0` | GGUF 路径依赖该包；Windows 可选 CPU/Vulkan/CUDA wheel，不同 wheel 不能混装 | 若 PC 开发环境也要测 GGUF，把它加入可选 extras 或 dev 文档 |
| `huggingface_hub` | 未显式列出 | 模型获取、dry-run、断点续传、`local_dir` 落盘都依赖它；0.32+ 默认整合 `hf_xet` | 在主依赖或下载脚本依赖中显式加入 `huggingface_hub>=0.32.0` |
| `safetensors` | 未显式列出 | 大模型仓库多数使用 safetensors；transformers 会间接依赖，但校验脚本直接读取时需要明确依赖 | 在下载/校验工具依赖中显式加入 `safetensors>=0.4.5` |
| `vllm` | 未列入 | 支持 DeepSeek/V2/V3/R1 架构，但 Windows 本地并不适合作为项目依赖 | 不加入主依赖；只写入 Linux 服务端部署文档 |

建议新增一个独立重模型环境文件，避免污染普通用户安装:

```text
# packaging/requirements-heavy-models.txt
torch>=2.4.0
transformers>=4.45.0,<5.0.0
accelerate>=1.0.0
bitsandbytes>=0.45.0
huggingface_hub>=0.32.0
safetensors>=0.4.5
```

GGUF 开发/测试环境另列，不与 CUDA PyTorch 环境强绑定:

```text
# packaging/requirements-gguf-dev.txt
llama-cpp-python>=0.3.0
huggingface_hub>=0.32.0
```

### 9.5 集成任务拆分更新

| 任务 | 文件 | 验收 |
|------|------|------|
| 注册 R1-Distill-Qwen-7B/14B/32B | `src/model_config.py` | `/api/models` 在 CUDA 环境展示；CPU 环境仍隐藏实验模型 |
| 增加模型获取脚本 | `scripts/download_model.py` 或 `src/model_downloader.py` | 支持 `--dry-run`、`--repo-id`、`--local-dir`、`--include`、`--sha256` |
| 增加模型 manifest | `models/model_manifest.json` | 记录 repo、revision、文件大小、SHA256、license 摘要 |
| tokenizer 模板验证 | `tests/test_model_config.py` / 手动脚本 | `apply_chat_template()` 能处理 user-only 对话；R1 建议不注入 system prompt |
| `trust_remote_code` 安全边界 | `src/model_module.py` / 下载脚本 | 仅官方 DeepSeek 仓库允许开启；manifest 记录 revision，避免静默拉取变更代码 |
| 依赖矩阵验证 | CI 或手动 venv | CPU 基线不安装 CUDA-only 依赖；CUDA 重模型 venv 可导入 torch/transformers/bnb |
| MoE 预研开关 | `src/model_module.py` | V2-Lite 默认标记 experimental；未通过 KV cache roundtrip 前不开放分布式管线 |

### 9.6 最小验收路径

P0 第一次落地只做 3 个模型:

1. `DeepSeek-R1-Distill-Qwen-7B`: 验证推理能力、R1 输出格式、8GB/12GB 设备门槛。
2. `DeepSeek-R1-Distill-Qwen-14B`: 验证 16GB/24GB 设备上限和分布式层拆分。
3. `deepseek-llm-7b-chat`: 验证 DeepSeek 原生 Dense 模型与 R1 蒸馏模型的差异。

通过标准:
- 模型下载脚本 dry-run 能给出体量，实际下载后 manifest 完整。
- `ModelConfig` 注册路径与实际落盘一致。
- PyTorch int4/int8/fp16 至少一种加载成功，能完成 32 token smoke test。
- 卸载模型后 VRAM 回收，无残留导致下一次切换 OOM。
- 分布式路径先只对 Dense/R1-Distill 开启；MoE 模型必须等待 KV cache roundtrip 和层拆分验证。

### 9.7 调研来源

- DeepSeek-R1 官方 HuggingFace 卡片: https://huggingface.co/deepseek-ai/DeepSeek-R1
- DeepSeek-LLM-7B-Chat 官方 HuggingFace 卡片: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat
- DeepSeek-V2-Lite-Chat 官方 HuggingFace 卡片: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
- DeepSeek-Coder-V2-Lite-Instruct 官方 HuggingFace 卡片: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- HuggingFace Hub 下载文档: https://huggingface.co/docs/huggingface_hub/guides/download
- bitsandbytes 安装文档: https://huggingface.co/docs/bitsandbytes/main/en/installation
- transformers bitsandbytes 量化文档: https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
- llama-cpp-python 安装文档: https://llama-cpp-python.readthedocs.io/en/latest/
- vLLM 支持模型列表: https://docs.vllm.ai/en/latest/models/supported_models/

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

---

## 10. DeepSeek 分布式推理修复计划（2026-07-13）

> 本节覆盖前文中“R1-Distill 可直接使用现有分层”的早期判断。模型架构能由
> Transformers 加载，不等于现有多节点协议已经支持该模型。当前流水线以
> Qwen-1.8B 的 24 层和默认模型路径为隐含前提，DeepSeek-R1-Distill-Qwen-7B
> 实际为 28 层，必须完成以下改造后才能开放分布式模式。

### 10.1 目标与边界

- 首个目标模型为 `deepseek-r1-distill-qwen-7b` 的 PyTorch Safetensors 版本。
- 仅 PC 计算节点参与层拆分；Android 和请求转发型 PC 客户端不要求落盘模型。
- 同一流水线中的计算节点必须使用相同 `model_id`、revision、权重、config 和 tokenizer。
- GGUF/llama.cpp 继续只支持单机推理，不与 PyTorch 层拆分混用。
- 模型缺失、摘要缺失或加载失败一律 fail closed，回退主节点单机推理，不允许降级放行。

### 10.2 P0：建立模型协商协议与就绪屏障

1. 定义 `ModelManifest`：包含 `model_id`、`engine`、`revision`、权重联合 SHA-256、
   config/tokenizer SHA-256、`num_hidden_layers`、`hidden_size`、`vocab_size` 和量化策略。
2. 主节点从当前 `ModelManager` 的活跃模型生成 manifest，不再从默认 Qwen GGUF 路径取摘要。
3. `LAYER_CONFIG` 对每个节点下发 manifest、层范围和配置版本；修正当前发送端传单个 assignment、
   接收端却按 `{node_id: assignment}` 读取的消息结构不一致。
4. 从节点先校验本地文件，再按 manifest 指定的 PyTorch 模型加载层范围，返回
   `LAYER_CONFIG_ACK {config_version, model_sha256, layer_range, status, error}`。
5. 主节点只有收到匹配 ACK 后才将节点标记 ready；“TCP 在线”和“消息已发送”不能代表模型已就绪。
6. 模型切换时暂停新任务、清理各节点 KV cache、广播新配置，全部 ACK 后原子切换流水线版本；
   超时或任一失败则回退主节点单机推理。

验收：Qwen/DeepSeek 模型不一致、从节点缺模型、摘要为空、加载异常时均不得进入流水线。

### 10.3 P1：移除固定 24 层假设

1. 调度器从活跃模型 config/`_total_model_layers` 获取总层数，并随 manifest 下发。
2. 将 API 的 `start_layer <= 23`、`end_layer <= 24` 改为基于活跃模型的运行时校验。
3. 前端手动分层编辑器从 `/api/config/layers` 读取总层数，不再显示固定 `0-24`。
4. 模型切换后使旧分层配置失效并重新计算，禁止把 Qwen 的 24 层配置复用于 DeepSeek 的 28 层。
5. 校验相邻节点的 hidden size、dtype、vocab size 和首尾 embedding/lm_head 所有权。

验收：Qwen-1.8B 完整覆盖 `[0,24)`；DeepSeek-R1-Distill-Qwen-7B 完整覆盖 `[0,28)`。

### 10.4 P2：从节点模型落盘与预检

1. `models_pc.7z` 只作为默认 Qwen 离线包，不继续膨胀为包含所有重模型的单一压缩包。
2. 增加 manifest 驱动的节点预检/下载命令，明确展示缺失文件、预计体积、revision 和 SHA-256。
3. 管理端展示每个计算节点的模型状态：`missing/downloading/verifying/loading/ready/error`。
4. 下载完成后先校验再注册计算能力；不允许只有主节点存在 DeepSeek 时给从节点分配层。
5. 保留人工预分发方式，避免通过控制协议直接传输 15GB 级权重。

### 10.5 P3：降低分层加载峰值内存

当前 `load_layer_range()` 会先完整加载模型再删除未分配层。短期内每个计算节点因此仍需完整
Safetensors 和一次完整加载的峰值内存。后续应改为按 safetensors index 选择性加载目标层、
embedding、norm 和 lm_head；完成前，节点预检必须按“完整模型峰值”评估内存，不能按分配层数估算。

### 10.6 测试矩阵

| 层级 | 场景 | 通过条件 |
|------|------|----------|
| 单元 | manifest 生成/摘要/协议序列化 | 字段完整、稳定、不同模型摘要不同 |
| 协议 | config 下发与 ACK | 发送、接收、加载、ACK 形成闭环；版本不匹配被拒绝 |
| 负向 | 缺模型/错模型/空摘要/28 层套 24 层配置 | 不进入 ready，自动回退且错误可见 |
| 数学 | tiny Qwen2 多段前向 | 流水线 logits 与完整模型一致 |
| 集成 | 两个本机进程经真实 TCP 跑 prefill+decode | token、KV cache、超时和清理均正确 |
| 实机 | 2-3 台 PC 跑 Qwen 24 层和 DeepSeek 28 层 | 连续 20 次请求无串层、无旧 KV、无静默降级 |
| 故障 | 推理中断开从节点/切换模型 | 当前任务明确失败或回退，后续任务能恢复 |

### 10.7 实施顺序与完成定义

实施顺序固定为：P0 协议闭环 -> P1 动态层数 -> P2 模型预检 -> Qwen 回归 -> DeepSeek 实机验证
-> P3 选择性加载。只有同时满足以下条件才可在 UI 中标记 DeepSeek 分布式“可用”：

- 所有计算节点 manifest 完全匹配并返回加载 ACK。
- 28 层分配连续、无重叠、完整覆盖。
- tiny 模型、真实 Qwen 和真实 DeepSeek 的前向/解码测试全部通过。
- 节点缺模型或掉线时不会继续用错误权重计算。
- 日志和管理面板能区分“在线”“模型已校验”“层已加载”和“流水线 ready”。
