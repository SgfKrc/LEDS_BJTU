# PyTorch 层流水线加载峰值优化方案

> 文档生命周期：**现行（Active）**
>
> 创建日期：2026-08-13
> 更新日期：2026-08-13
> 适用范围：分布式层间流水线（layer-wise pipeline）从节点的模型加载瞬时显存/内存峰值治理；直接约束 Surface 7 级轻量从节点在重模型（DeepSeek 7B / gemma-4 12B / Qwen 9B 等）下的可用性
>
> 目标：**加载瞬间峰值 ≈ 分配层段权重 + 常量开销**，与稳态占用同量级，杜绝"全模型峰值压垮从节点"

---

## 1. 背景与问题

- 从节点形态：Surface 7 级轻量设备（RAM/显存有限），是分布式的主要物理从节点形态。
- 担忧：重模型（≥7B）若在加载瞬间出现**全模型权重峰值**（如 fp16 7B ≈ 14GB、fp16 12B ≈ 24GB），会直接压垮从节点——即使稳态只占层段。
- 项目现有契约：`EngineHost.load_layer_range` / `ModelManager.load_layer_range` 按主节点下发的 `layer_assignments` 只加载对应层段（见 `scheduler.compute_layer_assignment` + `_check_node_vram_for_layers`）。

## 2. 现状实现（2026-08-13 代码实测）

| 路径 | 实现 | 瞬时峰值 |
|---|---|---|
| **Qwen / Qwen2 系**（含 DeepSeek-R1-Distill-Qwen，`model_type=qwen2`） | `_load_qwen2_layer_range` / `_load_qwen_layer_range`：`safetensors.safe_open` **按 key 过滤**，只物化 `model.layers.{i}.`（i ∈ [start, end)）权重 | ✅ **≈ 分配层段权重 + 单层缓冲**（已是最优路径，无全模型峰值） |
| Embedding / LM Head | 按 `has_embedding` / `has_lm_head` 决定保留；过滤路径下未分配端不物化 | ✅ 常量级 |
| 加载后裁剪 | `ModuleList(kept)` + `del` 释放（`:640-660` 附近） | 防御性二次保险（过滤路径下已无多余权重可裁） |
| **其他架构**（`gemma4_unified`、`qwen3`、`qwen3_vl` 等） | `load_layer_range` 走 `else` 分支：`RuntimeError("当前 PyTorch 层流水线不支持模型架构")` | 不适用（尚未接入） |

**结论**：Qwen/Qwen2 系**现状已无全模型峰值问题**；真正的风险在**未来新架构接入层流水线时**（若实现者图省事走"全加载 → 截断"，就会引入峰值）。本方案的任务是：**把"按层 key 过滤物化"固化为唯一允许的接入路径**，并治理其余加载缓冲。

## 3. 峰值构成分析（重模型接入时的四个来源）

| 来源 | 说明 | 治理 |
|---|---|---|
| ① 全模型物化 | `from_pretrained` 或全量 `state_dict` 加载后再裁剪 | **禁止**——必须按层 key 过滤（safetensors `safe_open` 或 `state_dict` 预过滤） |
| ② 未分配端 Embedding/LM Head | 首/末节点外加载双端 | 按 `has_embedding` / `has_lm_head` 过滤 key |
| ③ 加载缓冲 | 权重读取页、转置临时张量、量化中间张量（fp16→int8/Q4 转换） | 逐层处理 + 及时 `del`/释放；量化在过滤后按层进行 |
| ④ 峰值叠加 | 加载 + 优化器/计算图同时存在 | 加载期禁止并发推理（现有加载锁已覆盖） |

## 4. 优化方案（分层执行）

### 4.1 固化现状为强制模式（P0，零代码或小改）
- 将 `_load_qwen2_layer_range` 的"按层 key 过滤物化"确认为**层流水线唯一加载模式**，写入本方案为架构契约：
  - 任何新架构的 `load_layer_range` 实现必须从 safetensors 按 key 过滤，**禁止**"全加载后截断"路径。
  - 现有防御性裁剪代码保留（不删除，作为兜底断言：过滤后实际物化层数 == 分配层数）。
- 在 `load_layer_range` 的 `else` 分支错误消息中补充指引：新架构接入须按本方案 §4.2 模板实现。

### 4.2 新架构接入模板（P1，未来架构必用）
1. 解析 config 的 `num_hidden_layers`，校验与 assignment 一致（现有逻辑复用）；
2. 确定层 key 前缀（`model.layers.` / `transformer.h.` / `model.h.`——按架构）；
3. `safe_open` 过滤：仅物化 `{prefix}{i}.`（i ∈ [start, end)）+ 按 `has_embedding`/`has_lm_head` 过滤首尾 key；
4. 构建模型骨架（`AutoConfig` + 空模型或 meta device），用过滤后的 state_dict 加载到目标设备；
5. 量化（若启用）在层段内逐层执行，中间张量即时释放；
6. 加载后断言：`model.{layers}.__len__() == end - start`，失败 fail-closed。

### 4.3 加载缓冲治理（P2，可选增强）
- 单层流式物化：safetensors `mmap` 读页，逐层 `del` 前一层临时引用（当前已接近该行为，可补显式 `gc.collect()` 与 `torch.cuda.empty_cache()` 于加载末尾）；
- 峰值采样工具：加载期间以 `psutil` + `torch.cuda.memory_allocated()` 采样，峰值入日志/报告（供验收与回归）。

## 5. 验收标准

| 验收项 | 标准 | 方法 |
|---|---|---|
| 峰值上界 | 加载瞬时峰值 ≤ `层段权重 × 1.2 + 常量`（常量 = embedding/head/缓冲，实测标定） | 采样工具记录峰值，与理论层段权重比对 |
| 无全模型峰值 | 峰值 **不** 出现"全模型权重"量级（如 7B fp16 ≈ 14GB） | 峰值采样 + 断言 |
| Surface 7 可承载 | DeepSeek 7B 分 3 节点时，从节点峰值 ≤ Surface 7 可用内存 | 真机（下月从节点）实测 |
| 新架构接入 | 按 §4.2 模板实现，无"全加载截断"代码路径 | 代码评审 + 峰值回归 |

## 6. 测试与回归

- 现有：`test_layer_range` / 分布式层加载测试（Qwen 1.8B 层段加载、embed/lm_head 开关、无效范围 fail-closed）保持通过；
- 新增（P1 后）：峰值采样断言测试（加载过程峰值 ≤ 阈值）；新架构接入时的层过滤等价性测试（过滤加载 vs 全量加载逐层权重一致）；
- 真机：下月从节点（Surface 7）到位后，DeepSeek 7B 三层段分配的峰值实测入档（验收清单 A4 关联）。

## 7. 关联与登记

- 关联：`docs/Gemma-4-12B多模态支持方案.md` §2.4（PyTorch 候选 S2 接入层流水线时按本方案模板）；`docs/验收清单与资源限制登记.md` A4（双机验收含峰值实测）；`docs/张量并行外部辅助与混合拆分调研方案.md`（层间拆分 vs TP 的显存语义区分）。
- 优先级：P0（固化现状）本机即可做；P1 随未来架构接入票执行；P2 可选。
- 本方案不改变任何已验收行为；当前 Qwen 系分布式加载行为不变（已是目标形态）。
