# Qwen3.5 与 Qwen3-VL 小模型支持计划

> 状态：规划（调研与实施均未开始）
>
> 更新日期：2026-08-13
> 适用范围：为本项目补充 Qwen3.5 系列（0.8B / 2B / 4B / 9B）与 Qwen3-VL 系列（4B / 8B）的模型支持与实验接入；文本模型经既有 GGUF/Safetensors 引擎路径，VL 模型经 Ollama `external_api` 多模态路径（与 Gemma 4 同构）
>
> 目标：丰富项目与实验的模型梯度（0.8B→9B 文本 + 4B/8B 多模态），并借「非 R1 系 + 推理能力较强」的特性评估恢复 EX-N3 LLM 客观判题正确率口径

---

## 1. 背景与动机

- **模型梯度缺失**：当前注册表与实验基线只有 Qwen 1.8B（能力弱）与 DeepSeek-R1-Distill-7B（thinking 不可关闭）。项目缺少「架构新、推理能力尚可、无 thinking 污染」的中间梯度小模型。
- **架构较新**：Qwen3.5 / Qwen3-VL 与 Gemma 4 同属新架构代次，接入经验可相互复用（Ollama 兼容层、GGUF 量化、多模态判题链路）。
- **多模态补强**：Qwen3-VL 4B/8B 提供图像理解能力，可与 Gemma 4 判题构成**双模型对照**，缓解单一判题器偏差。
- **EX-N3 判题口径联动**（2026-08-13 方案 1 决策的升级通道）：LLM 侧客观正确率判题因本地模型不可用而暂停；Qwen3.5 非 R1 系且推理能力较强，**若其输出满足 `normalized_contains` 判题口径，可恢复正确率作为 LLM 侧质量判据**（§6.2.4 已预留该通道）。
- 补充：这些模型均非 R1 系列，不引入 thinking 污染问题。

## 2. 模型清单

| 模型 | 参数量 | 模态 | 预期推理形态 | 备注 |
|------|--------|------|-------------|------|
| Qwen3.5-0.8B | 0.8B | 文本 | GGUF CPU / Android | 最小梯度，LLM 侧轻量判题候选 |
| Qwen3.5-2B | 2B | 文本 | GGUF CPU / Android | 移动端本地推理梯度 |
| Qwen3.5-4B | 4B | 文本 | GGUF CPU / Safetensors CUDA | 单机主流梯度 |
| Qwen3.5-9B | 9B | 文本 | Safetensors CUDA / GGUF CPU | 本机上限梯度（接近 gemma4 12B 的资源量级） |
| Qwen3-VL-4B | 4B | 文本 + 图像 | Ollama `external_api` | 多模态判题 / 图像理解实验 |
| Qwen3-VL-8B | 8B | 文本 + 图像 | Ollama `external_api`（本机 VRAM 门内） | 多模态判题对照主力候选 |

> 工件来源、精确版本（revision/tag）、许可证与量化方案以 **S0 调研**结论为准，未冻结前不写入注册表。

## 3. 支持路径（复用现有能力）

| 环节 | 文本系列（Qwen3.5） | VL 系列（Qwen3-VL） |
|------|--------------------|--------------------|
| 模型注册 | `model_registry` 新增条目（现有 Qwen/DeepSeek 槽位模式） | 同左 |
| CPU/集显/Android | GGUF（llama.cpp / llama-cpp-python，复用 Qwen 1.8B GGUF 路径） | 不适用（VL 需视觉编码） |
| CUDA/分布式 | Safetensors（PyTorch 路径，复用 Qwen 1.8B 分布管线） | Ollama 官方工件（`external_api`，复用 Gemma 4 路径） |
| 多模态判题 | 不适用 | `experiment_gemma_judge_real.py` 判题器复用（模型名/契约参数化，仅非 R1 系） |
| 实验执行器 | `experiment_llm_quality_unit.py`（GGUF 执行器已支持任意本地工件 + SHA 锁） | Gemma 判题单元模式（Ollama 端点） |
| MODEL-TOOLS | `llm_smoke_matrix`、`gguf_convert`（已有注册表模型条目即自动覆盖） | 同左（Ollama 冒烟另行接线） |

## 4. 实验价值

1. **LLM 客观判题恢复**：Qwen3.5-4B/9B 重跑 `ps-v1` 客观子集三轮标定；若正确率 ≥ 基线，按 §6.2.5 重标并将 LLM 侧判据升级为「正确率 + 格式率」（替代方案 1 的格式率单判据）。
2. **多模态判题对照**：Qwen3-VL-8B 与 Gemma 4 对同一 SD 输出集判题，比较 topic/coverage 分布，评估判题器一致性（为质量 gate 引入双判题器投票或交叉验证提供数据）。
3. **模型对比矩阵**：0.8B/2B/4B/9B 文本梯度 + 4B/8B VL，可填充 EX 实验的「规模-能力」曲线（现有 Qwen 1.8B 与 DeepSeek 7B 两个数据点之外）。

## 5. 阶段划分

### S0 调研与基准（只读，无实施）
- 确认 Qwen3.5 / Qwen3-VL 官方工件来源、revision/tag、许可证（Apache-2.0 预期）、量化方案（Q4_K_M 兼容性）
- Ollama 兼容性验证（VL 系列模型名、`reasoning_effort` 支持、图像输入口径——**不启动模型**，仅查官方注册表/文档）
- 资源评估：各模型 Q4 内存/显存占用 vs 本机 16GB RAM / 8GB VRAM 门
- 产出：S0 调研结论 + 资源准入表

### S1 Qwen3.5 文本系列接入（0.8B / 2B / 4B / 9B）
- 注册表条目 + 配置（上下文、量化、引擎路由）
- GGUF 路径：`llm_smoke_matrix` 冒烟（复用既有 4 项 marker）
- Safetensors 路径：单机 CUDA 加载 + 分布管线接线（如资源允许）
- 验收：注册表 4 条目、冒烟 `execution_gate=true`、无回归

### S2 Qwen3-VL 系列接入（4B / 8B）
- Ollama 官方工件拉取（子进程代理，复用 Gemma 4 拉取流程）
- `external_api` 图像理解链路（复用 Gemma 4 的 `_describe_image` 路径，模型名参数化）
- 图像发送/回复的 UI 与 TUI 交互（如 Gemma 4 G4.2 已提供通用交互，则仅注册模型）
- 验收：VL 8B 实图冒烟通过、判题器可调用

### S3 实验接入（EX-N3 联动）
- Qwen3.5-4B/9B 跑 `ps-v1` 客观子集三轮标定 → `experiment_quality_calibrate.py` 汇总
- 若正确率达标：修订 §6.2.3 阈值表 + 升级 §6.2.4 LLM 侧判据（正确率恢复），以 plan 变更记录
- Qwen3-VL-8B 与 Gemma 4 双判题器对照（同一 SD 输出集，比较 topic/coverage 分布）
- 验收：LLM 侧判据升级或明确结论；双判题器对照报告入档

## 6. 资源评估（初步，S0 确认）

| 模型 | Q4_K_M 预估内存 | 本机（16GB RAM / 8GB VRAM）可行性 |
|------|----------------|----------------------------------|
| 0.8B | ~0.6 GB | ✅ CPU 无压力 |
| 2B | ~1.4 GB | ✅ CPU / Android |
| 4B | ~2.8 GB | ✅ CPU；CUDA 亦可 |
| 9B | ~6 GB | ⚠️ CPU 可跑（与 gemma4 12B 同量级，串行不并行）；CUDA Q4 在 8GB 门内 |
| Qwen3-VL-4B | ~3 GB + 视觉编码 | ✅ Ollama（gemma4 已证明路径） |
| Qwen3-VL-8B | ~6 GB + 视觉编码 | ⚠️ 需 Ollama 内存门评估（gemma4 12B 7.5GB 已近上限，VL 8B 需实测或换机） |

> 与 gemma4 12B **串行使用**（同一 Ollama 服务进程），不并行驻留；实测内存门在 S0/S2 确认。

## 7. 风险与决策点

| 风险/决策 | 说明 | 缓解 |
|-----------|------|------|
| 工件来源与许可证 | Qwen3.5/Qwen3-VL 官方工件与模型名未冻结 | S0 只读调研先行，不冻结不入库 |
| Ollama 兼容性 | VL 模型名、`reasoning_effort` 支持未验证 | S0 查官方注册表；不启动模型 |
| 资源门 | VL-8B 可能超本机内存门 | 以 gemma4 12B 实测为基准评估；不行则 4B 先行 |
| LLM 判题恢复失败 | Qwen3.5 正确率若仍不达标 | 结论登记，维持方案 1 判据（不强行升级） |
| 判题器偏差 | 双判题器不一致时以何为准 | 对照报告数据说话；人工复核兜底（既有通道） |

## 8. 验收标准（阶段门）

- **S0**：调研结论 + 资源准入表入档（doc 内更新本计划状态）
- **S1**：注册表 4 条目、`llm_smoke_matrix` 执行门通过、定向回归无退化
- **S2**：VL 8B（或 4B 兜底）实图冒烟 + 判题器参数化调用通过
- **S3**：LLM 侧判据升级或明确结论 + 双判题器对照报告；相关阈值变更随 plan 记录

## 9. 与现有工作的关系

- 复用：Gemma 4 多模态路径（Ollama/external_api/判题器）、MODEL-TOOLS 注册表与冒烟矩阵、EX-N3 质量链路（plan/rubric/calibrate）
- 不冲突：与 gemma-4 原生绑定开发（G4.3）并行无脏读写（不同文件域；本计划 S0/S1 只读为主）
- 联动：EX-N3 方案 1 的「恢复正确率判题」通道（§6.2.4 预留）
