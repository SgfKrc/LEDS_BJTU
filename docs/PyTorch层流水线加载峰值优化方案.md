# PyTorch 层流水线加载峰值优化方案

> 文档生命周期：**现行（Active）**
>
> 创建日期：2026-08-13
> 更新日期：2026-08-14
> 适用范围：分布式层间流水线（layer-wise pipeline）从节点的模型加载瞬时显存/内存峰值治理；直接约束 Surface 7 级轻量从节点在重模型（DeepSeek 7B / gemma-4 12B / Qwen 9B 等）下的可用性
>
> 目标：**加载瞬间峰值 ≈ 分配层段权重 + 常量开销**，与稳态占用同量级，杜绝"全模型峰值压垮从节点"；一期进一步把模型身份/容量发现从权重物化中剥离，向"集群总容量加载"演进。

> **实施复核（2026-08-13）**：P0 已完成，P2 的本机开发基线已完成。`ModelManager.load_layer_range` 现在强制 Qwen/Qwen2 架构加载器返回 `_LayerRangeLoadTracker`；每个 safetensors tensor 在 `get_tensor` 前先校验 key 是否属于分配层段，越界物化、缺失分配层或绕过守卫均 fail-closed。加载过程在 meta 骨架完成、tensor 读取、目标设备安装、裁剪和清理边界采样进程 RSS 与 CUDA allocated，并通过 `get_model_info().layer_load_metrics` 暴露层索引、tensor 数、源/目标字节和峰值增量，不记录文件路径或权重 key。定向回归 `95 passed / 1 skipped`，Python 全量 `2105 passed / 8 skipped`。本机开发门不替代 Surface 7、真实 7B/12B 与多机验收。

---

## 1. 背景与问题

- 从节点形态：Surface 7 级轻量设备（RAM/显存有限），是分布式的主要物理从节点形态。
- 担忧：重模型（≥7B）若在加载瞬间出现**全模型权重峰值**（如 fp16 7B ≈ 14GB、fp16 12B ≈ 24GB），会直接压垮从节点——即使稳态只占层段。
- 项目现有契约：`EngineHost.load_layer_range` / `ModelManager.load_layer_range` 按主节点下发的 `layer_assignments` 只加载对应层段（见 `scheduler.compute_layer_assignment` + `_check_node_vram_for_layers`）。

## 2. 现状实现（2026-08-13 代码实测）

| 路径 | 实现 | 瞬时峰值 |
|---|---|---|
| **Qwen / Qwen2 系**（含 DeepSeek-R1-Distill-Qwen，`model_type=qwen2`） | `_load_qwen2_layer_range` / `_load_qwen_layer_range`：`safetensors.safe_open` **按 key 过滤**；`_LayerRangeLoadTracker` 在读取前强制校验并在结束时核对实际层集合 | ✅ 代码路径只允许分配层段物化；真实峰值上界待目标硬件标定 |
| Embedding / LM Head | 按 `has_embedding` / `has_lm_head` 决定保留；过滤路径下未分配端不物化 | ✅ 常量级 |
| 加载后裁剪 | `ModuleList(kept)` + `del` 释放，并断言裁剪后层数等于 assignment | 防御性二次保险（过滤路径下未分配层仍为 meta） |
| 加载观测 | tensor/materialization 边界采样 RSS + CUDA allocated；统计源权重字节与目标 dtype 字节 | ✅ 本机可回归；不宣称捕获两个边界间的瞬时尖峰 |
| **其他架构**（`gemma4_unified`、`qwen3`、`qwen3_vl` 等） | `load_layer_range` fail-closed，并明确指向 §4.2；不能回退整模加载 | 不适用（尚未接入） |

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
- **状态：Completed（2026-08-13）**。
- 将 `_load_qwen2_layer_range` 的"按层 key 过滤物化"确认为**层流水线唯一加载模式**，写入本方案为架构契约：
  - 任何新架构的 `load_layer_range` 实现必须从 safetensors 按 key 过滤，**禁止**"全加载后截断"路径。
  - 现有防御性裁剪代码保留（不删除，作为兜底断言：过滤后实际物化层数 == 分配层数）。
- `load_layer_range` 要求加载器返回有效 `_LayerRangeLoadTracker`；未返回即拒绝进入后置裁剪。
- 在 `load_layer_range` 的 `else` 分支错误消息中补充指引：新架构接入须按本方案 §4.2 模板实现。

### 4.2 新架构接入模板（P1，未来架构必用）
**状态：Shared contract ready / architecture adapters pending。** 通用 key 守卫、字节统计、层集合断言和观测字段已经落地；Gemma 4/Qwen3 等具体 prefix、模型骨架与量化适配仍必须随各自架构票实现，不能把“通用守卫完成”登记成架构已支持。

1. 解析 config 的 `num_hidden_layers`，校验与 assignment 一致（现有逻辑复用）；
2. 确定层 key 前缀（`model.layers.` / `transformer.h.` / `model.h.`——按架构）；
3. `safe_open` 过滤：仅物化 `{prefix}{i}.`（i ∈ [start, end)）+ 按 `has_embedding`/`has_lm_head` 过滤首尾 key；
4. 构建模型骨架（`AutoConfig` + 空模型或 meta device），用过滤后的 state_dict 加载到目标设备；
5. 量化（若启用）在层段内逐层执行，中间张量即时释放；
6. 加载后断言：`model.{layers}.__len__() == end - start`，失败 fail-closed。

### 4.3 加载缓冲治理（P2，可选增强）
- **状态：Development baseline completed / hardware validation pending（2026-08-13）**。
- safetensors `mmap` 按 tensor 读取，安装到目标模块后立即 `del tensor`；加载结束已有 `gc.collect()` 与 `torch.cuda.empty_cache()`。
- 在 meta 骨架完成、每次 tensor 读取与目标设备安装、模型裁剪和最终清理边界，以 `psutil` + `torch.cuda.memory_allocated()` 采样；指标直接进入模型信息，不另生成报告文档。
- 边界采样用于发现整模量级回归和支撑真机标定，但不等同高频 profiler；两个采样点之间的极短瞬时尖峰属于最终验收残余风险。

## 5. 验收标准

| 验收项 | 标准 | 方法 |
|---|---|---|
| 峰值上界 | 加载瞬时峰值 ≤ `层段权重 × 1.2 + 常量`（常量 = embedding/head/缓冲，实测标定） | 采样工具记录峰值，与理论层段权重比对 |
| 无全模型峰值 | 峰值 **不** 出现"全模型权重"量级（如 7B fp16 ≈ 14GB） | 峰值采样 + 断言 |
| Surface 7 可承载 | DeepSeek 7B 分 3 节点时，从节点峰值 ≤ Surface 7 可用内存 | 真机（下月从节点）实测 |
| 新架构接入 | 按 §4.2 模板实现，无"全加载截断"代码路径 | 代码评审 + 峰值回归 |

开发门与验收门分开处理：P0/P2 代码与合成模型测试完成后可以继续其他开发；上述真实模型、目标硬件和多机结论保持 `Validating`，不阻塞工作流，也不得提前标为通过。

## 6. 测试与回归

- 已完成：`tests/test_model_module.py` 共 `95 passed / 1 skipped`，相邻模型 API/分布式前向 `91 passed`、scheduler `268 passed`，Python 全量 `2105 passed / 8 skipped`；覆盖真实 tiny Qwen2 safetensors 选择性加载、双层段与全模型 logits 等价、守卫在 `get_tensor` 前拒绝未分配 key、缺层 fail-closed、加载器不可绕过守卫、未知架构接入指引、指标结构和源/目标 dtype 字节核算。
- 新架构接入时必须新增：该架构的 key 过滤等价性测试（过滤加载 vs 全量加载逐层权重一致）及分段前向等价测试。
- 峰值数值阈值不能用 tiny 模型或当前开发机常量代替；真实工件到位后按 3 轮标定冻结常量，再启用 `层段权重 × 1.2 + 常量` 自动判定。
- 真机：下月从节点（Surface 7）到位后，DeepSeek 7B 三层段分配的峰值实测入档（验收清单 A4 关联）。

## 7. 关联与登记

- 关联：`docs/Gemma-4-12B多模态支持方案.md` §2.4（PyTorch 候选 S2 接入层流水线时按本方案模板）；`docs/验收清单与资源限制登记.md` A4（双机验收含峰值实测）；`docs/张量并行外部辅助与混合拆分调研方案.md`（层间拆分 vs TP 的显存语义区分）。
- 优先级：P0 已完成；P2 本机开发基线已完成；下一张相关开发票随首个 PyTorch 新架构选择执行 P1 adapter，硬件到位后再执行三轮峰值标定与阈值冻结。
- 本方案不改变任何已验收行为；当前 Qwen 系分布式加载行为不变（已是目标形态）。

## 8. 从“单机先整模”迁移到“集群总容量加载”的分期计划

### 8.1 目标不变量与本机工件核验

最终目标不是“主节点勉强能先装下模型，再把层切给其他节点”，而是：**任一节点都不需要先物化完整模型；只要通过准入的节点集合对各组件的可用 RAM/VRAM 总量足够，集群就能从冷态按最终分工加载。** 控制面只允许读取配置、索引、文件头和摘要，不得以 `from_pretrained`、完整 `state_dict` 或临时整模实例换取模型层数和容量信息。

2026-08-13 对现有 PyTorch/Safetensors 工件的核验如下。数值来自 Safetensors 文件头，不读取 tensor 内容：

| 本地工件 | 架构 | 解码层 | 权重字节 | 组件拆分 | 当前流水线执行状态 |
|---|---:|---:|---:|---|---|
| `models/qwen3-4b` | `qwen3` | 36 | 8,044,936,192 | 解码层 7,267,018,752；Embedding 777,912,320；Norm 5,120 | 可发现；adapter 未实现，拒绝准入 |
| `models/qwen3-vl-4b-instruct` | `qwen3_vl` | 36 | 8,875,631,616 | 文本同上；视觉塔 830,695,424 | 可发现；文本/视觉执行器未实现，拒绝准入 |
| `models/qwen3-5-2b` | `qwen3_5` | 24 | 4,548,144,832 | 解码层 2,746,532,544；Embedding 1,017,118,720；视觉塔 662,833,152；MTP 121,656,320 | 可发现；混合注意力/视觉/MTP 执行器未实现，拒绝准入 |

这三套都是官方 PyTorch/Safetensors 工件，不是 GGUF。主运行时 `transformers==4.47.x` 仍不能直接识别这些架构；Qwen3 sidecar 当前只有 config/tokenizer 预检能力，没有 torch/accelerate 分层执行环境。因此“已下载 PyTorch 工件”与“已能参与分布式 PyTorch 流水线”必须继续分开登记。

### 8.2 分期与依赖

| 阶段 | 开发票 | 内容 | 完成门 |
|---|---|---|---|
| **一期：清单优先冷启动控制面** | `PT-PIPE-C1` | 不加载权重地解析 config、Safetensors 索引与文件头；形成逐层/Embedding/LM Head/视觉/MTP 字节账本；增加显式 distributed-only 准备状态；调度器不再要求主节点 `is_loaded` 才能取得模型身份、层数和摘要 | 已准备状态下 `model is None`、`tokenizer is None`、`is_loaded=false`；Qwen/Qwen2 可下发；Qwen3 系仅发现且 fail-closed |
| **二期：集群总容量准入与两阶段提交** | `PT-PIPE-C2` | 以逐层真实字节、运行 dtype、常量开销和安全余量计算每节点容量；先生成完整可执行分配，再提交 worker；所有节点 ACK 后主节点只加载自身层段；任一节点不足则整轮不物化。取消“主节点至少一层”的硬假设，允许独立控制面/Tokenizer 节点 | 本机 RAM/VRAM 可低于整模，集群容量足够时仍可形成方案；失败时所有节点保持未加载或释放；无半提交代际 |
| **三期：Qwen3 文本分层执行器** | `PT-PIPE-QW3` | 在隔离运行时补齐 torch/accelerate/safetensors；实现 `model.layers.N` 按 key 物化、分段前向、KV cache、LM Head 与 `enable_thinking=false` tokenizer 契约；主运行时 4.47.x 不升级 | Qwen3-4B 两段/三段 logits 等价；无完整权重峰值；sidecar 断开时 fail-closed |
| **四期：按需工件同步** | `PT-PIPE-C3` | 将现有“每个 worker 先镜像全部 Safetensors”改为 assignment manifest；只同步本节点需要的 shard/重打包 layer bundle，并对文件、key 集和模型 revision 分层校验 | RAM/VRAM 与磁盘/网络都不再要求每节点持有整模；重连和 Range 续传不改变模型代际 |
| **五期：多模态组件编排** | `PT-PIPE-MM1` | 为 Qwen3-VL/Qwen3.5 将视觉塔、文本层、Embedding、LM Head、MTP 作为不同可放置组件；固定跨节点张量契约和视觉特征生命周期 | 视觉组件可以与文本层分置；纯文本请求不错误加载视觉塔；Qwen3.5 混合层按类型执行 |
| **六期：真机标定与默认启用** | `PT-PIPE-V1` | 在多台 CUDA PC/Surface 目标节点执行三轮冷启动、峰值、掉线、重连和结果等价验收；冻结安全余量 | 通过 §5 全部门；三轮稳定后才允许 UI 默认推荐 distributed-only 模式 |

顺序约束：`C2` 先解决架构无关的总容量事务，`QW3` 再接入新执行器；`C3` 解决磁盘和传输冗余，不能用它替代内存准入；多模态必须在文本 Qwen3 契约稳定后开始。没有外部真机不阻塞 C2/QW3/C3 的合成开发，但 V1 结论必须延后。

### 8.3 一期实施记录（`PT-PIPE-C1`，Completed，2026-08-13）

- 新增 `src/pipeline_model_descriptor.py`：仅读取 `config.json`、`model.safetensors.index.json` 和 `safe_open.get_slice()` 文件头；禁止 `get_tensor`，输出逐层和组件字节、架构、层前缀、工件总字节及执行器准入结果。
- `ModelManager.prepare_pipeline_model()` 新增显式 distributed-only 状态：在摘要和描述器验证成功后记录模型身份，但保持 `model/tokenizer=None`、`is_loaded=false`；普通卸载和普通模型切换均能清理该状态。
- 新增 `POST /api/models/prepare-pipeline`。它不把元数据准备伪装成完整模型加载；响应明确区分 `loaded=false` 与 `pipeline_prepared=true`。
- Scheduler 的层数、容量估算、模型身份和模型摘要可以来自描述器，不再要求主节点已经整模加载；现有 Qwen/Qwen2 下发白名单不变。distributed-only 请求只允许流水线执行，worker 未就绪或运行失败时禁止自动整模回退。
- Qwen3、Qwen3-VL、Qwen3.5 已登记元数据布局并通过真实本地工件检查，但 `pipeline_runtime_supported=false`；一期 API 不允许把它们准备成可执行流水线，避免“能识别 config”被误报成“能运行模型”。
- 回归：描述器 + 模型管理器 `99 passed / 1 skipped`；Scheduler `270 passed`；Python 全量 `2105 passed / 8 skipped`（约 5 分 43 秒）。真实多机、峰值与集群总容量验收继续标为 `Validating`。

### 8.4 二期实施记录（`PT-PIPE-C2`，开发门 Completed，2026-08-13）

- 新增 `src/pipeline_capacity.py` 纯函数求解器：消费描述器逐层字节、Embedding/Final Norm/LM Head 字节和节点当前可用容量，不打开权重、不导入 torch；每个参与节点收取运行时 reserve，CUDA 按 FP16/源字节，CPU 按 FP32 扩张，统一套用可配置安全余量 `QLH_PIPELINE_CAPACITY_SAFETY_MARGIN`（默认 `1.2`）。Final Norm 按每个实际层段节点计费，LM Head 只归属末节点。
- 求解结果是 all-or-nothing：连续覆盖 `[0,total_layers)`、Embedding/LM Head 各有唯一归属才返回 `admitted=true`；允许 master 没有层而作为控制面/Tokenizer 节点，返回 `control_only_nodes`。单机均不足但总容量足够时返回 `aggregate_only=true`；不足、容量未知、视觉/MTP 等未实现组件均返回结构化 `reason_code` 和空 assignment。
- Scheduler 新增 `get_pipeline_capacity_plan()` 与 `GET /api/cluster/pipeline-capacity`。规划使用实时 `gpu.vram_free_gb` 或 `ram.available_gb`，没有空闲容量证据时保守拒绝，不把物理总显存/RAM 冒充可用容量；`/api/cluster/layers` 在事务进行中复用同一份 capacity assignment。
- distributed-only 模式下，下发固定 `config_id + plan_id` 的 `phase=prepare`。worker 仅做本地摘要、描述器、层范围和容量复核，返回 `prepared`，不调用 `load_layer_range`；全部 prepared ACK 后主节点只加载自己的层段/Tokenizer 元数据，再下发 `phase=commit`。commit 必须命中同代 prepared 记录，才允许首次物化。
- 任一 prepare/commit error、worker 断连或主节点本地 commit 失败都会进入 abort/release 代际；worker 清理已物化层段但保留 distributed-only 描述器，禁止自动回退到主节点整模加载。事务状态和失败原因可通过容量 API 读取。
- 新增合成回归覆盖：单机都不足但聚合足够、总容量不足、CPU 扩张、主节点零层、全员 prepared 后 commit、阶段错配、worker 掉线和 abort 清理；本票定向 `279 passed` Scheduler、`102 passed / 1 skipped` ModelManager+capacity，API 模型回归 `55 passed`。

本票仍不宣称三项能力：按需 shard/bundle 尚未实现、Qwen3/Gemma4 等新架构执行器尚未准入、真实多机峰值/掉线/结果等价仍需 PT-PIPE-V1 三轮硬件验收。

### 8.5 三期实施记录（`PT-PIPE-C3`，开发门 Completed，2026-08-13）

- 新增 `src/pipeline_assignment_manifest.py`：主节点按 `config_id + plan_id + node_id + layer_range` 生成 assignment manifest，仅包含配置、Tokenizer 支持文件、过滤后的 Safetensors `weight_map` 和被分配层段实际所在 shard；manifest 同时绑定完整模型 revision `model_sha256`，并以 `manifest_sha256` 锁定清单内容。
- 新增 `/api/models/pipeline-assignment/{model_id}`，只接受可信模型 peer，且必须命中当前 `preparing` 事务、活动 `plan_id` 和精确节点/层段/Embedding/LM Head 合同；旧 `/api/models/downloadable` 全量清单接口保留给非事务兼容路径。
- `ensure_pipeline_assignment_available()` 将 worker 缓存放在 `.pipeline_assignments/<config>-<node>/` 代际目录，复用 `.part`、HTTP Range、单文件大小/SHA 校验和原子提交。prepare 阶段通过 assignment manifest 下载，不再要求 worker 先拥有整模目录；commit 继续只使用该代准备目录。
- Safetensors 描述器支持 partial assignment 检查；C3 不允许通过缺层目录冒充完整模型，层段范围、模型类型和 config 层数仍需一致。
- 回归：C3 manifest/model-sync/descriptor `12 passed`；Scheduler + API + 安全边界 + 同步专项 `357 passed`；Python 全量 `2126 passed / 8 skipped`（约 3 分 28 秒）。真实大模型传输、跨节点断点恢复和多机结果等价仍留在后续硬件门。

本票的边界：assignment manifest 目前只准入已经支持的 Qwen/Qwen2 PyTorch 执行器；视觉/MTP 组件仍 fail-closed；worker 目录中暂保留代际 assignment 缓存，空间回收策略和 Range 中断后的真实网络验收后置。

### 8.6 `PT-PIPE-C3.1` 缓存治理与恢复实施记录（开发门 Completed，2026-08-13）

- 新增 `QLH_PIPELINE_ASSIGNMENT_CACHE_MAX_MB`、`QLH_PIPELINE_ASSIGNMENT_MIN_FREE_MB` 和 `QLH_PIPELINE_ASSIGNMENT_STALE_SECONDS` 配置；下载前按 manifest 预计字节数检查磁盘保留空间及 assignment 缓存预算。
- 新增 `reconcile_pipeline_assignment_cache()`：只扫描模型目录下的 `.pipeline_assignments`，保护当前 config/正在下载目录，清理无 marker 或带 `.part` 的废弃代际，并按最旧优先回收超预算缓存；启动 Scheduler 时对当前模型执行一次 reconcile。
- 新增 `remove_pipeline_assignment_cache()`；C2 abort/release 将 `model_id + aborted_config_id + node_id` 传给 worker，失败代际立即清理，不影响旧代或用户完整模型目录。
- 过滤后的 `model.safetensors.index.json` 写入前校验实际大小/SHA；断点下载继续保留 `.part` 和 Range，清单/key 集/文件摘要任何不一致均拒绝提交。
- 回归：model sync/cache `11 passed`；C3 manifest/model-sync/descriptor `12 passed`；Python 全量基线仍为 `2126 passed / 8 skipped`。本机未执行真实大文件跨节点断线，真实网络矩阵与磁盘权限仍需后续外部门。

### 8.7 下一票（`PT-PIPE-QW3`）

1. 在隔离 Transformers sidecar 中实现 Qwen3 文本分层 adapter，保持主运行时 `transformers==4.47.x` 不变。
2. 复用 C1 描述器、C2 容量事务和 C3 assignment manifest，先完成 `enable_thinking=false` 的单机两段合成前向，再进入真实工件 smoke。
3. Qwen3-VL/Qwen3.5 视觉、MTP 和真实多机结果等价继续后置，不能因为文本 adapter 完成而放开多模态准入。

### 8.8 `PT-PIPE-QW3` Qwen3 文本 adapter 开发记录（开发门 Completed，2026-08-13）

- 新增 `scripts/model_tools/qwen3_pipeline_adapter.py` 与隔离 sidecar worker：主运行时 `transformers==4.47.x` 不变，sidecar 只接受 `transformers>=4.51`，并以 `local_files_only=true`、`trust_remote_code=false`、network-disabled、read-only 契约运行。
- adapter 在读取权重前校验 `model.layers.N` assignment key 集合、连续层范围、config 层数和 unknown/视觉/MTP key；真实 loader 使用 meta 骨架 + 过滤 index + `safe_open.get_tensor`，不调用 `from_pretrained` 整模物化，加载后断言非分配层未被物化。
- 已实现分段前向、RoPE/causal mask、官方 Transformers `DynamicCache` 与 legacy cache 兼容、LM Head；`tie_word_embeddings=true` 时末段显式复用 `model.embed_tokens.weight`，容量账本和 assignment manifest 同步计费/下发。
- tokenizer 通过 `apply_chat_template(..., enable_thinking=false)` 硬关闭 thinking；不接受该参数、产生非空 thinking 标记或 sidecar 未隔离时 fail-closed。
- 合成 Qwen3-like 模型已验证两段 logits 等价、KV cache 段内长度和 assignment 守卫；定向回归 `18 passed`。当前 `.venv-qwen3-sidecar` 实测 `transformers 4.57.6` 已就绪但缺 `torch/accelerate`，因此真实 Qwen3-4B 权重 smoke 尚未执行，不能宣称运行时准入。

### 8.9 下一票：`PT-PIPE-QW3.1` 执行 sidecar 与真实工件 smoke

1. 以 `scripts/setup_qwen3_sidecar_env.py --pipeline` 安装平台匹配的 torch/accelerate，并对 `models/qwen3-4b` 生成 C3 assignment manifest；先做单段 CPU/meta 预检，再做可用资源门控下的真实两段/三段前向。
2. 记录 full-model materialization 断言、每段 source/materialized bytes、RSS/VRAM 峰值、prefill/decode KV cache 和 `enable_thinking=false` 最终答案格式；任何资源不足或 sidecar 异常返回结构化 `resource_rejected/sidecar_failed`，不得回退主运行时整模加载。
3. 真实 smoke 通过后再接 scheduler prepare/commit；Qwen3-VL、Qwen3.5、跨节点结果等价和 PT-PIPE-V1 三轮硬件标定仍不在本票范围。

### 8.10 `PT-PIPE-QW3.1` 执行 sidecar 资源门开发记录（开发门 Completed，2026-08-13）

- 新增 `scripts/model_tools/qwen3_pipeline_smoke.py` 和隔离 `qwen3_pipeline_smoke_worker.py`，CLI 入口为 `qwen3-pipeline-smoke`。默认只读预检；只有显式 `--execute` 且资源/依赖门均通过时才允许调用 Qwen3 filtered-assignment loader，不会把 full index 当作 assignment 绕过。
- worker 使用 Safetensors `framework=np` header 路径计算所选层段的 tensor 数量、源字节、shard 数和 CPU/CUDA 预算；即使 sidecar 尚无 Torch，也能先完成资源门。资源不足返回 `resource_rejected`，执行依赖缺失返回 `runtime_unavailable`，所有结果均为结构化、read-only、network-disabled。
- 当前 `models/qwen3-4b` 实测 `layer_range=[0,1] + embedding` 选中 13 个 tensor、`979,779,072` bytes、2 个 shard；本机可用 RAM 约 `4,177,612,800` bytes，CPU 预算约 `2,888,340,685` bytes，资源门通过。隔离 venv 为 `transformers 4.57.6`，但缺 `torch/accelerate`，故本次真实物化未执行并正确返回 `runtime_unavailable`；没有整模 fallback。
- 新增回归覆盖依赖缺失仍能计量 assignment、容量不足先于依赖门拒绝、控制器结构化转发和 CLI 退出码；QW3 smoke/adapter/sidecar 专项 `13 passed`，`py_compile` 与 `git diff --check` 通过。全量 Python 回归 `2140 passed / 8 skipped`（约 218 秒）。

### 8.11 下一票：`PT-PIPE-QW3.2` sidecar 执行依赖与真实 assignment smoke

1. 仅在用户明确允许且存在平台匹配 wheel 时，通过 `scripts/setup_qwen3_sidecar_env.py --pipeline` 安装 Torch/Accelerate；主运行时 `transformers==4.47.x` 不变，禁止 sidecar 自动联网升级或回退主进程。
2. 复用 C3 assignment manifest 生成的 filtered index，在资源门通过后执行 Qwen3-4B 单段 CPU/CUDA prefill，再扩展两段/三段 decode、KV cache、峰值和 `enable_thinking=false` 格式门；完整索引、缺层、未知 key 或峰值超限继续 fail-closed。
3. 真实 smoke 通过后才接 Scheduler prepare/commit 和跨节点传输；Qwen3-VL/Qwen3.5、跨节点结果等价与 PT-PIPE-V1 三轮硬件标定继续后置。

### 8.12 `PT-PIPE-QW3.2` 真实 sidecar assignment smoke（开发门 Completed，2026-08-14）

- `.venv-qwen3-sidecar` 已按隔离原则补齐 `torch 2.13.0+cu126`、`accelerate 1.14.0`、`psutil 7.2.2`；主运行时仍为 `transformers 4.47.1`，未被升级。`setup_qwen3_sidecar_env.py` 新增 `--torch-index-url`、`--torch-spec` 及 `QLH_QWEN3_TORCH_INDEX_URL/QLH_QWEN3_TORCH_SPEC`，避免 `--pipeline` 无意安装 CPU Torch。
- worker 在执行前创建临时 filtered assignment：只复制 config/index，权重 shard 优先硬链接，跨卷才复制；执行结束自动清理，不修改 `models/qwen3-4b`，不把 full index 交给 loader。
- 真实 CPU 首段 `layer_range=[0,1] + embedding` 通过：13 tensor、`979,779,072` source bytes、FP32 materialized `1,959,558,144` bytes、RSS delta 约 `1.98 GB`，prefill 18 tokens + decode 1 token，KV cache 长度 `19`，`thinking_disabled=true`，`full_model_materialized=false`。
- 真实 CPU 末段 `[35,36] + lm_head` 通过，tied embedding 的 logits 路径和 BF16 物化通过：source `979,779,072` bytes、materialized `979,779,072` bytes、`full_model_materialized=false`。
- 真实 CUDA 中间段 `[1,2]` 通过：12 tensor、source/materialized `201,866,752` bytes，RTX 4060 Laptop 上 CUDA peak allocated `212,327,936`、reserved `216,006,656` bytes，未触发整模加载。首段 CUDA 在可用 RAM 约 `1.41 GB` 时因预算约 `1.71 GB` 返回 `resource_rejected`，没有物化；这是预期资源门行为。
- 回归新增 tokenizer/真实 staging/KV/RSS-CUDA 峰值/缓存长度和安装参数覆盖；QW3 smoke + adapter `9 passed`，真实命令均通过或按资源结构化拒绝。QW3.2 仍不宣称跨节点结果等价或生产准入。

### 8.13 下一票：`PT-PIPE-QW3.3` 两段/三段串联与跨节点 KV 合同

1. 复用 C3 assignment manifest，在同一隔离 worker 内加载首段和末段/中段 filtered assignment，验证 hidden-state handoff、两段/三段 logits 与单模型参考的等价性；继续断言每段 `full_model_materialized=false`。
2. 为跨节点协议固定 KV cache 的段索引、序列长度、dtype/device、prefill/decode 代际和失效清理字段；覆盖缓存缺失、长度不一致、跨段设备不匹配和中途 abort 的结构化拒绝。
3. 只有串联 smoke 通过后才接 Scheduler prepare/commit；真实双机传输、断线恢复、Qwen3-VL/Qwen3.5 和 PT-PIPE-V1 三轮硬件标定继续后置。

### 8.14 `PT-PIPE-QW3.3` 两段/三段串联与 KV 合同（开发门 Completed，2026-08-14）

- 新增 `qwen3_pipeline_chain.py` 与 `qwen3_pipeline_chain_smoke.py`。链路只接受 2 或 3 个连续段，首段唯一拥有 Embedding，末段唯一拥有 LM Head；段间缺层、重叠、空洞、组件归属错误均在物化前拒绝。
- 每段先独立通过 Safetensors header/本机 RAM-CUDA 预算门，再在隔离 sidecar 内按 `prefill -> release -> decode` 串行加载。权重段用临时 filtered assignment，硬链接优先，阶段结束清理；6 次真实段加载均报告 `full_model_materialized=false`。
- 新增 hidden handoff 合同：`chain_id`、段索引、`[batch, sequence, hidden]`、序列长度、dtype、device；新增 KV 合同：段层范围、batch、序列长度、prefill/decode phase、generation 和设备类型。字段不一致、缓存长度不符或跨段设备不匹配 fail-closed，不传输 prompt/权重/整模 KV。
- 真实 `models/qwen3-4b` 三段 CPU smoke：`[0,12]+embedding / [12,24] / [24,36]+lm_head`，tokenizer `enable_thinking=false`，prefill `18` tokens，decode 后每段 KV `19` tokens；两个 hidden handoff 均 `[1,18,2560]`，prefill/decode logits shape 分别 `[1,18,151936]` / `[1,1,151936]`，执行与清理通过。
- QW3 chain/adapter/smoke/sidecar 专项 `18 passed`，`py_compile` 和 `git diff --check` 通过；本票仍不宣称跨节点传输等价、Scheduler 已接入或生产准入。

### 8.15 下一票：`PT-PIPE-QW3.4` Scheduler prepare/commit 协议模拟与跨节点故障矩阵

1. 将 chain plan、filtered assignment revision、hidden/KV contract 摘要接入现有 C2 `prepare -> prepared ACK -> commit -> ready` 事务，但先保持 dry-run/仿真，不触发真实远端模型加载。
2. 增加跨节点合同签名、序列化上限、缓存代际和 abort/release 清理矩阵；覆盖缺段、重复 ACK、合同篡改、KV 长度/设备不一致、超时和断线重试，全部结构化拒绝且不整模回退。
3. 协议模拟稳定后再做单机 loopback HTTP/Range 传输；真实双机、异构设备结果等价、Qwen3-VL/Qwen3.5 和 PT-PIPE-V1 三轮标定继续后置。
