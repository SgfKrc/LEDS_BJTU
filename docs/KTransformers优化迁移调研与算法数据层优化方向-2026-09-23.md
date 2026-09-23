# KTransformers 算子级优化调研 + QLH 算法/数据层优化方向

> 状态：**现行（调研报告）**
>
> 更新日期：2026-09-23
>
> 结论摘要见 §1；可行性建议见 §6；联合排期见 §7；**待裁决的口径冲突**见 §8。

---

## 1. 结论摘要

**为什么做这份调研**：QLH 引擎层的**工程优化已经踩到边际效应** ——

- 同机实测：`llama.cpp + CUDA`（全层上 GPU）**26.6 ms/token**，而跨框架接力最优（torch GPU 上游 N=20 + llama.cpp CPU 下游）
  **129.1 ms/token** ⇒ 慢 **4.85×**（`README.md` §「与「全 llama + CUDA」的公平对照」）；
- 切点扫描：CPU 上游条件下总时间对切点几乎不敏感（N∈{4,8,12,16} 落在 214–225 ms/步，**±2.6%**），不存在中间谷底；
- 分段剖析：**99.3% 是两侧实际计算**，通信 + 同步 + batch 管理仅 **0.34%** ⇒ 优化必须打计算，**不打传输**。

所以下一步要看**算法层 / 数据层**的空间，而 KTransformers（下称 KT）以「算子级优化」著称，需要把它到底做了什么、哪些能迁移过来搞清楚。

**三条核心结论**：

1. **KT 不是、也不会成为 QLH 的第三个正式引擎**。它只作为公开研究参照，用来指导 QLH 自己的 **PyTorch 上游引擎**做算子、数据布局和异构调度实验；llama.cpp/GGUF 仍是稳定主线和无 CUDA/Edge 回退。
2. **KT 的价值是"CPU 侧算子与设备分工仍有数量级空间"的证据，而不是可复制的软件** —— 它的主要收益来自 MoE 专家下放 CPU + hot/cold 放置 + 专家调度；QLH 当前正式登记模型以 **dense ≤2B** 为主，因此先研究可迁移的 dense PyTorch 机制，MoE 仅作为科研扩展。
3. **最该做的一件事反而最便宜**：KT 相对 llama.cpp 的 27.79× 是**它自研 AMX kernel vs llama.cpp CPU 实现**的差距 —— 而同一份 AMX/AVX-512 实现**已经在主仓 vendor 的 llama.cpp 里**（`ggml/src/ggml-cpu/amx/mmq.cpp`、`ggml/src/CMakeLists.txt:387-401`）。所以**不要重写 llama.cpp kernel，要核实 QLH 实际构建是否打开这些变体**。
4. **KT 的算法级近似与 QLH 的默认判据直接冲突**：Expert Deferral、选择性专家激活可以作为科研对照，但不能绕过 QLH 的**逐 token argmax 一致、fail-closed**生产门。

**本报告的正式定位**：研究对象是 `model_module.py` 及其后续拆分后的 PyTorch 上游组件；研究结果必须能与未优化 PyTorch、整模 llama.cpp 和连续层分布式三组基线比较。研究代码受 feature gate 控制，不进入 Koakuma backend 枚举，不进入 Edge 最小发行，也不让 KTransformers 依赖进入主仓运行时。

---

## 2. KTransformers 的核心思路与架构分工

**核心命题**：MoE 模型「激活稀疏 + 权重巨大」⇒ 把**稀疏激活的专家**放在**大容量 DRAM 的 CPU** 上算，把**密集、计算密集的 attention / shared-expert / 路由**放在**高带宽的 GPU** 上算。论文摘要明确其适用域是 **low-concurrency** 场景。

| 层 | KT 用什么 |
| --- | --- |
| 模型加载 / 算子替换 | 基于 **Transformers/HF 权重**，用 YAML `optimize_rules` 做 **match/replace 注入**（如 `model.layers.*.mlp.experts` → `KTransformersExperts`），逐算子指定 `prefill_device: cuda` / `generate_device: cpu`；QLH 只借鉴这种声明式边界，不引入 KT loader |
| CPU 侧算子 | 自研 **AMX / AVX-512 / AVX2** MoE kernel；`LLAMAFILE` 后端直接吃 **GGUF** 权重 |
| GPU 侧算子 | PyTorch/CUDA + **FlashInfer**（含 variable-batch **CUDA Graph**）、Triton MLA、FP8/GPTQ/Marlin |
| 服务 / 调度 | 自研 `balance_serve`（C++ continuous batching + chunked prefill）→ **2025-10 起改为集成 SGLang**（`sglang-kt`） |
| 多 GPU 扩展 | 交给 **SGLang** 做 TP；KT 聚焦「单机 CPU+GPU 协作」 |
| Fine-tune | LLaMA-Factory 集成（`ktransformers[sft]`） |

**硬前提**：PyPI wheel 仅 **Linux x86-64**（AVX2 起）；CUDA 需 SM 8.0+；**AMX 需 Sapphire Rapids+**；
`kt-cpuinfer` = **物理核数（非超线程）**、`kt-threadpool-count` = NUMA 节点数。
**Windows 原生支持在 2026 Q2 roadmap 里仍是待做项**（issue #1921）。

---

## 3. 「算子级优化」逐项定性

把 KT 的手段按**层次**拆开很重要 —— 因为只有一部分与 QLH 的场景相容：

| 手段 | 内容 | 层次 |
| --- | --- | --- |
| AMX 专用 kernel | Tile 寄存器（TDPBSSD/TDPBF16PS）、**Tiling-aware 权重预处理**（按 Tile 形状重排、64B 对齐、group-wise 对称量化）、每元素只访 DRAM 一次的多级 cache 编排 | **算子级 + 数据布局** |
| AMX ↔ AVX-512 运行时切换 | decode 算术强度低时 AMX 的 tile 派发开销反而不划算 ⇒ 按算术强度动态选 kernel（同一内存布局） | **算子级**（+ 轻量自适应） |
| MoE 算子融合 + work-stealing | 一层所有专家的 gate/up/down 融合成 2 个大任务；再切细后线程间原子窃取，缓解 prefill 专家偏斜 | **算子级（融合）+ 工程级（调度）** |
| 量化 | CPU：INT4/INT8 对称 group-wise（**警告 FP8→INT4 会显著降精度，应从 BF16 量化**）；GPU：GPTQ/RTN W4A16/W8A16；CPU/GPU 共享同一份权重 | **数据层 / 量化** |
| NUMA-aware TP | 权重按 NUMA 节点切片落本地内存；双路服务器 decode 吞吐 **+63%** | **工程级（内存亲和）+ 数据布局** |
| CUDA Graph-backed 调度 | 把 CPU 任务/拷贝做成连续图，消除 graph breakpoint ⇒ kernel launch 开销 >20% → ≈0 | **工程级** |
| **Expert Deferral** | 跨层**重排**专家执行，与 GPU attention 重叠；靠残差容忍性，精度平均降 **≤0.5%**，吞吐 **+1.45×** | **算法级（引入近似）** |
| **选择性专家激活** | 基于 OOD 离线 profile 少激活专家（6 vs 8），宣称质量不变、速度提升 | **算法级（近似/剪枝）** |
| **专家放置策略** | `uniform/frequency/front-loading/random` + 动态更新（prefill 时采集真实路由分布重排 hot/cold） | **算法级（数据驱动调度）** |
| Dual prefill | 按 token 数阈值在 CPU-GPU hybrid prefill 与 Layerwise GPU prefill 间切换 | **工程级 / 调度** |
| 三层前缀缓存 | GPU-CPU-Disk KVCache 复用（`config.yaml` 的 `kvc2`） | **数据层（缓存）+ 工程级实现** |
| 与 SGLang 集成 | continuous batching / chunked prefill / mem-fraction-static / max-running-requests | **工程级** |

---

## 4. 公开收益数字**与前提条件**

⚠️ 这些"大倍数"全部是**低并发 + CPU 侧算力薄（无 GPU offload）场景下 vs llama.cpp/PyTorch 的对比**；
KT 论文自己在摘要里就把它限定为 low-concurrency。高并发下瓶颈很快转到 CPU 内存带宽或 GPU。

| 数字 | 前提 | 来源 |
| --- | --- | --- |
| prefill **4.62–19.74×**、decode **1.25–4.09×** | SOSP'25 摘要，baseline = "existing systems" | 论文页 |
| Expert Deferral **+1.45×**，精度平均降 **≤0.5%** | 论文摘要 + LMSYS blog（两条独立来源一致） | madsys / LMSYS |
| AMX kernel **21.3 TFLOPS** sustained（单 Xeon socket），比 PyTorch 原生快 **3.9×** | LMSYS blog | LMSYS |
| 双路 NUMA TP **+63% decode**；CUDA Graph **launch 开销 >20% → ~0** | LMSYS blog | LMSYS |
| DeepSeek-V3 Q4_K_M：14 GB VRAM + 382 GB DRAM；prefill 54.21→286.55 tok/s、decode 8.73→13.69 tok/s；对 **llama.cpp**（同机）最高 **27.79× prefill / 3.03× decode** | 单机、Xeon Gold 6454S、4090D、**测前充分预热**、**低并发** | `doc/en/DeepseekR1_V3_tutorial.md` |
| Qwen3-30B-A3B：**消费级** i9-14900KF + DDR5-4000 + 4090 可跑 | 显式提醒「内存降频到 4000MT」 | `doc/en/AMX.md` |

---

## 5. 为什么 KT **不能直接搬**过来（场景差异是结构性的）

| 维度 | KT | QLH | 后果 |
| --- | --- | --- | --- |
| 模型 | 大 **MoE**（DeepSeek-V3/R1、Kimi-K2、GLM-5、Qwen3-30B/235B） | 当前正式登记以 **dense ≤2B** 为主；MoE 只作为科研样本 | 专家卸载整套不能直接迁移；其热度统计、放置和预取思想可进入 PyTorch 科研线 |
| 拓扑 | **单机** CPU+GPU（多 GPU 交给 SGLang） | 跨机、异构、跨框架层接力 | KT **不含任何跨机/层接力语义** ⇒ 对分布式中枢零帮助 |
| 硬件 | 服务级 x86（AVX512 至少、AMX 最佳，Linux） | Windows x86_64（Surface）+ **ARM64 Android**（y700） | KT runtime 不在 Edge 支持面内；但 PyTorch 上游研究可在 PC/CUDA 节点做，结果必须保留无 Torch 回退 |
| 判据 | 允许精度换吞吐（≤0.5% 降质换 1.45×） | **逐 token argmax 一致、fail-closed** | 算法级近似**违反判据** |

**不直接引入清单**：KT runtime、`kt-kernel`、`sglang-kt`、AMX/AVX-512 私有 kernel、NUMA 专用执行器、MoE 专家卸载运行时、
Gate/Up 融合实现、GPTQ/Marlin/FP8 依赖栈。它们可以作为外部对照或科研原型，但不得成为 QLH 的第三个后端。

**可借鉴的是机制而不是依赖**：声明式算子替换、设备能力约束、prefill/decode 双计划、数据热度驱动的放置、KV 分层缓存和计算成本画像，见下节。

---

## 6. 对 QLH 的可行建议

### 6.1 PyTorch 上游算法 / 数据层

| 优先级 | 建议 | 理由与来源 | 预估工作量 |
| --- | --- | --- | --- |
| **P0** | **先裁决"上游 torch GPU 层是否真的更贵"的口径冲突**（见 §7），用已有的段画像 `fit_segment_profile`（固定开销 + 边际每层）替代「平均每层」做路由决策 | 两处结论**方向相反** | 0.5–1 人日（纯分析，可复用 `scripts/relay_cut_plan.py`） |
| **P1** | **prefix cache / 会话级 KV 复用落到 L 档主路径**（llama.cpp prompt cache / `src/paged_kv_cache.py`），量化"多轮对话重复 prefill"的节省 | KT 用三层 GPU-CPU-Disk 复用证明**数据层**收益；QLH 调研已把「本地 prefix cache」列为 ✅ 可行 | 2–4 人日 |
| **P1** | **量化资产管线纪律**：上游/下游量化源一律从 **BF16** 出发；核对 `quant_type="int4"` **静默回退 fp16** 的实际生效 dtype | KT 明确警告 FP8→INT4 明显掉精度；QLH 已记录该回退陷阱 | 0.5–1 人日（多为核对与文档） |
| **P2** | **投机解码从 PoC 走到真实链路**（`src/speculative.py` 尚未接入生产循环） | 属算法层提速；vLLM/SGLang 生态的 MTP 是同类杠杆 | 5–10 人日（含分布等价回归） |
| **P2** | **hidden 压缩真正上线**（`int8_block128` 数值往返已 32/32，但 `RELAY_WIRE_VERSION=1` 仍固定 f32） | **只对弱网/跨机有意义**（同机占比 <1%） | 2–3 人日（含协议版本升级与兼容门） |
| **科研档** | Expert Deferral、选择性专家激活、专家放置策略 | 先在 PyTorch 上游和 MoE 样本中验证；默认不能绕过逐 token 一致门 | 仅产出实验报告 |

### 6.2 工程层

| 优先级 | 建议 | 理由与来源 | 预估工作量 |
| --- | --- | --- | --- |
| **P0** | **核查并锁定 llama.cpp 构建的 CPU 指令变体**：在支持的 x86 上启用 AVX512/AMX 变体（`GGML_CPU_ALL_VARIANTS` 或 `sapphirerapids`），做同模型 A/B（每层耗时、decode ms/token） | vendor 内**已有** AMX/AVX512 实现，但 QLH 构建脚本只设了 `GGML_NATIVE`/`GGML_RPC`，**未显式启用**；KT 的 21 TFLOPS / 3.9× 说明该档位差异是**数量级**的 | 1–2 人日/节点 |
| **P0** | **给下游 CPU 侧做线程/亲和性标定**：`-t` = **物理核（非超线程）**、NUMA 策略、避开与 torch 上游争核（P2 交叠场景），产出「每层 ms / 线程数 / 是否与上游并发」表 | KT 把 `cpuinfer=物理核`、`threadpool=NUMA 数`、NUMA 放置当作一等参数，并称 NUMA TP +63% | 1–2 人日 |
| **P1** | **多请求/多序列交叠**（continuous batching + chunked prefill 的最小版）：把 `PIPELINE_MAX_CONCURRENT=1` 变成可解释可配置，先做队列级 micro-batch 交错 | KT 的 `balance_serve` 在 4 路并发下吞吐 +130%；QLH 已把「多请求流式交叠」列为 ✅ 可行但**改造量大** | 8–15 人日 |
| **P1** | **上游 torch 段引入 CUDA Graph / 减少 kernel launch**（不改数值，需与逐 token 判据并行验证；与已落地的 `torch.compile` 分开评估） | KT 用 CUDA Graph 把 launch 开销 >20% → ~0；QLH 上游 torch 每层耗时明显高于下游 | 3–5 人日 |
| **P2** | **权重 mmap 冷启动/缺页专项**（模型放磁盘/网络盘时的首 token 抖动） | KT roadmap 把「AI SSD / 慢 mmap 读盘」列为已识别瓶颈 | 2–3 人日 |

**一句话总纲**：落地路径应是「**先完成大文件拆解和行为基线 → 在 PyTorch 上游建立算子成本画像/注册表 → 做异构放置与 prefill/decode 双计划 → 再做 MoE 热度/预取科研**」；llama.cpp 继续承担稳定路径，KT 不作为运行时依赖。

---

## 7. 与大文件拆解计划的联合排期

大文件拆解必须先于 PyTorch 算子研究。原因不是形式上的代码整洁，而是当前 `scheduler.py`、`api_server.py` 和 PyTorch 执行路径之间仍有较大的装配耦合；在拆解前引入算子放置会把实验依赖继续埋进上帝对象，后续无法区分性能收益来自算法还是结构变化。

### 7.1 阶段与依赖

| 阶段 | 票号 | 交付 | 依赖 | 状态 |
| ---: | --- | --- | --- | --- |
| 0 | `REFACTOR-LARGEFILE-01` | 固化 scheduler/API OpenAPI、公共符号、monkeypatch 面、锁身份、导入/冷启动和全量定向测试基线 | 无 | 已完成 |
| 1 | `REFACTOR-LARGEFILE-02` | 拆出 `scheduler_layer_plan` 与 `scheduler_sidecars`；保留兼容门面及侧车工厂 patch 点 | 01 | **已完成（2026-09-23）** |
| 2 | `REFACTOR-LARGEFILE-03` | 拆出 task-worker、cluster/HA、pipeline mixin；保持实例私有属性、锁和调用点不变 | 02 | **已完成（2026-09-23）** |
| 3 | `REFACTOR-LARGEFILE-04` | 以 APIRouter 拆分 api_server，先 health/device/logs，再 cluster/models/auth/sessions/tasks/chat | 01 | **下一票** |
| 4 | `REFACTOR-LARGEFILE-05` | 重构收口：门面契约、OpenAPI 路径/方法集合、冷启动和完整回归，确认无逻辑夹带 | 03、04 | 排队 |
| 5 | `TORCH-OP-PROFILE-01` | 对项目 PyTorch 上游建立按算子形状、dtype、设备、阶段的成本画像；修正“平均每层”口径 | 05 | 排队 |
| 6 | `TORCH-OP-REGISTRY-01` | 建立逻辑算子到 eager/compile/实验实现的注册、能力声明和 fail-closed 回退合同 | OP-PROFILE-01 | 排队 |
| 7 | `TORCH-HETERO-PLAN-01` | 离线算子放置 planner：设备画像、内存、带宽、边界传输和正确性门；与连续层 planner 对照 | OP-REGISTRY-01 | 排队 |
| 8 | `TORCH-PHASE-PLAN-01` | prefill/decode 双计划和受控状态切换；失败时回退单一 PyTorch 计划或 llama.cpp | HETERO-PLAN-01 | 科研排队 |
| 9 | `TORCH-ACT-COMPRESS-01` | hidden/激活压缩实验；只在跨机带宽受限时启用，逐 token 和长序列门禁 | HETERO-PLAN-01 | 科研排队 |
| 10 | `TORCH-MOE-PLACEMENT-01` | 以可运行 MoE 样本验证专家热度、复制、预取和故障回退；不进入默认 dense 路径 | OP-REGISTRY-01、PHASE-PLAN-01 | 科研排队 |

### 7.2 共同门禁

- `REFACTOR-LARGEFILE-*` 期间不改变推理语义，不新增 KTransformers 依赖，不改默认 backend 选择。
- `TORCH-*` 只在 CUDA/PC 研究环境启用；Edge、Android 和无 CUDA 发行继续使用 llama.cpp/GGUF，不 import torch。
- 每个研究实现必须能退回未优化 PyTorch；整个 PyTorch 上游仍能退回 llama.cpp 或已有连续层计划。
- 研究报告必须同时记录正确性、首 token、decode、峰值内存、通信量和回退边界；性能数字不得跨设备、跨模型或跨阶段直接比较。
- MoE 研究不改变 dense 模型主线，不提前把 MoE 专家调度写进 `PipelineNode` 的稳定合同。

---

## 8. ⚠️ 本轮发现的待裁决口径冲突

**同一件事（"层该推给上游还是下游"）在两处文档里结论方向相反**：

| 出处 | 口径 | 结论 |
| --- | --- | --- |
| `README.md`（切点扫描节） | 上游 GPU **2.5–4.3 ms/层**，下游 CPU **5.8–8.6 ms/层** | **层应尽量推给 GPU 上游** |
| [跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) | 上游 **1.54 ms/层**，下游 **0.91 ms/层** | **方向相反** |

**很可能的原因**：「平均每层」被**固定开销**污染 —— 同一份文档的多轮表里，下游 24 层摊 1.13 ms/层、4 层摊 2.57 ms/层，
说明这个指标随切点变化，不能直接当边际成本用。

**建议**：用 `scripts/relay_cut_plan.py` 的段画像（**固定开销 + 边际每层**）重算一次，作为路由决策的唯一口径；
在裁决之前，**两处文档都应标注该冲突**而不是各自断言。

---

## 9. 明确标注为「未核实」的条目

1. KT 与 **vLLM** 是否另有官方整合（本轮只找到它用 vLLM 项目的 `llmcompressor` 做 GPU 量化）。
2. KT 的 **AMX kernel 是否已回流 llama.cpp**（旧文档只写"考虑贡献"，本轮未查到对应 PR）。
3. KT 的 **ARM64(KML) / Windows** 后端成熟度与实际性能（README 只列为可选构建，roadmap 仍在探索）。
4. §6 里所有**收益数字在 QLH 设备/模型上的外推**都属估计，**非实测**。
5. §7 的方向冲突**本轮未做新实测裁决**。

---

## 附：主要来源

- KT 官方仓库文档：`doc/en/AMX.md`、`doc/en/balance-serve.md`、`doc/en/prefix_cache.md`、
  `doc/en/kt-kernel/experts-sched-Tutorial.md`、`doc/en/kt-kernel/Native-Precision-Tutorial.md`、
  `doc/en/DeepseekR1_V3_tutorial.md`、`kt-kernel/README.md`、`kt-kernel/scripts/README.md`、issue #1921
- LMSYS blog（AMX kernel 21.3 TFLOPS / NUMA +63% / CUDA Graph / Expert Deferral +1.45×）
- SOSP'25 论文摘要与 madsys 论文页
- QLH 侧：[跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)、
  [分布式推理并行与跨框架路线调研汇总](分布式推理并行与跨框架路线调研汇总-2026-09-15.md)、`README.md`
