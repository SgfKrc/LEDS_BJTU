# 2-bit、3-bit 与 4-bit 量化调研与实施计划

> 文档状态：规划（`Candidate`，由《总体下一步计划》管理优先级）
>
> 更新日期：2026-07-30
>
> 适用范围：QLH 的 Qwen-1.8B、DeepSeek Distill 等 LLM，在 PyTorch/Transformers、llama.cpp/GGUF、Windows/Linux PC 和 Android Full 路线上的权重量化；2/3-bit 正式评估仅面向计划中的 14B 及以上模型

> 总计划入口：[总体下一步计划](总体下一步计划.md)

本文回答两个问题：QLH 是否能落地 2/3-bit 量化，以及低比特是否因为“采用浮点”而比 4-bit 精度更强。本文只定义研究、准入和止损规则，不代表当前已经支持任意 2/3-bit 模型。

## 0. 适用模型与产品策略

### 0.1 低比特只服务于大模型容量问题

2-bit/3-bit 的首要价值是让 **14B 及以上模型**在有限显存、内存或分布式容量条件下变得可加载，而不是把所有模型都压到最低位宽。QLH 的正常使用策略如下：

| 模型范围 | 2/3-bit 策略 | 日常默认策略 |
|---|---|---|
| Qwen-1.8B、DeepSeek-R1-Distill-Qwen-1.5B | 只做转换链路 smoke、格式回归和算法对照；不做产品质量结论 | Q4/NF4 或 FP16 |
| 7B 级模型 | 默认不做 2/3-bit；只有容量异常或算法论文复现实验才保留入口 | Q4/NF4、必要时 Q5/Q8 |
| Qwen2.5-14B、DeepSeek-R1-Distill-Qwen-14B | **正式 2/3-bit 候选**，做质量、容量、速度和部署评估 | Q4/NF4 仍是质量基线 |
| DeepSeek-V2-Lite/Coder-V2-Lite 16B | **正式 2/3-bit 候选**，需额外验证 MoE/MLA 架构和 router 保护 | Q4/NF4 或专用运行时 |
| DeepSeek-R1-Distill-Qwen-32B、后续 32B/67B/更大模型 | **优先研究对象**；低比特是否是唯一可加载档位由真实硬件决定 | Q4/NF4 作为对照，不承诺单机可运行 |

这里的“保留入口”是实验能力，不是推荐能力：小模型可以用于验证转换命令、loader、metadata、kernel 和回归测试，但前端、安装包和默认模型列表不得把 2/3-bit 标为更聪明、更快或更适合日常使用。低比特可能让原本已经能力有限的小模型进一步损失指令遵循、事实性和格式稳定性，不能为了节省并不紧张的几十到几百 MB 而主动降智。

---

## 1. 结论先行

1. **2-bit 量化可以做，但不是一种格式。** 至少要区分普通 INT2、带分组缩放的 2-bit 权重、llama.cpp 的 `Q2_K`、重要性矩阵驱动的 `IQ2_*`、AQLM/QuIP# 等向量码本方法，以及重新训练的 BitNet b1.58。它们的存储、内核、模型工件和精度不能互换。
2. **“2-bit 因为是浮点，精度比 4-bit 强”不是普遍成立的结论。** 2-bit 只有 4 个基本码值；4-bit 有 16 个码值。2-bit 某个模型或任务上优于某个 4-bit 结果，通常来自更好的校准、重要性加权、向量码本、异常值保护或不同模型，而不是因为“浮点位数更少反而更精确”。
3. **QLH 当前 PyTorch 路线不支持 `int2`。** `bitsandbytes` 的项目配置是 `fp16`、`int8`、`int4`，其中 `int4` 实际使用 NF4、double quant 和 FP16 compute；不能只把枚举值改成 `int2` 就得到可用的 2-bit kernel。
4. **QLH 当前最现实的 2-bit 入口是 GGUF/llama.cpp。** 上游已经提供 `Q2_K`、`IQ2_XXS`、`IQ2_XS`、`IQ2_S` 等格式。应从原始 FP16/BF16 GGUF 重新量化，并用 importance matrix；不能从 Q4 文件再次量化后宣称等价质量。
5. **第一阶段只做 GGUF 2/3-bit 实验，不改变默认档位。** `Q4_K_M` 仍是集显/Android 的默认质量与兼容性基线，PyTorch NF4 仍是独显路线基线。正式质量结论只对 14B 及以上计划模型生效；小模型仅作为实验参考。低比特只有同时通过质量、稳定性、设备兼容和内存收益门槛后，才能成为可选推荐。
6. **AQLM、QuIP# 和 BitNet 另行处理。** AQLM/QuIP# 是新的量化模型工件和专用推理实现；BitNet b1.58 是经过训练的三值模型和 `bitnet.cpp` 内核。它们不应塞进当前 `ModelManager` 的 `int4` 分支，也不能直接加入 PyTorch 层间流水线。

---

## 2. “2-bit”“4-bit”和“浮点”的准确含义

### 2.1 原始位宽不等于模型文件大小

如果只按权重数量计算，1.8B 参数模型的理想权重数据量约为：

| 权重表示 | 理想权重数据量 | 说明 |
|---|---:|---|
| FP16 | `1.8B × 16 / 8`，约 3.6 GB（十进制） | 不含 tokenizer、元数据和额外张量 |
| 4-bit | 约 0.9 GB | 仍需 scale、zero point、分组元数据和未量化张量 |
| 2-bit | 约 0.45 GB | 实际文件通常明显高于理想值 |

量化文件还要保存 scale、codebook、分组索引、重要性信息、embedding、norm、lm_head 和 metadata。因此应记录 **effective bits per weight（有效每权重位数，bpw）**，不能只按文件名中的 `Q2` 或 `Q4` 判断大小。

### 2.2 普通 INT2、FP2、NF4 和 IQ2 不是一回事

| 名称 | 基本思想 | 当前 QLH 关系 |
|---|---|---|
| INT2 | 2-bit 整数码值，通常配分组 scale/zero point | 没有当前 PyTorch 生产 kernel |
| FP2 | 2-bit 浮点编码，只有 4 种编码状态；不存在一个普适、主流的 LLM 推理标准 | 不作为当前入口；不能据此推断更高精度 |
| NF4 | 4-bit NormalFloat，16 个按正态分布划分的非均匀码值 | 当前 PyTorch `bitsandbytes` `int4` 的实际量化类型 |
| `Q2_K` | llama.cpp 的 K-quant 2-bit 家族，带 block/super-block 元数据；官方示例约 3.16 bpw | 可由 GGUF/llama.cpp 实验，但不能称为恰好 2.0 bpw |
| `IQ2_XXS/XS/S` | llama.cpp 的 importance-aware i-quant，使用重要性矩阵和非均匀码本 | 2-bit 首选候选，实际 bpw 高于 2 |
| `Q3_K_S/M/L`、`IQ3_*` | llama.cpp 的 3-bit 量化家族，通常作为 2-bit 与 4-bit 之间的质量/容量折中 | 14B+ 的第二候选；小模型只做格式实验 |
| AQLM | 多个向量码本相加表示一组权重 | 独立模型工件和 `aqlm` kernel |
| QuIP/QuIP# | 旋转使权重/Hessian 非相干，再使用自适应舍入或格码本 | 独立研究/sidecar，不是 bitsandbytes 开关 |
| BitNet b1.58 | 训练得到 `{-1, 0, +1}` 三值权重，约 1.58 bit | 独立模型架构/训练和 `bitnet.cpp`，不是普通模型 PTQ |

“浮点”只描述编码或计算类型，不自动代表精度。一个 FP2 码本只有四个状态；一个经过重要性优化的 2-bit 向量码本可能比朴素 4-bit 更适合某层，但这属于算法质量差异，必须逐模型、逐任务验证。

### 2.3 当前项目的 4-bit 基线本身不是朴素 INT4

QLH 的 PyTorch 配置使用：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

NF4 为标准正态分布构造非均匀码本，计算通常仍使用 FP16/BF16/FP32。因而将“2-bit 浮点”与“4-bit 整数”直接比较是不对称比较；本项目的实际对照应是：

```text
PyTorch:  FP16  ↔  bitsandbytes NF4 4-bit
GGUF:     F16    ↔  Q4_K_M          ↔  IQ2_XXS / IQ2_XS
```

---

## 3. 2/3-bit 与 4-bit 的工程对比

| 维度 | PyTorch NF4 4-bit | GGUF Q4_K_M | GGUF IQ2_XXS / IQ2_XS | AQLM/QuIP# 2-bit |
|---|---|---|---|---|
| 目标 | 通用 CUDA 加载时量化 | 通用 PC/CPU/集显/Android 推理 | 极限体积的 GGUF 推理 | 极限压缩质量/体积 |
| 当前 QLH 可用性 | 已实现 | 已实现 | 上游格式存在，QLH 需登记和验证 | 未接入 |
| 典型有效 bpw | 约 4 加元数据 | `Q4_K_M` 约 4.9 bpw（随版本/模型变化） | `IQ2_XXS` 约 2.38，`IQ2_XS` 约 2.59 | 取决于码本配置，不能用文件名推断 |
| 质量风险 | 中低，通常为默认低显存基线 | 低到中，质量/速度平衡 | 高，需要 imatrix、敏感张量保护和逐模型验收 | 中到高，需要专用校准和 kernel |
| 速度风险 | bitsandbytes kernel 受 GPU/版本影响 | llama.cpp 路径成熟 | 2-bit 解码/反量化 kernel 可能更慢 | kernel、模型布局和硬件强相关 |
| 模型兼容 | Transformers + bitsandbytes | llama.cpp 支持的 GGUF 架构 | 同左，但取决于嵌入的 llama.cpp 版本 | 仅支持已适配架构和版本 |
| 分布式层流水线 | 可参与同工件层段 | 不能与 PyTorch hidden state 混用 | 不能与 PyTorch hidden state 混用 | 不能直接混用 |
| 适合 Android | 不适合当前 Android 路线 | 是，需实机验证 | 可能适合，需实机验证 | 暂不适合 |

llama.cpp 官方示例对 Llama 3.1 8B 的测量中，`Q2_K` 约 3.1593 bpw，`IQ2_XXS` 约 2.3824 bpw，`IQ2_XS` 约 2.5882 bpw，`Q4_K_M` 约 4.8944 bpw。因此 `Q2_K`/`IQ2_*` 是 2-bit 量化家族，实际文件仍包含缩放和元数据。相同运行时下，低位宽不保证更快；解码 kernel、内存访问和 CPU/GPU backend 可能让 2-bit 的速度收益小于体积收益。

---

## 4. 可行性判断

### 4.1 Qwen-1.8B：只做工具链 smoke

Qwen-1.8B 适合做第一轮工具链 smoke：规模小、已有 Q4 GGUF、集显安装包已有 llama.cpp 路线，能快速验证 F16/Q4/Q3/IQ2 的转换、加载、metadata 和基本生成。但它**不属于 2/3-bit 正式目标模型**，其低比特结果只能回答“链路是否工作”，不能回答“产品是否应该给小模型降比特”。

准入顺序：

1. 固定当前 Qwen 模型的原始权重、tokenizer、chat template 和 llama.cpp commit。
2. 从原始 FP16/BF16 工件转换出 F16 GGUF。
3. 用同一版本 `llama-quantize` 生成 `Q4_K_M`、`Q3_K_M`、`Q2_K`、`IQ2_XXS`、`IQ2_XS`；i-quant 必须使用固定 calibration corpus 生成的 importance matrix。Qwen-1.8B 仅生成最小 smoke 样本，不投入完整质量评测。
4. 保留 embedding、norm 和 `output/lm_head` 的高精度策略，并记录是否使用 `--leave-output-tensor` 等选项。
5. 用固定 prompt、固定 seed、贪心解码和采样解码分别测质量、速度、内存和长文本稳定性。

### 4.2 14B+ 计划模型：正式目标

正式低比特评估优先覆盖 `Qwen2.5-14B`、`DeepSeek-R1-Distill-Qwen-14B/32B`、`DeepSeek-V2-Lite-Chat` 和 `DeepSeek-Coder-V2-Lite-Instruct`。这些模型的低比特研究应围绕“能否在目标硬件加载、能否维持可接受质量、是否值得增加维护复杂度”展开，而不是先假设 2-bit 必须成功。

### 4.3 DeepSeek Distill

DeepSeek Distill 不能直接沿用 Qwen-1.8B 的量化结论：架构、参数规模、tokenizer、思考 token 行为和敏感层都可能不同。至少要分别记录：

- Dense 还是 MoE、专家数和 active experts；
- embedding、attention、MLP/router、norm、lm_head 的保护策略；
- chat template、思考开关和最大输出长度；
- 同一 calibration corpus 是否覆盖中文、代码、数学和推理文本；
- 量化前后是否出现思考标签泄漏、重复循环、提前结束或格式破坏。

DeepSeek Distill 的 14B/32B 才是正式低比特目标；1.5B/7B 仅用于兼容性和转换回归。没有对应架构的稳定 GGUF 转换、真实显存窗口和 14B+ 质量集时，不为 DeepSeek 注册 2/3-bit 下载项。

### 4.4 PyTorch 2/3-bit

PyTorch 路线分两级：

1. **不做伪支持。** 不在 `src/config.py`、API 或前端增加 `int2`，除非后端能加载真实 2-bit 工件并完成正确性测试。`bitsandbytes` 当前公开路径是 8-bit 和 4-bit，不能作为通用 2-bit 后端。
2. **独立 sidecar 试验。** 选择 AQLM、HQQ、QuIP# 等有实际 loader/kernel 的方案，隔离依赖和模型目录，先只提供离线 benchmark。sidecar 不能改变默认 `ModelManager`、层间 Worker 或打包环境的依赖。

只有当 sidecar 能在一个 **14B+** Qwen/DeepSeek 目标架构上完成可重复加载、取消、错误恢复、质量门和性能门，才讨论接入完整 Worker。小模型 sidecar 只能做工具链参考。即使接入，也必须把量化算法、codebook、calibration 和 kernel 版本写入 `ModelIdentity`。

---

## 5. 计划阶段

### Q2-N0：定义与工件冻结

- 明确定义 `raw_bits`、`effective_bpw`、`quant_family`、`calibration_id`、`imatrix_sha256` 和 `kernel_revision`。
- 固定一个 Qwen-1.8B smoke 工件，以及一个 14B+ 正式目标的 FP16/BF16 源工件、tokenizer、chat template、llama.cpp commit 和测试机。
- 建立模型清单，不把 `Q2_K`、`IQ2_XXS`、AQLM、BitNet 归到同一个 `int2` 标签。
- 验收：同一输入能在两台测试机报告完全相同的 artifact/quant metadata；未满足则不进入转换。

### Q2-N1：GGUF 转换和静态检查

- 对 smoke 小模型生成最小 F16/Q4/Q3/IQ2 文件集；对 14B+ 正式目标生成 F16、Q4_K_M、Q3_K_M、Q2_K、IQ2_XXS、IQ2_XS 文件集。
- 2-bit 候选从 F16/BF16 直接量化；禁止 Q4→Q2 的重复量化作为质量基线。
- i-quant 使用固定 importance matrix；保留一份未量化 output/embedding 变体作为敏感张量对照。
- 用 `llama-model-loader`/现有 GGUF 检查工具确认 architecture、tokenizer、tensor 数、vocab 和量化类型。
- 验收：所有文件可加载；损坏、模型架构不匹配和缺 tokenizer 能在加载前明确失败。

### Q2-N2：单机质量与性能基准

14B+ 正式目标的基线固定为 F16 和当前 `Q4_K_M`，候选为 `Q3_K_M`、`Q2_K`、`IQ2_XXS`、`IQ2_XS`。小模型只运行短 smoke 和加载回归，不出正式质量排名。每组记录：

- 文件大小、effective bpw、加载时间、峰值 RSS/VRAM；
- prompt processing tok/s、首 token 延迟、decode tok/s、端到端时间；
- 2048/4096/8192 上下文的稳定性和 KV 内存；
- 固定中文、英文、代码、数学、长上下文和格式遵循样本的质量；
- 贪心输出、固定 seed 采样输出、异常退出、重复循环和空输出；
- CPU 集显、RTX 独显（若走 GGUF backend）以及 Android 真机结果。

建议质量门：

- 14B+ 的 2/3-bit 候选相对 Q4 基线的评测损失不超过预先冻结的阈值；初始建议困惑度相对增幅 ≤5%，任务准确率下降 ≤2 个百分点；小模型不套用该门槛做产品推荐；
- 不得出现大面积乱码、重复循环、chat template 破坏、思考标签失控或长上下文崩溃；
- 14B+ 的 2/3-bit 至少节省 25% 的模型常驻内存，或在目标设备上从无法加载变为可加载；小模型仅记录占用变化，不以此推动默认降比特；
- 速度可以不提升，但若 decode tok/s 比 Q4 低超过 15%，只能作为“容量档”，不能标为性能档。

阈值是初始门槛，不是跨模型保证。所有结论必须绑定模型工件、评测集、运行时、设备和量化参数。

### Q2-N3：QLH GGUF 与 Android 接入

- 扩展 `src/llama_engine.py` 的量化识别和模型清单，支持从 GGUF metadata 读取量化类型，不依赖固定文件名。
- 为 `Q3_K_M`、`Q2_K`、`IQ2_XXS`、`IQ2_XS` 增加显式实验标签和警告；仅当模型参数量 ≥14B 时允许进入正式低比特评估，默认仍推荐 `Q4_K_M`。
- 将 `quant_family`、`effective_bpw`、`imatrix_sha256`、llama.cpp commit 和 backend 写入模型 manifest/运行统计。
- Android 使用同一锁定的 llama.cpp 子模块和 GGUF 文件，在 armeabi-v7a/arm64-v8a 或实际发布 ABI 上做加载、生成、取消、后台恢复、温控和内存验收。
- 若 Android 内置 revision 不支持某 IQ 类型，必须在模型列表中隐藏该类型并给出原因，不能下载后才崩溃。

### Q2-N4：PyTorch 2-bit sidecar

- 分别调研 AQLM、QuIP#、HQQ/Quanto 等方案对 14B+ Qwen/DeepSeek 架构的覆盖、Windows/CUDA 支持、CPU 支持和许可证；小模型只作为 loader smoke。
- 只选一个有完整 loader、kernel、量化脚本和公开 benchmark 的方案做 PoC，避免同时引入多个量化框架。
- sidecar 以完整 Worker/外部 Provider 方式运行；不把 sidecar 的量化层塞进当前 PyTorch 层间协议。
- 记录反量化位置、codebook 传输、取消粒度、显存峰值和多节点 artifact 一致性。
- 质量或稳定性低于 GGUF IQ2，或者依赖无法进入 Windows/Android 发布环境时，直接冻结该路线。

### Q2-N5：Go/No-Go

| 结果 | 条件 | 产品动作 |
|---|---|---|
| Go：GGUF 2/3-bit 实验 | 14B+ 的 Q2-N2 质量、内存、加载和真机门全部通过 | 保留为高级/实验选项；Q4 仍为默认 |
| Go：GGUF 2/3-bit 容量档 | 14B+ 质量可接受，速度不占优但内存收益显著 | 只在对应大模型和低内存设备显示“容量优先”，不宣称更快/更准 |
| No-Go：GGUF 2/3-bit | 14B+ 质量退化、kernel 不稳定、Android 不可用或收益不足 | 不登记下载项；小模型不因 smoke 通过而开放推荐 |
| Go：PyTorch sidecar | 14B+ 独立 loader、kernel、模型和 Worker 验收通过 | 作为独立引擎/Provider，不能复用 `int4` 配置名 |
| Frozen：PyTorch sidecar | 依赖、架构或硬件门无法满足 | 不污染默认 requirements 和安装包 |

---

## 6. 分布式推理边界

量化是模型工件属性，不是一个可在节点之间随意切换的运行参数。

### 6.1 层间流水线

- PyTorch 层间流水线的所有参与节点必须使用相同模型架构、tokenizer、量化算法、quant revision、codebook、calibration 和兼容 runtime。
- 不能让主节点用 NF4、从节点用 GGUF IQ2，也不能让同一层段的一次 attempt 混用 Q4 和 Q2；hidden state 的数值误差会改变后续计算，协议也没有这种工件协商。
- 2/3/4-bit 的模型身份必须进入 ready ACK 和 `ModelIdentity`，而不是只在前端显示一个 `quant_type`；参数量和“正式目标/实验 smoke”也应可追溯。

### 6.2 完整 Worker 与任务链

- 完整 Worker 可以为不同任务选择不同量化工件，但任务统计必须显示实际 `artifact_id`、`quant_family`、参与节点和降级路径。
- 固定评测或演示任务必须锁定同一量化工件，否则输出差异不能归因到分布式调度。
- 任务级多 seed fan-out 可以把 Q4 与 Q2/Q3 作为明确的 A/B 实验组，但不能把它们的图片混称为同一模型的完全等价输出；小模型的 A/B 仅用于实验，不进入默认推荐。

### 6.3 传输与安全

- 量化权重不通过普通 Stage JSON 或 hidden-state 消息传输；节点只通过 artifact manifest、revision、哈希和本地路径映射准备模型。
- AQLM codebook、IQ2 importance metadata 等应作为模型资产的一部分登记，不能只同步主权重文件。
- 远端 Worker 未通过 quant manifest 校验时，调度器拒绝派发，而不是静默回退到另一种量化。

---

## 7. 必须更新的代码与测试（实施时）

本轮只写计划，不修改代码。进入 Q2-N3 时至少需要：

| 范围 | 计划改动 | 验收测试 |
|---|---|---|
| GGUF 识别 | 从 metadata/文件名识别 Q2/Q4/IQ2，未知类型不崩溃 | 每种 quant type 的 manifest、错误文件和旧 GGUF 回归 |
| 模型注册 | 增加 `quant_family/effective_bpw/calibration_id` | API 返回与本地加载结果一致 |
| 安装包 | 不默认下载 2/3-bit；仅对 14B+ 高级实验显示，模型列表可显示警告 | 集显版干净机启动、下载、取消和卸载 |
| Android | 以实际 llama.cpp revision 检查 IQ2 支持 | arm64 真机加载、生成、取消、温控和内存 |
| Worker | artifact identity 纳入 ready/lease/统计 | 错模型、错 quant、错 revision 拒绝；不影响 Q4/FP16 |
| 评测 | 固定 prompts/seeds/数据集和运行时版本 | 可重复生成报告，失败项保留原始日志 |

进入 Q2-N4 时，sidecar 必须单独安装和测试；不能把 AQLM/QuIP# 的 Python 依赖直接写进集显版或 Android 依赖。

---

## 8. 主要风险与止损

| 风险 | 处理 |
|---|---|
| 把 `Q2_K` 当成恰好 2 bpw | 所有报告同时写格式和 effective bpw；使用 `IQ2_*` 时注明 importance-aware |
| 2-bit 输出退化为重复/乱码 | 固定生成集、长上下文和异常样本门；失败即冻结，不通过调高 temperature 掩盖 |
| 2/3-bit 比 Q4 慢 | 将它定位为 14B+ 容量档；若无内存或可加载收益则 No-Go |
| 小模型低比特退化 | Qwen-1.8B/1.5B/7B 只保留 smoke 入口，不显示产品推荐，不用低比特掩盖模型能力不足 |
| 重新量化造成质量不可控 | 强制从 F16/BF16 源量化，禁止 Q4→Q2 作为正式基线 |
| embedding/lm_head 对低比特敏感 | 保留高精度对照和敏感张量白名单，并把策略写入 manifest |
| Android 的 llama.cpp revision/ABI 不支持 IQ2 | 构建期能力探测 + UI 隐藏，不在运行时猜测 |
| AQLM/QuIP# 依赖污染当前打包环境 | sidecar 独立 venv/Provider；不改默认集显和 Android 产物 |
| DeepSeek 架构与 Qwen 结论不同 | 每个架构独立 calibration、artifact 和质量门，禁止横向套用结论 |
| 用单一“精度”字段误导用户 | UI 显示“容量/质量/速度”三个维度，并标注实验性和基线 |

触发以下任一条件时，Q2 路线转 `Frozen`：

- 没有同一源工件、同一 tokenizer 和可复现转换命令；
- Q2 在固定评测中明显破坏格式、长文本或核心任务；
- 目标设备上的 2-bit kernel 不稳定，或加载/取消/恢复存在不可恢复崩溃；
- 相对 Q4 没有足够的内存或可加载收益；
- 只能通过隐式混用不同 artifact 才能完成分布式推理。

---

## 9. 待决策项

- 14B+ 正式目标中，首个 `IQ2_XXS`/`IQ2_XS` 或 `Q3_K_M` 候选由 Q2-N2 的质量/内存/速度数据决定；Qwen-1.8B 只保留 smoke 文件。
- Android 是否把 2-bit 作为高级下载项，还是只在开发版开放。
- 是否投入独立 AQLM/QuIP# sidecar；默认不与 GGUF 2-bit 并行投入。
- DeepSeek Distill 是否拥有足够显存和真实用户任务集进行 2-bit 校准。
- 2-bit 质量门的最终任务集、语言比例和长上下文长度。

---

## 10. 调研来源

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Hugging Face bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [bitsandbytes Linear4bit / NF4 reference](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit)
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](https://arxiv.org/abs/2307.13304)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)
- [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [Hugging Face AQLM documentation](https://huggingface.co/docs/transformers/quantization/aqlm)
- [llama.cpp quantization tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
- [Microsoft BitNet / bitnet.cpp](https://github.com/microsoft/BitNet)

外部项目的格式、kernel、模型支持和许可证会变化。真正实施前必须重新锁定 commit、转换命令、模型文件 SHA-256 和测试结果；本调研不能替代发布审计。
