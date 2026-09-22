# QLH

QLH 是一个面向异构边缘设备的分布式推理核心：主线是 GGUF/llama.cpp 轻量引擎，主仓同样拥有 PyTorch 分层分布式引擎与**层流水线**（含跨框架逐层接力），用户交互入口是跨平台 TUI。

> 状态：主仓基线重整中（2026-09-21）
>
> 本 README 只描述主仓当前边界和可复现入口。实验记录、历史实现和外置子项目不等同于主仓生产能力。
>
> English: [docs/README.en.md](docs/README.en.md)

## 主仓做什么

- 在 Windows/Linux PC、无 CUDA 设备和 Android 节点上运行或协调 GGUF 推理。
- 让单机装不下的模型由多个节点按已验证的层段合同共同承载；每个节点只应持有实际分配的模型部分。
- 让 Edge 节点默认使用 1B 以内模型完成本地推理，同时保留作为大模型 RPC worker 的能力。
- 用同一套层段合同管理三类节点（本机、远端 RPC、跨框架），并按设备画像与能力选择引擎。
- 通过设备画像、容量计划、模型身份、层段合同、租约和 epoch fencing 控制分布式准入与故障恢复。
- 用 Textual TUI 访问聊天、模型资产、节点、分布式布局、队列、设备、日志和设置；未形成专用交互的后端操作才进入调试兜底页。只读单命令走标准库薄层。

引擎是**双档**，两档都在主仓，不是"主路径 + 遗留对照"的关系：

| 档 | 引擎 | 能力 | 依赖 |
| --- | --- | --- | --- |
| **L 档** | llama.cpp / GGUF | 单机推理、RPC 部分驻留、TUI 对话、优化三件套 | 无 torch（Edge 默认） |
| **D 档** | PyTorch / Safetensors | 层拆分与张量放置、**层间流水线**、多节点层段承载、跨框架接力的上游侧，并承担与 llama.cpp 的对照实验 | torch |

**D 档不进入 Edge 默认依赖**，但层拆分、层流水线、跨框架接力这些能力实际由 PyTorch 系实现（`model_module.py`、`tcp_comm.py`、`qwen3_pipeline_*`），因此不是"仅对照路径"。同卡实测下 llama.cpp 单序列更快（约 4.3×），因此默认生产路径仍是 L 档。

## 架构总览

QLH 是**一个进程里的两层**：面向人的控制面，以及面向机器与协议的引擎层。两层之间只有一条
边界 —— 层段合同 `(layer_range, engine, location)`。

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 控制面（面向人）                                                           │
│ Textual TUI · 只读单命令 · HTTP API(/api/cluster/*)                        │
│ 设备画像 · 容量计划 · 层段合同 · 租约 / epoch fencing · 准入                │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ 层段合同 (layer_range, engine, location)
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ 引擎层（面向机器）                                                         │
│ L 档 llama.cpp / GGUF               D 档 PyTorch / Safetensors             │
│ ├ 单机推理（Edge 默认）             ├ 层拆分与张量放置                     │
│ ├ ggml RPC worker（借算力）         ├ 层间流水线（qwen3_pipeline_*）        │
│ └ 层段前向（keep-head shim）        └ 跨框架接力上游（model_module）        │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ 层段通道：Relay TCP（HIDDEN / HIDDEN_SEQ / TOKEN）
┌──────────────────────────────┴─────────────────────────────────────────────┐
│ 节点与传输（跨机、异构）                                                   │
│ 本机 loopback · SSH 隧道 · Surface(x86_64, Windows) · y700(ARM64, Termux)  │
│ hidden 压缩：f32 / f16 / bf16 / int8_block128 · 弱网带宽与额外延迟口径      │
└────────────────────────────────────────────────────────────────────────────┘
```

一条已实测通过的四段链路（记录与方法见 [文档入口](#文档入口) 里的接力基线文档）：

```
prompt → torch(0..7) →hidden→ Surface(8..15) →hidden→ y700(16..19) →hidden→ llama(20..23+head) → token
           本机 CUDA              x86_64 Windows         ARM64 Android           本机 llama.cpp
```

判据是**与同精度整模逐 token argmax 一致**（不得用 cosine 代替）；不一致即如实标 FAIL，
不做"可接受近似"。

## 这是什么软件：系统软件还是用户软件？

**分层回答**：QLH 是**系统软件内核 + 用户软件外壳**的同体交付。

- **控制面 ≈ 用户软件**：TUI、模型资产、节点/布局/队列/日志/设置页面、HTTP API。使用者是**人**，
  失败模式是"体验退化"（重试、换模型、换布局），接口可以演进。
- **引擎层 ≈ 系统软件**：层段合同、层流水线、跨框架接力、异构节点承载、hidden 压缩与传输。
  使用者是**其他软件**（控制面、上层编排、跨节点对端），失败模式是"静默算错"，因此接口是
  强契约、判据一律 fail-closed。

| 判据 | 控制面 | 引擎层 |
| --- | --- | --- |
| 主要使用者 | 人（终端用户 / 运维） | 其他软件（TUI、API 编排、跨节点对端） |
| 失败后果 | 体验退化，可重试 | 数值错误，可能**静默**错算 |
| 接口稳定性 | 可演进（页面/命令可改） | 强契约（层段合同、协议版本、记录 schema） |
| 可否单独替换 | 可以（换前端不动引擎） | 不可以（换引擎即换数值语义，须重跑逐 token 对照） |
| 类比 | 应用 / 管理面板 | 内核 + 运行时 + 分布式子系统 |

两个工程含义：

1. **改动落点决定验证强度**：改控制面跑 UI/合同测试即可；改引擎层必须逐 token 对照 +
   记录 schema 校验 + 矩阵化实测。
2. **"借算力"不是兜底而是节点类型**：`remote_rpc` / `cross_framework` 与 `local` 同构，
   所以引擎层是"一种子系统、多种节点"，而不是"主路径 + 降级路径"。

## 层流水线与跨框架逐层接力

### 统一节点抽象

层流水线把"谁持有哪些层、用什么引擎、在哪里、容量多大"统一成一条抽象：`(layer_range, engine, location)`。三种节点**同构**，共享同一套合同与校验：

| `kind` | 含义 | 引擎 | 通信 |
| --- | --- | --- | --- |
| `local` | 本机进程内的层段 | llama.cpp / pytorch | 进程内 |
| `remote_rpc` | **借来的算力**（Android/PC 上的 `ggml-rpc-server`） | llama.cpp | 网络（ggml RPC） |
| `cross_framework` | 跨引擎的层段接力（torch 上游 + llama.cpp 下游） | 两种 | 进程内或 stdio |

"借算力"不是兜底方案，而是层流水线里的**一种节点类型**；三种节点共用一个抽象，因此不存在"主路径 vs 兜底"的对立。

实现与端点：

- `src/pipeline_node_contract.py`：`PipelineNode` 合同、既有产物映射、布局 fail-closed 校验；
- `src/pipeline_capacity.py` 容量求解、`src/pipeline_assignment_manifest.py` 分配 manifest；
- `src/pipeline_reshard.py`：容量重解 + 工件就绪门 + epoch 原子提交；
- `GET /api/cluster/layers`、`GET /api/cluster/pipeline-capacity`、`GET /api/cluster/pipeline-reshard`。

### 跨框架逐层接力（D→L）

上游 PyTorch 层段算到第 N 层，把 hidden states 交给下游 llama.cpp 裁层模型继续算完。注入点是 llama.cpp 的**标准** `llama_batch.embd` 字段，因此 PyPI 版 `llama-cpp-python` 即可完成，**无需 fork 或重新编译**。

**为什么做它（意义）**：

1. **它是层流水线的孪生机制**——两者共用"层切分 + hidden 传递"这套机制；层流水线能**彻底摆脱整模加载**（llama.cpp RPC 需要 leader 持有完整 GGUF），因此这条路线决定的是"单个设备装不下的模型能否被多台设备承载"，而不是单序列延迟；
2. **它是异构设备唯一能拼进同一流水线的接口**——引擎能力不同的节点（`local` / `remote_rpc` / `cross_framework`）只有靠它才能共处一个层流水线；
3. **它是"可定制"的前提**——切点分配、混合精度、算子替换、批量交叠这些实验，都建立在"层间可传递 hidden"之上；
4. **学术上没有对口先例**——Petals 是同框架、KTransformers 是算子级、distributed-llama 是 TP；这条路上我们还给上游提交了缺陷并独立验证了修复（issue #28963）。

**当前有效数据（2026-09-21）**：旧的 182 s → 21.3 s 优化链保留为历史过程，不再作为主仓双引擎端到端性能基线。两端都走主仓引擎的样本是 qwen2.5-0.5B、12+12 层、gen=32：上游 `model_module.forward_layers` + 下游 `llama_engine.forward_layers_from_hidden`，D→L 为 **47.501 ms/步**，与纯 llama.cpp 整模逐 token 一致。**同日晚场复跑补齐了完整矩阵**：两模型 × 切点 / 负载（prefill 32/128/512、decode 32/64/256）/ batch（2/4）/ 混合精度（上游 fp16·f32·NF4 × 下游 Q4_K_M）共 **27 次全部逐 token 一致**，qwen3.5 hybrid 的 **K=8/12/16/20 全部 32/32**。⚠️ 引用档位前先核对**实际生效**项：主仓层流水线的 `quant_type="int4"` **静默回退 fp16**，上游 compile 在 <1.5B 参数时被规模门禁用。

**D→L 的容量价值**：qwen2.5-0.5B / qwen3-5-2b 两段切分的容量收益分别为 **1.568× / 1.547×**；同一受控 3.0 GB CUDA 预算下，整模拒绝而 12 层上游通过。该预算是可复现实验约束，不是物理 OOM。D→L 的定位是容量合并和异构能力组合，不是 CUDA 单机提速替代方案。

**证据与边界**：

| 项 | 当前口径 |
| --- | --- |
| 正确性 | 主仓双引擎 D→L 矩阵 **27/27 逐 token 一致**（qwen2.5-0.5B K=4/8/12/16/20、qwen3.5-2B K=8/12/16/20；负载 prefill 32/128/512、decode 32/64/256；batch 2/4；混合精度 fp16·f32·NF4 × Q4_K_M）；逐 token 判据与速度、容量分开记录 |
| 混合精度 | 上游 PyTorch 全精度、下游 GGUF 量化是有意的“不完整量化”策略；必须与同精度下游整模对拍 |
| L→L 上游通道 | pip 绑定的 `llama_get_embeddings_ith` 返回 `output_norm(H)`（实测 cos 0.999998）⇒ **不能**当层接力上游；补丁版 **keep-head 通道已打通**：`--path l2l_keep_head` / `d2l2l_keep_head` 实测 **32/32**（含「1 torch 上游 + 2 llama 下游」三段），旧 `l2l_llama` 保留为 fail-loud 反例 |
| 切点求解 | `scripts/relay_cut_plan.py` + `src/relay_cut_objective.py`：从**实测**拟合段画像（固定开销 + 每层耗时）再求解，输出 `capacity_feasible` / `latency_estimate` / `risk_penalty`；n 段、含 Qwen3.5 的 4 层倍数硬约束。2 段闭环在 qwen2.5（r² 0.96/0.99）与 qwen3.5（0.79/0.96）上均通过 |
| Windows 算子 | Windows 原生 `triton-windows==3.8.0.post28` 已实测可用；`PYTHONUTF8=1` 是编译路径前置条件；WSL2/fla 是并行路径，不是唯一方案 |
| 生产定位 | 正确性证据满足 Relay 合同准入；速度只影响默认路由倾向，长时、远端资产自动分发和多段故障验收仍待完成 |
| 当前文档 | 以[当前有效基线与后续优化计划](docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)为索引，旧报告中的矛盾数字按其有效性分级处理 |

**结论修正**：早期"IPC 是主要成本"的判断已被推翻——那是两侧都慢时被掩盖的假象。**杠杆在两侧的计算（切点、kernel、批量），不在传输层面**。详见 [同进程双后端接力实现与性能](docs/archive/同进程双后端接力实现与性能-2026-09-16.md) §12–§14。

### 切点扫描结论（P0，2026-09-18 实测）

对上游层数 N 做了完整扫描（N=0 表示**不接力**、只用 llama.cpp 跑整模；下游为对应的 f16 裁层 GGUF，CPU / 8 线程）：

| 上游层数 N | 上游 ms/步 | 下游 ms/步 | **合计 ms/步** | 64-token 序列 |
| ---: | ---: | ---: | ---: | --- |
| **0（不接力）** | 1.9 | 184.8 | **186.8** | 与基线逐 token 一致 |
| 4（当前默认） | 42.7 | 171.6 | **214.3** | 一致 |
| 8 | 77.9 | 145.9 | 223.8 | 一致 |
| 12 | 120.8 | 103.8 | 224.6 | 一致 |
| 16 | 137.2 | 84.6 | 221.9 | 一致 |
| 20 | 182.9 | 66.8 | 249.7 | 一致 |

- **正确性不随切点变化**：所有切点的贪心序列逐 token 相同；
- **总时间对切点几乎不敏感**（N∈{4,8,12,16} 仅在 214–225 ms/步之间，±2.6%），**不存在中间谷底**；现有默认 N=4 在"必须接力"的前提下已是最优点；
- **不接力反而最快**（186.8 ms/步，比默认 N=4 快 12.9%）⇒ 本机同机、单序列条件下，接力开销约 **+15%**（相对纯 llama.cpp CPU 口径）。这比"对比 llama.cpp 原生整模 GPU **慢约 25×**"（`333 ÷ 13.5`，见上表口径注）温和得多 —— **那 25× 主要是 CPU/GPU 之差，不是接力机制的成本**；
- 上游每层（torch/CUDA，8.8–10.7 ms）**不比**下游每层（llama.cpp/CPU，7.7–8.6 ms）便宜，所以"把层搬到 torch GPU"在本机不产生速度优势；
- **工程约束**：切点必须是 `full_attention_interval`（Qwen3.5 = 4）的整数倍，否则裁层 GGUF 的层类型错位而无法加载（N=2 实测）；
- **方法学警告**：脱离端到端链路的孤立测量不可信（本次把上游单步耗时测低了约 5.6×），切点类结论必须以端到端口径为准。

报告：`local_docs/CORE-RELAY-XFRAME-02-sweep-2026-09-18.json`；票：[验收清单 D29](docs/验收清单与资源限制登记.md)。

**⚠️ 同日修正（v2）—— 上面这一节（含表格）的结论仅在"上游跑在 CPU"时成立**：`relay_sameproc_4L.py` 的 `from_pretrained` 之后没有 `.to(device)`，`dev = tmodel.device` 于是是 **cpu**；而孤立脚本 `upstream_layer_cost.py` 显式 `.to("cuda")`。同一脚本同口径实测同 4 层：**cpu 34.9 ms / cuda 8.3 ms** ⇒ 那 5.6× 差异**由设备解释**（既不是 KV 形状，也不是空闲降频——两者已用对照实验否证：`shape_sensitivity` fixed 47.4 > growing 34.4；`idle_wakeup_and_overlap` idle 8.56 vs continuous 7.53 = 1.14×，SM 时钟全程 780/3105 MHz 不变）。

给 relay 加 `--upstream-device cuda`（配 f16 + `--upstream-partial` 只加载前 N 层，显存约 N/24 × 4.3 GB）后重扫：

| 上游层数 N | CPU 上游 合计 ms/步 | **GPU 上游 合计 ms/步** | 提升 |
| ---: | ---: | ---: | ---: |
| 8 | 223.8 | **169.0** | 1.32× |
| 12 | 224.6 | **155.0** | 1.45× |
| 16 | 221.9 | **146.4** | 1.52× |
| **20** | 249.7 | **129.1** | **1.93×** |

- 上游 **GPU ≈ 2.5–4.3 ms/层**，下游 **CPU llama.cpp ≈ 5.8–8.6 ms/层** ⇒ **应把层尽量推给 GPU 上游**；
- **修正后最优（已测）N=20 = 129.1 ms/步**，比**不接力**的 186.8 ms/步快 **1.45×** —— **接力首次显示出明确收益**；
- 全部切点的 64-token 序列仍**逐 token 一致**（含 GPU 上游）；
- 所以"不接力最快 / 切点无收益"**只在 CPU 上游条件下成立**，不可外推。下游在真实部署里多为 **无 CUDA 的边缘设备**，恰好支持"层往 GPU 上游放"这一方向——也因此 **P1（给下游加 GPU）适用面窄，真正值得做的是「上游 GPU 化」与 P2 交叠**。

报告：`local_docs/CORE-RELAY-XFRAME-02-p0-corrected-2026-09-18.json`（v2，取代 v1）。

### 与「全 llama + CUDA」的公平对照 + P2 交叠（2026-09-18 实测）

**问：集群里有 CUDA 节点时，全 llama.cpp 是不是不如接力？答：不是。**

| 配置 | ms/token | 相对 |
| --- | ---: | ---: |
| **llama.cpp + CUDA**（build-cuda，`-ngl 24` 全部层上 GPU，f16，t=8） | **26.6** | 1.0× |
| 跨框架接力最优（torch GPU 上游 N=20 + llama.cpp CPU 下游） | 129.1 | 4.85× 慢 |

`-ngl` 曲线单调（ms/token）：`0→91.1`、`4→71.0`、`8→60.5`、`12→52.9`、`16→44.2`、`20→34.7`、`24→26.6`。⚠️ 参照物必须**同构建**——build-cuda 的 llama-bench 即使 `-ngl 0` 也比 build-cpu 快约 2×（91.1 vs 194.2 ms/token）。

**P2 交叠**（软件流水线：上游 GPU torch 与下游 CPU llama.cpp 交错推进，各持一把锁；torch 的 CUDA 调用与 ctypes 的 llama.cpp 调用都释放 GIL，因此可真正并行）：

| 模式 | ms/token | 吞吐 |
| --- | ---: | ---: |
| serial | 101.11 | 154.5 tok/s |
| **threaded（交叠）** | **78.31** | **199.5 tok/s** |

加速 **1.291×**，且两条序列的 token 与单序列基线**完全一致**；理论天花板约 1.79×（完全重叠时取上游 72.3 / 下游 56.8 之较大者），实测达到约 72%。

**结论与定位**：上面 P0 那个 1.45× 只是「CPU llama.cpp → GPU **torch**」的局部收益；更好的做法是「CPU llama.cpp → GPU **llama.cpp**」（`-ngl`）。所以 **跨框架接力不是更快的推理路径**，而是「**只能用 torch 跑的层**」（hybrid/自定义算子）与「**容量合并 / 层流水线**（单机装不下）」的机制，外加实验平台。**集群里有 CUDA 节点时，最佳实践是把它作为 llama.cpp 的 CUDA worker（RPC/分片），而不是接力上游**；P2 交叠只在「不得不接力」的场景内把损失补回一部分（78.3 ms/token 仍慢于 26.6 约 3×）。

报告：`local_docs/CORE-RELAY-XFRAME-02-p2-2026-09-18.json`。⚠️ 以上均为**同机**数据；**跨机（GPU 节点 + 无 CUDA 边缘节点）的 RPC vs 接力对照仍未测**。

### torch.compile 与「层循环」开关（`USE_COMPILE` / `USE_MONOLITHIC_FORWARD`）

分段前向要吃到 `torch.compile` 的收益，靠这两个开关配合（都在 `src/config.py`，可用环境变量覆盖）：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `USE_COMPILE` | `True` | 启用编译。编译不可用时**告警并回退 eager**，不影响启动（Windows 未装 [`triton-windows`](requirements-compile.txt) 时即走此路径；**装了就可用** —— 2026-09-19 起 Windows 原生已实测编译成功，不再是「死开关」） |
| `USE_MONOLITHIC_FORWARD` | `False` | 打开后额外编译**「层循环」**（`_LayerLoop`），供 `forward_layers()` 使用；默认关 |

**为什么只编译「层循环」而不编译整个模型**：`Qwen2Model.forward()` 的返回值要经过 `self.norm`（完整模型语义），而分布式分段前向在 `has_lm_head=False` 时必须返回**未过 norm** 的 raw hidden。所以只包住层循环，前置/后置仍由 `forward_layers()` 负责，语义才与逐层版一致。（第一版直接编译整段 `Qwen2Model` 得到 2.325×，但**多算了一次 `self.norm`**，argmax 从 decode 第 1 步就分叉 —— ）

**实测收益**（`USE_MONOLITHIC_FORWARD=True`；见[图 3](docs/figures/cross-frame-relay/fig3-compile-gains.png)）：

| 场景 | 逐层 | 编译层循环 | 加速 | 逐 token argmax |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B 12 层（非 hybrid，prefill 64，repeats=5） | 10.269 ms/步 | **6.133 ms/步** | **1.674×** | 一致 |
| Qwen3.5-2B 24 层（hybrid，prefill 32，repeats=3） | 41.058 ms/步 | **32.349 ms/步** | **1.269×** | 一致 |

hybrid（Qwen3.5 的 18 层 `linear_attention` + 6 层 `full_attention`）需要**按层类型分别取 mask**，`_LayerLoop` 已支持（用「元组 + 每层的 mask 索引」，便于 `torch.compile` 做 guard）。⚠️ 两组口径不同（模型/层数/prefill），**加速比不可直接比较**；hybrid 的 `linear_attention`（GatedDeltaNet）可融合点比纯 attention+MLP 少，收益偏低属合理。

**⚠️ 三条必须知道的边界**：

1. **compile 与 eager 不是逐位一致**：hidden 差异量级恰为 **f16 的 1 ULP**（`0.015625 = 2^-6`）；逐项排除后唯一剩余来源是 attention 实现通路（`fuse_attention` 把 bmm+softmax 融回 aten SDPA）。**但不能说"compile 更差"** —— 上游对照 **float64** 基线时 compile 版本的 rtol **更好**；正确表述是「**与 eager 非逐位一致**」。
2. **有「逐 token 一致」验收判据的场景不得开启 compile**（例如跨框架接力的准入判据）。
3. **Windows 需要两件事**：`PYTHONUTF8=1`（否则 torch/inductor 内部按 GBK 解码失败、**静默回退 eager**）与 [`triton-windows`](requirements-compile.txt)（可选加速，`requirements-compile.txt` 声明；**已实测可用**，官方 PyPI 无 Windows wheel，用社区构建 `triton-windows-3.8.0.post28`）。两者缺一都不会崩，只是拿不到收益。

报告：`local_docs/CORE-RELAY-XFRAME-02-a4-layer-loop-2026-09-18.json`、`…-b14-hybrid-layer-loop-2026-09-18.json`、`…-compile-numerics-2026-09-18.json`。

### 顶层透明性

TUI 与 API 顶层只需知道**聚合资源**（GPU/CPU/内存）和"是否分布式"，不必知道谁在本地、谁在远端；引擎选择按**资源 + 能力 + 目标**决定，而不是"有 GPU 就用 torch"。可选策略（隐私、带宽）尚未实现，留给后续策略票。见 [层流水线的节点类型与顶层透明性](docs/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)。

## 当前状态

| 能力 | 当前口径 |
| --- | --- |
| Textual TUI | 已接入统一 `qlh` 入口；聊天、9 个功能屏和 1 个调试兜底屏共用一个进程，可在本机按需启动后端；模型下载/搜索/预检/登记、集群配置、节点管理、日志筛选/统计/导出、设备配置、用户设置等写操作经确认框闸门；旧自绘 ANSI TUI 已归档 |
| Edge <=1B 单机 | 已有模型画像、GGUF/llama.cpp 路径和边缘预检；默认不加载 torch |
| 同机双进程 RPC | 已有 llama host + `ggml-rpc-server` 模拟及合同测试；不等同于跨机生产准入 |
| PC RPC | 有设备评分、自动层数规划、租约/断线回退和资产同步合同；真实大模型吞吐收益仍不宣称，容量收益与 D→L 分开验收 |
| 层段合同与自动重分片 | 合同、布局 fail-closed 校验、容量重解与 epoch 原子提交的开发门已完成；真实 PC/Android 故障注入、长时和性能验收待做 |
| 跨框架逐层接力（D→L） | 合同正确性已准入；主仓双引擎样本为 qwen2.5-0.5B、12+12 层、**47.501 ms/步、32 步一致**，且 **2026-09-21 复跑补齐的完整矩阵 27/27 逐 token 一致**（qwen3.5-2B K=8/12/16/20 全部 32/32、负载档位、batch 2/4、混合精度档位）；长时、跨机与多段验收待补；默认路由仍优先可直接整模运行的 L/RPC 路径 |
| PyTorch D 档 | 层拆分/层间流水线/多节点层段承载的实际实现方，兼作对照实验；不进入 Edge 默认依赖 |
| Relay R | L→L、D→L、f32/采样矩阵和 SSH 跨机证据已完成正确性验证；容量收益已量化，默认路由仍不替代可直接整模的 L/RPC |
| Android | `qlh-android` P0 交叉编译/JNI 已完成；P1 的设备运行、RPC worker、断线、热/电和安全证据未完成 |
| 运行环境 | **主运行时 `transformers` 5.17.0**（`huggingface_hub` 1.32 / `tokenizers` 0.23）；全部 PyTorch 侧车（`.venv-qwen3-sidecar` / `.venv-gemma4-pipeline`）已统一到 5.17.0；`.venv-test` 含 `triton-windows`（compile 路径可测）。Windows 原生 `torch.compile` **可用** |
| 模型资产 | 模型文件不入 Git；主仓资产清单支持 Qwen2.5-0.5B、Qwen3-0.6B、MiniCPM4-0.5B、DistilQwen2.5-DS3-0324-7B 等登记模型 |

没有标注真实设备、跨机或生产验收的实验，只能作为开发证据或 PoC 使用。

## 主仓边界

| 保留在主仓 | 外置或不再回流 |
| --- | --- |
| llama.cpp/GGUF 引擎适配、RPC/层段合同、调度与故障恢复 | Android UI/JNI 工程：`qlh-android` |
| FastAPI 控制面、模型/节点/能力合同 | 产品壳和前端：`qlh-shell` |
| 跨平台 Textual TUI、只读命令薄层、Edge 入口和质量门 | 发布/安装器：`qlh-release` |
| 单机、同机双进程和 PC/Android 主线实验接口 | Toolbox：`qlh-toolbox` |
| **PyTorch D 档引擎（层拆分、层流水线）与跨框架接力实现** | 生图、Web 产品界面、邮件、运营工作台 |
| 模型注册、下载校验、设备画像和分布式观测 | Koakumix harness 的定制实验、生图和侧车能力 |

主项目不保留生图运行时和生图资产；生图唯一归属 Koakumix。多模态不作为固定主线依赖，由模型舰队按设备能力选择文本或视觉模型。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `src/` | QLH 主代码：控制面、引擎、层段/层流水线合同、TUI（分组见下） |
| `tests/` | pytest 套件（TUI、RPC/层段、调度、合同、文档门） |
| `scripts/` | 验证、实验、环境与文档工具（`edge_preflight.py`、`android_validation.py`、`llama_rpc_*.py`、`doc_maintenance_audit.py` 等） |
| `docs/` | 现行文档；历史与已迁移内容在 `docs/archive/` |
| `schemas/` | 跨进程/跨仓 JSON Schema 合同（artifact-manifest、cluster-profile、experiment-record 等） |
| `fixtures/` | 测试固件（含离线聊天事件回放，供 `qlh chat --fixture` 使用） |
| `local_docs/` | 本地实验与验收原始记录；不作为公开源码接口 |
| `runtime/` | 运行期日志与 llama.cpp 运行时目录 |
| `qlh.py` / `qlh_edge.py` | 交互 TUI/CLI 入口与 Edge 入口 |
| `qlh.bat` / `qlh.sh` / `bjtu.*` / `koakuma.*` | 启动器；`bjtu`、`koakuma` 为兼容别名，统一入口仍是 `qlh` |
| `start_tui.*` / `start_backend.bat` / `setup_all_envs.*` | 一键启动与多环境安装脚本 |
| `requirements*.txt` / `pytest.ini` / `pyrightconfig.json` / `reasonix.toml` | 依赖清单与工具配置 |
| `models/`、`chat_history/`、`dist/`、`build/`、`test-results/`、`logs/`、`_to_delete/` | 本地产物或归档区，不入 Git（`logs/`、`_to_delete/` 已 gitignore） |

**子模块（与主仓功能直接相关，4 个）** —— 由 `.gitmodules` 登记，`git submodule update --init` 拉取：

| 子模块路径 | 远端 |
| --- | --- |
| `android/` | `qlh-android` |
| `frontend_cybergothic/` | `qlh-shell` |
| `packaging/` | `qlh-release` |
| `harness_workbench/` | `Koakumix` |

**关联仓库（开发/实验工具，2026-09-20 起不再是子模块）** —— 需**另行 clone 到本地**，已被 `.gitignore` 覆盖、**不进主仓**：

| 本地目录 | 远端 | 说明 |
| --- | --- | --- |
| `tools/docagent/` | `qlh-docagent` | 文档维护扫描器 |
| `tools/toolbox/` | `qlh-toolbox` | 工具集 |
| `tools/reasonix-codex-bridge/` | `reasonix-codex-bridge` | Reasonix ↔ Codex 受控桥（MCP + ACP） |
| `tools/dsh-codex-bridge/` | `dsh-codex-bridge` | 孪生项目（DSH 侧，vendor 了桥接器运行时） |
| `packages/spawnledger/` | `spawnledger` | 进程归属票据工具 |

> **注意**：`tools/`、`packages/`、`logs/`、`docs/agent_tool/` 以及 `scripts/` 的一次性实验脚本
> 已从主仓裁掉（只留本地）。`scripts/` 中**被 `src/` 或测试 import 的**、以及上手/流水线要用的
> 工具（`setup_envs.py`、`cut_layers.py`、`run_test_channels.py` 等）仍保留入库。


`src/` 按职责分组（便于导航；逐模块接口见 [模块接口说明](docs/模块接口说明.md)）：

| 分组 | 代表模块 |
| --- | --- |
| 控制面与入口 | `api_server.py`（FastAPI 控制面）、`api_errors.py`、`config.py`、`bootstrap.py`、`model_api_access.py`、`review.py`、`local_store.py` |
| 调度与集群 | `scheduler.py`、`scheduler_svc_http.py`、`cluster_join.py`、`cluster_transport.py`、`edge_cluster.py`、`node_config.py`、`node_runtime.py`、`transport_runtime.py`、`transport_port.py`、`network_address.py`、`network_path.py`、`proxy_config.py`、`wss_loopback.py` |
| 引擎 | `llama_engine.py`（L 档）、`model_module.py`（D 档层拆分）、`island_engine.py`、`koakuma_engine.py`、`tcp_comm.py`（torch 张量传输）、`inference_client.py`、`inference_svc_main.py`、`inference_service/`、`paged_kv_cache.py`、`external_provider.py`、`speculative*.py` |
| RPC 与层接力 | `llama_rpc_contract.py`、`llama_rpc_device.py`、`llama_rpc_planner.py`、`relay_contract.py`、`relay_planner.py`、`relay_transport.py` |
| 层段/层流水线合同 | `pipeline_node_contract.py`、`pipeline_capacity.py`、`pipeline_assignment_manifest.py`、`pipeline_model_descriptor.py`、`pipeline_reshard.py`、`cache_unit_layout.py` |
| 跨框架/多模态流水线 | `qwen3_pipeline_*.py`、`qwen3_multimodal_*.py`、`gemma4_pipeline_*.py`、`multimodal.py` |
| 模型与资产 | `model_config.py`、`model_host.py`、`model_sync.py`、`model_downloader.py`、`model_download_jobs.py`、`model_search.py`、`model_registry_validation.py`、`model_runtime_contracts.py`、`local_model_assets.py` |
| 任务图与工作流 | `task_graph*.py`、`task_journal.py`、`task_provider.py`、`task_worker_*.py`、`graph_orchestrator.py` |
| TUI 与交互 | `tui_textual.py`、`tui_api.py`、`tui_shared.py`、`tui_sse.py`、`tui_backend.py`、`tui_commands.py` |
| 设备与压测 | `device_profiler.py`、`provider_soak.py`；RAG 已外置至 `harness_workbench`/Koakumix |

## 快速开始

### 1. 获取代码和子模块

```bash
git clone https://github.com/SgfKrc/qlh.git
cd qlh
git submodule update --init --recursive     # 只拉 4 个「子模块」（见上表）
```

主仓**子模块**的远端见 `.gitmodules`。**关联仓库**（开发/实验工具，非子模块）按需另 clone 到 `tools/`、`packages/` 下：

- 子模块：`https://github.com/SgfKrc/qlh-android.git` · `qlh-shell.git` · `qlh-release.git` · `Koakumix.git`
- 关联仓库：`https://github.com/SgfKrc/qlh-docagent.git` · `qlh-toolbox.git` · `reasonix-codex-bridge.git` · `dsh-codex-bridge.git` · `spawnledger.git`

### 2. 选择运行环境

主环境带 torch，适合 D 档 PyTorch 层流水线、跨框架接力和完整 API：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The main requirement set includes the pinned CPU/GGUF `llama-cpp-python==0.3.35`
fallback alongside the PyTorch D track. This is the default CPU wheel contract;
CUDA llama.cpp builds are optional, separately built artifacts and are not implied
by the main lock.

Edge 环境只安装 GGUF/llama.cpp 和控制面依赖，不安装 torch、Transformers 或 bitsandbytes：

```powershell
python -m venv .venv-edge
.\.venv-edge\Scripts\python.exe -m pip install -r requirements-edge.txt
.\.venv-edge\Scripts\python.exe scripts/edge_preflight.py --python .venv-edge\Scripts\python.exe --json
python scripts/llama_dependency_contract.py --json
```

The managed Gemma 4 MTMD profile is separate: `.venv-gemma4-native` uses the
frozen `llama-cpp-python==0.3.28` binding and must not reuse the ordinary CPU
wheel. Its ABI marker and lock are checked independently.

Linux/macOS 将 `Scripts\python.exe` 替换为 `bin/python`。交互 TUI 依赖 `Textual`；只读命令、协议层和 CI 检查不需要 Textual。`uvicorn/FastAPI` 只在本机自动启动后端时需要，远程 TUI 不会替远端启动本机后端。

若只需要交互 TUI，可在完整环境之外安装：

```powershell
python -m pip install -r requirements-tui.txt
```

### 3. 启动 TUI

```bash
python qlh.py chat
python qlh.py chat --route distributed_preferred --thinking
python qlh.py chat --fixture fixtures/chat.json
python qlh.py status
python qlh.py models
```

`qlh chat` 会在本机后端未运行时于当前进程的 daemon 线程中启动后端，并在 Textual 启动屏中显示探活阶段。后端日志不会刷入 TUI，而是保留到既有日志文件和日志页面。`status`、`models` 等单命令不会自动启动后端；`--fixture` 是不联网的聊天事件回放路径。

写操作从外壳发起：模型屏 `L` 加载 / `U` 卸载，队列屏 `P` 暂停-恢复 / `S` 策略 / `C` 清空排队，聊天屏可用 `/model`、`/queue`、`/new`、`/resume`、`/rename`、`/sessions`、`/delete-session`、`/reset`；破坏性与长耗时操作都会先弹确认框。模型控制接口按 loopback 默认放行，远程控制主节点需主节点配置 `QLH_MODEL_API_TRUSTED_CIDRS`。

Windows 可直接使用 `qlh.bat`，Linux/macOS 可使用 `qlh.sh`。`bjtu`/`koakuma` 是兼容启动器，主仓统一入口仍是 `qlh`。

TUI 的 9 个功能屏是主交互和验收边界：模型屏负责本地资产/预设/下载任务、搜索、预检、登记、加载和卸载；分布式/节点屏负责开关、容量、最大节点、邀请、连接、入群请求码/授权消费和注销；日志屏负责筛选、统计、导出和清理；设备屏负责自动配置和 GPU 选择；设置屏负责读取和写入用户设置。最后的「调试」屏从运行中后端 `/openapi.json` 动态读取路由，仅作为尚未形成专用交互的 JSON 兜底，不计作产品功能覆盖。当前主后端 OpenAPI 快照为 152 个操作，实际数量以目标后端返回为准；流式聊天和文件上传仍由聊天页专用处理。

## 模型与分布式

模型工件、下载缓存和大型 GGUF 文件不进入 Git。模型清单和能力画像由主仓 API/TUI 管理，模型必须通过格式、摘要、架构、模板、thinking、设备预算和来源校验后才能进入可用列表。

当前推荐验证顺序：

1. 单机加载一个 <=1B GGUF，验证模板、thinking 开关和流式输出。
2. 同机启动 host 与 `ggml-rpc-server`，验证部分驻留、容量合并、输出对拍和 worker 断开。
3. 在 PC 节点完成真实 RPC、资产同步、租约和故障恢复验收。
4. 涉及跨框架或层间流水线时，先在本机复核 D→L 接力的数值一致性与性能边界。
5. 再进入 ARM64/Android worker 验收。

不能用完整模型复制、任务图整请求并行或旧 PyTorch 双机结果宣称“模型已分片”。节点故障时的小模型绕行是调度策略，不是单独的 Lite 产品。

## Android 验证

Android 工程位于外置子模块 `android/`，开发机没有 Android 真机时使用分层证据：

```powershell
# JVM 协议/状态机/能力合同测试
python scripts/android_validation.py

# 同时构建 fullDebug APK
python scripts/android_validation.py --assemble

# 连接模拟器或 adb 真机后，安装并启动控制面
python scripts/android_validation.py --assemble --install --launch --serial emulator-5554

# 输出机器可读证据
python scripts/android_validation.py --assemble --json
```

x86_64 模拟器可验证 APK、UI、权限、网络和生命周期，但不能证明 `arm64-v8a` JNI RPC worker；ARM64 AVD/QEMU 只能补 ARM 兼容性，不能替代真实手机的热/电、后台回收、弱网和长时测试。Android P1 的完整判据和 AVD/QEMU/远程 adb 方案见 [Android 验证替代路径](android/Android验证替代路径-2026-09-18.md)。旧自绘 ANSI TUI 已移入 `_to_delete/`，不要再把它作为当前交互实现或测试入口。

## 测试

测试环境建议独立于主环境：

```powershell
python -m venv .venv-test
.\.venv-test\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv-test\Scripts\python.exe -m pytest -q
```

高风险主线的定向检查：

```powershell
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_tui_textual.py tests/test_tui_write_ops.py tests/test_tui_shared.py tests/test_tui_sse.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_llama_rpc_planner.py tests/test_llama_rpc_device.py
.\.venv-test\Scripts\python.exe -m pytest -q tests/test_pipeline_node_contract.py tests/test_pipeline_reshard.py tests/test_pipeline_capacity.py
```

当前**串行全量**基线：`2991 passed / 13 skipped / 0 failed`（`-n 0`；该数字来自 `610f4b3` 后的最近完整运行记录；xdist 并发下偶有 flaky，判定请以串行为准）。

真实硬件、跨机网络、Android ARM64、性能和长时 soak 必须另外保存原始命令、环境、模型摘要、拓扑、输出和失败边界，测试绿灯本身不替代这些证据。

## 文档入口

- [主线开发计划：分布式推理与边缘优化](docs/主线开发计划-分布式推理与边缘优化-2026-09-14.md)
- [整体架构](docs/整体架构.md)
- [层段协议立项（2026-09-17）](docs/层段协议立项-2026-09-17.md)
- [层流水线的节点类型与顶层透明性](docs/层流水线节点类型与顶层透明性-可行性确认-2026-09-17.md)
- [同进程双后端接力实现与性能](docs/archive/同进程双后端接力实现与性能-2026-09-16.md)
- [跨框架接力当前有效基线与后续优化计划](docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)
- [测试质量审计（2026-09-21）：并行 flaky 实测与竞态覆盖差距](docs/archive/测试质量审计-2026-09-21.md)
- [P4.5 立项：主节点动态选举与配套分布式管理](docs/主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- [引擎单序列与并发性能对比](docs/archive/引擎单序列与并发性能对比-2026-09-16.md)
- [分布式推理并行与跨框架路线调研汇总](docs/分布式推理并行与跨框架路线调研汇总-2026-09-15.md)
- [TUI 使用指南](docs/TUI使用指南.md)
- [TUI 功能屏与调试兜底说明](docs/TUI使用指南.md#调试兜底非功能验收)
- [TUI 指令集](docs/TUI指令集.md)
- [边缘设备模拟环境计划](docs/边缘设备模拟环境计划-2026-09-15.md)
- [Android 验证替代路径](android/Android验证替代路径-2026-09-18.md)
- [基线重写方案](docs/archive/基线重写方案-2026-09-16.md)
- [模块接口说明](docs/模块接口说明.md)
- [测试与评判标准](docs/测试与评判标准.md)
- [文档状态与清理清单](docs/文档状态与清理清单.md)

历史计划和已迁移能力位于 `docs/archive/`；本地实验产物和验收原始记录位于 `local_docs/`，不作为公开源码接口。

## 许可证

主仓使用 [MIT License](LICENSE)。各子模块和上游 `llama.cpp` 保持其自身许可证与版本锁定，不因主仓引用而改变。
