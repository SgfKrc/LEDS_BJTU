# 小模型轻量推理 Harness 工作台调研与方案

> 状态：**调研完成，方案已定（S1 可启动）**；定位为**面向玩具/自用、但可复现可审计的小模型实验工作台**：核心不是再造一个通用 Agent，而是针对不同小模型做可验证的定制化适配（模板、上下文、工具协议、资源预算和角色分工），以低成本展现技术力、创新力与工程能力；不重写推理引擎，不把单一模型的经验冒充通用能力。**独立性 = 经 OpenAI 兼容 API 包装/反代对接后端，仅共享模型工件，不依赖主项目代码**
>
> 创建日期：2026-09-04
> 适用范围：计划中的"轻量 harness 子项目"（工作台形态、模型定制化、上下文管理与 API 包装层）；不覆盖引擎内核重写、完整训练平台、分布式调度内核。这里的“定制化”默认指推理时适配和配置编排，不等于未经证据的权重微调。与 [SD 1.5引擎与分布式图像生成实施计划](SD%201.5引擎与分布式图像生成实施计划.md)、[TUI 适配实施计划](TUI适配实施计划.md) 的关系见 §3、§4、§6。

---

## 1. 问题与目标

成熟框架（Ollama、llama-server、Jan、LM Studio、text-generation-webui、Open WebUI）面向通用负载：调度好、性能好、生态好，但对**小模型（1-7B 级）的短板**几乎不设防：

1. **上下文窗口"垃圾"**：小模型原生 ctx 常见 2k-32k，且量化后可用预算再缩水；长对话/长文档下，框架普遍**静默丢头**（Ollama 从最旧消息截断、无 API 通知），系统提示与工具定义先被丢弃（[ollama#14259](https://github.com/ollama/ollama/issues/14259)）。
2. **生图/多模态割裂**：生图是独立工具（ComfyUI 等），与对话上下文、会话资产不连通。
3. **本地/远端体验不统一**：本机玩与连远端集群是两套入口。

本子项目目标：做一个**轻量 harness 工作台**，把上述三件事收进一个工具——"小模型友好"的会话上下文管理 + 生图工作区 + 本地主节点/远端推理双模式。**明确不追求**在吞吐/延迟上追平 ollama/llama-server（玩具性质，性能现状即由所选后端决定）。

**额外独立性目标**：子项目与主项目**低耦合**——harness 自己不 import 主项目任何模块，通过**包装/反代成主流 API 格式（OpenAI 兼容 `/v1`）**对外暴露，后端经可插拔适配器；与主项目只共享**模型工件**（GGUF / SD 离线资产，路径约定可配）。这样子项目可独立演进、可换后端（ollama/llama-server/主节点）。

### 1.1 核心使命：小模型定制化，而不是通用 Agent

“玩具”只表示工作台的部署和依赖保持轻量，不表示把模型当成一个不可解释的黑盒。工作台要回答的是：**同一个 8 GB 级设备上，为什么 QW1.8B、Qwen3-0.6B、Gemma 小模型和其他 GGUF 模型需要不同的提示模板、上下文策略和资源参数；这些差异能否通过固定实验复现**。

定制化分为五层，每层都要产出配置、证据和回退行为：

| 层 | 定制对象 | 典型产物 | 不能假设的内容 |
|---|---|---|---|
| 身份层 | 架构、revision、量化、tokenizer、chat template、特殊 token | `model_profile`、工件摘要、模板摘要 | 文件名或模型仓库名不能证明能力 |
| 协议层 | `/v1` 与后端原生字段、停止词、thinking 开关、工具/JSON/grammar 格式 | adapter 映射和 capability probe 报告 | “能输出 JSON”不等于能可靠调用工具 |
| 行为层 | system prompt、示例、回答长度、摘要方式、拒答和修复策略 | 可版本化 prompt/policy profile | 一个模型调好的 prompt 不自动适用于另一个模型 |
| 资源层 | `n_ctx`、输入/输出预算、KV cache、batch、线程、GPU layers、CPU offload、图片分辨率 | 资源预算和 Pareto 测量 | 更大的 ctx 或更高 batch 不必然更好 |
| 角色层 | 回答、工具路由、摘要压缩、视觉描述、embedding 等职责 | 角色路由矩阵和 fallback | 不能强迫一个小模型承担所有角色 |

首批画像不追求“所有模型都支持所有功能”，而是把差异写成明确策略：

| 候选模型类型 | 首选角色 | 首轮定制重点 | 默认回退 |
|---|---|---|---|
| QW1.8B / 普通小型指令模型 | 回答、短摘要 | Qwen 消息模板、较小 `n_ctx`、结构化输出修复、host-router 工具结果回灌 | 只做普通回答，不自主发工具请求 |
| Qwen3-0.6B / LittleLamb 类工具侧车 | 工具意图/参数路由 | thinking 开关、工具 envelope、schema/拒绝/回灌/错误恢复矩阵 | 未达 verified 时退回 host-router |
| Gemma 小模型 | 回答或摘要 | Gemma chat template、stop/eos、system 指令长度和事实保持 | 使用最小 system prompt + 明确格式修复 |
| 带 mmproj 的小型多模态模型 | 图像描述/视觉问答 | 图像 token 预算、图片缩略图、缓存失效和无图 fallback | 只发送摘要文本，不伪造视觉能力 |

因此，首期不做训练和微调框架；“模型定制”优先采用**配置、模板、grammar、上下文政策、角色分工和资源编排**。未来若验证某一模型确有 LoRA/蒸馏收益，只能作为独立实验适配器，必须与基础模型、数据、许可证和回归集一起登记，不能把训练产物偷偷混入通用 profile。

### 1.2 定制化流水线与准入状态

每个模型都按同一条可重复流水线进入工作台：

```text
inspect -> profile -> adapt -> fixture eval -> resource probe -> promote/rollback
```

1. **inspect**：读取 GGUF/Safetensors 元数据、tokenizer、模板、停止词、量化和后端能力，不加载全量权重也不联网。
2. **profile**：生成版本化画像，记录可用上下文、默认生成参数、thinking/工具/多模态状态和资产摘要。
3. **adapt**：选择该模型专属的 prompt family、消息序列、grammar、tool envelope、摘要和错误修复策略。
4. **fixture eval**：在固定的短对话、长上下文、结构化输出、工具拒绝、图片描述等样本上运行，不以人工挑选样本替代矩阵。
5. **resource probe**：测量首 token、生成速度、峰值 RSS/VRAM、输入预算和缓存命中；同时记录质量变化。
6. **promote/rollback**：只有满足质量门的 profile 才能标记为 `verified`；失败则回退到 `candidate` 或 `rejected`，不允许静默换模板。

profile 状态固定为 `unknown`、`candidate`、`verified`、`rejected` 四档。`unknown` 只能走保守 host-router/普通回答路径，`candidate` 只能用于实验，`verified` 也必须绑定特定后端、量化和模板摘要；任何资产摘要变化都触发重新评测。

## 2. 调研结论（成熟框架怎么做小模型上下文）

### 2.1 引擎层杠杆（llama.cpp / llama-server）

| 杠杆 | 说明 | 对本项目的适用性 |
|---|---|---|
| `--ctx-size` 动态预算 | 小 ctx 省 KV 显存，但低于模型原生会告警；按 256 token 对齐 | 中（llama_server 模式）：harness 自拉起时可配；qlh 模式属引擎域 |
| KV cache 量化 `-ctk/-ctv`（f16/q8_0/q4_0） | 小上下文省显存的主要杠杆，质量代价可控 | 中（llama_server 模式）：harness 暴露为配置项；qlh 模式属引擎域 |
| `--cache-prompt` / `--cache-reuse N` / `--cache-ram` | 前缀/分块复用，长 prompt 预填充从秒级降到亚秒；`--cache-reuse` 对 mmproj（多模态）与 hybrid 模型强制禁用（[PR #9866](https://github.com/ggml-org/llama.cpp/pull/9866)） | 高（llama_server 模式）：**默认为 harness 透传**（默认开 cache-prompt）；**多模态禁用约束是坑**，harness 需按会话是否含图开关 |
| `--context-shift`（KV 位移） | 旧方案；已被默认禁用，因会丢系统提示、破坏聊天模板与位置编码（[#19838](https://github.com/ggml-org/llama.cpp/issues/19838)） | 不采用：新代码不得依赖 |
| `--chat-truncate` | 替代方案：按消息列表截断（保留 system），不做 KV 位移 | 中：harness 层自己做更可控 |
| RoPE/YaRN 扩展（`--rope-scaling yarn`） | 静态 YaRN 全局恒定系数：**延长对短 prompt 反而变差**；主流框架都是静态实现 | 不做：玩具不冒险，短 prompt 是主场景 |
| SWA / iSWA（`llama_kv_cache_iswa`，Gemma2/3、Phi-3、部分 Qwen） | 架构级滑窗缓存，内存省数倍；代价：前缀缓存失效影响 speculative decoding 接受率、超窗内容不连贯 | 了解即可：随模型选择自然获得，harness 不应强制 `--swa-full` |
| `--kv-evict-sink/window`、QN1 RingBuffer | 物理 KV 封顶 + 逻辑上下文增长；perplexity 基本持平 | 观察：上游实验性，本项目不自研 |

### 2.2 harness 层杠杆（应用侧上下文管理，本项目主战场）

30 轮对话（关键决策前置）实测对比（[Context Management Strategies 实测](https://dev.to/wonderlab/agent-series-22-context-engineering-deep-dive-quantifying-three-context-management-strategies-45g3)；[SKILL.state](https://arxiv.org/abs/2608.26263) 预算匹配下 0.18 / 0.52 / 0.94 同向印证）：

| 策略 | 平均 token | 早期事实召回 | 减量 |
|---|---|---|---|
| 朴素全量历史 | 2513 | **80%** | — |
| 滑窗（近 12 条） | 604 | **20%** | −76% |
| 滚动摘要 | 1289 | 50% | −49% |

**结论**：滑窗是最便宜的但长任务召回崩塌（20%），只适合短闲聊；摘要召回/节省平衡最好但会丢"为什么"（保留"决定了什么"）；纯文本侧另有两类零成本手段——**observation/tool 输出用后即弃（masking）** 与 **verbatim 压缩**（3k+ tok/s、几乎零幻觉）；LLMLingua 式 token 剪枝会删掉 slot id 等语义必需 token（[SKILL.state](https://arxiv.org/abs/2608.26263)），不做。

社区成熟组合（供借鉴）：**预算公式** `input_budget = n_ctx - max_new_tokens - fixed_overhead`（system+tools）；**触发阈值**：摘要区占用超 ~70% 时做滚动摘要、保留近 ~35% 轮 verbatim、旧 tool 输出占用超 ~55% 时清空（[car-diagnostic-agent](https://huggingface.co/spaces/build-small-hackathon/car-diagnostic-agent)）；**截断必须按轮次边界**（保住 tool_call/tool_result 配对，[aiwisdom short-term memory](https://www.aiwisdom.dev/articles/agentic-systems/short-term-memory)）；**摘要建议用更大模型**（玩具可回退同模型短上下文摘要 call，注明质量代价）；摘要写成结构化 `STATE`（what/decisions/artifacts/open/next）落盘、配 `recall()` 检索。

> **小模型专属警告（[SKILL.state](https://arxiv.org/abs/2608.26263)，EMNLP）**：结构化 STATE 在**小开源模型**（Gemma-4-31B、Qwen-3-8B）上 68% 的失败源于**过早覆盖/删除状态字段**，作者建议 grammar-constrained decoding 兜底。本项目是小模型玩具：STATE 写回必须做 **schema 校验 + 拒绝非法补丁**，且删除操作显式确认（见 §4.3）。

### 2.3 必须避开的坑（写进设计约束）

1. **Ollama 式静默截断**：harness 必须**主动告知**用户"已裁剪/已摘要"，绝不静默丢内容。
2. **`/v1` 兼容路由忽略 per-request `num_ctx`**（Ollama）：harness 自己就是 OpenAI 兼容层（[hermes-agent#43900](https://github.com/NousResearch/hermes-agent/issues/43900)），**不得把 `num_ctx` 当标准字段隐式传递**——OpenAI 格式没有统一上下文字段，harness 必须显式映射：每会话固定上下文预算（配置/会话级），进入后端时转成 QLH 原生契约或 llama-server 的 `n_ctx` 参数，并保证 `/v1` 路由与原生路由行为一致。
3. **llama.cpp 对 `prompt+gen > n_ctx` 直接报错**：预算计算必须预留 `max_new_tokens + margin`。
4. **摘要丢失"为什么"**：链路里保留一份"决策日志"（who/when/why），摘要只折叠正文。
5. **SWA/多模态混合**：`--cache-reuse` 对 mmproj 强制禁用，harness 不得假设前缀复用总能命中。
6. **后端差异不可见**（新增，基于 §4.1）：harness 经多个 adapter 对接后端，同一请求在不同后端上的可用参数不同（QLH 有多模态/路由/生图高级编辑，llama-server 可能没有）；harness 的**能力探测**（capability 通告）必须来自后端真实响应，禁止在 harness 层伪造能力（如"生图可用"须由本地执行器探测或 qlh 后端真实响应决定）。

### 2.4 模型画像是定制化的唯一事实来源

工作台不维护一组隐藏的“万能默认值”，而是为每个**模型工件 × 量化 × 后端**建立画像。画像必须可读、可 diff、可回滚，至少包括以下信息：

```yaml
profile_schema: qlh.harness.model_profile.v1
model_id: QW1.8B
revision: local-2026-09-08
artifact_sha256: <sha256>
backend: llama_server
tokenizer_digest: <digest>
chat_template_digest: <digest>
context:
  n_ctx: 4096
  input_budget: 3072
  max_new_tokens: 768
  reserve_tokens: 256
generation:
  temperature: 0.7
  top_p: 0.9
  stop: ["<|im_end|>"]
  thinking: disabled
adaptation:
  prompt_family: qwen_chat_v1
  tool_mode: host_router
  structured_output: json_repair
  summary_mode: state_schema_v1
roles: [answer, summarizer]
resources:
  kv_cache: q8_0
  gpu_layers: auto
  max_batch: 128
evidence:
  fixture_set: small-model-core-v1
  status: candidate
  production_eligible: false
```

字段可按后端扩展，但必须遵守三条规则：

- profile 记录的是**测得的能力和选择的策略**，不是模型卡宣传语；未知字段保持 `unknown`，禁止猜测。
- 画像与资产摘要、运行时版本和适配器绑定。更换量化、chat template、llama-server 或 Transformers 版本，都要生成新 revision。
- profile 不能写入用户凭据、绝对路径或远端节点私有信息；模型资产和会话数据仍归用户本人管理。

画像让“定制化”成为工程对象：可以比较两个模型的模板差异、解释一次回退原因，也可以在不改引擎的情况下复现实验结果。

## 3. 主项目现状盘点（harness 仅复用的资源：HTTP 契约 / 模型工件；不 import 代码）

| 资产 | 现状 | harness 复用方式 |
|---|---|---|
| `src/api_server.py`（FastAPI） | `/api/chat`、`/api/chat/stream`（full/fast/interactive 流式）、`/api/chat/upload`（多模态）、`/api/sessions` | **qlh_adapter 反代目标**：仅契约，即本机进程或远端 `--host` |
| `ChatRequest` 契约 | `routing_preference`（auto/local_only/distributed_preferred/distributed_required）、`max_new_tokens`、`session_id`、`image_data_urls`（≤4）、`show_thinking`、`execution_mode/task_graph` | 适配器内**最小客户端**（复制字段 + 契约单测）；OpenAI 请求 → QLH 请求的映射 |
| SD 1.5 侧车与资产 | `/api/diffusion/*`（generate/edit/distributed/grid/mixed、blobs、资产目录/下载/导入，进程内托管 `src/diffusion/service.py`）；模型资产在 `models/sd15-*/`（manifest 校验） | **工件共享**（SD 离线资产包 + manifest）：harness 本地生图执行器**直驱共享工件，不经 HTTP 契约**；远端生图经 qlh_adapter 映射 `/api/diffusion/*` |
| RAG | `/api/rag/*`（FTS5 + 有界向量 + 容量预算 + ANN 决策门） | 长文档入口经 qlh_adapter（仅远端/主节点模式；本地独立模式 S4 评估） |
| 多模态 | `llama_engine.py`（llama.cpp mtmd/mmproj）、`qwen3_multimodal_*`、QW3-VL 契约 | 经 adapter 契约；本地纯玩模式走 llama-server（`--mmproj`），注意 §2.3-5 约束 |
| `src/tui_chat.py`（Textual） | 已是"HTTP 客户端 + 终端聊天"；`--host` 缺省连本地后端；Markdown 渲染防注入 | **仅作交互形态参考**（双进程入口、防注入、done 指标语义）；harness 自带 UI 不 import |
| `src/tui_shared.py` | 路由偏好/指标/图像加载公共件 | 参考实现；客户端逻辑由 harness 自持 |
| 模型工具 | `scripts/model_tools.py`、模型导入向导（HF/ModelScope 下载验证） | **共享工件**：下载的 GGUF 落公共模型目录，harness 扫描该目录（自解析元数据），不调主项目 `/api/models` |
| 前端 | `frontend_cybergothic`（React）、Android 客户端 | 二期可参考（Web 工作台），一期不做 |

## 4. 子项目方案

### 4.1 定位与边界

- **做**：上下文管理引擎（核心差异化）+ OpenAI 兼容 API 层（包装/反代）+ 会话/资产形态 + 生图工作区 + 本地/远端一体外壳（TUI 优先）。
- **不做**：引擎/调度重写、训练微调、性能对标、Web/Android 工作台（列为二期可选项）、**不 import 主项目任何模块**（`src/`、`scripts/`、前端均不依赖）。
- **形态决策**：**独立进程 + 包装/反代层**。harness 对外暴露 OpenAI 兼容 API（`/v1/chat/completions` + `/v1/images/generations`，SSE 流式），对内经**可插拔后端适配器（adapter）**转译：
  - `llama_server`（**本地纯玩，默认**）：harness 自行拉起 llama-server 子进程，直接消费共享的 GGUF 工件（`models/` 目录，路径可配）——**零主项目代码依赖，连主项目 api_server 都不需要**；
  - `qlh`（**主节点/远端**）：最小 HTTP 客户端（仅复制 QLH 私有契约字段 + 契约单测），反代到本机或远端 api_server，换取路由/分布式/生图高级编辑/RAG/多模态能力；
  - `ollama`（可选，S5）：作为易得后端的对照组，验证"换后端"成本。
- **共享工件约定**：模型工件（GGUF、SD 离线资产包）通过**目录约定**共享（默认同 `models/` 或环境变量指定）；harness **自解析工件元数据**（GGUF 头/sidecar），不调用主项目 `/api/models` 等内部接口。
- **能力通告**：harness 的 `/v1/models` 与能力信息来自**后端适配器真实探测**（qlh 走其契约端点、llama-server 走 `/props` 等），禁止在 harness 层伪造（如无生图后端的模式不得宣称 images 可用）。
- **定制化入口**：`model_profiles/` 是模型适配的单一事实来源。请求进入后先按 `model_id + artifact_sha256 + backend + profile_revision` 选择画像，再套用上下文、模板、生成、工具和资源策略；画像缺失时只允许保守默认值和 `unknown` 能力，不允许猜测后放行。
- **角色分工**：回答模型、工具路由模型、摘要模型、视觉描述模型和 embedding 模型可以是不同工件。harness 只负责编排和证据记录，不要求一个小模型包办全部任务；工具调用默认遵循 [联网搜索与轻量 Fetch 工具调用可行性调研与分期计划](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md) 的 host-router/sidecar 准入门。

### 4.2 模块划分

```
harness_workbench/                 # 独立子项目（独立包/仓库，仅共享工件目录）
├─ api_layer/                      # OpenAI 兼容入口：/v1/chat/completions(+SSE)、/v1/images/generations、/v1/models
│  └─ mapping.py                   # OpenAI 请求/响应 ↔ adapter 内部规范（含 num_ctx 显式映射，§2.3-2）
├─ adapters/                       # 可插拔后端适配器（统一 Capability/chat/images 接口）
│  ├─ base.py                      # adapter 接口 + 能力探测契约（禁伪造，§2.3-6）
│  ├─ llama_server.py              # 本地：自拉起 llama-server 子进程，消费共享 GGUF 工件（默认）
│  ├─ qlh.py                       # 主节点：QLH api_server 最小 HTTP 客户端（仅契约；含生图/路由映射）
│  └─ ollama.py                    # 可选对照（S5）
├─ model_profiles/                 # 模型 × 量化 × 后端的版本化画像与能力状态
│  ├─ registry.py                  # 选择、校验、diff、回滚 profile
│  ├─ schema.py                    # qlh.harness.model_profile.v1
│  └─ builtin/                     # 仅放可复现的候选画像，不放秘密和绝对路径
├─ context_engine/                 # 核心：上下文管理（后端无关，作用于会话消息层）
│  ├─ budget.py                    # 预算公式 input_budget = n_ctx - max_new_tokens - overhead
│  ├─ tokenizer.py                 # 本地 BPE 估算器 + 后端精确值回填（usage 字段）
│  ├─ policy.py                    # 策略管线：pinned 事实块 + 滚动摘要 + 近 N 轮 verbatim + tool 输出 masking
│  ├─ summarize.py                 # 摘要 call（结构化 STATE 输出；大模型优先、同模型回退并标记）
│  └─ notices.py                   # 所有裁剪/摘要动作的可见通知（禁止静默截断）
├─ session/                        # 会话状态（STATE 落盘 + 决策日志 + 资产引用）
├─ image_workbench/                # 生图工作区：/v1/images 封装、图片→会话资产、缩略卡入上下文
│  ├─ local_engine.py              # 本地生图执行器（子进程直驱共享 SD 工件，文生图基线；不经 HTTP 契约）
│  └─ remote_qlh.py                # 远端生图映射（qlh adapter → /api/diffusion/*）
├─ transport/                      # 连接管理（adapter 底座：重连/超时/退避；不感知具体后端）
├─ adaptation/                     # 定制化实验编排：模板、grammar、角色和资源策略组合
│  ├─ prompt_profiles.py            # 模型专属 prompt family 与停止词
│  ├─ capability_gate.py            # unknown/candidate/verified/rejected 准入
│  └─ repair.py                     # JSON/tool/STATE 的有界修复与失败回退
├─ eval/                            # 固定 fixture、回放、资源/质量指标和 promotion 报告
│  ├─ fixtures/                     # 短对话、长上下文、工具和多模态样本
│  ├─ replay.py                      # 记录工件/模板/参数摘要，支持 deterministic replay
│  └─ report.py                      # 质量-资源 Pareto 与回退原因
├─ ui/                             # 自带 Textual 外壳（交互参考 tui_chat.py，不 import；状态栏显示预算/裁剪/生图卡片）
└─ cli.py                          # 入口：python -m harness_workbench [--backend llama_server|qlh|ollama] [--model] [--host]
```

### 4.3 上下文策略管线（S1 交付核心）

预算阈值（借鉴 §2.2 社区组合，参数可在配置中调）：

1. **输入预算**：`input_budget = n_ctx - max_new_tokens - overhead`（overhead 含 system prompt + 生图卡片 + 检查点，默认 512）。
2. **pinned 事实块**：用户显式钉住的事实/决策（`/pin`），永不裁剪。
3. **近 N 轮 verbatim**：默认保近 12 轮（可配），**按轮次边界截断**。
4. **滚动摘要**：当"旧消息 + 摘要区"占用超 70% 时触发：旧消息折叠为结构化 `STATE`（decisions/artifacts/open/next），并行写**决策日志**（who/when/why，防 §2.3-4）；触发后近轮保留区收缩至 35%。**STATE 写回前做 schema 校验**（拒绝非法/越权字段补丁；删除字段需显式确认——小模型过早覆盖是最高频失败，见 §2.2 警告）。
5. **masking**：工具/生图输出用后即弃（默认保留最近 1 次），零成本先做。
6. **可见性**：每次触发输出一行 notice（`[已折叠 2026-09-04 12:00 前 23 轮 → 摘要 214 token]`），并支持 `/ctx` 查看预算构成。

**验收**：单测覆盖阈值触发、轮次边界、pinned 不被裁剪、max_new_tokens 溢出防护；模拟 30 轮对话断言 token 曲线 ≤ 预算且无异常。

### 4.4 生图工作区（S3 交付）

- harness 只暴露 OpenAI 兼容生图接口（`POST /v1/images/generations`），本地与远端走**两条独立路径**：
  - **本地（不经 HTTP 契约）**：harness 以子进程拉起**自带本地生图执行器**（`image_workbench/local_engine.py`，diffusers、延迟导入、CUDA venv），**直接消费共享 SD 工件**（`models/sd15-*/` + manifest 校验 + 资产目录可配）——不依赖主项目 api_server，也不 import `src/diffusion/`；首期只做**文生图基线**（txt2img），img2img/inpaint/IP-Adapter/指令编辑等高级编辑**不在本地执行器首期范围**（需要时走远端 qlh，或 S5 评估以"工件 + 独立脚本"方式补）；
  - **远端**：`remote_qlh.py` 映射到 `/api/diffusion/*`（含 `distributed` 参数透传）。
- 能力通告：`/v1/images` 是否可用由**本地执行器探测**（工件在+venv 就绪）或 qlh 后端真实响应决定，禁止伪造（§2.3-6）。
- 图片作为会话资产（blob id 引用 + 缩略图 + prompt 元数据），可 `/image show <id>` 查看、`/image edit <id>`（远端走现有 img2img/InstructPix2Pix 链路）。
- 图片入多模态上下文：缩略图卡（e.g. 256px base64）+ 摘要文本，控制 token 成本；遵循 §2.3-5（mmproj 下前缀缓存收益打折）。

### 4.5 本地 / 远端双模式（= 后端适配器选择）

- **本地纯玩**（默认）：`--backend llama_server` → harness 自拉 llama-server 子进程（共享 GGUF 工件），**不依赖主项目**；多模态经 `--mmproj`；生图经本地执行器直驱共享 SD 工件（同样不经 HTTP，§4.4）。
- **主节点本地/远端**：`--backend qlh [--host http://<master>:8000]` → 同一 UI 与上下文引擎，走 QLH 契约获得路由偏好（`routing_preference` 透传）/分布式/生图高级编辑/RAG；`--host` 缺省指向本机 8000。
- **能力差异可视化**：状态栏展示当前 adapter 与能力（models/images/multimodal/routing），不静默降级（如 qlh 断连不悄悄切 llama_server）。
- 断连：transport 底座统一指数退避重连 + 现有"连接中断"提示模式。

### 4.6 定制化工作台：把技术亮点变成可验证工件

定制化不是在 UI 里堆一组滑块，而是一个受准入门约束的组合器：

```text
model profile
    + prompt/template policy
    + context budget policy
    + tool/grammar policy
    + resource policy
    -> adapter request
    -> fixture replay + resource probe
    -> evidence report -> verified/candidate/rejected
```

每次运行的 evidence 至少记录：模型工件和 profile 摘要、后端/运行时版本、输入 token 预算、实际裁剪/摘要动作、生成参数、工具 envelope、首 token 与生成速度、峰值 RSS/VRAM、质量门结果和回退原因。报告不保存用户原文和密钥；需要复现时由用户在本地重新挂载资产并使用 fixture digest。

这条路线可以形成一组有技术含量、但不依赖大规模训练的展示点：

| 技术点 | 展示的工程能力 | 可提交的证据 |
|---|---|---|
| 模型画像与能力分级 | 识别模板/量化/后端差异，拒绝伪能力 | profile diff、capability probe、准入状态变更 |
| 上下文预算账本 | 对小窗口模型做主动资源管理，避免静默丢失 | 每轮 token ledger、裁剪 notice、STATE schema 校验 |
| 角色化小模型舰队 | 用多个专长小模型替代一个“大而全”模型 | role routing matrix、fallback 和隔离日志 |
| 有界工具/结构化输出修复 | 兼容未训练工具调用的小模型，不把错误 JSON 直接交给执行器 | grammar/repair 次数、拒绝越权率、原始输出摘要 |
| 质量-资源 Pareto | 同时优化可用性、速度和内存，不以单一 tok/s 造假 | fixture 报告、p50/p95、RSS/VRAM 与质量曲线 |
| 确定性回放 | 让“模型换了之后变好/变坏”可定位 | 工件/模板/参数摘要、seed、fixture digest、差异报告 |
| 用户资产本地化 | 不把模型、会话和评测数据交给开发组托管 | 本地 profile、SQLite/文件资产范围和导出记录 |

这里的“创新”必须落在可复现的差异上：一个 profile 能解释为什么参数不同，一份报告能说明收益是否超过代价，一次失败能安全回退。没有这些证据的 prompt 猜测、手工截图或单例 Demo 不计入完成度。

### 4.7 小模型定制化的推荐策略

1. **先做模板，再做参数**：先确认消息角色、特殊 token、stop/eos 和 thinking 开关，再调 temperature、top-p、上下文和 batch；否则参数实验会把模板错误误判成模型能力。
2. **先做 host-router，再开放模型自主工具调用**：工具调用能力为 `unknown` 的模型只接收标准化结果；只有通过 schema、拒绝、回灌和错误恢复矩阵的模型才可进入 sidecar 路径。
3. **摘要与回答分离**：回答模型质量不足时，使用另一小模型或远端模型做结构化压缩；同模型回退必须标记质量风险，不把摘要当成事实源。
4. **按资源预算选模型角色**：8 GB 设备优先让低参数模型做路由/摘要，让较强模型负责最终回答；显存不足时可切 CPU/GGUF，但 profile 必须重新测量，不能沿用 GPU 结果。
5. **以失败为一等结果**：格式错误、超预算、工具拒绝、能力未知和后端断连都要有稳定错误码与可读 notice；不使用“自动多试几次直到看起来成功”的隐式策略。

## 5. 分阶段计划

| 阶段 | 交付 | 验收 |
|---|---|---|
| **S0 调研**（本次） | 本文档 | 结论已定：包装/反代层 + 双 adapter + 上下文策略管线；模型定制化和证据链列为主线 |
| **S1 上下文引擎** | `context_engine/` + 单测 | §4.3 验收达成；接 SSE 回放对 harness 消息层验证；所有裁剪有 notice |
| **S1.5 模型画像与能力探测** | `model_profiles/` + `capability_gate` + profile schema | 至少登记 QW1.8B、Qwen3-0.6B、Gemma 小模型三类候选；模板/stop/thinking/工具/多模态状态可解释；unknown 不得伪装 verified |
| **S2 API 层 + 本地 adapter** | `api_layer/` + `adapters/llama_server` + `cli.py` + TUI | `curl /v1/chat/completions` 流式自测；本地 llama-server 子进程真机对聊；**不启动主项目也全程可玩** |
| **S2.5 定制化实验台** | `adaptation/` + `eval/` + fixture/replay 报告 | 同一模型至少比较两种 prompt/template、两种上下文策略和一组资源预算；报告同时给质量、延迟和 RSS/VRAM；失败可回滚 |
| **S3 生图工作区** | `image_workbench/`（local_engine + remote_qlh） | **本地**：`/v1/images/generations` → 本地执行器直驱共享 SD 工件文生图 → 会话资产 → 多模态追问闭环（**不启动主项目 api_server**）；**远端**：经 qlh 映射 `/api/diffusion/*` 可用 |
| **S4 远端与 RAG** | qlh 远端 `--host` + 契约单测 + 注入 chunk 预算控制 | 30k 文档对话不越预算、无静默截断；远端真机对聊 + 路由偏好透传 |
| **S5 评估与收口** | `adapters/ollama` 对照、契约漂移检测、文档、Pareto 总结 | 玩具定位复核 + "换后端成本"实测：不行则砍 ollama 而不是返工；至少保留一组可公开演示的定制化前后对照 |

> 注：S3 本地生图由 harness 自带执行器消费共享 SD 工件（资产 manifest 校验借用主项目产物），远端高级编辑（img2img/inpaint/IP-Adapter/指令编辑）依赖主节点 SD 侧车（参阅 SD 1.5 计划）；S2 起每个阶段都要求"可运行 + 有接受证据"再进下一阶段；harness 自始至终不 import 主项目代码，违背即视为回归。

### 5.1 定制化实验矩阵与质量门

实验必须固定模型工件、profile revision、运行时版本和 fixture digest。每次只改变一个主要变量，并保留未参与调参的 holdout 样本，避免“为了展示而挑样本”。建议首轮矩阵如下：

| 实验维度 | 对照 | 重点指标 |
|---|---|---|
| chat template / stop | 原生模板 vs profile 模板 | 首 token、截断率、提前停止率、格式正确率 |
| thinking | 开启 vs 关闭/预算限制 | 有效回答率、输出长度、p95 延迟 |
| 上下文策略 | 全量/滑窗 vs STATE 摘要 + pinned | 早期事实召回、输入 token、摘要覆盖错误 |
| 工具协议 | host-router vs sidecar/grammar | 选择准确率、schema 合法率、拒绝越权率、修复率 |
| 资源预算 | KV/ctx/batch/GPU layers 组合 | RSS/VRAM 峰值、tok/s、OOM、质量变化 |
| 角色分工 | 单模型 vs 路由/摘要/回答组合 | 端到端延迟、失败恢复率、总资源占用 |
| 多模态 | 无图、低分辨率图、超预算图 | 图像摘要正确率、上下文成本、缓存命中 |

最低 DoD：无静默截断；结构化输出 `schema_valid_rate >= 98%`；危险工具请求拦截率 100%；引用/资产 ID 不丢失；质量指标与资源指标同时有基线和差异；同一 fixture 回放得到可解释的 diff。任何模型若只在单个 Demo 上表现好、在 holdout 上退化，状态保持 `candidate`。

## 6. 风险与约束

1. **范围蔓延**（最大风险）：不做训练、不重写引擎、不做 Web 工作台——除非单独立项。**本地生图执行器只保文生图基线**（txt2img），img2img/inpaint/IP-Adapter/指令编辑进本地执行器即视为范围蔓延（该能力经远端 qlh 提供）。
2. **本地生图重复造轮子**（与主项目 SD 侧车能力重复的取舍）：代价是 harness 侧重实现并维护 diffusers 路径、SD 安全组合与离线加载；收益是本地零契约依赖。首期以"文生图基线 + manifest 复用"封顶；若后续高级编辑本地化，优先以"独立脚本 + 共享工件"扩展而非改写主项目。
3. **与主项目契约漂移**（解耦的新增风险）：`qlh` 适配器复制了私有契约（`/api/chat`、`/api/diffusion/*` 字段），主项目改端点/字段即可能静默破坏 → 适配器**契约单测固化**（请求/响应 JSON snapshot）+ 启动时**能力探测失败即明确报错**（不猜字段），并定期对主项目契约测试做差异对照。
4. **工件路径约定脆弱**：共享模型目录（默认 `models/`）若被主项目变更布局，harness 需自解析 GGUF 元数据兜底；路径必须可配（环境变量/CLI），不得硬编码。
5. **摘要质量**：小模型摘要会丢细节；靠"决策日志 + pinned + 可回滚完整日志（落盘不裁剪）"兜底。
6. **多模态前缀缓存失效**（§2.3-5）：性能波动不当作 bug，记入已知限制。
7. **静默截断的诱惑**：所有裁剪必须产生 notice，测试断言存在。
8. **与 TUI 适配计划的边界**：harness 与主项目**无代码依赖**；`tui_chat.py` 只作为交互形态参考，其契约（双进程入口、防注入、done 事件指标语义）不被 harness 修改或许诺一致。
9. **依赖**：Textual + httpx + uvicorn（harness 自带 API 服务）；生图本地执行器需要 diffusers + CUDA venv（独立环境，参考主项目 CUDA 侧车做法，不随 harness 分发）。
10. **单模型过拟合**：为 QW1.8B 调好的 prompt/grammar 可能让 Gemma 或 Qwen3 退化 → profile 必须按模型/量化/后端隔离，并使用 holdout fixture。
11. **画像爆炸**：模型、量化、后端组合过多会难以维护 → 先维护少量可复现 profile，重复字段上移为 schema，差异通过显式 override 表达；没有证据的组合不登记为 verified。
12. **指标作秀**：只报 tok/s 或只报单次 Demo 会掩盖质量和内存代价 → S2.5 起所有报告同时给质量、延迟、RSS/VRAM、回退和失败样本，禁止只以速度晋级。
13. **错误修复掩盖能力缺陷**：无限重试或强行 JSON 修复可能把错误请求送入工具 → 修复次数有界、原始输出摘要入审计，超限直接回退 host-router/普通回答。
14. **配置与资产混淆**：把用户模型、会话或评测内容上传到开发组 → 默认本地落盘，远端仅显式发送请求；导出 profile 时去除绝对路径、凭据和原文。

### 6.1 完成度判定：展示能力必须有工程证据

本工作台可以作为技术展示载体，但展示内容必须能被第三方复跑。以下情况不计入“已完成”：

- 只展示某一次回答，没有工件摘要、profile revision 和 fixture；
- 只说“支持工具调用/多模态”，没有 capability probe 和失败回退；
- 只报告速度，没有上下文预算、峰值内存和质量对照；
- 把人工修改后的 prompt 或结果截图当成自动化能力；
- 把 `candidate` 或离线 fixture 结果写成生产准入。

达到阶段 DoD 后，工作台应能生成一份短报告：**模型是谁、为它改了什么、为什么这样改、代价是什么、失败时如何回退、用户资产在哪里**。这比堆叠更多 UI 或接入更多模型更能体现创新和工程能力。

## 7. 变更记录

- 2026-09-04：初版（调研 + 方案 + S0-S5 计划）
- 2026-09-04：v2 —— 形态改为 **OpenAI 兼容包装/反代层 + 可插拔后端适配器**（llama_server 本地默认 / qlh 主节点 / ollama 可选），子项目与主项目解耦：不 import 代码、仅共享模型工件（目录约定），风险新增契约漂移与工件路径约定
- 2026-09-04：v3 —— 生图修正：**本地生图也不经 HTTP 契约**（harness 自带本地生图执行器子进程直驱共享 SD 工件，文生图基线；img2img/inpaint/IP-Adapter/指令编辑仍经远端 qlh），新增"本地生图重复造轮子"取舍风险
- 2026-09-08：v4 —— 将**小模型定制化**提升为主线：新增模型画像 schema、能力状态、适配流水线、角色分工、实验矩阵、质量-资源 Pareto 和可复现证据要求；明确“玩具”是轻量实验载体，不是无证据的通用 Agent 或性能宣传页
