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
│  ├─ qlh.py                       # 主节点：QLH api_server 最小 HTTP 客户端（仅契约；含聊天/路由映射）
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
├─ session/                        # 会话状态（SQLite 落盘 + 决策日志 + 资产引用）
├─ rag/                            # FTS5 优先检索、可替换 embedding provider、有界上下文
├─ image_workbench/                # 生图工作区：/v1/images 封装、图片→会话资产、缩略卡入上下文
│  ├─ local_engine.py              # 本地生图执行器（子进程直驱共享 SD 工件，文生图基线；不经 HTTP 契约）
│  └─ remote_qlh.py                # 远端生图映射（qlh adapter → /api/diffusion/*）
├─ transport/                      # 连接管理（adapter 底座：重连/超时/退避；不感知具体后端）
├─ ui_react/                       # 独立 React 工作台（赛博哥特结构，青蓝/洋红主题）
└─ tui.py                          # Textual 工作台（与 React 共享 /v1 合同）
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
  - **本地（不经 HTTP 契约）**：`image_workbench/local_engine.py` 先验证 QLH 生成的 `.qlh-sd-asset.json`，再调用注入式本地执行器直接消费共享 SD 工件；首期只做**文生图基线**（txt2img），img2img/inpaint/IP-Adapter/指令编辑等高级编辑继续走远端 qlh。
  - **远端**：`remote_qlh.py` 提交 `/api/diffusion/generate`，轮询 job 并读取 `/api/diffusion/blobs/{blob_id}`；能力探测映射 `/api/diffusion/capabilities`。
- **S3.1 本票边界**：`contracts.py` 固化尺寸、步数、提示词和响应格式校验；`manifest.py` 校验资产存在性、大小及可选 SHA-256；`local_engine.py` 保持与主项目解耦。默认执行器明确返回 `local_image_runtime_unavailable`，环境中偶然存在 `torch/diffusers` 不能替代真实执行器证明。
- 图片由 `ImageAssetStore` 写入用户指定根目录，响应可返回 `b64_json` 或用户资产 URL；绝对路径不出现在 API 响应和报告中。缩略图、会话引用与多模态追问闭环进入后续票。
- 图片入多模态上下文：缩略图卡（e.g. 256px base64）+ 摘要文本，控制 token 成本；遵循 §2.3-5（mmproj 下前缀缓存收益打折）。

### 4.5 本地 / 远端双模式（= 后端适配器选择）

- **本地纯玩**（默认）：`--backend llama_server` → harness 自拉 llama-server 子进程（共享 GGUF 工件），**不依赖主项目**；多模态经 `--mmproj`；生图经本地执行器直驱共享 SD 工件（同样不经 HTTP，§4.4）。
- **主节点本地/远端**：`--backend qlh [--host http://<master>:8000]` → 同一 UI 与上下文引擎，走 QLH 契约获得路由偏好（`routing_preference` 透传）/分布式/生图高级编辑/RAG；`--host` 缺省指向本机 8000。
- **能力差异可视化**：状态栏展示当前 adapter 与能力（models/images/multimodal/routing），不静默降级（如 qlh 断连不悄悄切 llama_server）。
- 断连：transport 底座统一指数退避重连 + 现有"连接中断"提示模式。

### 4.6 React / TUI 工作台（S6 交付）

- UI 是独立壳，不复制主项目 `frontend_cybergothic` 的数据层；React 与 TUI 都只调用 harness `/v1` 合同。
- React 首屏是可用工作台：会话栏、对话流、RAG 检索抽屉、运行时能力条和用户资产入口；API 不可用时显式进入 fixture/离线状态，不伪造“已连接”。
- 视觉沿用赛博哥特的低圆角、切角、分层和克制动效，但主题从荧光绿改为**高亮青蓝 + 洋红强调 + 暗金状态色**；深色保持黑底白字，浅色保持白底黑字，状态色不承担装饰功能。
- TUI 使用 Textual，复用同一状态术语和错误码；无 Textual 时只提示可选依赖缺失，不影响 API/CLI 核心。
- UI 开发票只覆盖工作台交互和视觉，不将真实模型质量、CUDA 生图或远端长时连接写成 UI 已验收。

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
| **S1 上下文引擎** | `context_engine/` + 单测 | **Completed（本机开发门）**；预算、轮次边界、pinned/system 保护、输出 masking、STATE 校验、摘要压缩和 notice/ledger 已通过专项测试；SSE 回放接入留给 S2 |
| **S1.5 模型画像与能力探测** | `model_profiles/` + `capability_gate` + profile schema | **Completed（本机开发门）**；已登记 QW1.8B、Qwen3-0.6B、Gemma-small 候选；本地元数据、模板/stop/thinking/工具/多模态状态可解释；unknown 不得进入 autonomous tools 或 production |
| **S2 API 层 + 本地 adapter** | `api_layer/` + `adapters/llama_server` + `cli.py` + TUI | **Completed（本机开发门）**；OpenAI 请求映射、非流式/SSE、能力通告、上下文显式映射、子进程生命周期和 fake transport 已通过专项测试；真实 llama-server 二进制/模型对聊后置验收；**不启动主项目也可独立运行** |
| **S2.5 定制化实验台** | `adaptation/` + `eval/` + fixture/replay 报告 | **Completed（离线本机开发门）**；同一模型可比较两种 prompt/template、两种上下文策略和两组资源预算；报告同时给质量、延迟、RSS/VRAM、回退和 holdout；真实模型 runner 后置 |
| **S3 生图工作区** | `image_workbench/`（contracts + manifest + assets + local_engine + remote_qlh） | **开发完成（离线本机门）**：`/v1/images/generations` 契约、本地 manifest 校验、注入式本地执行器、用户资产落盘、远端 qlh job/blob 映射和 `b64_json`/URL 响应已实现；真实 diffusers/CUDA 执行器、多模态追问和高级编辑仍后置验收 |
| **S4 远端与 RAG** | `adapters/qlh.py` + `session/` + `rag/` + `/v1/rag/*` + `/v1/sessions/*` | **开发完成（离线本机门）**：QLH `/api/chat`/SSE 映射、SQLite 用户会话与资产引用、FTS5 owner scope 硬过滤、可替换 embedding 契约、有界引用上下文；真实 30k 文档预算、远端真机对聊和 nomic 长时 provider 后置验收 |
| **S5 评估与收口** | `adapters/ollama` 对照、契约漂移检测、文档、Pareto 总结 | 玩具定位复核 + "换后端成本"实测：不行则砍 ollama 而不是返工；至少保留一组可公开演示的定制化前后对照 |
| **S6 工作台 UI** | `ui_react/` + `tui.py` + UI contract tests | React 工作台与 Textual TUI 共享 `/v1` 合同；主题、fixture/offline 状态、会话/RAG/资产入口一致；真实端到端质量和长时网络仍后置 |

### 4.7 S6 开发票

| 票 | 内容 | 验收门 |
|---|---|---|
| `HARNESS-UI-01` | React/Vite 工作台壳 + Textual TUI 壳；共享导航、连接状态、主题和 fixture 状态合同 | React typecheck/build；TUI import/smoke；青蓝/洋红主题无荧光绿主色；**已完成** |
| `HARNESS-UI-02` | React/TUI 对话流与会话切换，接 `/v1/chat/completions`、SSE 和 `/v1/sessions/*` | fake API 下新建/切换/发送/断连错误可复现；不静默降级；**已完成本机开发门** |
| `HARNESS-UI-03` | RAG 工作区与引用上下文，接 `/v1/rag/*`；图片/资产抽屉接 `/v1/images/*` | owner scope、预算遗漏、资产 URL 错误状态可见；桌面/窄屏布局稳定；**已完成本机开发门** |
| `HARNESS-UI-04` | 主题/可访问性/视觉回归与 TUI parity；文档、启动脚本、独立依赖锁定 | 深浅色对比、键盘导航、减少动效、React/TUI 术语一致；不改主项目前端；**已完成本机开发门** |

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
- 2026-09-08：v5 —— 完成 **S1 上下文引擎本机开发门**：新增独立 `harness_workbench/context_engine/` 纯 Python 核心，提供预算账本、轮次裁剪、pinned 保护、工具/图片输出 masking、结构化 STATE 校验、摘要压缩、可见 notice 和 per-message ledger；专项测试 8 passed，未引入主项目代码依赖
- 2026-09-08：v6 —— 完成 **S1.5 模型画像与能力探测本机开发门**：新增 `model_profiles/` schema、用户本地 registry、QW1.8B/Qwen3-0.6B/Gemma-small 保守候选画像、静态本地工件探测和 fail-closed capability gate；专项测试 9 passed，不加载权重、不联网、不依赖主项目代码
- 2026-09-08：v7 —— 完成 **S2 API 层与 llama-server adapter 本机开发门**：新增 OpenAI 兼容 `/v1/models`、`/v1/capabilities`、`/v1/chat/completions`（含 SSE）、严格请求映射、llama-server 子进程生命周期和可注入 HTTP transport；专项测试 7 passed，真实模型对聊后置，不依赖主项目代码
- 2026-09-08：v8 —— 完成 **S2.5 定制化实验台离线开发门**：新增 adaptation 变体矩阵、内置小模型 prompt/context/resource profiles、fixture/replay runner、质量/格式/截断/回退/延迟/RSS/VRAM 指标、holdout promotion gate 和 Pareto frontier；专项测试 7 passed，不加载权重、不联网、不把 fake runner 结果写成生产能力
- 2026-09-08：v9 —— 完成 **S3 生图工作区离线开发门**：新增独立 image contracts、QLH `.qlh-sd-asset.json` manifest 校验、注入式本地 txt2img 边界、用户资产原子落盘、远端 `/api/diffusion/generate` job/blob 映射和 `/v1/images/generations` 的 `b64_json`/URL 响应；专项测试 7 passed，真实 diffusers/CUDA 执行器、多模态追问和高级编辑后置
- 2026-09-08：v10 —— 完成 **S4 远端与 RAG 离线开发门**：新增 QLH `/api/chat` 与 SSE adapter、bounded role transcript、用户-owned SQLite session/asset refs、FTS5 owner scope 检索、可替换 embedding provider 契约、有界 citation context 及 `/v1/rag/*`、`/v1/sessions/*`；专项测试 8 passed，真实远端对聊、30k 文档容量、nomic 长时 provider 和跨进程并发后置
- 2026-09-08：v11 —— 增加 **S6 React/TUI 工作台分票**：React/Vite 与 Textual 共享 `/v1` 合同，首票先做壳、fixture/offline 状态、会话/RAG/资产入口和青蓝/洋红赛博哥特主题；荧光绿不再作为主强调色。
- 2026-09-09：v12 —— 完成 **HARNESS-UI-02 对话与会话工作流本机开发门**：会话列表/新建/切换/SQLite 恢复、`/v1/chat/completions` SSE 增量渲染、消息落盘、停止生成和 Vite 本地 API 代理；真实模型质量与长时网络仍后置。
- 2026-09-09：v13 —— 完成 **HARNESS-UI-03 RAG 与图像资产工作区本机开发门**：RAG owner scope/引用上下文/预算省略可见，图像能力准入、URL-only 生成、用户-owned 资产预览和 URL 错误状态接入；真实 CUDA 采样与长时资产服务仍后置。
- 2026-09-09：v14 —— 完成 **HARNESS-UI-04 主题、可访问性、视觉回归与 TUI parity 本机开发门**：跳过链接、主内容焦点、实时区域语义、深浅色/强制颜色/减少动效规则、TUI 能力状态和 Playwright visual smoke；不改主项目前端。
- 2026-09-11：v15 —— 完成 **HW-CTX-SQZ-01 上下文进一步压榨本机开发门**：新增显式 `adaptive/state/verbatim/mask` 压缩策略、`compact/lines/nonempty` STATE 变体、memory/RAG/STATE/context 四层预算账本、owner scope 有界记忆召回和可序列化降级曲线；30 轮离线 fixture 与邻接回归 `27 passed`，未加载模型、未联网。
- 2026-09-11：v16 —— 完成 **HW-RAG-SQZ-01 harness RAG 进一步压榨本机开发门**：新增确定性查询改写、多路 FTS/可替换 embedding 混合召回与加权 RRF 去重、fixed/paragraph/sentence 分块、元数据过滤、快照失效跨会话缓存和字符/token 双预算引用边界；专项 `7 passed`，完整 harness/docagent/doc-maintenance 回归 `232 passed, 1 skipped`，未加载模型、未联网。
- 2026-09-11：v17 —— 完成 **HW-R1 榨干小模型潜力首轮研究设计门**：新增 `research/ceiling.py`，固定 5 个模型对象、6 个上限问题、6 个因素消融、v1/v2 判题口径、单模型/草稿-校验角色 Pareto 和公开证据登记；study digest、holdout、seed、artifact/profile/fixture evidence gate 可复现；专项 `6 passed`，当前不加载模型、不联网。
- 2026-09-11：v18 —— 根据当前开发机资源收口 **21～37 号票的执行策略**：仅 `QW1.8B` 可用，优先执行 `EX-CTX-MEAS-01`、工具/回放/文档类纯软件票；Qwen3-4B、DS3-7B、Qwen3-0.6B 和多模型对比票保持模型门后置，不下载、不冒烟、不以 QW1.8B 冒充替代。
- 2026-09-11：v19 —— 完成 **EX-CTX-MEAS-01 上下文策略测度开发门**：新增固定 30 轮 fixture、滑窗/STATE/长期记忆三策略的 6 档预算曲线、早期事实召回度量、JSON/Markdown/绘图 series 产物；全程 fixture-only、无模型权重、无网络。
- 2026-09-11：v20 —— 完成 **TOOL-JUDGE-POLICY-01 判题口径差异工具**：新增 v1/v2 rubric loader、同输出双政策判定、rescue/regression/invalid 分类、completion 脱敏 hash 和答辩 Markdown 报告；全程 fixture-only、无模型权重、无网络。
- 2026-09-11：v21 —— 完成 **TOOL-MANIFEST-HLTH-01 模型资产体检工具**：新增只读 manifest/lock/index、sidecar SHA 声明、磁盘元数据与仓库级 `.gitignore` 命中报告；默认不读取完整权重，显式 `verify_hash=True` 才计算 SHA-256，并固定声明无模型加载、无网络。
- 2026-09-11：v22 —— 完成 **TOOL-CTX-RESS-01 ContextPolicy 压力测试工具**：新增 30 轮 fixture 的 window/state/memory 预算矩阵、早期事实召回曲线、六项不变量检查和 JSON/Markdown CLI 报告；全程 fixture-only、无模型权重、无网络。
- 2026-09-11：v23 —— 完成 **TOOL-REDTEAM-LAB-01 红队样本演练工具**：新增四类 12 条攻击样本的逐条 block/reason 报告、1 条安全 allowlist 对照、fixture/family 过滤和 JSON/Markdown CLI；payload 脱敏、无模型权重、无网络。
- 2026-09-11：v24 —— 完成 **TOOL-TRACE-RPL-01 双机验收时间线回放工具**：将项目进展/验收清单中的 8 月 20 日分层验收、8 月 21 日重启恢复/断连重派/Tailnet IPv6 事实归一化为 4 个脱敏 fixture、13 个事件，提供场景/事件过滤及 JSON/Markdown CLI；原始地址、绝对路径和凭据 fail-closed，fixture-only、无模型权重、无网络。
- 2026-09-11：v25 —— 完成 **TOOL-PROMPT-LAB-01 多模板 A/B 渲染工具**：复用 `PromptProfile`/`render_prompt_messages()` 对固定 case 矩阵做模板字段差异、system/消息长度和确定性 token 估算；输入 schema、路径、重复 ID fail-closed，正文只保留 digest，fixture-only、无模型权重、无网络。
- 2026-09-11：v26 —— 完成 **TOOL-BENCH-LDG-01 benchmark ledger**：将 P3 控制面、物理双机 `not_run`、真实模型 `not_run` 与可扩展实验 record JSON 聚合为单机/双机/多模型答辩引用表；source/report digest、claim scope 和缺失指标保持显式，fixture-only、无模型权重、无网络。
- 2026-09-11：v27 —— 完成 **TOOL-MODEL-CARD-01 模型卡生成器**：从 manifest health、SHA sidecar、模型画像和有界 GGUF 头自动生成 JSON/Markdown；白名单提取架构、量化、上下文、tokenizer 等声明，manifest 缺口显式标记 `incomplete`，不加载 tensor、不联网、不声明模型质量或性能。
- 2026-09-11：v28 —— 完成 **TOOL-DL-RUNNER-01 下载编排器**：规范化 revision/SHA pin，提供 `.part`/Range 续传、连续失败重试、SHA sidecar、原子发布和完成后 manifest health gate；真实 HTTPS 默认禁用，fixture transport 全程无网络。
- 2026-09-11：v29 —— 完成 **TOOL-API-WB-01 API 契约工作台**：用显式 case 对照 harness `/v1` 与主项目 `/api` 的健康、模型列表、聊天、错误和 SSE 流式契约，采样状态码、错误码、响应形状和有界耗时；默认 fixture-only，报告脱敏且固定无网络、无权重加载。
- 2026-09-11：v30 —— 完成 **FUN-CLI-01 趣味 CLI**：新增零依赖 ASCII 横幅/进度样式和固定 fixture 语录卡片，提供 `qlh_say`/`model_quotes` 入口、输入 schema、脱敏与 JSON/Markdown 产物；不调用模型、不联网。

- 2026-09-11：v31 —— 完成 **DOC-S1-GAP-01 S1 摘要能力边界收口**：在 S1 实施记录显式标注当前仅有规则摘要，LLM 摘要 call 仍为规划项，并链接实现核查调研，避免把 STATE 摘要误读为模型调用已完成。

## 8. S1 实施记录

本票只实现后端无关的上下文消息层，不启动模型、不连接网络、不引入主项目运行时。核心接口如下：

- `ContextBudget`：按 `n_ctx - max_new_tokens - overhead` 计算输入预算，支持对齐，并在生成预算耗尽时 fail-closed。
- `ContextPolicy.build()`：保留 system/pinned 内容，按显式或隐式 turn 边界选择最近轮次；旧轮次进入 STATE 摘要，无法放入摘要时保留最近轮次并发出 `context.summary_omitted`。
- `ContextMessage` 与 `ContextLedgerEntry`：为每条消息保留稳定身份、轮次、token 估算、保留原因和 masking/summary 证据。
- `validate_state()` / `apply_state_patch()`：只允许 `what/decisions/artifacts/open/next` 五个字段；未知字段拒绝，删除必须显式确认。
- `ContextNotice`：所有裁剪、masking、摘要和异常都以稳定 code 对外报告，禁止静默丢失内容。
- 摘要调用边界：当前仅实现无模型的 `RuleBasedSummarizer`；**LLM 摘要 call 未实现（规划中，见[上下文压缩实现核查与摘要模型必要性](harness上下文压缩实现核查与摘要模型必要性.md)）**，S1 的 STATE 摘要不代表已接入模型。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_engine.py -q
8 passed
```

本票不声明模型真实生成能力已完成；模型画像和静态能力合同在 S1.5 单独收口，下一票进入 S2 API 层与本地 adapter。

### HW-CTX-SQZ-01 实施记录（2026-09-11）

第 18 票在既有 S1 引擎上增量实现，保持默认 `adaptive` 行为和旧三层 `LayeredBudget` 调用兼容：

- `context_engine/compression.py` 提供无模型的完整消息 verbatim 压缩、STATE 渲染变体和 `CompressionStep`；策略只在轮次边界上选择，不切片 pinned/system 或单条记忆。
- `ContextPolicyConfig` 可显式选择 `adaptive`、`state`、`verbatim`、`mask`；`adaptive` 按 STATE → verbatim → window 的顺序降级，并在每步记录 before/after token、策略、遗漏数和 warning notice。
- `LayeredBudget` 增加可选 `state_budget`，四层总和严格等于输入预算；适配变体 digest 包含压缩策略、STATE 变体、verbatim 上限和 memory recall 参数，防止实验结果在策略变化后误复用。
- `ContextPolicy.build()` 增加 `memory_query`：按 `owner_scope` 查询长期记忆，只注入能完整放入预留预算的条目并保留 entry id；无条目适配或预算不足时报告 `context.memory_recall_omitted`，不跨 scope、不静默截断。

离线验收命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_squeeze.py tests/test_harness_context_engine.py tests/test_harness_memory_retrieve.py tests/test_harness_memory_extract.py tests/test_harness_adaptation_eval.py -q
27 passed
```

本票仍不启动模型或 provider；`HW-SUMM-01` 的真实/可替换摘要模型 adapter、质量测度和 `EX-CTX-MEAS-01` 的 30 轮召回曲线属于后续票。

## 9. S1.5 实施记录

`model_profiles/` 是模型适配的单一事实来源，画像不包含用户绝对路径、凭据或会话原文：

- `schema.py` 定义 `qlh.harness.model_profile.v1`，区分 profile admission（`unknown/candidate/verified/rejected`）与 capability 状态（`unknown/declared/verified/rejected`），并对 artifact/tokenizer/template digest 做格式校验。
- `probe.py` 只读取本地配置、tokenizer、chat template、generation config 和文件清单；默认对目录权重做 inventory digest，不把 tensor 加载进框架，`hash_weights=True` 才执行完整流式权重哈希。报告固定声明 `weights_loaded=false`、`network_used=false`。
- `builtin.py` 登记 QW1.8B、Qwen3-0.6B、Gemma-small 三个候选画像，均不宣称工具调用或多模态已 verified；模型专属 prompt family、tool mode、角色和资源策略显式可 diff。
- `capability_gate.py` 允许未知模型做普通回答或 host-router，只有工具调用生成和结果回灌同时 verified 才能进入 `autonomous_tools`；目录 inventory digest 不能进入 verified/production，必须显式完成 `full_stream` 内容哈希；拒绝状态永远 fail-closed。
- `registry.py` 使用用户指定目录保存带内容摘要文件名的 JSON profile，支持精确选择、diff、promotion 和 rollback；篡改文件名摘要或 profile 内容都会被忽略/拒绝。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_engine.py tests/test_harness_model_profiles.py -q
17 passed
```

本票完成的是静态画像和准入合同，不是模型真实生成能力验收。真实模型进程、模板运行、工具调用和资源 Pareto 进入 S2.5 定制化实验台。

## 10. S2 实施记录

本票把 harness 变成可独立启动的 API 外壳，但没有把后端能力伪装成主项目能力：

- `api_layer/mapping.py` 将 OpenAI 请求严格转换为 adapter request，显式处理 `max_tokens`、`stop`、`temperature`、`top_p`、`extra_body.num_ctx` 和 `cache_prompt`；未知扩展字段拒绝，不静默吞参数。
- `api_layer/app.py` 提供 `/healthz`、`/v1/models`、`/v1/capabilities` 和 `/v1/chat/completions`。非流式响应遵循 OpenAI completion envelope；流式响应逐块输出 SSE，错误以稳定 error envelope 结束并发送 `[DONE]`。
- `adapters/base.py` 固化后端无关的 capability/model/request/response/chunk 合同；`AdapterError` 只暴露稳定 code、retryable 和状态，不泄露本地模型路径。
- `adapters/llama_server.py` 使用独立的 `llama-server` 子进程参数构造（`--model/--ctx-size/--n-predict/--jinja/--cache-prompt/--mmproj`），`num_ctx` 明确视为进程级配置；支持 `/props`、`/v1/models`、completion 和 SSE 解析，HTTP transport 可注入测试替身。
- `cli.py` 提供 `python -m harness_workbench.cli --model ...` 入口，主项目 `api_server.py` 不参与启动路径；真实二进制、权重和 CUDA 只在后置真机验收时启用。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_engine.py tests/test_harness_model_profiles.py tests/test_harness_api_layer.py -q
24 passed
```

本票已完成协议和进程工程化，不等于真实模型质量验收。下一票进入 S2.5 定制化实验台，建立 prompt/template、上下文策略和资源预算的可复现实验矩阵。

## 11. S2.5 实施记录

本票实现的是“可展示但不作秀”的定制化实验台：

- `adaptation/profiles.py` 定义 `PromptProfile`、`ContextStrategy`、`ResourceProfile` 和 `AdaptationVariant`；变体 ID 由模型 profile digest、prompt、上下文和资源配置规范化计算，不能手工覆盖。`builtin.py` 提供 Qwen/Gemma/generic 两组 prompt 对照、recent-window/STATE-summary 两组上下文策略和 8 GB safe/balanced 两组资源预算。
- `eval/fixtures.py` 冻结短回答、结构化 JSON、工具拒绝、长上下文召回和 holdout 格式/grounding 样本；fixture digest 进入报告，fixture 不携带本地路径、权重或凭据。
- `eval/replay.py` 定义可注入 `ReplayRunner`。每次回放绑定 variant、fixture digest、消息 digest、seed、延迟和资源指标；默认报告去除原始输出，固定声明 `weights_loaded=false`、`network_used=false`。
- `eval/report.py` 统计质量率、格式率、截断率、回退率、延迟 P50/P95、RSS/VRAM 峰值；holdout 不达门时保持 `candidate`；即使通过离线质量门，promotion 也只返回 `verified` candidate，生产路由仍需真实运行时验收。
- `pareto_frontier()` 同时考虑质量和资源代价，避免只用 tok/s 或单个成功 Demo 选择 profile；被更高质量且更低资源代价的变体标记为 dominated。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_engine.py tests/test_harness_model_profiles.py tests/test_harness_api_layer.py tests/test_harness_adaptation_eval.py -q
31 passed
```

当前仍未做真实 QW1.8B/Gemma/其他小模型权重回放、真实 RSS/VRAM 采样或真实 llama-server 质量结论；这些是后置真机验收，不影响本票先完成工程化实验工作台。

## 12. S3 实施记录

本票完成生图工作区的工程合同和两条后端路径，不把 fake executor 或异步 job 结果写成真实本地生图能力：

- `image_workbench/contracts.py` 固化 prompt、尺寸、步数、引导强度、seed、模型和响应格式；非法尺寸、越界参数和不受支持的响应格式在 API 边界 fail-closed。
- `image_workbench/manifest.py` 独立解析主项目资产工具生成的 `.qlh-sd-asset.json`，检查相对路径、重复项、文件存在性、大小和可选 SHA-256；不 import `src/diffusion/`，也不把绝对资产路径返回给调用方。
- `image_workbench/local_engine.py` 只负责 manifest 闸门和 executor 生命周期。当前默认 executor 明确返回 `local_image_runtime_unavailable`；测试替身可以证明请求、工件和生成结果的连接，但不能替代 CUDA/diffusers 验收。
- `image_workbench/remote_qlh.py` 将请求映射到 `/api/diffusion/generate`，轮询 `/api/diffusion/jobs/{job_id}`，读取 `/api/diffusion/blobs/{blob_id}` 或直接解析 base64；能力探测只接受 QLH `/api/diffusion/capabilities` 的真实响应。
- `ImageAssetStore` 将图片和最小 prompt/尺寸/seed 元数据写入用户指定根目录，先写临时文件再原子替换，读回时重新校验 SHA-256。`api_layer/app.py` 新增 `/v1/images/capabilities`、`/v1/images/generations` 和资产读取端点。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_image_workbench.py -q
7 passed
```

本票仍未完成真实本地 diffusers/CUDA executor、实际 SD 采样质量、缩略图/会话多模态回灌和本地高级编辑；这些能力进入后续票并要求真机验收。

## 13. S4 实施记录

本票完成远端主节点和本地知识/会话边界，不把主项目数据库或远端状态偷偷变成 harness 的隐式依赖：

- `adapters/qlh.py` 探测 `/api/status`，将 OpenAI 风格的 messages 明确序列化为有界 role transcript，再映射 `/api/chat`；SSE 同时支持逐 token 和 `done.response` 完整回答，异常保持稳定错误码。QLH 的路由偏好、外部数据授权和客户端类型由配置显式控制。
- `session/store.py` 使用用户指定的 SQLite WAL 文件保存会话、消息和资产引用。返回对象只有 session/message/asset ID、scope 和元数据，不暴露本地绝对路径；scope 不匹配时硬拒绝读取。
- `rag/store.py` 默认 FTS5，source/chunk ID 由内容摘要稳定生成，owner scope 在 SQL 查询中硬过滤；`providers.py` 只定义 embedding provider，不因为存在普通聊天模型就宣称 embedding 可用。
- `rag/context.py` 按完整 chunk 组装引用上下文，超出预算时返回 `omitted_count` 和 `truncated=true`，不静默截断；API 提供 `/v1/rag/health`、`/v1/rag/sources`、`/v1/rag/search` 与 `/v1/sessions`、`/v1/sessions/{id}/messages`、`/v1/sessions/{id}/assets`。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_s4_remote_rag.py -q
8 passed
```

真实 QLH 主节点、30k 文档容量与长时 embedding provider 仍是后置环境验收；本票不宣称网络可达、nomic 质量或跨进程压力已通过。

### HW-RAG-SQZ-01 实施记录（2026-09-11）

本票只在 harness RAG 检索管线层做确定性增强，不训练或微调 embedding，不启动模型，不连接网络：

- `rag/query.py` 对查询做 NFKC/空白归一化，并按有限别名表生成最多 8 路变体；原始查询、规范化查询和变体均进入检索结果记录，便于复核改写影响。
- `rag/retriever.py` 的 `HybridRagRetriever` 将每路 FTS 结果与可选 `EmbeddingProvider` 候选以加权 RRF 融合，按 chunk ID 去重；provider 维度/数量异常或运行失败时降级到 FTS，并保留 `route_counts`/`candidate_count` 证据。
- `rag/store.py` 增加固定/段落/句子三种分块粒度、`source_ids`/`title_prefix` 过滤和 owner scope 硬隔离；查询缓存以配置和文档快照摘要为键，跨 session 可复用，任一文档变更即失效。
- `rag/context.py` 对完整 chunk 同时施加字符与 token 预算，引用只与实际纳入块对应；超限块逐项记录 `chars`、`tokens` 或 `invalid` 原因，不静默截断。HTTP API 与 MCP 的增强入口均为显式依赖注入，旧 FTS 默认路径保持兼容。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_rag_squeeze.py -q
7 passed
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_s4_remote_rag.py -q
9 passed
```

本票完成的是离线检索管线、引用边界和可复用缓存开发门；真实 embedding 长时质量、30k 文档容量、跨进程并发与主项目 `src/rag_store.py` 双侧 hit@5/MRR 基准进入后续 `RAG-BASE-01` 等票，不把 fake provider 结果宣称为生产质量。

### HW-R1 实施记录（2026-09-11）

本票完成的是“上限研究如何被正确测量”的合同，不执行真实模型冒烟：当前开发机缺少合适小模型且性能一般，研究模块严格保持 model-free。

- `research/ceiling.py` 的 `CeilingStudyPlan` 固定五个研究对象：Qwen2.5-0.5B、Qwen3-0.6B、MiniCPM4-0.5B、QW1.8B、DS3-0324-7B；计划 digest 绑定问题、因素、实验单元、fixture digest、v2 policy 和 seed。
- 六个首轮因素为 context、template、memory、role、quantization、policy；每个模型生成 baseline 和单因素 ablation，另生成 v1/v2 判题对照与 single/draft_verify 角色对照。所有单元要求 holdout，不允许把 prompt、角色、量化和上下文同时改变后归因。
- `compare_factor()` 只输出方向性 delta；`pareto_points()` 只标记被支配点，不选择生产赢家；`EvidenceRecord` 明确记录 profile/artifact/runtime/fixture digest、runner、权重和网络状态，fixture/injected 结果通过 `evaluate_evidence_gate()` 保持 candidate。
- 公开证据登记仅用于定义可核对主张： [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)、[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)、[MiniCPM 官方项目资料](https://github.com/OpenBMB/MiniCPM)、[llama.cpp quantization README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)。公开分数、模型卡和量化文件大小都不能替代本机同口径实测。

验证命令：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_ceiling_research.py -q
6 passed
```

本票不宣称任何模型已加载、质量已提升、真实 RSS/VRAM 或 tok/s 已测量，也不改变既有 R1/DS3 生产引用点；按当前 QW1.8B-only 资源门，下一票优先进入 `EX-CTX-MEAS-01`，真实三轮标定和公开证据复核仍后置。

### EX-CTX-MEAS-01 实施记录（2026-09-11）

本票测量的是上下文装配策略对早期事实的保留能力，不执行 QW1.8B 生成，不把 fixture 曲线当作模型质量。

- `research/context_measure.py` 固定 `context-30-round-early-facts`：30 轮、3 条早期事实、固定 `seed=17`；输入预算为 64/96/128/192/256/384 token。
- 三条策略均复用既有 `ContextPolicy`：`window` 只保留最近轮次，`state` 使用 schema-validated rule-based STATE，`memory` 使用隔离 SQLite FTS memory 写入/召回；每个 cell 记录输入、STATE、记忆和近期消息 token，以及遗漏/写入/召回数量。
- `ContextMeasureReport` 的 `as_dict()` 输出 digest、provenance 和 chart-ready `series`；`to_markdown()` 输出可直接放入论文/答辩材料的表格。`runner_kind=fixture`、`weights_loaded=false`、`network_used=false` 是硬编码边界。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_context_measure.py -q
5 passed
```

本票的早期事实召回率是 context policy 的离线装配指标；真实 QW1.8B 回答正确率、长时延迟、RSS/VRAM 与生产路由继续后置。下一票进入 `TOOL-JUDGE-POLICY-01`。

### TOOL-JUDGE-POLICY-01 实施记录（2026-09-11）

本票把“判题口径问题”独立成可复现的工具，不运行模型，不把 rubric 通过率解释为模型能力。

- `tools/judge_policy.py` 固定 v1 `normalized_contains`/192 与 v2 `loose_contains`/512；v2 保留时间表达（如 `13时54分`）和独立中文数字归一化，并从答案标记/末行提取候选。
- `load_judge_rubric()` 支持 mapping 或 JSON 文件、correctness 条目筛选和 SHA-256 门；`run_judge_policy_diff()` 对公共 prompt ID 生成 v1/v2 状态、rescue/regression/invalid 原因和计数。completion 只参与内存判定，报告仅保留每条输出的 SHA-256。
- `JudgePolicyReport.to_markdown()` 输出可答辩表格；内置 fixture 同一批 4 条结果为 v1 `2/4`、v2 `4/4`、rescue `2`。实际项目 v1/v2 rubric prompt-set 不一致时报告显式标记 `prompt_set_match=false`，不隐藏该混杂因素。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_judge_policy.py -q
5 passed
```

本票只证明判题工具可复现、可审计和能展示口径差异；真实模型三轮重标、人工复核、质量门升级继续后置。下一票进入 `TOOL-MANIFEST-HLTH-01`。

### TOOL-MANIFEST-HLTH-01 实施记录（2026-09-11）

本票把“模型文件在不在、清单是否自洽、是否被版本控制忽略”独立成只读体检工具，不加载权重、不联网，也不把静态资产声明解释为模型可用性或质量结论。

- `tools/manifest_health.py` 提供 `scan_manifest_health()`/`build_manifest_health_report()`；扫描模型后缀文件、`.sha256` sidecar/聚合声明、`.manifest.json`/`.lock.json`/`model.safetensors.index.json`，检查安全相对路径、文件存在性、尺寸和可选完整 SHA-256。
- `.gitignore` 按仓库根路径匹配，报告每个模型文件的 `ignored`/命中规则、字节数和 SHA 状态；默认只读 sidecar 声明，避免在 QW1.8B-only 开发机上误触多 GB 权重读取。symlink/junction 会被跳过并单独记录。
- 报告 schema 为 `qlh.harness.manifest_health.v1`，提供稳定 digest、JSON/Markdown 产物和固定边界 `read_only=true`、`network_used=false`、`weights_loaded=false`。

当前本机静态结果：`models/` 扫描 22 个文件、识别 3 个模型资产且全部命中 `.gitignore`；Qwen GGUF sidecar 为 `declared`，两片 safetensors 为 `aggregate_declared`，safetensors index 检查 195 个条目通过；Gemma lock 引用的 `main_gguf`/`mmproj` 缺失，因此报告明确为不通过。该结果只说明资产缺口，不替代模型下载、加载和质量验收。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_manifest_health.py -q
6 passed
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_manifest_health.py tests/test_harness_judge_policy.py tests/test_harness_tool_context.py tests/test_harness_network_tools.py tests/test_harness_remote_tool.py -q
36 passed
```

本票完成静态资产与发布边界体检；下一票进入 `TOOL-CTX-RESS-01`，继续保持 QW1.8B-only、无模型冒烟约束。

### TOOL-CTX-RESS-01 实施记录（2026-09-11）

本票把 `ContextPolicy` 的 30 轮会话压测包装为可重复 CLI 和答辩报告，度量上下文装配行为，不运行模型、不联网，也不把早期事实召回率写成模型质量。

- `tools/ctx_ressure.py` 复用 `research/context_measure.py` 的固定 fixture、确定性 tokenizer、规则摘要器和隔离 SQLite memory store，默认按 64/96/128/192/256/384 token 运行 `window`、`state`、`memory` 三策略。
- `ContextPressureReport` 输出每个 cell 的输入 token、遗漏消息、早期事实召回、记忆写入/召回和 `compression_strategy`，并执行 `cells_complete`、`input_budget_bound`、`fixture_provenance`、`recall_curve_monotonic`、`memory_extract_recall_bound`、`folding_observed` 六项检查。
- CLI：`python -m harness_workbench.tools.ctx_ressure --json build/ctx-ressure.json --markdown build/ctx-ressure.md`；未给输出参数时打印 Markdown，退出码在不变量失败时为 1。顶层 `harness_workbench.tools` 保留兼容导出，模块入口使用 lazy import，避免 `python -m` 预加载警告。

当前默认矩阵 18 个 cell 全部通过：`window` 在本 fixture 未召回早期事实，`state` 首次达到 3/3 召回的预算为 128，`memory` 首次达到 3/3 的预算为 192。专项及相邻回归：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_ctx_ressure.py -q
6 passed
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_ctx_ressure.py tests/test_harness_manifest_health.py tests/test_harness_judge_policy.py tests/test_harness_tool_context.py tests/test_harness_network_tools.py tests/test_harness_remote_tool.py -q
42 passed
```

本票完成压力矩阵、折叠边界和早召回报告入口；真实 QW1.8B 回答正确率、长时资源和生产路由仍后置。下一票进入 `TOOL-REDTEAM-LAB-01`。

### TOOL-REDTEAM-LAB-01 实施记录（2026-09-11）

本票把既有 `eval/red_team.py` 安全门变成可逐条复演的离线实验台，不调用模型、不执行真实工具、不联网，也不把攻击 payload 写入报告。

- `tools/red_team_lab.py` 默认运行 12 条内置攻击 fixture：prompt injection、tool authorization、image path、context injection 各类均覆盖；另加 `safe-tool-local-v1`，只在 `web_search` allowlist、`local` scope、profile/capability `verified` 且 `production_eligible=true` 时放行。
- `RedTeamLabDecision` 比较 expected 与 observed 的 block/allow/reason，标记 `blocked`、`allowed`、`mismatch`；`RedTeamLabReport` 输出 fixture digest、schema、未授权放行计数和脱敏决策表，不输出 payload。
- CLI：`python -m harness_workbench.tools.red_team_lab --json build/red-team-lab.json --markdown build/red-team-lab.md`；可用 `--list`、`--family`、`--fixture`、`--without-safe` 缩小演练范围，期望漂移时退出码为 1。

当前默认演练为 12/12 攻击拦截、1 条安全调用放行、`mismatch=0`、`unauthorized_pass=0`。专项及原红队回归：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_red_team_lab.py tests/test_harness_red_team.py -q
11 passed
```

本票完成安全门的逐条演练和答辩报告入口；真实模型生成安全性、真实工具执行和生产授权仍需后置验收。下一票进入 `TOOL-TRACE-RPL-01`。

### TOOL-TRACE-RPL-01 实施记录（2026-09-11）

本票把“答辩时如何复盘双机验收”独立成只读、可复现的时间线回放工具。仓库当前没有原始 8 月日志流，因此内置数据明确标记为基于文档证据的 normalized fixture，不伪造实时日志采集或完整原始日志。

- `tools/trace_replay.py` 固定四条场景：8 月 20 日 QW1.8B layer pipeline、Full Worker controlled restart/recovery、remote disconnect → expired → one reassignment → master fallback completion、Tailnet IPv6 dual-machine completion。事件保留 timestamp、kind、actor、status、summary、sequence 和稳定 hash workflow ref，不输出原始 workflow ID、地址或路径。
- `load_trace_events()`/`--input` 接收规范化 JSON 事件；输入中出现绝对路径、IPv4/IPv6、token/secret/password 等字段会在报告前拒绝。`TraceReplayReport` 检查事件顺序、终态闭合、来源相对路径、载荷脱敏和 fixture/offline 边界。
- CLI：`python -m harness_workbench.tools.trace_replay --list`、`--scenario`、`--kind`、`--date`、`--input`、`--json` 和 `--markdown`；默认报告适合直接贴入答辩材料，五项检查全部有显式结果。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_trace_replay.py -q
10 passed
```

当前默认回放为 4 个场景、13 个事件，全部检查通过；未启动服务、未访问网络、未加载 QW1.8B。该工具只证明既有验收事实可以稳定复演和脱敏展示，不替代原始日志归档、长时双机复验或真实模型质量验收。下一票进入 `TOOL-PROMPT-LAB-01`。

### TOOL-PROMPT-LAB-01 实施记录（2026-09-11）

本票把 PromptProfile 的模板差异和渲染成本做成可复现、可脱敏的 A/B 报告，不执行模型生成。默认使用 QW1.8B 内置 profile 家族的 minimal/structured 两个 profile 和 3 个固定对话 case。

- `tools/prompt_lab.py` 复用 `PromptProfile` 与 `render_prompt_messages()`，对所有 profile × case 组合渲染；每个结果保留 profile/case ID、输入/渲染 digest、system 注入次数、消息数、字符数和 `HeuristicTokenizer` 估算 token，不保存 system prompt 或消息正文。
- `PromptProfileDiff` 逐字段对比 family/version/system_prompt/stop/thinking/tool_mode/structured_output；system prompt/stop 的变化只展示字符数、数量和 digest。`PromptCaseDelta` 给出每个 case 的 B-A 字符/token 差异，避免把长度变化误报为质量收益。
- CLI：`python -m harness_workbench.tools.prompt_lab --list`、`--profile`、`--case`、`--profile-file`、`--case-file`、`--json`、`--markdown`；规范化 JSON 使用 `qlh.prompt_lab.v1`，原文、绝对路径、未知 schema 和重复 ID 均拒绝。
- 报告固定检查矩阵完整、身份引用、system 只注入一次、profile digest 有差异、字段 diff 存在、payload 省略和 offline 边界；默认 2 × 3 矩阵为 6 个渲染单元。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_prompt_lab.py -q
9 passed
```

当前默认 A/B 差异只有 system prompt 与 structured-output policy；structured profile 在三个 case 各增加 27 字符、3 个估算 token。该数字是确定性估算，不代表真实 tokenizer、模型质量或延迟收益。下一票进入 `TOOL-BENCH-LDG-01`。

### TOOL-BENCH-LDG-01 实施记录（2026-09-11）

本票把散落的结构化实验结果整理为答辩可引用 ledger，不解析自由文本或启动任何运行时。默认输入为既有 P3 控制面 benchmark 和真实模型性能 `not_run` 合同。

- `tools/benchmark_ledger.py` 支持 `qlh.defense_benchmark.v1`、`qlh.real_model_performance.v1`、通用 `qlh.experiment_record.v1` 与 `qlh.benchmark_ledger.v1 records[]`；按 claim class 聚合 `single_host`、`dual_host`、`not_run` 和 `multi_model`，指标只从 JSON 数值字段读取。
- 每条 `BenchmarkRecord` 保留 source 相对路径、source digest、模型/拓扑/host/process/sample、status、有限 metrics、claim scope 和 `eligible_for_claim`。P3 的控制面 `throughput_tasks_per_second` 仍明确是 tasks/s；真实模型 TTFT/tokens/s 与物理双机缺失保持 `NOT RUN`。
- CLI：`python -m harness_workbench.tools.benchmark_ledger` 默认读取 P3 与 real-model-not-run；`--input PATH` 可重复添加文件，`--root PATH` 递归扫描并忽略不支持 schema，`--strict` 对显式输入 fail-closed；`--json`/`--markdown` 输出汇总表。
- 安全与资格检查拒绝绝对路径、IPv4/IPv6、凭据、未知 schema、非有限数字和重复 record ID；报告固定验证 source 相对路径、数值有限、claim scope、跨模型聚合及离线边界。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_benchmark_ledger.py -q
10 passed
```

当前默认 ledger 为 4 条记录：2 条单机控制面 fixture、1 条物理双机 `not_run`、1 条真实模型 `not_run`；未启动服务、未访问网络、未加载 QW1.8B。该表只提供结构化证据索引，不将控制面指标解释为模型性能。下一票进入 `TOOL-MODEL-CARD-01`。

### TOOL-MODEL-CARD-01 实施记录（2026-09-11）

本票把模型工件身份与发布缺口生成为可直接用于答辩的模型卡，不调用推理引擎。卡片中的字段来源保持可追踪，未知能力继续为 `unknown`，文件名推断明确标为 hint。

- `tools/model_card.py` 复用 `scan_manifest_health()`，将工件大小、ignore 命中、SHA sidecar 状态、manifest/lock/index 结果和 `ModelProfile` 身份汇入 `qlh.harness.model_card.v1`。证据仅保留仓库相对路径和 digest。
- GGUF 解析器最多读取有界头部，按 GGUF v3 类型结构跳过 tokenizer 大数组，只物化白名单键；tensor data 不解析。当前 QW1.8B 识别为 GGUF v3、195 tensors、20 项元数据、`qwen` 架构、`Q4_K_M`、`gpt2` tokenizer、声明上下文 8192。
- 内置 QW1.8B profile 记录 `builtin-qw1-v1`、`llama_server`、`candidate`、`production_eligible=false` 和运行配置上下文 4096；卡片同时展示 GGUF 8192 与运行配置 4096 的差异，不把声明窗口写成已经验证的有效上下文。
- CLI：`python -m harness_workbench.tools.model_card --root models --model-id QW1.8B --json build/model-card/latest.json --markdown build/model-card/latest.md`；支持 `--health-json` 与 `--profile`，结构错误、绝对路径和未知 schema fail-closed。
- 当前模型卡结构有效，但资产状态为 `incomplete`：22 个文件中识别 3 个权重工件和 2 个 manifest，Gemma lock 引用的 `main_gguf`/`mmproj` 缺失。该状态保留在 Markdown，而不是隐去非目标模型的发布缺口。

离线验收：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_model_card.py -q
11 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
194 passed, 1 skipped
```

本票固定 `runner_kind=metadata`、`weights_loaded=false`、`network_used=false`；未测 QW1.8B 的回答质量、TTFT、tokens/s、RSS 或 VRAM。下一票进入 `TOOL-DL-RUNNER-01`。

### TOOL-DL-RUNNER-01 实施记录（2026-09-11）

本票把模型工件下载从一次性脚本收口为可审计、可恢复的传输契约；本轮只验证离线 fixture，不连接公网或下载新模型。

- `tools/download_runner.py` 接收 `qlh.download_manifest.v1`：模型 ID、非浮动 revision、每个文件的 HTTPS URL、相对目标路径、size 和 SHA-256 必须齐全；`source_base_url` 只能拼出 HTTPS 无凭据 URL。
- `DownloadRunner` 先检查已有目标：size/SHA 匹配则补写 sidecar 并返回 `already_ready`，不匹配则拒绝覆盖。新传输写 `.part`/`.part.json`，每次请求带有界 `Range`；成功分片会重置连续失败计数，网络临时错误按 `max_attempts` 重试。
- staging 完整后计算全量 SHA-256，随后使用 `os.replace` 原子发布目标和 sidecar；状态文件只在发布成功后删除。目标/根目录/staging 路径经过 symlink/junction 与路径穿越检查。
- 发布后调用 `scan_manifest_health()`；体检 error 会让报告的 `post_download_health` 和整体 `valid` 失败，不允许带着缺口进入注册。报告保留每文件状态、attempts、是否续传、hash、sidecar、health digest 和限制项。
- `MemoryDownloadTransport` 可注入 Range、短响应和 5xx/timeout 失败；`UrllibDownloadTransport(enabled=False)` 是默认安全边界。CLI 默认 `--manifest PATH` 只产出 `planned` JSON/Markdown，`--execute` 没有显式 transport 仍保持 blocked。

离线验收：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_download_runner.py -q
13 passed
```

本票固定 `weights_loaded=false`；本轮未执行真实 HTTPS、未下载 DS3/Qwen3 等缺失模型、未运行任何模型冒烟。下一票进入 `TOOL-API-WB-01`。

### TOOL-API-WB-01 实施记录（2026-09-11）

本票把两个已有 HTTP 面的最小契约差异收口为可复现、可审计的离线工作台；不启动 harness 或主项目服务，也不通过 API 触发 QW1.8B 推理。

- `harness_workbench/tools/api_workbench.py` 用 `ProbeCase` 固定 5 组映射：`/healthz` ↔ `/api/health`、`/v1/models` ↔ `/api/models`、OpenAI `/v1/chat/completions` ↔ `/api/chat`、无效请求和 `/v1/chat/completions` SSE ↔ `/api/chat/stream`。
- `MemoryAPITransport` 提供稳定 fixture；`UrllibAPITransport` 默认禁用，只有调用方显式开启才允许 HTTP。相对路径、非凭据 header、payload 安全边界和输入 schema 均 fail-closed。
- 报告只保留请求 digest、方法/路径/脱敏形状、状态码、错误码、响应形状、content type 和非负耗时；不写入 prompt、message、响应正文、凭据、绝对地址或模型质量指标。
- 默认 fixture 结果为 `4 matched / 1 drifted / 0 failed`。唯一漂移是无效聊天：harness 返回 `400 invalid_messages`，主项目返回 `422 validation_error`；这被显式记录为 `status_code; error_code`，不是静默吞掉。两端流式完成事件归一化后匹配。

离线验收：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_api_workbench.py -q
9 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
216 passed, 1 skipped
```

本票固定 `runner_kind=fixture`、`network_used=false`、`weights_loaded=false`；没有真实网络请求、模型加载或质量/延迟结论。下一票进入 `FUN-CLI-01`。

### FUN-CLI-01 实施记录（2026-09-11）

本票把答辩开场需要的趣味输出收口为纯标准库、可复现的 fixture 工具；语录不是模型实测结果，不触发 QW1.8B，也不连接 API。

- `harness_workbench/tools/fun_cli.py` 提供 `say` 与 `quotes` 子命令。`say` 支持 `cyber/classic/minimal` 三种 ASCII banner、`bar/blocks/dots/steps/none` 五种进度样式、消息 digest 和 JSON/Markdown 输出。
- `quotes` 默认输出 3 张固定卡片，并支持 `qlh.fun_quotes.v1` 输入和 `--model` 筛选。卡片保留脱敏后的 prompt/quote、digest、source/claim scope；URL、地址和常见凭据在终端/报告中均被替换。
- `scripts/qlh_say.py` 与 `scripts/model_quotes.py` 是直接入口，仓库根目录可运行；报告固定 `network_used=false`、`weights_loaded=false`、`model_invoked=false`，不声明任何生成质量。

离线验收：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_fun_cli.py -q
10 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
226 passed, 1 skipped
```

本票完成 F1/F2 趣味工具合并；下一票进入 `DOC-S1-GAP-01`。

### DOC-S1-GAP-01 实施记录（2026-09-11）

本票是文档口径收口，不改代码、不启动模型、不联网。

- 在 `## 8. S1 实施记录` 增加摘要调用边界：当前 `RuleBasedSummarizer` 是唯一实现，LLM 摘要 call 尚未实现，仍属于规划项。
- 该表述与[上下文压缩实现核查与摘要模型必要性](harness上下文压缩实现核查与摘要模型必要性.md) §1/§2 结论一致，保留 `HW-SUMM-01` 作为后续模型 adapter 票，避免把 STATE 规则压缩误报为模型能力。

## 14. S6-HARNESS-UI-01 实施记录

本票实现工作台的第一层交互壳，不绑定真实模型进程：

- `ui_react/` 是独立 Vite/React 包，首屏直接进入工作台，包含会话侧栏、对话流、RAG 检索入口、运行时能力面板和资产入口；接口失败时显示离线/fixture 状态，不能把 mock 状态标为在线。
- 视觉契约采用黑/白底、低圆角、切角分层和少量动效；强调色使用高亮青蓝 `#63e6ff`、洋红 `#ff5bd7`，状态色使用暗金/琥珀，明确不使用荧光绿作为主色。深色和浅色均通过文字/边框对比保持可读。
- `tui.py` 提供 Textual 工作台壳，连接状态、会话、对话和 RAG 术语与 React 共用；Textual 不可用时只返回可操作的依赖提示。
- 本票不声明真实 QLH、模型权重、图像采样或 RAG provider 已通过 UI 端到端验收；后续票接入 fake API 后再做交互回归。

验证证据：

```text
ui_react: npm run build       # tsc --noEmit + vite build 通过
TUI/UI 合同: tests/test_harness_ui.py 2 passed
Playwright: 1440x900 与 390x844 截图通过；scrollWidth == innerWidth，无横向溢出
```

## 15. S6-HARNESS-UI-02 实施记录

本票把 UI-01 的交互壳接入可恢复的本地会话工作流，并保持离线状态可辨识：

- `SessionStore.list()` 和 `GET /v1/sessions` 按 `owner_scope`、更新时间倒序返回有界会话摘要；React 侧栏支持新建、切换和从 SQLite 恢复消息，不暴露本地路径。
- `data.ts` 增加会话 CRUD 边界和 `streamChat()` SSE 解析器，严格消费 OpenAI 兼容的 `choices[].delta.content`，错误或空流不会伪造助手回答；Vite 开发服务器将 `/healthz`、`/v1` 代理到本地 harness API。
- 对话发送先持久化用户消息，再以增量占位渲染助手输出，完成后落盘助手消息。停止按钮通过 `AbortController` 中止流并留下系统状态；API 不可用时仍显式使用 fixture，不把 fixture 标为在线。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_api_layer.py tests/test_harness_s4_remote_rag.py tests/test_harness_ui.py -q
17 passed
ui_react: npm run build       # tsc --noEmit + vite build 通过
```

本票完成的是本机协议、状态恢复和 UI 工作流开发门；真实 QW1.8B/Gemma 输出质量、跨设备 SSE 稳定性、长时断线重连和生产资产回灌仍进入后置验收。下一票为 `HARNESS-UI-03`，聚焦 RAG 引用工作区与图片/资产抽屉。

## 16. S6-HARNESS-UI-03 实施记录

本票把 RAG 与图像资产从占位入口接入真实 API，并保持能力和资产边界显式：

- RAG 工作区调用 `/v1/rag/health` 与 `/v1/rag/search`，固定 `owner_scope=local`，显示 FTS5 后端、命中分数、source/chunk 引用、上下文文本、included/omitted 数量和预算截断状态；API 失败时只显示错误，不伪造结果。
- 图像工作区调用 `/v1/images/capabilities` 和 `/v1/images/generations`，只在 `runtime_available && supports_txt2img` 时启用生成；请求固定使用 `response_format=url`，缺少 `asset_id`/URL 或返回 503 时保持失败状态，不回退为 fixture 图片。
- 成功图像通过用户-owned URL 预览，并把 asset 引用写入活动 SQLite session；浏览器加载失败会显示“资产 URL 不可用”，原始 URL 可单独打开。尺寸、步数和 seed 使用明确控件，窄屏下改为单列。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_image_workbench.py tests/test_harness_s4_remote_rag.py tests/test_harness_ui.py -q
18 passed
ui_react: npm run build       # tsc --noEmit + vite build 通过
```

本票完成 API 消费、状态呈现和本机布局开发门；真实 SD/diffusers/CUDA 采样质量、远端图像长时任务、真实浏览器回读大图和 30k 文档容量仍需后置验收。下一票为 `HARNESS-UI-04`，聚焦主题/可访问性、视觉回归与 React/TUI parity。

## 17. S6-HARNESS-UI-04 实施记录

本票完成工作台的 UI 收口和可重复回归门：

- React 增加跳过链接、`main-content` 焦点落点、导航 `aria-current`、消息 `role=log`/`aria-busy`、状态播报和输入标签；系统消息不再通过整体透明度降低对比度。
- 深浅色仍使用黑/白底与青蓝/洋红/暗金强调，补充 `forced-colors`、`prefers-reduced-motion` 和统一 focus-visible 样式；减少动效时关闭动画/过渡，窄屏与键盘焦点不改变布局。
- TUI 增加键盘焦点样式和 RAG/TXT2IMG 能力状态，与 React 共用 `ONLINE/FIXTURE/RAG/ASSETS/TXT2IMG` 术语；API 不可用时仍明确显示不可用，不伪造能力。
- `scripts/visual_smoke.mjs` 使用显式 Playwright 模块覆盖桌面/移动端、API fixture、横向溢出、导航后焦点、减少动效、浅色切换和按钮命名；截图写入 ignored 的 `build/ui-visual-smoke/`，不进入用户资产或仓库。

验证证据：

```text
npm run build                                  # tsc --noEmit + vite build 通过
npm run visual:smoke -- http://127.0.0.1:5181/ # desktop/mobile 通过
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
49 passed
```

S6 UI 四张开发票均已完成本机开发门；真实模型质量、真实 SD/CUDA、生图大图回读、跨设备长时网络和生产部署仍按前置计划后置验收。

## 18. RAG-BASE-01 实施记录

本票冻结主项目与 harness 的双侧离线召回基准，不启动模型、不访问网络，也不把 fixture 结果解释成模型质量：

- `harness_workbench/tools/rag_baseline.py` 固定 6 份目标文档与 30 条问题集；同一 source reference 分别写入 `src.rag_store.RagStore` 与 harness `rag.store.RagStore`，两侧均只使用现有 SQLite FTS5 检索契约。
- `RagBaselineReport` 使用 `qlh.rag_baseline.v1`，输出 corpus/case-set digest、query SHA-256、目标引用、命中位置和计数，不输出问题原文、文档正文、绝对临时路径或模型信息；默认两侧 hit@5 与 MRR 均为 `1.000000`。
- CLI：`python -m harness_workbench.tools.rag_baseline --json build/rag-baseline/latest.json --markdown build/rag-baseline/latest.md`；输入 schema、相对路径、重复标识和 30 条问题数均有 fail-closed 校验。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_rag_baseline.py -q
8 passed
```

后续 `RAG-META-01`、`RAG-CHUNK-01` 等票必须复用该问题集和报告，改动前后同时复测主项目与 harness，不静默替换度量口径。

## 19. RAG-META-01 实施记录

本票为主项目与 harness 增加结构化元数据过滤，仍保持本机 SQLite、FTS5 优先和无模型/无网络边界：

- 两侧新增 `rag_metadata_index` 倒排表，固定支持 `source`、`scope`、`type`、`tag`、`time` 字段；`tag` 可写入/匹配多值，过滤条件使用 SQL `EXISTS` 下推到 FTS/候选召回阶段。
- 主项目旧库初始化时按 `metadata_index_version` 回填索引；revision 元数据变化显式报冲突，删除会同步清理索引。harness 混合检索、cache key 和 `/v1/rag` source/search API 均透传 `metadata_filters`。
- 过滤字段和值有数量、长度、类型边界；未知字段和非法值 fail-closed。主项目查询审计只保留过滤字段名，不记录过滤值或查询原文。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_rag_store.py tests/test_rag_api.py tests/test_harness_s4_remote_rag.py tests/test_harness_mcp_api.py tests/test_harness_mcp_server.py tests/test_harness_rag_squeeze.py -q
62 passed
```

下一票进入 `RAG-CHUNK-01`；继续复用 `RAG-BASE-01` 的双侧 hit@5/MRR 口径。

## 20. RAG-CHUNK-01 实施记录

本票把主项目与 harness 的分块边界统一为可配置、可复现的本地规则，并将粒度信息随 chunk 持久化：

- 两侧均支持 `fixed`、`paragraph`、`sentence`、`section`、`adaptive`、`semantic` 六种策略；`section` 识别 Markdown/大写标题，`adaptive` 综合章节、段落和句子边界，`semantic` 在没有模型的开发机上仅使用确定性标题/段落/标点启发式。
- 重叠窗口继续受 `max_chunk_chars` 和 overlap 上限约束，所有 chunk 保存原文 `start_offset/end_offset`；主项目 `rag_chunks.granularity` 与 harness 同名字段记录实际策略，旧 SQLite 初始化自动补列。
- 主项目 FTS/CJK FTS/向量返回与 API 引用、harness source API 和 `list_chunks` 均透传粒度；相同 source/revision 以不同策略重复写入会报告 `revision_conflict`，避免无声改变引用边界。
- 新增 `run_rag_chunk_comparison()` 复用冻结的 6 份文档/30 条问题集，逐策略比较主项目与 harness；六种策略两侧均达到 `hit@5=1.000000`、`MRR=1.000000`，且详情签名一致。

离线验收：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_rag_chunking.py -q
10 passed
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_rag_store.py tests/test_rag_api.py tests/test_harness_s4_remote_rag.py -q
44 passed
```

本票不下载、不加载 QW1.8B、不联网；`RAG-BASE-01` 的双侧 hit@5/MRR 口径保持不变。下一票为 `RAG-QRW-01`，继续在规则和 FTS/候选管线层推进。
