# 联网搜索与轻量 Fetch 工具调用可行性调研与分期计划

> 更新日期：2026-09-08
> 状态：调研与计划门，尚未接入生产聊天路径
> 适用范围：主节点联网搜索、轻量网页 Fetch、QW1.8B/其他 8GB 级模型的工具调用，以及与现有任务图、RAG、Provider 和投机解码的关系

## 0. 结论先行

这条支线可做，但不应把“联网能力”寄托在 QW1.8B 自己是否会生成工具调用，也不应把 LittleLamb 直接当成 QW1.8B 的投机 draft。推荐的产品形态是：

1. 主节点提供一个受策略约束的 **Tool Gateway**，统一执行 `web_search` 和 `web_fetch`，模型只提交经过 schema 校验的意图，不能直接打开任意 socket。
2. LittleLamb 0.3B Tool-Calling 作为可选的轻量规划/路由 sidecar；它已经声明以 Qwen3-0.6B 为基础并进行工具调用和结构化输出微调，适合先做“是否调用工具、调用哪个工具、参数是什么”。
3. QW1.8B 默认定位为回答模型。若模型卡、chat template 和离线冒烟没有证明它支持工具调用，就由主节点规则/分类器决定是否调用工具，再把工具结果作为受控上下文交给 QW1.8B。
4. 投机解码只做速度优化，不做能力迁移。LittleLamb 作为 draft 不会把工具调用训练注入 QW1.8B；如果 target 不会生成工具调用，投机验证也不会改变这一点。
5. 不新造一个独立 harness。复用现有 TaskGraph/Stage/lease/audit、Provider 数据作用域、TaskPayloadStore、主节点 SQLite RAG、llama.cpp/Transformers 的 chat template/grammar，以及可选的 Qwen-Agent/MCP 适配层。新增的只是薄 Tool Gateway、契约和能力探测。

因此本计划的默认路线是 **“工具网关 + 小模型路由 + 任意小模型回答”**，而不是 **“LittleLamb draft + QW1.8B target”**。

## 1. 调研依据与已知事实

### 1.1 LittleLamb 0.3B Tool-Calling

官方 Hugging Face 模型卡（[MultiverseComputingCAI/LittleLamb-ToolCalling](https://huggingface.co/MultiverseComputingCAI/LittleLamb-ToolCalling)）给出的事实是：

- 约 290M 参数，基于 Qwen3-0.6B 的压缩模型；
- 额外进行 function calling、structured outputs 和 agentic workflow 微调；
- 使用 Qwen3 风格工具 schema，能输出结构化工具调用并消费工具结果；
- Transformers 运行路径要求 `transformers>=4.51.0`，与项目主运行时的 4.47.x 锁不兼容，因此必须走独立 sidecar/venv；
- 官方模型卡主要证明 Transformers/ONNX 路径。社区 GGUF 不能自动视为官方工件，准入仍要经过模型身份、chat template、SHA 和工具调用冒烟。

结论：LittleLamb **可以作为候选工具路由器**，但“能生成格式正确的 JSON”不等于“能安全执行网络请求”。执行权必须在主节点 Tool Gateway。

### 1.2 QW1.8B 与其他小模型

- 项目已部署的 QW1.8B 主要资产是旧 Qwen-1.8B-Chat Safetensors/GGUF。当前登记信息没有证明它经过专门 function-calling 训练，不能因为能输出 JSON 就直接登记为 tool-capable。
- Qwen2.5-0.5B-Instruct 官方卡强调结构化输出和 JSON 能力，但使用 `tools` 时仍需检查具体 chat template 和解析器；“能写 JSON”与“稳定调用工具”是两个质量门。
- Qwen3-0.6B 官方卡明确写出 agent/tool use 能力，并推荐 Qwen-Agent；它可作为 LittleLamb 之外的对照候选，但不改变当前主运行时版本隔离要求。
- Gemma、Qwen1.8B、普通 base/chat 模型和量化社区模型一律按 `unknown` 处理，直到能力探测报告同时确认：工具模板、工具调用标记、参数 schema、tool result 回灌和错误恢复。

### 1.3 现成工具调用基础设施

- Hugging Face Transformers 的 [tool/chat template 文档](https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents) 已定义 `tools` schema、assistant `tool_calls` 和 `tool` result 的消息循环，同时明确不同模型的输出包装可能不同，需要解析器适配。
- llama.cpp 官方 [function-calling 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) 已支持 OpenAI 风格工具调用：已识别模板走 native handler，未识别模板可以走 Generic；Generic 会增加 token 消耗且不保证模型本身具备可靠的工具选择能力。`--jinja`、`chat_template` 和 `/props` 能力检查应优先复用。
- Qwen-Agent 官方仓库已提供 Function Calling、MCP、RAG 和 OpenAI 兼容服务接入，[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) 也直接推荐它。可以把它作为可选解析/适配层，不将其升级为 QLH 的第二套任务调度器。
- MCP 官方 [fetch server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch) 可以把网页抽取为 Markdown 并分段返回，但其文档明确警告默认实现可能访问本地/内部 IP。它只能放在 QLH Tool Gateway 的受限适配器后面，不能直接暴露给模型。
- SearXNG 官方 [Search API](https://github.com/searxng/searxng/blob/master/docs/dev/search_api.rst) 提供 JSON 搜索接口，适合用户自托管或用户自行选择的搜索 Provider；搜索结果只返回标题、摘要、URL、来源等小对象，正文再交给 Fetch。

## 2. LittleLamb + 投机解码是否可行

### 2.1 严格答案：不能用投机解码迁移工具能力

标准 speculative decoding 的角色是：draft 先提出 token，target 并行验证，最终仍由 target 的分布决定输出。[Leviathan 等人的原始论文](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)把“保持 target 分布”作为核心性质；实际实现还要求 draft/target 词表或 tokenizer 具备兼容映射。项目现有《投机解码外部辅助实施说明》也把共享 tokenizer、verify 能力和外发作用域列为硬前提。

对当前设想的影响如下：

| 组合 | 判断 | 原因 |
|---|---|---|
| LittleLamb draft -> QW1.8B target | **默认 No-Go** | LittleLamb 是 Qwen3 tokenizer/模板，QW1.8B 是旧 Qwen tokenizer/模板，不能假设 token 对齐；即使做异构词表映射，也只优化速度，不增加 target 的工具训练能力。 |
| QW1.8B draft -> LittleLamb target | 可做实验，不是默认产品路线 | target 具备较强工具调用训练，但 QW1.8B 草稿可能在工具 JSON 边界和参数上低接受率，额外 draft 成本可能抵消收益。 |
| LittleLamb 独立 router -> QW1.8B answer | **推荐** | 两者通过 `tool_request/tool_result` 结构交互，不要求共享 tokenizer；LittleLamb 负责小范围决策，QW1.8B 负责回答。 |
| Qwen3-0.6B/Qwen-Agent -> QW1.8B answer | 可选对照 | 复用官方 agent/tool parser，但依赖和版本必须放在 sidecar，不污染主运行时。 |
| QW1.8B 自己直接 tool call | 未知，不能准入 | 必须先做模板、schema、参数、错误恢复和拒绝越权的离线矩阵；失败时自动切到 host router。 |

投机解码可以在 **工具结果已经获取之后** 用于加速最终自然语言回答，但仍需同 tokenizer、同输出契约、接受率和端到端延迟门；这与“让 QW1.8B 学会调用工具”是两件事，不能合并成一个卖点。

### 2.2 LittleLamb 是否值得引入

值得作为可卸载、可替换的 router 候选，但不应成为强依赖：

- 0.3B 参数量适合 8GB 机器和无独显从节点，推理预算远小于完整回答模型；
- 工具调用微调比普通小模型的 few-shot prompt 更可控；
- sidecar 隔离 Transformers 版本，符合 QW3 既有隔离策略；
- 仍需验证中文搜索意图、长 URL、数字参数、拒绝危险 URL 和多轮 tool result 回灌，官方 BFCL/τ²-Bench 分数不能代替项目自己的工具契约门。

## 3. 推荐总体架构

```text
用户请求
  -> 主节点 Intent Gate（规则/模型能力/用户授权）
      -> 无需联网：现有本地聊天/任务图
      -> 需要联网：Tool Gateway
           -> web_search（SearXNG/用户配置 Provider）
           -> web_fetch（受限 HTTP/可选 MCP fetch adapter）
           -> 结果规范化、引用、缓存、RAG 入库（可选）
      -> tool_result 回灌
           -> QW1.8B/Gemma/其他小模型生成最终答案
```

### 3.1 Tool Gateway 最小契约

工具调用不直接复用任意模型的原始输出，先规范化为版本化 envelope：

```json
{
  "schema": "qlh.tool_request.v1",
  "request_id": "...",
  "tool_name": "web_search",
  "arguments": {"query": "...", "top_k": 5},
  "user_scope": "local_user",
  "network_scope": "explicit_opt_in",
  "deadline_ms": 5000
}
```

返回值只允许摘要、引用和可选截断正文：

```json
{
  "schema": "qlh.tool_result.v1",
  "request_id": "...",
  "status": "ok",
  "items": [{"title": "...", "url": "...", "snippet": "..."}],
  "citations": [{"url": "...", "sha256": "..."}],
  "truncated": false,
  "policy": {"redirects": 0, "bytes": 12345}
}
```

第一版只注册两个工具：`web_search` 和 `web_fetch`。工具名、参数字段、最大长度、`top_k`、超时和返回字节数全部由主节点 schema 固定，未知工具、未知字段、越界数字和模型附加指令均拒绝。

### 3.2 两种模型路径

**路径 A：有工具训练的 sidecar。** LittleLamb/Qwen3 sidecar 接收有限工具 schema，输出 `tool_request.v1`；主节点验证后执行；结果作为 `tool` 消息或受控引用上下文回给 sidecar/回答模型。

**路径 B：没有工具训练的回答模型。** 主节点根据显式用户按钮、URL/搜索意图规则或一个轻量分类器决定调用哪个工具；QW1.8B 只收到脱敏、限长、带引用的结果，不需要生成 function call。工具失败时返回可解释的“未获取到资料”，不能让模型自行猜测或构造下一条网络请求。

路径 B 是 QW1.8B 的默认准入路径，也是其他未验证小模型的统一 fallback。

### 3.3 与现有模块的复用边界

| 能力 | 复用对象 | 本支线新增内容 |
|---|---|---|
| 阶段、取消、租约、重试、审计 | `TaskGraphCoordinator`、Stage/attempt/lease、现有 task graph projection | 新增 `tool_request`/`tool_result` Stage 类型及稳定错误码 |
| 外部数据授权 | `external_provider` 的 `deny/opt_in/allow_all` 和用户主节点配置 | 将网络工具也纳入同一 `network_scope`，默认不出网 |
| 大对象/正文生命周期 | `TaskPayloadStore` | 只保存摘要、SHA 和受控临时正文，不把原始网页塞进任务 envelope |
| 本地知识库 | 主节点 SQLite RAG、FTS5/向量 provider | 可选把用户明确保存的网页按 source/revision 入库，默认不自动持久化 |
| 工具模板/解析 | llama.cpp `--jinja`、Transformers `apply_chat_template`、可选 Qwen-Agent/MCP | 能力探测和统一 envelope 适配，不新增第二套 Agent runtime |
| 测试/评判 | `llm_smoke_matrix`、任务图和网络故障矩阵 | 增加工具调用 schema、SSRF、引用完整性和时序用例 |

## 4. 联网搜索与轻量 Fetch 的安全边界

联网工具的风险不低于模型本身。OWASP [SSRF 防护建议](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) 和 [OWASP Top 10 A10](https://owasp.org/Top10/2021/A10_2021-Server_Side_Request_Forgery_%28SSRF%29/) 要求使用正向约束、处理 DNS rebinding/TOCTOU 并重新校验跳转；本项目应采用以下默认值：

- 只允许 `https`；`http` 仅在用户明确打开兼容开关后允许；拒绝 `file://`、`data://`、`gopher://`、`ftp://` 等 scheme；
- DNS 解析后同时检查 IPv4/IPv6，拒绝 loopback、link-local、RFC1918、ULA/Tailscale 私网、multicast、metadata endpoint；连接前再次绑定解析结果，避免 DNS rebinding；
- 默认不自动跟随跳转；若允许，最多 3 跳且每跳重新执行 scheme、host、解析地址和端口策略；
- 单次请求超时 5 秒、响应上限 1 MiB、文本抽取上限 32 KiB、并发上限 2；首版只接收 HTML/text/JSON，脚本、二进制附件和可执行内容拒绝；
- 尊重 robots.txt 和用户配置的代理；7897 只作为用户显式配置的出站代理，不能由模型参数覆盖；
- 搜索结果先返回标题/摘要/URL，正文必须由第二次 `web_fetch` 且通过相同策略；每个结果保留来源 URL、抓取时间、响应摘要 SHA 和截断标记；
- 日志只记录 request/tool/provider/status/bytes/latency/URL digest，不记录认证头、完整正文、Cookie 或用户原始 prompt；
- 默认不把联网内容写入 SQLite RAG。只有用户点击保存或任务明确声明 `persist=true` 时才创建 source/revision，并先做敏感内容和提示注入标记。

MCP fetch 可以作为实现参考或隔离进程 adapter，但不能绕过以上策略。其官方文档已经提示本地/内部 IP 风险，QLH 必须在 MCP client 之前再做一次 URL 和连接策略校验。

## 5. 不自建 harness 的落地原则

这里的“harness”如果是指一套重新定义模型、工具、重试、状态、观测和 UI 的 Agent 框架，确实属于重复建设。建议只增加三个薄层：

1. `ToolCapabilityProbe`：复用现有 `llm_smoke_matrix` 的隔离子进程和报告格式，增加模型 chat template、工具 schema、调用/回灌/拒绝矩阵；
2. `ToolGateway`：只负责策略、provider adapter、结果规范化和缓存，不负责模型对话循环；
3. `TaskGraph` 适配：把一次搜索/Fetch 表达为现有任务图里的受控 Stage，使用已有 lease、cancel、retry、audit 和 payload 引用。

优先复用 Qwen-Agent 的 parser/MCP client 或 llama.cpp/Transformers 的原生模板；只有当版本、许可证或隔离依赖无法接受时，才实现一个小型兼容 parser。无论是否使用 MCP，QLH 的安全策略、数据归属和审计都必须留在主节点，而不是交给第三方 Agent runtime。

## 6. 分期开发票与验收门

本计划只排除“没有联网/没有第二张 CUDA 卡”的开发阻塞；真实搜索 Provider、长时间网络、跨平台和生产路由可后置验收。

| 票 | 状态 | 内容 | 验收门 |
|---|---|---|---|
| `WEB-TOOL-G0` | Completed（本票，文档） | 调研 LittleLamb、Qwen 小模型、chat template、MCP/搜索/Fetch、投机解码边界，冻结路线与 No-Go | 文档结论、来源、复用边界和安全原则完整 |
| `WEB-TOOL-G1` | Completed (local gate) | `scripts/model_tools/tool_capability.py` 与 `tool-capability-probe` CLI 已完成：读取本地 GGUF/Safetensors 元数据、template/tool schema/tokenizer digest/sidecar 版本；支持离线 transcript fixture，不联网、不加载权重 | 报告独立区分 JSON、工具调用、tool-result 回灌三项状态；`unknown` 不准入，fixture 只能形成 `candidate`，仍需运行时门 |
| `WEB-TOOL-G2` | Completed (local gate) | `src/tool_gateway.py` 完成 Tool Gateway 请求/结果合同、`web_search`/`web_fetch` 参数边界、scope 授权、SSRF/DNS/redirect/size/deadline 和稳定错误码；只做策略与规范化，不发起网络 | schema、私网/ULA/loopback/metadata、IPv4/IPv6、重定向、代理、结果大小和 content-type 的 fail-closed 矩阵通过；真实 Provider/HTTP 执行留在 G3 |
| `WEB-TOOL-G3` | Completed（本机开发门） | `src/tool_gateway_adapters.py` 完成 SearXNG JSON adapter、受限 Fetch adapter、显式用户代理/代理传输、脱敏引用和 provider fallback；真实出口仍不挂生产路由 | fake transport/provider 全覆盖；每跳沿用 G2 SSRF/大小/content-type 策略；失败仅对 retryable provider error 回退；真实 Provider 仍后置验收 |
| `WEB-TOOL-G4` | Completed（本机开发门） | `src/tool_task_graph.py` 复用 TaskGraph Stage/attempt/lease/cancel/journal audit；加入 sidecar capability gate、host-router fallback 与 `qlh.tool_context.v1` 结果回灌 | 未验证 sidecar 不得接管；仅 retryable sidecar 错误可回退；取消、预取消和迟到结果不提交；普通回答模型只消费标准化 tool context |
| `WEB-TOOL-G5` | Completed（本机开发门） | `src/tool_rag_cache.py` 与 `/api/tool-cache/*` 接入用户-owned SQLite RAG：显式持久化、引用展示、scope、容量、TTL/过期清理、删除/重建 | 未传 `persist=true` 永不落盘；每个 item 必须有 citation；owner/project scope 硬过滤；容量/TTL/删除/重建和 path-free API 通过 |
| `WEB-TOOL-G6` | Completed（本机准入门） | 候选模型/Provider 离线质量评测、指标阈值和生产候选矩阵；真实网络与权重运行仍后置 | 工具选择/schema/拒绝越权/引用/grounding/延迟均有固定报告；未达门槛或未有真实网络证据时 fail-closed |
| `WEB-TOOL-AUDIT-01` | Completed（本机联合审计） | 对 G2-G6 的安全边界、API surface、sidecar fallback、显式持久化和联合测试结果审计 | 审计无阻断项；明确真实 Provider、sidecar 模型和生产路由残余风险 |
| `WEB-TOOL-SPEC` | Optional / 后置 | 仅在 G1/G4 证明同 tokenizer、工具边界可验证且接受率有收益时，评估“工具结果后的最终回答”投机解码 | `tokens_per_round`、p95 延迟和总 CPU/VRAM 成本优于非投机；否则永久保留为实验路线 |

### 6.1 建议质量门

- `tool_selection_accuracy`：在冻结的搜索/Fetch/无需联网/拒绝 URL 集上统计；低于 90% 只允许 host router，不允许模型自主调用。
- `schema_valid_rate`：工具名、必需参数、类型和范围全部合法；生产候选至少 98%，否则进入 grammar/规则 fallback。
- `unsafe_request_block_rate`：私网、重定向、特殊 scheme、过大响应和超时样本必须 100% 拦截或归类为稳定错误。
- `citation_rate`：使用联网工具的回答必须带来源摘要；工具失败不得伪造引用。目标 100%。
- `answer_grounded_rate`：回答中的可验证事实必须能回指 `tool_result` 或用户明确提供的内容；先做离线人工抽样，再决定生产阈值。
- `latency`：不把网络 RTT 与模型生成混成一个指标，至少记录 intent、provider、fetch、回灌和最终生成的 p50/p95。
- `speculation`：只看端到端收益；接受率低或 `tokens_per_round` 接近 1 时自动回退，不以单独的 draft 速度宣称收益。

## 7. 当前决策

1. **下一步推进**：G6 本机准入门和 `WEB-TOOL-AUDIT-01` 已完成；下一步仅在真实 Provider、候选模型进程和生产路由验收条件具备时，运行同一报告合同，不允许以离线 candidate 替代生产准入。
2. **LittleLamb 定位**：独立 sidecar 的可选工具路由器；复用 `.venv-gemma4-native` 的原则但新建/锁定自己的运行时，避免 Qwen3 `transformers>=4.51` 污染主环境。
3. **QW1.8B 定位**：先作为回答模型；工具调用能力登记为 `unknown`，不得因为能生成 JSON 或 llama.cpp Generic parser 存在就自动升级。
4. **投机解码**：不纳入联网工具调用主路径。只有工具结果已取得、模型/词表/模板满足兼容条件，且实验数据证明端到端收益时才做 `WEB-TOOL-SPEC`。
5. **工程形态**：不建立新的通用 Agent harness；以现有任务图、Provider scope、RAG、模型冒烟和 MCP/Qwen-Agent/llama.cpp 适配为基础，新增薄 gateway、probe 和契约。

## 8. 参考资料

- [LittleLamb 0.3B Tool-Calling model card](https://huggingface.co/MultiverseComputingCAI/LittleLamb-ToolCalling)
- [Qwen3-0.6B model card（Agentic Use）](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Transformers: Expanding Chat Templates with Tools and Documents](https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents)
- [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [MCP reference Fetch server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [SearXNG Search API](https://github.com/searxng/searxng/blob/master/docs/dev/search_api.rst)
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)

## 9. WEB-TOOL-G1 实施记录

本票完成的是离线静态能力注册与合同门，不是模型真实工具调用准入。

- `scripts/model_tools/tool_capability.py` 只读取本地模型目录或 GGUF 头部；不读取权重张量、不访问 Hugging Face/ModelScope/外部网络。
- Safetensors 侧读取 `config.json`、`tokenizer_config.json`、`chat_template.jinja` 和 tokenizer 文件摘要；GGUF 侧复用现有 `inspect_gguf`，只登记 tokenizer/template 元数据摘要，不输出原始模板。
- 报告固定包含 `json_output`、`tool_call_generation`、`tool_result_reinjection` 三个能力状态，状态为 `unknown`、`declared`、`verified` 或 `rejected`。
- `--fixture` 只校验有界的 `web_search`/`web_fetch` transcript 合同。fixture 全部通过时 admission 为 `candidate`，`production_eligible` 仍为 `false`；没有 fixture 的 QW1.8B 或 LittleLamb 资产不会因为存在 JSON/template 标记而自动准入。
- CLI 支持 `tool-capability-probe` 与 `tool_capability_probe`，可对注册模型矩阵或显式本地资产运行；报告仅保留本地路径摘要，避免泄露绝对路径。
- `.venv-test` 新增能力合同测试 `4 passed`，既有 LLM smoke 定向回归 `10 passed`。真实 sidecar 生成、Provider、网络与生产路由留在 G2-G6。

## 10. WEB-TOOL-G2 实施记录

本票完成 Tool Gateway 的策略与契约层，仍未打开生产网络出口。

- `src/tool_gateway.py` 新增 `qlh.tool_request.v1` / `qlh.tool_result.v1` envelope、严格字段集、`web_search`/`web_fetch` 参数校验和稳定 `ToolGatewayError.code`。
- 默认 `data_scope=opt_in`；`deny` 永远拒绝，`opt_in` 必须由上层显式传入 `allow_external=true`，请求中的 `network_scope` 不能扩大权限。
- URL 默认仅允许 HTTPS；拒绝凭据、fragment、loopback、RFC1918、CGNAT、link-local、ULA、multicast、metadata 主机名和非公开 IPv4/IPv6 字面量。DNS 解析结果通过 `validate_resolved_addresses` 单独复核，Provider 每次连接和重定向都必须调用。
- 重定向最多 3 跳，每跳重新校验 scheme/host；响应限制为 1 MiB，文本最多 32 KiB，仅接受 HTML/text/JSON/Markdown；原始响应正文不会进入标准化结果。
- `QLH_TOOL_PROXY` 仅作为服务端显式配置读取，允许用户本机 `127.0.0.1:7897` 代理；请求/模型参数不能覆盖代理，代理不允许携带凭据。
- 本票没有 socket、DNS、HTTP、Provider 或生产路由副作用；G3 才接 fake Provider 和受限 adapter。
- `.venv-test` Tool Gateway 专项测试 `15 passed`；G1 能力探测 `4 passed` 与 MODEL-TOOLS 回归 `61 passed, 2 skipped` 保持通过。

## 11. WEB-TOOL-G3 实施记录

本票完成 provider 适配层，但没有把网络出口直接接入聊天生产路由。

- `SearxSearchAdapter` 接受 SearXNG JSON，固定 `format=json`，仅输出有界标题/摘要/安全 URL 和 SHA-256 引用；不安全或空结果会被过滤并形成稳定错误。
- `RestrictedFetchAdapter` 只接受 HTML、纯文本、Markdown、JSON；正文受字节和字符上限约束，HTML 会移除 script/style/noscript/template，原始正文不会进入结果 envelope。
- `UrllibTransport` 禁用环境代理，代理必须由服务端显式传入（可使用 `http://127.0.0.1:7897`），每个请求/重定向 hop 先走 G2 URL/DNS/SSRF 门，并手动限制重定向次数和响应大小。
- `ToolGatewayExecutor` 只在 `ToolProviderError.retryable=true` 时切换下一个同类 provider；scope、URL、内容类型和合同错误直接失败，保留 provider/状态/错误码尝试记录。
- 新增 `tests/test_tool_gateway_adapters.py`，使用 fake transport 覆盖搜索、过滤、HTML 脱敏、内容类型拒绝、可解释回退和 scope 拒绝；`.venv-test` 专项 `7 passed`。
- 仍未完成：真实 SearXNG/外网验收、DNS 连接 pinning、真实 sidecar/回答模型进程、多阶段生产聊天路由和 RAG 持久化；这些进入 G5-G6，避免网络条件改变生产行为。

## 12. WEB-TOOL-G4 实施记录

本票完成 TaskGraph 适配与模型回灌边界，仍不启动 LittleLamb/QW1.8B 进程，也不打开真实网络生产出口。

- `ToolRouter` 固定两条路由：sidecar 只有能力状态 `verified` 才能运行；`unknown`、`declared`、`rejected` 均走主节点 host-router。sidecar 结果为 retryable error 且启用 fallback 时才切换 host，scope/合同/不安全 URL 错误不会被吞掉。
- `ToolTaskGraphAdapter` 将请求包装成 `tool_request` Stage，直接复用既有 attempt、lease、cancel、SQLite journal 和结果提交状态机；请求在注册 workflow 前先过 G2 Gateway，避免非法请求消耗状态序列。
- 成功结果生成 `qlh.tool_context.v1`，仅含工具名、请求 ID、有界 items/citations 和截断标记，供 QW1.8B 等普通回答模型消费；不复制正文、Cookie、认证头或 provider 原始字段。
- `audit()` 只读调用现有 `task_graph_attempt_audit`，不重试、不续租、不提交结果；预取消、执行中取消和迟到返回均由 TaskGraph fencing 处理。
- 新增 `tests/test_tool_task_graph.py`，覆盖 capability gate、sidecar retryable fallback、非 retryable 不回退、scope 前置拒绝、journal audit、预取消和执行中取消；`.venv-test` 专项 `5 passed`。
- 仍未完成：真实 sidecar/回答模型进程、TaskGraph 多阶段回灌、真实网络取消中断、生产 HTTP 路由和 RAG 持久化；进入 G5-G6 后置验收。

## 13. WEB-TOOL-G5 实施记录

本票完成联网结果到用户本地 RAG 的显式持久化边界，默认仍不自动保存。

- 新增 `src/tool_rag_cache.py`，在同一用户-owned SQLite/WAL 文件中建立 `rag_tool_cache` ledger；只保存标准化 items/citations、来源摘要 SHA、scope、内容字节数、TTL 和容量信息，不保存 provider 原始响应、Cookie、认证头或请求正文。
- `persist=true` 是硬门。缺省或 `false` 直接返回 `persistence_not_explicit`；结果必须通过 G2 `normalize_tool_result`，且每个 item 都必须能匹配 citation URL。
- 缓存正文通过现有 `RagStore.ingest_document` 写入，复用敏感字段拒绝、FTS5、revision、owner/access scope 和删除事务；检测到提示注入样式只登记 `prompt_injection_suspected`，不会把外部文本当作系统指令。
- 提供容量/列表/读取/搜索/显式删除/FTS 重建/过期清理 API：`/api/tool-cache/health`、`/api/tool-cache`、`/api/tool-cache/search`、`/api/tool-cache/rebuild`、`/api/tool-cache/purge`。API 只返回 SQLite 摘要和引用，不返回本地绝对路径或 vector blob。
- TTL 默认 7 天，可由用户明确设置但不超过 90 天；条目数和总字节数均有服务端上限，超过预算 fail-closed，不自动淘汰未过期用户资产。
- 新增 `tests/test_tool_rag_cache.py` `6 passed` 与 `tests/test_tool_rag_cache_api.py` `2 passed`；联合 RAG 回归 `38 passed`。真实网络内容质量、长时 TTL、跨进程并发和生产 UI 接入仍后置到 G6/产品端。

## 14. WEB-TOOL-G6 实施记录

本票完成离线质量校准和生产候选矩阵，未打开真实网络出口，也未加载模型权重。

- 新增 `src/tool_quality_gate.py`：冻结 `qlh.tool_quality.v1` 报告，指标包括工具选择准确率、工具调用 schema 合法率、危险请求拦截率、引用完整率、回答 grounding 率和延迟 P50/P95。报告只保存 fixture digest、计数、比例和错误码，不保存 prompt、URL、正文、绝对路径。
- 默认阈值与本计划 §6.1 对齐：选择 `>=90%`、schema `>=98%`、危险请求拦截 `100%`、引用 `100%`、grounding `>=90%`；P95 延迟只在显式提供阈值时作为硬门，避免把网络 RTT 和模型生成混成单一指标。
- 离线 fixture 覆盖 search、Fetch、无需联网、私网/危险 URL 拒绝、retryable provider、畸形调用和无工具回答。拒绝样本不计入工具 schema 分母，但计入越权拦截率，避免安全拒绝被误判成调用失败。
- `assess_candidate`/`build_admission_matrix` 将模型分为 `model_autonomous` 与 `host_router`：QW1.8B 等 `tool_call_generation != verified` 的模型只能走 host-router；即使 LittleLamb 达到 verified，没有真实 Provider 证据也只标记 candidate，`production_eligible=false`。
- 新增 `tests/test_tool_quality_gate.py`，专项 `9 passed`；G6 质量门和模型/Provider 矩阵不创建 socket、不读权重。

## 15. WEB-TOOL-AUDIT-01 联合审计记录

G6 完成后对 G2-G6 新增功能做只读联合审计，新增 `src/tool_feature_audit.py` 与 `tests/test_tool_feature_audit.py`。

- 审计检查 G2 私网 SSRF 拦截、G4 sidecar 未验证时的 host-router 兜底、G5 `persist` 默认关闭、G5 API 路由完整性和 G6 离线质量门；全部通过。
- 联合审计报告固定声明 `network_used=false`、`weights_loaded=false`、`production_network_enabled=false`，残余风险仅为真实 Provider、真实 sidecar 模型进程和生产聊天路由尚未验收。
- 本次没有发现需要新增已知问题记录的阻断缺陷。G2-G6 联合专项为 `145 passed, 2 skipped`；在 `NO_PROXY=*` 隔离环境下全量 `.venv-test` 为 `3219 passed, 19 skipped`。真实验收时必须复用同一质量报告合同，并补充真实网络、DNS pinning、端到端取消和模型生成证据；不能手工把 candidate 改成 production。
