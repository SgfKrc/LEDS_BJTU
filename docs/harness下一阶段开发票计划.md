# Harness 下一阶段开发票计划（S3.2 生图后置+红队 / S7 联网+MCP / S8 长期记忆）

> 状态：**已拆票；`S7-MCP-02` 本机开发门已完成，当前进入 `S5-CLOSE-01`**；真实第三方 MCP 服务、认证与权限仍属于后置验收；前置状态回顾：S1-S4 本机/离线开发门已完成（v10 记录），S6（React 工作台 + Textual TUI）已完成本机开发门；本文档新增三条支线：**S3.2（生图真机后置 + 安全红队合并）、S7（联网搜索 + 轻量 MCP 服务）、S8（小模型长期记忆：RAG + 上下文压缩 + 本地 memory）**
>
> 创建日期：2026-09-08
> 适用范围：harness 子项目下一阶段票；不与 [WEB-TOOL 联网支线](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md) 合并（那些是主项目运行时工具，本票是 harness 侧能力与对外服务）；不覆盖训练微调。

---

## 1. 背景（现状盘点）

- **已完成**：S1 上下文引擎、S1.5 模型画像与能力门、S2 API 层 + llama-server adapter、S2.5 定制化实验台（adaptation/eval/Pareto）、S3 生图工作区（离线门）、S4 远端与 RAG（离线门）——全部"本机/离线开发门"，**真实运行时验收整体后置**。
- **已列计划**：S5 评估收口（ollama 对照/契约漂移/Pareto）、S6 工作台 UI（`ui_react/` + `tui.py`）。
- **验证命令基线**：`.venv-test\Scripts\python.exe -m pytest tests/test_harness_*.py -q`（`S7-MCP-02` 完成后全量 105 passed；全仓库回归 3324 passed, 19 skipped）。

## 2. S3 后置项现状说明（归票前澄清）

S3 离线门已封闭：contracts/manifest/注入式执行器/资产原子落盘/远端 QLH job-blob 映射/`b64_json`+URL 响应均有测试；**当前默认本地执行器明确返回 `local_image_runtime_unavailable`**（不伪造能力）。后置项 = **真实 diffusers/CUDA 执行器验证、生图→多模态追问闭环、高级编辑（img2img/inpaint/IP-Adapter/指令编辑）**。S3.5 之前的表述把"安全红队"单独列为候选——现将两者合并为一张票（S3.2），理由：都属"把 demo 边界变成可信证据"（真机证据 + 对抗证据），且都需要先在 harness 侧落固定契约样本再进真机。

## 3. 票 S3.2：生图真机后置 + 安全红队扩展

**目标**：把生图从"离线合同"推进到"真实执行器可演示"，同时用红队样本证明工具/图片路径的越权与注入边界。

| 项 | 交付 | 验收 |
|---|---|---|
| 生图真机后置 | 接入真实 diffusers/CUDA 执行器（独立 venv，消费共享 SD 工件，`manifest` 校验保持）；修复真机暴露的 executor 生命周期问题 | 真实 txt2img 生成 → `ImageAssetStore` 落盘 → 读回 SHA 校验通过；`/v1/images/generations` 端到端（不启动主项目 API）；**默认执行器不再返回 unavailable 才可关闭该码** |
| 多模态追问闭环 | 生图结果 → 缩略卡 → 上下文（CTRL `image_ref` 资产引用）→ 多模态模型回答 | 生图→引用→追问端到端 1 轮闭环；上下文预算含缩略卡（预算公式更新） |
| 高级编辑 | img2img / inpaint / IP-Adapter / 指令编辑 的 contracts + manifest + 远端 qlh 映射（本地执行器后置或明确不可用） | 至少一条路径真机验证（img2img 或 inpaint）；其余登记"本地不可用/远端可用" |
| **安全红队扩展** | `eval/fixtures.py` 新增红队样本族：① **prompt injection**（系统提示覆盖/角色劫持/越狱）；② **工具越权**（工具调用请求篡改 scope、绕过 host_router、虚构 verified）；③ **图片路径**（manifest 遍历/符号链接/超大图资产）；④ **上下文注入**（伪造 STATE 字段/schema 破坏/删除键越权） | 危险请求拦截 100%（红队样本全被拒绝或 fail-closed）；`schema_valid_rate >= 98%`、越权样本 0 通过；报告含红队检出率（新增指标 `red_team_blocked`） |
| 边界 | 不做真实生图的安全模型审计（如 prompt 注入到图片），只做 harness 契约层拦截 | — |

**依赖**：真机 CUDA 环境（与主项目 SD 侧车共享资产但不同时运行，互斥避免显存冲突）；S3.2 红队部分不依赖真机，可先行。

## 4. 票 S7：联网搜索功能 + 轻量 MCP 服务

**目标**：给 harness 增加"信息获取"能力（联网搜索/受限 Fetch），并把这些能力做成**自研好玩的轻量 MCP 功能集**（harness 作为 MCP server，让 Claude Code / opencode 等外部客户端把 harness 当工具集合玩——反向展示：harness 不只消费模型，也对外提供 K 个工具）；同时**预留接入其他 MCP 服务的扩展点**（本期只落地接口/契约，真实第三方接入后续）。

| 项 | 交付 | 验收 |
|---|---|---|
| 联网搜索工具 | `tools/`（或 `adapters/tool_*`）：`web_search` / `web_fetch` 工具实现；**本地模式** = 标准库受限 HTTP（仅 HTTPS、显式 opt-in、拒绝 loopback/RFC1918/metadata、DNS+重定向复检、大小/content-type 门——约束思路沿用主项目 [Tool Gateway 方案](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md)§G2，但 harness **不 import 主项目代码**，自实现轻量版）；**远端模式** = 经 qlh adapter 走主项目 `/api/tool-*`（复用已验证的 G2-G5 链路） | 本地：fake transport 下 SSRF 拦截矩阵 + 大小/类型门测试全过；远端：真实 qlh 契约映射测试；**生产网络默认关闭**（`production_network_enabled=false` 直到显式验收） |
| 工具协议接入 | 工具结果 → `qlh.tool_context.v1` 有界注入回答模型（对齐 harness 角色分工：模型不自行生成工具调用，**host_router 模式**）；能力通告随 model_profiles capability 门（未 verified 不启用 autonomous_tools） | 与 S1.5 capability_gate 联动；无 verified 时工具不可用（fail-closed）；红队样本（越权/伪工具）复用 S3.2 红队族 |
| **轻量 MCP 服务** | 新增 `mcp_server/`：**harness 自带的好玩 MCP 功能集**——chat、sessions、rag/search、images、web_search/web_fetch、memory(§S8) 全部以 MCP 工具暴露（stdio 与 SSE 两种 transport，工具名/schema 与 `/v1/*` 合同一致、可离线玩耍）；**同时预留接入其他 MCP 服务的通道**：MCP tool registry + 外部 MCP server 端点配置声明（本期只落地接口、契约、schema 校验与能力通告，不实现真实第三方接入） | ① server 角色：任一 MCP 客户端（如 Claude Code）连接后可列出工具并调用 1 个读工具（rag/search）+ 1 个写工具（session create）成功；② **接入预留**：以 fixture fake MCP server 证明"仅配置即可接入并隔离失败"的通道可用（工具发现/调用/错误传播），真实第三方 MCP 服务留后续票；**无密钥泄漏、无绝对路径**；harness 不 import 主项目代码约束保持 |
| 边界 | 不做通用 Agent 循环（模型自治选择工具由客户端侧承担；harness 只提供工具与 gate）；不做主项目那套完整 Tool Gateway 策略（复用思路，简化实现） | — |

## 5. 票 S8：小模型长期记忆（RAG + 上下文压缩 + 本地 memory）

**用户想法正式立项**：小模型上下文窗口有限，单一" STATE 摘要 + 近 N 轮"（S1）只解决**会话内**压缩；跨会话的事实、偏好、决策会随会话结束丢失。本票把"上下文有限"问题的解从"压缩"扩展到"**压缩 + 检索 + 持久化**"三层组合。

| 层 | 交付 | 验收 |
|---|---|---|
| **工作记忆**（已有，增强） | S1 `ContextPolicy` 的 STATE 摘要（decision log）+ 本票增加**会话内 pinned 事实自动建议**（`/pin` 显式为主，建议提示为辅） | 原 S1 测试不回退；pinned 永不裁剪 |
| **长期记忆**（新增） | `memory/store.py`（用户-owned SQLite，独立于 session 库）：`facts`（asymmetric 断言）/`preferences`（用户显式声明）/`decisions`（谁/何时/为什么）三类条目，带 scope、来源会话、时间戳、**可删除/可失效**（软删除） | 跨会话：A 会话存的事实，B 会话可检索；删除后检索不到；scope 硬过滤（沿用 S4 模式） |
| **检索注入** | `memory/retrieve.py`：FTS5 优先 + 可替换 embedding provider（与 S4 rag/provider 契约共用 interface）；hit 注入遵循完整 chunk 边界、超预算 `omitted_count`+`truncated=true`（不静默截断）；**预算分配**：memory 块 + RAG 引用块 + 上下文策略三者共享 `input_budget`（预算公式升级为分层预算） | 30 轮+ 跨 2 会话场景：早期事实召回（滑窗基线 vs memory+压缩）实测；预算 0 溢出；`hit@5` 基线复测（S4 门不回退） |
| **上下文压缩联动** | `ContextPolicy` 触发时：旧轮 → STATE 摘要 + **高价值事实抽取写入长期记忆**（一次压缩双写） | 抽取准确性人工抽样 ≥ 90%（固定 fixture）；重复抽取去重（事实指纹） |
| 边界 | 不做 embedding 训练/不做自动改写用户记忆（只可由用户删除/失效）；不在无证据时把 memory 内容当事实（来源+时间戳必须保留） | — |

**验收（联合）**：跨会话记忆场景端到端演示：会话 1 建立事实 → 会话 2 在新上下文预算下检索+注入 → 回答引用正确；附 S8 专项 pytest + 红队注入样本（伪造 memory 条目/越权删除）纳入 S3.2 红队族。

## 6. 排序与依赖

```
S3.2 红队部分（离线先行）───┐
S7 联网+MCP（依赖已验证 G2 思路；红队样本复用）───┤
S8 长期记忆（依赖 S1/S4 已有基础；检索与预算升级）──┼─> S5 收口（Pareto/展示）
S6 工作台 UI（依赖 S7/S8 的 API 面）─────────────────┘
```

- 建议顺序：**S3.2 红队 → S8（核心亮点，用户关注）→ S7（外部可演示）→ S3.2 真机生图（等 CUDA 环境）→ S5**。
- 每票仍按项目惯例"本机/离线门 + 后置真机"，所有"可以做"都必须有接受证据后才关闭。

## 6.1 开发票拆分与当前入口

分票遵循“先完成不依赖硬件的契约和安全门，再做真机后置验收”；每票只改变一个边界，禁止把真实 CUDA、第三方网络或第三方 MCP 服务作为 core 开发前置。

| 票 | 支线 | 内容 | 依赖 | 状态/开发门 |
|---|---|---|---|---|
| `S3.2-RT-01` | S3.2 红队 | 四类红队 fixture（prompt injection、工具越权、图片路径、上下文注入）与统一 fail-closed gate | S1/S2/S3/S4 已有契约 | **已完成本机开发门**；fixture digest 稳定，危险样本全部 blocked，决策 schema 可审计 |
| `S3.2-RT-02` | S3.2 红队 | `red_team_blocked`/越权通过率/`schema_valid_rate` 指标接入 evaluation report 与 promotion gate | RT-01 | **已完成本机开发门**；默认 block rate=100%、schema valid≥98%、越权通过=0 |
| `S3.2-IMG-01` | S3.2 生图 | 独立 executor 生命周期、真实 diffusers/CUDA 适配边界和互斥锁 | RT-01 | **硬件后置**；无 CUDA 时保持 unavailable |
| `S3.2-MM-01` | S3.2 多模态 | image asset ref、缩略卡和多模态追问上下文合同 | IMG-01 | 无硬件可先做合同/fixture，真机后置 |
| `S3.2-EDIT-01` | S3.2 编辑 | img2img/inpaint/IP-Adapter/指令编辑 contracts 与远端映射 | S3 contracts | 本机契约门；至少一条真机路径后置 |
| `S8-MEM-01` | S8 记忆 | 用户-owned SQLite facts/preferences/decisions schema、scope、软删除/失效 | S4 session/RAG | **已完成本机开发门**；独立 memory SQLite、scope 硬过滤、显式删除确认、失效/过期保留审计行 |
| `S8-MEM-02` | S8 记忆 | FTS5 检索、citation、分层预算与 `omitted_count`/`truncated` | MEM-01 | **已完成本机开发门**；FTS5 scope/lifecycle 过滤、可选 embedding rerank、三层共享预算 |
| `S8-MEM-03` | S8 记忆 | ContextPolicy 压缩时的事实抽取双写与去重 | MEM-01/02 | **已完成本机开发门**；保守抽取、显式候选覆盖摘要、fingerprint 去重 |
| `S8-E2E-01` | S8 记忆 | 跨会话召回、删除/失效、scope 隔离与红队注入联合场景 | MEM-01~03 | **已完成本机 E2E 门**；workflow 闭环、跨 scope fail-closed、注入拦截 |
| `S7-NET-01` | S7 联网 | 本地 HTTPS `web_fetch`/`web_search`、SSRF/DNS/重定向/大小/content-type 门 | RT-01 | **已完成本机开发门**；fake transport 门通过；生产网络默认关闭 |
| `S7-NET-02` | S7 联网 | 远端 QLH tool adapter 映射与错误合同 | NET-01 | **已完成本机开发门**；fake QLH 门通过；真实网络后置 |
| `S7-TOOL-01` | S7 工具 | `qlh.tool_context.v1` 有界结果注入与 capability gate 联动 | NET-01/02 | **已完成本机开发门**；无 verified 能力时 fail-closed |
| `S7-MCP-01` | S7 MCP | 内置 MCP registry、schema 校验、stdio transport、读/写工具最小集 | NET/TOOL 合同 | **已完成本机开发门**；fixture MCP 门 |
| `S7-MCP-02` | S7 MCP | SSE transport 与外部 MCP 端点后置验收、第三方客户端兼容加固 | MCP-01 | **已完成本机开发门**；真实第三方服务/认证后置 |
| `S5-CLOSE-01` | S5 收口 | Ollama 对照、契约漂移、Pareto 与公开演示 evidence 汇总 | S3.2/S7/S8 可用门 | 真机质量可后置，先做报告骨架 |

**当前执行顺序**：`S7-NET-01 → S7-NET-02 → S7-TOOL-01 → S7-MCP-01 → S7-MCP-02 → S5-CLOSE-01`；当前入口为 `S5-CLOSE-01`；`S3.2-IMG-01`、`S3.2-MM-01`、`S3.2-EDIT-01` 按硬件可用性插入，不阻塞其余票。

## 7. 风险与边界

1. 范围控制：S7 的联网工具**不做**主项目 G2 全套策略复刻（只保留核心 SSRF 门）；MCP 服务不替代 harness 自身 run-loop（客户端自治）。
2. S8 风险：memory 内容污染/过期事实 → 软删除 + 来源/时间戳 + 显式失效；"建议 pin"只做建议不自动写。
2b. S7 边界：MCP **预留通道**（registry + 端点配置 + schema 校验）本期只做接口与 fixture 验证，**真实第三方 MCP 服务接入**（依赖外部 server 地址/认证/权限）登记后置票，不提前引入不可控依赖。
3. 生图真机（S3.2）：与主项目 SD 侧车共享工件但**不同时运行**（显存互斥）；真实执行器验证必须真 CUDA 卡——不可用则保持 `unavailable` 码（不关门）。
4. 全票贯穿约束：harness 不 import 主项目代码；能力通告来自真实探测；无绝对路径/凭据泄漏。

## 8. 变更记录

- 2026-09-08：初版（S3.2/S7/S8 三票 + 排序与边界）
- 2026-09-09：v2 —— 将三条支线拆为 15 张开发票，先行实施不依赖硬件的 `S3.2-RT-01` 红队 fixture 与 fail-closed gate；真实 CUDA、第三方网络和 MCP 服务均后置。
- 2026-09-09：v3 —— 完成 `S3.2-RT-01`：新增四类共 12 个无网络/无权重红队 fixture、统一 gate 和稳定报告对象；专项测试 3 passed，下一票为 `S3.2-RT-02`。
- 2026-09-09：v4 —— 完成 `S3.2-RT-02`：evaluation report/promotion gate 接入红队 blocked rate、schema valid rate、越权通过数；新增集成断言，默认红队门为 100% 拦截、schema valid≥98%、越权通过=0；下一票为 `S8-MEM-01`。
- 2026-09-09：v5 —— 完成 `S8-MEM-01`：新增独立用户-owned SQLite memory store，落地 fact/preference/decision 三类条目、scope 硬过滤、来源与时间戳、软删除/显式失效/过期状态和审计保留；专项测试 4 passed，下一票为 `S8-MEM-02`。
- 2026-09-09：v6 —— 完成 `S8-MEM-02`：memory FTS5 检索与生命周期过滤、citation、有界三层预算（memory/RAG/recent context）和 omitted/truncated 输出落地；支持可选 embedding provider 对 FTS 候选 rerank，失败自动回退 FTS；专项测试 62 passed，下一票为 `S8-MEM-03`。
- 2026-09-09：v7 —— 完成 `S8-MEM-03`：ContextPolicy 压缩时接入用户-owned memory 双写；仅保存用户明确事实/偏好/决定或程序显式候选，助手普通回答不自动入库；同源显式候选覆盖摘要候选，store fingerprint 负责跨次去重；专项测试 65 passed，下一票为 `S8-E2E-01`。
- 2026-09-09：v8 —— 完成 `S8-E2E-01`：新增 MemoryWorkflow 串联 remember/recall/delete/invalidate；验证跨 session SQLite 重开后的召回、生命周期清除、scope 越权 fail-closed，以及 prompt injection 与 12 个内置红队样本联合拦截；全量专项测试 68 passed，下一票为 `S7-NET-01`。
- 2026-09-09：v9 —— 完成 `S7-NET-01`：新增 harness 独立 `tools/network.py`，提供生产网络默认关闭的 `web_fetch`/`web_search`；每跳执行 HTTPS、主机、DNS 公网地址、重定向复检，拒绝 loopback/RFC1918/ULA/CGNAT/metadata/保留地址，并执行响应大小、文本长度、content-type、超时和代理凭据门；搜索结果做 URL 复检、长度限制和 SHA-256 citation。定向测试 5 passed，全量 Harness 测试 73 passed；下一票为 `S7-NET-02`。
- 2026-09-09：v10 —— 完成 `S7-NET-02`：新增独立 `QLHToolAdapter`，向可配置 QLH `/api/tool` 端点映射 `qlh.tool_request.v1`；默认使用离线占位 transport，不创建真实网络副作用。远端结果严格校验 schema、request_id、items/citations、响应策略和错误码，再映射为 `qlh.harness.tool_result.v1`；scope、参数扩权、无效 URL、结果超限和 retryable 错误均保持稳定合同。fake QLH 专项 8 passed，全量 Harness 测试 81 passed；下一票为 `S7-TOOL-01`。
- 2026-09-09：v11 —— 完成 `S7-TOOL-01`：新增 `ToolContextBuilder` 和 `ToolContextPolicy`，将本地/远端成功工具结果转换为固定字段集 `qlh.tool_context.v1`；只保留有界 title/url/snippet 与 citation，fetch 富结果中的正文/headers/provider 字段不会透传，超限裁剪显式标记 `truncated`。注入前通过 `CapabilityGate`，要求 `tool_result_reinjection=verified`；自治模式额外要求 `autonomous_tools`，candidate/declared/unknown/无 profile 均 fail-closed。专项 12 passed，全量 Harness 测试 93 passed；下一票为 `S7-MCP-01`。
- 2026-09-09：v12 —— 完成 `S7-MCP-01`：新增独立 `harness_workbench/mcp_server/`，以标准 JSON-RPC 方法 `initialize`、`tools/list`、`tools/call` 暴露 chat、sessions、rag、memory、images 和 web 工具；内置工具由 harness store/adapter 注入，未配置能力只通告合同且调用 fail-closed。新增严格对象 schema 校验、未知参数拒绝、只读/写入/破坏性/open-world annotations；session create 与 rag search fixture 成功闭环。新增 loopback-only SSE transport、外部 MCP 配置声明和注入式 fake discovery/call 隔离通道，未创建真实第三方进程/连接。专项 10 passed，全量 Harness 测试 103 passed；下一票为 `S7-MCP-02`。
- 2026-09-09：v13 —— 完成 `S7-MCP-02` 本机开发门：`api_layer` 新增 `/v1/mcp/manifest`、`/v1/mcp/tools`、`/v1/mcp/call`、`/v1/mcp/rpc`，全部复用同一注入式 MCP registry/store/adapter；manifest 显式通告 stdio、loopback SSE 和 external MCP configuration-only 状态。React Harness 工作台新增 `MCP 控制面`，按后端 manifest 展示工具、schema、configured/read-write 状态，支持 JSON 参数调用和错误/结果回显；桌面与窄屏 smoke 覆盖 MCP 导航、调用、主题、焦点和横向溢出。MCP/API/UI 定向 15 passed，Harness 全量 105 passed，全仓库 3324 passed、19 skipped；真实第三方 MCP 服务、认证与权限仍留后置验收。

## 9. `S3.2-RT-01` 实施记录

本票只实现契约层安全边界，不启动模型、网络或 CUDA：

- `harness_workbench/eval/red_team.py` 固化 `qlh.harness.red_team.v1`，登记 prompt injection、工具越权、图片路径、上下文注入四类 12 个 fixture；fixture digest 可复现，报告只返回决策摘要，不携带真实用户路径或凭据。
- `RedTeamGate` 在模型/adapter 之前执行：不受信 system role、越狱标记、非 local scope、未 verified capability、非 allowlist 工具、路径遍历/绝对路径/符号链接/超大资产、非法 STATE 字段和未经确认删除均 fail-closed；gate checker 自身异常也转为 blocked。
- `RedTeamReport` 输出 blocked 数量/比例、决策 schema 有效率和越权通过数，为下一票接入 evaluation report 保留稳定字段；允许 fully-verified、local、allowlist 工具通过，避免把安全门误写成全拒绝。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_red_team.py tests/test_harness_context_engine.py tests/test_harness_adaptation_eval.py -q
18 passed
```

## 10. `S3.2-RT-02` 实施记录

本票把 RT-01 的安全结果接入现有评估报告和晋级门，不启动模型、网络或 CUDA：

- `build_evaluation_report(..., red_team_report=...)` 在 `metrics` 中落盘 `red_team_blocked`、`red_team_fixture_count`、`red_team_block_rate`、`schema_valid_rate`、`unauthorized_pass_count`，并保留完整 `red_team` 摘要；未传入红队报告时保持原有报告兼容行为。
- `promotion_gate` 增加可配置阈值，默认要求危险样本拦截率 100%、决策 schema 有效率至少 98%、越权通过数为 0；空报告、阈值不足或越权通过会返回 `candidate`，不会进入生产资格。
- 集成测试覆盖安全报告字段和晋级拒绝路径，避免只测红队执行器而遗漏报告落盘/晋级判断之间的断链。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_red_team.py tests/test_harness_adaptation_eval.py -q
11 passed
```

本票完成后进入 `S8-MEM-01`；其实施记录见 §11。真实生图执行器仍不作为前置。

## 11. `S8-MEM-01` 实施记录

本票只实现长期记忆的用户-owned SQLite 基础层，不接入 embedding、检索或上下文预算：

- 新增 `harness_workbench/memory/store.py` 与 `memory/__init__.py`。`MemoryStore` 使用用户指定的独立 SQLite 文件，启用 WAL、外键、busy timeout 和 FULL synchronous，不复用 session 数据库。
- `memory_entries` 固化 `fact`、`preference`、`decision` 三类条目，记录 `owner_scope`、来源 session/message、创建/更新时间、指纹、metadata、有效期和生命周期状态；scope 在所有读取与修改入口硬过滤。
- 删除是软删除且必须传 `confirm=True`；`invalidate()` 用于显式失效；过期、失效、删除条目保留审计内容，但默认不会进入 active 列表。路径形态 scope/source 标识和不可序列化 metadata 会被拒绝。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_memory_store.py tests/test_harness_red_team.py tests/test_harness_adaptation_eval.py -q
15 passed
```

本票完成后进入 `S8-MEM-02`；其实施记录见 §12。本票暂不改变 API 层和生产路由。

## 12. `S8-MEM-02` 实施记录

本票在 MEM-01 基础上实现离线检索和有界注入，不接入生产 embedding 服务或改变 API 路由：

- `memory_entries_fts` 使用 SQLite FTS5 索引 memory 内容；`MemoryStore.search()` 以 FTS 候选为入口，再由主表强制校验 owner scope、软删除、显式失效和有效期，返回带 score、fingerprint、来源 session/message 的 `MemoryHit` citation。
- `MemoryRetriever` 默认只使用 FTS；传入与 S4 相同形状的 embedding provider 时，仅对 FTS 候选做 cosine rerank。provider 缺失、维度异常或运行失败均回退 FTS，不把网络或模型依赖引入本票。
- `LayeredBudget` 将单一 `input_budget` 分成 memory、RAG、recent context 三层；`build_layered_context` 按 tokenizer 计数，只接纳完整条目，超预算记录 `omitted_count` 并置 `truncated=true`，不会静默截断内容；报告保留各层 token/省略计数和 citation。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
62 passed
```

本票完成后进入 `S8-MEM-03`；其实施记录见 §13。本票已验证跨 scope 和生命周期过滤，未宣称跨会话端到端验收完成。

## 13. `S8-MEM-03` 实施记录

本票把上下文压缩与长期记忆接起来，但不把摘要或助手回答自动视为事实：

- `ContextPolicy.build()` 增加可选 `memory_store`、`memory_owner_scope` 和 `memory_source_session_id` 参数；只有发生压缩且存在被省略的旧轮时才触发双写，旧调用不改变行为。
- `memory/extract.py` 提供规则化 `MemoryCandidate`：识别用户明确的事实、偏好、决定表达，或程序显式的 `metadata.memory={kind,content}` 候选；摘要只负责规范化，不能凭空制造证据。候选、来源 message id、写入 entry id 和失败类型会进入 `ContextSnapshot`/notice 审计字段。
- `MemoryStore.add()` 默认按 scope/kind/fingerprint 复用 active 条目；显式候选优先于同源 STATE 摘要候选，避免一次压缩写入两条语义相同的记录。写入失败不阻断对话，但产生 warning notice。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
65 passed
```

本票完成后进入 `S8-E2E-01`；其实施记录见 §14。本票仍不宣称跨会话生产 API 或自动记忆质量验收完成。

## 14. `S8-E2E-01` 实施记录

本票新增跨会话 memory workflow，把前面三张 memory 票和红队门串成可执行闭环：

- `memory/workflow.py` 提供 `MemoryWorkflow.remember/recall/delete/invalidate`。同一用户-owned SQLite 文件被第二个 workflow 实例重新打开后，仍能按 owner scope 检索第一会话写入的条目，citation 保留来源 session/message。
- `recall()` 使用 FTS-first `MemoryRetriever` 和有界 `LayeredContext`；删除/失效后检索立即排除条目，跨 scope 读取或生命周期修改均由 store 硬过滤并返回 `KeyError`。
- `remember()` 对 untrusted memory content 复用 `RedTeamGate` 的 prompt-injection 检查；注入内容在落盘前拒绝，内置 12 个红队 fixture 仍保持全量 blocked。workflow 对红队的导入采用懒加载，避免 context engine 与 eval 的循环依赖。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
68 passed
```

本票完成后进入 `S7-NET-01`；实施记录见 §15。

## 15. `S7-NET-01` 实施记录

本票只实现 harness 侧的受限联网工具和离线可验证安全门，不开启真实生产网络：

- 新增 `harness_workbench/tools/network.py` 与 `tools/__init__.py`。模块不导入主项目运行时，使用可注入 transport 与 DNS resolver 作为确定性测试边界；默认 `UrllibTransport` 只有在服务端策略 `production_network_enabled=true` 且请求显式 `allow_external=true` 时才允许运行。
- `web_fetch` 支持 HTTPS、有限重定向、DNS 每跳复检、最大响应字节数、文本字符数、UTF-8 解码、HTML 脚本/样式剥离、允许 content-type 和 SHA-256 摘要。URL 与每次重定向均拒绝凭据、fragment、loopback、RFC1918、link-local、ULA、CGNAT、metadata、非公网及 RFC 保留地址。
- `web_search` 对 SearXNG 风格 JSON 结果执行结果 URL 安全复检、标题/摘要长度限制、top-k 上限和 citation digest；无安全结果时 fail-closed。代理只能使用无凭据 HTTP(S) URL，不能由请求参数临时放宽策略。
- `execute()` 输出稳定的 `qlh.harness.tool_result.v1` 信封，携带 request id、工具名、items、citations、截断状态及策略观测；生产网络开关按实际 policy 反映，避免把离线 fixture 与真实网络状态混淆。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_network_tools.py -q
5 passed
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
73 passed
```

本票没有宣称真实外网、真实搜索服务或主项目路由已验收；实施记录见 §16。

## 16. `S7-NET-02` 实施记录

本票只实现远端 QLH 映射和错误合同，不启动真实 HTTP transport：

- 新增 `harness_workbench/tools/remote.py` 与公开导出。`QLHToolAdapter` 接受与主项目 ToolGateway 相同的 `qlh.tool_request.v1` 请求，只允许 `web_search`/`web_fetch`，严格校验 `local_user`、`explicit_opt_in`、deadline、参数字段和 URL 语法；未显式 `allow_external=true` 时在 transport 前拒绝。
- 远端端点默认 `/api/tool`，路径可由 `QLHToolAdapterConfig` 配置；base URL 只做语法/凭据/query/fragment 校验。默认 transport 是离线占位实现，只有调用方显式注入 HTTP/QLH transport 才可能发起远端请求，真实网络验收因此保持后置。
- 返回值必须是 `qlh.tool_result.v1`。适配器拒绝未知字段、schema/request_id 不匹配、无界文本/条目、私有或非 HTTPS URL、非法 citation digest、超限 bytes/redirects/content-type，并将合法结果映射为 harness 的 `qlh.harness.tool_result.v1`，不透传远端原始字段。
- 远端 `status=error` 被映射为 `RemoteToolError`，保留稳定 `code`、`message`、`retryable` 和 502/403/400 分类；transport 未配置或失败分别使用 `remote_transport_unavailable` / `remote_transport_error`，便于后续 ToolRouter 做明确回退。

验证证据：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_remote_tool.py -q
8 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
81 passed
```

本票完成后进入 `S7-TOOL-01`；实施记录见 §17。

## 17. `S7-TOOL-01` 实施记录

本票实现工具结果到普通回答模型的安全回灌边界，不接生产聊天路由：

- 新增 `harness_workbench/tools/context.py`。`ToolContextBuilder` 同时接受主项目 `qlh.tool_result.v1` 与 harness `qlh.harness.tool_result.v1`，先校验 tool/request identity、成功状态和 schema，再输出固定的 `qlh.tool_context.v1`：`schema`、`role`、`name`、`request_id`、`items`、`citations`、`truncated`。
- 结果只允许有界 `{title,url,snippet}` 和 `{url,sha256}`。本地 `web_fetch` 的 `text/final_url` 可转换为 snippet，但 headers、Cookie、原始 provider 字段、响应正文之外的 metadata 均不透传；项目策略限制 item/citation 数量、单项字符数和总字符数，裁剪均显式设置 `truncated=true`。
- 注入前调用既有 `CapabilityGate`。host-router 需要 profile/decision 状态为 `verified` 且 `tool_result_reinjection` 为 `verified`；`autonomous_tools` 模式还必须满足 `tool_call_generation` verified、完整绑定证据和 gate 的 autonomous admission。没有证据、candidate、declared、unknown、rejected 或 scope/身份不匹配均 fail-closed。
- citation 只保留对应已注入 item 的 URL，digest 必须为小写 SHA-256；成功结果缺引用、私有/非 HTTPS URL、错误 schema、request ID 不一致和 provider error 均不会进入模型上下文。

验证证据：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_tool_context.py -q
12 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
93 passed
```

## 18. `S7-MCP-01` 实施记录

本票实现 harness 作为 MCP server 的本机/离线开发门，不接生产聊天路由，也不启动真实第三方 MCP 进程或网络连接：

- 新增 `harness_workbench/mcp_server/`。`MCPServer` 实现 JSON-RPC `initialize`、`ping`、`tools/list`、`tools/call`，`StdioMCPTransport` 使用 newline-delimited JSON；notification 不回包，解析错误、非法请求和未知方法遵循稳定 JSON-RPC error code。
- `ToolRegistry` 用严格 `ToolDefinition` 管理名称、输入 schema、handler、读写/破坏性/open-world annotations 和 capability notice。schema 只接受 `type=object` 且 `additionalProperties=false` 的有界子集，顶层/数组/嵌套对象参数均递归校验；未知字段、缺失字段、类型、范围、pattern、enum 错误均拒绝。
- `register_builtin_tools()` 暴露 `chat`、`session_create`、`sessions_list`、`session_get`、`rag_search`、`rag_add_source`、`memory_search`、`memory_add`、`memory_invalidate`、`memory_delete`、`web_search`、`web_fetch`、`image_capabilities`、`image_generate`。session/RAG/memory/image/network/chat 均通过 `HarnessMCPDependencies` 注入，未配置项仍可列出合同，但调用返回 `isError=true` 的稳定 code，不伪造后端可用性。
- 读工具通过 fixture SQLite 成功验证 `rag_search`；写工具成功验证 `session_create`。memory 删除保留既有 `confirm=true` 显式确认要求，image 与 network 继续复用 harness 自有合同和默认离线/不可用策略。
- `SSEMCPTransport` 仅绑定 `127.0.0.1`/`localhost`/`::1`，提供 `/sse` endpoint event 和 `/messages?sessionId=...` POST 回传 message event；不提供公网 bind、认证或真实第三方连接承诺。
- `ExternalMCPServerConfig` 只保存 schema、server id、stdio/SSE transport、bare command 或无凭据 endpoint、参数名声明和 enabled 状态；拒绝绝对路径、凭据/query/fragment、secret-bearing env key。`ToolRegistry.declare_external()` 只登记配置，`mount_external()` 接受注入式 fake client 做工具发现和调用，并以 `<server_id>/<tool_name>` namespace 隔离碰撞；发现/调用/结果错误均转换为稳定 `MCPToolError`，不泄露原始异常。
- MCP 初始化 capability notice 明确 `external_mcp.configuration_only=true`、`real_connections=false`；本票不把 fixture discovery 记为第三方服务验收。

验证证据：

```text
.\\.venv-test\\Scripts\\python.exe -m pytest tests/test_harness_mcp_server.py -q
10 passed
.\\.venv-test\\Scripts\\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
103 passed
```

## 19. `S7-MCP-02` 实施记录

本票完成 harness MCP API bridge 和 React 前端跟随后端能力的本机开发门；不宣称真实第三方 MCP 服务已接入：

- `harness_workbench/api_layer/app.py` 新增 MCP HTTP bridge。`/v1/mcp/manifest` 和 `/v1/mcp/tools` 从同一个 `MCPServer` registry 生成工具合同；`/v1/mcp/call` 和 `/v1/mcp/rpc` 分别提供简化调用与标准 JSON-RPC 入口，notification 返回 204。API 注入的 session、RAG、memory、network、image 和 chat 依赖与 MCP server 共用，避免前后端目录漂移。
- `harness_workbench/ui_react/src/data.ts` 新增 manifest/call 类型和请求函数；`App.tsx` 新增 `MCP 控制面`，动态展示工具目录、configured 能力、读写/破坏性/open-world annotations、输入 schema、JSON 参数编辑器、调用状态和结构化结果。未配置工具保持禁用，不把声明当成可用能力。
- `styles.css` 新增 MCP 双栏 inspector、状态条、schema/result 面板及窄屏单列布局；`scripts/visual_smoke.mjs` 使用 manifest/call fixture 验证桌面与移动端可浏览、可调用且无横向溢出。截图产物位于 `harness_workbench/ui_react/build/ui-visual-smoke/`。
- API 测试覆盖 manifest、session create、RAG add/search、chat、JSON-RPC initialize 和 notification；前端 smoke 覆盖 `rag_search` 调用。External MCP 仍显示 `configuration_only=true`、`real_connections=false`，stdio/SSE 只验证 loopback/进程入口合同。

验证证据：

```text
.\.venv-test\Scripts\python.exe -m pytest tests/test_harness_mcp_api.py tests/test_harness_mcp_server.py tests/test_harness_ui.py -q
15 passed
.\.venv-test\Scripts\python.exe -m pytest (Get-ChildItem tests -Filter 'test_harness_*.py').FullName -q
105 passed
.\.venv-test\Scripts\python.exe -m pytest -q
3324 passed, 19 skipped
cd harness_workbench/ui_react
npm run build
npm run visual:smoke -- http://127.0.0.1:5181/
desktop/mobile: MCP navigation, tool call, theme, focus, no overflow passed
```

本票完成后进入 `S5-CLOSE-01`；真实第三方 MCP 端点、认证、权限和客户端互操作仍需外部服务验收票，不作为本机开发门结论。
