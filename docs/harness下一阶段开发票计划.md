# Harness 下一阶段开发票计划（S3.2 生图后置+红队 / S7 联网+MCP / S8 长期记忆）

> 状态：**计划已定（S3.2 / S7 / S8 待排期）**；前置状态回顾：S1-S4 本机/离线开发门已完成（v10 记录），S5（收口评估）与 S6（React 工作台 + Textual TUI）已列计划表，本文档新增三张票：**S3.2（生图真机后置 + 安全红队合并）、S7（联网搜索 + 轻量 MCP 服务）、S8（小模型长期记忆：RAG + 上下文压缩 + 本地 memory）**
>
> 创建日期：2026-09-08
> 适用范围：harness 子项目下一阶段票；不与 [WEB-TOOL 联网支线](联网搜索与轻量Fetch工具调用可行性调研与分期计划.md) 合并（那些是主项目运行时工具，本票是 harness 侧能力与对外服务）；不覆盖训练微调。

---

## 1. 背景（现状盘点）

- **已完成**：S1 上下文引擎、S1.5 模型画像与能力门、S2 API 层 + llama-server adapter、S2.5 定制化实验台（adaptation/eval/Pareto）、S3 生图工作区（离线门）、S4 远端与 RAG（离线门）——全部"本机/离线开发门"，**真实运行时验收整体后置**。
- **已列计划**：S5 评估收口（ollama 对照/契约漂移/Pareto）、S6 工作台 UI（`ui_react/` + `tui.py`）。
- **验证命令基线**：`.venv-test\Scripts\python.exe -m pytest tests/test_harness_*.py -q`（当前各票 7~8 passed）。

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

## 7. 风险与边界

1. 范围控制：S7 的联网工具**不做**主项目 G2 全套策略复刻（只保留核心 SSRF 门）；MCP 服务不替代 harness 自身 run-loop（客户端自治）。
2. S8 风险：memory 内容污染/过期事实 → 软删除 + 来源/时间戳 + 显式失效；"建议 pin"只做建议不自动写。
2b. S7 边界：MCP **预留通道**（registry + 端点配置 + schema 校验）本期只做接口与 fixture 验证，**真实第三方 MCP 服务接入**（依赖外部 server 地址/认证/权限）登记后置票，不提前引入不可控依赖。
3. 生图真机（S3.2）：与主项目 SD 侧车共享工件但**不同时运行**（显存互斥）；真实执行器验证必须真 CUDA 卡——不可用则保持 `unavailable` 码（不关门）。
4. 全票贯穿约束：harness 不 import 主项目代码；能力通告来自真实探测；无绝对路径/凭据泄漏。

## 8. 变更记录

- 2026-09-08：初版（S3.2/S7/S8 三票 + 排序与边界）
