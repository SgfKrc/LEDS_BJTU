# 外部推理服务 Provider 接入指南（路线 B 阶段 1 PoC）

> 状态：PoC 已实现（external_provider + 聊天路径外部路由）
> 更新日期：2026-07-27
> 适用范围：路线 B 阶段 1 PoC 的 external_api 路由、数据作用域门控与排障
>
> 前置阅读：[张量并行外部辅助与混合拆分调研方案](张量并行外部辅助与混合拆分调研方案.md) §2.2
>
> 一句话：集群把**整条请求**作为路由决策交给一个 OpenAI 兼容的外部端点（租用 GPU 盒子、实验室服务器、云 API），不要求对方运行任何 QLH 代码，也无专职网关节点；**数据作用域门控是硬前提**——默认 opt_in，未显式授权的请求绝不离开集群。

---

## 1. 与 A 方案（TP 孤岛）的区别

| 维度 | 路线 A：TP 孤岛 | 路线 B：外部推理服务 |
|---|---|---|
| 接入形态 | 网关**节点**加入集群，孤岛=集群内逻辑高算力节点 | 无专职节点，主/服务节点按请求**路由**到外部端点 |
| 信任域 | 孤岛在集群信任域内（Tailscale/内网） | 外部端点在集群信任域**之外** |
| 数据边界 | 会话数据不出 mesh | 数据出集群 → **数据作用域门控（默认 opt_in）** |
| 引擎标识 | `engine="island"`（节点本地引擎） | `engine="external_api"`（按请求的执行方式，不可 load） |
| 路由粒度 | 主节点按节点权重把整请求派给孤岛节点 | 每条请求独立决策：flag / 长上下文阈值 |
| 取消语义 | 断 SSE，孤岛可能算完（best-effort） | 相同（OpenAI 端点无服务端取消原语） |
| 代码复用 | `island_engine.IslandEngine` | **组合复用**同一客户端作传输层，错误重分类 |

## 2. 架构

```text
前端 / Android / curl
   │ POST /api/chat  { message, allow_external, prefer_external }
   ▼
api_server（服务节点）
   │ decide_external_route()  ← 纯函数路由决策
   │   不满足 → 原本地 / 流水线 / 孤岛路径，行为不变
   │   满足   → ExternalChatClient（数据作用域最后关口在 chat() 内）
   ▼
外部 OpenAI 兼容端点（vLLM / SGLang / 云 API，集群信任域之外）
        POST /v1/chat/completions（非流式 / SSE）
```

- 传输层组合复用 `IslandEngine`（凭据脱敏、URL 内嵌账号→BasicAuth、SSE 解析、chunk 边界取消）；孤岛语义与测试逐字节不变。
- 任务链侧：`external_provider.ExternalOpenAIProvider` 实现 `ExecutionProvider` 契约（`full_inference`/`aggregate`），可注册进 `ProviderRegistry` 供任务系统选择；Stage 的 `root_input` 必须携带 `"allow_external": true` 才可外发（与聊天路径同一门控函数）。PoC 不自动注册进聊天任务链协调器（避免"首个兼容者"字典序选择劫持默认本地工作流）。

## 3. 配置项（QLH_EXTERNAL_*，与 QLH_ISLAND_* 同风格）

| 变量 | 必填 | 说明 | 示例 |
|---|---|---|---|
| `QLH_EXTERNAL_ENABLED` | 是 | 外部推理服务总开关 | `1` |
| `QLH_EXTERNAL_BASE_URL` | 是 | OpenAI 兼容端点 | `https://gpu-box.example.com:8000` |
| `QLH_EXTERNAL_API_KEY` | 否 | Bearer 凭据（日志/状态自动脱敏） | `sk-xxxx` |
| `QLH_EXTERNAL_MODEL` | 否 | 服务端模型名；留空自动取 `/v1/models` 首个 | `Qwen/Qwen2.5-7B-Instruct` |
| `QLH_EXTERNAL_TIMEOUT` | 否 | 请求超时秒数（默认 120） | `180` |
| `QLH_EXTERNAL_CONNECT_TIMEOUT` | 否 | 连接超时秒数（默认 5） | `5` |
| `QLH_EXTERNAL_DATA_SCOPE` | 否 | 数据作用域：`deny` / `opt_in`（默认）/ `allow_all`；**取值写错（如 `denied`/`off`）一律 fail-closed 回落 `deny` 并打印 WARN**，未设置才是 `opt_in` | `opt_in` |
| `QLH_EXTERNAL_MIN_PROMPT_CHARS` | 否 | 长上下文卸载阈值，0=关（默认）。统计口径是**本轮消息**字符数（CJK 按 1 字符计），不含历史与系统提示词 | `2048` |
| `QLH_EXTERNAL_LABEL` | 否 | 展示名 | `实验室vLLM` |

## 4. 数据作用域说明（隐私默认）

**默认 opt_in：没有任何请求会在你不知情的情况下离开集群。**

| 档位 | 未带 flag 的请求 | 带 `allow_external=true` 的请求 |
|---|---|---|
| `deny` | 不出集群 | **仍不出集群**（硬禁用） |
| `opt_in`（默认） | 不出集群 | 可路由到外部端点 |
| `allow_all` | 可路由到外部端点 | 可路由到外部端点 |

- 什么会离开集群：被放行请求的**对话消息**（history + 当前消息）与采样参数；`Authorization` 凭据只发往所配置端点。
- 什么不会离开集群：未放行请求的任何内容；集群拓扑/节点画像/日志；凭据不落日志，`base_url` 在 `/api/status`、metrics、日志中一律脱敏。
- 双重防线：路由决策（`decide_external_route`）之外，`ExternalChatClient.chat/chat_stream` 内部在**发出请求前的最后一行**再次执行 `ensure_external_scope_allowed`——未来任何重构都无法绕过。
- 被作用域拒绝的带 flag 请求会记一条 INFO（`数据作用域拒绝外部路由: reason=...`），**不含消息正文**，随后走原本地路径。
- 路由触发条件（作用域放行后）：`prefer_external=true`，或消息字符数 ≥ `QLH_EXTERNAL_MIN_PROMPT_CHARS`（>0 时，"长上下文卸载"启发式）。

## 5. 前端 / HTTP 调用示例

```bash
# 1) 显式授权 + 优先外部（opt_in 档位下的标准用法）
curl -X POST http://<节点IP>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "介绍一下张量并行", "allow_external": true, "prefer_external": true}'
#   → metrics.engine = "external_api", metrics.execution_mode = "external_api"

# 2) 只授权不强制：仅当消息达到 QLH_EXTERNAL_MIN_PROMPT_CHARS 才外发
curl -X POST http://<节点IP>:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "<很长的上下文...>", "allow_external": true}'

# 3) 流式（fast 模式为真流式逐 token；full 模式为单 done 事件）
curl -N -X POST http://<节点IP>:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "介绍一下张量并行", "streaming_mode": "fast",
       "allow_external": true, "prefer_external": true}'

# 4) 状态检查（端点已脱敏；reachable 为 ~30s 缓存的健康检查结果）
curl http://<节点IP>:8000/api/status | python -m json.tool
#   → "external": {"enabled": true, "data_scope": "opt_in", "reachable": true, ...}
```

前端 JS（`frontend/src/api/client.js`）：`sendMessage(msg, { allowExternal: true, preferExternal: true })`，缺省均为 `false`，旧客户端行为不变。

## 6. 取消语义（best-effort）

- OpenAI 兼容端点**无服务端取消原语**。QLH 侧取消（`POST /api/chat/generations/{id}/cancel` 或断开 SSE）在 **chunk 边界断流并关闭连接**；外部端可能继续算完当前请求（计费按外部端计量）。
- 非流式调用在携带取消事件时内部改走流式消费，因此同样具备 chunk 边界取消能力；取消时返回已收到的部分内容，`finish_reason="cancelled"`，token 统计标注 `usage_estimated=true`。
- 任务链 Provider 的 `cancel(attempt_id)` 语义相同。

## 7. 故障回退行为

| 场景 | 行为 |
|---|---|
| 外部调用失败 + 本地引擎可用 | 回退本地路径继续推理；metrics `fallback=true`、`fallback_reason="external_api_failed: ..."` |
| 外部调用失败 + 本地无引擎 + `prefer_external` | 干净的 502 中文错误（不再尝试加载本地模型） |
| 外部调用失败 + 本地无引擎（阈值触发） | 尝试自动加载本地默认模型；仍失败 → 502 中文错误 |
| 作用域拒绝（deny / opt_in 无 flag） | 与外部功能不存在时完全一致：走原本地路径 |
| `QLH_EXTERNAL_ENABLED` 未设 | 全部路由逻辑短路，零行为变化 |

## 8. 排障表

| 现象 | 中文错误 | 可能原因 | 处置 |
|---|---|---|---|
| "外部推理服务不可达" | `ExternalUnreachableError` | 端点未启动 / 防火墙 / URL 写错 | `curl <BASE_URL>/v1/models` 复核连通性 |
| "外部推理服务超时" | `ExternalTimeoutError` | 外部过载、上下文过长 | 调大 `QLH_EXTERNAL_TIMEOUT` |
| "外部推理服务 HTTP 错误：… 401/403" | `ExternalHTTPError` | `QLH_EXTERNAL_API_KEY` 缺失或错误 | 核对外部端鉴权 |
| 401 且日志有"同时配置了 URL 内嵌凭据和 API Key" | — | HTTP 只允许一个 `Authorization` 头，URL 里的 `user:pass` 会顶掉 Bearer，API Key 发不出去 | 二选一：从 BASE_URL 去掉 `user:pass`，或改用反代白名单放行 |
| 设了 `DATA_SCOPE` 却仍被拒绝出网 | `ExternalScopeDeniedError` | 取值拼写错误被 fail-closed 成 `deny`（启动日志有 `[QLH][WARN]`） | 改成 `deny`/`opt_in`/`allow_all` 三者之一 |
| "外部推理服务 HTTP 错误：… 404" | `ExternalHTTPError` | BASE_URL 多写/漏写 `/v1` 前缀 | BASE_URL 只写到端口 |
| "外部推理服务流式响应中断" | `ExternalStreamInterruptedError` | 外部端崩溃 / 反代缓冲截断 SSE | 查外部服务日志；nginx 关缓冲 |
| "数据作用域禁止外部路由" | `ExternalScopeDeniedError` | scope=deny，或 opt_in 未带 `allow_external` | 确认策略后带 flag 重试或调整档位 |
| 请求总走本地、flag 已带 | — | `MIN_PROMPT_CHARS=0` 且未设 `prefer_external` | 设 `prefer_external:true` 或配置阈值 |
| `/api/status` 无 `external` 段 | — | `QLH_EXTERNAL_ENABLED` 未设或非 `1/true` | 检查环境变量后重启 |
| 连接日志出现"孤岛后端/孤岛引擎"字样 | — | 传输层组合复用孤岛客户端（仅连接期日志） | 正常现象；面向调用方的错误均为"外部推理服务"文案 |

## 9. 限制与后续

- **多轮会话粘性**：外部端点不保证跨请求 KV，长对话反复外发=每轮重付 prefill；建议只把新发起的长任务路由出去（调研方案 RQ8）。
- **并发闸门**：聊天路径当前与本地推理共享执行互斥；任务链 Provider 的 `max_concurrency` 为客户端侧闸门（外部端无排队信息）。
- **verify Stage**：调研方案设想的 `verify` 类型在任务系统中尚未定义，Provider 暂只声明 `full_inference`/`aggregate`。
- **成本计量**：PoC 只落 usage 到 metrics/journal，未做预算硬限。
