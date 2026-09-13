# reasonix-codex-bridge Harness 工具扩展与能力补齐排期（2026-09-13）

> 状态：已完成能力调研；`TOOL-RXB-EXEC-01`、`TOOL-RXB-TOOL-01` 与 `TOOL-RXB-NET-01` 已实现并通过离线回归与真实 CLI/doctor/原生抓取验收。
>
> 范围：只讨论桥接器如何安全复用 Reasonix Harness 已存在的能力，不把未验证的模型工具调用能力包装成“原生等价”。

## 1. 调研结论

本机 Reasonix CLI 为 `v1.38.7`。CLI 的 Harness 能力不是一个独立的 `reasonix tool` 子命令，而是由运行时 capability/profile 和 MCP 挂载提供：

| 能力 | 本机实际发现 | 对 bridge 的含义 |
| --- | --- | --- |
| Shell/测试/构建 | `reasonix run --allowed-tools shell` 可调用 shell；`subagent` 的既有 read/write profile 刻意没有 shell | 不能把普通 shell 直接加入写 profile；先做命名命令、argv-only、无 shell 的 host 通道 |
| 指定 URL 抓取 | `web_fetch` 可用，实测 `https://example.com` 返回 200 和标题 | 复用现有主仓 Tool Gateway 的 SSRF/重定向/大小/content-type 策略；不在 bridge 内接受任意 URL |
| 网络搜索 | `web_search` 在本机 capability catalog 中不可用；Reasonix 报告没有搜索后端。**归属确认**：按内置文档 `docs/WEB_SEARCH.md`，搜索是 **provider 侧能力**（另发一次模型请求、使用后端原生搜索，查询交给 provider 并按搜索请求计费） | **不在 bridge 自建任何搜索后端**；接通 provider 时只做透传，未接通时稳定返回 unavailable——细节见 [工具面现状](reasonix-codex-bridge工具面现状-2026-09-13.md) |
| MCP 外部工具 | `reasonix mcp list` 显示 `gitcontext` stdio 与 `shizi-wiki` HTTP；后者当前 failed | 只能做显式配置、命名空间隔离和 fail-closed 状态透传 |
| ACP/长会话 | ACP client/coordinator/registry 已有离线门；真实 provider 空会话跨进程恢复仍 blocked | 继续保留 opt-in，不把“会话创建成功”当作恢复通过 |
| 子智能体工具集 | read profile 实际生效 6 件：`read_file, grep, glob, ls, code_index, web_fetch`；write profile 追加 `edit_file, write_file`（8 件）。**`git_log`/`git_diff` 不是 Reasonix 已知工具身份**（doctor 曾连续 4 条警告），从未生效 | shell、git 只读、搜索、动态派生与消息协作仍是缺口；`web_fetch` 已由 NET-01 接入并仍归 Reasonix |

## 2. 开发票排期

优先级按“补齐写后验证闭环”与安全风险排序：

| 票号 | 优先级 | 目标 | 主要验收门 | 状态 |
| --- | --- | --- | --- | --- |
| `TOOL-RXB-EXEC-01` | P0 | 受控 shell/测试执行：命名 executable profile、argv 数组、`shell:false`、cwd/clean-tree、超时/输出上限、取消、变更检测、脱敏结果 | 离线 fixture + 真实 Reasonix CLI 启动；命令注入、越界 cwd、dirty tree、超时、截断、变更检测全覆盖 | **已完成** |
| `TOOL-RXB-NET-01` | P1 | **能力归属 Reasonix**：worker 侧直接使用 Reasonix 自带 `web_fetch`（host registry 可选工具），bridge **不自建网络栈**，只做透传与结果摘要；若未来确需自建，必须复用主仓 Tool Gateway 的 HTTPS 强制/SSRF/DNS/redirect 策略 | 失败态稳定透传；不新增网络代码路径；无任意 socket/代理覆盖 | **已完成** |
| `TOOL-RXB-NET-02` | P1 | **搜索归属 provider**：按 Reasonix 内置文档，`web_search` 由 provider 侧执行（查询外发、按搜索请求计费）。**本机不自建搜索后端**；bridge 只做能力探测（不可用即稳定返回 unavailable）与结果透传 | 无后端时不得伪造成结果；接通后 summary/sources/truncated 完整 | 排队（**已撤销"本机后端"前置**） |
| `TOOL-RXB-TOOL-01` | P1 | 工具身份对齐：`READ_ONLY_PROFILE_TOOLS` 收敛为真实有效集合（移除 `git_log`/`git_diff`；如需 git 只读走 MCP `gitcontext` 或受控 exec）；README/profile 同步；`configure verify` 增加对照 Reasonix 真实清单的校验 | 修正后 `doctor` 对两个 profile 零警告；`verify --role read/write` 均通过；`npm test` 全绿 | **已完成** |
| `TOOL-RXB-LOOP-01` | P1 | 主 agent 显式阶段编排模板：plan -> implement -> exec/test -> review；每阶段 job/checkpoint/audit 可见 | 任一阶段失败可定位、可取消、可回滚；不隐式重试或自动扩大权限 | 排队 |
| `TOOL-RXB-EVT-01` | P2 | 长任务事件/进度轮询：只暴露 job id、阶段、计数和状态，不透传敏感正文 | 中途取消、重连、事件顺序和脱敏检查 | 排队 |
| `TOOL-RXB-ACP-07` | P2 | provider-backed 非空 ACP fixture，验证强杀后的跨进程 `resume/load`，再评估生产 registry 接线 | prompt -> kill -> resume/load -> close/delete 全链路真实通过 | 受真实 provider 持久化能力阻塞 |
| `TOOL-RXB-MSG-01` | P2 | 受控多 agent 派生/消息协议；固定角色、任务作用域和并发额度 | 动态派生不越权、消息不带凭据/正文泄漏、失败可回收 | 排队，风险高于收益 |

### 2.1 明确不做

- 不把 `reasonix run --allowed-tools shell` 直接透传为任意 shell MCP 工具。
- 不在 bridge 内自建第二套网络栈，也不自建搜索后端；`web_fetch` 直接用 Reasonix 自带工具，`web_search` 归属 provider——**能丢给 Reasonix 的都交给 Reasonix**。
- 不因 QW1.8B 能输出 JSON 就宣称它具备稳定 tool-calling；小模型仍走 host-router/fallback。
- 不把 ACP 空会话、离线 fixture 或一次成功的模型调用当作跨进程持久恢复证据。

## 3. 第一票 `TOOL-RXB-EXEC-01`

实现位置：`tools/reasonix-codex-bridge/src/config.mjs`、`src/server.mjs`、`test/bridge.test.mjs`、README/CHANGELOG。

配置示例：

```json
{
  "execPolicy": {
    "enabled": true,
    "allowedPaths": ["tools/reasonix-codex-bridge"],
    "commands": [
      { "name": "bridge-test", "executable": "node", "argsPrefix": ["--test"], "maxArgs": 8 }
    ],
    "requireCleanTree": true,
    "timeoutSeconds": 300,
    "outputCharCap": 12000
  }
}
```

`reasonix_exec` 只接受已配置的命令名和字符串数组参数。桥接器解析可执行文件但不接受调用方覆盖，使用 `shell:false` 启动，执行前要求可验证的干净 Git 树，执行后返回 `qlh.reasonix.exec.v1`（退出码、截断标记和变更路径）。发现工作区变化时返回 `workspace_modified` 并停止把结果当作成功；不会替用户自动回滚。状态只展示命令名、参数前缀和有界限制。

## 4. 验收记录

- 离线定向回归：`reasonix_exec` 禁用门、命令白名单、argv 执行、输出脱敏和工作区变更检测已通过。
- 全量回归：`npm test`，99/99 通过；`npm run check`、`npm run check:links` 和 `git diff --check` 均通过。
- 真实 CLI：使用本机 Reasonix `v1.38.7` 启动 bridge，在项目树内 `build/bridge-test/exec-real-fixture/` 的干净 fixture 中执行 `node-version` 命名 profile；MCP 暴露 6 个工具，返回 `qlh.reasonix.exec.v1`、`outcome=success`、`exitCode=0`、`changedPaths=[]`；不发送模型 prompt，不访问网络。fixture 已清理。
- 工具身份：同步本机 `%APPDATA%/reasonix/skills/deepseek-worker*` 两个全局 profile 后，`reasonix doctor --json` 不再报告未知工具，`reasonix doctor capabilities --json` 为 `errors=0,warnings=0`；`configure verify --role read` 与 `--role write` 均通过。
- NET-01：两个全局 profile 已同步为 read 6 / write 8 个有效身份，其中 `web_fetch` 由 Reasonix 原生 host registry 提供；原生 `reasonix run --allowed-tools web_fetch` 与真实 bridge MCP `reasonix_run(mode=inspect)` 对 `https://example.com` 均返回 `HTTP status: 200 OK` 与标题 `Example Domain`，对保留域名 `https://example.invalid` 稳定返回 transport error 且按任务要求不重试。bridge 的 `tools/list` 仍只有 6 个 host 工具，源码未新增 URL/socket/fetch 网络路径。

## 5. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-13 | 完成 Reasonix Harness capability 调研：shell 可由 `reasonix run --allowed-tools shell` 使用，`web_fetch` 可用，`web_search` 本机不可用；建立 EXEC/NET/LOOP/EVT/ACP/MESSAGE 排期。 |
| 2026-09-13 | 开始 `TOOL-RXB-EXEC-01`：bridge 新增默认关闭的命名命令执行器与结构化结果契约。 |
| 2026-09-13 | 按内置文档与 `doctor` 实测修订工具面结论：`git_log`/`git_diff` 非有效工具身份（新增 `TOOL-RXB-TOOL-01`）；`web_fetch` 归属 Reasonix、`web_search` 归属 provider，NET-01/NET-02 改为透传与门控，撤销"本机搜索后端"前置。 |
| 2026-09-13 | 完成 `TOOL-RXB-TOOL-01`：profile 工具集收敛为 5/7 个真实身份，read/write 全局 profile 已同步；`doctor` 告警归零，`configure verify` 增加未知身份 fail-closed，99/99 回归通过。 |
| 2026-09-13 | 完成 `TOOL-RXB-NET-01`：read/write profile 接入 Reasonix 原生 `web_fetch`（6/8）；不新增 bridge 网络栈或任意 URL MCP 工具，真实 `example.com` 抓取返回 200/标题，doctor/verify 零告警。 |
