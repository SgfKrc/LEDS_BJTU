# reasonix-codex-bridge ACP 会话级恢复专项计划（2026-09-13）

> 状态：**ACP-01～ACP-04 已完成（2026-09-13），ACP-05～ACP-06 未开始**；本文把 ACP 会话级恢复的现状证据、约束、分层交付与验收门写清楚，后续仍逐票推进，不把客户端层、协调器、注册表或安全门误记为生产会话接入。
>
> 创建日期：2026-09-13
> 适用范围：`tools/reasonix-codex-bridge`（独立子项目）从"stateless per call"扩展到"ACP 持久会话 + 会话级恢复"的规划。**不覆盖** Reasonix 本体的 ACP 实现（已可用，见 §2.1），也不覆盖已落地的**任务级**续跑（checkpoint，见 §2.3）。
> 关联：[完善方向](reasonix-codex-bridge完善方向-2026-09-12.md)（票 `TOOL-RXB-E2` 即本文前身——其要求"先登记会话生命周期、128MB 前 compact/rotate、stateless 回退和失败无副作用边界，不立即接生产路径"）、[审计报告](reasonix-codex-bridge审计报告-2026-09-12.md)。

---

## 1. 结论摘要

1. **ACP 本身可用**：`reasonix acp` 实测可 initialize、`session/new`，能力声明含 `loadSession: true` 与 `sessionCapabilities.{list,resume,close,delete}`。
2. **不可用的是 bridge 层**：`src/acp-prototype.mjs` 为 design-only 模块（源码注释明示"intentionally not imported by server.mjs"），bridge 从未接入 ACP；MCP 工具面始终走 `reasonix subagent run` 的一次性子进程路径。
3. **现状替代**：`012f3c1` + `a596921` 已交付**任务级**续跑（durable task checkpoints，原子 claim）——解决"长任务中断即丢"，但不保留会话上下文（对话与工具轨迹）。
4. **本文交付**：6 张票 `TOOL-RXB-ACP-01`～`06`，覆盖客户端层、compact 真实化、生命周期、安全回归、共存切换与验收演练。

---

## 2. 现状证据

### 2.1 ACP 侧（本机实测，2026-09-13）

最小握手（`reasonix acp` + `initialize` + `session/new`，CLI v1.38.7）关键返回：

| 能力 | 实测值 |
| --- | --- |
| 会话加载 | `agentCapabilities.loadSession: true` |
| 会话能力 | `sessionCapabilities: { list, resume, close, delete }` |
| 提示能力 | `image: false`、`audio: false`、`embeddedContext: true` |
| MCP 能力 | `http: true`、`sse: false` |
| 扩展 | `_reasonix.io/session/status`、`sessionSteer`、`sessionInbox`（enqueue/retry/setPaused…）、`sessionReloadExtensions` |
| `session/new` 返回 | `sessionId`（如 `2759994c-2465-4691-a5fb-03e9bdd61eac`）、12 个可选模型、3 个模式（normal/plan/goal）、`tool_approval` 三档（ask/auto/yolo） |
| 方法表（二进制内字符串） | `session/load`（打开持久会话**并重放** transcript）、`session/resume`（打开但**不重放**）、`session/prompt`、`session/update`、`session/cancel`、`session/delete`、`session/set_mode` |

### 2.2 bridge 侧

- `src/acp-prototype.mjs`（91 行）保留**纯决策原型**：定义 `ACP_HISTORY_HARD_CAP_BYTES = 128MB`、`ACP_COMPACT_TRIGGER_RATIO = 0.75`、事务式 `compactHistory`（摘要失败不改原历史）与 `prepareSessionContinuation` 的四分支（`append` / `compact` / `rotate` / `per_call` 回退）。`src/acp-session.mjs`（ACP-02）将该决策接入 `AcpClient`，提供有界规则摘要、替换会话事务和显式 stateless 回退，但仍未由 server 生产接线。
- `src/acp-client.mjs` 已提供 ACP-01 的零持久化客户端；bridge server 尚未 import 它，因此仍不存在生产会话注册表、compact 或 MCP transport 切换。
- 现有 MCP 工具面：`reasonix_run` / `reasonix_status` / `reasonix_rollback`，其中 worker 调用固定为 `reasonix subagent run <profile> --model <ref> --max-steps N --dir <cwd> -- <task>`。

### 2.3 已落地的替代（任务级续跑）

- `012f3c1 feat: add durable task checkpoints`、`a596921 fix: claim checkpoints atomically`：checkpoint 落盘 + 原子 claim，任务中断后可续跑。
- 边界：续跑需要调用方重新提供上下文；**没有**对话历史与工具轨迹的恢复能力——这正是 ACP `session/resume` 能补的部分。

---

## 3. 为什么现状没接（三条约束）

| 约束 | 内容 | 对应票 |
| --- | --- | --- |
| **128MB 会话历史硬上限** | 持久会话会累积历史，必须在 0.75 触发比与 128MB 硬上限下完成 compact/rotate 的验证；ACP-02 已接入有界规则 summarizer 与替换会话协调器，后续仍需在 ACP-06 做长会话演练 | `ACP-06` |
| **stateless 是现有安全契约** | 一次性调用、可审计、无残留。改长驻会话需新增崩溃恢复、并发串行化、会话/进程清理，以及持久历史下的越权风险控制（写白名单目前按调用检查） | `ACP-05`、`ACP-06` |
| **无验收票** | 会话生命周期验证（崩溃后 resume、compact 正确性、rotate 边界、取消/并发、零泄漏）此前无票覆盖 | `ACP-06` |

---

## 4. 目标与非目标

**目标**：在 bridge 内接入 ACP 持久会话，提供**会话级恢复**（`session/load` / `session/resume`），并与现有 stateless 路径**共存、可切换、可回退**。

**非目标**：
- 不替换现有 MCP 工具面（`reasonix_run/status/rollback` 语义保持不变）；
- 不把 128MB 历史上限做成可配置；
- 不允许"跳过 compact 直接长会话"；
- 不在 `ACP-05` 完成前默认启用 ACP。

---

## 5. 分层交付（票）

| 票号 | 主题 | 交付要点 | 依赖 | 验收门 |
| --- | --- | --- | --- | --- |
| `TOOL-RXB-ACP-01` | ACP 客户端层 | spawn `reasonix acp`；JSON-RPC over stdio 客户端（initialize → 能力探测 → `session/new`/`load`/`resume`）；`session/prompt` 的 `session/update` 事件流汇总为调用结果；`session/cancel` 接超时 | — | **已完成**：create→prompt→事件流→干净关闭 fixture 与真实握手；旧 CLI（`loadSession=false`）能力可探测，实际 per-call 回退由 ACP-05 接线；stdin 关闭无残留进程 |
| `TOOL-RXB-ACP-02` | compact 策略真实化 | `src/acp-session.mjs` 为 `prepareSessionContinuation` 接入有界规则 summarizer（保留可注入 LLM 接口）；四分支接入会话循环；记录每次决策（字节、动作、耗时） | ACP-01 | **已完成**：离线回归覆盖 0.75 触发、append、compact、rotate、摘要失败与替换会话失败回退；摘要/替换失败均回退 `per_call` 且历史不变；替换成功后旧会话 close/delete，rotate 后协调器历史低于硬上限 |
| `TOOL-RXB-ACP-03` | 会话生命周期管理 | `src/acp-registry.mjs` 提供 metadata 注册表（sessionId ↔ cwd/profile/model）；封装 `session/list`/`close`/`delete`；孤儿标记与显式 shutdown；崩溃后 capability-gated `session/resume`/`load`；同会话并发串行化 | ACP-01 | **开发门完成（opt-in）**：持久元数据不含 task/response；重启条目标记 orphaned 并可 resume/load；delete 在关闭进程前发送；并发 prompt 串行，删除后的排队请求被阻断；真实强杀 bridge 后的进程表验收留给 ACP-06；server 仍未生产接线 |
| `TOOL-RXB-ACP-04` | 安全回归 | `src/acp-security.mjs` 提供 opt-in 会话作用域、workspace/配置钉死、`WRITE_POLICY` 写路径预检与敏感历史脱敏；服务端 Git diff/rollback 仍是实际写入权威 | ACP-01 | 6 项新增安全回归通过：越界写预检拒绝、跨任务历史隔离、cwd/profile/model 漂移拒绝、`.env`/凭据/私钥脱敏；server 仍未生产接线 |
| `TOOL-RXB-ACP-05` | 共存与切换 | `bridge.config.json` 增 `transport: "per-call" \| "acp"`（默认 per-call）；两条路径共享限额/审计/日志；ACP 初始化失败或超时自动降级 | ACP-02/03/04 | 默认行为与现状回归一致；切到 acp 后既有测试全绿或明确标注不适用项；降级路径有测试 |
| `TOOL-RXB-ACP-06` | 验收与演练 | 崩溃 resume 演练、compact 正确性对照、零泄漏检查（进程/会话/句柄）、并发与取消演练；结果登记回本文档 | ACP-05 | 见 §7 汇总 |

---

## 6. 执行序与依赖

```
TOOL-RXB-ACP-01 ─┬─> TOOL-RXB-ACP-02 ─┬─> TOOL-RXB-ACP-05 ─> TOOL-RXB-ACP-06
                 └─> TOOL-RXB-ACP-03 ─┤
                 └─> TOOL-RXB-ACP-04 ─┘
（ACP-03 与 ACP-04 已完成；ACP-02 与 ACP-03/04 亦可并行）
```

与 `TOOL-RXB-R3-EXT-01`（并行 worker 池）的关系：并行池是"多 worker 同时跑"，ACP 是"单 worker 长会话"，两者正交。R3 已完成；ACP-01 交付客户端和只读协议 fixture，ACP-02 交付协调器，ACP-03 交付 opt-in 会话注册表，ACP-04 交付 opt-in 安全门，生产接线仍由 ACP-05 负责。

---

## 7. 验收门汇总

| 方向 | 验收门 | 证据形式 |
| --- | --- | --- |
| ACP-01 | create→prompt→事件流→关闭全链路；旧 CLI 能力探测；无残留进程（per-call 降级由 ACP-05 接线） | fixture 回归 + 真实 `initialize/session/new/session/close` |
| ACP-02 | 0.75/128MB 不越线；摘要失败回退且历史不变；rotate 有效 | `AcpSessionCoordinator` 决策记录 + 历史字节断言 + 五项回归 |
| ACP-03 | 孤儿状态可识别；可 resume；并发串行；shutdown 有收口路径 | `AcpSessionRegistry` 8 项回归 + 持久注册表/恢复顺序断言；强杀进程表演练留 ACP-06 |
| ACP-04 | 越界写被拒并回滚；跨任务历史不混；敏感内容不入历史 | `AcpSecurityPolicy` 负向测试 + 协调器历史抽样；真实生产回滚仍由 ACP-05/06 演练 |
| ACP-05 | 默认路径回归一致；切换可用；降级有测试 | 回归输出 + 开关测试 |
| ACP-06 | resume/compact/零泄漏/并发取消四类演练通过 | 演练记录（登记回本文档） |

---

## 8. 风险与边界

1. **历史泄漏**：持久会话可能把上一任务的敏感内容带进下一任务——ACP-04 已用"按调用隔离 + 脱敏"关闭本地历史风险，生产接线仍需 ACP-05 复核。
2. **摘要失真**：compact 可能丢关键约束；契约是"摘要失败/可疑即回退 `per_call`"，不得"尽力压缩"。
3. **进程/会话泄漏**：长驻 CLI 是新的资源面；bridge 崩溃、Codex 退出、会话 idle 都要有回收路径。
4. **CLI 版本耦合**：`loadSession`/`resume` 等能力随版本变化——必须**能力探测 + 降级**，不得假设。
5. **启用顺序**：`ACP-05` 未完成前，ACP 不得作为默认 transport；`transport` 开关默认值必须保持 `per-call`。
6. **不做的事**：不把 128MB 上限变成配置项；不用 ACP 替代 MCP 工具契约；不在未验证 compact 的情况下开启长会话。

---

## 9. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-13 | 首版：登记 ACP 侧实测能力（`loadSession`、`session/{list,resume,close,delete}`）与 bridge 侧 design-only 现状；给出三条约束、6 张票（`TOOL-RXB-ACP-01`～`06`）、执行序、验收门与五条边界；明确与任务级续跑（checkpoint）和并行 worker 池（`R3-EXT-01`）的分工 |
| 2026-09-13 | `TOOL-RXB-ACP-01` 完成：新增 `src/acp-client.mjs`，实现 newline JSON-RPC `initialize`、能力探测、`session/new`/`load`/`resume`、`session/prompt` 更新汇总、默认拒绝权限、超时 `session/cancel` 与进程清理；离线回归 2 项通过，真实 Reasonix 完成不发 prompt 的 `initialize → session/new → session/close`。server 仍默认 stateless，未启用 ACP |
| 2026-09-13 | `TOOL-RXB-ACP-02` 完成：新增 `src/acp-session.mjs` 与 `session/delete` 客户端封装；有界确定性摘要接入四分支决策，compact/rotate 采用新会话成功后再关闭/删除旧会话的事务顺序，替换失败和摘要失败显式回退 stateless per-call；协调器回归 6 项通过，bridge 全量回归 67 项通过。server 仍默认 stateless，ACP-03/04/05 生产接线与安全回归未开始 |
| 2026-09-13 | 真实 Reasonix 子智能体受控验证：只读任务跨 README、server、config、checkpoint、ACP client、测试文件完成架构与风险审计，桥接仓库工作树无改动；写角色任务在一次性临时 Git 仓库读取 4 个文件并仅新增 `docs/architecture.md`（49 行），返回 `qlh.reasonix.changes.v1` 与 `rollback_id`，同一 bridge 进程调用回滚后仓库恢复干净，`src/app.txt` 与 `tests/README.md` 哈希不变。该证据验证当前 per-call 读/写角色链路，不等同于 ACP 持久会话安全验收 |
| 2026-09-13 | `TOOL-RXB-ACP-03` 开发门完成：新增 `src/acp-registry.mjs`，提供 metadata-only 持久注册表、孤儿识别、resume/load 能力回退、按会话串行 prompt、close/delete/shutdown 和进程生命周期钩子；测试夹具优先使用项目树内 `build/bridge-test/`，`tmpdir()` 仅作兜底。注册表回归 8 项通过，全量 bridge 回归 `75 passed / 0 failed`；真实强杀 bridge 后的进程表无孤儿演练与生产接线留 ACP-06/05，server 仍默认 stateless |
| 2026-09-13 | `TOOL-RXB-ACP-04` 开发门完成：新增 `src/acp-security.mjs`，以 opaque `owner/taskId` 绑定会话，钉住 cwd/profile/model，复用 `resolveWritePolicy` 的 fail-closed 白名单做 implement 预检，并对 `.env`/凭据/私钥及协调器响应做有界脱敏；注册表只持久化作用域元数据，并可在每个串行 prompt lane 内重检安全策略，协调器可选传输前脱敏。新增 6 项 ACP-04 测试，全量 bridge 回归 `81 passed / 0 failed`；server 仍默认 stateless，ACP-05/06 负责生产切换和强杀演练 |
