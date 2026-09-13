# reasonix-codex-bridge 全面审计与多次实测报告（2026-09-13）

## 结论

桥接器已经可以作为“受控、低价、文件型开发助手”使用：只读勘察、review/plan、白名单写入、结构化变更证据、显式回滚、失败 checkpoint/续跑、取消和只读并行均有实现与回归证据。它还不是 Codex 原生子智能体的等价替代：默认 worker 没有 shell、测试执行、网络、动态子智能体派生、子智能体间消息和流式进度；ACP 跨进程恢复也未通过真实 provider 持久化门。

因此当前适用边界是：主 agent 拆分任务并审查结果，bridge worker 完成局部读写工作。需要执行测试/构建/安装、访问网络或跨多个自主阶段迭代时，仍必须由主 agent 或后续受控工具完成。

## 审计范围

- MCP stdio 生命周期、工具 schema、错误和状态返回；
- Reasonix CLI/模型/profile 解析、版本门、预算和上下文预检；
- per-call worker、任务级 checkpoint/resume、取消、队列与只读并行；
- implement 写白名单、clean-tree、变更证据、哈希校验和 rollback；
- ACP client、compact/rotate、进程内 transport、registry、强杀清理；
- 日志/状态脱敏、测试夹具路径、离线可复现性和真实 provider 边界。

主要代码入口：`tools/reasonix-codex-bridge/src/server.mjs:62-68`、`:538-736`、`:741-972`，以及 `src/acp-client.mjs`、`src/acp-session.mjs`、`src/acp-registry.mjs`、`src/acp-transport.mjs`。

## 能力矩阵

| 能力 | 结论 | 证据与边界 |
| --- | --- | --- |
| inspect / review | 通过 | `reasonix_run` 四模式 schema 与只读 profile；真实 inspect 成功 |
| plan | 通过但需复核 | 真实 plan 返回 `qlh.reasonix.plan.v1`；模型可能给出未验证的静态推断 |
| 受控写入 | 通过 | write profile + `allowWrite` + `allowedPaths` + clean Git tree；只返回变更证据 |
| 回滚 | 通过 | 一次性 `rollback_id`，回滚前做文件 hash/状态冲突检查 |
| 任务级续跑 | 通过 | timeout/worker_exit/step_limit/cursor_error 生成 checkpoint，`reasonix_resume` 显式消费 |
| 会话级续跑 | 部分通过 | ACP-06 离线 registry orphan/resume 通过；真实 Reasonix 空会话跨进程 resume 返回 `unknown session` |
| 取消 | 通过 | 队列任务可取消；运行中 worker 终止进程树且不生成 checkpoint |
| 并行 | 受限通过 | `parallel=true` 仅 inspect/review/plan，只读槽位并发；implement/resume/rollback 独占 |
| compact / rotate | 通过（适配器层） | 固定 128 MiB 上限、75% 触发，事务式替换；不是 Reasonix 原生历史替换声明 |
| 观测 | 通过 | `reasonix_status`、可选 JSONL 脱敏日志、job 状态；无流式 partial output |
| 工具生态 | 有意受限 | read profile 7 项工具，write profile 仅增加 `edit_file`/`write_file`；无 shell、测试、网络 |
| 自主迭代 | 部分通过 | 单次 worker 可在预算内多轮；跨调用续跑需主 agent 显式编排，无自动 plan→implement→test loop |

## 实测记录

### 1. MCP 控制面

使用本机 Reasonix CLI v1.38.7 启动真实 `src/server.mjs`，发送 `initialize`、`tools/list`、`reasonix_status`：

- MCP 握手正常，暴露 5 个工具：`reasonix_run`、`reasonix_resume`、`reasonix_cancel`、`reasonix_rollback`、`reasonix_status`；
- 版本门 `1.38.6` 通过；当前 profile 为 `deepseek-worker` / read；模型能力窗口报告为 1,000,000 tokens；
- 默认 transport 为 `per-call`，写策略状态可见但只读角色不会被当作写角色；
- 状态显示本机有 1 个历史 `cursor_error` checkpoint。本轮未消费或修改它。

### 2. 真实只读 inspect

任务：读取 `tools/reasonix-codex-bridge/package.json`，只报告 name/version，不修改文件。

- 退出码：0；worker elapsed 约 6.7s；输出带 `package.json:2`、`:4` 行号依据；
- bridge 进程正常收口，无文件变化。

### 3. 真实 plan

任务：审阅 `src/acp-session.mjs`，输出最多三个风险的 `qlh.reasonix.plan.v1` JSON，不写文件。

- MCP 调用成功，返回机器可消费的 plan JSON 和文件/行号依据；
- 输出同时标注了“未验证”推断，说明模型产出不能直接视为审计结论；
- 当前低价模型该次调用约 60s，性能不作为功能通过条件。

### 4. 真实受控写入与回滚

在 `build/bridge-test/audit-write-fixture/` 建立干净 Git 仓库，配置 `allowedPaths=["docs"]`，使用真实 write profile 执行：仅创建 `docs/audit-note.md`，内容为一行固定文本。

- implement 成功，返回 `qlh.reasonix.changes.v1`；仅有 `docs/audit-note.md` 一项 added，含 additions、SHA-256、`hash_status=readable` 和 `rollback_id`；
- 同一 bridge 进程立即调用 `reasonix_rollback` 成功；文件删除，Git 树恢复干净；
- 没有 commit、网络或白名单外路径变化。

### 5. 真实 checkpoint / resume

在干净 bridge 子仓故意以 `tool_rounds=1` 运行需要多次读取的 inspect 任务，触发 Reasonix 的 cursor 失败边界。

- 首次结果为 `cursor_error`，输出只给脱敏诊断并返回 `checkpoint_id`；
- 显式调用 `reasonix_resume` 后成功完成同一任务，未隐式重放；
- checkpoint 文件在指定隔离目录生成并被消费，原始任务正文没有出现在 worker 输出中。

### 6. ACP-06 离线和真实控制面

```text
npm run check
npm run acceptance:acp
node --test test/bridge.test.mjs
```

结果：语法/链接检查通过；ACP-06 离线报告 `status=passed`，覆盖真实子进程强杀、orphan/resume、metadata-only、compact/rotate、并发串行、取消、child cleanup 和工件删除；全量测试 `92 passed / 0 failed`。

```text
npm run acceptance:acp:real
```

结果：Reasonix v1.38.7 完成 `initialize`、`session/new` 和强杀，但空会话跨进程 `session/resume` 返回 `unknown session`；命令按契约返回 `status=blocked`、`reason=empty_session_not_persisted`、退出码 2。该探测没有发送模型 prompt，因此不能据此宣称真实 provider 会话恢复通过。

## 审计发现与优先级

### P1：真实 ACP 跨进程恢复未闭环

`AcpSessionRegistry` 已有 metadata-only、orphan、resume/load、shutdown 和并发串行，但尚未接入 production server。真实 provider 对未产生 prompt 的会话不持久化，导致强杀后 resume 不成立。生产默认保持 `per-call` 是正确的 fail-closed 行为。

### P1：写入后不能由 worker 自己执行测试

write profile 没有 shell、测试或构建工具，worker 只能编辑文件；实现结果必须由主 agent 审查并运行测试。这提高安全性，但与原生 coding subagent 的“编辑→运行测试→修复”闭环有明显差距。

### P1：没有跨调用自主编排

checkpoint 是一次性、显式 resume；parallel 是显式只读槽位；bridge 不会自行生成子任务、安排 plan→implement→test 阶段或在失败后自动改变策略。这要求 Codex 主 agent 保留编排和最终判断权。

### P2：ACP coordinator 直接使用时仍需调用方串行化

production 的 `AcpTransportManager` 已按会话串行 prompt，并发回归通过；但 `AcpSessionCoordinator` 是公开模块，直接调用者应继续通过 manager 或外层队列使用，不能把它当作任意并发安全对象。

### P2：没有流式进度事件

MCP `tools/call` 当前只返回最终文本；job 状态可轮询，取消请求先返回 acknowledgement。长任务期间没有原生子智能体式的增量进度/中间产物通道。

### P2：checkpoint 任务正文落盘

checkpoint 不保存 stdout/stderr，但会保存原始任务文本、模式、预算和指纹到用户状态目录。涉及敏感任务时应由调用方避免把凭据放进任务，或后续增加本地加密/保留期策略。

## 建议下一步

1. 新增受控 `reasonix_exec`/测试工具：只允许配置的可执行文件和 argv 数组，禁止 shell，固定 cwd 根，继承 timeout/output caps，记录脱敏审计，并默认关闭；这是补齐“写后验证”能力的最高收益改造。
2. 为 ACP 增加 provider-backed 非空会话 fixture，先验证“产生 prompt 后强杀→resume/load→close/delete”，再评估 registry 的生产接线；在此之前不改默认 transport。
3. 在主 agent 层提供显式阶段编排模板（plan→implement→test→review），保持每阶段可审查、可回滚，不在 bridge 内隐式重试。
4. 长任务若需要进度，增加仅含 job id、阶段和计数的通知/轮询契约，不传任务正文或 worker 输出正文。

## 文档质量门

本轮新报告链接已存在。主仓 `python scripts/check_doc_links.py` 仍报告 7 个既有失效引用（Android 侧 llama.cpp 文档 5 个、旧 `local_docs` 引用 2 个）；它们与 bridge 代码无关，本轮没有擅自修复。`scripts/check_readme_l10n.py` 仍报告 6 个既有中英文章节映射缺口。bridge 子项目自身 `npm run check:links` 通过。

## 最终判定

对文档、配置、代码局部修改和受控审查任务：**可作为低价替代投入使用**。对需要 shell/测试/网络、长时间跨进程上下文、自主多 agent 协作的任务：**当前只能作为受限执行器，不能宣称达到 Codex 原生子智能体的大多数能力**。本轮没有把小模型响应质量或性能冒充为桥接器能力结论。
