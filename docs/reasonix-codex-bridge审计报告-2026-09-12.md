# Reasonix-Codex Bridge 审计报告

日期：2026-09-12  
范围：`tools/reasonix-codex-bridge` 的源码、离线测试、配置/说明文档及本机桥接调用。  
审计方式：主 agent 静态复核、最小临时仓库复现、Node 测试覆盖率，以及通过桥接工具自身发起只读 `review` 子任务。

## 结论

初审时默认配置没有 `allowWrite: true`，默认 profile 也是 read profile，因此本机默认暴露面较低；当时发现 2 项可复现的 P1 缺陷，均涉及越权写入或越权修改保留。2026-09-13 复验确认 BR-001～BR-005 的代码级修复和回归测试已完成，写入策略仍保持关闭，未将受控写 profile 作为默认生产通道开放。随后针对“5 轮工具调用后退出码 1”补查，确认是 Reasonix raw `max_steps` 预算耗尽，不是 bridge 120 秒超时；该边界已由 AUD-07 固化。针对 `read_file` continuation cursor malformed 的重复失败已由 AUD-08 增加 prompt 约束、错误分类与 token 脱敏，未改变 Reasonix 上游策略。

未发现 P0；未发现 MCP 输入可直接绕过 workspace 根目录校验的证据。

## 发现项

### BR-001 P1：非 implement 模式不会强制使用只读 profile

位置：`src/server.mjs:474-490`，尤其是 `:488-490`。

`allowWrite` 只在 `mode=implement` 分支被检查；`inspect`、`review`、`plan` 都直接调用同一个 `runWorker`，使用环境变量/配置选择的 `SUBAGENT_NAME`。因此把 `REASONIX_SUBAGENT` 指向带 `edit_file`/`write_file` 的 write profile 后，`mode=review` 仍可执行写操作，且没有前后 Git 快照或白名单回滚。

复现：临时 Git 仓库选用 `deepseek-worker-write`，`allowWrite` 保持默认 false；伪 Reasonix worker 在 `mode=review` 写入 `outside.txt`。桥接器返回 `isError=false`，文件内容变为 `write-capable worker wrote during review`。

影响：调用方以 review/plan/inspect 语义发起请求时，实际可能修改仓库；这是模式授权边界失效。

建议：非 `implement` 模式强制 read-role（或强制解析到 read profile）；`implement` 模式反向要求显式 write-role，并在启动/调用时拒绝 role 与模式不匹配。增加 write profile + 三种只读模式的拒绝测试。

### BR-002 P1：`requireCleanTree=false` 时会漏掉状态不变的越权修改

位置：`src/server.mjs:188-190`、`:407-420`。

`changedEntries` 只比较 Git status 的 `path -> status` 映射。如果调用前某个不在白名单的文件已经是 ` M`，worker 再改写其内容，调用前后状态仍为 ` M`，该路径不会进入 `changed`，也就不会触发白名单检查或回滚。

复现：设置 `allowWrite=true`、`allowedPaths=["allowed.txt"]`、`requireCleanTree=false`；调用前将 `unrelated.txt` 改为 dirty，worker 同时写 `allowed.txt` 和 `unrelated.txt`。桥接器返回成功，change set 只报告 `allowed.txt`，而 `unrelated.txt` 保留了 `worker unauthorized` 内容。

影响：显式关闭 clean-tree 保护后，worker 可以修改既有 dirty 的白名单外文件而不被发现，违反文档中“out-of-scope paths are zero”的写入契约。

建议：最稳妥的做法是移除该 opt-out 并始终要求 clean tree；若必须保留，则在调用前后对所有既有工作区文件保存并比较哈希/类型，同时单独处理新增、删除、重命名和符号链接，不能只依赖 status 字符串。

### BR-003 P2：显式 rollback 绕过串行队列

位置：`src/server.mjs:445-465`。

`reasonix_run` 经过 `enqueue`，但 `reasonix_rollback` 在 `callTool` 中直接调用 `explicitRollback`。当旧 rollback record 与另一个 implement 调用同时作用于同一工作区时，rollback 的 hash 检查和 `git restore` 可能与 worker 写入交错，造成恢复旧状态或覆盖新状态。当前测试只覆盖“串行 rollback”和“后续手工编辑冲突”，没有并发场景。

建议：rollback 与 implement 共用按 workspace 的串行队列/锁；在执行 restore 前后再次确认文件哈希，并增加并发回归测试。

### BR-004 P2：Windows `.cmd` CLI 路径启用 shell，任务文本进入 shell 参数

位置：`src/config.mjs:86-91`、`src/server.mjs:366-370`。

为支持 `.cmd`，`cliSpawnOptions` 设置 `shell=true`，而 worker task 作为参数传给 CLI。`REASONIX_EXE` 可由环境变量指定，因而不应把它视为永远可信的 `.exe`。在 shell 路径下，包含 cmd 元字符的外部任务文本存在参数解释/注入风险。当前 Node 运行也出现了 `DEP0190` 关于 shell 参数未转义的弃用警告。

建议：优先只接受已解析的 `.exe`；若必须支持 `.cmd`，使用明确的 `cmd.exe` 调用和经过验证的参数编码，并增加带 `&`, `|`, `%` 等字符的 Windows 回归测试。

### BR-005 P3：哈希读取失败与“文件不存在”使用同一个 null 哨兵

位置：`src/server.mjs:223-232`、`:303-309`。

`hashFile` 在权限/读取错误时返回 `null`，rollback 冲突检查把“当前不可读”和“文件不存在”视为同一种状态。对新增文件而言，这可能让不可读文件通过冲突检查后进入删除路径。未在本机构造 ACL 失败复现，属于需要防御性收紧的边界。

建议：区分 `missing`、`unreadable` 和实际 SHA-256；任何 unreadable 状态默认拒绝 rollback。

### BR-006 P2：Reasonix raw `max_steps` 与工具调用轮次口径不一致

位置：`src/server.mjs` 的 `reasonix_run` 步数参数与 worker 非零退出处理。

桥接层原先直接把调用方的 `max_steps` 传给 Reasonix，但 Reasonix CLI 将其作为内部 agent step 预算，而不是工具调用轮次。对本机 Reasonix `v1.38.7` 的直接复现显示：`--max-steps 10` 在 5 轮工具调用后以退出码 1 暂停；同类 8 次工具调用任务在 `--max-steps 20` 下成功。两次均在 120 秒前结束，因此不是 bridge timeout。

影响：调用方按“工具轮次”估算任务大小时会过早触发退出码 1，日志也只能看到笼统的 `worker_exit`，难以区分步数耗尽与真正的进程失败或超时。

修复：子项目 commit `6d093c1` 新增 `tool_rounds` 参数，按当前 CLI 每轮折算 2 个 raw steps；`reasonix_status.limits` 暴露 `toolRoundsCap`；匹配 Reasonix `paused after ... tool-call rounds (max_steps)` 的 stderr 时记录 `step_limit`、轮次数和“120 秒未到”的诊断。原始 `max_steps` 仍保留以兼容直接 CLI 口径。

### BR-007 P2：continuation cursor 失败未分类且可能回显失效 token

位置：`src/server.mjs` 的 worker 非零退出处理，以及 `prompts/deepseek-worker-prompt.md`。

Reasonix 的 `read_file` continuation cursor malformed/invalid 错误此前只会作为普通 worker stderr 透传，bridge 没有独立失败语义；read-only 与 write prompt 也没有明确要求 cursor 原样回传。若失效 cursor 被再次改写或错误重放，读取会反复失败；若错误文本包含长 token，原样回传还会把该 token 暴露给上层模型。

修复：子项目 commit `491c435`（含前序 cursor 处理）在两个 worker prompt 中固化 cursor 不透明值契约，要求逐字原样传回，失效时从明确路径/范围重新读取，并修正 prompt 同步命令路径。bridge 将匹配错误归类为 `cursor_error`，响应中隐藏原始 cursor 诊断，明确不自动重放任务；脱敏日志保留 `cursorError=true`，implement 路径同步保留该分类。该修复不尝试修改 Reasonix 上游工具策略。

## 子 agent 桥接审查证据

主 agent 启动了桥接服务器并调用 `reasonix_run(mode=review)`，服务器日志确认使用了当前 bridge、`role=read`、本机 Reasonix CLI 和配置模型。两次只读审查均未产出可采纳的报告：

1. 全量请求曾在工具调用轮次耗尽后返回 `paused`/exit code 1；stderr 同时报告 Reasonix 配置迁移临时文件 `Access is denied`。后续直接 CLI 复现将“5 轮暂停”与 120 秒计时拆开，确认前者是 raw `max_steps` 耗尽。
2. 缩小到 `src/server.mjs` 与测试文件后，worker 又因 `read_file` continuation cursor malformed 退出；该现象现已由 AUD-08 分类和脱敏，真实上游工具重试行为仍需单独验证。

因此本报告没有把子 agent 的未完成输出当作结论；缺陷均由主 agent 源码证据和独立复现确认。调用过程未修改桥接仓库，审查后工作树保持 clean。

## 测试质量审计

已执行：

| 检查 | 结果 |
|---|---|
| `npm test` | 初审 40 passed；R4 收口后复验 59 passed, 0 failed |
| `npm run check` | 通过，所有模块语法检查通过 |
| `npm run check:links` | README 本地链接通过 |
| `node --test --experimental-test-coverage` | 行 91.92%，分支 56.44%，函数 90.32% |
| `git diff --check` | 通过 |

现有测试覆盖较好的部分包括：配置解析与 doctor cache、Codex TOML 合并、profile 漂移、MCP 工具/状态、任务和输出限制、队列容量与脱敏日志、默认禁写、clean-tree、白名单、手工修改后的 rollback 拒绝，以及 ACP 原型的压缩/轮换。

初审缺口与剩余验证：

- 初审时没有验证 write profile 在 `inspect`/`review`/`plan` 下必然不能写（BR-001）；已由 AUD-01 回归关闭。
- 初审时没有 `requireCleanTree=false` + 调用前已有 dirty 白名单外文件的测试（BR-002）；已由 AUD-02 回归关闭。
- 初审时没有 rollback 与 implement 并发、同一工作区锁竞争的测试（BR-003）；已由 AUD-03 回归关闭。
- 初审时没有 Windows `.cmd` shell 参数元字符测试（BR-004）；已由 AUD-04 回归关闭。
- rename/copy 回滚目标的 staged index 缺口已由 AUD-06 关闭；symlink、权限失败、不可读文件和二进制新增/删除仍未在本机完整覆盖。
- 测试使用伪 CLI/worker，未覆盖真实 Reasonix 版本、全局 profile 解析、provider 失败、模型超时和网络故障；本机真实子 agent 调用还受到配置迁移权限和工具 cursor 错误影响。
- 已直接覆盖真实 Reasonix `v1.38.7` 的步数口径：`max_steps=10` 稳定表现为 5 轮后暂停，`max_steps=20` 可完成 8 次工具调用任务；这证明了步数耗尽与 120 秒超时是不同故障形态。真实 worker 的 `read_file` continuation cursor malformed 仍是独立的工具集成问题。
- 已用 fixture 覆盖 malformed/invalid continuation cursor：bridge 归类为 `cursor_error`、不自动重放、响应不回显失效 token，read/write prompt 均包含原样回传和失效后重新读取约束。真实 Reasonix 上游工具对该 prompt 的长期成功率仍未测量。
- CI 文档声明 Node 20，当前本机测试运行时为较新的 Node 版本；至少应在 Node 20 和当前支持的 Windows/WSL 环境各跑一次。

## 整改顺序

1. 修复 BR-001：把 profile role 纳入模式授权，默认只读模式不可调用 write profile。**已完成（2026-09-12）**。
2. 修复 BR-002：默认强制 clean tree，或改为全工作区快照比较；为该行为补回归测试。**已完成（2026-09-12）**。
3. 将 rollback 纳入同一串行队列，并补并发测试。**已完成（2026-09-13）**。
4. 收紧 Windows CLI 启动方式并补 shell 参数测试。**已完成（2026-09-13）**。
5. 增加失败注入、权限/链接/二进制和真实 CLI 的分层集成测试；解决本机 Reasonix 配置目录的权限迁移问题后，再重新运行桥接子 agent 审查。**部分完成**：本轮补齐并发、类型变化/缺失、Windows 参数、restore 后哈希和 staged rename 回归；ACL 权限失败、symlink、二进制边界、POSIX/WSL、真实 provider/model 质量证据仍待对应环境。

审计判定：BR-001～BR-007 的代码级风险已由复验关闭，默认只读 inspect/review 可继续使用；写 profile 和 `allowWrite=true` 仍保持受控，直至 G3 跨 POSIX 环境和真实集成证据补齐后再评估生产开放。

## 修复跟踪

| 修复票 | 对应发现 | 状态 | 证据 |
|---|---|---|---|
| `TOOL-RXB-AUD-01` | BR-001 非 implement 模式未强制只读 profile | **已完成（2026-09-12）** | 子项目 commit `f6bacd7`；回归测试 `43 passed` |
| `TOOL-RXB-AUD-02` | BR-002 dirty-tree 状态不变修改漏报 | **已完成（2026-09-12）** | 子项目 commit `a361634`；`requireCleanTree=false` fail-closed 回归通过 |
| `TOOL-RXB-AUD-03` | BR-003 rollback 绕过串行队列 | **已完成（2026-09-13）** | 回滚接入 implement 共用队列；同一工作区并发回归通过 |
| `TOOL-RXB-AUD-04` | BR-004 Windows `.cmd` shell 参数 | **已完成（2026-09-13）** | `shell:false` + 显式 `cmd.exe`；元字符拒绝与 `.cmd` 启动回归通过 |
| `TOOL-RXB-AUD-05` | BR-005 hash 读取失败哨兵混用 | **已完成（2026-09-13）** | `readable/missing/unreadable` 三态；类型变化、缺失和 restore 后 SHA-256 回归通过 |
| `TOOL-RXB-AUD-06` | rename/copy 目标回滚的 index 状态未覆盖 | **已完成（2026-09-13）** | 子项目 commit `8e7693c`；rename 目标按新增路径清理并撤销 staged index，回归通过 |
| `TOOL-RXB-AUD-07` | Reasonix raw `max_steps` 与工具调用轮次口径不一致 | **已完成（2026-09-13）** | 子项目 commit `6d093c1`；`tool_rounds` 映射、`step_limit` 分类和 51 项回归通过 |
| `TOOL-RXB-AUD-08` | Reasonix continuation cursor 失败未分类且可能回显失效 token | **已完成（2026-09-13）** | 子项目 commit `491c435`；prompt 原样回传约束、同步路径修正、`cursor_error` 分类/脱敏、禁止重放和 53 项回归通过 |
| `TOOL-RXB-R2-EXT-01` | 审计后长任务预算仍偏窄 | **已完成（2026-09-13）** | 子项目 commit `26b4113`；有限上限扩展至 256 raw steps/128 工具轮次/1800 秒，54 项回归通过 |
| `TOOL-RXB-E2-EXT-01` | 长任务失败后无法显式续跑 | **已完成（2026-09-13）** | 子项目 commit `a596921`；任务级 checkpoint、配置/工作区漂移拒绝、原子 claim、一次性 `reasonix_resume` 和 57 项回归通过 |
| `TOOL-RXB-R3-EXT-01` | 并行 worker 无隔离、取消和回收观测 | **已完成（2026-09-13）** | 子项目 commit `782d6b0`；显式只读并行槽位、job 状态、`reasonix_cancel`、exclusive 写入 lane 和 59 项回归通过 |
| `TOOL-RXB-R4` | 输出截断跨平台结果不确定 | **已完成（2026-09-13）** | 子项目 commit `fa06fd2`；输出超限改为有界缓冲并返回成功+`truncated=true`，Windows `npm test` 59/59、check、links 全通过 |

本次初审关闭 BR-001、BR-002；2026-09-13 复验关闭 BR-003、BR-004、BR-005、BR-006、BR-007，G3 跨 POSIX 实跑与 R4 输出截断竞态也已收口。写 profile 和 `allowWrite=true` 仍不作为默认生产通道开放；当前 Windows shell 未安装 Linux Node，不能在本轮独立重跑 WSL 证据。

## 复验记录（2026-09-13）

- 主节点 Codex 只读审计复核确认修复方向：AUD-03 的回滚请求进入与 implement 相同的串行队列；AUD-04 的 `.cmd/.bat` 调用不再启用 `shell=true`，危险元字符在启动前 fail-closed；AUD-05 的回滚目标区分 `missing`、`unreadable` 和 SHA-256，恢复后再次检查状态与哈希。
- 本机 Reasonix 只读复核本轮以 worker exit code 1 结束，没有产出可采纳报告；未把该失败当作通过证据，也未开放 Reasonix 写入配置。
- `npm test`：R4 收口后 59 passed, 0 failed；`npm run check`：通过；`npm run check:links`：通过；`git diff --check`：通过。
- 新增回归覆盖：同一 MCP 进程中 implement 与 rollback 并发、Windows `.cmd` 参数元字符、非规则文件/缺失目标的 rollback 拒绝、restore 后 SHA-256 复核，以及 staged rename 目标的 index 清理。
- 新增步数回归覆盖：Reasonix `max_steps` 暂停识别为 `step_limit`、`timeout_seconds` 未到的诊断，以及 `tool_rounds` 到 raw `--max-steps` 的映射。
- 新增 cursor 回归覆盖：prompt 原样回传约束、malformed/invalid 错误 `cursor_error` 分类、失效 token 不回显、禁止自动重放，以及 implement 日志元数据保留。
- 审计后增强 `TOOL-RXB-R2-EXT-01` 将有限 runtime budget 扩展至 256 raw steps/128 工具轮次/1800 秒；`TOOL-RXB-E2-EXT-01` 已接入任务级 checkpoint/续跑，`TOOL-RXB-R3-EXT-01` 已接入显式只读并行与取消回收，`TOOL-RXB-R4` 已统一输出超限结果，ACP-01 已提供未接入 server 的客户端层；写入/续跑/回滚仍保持独占，原生 ACP 会话恢复与 transport 切换仍未接入。
- 未伪造 ACL 权限失败或真实 provider/model 质量证据；WSL 实跑证据已由并行主节点记录，但当前 Windows shell 无 Linux Node，无法本轮独立重跑。
