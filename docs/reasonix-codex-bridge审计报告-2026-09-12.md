# Reasonix-Codex Bridge 审计报告

日期：2026-09-12  
范围：`tools/reasonix-codex-bridge` 的源码、离线测试、配置/说明文档及本机桥接调用。  
审计方式：主 agent 静态复核、最小临时仓库复现、Node 测试覆盖率，以及通过桥接工具自身发起只读 `review` 子任务。

## 结论

当前默认配置没有 `allowWrite: true`，默认 profile 也是 read profile，因此本机默认暴露面较低。但实现路径目前不应作为“写入安全”已收口：发现 2 项可复现的 P1 缺陷，均涉及越权写入或越权修改保留。修复并补充回归测试前，建议继续保持写入关闭。

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

## 子 agent 桥接审查证据

主 agent 启动了桥接服务器并调用 `reasonix_run(mode=review)`，服务器日志确认使用了当前 bridge、`role=read`、本机 Reasonix CLI 和配置模型。两次只读审查均未产出可采纳的报告：

1. 全量请求在 8 轮工具调用后返回 `paused`/exit code 1；stderr 同时报告 Reasonix 配置迁移临时文件 `Access is denied`。
2. 缩小到 `src/server.mjs` 与测试文件后，worker 又因 `read_file` continuation cursor malformed 退出。

因此本报告没有把子 agent 的未完成输出当作结论；缺陷均由主 agent 源码证据和独立复现确认。调用过程未修改桥接仓库，审查后工作树保持 clean。

## 测试质量审计

已执行：

| 检查 | 结果 |
|---|---|
| `npm test` | 40 passed, 0 failed |
| `npm run check` | 通过，所有模块语法检查通过 |
| `npm run check:links` | README 本地链接通过 |
| `node --test --experimental-test-coverage` | 行 90.80%，分支 53.73%，函数 90.34% |
| `git diff --check` | 通过 |

现有测试覆盖较好的部分包括：配置解析与 doctor cache、Codex TOML 合并、profile 漂移、MCP 工具/状态、任务和输出限制、队列容量与脱敏日志、默认禁写、clean-tree、白名单、手工修改后的 rollback 拒绝，以及 ACP 原型的压缩/轮换。

关键缺口：

- 没有验证 write profile 在 `inspect`/`review`/`plan` 下必然不能写（BR-001）。
- 没有 `requireCleanTree=false` + 调用前已有 dirty 白名单外文件的测试（BR-002）。
- 没有 rollback 与 implement 并发、同一工作区锁竞争的测试（BR-003）。
- 没有 Windows `.cmd` shell 参数元字符测试（BR-004）。
- 没有 symlink、rename/copy、权限失败、不可读文件、二进制新增/删除等 rollback 边界测试。
- 测试使用伪 CLI/worker，未覆盖真实 Reasonix 版本、全局 profile 解析、provider 失败、模型超时和网络故障；本机真实子 agent 调用还受到配置迁移权限和工具 cursor 错误影响。
- CI 文档声明 Node 20，当前本机测试运行时为较新的 Node 版本；至少应在 Node 20 和当前支持的 Windows/WSL 环境各跑一次。

## 整改顺序

1. 修复 BR-001：把 profile role 纳入模式授权，默认只读模式不可调用 write profile。
2. 修复 BR-002：默认强制 clean tree，或改为全工作区快照比较；为该行为补回归测试。
3. 将 rollback 纳入同一串行队列，并补并发测试。
4. 收紧 Windows CLI 启动方式并补 shell 参数测试。
5. 增加失败注入、权限/链接/二进制和真实 CLI 的分层集成测试；解决本机 Reasonix 配置目录的权限迁移问题后，再重新运行桥接子 agent 审查。

审计判定：功能测试通过不等于写入安全收口。当前可继续用于默认只读 inspect/review，但写 profile 和 `allowWrite=true` 应保持受控，不建议在修复前作为生产写入通道启用。

## 修复跟踪

| 修复票 | 对应发现 | 状态 | 证据 |
|---|---|---|---|
| `TOOL-RXB-AUD-01` | BR-001 非 implement 模式未强制只读 profile | **已完成（2026-09-12）** | 子项目 commit `f6bacd7`；回归测试 `43 passed` |
| `TOOL-RXB-AUD-02` | BR-002 dirty-tree 状态不变修改漏报 | **已完成（2026-09-12）** | 子项目 commit `a361634`；`requireCleanTree=false` fail-closed 回归通过 |
| `TOOL-RXB-AUD-03` | BR-003 rollback 绕过串行队列 | 待处理 | 设计风险，待并发回归 |
| `TOOL-RXB-AUD-04` | BR-004 Windows `.cmd` shell 参数 | 待处理 | 需 Windows 参数回归 |
| `TOOL-RXB-AUD-05` | BR-005 hash 读取失败哨兵混用 | 待处理 | 需权限失败边界测试 |

本次已关闭 BR-001、BR-002；BR-003 及后续发现仍按优先级执行。
