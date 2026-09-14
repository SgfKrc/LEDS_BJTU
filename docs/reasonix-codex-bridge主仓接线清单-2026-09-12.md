# reasonix-codex-bridge 主仓接线清单（2026-09-12）

> 用途：把 Codex、Reasonix CLI、profile 与 bridge workspace 的接线登记成可照抄步骤。
> 本清单不记录 API key、token、完整 endpoint 或实际 provider 凭据；模型 ref 由目标机 `reasonix doctor` 选择。

## 当前机确认

> 2026-09-14 改写为**本机（`G:\C\PYT\qlh`）实际配置**；此前的 Surface 记录（`C:\Users\surface\…`）已替换，历史值见 git 历史。

| 项目 | 值/结论 |
| --- | --- |
| 主仓 workspace root | `G:\C\PYT\qlh` |
| bridge 子项目 | `G:\C\PYT\qlh\tools\reasonix-codex-bridge`（submodule，`5104f3d`） |
| Node | `v24.16.0`（项目要求 ≥20） |
| Reasonix CLI 探测 | **非标准路径**：`H:\Reasonix\versions\v1.38.7\reasonix-cli.exe`（桌面版目录；标准探测顺序不含 H 盘，**必须显式 `REASONIX_EXE`**；npm 全局 `reasonix` 为 v1.21.3，版本过旧不可用） |
| Codex 配置 | `C:\Users\Koakuma\.codex\config.toml` → `[mcp_servers.reasonix_local]`（2026-09-14 写入，含自动备份 `config.toml.bak-*`） |
| Reasonix profile | `%APPDATA%\reasonix\skills\deepseek-worker\SKILL.md`（读角色）与 `…\deepseek-worker-write\SKILL.md`（写角色，2026-09-14 创建） |
| bridge 本机配置 | `bridge.config.json` 当前实际 `modelRef = opencode-go-2ae…/deepseek-flash`（项目标准 V4.1-Flash API ref）；**写策略已恢复关闭（2026-09-14，按并行线审查建议）**：`allowWrite=false` + 白名单预置最小集 `["src/","harness_workbench/","tests/"]` + `requireCleanTree=true` → 解析 `enabled=false`；剩余 P1 审计问题关闭后再以最小白名单重开） |
| **子智能体权限（当前）** | **读角色默认**：`deepseek-worker`（只读：`read_file, grep, glob, ls, code_index, git_log, git_diff`）；**写角色已创建**：`deepseek-worker-write`（读集 + `edit_file, write_file, web_fetch`，无 `read-only`），仅在 `mode=implement` 显式授权时使用；**写入前要求工作树 clean**（`requireCleanTree=true`，⚠️ **开发阶段临时口径**：agent 场景"一写即脏"，连续写入会被拒，已登记 `TOOL-RXB-W1-EXT-01` 将来必须优化/砍掉），主 agent 负责审查 diff |
| **预算默认（bridge 新版口径）** | **`tool_rounds`：inspect 40 / review·plan·implement 48**（raw `max_steps` 80/96，1 round = 2 steps）；timeout 600s / 900s；硬上限 128 rounds / 256 steps / 1800s。**调用省略参数即用以上默认**——Codex 侧无需传预算（此前 120s 超时是调用方显式传入的保守值） |
| workspace 传递 | `REASONIX_ROOT = G:\C\PYT\qlh`；bridge 只允许该根及显式 `REASONIX_ADD_DIRS` |
| G3 状态 | **已完成（2026-09-13）**：记录 WSL Ubuntu 22.04 的 POSIX CLI 探测、无 CLI 启动拒绝、WSL interop 下 `--version`/doctor/verify 与跨平台测试夹具修复；当前 Windows shell 的 Linux Node 未安装，不能在本轮独立复跑 |

## 标准接线

在目标机执行：

```powershell
git clone <主仓地址>
cd <主仓目录>
git submodule update --init --recursive
cd tools\reasonix-codex-bridge

node src\configure.mjs list
node src\configure.mjs use <provider/model-from-list>
node src\configure.mjs codex --write
node src\configure.mjs profile --role read --create --write
node src\configure.mjs verify

npm run check
npm test
npm run check:links
```

已有 profile 时，将 `profile --create --write` 换成：

```powershell
node src\configure.mjs profile --sync --write
```

需要受控写入时，先单独生成写角色；它不会改写默认只读 profile：

```powershell
node src\configure.mjs profile --role write --create --write
node src\configure.mjs verify --role write
```

`configure codex --write` 会先校验、备份并原子更新 `%USERPROFILE%\.codex\config.toml`；不希望自动写入时先运行不带 `--write` 的预览命令。`configure profile` 的写入同样必须显式带 `--write`。

## Codex 环境契约

生成的 `[mcp_servers.reasonix_local.env]` 至少登记以下四项：

```toml
REASONIX_EXE = "<目标机 reasonix-cli 路径>"
REASONIX_ROOT = "<目标 workspace root>"
REASONIX_SUBAGENT = "deepseek-worker"  # 默认只读；受控 implement 才显式切换为 deepseek-worker-write
REASONIX_MODEL_REF = "<configure list 选定的 provider/model>"
```

`REASONIX_EXE` 也可以省略，让 bridge 按 README 的标准路径顺序探测。`REASONIX_MODEL_REF` 不应复制另一台机器的值；目标机必须重新运行 `configure list`。当前默认 read profile 的工具集合由 `configure profile --role read --sync --write` 固定为 Reasonix v1.38.7 实际识别的 `read_file, grep, glob, ls, code_index`；write profile 只额外增加 `edit_file, write_file` 且不带 `read-only`。`verify --role read/write` 会同时检查 profile 漂移与 doctor 未知工具身份并 fail-closed。主 agent 负责指挥、diff/测试审查和越界检查，子 agent 只在 W1/W2 策略已显式开启时执行写入。

## 接线后验收

1. `configure verify` 输出 CLI、model ref、profile model/read-only 和实际 `allowed-tools`，且无 `FAIL`。
2. Codex 重启后，MCP `tools/list` 出现 `reasonix_run`、`reasonix_resume`、`reasonix_rollback` 与 `reasonix_status`；`mode=implement` 只有在写入策略显式启用时才允许。
3. `reasonix_status` 的 `workspaceRoot`、`modelRefSource`、能力摘要和限额与目标机配置一致。
4. `npm run check`、`npm test`、`npm run check:links` 全绿；这些检查不下载模型、不联网调用 provider。

## 审计复验（2026-09-13）

- AUD-03：`reasonix_rollback` 与 implement 共用串行队列，同一工作区并发回归通过。
- AUD-04：Windows `.cmd/.bat` 通过显式 `cmd.exe`、`shell:false` 启动；含 `&`、`|`、`%`、`!` 等元字符的参数在启动前拒绝。
- AUD-05：回滚变更集登记 `hash_status`，区分 `readable`、`missing`、`unreadable`；不可读/类型变化默认拒绝，恢复后复核 Git 状态和 SHA-256。
- AUD-06：Git rename/copy 目标按新增路径处理，清理前撤销 staged index；staged rename 回归通过。
- AUD-07：Reasonix raw `max_steps` 与工具调用轮次分离；`tool_rounds` 映射和 `step_limit` 失败分类已落地。
- AUD-08：continuation cursor 必须原样传回；malformed/invalid cursor 分类为 `cursor_error`，失效 token 不回显且不自动重放。
- 本机复验为 `npm test` 53/53、`npm run check`、`npm run check:links` 全绿；`allowWrite` 默认仍为 `false`，没有启用 Reasonix 写入 profile。

## 变更边界

- `reasonix_resume` 仅接受失败响应中的 checkpoint id；恢复前会校验 workspace/config 指纹，并消费 checkpoint 后才启动新的显式续跑。
- 本清单只登记接线和验收，不执行全局 Codex/profile 写入。
- 生产 bridge 仍是 stateless，写入默认关闭；G3 的 POSIX/WSL 运行证据已登记，R4 的输出截断竞态已在子项目 `fa06fd2` 修复并由 Windows 回归覆盖。
