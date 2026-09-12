# reasonix-codex-bridge 主仓接线清单（2026-09-12）

> 用途：把 Codex、Reasonix CLI、profile 与 bridge workspace 的接线登记成可照抄步骤。
> 本清单不记录 API key、token、完整 endpoint 或实际 provider 凭据；模型 ref 由目标机 `reasonix doctor` 选择。

## 当前机确认

| 项目 | 值/结论 |
| --- | --- |
| 主仓 workspace root | `C:\Users\surface\Documents\LEDS_BJTU` |
| bridge 子项目 | `C:\Users\surface\Documents\LEDS_BJTU\tools\reasonix-codex-bridge` |
| Node | `C:\Program Files\nodejs\node.exe`（本机 26.x；项目要求 >=20） |
| Reasonix CLI 探测 | `C:\Users\surface\AppData\Local\Programs\Reasonix\reasonix-cli.exe`；版本目录另有 `v1.38.6`、`v1.38.7` |
| Codex 配置 | `%USERPROFILE%\.codex\config.toml`（本机文件存在） |
| Reasonix profile | `%APPDATA%\reasonix\skills\deepseek-worker\SKILL.md`（本机文件存在） |
| bridge 本机配置 | `tools/reasonix-codex-bridge/bridge.config.json`（机器文件，已被 `.gitignore` 忽略） |
| workspace 传递 | `REASONIX_ROOT` 指向目标 workspace；bridge 只允许该根及显式 `REASONIX_ADD_DIRS` |
| G3 状态 | 本机 `wsl.exe --list --quiet` 返回空列表，Docker/QEMU/VirtualBox 不可用；未伪造 POSIX 证据，G3 保留等待 |

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

`REASONIX_EXE` 也可以省略，让 bridge 按 README 的标准路径顺序探测。`REASONIX_MODEL_REF` 不应复制另一台机器的值；目标机必须重新运行 `configure list`。默认 read profile 的工具集合由 `configure profile --role read --sync --write` 固定为 `read_file, grep, glob, ls, code_index, git_log, git_diff`；write profile 只额外增加 `edit_file, write_file` 且不带 `read-only`。`verify --role read/write` 会列出对应实际集合并拒绝漂移。主 agent 负责指挥、diff/测试审查和越界检查，子 agent 只在 W1/W2 策略已显式开启时执行写入。

## 接线后验收

1. `configure verify` 输出 CLI、model ref、profile model/read-only 和实际 `allowed-tools`，且无 `FAIL`。
2. Codex 重启后，MCP `tools/list` 出现 `reasonix_run`、`reasonix_rollback` 与 `reasonix_status`；`mode=implement` 只有在写入策略显式启用时才允许。
3. `reasonix_status` 的 `workspaceRoot`、`modelRefSource`、能力摘要和限额与目标机配置一致。
4. `npm run check`、`npm test`、`npm run check:links` 全绿；这些检查不下载模型、不联网调用 provider。

## 审计复验（2026-09-13）

- AUD-03：`reasonix_rollback` 与 implement 共用串行队列，同一工作区并发回归通过。
- AUD-04：Windows `.cmd/.bat` 通过显式 `cmd.exe`、`shell:false` 启动；含 `&`、`|`、`%`、`!` 等元字符的参数在启动前拒绝。
- AUD-05：回滚变更集登记 `hash_status`，区分 `readable`、`missing`、`unreadable`；不可读/类型变化默认拒绝，恢复后复核 Git 状态和 SHA-256。
- 本机复验为 `npm test` 47/47、`npm run check`、`npm run check:links` 全绿；`allowWrite` 默认仍为 `false`，没有启用 Reasonix 写入 profile。

## 变更边界

- 本清单只登记接线和验收，不执行全局 Codex/profile 写入。
- 生产 bridge 仍是 stateless、只读；G3 的 POSIX/WSL 运行证据必须在具备实际 Linux 环境后补齐。
