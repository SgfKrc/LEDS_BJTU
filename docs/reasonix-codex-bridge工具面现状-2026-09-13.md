# reasonix-codex-bridge 工具面现状与能力归属（2026-09-13）

> 状态：**现状登记，`TOOL-RXB-TOOL-01`、`TOOL-RXB-NET-01`、`TOOL-RXB-NET-02` 与 `TOOL-RXB-LOOP-01` 已完成**；本文记录 Reasonix v1.38.7 的真实工具面、bridge 声明与现实的差异，以及"能力归属"原则（能丢给 Reasonix 的都交给 Reasonix）。NET-02 当前为本机 provider unavailable，接通 provider 后需补真实搜索验收。
>
> 创建日期：2026-09-13
> 证据类型：Reasonix **内置文档**（`docs/TOOL_CONTRACT.md` 等，随 v1.38.7 打包）+ 本机 `reasonix doctor` / `doctor capabilities` 实测输出。

---

## 1. 结论摘要

1. Reasonix 的工具面分**两层**：provider 可见的 **core**（每次任务固定出现）与留在 host registry、经 `use_capability` 调度的**可选工具**。
2. bridge 契约曾错误声明 **`git_log` 与 `git_diff`**；本机 `doctor` 曾连续报出 4 条警告（read/write 两个 profile × 2），即这两个工具**从未真正生效**。`TOOL-RXB-TOOL-01` 已移除它们并同步 profile，当前 doctor 告警为 0。
3. `read_file, grep, glob, ls, code_index, web_fetch, edit_file, write_file` 均为**有效**身份（doctor 清单未对其报警）。
4. **`web_fetch` 是 Reasonix 自带的可选工具**；搜索（`web_search`）按官方文档是 **provider 侧能力**（搜索会另发一次模型请求，查询交给 provider）。因此网络能力**不落在 bridge**——bridge 只做门控与透传。
5. bridge 的 `configure verify` 按**自己的期望集**比对 profile，并消费 Reasonix doctor 的未知身份诊断；当前两个 profile 均已通过 fail-closed 校验。

---

## 2. Reasonix 真实工具面（v1.38.7 内置文档）

来源：`docs/TOOL_CONTRACT.md:158-171`（"Unified Boot Surface (every task)"）。

### 2.1 core（provider 可见，每次任务固定）

```
bash, bash_output, edit_file, kill_shell, read_file, view_image,
wait, write_file, compress (when registered), use_capability
```

### 2.2 可选工具（留在 host registry，经 `use_capability` 调度）

文档原文列举：`glob`、`grep`、`ls`、**`web_fetch`**、MCP、skills、subagents、docs、session history、memory mutation、workflow 等。要点：

- 可选工具**不改变 provider 工具列表**；模型通过 `use_capability` 发现/调用/拒绝它们；
- `doctor` / `doctor capabilities` 检查 `allowed-tools` 时，"inventory combines compile-time tools with host-managed tool identities"（`docs/CAPABILITY_DIAGNOSTICS.md:43-73`）；
- 被检查出的"未知身份"就是本节 §3 的差异来源。

### 2.3 本机 MCP 与相关配置（实测）

| 项 | 实测值 |
| --- | --- |
| MCP servers | `gitcontext`（stdio，auto_start）、`shizi-wiki`（http，当前 failed） |
| 可用的 git 只读能力 | 经 MCP：如 `mcp-tool:gitcontext/git_pickaxe`（capability catalog 可见） |
| 网络出口 | `network.proxy_mode=auto`（env 代理） |
| 权限模式 | `permission.mode=ask` |

---

## 3. bridge 声明 vs Reasonix 现实（差异表）

| 工具 | bridge profile / 契约声明 | Reasonix 现实 | 结论 |
| --- | --- | --- | --- |
| `read_file` `grep` `glob` `ls` `code_index` `web_fetch` | read profile 6 件套 | 有效身份（同步后 doctor 无警告） | ✅ 一致 |
| `edit_file` `write_file` | write profile 追加 | core 工具 | ✅ 一致 |
| **`git_log`** | 历史 profile/README 曾声明 | **"not a known tool identity"**；已从 profile 与常量移除 | ⚠️ **历史错误，已修复** |
| **`git_diff`** | 历史 profile/README 曾声明 | **"not a known tool identity"**；已从 profile 与常量移除 | ⚠️ **历史错误，已修复** |

本机 `doctor` 原始警告（4 条，原文）：

```
skill "deepseek-worker"       allowed-tools reference "git_log"  is not a known tool identity
skill "deepseek-worker"       allowed-tools reference "git_diff" is not a known tool identity
skill "deepseek-worker-write" allowed-tools reference "git_log"  is not a known tool identity
skill "deepseek-worker-write" allowed-tools reference "git_diff" is not a known tool identity
```

**实际生效的工具集**因此是：read = `read_file, grep, glob, ls, code_index, web_fetch`（6）；write = 上述 + `edit_file, write_file`（8）。`web_fetch` 仍由 Reasonix host registry 执行，bridge 不接收 URL 参数，也不实现第二套网络栈。

### 3.1 影响

1. **能力缺口**：worker 没有任何 git 只读能力；"让子智能体看 git log/diff"的预期从未成立。当前由主 agent 通过 host/MCP 或受控 exec 审查。
2. **校验失真（已修复）**：旧版 `configure verify` 只对照 bridge 自己的常量；当前会消费 `reasonix doctor` 的 profile 工具告警并 fail-closed。
3. **文档契约**：README、prompt 与全局 profile 已同步为上述 6/8 集合；`configure verify` 对工具漂移和 doctor 未知身份均 fail-closed。

---

## 4. 能力归属原则（本文件的核心约定）

> **能丢给 Reasonix 的都交给 Reasonix；bridge 只做门控、限额与证据。**

| 能力 | 归属 | 理由 |
| --- | --- | --- |
| 文件读写、shell、测试执行 | Reasonix（core/可选工具） | bridge 通过 `EXEC-01` 只做"命名命令 + argv + shell:false"的受控入口，不复制 Reasonix 的能力 |
| **`web_fetch`** | **Reasonix**（可选工具） | 无需 bridge 自建网络栈；抓取策略由 Reasonix 与其配置决定 |
| **`web_search`** | **provider 侧**（Reasonix 文档：搜索会另发一次模型请求，使用后端原生搜索；查询交给 provider，按搜索请求计费） | bridge 不具备也不必具备搜索后端；未接通 provider 时**稳定返回不可用**，不伪造结果 |
| git 只读查看 | Reasonix（MCP `gitcontext`）或受控 `bash` | 用**真实存在的工具身份**替代无效的 `git_log`/`git_diff` |
| 会话持久化/恢复 | Reasonix（ACP） | bridge 仅按 opt-in 透传（ACP-01~06 已落地为可选） |

**明确不做**（与排期文档 §2.1 一致）：不在 bridge 内自建第二套网络栈；不把 `reasonix run --allowed-tools shell` 直接透传；不擅自扩大子智能体工具面。

---

## 5. 对齐动作（票）

| 票号 | 内容 | 验收门 |
| --- | --- | --- |
| `TOOL-RXB-TOOL-01`（P1） | 工具身份对齐：把 `READ_ONLY_PROFILE_TOOLS` 收敛为**真实有效**集合（去掉 `git_log`/`git_diff`）；README 与 profile 同步；`configure verify` 对照 Reasonix 真实清单 | **已完成**：两个 profile doctor 零警告，verify 覆盖新增校验，`npm test` 99/99 |
| `TOOL-RXB-NET-01`（修订） | worker 侧直接使用 Reasonix 自带 `web_fetch`；bridge 只做透传与结果摘要（引用/unsafe URL 策略由 Reasonix 决定），**不自建网络栈** | **已完成**：两个全局 profile 已同步，原生 `example.com` 抓取返回 200/标题；bridge 未增加 URL/socket 网络路径 |
| `TOOL-RXB-NET-02`（修订） | `web_search` 归属 provider：未接通时稳定返回 unavailable；接通后只透传 provider 结果（summary/sources/truncated） | **已完成本机门控**：`reasonix_status.providerSearch` 返回 unavailable；原生探测和子智能体均未得到结果，不伪造、不回退 |
| `TOOL-RXB-LOOP-01` | 主 agent 显式标记 `plan -> implement -> exec -> review` 阶段；job/checkpoint/audit 可见，阶段不自动推进 | **已完成**：`reasonix_status.workflow`、`jobs[*].stage`、`lastRun.stage` 与 checkpoint 均记录阶段；失败仍由调用方显式取消、续跑、审查或回滚 |

---

## 6. 证据附录

| 证据 | 获取方式 | 关键输出 |
| --- | --- | --- |
| 工具面两层结构 | 内置文档 `docs/TOOL_CONTRACT.md:158-171` | core 10 个身份 + 可选工具（含 `web_fetch`） |
| 身份校验口径 | 内置文档 `docs/CAPABILITY_DIAGNOSTICS.md:43-73` | "inventory combines compile-time tools with host-managed tool identities" |
| 无效身份（历史证据） | `reasonix doctor --json` → `warnings` | 曾有 4 条 `is not a known tool identity`（`git_log`/`git_diff`）；修复后为 0 |
| 能力清点（当前） | `reasonix doctor capabilities --json` | `summary.mcp_servers=2`、`skills=10`、`warnings=0` |
| 搜索归属 | 内置文档 `docs/WEB_SEARCH.md` + changelog v1.19.7 | "opens a separate model request … backend's native search tool"；官方端点上查询发给 provider 并按搜索计费 |
| NET-01 原生抓取 | 原生 `reasonix run --allowed-tools web_fetch` + 真实 bridge MCP `reasonix_run(mode=inspect)` | `https://example.com` 返回 `HTTP status: 200 OK`、标题 `Example Domain`；`https://example.invalid` 返回 transport error 且未重试；`doctor capabilities` 为 `errors=0,warnings=0` |
| NET-02 provider 搜索 | `reasonix_status.providerSearch` + `reasonix run --allowed-tools web_search` + `reasonix subagent try deepseek-worker` | `available=false,status=unavailable,reason=provider_capability_not_advertised`；provider 返回 no native search results；worker 不回退 `web_fetch`，不输出伪造来源 |
| LOOP-01 阶段审计 | 离线 `npm test` + 真实 bridge `reasonix_status` | `npm test` 101/101；workflow 模板为 `plan,implement,exec,review`；默认预算 hard cap 为 256 raw steps / 128 tool rounds；阶段错配在 spawn 前拒绝 |
| MCP git 能力 | capability catalog | `mcp-tool:gitcontext/git_pickaxe` 可用 |

---

## 7. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-13 | 首版：登记 Reasonix v1.38.7 真实工具面（core 10 + 可选）、`git_log`/`git_diff` 无效身份的实测证据、bridge 校验失真的原因、能力归属原则，并提出 `TOOL-RXB-TOOL-01` 与 NET-01/NET-02 的修订方向 |
| 2026-09-13 | `TOOL-RXB-TOOL-01` 收口：常量、README、prompt 与两个全局 profile 已同步为 5/7 个有效身份；`doctor capabilities` 告警从 4 降为 0，`configure verify` 增加 fail-closed 诊断。 |
| 2026-09-13 | `TOOL-RXB-NET-01` 收口：profile 接入 Reasonix 原生 `web_fetch`，read/write 为 6/8；bridge 仍不提供任意 URL MCP 工具或自建网络栈，真实抓取验收通过。 |
| 2026-09-13 | `TOOL-RXB-NET-02` 收口：新增 `providerSearch` fail-closed 摘要；本机 provider 搜索不可用时稳定透传 unavailable，不自建搜索后端、不伪造结果。 |
| 2026-09-13 | `TOOL-RXB-LOOP-01` 收口：新增显式阶段元数据与 workflow 状态摘要；不自动推进、不隐式重试、不扩大权限，保留 job/checkpoint/exec/回滚审计门。 |
