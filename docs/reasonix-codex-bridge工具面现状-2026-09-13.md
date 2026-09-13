# reasonix-codex-bridge 工具面现状与能力归属（2026-09-13）

> 状态：**现状登记**；本文记录 Reasonix v1.38.7 的真实工具面、bridge 声明与现实的差异，以及"能力归属"原则（能丢给 Reasonix 的都交给 Reasonix）。修复动作见 [排期文档](reasonix-codex-bridge-Harness工具扩展与能力补齐排期-2026-09-13.md) 的 `TOOL-RXB-TOOL-01` 与 `TOOL-RXB-NET-01/02`。
>
> 创建日期：2026-09-13
> 证据类型：Reasonix **内置文档**（`docs/TOOL_CONTRACT.md` 等，随 v1.38.7 打包）+ 本机 `reasonix doctor` / `doctor capabilities` 实测输出。

---

## 1. 结论摘要

1. Reasonix 的工具面分**两层**：provider 可见的 **core**（每次任务固定出现）与留在 host registry、经 `use_capability` 调度的**可选工具**。
2. bridge 契约中的 **`git_log` 与 `git_diff` 不是 Reasonix 已知的工具身份**——本机 `doctor` 连续报出 4 条警告（read/write 两个 profile × 2），即这两个工具**从未真正生效**。
3. `read_file, grep, glob, ls, code_index, edit_file, write_file` 均为**有效**身份（医生清单未对其报警）。
4. **`web_fetch` 是 Reasonix 自带的可选工具**；搜索（`web_search`）按官方文档是 **provider 侧能力**（搜索会另发一次模型请求，查询交给 provider）。因此网络能力**不落在 bridge**——bridge 只做门控与透传。
5. bridge 的 `configure verify` 目前按**自己的期望集**比对，因而对上述差异报"一致"；它尚未对照 Reasonix 的真实清单。

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
| `read_file` `grep` `glob` `ls` `code_index` | read profile 5 件套 | 有效身份（doctor 无警告） | ✅ 一致 |
| `edit_file` `write_file` | write profile 追加 | core 工具 | ✅ 一致 |
| **`git_log`** | read + write profile、README "canonical read-only tool set"、`READ_ONLY_PROFILE_TOOLS` | **"not a known tool identity"** | ❌ **无效** |
| **`git_diff`** | 同上 | **"not a known tool identity"** | ❌ **无效** |

本机 `doctor` 原始警告（4 条，原文）：

```
skill "deepseek-worker"       allowed-tools reference "git_log"  is not a known tool identity
skill "deepseek-worker"       allowed-tools reference "git_diff" is not a known tool identity
skill "deepseek-worker-write" allowed-tools reference "git_log"  is not a known tool identity
skill "deepseek-worker-write" allowed-tools reference "git_diff" is not a known tool identity
```

**实际生效的工具集**因此是：read = `read_file, grep, glob, ls, code_index`（5）；write = 上述 + `edit_file, write_file`（7）。

### 3.1 影响

1. **能力缺口**：worker 没有任何 git 只读能力；"让子智能体看 git log/diff"的预期从未成立。
2. **校验失真**：`configure verify` 的"工具集一致"只对照 bridge 自己的常量，**不能证明 Reasonix 认这些工具**。
3. **文档失真**：bridge README 的"canonical read-only profile tool set"与"verify 会在工具集漂移时失败"两处描述，需要按真实现实修订。

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
| `TOOL-RXB-TOOL-01`（新增，P1） | 工具身份对齐：把 `READ_ONLY_PROFILE_TOOLS` 收敛为**真实有效**集合（去掉 `git_log`/`git_diff`）；README 与 profile 同步；`configure verify` 增加"对照 Reasonix 真实清单"的可选校验（或至少在 verify 输出中标注"未对照 CLI 清单"） | 修正后 `doctor` 对两个 profile **零警告**；verify 覆盖新增校验；`npm test` 全绿 |
| `TOOL-RXB-NET-01`（修订） | worker 侧直接使用 Reasonix 自带 `web_fetch`；bridge 只做透传与结果摘要（引用/unsafe URL 策略由 Reasonix 决定），**不自建网络栈** | 失败态稳定透传；不引入新网络代码路径 |
| `TOOL-RXB-NET-02`（修订） | `web_search` 归属 provider：未接通时稳定返回 unavailable；接通后只透传 provider 结果（summary/sources/truncated） | 无搜索后端时不得伪造；接通后 citation 完整 |

---

## 6. 证据附录

| 证据 | 获取方式 | 关键输出 |
| --- | --- | --- |
| 工具面两层结构 | 内置文档 `docs/TOOL_CONTRACT.md:158-171` | core 10 个身份 + 可选工具（含 `web_fetch`） |
| 身份校验口径 | 内置文档 `docs/CAPABILITY_DIAGNOSTICS.md:43-73` | "inventory combines compile-time tools with host-managed tool identities" |
| 无效身份 | `reasonix doctor --json` → `warnings` | 4 条 `is not a known tool identity`（`git_log`/`git_diff`） |
| 能力清点 | `reasonix doctor capabilities --json` | `summary.mcp_servers=2`、`skills=10`、`warnings=4` |
| 搜索归属 | 内置文档 `docs/WEB_SEARCH.md` + changelog v1.19.7 | "opens a separate model request … backend's native search tool"；官方端点上查询发给 provider 并按搜索计费 |
| MCP git 能力 | capability catalog | `mcp-tool:gitcontext/git_pickaxe` 可用 |

---

## 7. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-13 | 首版：登记 Reasonix v1.38.7 真实工具面（core 10 + 可选）、`git_log`/`git_diff` 无效身份的实测证据、bridge 校验失真的原因、能力归属原则，并提出 `TOOL-RXB-TOOL-01` 与 NET-01/NET-02 的修订方向 |
