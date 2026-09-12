# reasonix-codex-bridge 完善方向（2026-09-12）

> 状态：方向文档已转票；`TOOL-RXB-T1`/`TOOL-RXB-T2`/`TOOL-RXB-T3`/`TOOL-RXB-C1`/`TOOL-RXB-C2`/`TOOL-RXB-C3`/`TOOL-RXB-C4`/`TOOL-RXB-R1`/`TOOL-RXB-R2`/`TOOL-RXB-R3`/`TOOL-RXB-E1`/`TOOL-RXB-E2` 已完成，E3–E4/G1–G4 已登记为后续票池。本文件仍保留完整方向与验收门，具体进度以开发票计划为准。
>
> 创建日期：2026-09-12
> 适用范围：`tools/reasonix-codex-bridge`（独立子项目，https://github.com/SgfKrc/reasonix-codex-bridge）及其在 Codex / Reasonix 之间的接线方式。不覆盖 Reasonix 本体的模型、运行时与权限能力。

---

## 1. 基线

基线 commit：`1d1e2a6`（可配置模型预设 + configure 工具），主仓以 submodule gitlink 引用（`tools/reasonix-codex-bridge`）。

### 1.1 已具备且已验证

| 能力 | 验证方式（本机实跑） |
| --- | --- |
| stdio MCP facade（`reasonix_run` / `reasonix_status`） | stdin 喂 JSON-RPC，返回 `initialize`/`tools/list`/`tools/call` 均正常 |
| 只读子智能体 profile `deepseek-worker` | `reasonix subagent list` 显示 `[global, manual, read-only]`，工具集 `read_file,grep,glob,ls,code_index` |
| CLI 路径探测（不硬编码） | 未设 `REASONIX_EXE` 时解析到 `%LOCALAPPDATA%\Programs\Reasonix\reasonix-cli.exe`；`REASONIX_EXE` 指向缺失文件 → exit 2；清空 `LOCALAPPDATA`+`PATH` → exit 2 |
| 模型解析链 env → `bridge.config.json` → doctor `default_model` | 三条路径分别实跑并核对 `reasonix_status.modelRefSource`；畸形 ref（无 `/`、含空格）→ exit 2 |
| `configure` 六个子命令 | `list` 枚举 13 个本机 ref 并标记 `current`/`reasonix default`/`no api key`；`use` 写文件；`codex` 打印片段；`verify` 三行 OK、exit 0 |
| 配置写入安全 | `codex --write` 先做时间戳备份、只替换 `[mcp_servers.reasonix_local*]` 段；`bridge.config.json` / `presets.json` 由 `.gitignore` 排除 |

### 1.2 未验证 / 未知（本文档的驱动）

| 项 | 风险 | 现状 |
| --- | --- | --- |
| 端到端真实调用（`reasonix_run` 真跑一次子智能体） | 桥接链路可能在某处断（参数拼装、CLI 交互、输出解析），目前只有 `reasonix_status` 自检 | **未跑过**，消耗额度故未做 |
| 自动化测试 | 任何重构都可能静默回归；`npm run check` 只做 `node --check` | 仓库目前 **0 个测试** |
| CLI 版本门 | README 要求 Reasonix ≥1.38.6；本机 npm 全局版是 **1.38.3**（桌面版 1.38.7），说明"装到旧版"是真实场景 | 代码不校验版本 |
| profile 与模型 ref 的一致性 | `configure use` 只改 `bridge.config.json`，不会同步 profile frontmatter 的 `model` | 可能"A 配置选了新模型，profile 还指向旧模型" |
| POSIX 分支 | `terminate()` 的 kill 路径、Unix CLI 探测位置从未在真实 POSIX 机器上跑过 | 仅代码审查 |
| doctor 调用开销 | 每次启动与每个 configure 命令都 spawn 一次 doctor（实测 1–2 秒量级） | 无缓存 |

---

## 2. 完善方向

### 2.1 P0 — 可信度与回归

- **T1 自动化测试（零依赖，`node:test`）**
  - 做什么：三组测试。① `config.mjs` 纯函数：解析链三条优先级、`validateModelRef` 边界、`doctorRefs` 归一化（`models[]` / 单 `model` 两种形态）、`upsertReasonixBlock` 的追加/替换/相邻段边界；② `configure.mjs` 命令级：把 `BRIDGE_CONFIG`/`CODEX_CONFIG` 指向 tmp，跑 `use` / `codex --write` 后回读断言；③ MCP 会话级：spawn `src/server.mjs`，用 PATH 上的 doctor stub 替代真实 CLI，断言 `tools/list` 与 `reasonix_status` 输出结构。
  - 验收门：`node --test` 全绿；覆盖解析链三分支与 TOML upsert 三形态；测试不依赖真实模型、不联网。

- **T2 CLI 版本兼容门**
  - 做什么：解析 `reasonix --version`，低于 1.38.6 时 `verify` 报 fail；server 启动默认 refuse（与现有 fail-closed 一致），可用 `REASONIX_MIN_VERSION` 显式放宽并打印警告。版本探测失败不阻断（CLI 可能输出格式变化），但要在 `reasonix_status` 里标注 `versionCheck: unknown`。
  - 验收门：用低版本 stub 时 `verify` fail、server refuse；真实 1.38.7 通过；`REASONIX_MIN_VERSION=1.0` 时降级为警告。

- **T3 端到端冒烟（低频、手动）**
  - 做什么：跑一次 `mode=inspect` 的小任务（例如"列出 src/config.mjs 的导出函数"），记录耗时、输出长度、是否触发截断，并把结果登记回本文档 §1.1。
  - 验收门：一次真实调用成功且输出可直接引用；失败时把失败形态（超时 / 退出码 / CLI 报错）写成新的 §1.2 条目，而不是沉默。

### 2.2 P1 — 配置与集成体验

- **C1 profile 同步与一致性检查**
  - 做什么：新增 `configure profile [--create|--sync] [--write]`（默认打印 `reasonix subagent create|edit` 命令，`--write` 才执行并回读校验）；`verify` 增加一致性检查——读 `%APPDATA%\reasonix\skills\<name>\SKILL.md` 的 frontmatter，比对 `model` 与当前 ref、`read-only` 是否存在。
  - 验收门：人为制造 drift（profile model 与 bridge.config.json 不一致）时 `verify` 报 fail；`--sync` 后恢复一致并留痕。

- **C2 Codex 配置写入健壮性**
  - 做什么：写入前检测重复段并合并；写入后回读做轻量结构校验（段名集合 + 必需键存在）；CRLF/LF 混合与段位于文件首/尾都要正确；文件不可写时明确报错（不半写）。
  - 验收门：四类样例（重复段、CRLF、段在首位、段在末位）全部通过，且写到临时文件后再原子替换。

- **C3 doctor 结果缓存**
  - 做什么：在 `bridge.config.json` 旁缓存 doctor 摘要（`cliPath` + CLI mtime + version + 抓取时间），TTL 默认 10 分钟、`configure` 侧可 `--refresh` 强制刷新；缓存失效条件包含 CLI 文件变更。
  - 验收门：二次启动可测地变快；改 CLI 文件或过期后自动重取；缓存损坏时回退为重新抓取而非报错。

- **C4 环境摘要导出/导入（可选）**
  - 做什么：`configure export` 输出脱敏环境摘要（平台、CLI 版本、provider 名与模型名、当前 ref、profile 名），便于贴到 issue/群聊；`configure import` 用于对照他人环境。
  - 验收门：导出内容不含 key、不含完整 endpoint、不含用户路径；导入不直接改配置（只打印差异）。

### 2.3 P1 — 运行时与观测

- **R1 结构化调用日志（脱敏）**
  - 做什么：可选 `BRIDGE_LOG`（JSONL），每次调用写一条：时间、mode、cwd 根标签、maxSteps、timeout、退出码、耗时、输出字节、是否截断。**不记录 task 正文与输出正文**。
  - 验收门：日志可被 `jq` 解析；正文不出现在日志；关闭时零写入。

- **R2 限额可配置（带硬上限）**
  - 做什么：`MAX_STEPS_CAP` / `TIMEOUT_SECONDS_CAP` / `OUTPUT_CHAR_CAP` / `queueCap` 允许在 `bridge.config.json` 覆盖，但仍被代码内的硬上限夹紧；非法值回退默认并打印一次警告。
  - 验收门：覆盖生效且越界被夹紧；`reasonix_status.limits` 反映实际生效值。

- **R3 队列与在途可观测**
  - 做什么：`reasonix_status` 暴露 `queueDepth`、`inFlight`、`lastRun`（脱敏摘要）。
  - 验收门：并发灌入请求时 status 反映真实深度；队列满的错误信息附带当前深度与建议重试时间。

### 2.4 P2 — 能力扩展

- **E1 受控方案模式（`mode=plan`）**
  - 做什么：仍只读，但要求 worker 输出可机器解析的"改动建议清单"（文件、位置、理由、最小 patch 摘要），桥接层只透传、不写盘；`implement` 保持禁用。
  - 验收门：输出含结构化清单；桥接层无任何写路径（代码审查 + 测试断言无 `writeFile` 调用）。

- **E2 持久 ACP transport（设计先行）**
  - 做什么：先写设计：会话生命周期、compact/rotate 触发点（必须在 128MB 前主动压缩）、与 stateless 模式的关系、失败时回退到 per-call。
  - 验收门：设计文档评审通过；原型能在超限前主动 compact，且 compact 失败时能无副作用回退。

- **E3 provider 能力透传**
  - 做什么：`reasonix_status` 暴露当前 ref 的 `contextWindow`、是否 vision、所属 provider 的 `base_url_host`；对明显超过 context window 的 task 在调用前给出明确拒绝或警告。
  - 验收门：超长任务被调用前拦截并给出具体上限数值。

- **E4 只读工具集扩展（按需）**
  - 做什么：如确需，扩展 profile 工具集（例如只读的 `git log` / `git diff` 查看器），每加一项都要在 profile frontmatter 与本文档登记。
  - 验收门：`verify` 能列出实际工具集，且与文档一致。

### 2.5 P2 — 工程化与发布

- **G1 CI**：GitHub Actions 跑 `node --check`（三个 mjs）+ `node --test` + README 内链接检查；全部离线可跑。
- **G2 版本与变更记录**：`package.json` 版本语义化 + `CHANGELOG.md`；子仓打首个 tag（v0.1.0）。
- **G3 跨平台验证**：至少在 WSL 或 CI 上跑一次 CLI 探测 + 启动自检 + `configure verify`，覆盖 `terminate()` 的 POSIX 分支。
- **G4 主仓集成登记**：在主仓 `docs/` 登记接线清单（Codex 配置位置、profile 名、本机 CLI 与 workspace root），换机时照抄即可，避免重做。

---

## 3. 边界（明确不做）

1. **不给子智能体写权限**：桥接层永远是只读执行器，"谁改代码"始终是 Codex（主智能体）。
2. **不把 128MB 会话历史上限做成可配置**：它是 Reasonix 的硬约束，只能压缩或轮换，不能声明为"可调"。
3. **不引入运行时依赖**：保持 zero-dependency（Node 内置模块 + `node:test`）；需要外部能力时用 CLI 而非 SDK。
4. **不在桥接层做自动模型选择/降级**：模型必须显式选择；唯一的"自动"是回退到本机 `default_model`，且必须在 `reasonix_status` 里标注来源。
5. **不代管他人进程**：任何终止动作只针对本 bridge 自己登记的 PID，且先复核身份（沿用主仓演示工具链的重置哲学）。

---

## 4. 建议执行序

1. **T1 自动化测试** —— 先建回归网，后续改动才有底气
2. **T2 CLI 版本门** —— 低版本误装是真实现场（本机 1.38.3/1.38.7 并存已经暴露）
3. **T3 端到端冒烟** —— 拿到第一条真实链路证据，回填 §1.1
4. **C1 profile 同步 + verify drift 检查** —— 消除"配置两处不一致"这一最常见误配
5. **R1 结构化日志 + R3 队列可观测** —— 先有观测，才谈优化
6. **C3 doctor 缓存** —— 降低每次启动/配置的固定开销
7. 其余（C2/C4/R3/E3–E4/G1–G4）按需插入

---

## 5. 验收门汇总

| 方向 | 验收门 | 证据形式 |
| --- | --- | --- |
| T1 | `node --test` 全绿，覆盖解析链三分支 + upsert 三形态 | 测试输出 |
| T2 | 低版本 stub → verify fail / server refuse；放宽开关 → 警告 | 命令输出 + exit code |
| T3 | 一次真实 `mode=inspect` 调用成功并登记耗时/长度 | 调用输出 + 本文档回填 |
| C1 | 人为 drift → verify fail；`--sync` 后一致 | verify 前后对照 |
| C2 | 重复段 / CRLF / 首尾段四类样例通过，且原子替换 | 测试输出 + 文件 diff |
| C3 | 二次启动变快，CLI 变更或过期自动重取 | 计时 + 缓存文件 |
| C4 | 导出无 key / 无完整 endpoint / 无用户路径 | 导出样本审查 |
| R1 | JSONL 可解析且不含正文 | 日志样本 |
| R2 | 覆盖生效且被硬上限夹紧，status 反映实际值 | status 输出 |
| R3 | 并发时 status 反映真实深度 | status 采样 |
| E1 | 输出含结构化清单，代码无写路径 | 调用输出 + 代码审查 |
| E2 | 设计评审通过 + 原型在超限前 compact | 设计文档 + 原型日志 |
| E3 | 超长任务调用前被拦截并给出上限值 | 调用输出 |
| E4 | verify 列出的工具集与文档一致 | verify 输出 |
| G1–G3 | CI 全绿；首个 tag；POSIX 至少跑通一次 | CI 记录 |
| G4 | 主仓接线清单可照抄完成新机接入 | 文档 + 一次实操 |

---

## 6. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-12 | 首版：基线 commit `1d1e2a6`，列 P0（T1–T3）/P1（C1–C4、R1–R3）/P2（E1–E4、G1–G4）方向与验收门，明确五条边界 |
| 2026-09-12 | 按资源约束转出 `TOOL-RXB-T1`/`T2`/`T3`；T1/T2 以离线 stub 完成，真实调用 T3 交主节点，不在本机消耗模型额度 |
| 2026-09-12 | `TOOL-RXB-C1` 完成：profile 预览/同步、model/read-only drift 门与显式写后回读校验落地；本机只做预览与 verify，未改全局 profile |
| 2026-09-12 | `TOOL-RXB-T3` 完成：本机 bridge 通过 Reasonix `v1.38.7` 执行 ASCII-only 真实 `mode=inspect` 只读任务，退出码 `0`，登记非空未截断输出与 `10.3s` 耗时；一次 `max_steps=6` 暂停按失败形态保留，不作模型质量结论 |
| 2026-09-12 | `TOOL-RXB-C2` 完成：Codex bridge 配置写入增加必需键校验、重复段合并、换行风格保持和同目录原子替换/失败恢复；`npm test` 17 项通过 |
| 2026-09-12 | `TOOL-RXB-C3` 完成：doctor 摘要缓存加入 CLI mtime/version/抓取时间元数据，默认 TTL 10 分钟，支持 `--refresh`，损坏/过期/CLI 变更自动重取；本机 `configure list` 首次/命中/强刷为 `2317ms`/`90ms`/`2227ms`，`npm test` 19 项通过 |
| 2026-09-12 | `TOOL-RXB-C4` 完成：新增路径无关、脱敏的 `configure export` JSON 摘要与只读 `configure import <file|->` 对照；拒绝 key、endpoint、用户路径等外部敏感值，`npm test` 20 项通过 |
| 2026-09-12 | `TOOL-RXB-R1` 完成：`BRIDGE_LOG` 可选 JSONL 脱敏调用日志覆盖成功、拒绝、非零退出，不记录 task/输出正文、模型 ref 或绝对路径；未设置时零写入，`npm test` 21 项通过 |
| 2026-09-12 | `TOOL-RXB-R2` 完成：`bridge.config.json.limits` 支持 steps/timeout/output/queue 覆盖，非法值回退并一次告警，超过代码硬上限夹紧；`reasonix_status.limits` 反映有效值，离线回归 `npm test` 23 项通过 |
| 2026-09-12 | `TOOL-RXB-R3` 完成：`reasonix_status` 增加 queueDepth/inFlight/lastRun 脱敏摘要，队列满错误附当前深度、容量和 retry-after 提示；离线并发 stub 验证状态转移，`npm test` 24 项通过 |
| 2026-09-12 | `TOOL-RXB-E1` 完成：新增只读 `mode=plan`，原样透传机器可解析的 `qlh.reasonix.plan.v1` 建议清单，桥接层不解析或写盘，`implement` 继续禁用；`npm test` 25 项通过 |
| 2026-09-12 | `TOOL-RXB-E2` 完成：新增设计专文与未接入生产的纯函数 ACP 原型；在固定 128 MiB 历史上限的 75% 触发事务性 compact，compact 后仍超限则 rotate，compact 失败回退 per-call 且持久历史无副作用；`npm test` 28 项通过 |
