# reasonix-codex-bridge 完善方向（2026-09-12）

> 状态：方向文档已转票；`TOOL-RXB-T1`/`TOOL-RXB-T2`/`TOOL-RXB-T3`/`TOOL-RXB-C1`/`TOOL-RXB-C2`/`TOOL-RXB-C3`/`TOOL-RXB-C4`/`TOOL-RXB-R1`/`TOOL-RXB-R2`/`TOOL-RXB-R3`/`TOOL-RXB-E1`/`TOOL-RXB-E2`/`TOOL-RXB-E3`/`TOOL-RXB-E4`/`TOOL-RXB-G1`/`TOOL-RXB-G2`/`TOOL-RXB-G3`/`TOOL-RXB-G4`/`TOOL-RXB-W1`/`TOOL-RXB-W2`/`TOOL-RXB-W3`/`TOOL-RXB-AUD-01`/`TOOL-RXB-AUD-02`/`TOOL-RXB-AUD-03`/`TOOL-RXB-AUD-04`/`TOOL-RXB-AUD-05`/`TOOL-RXB-AUD-06`/`TOOL-RXB-AUD-07`/`TOOL-RXB-AUD-08`/`TOOL-RXB-R2-EXT-01`/`TOOL-RXB-E2-EXT-01`/`TOOL-RXB-R3-EXT-01`/`TOOL-RXB-R4` 已完成；G3 已在 WSL Ubuntu 22.04 完成跨 POSIX 实跑。本文件仍保留完整方向与验收门，具体进度以开发票计划为准。写入策略继续默认关闭。
>
> 创建日期：2026-09-12
> 适用范围：`tools/reasonix-codex-bridge`（独立子项目，https://github.com/SgfKrc/reasonix-codex-bridge）及其在 Codex / Reasonix 之间的接线方式。不覆盖 Reasonix 本体的模型、运行时与权限能力。

---

## 1. 基线

基线 commit：`1d1e2a6`（可配置模型预设 + configure 工具），主仓以 submodule gitlink 引用（`tools/reasonix-codex-bridge`）。

### 1.1 已具备且已验证

| 能力 | 验证方式（本机实跑） |
| --- | --- |
| stdio MCP facade（`reasonix_run` / `reasonix_resume` / `reasonix_cancel` / `reasonix_status`） | stdin 喂 JSON-RPC，返回 `initialize`/`tools/list`/`tools/call` 均正常 |
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

### 2.6 网络能力委派修复（2026-09-14，修复票）

> 背景（2026-09-14 诊断）：`web_search` 现只认 **provider 原生声明**（`resolveProviderSearchCapability` 检查 provider 的 `web_search/webSearch/capabilities.*/tools.*`；缺失即 fail-closed，不推断），而 `reasonix doctor --json` 的 provider schema 不含这类字段 → 对**所有**配置 provider 稳定 `provider_capability_not_advertised`；同时 Reasonix 直连时宿主层联网可用（宿主/插件/网关通道），造成"直连能搜、桥接内稳定 unavailable"的观感差异。`web_fetch` 已由 NET-01 委派宿主（`f6a539c`），`web_search` 缺对应委派。

- **`TOOL-RXB-NET-03` host search 委派通道（P0）**
  - 做什么：按 NET-01 对 `web_fetch` 的委派模式，探测 **Reasonix 宿主声明的 search 能力**并委派执行；通道选择明确化：provider 原生声明优先 → host 委派兜底 → 无通道则 fail-closed（原因码保持稳定）；`reasonix_status` 增加 search 通道来源（`provider` / `host` / `unavailable`）；修正探测口径（不再只吃 doctor 的 provider 字段，需覆盖宿主能力渠道）。
  - 验收门：宿主 search 可用时 `web_search` 返回真实结果且标注通道来源；宿主不可用时 fail-closed 且原因码与现行为一致；与 NET-01 fetch 委派的脱敏/边界合同一致；回归覆盖"provider/host/无通道"三态选择矩阵。

- **`TOOL-RXB-NET-04` Codex 主 agent 联网回退（P1）**
  - 背景：本地两通道（provider + host）都不可用时，子 agent 需要外部信息却拿不到；但主 agent（Codex）本身有联网能力，可作为兜底供料方。
  - 做什么：设计"联网求助"协议——bridge 在子 agent 明确需要联网而本地通道不可用时，返回**结构化缺口信号**（如 `needs_network: true` + 查询意图/URL 列表；不夹带无关上下文）；主 agent 用自身联网获取信息后，把结果作为**一次性补充上下文**经新参数/工具（如 `reasonix_run` 的 `context`/`attachments` 字段或独立 `reasonix_inject`）回灌给随后调用的 Reasonix 子 agent；全链路审计（记录"本次调用由主 agent 供料"）。
  - 边界：回退通道不替代 NET-03；主 agent 供料遵守数据边界策略（敏感内容不默认外发）；一次性注入，不做持久共享；默认只读/写入边界不变。
  - 验收门：无本地通道时子 agent 收到结构化缺口信号（不空转、不乱猜）；主 agent 供料后子 agent 能基于供料完成回答；审计与事件流标注供料来源；不改变现有写策略与脱敏合同。

### 2.7 写入策略演进（2026-09-14，修复票）

> 背景（2026-09-14 实测与用户判定）：`requireCleanTree=true` 要求**每次写入调用前工作树 clean**——但 agent 场景中**一写即脏**，第二次写入起即被拒（"写入 → 树变脏 → 后续写入全被 clean-tree 门拦死"），与真实使用不兼容。**当前开发阶段**（工具不成熟）接受这种严格受控的写入；**将来不允许**保留该前置门，必须砍掉/优化。

- **`TOOL-RXB-W1-EXT-01` clean-tree 前置门优化（P1，当前保留、将来必改）**
  - 问题：clean-tree 作为**写入前置门**与 agent 工作模式根本冲突——任何一次成功写入都会让工作树变脏，使后续写入被自己造成的脏树拒绝；实际使用中无法连续写入。
  - 优化方向（择一或组合，实现时再定）：
    - **a) 前置门 → 回滚依据**：允许脏树写入，改为写入前记录**目标路径的内容/哈希快照**，回滚令牌与冲突检测基于快照（目标文件在写入后被外部修改 → 拒绝回滚，fail-closed），不再要求整树 clean；
    - **b) 范围收窄**：仅校验 `allowedPaths` 内、且为**本次目标路径**的 dirty 状态；工作树其他部分的未提交改动不阻塞；
    - **c) 策略档位**：区分"开发/测试态"（允许脏树 + 快照回滚）与"生产态"（保留 clean-tree 前置门），默认档位显式配置并在 `reasonix_status` 可见。
  - 验收门：**连续两次写入**（第一次后工作树已脏）第二次不被 clean-tree 拒绝；回滚令牌仍能精确回滚本次变化且**不误伤既有未提交改动**；目标文件被第三方修改后回滚/冲突检测仍 fail-closed；`reasonix_status` 明确展示当前 clean-tree 策略与档位。迁移前保持现状并在文档/票中标注"严格受控仅为开发阶段临时口径"。

---

## 3. 边界（明确不做）

1. **默认不给子智能体写权限**：桥接层默认是只读执行器；只有 W1 的双开关、路径白名单、干净树和回滚门全部满足时，才允许显式 `mode=implement`，主 agent 仍负责审查。
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
7. 其余（C2/C4/R3/G3）按需插入；G4 已通过主仓接线清单收口

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
| G1–G2 | CI 全绿；首个 tag | CI/tag 记录 |
| G3 | POSIX/WSL 实跑 CLI 探测、启动自检、verify 与 terminate 分支 | WSL Ubuntu 22.04 实跑记录；当前 shell 复跑因 Linux Node 未安装而受限 |
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
| 2026-09-12 | `TOOL-RXB-E3` 完成：`reasonix_status` 透传当前 ref 的 `contextWindow`/`vision`/`base_url_host` 能力摘要；调用前按 UTF-8 字节保守估算 task tokens，明显超过上限时拒绝且不启动 worker；能力探测不落盘 doctor cache，`npm test` 29 项通过 |
| 2026-09-12 | `TOOL-RXB-E4` 完成：profile 只读白名单扩展为 `read_file,grep,glob,ls,code_index,git_log,git_diff`，README/worker prompt 同步登记；`profile --sync` 固定该集合，`verify` 列出实际工具并对集合漂移 fail-closed；`npm test` 30 项通过 |
| 2026-09-12 | `TOOL-RXB-G1` 完成：新增 `.github/workflows/ci.yml`，Node 20 离线执行 `npm run check`、`npm test`、`npm run check:links`；链接检查只解析 README 本地相对路径，外部 URL/anchor/mailto 不触网；`npm test` 31 项通过 |
| 2026-09-12 | `TOOL-RXB-G2` 完成：package 版本固定为 semver `0.1.0`，新增首版 `CHANGELOG.md` 与版本一致性回归；子仓创建注释 tag `v0.1.0`；`npm test` 32 项通过 |
| 2026-09-12 | `TOOL-RXB-G3` 暂缓：当时本机 `wsl.exe --list --quiet` 返回空列表，未伪造 POSIX 运行证据 |
| 2026-09-12 | `TOOL-RXB-G4` 完成：主仓新增接线清单，登记 Codex 配置、`deepseek-worker` profile、Reasonix CLI、workspace root、换机命令与脱敏边界；通过 `configure show`/`codex` 无写入预览核对 |
| 2026-09-12 | `TOOL-RXB-W1` 完成：bridge 默认关闭 `mode=implement`，显式 `allowWrite` + 非空 `allowedPaths` + 干净 Git 树后才允许写入；越界路径或 worker 失败回滚本次变化，离线回归 36 项通过 |
| 2026-09-12 | `TOOL-RXB-W2` 完成：成功写调用仅返回 `qlh.reasonix.changes.v1` 结构化变更集（路径、增删行数、diff stat、SHA-256），不返回 worker 输出或文件正文；新增 `reasonix_rollback` 一次性回滚令牌，回滚前校验后续修改并冲突拒绝；令牌仅存当前 bridge 进程，离线回归 37 项通过 |
| 2026-09-12 | `TOOL-RXB-W3` 完成：新增独立 write profile 与专用 prompt，`configure profile --role read/write` 可分别生成/同步并验证角色 guard；write profile 只在 read 工具集合上增加 `edit_file,write_file`，不带 `read-only`，README 固化主 agent 指挥、子 agent 执行、主 agent 审查/回滚流程，离线回归 40 项通过 |
| 2026-09-12 | `TOOL-RXB-AUD-01` 完成：审计发现的模式授权边界已修复；inspect/review/plan 强制 read-role，implement 要求显式 write-role，canonical `-write` profile 不可被 `REASONIX_SUBAGENT_ROLE=read` 降级；离线回归 43 项通过，子项目 commit `f6bacd7` |
| 2026-09-12 | `TOOL-RXB-AUD-02` 完成：拒绝不安全的 `requireCleanTree=false` opt-out，受控写入统一要求 clean Git tree；新增 worker 不启动回归，子项目 commit `a361634` |
| 2026-09-13 | `TOOL-RXB-AUD-03` 完成：显式 `reasonix_rollback` 纳入与 implement 共用的进程内串行队列；新增同一工作区并发回归，验证 rollback 不与 worker 写入交错 |
| 2026-09-13 | `TOOL-RXB-AUD-04` 完成：Windows `.cmd/.bat` 通过显式 `cmd.exe /d /s /c`、`shell:false` 启动；命令元字符在进程创建前拒绝，`configure`、doctor、version、worker 共用安全调用解析器，并补启动/参数回归 |
| 2026-09-13 | `TOOL-RXB-AUD-05` 完成：rollback 哈希改为 `readable/missing/unreadable` 三态；不可读和类型变化默认拒绝，缺失与哈希冲突分别返回语义化错误，restore 后复核 Git 状态与 SHA-256 |
| 2026-09-13 | `TOOL-RXB-AUD-06` 完成：Git rename/copy 目标按新增路径处理；回滚清理前撤销 staged index，避免 rename 目标残留导致 post-check 误报；新增 staged rename 回归，子项目 commit `8e7693c`，`npm test` 48 项通过 |
| 2026-09-13 | `TOOL-RXB-AUD-07` 完成：确认 Reasonix `--max-steps` 是 raw 内部步数而非工具轮次；`max_steps=10` 复现 5 轮后暂停且未到 120 秒，新增 `tool_rounds` 映射、`step_limit` 失败语义与回归，子项目 commit `6d093c1`，`npm test` 51 项通过 |
| 2026-09-13 | `TOOL-RXB-AUD-08` 完成：read/write prompt 固化 continuation cursor 原样回传约束；bridge 将 malformed/invalid cursor 分类为 `cursor_error`、隐藏失效 token 且禁止自动重放，并修正 prompt 同步命令路径，子项目 commit `491c435`，`npm test` 53 项通过 |
| 2026-09-13 | `TOOL-RXB-R2-EXT-01` 完成：预算边界放宽为最多 256 raw steps/128 工具轮次和 1800 秒；默认 mode 预算不变，仍需显式传 `tool_rounds`/`timeout_seconds`；配置、夹紧和长预算映射回归共 `npm test` 54 项通过，子项目 commit `26b4113` |
| 2026-09-13 | `TOOL-RXB-E2-EXT-01` 完成：新增任务级持久 checkpoint 与显式 `reasonix_resume`；失败时保存任务、预算、版本/配置和 Git 状态指纹，不保存 stdout/stderr；恢复前拒绝 workspace/config drift，checkpoint 一次性消费并以原子 claim 锁防并发重复；跨进程 fixture 和派生 write profile 回归后 `npm test` 57 项通过，子项目 commit `a596921` |
| 2026-09-13 | `TOOL-RXB-R3-EXT-01` 完成：新增显式 `parallel=true` 只读 worker 槽位、job 状态展开和 `reasonix_cancel`；implement/resume/rollback 保持 workspace 独占，取消终止进程树且不生成 checkpoint；并行重叠、取消回收和既有串行写入回归后 `npm test` 59 项通过，子项目 commit `782d6b0` |
| 2026-09-13 | `TOOL-RXB-G3` 完成：在 WSL Ubuntu 22.04 记录 POSIX CLI 探测、无 CLI 启动拒绝、WSL interop 下 `--version`/doctor/verify 和测试结果；跨平台夹具修复后记录为 58/59，剩余输出截断竞态转为 R4 |
| 2026-09-13 | `TOOL-RXB-R4` 完成：输出超过 `OUTPUT_CHAR_CAP` 时改为有界缓冲并以成功结果标记 `truncated=true`，不因输出超限终止 worker；timeout/cancel 仍终止。Windows `npm test` 59/59、`npm run check`、`npm run check:links` 通过，子项目 commit `fa06fd2` |
| 2026-09-14 | `TOOL-RXB-EVT-01` 完成：新增 `reasonix_events` 有界事件流（`after_seq`/`limit` 增量轮询；仅返回 job id、阶段、状态、终态与有界计数，任务文本/模型引用/路径/worker 输出一律不返回；每 job 环形缓冲上限 32 条、完成即清）；工具面现为 `run/resume/rollback/events/exec/cancel/status` 七项；子项目 commit `5104f3d`，`npm test` `103 passed / 0 failed` |
| 2026-09-14 | 登记修复票 `TOOL-RXB-NET-03`（host search 委派通道，P0）与 `TOOL-RXB-NET-04`（Codex 主 agent 联网回退，P1），详见 §2.6；背景为 2026-09-14 诊断"`web_search` 仅认 provider 原生声明（doctor 不含该字段）→ 对全部 provider 稳定 fail-closed，而 Reasonix 直连宿主联网可用"的通道归属差异 |
| 2026-09-14 | 本机开放写权限用于真实测试（write profile `deepseek-worker-write` 创建、`bridge.config.json` 写策略 `enabled=true`）；同时登记修复票 `TOOL-RXB-W1-EXT-01`（clean-tree 前置门优化，详见 §2.7）：`requireCleanTree` 使"一写即脏、后续写入被拒"与 agent 场景不兼容，**当前开发阶段接受严格受控，将来必须砍掉/优化**（方向：前置门→快照回滚依据 / 范围为本次目标路径 / 开发态与生产态策略档位） |
