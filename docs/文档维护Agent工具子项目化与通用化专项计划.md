# 文档维护 Agent 工具子项目化与通用化专项计划

> 状态：**`DOCAGENT-P5B` 已完成本机开发门，P5 规则演进门控阶段完成**；目标 = 将本仓库已有 `docs/agent_tool` 套件抽成**通用子项目 docagent**（各项目通用的文档维护 agent 工具），规则**数据化**为初始规则文档，并交付**规则变更后的机械扫描工具**；"允许 agent 修改维护规则"列为将来门控能力
>
> 创建日期：2026-09-08
> 适用范围：文档维护工具的子项目化与规则演进机制；不覆盖文档内容自动改写、语义总结生成、跨项目内容标准统一。与 [文档维护 Agent 工具设计](文档维护Agent工具设计.md)（M1-M3 原始设计）、[小模型轻量推理harness工作台调研与方案](小模型轻量推理harness工作台调研与方案.md) 的关系见 §9。

---

## 1. 目标与定位

本仓库已有一套约 2500 行的文档维护工具（§2），但它是**本仓库专用**：规则内嵌代码、路径硬编码、约定写死。本专项把它升级为**通用子项目**：

1. **子项目化**：抽取为独立包 `docagent`（先独立目录 + `pyproject.toml`，后续可独立仓库/PyPI），主项目只保留薄壳入口（可选依赖，`pip install docagent` 或路径安装）。
2. **通用化**：任意仓库可用——`--root <repo>` 指定目标、**规则即数据**（初始规则文档 + 结构化规则文件）、profile 表达项目差异（docs 目录、例外路径、词汇表、是否启用 git 关联）。
3. **初始规则文档**：把本仓库既有约定（状态行/更新日期/适用范围/变更记录/链接/命名/术语）固化为**可读规则集 v1 + 机器可读 rules.yaml**，规则不再藏在 Python 代码里。
4. **规则演进机制（本期交付）**：规则变更走"**预演-门控-发布**"——用新规则对全库机械扫描，产出**违规增量矩阵**（新增/消失/等级变化），防误报爆炸与静默漏检；规则版本化（semver + schema_version）。
5. **允许 agent 修改维护规则（将来）**：agent 只改**规则数据**（YAML），不改扫描器代码；变更经预演门控（低风险自动、高风险人工确认），回滚 = git revert + 重扫。

**边界**：本工具**不改写文档内容**（M2 LLM 判定只输出"疑似"建议，不自动落笔）；不统一各项目的内容风格，只统一"维护状态可机器核查"这一件事。

## 2. 现状盘点（本仓库已有资产）

| 模块 | 行数 | 作用 | 通用化阻塞点 |
|---|---|---|---|
| `docs/agent_tool/doc_maintenance_audit.py` | 480 | M1 扫描器：R1-R5 规则、git 脏文件/源码关联 | 规则表、豁免词、R1/R5 正则**内嵌代码**；`REPO_ROOT/DOCS_DIR/OUT_DIR` 硬编码 |
| `doc_maintenance_llm.py` | 354 | M2 本地/远程 LLM 判定（DeepSeek 远程 + Ollama 兜底） | provider 配置在 `.env.docagent`，路径假定 |
| `doc_maintenance_events.py` | 667 | M3 事件库（SQLite schema、写入/检索） | 与 `local_store` 先例绑定的路径与建表 |
| `doc_maintenance_retrieval_gate.py` / `semantic_gate.py` | 188/214 | RAG 检索质量门 / 语义判定门 | fixture 与文档固定 |
| `doc_maintenance_cache.py` / `embeddings.py` / `suggestions.py` / `runtime.py` | 186/66/95/354 | 缓存、embedding、建议生成、运行配置 | 同上 |
| `scripts/doc_maintenance_audit.py` | 16 | 薄壳入口（`sys.path` 指向 `docs/agent_tool`） | 子项目化后改为 delegate |
| `scripts/check_doc_links.py`、`check_readme_l10n.py` | — | 链接/README 语言检查（独立脚本） | 可并入 docagent 作为规则插件 |
| `docs/文档维护Agent工具设计.md` | 275 | M1-M3 原始设计（分层架构、缓存、里程碑） | 作为设计依据保留 |

现有规则 R1-R5（等级 info/warn/error）：**R1 完成未收口、R2 未提交登记、R3 状态行滞后（git 粗关联）、R4 链接失效、R5 状态行缺失**；另有豁免词表（EXEMPT_HINTS）与"只看状态行主干"的防误报设计——这些都要进初始规则文档。

## 3. 子项目形态决策

- **形态**：独立 Python 包 `docagent/`（`pyproject.toml`，零运行时依赖或仅 stdlib + 可选 `pyyaml`；M2/M3 的 LLM/SQLite 依赖为**可选 extras**），CLI 入口 `docagent`。
- **目录**：先放本仓库 `tools/docagent/`（独立包边界清晰，不混 src/），主项目 `scripts/doc_maintenance_audit.py` 改为 thin delegate（`sys.path` 指向包或 pip 安装）。
- **与 harness 子项目平行**：两者都遵守"低耦合、可独立演进"，但无代码依赖；docagent 不消费 harness 的 OpenAI 兼容层（M2 判定自持 provider 抽象）。

```
docagent/
├─ pyproject.toml
├─ docagent/
│  ├─ cli.py            # docagent scan|rules|audit|init
│  ├─ scanner.py        # 规则引擎（读 rules.yaml，逐文档执行）
│  ├─ rules.py          # 规则 schema/加载/校验/版本
│  ├─ report.py         # text/json/markdown 输出
│  ├─ profiles.py       # 项目差异覆盖（docs_dir/exempt/词表/git 开关）
│  ├─ gitlink.py        # git 集成（脏文件/提交历史；可禁用）
│  ├─ llm.py            # [optional] M2 判定（provider 抽象保留）
│  └─ events.py         # [optional] M3 事件库/RAG（保留，默认不启用）
├─ rules/
│  ├─ RULES.md          # 初始规则文档（人读）
│  ├─ rules.yaml        # 机器可读规则集 v1（R1-R5 + 约定）
│  └─ profiles/
│     └─ qlh.yaml       # 本仓库 profile（docs 约定/豁免）
└─ tests/
```

## 4. 初始规则文档设计（规则即数据）

### 4.1 双层结构

- **`RULES.md`（人读）**：规则语义、判定依据、豁免原则、等级定义（error=CI 必拦 / warn=疑似需复核 / info=提示）、变更历史。本仓库约定直接搬入：状态行格式、创建/更新日期、适用范围、变更记录、相对链接、文档命名、术语表。
- **`rules.yaml`（机器读）**：每条规则结构化，schema 如下（v1 冻结）：

```yaml
rules_schema_version: 1
rule_set_version: 1.0.0
rules:
  - id: R1
    title: 完成未收口
    level: warn                    # error|warn|info
    scope: docs                    # docs|file|git
    params:
      status_head_lines: 12
      stale_hints: [进行中, 规划, 待, Candidate]
      done_marks: [完成, passed, Completed]
      exempt_hints: [故意不更新, 历史, 冻结]
      stem_only: true              # 只看状态行主干（去括号）
  - id: R4
    title: 链接失效
    level: warn
    scope: docs
    params:
      rel_links: true
      check_external: false        # v1 不查外链（网络副作用）
```

- **profile（项目差异）**：`rules/profiles/qlh.yaml` 覆盖 `docs_dir: docs`、`exclude: [local_docs]`、词表、`git_enabled: true`；其他项目用 `minimal` profile（只启用 R4/R5 与链接检查，无 git 关联）。

### 4.2 等价回归（行为不漂移的硬门）

P1 改造完成后：**旧代码（git HEAD 前）扫描结果 == 新引擎同参数扫描结果**（同一 git 快照、同 baseline），逐条 diff 必须为空。这是"规则数据化"的验收门槛，防止隐性行为变化。

## 5. 规则变更机械扫描工具（本期交付核心）

### 5.1 命令形态

```
docagent rules diff --old rules.yaml --new rules-next.yaml   # 规则变更差异（结构性）
docagent scan --root <repo> --rules rules-next.yaml \
        --baseline audit-baseline.json --dry-run             # 全库预演
```

### 5.2 违规增量矩阵（预演报告）

对全库用新规则集扫描，与基线（旧规则集结果或锁定快照）对比，输出：

| 维度 | 含义 | 用途 |
|---|---|---|
| `new` | 新规则新增命中的文档×规则 | 检测误报爆炸（新规则过严） |
| `gone` | 基线命中但新规则不再命中 | 检测静默漏检（规则被放宽/误删） |
| `changed` | 等级或 message 变化 | 检测语义漂移 |
| `affected_docs` | 受影响文档集合 | 评估人工复核范围 |

- **dry-run 默认**：预演绝不写入报告文件之外的任何状态（不动 docs、不动 git）。
- **阈值门**：`--max-new N` / `--max-gone M`（超限 exit 非零），用于 CI 化"规则变更门"。

### 5.3 基线锁定

`docagent audit --lock baseline.json` 在规则集 vN 上生成全量扫描快照（文档清单、命中明细、规则集指纹）；规则变更预演必须携带 baseline，否则拒绝生成"无对比报告"（防止凭空论断）。

## 6. 规则演进机制（将来：允许 agent 修改维护规则）

### 6.1 状态机

```
proposed ──preflight(§5)──> gates ──> approved ──> released
   ▲                           │
   └────── rejected + 理由 ─────┘
```

- **agent 允许做的事**：改 `rules.yaml` 数据（增/改/删规则、调等级、改词表）；**不允许**改 `scanner.py` / `rules.py` 等代码。
- **门控**（本期就实现，将来只放开 agent 入口）：
  - 新增 `error`/`warn` 级规则 → **人工批准**（误报风险高）；
  - 等级放宽/删除规则 → **人工批准**（静默漏检风险）；
  - 词表/参数微调且增量矩阵为"零新增、零消失"或仅 `changed` → 可自动放行（低风险）。
- **版本语义**：`rule_set_version` semver；`rules_schema_version` 与扫描器版本绑定（扫描器拒绝未知 schema，防旧引擎误跑新规则）。每条规则变更必须带 `change_note`（why），写入 `RULES.md` 变更历史。
- **回滚**：规则文件走 git 提交；回滚 = revert + 重扫（基线仍在，直接对比）。

### 6.2 防"agent 自己改规则自己过"的审计

- 门控输出（预演报告 + 批准记录）落到 `build/docagent-gates/`（或 CI artifact），含规则集指纹与（若启用 M3）事件库记录；
- 规则文件变更必须与"预演报告"同 PR/同提交，CI 校验报告指纹与规则指纹一致（防补报告）。

## 7. 分期计划

| 阶段 | 交付 | 验收 |
|---|---|---|
| **P0 基线固化**（本次） | 本文档 + 现状盘点 | 规则表/硬编码点清单与代码一致 |
| **P1 规则数据化** | `rules.yaml`（R1-R5+约定）+ 规则引擎重构（scanner 读数据） | **等价回归**：旧代码 vs 新引擎扫描结果逐条一致（§4.2）；专项测试通过 |
| **P2 子项目化** | `tools/docagent/` 独立包 + CLI（scan/rules/audit）+ 主项目薄壳 delegate | 从任意 cwd `docagent scan --root <repo>` 可用；主项目 `scripts/doc_maintenance_audit.py` 行为不变（同一回归 + 冒烟） |
| **P3 通用化** | profile 机制 + `--root`/`--profile`/`--json`/`--markdown`/`--fail-on` + CI 用法文档 | 在第二个项目（最小 profile）跑通 R4/R5；输出不含本仓库路径 |
| **P4 变更机械扫描** | `rules diff` + `scan --baseline --dry-run` + 增量矩阵 + 阈值门 + `audit --lock` | 人工构造"过严/放宽"规则变更，预演正确报出 new/gone；CI 门 exit 码正确 |
| **P5 规则演进门控（将来）** | 门控状态机 + agent 入口（LLM 生成规则草案）+ 预演报告指纹绑定 | 低风险规则变更自动通过；高风险被拒且留审计记录；回滚重扫一致 |

> 依赖：P1 依赖基线扫描记录可用（现有工具即可产）；P3 的"第二个项目"可先用本仓库 `tools/` 或 `docs/agent_tool` 自身做最小化验证（自举）。

### 7.1 开发票拆分（执行入口）

原有 P0-P5 是阶段门，不直接作为单次开发任务。后续以以下开发票推进；每票只改变工具边界，不把文档内容自动改写混入扫描器：

| 票 | 所属阶段 | 内容 | 依赖 | 开发门 |
|---|---|---|---|---|
| `DOCAGENT-P0` | P0 | 现状盘点、规则表/硬编码点清单、子项目边界冻结 | 无 | 本文档与原始设计一致；已完成 |
| `DOCAGENT-P1A` | P1 | `rules.yaml`/`RULES.md` v1 schema、版本校验、规则加载 API；搬入 R1-R5 初始参数 | P0 | 旧规则字段可完整表达；未知 schema fail-closed |
| `DOCAGENT-P1B` | P1 | scanner 改为规则数据驱动，保留 R1-R5 行为适配层 | P1A | 同一 git 快照旧/新逐条 finding 等价；专项回归 |
| `DOCAGENT-P2A` | P2 | `tools/docagent` 独立包骨架、`pyproject.toml`、stdlib core、CLI `scan/rules/audit/init` | P1B | 任意 cwd 指定 `--root` 可运行；不写目标文档 |
| `DOCAGENT-P2B` | P2 | 主项目脚本 thin delegate、输出格式和退出码兼容层 | P2A | 旧入口与新 CLI 同一报告摘要、失败级别一致 |
| `DOCAGENT-P3A` | P3 | profile schema 与 qlh/minimal profile、docs_dir/exclude/词表/git 开关 | P2A | 本仓库和第二个最小项目均能扫描；输出无绝对路径 |
| `DOCAGENT-P3B` | P3 | `--json/--markdown/--fail-on`、配置错误诊断、CI 使用文档 | P3A | 格式稳定；warn/error 门 exit code 可断言 |
| `DOCAGENT-P4A` | P4 | baseline lock、规则指纹、`rules diff` 结构差异 | P3B | 缺 baseline 或指纹不符时拒绝比较 |
| `DOCAGENT-P4B` | P4 | dry-run 增量矩阵（new/gone/changed/affected_docs）与 max-new/max-gone 门 | P4A | 人工过严/放宽样例矩阵正确；dry-run 不改状态 |
| `DOCAGENT-P5A` | P5 | 规则演进状态机、change_note、审批记录和 agent 规则草案入口 | P4B | 低风险自动放行；新增/删除 warn/error 必须人工批准 |
| `DOCAGENT-P5B` | P5 | 规则/报告指纹绑定、回滚重扫、审计事件适配（M3 可选） | P5A | 篡改报告或规则后门控失败；revert 后可重建一致 |

**当前执行顺序**：`DOCAGENT-P1A → P1B → P2A → P2B → P3A → P3B → P4A → P4B → P5A → P5B`。P1A/P1B/P2A/P3A 不依赖 GPU、网络、LLM 或第二台设备，可连续开发；M2/M3 provider 只在对应可选票中接入，不阻塞 core。

**当前票状态**：`DOCAGENT-P5B` 已完成：扫描报告新增完整内容 `report_fingerprint`，新增 `gate` artifact 与 `gate verify` 规则/报告/基线/演进绑定门；报告或规则文件篡改会 fail-closed，`gate rescan` 提供 revert 后基线重扫路径；新增独立可选 M3 SQLite 事件适配器，默认扫描仍不依赖 SQLite。P5 规则演进门控阶段完成。

## 8. 风险与边界

1. **行为漂移**（P1 最大风险）：规则从代码搬 YAML 时语义走样 → 等价回归硬门 + 全量 baseline 对照。
2. **规则变更误伤**：预演新增命中爆炸 → `--max-new` 门 + 新增 warn/error 规则人工批准。
3. **静默漏检**：删除/放宽规则后没人发现 → `--max-gone` 门 + 基线锁定 + 变更报告必须带 change_note。
4. **agent 越权改代码**：规则引擎代码与规则数据物理分离（规则只读 YAML），CLI 校验 `rules_schema_version`；agent 修改范围由门控约束。
5. **M2/M3 通用化成本**：LLM provider 与 SQLite 事件库是本仓库定制较多部分 → 留在可选 extras，默认不进 core，不承诺跨项目开箱即用。
6. **范围边界**：不做文档内容自动改写、不做语义总结生成、不做跨项目内容风格统一；`check_readme_l10n.py` 等语言检查是否并入由 P3 评估。

## 9. 与既有文档/子项目关系

- **M1-M3 原始设计**（[文档维护 Agent 工具设计](文档维护Agent工具设计.md)）：本计划是其实施升级版——M1 规则数据化、M2/M3 保留为可选模块，分层架构与缓存设计沿用，不推翻。
- **harness 子项目**（[小模型轻量推理harness工作台调研与方案](小模型轻量推理harness工作台调研与方案.md)）：平行子项目，无代码依赖；docagent 不走 OpenAI 兼容层，M2 自持 provider。
- **WEB-TOOL 支线**：无交集（联网工具服务于运行时 Agent，docagent 服务于开发期文档）。

## 10. 变更记录

- 2026-09-08：初版（现状盘点 + 子项目化方案 + 初始规则文档设计 + 变更机械扫描工具 + P0-P5 计划）
- 2026-09-08：v2 —— 将 P0-P5 阶段拆为 `DOCAGENT-P0` 至 `DOCAGENT-P5B` 十一张开发票，冻结依赖顺序、core 与可选 provider 的边界，以及每票开发门；下一票为 `DOCAGENT-P1A`。
- 2026-09-09：完成 `DOCAGENT-P1A` 本机开发门：规则数据文件、规则人读说明、stdlib 优先的加载/校验/指纹 API 与 6 项专项测试；下一票切换为 `DOCAGENT-P1B`。
- 2026-09-09：完成 `DOCAGENT-P1B` 本机开发门：scanner 改为读取规则数据，审计报告绑定规则集指纹，新增默认等价与规则参数生效回归；下一票切换为 `DOCAGENT-P2A`。
- 2026-09-09：完成 `DOCAGENT-P2A` 本机开发门：新增独立 `tools/docagent` 包、stdlib core、规则合同副本、`scan/rules/audit/init` CLI、`pyproject.toml` 和外部 cwd 回归；下一票切换为 `DOCAGENT-P2B`。
- 2026-09-09：完成 `DOCAGENT-P2B` 本机开发门：主项目 M1 入口改为独立包兼容委托，保留 `build/doc-audit` 输出、退出码和 M2/M3 扩展分流；新增旧/新逐文档 finding 等价回归；下一票切换为 `DOCAGENT-P3A`。
- 2026-09-09：完成 `DOCAGENT-P3A` 本机开发门：新增 profile schema、`qlh`/`minimal` 内置 profile、项目级 `.docagent/profile.yaml`、`docs_dir`/递归扫描/exclude/词表/Git 开关，以及跨 cwd、嵌套文档和绝对路径泄漏回归；下一票切换为 `DOCAGENT-P3B`。
- 2026-09-09：完成 `DOCAGENT-P3B` 本机开发门：新增统一 JSON/Markdown/text renderer、`--markdown` 与格式互斥约束、按格式写出报告、配置错误诊断和 `tools/docagent/CI.md`；新增 15 项输出/门控/错误回归，确认 JSON stdout 可直接被 CI 解析；下一票切换为 `DOCAGENT-P4A`。
- 2026-09-09：完成 `DOCAGENT-P4A` 本机开发门：新增 baseline schema、`audit --lock`、`--baseline`/`--dry-run` 校验、rules fingerprint/profile 匹配门和 `rules diff` 结构化字段差异；新增 48 项相关回归，并验证 baseline 与 diff 输出不含绝对路径；下一票切换为 `DOCAGENT-P4B`。
- 2026-09-09：完成 `DOCAGENT-P4B` 本机开发门：新增 baseline dry-run 增量矩阵 `new/gone/changed/affected_docs`、文档 hash/新增/删除明细和 `--max-new`/`--max-gone` 阈值门；新增 49 项相关回归，确认 dry-run 不改 baseline 且规则/profile 指纹不匹配仍拒绝比较；下一票切换为 `DOCAGENT-P5A`。
- 2026-09-09：完成 `DOCAGENT-P5A` 本机开发门：新增 `rules evolve` 状态机与 `qlh.docagent.evolution.v1`/`qlh.docagent.approval.v1` 合同，强制 `change_note`，绑定候选规则集指纹与人工审批，低风险变更自动放行，新增/删除 `warn/error` 规则和等级变化 fail-closed；新增 4 项专项回归并更新 README/CI 用法；下一票切换为 `DOCAGENT-P5B`。
- 2026-09-09：完成 `DOCAGENT-P5B` 本机开发门：扫描报告加入自指纹，新增 `qlh.docagent.gate.v1` / `gate verify` 的规则、报告、profile、baseline、evolution 绑定，新增 `gate rescan` 回滚重扫入口与可选 `qlh.docagent.events.v1` SQLite M3 事件适配；篡改报告/规则/门禁 artifact 均被拒绝，revert 后可重建零差异；新增 4 项专项回归，P5 阶段完成。
