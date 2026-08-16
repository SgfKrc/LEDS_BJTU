# 文档维护 Agent 工具设计

> 状态：实施中（**M1 已复核收口，M2.1-M2.4 已完成并通过 opencode 实网传输验收；M3.1-M3.4 开发票完成，M2 语义基准与 M3 embedding 质量待后置验收**）
>
> 更新日期：2026-08-16
> 适用范围：仓库 `docs/` 文档元数据一致性维护（状态行、完成登记、链接、提交记录）；不替代文档内容创作、不替代人工审核、不改变"git 是唯一事实源"的纪律
>
> 立项背景：2026-08-16 批量筛查 46 份文档发现 4 类遗漏——① 完成提交但头部状态行未收口（SD 离线包阶段 2）；② 整个实现+登记未提交（TG-OPT-G1）；③ 状态行过时（自动化实验"仍受资源门阻塞"）；④ 正文完成但状态行未同步（答辩演示 P5-P8）。此类问题靠人工周期性排查成本高、易漏。

## 1. 目标与不做什么

### 目标
1. **机械化扫描**（M1，零 LLM）：自动扫描文档头状态行与 git 提交记录/正文完成标记/工作区未提交改动的矛盾，产出**人工核对清单**，把"发现遗漏"从人工巡检变成脚本输出。
2. **智能化判定**（M2，本地 LLM）：对机械化信号做语义判定（状态行措辞是否准确描述现状、哪些"疑似"是误报），分级输出（过时/准确/需人工）。
3. **知识库与 RAG**（M3，远期）：SQLite 记录文档变更事件与元数据，embedding 检索辅助"哪些文档受某次改动影响"。

### 不做什么（边界）
- **不自动改写文档**：M1-M3 全部输出核对清单与建议 diff，最终落笔必须人工确认（或显式 `--apply` + 生成 diff 走 git 评审）；避免"agent 自己改自己"的循环污染。
- **不替代人工审核**：LLM 判定只是排序与预筛，结论置信度低于阈值的全部转人工。
- **不重复造轮子**：与既有 `scripts/check_links.py`、`文档状态与清理清单.md`、`已知问题记录.md` 分工——本工具管**元数据一致性**，不接管内容清理与问题登记（发现的问题仍按既有渠道登记）。
- **不做运行时接入**：这是开发期维护工具，不进入安装包、不依赖 GPU/显存门（LLM 判定走 Ollama 可选，缺失时机械化部分照常工作）。

## 2. 分层架构

```text
┌─ L0 机械化扫描器（纯 Python 标准库 + git CLI，无外部依赖）────────┐
│   信号采集：文档头状态行 / 正文完成标记 / git log 提交 / 工作区状态    │
│   矛盾检测：状态行 vs 提交记录 / 状态行 vs 正文 / 未提交登记检测        │
│   输出：核对清单（markdown + JSON），可进 CI 或手动入口               │
└──────────────────────────────────────────────────────────────┘
                         │ 疑似项（结构化 JSON）
┌─ L1 本地 LLM 判定（可选，Ollama）──────────────────────────────┐
│   输入：疑似项 + 相关文档片段 + 相关提交摘要（脱敏，不含密钥）        │
│   输出：判定（过时/准确/需人工）+ 置信度 + 建议修正（建议 diff）       │
│   fail-closed：LLM 不可用/超时/输出非法 → 降级为"需人工"             │
└──────────────────────────────────────────────────────────────┘
                         │ 核对清单（分级排序）
┌─ M3 事件库（远期，SQLite + RAG）──────────────────────────────┐
│   docs_meta / doc_events / check_runs 表；embedding 索引；        │
│   "某改动影响哪些文档"检索；历史核对结论复用                         │
└──────────────────────────────────────────────────────────────┘
```

## 3. M1 机械化扫描器（近期实施）

### 3.1 信号采集（全部只读）
| 信号 | 来源 | 说明 |
|---|---|---|
| 状态行 | 每份 `docs/*.md` 前 12 行含"状态"的行（兼容 `状态：` / `**状态**：`，跳过"文档生命周期"） | 记录原文供人工核对 |
| 正文完成标记 | 前 200 行内 `✅ / Completed / 已完成 / 已关闭 / 已验收 / 开发门完成` | 与状态行对比 |
| 提交记录 | `git log --name-only -- src/` | R3 提供关联代码路径、提交 hash 与日期 |
| 工作区状态 | `git status --short docs/` | 只登记**当前文档**的未提交改动，避免全库重复告警 |
| 代码改动 | `git log --since=<文档更新时间> -- src/`（可选） | 仅按文档文件名/标题 token 或正文显式 `src/` 引用做路径粗关联；无关联线索则不猜测 |

### 3.2 矛盾检测规则（规则表，命中即列为"疑似"）
1. **完成未收口**：正文含完成标记，但状态行主干含 `规划 / Candidate / Blocked`，且状态行自身未声明完成（人工判定是否该升级）。`进行中/实施中/待` 允许已完成子阶段，故不作为机械化命中词；历史参考类文档按豁免词跳过。
2. **未提交登记**：`git status` 显示**该 `docs/*.md`** 有改动（文档写了完成记录但代码/登记没提交）。
3. **状态行滞后于代码**：文档"更新日期"早于 `git log -1 -- src/` 中相关模块的提交（需主题关联，M1 用目录名/文件名前缀粗关联，标"需人工确认"）。
4. **链接失效**：内置检查 `docs/` 内 Markdown 相对链接的目标存在性（外链与页内锚点跳过）。
5. **状态行缺失**：前 12 行无"状态"行（不是所有文档必须有，标"建议补充"）。

### 3.3 输出
- `docs/维护核对清单.md`（或 `build/doc-audit/` 下）：按文档分组的疑似项表 + 证据（引用原文行、commit hash、git 状态行）。
- `build/doc-audit/audit.json`：结构化结果（文档、信号、规则命中、证据路径），供 L1 消费或 CI 断言（`--fail-on <规则>` 可配）。

### 3.4 入口
```bash
python scripts/doc_maintenance_audit.py            # 全量扫描，输出清单
python scripts/doc_maintenance_audit.py --json     # 结构化输出
python scripts/doc_maintenance_audit.py --since 7d # 只看近 7 天有改动的文档
python scripts/doc_maintenance_audit.py --fail-on R4 # 指定规则命中时供 CI 返回非零
```
- 手动入口为主；CI 选配（只读，无副作用）。
- 实现位于 `docs/agent_tool/`，`scripts/doc_maintenance_audit.py` 是稳定 CLI 入口；输出固定写入已忽略的 `build/doc-audit/`。

## 4. M2 本地/远程 LLM 判定（中期）

### 4.1 Provider 抽象（远程 DeepSeek V4 Flash 优先 + 本地 Ollama 兜底）

判定引擎通过 OpenAI 兼容接口抽象 provider，支持两种后端。**默认优先走 opencode go 套餐的远程通道**（本地 Ollama 模型如 gemma4:12b 上下文窗口有限，判题输入超限会截断误判；远程 DS V4 Flash 上下文充足），Ollama 作为无密钥/离线环境兜底：

| Provider | 用途 | 接入方式 | 说明 |
|---|---|---|---|
| **远程 opencode go 网关 / DeepSeek V4 Flash**（**默认优先**） | 主判定通道 | OpenAI 兼容 `base_url=https://opencode.ai/zen/go/v1`（完整端点 `/chat/completions`）+ `sk-*` 密钥 | 上下文充足、判定质量高；**判题文本会外发**——只允许发送文档片段与提交标题，且必须在独立 env 中显式配置密钥才启用 |
| **本地 Ollama**（兜底） | 离线/无密钥环境（`gemma4:12b`、`qwen3:4b` 等） | `http://127.0.0.1:11434/v1`，无需密钥 | 无外发数据、无网络依赖；Ollama 未运行或上下文不足时自动降级为"需人工" |

配置独立于主 `.env`，放在 **`.env.docagent`**（已入 `.gitignore`，随 `.env` 一行忽略规则覆盖）：

```bash
# .env.docagent（示例；sk 不落 git）
DOCAGENT_PROVIDER=opencode        # opencode（默认优先，需完整配置）| ollama；deepseek 为兼容别名
DOCAGENT_DEEPSEEK_BASE_URL=https://opencode.ai/zen/go/v1
DOCAGENT_DEEPSEEK_MODEL=deepseek-v4-flash   # 以 opencode go 套餐实际模型名为准
DOCAGENT_DEEPSEEK_API_KEY=sk-xxx
DOCAGENT_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
DOCAGENT_OLLAMA_MODEL=gemma4:12b  # 或 qwen3:4b
DOCAGENT_CONFIDENCE_FLOOR=0.6     # 低于此值一律 needs_review
```

- 读取规则：`--provider` 显式指定 > `.env.docagent` 的 `DOCAGENT_PROVIDER` > 默认 deepseek；远程通道**必须**有 base_url+api_key 且 `DOCAGENT_PROVIDER=opencode`（或旧别名 `deepseek`）才启用（fail-closed：配置缺失自动回退 ollama；ollama 也不可用则跳过 LLM 判定，机械化清单照常输出）。
- 密钥纪律：sk 只从 `.env.docagent` 读取，**不进入**命令行参数、日志、audit.json 与任何输出；`--llm` 输出只含判定与建议文本。
- 数据边界：无论哪个 provider，发送内容仅限「疑似项 + 文档头部/相关段落（≤4KB/项）+ 提交标题」；正文、密钥、路径、URL、blob、grant 类字段一律不发送（沿用投影脱敏白名单思路）。

### 4.2 输入与输出
- 输入：audit.json 的疑似项 + 文档头部/相关段落（截断到 ≤4KB/项）+ 相关提交 message 列表；**脱敏**：不传 .env 值、密钥、日志正文，仅传提交标题与文档文本。
- 输出（JSON）：`{"doc": ..., "judgement": "stale|accurate|needs_review", "confidence": 0-1, "suggestion": "建议的状态行文本或说明"}`。

### 4.3 判定协议（防误报）
- 只允许三种判定；`confidence < 0.6` 一律 `needs_review`。
- `suggestion` 仅作为建议 diff 呈现，不自动应用。
- 校验输出 schema，非法 JSON 视为 `needs_review`（fail-closed）。

### 4.4 M2 分票与当前边界

| 票号 | 交付 | 状态 | 边界 |
|---|---|---|---|
| **M2.1** | `doc_maintenance_llm.py` 的独立配置读取、DeepSeek→Ollama 计划解析、匿名 `doc_ref`、发送前脱敏、单文档批处理、canonical hash 与严格 JSON 判定 | ✅ 2026-08-16 | **零网络调用**；不读取主 `.env` 或进程环境；provider 输入不含本地路径、URL、密钥、链接目标或原始 finding 文本 |
| **M2.2** | `build/docagent-cache.sqlite` 的 L1/L2 表、懒失效、人工确认回写、`cost_log` | ✅ 2026-08-16 | 纯本地 SQLite；完全相同输入的 L1 复用合法 `needs_review`，输入变化后的 L2 不复用；`human` 结论优先，文档/提交指纹变化后自然 miss |
| **M2.3** | `--llm`、OpenAI 兼容 HTTP client、DeepSeek→Ollama→跳过回退、超时与用量采集 | ✅ 2026-08-16 | 仅以 loopback fake provider 测试；真实远程调用留到用户显式配置后验收；失败类别不回显 URL/响应体 |
| **M2.4** | 建议 diff 生成与人工确认导入 | ✅ 2026-08-16 | `--apply` **只写** `build/doc-audit/suggestions.patch` 与 manifest；不写源文档、不执行 `git apply`、不提交 |

M2.1 的 provider 输入只使用匿名文档引用；返回协议中 `doc` 同样必须回显该 `doc_ref`，本地报告阶段才映射回实际相对路径。confidence 低于阈值、非法 JSON、字段额外/缺失或 `doc_ref` 不一致，全部降级为 `needs_review`。

### 4.5 缓存与省钱设计（远程按量计费的关键）

opencode go 远程通道按量计费，缓存命中直接决定成本。设计**三级缓存**，全部落本地 SQLite（M2 即可用独立 `build/docagent-cache.sqlite`，M3 并入事件库）：

#### 4.5.1 三级缓存
| 级 | 机制 | 命中条件 | 成本 |
|---|---|---|---|
| **L1 精确缓存** | 输入 canonical hash（疑似项 + 文档片段截断 + 相关提交标题，规范化排序去空白）→ 上次判定 | 同一文档同一疑似项，输入指纹一致 | 零（本地查表） |
| **L2 状态缓存** | 缓存键 = 文档 sha256 + 相关 commit 列表 + 规则 ID；键未变且上次判定非 `needs_review` | **文档与提交都没变**时重扫/人工核对后重跑 | 零 |
| **L3 语义缓存**（远期，M3 向量） | embedding 相似度 ≥ 阈值时复用"参考建议" | 相似状态行/相似规则模式 | 零（仅本地向量检索） |

**失效策略（懒失效）**：缓存条目记录 `doc_sha256` 与 `related_commits`；每次命中先比对当前值，变了则标记 stale 不命中（**提交后自然失效**，但同一提交状态下的重复扫描/核对重跑全部命中）。

#### 4.5.2 提高命中率的输入设计（省钱核心）
1. **输入规范化**：文档片段固定截断窗口（≤4KB，从头取）、列表排序、空白归一 → 同一状态行在多次扫描中产生**相同指纹**（不做规范化则时间戳/行号会让缓存永远 miss）。
2. **批量合并**：同一文档的多个疑似项**合并为一次调用**（一条 prompt 判多项），减少调用次数；合并键 = 文档 sha256，任一疑似项变化才 miss 该批。
3. **模式去重**：相同规则 + 相似状态行措辞（如多个文档都是"完成未收口"模式）→ 先按模板分类（规则 ID + 状态行关键词聚类），同类只抽样判定一次，其余复用模板结论并标注"模板复用"。
4. **人工确认回写**：人工核对后的最终结论写入缓存（`source=human` 优先于 `source=llm`），后续命中直接用人工结论——**人工核对一次，终身复用**。
5. **预检跳过**：扫描时先比对缓存键，命中且上次判定 `accurate` 的文档**完全跳过 LLM 调用**（连批量合并都不做）。

#### 4.5.3 缓存表结构（M2 落地版）
```sql
CREATE TABLE llm_judgements (
  cache_key TEXT PRIMARY KEY,        -- sha256(canonical input)
  doc_id TEXT, rule_id TEXT,
  doc_sha256 TEXT, related_commits TEXT,  -- 懒失效依据
  judgement TEXT, confidence REAL, suggestion TEXT,
  source TEXT,                        -- llm | human | semantic
  provider TEXT, model TEXT,
  prompt_tokens INT, completion_tokens INT,  -- 成本核算
  judged_at TEXT
);
CREATE TABLE cost_log (              -- 每次远程调用一条
  run_id TEXT, ts TEXT, provider TEXT, model TEXT,
  prompt_tokens INT, completion_tokens INT,
  hits INT, misses INT
);
```
- **安全**：缓存只存判定结果 + 脱敏输入片段（≤4KB 文档头）；sk、正文、密钥类字段**永不落缓存**；`--llm` 输出报告可加 `--cost` 显示"本次扫描命中 N 次、远程调用 M 次、估算 tokens"。

#### 4.5.4 预期效果（验收基准）
- 同一提交状态下连续两次扫描：第二次 LLM 调用数 ≈ 0（L1/L2 全命中）。
- 人工核对后的重扫：被核对文档不再产生远程调用（source=human 命中）。
- 提交一次新 commit 后：仅该 commit 涉及的文档重新调用（其余仍命中）。

### 4.6 使用流程
```bash
python scripts/doc_maintenance_audit.py --llm                # 机械化 + LLM 分级（默认 deepseek/opencode go，需 .env.docagent 密钥）
python scripts/doc_maintenance_audit.py --llm --provider ollama    # 本地 Ollama（离线/无密钥环境）
python scripts/doc_maintenance_audit.py --llm --apply              # 仅生成建议 diff（不改文档、不提交）
```

## 5. M3 事件库与 RAG（远期）

### 5.1 SQLite schema 草案（沿用项目 local_store 的 sqlite 先例）
```sql
CREATE TABLE doc_meta (
  doc_id TEXT PRIMARY KEY,          -- docs/ 相对路径
  title TEXT, status_line TEXT, updated_at TEXT,
  last_commit TEXT, last_commit_ts TEXT,
  sha256 TEXT,                      -- 当前内容哈希
  indexed_at TEXT
);
CREATE TABLE doc_events (           -- 每次扫描/修改的事件
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT, ts TEXT, kind TEXT,  -- scan|manual_edit|llm_suggestion|applied
  payload TEXT                      -- 结构化变更说明（JSON）
);
CREATE TABLE check_runs (           -- 核对历史，避免重复人工核对
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, rules TEXT, findings TEXT, decisions TEXT
);
-- RAG（远期）：doc_chunks(doc_id, chunk_no, text) + doc_embeddings(chunk_id, model, dim, vec BLOB)
```
- 事件表与 git 的关系：git 是事实源，`doc_events` 只是检索/审计索引；重建 = 从 git log 重放（脚本提供 `--rebuild`）。

### 5.1.1 M3 分票与边界

| 票号 | 交付 | 状态 | 边界 |
|---|---|---|---|
| **M3.1** | `build/docagent-events.sqlite`，`doc_meta` / `doc_events` / `check_runs` 初始化，M1 快照索引与 `--index` | ✅ 2026-08-17 | 只写本地 `build/`；实仓索引 57 份文档、57 条 scan 事件、1 次 check run，不修改 `docs/` |
| **M3.2** | `--rebuild` 从 git 历史重建事件索引、重建前备份与可重复性校验 | ✅ 2026-08-17 | 实仓重放 251 个提交、595 条历史事件；临时库成功后原子替换，保留 `.bak-*` 旧库 |
| **M3.3** | `doc_chunks` 与本地 FTS 检索 CLI，先提供无 embedding 的关键词候选 | ✅ 2026-08-17 | 实仓 57 份文档 / 798 chunks，`task graph` 查询返回 5 个候选；无网络、无 GPU；检索只生成核对候选 |
| **M3.4** | Ollama embedding adapter、`doc_embeddings` 与 top-5 语义检索 | 开发票 ✅ 2026-08-17 | fake provider 完成增量/替换/cosine 测试；实仓无 Ollama 时安全降级，真实 embedding 质量与模型可用性后验收；FTS 不受影响 |

M3.1 的索引是派生数据：重复执行只新增本次 `check_runs`/扫描审计事件，`doc_meta` 始终代表当前快照；数据库可随时删除并从仓库重建。

```bash
python scripts/doc_maintenance_audit.py --index --json --fail-on none
python scripts/doc_maintenance_audit.py --rebuild --json --fail-on none
python scripts/doc_maintenance_audit.py --index-chunks --search "task graph" --search-limit 5
python scripts/doc_maintenance_audit.py --embed --semantic-search "慢网任务调度"
```

`--embed` 和 `--semantic-search` 只显式调用本机 Ollama；provider 不可用时报告降级，不影响 M1、事件库或 FTS。

### 5.2 embedding 选型
- Ollama 本地模型（用户侧已具备 Ollama 环境；具体模型待实测确认，候选 `nomic-embed-text` / `bge-m3`，以检索质量与体积权衡）。
- 用途：① "这次 src 改动影响哪些文档"（文档↔代码主题相似度）；② 核对清单历史问题聚类（同类遗漏合并处理）；③ 新文档建立时找"最相关的既有文档"防重复立项。
- 不做：不把 RAG 输出当成事实，检索结果只进人工核对清单。

### 5.3 检索场景（远期验收）
- 输入一段变更描述（如"task_graph 新增 shadow 优化器"）→ 输出应关联的文档清单（top-5）与理由。
- 历史核对：某文档"最近 3 次核对结论"一键回看，避免重复判定同一状态行。

## 6. 里程碑与验收

| 里程碑 | 交付 | 验收口径 |
|---|---|---|
| **M1 机械化扫描器** | `scripts/doc_maintenance_audit.py` + 规则表 + 清单模板 | ✅ 2026-08-16 复核：54 份文档全量扫描 0.76s；20 项定向测试通过；R2 仅命中实际改动文档、R3 仅按 `src/` 路径粗关联；Windows GBK 父终端下 CLI 强制 UTF-8 输出；零改写 `docs/` 内容 |
| **M2 LLM 判定** | `--llm` 路径 + 判定协议 + 脱敏检查 + **三级缓存**（L1/L2 本地 SQLite，L3 随 M3 接入） | 开发票 + 实网传输 ✅：opencode 4/4 返回合法 JSON，低置信度均转人工；待语义验收：人工冻结 4 个带结论的样本后，判定一致率 ≥ 3/4；远程缺失时回退 Ollama、均不可用时机械化照常；同提交重扫远程调用 0、人工结论重扫 0、新提交仅关联文档重调 |
| **M3 事件库 + RAG** | SQLite 三表 + embedding 索引 + 检索 CLI | 变更描述→关联文档 top-5 命中率 ≥ 60%（30 条抽样人工标注）；`--rebuild` 从 git log 重建与 git 一致 |
| 收口 | 全量核对清单清零（或逐项登记为已知/有意） | 2026-08-16 发现的 4 类问题模式在后续扫描中不再以"未发现"状态存在 |

## 7. 依赖与风险

| 项 | 说明 |
|---|---|
| 依赖 | M1：Python 3.10+ 标准库 + git CLI（本机已具备）；M2：Ollama（可选，已具备）；M3：sqlite3（已有先例） |
| 磁盘 | 极小（脚本 + 索引库 <50MB）；RAG 向量库另计 |
| 冲突边界 | 全程只读/建议；`--apply` 只生成 diff 不提交；不碰业务 `src/`；维护工具的专属单测限 `tests/test_doc_maintenance_*.py`；与并行组工作区无交集（实现限 `docs/agent_tool/`、稳定入口限 `scripts/doc_maintenance_audit.py`，生成物落已忽略的 `build/`） |
| 风险 | ① 状态行"故意不更新"（如历史参考文档）→ 规则 1 需"已声明历史"豁免词表（`历史参考/待拆分/不作为当前能力`）；② LLM 误判 → 置信度门 + 人工兜底；③ 过度设计 → M1 先落地解决眼前痛点，M2/M3 按需推进，不提前实现 |
| 与既有工具分工 | `check_links.py`（链接）、`文档状态与清理清单.md`（内容清理）、`已知问题记录.md`（问题登记）均不接管；本工具只管"状态行/登记/提交的一致性" |

## 8. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-16 | 建立本文档（M1-M3 设计基线）；立项背景为当日批量筛查发现的 4 类遗漏 |
| 2026-08-16 | M2 扩展为双 provider：本地 Ollama（GPU/CPU）默认 + 远程 DeepSeek V4 Flash（opencode go 套餐）可选；配置独立 `.env.docagent`（入 gitignore），远程通道 fail-closed，密钥不落任何输出 |
| 2026-08-16 | M2 优先级调整：**opencode go 远程通道（`https://opencode.ai/zen/go/v1`）默认优先**（本地 Ollama 上下文窗口有限易截断误判），Ollama 降为离线/无密钥兜底；回退链 deepseek→ollama→跳过 |
| 2026-08-16 | M2 新增 §4.5 缓存与省钱设计：三级缓存（精确/状态/语义）+ 输入规范化、批量合并、模式去重、人工确认回写、预检跳过；`llm_judgements`/`cost_log` 表；验收基准（同状态重扫零调用） |
| 2026-08-16 | 复核并收口并行组 M1：修正 R2 全库重复告警、R3 误用文档自身提交、无效 `--fail-on` 与 Windows GBK stdout 崩溃；补稳定 `scripts/` 入口，20 项 M1 测试与真实全量扫描通过 |
| 2026-08-16 | 完成 M2.1（零网络）：独立 `.env.docagent` 读取与 fail-closed provider 计划、匿名批次、路径/URL/密钥脱敏、稳定 cache key、严格 schema + confidence 降级；13 项定向测试通过 |
| 2026-08-16 | 完成 M2.2：本地 SQLite L1/L2 缓存、`human` 优先、懒失效与 `cost_log`；不存密钥或原始正文，6 项定向测试通过 |
| 2026-08-16 | 完成 M2.3：`--llm` 与标准库 OpenAI 兼容 client，DeepSeek→Ollama→人工回退，20-60s 有界超时、用量记录；以 loopback fake provider 验证脱敏载荷、缓存命中和回退，不访问真实远程服务 |
| 2026-08-16 | 完成 M2.4：仅生成可 `git apply --check` 的建议 patch，含糊建议跳过；源文档不写入，5 项定向测试通过 |
| 2026-08-17 | 修复 M2 配置兼容：支持用户已配置的 `DOCAGENT_PROVIDER=opencode`，映射既有 `DOCAGENT_DEEPSEEK_*` 网关字段；保留 `deepseek` 兼容别名，并支持未加引号的行尾 `#` 注释 |
| 2026-08-17 | M2 真实验收前复核 M1 输入，修复 Markdown 加粗状态标签 `**状态**：` 被误报为 R5 的解析缺口 |
| 2026-08-17 | M2 真实网关诊断：opencode 在 Python 默认请求头下返回 HTTP 403；加入固定、无用户信息的 `User-Agent: QLH-DocAgent/0.1` 后同一匿名批次返回 HTTP 200，认证与模型配置可用 |
| 2026-08-17 | M2 实网受控验收：4 个匿名文档批次全部由 opencode 返回合法 JSON，未降级到 Ollama；2 个 R1 与 README 因 confidence 0.3/0.5/0.4 降级 `needs_review`，PyTorch 生命周期场景以 0.8 判定 `accurate`。缺少人工冻结标签，语义一致率门保留待验收 |
| 2026-08-17 | M3 分票收敛为本地优先路径：M3.1 当前快照事件库、M3.2 git 重建、M3.3 FTS、M3.4 可选本地 embedding；不以模型/硬件可用性阻塞前 3 票开发 |
| 2026-08-17 | M2 实网重扫修复缓存语义：合法低置信度 `needs_review` 现在可被完全相同输入的 L1 精确缓存复用；L2 状态缓存仍拒绝复用，避免输入变化后沿用不确定结论 |
| 2026-08-17 | 完成 M3.1：新增 `docagent-events.sqlite` 三表、`--index` 当前快照入口、最近 git 提交索引、scan/LLM 事件与人工 decisions 回写；实仓 57 份文档索引耗时 3.24s |
| 2026-08-17 | 完成 M3.2：`--rebuild` 正序重放 git 的 `docs/*.md` 历史为 `manual_edit` 事件，重建前备份旧库、临时库完成后 `os.replace`；实仓 251 提交 / 595 历史事件，耗时 3.53s |
| 2026-08-17 | 完成 M3.3：新增确定性 Markdown chunks、SQLite FTS5、`--index-chunks` 与 `--search`；实仓生成 798 chunks，FTS 查询耗时 1.12s |
| 2026-08-17 | 完成 M3.4 开发票：新增 Ollama `/api/embed` 适配、float32 向量存储、增量 embedding 与 cosine top-N；fake provider 35 项组合回归通过，实仓无 Ollama 时机械/FTS 路径保持可用 |
