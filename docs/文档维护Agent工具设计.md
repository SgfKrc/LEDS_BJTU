# 文档维护 Agent 工具设计

> 状态：规划（M1 未开始；本页为设计基线）
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
| 状态行 | 每份 `docs/*.md` 前 12 行含"状态"的行（跳过"文档生命周期"） | 记录原文供人工核对 |
| 正文完成标记 | 前 200 行内 `✅ / Completed / 已完成 / 已关闭 / 已验收 / 开发门完成` | 与状态行对比 |
| 提交记录 | `git log -1 -- <doc>` | 文档最近改动 |
| 工作区状态 | `git status --short docs/` | 未提交的文档改动（登记未落盘） |
| 代码改动 | `git log --since=<文档更新时间> -- src/`（可选） | 状态行"更新日期"之后 src 是否有大改 |

### 3.2 矛盾检测规则（规则表，命中即列为"疑似"）
1. **完成未收口**：正文/提交含完成标记，但状态行含 `进行中 / 规划 / 待 / Candidate`（人工判定该状态行是否该升级）——本次筛查发现 4 例的同一模式。
2. **未提交登记**：`git status` 显示 `docs/*.md` 有改动（文档写了完成记录但代码/登记没提交）。
3. **状态行滞后于代码**：文档"更新日期"早于 `git log -1 -- src/` 中相关模块的提交（需主题关联，M1 用目录名/文件名前缀粗关联，标"需人工确认"）。
4. **链接失效**：复用既有 `scripts/check_links.py` 结果合并进清单。
5. **状态行缺失**：前 12 行无"状态"行（不是所有文档必须有，标"建议补充"）。

### 3.3 输出
- `docs/维护核对清单.md`（或 `build/doc-audit/` 下）：按文档分组的疑似项表 + 证据（引用原文行、commit hash、git 状态行）。
- `build/doc-audit/audit.json`：结构化结果（文档、信号、规则命中、证据路径），供 L1 消费或 CI 断言（`--fail-on <规则>` 可配）。

### 3.4 入口
```bash
python scripts/doc_maintenance_audit.py            # 全量扫描，输出清单
python scripts/doc_maintenance_audit.py --json     # 结构化输出
python scripts/doc_maintenance_audit.py --since 7d # 只看近 7 天有改动的文档
```
- 手动入口为主；CI 选配（只读，无副作用）。

## 4. M2 本地 LLM 判定（中期）

### 4.1 输入与输出
- 输入：audit.json 的疑似项 + 文档头部/相关段落（截断到 ≤4KB/项）+ 相关提交 message 列表；**脱敏**：不传 .env 值、密钥、日志正文，仅传提交标题与文档文本。
- 输出（JSON）：`{"doc": ..., "judgement": "stale|accurate|needs_review", "confidence": 0-1, "suggestion": "建议的状态行文本或说明"}`。
- 模型：Ollama 本地（gemma4:12b 或 qwen3 4b 均可；缺 Ollama 时跳过 L1，清单保持原样并标注）。

### 4.2 判定协议（防误报）
- 只允许三种判定；`confidence < 0.6` 一律 `needs_review`。
- `suggestion` 仅作为建议 diff 呈现，不自动应用。
- 校验输出 schema，非法 JSON 视为 `needs_review`（fail-closed）。

### 4.3 使用流程
```bash
python scripts/doc_maintenance_audit.py --llm          # 机械化 + LLM 分级
python scripts/doc_maintenance_audit.py --llm --apply  # 生成建议 diff（不自动提交）
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
| **M1 机械化扫描器** | `scripts/doc_maintenance_audit.py` + 规则表 + 清单模板 | 对当前仓库实测：能复现 2026-08-16 筛查发现的 4 类遗漏中的至少 3 类；全量扫描 <10s；零误删零改写（只读）；单测覆盖 5 条规则各至少 2 例（命中+不命中） |
| **M2 LLM 判定** | `--llm` 路径 + 判定协议 + 脱敏检查 | 对已知 4 例遗留样本：判定与人工结论一致率 ≥ 3/4；confidence<0.6 全部转人工；Ollama 缺失时机械化照常 |
| **M3 事件库 + RAG** | SQLite 三表 + embedding 索引 + 检索 CLI | 变更描述→关联文档 top-5 命中率 ≥ 60%（30 条抽样人工标注）；`--rebuild` 从 git log 重建与 git 一致 |
| 收口 | 全量核对清单清零（或逐项登记为已知/有意） | 2026-08-16 发现的 4 类问题模式在后续扫描中不再以"未发现"状态存在 |

## 7. 依赖与风险

| 项 | 说明 |
|---|---|
| 依赖 | M1：Python 3.10+ 标准库 + git CLI（本机已具备）；M2：Ollama（可选，已具备）；M3：sqlite3（已有先例） |
| 磁盘 | 极小（脚本 + 索引库 <50MB）；RAG 向量库另计 |
| 冲突边界 | 全程只读/建议；`--apply` 只生成 diff 不提交；不碰 `src/` 与 `tests/`；与并行组工作区无交集（新文件限 `scripts/doc_maintenance_audit.py`、`docs/维护核对清单.md`（生成物，可 .gitignore 或提交均可）） |
| 风险 | ① 状态行"故意不更新"（如历史参考文档）→ 规则 1 需"已声明历史"豁免词表（`历史参考/待拆分/不作为当前能力`）；② LLM 误判 → 置信度门 + 人工兜底；③ 过度设计 → M1 先落地解决眼前痛点，M2/M3 按需推进，不提前实现 |
| 与既有工具分工 | `check_links.py`（链接）、`文档状态与清理清单.md`（内容清理）、`已知问题记录.md`（问题登记）均不接管；本工具只管"状态行/登记/提交的一致性" |

## 8. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-16 | 建立本文档（M1-M3 设计基线）；立项背景为当日批量筛查发现的 4 类遗漏 |
