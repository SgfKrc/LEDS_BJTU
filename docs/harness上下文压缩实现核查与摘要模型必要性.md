# Harness 上下文压缩实现核查与摘要模型必要性

> 状态：**调研完成**（实现核查：上下文压缩为纯规则实现，无任何模型参与；结论：**"专用压缩小模型"作为独立目标没有必要**；值得做的是"把现成小模型接成 summarizer 角色 + A/B 度量"，且已有规划票承接）
>
> 创建日期：2026-09-09
> 适用范围：harness `context_engine/` 的摘要/折叠路径与记忆抽取的现状与演进建议；不涉及主项目判题/工具支线。关联票：[HW-CTX-SQZ-01](开发票计划-审计收口与答辩演示-2026-09-09.md)（上下文压榨）、`EX-CTX-MEAS-01`/`TOOL-CTX-RESS-01`（测度与压力测试）。

---

## 1. 实现现状（代码核查，2026-09-09）

| 位置 | 实现 | 模型参与 |
|---|---|---|
| `context_engine/summarize.py` — `SummaryProvider`（Protocol） | 摘要提供者接口：`summarize(messages) -> SummaryResult`（含 STATE 校验/渲染，模型无关） | —（接口层） |
| `context_engine/summarize.py` — `RuleBasedSummarizer`（**唯一实现**） | 确定性规则：按角色归类（user→`what`、输出→`artifacts`、assistant→`decisions`、其他→`open`），每字段 ≤4 条 × 160 字符；`next` 记录"review omitted messages: N"；`source_message_ids` 溯源 | **无**（纯规则，docstring 自注"used before a dedicated summary model exists"） |
| `context_engine/policy.py` | `ContextPolicyConfig.summarizer: SummaryProvider = default_factory(RuleBasedSummarizer)`；折叠触发时 `_summarize(old_messages, current_state)`，STATE 经 `validate_state`/`apply_state_patch`（未知字段拒绝、删除需显式确认——防小模型注入的护栏已就位） | 无 |
| `memory/extract.py` | `extract_memory_candidates(summary, ...)`：从 SummaryResult 的规范化 STATE 抽取候选（显式候选 + 摘要字段清洗/去重标记），**同样无模型** | 无 |
| `adapters/`、`api_layer/` | 无任何 summarize/sub-summary 调用 | 无 |

**结论**：harness 的上下文压缩 = **确定性规则压缩**（安全、零成本、零幻觉，但信息保留粗糙）；**没有用任何模型做摘要，更没有专门用来压缩的小模型**。

**与 S1 方案设计的差异**（需登记）：S1 设计原文写"摘要 call（结构化 STATE 输出；大模型优先、同模型回退并标记）"——实现只落地了"无模型回退"（RuleBased），**LLM 摘要 call 从未实现**；`SummaryProvider` 协议与 STATE 校验护栏（validate_state/apply_state_patch）是为它预留的插槽，且针对"小模型压缩"的防注入设计已就绪。

## 2. "专用压缩小模型"是否有必要

### 2.1 结论：**作为独立目标没有必要**；现成小模型担任 summarizer 角色值得做

| 候选方案 | 成本 | 收益 | 判定 |
|---|---|---|---|
| A. 保持纯规则压缩（现状） | 零 | 安全/确定性；跨轮语义折叠、事实归并缺失 | 基线可用 |
| B. 训练/引入**专用压缩小模型**（LLMLingua 式 token 剪枝、专用摘要模型） | 需新模型工件/训练；本项目玩具定位且无训练计划 | 压缩率提升，但本项目调研（§2.2）已记录：**token 剪枝会删语义必需 token**、摘要型模型在小模型上 68% 失败源于过早覆盖状态字段 | **不做**（风险 > 收益） |
| C. 把**已下载的现成小模型（Qwen3-0.6B）**接为 harness `summarizer` 角色 | 低（实现一个 `SummaryProvider` 实现类：经 adapter/chat 产 STATE JSON → schema 校验 → fail-closed 回退 RuleBased）；模型已在手（0.6B 快且模板支持 thinking 关闭） | 语义折叠/去重/跨轮事实归并质量提升；与回答模型**角色分工**（呼应"7B 写草稿、小模型精修"思路的镜像） | **值得做**（作为 `HW-CTX-SQZ-01` 首增量） |
| D. C 的大模型版本（更大模型做摘要，社区建议） | 本项目无更大模型；远端 qlh 可代但引入数据作用域问题 | 摘要质量上限更高 | 后置（远端模式可选） |

### 2.2 理由展开

1. **现状压缩已经成立**：规则压缩 + STATE schema + 删除确认 = 让"小模型上下文有限"问题**可工作**的最小闭环；质量缺口（语义折叠）由 C 补齐而非替换。
2. **专用压缩模型在小模型场景是负期望**：本项目 2026-08 调研（社区组合 + [SKILL.state 论文证据](小模型轻量推理harness工作台调研与方案.md) §2.2）明确——压缩模型引入的是"删错 token / 覆盖状态"类新风险，且需要额外工件与维护，违背玩具定位。
3. **"专门"的价值应来自角色分工而非专用模型**：用现成 0.6B 当 `summarizer`（专用角色、不训练），保留规则压缩为 fallback；质量差异用 A/B 度量（`EX-CTX-MEAS-01` 的早召回-预算曲线）验证——**用数据决定是否继续**，不给模型承诺。
4. **护栏复用**：`validate_state`/`apply_state_patch` 已针对"小模型压缩输出不可信"实现（未知字段拒绝、删除显式确认、字段类型/长度边界），LLM 摘要器接入无需新安全设计。

## 3. 建议（登记为规划，不在此落地）

1. `HW-CTX-SQZ-01` 首增量：实现 `LLMSummarizer(SummaryProvider)`——调用 harness 现有 chat 路径（本地 llama_server/qlh 后端无所谓），**summarizer 角色固定为 Qwen3-0.6B**（配置可换），输出 JSON STATE → `validate_state` 校验 → 失败/超时/非法 JSON **fail-closed 回退 RuleBasedSummarizer** 并发出 `context.summary_fallback` notice；
2. `EX-CTX-MEAS-01`/`TOOL-CTX-RESS-01`：A/B（RuleBased vs 0.6B-summarizer）× 30 轮 fixture → 早召回-预算曲线 + 摘要覆盖错误率；**只有度量通过才把 summarizer 角色设为默认**；
3. 设计差异收口：在 harness 方案文档 S1 实施记录补一行"LLM 摘要 call 未实现（规划中，见本调研）"，避免"摘要已完成"的误读。

## 4. 变更记录

- 2026-09-09：初版（实现核查 + 必要性结论 + 建议与关联票）
