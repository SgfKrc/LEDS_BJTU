# DOC-AUDIT-ENTRY-01 定点审计实现与验收记录

> 状态：已完成
> 更新日期：2026-09-11
> 范围：`tools/docagent` 独立子项目及主项目票据文档；未执行模型加载、网络请求或外部写操作。

## 交付结论

第 17 票已新增 `docagent audit-entry`。入口面向“已知某一条文档声明需要核实”的场景，而不是把全库扫描结果再人工过滤：调用者必须给出一个 Markdown 文档，并以文字条目或标题锚点唯一定位审查范围。

命令输出将文档表述和仓库实际状态拆开：声明行与生命周期状态属于 `claims`；R1/R3 复用结果属于 `signals`；Git/工作区/路径存在性属于 `evidence`；所有不一致或尚不能独立证明的内容属于 `differences`。文档中的测试计数不会触发测试执行，也不会自动获得“已验证”资格。

## 入口与边界

```text
python tools/docagent/run.py audit-entry --root . \
  --doc docs/开发票计划-审计收口与答辩演示-2026-09-09.md \
  --anchor 已完成票-doc-audit-entry-01 \
  --json --output build/docagent/DOC-AUDIT-ENTRY-01.json --fail-on error
```

- `--entry` 与 `--anchor` 必须二选一；`--entry` 多处命中时必须提供 `--occurrence`。
- `--doc`、`--evidence` 只接受仓库相对路径；解析后的文档必须位于 profile 的 `docs_dir`，符号链接或 `..` 不能越出仓库。
- 输出不得写进文档树，推荐放在 `build/docagent/` 作为可再生工件。
- Git 不可用时显式返回 `git_available=false` 和未知 dirty 状态，不将“无法检查”伪装成干净。
- 报告只保留仓库相对路径；声明中的 API-key 形态、Bearer token 和绝对路径会被替换。

## 测试与复核

- 独立子项目：`tools/docagent/tests`，`39 passed`；主项目 docagent package/profile/report/gate/baseline/compat/evolution 与旧审计适配器邻接回归，`53 passed`。
- 覆盖项：标题锚点/重复标题规则、文字条目歧义与 occurrence、越界路径、缺失必需证据、未绑定测试结果、JSON/JUnit 测试计数匹配与不匹配、真实临时 Git 提交、R3 新提交信号、dirty evidence、R2 unstaged 状态回归、专用 `.env.docagent` profile、输出目录门及敏感内容脱敏。
- 静态检查：独立包 `compileall` 与 `git diff --check` 通过。
- 主项目定点自审输出：`build/docagent/DOC-AUDIT-ENTRY-01.json`。该工件处于忽略目录，可随时按上方命令重建；在未提交开发工作区执行时，完成声明对应 dirty 文档/子模块会产生 `COMPLETED_CLAIM_WITH_DIRTY_STATE`，这是如实证据，不是工具误报。

## 已知边界

- 工具只验证仓库可观察证据，不解释外部 CI 页面或口头声明。
- `tests_executed=false` 是固定合同；如需绑定测试结果，应先由测试系统生成 JSON/XML/log 工件，再通过 `--evidence` 指定。
- R1/R3 只读取文档元数据行与命中条目/章节；目标外章节中的源码引用不会污染定点结论。
