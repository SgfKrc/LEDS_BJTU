# Patchouli — 知识库/文档库管理 TUI（只读）

> 设计与票分解见主仓 `docs/Patchouli知识库管理TUI专项计划-2026-09-14.md`。
> 纪律：只读、零第三方依赖、fail-soft（未知格式不崩溃）。

## 现状

| 票 | 内容 | 状态 |
| --- | --- | --- |
| `PATCH-01` | 只读数据层（catalog）：docs/ 扫描 + 状态行/更新日期/票号/互链/分类解析 | ✅ 完成（2026-09-14） |
| `PATCH-02` | 书架 TUI（三栏：列表→文档卡→预览；分类过滤/归档开关） | ✅ 完成（2026-09-14，Textual） |
| `PATCH-03` | 检索台（对接 RAG 检索全链路） | 待做 |
| `PATCH-04` | 编目（docagent scan/audit 集成） | 待做 |
| `PATCH-05` | 流通记录（git log/变更记录） | 待做 |
| `PATCH-06` | 馆藏统计 + 收口 | 待做 |

## 用法

```bash
python -m patchouli --root <repo> --summary   # 馆藏摘要（分类/票号/缺状态行）
python -m patchouli --root <repo> --json      # 全量结构化（qlh.patchouli.catalog.v1）
python -m patchouli.bookshelf --root <repo>   # 书架 TUI（↑↓ 选择 · 1-7 分类 · 0 全部 · a 归档 · r 刷新 · q 退出）
```

> **依赖边界**：Patchouli 是**开发期工具**（与 docagent 同级），书架 TUI 允许 Textual；
> **引擎产品 TUI（`qlh chat`）仍遵守零第三方依赖基调**（见主仓《QLH-TUI跨平台基调》），两者定位不同。

## 示例输出（主仓 2026-09-14）

```
docs: 90（archive 10） 票号 510
分类: {decision 5, guide 16, other 10, reference 7, report 7, special-plan 43, ticket-plan 2}
缺状态行: 31
```

## 测试

```bash
python -m pytest tools/patchouli/tests -q   # 6 passed（含真实仓库冒烟）
```
