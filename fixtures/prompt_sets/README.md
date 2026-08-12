# 固定提示词集管理规则

提示词集是**版本化资产**：任何修改（增删条目、改措辞、改长度分布）必须升版本号，禁止原地覆盖。

## 当前版本：ps-v1-zh-en-code

- 位置：`fixtures/prompt_sets/ps-v1-zh-en-code/prompts.jsonl`
- 类别：`zh_qa` / `en_qa` / `code` / `math_reasoning` / `long_context` / `format`，每类 5 条，共 30 条
- 每条记录：`id`、`category`、`prompt`、`prompt_token_estimate`
- `prompt_token_estimate` 为字符级估算（中文按 1 字 ≈ 1.4 token、英文按 4 字符 ≈ 1 token 的近似口径），仅用于输入长度分级；**正式实验的实测 token 数由 runner 在实验记录中写入**（`record.prompt_set` 摘要），不得回写本文件（保持哈希锁定）
- 长上下文类（`long-*`）的实际长文由实验计划在 runner 命令中注入固定内容（如 `fixtures/prompt_sets/ps-v1-zh-en-code/long_docs/`），提示词只含指令模板

## 使用

- 实验计划（manifest）引用 `prompt_set.id`，框架校验文件 SHA-256 与 manifest 声明一致；不匹配直接拒绝运行
- 报告必须声明 `prompt_set_id` 与 SHA-256；不同版本的数据不得横向合并

## 锁定

当前 SHA-256（2026-08-09 冻结）：
`8cc555f57fa23d45c16820f77cc90b507da04578a1e90eebf46e19d4eb2568a3`（30 条）

## 版本：ps-sd-v1（SD→Gemma 三轮标定图像提示词集）

- 位置：`fixtures/prompt_sets/ps-sd-v1/prompts.jsonl`
- 类别：`image_text`，10 条固定图像描述提示词
- 每条记录：`id`、`category`、`prompt`、`key_elements`（2 个/条，用于 EX-N3 Gemma 判题的归一化包含匹配）
- 用途：SD 1.5 十 seed 文生图 + `experiment_gemma_judge_real.py` 图生文判题的固定输入；`prompt` 与 `key_elements` 均不进入质量证据（契约白名单禁止）
- 选择原则：要素客观、SD 1.5 可稳定生成、Gemma 描述中大概率出现（可子串匹配）

## 锁定

当前 SHA-256（2026-08-13 冻结）：`8b1775e0d64e357a854a62870ca7e3c2f58667d96219773d03374bf41ecfabd6`
