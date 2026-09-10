# AUD-RT-01 / AUD-SW-01 真实模型验收结论

日期：2026-09-10
状态：两票均已完成

## 验收环境

- Windows 开发机，Python 3.12。
- 真实运行时：`llama-cpp-python 0.3.28`。
- 本地固定 GGUF：Qwen2.5-0.5B Q4_K_M、Qwen3-0.6B Q8_0、MiniCPM4-0.5B Q4_K_M、DistilQwen2.5-DS3-0324-7B Q4_K_M。
- 模板 worker：`windows-restricted-token-low-integrity`，离线 Transformers 环境。
- 本报告不保存模型原始输出、绝对路径或凭据；可再生的机器报告位于被忽略的 `build/audits/`。

## AUD-RT-01 结果

三个亚 1B 模型的 tokenizer/chat template 均在低权限沙箱中实际执行，静态工件、manifest 和 GGUF 架构门全部通过：

| 模型 | 模板执行 | thinking 结论 | GGUF/runtime |
| --- | --- | --- | --- |
| Qwen2.5-0.5B | 通过 | 模板未声明开关；运行时接受但不宣称支持 | 加载与生成通过 |
| Qwen3-0.6B | enabled/disabled 均完成渲染 | 声明 `enable_thinking`，开关探测通过 | 加载与生成通过 |
| MiniCPM4-0.5B | 通过 | 模板未声明开关；运行时接受但不宣称支持 | `minicpm` 架构映射、加载与生成通过 |

真实 GGUF 矩阵中，三个亚 1B 模型共完成 12/12 次非空生成，runtime 错误为 0。MiniCPM4 单模型 B1 runtime gate 为通过。严格输出质量是另一道门：Qwen3 与 MiniCPM4 未完全遵循 exact JSON 和长上下文 marker；这说明小模型能力边界，不等同于模板、架构或 llama.cpp 不兼容。代码现分别输出 `runtime_*` 与 `quality_gate_passed`，避免误分类。

DSW-D1 同轮完成 DistilQwen2.5-DS3-0324-7B 的真实模板执行、GGUF 结构与 sidecar SHA 校验、Qwen2 架构映射、转换器/量化器来源验证及 Q4_K_M 磁盘计划，结果通过。四模型矩阵尝试加载 DS3 时，本机可用内存 5.39 GiB 低于 4.68 GiB GGUF 的保守加载预算，工具按设计返回 `insufficient_ram`，未强行制造系统内存压力；因此不声明 DS3 runtime 或性能结论。该项不影响本票规定的 MiniCPM4 runtime 验收结论。

可再生证据：

- `build/audits/AUD-RT-01-b1-runtime-2026-09-10.json`
- `build/audits/AUD-RT-01-four-model-runtime-2026-09-10.json`
- `build/audits/AUD-RT-01-dsw-d1-2026-09-10.json`

## AUD-SW-01 发现与修复

真实预验收发现：健康的 Qwen3-0.6B 已加载时，切换到未注册模型会先执行卸载，再从加载分支提前返回 `MODEL_NOT_REGISTERED`；结果是活跃模型被清空且没有回滚。

修复后，目标注册预检位于卸载之前：

- 未注册目标返回 `MODEL_NOT_REGISTERED`、`requested_model_id` 和 `active_model_preserved=true`。
- 当前模型 ID、引擎对象和已加载状态保持不变。
- 已注册但工件缺失的目标仍进入真实加载失败路径，并按原事务逻辑回滚到旧模型，返回 `MODEL_LOAD_FAILED_ROLLED_BACK`。

真实模型用例完成以下断言：

1. Qwen2.5-0.5B 真实加载并生成非空内容。
2. Qwen2.5-0.5B 成功切换到 Qwen3-0.6B。
3. 未注册目标在卸载前被拒绝，Qwen3 继续可用。
4. 已注册但缺失 GGUF 的目标加载失败，随后真实重新加载 Qwen3 并完成回滚。
5. `pytorch` 引擎与 `Q4_K_M` 量化错配返回 HTTP 400，当前 llama.cpp 模型状态不变。

真实测试命令最终为 `2 passed`；代码级专项回归为 `78 passed, 4 skipped`，其中真实模型测试在普通测试环境按设计跳过，随后通过显式 opt-in 在含 `llama_cpp` 的运行时执行。最终全量 Python suite 为 `3448 passed, 22 skipped`。

## 票据边界

- `AUD-RT-01` 验收的是模板/thinking/架构与真实加载生成兼容性，不承诺亚 1B 模型的严格指令遵循率。
- `AUD-SW-01` 验收的是单进程模型生命周期事务，不代表物理双机性能。
- 本轮未按 `qlh.real_model_performance.v1` 采集 TTFT、inter-token、tokens/s 或峰值内存；性能结题表继续保留 `NOT RUN`，不得用本次 smoke 延迟替代。
- 原审计 1～15 票现为 15/15 完成，待完成工作为 0 项。
