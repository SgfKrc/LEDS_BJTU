# Qwen3、Qwen3-VL 与 Qwen3.5 模型支持计划

> 状态：`QW3-S0 Completed（调研，2026-08-13）`；**`QW3-D1a` 双格式 Completed（2026-08-13）**——① Qwen3-4B 官方 Safetensors 经 ModelScope 直连（14 文件、7.51 GB、逐文件 SHA-256 对照官方哈希全通过，manifest `models/qwen3-4b/.qlh-model-asset.json`，revision `2c54d5a09e…`）；② Qwen3-4B-GGUF 官方 Q4_K_M 经 HF+7897 代理（2.326 GB、SHA-256 `7485fe6f…` 对照官方 LFS oid 全通过，`gguf_inspect`：architecture=qwen3、40K 上下文，manifest `models/qwen3-4b-gguf/.qlh-model-asset.json`，revision `bc640142c6…`）。`QW3-D1b`（Qwen3-VL-4B-Instruct 双格式/mmproj）与 `QW35-D1`（Qwen3.5-2B）下一步。
>
> 更新日期：2026-08-13
>
> 适用范围：Qwen3 文本系列、Qwen3-VL 系列与 Qwen3.5 原生多模态系列的工件获取、引擎准入和 EX-N3 质量标定。Gemma 4 的新格式工件另立候选票，不在本文引用或预设未建立的 Gemma 子票。
>
> 总计划入口：[总体下一步计划](总体下一步计划.md)；下载与用户代理规则：[一键模型部署与自治集群远期计划](一键模型部署与自治集群远期计划.md) §7.1。

---

## 1. 本轮结论

- `Qwen-1.8B Q4_K_M` 三轮客观子集均为正确率 `0/4`、格式率 `4/11`；它可继续作为弱工件回归样本，不能担当答案判定基线。
- `DeepSeek-R1-Distill-Qwen-7B Q4_K_M` 三轮同为正确率 `0/4`、格式率 `2/11`。实测根因是 R1 蒸馏模型无法被当前路径可靠地关闭 thinking，输出预算耗尽在 `<think>`，没有最终答案可按 rubric 检验。
- **首选替代不是再调 R1，而是官方 Qwen3-4B。** 它有官方 Safetensors 与官方 GGUF，且模型模板提供硬 `enable_thinking=false`，可以从机制上避免“只输出思考链”的已知问题。是否能通过答案检验仍必须实测，不能由模型卡或参数量推断。
- Qwen3-VL 是独立的视觉语言系列，官方既提供 Transformers 工件，也提供带 `mmproj` 的官方 GGUF；Qwen3.5 则是原生多模态系列，不能再误写成纯文本系列。
- Qwen3.5 也支持非 thinking 模式，但当前主运行时 `transformers==4.47.1` 不满足 Qwen3 的官方最低 `4.51` 要求；Qwen3.5 官方明确要求使用最新 Transformers，且它的 `qwen3_5` 混合注意力/多模态路径不能直接塞进现有进程。下载、运行时升级与质量标定必须分票。

当前质量门不变：在新模型完成固定工件的三轮标定和人工复核前，LLM 侧仍只使用现有“非 R1 格式率 + 人工复核”临时规则；**不得**因为下载成功就恢复客观正确率门。

## 2. 官方工件调研

| 家族 / 首选型号 | 官方 Transformers/Safetensors | 官方 GGUF | 当前定位 |
|---|---|---|---|
| Qwen3-4B | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)，Apache-2.0 | [`Qwen/Qwen3-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-4B-GGUF)，含 Q4_K_M（约 2.5 GB） | **D1 首个双格式目标**；文本答案判定候选 |
| Qwen3-0.6B / 1.7B / 8B | Qwen 官方 Qwen3 collection | 对应 Qwen 官方 `-GGUF` 仓库 | 后续容量梯度；不抢占 4B 的首轮标定 |
| Qwen3-VL-4B-Instruct | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 官方 Transformers 工件 | [`Qwen/Qwen3-VL-4B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)：语言模型 Q4_K_M 约 2.5 GB；视觉编码器 `mmproj` 是独立文件（FP16 或 Q8_0） | D1 次序二；Gemma 判题器的交叉对照候选 |
| Qwen3-VL-8B-Instruct | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 官方 Transformers 工件 | [`Qwen/Qwen3-VL-8B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)：Q4_K_M 约 5.03 GB，另须 `mmproj` | 资源条件票；Ollama 标称包约 6.1 GB，不与 SD/Gemma 并驻留 |
| Qwen3.5-2B | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)，官方 Transformers/Safetensors | 本轮未发现 Qwen 官方 GGUF 发布；第三方转换仅作候选 | 原生多模态、默认非 thinking；先走隔离 PyTorch/服务端运行时 |
| Qwen3.5-4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)，4B 语言模型 + 视觉编码器 | 本轮未发现 Qwen 官方 GGUF 发布；第三方转换不得直接纳入 baseline | 原生多模态；默认 thinking，需 API 级显式关闭 |
| Qwen3.5-0.8B / 9B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) / [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) 官方 Transformers/Safetensors | 本轮未发现 Qwen 官方 GGUF；第三方转换不进入 baseline | 后续梯度，不进入 D1 |

“PyTorch 模型”在本计划中指官方 Hugging Face Transformers/Safetensors 工件，不接受来源不明的 `.pt`/pickle 权重。GGUF 与 Safetensors 是两个独立 artifact：各自锁 revision、文件清单、总大小和 SHA-256，不能共享一次准入结论。

## 3. 思考模式与答案检验边界

| 模型 | 可控性 | EX-N3 要求 |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 当前实测无法可靠关闭 thinking | 不再作为客观答案判定基线；保留历史失败证据 |
| Qwen3 | `tokenizer.apply_chat_template(..., enable_thinking=False)` 为硬关闭；`/no_think` 只是 thinking 开启时的软切换 | `QW3-G3` 必须验证 GGUF 与 Safetensors 路径都不产生 `<think>`，并且答案落在既有输出预算内 |
| Qwen3.5-2B | 官方说明默认非 thinking；思考需 API 显式启用 | 仍须明确传递运行时参数；官方已提示 2B 在 thinking 模式更易陷入循环，不能把默认非 thinking 当作质量通过 |
| Qwen3.5-4B / 9B | 默认 thinking；官方 OpenAI 兼容路径用 `chat_template_kwargs: {enable_thinking: false}` 获取直接回答 | QLH 当前本地聊天接口尚未证明会透传此参数；未实现/验证前不能用于答案判定 |

Qwen3 非 thinking 模式的推荐采样为 `temperature=0.7`、`top_p=0.8`、`top_k=20`、`min_p=0`；Qwen3.5 的推荐参数因型号和文本/VL 任务不同。现有 `experiment_llm_quality_unit.py` 是固定贪心 Qwen 1.8B 执行器，**不能直接把新工件名替换进去**。`QW3-G3` 必须把模型族、chat template、thinking 开关、采样参数和 `max_new_tokens` 全部写入 plan 并锁定。

## 4. 工件下载与信任策略

### 4.1 下载顺序

1. **QW3-D1a**：`Qwen/Qwen3-4B` 官方 Safetensors 与 `Qwen/Qwen3-4B-GGUF` 官方 Q4_K_M；先完成两个独立 artifact 的 resolve、下载、摘要和 inspection。
2. **QW3-D1b**：`Qwen/Qwen3-VL-4B-Instruct` 官方 Safetensors，以及同一官方 GGUF 仓库中、由上游兼容性声明覆盖的 Q4_K_M 语言模型与 `mmproj`。两个仓库的 Git revision 不可假定相同；必须分别锁定 revision、文件名、大小、SHA-256，并校验 `qwen3vl` 架构与同一 Instruct 型号。
3. **QW35-D1**：`Qwen/Qwen3.5-2B` 官方 Safetensors。Qwen3.5 官方未锁定 GGUF 时，不自动下载第三方转换；若以后引入，必须以“第三方转换 artifact”独立登记、完整哈希并和官方 Transformers 输出做语义对照。
4. **条件扩展**：Qwen3-VL-8B 和 Qwen3.5-4B 只在 D1a/D1b 通过磁盘与运行时预检后排队。Gemma 4 新格式工件单列专项，不与本票混传或共享准入结论。

每个任务先做 resolve/dry-run，确认精确 commit revision、许可证、文件 pattern、合计大小、staging 预算和目标目录可写，再允许传输。默认每次只传一个“大工件组合”；Safetensors 与 GGUF 不并行拉取，以免 staging 与磁盘预算相互挤占。下载完成只代表 artifact 已受管，仍须通过 format/架构检查、隔离试加载和引擎 smoke，才可能进入部署。

### 4.2 7897 用户代理

项目已有模型下载代理优先级：`QLH_HTTP_PROXY` > 用户持久化模型代理 > 直连。直连失败时，用户可在模型管理“网络”页设置 `http://127.0.0.1:7897`，或只为本次受控下载进程设置：

```powershell
$env:QLH_HTTP_PROXY = 'http://127.0.0.1:7897'
```

- 代理只影响模型 resolver/downloader 子进程，不修改系统代理、`HF_ENDPOINT`、全局 Hugging Face 配置或模型运行时。
- 7897 可在下载完成后关闭；已持久化的 partial/Range 元数据按**同一 source、revision 和文件 ETag**续传，换代理不允许换源或绕过重新 resolve。
- token 仍只通过 `credential_ref`/系统凭据保管；代理地址、访问 token、响应正文和本地绝对路径不进入实验记录。loopback HTTP 仅用于本机代理，远程模型源仍必须 HTTPS。

## 5. 分期计划

| 票 | 状态 | 范围 | 退出条件 |
|---|---|---|---|
| QW3-S0 | Completed（调研） | 官方源、工件格式、thinking 控制、容量与现有运行时差距 | 本文 §1-§4；不下载、不改依赖 |
| QW3-D1 | Ready（下载与登记） | Qwen3-4B 双格式，再按顺序 Qwen3-VL-4B 双格式/投影器 | 每个 artifact 具备 revision、许可、文件清单、大小、SHA、受管 manifest；下载失败可经 7897 恢复 |
| QW3-R2 | Ready（隔离运行时设计） | 为 Qwen3 Safetensors 准备 `transformers>=4.51` 的隔离 sidecar；主安装运行时继续锁在 4.47.x | sidecar 不改变主应用/打包依赖；Qwen3 配置、tokenizer 与 `enable_thinking=false` 可被预检 |
| QW3-G3 | Conditional（真实工件后） | Qwen3-4B GGUF/Transformers 单机 smoke、thinking 硬关闭、输出预算/格式契约 | 双路径各自产生最终答案；无 `<think>`、无 prompt/正文持久化；失败 fail-closed |
| QW3-E4 | Conditional（G3 后） | Qwen3-4B 三轮客观子集标定与人工复核 | 仅在三轮稳定通过后，才讨论恢复 LLM 正确率质量门 |
| QWVL-J1 | Conditional（D1b/G3 后） | Qwen3-VL-4B 与 Gemma 4 对同一 SD 输出作交叉判题 | 只记录受限计数；不以任一模型单次结果覆盖人工复核 |
| QW35-R1 | Conditional | Qwen3.5-2B/4B 的最新 Transformers、多模态 processor、显式 thinking 控制和资源准入 | 先隔离运行时和本机 smoke；第三方 GGUF 不得替代官方 PyTorch 基线 |

## 6. 当前运行时前置

| 组件 | 当前事实 | 新模型要求 / 处理 |
|---|---|---|
| 主 Python/打包依赖 | `transformers==4.47.1`、`torch==2.13.0+cu126` | Qwen3 官方提示低于 `transformers 4.51` 会缺 `qwen3`；Qwen3.5 官方要求最新 Transformers。Safetensors 均先走隔离 sidecar，不能直接升级主依赖 |
| 现有 PyTorch loader | 以 `AutoModelForCausalLM` 和 Qwen2 兼容窗口为主；遗留 Qwen 配置仍有 `TRUST_REMOTE_CODE=True` | Qwen3.5 需要官方多模态模型类/processor，另立适配票；新 Qwen3/Qwen3.5 准入不得因遗留开关而自动启用 `trust_remote_code` |
| llama.cpp 源 | 锁定 `47e1de77…f89fcbe`，源码已枚举 `qwen3`、`qwen3vl`、`qwen35`、`gemma4` 架构 | 源码枚举不是运行时能力；本机 `llama-cpp-python==0.3.28` 必须与目标 GGUF/mmproj 组合实际 smoke 后再登记支持 |
| Ollama（Qwen3-VL 可选路径） | 本机已有 Ollama 路线 A 与通用 `external_api` 图像消息能力；尚未拉取 Qwen3-VL | 官方要求 Ollama `>=0.12.7`；`qwen3-vl:4b` 约 3.3 GB、`qwen3-vl:8b` 约 6.1 GB。D1b 先受管官方 Hugging Face 工件，是否额外拉取 Ollama 标签由 QWVL-J1 单独决定 |
| 8 GB RTX 4060 | Qwen3-4B/Q4 与 Qwen3-VL-4B/Q4 是优先尝试范围；不能与 SD/Gemma 并驻留 | 每次运行前执行现有资源准入；Qwen3-VL-8B、Gemma 4 Safetensors 及长上下文保持条件票 |

## 7. EX-N3 恢复路径

Qwen3-4B 是“恢复答案检验”的候选，不是预设结论。执行顺序固定为：

```text
官方双格式 artifact 受管
  -> 隔离运行时 / llama.cpp 预检
  -> 明确关闭 thinking 的单机 smoke
  -> 固定 chat template 与采样的三轮客观子集
  -> 人工复核与阈值修订
  -> 才能改动 quality.required 的 LLM 正确率规则
```

若 Qwen3-4B 仍不能产生可检验最终答案，记录失败并维持当前格式率方案；不得以增大输出预算、剥离 `<think>` 文本或放宽 rubric 来制造“通过”。

## 8. 参考

- [Qwen3-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B) 与 [官方 GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Qwen3-VL-4B 官方 GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) 与 [Qwen3-VL-8B 官方 GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B) 与 [Qwen3.5-4B 模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Ollama Qwen3-VL 标签与大小](https://ollama.com/library/qwen3-vl)
