# Qwen3、Qwen3-VL 与 Qwen3.5 模型支持计划

> 状态：`QW3-S0 Completed`；**`QW3-D1a` / `QW3-D1b` / `QW35-D1` Completed（2026-08-13）**——① D1a：Qwen3-4B 官方 Safetensors（ModelScope 直连，14 文件、7.51 GB，revision `2c54d5a09e…`）+ 官方 GGUF Q4_K_M（HF 代理，2.326 GB，revision `bc640142c6…`，qwen3/40K ctx），manifest 分别登记；② D1b：Qwen3-VL-4B-Instruct 官方 Safetensors（ModelScope 直连，15 文件、8.28 GB，`qwen3_vl` + 内置视觉编码器）+ 官方 GGUF `Q4_K_M`（2.326 GB，qwen3vl/256K ctx）+ `mmproj-F16`（0.779 GB，clip），两仓库 revision 独立锁定；③ QW35-D1：Qwen3.5-2B 官方 Safetensors（ModelScope 直连，14 文件、4.26 GB，`qwen3_5`，官方无 GGUF）。**`QW3-R2` Completed（2026-08-13）**——隔离 sidecar 已就绪：`setup_qwen3_sidecar_env.py` 建独立 venv（`transformers 4.57.6` 隔离，主运行时仍锁 4.47.x），`qwen3-sidecar-probe` 对 `models/qwen3-4b` 真实预检通过：tokenizer 加载、chat template 可用、`enable_thinking=False` 硬关闭且渲染无 `<think>` 脚手架，状态 `ready_for_qwen3_smoke`。**`QW3-G3` 闸门已开（工件 + sidecar 就绪）**。`QW3-G3` GGUF 半程 **Completed（2026-08-13）**，Safetensors 半程 **Completed（2026-08-14）**：隔离 sidecar 已固定 `torch 2.13.0+cu126`、`accelerate 1.14.0`、`psutil 7.2.2`；真实 Qwen3-4B CPU 首段/末段与 CUDA 中间段 assignment smoke 通过，涵盖 `enable_thinking=false`、KV prefill/decode、tied embedding logits、RSS/CUDA 峰值和 `full_model_materialized=false`。首段 CUDA 在本机 RAM 不足时结构化拒绝，不能据此标记完整单机/多机准入。**G3 整票关闭（2026-08-14）**：另以 sidecar（transformers 4.57.6 + bnb 0.50.1 `BitsAndBytesConfig` 4bit）全模型加载（device_map auto，15s）完成**最终答案对照**——与 GGUF 半程同一渲染 prompt（`enable_thinking=False`），数学题输出 `2`（1.0s）、一句话题输出 `TCP 是传输控制协议，用于在网络上可靠地传输数据。`（1.1s），**两路径逐字一致、无 `<think>` 正文、预算/格式通过**；fp16 全模型资源门（峰值方案 P0）双路径实证 fail-closed（GPU 9.95GB 需求 > 7.45GB 可用、CPU 19.4GB > 10.4GB）。**`QWVL-J1` Completed（2026-08-14）**：Qwen3-VL-4B 与 Gemma 4 对同一 SD 输出集（`build/exp-calibration/round-1`，10 图、`gemma-judge-counts-v1` 契约）交叉判题：Gemma4:12b `topic 3/10`、`coverage 3/20`（EX-N3 v2 基线）；**Qwen3-VL-4B（Ollama，num_ctx=4096）`topic 8/10`、`coverage 9/20`**——同一图集下显著领先，但按 J1 规则**只记录受限计数，不以单次结果覆盖人工复核**。实测登记：① qwen3-vl:4b 在 Ollama 默认 **256K 上下文 → 44GB KV、89% CPU**，须显式 `num_ctx` 受限；② 该模型 Ollama 路径**偶发 finish=length 空输出**（判题器已加重试，J1 最终 0 failures）；③ `reasoning_effort` 字段对 qwen3-vl 有害（空输出），判题器已参数化。证据 `build/exp-calibration/round-1/qwen3vl-evidence-v3.json`。**`QW3-E4`（三轮客观子集标定）Completed（2026-08-14）**：Qwen3-4B GGUF Q4_K_M（非思考渲染 prompt + 锁定采样 0.7/0.8/20/0）三轮 `ps-v1` 客观子集**完全稳定**——correctness **0/4 ×3**、format **4/11 ×3**（与 Qwen 1.8B 标定结果逐项相同 0.0 / 0.364）。人工复核 math-001：模型输出完整推理过程（1200km/8:00/120km/h 逐步推导），192 token 预算内未收敛到 `normalized_contains` 精确串 `13:54` → 判错。**结论：`normalized_contains` 判题口径存在地板效应（对推理风格输出不匹配），当前不能区分模型能力；按 EX-N3 规则维持「非 R1 格式率 + 人工复核」临时质量规则，客观正确率门不恢复**。标定证据 `build/experiments/qwen3-e4-round{1,2,3}.json`；后续若换判题口径（宽松匹配/模型族感知）需重标。
>
> 更新日期：2026-08-14
>
> 适用范围：Qwen3 文本系列、Qwen3-VL 系列与 Qwen3.5 原生多模态系列的工件获取、引擎准入和 EX-N3 质量标定。Gemma 4 的新格式工件另立候选票，不在本文引用或预设未建立的 Gemma 子票。
>
> 总计划入口：[总体下一步计划](总体下一步计划.md)；下载与用户代理规则：[一键模型部署与自治集群远期计划](一键模型部署与自治集群远期计划.md) §7.1。

> **2026-08-14 `PT-PIPE-QW3.4` 增量**：Scheduler dry-run canonical 合同与 prepare/commit/abort/release 故障矩阵已完成；真实 Qwen3-4B 三段 C3 manifest 摘要合同约 2.9 KiB。该票明确 `network_dispatch=false / weight_materialization=false / full_model_fallback=false`，仅证明控制协议可承接真实 revision，不构成跨节点或生产准入。

> **2026-08-14 `PT-PIPE-QW3.5` 增量**：已完成单机认证 loopback：只接受 HMAC 注册完成的 loopback TCP peer，控制帧/ACK 绑定 peer、合同、代际、阶段、时间窗口和 nonce；worker 先验 C3 assignment manifest，再以严格 HTTP Range 只读 Safetensors 头部。真实 Qwen3-4B shard 只读 20,008 bytes，未物化 tensor；专项 `48 passed`，扩大回归 `481 passed`。Qwen3 生产准入、真实双机 hidden tensor/KV 数据面、Qwen3-VL 和 Qwen3.5 仍未因此放开；下一票是 `PT-PIPE-QW3.6` node-local 隔离 sidecar 执行生命周期。

> **2026-08-14 `PT-PIPE-QW3.6` 增量**：Qwen3 现在有独立 sidecar session/runtime worker，以 `prepare -> commit -> release/abort` 管理 filtered assignment 和段物化；合同默认 metadata-only，只有显式 `node_local_sidecar` 才改变段物化标记。断线、超时、会话重放和 staging 清理已覆盖；QW3 扩大回归 `488 passed`，完整 unit 通道 `2201 passed / 6 skipped`。该票仍未开启真实双机 hidden tensor/KV 传输、Qwen3-VL 或 Qwen3.5 生产准入；下一票 `PT-PIPE-QW3.7` 做单机多 sidecar hidden handoff/KV 执行。

> **2026-08-14 `PT-PIPE-QW3.7` 增量**：新增单机 2/3 sidecar 编排和 `from_contract()` 绑定，hidden/KV 只通过本机受控 artifact 交接，JSONL 控制帧绑定输入/输出摘要、chain/segment、shape、dtype/device、sequence length 与 generation。prefill/decode 顺序、独立 KV、取消、重启恢复、摘要篡改和任一段失败全量清理已覆盖；专项 `10 passed`，完整 unit `2208 passed / 6 skipped`。该票仍未开启真实双机 hidden/KV、CUDA 结果等价、异构转换、Qwen3-VL/Qwen3.5 或生产准入；下一票 `PT-PIPE-QW3.8` 做 Scheduler 本地链入口与 CPU parity gate。

> **2026-08-14 `PT-PIPE-QW3.8` 增量**：主节点现在有显式本地链 Scheduler/API，用户取消、SQLite 元数据投影、服务重启 reconcile、遗留工件定向回收、同合同/陈旧代际 fencing 均已收口；默认聊天与生产流水线不自动进入。CPU parity gate 对 2/3 段 logits、artifact SHA/字节、hidden shape 和 KV phase/generation/length 做 fail-closed 对照，失败整链取消且不回退整模。合成 Qwen3-like 同权重整链/分段对照通过；新增专项 `15 passed`、QW3 专项 `102 passed`、完整 unit `2223 passed / 6 skipped`。独立 Worker 的运行时合同已移出 `scripts` 导入路径。真实 CUDA/双机仍未准入；下一票 `PT-PIPE-QW3.9` 做不依赖外部硬件的认证网络工件传输合同与 loopback 故障矩阵。

> **2026-08-14 `PT-PIPE-QW3.9` 增量**：Qwen3 已有独立的认证工件传输合同和 loopback 数据面。HMAC 票据绑定 peer/chain/generation/phase/相邻 segment/bytes/SHA/TTL/nonce；接收端以 4 MiB 上限顺序写入 `.part`，支持 offset 查询、断线续传、最终摘要复核和原子提交，控制面不聚合完整 hidden/KV。FastAPI adapter 只信任连接认证层注入的 peer identity；跨 peer/跨 session 不得破坏合法 staging，协议损坏、摘要失败、取消和过期均定向清理。专项 `14 passed`、QW3 专项 `107 passed`、完整 unit `2237 passed / 6 skipped`。该 router 尚未注册到生产 API 或真实 Scheduler peer，真实双机/CUDA/生产准入仍关闭；下一票 `PT-PIPE-QW3.10` 做认证 Scheduler 接线和同机模拟多节点 sidecar 链。

> **2026-08-14 `PT-PIPE-QW3.10` 增量**：Scheduler 现在可显式接入 peer verifier、活动 network coordinator 和可选 handoff transport；请求级 HMAC proof 绑定 method/path、Bearer 摘要、时间窗口/nonce，控制体再以规范 JSON + SHA-256 绑定，目标端只接受活动 canonical chain 的相邻已认证 peer。`Qwen3PipelineMultiSidecar` 输出 path-free `local`/`network` artifact reference，目标摘要复核并原子提交后才允许下一段消费。测试 helper 启动独立目标进程，2/3 节点 prefill/decode、断线续传、失败/取消、重启孤儿回收和陈旧合同拒绝通过；网络专项 `14 passed`、QW3 `121 passed`、全量 `2252 passed / 8 skipped`。实际生产路由、真实 TCP 注册生命周期、目标端 sidecar 消费、真实双机/CUDA parity 与 Qwen3-VL/Qwen3.5 仍关闭；下一票 `PT-PIPE-QW3.11` 收口 TCP peer epoch 和目标端 sidecar 执行边界。

> **2026-08-14 `PT-PIPE-QW3.11` 增量**：TCP 注册、Qwen3 peer proof、artifact ticket 和 network control 现已共享单调 live epoch，重连或撤销会使旧 proof/ticket 失效。目标节点新增 path-free consume 边界，目标 executor 本地解析 artifact，控制面只返回 SHA/bytes 和 hidden/KV shape/phase 摘要。网络/传输专项 `30 passed`；Qwen3.5/Qwen3-VL 尚未接入该执行面，真实双机/CUDA parity 与生产准入仍关闭。下一票 `PT-PIPE-QW3.12` 先完成目标 sidecar 真执行和消费后工件生命周期，再评估多模态组件接入。

> **2026-08-14 `PT-PIPE-QW3.12` 增量**：目标 sidecar executor 已接入 network consume，filtered assignment/hidden/KV artifact 在目标进程内解析，目标段 decode 使用本段 prefill KV；消费状态机支持同合同幂等、重复合同拒绝、执行异常/取消回收、epoch 变化 fencing 和目标重启遗留工件清理。目标输出经本地 SHA/bytes 复核后登记新的 path-free `output_reference`，不把路径、ticket 或 tensor 返回控制面。QW3 专项 `144 passed`；Qwen3.5/Qwen3-VL 仍未接入真实执行面，远端下一跳搬运、双机/CUDA parity 和生产准入继续关闭。下一票 `PT-PIPE-QW3.13` 先完成 path-free output reference 的远端下一跳传输，再评估多模态组件接入。

> **2026-08-14 `PT-PIPE-QW3.13` 增量**：源端 `output_reference` 现在可由认证目标 peer 按有界 offset 分块读取，每块复核输出 bytes/SHA；摘要变化、输出缺失、offset 越界和 peer 越界均 fail-closed 并使 reference 失效。目标端 `upload_chunks` 复用 `.part` offset/PATCH/commit 断线续传，二进制 output route 只返回 chunk 与摘要/offset/EOF headers；`transfer_registered_output` 完成 2/3 节点 path-free 下一跳搬运，连接失败保留 staging 可重试。专项 `36 passed`；持久 generation/KV ledger、Qwen3.5/Qwen3-VL 真实执行、双机/CUDA parity 和生产准入仍关闭。下一票 `PT-PIPE-QW3.14` 接入持久账本、多阶段重试与逐节点 release。

> **2026-08-14 `PT-PIPE-QW3.14` 增量**：用户主节点 SQLite `qwen3_network_ledger_v1` 已接入 network coordinator，保存 generation/phase、transfer descriptor/确认 offset、KV 摘要和 output lease/next transfer 关系，不保存路径、ticket 或 tensor。重启后旧 transfer/output reference 标为 `invalidated`，同 canonical contract 可重新 activate；prefill/decode 代际与 KV 投影、重复 lease/commit、跨调用断线续传、永久错误双端清理和逐节点 release 已覆盖。账本/网络专项 `44 passed`、QW3 全线 `153 passed`；Qwen3.5/Qwen3-VL 真实执行、双机/CUDA parity 和生产准入仍关闭。下一票 `PT-PIPE-QW3.15` 接入独立进程端到端 sidecar 自动串联。

> **2026-08-14 `PT-PIPE-QW3.15` 增量**：独立进程 helper 已完成 2/3 节点 path-free output reference 下一跳自动串联，覆盖 prefill/decode、KV generation 和 source release；目标端使用本地 synthetic executor 验证进程边界，未把测试结果冒充真实 Qwen3 模型执行。目标重启后 ledger 会使旧 output reference 失效并清理消费工件/staging。`.venv-test` 下网络专项 `28 passed`、QW3 全线 `156 passed`；此前模拟器使用主 Python 导致 `pytest-timeout` 缺失，已修复为优先调用 `.venv-test`。Qwen3.5/Qwen3-VL 真实执行、TTL/epoch/撤销等完整故障矩阵、双机/CUDA parity 和生产准入仍关闭。下一票 `PT-PIPE-QW3.16` 补故障矩阵并接入真实隔离 sidecar CPU smoke。

> **2026-08-14 `PT-PIPE-QW3.16` 增量**：receiver TTL 过期现在会同步清理 network coordinator 的活动 transfer，并在主节点 ledger 留下 `expired` 终态；peer registration epoch 前进会 fencing 活动 transfer 为 `invalidated`，独立进程验证 `.part`、lease 和旧引用均不残留。网络专项 `30 passed`、QW3 全线 `158 passed`。隔离 sidecar 对真实 Qwen3-4B assignment `[0,1]+embedding` 的 CPU smoke 已通过（13 tensors、约 0.98 GB、KV prefill/decode、thinking-off 预检、`full_model_materialized=false`）；这是受控 synthetic forward，不是 Qwen3.5/Qwen3-VL 或真实多节点质量证据。下一票 `PT-PIPE-QW3.17` 接入真实 network sidecar executor 进程链和断线续传。

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
| llama.cpp 源 | 锁定 `47e1de77…f89fcbe`，源码已枚举 `qwen3`、`qwen3vl`、`qwen35`、`gemma4` 架构 | 源码枚举不是运行时能力；**本机 `llama-cpp-python==0.3.28` 已实测支持 qwen3（2026-08-13：Q4_K_M 加载 + 生成 smoke 通过）**；qwen3vl/mmproj 与 qwen35 组合仍待实测后登记 |
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
