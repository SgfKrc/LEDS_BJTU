# MODEL-FLEET M2 运行时侧车报告（2026-08-08）

> **状态**：历史报告，使命完成（2026-08-08 登记）；DeepSeek 7B GGUF 真实加载 `ready`，Safetensors 按资源门 `resource_rejected`
>
> **生命周期**：废弃（待手动删除）
>
> **替代入口**：结论与验收矩阵已并入《一键模型部署与自治集群远期计划》§16（M2.2）与《总体下一步计划》L4-MODEL-FLEET 条目；确认无必要引用后可手动删除本文件

## 1. 结论

M2 runtime sidecar 基础能力已完成。本轮没有访问网络，也没有使用 `127.0.0.1:7897` 或其他代理：

- DeepSeek-R1-Distill-Qwen-7B GGUF 工件完成逐文件 SHA256 复核、独立进程真实权重加载和 tokenizer 探针，登记为 `ready`；
- 同模型 Safetensors 工件进入同一侧车协议，但本机可用 RAM/VRAM 不满足加载安全余量，登记为 `resource_rejected/insufficient_memory`，未冒险触发 OOM 或系统换页；
- 子进程崩溃、超时、协议身份错配、输出越界、缺 blob 和资源不足均不会终止 control-svc；终态写入 SQLite `artifact_runtime_checks`，不修改不可变 artifact manifest；
- 试加载结束后运行时硬链接视图为 0、残留侧车进程为 0。

这证明当前主机上的 DeepSeek 7B GGUF 工件可由 `llama_cpp` 运行，不证明 Safetensors 已加载成功，也不等于跨 PC 分发、激活或完整推理链已经验收。

## 2. 实现

| 组件 | 作用 |
|---|---|
| `src/inference_service/model_runtime_sidecar.py` | 单请求/单结果 JSON 协议；强制 `trust_remote_code=false` 和离线加载；GGUF 实际加载、Safetensors 资源预检与可加载路径 |
| `control/src/data/model-runtime-sidecar.ts` | 独立进程监管、240s 默认超时、输出上限、崩溃回收、协议身份校验、同盘硬链接运行时视图与清理 |
| `control/src/data/artifact-runtime-repository.ts` | 按 `artifact_id + node_id + runtime_profile` 保存本机运行时事实 |
| `control/src/data/model-runtime-check.service.ts` | 运行 sidecar 并原子登记终态 |
| `control/src/model-fleet-runtime-check.ts` | `npm run model-fleet:runtime-check` 本地执行入口 |
| `control/src/data/sqlite-store.ts` | schema v2 新增 `artifact_runtime_checks`；已有 v1 数据库事务化升级且保留原数据 |

子进程环境显式设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，并移除代理环境变量。工件视图只在 `<model-store>/runtime/check-*` 创建硬链接，既不复制大文件，也不修改 blob；准备失败和进程结束都会清理。

## 3. DeepSeek 7B 实测

运行事实源：`build/model-fleet/runtime-check-20260808.sqlite3`（本机生成，不进入 Git）。

| 格式 | artifact_id | runtime | 结果 | 关键证据 |
|---|---|---|---|---|
| GGUF | `sha256:ca26e10ae3743995f880d2082a2ac5e316e8a8a755e026d182e975f97e92127a` | `llama-cpp-python/0.3.28`、`llm-cpu-v1` | `ready` | 4,683,073,248-byte blob；SHA256 复核 3,634ms；实际加载 2,720ms；`qwen2`、`general.file_type=15`、`Q4_K_M`、训练上下文 131072；token probe=3 |
| Safetensors | `sha256:f1b0869a7d85b90ed2a117c1f8bc3bd32e6de5a9aee62b77fb050edf9d064bfe` | `pytorch_transformers`、`llm-cuda-v1` | `resource_rejected` | 权重 15,231,271,850 bytes；安全需求 17,595,895,384；可用 RAM 7,282,683,904、可用 VRAM 7,451,181,056，扣除保留量后合计 12,586,381,312；`trust_remote_code=false` |

GGUF 执行命令：

```powershell
cd control
npm run model-fleet:runtime-check -- `
  --manifest ../build/model-fleet/model-store-20260808/manifests/migration/deepseek-r1-distill-qwen-7b-gguf/builtin-20260808.json `
  --model-store ../build/model-fleet/model-store-20260808 `
  --sqlite ../build/model-fleet/runtime-check-20260808.sqlite3 `
  --node-id local-rtx4060 --timeout-ms 300000
```

Safetensors 使用相同命令，仅替换 manifest 路径。CLI 对 `ready` 返回 0，对 `resource_rejected/load_failed` 返回 2，便于 CI 或部署编排区分。

## 4. 元数据修正

旧迁移 manifest 把文件名为 `Q4_K_M` 的 GGUF 记录成了 `q8_k`。实际 GGUF 元数据为 `general.file_type=15`，固定 llama.cpp 头文件和当前 loader 均定义为 `Q4_K_M`。本轮已修正 `ModelInspector` 的完整 `llama_ftype` 映射，并支持 `qwen2.context_length` 等架构专属上下文字段。

旧 manifest 作为历史不可变迁移事实未被原地改写；本机运行时记录保存真实的 `q4_k_m/131072`。后续重新索引必须生成新 tag/manifest，不能静默覆盖旧清单。

## 5. 质量门

- `npm run build`：通过；
- `control/test/model-runtime-sidecar.e2e-spec.ts`：6/6，通过；
- M2 工件/导入/sidecar 三套专项：19/19，通过；
- control-svc 全量：26 套、257/257，通过；
- Python MODEL-FLEET schema：25/25，通过；
- `python -m py_compile src/inference_service/model_runtime_sidecar.py`：通过；
- `npm audit --omit=dev`：0 vulnerability；
- `git diff --check`：通过。

## 6. 后续状态

原定 import/pull `adapting`、状态 query/retry/invalidate、运行时指纹和部署过滤已在后续批次完成，见 [M2 运行时准入接线报告](MODEL-FLEET-M2运行时准入接线报告-20260808.md)。下一本地阶段为模型管理控制面/UI，仍不需要代理。

DeepSeek 7B Safetensors 只在满足至少 17,595,895,384 bytes 可用组合余量的主机上重试，或另立量化工件；本机 `resource_rejected` 不得改写为 `ready`。真实模型权重完整 pull、gated 账号和代理掉线验证仍是后续网络门，进入这些任务前再启用代理。
