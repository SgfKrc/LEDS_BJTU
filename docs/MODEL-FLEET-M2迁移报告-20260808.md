# MODEL-FLEET M2 内置目录迁移报告（2026-08-08）

> **状态**：历史报告，使命完成（2026-08-08 登记）
>
> **生命周期**：废弃（待手动删除）
>
> **替代入口**：M2 结论与验收矩阵已并入《一键模型部署与自治集群远期计划》§16 与《总体下一步计划》L4-MODEL-FLEET 条目；确认无必要引用后可手动删除本文件

## 1. 结论

`build/model-fleet/catalog-seed.json` 中的 8 个内置模型条目已完成一次真实 catalog 迁移检查。catalog 共展开 10 个预期来源（两个 `both` 条目各包含 Safetensors 与 GGUF）：

- 2 个模型 ID、4 个实际来源迁移成功，共提交 46 个 blob 和 4 份 manifest；
- 6 个模型 ID 的本地源路径不存在，按 `missing` 登记，未创建伪工件、未进入 quarantine；
- 最终报告中 `failed_sources=0`、`quarantined_sources=0`；首轮两个 GGUF 因 1 MiB 元数据预算过小被隔离，解析预算保持有界并提升到 64 MiB 后重试成功；
- 本次完成的是 M2“8 条目实际运行并登记报告”证据，不代表 8 个模型权重已全部具备，也不代表 runtime sidecar 试加载已经通过。

最终本地报告：`build/model-fleet/catalog-migration-20260808-final.json`。首轮与 GGUF 重试报告分别为 `build/model-fleet/catalog-migration-20260808.json`、`build/model-fleet/catalog-gguf-retry-20260808.json`。`build/` 是本机生成目录，不作为发布包或 Git 工件。

## 2. 条目结果

| model_id | 格式 | 状态 | artifact_id / 原因 |
|---|---|---|---|
| `qwen-1_8b` | Safetensors | succeeded（inspection only） | `sha256:8c3f97c3c8f64ff8d908c37426eed87537c162f6a40b2caf5b3ca27613d6f090` |
| `qwen-1_8b` | GGUF | succeeded（inspection only） | `sha256:9511b2bcf452a96bbdc1e197c51c661507f8a11db05b7fb86776c2e995090e87` |
| `qwen2.5-7b` | Safetensors | missing | `models/qwen2.5-7b-instruct` 不存在 |
| `qwen2.5-14b` | Safetensors | missing | `models/qwen2.5-14b-instruct` 不存在 |
| `qwen2.5-7b-gguf` | GGUF | missing | `models/qwen2.5-7b-instruct-Q4_K_M.gguf` 不存在 |
| `deepseek-r1-distill-qwen-1.5b` | Safetensors | missing | `models/deepseek-r1-distill-qwen-1.5b` 不存在 |
| `deepseek-r1-distill-qwen-7b` | Safetensors | succeeded | `sha256:f1b0869a7d85b90ed2a117c1f8bc3bd32e6de5a9aee62b77fb050edf9d064bfe` |
| `deepseek-r1-distill-qwen-7b` | GGUF | succeeded | `sha256:ca26e10ae3743995f880d2082a2ac5e316e8a8a755e026d182e975f97e92127a` |
| `deepseek-r1-distill-qwen-14b` | Safetensors | missing | `models/deepseek-r1-distill-qwen-14b` 不存在 |
| `deepseek-r1-distill-qwen-32b` | Safetensors | missing | `models/deepseek-r1-distill-qwen-32b` 不存在 |

Qwen 1.8B 的 `model_type/general.architecture=qwen` 不在当前自动部署白名单，两个 manifest 均保持能力位全 false；DeepSeek 7B Safetensors 判定为 `qwen2/full_worker=true`，GGUF 判定为 `qwen2/llama_cpp=true`。这符合未知或旧架构 fail-closed 的准入规则。

## 3. 执行与校验

执行入口：

```powershell
cd control
npm run build
node dist/model-fleet-import.js --catalog ..\build\model-fleet\catalog-seed.json `
  --store ..\build\model-fleet\model-store-20260808 `
  --report ..\build\model-fleet\catalog-migration-20260808.json `
  --namespace migration --tag builtin-20260808
```

真实权重文件通过流式文件复制和 8 MiB 分块 SHA256 计算进入 staging/blob，避免将 6-9 GB 分片整体读入内存。校验结果：

- 4 份 manifest 与最终报告中的 artifact_id 一致；成功来源文件数为 `18 + 1 + 26 + 1 = 46`；
- 两个 GGUF 源文件 SHA256 与其 `.sha256` sidecar 一致，源文件大小与修改时间未变化；
- 内容存储当前约 `30,766,722,857` bytes，其中包含首轮两个 GGUF quarantine 副本；未执行破坏性清理，G 盘复核剩余约 `9,411,092,480` bytes；
- `npm run build` 通过，M2 两套专项 13/13 通过，control-svc 全量 21 套、226/226 通过。

## 4. 下一步

1. M2 runtime sidecar 对 DeepSeek 7B 的 Safetensors 与 GGUF 各做一次隔离试加载；Qwen 1.8B 在专用 legacy adapter/远程代码信任策略完成前保持 inspection only。
2. M3 Windows 安全门已在后续批次完成，见 [M3 凭据、代理与 gated 许可安全门](MODEL-FLEET-M3安全门报告-20260808.md)；真实 Hub/代理/gated 账号仍待外部环境。
3. 六个缺失模型仅在权重真实到位后重跑 catalog；不得把 `missing` 改写成已迁移或已部署。
