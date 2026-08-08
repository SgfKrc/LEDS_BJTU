# QLH 模型部署契约 Schema（M0 冻结，2026-08-06）

本目录是 [一键模型部署与自治集群远期计划](../docs/一键模型部署与自治集群远期计划.md)
§16 M0「冻结契约与事实源」的产物：六个 JSON Schema（draft-07）及
`model-fleet-compatibility.json` 作为模型生命周期、迁移和集群档案的**跨语言契约**。
Python（jsonschema）与 TS（ajv）两侧使用同一批 fixture 验证，结果必须一致。

## Schema 清单

| 文件 | 用途 | 关键状态机 |
|---|---|---|
| `artifact-manifest.schema.json` | 不可变模型工件清单（内容寻址） | `artifact_id` 必须为 `sha256:<64 hex>` |
| `pull-job.schema.json` | 下载/校验/注册任务 | `queued → resolving → downloading → verifying → adapting → registered`；终止态 `failed / cancelled / rejected / quarantined / rolled_back` |
| `deployment.schema.json` | 单节点部署记录 | `planned → preparing → distributing → ready → active`；`failed / partial / rolled_back` |
| `cluster-profile.schema.json` | Tailscale 主节点档案 | `active / pending_verification / unreachable`；只存 `key_ref` 不存明文 |
| `migration-map.schema.json` | 旧事实源到 SQLite 的一次性映射 | source_id 幂等；未知根字段拒绝 |
| `fetcher-progress.schema.json` | 独立 fetcher 结构化事件 | `started → resolved → progress → verified → completed`；终止事件不可续写 |

`model-fleet-compatibility.json` 是版本演进和未知字段策略的机器可读索引；它本身不是业务
payload schema，不替代六个文件的字段校验。

## 能力枚举（禁止用 model_type 推导能力）

`artifact-manifest` 的 `capabilities` 是**冻结枚举**（`additionalProperties: false`），
当前仅四个布尔字段：

- `full_worker`：可承担完整 Worker 推理。
- `pytorch_layer_pipeline`：可参加 PyTorch hidden-state 层间流水线（必须全节点
  同 artifact digest，不能仅凭 model family 宣称）。
- `llama_cpp`：当前锁定 commit 的 llama.cpp 可加载该工件。
- `task_stage`：可作为任务链 Stage 参与。

规则：

1. 调度与运行只信 `artifact_id` 与 `capabilities`，**禁止**从 `model_type` /
   `family` / `quantization` 推导任何分布式能力。
2. 新增能力必须同时更新本 schema、两语言验证测试与本文档，并提升
   `schema_version` 或走兼容演进（见下）。
3. `additionalProperties: false` 意味着未知能力键会被拒绝——这是有意的
   fail-closed 行为，不是疏漏。

## 演进规则

- `schema_version` 当前为 1。**破坏性变更**（删除必填字段、改枚举值、
  收紧 pattern）必须递增版本号，并保留旧版本迁移器。
- **兼容演进**（开放对象新增可选字段、非身份数值范围放宽、纯描述澄清）允许在原版本内进行；
  枚举成员变化即使是扩展也视为破坏性变更，必须提升版本。
- 未知根字段策略由兼容清单锁定：artifact/pull/deployment/cluster profile 当前允许扩展，
  migration-map/fetcher-progress 当前拒绝扩展。
- 任何变更后必须同时通过：
  - `python -m pytest -q tests/test_model_fleet_schemas.py`（25 用例）
  - `cd control && npm test -- --runTestsByPath test/schemas.e2e-spec.ts`（25 用例）

## Fixture 约定

`fixtures/model-fleet/` 命名规则：`<kind>-valid*.json` 必须通过校验，
`<kind>-invalid-<reason>.json` 必须被拒绝；状态时序使用
`fetcher-sequence-valid*.json` / `fetcher-sequence-invalid-<reason>.json`，除 JSON Schema
外还验证 job_id 一致性、合法迁移、终止态封闭和进度单调性。新增 schema 或状态边界时必须成对添加
fixture，并在 Python/TS 同一变更中更新。
