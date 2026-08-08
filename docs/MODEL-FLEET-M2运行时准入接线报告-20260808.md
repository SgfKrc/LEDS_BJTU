# MODEL-FLEET M2 运行时准入接线报告（2026-08-08）

> **状态**：历史报告，使命完成（2026-08-08 登记）；本地 import/pull/runtime API/部署过滤已接入同一运行时事实源
>
> **生命周期**：废弃（待手动删除）
>
> **替代入口**：结论与验收矩阵已并入《一键模型部署与自治集群远期计划》§16（M2.3）与《总体下一步计划》L4-MODEL-FLEET 条目；确认无必要引用后可手动删除本文件

## 1. 结论

M2.3 运行时准入接线已完成。本轮只读取本地工件，没有访问网络，也没有使用 `127.0.0.1:7897` 或其他代理：

- interactive local import 在 manifest 提交后自动运行 sidecar，响应同时返回静态导入报告和节点运行时终态；
- pull job 在 `verifying → adapting → registered` 的 `adapting` 阶段运行 sidecar。`registered` 表示已校验工件已保留，是否可部署只看 `runtime_check.status`；`resource_rejected/load_failed` 不会删除有效 blob，也不会伪报可运行；
- 新增运行时状态查询、显式重试和失效 API。重试/导入/失效仅允许 loopback，请求不能替远端节点写入本机检查结果；
- 每次检查生成包含 OS、CPU、总内存、loader/profile 和相关 GPU 信息的 SHA256 运行时指纹；版本或硬件画像变化可显式标记 `stale`，指纹不匹配也会被部署过滤拒绝；
- 部署 prepare 同时要求 digest、capability、`runtime_profile`、`ready` 状态和指纹一致；prepare 后检查失效时，activate 再次校验并拒绝 TOCTOU 激活。

本阶段完成的是本地控制面和模拟部署的准入闭环，不代表真实 scheduler、远端 Worker、跨 PC 分发或 Safetensors 已通过。

## 2. API 与语义

| 方法与路径 | 作用 | 写入边界 |
|---|---|---|
| `GET /models/runtime-checks` | 按 artifact/node/profile/status 查询当前运行时事实 | 只读 |
| `POST /models/runtime-checks/retry` | 用 namespace/name/tag 对本机 manifest 显式重试 | loopback；`node_id` 必须等于 `QLH_NODE_ID`/`local` |
| `DELETE /models/runtime-checks` | 按 artifact 或 node 将记录标记为 `stale`，保留审计字段 | loopback；禁止无范围全量失效 |
| `POST /models/imports` | 本地路径导入、静态检查、manifest、sidecar 和状态登记 | loopback；源文件只读 |
| `POST /models/deployment-simulations` | 创建带 `runtime_profile` 和节点 runtime fingerprint 的计划 | prepare/activate 都重新读取 runtime check |

| 状态 | 含义 | 可部署 |
|---|---|---|
| `ready` | 当前节点/profile/指纹上真实 loader 已通过 | 是，仍需 digest/capability/available 同时通过 |
| `resource_rejected` | 工件有效，但当前资源安全余量不足 | 否 |
| `load_failed` | loader、协议、超时、崩溃或工件视图失败 | 否 |
| `stale` | loader/硬件/配置变化后显式失效 | 否，必须重试 |

部署拒绝码新增 `runtime_unchecked`、`runtime_not_ready`、`runtime_context_changed` 和 `runtime_admission_changed`。最后一项覆盖 prepare 后记录变化，禁止旧 `ready` 越过 activate。

## 3. 真实本地纵切

使用上一阶段的 DeepSeek-R1-Distill-Qwen-7B GGUF 工件：

- artifact：`sha256:ca26e10ae3743995f880d2082a2ac5e316e8a8a755e026d182e975f97e92127a`；
- `POST /models/runtime-checks/retry`：HTTP 200，`ready`，`llama-cpp-python/0.3.28`，实际加载 2,660ms；
- runtime fingerprint：`sha256:dc0568409aad328716e5cbdf3b07f6f189bce3ae14caf5ea30799c6b461a03d9`；
- API 总耗时约 6.7 秒，包含 4.68GB blob 的 SHA256 复核、loader 和状态写入；
- 同一 artifact/profile/fingerprint 创建单节点部署计划后，prepare=`ready`、error=`null`；
- 执行后残留 runtime view=0、sidecar process=0。

该记录保存在本机 `build/model-fleet/runtime-check-20260808.sqlite3`，不进入 Git。

## 4. 实现位置

- `control/src/modules/models/runtime-admission.controller.ts`：四个本地运行时/导入 API；
- `control/src/data/model-import-admission.service.ts`：interactive import 后置准入；
- `control/src/data/model-runtime-check.service.ts`：manifest ref、运行时指纹和状态登记；
- `control/src/data/artifact-runtime-repository.ts`：组合键查询、过滤和 `stale` 失效；
- `control/src/data/pull-job-executor.ts`：`adapting` 阶段运行 sidecar，保留非 ready 工件；
- `control/src/data/deployment-simulator.ts`：prepare/activate 双重 runtime admission；
- `schemas/pull-job.schema.json`、`schemas/deployment.schema.json`：兼容新增可选运行时字段。

## 5. 质量门

- `npm run build`：通过；
- runtime admission 相关 6 套专项：53/53，通过；
- control-svc 全量：27 套、264/264，通过；
- Python MODEL-FLEET schema：25/25，通过；
- 生产依赖审计：0 vulnerability；
- `git diff --check`：通过。

新增证据覆盖 sidecar AbortSignal 取消回收、API retry/query/invalidate/import、pull 非 ready 工件保留、未检查/非 ready/指纹变化拒绝，以及 prepare 后失效阻止 activate。

## 6. 下一阶段

进入本地模型管理控制面/UI，不需要代理：

1. 提供 artifact/manifest/runtime 汇总读模型，让页面能区分“已入库”“当前节点可运行”“资源不足”“需重试”；
2. 接入本地导入、运行时重试/失效、pull 任务与进度查看；
3. 展示并编辑 Git 风格用户代理，明确优先级 `QLH_HTTP_PROXY > user > direct`，环境代理只读；
4. UI 自动化只走本地 fixture/mock，不执行真实下载。真实模型权重 pull 或代理掉线门开始前再启用代理。
