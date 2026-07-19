# PC 完整任务 Worker 接线准入报告

> 结论日期：2026-07-15
>
> 更新日期：2026-07-17
>
> 适用阶段：TC-N2.4 本机准入完成后，物理 PC 准入执行前
>
> 结论：**本机独立进程与实验开关门槛已通过；N2.4 尚未整体完成，不允许声明真实 Worker 已生产可用或默认启用**

## 1. 准入结论

当前任务链内核已经具备真实 PC Full Worker adapter 所需的稳定边界：Provider reservation、accepted attempt、真实 accept/execute timeout、lease epoch/renew、唯一 winner、迟到结果 fencing、受控重派、journal 恢复和按会话隔离的前端可观测状态均已有测试保护。Worker v1 保留兼容解码，修正后的 v2 是新 adapter 必选版本。

因此，TC-N2 可以开始实现真实传输和 Provider adapter，不需要再次修改 TaskGraph 核心状态机。

N2.4 已补充默认关闭的实验门、本机独立 Scheduler 进程集成和连续槽位回收测试。以下能力仍未实现，不得据此宣称“真实分布式任务链生产可用”：

- 未使用真实从节点验证模型一致性、超时、取消、重连和槽位回收；
- Android Full Worker 仍未接入，Android Lite 继续只作为请求端。

## 2. v1 兼容与 v2 接线协议

协议名为 `qlh.task_worker`，支持版本范围为 v1-v2。顶层 envelope 固定包含：

```text
protocol / version / message_type / message_id / sent_at_ms / payload
```

九类固定消息：

| 消息 | 方向 | 用途 |
|------|------|------|
| `hello` | Worker -> 主节点 | 上报版本范围、PC Full Worker 类型、引擎、模型和并发能力 |
| `hello_ack` | 主节点 -> Worker | 返回协商版本或稳定拒绝原因 |
| `stage_offer` | 主节点 -> Worker | 下发 Stage 输入、attempt、lease、epoch、输入摘要和精确模型身份 |
| `stage_accept` | Worker -> 主节点 | 接受或拒绝 reservation，并明确拒绝是否可重试 |
| `lease_renew` | 主节点 -> Worker | 延长当前 attempt 的 Coordinator 时间租约 |
| `stage_result` | Worker -> 主节点 | 返回带 attempt/epoch/Provider 身份和 SHA-256 的结果 |
| `stage_error` | Worker -> 主节点 | 返回稳定错误码和明确的 `retryable`，不传原始异常正文 |
| `stage_cancel` | 主节点 -> Worker | 协作取消指定 attempt |
| `stage_cancelled` | Worker -> 主节点 | 确认取消处理结果 |

golden fixture 位于 `tests/fixtures/task_worker_protocol_v1.json`，用于历史 v1 兼容。v2 在 `stage_offer` 强制增加 `model_identity`，在 `stage_accept` 强制增加 `retryable`；新 adapter 必须协商 v2，不得以错误码文本猜测重试，也不得仅依赖 hello 时的模型清单。

## 3. 协议与状态机映射

| Worker 协议动作 | TaskGraph/Provider 动作 | 不变量 |
|-----------------|-------------------------|--------|
| `hello` 通过 | 注册/刷新远端 Provider 能力 | 节点、引擎、模型 manifest 可解释且 Provider ID 唯一 |
| `stage_offer` | `reserve()` 后创建 accepted attempt | lease epoch 单调增加；模型身份必须与 offer 完全一致 |
| `stage_accept.accepted=false` | reservation 失败 | 只有 `retryable=true` 且 Stage 为纯 Stage 才能 fallback |
| `lease_renew` | `renew_stage_lease()` | 只能延长当前 running attempt 的同一 lease/epoch |
| `stage_result` | `submit_stage_result()` | 所有远端结果必须经过现有 fencing 闸门，不得直接写 Stage output |
| `stage_error` | attempt 失败/过期 | adapter 不得根据错误文本猜测是否重试 |
| `stage_cancelled` | 协作取消收敛 | 无论远端是否确认，主节点最终都必须释放 reservation |
| 断线/lease 到期 | 当前 attempt `expired` | 旧 epoch 后续返回只能记录拒绝，不能覆盖 winner |

adapter 不得创建第二套重试、winner 或聚合状态。传输只负责把协议消息映射到现有 Provider/Coordinator 接口。

## 4. 身份、模型与网络边界

- 设备进入现有 Tailscale 网络已经过网络所有者审核；TC-N2 沿用当前共享集群认证，不增加逐设备证书审批或重复配对流程。
- 集群共享密钥仍不得为空。它可以随受信设备配置下发，但日志、journal、fixture 和 API 响应不得包含密钥。
- Worker 必须声明 `worker_kind=pc_full_worker`；层拆分节点和 Android 请求端不能注册成完整任务 Worker。
- 模型准入按 `model_id + engine + format + revision + sha256` 判断。仅模型名称相同不代表可执行同一个 Stage。
- 主节点使用自己的接收时间和 lease deadline 做 fencing；不得信任 Worker 本地时钟决定结果是否过期。
- `message_id` 用于 adapter 去重，`attempt_id + lease_epoch + digest` 用于结果幂等；两者不能互相替代。
- Worker 不直接访问任务 journal、聊天历史数据库或外部 PostgreSQL；数据库离线不能阻断 Stage 执行。

## 5. 已通过门槛

| 门槛 | 状态 | 证据 |
|------|------|------|
| Workflow/Stage/Attempt 显式状态与 journal | 通过 | 重启恢复、终态不可逆和 sequence 测试 |
| Provider reservation 与释放 | 通过 | accept 超时、晚到释放、失败、取消、并行、close 和故障循环测试 |
| lease/epoch/winner fencing | 通过 | 执行超时、续租、断线、重复、迟到、错误身份和错误 digest 测试 |
| 可重试边界 | 通过 | 仅 `retryable + pure + fallback` 可重派 |
| API 可观测性 | 通过 | journal、恢复、重派、拒绝、winner、实际 Provider 投影测试 |
| UI 可观测性 | 通过 | session 隔离、`result_ready`、恢复失败、重派和结果拒绝格式化测试及生产构建 |
| Worker v1/v2 schema | 通过 | v1 golden 兼容；v2 模型身份、retryable 和负向校验 |
| 版本协商 | 通过 | 最高公共版本选择、无交集和非法范围拒绝测试 |
| 线上取消与租约续期 | 通过 | cancel/ack、自动续租、过期不可复活和 journal 同源测试 |
| Stage 消息去重/重放 | 通过 | 重复 offer 不重算、结果发送失败后原响应重放、ID 冲突拒绝 |
| 断线/超时槽位收敛 | 通过 | accept 超时、执行断线、lease 到期和 reservation 归零测试 |
| 打包声明 | 通过 | CPU/集显和 CUDA spec 均包含协议模块 |

2026-07-15 修复验收：任务链专项 `116 passed`，完整 Python 回归 `817 passed, 23 skipped`，前端 `7 passed` 且生产构建成功。

## 6. TC-N2.0 完成状态

- 传输复用现有 HMAC 认证、4 字节长度前缀 TCP；外层 `task_worker` 仅承载内层严格协议 envelope。
- 新 PC adapter 强制 v2；hello 绑定 TCP 注册身份、PC 类型和 client 角色。
- 主节点保存脱敏能力与健康投影；模型切换触发 generation 化能力刷新，分层模型不冒充完整模型。
- 消息 ID 有界去重、冲突拒绝、8 MiB 内层协议上限和异常隔离已有测试。
- 真实认证回环 TCP 已跑通注册、hello、hello_ack；Stage 消息保持硬拒绝。
- `adapter_connected=false` 和 `task_dispatch_enabled=false` 是当前准确状态。

N2.0 完成不代表下述生产启用门槛完成。尤其是远端 Provider、双进程 Stage、取消/lease、实机模型切换和统计仍待后续阶段。

2026-07-17 验收：N2.0/协议/TCP/调度器/API 联合专项 `402 passed`；完整 Python 回归 `857 passed, 23 skipped`；前端状态测试 `7 passed`。

## 7. TC-N2.1 完成状态

- `RemoteFullWorkerProvider` 只对应通过 v2 hello 的已认证 PC client，并固定为单受控槽位。
- reservation 校验 Stage 类型与完整模型身份；Worker 在收到 offer 时按当前模型再次校验。
- API 必须同时显式指定一个 Stage 和远端 Provider；默认模板永远保持本地，不自动选择远端。
- 显式远端 Stage 没有 fallback，且正确结果仍由既有 TaskGraph attempt/epoch/lease/winner 闸门唯一提交。
- API 指标只统计实际完成的远端 attempt；已连接但未参与的节点不会进入 `workers_used`。
- 认证 TCP 回环已跑通完整 offer/accept/result；错模型、错误 epoch、断线唤醒和槽位释放已有专项保护。
- N2.1 验收时，线上 cancel、lease renew、消息重放、双进程和实机测试尚未完成；前三项已在 N2.2 补齐。

N2.1 验收时的状态为 `worker_protocol.admission_state=n2_1_manual_stage_only`；当前状态见下一节。`adapter_connected=false` 和自动 `task_dispatch_enabled=false` 始终保持。

2026-07-17 验收：N2.1 联合专项 `460 passed`；打包声明与 Worker adapter 专项 `27 passed`；完整 Python 回归 `867 passed, 23 skipped`；前端状态测试 `7 passed`。

## 8. TC-N2.2 完成状态

- Coordinator 仅为声明支持续租的远端 Provider 自动发送 `lease_renew`，再通过既有 TaskGraph lease 校验和 journal 事件延长本地 deadline。
- Worker 对续租执行完整身份校验；原 lease 已过期时设置协作取消，拒绝用迟到 renew 复活 attempt。
- 远端 `stage_accept` 等待使用 `StageSpec.accept_timeout_seconds`，不再被 Provider 固定默认值覆盖。
- Coordinator 发送 `stage_cancel` 后先完成本地取消收敛；Worker 返回 `stage_cancelled` 并停止提交结果，取消确认丢失不阻塞 reservation 释放。
- Stage 请求和响应均按消息 ID 有界去重；相同 offer 重放已缓存的 accept/result，结果发送失败后不重新执行模型。
- Worker 将主节点发送的 lease 时长转换为本地单调时钟 deadline，不依赖两台设备的绝对系统时间一致；Coordinator 仍是结果过期判定的唯一权威。
- 断线会同时唤醒 Coordinator pending attempt 和 Worker active attempt；accept 超时、执行断线与 lease 到期均有槽位归零测试。
- N2.2 验收时自动 Provider 选择仍关闭，当时状态为 `n2_2_manual_stage_fault_handling`；当前状态见下一节。

2026-07-17 验收：Worker adapter 专项 `26 passed`；核心专项 `81 passed`；完整 Python 回归 `877 passed, 23 skipped`。本阶段未打包，也未进行真实双进程/物理从节点模型推理。

## 9. TC-N2.3 完成状态

- `task_graph_auto_remote` 是默认关闭的请求级开关；与 N2.1 手动 Stage/Provider 字段互斥，旧请求行为不变。
- 自动规划只允许两个候选 Stage 成为纯 Stage；aggregate 固定本地且不可自动重派。
- Worker 资格同时检查健康、空闲槽、Stage 类型和完整模型身份；候选按当前负载与稳定 ID 排序。
- 两个 Worker 分担两个候选；单 Worker 与主节点各承担一个候选，避免单槽远端串行化。
- retryable 远端失败按其他 Worker、最后本地的有界顺序 fallback；非 retryable、安全 fencing 拒绝和非纯 Stage 不重派。
- Stage 快照保留最后重派错误码，metrics 分离计划节点、实际参与节点、成功 Worker 和 fallback 原因。
- Provider 类型统一为 `remote_full_worker`；manual/auto 仅表示选择策略，不创建第二套执行 Provider。
- 前端已提供“仅本地 / 自动 Worker”控制，默认仅本地。
- N2.3 验收时状态为 `n2_3_controlled_auto_fallback`；当前 N2.4 状态见下一节。`adapter_connected=false` 和全局 `task_dispatch_enabled=false` 继续保持。

2026-07-17 验收：核心专项 `103 passed`；完整 Python 回归 `883 passed, 23 skipped`；前端 `8 passed` 且生产构建成功。本阶段未打包，也未进行真实双进程/物理从节点模型推理。

## 10. TC-N2.4 部分完成状态

- `QLH_TASK_WORKER_EXPERIMENTAL_ENABLED=false` 是默认值，远端任务调度不能因 Worker 已连接而自行开启。
- 手动远端请求在实验门关闭时明确拒绝；自动请求安全回退本地并记录稳定原因。
- 公共状态分离“控制面连接”“实验调度可用”和“生产 adapter 准入”；当前 `adapter_connected=false`、`task_dispatch_enabled=false`。
- 独立子进程使用真实 `Scheduler + TCPClient`，主进程使用真实 `TCPServer + RemoteFullWorkerProvider + TaskGraphCoordinator`，通过 HMAC TCP 完成单 Stage。
- 子进程结果 PID 与主进程不同，不共享模型 executor、TaskGraph、Provider 或进程内锁。
- 30 轮成功/失败循环后 reservation 始终归零；原有取消、续租、断线和消息重放测试继续通过。
- 实验开关已同时覆盖 Worker hello 和 Worker 端 offer 执行；关闭时不再上报 Full Worker 能力，旧会话也会稳定拒绝 Stage。
- accept 响应丢失或超时后主节点会补发 `stage_cancel`；cancel/lease renew 通过有界后台发送队列发出，不阻塞 Coordinator 锁。
- 取消请求先写 journal，再调用 Provider cancel；持久化失败不会留下仅内存可见的取消副作用。
- TCP 双端在读取正文前校验普通包和张量包长度；手动远端 Stage 在工作流创建前完成精确模型身份预检。
- 前端自动 Worker 控制受实验门状态约束；门关闭时按钮禁用。

当前准确状态为：实验门关闭时 `n2_4_experiment_disabled`；门打开但无 Worker 时 `n2_4_experiment_enabled_not_connected`；门打开且 Worker 已连接时 `n2_4_experimental_physical_validation_pending`。

2026-07-17 BUG 审查修复后本机验收：任务图/Worker/API/TCP 专项连续两轮 `138 passed`；完整 Python 回归 `893 passed, 23 skipped`；前端 `8 passed` 且生产构建成功。本阶段未打包。

仍需物理 PC 完成真实 Qwen/DeepSeek 模型、错/缺/切换模型、Tailscale 断线重连、长时性能和安装包双端测试，N2.4 才能整体完成。

## 11. TC-N2 生产启用前硬门槛

以下项目全部通过前，API 中 `worker_protocol.adapter_connected` 必须保持 `false`：

1. 冻结传输 framing、单消息大小、连接关闭和背压规则。
2. 实现 `RemoteFullWorkerProvider`，并确保所有结果只进入 `submit_stage_result()`。
3. 实现 hello/hello_ack、消息 ID 去重、心跳/断线和 reservation 清理。
4. 使用两个本机独立进程跑通 full inference Stage，不共享模型对象或内存状态。
5. 注入 accept 超时、执行超时、断线、重复消息、旧 epoch、取消和主节点重启。
6. 使用真实 PC 从节点验证正确模型、错模型、缺模型、模型切换和并发槽位。
7. UI 和任务统计如实展示实际参与节点；fallback 到主节点时不得继续标记远端参与。
8. 功能默认关闭；关闭时不得影响现有本地聊天、PyTorch 层流水线或 Android 请求路径。

## 12. 建议实施顺序

1. **TC-N2.0 传输适配（已完成）**：只实现 hello、能力查询和健康状态，不接任务。
2. **TC-N2.1 单 Stage 手动分发（已完成）**：关闭自动 fallback，验证身份、模型和 result fencing。
3. **TC-N2.2 取消与故障（已完成）**：接入 cancel、lease renew、断线和消息去重。
4. **TC-N2.3 自动 Provider 选择（已完成）**：只对显式纯 Stage 开启受控 fallback。
5. **TC-N2.4 实机准入（部分完成）**：本机进程、统计/UI 和关闭开关已通过；物理模型、网络、压测和安装包待执行。

TC-N2 以 v2 为接线基线，不修改其核心状态语义。若传输实现发现必须修改 attempt、lease 或 winner 规则，应先停止接线并升级协议，而不是在 adapter 内增加旁路。
