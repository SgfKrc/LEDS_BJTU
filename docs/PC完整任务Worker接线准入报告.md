# PC 完整任务 Worker 接线准入报告

> 结论日期：2026-07-14
>
> 适用阶段：TC-N1 完成后，TC-N2 真实 PC Full Worker adapter 开始前
>
> 结论：**允许开始 adapter 开发；不允许声明真实 Worker 已可用或默认启用**

## 1. 准入结论

当前任务链内核已经具备真实 PC Full Worker adapter 所需的稳定边界：Provider reservation、accepted attempt、lease epoch、唯一 winner、迟到结果 fencing、受控重派、journal 恢复和前端可观测状态均已有测试保护。Worker v1 消息格式和版本协商也已由 golden fixture 固定。

因此，TC-N2 可以开始实现真实传输和 Provider adapter，不需要再次修改 TaskGraph 核心状态机。

以下能力仍未实现，不得据此宣称“真实分布式任务链可用”：

- 未在现有 TCP 连接或隔离通道上注册 Worker 消息处理器；
- 未实现远端 `ExecutionProvider` adapter、消息去重表和断线检测；
- 未完成主从两个独立进程的 offer/accept/result 集成测试；
- 未使用真实从节点验证模型一致性、超时、取消、重连和槽位回收；
- Android Full Worker 仍未接入，Android Lite 继续只作为请求端。

## 2. 已冻结的 v1 协议

协议名为 `qlh.task_worker`，当前唯一支持版本为 v1。顶层 envelope 固定包含：

```text
protocol / version / message_type / message_id / sent_at_ms / payload
```

v1 固定消息：

| 消息 | 方向 | 用途 |
|------|------|------|
| `hello` | Worker -> 主节点 | 上报版本范围、PC Full Worker 类型、引擎、模型和并发能力 |
| `hello_ack` | 主节点 -> Worker | 返回协商版本或稳定拒绝原因 |
| `stage_offer` | 主节点 -> Worker | 下发 Stage 输入、attempt、lease、epoch 和输入摘要 |
| `stage_accept` | Worker -> 主节点 | 接受或拒绝 reservation |
| `lease_renew` | 主节点 -> Worker | 延长当前 attempt 的 Coordinator 时间租约 |
| `stage_result` | Worker -> 主节点 | 返回带 attempt/epoch/Provider 身份和 SHA-256 的结果 |
| `stage_error` | Worker -> 主节点 | 返回稳定错误码和明确的 `retryable`，不传原始异常正文 |
| `stage_cancel` | 主节点 -> Worker | 协作取消指定 attempt |
| `stage_cancelled` | Worker -> 主节点 | 确认取消处理结果 |

golden fixture 位于 `tests/fixtures/task_worker_protocol_v1.json`。v1 内禁止静默增加字段；新增字段需要新协议版本或先定义明确的兼容策略。

## 3. 协议与状态机映射

| Worker 协议动作 | TaskGraph/Provider 动作 | 不变量 |
|-----------------|-------------------------|--------|
| `hello` 通过 | 注册/刷新远端 Provider 能力 | 节点、引擎、模型 manifest 可解释且 Provider ID 唯一 |
| `stage_offer` | `reserve()` 后创建 accepted attempt | lease epoch 由主节点生成且单调增加 |
| `stage_accept.accepted=false` | reservation 失败 | 只有稳定错误明确可重试且 Stage 为纯 Stage 才能 fallback |
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
| Provider reservation 与释放 | 通过 | 成功、失败、取消、并行、close 和故障循环测试 |
| lease/epoch/winner fencing | 通过 | 超时、断线、重复、迟到、错误身份和错误 digest 测试 |
| 可重试边界 | 通过 | 仅 `retryable + pure + fallback` 可重派 |
| API 可观测性 | 通过 | journal、恢复、重派、拒绝、winner、实际 Provider 投影测试 |
| UI 可观测性 | 通过 | `result_ready`、恢复失败、重派和结果拒绝格式化测试及生产构建 |
| Worker v1 schema | 通过 | 全部九类 golden message 往返和负向校验 |
| 版本协商 | 通过 | 最高公共版本选择、无交集和非法范围拒绝测试 |
| 打包声明 | 通过 | CPU/集显和 CUDA spec 均包含协议模块 |

## 6. TC-N2 生产启用前硬门槛

以下项目全部通过前，API 中 `worker_protocol.adapter_connected` 必须保持 `false`：

1. 冻结传输 framing、单消息大小、连接关闭和背压规则。
2. 实现 `RemoteFullWorkerProvider`，并确保所有结果只进入 `submit_stage_result()`。
3. 实现 hello/hello_ack、消息 ID 去重、心跳/断线和 reservation 清理。
4. 使用两个本机独立进程跑通 full inference Stage，不共享模型对象或内存状态。
5. 注入 accept 超时、执行超时、断线、重复消息、旧 epoch、取消和主节点重启。
6. 使用真实 PC 从节点验证正确模型、错模型、缺模型、模型切换和并发槽位。
7. UI 和任务统计如实展示实际参与节点；fallback 到主节点时不得继续标记远端参与。
8. 功能默认关闭；关闭时不得影响现有本地聊天、PyTorch 层流水线或 Android 请求路径。

## 7. 建议实施顺序

1. **TC-N2.0 传输适配**：只实现 hello、能力查询和健康状态，不接任务。
2. **TC-N2.1 单 Stage 手动分发**：关闭自动 fallback，先验证身份、模型和 result fencing。
3. **TC-N2.2 取消与故障**：接入 cancel、lease renew、断线和消息去重。
4. **TC-N2.3 自动 Provider 选择**：只对显式纯 Stage 开启受控 fallback。
5. **TC-N2.4 实机准入**：完成统计、UI、压测和关闭开关后再允许实验启用。

TC-N2 不修改 v1 核心状态语义。若传输实现发现必须修改 attempt、lease 或 winner 规则，应先停止接线并重新审查协议，而不是在 adapter 内增加旁路。
