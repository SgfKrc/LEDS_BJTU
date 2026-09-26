# 2026-09-24 至 2026-09-26 R-Rx / R-R3 提交复核

复核范围：该日期区间的 git 提交、`R-Rx` relay/HA/RPC 工作、R-R3 双机复验及其 #28 修复尝试；子 agent 只读完成，未修改工作树。定向测试结果：`58 passed`、`19 passed`。

## 结论

### P0

- **R-R3 的成功证据不是自动两节点层计划。** `distributed_used=true` 对应的是显式 `execution_mode=task_graph` 下的 `task_graph_remote_auto + Full Worker`。两台 PC 的层流水线仍因 Full Worker opt-out 与“至少两个可用 PC”条件不满足而拒绝。涉及 `9d57cc4`、`d812185`，代码见 `src/scheduler_pipeline.py`、`src/pipeline_capacity.py`。
- **#28 代码修复仍失败。** `8fdecf4` 的 A/B 方案经 `0515063`、`db443dc` 实机证实会触发 hello/release 死循环，已回退；当前 R-R3 依赖“先加载整模、后连接集群”的人工顺序，不能宣称代码级修复完成。现行双源状态分别见 worker 实时 capabilities 与 master `_task_worker_full_model_ids()`。

### P1

- **y700 Termux 动态 SSH 端口未接入。** `scripts/ha_crosshost_physical_smoke.py` 的相关 SSH 路径未统一使用显式端口，动态端口会走默认 22 或超时，现有测试覆盖的是动态 ADB 端口而非 Termux SSH。
- **R-R2 全部重试失败时可能遗留最后一个 SSH worker。** `_connect_with_retry()` 把最后进程挂在异常上，但 `_run_surface()` 没有消费并回收该进程；测试只覆盖 helper，不覆盖上层全失败清理。
- **R-R6 远端后端 ready 判定不足。** `scripts/llama_pc_rpc.py` 目前主要检查远端进程和隧道端口，不做 RPC 协议握手/ready 探针，进程未监听或已失活会延迟到请求阶段才暴露。
- **Worker 环境一致性不在准入合同中。** R-R3 只验证模型文件摘要；capabilities 未绑定 Python/Torch/Transformers/运行时构建和加载配置指纹，模型相同不等于数值运行环境相同。
- **R-R2 物理弱网仍未验收。** 5% 丢包下多轮重试仍失败或长时间卡住；目前只能称代码侧重试已实现，不能称物理可用性闭环。

### P2

- `task_graph_template` 在 `execution_mode=auto` 下仍可能被静默忽略并进入层流水线，已知问题 #29。
- R-R3 的 `distributed_used_proof.json` 没有生成/校验脚本且未纳入 Git 跟踪，人工证据不可重复生成。

## 处理边界

本轮未直接修改 R-Rx/R-R3 分布式路径。下一组修复应先处理动态 SSH 端口与 RPC ready 探针，再设计 #28 的 master 侧时效/幂等状态方案；不能重新启用已被真机否掉的 worker A/B 方案。

## Reasonix 增量复核

第二份只读复核确认窗口内提交没有“代码不存在却宣称完成”的假完成项，且 A/B 回退无残留。新增风险如下：

- **P1：#28 正反馈触发机制仍存于整模变更 refresh 路径。** `src/scheduler_task_worker.py` 的 refresh generation/hello-ack 重发逻辑及 `api_server.py`、`api/routes_models.py`、`inference_service/engine_host.py` 的整模变更调用点仍在。当前没有层段路径主动 refresh，因此风险未被日常路径触发；但层段与 release 过渡期若叠加整模变更，理论上仍可重放 hello/release 正反馈。需 master 侧幂等快照/角色状态设计，不能重新启用已被真机否掉的 A/B。
- **P2：B5 行文滞后。** R-R3 源码同步已完成，但验收清单仍写“等待 R-R3 同步源码”；本轮改为“同步已完成，双机身份重置本身待复验”。
- **P2：R-R6 `passed` 不覆盖远端 RPC 进程存活。** 现有证据虽与文档判据一致，但应在后续 ready/supervisor 票中显式区分“请求判据通过”和“服务常驻通过”。
- **P2：R-R10 提交叙事与逐项复跑结果存在落差。** D4/D12/D15 等仍未通过，文档已如实记录；后续引用提交时必须同时引用验收行，不能把“复跑”理解成“验收通过”。
- **维护风险：** Reasonix 发现 `.git/REBASE_HEAD`、`.git/COMMIT_EDITMSG.swp` 等残留元数据；本轮未删除，避免对仓库元数据做不可逆清理。

## 证据

- #28 失败链：`8fdecf4` → `0515063` → `db443dc`。
- R-R3 当前操作顺序与限制：`docs/已知问题记录.md` §22.10、§28.4、§28.5。
- R-R2 物理失败记录：`docs/未完成工作备忘-2026-09-23.md` §4.1 / 2026-09-24 条目。
