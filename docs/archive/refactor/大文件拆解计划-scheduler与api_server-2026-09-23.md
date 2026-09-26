# 大文件拆解计划：`scheduler.py` 与 `api_server.py`

> 状态：**历史参考（REFACTOR-LARGEFILE-01 至 05 已完成，计划收口后归档）**
>
> 归档（2026-09-23）：01–05 全部落地（`src/api/` 领域路由拆分 + 收口验收，全量回归
> 990 passed / 4 skipped）⇒ 移入 `docs/archive/refactor/`，**不再随实现维护**；
> 文中源码行号与路径均为**拆分前与收口时**的口径。
> 现行入口：[主线开发计划](../../主线开发计划-分布式推理与边缘优化-2026-09-14.md)、
> [文档状态与清理清单](../../文档状态与清理清单.md)、
> [跨框架接力当前有效基线](../../跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)。
>
> 更新日期：2026-09-23
>
> 背景：主仓代码量统计显示这两个文件严重超标（`src/` 平均约 840 行/文件）：
> `REFACTOR-LARGEFILE-02` 前 `src/scheduler.py` 为 14,786 行（`Scheduler` 类 13,695 行）；完成 `REFACTOR-LARGEFILE-03` 后门面约 3,736 行，`scheduler_task_worker.py` 约 907 行、`scheduler_cluster.py` 约 3,597 行、`scheduler_pipeline.py` 约 4,937 行、`scheduler_types.py` 约 173 行，`scheduler_sidecars.py` 约 1,518 行。
> 完成 `REFACTOR-LARGEFILE-04` 全量路由迁移后，`src/api_server.py` 约 4,731 行；原 144 个 OpenAPI 操作均由领域 `APIRouter` 提供。
> 本计划给出**可独立回滚的分步方案**与**动手前必须先建的安全网**。

> `REFACTOR-LARGEFILE-01` 证据：`tests/test_refactor_largefile_baseline.py`、
> `local_docs/evidence/refactor-largefile/REFACTOR-LARGEFILE-01-openapi-baseline-2026-09-23.json`。

---

## 1. 结论摘要

| 文件 | 形状 | 本质 |
| --- | --- | --- |
| `src/scheduler.py` | 拆分前 14,786 行，**主要逻辑仍集中在单个 `Scheduler` 类**，模块级公共符号已建立显式门面契约 | **上帝对象** —— 拆分的本质是给 `Scheduler` 减负，不是给文件分堆 |
| `src/api_server.py` | 拆分前 9,774 行、144 个操作；当前约 4,731 行，144 个操作由领域 `APIRouter` 接入 | 作为应用组合根与兼容门面，保留共享状态、lifespan、schema 和 patch 点 |

**共同的兼容策略**：`src/scheduler.py` 与 `src/api_server.py` **永不删除**，始终作为**唯一对外模块名**做兼容门面；
优先移动定义、不改调用点；如需保留 monkeypatch 或依赖注入行为，则在门面留显式转发点并以测试锁定。

**两条硬约束**（违反则等于拆解失败）：

1. `src/api_server.py` 拆分**不得改变端点集合与路径**；本票已固化主后端 OpenAPI 快照为 **123 条路径 / 144 个操作**；
2. `src/scheduler.py` 拆分**不得改变测试可 patch 的名字面**（模块级 `NODE_ROLE` / `NODE_ID` / `RUN_MODE` / `PIPELINE_PREEMPT_ENABLED`、
   实例方法名、私有属性名 —— 见 §4 风险清单）。

**与 PyTorch 科研优化的依赖**：本计划是后续 `TORCH-OP-PROFILE-01`、`TORCH-OP-REGISTRY-01`
和异构算子放置研究的前置工程票。拆解阶段只做安全网、定义移动和门面兼容，不把
KTransformers、算子替换、设备调度或新的 PyTorch 执行语义塞进 `Scheduler`/`api_server`。
研究对象是拆解后的项目自有 PyTorch 上游；KTransformers 仅提供设计参照，不作为第三个后端或依赖。

---

## 2. `src/scheduler.py` 职责地图

### 2.1 REFACTOR-LARGEFILE-01 基线结构

以下行区间为 `REFACTOR-LARGEFILE-02` 前的结构快照；当前边界以新拆分模块和各票收口记录为准。

| 结构 | 行区间 | 行数 | 职责 |
| --- | --- | ---: | --- |
| docstring + imports + 常量 | 1–145 | 145 | 依赖装配；`torch = LazyTorch()`、`logger`、Android 常量 |
| `_TaskWorkerActiveAttempt`（dataclass） | 101–113 | 13 | Task-Worker 租约/取消事件载体 |
| 顶层函数 ×5 | 159–247 | 89 | logits 采样、bootstrap 端口推导、注册失败判定、运行时写回 `NODE_ID` |
| `NodeState` / `NodeRole` / `NodeInfo` | 249–308 | 60 | 节点枚举与画像 DTO（被 api_server 与 30+ 测试 import） |
| `_node_supports_forward_layers` | 311–355 | 45 | 层拆分资格门（测试直接 import） |
| `InferenceTask` / `QueueTask` / `PreemptState` | 359–457 | 99 | 任务与抢占 DTO |
| `PipelineQueue` | 460–1089 | **630** | MLFQ 三级反馈队列 + 结果 TTL + 抢占字段 |
| **`Scheduler`** | **1092–14786** | **13,695** | 见下表 |

### 2.2 `Scheduler` 的子职责分区

| # | 子职责 | 拆分结果 |
| --- | --- | --- |
| A | 构造 / 依赖注入 / **~40 个锁与状态字典** | 状态仍由 Scheduler 实例持有 |
| B | 集群角色与 HA 注入点（control fence / auto-role / handoff） | 已移至 `scheduler_cluster`；实例状态和依赖仍由 Scheduler 持有 |
| C | 生命周期 start/stop + TCP server 装配 | 留待 `scheduler_core` 拆分 |
| D | 节点注册 / 心跳 / Android presence / 设备画像 | 已移至 `scheduler_cluster` |
| E | 节点权重、GPU 选择与 assignment 锚点 | 无状态 helper 已移至 `scheduler_layer_plan` |
| F | 层分配计算 + 容量规划 + 手工覆盖 | 本票保持原位 |
| G | 层配置下发 / 权威同步 / 版本栅栏 | 对外操作移至 `scheduler_pipeline`；锁与运行时状态仍由 Scheduler 实例持有 |
| H–K | Gemma4/Qwen3/model-runtime/loopback/dry-run | 60 个方法已移至 `SchedulerSidecarMixin` |
| L+ | Task-Worker、cluster/client-mode、pipeline/forward、HA … | 已拆至 `scheduler_task_worker`、`scheduler_cluster`、`scheduler_pipeline`；`_effective_role` 与跨域 TCP dispatch 留在门面 |

**关键观察**：模块级可变全局只有 `from config import RUN_MODE, NODE_ROLE, NODE_ID, MAX_NODES, PIPELINE_*` 这几个引用
（测试会 `monkeypatch.setattr(scheduler_mod, "NODE_ROLE", …)`），其余状态全在**单个实例**上。

---

## 3. `scheduler.py` 拆分方案

### 3.1 目标结构（依赖单向、无环）

```
src/scheduler.py              门面 + re-export + 编排/状态查询（当前约 3,737 行）
src/scheduler_core.py         __init__ 状态、start/stop、status/config、节点注册查询
src/scheduler_layer_plan.py   纯函数：节点权重、GPU 选择、master 锚点与区间重排（约 257 行）
src/scheduler_sidecars.py     Qwen3 / Gemma4 / model-runtime-contract / loopback / dry-run（约 1520 行）
src/scheduler_task_worker.py  Task-Worker 控制面（约 907 行）
src/scheduler_cluster.py      节点注册/心跳/Android、client-mode、角色转让、HA（约 3,597 行）
src/scheduler_pipeline.py     forward、layer_config、layer_forward、chain、run_pipeline、全模型回退（约 4,937 行）
src/scheduler_types.py        scheduler 共享 DTO 与队列状态类型（约 173 行）
```

```
scheduler.py（门面/编排）
  ├─> scheduler_core ──> scheduler_layer_plan（纯函数，无回边）
  ├─> scheduler_sidecars
  ├─> scheduler_task_worker
  ├─> scheduler_cluster
  └─> scheduler_pipeline ──> scheduler_layer_plan
```

**手法选择**：现状是「一个类 + 海量 `self._x`」⇒ 子模块用 **Mixin 类**（`class SidecarMixin:`）
比「组合 + 委托」改动小一个数量级 —— 方法体内 `self._qwen3_local_chain` 等**无需改写**，
只要 `class Scheduler(SidecarMixin, TaskWorkerMixin, …)`。纯函数族（layer_plan）直接搬成模块级函数。

### 3.2 最小可行第一步（MVS）

**已抽出 `src/scheduler_sidecars.py`（拆分前 `scheduler.py:4244–5720`，60 个方法）**。选择该边界的理由：

1. **自包含度最高**：只依赖注入接口 `self._host`、`self._qwen3_*` / `_gemma4_*` / `_model_runtime_*` 私有状态，
   以及已独立的 Qwen3/Gemma4 sidecar 协议模块；
2. **不触碰核心路径**：`run_pipeline` / `layer_forward` / 节点注册 / HA 全都不动；
3. **已有专属测试做安全网**：`test_qwen3_local_chain_scheduler.py`、`test_qwen3_pipeline_loopback.py`、
   `test_qwen3_pipeline_network.py`、`test_qwen3_pipeline_transaction.py`、`test_gemma4_pipeline_sidecar.py`、
   `test_model_runtime_sidecar_control.py`（全部 `from scheduler import Scheduler`）；
4. **patch 面已保留**：测试会替换 `scheduler.Qwen3PipelineMultiSidecar`；`Scheduler._qwen3_multisidecar_factory()` 保留旧模块级替换入口，sidecar mixin 不反向导入门面。

### 3.3 拆分顺序（每步独立可回滚）

| 步 | 抽什么 | 手法 | 安全网 |
| ---: | --- | --- | --- |
| 1 | `scheduler_layer_plan`（纯函数） | **已完成**：搬成模块级函数；`Scheduler` 内保留同名兼容转发器 | `test_scheduler.py`、`test_island_engine.py` |
| 2 | **`scheduler_sidecars`（MVS）** | **已完成**：Mixin；工厂经 Scheduler 门面注入 | §3.2 列出的 6 个测试文件 |
| 3 | `scheduler_task_worker` | **已完成**：Mixin；`_task_worker_*` 实例状态仍在 `__init__` | Task-Worker 协议、适配器及 layer-forward 回归 |
| 4 | `scheduler_cluster`（最大一刀） | **已完成**：Mixin；`_effective_role` 与跨域 TCP dispatch 留在门面，保留 facade 模块级 patch 点 | `TestEffectiveRole`、`TestProvisionalMasterRole`、HA 系列、`test_connect_to_master_bootstrap_recovery` |
| 5 | `scheduler_pipeline` | **已完成**：Mixin；实例 pipeline 状态不迁移，方法实现仅搬移 | `TestPipelineOrchestrationIntegration`、`TestChainTopology`、`TestPipelineMessageDispatch` |
| 6 | `scheduler.py` 门面收口 | `__all__`/re-export 已由 01 固化；最终 API 迁移后的门面回归归 05 | 门面契约测试与 OpenAPI 快照 |

---

## 4. `api_server.py` 拆分方案

### 4.1 关键前置事实

| 事实 | 含义 |
| --- | --- |
| 拆分前路由全部挂在模块级 `app`（`api_server.py:341`），现有 health/device/logs 已用 `APIRouter` | 后续迁移继续核验注册顺序，尤其 logs 的 literal/path 通配路径 |
| 生命周期用 `lifespan`（318–338）而非 `on_event` | 迁移时不要改写风格 |
| 已有一次「边界治理」：`model_host` 单例 + 尾部回调注入（526、10073–10083） | 这是**依赖倒置的先例**，拆分应沿用同一手法 |
| **唯一运行时 `import api_server` 的是 `src/tui_backend.py:80,92,101`** | 消费者极少 ⇒ 门面策略可行 |
| 已存在并行的 `src/inference_service/`（`/v1/*`） | 那是**另一条产品线（复制而非拆分）**，不是本次的替代品；但它提供了「已拆好的参考实现」 |
| ⚠️ **实测装饰器 144 个，而 `README.md` 写 152** | **先固化 baseline 并核对差异**，再动刀 |

### 4.2 路由地图（实测 144 个）

| 域 | 数量 | 备注 |
| --- | ---: | --- |
| `/api/cluster/*` | **65** | 最大一块；含 `nodes/log-aggregate`（实现位于日志区段，建议归 cluster） |
| `/api/models/*` | 23 | 含下载/搜索/预检/登记 |
| `/api/logs/*` | 11 | 含 `{filename:path}` 通配 ⇒ **注册顺序敏感** |
| `/api/sessions/*` | 7 | |
| `/api/auth/*` | 6 | |
| `/api/chat*` | 5 | 含 SSE 流式（`StreamingResponse`） |
| `/api/workflows*` | 4 | |
| `/api/users*` | 4 | |
| 其余（health/ready/presets/status/device/conversations/settings/bootstrap/experimental/db/storage/shutdown） | 19 | |
| **合计** | **144** | |

### 4.3 目标结构与顺序

```
src/api/__init__.py
src/api/_routing.py        router 对兼容门面的窄解析
src/api/routes_health.py   health/ready/status/presets（4 个操作）
src/api/routes_device.py   device/*（3 个操作）
src/api/routes_logs.py     logs/*（12 个操作，通配路由保持末位）
src/api/routes_cluster.py  cluster/bootstrap（66 个操作）
src/api/routes_models.py   models/*（23 个操作）
src/api/routes_auth.py     auth/*、users/*、user/settings（12 个操作）
src/api/routes_sessions.py sessions/*、conversations/*（10 个操作）
src/api/routes_tasks.py    workflows/*（4 个操作）
src/api/routes_chat.py     chat/experimental（7 个操作）
src/api/routes_system.py   system/db/storage（3 个操作）
```

**推进顺序（薄 → 厚）**：health/device/logs → cluster/models/auth/sessions/tasks/chat/system 已全部迁移。`REFACTOR-LARGEFILE-05` 完成门面验收，但没有把 FastAPI app/lifespan、共享状态与 schema 迁出 `api_server`：router 仍通过显式配置的兼容门面解析这些运行时依赖，以保持现有单例与 monkeypatch 契约。它们不是已拆出的 `api/*` 模块；若以后迁移，必须单独设计注入与兼容边界，不属于本轮验收。

**第一件事（强制）**：固化 **OpenAPI baseline** ——

```python
TestClient(api_server.app).get("/openapi.json").json()["paths"]
```

写出快照并断言「路径集合 + 每个路径的 method 集合」不变；同时核对**真实操作数**（当前实测 144，README 写 152）。

---

## 5. 动手前必须先建的安全网

### 5.1 风险清单（有实锤）

| 风险类型 | 证据 | 后果 |
| --- | --- | --- |
| **模块级 monkeypatch** | `tests/test_scheduler.py` 对 `NODE_ROLE` / `NODE_ID` / `RUN_MODE` / `PIPELINE_PREEMPT_ENABLED` / `scheduler_mod.time.time` 的 patch | 若 `_effective_role` 等移到"用自己的模块名"的子模块 ⇒ patch 变无效，测试**假通过或失败** |
| **实例方法改名** | `test_scheduler.py` 大量 `monkeypatch.setattr(sched, "_run_pipeline", …)`、`_get_pipeline_readiness`、`_all_pipeline_nodes_ready`、`_run_full_model_inference` 等 | 方法名是**测试契约**，重命名即断裂 |
| **源码文本扫描型测试（最隐蔽）** | `tests/test_database_fallback.py:35`、`tests/test_koakuma_engine.py:101` 直接 `read_text("src/scheduler.py")` 做 token 断言 | 代码搬走后**断言仍然通过但覆盖出现漏洞** ⇒ **静默失去门禁** |
| **私有属性直读** | `api_server.py` 与 `scheduler_svc_http.py` 用 `getattr(scheduler, "_qwen3_artifact_transfer_runtime")` 之类字符串访问 | 改名/改归属**不报错，只静默返回 None** ⇒ 功能降级难排查 |
| **锁归属** | `_inference_lock`、`_layer_config_lock`、`_layer_execution_lock` 被 api_server 直接当上下文管理器用 | Mixin 拆分时必须保证锁仍挂在**实例**上（不能变模块级） |
| **路由注册顺序** | `/api/logs/{filename:path}` 与 `/api/logs/recent` 共存 | 顺序变化会改变匹配结果 |
| **OpenAPI 快照** | `README.md` 的旧数字待后续同步；本票已记录实测 **123 条路径 / 144 个操作**，并加入 JSON 快照摘要 | 拆分前后"端点集合是否变化"由 `test_refactor_largefile_baseline.py` 自动验证 |
| **测试辅助进程** | `tests/helpers/task_worker_process.py:17` `from scheduler import Scheduler` | 门面缺 re-export 会直接 ImportError |

### 5.2 需要先补的测试

**scheduler 侧**：

- **T1 门面契约测试**（新文件）：`import scheduler` 后断言 `__all__ ⊇ {Scheduler, PipelineQueue, NodeInfo, NodeState, NodeRole, _node_supports_forward_layers, _bootstrap_api_port}`，
  并用 `git show HEAD:src/scheduler.py` 生成 `dir(scheduler)` 基准做 diff；
- **T2 源码扫描清单更新**：把 `test_database_fallback.py` / `test_koakuma_engine.py` 的扫描目标从硬编码 `"src/scheduler.py"`
  改为「`scheduler.py` + 所有 `scheduler_*.py`」，并断言新清单非空 —— **这一条最容易被忽略且失败方式最隐蔽**；
- **T3 monkeypatch 面测试**：断言 `_effective_role()` 读的是**`scheduler` 模块**的 `NODE_ROLE`（把 7 处隐式依赖变成显式契约）；
- **T4 `handle_infer_forward` 全路径测试**：走 `INFER_FORWARD → run_pipeline_safe → INFER_RESULT`，含并发上限与取消路径
  （现有测试**只断言"被调用"**，没有成功路径覆盖）；
- **T5 锁语义测试**：断言 `_inference_lock` / `_layer_config_lock` / `_layer_execution_lock` 是**同一实例上的同一对象**。

**api_server 侧**：

- **OpenAPI baseline 快照测试**（§4.3）；
- **路由顺序不变量**：断言 `/api/logs/{filename:path}` 在 `/api/logs/recent` 之后注册；
- **公共符号 re-export 冒烟**：把"被其他模块 import 的公共函数"清单固化为测试。

---

## 6. 验收方式

### 6.1 REFACTOR-LARGEFILE-01 收口记录

- 已完成 scheduler 门面 `__all__`、HEAD 符号保留、模块级角色 monkeypatch、实例锁身份和 `handle_infer_forward` 成功/拒绝/取消测试。
- 已将 scheduler 源码扫描门禁扩展到 `scheduler.py` 与所有 `scheduler_*.py`；当前仓库已有 `scheduler_svc_http.py`，后续新增拆分模块会自动纳入扫描。
- 已固定 API OpenAPI 路径/方法摘要，并固定 `/api/logs/recent` 必须早于 `/api/logs/{filename:path}` 的注册顺序。
- 定向验收：`30 passed`（本票基线、数据库退场门禁、引擎门禁）。

### 6.2 REFACTOR-LARGEFILE-02 收口记录

- 抽出 `scheduler_layer_plan.py` 的无状态评分/GPU 选择/锚点/区间函数；`Scheduler` 保留兼容转发方法。
- 抽出 `scheduler_sidecars.py` 的 60 个 Gemma4/Qwen3/model-runtime/loopback/dry-run 方法为 mixin；保留 `scheduler.Qwen3PipelineMultiSidecar` monkeypatch 面，由 `Scheduler._qwen3_multisidecar_factory()` 读取门面符号。
- 新 mixin 不导入 `scheduler`；定向联合回归 `424 passed`，覆盖 baseline、scheduler、Qwen3、Gemma4 与 model-runtime sidecar。
- `py_compile` 与 `git diff --check` 通过；AST 对比确认搬出的 60 个方法实现一致，唯一方法体适配是侧车 factory 注入。

### 6.3 REFACTOR-LARGEFILE-03 收口记录

- 拆出 `scheduler_task_worker.py`、`scheduler_cluster.py`、`scheduler_pipeline.py` 三个 mixin，并将共享节点/任务/抢占 DTO 放入 `scheduler_types.py`；`Scheduler` 只增加 mixin 继承，状态仍由原实例持有。
- 搬移 172 个方法；模块级配置和 helper 通过 facade lookup 动态读取，保持 `scheduler` 模块 monkeypatch 行为。`_effective_role`、`_on_tcp_message`、`_on_tcp_disconnect`、`_on_master_connection_lost` 保留在 `Scheduler`，避免跨领域 dispatch 被拆散。
- AST 对照确认搬移方法体等价（除显式 facade-global lookup 适配）；新增基线测试锁定 MRO、代表性方法归属和运行时 patch 点。
- Task-Worker、cluster/HA、pipeline 联测：`647 passed`；收口基线测试 `20 passed`。全仓并行测试在 `1071 passed, 16 skipped` 后遇到两个非本票基线/环境失败：缓存命名测试引用的计划文档缺失，以及 Windows `llama.dll` 加载 WinError 127；因此不宣称全仓通过。
- 清理机械搬移产生的空白行后，`scheduler.py` 约 3,736 行；`import scheduler` 成功，OpenAPI 保持 123 条路径，静态编译通过。

### 6.4 REFACTOR-LARGEFILE-04 全量路由迁移

- 按 health/device/logs/cluster/models/auth/sessions/tasks/chat/system 拆出 10 个领域 router；原 144 个操作 handler 已移出 `api_server.py`，兼容门面仍 re-export 同名 handler。
- 共享状态、日志 helper、Scheduler 和可变运行时配置仍由 `api_server` 持有；router 通过窄类型解析与 facade 动态读取保留 monkeypatch 和单例语义。session 串行化装饰器及依赖/响应模型元数据保持原注册行为。
- OpenAPI 路径/方法摘要保持 `123 paths / 144 operations`；router operations 与 OpenAPI operation 集合逐项相等，logs 的 literal 路由仍早于 `/api/logs/{filename:path}`。
- 验证：API/模型/集群/会话/聊天/task-graph 定向回归 `251 passed`；API cold-start/auth/bootstrap/TUI SSE `86 passed`；cache naming + keep-head/relay `30 passed, 7 skipped`，其中真实 shim/head 模型隔离后宿主 `llama_cpp` 导入通过；125 个本轮迁移 handler 的 AST 实现体对照无差异。

### 6.5 REFACTOR-LARGEFILE-05 门面与联合回归收口

- 门面契约：scheduler 保留 HEAD 模块符号、`__all__`、运行时 monkeypatch 点与实例锁身份；API 门面 re-export 全部 router handlers 及历史调用符号。当前路由均由 10 个领域 router 注册，`api_server.py` 不再定义 endpoint handler。
- API 契约：OpenAPI 快照固定为 `123 paths / 144 operations`；router operation 与 OpenAPI 的 path/method 多重集合逐项相等；`/api/logs/recent` 仍先于通配 filename 路由。
- 启动契约：API 冷启动测试覆盖默认 GGUF 路径不导入 torch、后台启动期间 health/readiness 可响应；默认引擎和无 Torch 边缘路径未改变。
- 联合回归（Windows，单进程 `pytest -n 0`）：API/chat/task-graph/core-cutover/baseline `275 passed`；scheduler/task-worker/reshard/Qwen3/Gemma4/HA `715 passed, 4 skipped`；合计 `990 passed, 4 skipped`。跳过项属于需要外部模型或设备条件的 smoke/门禁，不计作通过。
- 静态审计：原 125 个迁移 handler 的 AST 函数体对比无行为差异；当前门面 AST 不含 endpoint 定义。`api_server` 的共享状态、schema 与 lifespan 仍是有意保留的兼容边界，不能据此宣称它们已模块化。

`REFACTOR-LARGEFILE-01` 至 `05` 已完成。本批目标是 scheduler mixin 拆分、API endpoint 按域迁移及兼容验收；没有承诺把所有运行时状态和 schema 拆成独立模块。

每步之后必须同时满足：

1. 定向测试通过 —— scheduler 侧：
   `test_scheduler.py` + `test_qwen3_*` + `test_gemma4_*` + `test_task_worker_adapter.py` + HA 系列；
   api_server 侧：`test_api_*.py` + `test_task_graph_api.py` + `test_chat_interactive.py` + `test_core_cutover.py`；
2. **冷启动可用**：`python -c "import scheduler"` / `python -c "import api_server"`；
3. **端点集合不变**（api_server）：OpenAPI 快照对比；
4. **实现语义不变**：用 `git diff -M` 对照移动定义；门面转发、monkeypatch 工厂或依赖注入适配允许有小范围改动，但必须有专项测试锁定，不能夹带调度/推理逻辑变化。

---

## 7. 与既有 `src/inference_service/` 的关系（避免重复造轮子）

`src/inference_service/` 是**另一条产品线**：它把 api_server 的执行段**复制**成 `/v1/*` 新契约
（`engine_host.py` 注释明写"复制自 api_server…源文件保持不动"），入口 `src/inference_svc_main.py`，与 `api_server.py` **并行共存**。

它提供了两份**已拆好的参考实现**：

- `src/scheduler_svc_http.py` —— cluster/device 域的薄壳端点，可直接作为 `routes_cluster.py` 的实现参考；
- `src/inference_service/routes.py` —— chat/models 端点 + 引擎宿主解耦范式。

**建议**：本次只做「移动 + re-export」，**不要**顺手把 `/api/*` 改成 `/v1/*` —— 两套契约并存是既定架构。

---

## 8. 联合排期与当前下一票

本计划与《KTransformers 算子级优化调研 + QLH 算法/数据层优化方向》采用同一排期。顺序固定为：

1. `REFACTOR-LARGEFILE-01`：**已完成**；已固化 OpenAPI、公共符号、锁身份、monkeypatch 面、导入和回归基线。
2. `REFACTOR-LARGEFILE-02` 至 `REFACTOR-LARGEFILE-05`：**已完成**；scheduler/API 拆解、门面兼容、端点集合不变和联合回归均已验收。
3. `TORCH-OP-PROFILE-01`：**当前下一票**；建立项目 PyTorch 上游的算子成本画像，替代平均每层的路由依据。
4. `TORCH-OP-REGISTRY-01`：建立逻辑算子、候选实现、设备能力、误差边界和回退实现的合同。
5. `TORCH-HETERO-PLAN-01` 及后续科研票：研究算子放置、prefill/decode 双计划、激活压缩和 MoE 热度/预取。

`REFACTOR-LARGEFILE-01/02/03/04/05` 已完成；当前下一票为 `TORCH-OP-PROFILE-01`。PyTorch 算子研究只在 PC/CUDA 研究环境推进；不得改变 llama.cpp/GGUF 默认路径、Edge 无 Torch 边界或 Koakuma 正式 backend 枚举。

### 联合验收顺序

- 重构票：定向测试、门面导入、OpenAPI 路径/方法集合、路由顺序、锁身份和 `git diff` 移动行检查。
- PyTorch 研究票：未优化 PyTorch、优化 PyTorch、整模 llama.cpp、连续层分布式四组对照；记录正确性、首 token、decode、峰值内存、通信和回退。
- MoE 研究票：必须单独声明实验模型和硬件，不能用 dense 模型数字外推；任何近似策略先作为研究档，不能绕过逐 token/fail-closed 门。
