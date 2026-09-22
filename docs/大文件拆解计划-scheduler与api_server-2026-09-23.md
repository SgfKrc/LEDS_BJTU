# 大文件拆解计划：`scheduler.py` 与 `api_server.py`

> 状态：**现行（重构计划）**
>
> 更新日期：2026-09-23
>
> 背景：主仓代码量统计显示这两个文件严重超标（`src/` 平均约 840 行/文件）：
> `src/scheduler.py` **14,775 行**、`src/api_server.py` **10,167 行**。
> 本计划给出**可独立回滚的分步方案**与**动手前必须先建的安全网**。

---

## 1. 结论摘要

| 文件 | 形状 | 本质 |
| --- | --- | --- |
| `src/scheduler.py` | 14,775 行，**92.7% 是单个 `Scheduler` 类**（13,697 行 / ≈302 个方法），模块级可变全局 **0 个** | **上帝对象** —— 拆分的本质是给 `Scheduler` 减负，不是给文件分堆 |
| `src/api_server.py` | 10,167 行，**144 个路由全部挂在模块级单例 `app` 上**（无 `APIRouter` / `include_router` / `mount`） | 单体 FastAPI 应用 —— 拆分的本质是**按领域切路由 + 抽共享状态** |

**共同的兼容策略**：`src/scheduler.py` 与 `src/api_server.py` **永不删除**，始终作为**唯一对外模块名**做 re-export 门面；
每步只搬**定义**、不动调用点，使 `git diff` 只含移动行 —— 这样 `git revert` 单步即可回滚。

**两条硬约束**（违反则等于拆解失败）：

1. `src/api_server.py` 拆分**不得改变端点集合与路径**（`README.md` 记录主后端 OpenAPI 快照为 152 个操作）；
2. `src/scheduler.py` 拆分**不得改变测试可 patch 的名字面**（模块级 `NODE_ROLE` / `NODE_ID` / `RUN_MODE` / `PIPELINE_PREEMPT_ENABLED`、
   实例方法名、私有属性名 —— 见 §4 风险清单）。

---

## 2. `src/scheduler.py` 职责地图

### 2.1 顶层结构

| 结构 | 行区间 | 行数 | 职责 |
| --- | --- | ---: | --- |
| docstring + imports + 常量 | 1–99 | 99 | 依赖装配；`torch = LazyTorch()`(118)、`logger`(138)、Android 常量(140–142) |
| `_TaskWorkerActiveAttempt`（dataclass） | 100–114 | 15 | Task-Worker 租约/取消事件载体 |
| 顶层函数 ×4 | 146–235 | 90 | logits 采样、bootstrap 端口推导、注册失败判定、运行时写回 `NODE_ID` |
| `NodeState` / `NodeRole` / `NodeInfo` | 236–297 | 62 | 节点枚举与画像 DTO（被 api_server 与 30+ 测试 import） |
| `_node_supports_forward_layers` | 298–345 | 48 | 层拆分资格门（测试直接 import） |
| `InferenceTask` / `QueueTask` / `PreemptState` | 346–446 | 101 | 任务与抢占 DTO |
| `PipelineQueue` | 447–1078 | **632** | MLFQ 三级反馈队列 + 结果 TTL + 抢占字段（24 个方法） |
| **`Scheduler`** | **1079–14775** | **13,697** | 见下表 |

### 2.2 `Scheduler` 的子职责分区

| # | 子职责 | 行区间 | 约行数 |
| --- | --- | --- | ---: |
| A | 构造 / 依赖注入 / **~40 个锁与状态字典** | 1092–1286 | 195 |
| B | 集群角色与 HA 注入点（control fence / auto-role / handoff） | 1287–1499 | 213 |
| C | 生命周期 start/stop + TCP server 装配 | 1500–1831 | 332 |
| D | 节点注册 / 心跳 / Android presence / 设备画像 | 1832–2392 | 561 |
| E | 节点权重 & VRAM & 层内存估算（多为 `staticmethod`） | 2393–2847 | 455 |
| F | 层分配计算 + 容量规划 + 手工覆盖 | 2848–3940 | 1,093 |
| G | 层配置下发 / 权威同步 / 版本栅栏 | 3941–4451 | 511 |
| H | Gemma4 sidecar | 4452–4657 | 206 |
| I | Qwen3 sidecar / dry-run / loopback / artifact transfer | 4658–5200 | 543 |
| J | Model-runtime contract 持久化 + sidecar 控制 | 5200–5628 | 429 |
| K | Qwen3 loopback 消息处理 & dry-run ack | 5629–5929 | 301 |
| L | Pipeline load transaction / 层配置重试监控 | 5930–6090 | 161 |
| M+ | Task-Worker、cluster/client-mode、pipeline/forward、HA … | 6091–14775 | 其余 |

**关键观察**：模块级可变全局只有 `from config import RUN_MODE, NODE_ROLE, NODE_ID, MAX_NODES, PIPELINE_*` 这几个引用
（测试会 `monkeypatch.setattr(scheduler_mod, "NODE_ROLE", …)`），其余状态全在**单个实例**上。

---

## 3. `scheduler.py` 拆分方案

### 3.1 目标结构（依赖单向、无环）

```
src/scheduler.py              门面 + re-export + 编排/状态查询（约 1500 行）
src/scheduler_core.py         __init__ 状态、start/stop、status/config、节点注册查询
src/scheduler_layer_plan.py   纯函数：节点权重、GPU 选择、层分配、容量规划（约 1200 行）
src/scheduler_sidecars.py     Qwen3 / Gemma4 / model-runtime-contract / loopback / dry-run（约 1480 行）
src/scheduler_task_worker.py  Task-Worker 控制面（约 875 行）
src/scheduler_cluster.py      节点注册/心跳/Android、client-mode、角色转让、HA（约 4400 行）
src/scheduler_pipeline.py     forward、layer_config、layer_forward、chain、run_pipeline、全模型回退（约 3600 行）
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

**抽出 `src/scheduler_sidecars.py`（`scheduler.py:4452–5929`，约 1,480 行）**。选它的四条理由：

1. **自包含度最高**：只依赖注入接口 `self._host`、`self._qwen3_*` / `_gemma4_*` / `_model_runtime_*` 私有状态，
   以及 4 个**已是独立模块**的 `qwen3_pipeline_*` / `gemma4_pipeline_*`；
2. **不触碰核心路径**：`run_pipeline` / `layer_forward` / 节点注册 / HA 全都不动；
3. **已有专属测试做安全网**：`test_qwen3_local_chain_scheduler.py`、`test_qwen3_pipeline_loopback.py`、
   `test_qwen3_pipeline_network.py`、`test_qwen3_pipeline_transaction.py`、`test_gemma4_pipeline_sidecar.py`、
   `test_model_runtime_sidecar_control.py`（全部 `from scheduler import Scheduler`）；
4. **零 monkeypatch 命中**：这 46 个方法**没有出现在任何测试的 patch 列表**里 ⇒ 挪位不会碰测试桩。

### 3.3 拆分顺序（每步独立可回滚）

| 步 | 抽什么 | 手法 | 安全网 |
| ---: | --- | --- | --- |
| 1 | `scheduler_layer_plan`（纯函数） | 搬成模块级函数；`Scheduler` 内保留同名 `staticmethod` 转发 | `TestComputeNodeWeight`、`TestGpuSelection`、`TestGpuIsIntegrated`、`TestNormalizeMasterAnchor` |
| 2 | **`scheduler_sidecars`（MVS）** | Mixin | §3.2 列出的 6 个测试文件 |
| 3 | `scheduler_task_worker` | Mixin（`_task_worker_*` 状态仍在 `__init__`） | `test_task_worker_adapter.py`（>2,500 行）、`tests/helpers/task_worker_process.py` |
| 4 | `scheduler_cluster`（最大一刀） | Mixin；**`_effective_role` 必须留在 `scheduler.py`**，否则 monkeypatch 全线失效 | `TestEffectiveRole`、`TestProvisionalMasterRole`、HA 系列、`test_connect_to_master_bootstrap_recovery` |
| 5 | `scheduler_pipeline` | Mixin；`_run_pipeline` / `_handle_layer_forward_locked` **最后搬** | `TestPipelineOrchestrationIntegration`、`TestChainTopology`、`TestPipelineMessageDispatch` |
| 6 | `scheduler.py` 收敛为门面 | 显式 `__all__` + re-export | 新增门面契约测试（§4.3 T1） |

---

## 4. `api_server.py` 拆分方案

### 4.1 关键前置事实

| 事实 | 含义 |
| --- | --- |
| 全部路由挂在模块级 `app`（`api_server.py:341`），**无 `APIRouter`** | 拆分必须引入 `APIRouter` + `include_router`，且**注册顺序**要复现 |
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
src/api/app.py             create_app：FastAPI 实例 + 中间件 + lifespan + include_router
src/api/state.py           全局单例与共享状态（scheduler / model_host / kv_cache / session_histories …）
src/api/deps.py            依赖注入与权限/边界校验
src/api/schemas.py         Pydantic 模型
src/api/routes_health.py   health/ready/status/presets
src/api/routes_device.py   device/*
src/api/routes_logs.py     logs/*（先抽，自包含）
src/api/routes_cluster.py  cluster/*（65 个，可再按子域分）
src/api/routes_models.py   models/*
src/api/routes_auth.py     auth/*、users/*
src/api/routes_sessions.py sessions/*、conversations/*
src/api/routes_tasks.py    workflows/*、task-graph
src/api/routes_chat.py     chat/*（最重，单独分支）
```

**推进顺序（薄 → 厚）**：`api/app.py` → `routes_logs` → `routes_health` / `routes_device` / `routes_cluster`
→ `schemas` / `state` / `routes_models` / `routes_auth` / `routes_sessions` / `routes_tasks` → `routes_chat`。

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
| **OpenAPI 快照** | `README.md:422` 记 152，实测 144，仓内**无快照文件** | 拆分前后"端点集合是否变化"**当前无法自动验证** |
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

每步之后必须同时满足：

1. 定向测试通过 —— scheduler 侧：
   `test_scheduler.py` + `test_qwen3_*` + `test_gemma4_*` + `test_task_worker_adapter.py` + HA 系列；
   api_server 侧：`test_api_*.py` + `test_task_graph_api.py` + `test_chat_interactive.py` + `test_core_cutover.py`；
2. **冷启动可用**：`python -c "import scheduler"` / `python -c "import api_server"`；
3. **端点集合不变**（api_server）：OpenAPI 快照对比；
4. **`git diff` 只含移动行**：用 `git diff --stat` 与 `git diff -M` 确认没有夹带逻辑改动。

---

## 7. 与既有 `src/inference_service/` 的关系（避免重复造轮子）

`src/inference_service/` 是**另一条产品线**：它把 api_server 的执行段**复制**成 `/v1/*` 新契约
（`engine_host.py` 注释明写"复制自 api_server…源文件保持不动"），入口 `src/inference_svc_main.py`，与 `api_server.py` **并行共存**。

它提供了两份**已拆好的参考实现**：

- `src/scheduler_svc_http.py` —— cluster/device 域的薄壳端点，可直接作为 `routes_cluster.py` 的实现参考；
- `src/inference_service/routes.py` —— chat/models 端点 + 引擎宿主解耦范式。

**建议**：本次只做「移动 + re-export」，**不要**顺手把 `/api/*` 改成 `/v1/*` —— 两套契约并存是既定架构。
