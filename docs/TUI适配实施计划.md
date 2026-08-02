# TUI 适配实施计划（阶段 2 网关专项）

> **状态**：现行（T1-T8 全部完成，2026-08-02 验收通过：契约测试 44/44 + TUI 7 屏 × 2 角色自动化走查 + Python 回归 1086 passed / 3 skipped + 前端 9/9）
>
> **更新日期**：2026-08-02
>
> **关联文档**：[微服务架构改造计划](微服务架构改造计划.md)（§2.2 / §2.4 / §4.2 / §6.2）· [模块接口说明](模块接口说明.md) · [Python后端冷启动优化方案](Python后端冷启动优化方案.md)
>
> **前置条件**：主计划阶段 1 完成（scheduler-svc / inference-svc 进程化）；阶段 2 网关工程初始化（2.1/2.2 按域迁移）进行中。本文档为 **TUI 相关适配**的专项细化，其他网关域（chat / models / sessions 等）见主计划 §2.2，不在此展开。

---

## 一、背景与目标

### 1.1 背景

`tui_admin.py`（70KB，纯标准库）是 QLH 的终端管理面，经 HTTP 操作集群，覆盖 7 个屏幕。主计划承诺"TUI 零改动"——TS 网关替换 FastAPI 后 TUI 照常运行。2026-08-02 已完成 TUI 依赖面审计（38 处调用点 / 25 个唯一端点），结论：

- **可行性高**：全部同步 JSON、零流式/SSE/WebSocket、零 cookie、错误解析只依赖 JSON `detail` 且有容错。
- **3 类真实风险**：① 网关若按 §4.2 内部契约实现则 TUI 端点大面积 404（内部契约 ≠ 对外契约）；② DELETE 空响应/204 致 TUI 误报；③ 数值字段类型、`/api` 前缀、`X-QLH-Log-Token` 语义必须原样保留。

### 1.2 目标

1. 网关对外端点按 **api_server 现有路径/方法/字段** 原样实现，覆盖 TUI 全部 25 个唯一端点。
2. `tui_admin.py` **零改动**跑通 7 屏 × 2 角色（主节点 / 从节点视角）。
3. 以 TUI 38 个调用点为用例建立**契约测试**，纳入阶段 2 Go/No-Go 验收。

### 1.3 非目标

- 不改 `tui_admin.py` 代码（零改动是验收项，不是可选项）。
- 不实现 TUI 未调用的端点（如 chat/stream、models/download 等按主计划其他域处理）。
- 不做 TUI 功能增强（如新增屏幕、流式日志）。

---

## 二、TUI 依赖面审计基线（2026-08-02 复核）

### 2.1 客户端实现事实（`src/tui_admin.py`）

| # | 事实 | 位置 | 对网关的要求 |
|---|------|------|--------------|
| 1 | URL 硬编码 `base_url + "/api" + path` | :218 | **`/api` 前缀必须原样保留**，无 `base_url` 可配路径前缀 |
| 2 | query 用 `urlencode` 过滤空值 | :220-222 | 空值参数（None/""）不发送，网关不得因缺参 400（参数均有默认值语义） |
| 3 | 请求头 `Accept: application/json`；有 body 时 `Content-Type: application/json` | :224-227 | 响应必须 JSON；无 body 的 POST/DELETE 不要求 Content-Type |
| 4 | `X-QLH-Log-Token` 仅 `with_log_token and self.log_token` 才带 | :228-229 | **允许无 token 请求日志端点**；有 token 时按原语义校验 |
| 5 | 错误解析：HTTPError → 读 `detail`（dict 时取 `message`） | :234-244 | 错误体必须是 JSON `detail`；**禁止 302/纯文本/204 空体** |
| 6 | 空响应返回 `{}`；非 JSON 回退 `{"detail": text}` | :254-259 | 空响应会使 TUI 显示"—"，不崩但功能退化；**DELETE 必须返回 JSON** |
| 7 | 3s 轮询 3 个屏幕（Dashboard/Nodes/Queue） | :552、:599、:675、:992 | 网关不得引入 429/限流（urllib 无重试，直接报错） |

### 2.2 端点依赖全表（38 调用点 → 25 唯一端点）

> 行号以 2026-08-02 源码为准。`*` 标注的端点路径参数经 `urllib.parse.quote` 编码（:800、:809、:1076）。

| # | 端点（隐含 `/api` 前缀） | 方法 | 调用点 | 屏幕 | 读取的响应字段 |
|---|--------------------------|------|--------|------|----------------|
| 1 | `/health` | GET | :602、:1407 | 总览/设置 | `status=="ok"` |
| 2 | `/status` | GET | :603 | 总览 | `run_mode`、`node_role`、`node_id`、`max_nodes`、`model_loaded`、`model_name`、`active_model_id`、`current_quant`、`gpu{name,allocated_mb,total_mb,utilization}`、`kv_cache{total_tokens,allocated_pages,max_pages,estimated_memory_mb,rounds}`、`device{tier_label,tier,score,warnings}` |
| 3 | `/models/current` | GET | :604 | 总览 | `engine`、`device`、`total_params`、`gpu_allocated_gb`（float） |
| 4 | `/cluster/my-role` | GET | :605、:679 | 总览/节点 | `is_master`、`is_provisional`、`runtime_node_role`、`node_role`、`node_id` |
| 5 | `/cluster/nodes` | GET | :678 | 节点 | `count`、`online_count`、`offline_count`、`nodes[]{node_id,role,node_type,state,address,network_type,avg_rtt_ms,task_count,error_count,last_heartbeat}` |
| 6 | `/cluster/invite` | GET | :683 | 节点（主） | `master_host`、`master_port`、`node_count`、`max_nodes`、`identity_verified` |
| 7 | `/cluster/spare-master` | GET | :684 | 节点（主） | `spare_master{node_id,hostname,is_online}` |
| 8 | `/cluster/master-health` | GET | :686 | 节点（从） | `master_online`、`master_host`、`master_port`、`last_seen_seconds_ago`、`stale` |
| 9 | `/cluster/discover` | GET | :758、:848 | 节点 | `found`、`master_host`、`master_port`、`source`、`stale` |
| 10 | `/cluster/connect` | POST | :777 | 节点 | `status`、`message` |
| 11 | `/cluster/nodes/register` | POST | :789 | 节点 | `status`、`reason`、`message` |
| 12 | `/cluster/nodes/{id}/deregister` * | POST | :800 | 节点 | `status` |
| 13 | `/cluster/nodes/{id}` * | DELETE | :809 | 节点 | `status` |
| 14 | `/cluster/transfer-master` | POST | :818 | 节点 | `status`、`message` |
| 15 | `/cluster/spare-master` | POST | :827 | 节点 | `status`、`message` |
| 16 | `/cluster/spare-master` | DELETE | :833 | 节点 | `status`、`message` |
| 17 | `/cluster/config/max-nodes` | PUT | :844 | 节点 | `status` |
| 18 | `/cluster/transfer-logs` | GET | :856 | 节点 | `logs[0]{direction,from_role,to_role,related_node}`、`count` |
| 19 | `/cluster/email-test` | POST | :869 | 节点 | `message`、`status` |
| 20 | `/cluster/reset-identity` | POST | :876 | 节点 | `status` |
| 21 | `/cluster/config/distributed-inference` | GET | :889 | 分布式 | `enabled`、`default` |
| 22 | `/cluster/config/distributed-inference` | PUT | :953 | 分布式 | `status` |
| 23 | `/cluster/layers` | GET | :890 | 分布式 | `total`、`strategy`、`computed_at`、`assignments[]{node_id,role,start_layer,end_layer,has_embedding,has_lm_head,score}` |
| 24 | `/cluster/layers` | PUT | :976 | 分布式 | `status`、`message` |
| 25 | `/cluster/layers` | DELETE | :982 | 分布式 | `status` |
| 26 | `/cluster/config` | GET | :891 | 分布式 | `network{server_ip,server_port,heartbeat_interval_s}`、`model{quant_type,page_size,max_page_num,max_seq_len}` |
| 27 | `/cluster/queue` | GET | :995 | 队列 | `paused`、`strategy`、`current_task`、`queue_size`、`max_size`、`q0_depth`、`q1_depth`、`q2_depth`、`completed_count`、`aging_params{q0_max_tokens,q1_max_tokens,q1_to_q0_s,q2_to_q1_s}`、`preempt_stats{count,total_overhead_ms}`、`q0/q1/q2[]{task_id,original_level,wait_seconds,max_new_tokens,is_aged,session_id}` |
| 28 | `/cluster/queue/strategy` | POST | :1053 | 队列 | `strategy` |
| 29 | `/cluster/queue/pause` | POST | :1057 | 队列 | （不读） |
| 30 | `/cluster/queue/resume` | POST | :1061 | 队列 | （不读） |
| 31 | `/cluster/queue/clear` | POST | :1067 | 队列 | `cleared` |
| 32 | `/cluster/queue/task/{id}` * | DELETE | :1076 | 队列 | `message`、`success` |
| 33 | `/device/profile` | GET | :1089 | 画像 | `os{system,release}`、`hostname`、`cpu{model/brand,physical_cores,logical_cores}`、`ram/memory{total_gb,available_gb}`、`disk{free_gb,total_gb}`、`gpus[]{name,gpu_type,cuda_available,vram_total_gb}`、`selected_gpu_index`、`tier_label`、`tier`、`score_total`、`recommendations`、`warnings` |
| 34 | `/device/select-gpu` | POST | :1157 | 画像 | `selected_gpu{name}`、`selected_gpu_index`、`warning` |
| 35 | `/device/auto-configure` | POST | :1166 | 画像 | `applied_config{description}`、`tier`、`score` |
| 36 | `/logs/recent?limit&level` | GET | :1193-1196 | 日志（远程） | `count`、`matched`、`buffer_size`、`buffer_capacity`、`logs[]{level,time/timestamp,name,message}` |
| 37 | `/logs` | GET | :1198 | 日志（远程） | `files[]{name,size,modified}` |
| 38 | `/logs/stats` | GET | :1199 | 日志（远程） | `log_dir`、`files_count`、`files_total_bytes`、`buffer_size`、`buffer_capacity`、`buffer_total_seen`、`buffer_dropped_estimate`、`levels{}`、`nodes{}` |

> 36-38 三个日志端点在**本地模式**（TUI 与后端同机）下不走 HTTP（直接读 `logs/` 目录，:1201-1230），仅**远程模式**（`--host` 指向远端主节点）走 HTTP 并带 token（:1193-1199）。

---

## 三、网关适配要求

### 3.1 五个必须守住的细节（对应 §2.1 风险）

| # | 细节 | 违反后果 | 网关实现要求 |
|---|------|----------|--------------|
| ① | `/api` 前缀原样保留 | TUI 全部请求 404 | 网关挂载路径 `/api/*`，无额外前缀配置 |
| ② | 错误体必须是 JSON `detail`；DELETE 必须返回 JSON | TUI 误报"失败"或显示内部错误 | 统一异常过滤器输出 `{"detail": ...}`；`queue/task/{id}`、`nodes/{id}`、`spare-master`、`layers` 的 DELETE 返回 `{"status": ...}`/`{"message","success"}` 等原字段，**禁止 204** |
| ③ | 数值字段保持 number | `float()` 转换抛 ValueError → 屏幕显示"内部错误" | `avg_rtt_ms`、`wait_seconds`、`gpu_allocated_gb`、`utilization`、`score`、`q*_depth`、`size` 等序列化时禁止字符串化；TS 侧用 `number \| null` 类型定义 |
| ④ | `/api/health` 返回 JSON `status=="ok"`（网关先行拓扑下亦然） | 健康检查屏显示异常 | 网关自身存活即返回 200 `{"status":"ok"}`，**不得 503/纯文本**；模型/调度未就绪是 `/status` 的 `model_loaded` 等字段的事 |
| ⑤ | `/api/status` 的 `gpu`/`kv_cache`/`device` 内嵌对象结构原样 | TUI 显示"—"功能退化 | 聚合响应时保持嵌套字段名；扁平化即视为契约破坏 |

**软要求**：① 响应必须 UTF-8 JSON（TUI 硬编码 utf-8 decode，:233）；② 3s 轮询不触发 429 限流；③ 无 token 的日志请求必须返回 JSON（允许放行或 401 JSON，**禁止 302 重定向**——urllib 会跟随重定向导致行为不可控）。

### 3.2 端点 → 后端服务映射（网关实现方式）

| TUI 端点（#） | 后端 | 网关实现方式 |
|---------------|------|--------------|
| `/health`（1） | 无 | **内嵌**：网关自身返回 `{"status":"ok"}` |
| `/status`（2） | scheduler-svc + inference-svc + 本地 | **聚合**：scheduler `/v1/status` + inference `/v1/status`（模型/显存）+ 网关本机画像缓存；嵌套字段按 §2.2 #2 原样输出 |
| `/models/current`（3） | inference-svc | 代理：`GET /v1/models/current` |
| `/cluster/my-role`、`nodes`、`invite`、`spare-master`(GET)、`master-health`、`discover`（4-9） | scheduler-svc | 代理（scheduler-svc 需按 §2.2 字段提供；内部端点形态自行决定，见主计划 §4.2 对外适配原则） |
| `/cluster/connect`、`nodes/register`、`nodes/{id}/deregister`、`nodes/{id}`(DELETE)、`transfer-master`、`spare-master`(POST/DELETE)、`config/max-nodes`、`transfer-logs`、`email-test`、`reset-identity`（10-20） | scheduler-svc（email-test 阶段 2 由 legacy-control 承载，见 3.3） | 代理 |
| `/cluster/config/distributed-inference`(GET/PUT)、`layers`(GET/PUT/DELETE)、`config`(GET)（21-26） | scheduler-svc | 代理；**保留 PUT/DELETE 方法语义**（对外 ≠ 内部 POST 设计） |
| `/cluster/queue*`（27-32） | scheduler-svc | 代理；`queue/task/{id}` DELETE 透传 JSON 响应 |
| `/device/profile`、`select-gpu`、`auto-configure`（33-35） | scheduler-svc（采集库 device_profiler 留 Python） | 代理 |
| `/logs/recent`、`/logs`、`/logs/stats`（36-38） | legacy-control（阶段 2）→ control-svc（阶段 3） | 代理，**透传 `X-QLH-Log-Token`**，允许无 token |

### 3.3 时序决策：日志端点的承载进程

`/logs/*` 的 `buffer_*` 字段来自 FastAPI 进程内 logging 内存缓冲（非纯文件读取），TS 侧无法在阶段 2 低成本复刻。因此：

- **阶段 2 期间**：主计划 §2.2 允许"控制面域端点暂由 FastAPI 遗留进程承载"——从 api_server 剥离出**控制面遗留进程 `legacy-control`**（端口 `QLH_LEGACY_CONTROL_PORT`=8040，仅挂 logs/review/email/sessions/conversations/settings/bootstrap 等控制面路由），网关对 `/logs/*` 反向代理并透传请求头。迁移域端点（chat/cluster/queue/device 等）从遗留进程移除，实现主计划 §2.5"端点壳退役"。
- **阶段 3**：control-svc 就绪后，`/logs/*` 改代理到 control-svc，legacy-control 退役。**契约（路径/字段/token 语义）全程不变**。

---

## 四、契约测试设计

### 4.1 用例清单

测试文件：`gateway/test/tui-contract.e2e-spec.ts`（Jest + Supertest，连真实网关与各服务）

- **用例 1-38**：§2.2 全表 38 个调用点逐一用例——方法 + 路径 + 参数（含 `urllib.parse.quote` 编码的路径参数）命中 200/预期码，响应体 JSON 可解析。
- **用例 39-42（5 项细节断言）**：
  - 39：DELETE `/api/cluster/queue/task/{id}`、`/api/cluster/nodes/{id}`、`/api/cluster/spare-master`、`/api/cluster/layers` 均返回 JSON 且非空体（`Content-Type: application/json`，body ≠ ""）。
  - 40：数值字段类型断言——`/status.gpu.utilization`、`/status.device.score`、`/cluster/nodes.nodes[].avg_rtt_ms`、`/cluster/queue.q0[].wait_seconds`、`/models/current.gpu_allocated_gb` 均为 `number`（`typeof === "number"`，null 允许，字符串禁止）。
  - 41：`GET /api/health` 返回 `{"status":"ok"}`；无 `X-QLH-Log-Token` 请求 `/api/logs/recent` 返回 JSON（非 302）。
  - 42：`/api/status` 响应含 `gpu`/`kv_cache`/`device` 三个嵌套对象且字段名与 §2.2 #2 一致。
- **用例 43（错误契约）**：构造 4xx（如不存在的节点 id DELETE），断言响应体为 JSON 且含 `detail` 字段。

### 4.2 断言模板（TS）

```typescript
// gateway/test/tui-contract.e2e-spec.ts（节选）
import request from 'supertest';
import { app } from '../src/app';

describe('TUI 契约：/api 前缀与 JSON 错误', () => {
  it('用例 1: GET /api/health 返回 status=ok', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
  });

  it('用例 32: DELETE /api/cluster/queue/task/:id 返回 JSON 非空体', async () => {
    const res = await request(app).delete('/api/cluster/queue/task/nonexistent');
    expect(res.status).toBe(404);                    // 或 200，视 scheduler 语义
    expect(res.headers['content-type']).toContain('application/json');
    expect(res.text).not.toBe('');                    // 禁止 204 空体
  });

  it('用例 40: 数值字段类型必须为 number', async () => {
    const res = await request(app).get('/api/status');
    expect(typeof res.body.gpu?.utilization).toBe('number');
    expect(typeof res.body.device?.score).toBe('number');
  });

  it('用例 41: 无 token 日志请求返回 JSON 而非重定向', async () => {
    const res = await request(app).get('/api/logs/recent');
    expect(res.status).not.toBe(302);
    expect(res.headers['content-type']).toContain('application/json');
  });
});
```

> 说明：§2.2 中"（不读）"的端点（queue/pause、queue/resume）只需断言状态码与 JSON 响应，不校验字段。

---

## 五、实施步骤（任务清单）

> 执行顺序：契约测试先行（TDD），再实现网关适配，最后双网关对比与 TUI 实测。每任务完成后 commit。

- [x] **T1 建立契约测试骨架** ✅ 2026-08-02
  - 新建 `gateway/test/tui-contract.e2e-spec.ts`，按 §4.1 列出 43 个用例（未实现网关适配前允许 skip 标记）。
  - 验收：`npm test -- tui-contract` 可运行（允许 fail/skip），测试清单与 §2.2 全表一一对应（人工核对 38 项）。

- [x] **T2 网关基础设施（若主计划 2.1 已完成则跳过）** ✅ 2026-08-02
  - NestJS + Fastify adapter、`/api` 前缀挂载、request-id 中间件、异常过滤器输出 JSON `detail`（对齐 `api_server.py:319-366`）。
  - 验收：`curl -H "Accept: application/json" localhost:8000/api/health` 返回 `{"status":"ok"}`；未匹配路由返回 JSON 404 `{"detail": ...}`。

- [x] **T3 集群/队列/分层端点代理（#4-32）** ✅ 2026-08-02
  - 实现 scheduler-svc 客户端模块 `gateway/src/clients/scheduler.client.ts`（透传 + 502 兜底 + 非 2xx 转 HttpException）；`gateway/src/modules/cluster/cluster.controller.ts`（`@All('cluster/*')` 泛化转发，保留 PUT/DELETE 方法语义）。
  - **实现方式：1:1 透传代理**——内部端点 = 对外路径去掉 `/api` 前缀（见主计划 §4.2 落地决策）；scheduler-svc 未来实现按此 + §2.2 字段对齐。
  - 测试桩 `gateway/test/fake-scheduler.ts`：§2.2 字段对齐 + 真实 DELETE 语义（queue/task 不存在 → 200 success:false；nodes/{id} 不存在 → 404 detail，对齐 `api_server.py:5715-5732/:5266-5285`）。
  - 验收：用例 4-32、39 转绿（34 passed / 10 skipped）；`contract_diff.py` 双网关对比留待阶段 2 整体验收（旧网关尚未部署）。

- [x] **T4 设备画像端点代理（#33-35）** ✅ 2026-08-02
  - `gateway/src/modules/device/device.controller.ts`：与 cluster 同模式的 `@All('device/*')` 透传代理到 scheduler-svc（采集库 device_profiler 留 Python，见 §3.2 决策）。
  - fake-scheduler 补 3 条 device 路由（profile 含 os/cpu/ram/memory/disk/gpus/tier/score 全字段，数值字段 number）。
  - 验收：用例 33-35 转绿（37 passed / 7 skipped）。

- [x] **T5 状态聚合端点（#1-3）** ✅ 2026-08-02
  - `gateway/src/clients/forward-client.ts`：公共转发基类（SchedulerClient/InferenceClient 继承，消除复制）；`inference.client.ts` 新增（QLH_INFERENCE_URL，默认 :8010）。
  - `gateway/src/modules/status/status.controller.ts`：GET /api/status 并行聚合 scheduler `/cluster/status`（run_mode/node_role/node_id/max_nodes）+ inference `/v1/status`（model/gpu/kv_cache）+ `/device/profile`（device 摘要），单源失败回落默认值不 500（网关先行语义）；`modules/models/models.controller.ts`：/api/models/current 透传 inference `/v1/models/current`。
  - fake-inference.ts 测试桩（/v1/status、/v1/models/current，数值字段 number）；fake-scheduler 补 /cluster/status。
  - 验收：用例 2、3、40、42 转绿（41 passed / 3 skipped）。

- [x] **T6 日志端点代理（#36-38）** ✅ 2026-08-02
  - **legacy-control Python 桩**：`src/legacy_control.py`（纯标准库零依赖，未来 legacy-control 进程原型；`/logs/recent`、`/logs`、`/logs/stats`，X-QLH-Log-Token 可选，stdout 探活 `LEGACY_CONTROL_LISTENING:<port>`）。
  - 网关 `gateway/src/modules/logs/logs.controller.ts`：`/api/logs` 与 `/api/logs/*` 均透传 legacy-control（拆两个方法——fastify adapter 下同一方法叠加多个 @All 会覆盖）；`ForwardClient` 增加 `extraHeaders` 参数透传 X-QLH-Log-Token；`clients/legacy.client.ts`（QLH_LEGACY_CONTROL_URL，默认 :8040）。
  - 测试：jest 直接 spawn Python 桩（真实网关→Python 链路）；用例 36-38 打开；用例 41 补 logs 段 status=200 + 带 token 透传断言。
  - 验收：**44/44 全部用例通过（0 skipped）**。

- [x] **T7 TUI 实测（验收主体）** ✅ 2026-08-02（自动化走查通过）
  - 主节点角色：7 屏全走查（总览/节点/分布式/队列/画像/日志/设置），动作面全执行一遍（连接、注册、注销、删除、分层覆盖、队列策略、备用主节点设置等）。
  - 从节点视角：`tui_admin.py --host <主节点> --port 8000`（或 Tailscale IP），重点验证 `/cluster/master-health`（#8）与远程日志（#36-38）。
  - 验收：**tui_admin.py 零改动**，7 屏 × 2 角色全部通过；无"内部错误"屏（数值字段类型问题会在此暴露）。
  - **自动化走查**：`scripts/tui_walkthrough.py --mode master|client`（驱动 `tui_admin.py --plain`，喂入全动作输入序列，断言屏幕齐全 + 无错误屏 + 动作结果全出现）。配套 `scripts/dev_stubs.py --client-mode`（模拟从节点身份）。实测：master 7 屏 × 23 动作全通过；client 从节点视角（Dashboard + Nodes master-health 分支 + 远程日志）通过。
  - 排障记录：① TUI 对 `distributed-inference`/`layers`/`max-nodes` 用 **PUT**（act_toggle/act_override/act_max_nodes），桩补 PUT 路由；② `act_connect` 在主节点身份下有隐藏 confirm（切换为从节点），走查序列需补输入；③ 日志屏渲染桩日志 ERROR 级记录为 `[错误] 20xx-...` 行，属合法日志内容，错误屏断言需排除时间戳格式行。

- [x] **T8 回归与收尾** ✅ 2026-08-02
  - `pytest -q`（Python 侧回归）；前端 9/9（确认网关改动未破坏 Web 面）。
  - 更新 [微服务架构改造计划](微服务架构改造计划.md) §2.4 勾选状态与本文档状态为"现行"。
  - 验收：全部测试绿，提交。
  - 2026-08-02 实测：Python 全量回归 **1086 passed / 3 skipped**（112.6s）；前端 `npm test` **9/9** + 生产构建成功；网关契约测试 **63/63**（tui-contract + rest-contract）；TUI 7 屏 × 2 角色自动化走查通过（`scripts/tui_walkthrough.py`）。已提交。

---

## 六、验收标准（Go/No-Go）

| # | 项 | 通过标准 |
|---|----|----------|
| 1 | TUI 零改动 | `git diff src/tui_admin.py` 为空 |
| 2 | 端点覆盖 | §2.2 全表 38 个调用点全部可用（HTTP 状态 + JSON 响应） |
| 3 | 5 项细节 | §3.1 ①-⑤ 全部满足，用例 39-42 绿 |
| 4 | 双角色实测 | 7 屏 × 主/从角色走查通过，无"内部错误"屏 |
| 5 | 日志远程模式 | 无 token 可访问（JSON 响应）；带 token 行为与旧网关一致 |
| 6 | 性能 | 3s 轮询无 429；`/status` 聚合响应 P95 < 500ms（本地） |
| 7 | 回归 | Python `pytest -q` 绿；前端 9/9；主计划阶段 2 其他域不受影响 |

---

## 七、风险与回退

| # | 风险 | 缓解 / 回退 |
|---|------|--------------|
| 1 | scheduler-svc 内部端点形态与 TUI 对外字段不一致 | 网关适配层承担映射（对外字段以 §2.2 为准），scheduler-svc 不背 TUI 契约；契约测试兜底 |
| 2 | `/logs/*` buffer 统计无法在 TS 复刻 | 已决策：阶段 2 由 legacy-control 承载（§3.3），网关仅代理；若 legacy-control 构建受阻，回退为阶段 2 内嵌 TS 实现 `logs/recent`（文件尾部读取）+ 容错 `buffer_*` 字段（返回 0/空对象，TUI 显示为 0 不崩） |
| 3 | DELETE 响应被 NestJS 默认序列化吃掉（204） | 控制器显式 `return { ... }`，禁止 `@HttpCode(204)`；用例 39 强制非空 JSON |
| 4 | 数值字段被 TS 序列化为字符串（如大数/undefined） | 类型定义 `number \| null` + 用例 40 断言；网关 JSON 序列化用 `JSON.stringify` 默认行为（undefined 字段直接省略——TUI `.get()` 容错为"—"，可接受） |
| 5 | 聚合 `/status` 延迟影响轮询 | 聚合只读内存态/轻接口；若 scheduler `/v1/status` 慢，加 500ms 本地缓存，字段仍以 §2.2 #2 为准 |
| 6 | 从节点角色访问主节点端点被 CORS/鉴权拦截 | 网关不引入新鉴权（现无）；`master-health` 等从节点视角端点按旧语义放行 |

### 7.1 已知技术坑（2026-08-02 T1 排障记录）

> 完整排障链与验证方式见 [gateway/README.md](../gateway/README.md)。任何改动 fastify 版本、
> beforeAll 初始化或 404 处理的 PR 必须先跑 `npm run test:tui`。

| # | 坑 | 根因 | 修复 |
|---|----|------|------|
| 1 | `@nestjs/platform-fastify@11.1.28` 硬编码 `fastify@5.10.0`（精确版本），npm 嵌套安装两个 fastify | platform-fastify 的 `dependencies` 是精确版本，无法 dedupe | `package.json` 加 `"overrides": { "fastify": "^5.11.0" }` 统一 |
| 2 | supertest 首个请求崩溃 `Cannot read properties of undefined (reading 'length')`，且 jest `did not exit` | NestJS `app.init()` 不等待 fastify `ready()`；fourOhFour 404 context 的 hooks 在 `preReady` 才初始化，ready 前 `context.preParsing` 为 undefined 而非 null | 测试 `beforeAll` 中 `await (app.getHttpAdapter().getInstance() as any).ready()`（**T2-T6 每个测试文件都必须带**）；生产 `app.listen()` 内部会等 ready，不受影响 |

---

## 八、决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-08-02 | 网关对外端点以 api_server 现有路径/方法/字段为准，§4.2 内部契约不承担对外适配 | TUI 38 调用点审计证明对外面是唯一可信契约；内部端点形态由各服务自定 |
| 2026-08-02 | `/logs/*` 阶段 2 由 legacy-control 承载、网关代理；阶段 3 切 control-svc | `buffer_*` 是进程内内存统计，TS 复刻成本高；代理保持契约不变 |
| 2026-08-02 | 5 项细节（§3.1）以契约测试固化，不做口头约定 | 防止网关实现漂移；用例 39-42 进阶段 2 Go/No-Go |
| 2026-08-02 | TUI 零改动为硬验收（`git diff` 为空） | TUI 是纯 urllib 客户端，改动任何一处都意味着契约已破坏 |

---

**文档版本**：1.0
**维护者**：QLH 开发团队
**下次复核触发**：T6 完成（日志代理）、T7 完成（TUI 实测）各一次
