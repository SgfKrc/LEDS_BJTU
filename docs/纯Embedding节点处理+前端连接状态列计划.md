# 纯 Embedding / LM Head 节点处理 + 前端连接状态列 — 实施计划

> 日期：2026-06-17  
> 关联：`_apply_vram_constraints` 层转移逻辑、`LAYER_FORWARD` 协议、前端 AdminPanel 节点列表

---

## 1. "纯 Embedding / LM Head 节点"是什么

在 QLH 分布式流水线中，每个节点承担一个**连续的 Transformer 层区间**（如 Layer 0-7），并可以附带两项特殊职责：

| 标记 | 含义 | 额外显存开销 |
|------|------|-------------|
| `has_embedding: True` | 首节点：负责 `input_ids → hidden_states` 的 Token Embedding 查找 | ~128 MB (fp16) |
| `has_lm_head: True` | 末节点：负责 `hidden_states → logits` 的 LM Head 投影 | ~128 MB (fp16) |

**纯 Embedding/LM Head 节点**是指：经过 `_apply_vram_constraints` 显存约束重分配后，某个节点的 Transformer 层被全部转移给其他节点，只剩下 Embedding 或 LM Head 职责（`layers_count == 0`，但 `has_embedding == True` 或 `has_lm_head == True`）。

### 当前代码的行为

[scheduler.py:1199-1208](src/scheduler.py#L1199-L1208) 会**直接删除** `layers_count == 0` 的节点，然后自动将 `has_embedding` 赋给新列表的首节点、`has_lm_head` 赋给末节点：

```python
assignments[:] = [a for a in assignments if a["layers_count"] > 0]  # 删掉 0 层节点
for i, a in enumerate(assignments):
    a["has_embedding"] = (i == 0)          # 新的首节点继承 embedding
    a["has_lm_head"] = (i == len(assignments) - 1)  # 新的末节点继承 lm_head
```

### 这有什么问题

1. **接盘节点未加载 Embedding/LM Head 权重**：假设 master 原为首节点（has_embedding=True, layers 0-7），因显存不足把 7 层全部转给 client1。client1 变成新的首节点，被标记 `has_embedding=True`。但 client1 侧 `load_layer_range(0, 7, has_embedding=False, ...)` 从未加载过 Embedding 权重。当主节点发来 `LAYER_FORWARD`（含 `input_ids`）时，client1 调用 `forward_layers(hidden_states=embedding(input_ids))` 会失败——`model.embed_tokens` 不在已加载的模型部分中。

2. **对称问题在 LM Head**：末节点转移所有层后，新末节点未加载 `lm_head` 权重，最终 logits 输出会缺失或崩溃。

3. **转发协议不支持拆分**：当前 `LAYER_FORWARD` 协议假设首节点接收 `input_ids`（做 embedding lookup），中间节点接收 `hidden_states`（做 transformer forward），末节点输出 `hidden_states`（由主节点做 lm_head）。如果 embedding 和第一个 transformer 层不在同一节点上，现有协议无法表达"节点 A 只做 embedding → 发给节点 B 做 layer 0"的语义。

---

## 2. 处理方案

### 方案 A（推荐，短期）：保证 Embedding/LM Head 不脱离 Transformer 层

**思路**：在 `_apply_vram_constraints` 中禁止将首节点的**最后一个 Transformer 层**转移走，禁止将末节点的**第一个 Transformer 层**转移走。这样首节点至少保留 1 层 Transformer + Embedding，末节点至少保留 1 层 Transformer + LM Head。

**改动点**：

1. [scheduler.py:1174-1197](src/scheduler.py#L1174-L1197) — `_apply_vram_constraints` 增加保护：
   ```python
   # 保护：首节点至少保留 1 层（与 Embedding 共存）
   if a["has_embedding"] and a["layers_count"] <= 1:
       continue  # 不转移，跳过该节点
   # 保护：末节点至少保留 1 层（与 LM Head 共存）  
   if a["has_lm_head"] and a["layers_count"] <= 1:
       continue
   ```

2. 显存不足时打 WARNING 日志并降级为本地推理，而非强制转移导致协议不兼容。

**优点**：改动最小，不涉及协议变更，不引入新的消息类型。  
**缺点**：首/末节点显存极度紧张时（连 1 层 Transformer + Embedding 都装不下）无法自动恢复，需人工介入。

### 方案 B（中期）：引入专门的 EMBED_FORWARD / LMHEAD_FORWARD 消息

**思路**：将 Embedding 和 LM Head 从 Transformer 层区间中独立出来，作为单独的流水线步骤。

**新增消息类型**：

| 消息 | 方向 | 载荷 | 说明 |
|------|------|------|------|
| `EMBED_FORWARD` | master → embedding 节点 | `{task_id, input_ids}` | 只做 token embedding |
| `EMBED_RESULT` | embedding 节点 → master | `{task_id, hidden_states}` | 返回 hidden states |
| `LMHEAD_FORWARD` | master → lm_head 节点 | `{task_id, hidden_states}` | 只做 lm_head 投影 |
| `LMHEAD_RESULT` | lm_head 节点 → master | `{task_id, logits}` | 返回 logits |

**流水线拓扑变更**：

```
原来（嵌入式）:
  master(embed + L0-7) → client1(L8-15) → client2(L16-23 + lm_head)

方案B（独立式）:
  embed_node(embed only) → client1(L0-11) → client2(L12-23) → lm_head_node(lm_head only)
```

**优点**：灵活度最高，embedding/lm_head 可以部署在任何节点上（包括极低显存设备）。  
**缺点**：增加 4 个消息类型，`run_pipeline` 逻辑变复杂，前端拓扑展示需适配。

### 方案 C（长期，可跨引擎）：统一为 "subgraph dispatch"

**思路**：将模型的不同子图（Embedding、Transformer 层组、LM Head）抽象为 `ComputeUnit`，调度器按 DAG 依赖关系把 `ComputeUnit` 分发到各节点。这本质上是把 `LAYER_FORWARD` 泛化为 `SUBGRAPH_EXEC`。

**优点**：架构最通用，可支持非 Transformer 模型。  
**缺点**：工程量大，与当前简单的链式流水线设计理念有冲突。

---

## 3. 推荐实施顺序

| 阶段 | 方案 | 工作量 | 时间 |
|------|------|--------|------|
| **Phase 1** | 方案 A — 保护首/末节点层数 | ~10 行代码 | 立即 |
| **Phase 2** | 前端连接状态列 | ~30 行 JSX+CSS | 1-2h |
| **Phase 3** | 方案 B — 独立 EMBED/LMHEAD 消息 | ~150 行 | 看需求 |

---

## 4. 前端连接状态列 — 实施细节

### 当前问题

AdminPanel 节点列表（[AdminPanel.jsx:1499](frontend\src\components\AdminPanel.jsx#L1499)）只显示节点名称、角色、状态标签，不显示 TCP 连接状态、RTT、网络类型。从节点视角：
- 查看主节点连接质量（RTT、是否已注册）需要到浏览器 DevTools Network 面板看 `/api/cluster/status` 响应
- 主节点视角：看不出哪些从节点 TCP 已断开 vs DB 记录仍显示 online

### 实施内容

**后端**（已完成）：`/api/cluster/status` 已返回 `tcp_client` 字段（[scheduler.py:2687-2704](src/scheduler.py#L2687-L2704)），包含 `connected`、`running`、`avg_rtt_ms`。`/api/cluster/nodes` 可通过 `get_client_info()` 补充 TCP 连接详情。

**前端改动**（[AdminPanel.jsx](frontend\src\components\AdminPanel.jsx) 节点列表区域）：

1. **新增列**（在现有"角色"和"状态"列之间插入）：

   | 列名 | 数据来源 | 显示内容 |
   |------|---------|---------|
   | 🔗 连接 | `status.tcp_server.client_details[nid]` 或 `node.connected_at` | 绿色圆点 + "已连接" 或 灰色圆点 + "未连接" |
   | ⏱ RTT | `status.tcp_server.client_details[nid].last_heartbeat` 或 `status.tcp_client.avg_rtt_ms` | 主节点视角：上次心跳距今秒数；从节点视角：RTT ms |
   | 🌐 网络 | `node.network_type` | WiFi / 以太网 / 本地 / Tailscale |

2. **状态列增强**：将现有的纯文字 "在线/离线" 改为带颜色指示灯：
   - 🟢 在线 + TCP 已连接
   - 🟡 在线 + TCP 未连接（假在线，需告警）
   - 🔴 离线

3. **Tooltip**：悬浮在连接状态上显示详细信息（连接建立时间、心跳丢失次数、客户端地址）。

### 前端代码改动要点

```jsx
// 在节点列表的 table header 增加列
<th>🔗 连接</th>
<th>⏱ 延迟</th>
<th>🌐 网络</th>

// 每行数据
const tcpDetail = status?.tcp_server?.client_details?.[node.node_id];
const isTcpConnected = tcpDetail && (Date.now()/1000 - tcpDetail.last_heartbeat) < 15;

<td>
  <span className={`conn-dot ${isTcpConnected ? 'conn-ok' : 'conn-bad'}`} />
  {isTcpConnected ? '已连接' : '未连接'}
</td>
<td>{status?.tcp_client?.avg_rtt_ms ? `${status.tcp_client.avg_rtt_ms}ms` : '—'}</td>
<td>{NETWORK_LABELS[node.network_type] || '—'}</td>
```

### CSS 新增

```css
.conn-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.conn-ok { background: var(--success); }
.conn-bad { background: var(--text-muted); }
```

---

## 5. 总结

| 项目 | 优先级 | 状态 |
|------|--------|------|
| 纯 Embedding/LM Head 节点保护（方案 A） | 🔴 高 | 待实现 |
| EMBED_FORWARD / LMHEAD_FORWARD 协议（方案 B） | 🟡 中 | 按需启动 |
| Subgraph dispatch 泛化（方案 C） | 🟢 低 | 远期参考 |
| 前端节点列表连接状态列 | 🟡 中 | 后端数据已就绪，前端待实现 |
