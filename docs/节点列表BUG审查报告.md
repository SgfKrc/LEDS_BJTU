# 节点列表 BUG 审查报告

> 日期: 2026-07-08  
> 审查范围: `src/scheduler.py`（init_nodes / register_node / deregister_node / delete_node / manual_register_node / update_max_nodes）+ `src/api_server.py` + `frontend/src/components/AdminPanel.jsx`  
> 测试状态: 160 passed

---

## 总览

| 级别 | 数量 |
|------|:----:|
| 🔴 Critical | 2 |
| 🟡 Medium | 3 |
| 🔵 Low | 4 |

**共 9 个**

---

## 🔴 Critical

### C1 — `update_max_nodes()` 扩容时创建幽灵 `client{i}` 槽位，无法彻底清理

**文件**: [scheduler.py:3874-3881](src/scheduler.py#L3874-L3881)

扩容时自动创建 `NodeState.OFFLINE` 的空槽位（`client2`、`client3` 等）：

```python
new_client_ids = [f"client{i}" for i in range(old_max, new_max)]
for cid in new_client_ids:
    if cid not in self.nodes:
        self.nodes[cid] = NodeInfo(node_id=cid, role=NodeRole.CLIENT, state=NodeState.OFFLINE)
```

缩容时虽然会尝试删除这些槽位，但条件严苛：必须满足 `state == OFFLINE and not address`（即从未被连接过）。一旦某个槽位被 TCP 连接过一次（写了 address），缩容就删不掉它。且这些无用的 client2/client3 条目永远不会被自动清理。

**影响**: 扩容后再缩容，节点列表里会残留永久离线幽灵节点。

**修复方向**: 扩容不再创建空槽位，只修改 `self._max_nodes` 上限。TCP 注册时动态接受节点直到达到上限即可，不需要预创建占位符。

---

### C2 — `register_node()` TCP 注册路径无容量检查

**文件**: [scheduler.py:1159-1171](src/scheduler.py#L1159-L1171)

```python
if node_id not in self.nodes:
    ...
    expected_client_ids = NodeRole.client_ids(MAX_NODES)  # ← 死代码，从未使用
    # 也允许用户在 MAX_NODES 之后添加的自定义节点
    logger.info(f"动态添加节点: {node_id} (type={node_type})")
    self.nodes[node_id] = NodeInfo(node_id=node_id, role=NodeRole.CLIENT, ...)
```

计算了 `expected_client_ids` 但从未用它来限制注册数量。任何 TCP 客户端都可以无限制注册，完全绕过 `max_nodes` 限制。

**影响**: 测试环境反复连接会无限堆积节点；生产环境恶意或错误客户端可注册任意数量节点。

**修复方向**: 在创建新节点前检查 `len([n for n in self.nodes.values() if n.role != "master"]) >= self._max_nodes - 1`，超限则返回 `False` 拒绝。

---

## 🟡 Medium

### M1 — `init_nodes()` 从数据库恢复所有历史节点，含陈旧/测试数据

**文件**: [scheduler.py:1043-1051](src/scheduler.py#L1043-L1051)

```python
for nid, row in db_nodes.items():
    if nid == "master":
        continue
    node = self._node_from_db(row)
    if RUN_MODE == "distributed":
        node.state = NodeState.OFFLINE
    self.nodes[nid] = node
```

数据库里存过的**所有**从节点都会被恢复到内存，然后标记为 OFFLINE。这意味着：

- 测试运行时注册过的节点会在下次启动时出现
- 临时连过一次的节点永远留在列表里
- 没有任何 TTL 或过期机制

**影响**: 跨重启累积，节点列表越来越长。

**修复方向**: 增加可选的离线过期逻辑（如 `last_heartbeat` 超过 N 天则跳过恢复）；或在 `get_nodes()` 中允许前端按状态过滤。

---

### M2 — 离线节点占用容量，阻塞新注册

**文件**: [scheduler.py:5927-5930](src/scheduler.py#L5927-L5930)

```python
non_master = [n for n in self.nodes.values() if n.role != "master"]
if len(non_master) >= self._max_nodes - 1:
    return {"status": "full", ...}
```

此容量检查统计了**所有**非 master 节点，包括永不再上线的离线/幽灵/陈旧节点。如果历史累积了 2 个离线节点且 `max_nodes=3`，则新节点无法注册。

**影响**: 结合 C1 和 M1，用户会发现扩容过、测试过之后，再也注册不了新节点，UI 提示"已达到最大从节点数量"。

**修复方向**: 容量检查仅统计在线节点，允许离线节点被新注册覆盖或手动删除后腾出位置。

---

### M3 — `get_nodes()` 无过滤机制，前端展示所有历史节点

**文件**: [api_server.py:2324-2331](src/api_server.py#L2324-L2331) + [AdminPanel.jsx:844-847](frontend/src/components/AdminPanel.jsx#L844-L847)

后端 `get_cluster_nodes()` 返回全量 `self.nodes`，前端 master 视图展示所有节点。离线、幽灵、陈旧、测试节点混在一起，用户无法区分哪些是真正活跃的节点。

**修复方向**: 前端增加"仅显示在线"过滤；增加"清理所有离线节点"批量操作按钮。

---

## 🔵 Low

### L1 — `deregister_node()` 仅标记 offline，不删除记录

**文件**: [scheduler.py:1223-1268](src/scheduler.py#L1223-L1268)

设计如此 — 注销保持记录以供审计。但结合 M1（DB 恢复）意味着这条记录永远不会消失，除非管理员手动调用 delete API。

**修复方向**: 这是可接受的设计选择（已补充 delete_node），但建议在 UI 中明确区分"注销"和"删除"并提示区别。

---

### L2 — `DEFAULT_LAYER_CONFIG` 硬编码 `client1`/`client2`

**文件**: [config.py:90-94](src/config.py#L90-L94)

```python
DEFAULT_LAYER_CONFIG = {
    "master":  (0, 8),
    "client1": (8, 16),
    "client2": (16, 24),
}
```

这是旧固定槽位架构的遗留默认值。当前运行时不会被使用（分层由 `compute_layer_assignment` 动态计算），但如果有代码路径回退到这个默认值，会为不存在的节点创建分层。

**修复方向**: 移除或改为空字典，确保任何回退路径都会报错而非静默使用幽灵节点配置。

---

### L3 — `NodeRole.client_ids()` 死代码

**文件**: [scheduler.py:87-89](src/scheduler.py#L87-L89)

`generates ["client1", "client2", ...]` 仅在 `register_node()` 中被赋给一个未使用的变量。相同的命名约定在 `update_max_nodes()` 中被独立重复实现。建议统一并使其有实际作用，或直接删除。

---

### L4 — `manual_register_node()` 可"更新"任意已存在节点

**文件**: [scheduler.py:5900-5925](src/scheduler.py#L5900-L5925)

已存在的节点（包括曾经在线又离线的真实节点）可被手动注册表单"更新"元数据。返回 `status: "updated"` 前端视为成功。但节点仍是 offline，用户体验困惑。

**修复方向**: 对真实 TCP 注册过的节点，手动更新不应改变关键标识信息（如 address），或至少提示用户该节点曾经在线。

---

## 修复优先级

1. **立即修复**: C1（幽灵槽位）、C2（TCP 注册无容量限制）、M2（离线占用容量）
2. **本迭代修复**: M3（前端过滤/批量删除）
3. **可延后**: M1（DB 恢复 TTL）、L1-L4（设计/代码清理）

---

## 验证

```bash
python -m pytest tests/test_scheduler.py tests/test_distributed_inference.py -q
```
