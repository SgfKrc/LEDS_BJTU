"""★ A1 / X 档（Y 档第二条缺口 1）：relay 段节点必须**不占层**、且被标 `engine="relay_middle"`。

背景（实测，2026-09-28）：产品路径跨机 relay 第一次跑时，master 把 relay worker 当**普通层节点**
分层（日志 `client_TABLET-2TLUCNU8: Layer 18-24 (6层)`），而它没有本地模型 ⇒ 无法确认
「分层释放 ACK」⇒ `pipeline_distributed_workers_unavailable` ⇒ `/api/chat` 503。
根因：「动态分层」这条路（`compute_layer_assignment` / `_simple_weight_assignment`）**不认 relay 段** ——
第 2/3 批的 `engine="relay_middle"` 只接到了 `scheduler_pipeline` 里的 assignment 构造路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import api_server  # noqa: F401,E402 - 与 tests/test_scheduler.py 保持同样的导入副作用
from scheduler import NodeInfo, NodeState, Scheduler  # noqa: E402

RELAY_SPEC = {
    "role": "middle", "host": "127.0.0.1", "port": 50283,
    "n_embd": 896, "timeout": 60.0,
}


def _node(node_id: str, role: str) -> NodeInfo:
    return NodeInfo(
        node_id=node_id, role=role, state=NodeState.ONLINE,
        node_type="pc", device_info={},
    )


def _scheduler(monkeypatch: pytest.MonkeyPatch, relay_ids: set[str]) -> Scheduler:
    """造一个三节点调度器；`relay_ids` 里的节点被 `_relay_segment_for_worker` 认成 relay 段。"""
    sched = Scheduler()
    sched.nodes = {
        "master": _node("master", "master"),
        "client1": _node("client1", "client"),
        "client2": _node("client2", "client"),
    }

    def _relay_for(node_id, routing_preference="auto"):
        return RELAY_SPEC if node_id in relay_ids else None

    monkeypatch.setattr(sched, "_relay_segment_for_worker", _relay_for, raising=True)
    return sched


def test_relay_worker_gets_no_layers_and_is_marked_relay_middle(monkeypatch):
    """★ 缺口 1 的判据：relay 段节点 `layers_count == 0` 且 `engine == "relay_middle"`。"""
    sched = _scheduler(monkeypatch, {"client2"})

    result = {a["node_id"]: a for a in sched.compute_layer_assignment()}

    relay = result["client2"]
    assert relay["layers_count"] == 0, relay
    assert relay["engine"] == "relay_middle"
    assert relay["start_layer"] == relay["end_layer"]
    assert relay["has_embedding"] is False
    assert relay["has_lm_head"] is False

    # 真正跑层的两个节点仍覆盖全部 24 层（relay 节点**不**吃层）。
    learning = [a for a in result.values() if a["node_id"] != "client2"]
    assert sum(a["layers_count"] for a in learning) == 24
    assert max(a["end_layer"] for a in learning) == 24


def test_relay_worker_alone_still_appears_in_the_plan(monkeypatch):
    """只有 relay 节点参与时也必须出条目 —— 否则流水线直接空转（`return []`）。"""
    sched = _scheduler(monkeypatch, {"master", "client1", "client2"})

    result = {a["node_id"]: a for a in sched.compute_layer_assignment()}

    assert set(result) == {"master", "client1", "client2"}
    assert all(a["layers_count"] == 0 for a in result.values())
    assert all(a["engine"] == "relay_middle" for a in result.values())


def test_switch_off_keeps_layer_plan_unchanged(monkeypatch):
    """开关默认关（`_relay_segment_for_worker` 恒 None）⇒ 分层结果里**没有** relay 条目。"""
    sched = _scheduler(monkeypatch, set())

    result = sched.compute_layer_assignment()

    assert all(a.get("engine") != "relay_middle" for a in result)
    assert all(a["layers_count"] > 0 for a in result)
    assert sum(a["layers_count"] for a in result) == 24


def test_guard_split_relay_nodes_uses_the_real_predicate(monkeypatch):
    """★「该红必须红」守卫：`_split_relay_nodes` 必须用**真判据**摘节点。

    若把它改成恒返回 `(node_list, [])`，上面两条会**静默失效**（relay 节点又会被分层），
    而本用例立刻红 —— 它直接断言"谁被摘出来"。
    """
    sched = _scheduler(monkeypatch, {"client2"})

    remaining, relay = sched._split_relay_nodes([
        {"node_id": "master", "role": "master"},
        {"node_id": "client2", "role": "client"},
    ])

    assert [n["node_id"] for n in remaining] == ["master"]
    assert [n["node_id"] for n in relay] == ["client2"]
