"""集群状态里的容量投影契约（2026-10-08 第 4 条静默路径：节点 online 却不参与）。

`_pipeline_capacity_status_snapshot()` 的两条硬约束：
1. 缓存优先 —— `/api/cluster/status` 会被 App/TUI 反复轮询，有权威决策在缓存里时
   不得再求解、更不得改变准入状态；
2. 没有缓存时要给出真实原因 —— 一律报 `unknown` 等于把"为什么没在干活"又藏回去。

本测试用最小 stub 承载 mixin，直接锁定这两条与节点级判定顺序。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scheduler_cluster import SchedulerClusterMixin


class _NodeInfoStub:
    def __init__(self, node_id, role="client"):
        self.node_id = node_id
        self.role = role

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "role": self.role,
            "state": "online",
            "is_available": True,
        }


class _SchedulerStub(SchedulerClusterMixin):
    """只携带投影会读到的属性。"""

    def __init__(self, nodes, *, active=None, transaction=None, solved=None,
                 solver_error=None):
        self.nodes = {node.node_id: node for node in nodes}
        self._nodes_lock = threading.RLock()
        self._layer_config_lock = threading.RLock()
        self._active_pipeline_capacity_plan = active
        self._pipeline_load_transaction = transaction
        self._solved = solved
        self._solver_error = solver_error
        self.solver_calls = 0

    def get_pipeline_capacity_plan(self, *args, **kwargs):
        self.solver_calls += 1
        if self._solver_error is not None:
            raise self._solver_error
        return self._solved


def _plan(**overrides):
    plan = {
        "status": "admitted",
        "admitted": True,
        "reason_code": "",
        "assignments": [],
        "control_only_nodes": [],
    }
    plan.update(overrides)
    return plan


def test_unavailable_solver_result_is_reported_as_not_computed():
    stub = _SchedulerStub([_NodeInfoStub("master", role="master")])

    snapshot = stub._pipeline_capacity_status_snapshot()

    assert stub.solver_calls == 1
    assert snapshot["status"] == "unknown"
    assert snapshot["admitted"] is None
    assert snapshot["reason_code"] == "pipeline_capacity_not_computed"
    assert snapshot["participating_node_ids"] == []


def test_cold_cache_projects_the_solved_plan():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master"), _NodeInfoStub("android-1")],
        solved=_plan(
            assignments=[{"node_id": "master", "start_layer": 0, "end_layer": 24}],
            control_only_nodes=["android-1"],
        ),
    )

    snapshot = stub._pipeline_capacity_status_snapshot()
    nodes = stub._snapshot_nodes(capacity=snapshot)

    assert stub.solver_calls == 1
    assert snapshot["admitted"] is True
    assert snapshot["participating_node_ids"] == ["master"]
    assert nodes["master"]["pipeline_participating"] is True
    assert nodes["master"]["pipeline_exclusion_reason"] == ""
    assert nodes["android-1"]["pipeline_participating"] is False
    assert nodes["android-1"]["pipeline_exclusion_reason"] == "capacity_plan_control_only"


def test_active_plan_short_circuits_the_solver():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master")],
        active=_plan(assignments=[{"node_id": "master", "start_layer": 0, "end_layer": 24}]),
        solved=_plan(
            assignments=[{"node_id": "android-1", "start_layer": 0, "end_layer": 24}]
        ),
    )

    snapshot = stub._pipeline_capacity_status_snapshot()

    assert stub.solver_calls == 0
    assert snapshot["participating_node_ids"] == ["master"]


def test_transaction_short_circuits_the_solver():
    transaction = {
        "phase": "preparing",
        "prepared_nodes": {"master"},
        "ready_nodes": set(),
        "worker_ids": {"master", "android-1"},
        "plan": _plan(
            assignments=[{"node_id": "android-1", "start_layer": 0, "end_layer": 24}]
        ),
    }
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master"), _NodeInfoStub("android-1")],
        transaction=transaction,
        solver_error=RuntimeError("must not be called"),
    )

    snapshot = stub._pipeline_capacity_status_snapshot()

    assert stub.solver_calls == 0
    assert snapshot["transaction_phase"] == "preparing"
    assert snapshot["prepared_node_count"] == 1
    assert snapshot["ready_node_count"] == 0
    assert snapshot["worker_count"] == 2
    assert snapshot["participating_node_ids"] == ["android-1"]


def test_rejected_plan_marks_online_node_as_control_only():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master")],
        active=_plan(
            status="rejected",
            admitted=False,
            reason_code="pipeline_distributed_workers_unavailable",
            control_only_nodes=["master"],
        ),
    )

    nodes = stub._snapshot_nodes()

    assert nodes["master"]["state"] == "online"
    assert nodes["master"]["pipeline_participating"] is False
    assert nodes["master"]["pipeline_exclusion_reason"] == "capacity_plan_control_only"


def test_node_outside_admitted_plan_is_reported_as_unassigned():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master"), _NodeInfoStub("android-1")],
        active=_plan(
            assignments=[{"node_id": "master", "start_layer": 0, "end_layer": 24}],
        ),
    )

    nodes = stub._snapshot_nodes()

    assert nodes["android-1"]["pipeline_participating"] is False
    assert nodes["android-1"]["pipeline_exclusion_reason"] == "capacity_plan_unassigned"


def test_node_excluded_by_a_rejected_plan_reports_the_plan_reason():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master"), _NodeInfoStub("android-1")],
        active=_plan(
            status="rejected",
            admitted=False,
            reason_code="pipeline_distributed_workers_unavailable",
        ),
    )

    nodes = stub._snapshot_nodes()

    assert nodes["android-1"]["pipeline_exclusion_reason"] == (
        "pipeline_distributed_workers_unavailable"
    )


def test_solver_failure_degrades_to_unknown_instead_of_raising():
    stub = _SchedulerStub(
        [_NodeInfoStub("master", role="master")],
        solver_error=RuntimeError("boom"),
    )

    snapshot = stub._pipeline_capacity_status_snapshot()
    nodes = stub._snapshot_nodes(capacity=snapshot)

    assert stub.solver_calls == 1
    assert snapshot["admitted"] is None
    assert snapshot["reason_code"] == "pipeline_capacity_not_computed"
    assert nodes["master"]["pipeline_participating"] is False
    assert nodes["master"]["pipeline_exclusion_reason"] == "capacity_plan_unknown"
