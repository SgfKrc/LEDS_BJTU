"""DIST-NEXT-3 回归：`WorkerAssignmentState` 的换代际、相位推进与单一 reason code。

审计 P1-1 / P0-3：同一节点重连或重配时只有「config/generation 的组合」可以间接隔离，
且撤销原因散落多处。本文件的判据：

1. 每次下发新配置都换 `assignment_id`（旧 lease 不被复用）；
2. 相位按序前进，越级/回退/终止态变更一律拒绝（这类事件就是「不属于当前 assignment」）；
3. 进入终止态只记**一个** reason code，且重复撤销幂等；
4. 现有写路径（publish / ACK / clear）已同步进权威视图（读路径尚未切换 ⇒ 零行为变化）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402
from scheduler import Scheduler  # noqa: E402
from worker_assignment_state import (  # noqa: E402
    PHASE_ABORTED,
    PHASE_ACKED,
    PHASE_PUSHING,
    PHASE_READY,
    PHASE_RELEASED,
    REASON_CONFIG_ACKED,
    REASON_CONFIG_CLEARED,
    REASON_CONFIG_PUBLISHED,
    REASON_RECONFIGURED,
    REASON_WORKER_RELEASED,
    WorkerAssignmentRegistry,
    WorkerAssignmentState,
    assignment_state_summary,
)


def _registry():
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"asg_test{counter['n']:04d}"

    return WorkerAssignmentRegistry(id_factory=next_id), counter


def test_begin_assigns_a_fresh_identity_every_time():
    """同一节点重配 ⇒ 新 assignment_id、旧 lease 不复用（换代际）。"""
    registry, counter = _registry()

    first = registry.begin(
        "worker_01", config_id="cfg-1", connection_generation=7,
    )
    assert first.assignment_id == "asg_test0001"
    assert first.phase == PHASE_PUSHING
    assert first.config_id == "cfg-1"
    assert first.connection_generation == 7
    assert first.reason_code == REASON_CONFIG_PUBLISHED

    registry.attach_lease(
        "worker_01", lease_id="lease_aaa", lease_epoch=3,
    )
    assert registry.state("worker_01").lease_id == "lease_aaa"

    second = registry.begin(
        "worker_01", config_id="cfg-2", connection_generation=8,
    )
    assert second.assignment_id == "asg_test0002"
    assert second.assignment_id != first.assignment_id
    # 旧 lease 不得跟着新 assignment 走
    assert second.lease_id == ""
    assert second.lease_epoch == 0
    assert second.reason_code == REASON_RECONFIGURED
    assert counter["n"] == 2


def test_phase_progress_only_forward_and_only_on_the_current_assignment():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")

    # 允许**跨级前进**（真实路径里 ACK 可能直接给出就绪，没有单独的 acked 阶段）
    assert registry.transition("worker_01", phase=PHASE_READY) is not None
    assert registry.is_ready("worker_01")

    # 回退被拒
    assert registry.transition("worker_01", phase=PHASE_ACKED) is None
    assert registry.phase("worker_01") == PHASE_READY
    # 未知相位被拒
    assert registry.transition("worker_01", phase="nonsense") is None
    # 未登记的节点不凭空产生状态
    assert registry.transition("worker_99", phase=PHASE_ACKED) is None

    # 正常顺序前进也成立
    registry.begin("worker_02", config_id="cfg-1")
    assert registry.transition("worker_02", phase=PHASE_ACKED) is not None
    assert registry.transition("worker_02", phase=PHASE_READY) is not None
    assert registry.phase("worker_02") == PHASE_READY


def test_release_records_exactly_one_reason_and_is_idempotent():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")
    registry.transition("worker_01", phase=PHASE_ACKED)

    released = registry.release("worker_01", reason_code=REASON_CONFIG_CLEARED)

    assert released.phase == PHASE_RELEASED
    assert released.reason_code == REASON_CONFIG_CLEARED
    assert released.terminal
    assert released.lease_id == ""
    # 重复撤销：状态不变、原因不被覆盖
    again = registry.release("worker_01", reason_code="something_else")
    assert again.reason_code == REASON_CONFIG_CLEARED
    # 终止态不接受相位推进，也不接受新 lease
    assert registry.transition("worker_01", phase=PHASE_READY) is None
    assert registry.attach_lease(
        "worker_01", lease_id="lease_late", lease_epoch=9,
    ) is None


def test_aborted_release_uses_the_aborted_phase():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")

    state = registry.release(
        "worker_01", reason_code="pipeline_local_commit_failed", aborted=True,
    )

    assert state.phase == PHASE_ABORTED
    assert state.reason_code == "pipeline_local_commit_failed"


def test_snapshot_and_summary_are_serializable():
    registry, _ = _registry()
    registry.begin("worker_01", config_id="cfg-1")
    registry.transition("worker_01", phase=PHASE_READY)
    registry.begin("worker_02", config_id="cfg-1")

    snapshot = registry.snapshot()

    assert set(snapshot) == {"worker_01", "worker_02"}
    assert snapshot["worker_01"]["phase"] == PHASE_READY
    assert snapshot["worker_01"]["assignment_id"] == "asg_test0001"
    assert assignment_state_summary(snapshot) == {PHASE_READY: 1, PHASE_PUSHING: 1}
    # 传 dataclass 也支持
    assert assignment_state_summary({
        "worker_01": WorkerAssignmentState(
            node_id="worker_01", assignment_id="asg_x", phase=PHASE_RELEASED,
        ),
    }) == {PHASE_RELEASED: 1}


def test_scheduler_write_paths_feed_the_authority_view():
    """现有写路径（publish / ACK / clear）已同步进权威视图。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    assert isinstance(registry, WorkerAssignmentRegistry)

    sched._publish_layer_configs({
        "worker_01": {
            "config_id": "cfg-1",
            "generation": 11,
            "nodes": [],
        },
    })
    state = registry.state("worker_01")
    assert state is not None
    assert state.phase == PHASE_PUSHING
    assert state.config_id == "cfg-1"
    assert state.assignment_id.startswith("asg_")

    # 清配置 ⇒ 终止态 + 单一 reason
    sched._clear_layer_config_state("worker_01")
    released = registry.state("worker_01")
    assert released.phase == PHASE_RELEASED
    assert released.reason_code == REASON_CONFIG_CLEARED


def test_reconfiguration_replaces_the_previous_assignment():
    """同节点二次下发 ⇒ 权威视图换成新代际（旧 ACK 无法冒充）。"""
    sched = Scheduler()
    registry = sched._worker_assignments

    sched._publish_layer_configs({
        "worker_01": {"config_id": "cfg-1", "generation": 11, "nodes": []},
    })
    first = registry.assignment_id("worker_01")
    sched._publish_layer_configs({
        "worker_01": {"config_id": "cfg-2", "generation": 12, "nodes": []},
    })
    second = registry.assignment_id("worker_01")

    assert first and second and first != second
    assert registry.state("worker_01").config_id == "cfg-2"
    assert registry.state("worker_01").reason_code == REASON_RECONFIGURED


def test_ack_path_reaches_acked_and_released_states():
    """ACK 路径：非 released ⇒ acked；released ⇒ 终止态（reason=worker_released）。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    # `release: True` 的配置才会走 released ACK 分支（否则要求层区间等 legacy 字段）
    sched._publish_layer_configs({
        "worker_01": {
            "config_id": "cfg-1", "generation": 11, "release": True, "nodes": [],
        },
    })

    def ack(**extra):
        return {"data": {
            "node_id": "worker_01",
            "config_id": "cfg-1",
            "generation": 11,
            **extra,
        }}

    # released ACK（worker 确认释放该层配置）⇒ 权威视图进终止态 + 单一 reason。
    # 非 released 的 ready ACK 还要求层区间等 legacy 字段，其相位推进由 registry
    # 单测直接覆盖（见 test_phase_progress_only_forward...）。
    sched._handle_layer_config_ack("worker_01", ack(status="released", release=True))
    state = registry.state("worker_01")
    assert state.phase == PHASE_RELEASED
    assert state.reason_code == REASON_WORKER_RELEASED
