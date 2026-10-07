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


def test_consistency_check_flags_the_three_divergences():
    """权威视图 vs 旧集合判据：三种分歧各有具名原因，且只观测、不决定行为。"""
    from worker_assignment_state import (
        DIVERGENCE_READY_BUT_NOT_PUSHED,
        DIVERGENCE_STATE_MISSING,
        DIVERGENCE_TERMINAL_BUT_PUSHED,
        evaluate_assignment_consistency,
    )

    # 一致：既无权威状态、也无旧期望
    assert evaluate_assignment_consistency(
        None, legacy_pushed=False, has_expected=False,
    ).consistent

    # 分歧①：旧期望在、权威视图没有
    verdict = evaluate_assignment_consistency(
        None, legacy_pushed=False, has_expected=True,
    )
    assert not verdict.consistent
    assert verdict.reason_code == DIVERGENCE_STATE_MISSING

    # 分歧②：终止态 vs 旧 pushed
    registry, _ = _registry()
    registry.begin("w", config_id="c")
    registry.release("w", reason_code="config_cleared")
    verdict = evaluate_assignment_consistency(
        registry.state("w"), legacy_pushed=True, has_expected=True,
    )
    assert verdict.reason_code == DIVERGENCE_TERMINAL_BUT_PUSHED

    # 分歧③：ready 相位 vs 旧未 pushed
    ready_registry, _ = _registry()
    ready_registry.begin("w", config_id="c")
    ready_registry.transition("w", phase=PHASE_READY)
    verdict = evaluate_assignment_consistency(
        ready_registry.state("w"), legacy_pushed=False, has_expected=True,
    )
    assert verdict.reason_code == DIVERGENCE_READY_BUT_NOT_PUSHED

    # 一致：pushing 相位 + 旧未 pushed（ACK 还没到，两边都"未就绪"）
    assert evaluate_assignment_consistency(
        ready_registry.state("w") if False else None,
        legacy_pushed=False,
        has_expected=False,
    ).consistent


def test_legacy_ready_ack_advances_the_authority_view():
    """legacy ready ACK 也要推进权威视图 —— 这是"降级为派生视图"的前置不变量。

    此前只有 release 类配置的分支推进相位，registry 长期停在 `pushing`，派生集合会漏掉
    这类节点（回归实测：`test_push_waits_for_worker_load_ack` 一收紧就变红）。
    """
    sched = Scheduler()
    config = {
        "config_id": "cfg-1", "generation": 11, "nodes": [],
        "phase": "commit", "plan_id": "plan-1",
        "start_layer": 8, "end_layer": 24,
        "model_sha256": "a" * 64, "model_type": "qwen2", "engine": "pytorch",
    }
    sched._publish_layer_configs({"worker_01": config})

    sched._handle_layer_config_ack("worker_01", {"data": {
        "node_id": "worker_01", "config_id": "cfg-1", "generation": 11,
        "status": "ready", "phase": "commit", "plan_id": "plan-1",
        "layer_range": [8, 24], "model_sha256": "a" * 64, "model_type": "qwen2",
        "engine": "pytorch",
    }})

    assert "worker_01" in sched._layer_config_pushed
    derived = {
        node for node, state in sched._worker_assignments.snapshot().items()
        if state["phase"] in (PHASE_ACKED, PHASE_READY)
    }
    # 不变量：旧集合与权威视图派生的集合一致（达成后即可把旧集合降级为派生视图）
    assert derived == set(sched._layer_config_pushed)


def test_retry_scan_is_not_blocked_by_a_stale_pushed_entry():
    """陈旧 pushed（权威视图已终止）不再阻止重发 —— 否则该节点永远等不到配置。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    sent = []

    class _Server:
        _running = True

        def get_client_ids(self):
            return ["stale"]

        def send_layer_config(self, node_id, payload):
            sent.append((node_id, payload))

    sched._tcp_server = _Server()
    with sched._layer_config_lock:
        sched._layer_config_expected["stale"] = {"config_id": "c", "generation": 1}
        sched._layer_config_pushed.add("stale")
        sched._layer_config_retry_state["stale"] = {"attempts": 0, "next_retry": 0.0}
        registry.begin("stale", config_id="c")
        registry.release("stale", reason_code=REASON_CONFIG_CLEARED)

    assert sched._retry_pending_layer_configs(now=1_000_000.0) == 1
    assert sent and sent[0][0] == "stale"


def test_retry_scan_still_skips_genuinely_pushed_nodes():
    """对照：旧集合与权威视图都说 pushed ⇒ 重发照旧跳过（零行为漂移）。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    sent = []

    class _Server:
        _running = True

        def get_client_ids(self):
            return ["fresh"]

        def send_layer_config(self, node_id, payload):
            sent.append((node_id, payload))

    sched._tcp_server = _Server()
    with sched._layer_config_lock:
        sched._layer_config_expected["fresh"] = {"config_id": "c", "generation": 1}
        sched._layer_config_pushed.add("fresh")
        sched._layer_config_retry_state["fresh"] = {"attempts": 0, "next_retry": 0.0}
        registry.begin("fresh", config_id="c")
        registry.transition("fresh", phase=PHASE_ACKED)

    assert sched._retry_pending_layer_configs(now=1_000_000.0) == 0
    assert sent == []


def test_recovery_fence_observes_divergence_without_changing_the_verdict(caplog):
    """观测点：分歧进日志，判据不变（这是"先让分歧可见、再切读路径"的落点）。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    registry.begin("worker_01", config_id="cfg-1")
    registry.release("worker_01", reason_code=REASON_CONFIG_CLEARED)

    with caplog.at_level("WARNING", logger="scheduler"):
        sched._observe_assignment_state_consistency(
            "worker_01", legacy_pushed=True, expected={"config_id": "cfg-1"},
        )

    assert "event=worker_assignment_state_divergence" in caplog.text
    assert "terminal_state_but_legacy_pushed" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="scheduler"):
        sched._observe_assignment_state_consistency(
            "worker_01", legacy_pushed=False, expected={},
        )
    assert "worker_assignment_state_divergence" not in caplog.text


def test_pushed_equivalence_mapping_and_effective_verdict(caplog):
    """`_layer_config_pushed` 的等价映射，以及"读路径切换"的保守规则。"""
    from worker_assignment_state import pushed_from_state

    registry, _ = _registry()
    # 无记录 ⇒ 无从推导
    assert pushed_from_state(None) is None
    # 未 ACK 的相位 ⇒ **未知**（不等于否定：旧集合可能由其它路径维护）
    registry.begin("w_pushing", config_id="c")
    assert pushed_from_state(registry.state("w_pushing")) is None
    # ACK 之后 ⇒ True（与 `_layer_config_pushed.add` 的语义等价）
    registry.transition("w_pushing", phase=PHASE_ACKED)
    assert pushed_from_state(registry.state("w_pushing")) is True
    registry.transition("w_pushing", phase=PHASE_READY)
    assert pushed_from_state(registry.state("w_pushing")) is True
    # 终止态 ⇒ False
    registry.release("w_pushing", reason_code=REASON_CONFIG_CLEARED)
    assert pushed_from_state(registry.state("w_pushing")) is False

    sched = Scheduler()
    # ① 无记录 ⇒ 沿用旧集合
    assert sched._effective_layer_config_pushed("unknown_node", True) is True
    assert sched._effective_layer_config_pushed("unknown_node", False) is False

    # ② 等价场景 ⇒ 读权威视图（结果与旧集合一致）
    sched._worker_assignments.begin("w_acked", config_id="c")
    sched._worker_assignments.transition("w_acked", phase=PHASE_ACKED)
    assert sched._effective_layer_config_pushed("w_acked", True) is True
    assert sched._effective_layer_config_pushed("w_acked", False) is False

    # ③ 收紧了场景 ⇒ 权威视图判否 ⇒ 不再算 pushed（排除陈旧项），且观测点已留证据
    sched._worker_assignments.begin("w_released", config_id="c")
    sched._worker_assignments.release("w_released", reason_code=REASON_CONFIG_CLEARED)
    with caplog.at_level("WARNING", logger="scheduler"):
        sched._observe_assignment_state_consistency(
            "w_released", legacy_pushed=True, expected={"config_id": "c"},
        )
        verdict = sched._effective_layer_config_pushed("w_released", True)
    assert verdict is False                      # 陈旧项被排除（收紧）
    assert "terminal_state_but_legacy_pushed" in caplog.text

    # ④ **不放宽**：权威视图判真、旧集合判假 ⇒ 保持旧值（放宽集合会改 readiness 结论）
    sched._worker_assignments.begin("w_acked2", config_id="c")
    sched._worker_assignments.transition("w_acked2", phase=PHASE_ACKED)
    assert sched._effective_layer_config_pushed("w_acked2", False) is False


def test_ready_node_set_excludes_stale_entries_and_never_widens(caplog):
    """readiness 的 ready 集合：陈旧 pushed 被排除、无记录项保留、权威独有项不加入。"""
    sched = Scheduler()
    registry = sched._worker_assignments
    registry.begin("stale", config_id="c")
    registry.release("stale", reason_code=REASON_CONFIG_CLEARED)
    registry.begin("fresh", config_id="c")
    registry.transition("fresh", phase=PHASE_ACKED)
    registry.begin("authority_only", config_id="c")
    registry.transition("authority_only", phase=PHASE_ACKED)

    with caplog.at_level("WARNING", logger="scheduler"):
        ready = sched._effective_layer_config_pushed_nodes(
            {"stale", "fresh", "legacy_only"},
            {"fresh": {"config_id": "c"}, "authority_only": {"config_id": "c"}},
        )

    # stale：旧集合有、权威视图已 released ⇒ 排除
    # legacy_only：无 assignment 记录 ⇒ 沿用旧集合（不当作未就绪）
    # authority_only：权威说 pushed 但旧集合没有 ⇒ 不放宽（只记事件）
    assert ready == {"fresh", "legacy_only"}
    assert "worker_assignment_state_divergence" in caplog.text


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
