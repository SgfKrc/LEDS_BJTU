"""★ 2026-10-07（DIST-NEXT-3）：`WorkerAssignmentState` —— 单个 worker 的 assignment 权威视图。

背景（`docs/DIST后续核心链路缺陷审计-2026-10-07.md` P1-1 / P0-3）
----------------------------------------------------------------
主仓为「节点是否已拿到层配置」同时维护 `_layer_config_pushed` / `_layer_config_expected` /
`_layer_config_acks` / `_pipeline_load_transaction` / `_active_pipeline_capacity_plan` /
`_pipeline_recovery_state` 等集合，并且**没有 `assignment_id` 维度** —— 同一节点重连或重配时
只能靠 config/generation 的组合间接隔离，于是出现「某处判 ready、另一处仍 not_configured」
与 release/re-ACK 竞态。

本模块给出**单一事实源**的形状与状态机：

* 身份：`assignment_id`（每次下发新配置都换）+ `config_id` + `connection_generation` + lease；
* 相位：`staged → pushing → acked → ready`，终止态 `released` / `aborted`；
* **单一 reason code**：每次进入终止态只记录一个原因（复用 `task_worker_adapter` 的
  `RELEASE_REASON_*` 闭集，不另造第三套命名）。

**接线纪律（本步只写不读）**：registry 先在现有写入点同步维护（`_publish_layer_configs` /
`_clear_layer_config_state` / ACK / abort），**读路径暂不切换** ⇒ 零行为变化；后续轮次再把
容量、readiness、dispatch、release 的判据逐条切到本视图。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

#: 相位：非终止态按序前进；终止态只能从非终止态进入。
PHASE_STAGED = "staged"
PHASE_PUSHING = "pushing"
PHASE_ACKED = "acked"
PHASE_READY = "ready"
PHASE_RELEASED = "released"
PHASE_ABORTED = "aborted"

ACTIVE_PHASES = (PHASE_STAGED, PHASE_PUSHING, PHASE_ACKED, PHASE_READY)
TERMINAL_PHASES = (PHASE_RELEASED, PHASE_ABORTED)

#: 相位之间的允许前进方向（同相位重复设置视为幂等，不算越级）。
_PHASE_ORDER = {
    PHASE_STAGED: 0,
    PHASE_PUSHING: 1,
    PHASE_ACKED: 2,
    PHASE_READY: 3,
}

#: 本模块自有的 reason code（撤销原因复用 `task_worker_adapter.RELEASE_REASON_*`）。
REASON_CONFIG_PUBLISHED = "config_published"
REASON_CONFIG_ACKED = "config_acked"
REASON_CONFIG_CLEARED = "config_cleared"
REASON_RECONFIGURED = "reconfigured"
#: worker 侧确认释放该层配置（`stage` 级 released ACK）。
REASON_WORKER_RELEASED = "worker_released"


def new_assignment_id() -> str:
    return f"asg_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class WorkerAssignmentState:
    """一个 worker 当前 assignment 的完整身份与相位（不可变快照）。"""

    node_id: str
    assignment_id: str
    config_id: str = ""
    connection_generation: int = 0
    lease_id: str = ""
    lease_epoch: int = 0
    phase: str = PHASE_STAGED
    reason_code: str = ""
    updated_at: float = 0.0

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def snapshot(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "assignment_id": self.assignment_id,
            "config_id": self.config_id,
            "connection_generation": int(self.connection_generation),
            "lease_id": self.lease_id,
            "lease_epoch": int(self.lease_epoch),
            "phase": self.phase,
            "reason_code": self.reason_code,
            "updated_at": float(self.updated_at),
        }


class WorkerAssignmentRegistry:
    """per-node assignment 状态的唯一写入点（线程安全由调用方的配置锁保证）。

    ⚠️ **调用方必须在持有 `_layer_config_lock` 时使用**：本类不自己加锁，以免与既有
    锁序（配置锁 → 其它锁）冲突。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = new_assignment_id,
    ) -> None:
        self._clock = clock
        self._id_factory = id_factory
        self._states: dict[str, WorkerAssignmentState] = {}

    # -- 写入 ---------------------------------------------------------------

    def begin(
        self,
        node_id: str,
        *,
        config_id: str = "",
        connection_generation: int = 0,
        reason_code: str = REASON_CONFIG_PUBLISHED,
    ) -> WorkerAssignmentState:
        """开始（或换代）一个节点的 assignment —— **每次都换 `assignment_id`**。

        这是「同节点重连/重配必须换代际」的落点：旧 assignment 的身份与 lease 不会被
        新 assignment 复用，因此迟到的 ACK 无法冒充当前配置的就绪。
        """
        key = str(node_id or "")
        previous = self._states.get(key)
        if previous is not None and not previous.terminal:
            reason = REASON_RECONFIGURED
        else:
            reason = reason_code
        state = WorkerAssignmentState(
            node_id=key,
            assignment_id=self._id_factory(),
            config_id=str(config_id or ""),
            connection_generation=int(connection_generation),
            lease_id="",
            lease_epoch=0,
            phase=PHASE_PUSHING,
            reason_code=reason,
            updated_at=self._clock(),
        )
        self._states[key] = state
        return state

    def attach_lease(
        self, node_id: str, *, lease_id: str, lease_epoch: int,
    ) -> Optional[WorkerAssignmentState]:
        """把 lease 绑到当前 assignment（终止态不再接受变更）。"""
        state = self._states.get(str(node_id or ""))
        if state is None or state.terminal:
            return None
        updated = replace(
            state,
            lease_id=str(lease_id or ""),
            lease_epoch=int(lease_epoch),
            updated_at=self._clock(),
        )
        self._states[updated.node_id] = updated
        return updated

    def transition(
        self, node_id: str, *, phase: str, reason_code: str = "",
    ) -> Optional[WorkerAssignmentState]:
        """按序前进相位；越级/回退/终止态变更一律**拒绝**（返回 `None`）。

        拒绝而不是抛错：调用方（既有 ACK/重试路径）把它们当作「不是当前 assignment 的
        事件」忽略即可 —— 这正是替代「多集合交叉判定」的地方。
        """
        state = self._states.get(str(node_id or ""))
        if state is None or state.terminal:
            return None
        if phase in TERMINAL_PHASES:
            return None
        current = _PHASE_ORDER.get(state.phase, -1)
        target = _PHASE_ORDER.get(phase, -1)
        if target < 0 or target < current:
            return None
        updated = replace(
            state,
            phase=phase,
            reason_code=str(reason_code or ""),
            updated_at=self._clock(),
        )
        self._states[updated.node_id] = updated
        return updated

    def release(
        self, node_id: str, *, reason_code: str, aborted: bool = False,
    ) -> Optional[WorkerAssignmentState]:
        """进入终止态并记录**单一** reason code（重复撤销幂等返回当前状态）。"""
        key = str(node_id or "")
        state = self._states.get(key)
        if state is None:
            return None
        if state.terminal:
            return state
        updated = replace(
            state,
            phase=PHASE_ABORTED if aborted else PHASE_RELEASED,
            reason_code=str(reason_code or ""),
            lease_id="",
            lease_epoch=0,
            updated_at=self._clock(),
        )
        self._states[key] = updated
        return updated

    # -- 读取 ---------------------------------------------------------------

    def state(self, node_id: str) -> Optional[WorkerAssignmentState]:
        return self._states.get(str(node_id or ""))

    def assignment_id(self, node_id: str) -> str:
        state = self.state(node_id)
        return "" if state is None else state.assignment_id

    def phase(self, node_id: str) -> str:
        state = self.state(node_id)
        return "" if state is None else state.phase

    def is_ready(self, node_id: str) -> bool:
        state = self.state(node_id)
        return state is not None and state.phase == PHASE_READY

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            node_id: state.snapshot()
            for node_id, state in sorted(self._states.items())
        }

    def drop(self, node_id: str) -> None:
        """移除记录（测试与节点彻底退场时使用）。"""
        self._states.pop(str(node_id or ""), None)


def assignment_state_summary(
    states: Mapping[str, WorkerAssignmentState] | Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """按相位计数（诊断/日志用）。"""
    counts: dict[str, int] = {}
    for value in states.values():
        phase = value.phase if isinstance(value, WorkerAssignmentState) else str(
            value.get("phase", "")
        )
        counts[phase] = counts.get(phase, 0) + 1
    return counts


#: 一致性分歧的具名原因（`event=worker_assignment_state_divergence`）。
DIVERGENCE_STATE_MISSING = "state_missing_but_legacy_expected"
DIVERGENCE_TERMINAL_BUT_PUSHED = "terminal_state_but_legacy_pushed"
DIVERGENCE_READY_BUT_NOT_PUSHED = "state_ready_but_legacy_not_pushed"


@dataclass(frozen=True)
class AssignmentConsistency:
    """权威视图与旧集合判据的比对结果（仅用于**观测**，不改变任何判据）。"""

    consistent: bool
    reason_code: str = ""
    state_phase: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "reason_code": self.reason_code,
            "state_phase": self.state_phase,
        }


def evaluate_assignment_consistency(
    state: Optional[WorkerAssignmentState],
    *,
    legacy_pushed: bool,
    has_expected: bool,
) -> AssignmentConsistency:
    """把「权威视图」与「`_layer_config_expected` / `_layer_config_pushed`」对照。

    刻意**只判等价关系**，不决定任何行为：读路径切换前，先让真实分歧在日志里可见
    （`event=worker_assignment_state_divergence`），而不是继续靠多集合各自推断。
    """
    if state is None:
        if has_expected:
            return AssignmentConsistency(
                consistent=False, reason_code=DIVERGENCE_STATE_MISSING,
            )
        return AssignmentConsistency(consistent=True)
    if state.terminal and legacy_pushed:
        return AssignmentConsistency(
            consistent=False,
            reason_code=DIVERGENCE_TERMINAL_BUT_PUSHED,
            state_phase=state.phase,
        )
    if state.phase == PHASE_READY and not legacy_pushed:
        return AssignmentConsistency(
            consistent=False,
            reason_code=DIVERGENCE_READY_BUT_NOT_PUSHED,
            state_phase=state.phase,
        )
    return AssignmentConsistency(consistent=True, state_phase=state.phase)
