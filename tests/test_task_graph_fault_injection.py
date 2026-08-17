"""TG-OPT-G5.2 动态故障注入矩阵（本机，无网络/GPU）。

对照 G5.1 只读审计的 reason code 清单，对真实 TaskGraphCoordinator 注入
故障并断言运行时全部 fail-closed（拒绝 + 正确 reason code + 状态未被污染），
且注入后的 workflow 快照再过 G5.1 审计应无意外 gap（审计与运行时闭环一致）。

覆盖矩阵：
  M1 迟到结果（旧 attempt 在 winner 提交后到达）  -> winner_already_committed
  M2 重复提交（同 attempt 同 digest）             -> already_committed（幂等）
  M3 冲突重复（同 attempt 不同 digest）           -> winner_digest_mismatch
  M4 winner 后第二个 attempt 结果到达             -> winner_already_committed
  M5 epoch 回退（旧 epoch 重放）                  -> attempt_epoch_mismatch
  M6 stale lease（attempt epoch < stage epoch）   -> stale_lease_epoch
  M7 provider 身份错配                            -> provider_identity_mismatch
  M8 跨 stage/工作流 attempt                      -> attempt_not_owned_by_stage
  M9 工作流终态后提交（无 winner）                -> workflow_terminal
  M10 非法输出 schema                             -> invalid_result_schema
  M11 非 running attempt 提交                     -> attempt_not_running

结论分叉：矩阵全拒绝 -> G5.2 No-Go 升级为"动态故障注入验证无缺口"；
发现未拒绝路径 -> 按 reason code 开修复票。
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_graph import (  # noqa: E402
    StageSpec,
    TaskGraphCoordinator,
    WorkflowExecutionError,
)
from tests.helpers.task_graph_common import single_stage  # noqa: E402
from task_graph_attempt_audit import audit_task_graph_attempts  # noqa: E402
from task_provider import (  # noqa: E402
    DeterministicFakeProvider,
    ProviderRegistry,
    StageResult,
)


@pytest.fixture
def winner_workflow():
    """primary 失败 1 次 -> fallback winner；返回 (coordinator, workflow, old_attempt, winner)。"""
    registry = ProviderRegistry()
    primary = DeterministicFakeProvider("primary", execution_failures=1)
    fallback = DeterministicFakeProvider(
        "fallback",
        output_factory=lambda request, cancel_event: {"content": "winner"},
    )
    registry.register(primary)
    registry.register(fallback)
    coordinator = TaskGraphCoordinator(provider_registry=registry)

    _, workflow = coordinator.run(
        single_stage(),
        "answer",
        {"message": "question"},
        workflow_id="wf_fault_matrix",
    )
    stage = workflow["stages"][0]
    old_attempt, winner = stage["attempts"]
    yield coordinator, workflow, old_attempt, winner
    coordinator.close()


@pytest.fixture
def running_stage():
    """stage running、attempt running、无 winner——运行中注入窗口。

    prior_failures=1 使第一个 attempt 失败（epoch1, expired），第二个 attempt
    （epoch2）阻塞在 execute——stage.lease_epoch=2 > 旧 attempt.epoch=1。
    返回 (coordinator, wf_id, attempts)。
    """
    release = threading.Event()
    registry = ProviderRegistry()
    # primary 第一次 execute 失败（retryable，attempt 已创建/已开始）
    registry.register(DeterministicFakeProvider(
        "primary", execution_failures=1,
        execution_error_code="fake_worker_disconnected"))
    # retry 后的 fallback attempt 阻塞在 block_event（epoch2, running, 无 winner）
    registry.register(DeterministicFakeProvider(
        "fallback", block_event=release))
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    completed = []
    errors = []

    def run_workflow():
        try:
            completed.append(coordinator.run(
                [StageSpec("answer", "full_inference", provider="primary",
                           fallback_providers=("fallback",), pure=True)],
                "answer", {"message": "q"}, workflow_id="wf_running01"),
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=run_workflow)
    thread.start()
    # 轮询等待 fallback attempt 进入 running（替代 Barrier，消除竞态）
    stage = None
    deadline = time.time() + 8
    while time.time() < deadline:
        stage = coordinator.get("wf_running01")["stages"][0]
        attempts = stage["attempts"]
        if (stage["state"] == "running" and len(attempts) >= 2
                and attempts[-1]["state"] == "running"
                and attempts[-1]["lease_epoch"] == 2):
            break
        time.sleep(0.01)
    assert stage is not None and stage["state"] == "running"
    assert stage["winner_attempt_id"] == ""
    assert len(attempts) >= 2
    yield coordinator, "wf_running01", attempts, release, thread, completed, errors
    release.set()
    thread.join(5)


def _submit(coordinator, attempt, *, provider=None, output=None, epoch=None,
            workflow_id="wf_fault_matrix", stage_id="answer"):
    return coordinator.submit_stage_result(
        workflow_id,
        stage_id,
        StageResult(
            output={"content": "x"} if output is None else output,
            provider_id=provider or attempt["provider"],
            attempt_id=attempt["attempt_id"],
            lease_epoch=epoch if epoch is not None else attempt["lease_epoch"],
        ),
    )


def _audit_clean(coordinator, workflow_id="wf_fault_matrix") -> list:
    """注入后的 snapshot 再过 G5.1 审计，返回 gap 列表（应为空）。"""
    snapshot = coordinator.get(workflow_id)
    report = audit_task_graph_attempts(snapshot)
    assert report["status"] == "passed", report["gaps"]
    assert not report["gaps"], report["gaps"]
    return report["gaps"]


# ---- M1-M4：winner 已提交后的迟到/重复/冲突/双赢家 ----

def test_m1_late_result_is_fenced(winner_workflow):
    coordinator, _, old_attempt, _ = winner_workflow
    r = _submit(coordinator, old_attempt)
    assert r["status"] == "rejected"
    assert r["reason"] == "winner_already_committed"
    _audit_clean(coordinator)


def test_m2_duplicate_commit_is_idempotent(winner_workflow):
    coordinator, _, _, winner = winner_workflow
    # winner 输出 = fallback output_factory 固定产物；同 digest 重提 → 幂等
    r = _submit(coordinator, winner, output={"content": "winner"})
    assert r["status"] == "idempotent"
    assert r["reason"] == "already_committed"
    _audit_clean(coordinator)


def test_m3_conflicting_duplicate_is_rejected(winner_workflow):
    coordinator, _, _, winner = winner_workflow
    r = _submit(coordinator, winner, output={"content": "different"})
    assert r["status"] == "rejected"
    assert r["reason"] == "winner_digest_mismatch"
    _audit_clean(coordinator)


def test_m4_second_attempt_after_winner_is_rejected(winner_workflow):
    coordinator, _, old_attempt, _ = winner_workflow
    # 当前运行时没有并发 speculative attempt；这里验证已存在的另一个
    # attempt 在 winner 后返回时不能成为第二个 winner。
    r = _submit(coordinator, old_attempt, output={"content": "second-winner"})
    assert r["status"] == "rejected"
    assert r["reason"] == "winner_already_committed"
    _audit_clean(coordinator)


# ---- M5-M6：epoch 类 ----

def test_m5_epoch_rollback_is_rejected(running_stage):
    coordinator, wf_id, attempts, release, thread, completed, errors = (
        running_stage)
    running = attempts[-1]
    r = _submit(coordinator, running, workflow_id=wf_id,
                epoch=running["lease_epoch"] - 1)
    assert r["status"] == "rejected"
    assert r["reason"] == "attempt_epoch_mismatch"
    release.set()
    thread.join(5)
    assert not errors
    assert completed
    _audit_clean(coordinator, wf_id)


def test_m6_stale_lease_epoch_rejected_before_winner(running_stage):
    coordinator, wf_id, attempts, release, thread, completed, errors = (
        running_stage)
    old_attempt = attempts[0]  # epoch1 expired；stage.lease_epoch=2
    r = _submit(coordinator, old_attempt, workflow_id=wf_id)
    assert r["status"] == "rejected"
    assert r["reason"] == "stale_lease_epoch"
    release.set()
    thread.join(5)
    assert not errors
    assert completed
    _audit_clean(coordinator, wf_id)



# ---- M7-M8：身份类 ----

def test_m7_provider_identity_mismatch_rejected(running_stage):
    coordinator, wf_id, attempts, release, thread, completed, errors = (
        running_stage)
    running = attempts[-1]
    r = _submit(coordinator, running, provider="intruder",
                workflow_id=wf_id)
    assert r["status"] == "rejected"
    assert r["reason"] == "provider_identity_mismatch"
    release.set()
    thread.join(5)
    assert not errors
    assert completed
    _audit_clean(coordinator, wf_id)


def test_m8_attempt_not_owned_by_stage(running_stage):
    coordinator, wf_id, attempts, release, thread, completed, errors = (
        running_stage)
    r = coordinator.submit_stage_result(
        wf_id,
        "answer",
        StageResult(
            output={"content": "x"},
            provider_id="primary",
            attempt_id="attempt_never_owned",
            lease_epoch=1,
        ),
    )
    assert r["status"] == "rejected"
    assert r["reason"] == "attempt_not_owned_by_stage"
    release.set()
    thread.join(5)
    assert not errors
    assert completed
    _audit_clean(coordinator, wf_id)


# ---- M9：终态 ----

def test_m9_workflow_terminal_rejects_submission():
    registry = ProviderRegistry()
    failing = DeterministicFakeProvider("primary", execution_failures=99)
    failing2 = DeterministicFakeProvider("fallback", execution_failures=99)
    registry.register(failing)
    registry.register(failing2)
    coordinator = TaskGraphCoordinator(provider_registry=registry)
    try:
        try:
            coordinator.run(single_stage(), "answer", {"message": "q"},
                            workflow_id="wf_terminal_matrix")
        except WorkflowExecutionError:
            pass  # 全失败 -> 终态，从快照取
        workflow = coordinator.get("wf_terminal_matrix")
        stage = workflow["stages"][0]
        attempt = stage["attempts"][0]
        assert workflow["state"] in ("failed", "cancelled")
        r = _submit(coordinator, attempt,
                    workflow_id="wf_terminal_matrix")
        assert r["status"] == "rejected"
        assert r["reason"] == "workflow_terminal"
        _audit_clean(coordinator, "wf_terminal_matrix")
    finally:
        coordinator.close()


# ---- M10：schema ----

def test_m10_invalid_output_schema_rejected(winner_workflow):
    coordinator, _, _, winner = winner_workflow
    r = _submit(coordinator, winner, output="not-a-dict")
    assert r["status"] == "rejected"
    assert r["reason"] == "invalid_result_schema"
    _audit_clean(coordinator)


# ---- M11：非 running attempt ----

def test_m11_non_running_attempt_is_rejected(running_stage):
    coordinator, wf_id, attempts, release, thread, completed, errors = (
        running_stage)
    running = attempts[-1]

    # attempt_not_running 位于 epoch/stage fencing 之后。合法状态机不会留下
    # “当前 epoch、Stage running、Attempt terminal”的组合，因此本故障注入
    # 在锁内暂时破坏该单一状态，再在释放 Provider 前恢复。
    with coordinator._lock:
        workflow = coordinator._workflows[wf_id]
        with workflow.lock:
            attempt = workflow.stages["answer"].attempts[-1]
            assert attempt.attempt_id == running["attempt_id"]
            assert attempt.state == "running"
            attempt.state = "expired"
    try:
        r = _submit(coordinator, running, workflow_id=wf_id)
        assert r["status"] == "rejected"
        assert r["reason"] == "attempt_not_running"
    finally:
        with coordinator._lock:
            workflow = coordinator._workflows[wf_id]
            with workflow.lock:
                workflow.stages["answer"].attempts[-1].state = "running"
    release.set()
    thread.join(5)
    assert not errors
    assert completed
    _audit_clean(coordinator, wf_id)


# ---- 矩阵汇总：全部注入路径必须拒绝（fail-closed 总量断言） ----

def test_winner_guard_batch_has_zero_unaccepted_conflicts(winner_workflow):
    """批量复核 winner 已提交分支，冲突结果不得被接受。"""
    coordinator, _, old_attempt, winner = winner_workflow
    attempts = [
        ("M1", _submit(coordinator, old_attempt)),
        ("M2", _submit(coordinator, winner, output={"content": "winner"})),
        ("M3", _submit(coordinator, winner, output={"content": "different"})),
        ("M4", _submit(coordinator, old_attempt,
                       output={"content": "second-winner"})),
        ("M5", _submit(coordinator, winner,
                       epoch=winner["lease_epoch"] - 1)),
        ("M8", coordinator.submit_stage_result(
            "wf_fault_matrix", "answer",
            StageResult(output={"content": "x"}, provider_id="fallback",
                        attempt_id="attempt_never_owned", lease_epoch=1))),
        ("M10", _submit(coordinator, winner, output="not-a-dict")),
    ]
    for name, r in attempts:
        assert r["status"] in ("rejected", "idempotent"), (
            f"{name} 未被拒绝: {r}")
        if name != "M2":
            assert r["status"] == "rejected", f"{name} 未 fail-closed: {r}"
    # 所有拒绝都有 reason code
    assert all(r.get("reason") for _, r in attempts)
    # 注入后 winner 未被污染、审计闭环干净
    current = coordinator.get("wf_fault_matrix")
    stage = current["stages"][0]
    assert stage["winner_attempt_id"] == winner["attempt_id"]
    assert stage["state"] == "completed"
    _audit_clean(coordinator)
