"""DIST-NEXT-4 回归：worker 存活阈值必须来自同一配置源且互相自洽。

背景
----
心跳容忍上限、task-worker 控制面 health 超时、Android 心跳间隔分属三处，历史上
两次错配都靠真机实测才发现：

* presence 心跳 45s 而容忍上限写死 10s ⇒ 节点被持续判过期（实测 `11.4s > 10s`）；
* 容量候选只看 `NodeState`、不看心跳 ⇒ TCP 半开（对端进程已死、不发 FIN）的节点
  在巡检窗口里**一直占容量**（实测：Y700 被带走后 master 仍把它算进规划）。

本文件把「心跳间隔 → 容忍上限」的关系固化下来：容量/readiness 用同一个上限，
控制面 health 先于它翻转，且心跳判据不晚于 TCP 巡检。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import HEARTBEAT_INTERVAL, PIPELINE_STEP_TIMEOUT  # noqa: E402
from scheduler_types import (  # noqa: E402
    ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS,
    ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS,
    TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS,
    WORKER_HEARTBEAT_MAX_AGE,
    assert_worker_liveness_thresholds,
    task_worker_control_plane_health_timeout_seconds,
    worker_heartbeat_max_age_seconds,
    worker_liveness_thresholds,
)


def test_thresholds_are_self_consistent():
    assert_worker_liveness_thresholds(HEARTBEAT_INTERVAL)
    thresholds = worker_liveness_thresholds(HEARTBEAT_INTERVAL)

    assert thresholds["worker_heartbeat_max_age_seconds"] == WORKER_HEARTBEAT_MAX_AGE
    for key in (
        "android_task_worker_heartbeat_interval_seconds",
        "android_presence_heartbeat_interval_seconds",
    ):
        assert thresholds["worker_heartbeat_max_age_seconds"] >= 2 * thresholds[key]


def test_android_task_worker_heartbeat_matches_the_device_constant():
    """主仓记录的心跳间隔必须等于设备侧常量（15s）—— 否则判据无意义。"""
    assert ANDROID_TASK_WORKER_HEARTBEAT_INTERVAL_SECONDS == 15.0
    # presence 心跳（45s）与 task-worker 心跳（15s）是不同的两个通道
    assert ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS == 45.0


def test_control_plane_layer_flips_before_the_planning_layer():
    """控制面 health（先停止派 stage）必须早于规划/readiness 的容忍上限。

    两层不同时翻转是**有意**的：若并成同一个数，节点会在「仍被规划进层区间」的
    同时「已不再接受 stage」，边界更难推理。
    """
    control = task_worker_control_plane_health_timeout_seconds(HEARTBEAT_INTERVAL)

    assert control == max(
        TASK_WORKER_HEALTH_TIMEOUT_FLOOR_SECONDS, HEARTBEAT_INTERVAL * 4.0,
    )
    assert control <= worker_heartbeat_max_age_seconds()
    # 现值：下界 30s 生效（心跳 3s × 4 = 12s），规划层容忍 120s
    assert control == 30.0
    assert worker_heartbeat_max_age_seconds() == 120.0


def test_heartbeat_judgement_is_not_later_than_tcp_inspection():
    """静默节点不能只在 TCP 巡检窗口里被发现，也不能被巡检窗口放过。

    巡检上限 = `HEARTBEAT_INTERVAL * (MAX_HEARTBEAT_MISSED + 1) + 2`；默认配置
    （step timeout 120s、心跳 3s）下为 131s > 120s ⇒ 心跳判据先翻转。
    若本机把 `QLH_PIPELINE_STEP_TIMEOUT` 调小，巡检窗口会更短、由巡检先判离线，
    同样是 fail-closed。
    """
    from tcp_comm import TCPServer

    inspection_window = (
        HEARTBEAT_INTERVAL * (TCPServer.MAX_HEARTBEAT_MISSED + 1) + 2
    )
    if inspection_window <= worker_heartbeat_max_age_seconds():
        pytest.skip(
            "PIPELINE_STEP_TIMEOUT 被调小 ⇒ 巡检窗口短于心跳容忍，"
            "静默节点由巡检先判离线（等价 fail-closed）"
        )
    assert worker_heartbeat_max_age_seconds() < inspection_window
    # 默认配置的口径写死一次，便于日志里出现「约 131s」时对照
    if PIPELINE_STEP_TIMEOUT == 120.0 and HEARTBEAT_INTERVAL == 3:
        assert inspection_window == 131


def test_scheduler_alias_points_at_the_single_source():
    """`scheduler.ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS` 只是别名。"""
    import scheduler

    assert (
        scheduler.ANDROID_HTTP_CLIENT_HEARTBEAT_INTERVAL_SECONDS
        == ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS
    )
    assert scheduler.WORKER_HEARTBEAT_MAX_AGE == WORKER_HEARTBEAT_MAX_AGE


def test_misconfigured_thresholds_fail_loud(monkeypatch):
    """错配必须在自检里直接失败，而不是等真机实测。"""
    import scheduler_types

    monkeypatch.setattr(scheduler_types, "WORKER_HEARTBEAT_MAX_AGE", 45.0)
    with pytest.raises(ValueError) as uncovered:
        scheduler_types.assert_worker_liveness_thresholds(HEARTBEAT_INTERVAL)
    assert "WORKER_HEARTBEAT_MAX_AGE" in str(uncovered.value)

    monkeypatch.setattr(scheduler_types, "WORKER_HEARTBEAT_MAX_AGE", 120.0)
    monkeypatch.setattr(
        scheduler_types, "ANDROID_PRESENCE_HEARTBEAT_INTERVAL_SECONDS", 200.0,
    )
    with pytest.raises(ValueError):
        scheduler_types.assert_worker_liveness_thresholds(HEARTBEAT_INTERVAL)
