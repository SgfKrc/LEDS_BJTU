"""TaskGraph 测试公共 helper（T13 收敛：消除跨文件重复实现与参数漂移）。

原分布：test_task_graph_fencing.py 与 test_task_graph_fault_injection.py
各有一份 `_single_stage`（fencing 版有 pure 参数、fault 版硬编码 pure=True
——参数默认已漂移）。收敛到单一实现：统一签名 single_stage(*, pure=True)。
"""
from __future__ import annotations

from task_graph import StageSpec  # noqa: F401


def single_stage(*, pure: bool = True, lease_timeout_seconds: float = 1.0):
    """单 stage（answer/full_inference，primary + fallback）StageSpec 列表。

    pure 参数化：默认 True（与 fencing 版一致），fault 注入需要时传 False。
    """
    return [
        StageSpec(
            "answer",
            "full_inference",
            provider="primary",
            fallback_providers=("fallback",),
            pure=pure,
            lease_timeout_seconds=lease_timeout_seconds,
        ),
    ]


def assert_no_active_reservations(coordinator) -> None:
    """断言无活跃 reservation（fencing 收敛）。"""
    assert all(
        status["active_reservations"] == 0
        for status in coordinator.provider_status()
    )
