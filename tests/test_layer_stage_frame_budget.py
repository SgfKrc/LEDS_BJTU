"""DIST-NEXT-2 回归：层段 hidden 的 **dispatch 前** wire 大小预检。

背景
----
层段的 `root_input.hidden_f32` 与中间段的 `output.hidden_out_f32` 都是 raw 数据经
base64 承载，而单条 task-worker 消息的上限是 `MAX_MESSAGE_BYTES`（8 MiB）。旧实现
没有任何预检 ⇒ 合法的长 prefill / 大 `n_embd` 会**先通过准入**，在 worker 执行完成
后才由协议层抛 `message_too_large`（表现为 stage 超时、回退或被 `distributed_required`
拒绝）—— 这是过晚的契约发现，而不是有效的 fail-closed。

本文件锁定两条不变量：
1. 预算函数与协议常量同源（`task_worker_protocol`），f32 / f16 双档一致；
2. `_execute_layer_stage_offer` 在 **reserve/execute 之前**就以稳定 reason code 拒绝。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

import api_server  # noqa: F401,E402  （加载 API composition root）
from scheduler import Scheduler  # noqa: E402
from scheduler_pipeline import (  # noqa: E402
    LayerStageFrameTooLarge,
    _assert_layer_stage_offer_fits_frame,
)
from task_worker_protocol import (  # noqa: E402
    MAX_MESSAGE_BYTES,
    STAGE_FRAME_RESERVE_BYTES,
    WorkerProtocolError,
    hidden_fits_stage_frame,
    hidden_wire_bytes,
    max_hidden_tokens,
    stage_payload_budget_bytes,
)


def test_hidden_wire_bytes_includes_base64_expansion():
    # 1024 个 f32 元素 = 4096 B ⇒ base64 后 5462 B（向上取整）
    assert hidden_wire_bytes(1, 1024, "float32") == 5462
    # f16 是同一公式的一半
    assert hidden_wire_bytes(1, 1024, "float16") == 2731
    with pytest.raises(WorkerProtocolError) as captured:
        hidden_wire_bytes(1, 1024, "bfloat16")
    assert captured.value.code == "unsupported_hidden_dtype"
    with pytest.raises(WorkerProtocolError) as invalid:
        hidden_wire_bytes(0, 1024)
    assert invalid.value.code == "invalid_hidden_spec"


def test_stage_frame_budget_tracks_protocol_limit():
    assert stage_payload_budget_bytes() == MAX_MESSAGE_BYTES - STAGE_FRAME_RESERVE_BYTES
    # 单帧预算内的最大 token 数：再 +1 就装不下（判据自洽）
    n_embd = 2048
    limit = max_hidden_tokens(n_embd, "float32")
    assert hidden_fits_stage_frame(limit, n_embd, "float32")
    assert not hidden_fits_stage_frame(limit + 1, n_embd, "float32")
    # f16 能把同一预算装下约两倍的 token
    assert max_hidden_tokens(n_embd, "float16") >= 2 * limit - 1


def test_precheck_passes_within_budget():
    _assert_layer_stage_offer_fits_frame(
        node_id="worker-b", n_tokens=512, n_embd=2048,
    )


def test_precheck_rejects_oversized_hidden_with_stable_reason():
    with pytest.raises(LayerStageFrameTooLarge) as captured:
        _assert_layer_stage_offer_fits_frame(
            node_id="worker-b", n_tokens=1, n_embd=2_000_000,
        )
    error = captured.value
    assert error.code == "route_a_stage_frame_too_large"
    text = str(error)
    assert text.startswith(error.code)
    assert f"wire={error.wire_bytes}" in text
    assert f"budget={error.budget_bytes}" in text
    assert error.wire_bytes > error.budget_bytes


def test_layer_stage_offer_rejects_before_touching_the_provider():
    """超预算的 hidden 必须在 reserve/execute **之前**结束。"""
    sched = Scheduler()
    provider_calls = []

    def _provider(node_id):
        provider_calls.append(node_id)
        raise AssertionError("预检失败时不得触碰远端 provider")

    sched._ensure_remote_task_worker_provider = _provider  # type: ignore[assignment]

    hidden = np.zeros((1, 2_000_000), dtype=np.float32)

    with pytest.raises(LayerStageFrameTooLarge) as captured:
        sched._execute_layer_stage_offer(
            node_id="worker-b",
            assignment={"start_layer": 2, "end_layer": 4},
            hidden_states=hidden,
            model_identity=object(),
            workflow_id="wf_framebudget01",
            request_id="request-framebudget01",
            stage_id="worker-b:step:0",
            context_size=2048,
        )

    assert captured.value.code == "route_a_stage_frame_too_large"
    assert provider_calls == []
