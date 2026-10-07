"""#52 回归：stage-only 路径必须为本地节点补 prepare。

背景
----
`src/scheduler_pipeline.py` 在「只有 A3 stage worker、没有 legacy ACK」时直接调用
`_commit_pipeline_load_transaction()`。而该函数**先**调 `prepare_pipeline_tokenizer()`，
它在 `src/model_module.py` 因 `is_pipeline_prepared` 为假而抛
`当前没有已准备的 distributed-only 流水线模型`
⇒ 真机表现为 `503 reason_code=pipeline_local_commit_failed`。

`is_pipeline_prepared` 需要 `_pipeline_distributed_only and _pipeline_descriptor`，
两者**只**由 `prepare_pipeline_model()` 设置，而该调用在 legacy 路径由远端 ACK 事务驱动
—— stage-only 路径没有任何东西驱动它。

修复：commit 前若本地 `local_assignment` 存在且 `is_pipeline_prepared` 为假，
先为本地补一次 `prepare_pipeline_model()`（已 prepared 时为 no-op）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402  （加载 API composition root，配置 SchedulerCallbackSet）
from scheduler import Scheduler  # noqa: E402


def _sched_with(host_attrs):
    sched = Scheduler()
    sched._tcp_server = type("Server", (), {
        "_running": True,
        "send_layer_config": lambda self, node_id, payload: None,
    })()
    sched._host = type("Host", (), host_attrs)()
    sched._pipeline_load_transaction = {
        "config_id": "cfg-1",
        "generation": 1,
        "phase": "preparing",
        "plan": {
            "admitted": True,
            "plan_id": "plan-1",
            "model_id": "qwen-test",
            "total_layers": 4,
            "assignments": [
                {
                    "node_id": "master", "start_layer": 0, "end_layer": 2,
                    "has_embedding": True, "has_lm_head": False,
                },
                {
                    "node_id": "worker-b", "start_layer": 2, "end_layer": 4,
                    "has_embedding": False, "has_lm_head": True,
                },
            ],
        },
        "worker_ids": {"worker-b"},
        "prepared_nodes": {"worker-b"},
    }
    return sched


def test_commit_prepares_local_segment_when_not_prepared():
    """stage-only：本地未 prepared ⇒ commit 前补一次 prepare_pipeline_model。"""
    calls = []

    sched = _sched_with({
        "is_pipeline_prepared": False,
        "_full_model_path": "C:/models/qwen-test",
        "quant_type": "none",
        "prepare_pipeline_model": lambda self, **kw: calls.append(kw),
        "prepare_pipeline_tokenizer": lambda self: object(),
        "load_layer_range": lambda self, *a, **kw: None,
    })

    sched._commit_pipeline_load_transaction("cfg-1")

    assert calls, "stage-only 路径未为本地节点补 prepare（#52 回归）"
    assert calls[0]["layer_range"] == (0, 2)
    assert calls[0]["model_path"] == "C:/models/qwen-test"
    assert calls[0]["model_id"] == "qwen-test"


def test_commit_skips_prepare_when_already_prepared():
    """已 prepared ⇒ 不再重复 prepare。"""
    calls = []

    sched = _sched_with({
        "is_pipeline_prepared": True,
        "_full_model_path": "C:/models/qwen-test",
        "quant_type": "none",
        "prepare_pipeline_model": lambda self, **kw: calls.append(kw),
        "prepare_pipeline_tokenizer": lambda self: object(),
        "load_layer_range": lambda self, *a, **kw: None,
    })

    sched._commit_pipeline_load_transaction("cfg-1")

    assert calls == []


def test_commit_records_local_commit_even_without_prepare_api():
    """host 不提供 prepare_pipeline_model 时不得崩，仍要走完 commit。"""
    sched = _sched_with({
        "is_pipeline_prepared": False,
        "_full_model_path": "C:/models/qwen-test",
        "quant_type": "none",
        "prepare_pipeline_tokenizer": lambda self: object(),
        "load_layer_range": lambda self, *a, **kw: None,
    })

    sched._commit_pipeline_load_transaction("cfg-1")  # 不应抛异常
