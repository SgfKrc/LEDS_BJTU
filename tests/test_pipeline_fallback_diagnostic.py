"""`#78` 缺口 2：流水线不可用时的**回退拒绝**必须带出真实 `reason_code` 与可读原因。

背景（2026-10-09 实测）：TUI 只显示
「当前模型仅以分布式流水线模式准备，禁止整模回退；请等待从节点就绪」，
而真因（如 `pipeline_capacity_workers_unavailable` = 从节点未声明工件）只留在 logcat，
用户无从自助诊断。本文件锁定修复后的契约：

1. 无诊断信息时，**不许乱编** —— 退回稳定的兜底 code；
2. 有 `_pipeline_load_transaction`（容量准入被拒）时，`error` 必须含 **`reason_code` 与中文说明**；
3. **非流式**与**流式**两条回退路径必须**一致**（同一处修复不能只改一半）；
4. 更早的 `_active_pipeline_capacity_plan` 作为兜底来源也要能取到。

做法：沿用 `test_scheduler_relay_segment.py` 的最小假实例模式（`object.__new__` + 只 stub 协作方法），
毫秒级、不需要模型与拓扑。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scheduler_pipeline import SchedulerPipelineMixin  # noqa: E402


def _make(**attrs):
    """最小假实例：只带 `_run_full_model_inference`/`_run_full_model_inference_stream` 用到的属性。"""
    obj = object.__new__(SchedulerPipelineMixin)
    obj._host = types.SimpleNamespace(is_pipeline_prepared=True, is_loaded=False)
    obj._pipeline_load_transaction = None
    obj._active_pipeline_capacity_plan = None
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_diagnostic_empty_when_nothing_known():
    """没有任何容量信息时返回空 —— 不许凭空编造原因。"""
    code, detail = _make()._pipeline_unavailable_diagnostic()
    assert code == ""
    assert detail == ""


def test_diagnostic_from_rejected_transaction():
    """容量准入被拒（`_pipeline_load_transaction`）⇒ 取出 code + 中文说明。"""
    obj = _make(
        _pipeline_load_transaction={
            "phase": "rejected",
            "reason_code": "pipeline_capacity_workers_unavailable",
            "plan": {"reason_code": "pipeline_capacity_workers_unavailable"},
        }
    )
    code, detail = obj._pipeline_unavailable_diagnostic()
    assert code == "pipeline_capacity_workers_unavailable"
    assert "分层 worker" in detail  # 人话说明必须真的给出


def test_diagnostic_includes_english_reason_and_excluded_nodes():
    """求解器给的英文 `reason` 与 `excluded_nodes` 也要带出来（那是定位线索）。"""
    obj = _make(
        _pipeline_load_transaction={
            "reason_code": "pipeline_distributed_workers_unavailable",
            "plan": {
                "reason_code": "pipeline_distributed_workers_unavailable",
                "reason": "distributed placement requires at least two usable PC nodes",
                "excluded_nodes": [
                    {"node_id": "android-x", "reason": "layer_artifact_missing"},
                ],
            },
        }
    )
    code, detail = obj._pipeline_unavailable_diagnostic()
    assert code == "pipeline_distributed_workers_unavailable"
    assert "two usable PC nodes" in detail
    assert "android-x" in detail and "layer_artifact_missing" in detail


def test_diagnostic_falls_back_to_active_plan():
    """事务已被清理时，仍能从 `_active_pipeline_capacity_plan` 取到原因。"""
    obj = _make(
        _active_pipeline_capacity_plan={
            "reason_code": "pipeline_layer_range_coverage_insufficient",
        }
    )
    code, detail = obj._pipeline_unavailable_diagnostic()
    assert code == "pipeline_layer_range_coverage_insufficient"
    assert detail  # 映射表里有说明


def test_unknown_code_still_reports_code():
    """映射表没收录的 code 也必须原样带出（不能因为没文案就丢掉 code）。"""
    obj = _make(
        _pipeline_load_transaction={
            "reason_code": "some_brand_new_reason",
            "plan": {"reason_code": "some_brand_new_reason"},
        }
    )
    code, detail = obj._pipeline_unavailable_diagnostic()
    assert code == "some_brand_new_reason"
    assert detail == ""


def test_non_stream_error_carries_reason_code():
    """非流式回退拒绝：`error` 里要有人能看懂的原因，且返回结构化 `reason_code`。"""
    obj = _make(
        _pipeline_load_transaction={
            "reason_code": "pipeline_capacity_workers_unavailable",
            "plan": {"reason_code": "pipeline_capacity_workers_unavailable"},
        }
    )
    result = obj._run_full_model_inference("hi")
    assert result["response"] == ""
    assert "禁止整模回退" in result["error"]
    assert "pipeline_capacity_workers_unavailable" in result["error"]
    assert result["reason_code"] == "pipeline_capacity_workers_unavailable"


def test_stream_error_carries_reason_code_and_matches_non_stream():
    """流式路径必须与非流式**同源**（同一修复不能只改一半）。"""
    txn = {
        "reason_code": "pipeline_capacity_workers_unavailable",
        "plan": {"reason_code": "pipeline_capacity_workers_unavailable"},
    }
    obj = _make(_pipeline_load_transaction=dict(txn))
    streamed = next(iter(obj._run_full_model_inference_stream("hi")))
    assert streamed["done"] is True
    assert "pipeline_capacity_workers_unavailable" in streamed["error"]
    assert streamed["reason_code"] == "pipeline_capacity_workers_unavailable"

    non_stream = _make(_pipeline_load_transaction=dict(txn))._run_full_model_inference("hi")
    # 两条路径的 error 文案应当一致（去掉各自 payload 结构差异后逐字相同）
    assert streamed["error"] == non_stream["error"]


def test_no_diagnostic_falls_back_to_stable_code():
    """拿不到任何原因时，也不能给出空 `reason_code`（否则前端无法判分支）。"""
    obj = _make()
    assert obj._run_full_model_inference("hi")["reason_code"] == (
        "pipeline_full_model_fallback_forbidden"
    )
    streamed = next(iter(_make()._run_full_model_inference_stream("hi")))
    assert streamed["reason_code"] == "pipeline_full_model_fallback_forbidden"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
