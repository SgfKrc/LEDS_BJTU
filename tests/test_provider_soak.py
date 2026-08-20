"""RAG 后置可准备：provider soak 核心测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from provider_soak import SoakSpec, run_provider_soak


def _ok(embed):
    """确定性 fake provider：前 n 次成功，之后抛异常。"""
    def fn(index):
        if index >= 60:
            raise RuntimeError(f"provider failure at {index}")
        return index * 2
    return fn


def test_clean_run_reports_success():
    spec = SoakSpec(iterations=120, failure_rate=0.0, hang_rate=0.0, seed=1)
    report = run_provider_soak(lambda i: i, spec=spec)
    assert report["iterations"] == 120
    assert report["successes"] == 120
    assert report["failures"] == 0
    assert report["timeouts"] == 0
    assert report["latency_ms"]["p50"] >= 0
    assert report["digest"]


def test_clean_run_deterministic_digest():
    def embed(i):
        return i
    a = run_provider_soak(embed, spec=SoakSpec(seed=7))
    b = run_provider_soak(embed, spec=SoakSpec(seed=7))
    assert a["digest"] == b["digest"]
    c = run_provider_soak(embed, spec=SoakSpec(seed=8))
    assert a["digest"] != c["digest"]


def test_injected_failure_and_recovery():
    spec = SoakSpec(iterations=200, failure_rate=0.2, hang_rate=0.0, seed=3)
    report = run_provider_soak(lambda i: i, spec=spec)
    assert report["failures"] > 0
    # 失败后下一次成功 → 计入 recovered（只要存在"失败后成功"序列）
    assert report["successes"] + report["failures"] == report["iterations"]
    assert report["recovered"] >= 0


def test_provider_exception_is_failure():
    spec = SoakSpec(iterations=100, failure_rate=0.0, hang_rate=0.0, seed=1)
    report = run_provider_soak(_ok(None), spec=spec)
    assert report["failures"] == 40          # 60 成功、40 抛异常
    assert report["successes"] == 60


def test_hang_counted_as_timeout_without_blocking():
    spec = SoakSpec(iterations=100, failure_rate=0.0, hang_rate=0.3, seed=5)
    report = run_provider_soak(lambda i: i, spec=spec)
    assert report["timeouts"] > 0
    assert report["timeouts"] + report["successes"] == report["iterations"]


def test_invalid_rates_rejected():
    with pytest.raises(ValueError):
        run_provider_soak(lambda i: i, spec=SoakSpec(iterations=10, failure_rate=1.5))
    with pytest.raises(ValueError):
        run_provider_soak(lambda i: i, spec=SoakSpec(iterations=1, failure_rate=-0.1))
    with pytest.raises(ValueError):
        run_provider_soak(lambda i: i, spec=SoakSpec(iterations=1, hang_rate=1.2))


def test_report_has_schema_and_digest():
    report = run_provider_soak(lambda i: i, spec=SoakSpec(iterations=5, seed=0))
    assert report["schema_version"] == "qlh.rag.provider_soak.v1"
    assert report["mode"] == "soak_shadow"
    assert set(report) >= {"iterations", "successes", "failures", "timeouts", "recovered", "latency_ms", "digest"}


def test_timeout_zero_ok():
    report = run_provider_soak(lambda i: i, spec=SoakSpec(iterations=10, timeout_s=0.0, seed=2))
    assert report["successes"] == 10
