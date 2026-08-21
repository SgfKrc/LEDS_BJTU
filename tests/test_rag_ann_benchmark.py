"""RAG 后置可准备：sqlite-vec 采用决策基准测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

import rag_ann


def _import_benchmark():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rag_ann_benchmark",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "rag_ann_benchmark.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_ann_benchmark"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    return module


benchmark = _import_benchmark()


def test_no_go_when_corpus_within_budget():
    report = benchmark.format_report(corpus_chunks=500, scan_budget=1000)
    assert report["decision"] == "NO_GO"
    assert report["reason"] == "bounded_cosine_within_scan_budget"
    assert report["schema_version"] == "qlh.rag.ann_benchmark.v1"


def test_gate_only_go_when_over_budget_and_extension_available(monkeypatch):
    monkeypatch.setattr(benchmark, "sqlite_vec_available", lambda: True)
    report = benchmark.format_report(corpus_chunks=50000, scan_budget=1000)
    assert report["decision"] == "GO"
    assert report["reason"] == "benchmark_gate_only"
    assert report["sqlite_vec_available"] is True


def test_no_go_when_extension_missing(monkeypatch):
    monkeypatch.setattr(benchmark, "sqlite_vec_available", lambda: False)
    report = benchmark.format_report(corpus_chunks=50000, scan_budget=1000)
    assert report["decision"] == "NO_GO"
    assert report["reason"] == "sqlite_vec_not_installed"


def test_report_fields_present():
    report = benchmark.format_report(corpus_chunks=100, scan_budget=50)
    assert set(report) >= {
        "schema_version", "corpus_chunks", "scan_budget",
        "sqlite_vec_available", "decision", "reason", "baseline",
    }
    assert report["baseline"] == "FTS5 + bounded cosine scan"


def test_invalid_corpus_rejected():
    with pytest.raises(ValueError):
        benchmark.format_report(corpus_chunks=-1, scan_budget=10)
