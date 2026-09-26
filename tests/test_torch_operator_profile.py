from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.torch_operator_profile import (  # noqa: E402
    collect_operator_rows,
    make_dispatch_capture_mode,
    summarize_samples,
)


def test_operator_capture_reports_shape_dtype_device_and_phase():
    torch = pytest.importorskip("torch")
    from torch.profiler import ProfilerActivity, profile

    capture = make_dispatch_capture_mode("decode")
    left = torch.ones((2, 3), dtype=torch.float32)
    right = torch.ones((3, 4), dtype=torch.float32)
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with capture:
            result = left @ right
    rows = collect_operator_rows(
        profiler, capture, phase="decode", invocations_per_sample=1,
    )

    matmul = [row for row in rows if row["operator"] == "aten::mm"]
    assert result.shape == (2, 4)
    assert len(matmul) == 1
    assert matmul[0]["phase"] == "decode"
    assert matmul[0]["calls"] == 1
    assert matmul[0]["profiler_operator_events"] == 1
    assert matmul[0]["inputs"]["args"]["tuple"][0]["tensor"] == {
        "shape": [2, 3],
        "stride": [3, 1],
        "dtype": "float32",
        "device": "cpu",
        "layout": "strided",
    }
    assert matmul[0]["self_cpu_us_total"] > 0
    assert matmul[0]["self_device_us_total"] == 0


def test_profile_signature_distinguishes_scalar_operation_arguments():
    torch = pytest.importorskip("torch")
    from torch.profiler import ProfilerActivity, profile

    capture = make_dispatch_capture_mode("prefill")
    tensor = torch.ones((2, 3))
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with capture:
            torch.sum(tensor, dim=0)
            torch.sum(tensor, dim=1)
    rows = collect_operator_rows(
        profiler, capture, phase="prefill", invocations_per_sample=1,
    )
    reductions = [row for row in rows if row["operator"] == "aten::sum"]

    assert len(reductions) == 2
    assert {
        row["inputs"]["args"]["tuple"][1]["list"][0]
        for row in reductions
    } == {0, 1}


def test_profile_rejects_dispatch_profiler_event_count_mismatch():
    torch = pytest.importorskip("torch")
    from torch.profiler import ProfilerActivity, profile

    capture = make_dispatch_capture_mode("decode")
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with capture:
            torch.ones((2, 2)) + 1
    label = next(iter(capture.records))
    capture.records[label]["calls"] += 1

    with pytest.raises(RuntimeError, match="event mismatch"):
        collect_operator_rows(profiler, capture, phase="decode", invocations_per_sample=1)


def test_multiple_aten_children_still_match_one_dispatch_scope():
    label = "qlh.operator.test"
    scope = SimpleNamespace(
        name=label,
        device_type=SimpleNamespace(name="CPU"),
        cpu_children=[
            SimpleNamespace(name="aten::add", self_cpu_time_total=2.0),
            SimpleNamespace(name="aten::mul", self_cpu_time_total=3.0),
        ],
    )
    profile = SimpleNamespace(events=lambda: [scope])
    capture = SimpleNamespace(records={label: {
        "operator": "aten::composite",
        "inputs": {},
        "calls": 1,
        "outputs": Counter(),
    }})

    rows = collect_operator_rows(
        profile, capture, phase="prefill", invocations_per_sample=1,
    )

    assert rows[0]["profiler_operator_events"] == 2
    assert rows[0]["unmatched_dispatch_calls"] == 0


def test_sample_summary_is_sorted_and_reports_variance():
    result = summarize_samples([3, 1, 2])

    assert result["samples_ms"] == [1.0, 2.0, 3.0]
    assert result["median_ms"] == 2.0
    assert result["population_stddev_ms"] == pytest.approx(0.816497, abs=1e-6)


def test_script_import_does_not_import_torch_for_edge_safe_tool_discovery():
    import ast

    source = (ROOT / "scripts" / "torch_operator_profile.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_torch_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "torch"
    ]
    assert top_level_torch_imports == []
