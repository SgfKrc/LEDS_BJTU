from __future__ import annotations

import ast
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "torch_hardware_admit_under_test", ROOT / "scripts" / "torch_hardware_admit.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report(device: str, *, host: str = "workstation", precision: str = "fp32") -> dict:
    phase_row = {
        phase: {"uninstrumented_wall": {"samples_ms": [1.0, 1.1, 0.9], "cv": 0.08}}
        for phase in ("prefill", "decode")
    }
    cache_row = {
        phase: {"structural_layout_fingerprint": f"layout-{phase}"}
        for phase in ("prefill", "decode")
    }
    range_row = {**phase_row, "actual_kv_cache": cache_row}
    return {
        "schema_version": MODULE.REPORT_SCHEMA,
        "model_identity": {"weights_sha256": "weights"},
        "tokenizer_sha256": "tokenizer",
        "precision_mode": precision,
        "runtime": {
            "device": {"type": device}, "host_identity": host, "threads": 8,
            "python": "3.12.10", "torch": "2.13.0+cpu" if device == "cpu" else "2.13.0+cu126",
            "transformers": "5.17.0", "torch_cuda_runtime": None if device == "cpu" else "12.6",
        },
        "workloads": {
            key: {
                "prefill_tokens": int(key),
                "decode_steps": 8,
                "prompt_token_sha256": f"prompt-{key}",
                "reference": {"generated_token_ids": [1, 2, 3]},
            }
            for key in ("64", "256")
        },
        "range_profiles": {
            f"{workload}:{layer_end}": range_row
            for workload in ("64", "256")
            for layer_end in (4, 12, 24)
        },
        "heterogeneous_pairs": {},
        "provenance": {"commit": "abc", "working_tree_dirty": True},
    }


def _paired_reports():
    cpu = _report("cpu")
    cuda = _report("cuda")
    direction = {
        "prefill_argmax_exact": True,
        "generated_tokens_exact": True,
        "prefill": {"samples_ms": [3.0, 3.1, 2.9], "cv": 0.03},
        "decode": {"samples_ms": [1.0, 1.1, 0.9], "cv": 0.08},
        "kv_cache_after_decode": {
            "first_stage_cache": {"structural_layout_fingerprint": "stage-1"},
            "second_stage_cache": {"structural_layout_fingerprint": "stage-2"},
        },
    }
    cuda["heterogeneous_pairs"] = {
        key: {"directions": {"cpu_to_cuda": direction, "cuda_to_cpu": direction}}
        for key in ("64", "256")
    }
    return cpu, cuda


def test_sample_summary_includes_spread_and_rejects_invalid_values():
    result = MODULE.summarize_samples([3.0, 1.0, 2.0])

    assert result["samples_ms"] == [3.0, 1.0, 2.0]
    assert result["median_ms"] == 2.0
    assert result["cv"] == pytest.approx(0.408248, abs=1e-6)
    with pytest.raises(ValueError, match="finite positive"):
        MODULE.summarize_samples([1.0, float("nan")])


def test_phase_measurement_order_balances_and_pairs_each_phase():
    order = MODULE._phase_measurement_order(5)

    assert order == [
        "prefill", "decode", "decode", "prefill", "prefill", "decode",
        "decode", "prefill", "prefill", "decode",
    ]
    assert order.count("prefill") == order.count("decode") == 5


def test_diagnostic_clock_summary_accepts_quantized_zero_samples():
    result = MODULE.summarize_nonnegative_samples([0.0, 0.0, 1.0])

    assert result["samples_ms"] == [0.0, 0.0, 1.0]
    assert result["cv"] == pytest.approx(1.41421356237)
    assert MODULE.summarize_nonnegative_samples([0.0, 0.0])["cv"] is None


def test_cpu_affinity_parser_rejects_duplicates_and_accepts_ordered_indices():
    assert MODULE._parse_cpu_affinity("0, 2,10") == [0, 2, 10]
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        MODULE._parse_cpu_affinity("0,0")


def test_cpu_affinity_mismatch_fails_closed_when_both_reports_attest_it():
    cpu, cuda = _paired_reports()
    cpu["runtime"]["cpu_affinity"] = {"requested_logical_cpus": [0, 2], "mask": 5, "applied": True}
    cuda["runtime"]["cpu_affinity"] = {"requested_logical_cpus": [0, 1], "mask": 3, "applied": True}

    result = MODULE.compare_reports(cpu, cuda, max_cv=0.1)

    assert not result["same_host_cpu_cuda_split_admitted"]
    assert "cpu_affinity_mismatch" in result["reasons"]


def test_interop_thread_mismatch_fails_closed_when_both_reports_attest_it():
    cpu, cuda = _paired_reports()
    cpu["runtime"]["interop_threads"] = 1
    cuda["runtime"]["interop_threads"] = 2

    result = MODULE.compare_reports(cpu, cuda, max_cv=0.1)

    assert not result["same_host_cpu_cuda_split_admitted"]
    assert "interop_thread_count_mismatch" in result["reasons"]


def test_hardware_telemetry_parsers_accept_realistic_windows_outputs():
    gpu = MODULE._parse_nvidia_smi_csv(
        "0, NVIDIA GeForce RTX 4060 Laptop GPU, 780, 3105, 60, 22.4, [N/A], 45, 1234, 0x1, 0x1\n",
    )
    assert gpu["clock_sm_mhz"] == 780.0
    assert gpu["temperature_c"] == 60.0
    assert gpu["power_limit_w"] is None
    assert gpu["event_reasons_active"] == "0x1"
    assert gpu["throttle_reasons_active"] == "0x1"

    system = MODULE._parse_typeperf_csv(
        '"(PDH-CSV 4.0)","\\\\HOST\\Processor(_Total)\\% Processor Time",'
        '"\\\\HOST\\Thermal Zone Information(\\_TZ.TZ00)\\Temperature"\n'
        '"09/24/2026 12:00:00.000","12.5","301.0"\n',
    )
    assert system["\\\\HOST\\Processor(_Total)\\% Processor Time"] == 12.5
    assert system["\\\\HOST\\Thermal Zone Information(\\_TZ.TZ00)\\Temperature"] == 301.0


def test_hardware_telemetry_sampler_stops_and_records_empty_source():
    sampler = MODULE._HardwareTelemetrySampler("none")

    sampler.start()
    result = sampler.stop()

    assert result["source"] == "none"
    assert result["sample_count"] == 0
    assert result["sampling_complete"] is True


def test_cache_summary_captures_actual_tensor_layout_and_bytes():
    torch = pytest.importorskip("torch")
    cache = SimpleNamespace(
        layers=[SimpleNamespace(keys=torch.zeros((1, 2, 3, 4), dtype=torch.float16), values=None)],
    )

    result = MODULE.summarize_cache(cache, torch)

    assert result["tensor_count"] == 1
    assert result["tensor_bytes"] == 48
    assert result["tensors"][0]["shape"] == [1, 2, 3, 4]
    assert result["tensors"][0]["dtype"] == "float16"
    assert len(result["structural_layout_fingerprint"]) == 64
    assert len(result["placement_fingerprint"]) == 64


def test_cache_structure_fingerprint_is_independent_of_device_placement():
    torch = pytest.importorskip("torch")
    left = MODULE.summarize_cache(
        SimpleNamespace(keys=torch.zeros((1, 2, 3), dtype=torch.float32)), torch,
    )
    right_tensor = torch.zeros((1, 2, 3), dtype=torch.float32, device="meta")
    right = MODULE.summarize_cache(SimpleNamespace(keys=right_tensor), torch)

    assert left["structural_layout_fingerprint"] == right["structural_layout_fingerprint"]
    assert left["placement_fingerprint"] != right["placement_fingerprint"]


def test_cpu_cuda_pair_admits_only_same_host_matched_fp32_and_both_exact_directions():
    cpu, cuda = _paired_reports()

    result = MODULE.compare_reports(cpu, cuda, max_cv=0.1)

    assert result["matched_fp32_cpu_cuda_pair"]
    assert result["same_host_cpu_cuda_split_admitted"]
    assert not result["cross_host_admitted"]
    assert not result["production_runtime_enabled"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda cpu, cuda: cuda["runtime"].update(host_identity="another-host"), "host_identity_mismatch"),
        (lambda cpu, cuda: cuda["runtime"].update(threads=4), "thread_count_mismatch"),
        (lambda cpu, cuda: cuda.update(precision_mode="fp16"), "matched_fp32_control_missing"),
        (lambda cpu, cuda: cuda["heterogeneous_pairs"]["64"]["directions"].pop("cuda_to_cpu"), "matrix_incomplete"),
        (lambda cpu, cuda: cuda["heterogeneous_pairs"]["64"]["directions"]["cpu_to_cuda"].update(generated_tokens_exact=False), "heterogeneous_correctness_failed"),
    ],
)
def test_admission_fails_closed_on_identity_precision_completeness_or_correctness(mutation, reason):
    cpu, cuda = _paired_reports()
    mutation(cpu, cuda)

    result = MODULE.compare_reports(cpu, cuda, max_cv=0.1)

    assert not result["same_host_cpu_cuda_split_admitted"]
    assert any(reason in item for item in result["reasons"])


def test_script_import_does_not_eagerly_import_torch():
    source = (ROOT / "scripts" / "torch_hardware_admit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager_torch_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "torch"
    ]

    assert eager_torch_imports == []


def test_script_is_not_ignored():
    result = __import__("subprocess").run(
        ["git", "check-ignore", "scripts/torch_hardware_admit.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 1
