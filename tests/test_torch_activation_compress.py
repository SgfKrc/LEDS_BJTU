from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "torch_activation_compress_under_test",
    ROOT / "scripts" / "torch_activation_compress.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_current_tensor_serializer_round_trip_preserves_shape_dtype_and_values():
    torch = pytest.importorskip("torch")
    hidden = torch.randn((2, 3, 257), dtype=torch.float32)

    result = MODULE.roundtrip_activation(hidden, "none", torch)

    assert result["tensor"].shape == hidden.shape
    assert result["tensor"].dtype == hidden.dtype
    assert torch.equal(result["tensor"], hidden)
    assert result["payload_bytes"] == result["legacy_serialized_bytes"]
    assert result["source_tensor_bytes"] == hidden.numel() * hidden.element_size()
    assert result["max_abs_error"] == 0


@pytest.mark.parametrize(
    ("mode", "max_abs_error"),
    [("f16", 0.01), ("int8_block128", 0.04), ("int4_block128", 0.4)],
)
def test_compressed_round_trip_reports_bounded_error_and_payload(mode, max_abs_error):
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    hidden = torch.randn((2, 3, 257), dtype=torch.float32)
    MODULE.roundtrip_activation(hidden, "none", torch)

    result = MODULE.roundtrip_activation(hidden, mode, torch)

    assert result["tensor"].shape == hidden.shape
    assert result["tensor"].dtype == torch.float32
    assert result["payload_bytes"] < result["legacy_serialized_bytes"]
    assert result["payload_ratio_vs_raw_activation"] > 0
    assert result["max_abs_error"] <= max_abs_error
    assert result["codec_roundtrip_ms"] >= 0


def test_compressed_mode_requires_a_cached_legacy_baseline():
    torch = pytest.importorskip("torch")
    hidden = torch.zeros((1, 1, 333), dtype=torch.float32)
    MODULE._LEGACY_SIZE_CACHE.pop((tuple(hidden.shape), str(hidden.dtype)), None)

    with pytest.raises(ValueError, match="none-mode baseline"):
        MODULE.roundtrip_activation(hidden, "int8_block128", torch)


def test_argmax_comparison_fails_closed_on_prefill_shape_mismatch():
    reference = {"prefill_argmax": [[1, 2, 3]], "generated": [4, 5]}
    candidate = {"prefill_argmax": [[1, 2, 3, 4]], "generated": [4, 5]}

    result = MODULE._comparison(reference, candidate)

    assert not result["exact"]


def test_argmax_comparison_reports_first_autoregressive_divergence():
    reference = {"prefill_argmax": [[1, 2, 3]], "generated": [4, 5, 6]}
    candidate = {"prefill_argmax": [[1, 2, 3]], "generated": [4, 9, 6]}

    result = MODULE._comparison(reference, candidate)

    assert result["prefill_argmax_matches"] == 3
    assert result["generated_token_matches"] == 2
    assert result["first_generated_mismatch"] == 1
    assert not result["exact"]


def test_link_test_candidate_requires_exactness_and_stable_candidate_and_baseline():
    args = {
        "mode": "f16",
        "exact": True,
        "device_type": "cuda",
        "repeats": 3,
        "prefill_tokens": 128,
        "decode_steps": 32,
        "candidate_cv": 0.04,
        "baseline_cv": 0.05,
        "max_runtime_cv": 0.1,
    }

    assert MODULE._candidate_for_cross_device_link_test(**args)
    assert not MODULE._candidate_for_cross_device_link_test(**{**args, "baseline_cv": 0.2})
    assert not MODULE._candidate_for_cross_device_link_test(**{**args, "mode": "none"})
    assert not MODULE._candidate_for_cross_device_link_test(**{**args, "exact": False})
