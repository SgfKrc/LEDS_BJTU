from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

pytest.importorskip("transformers.models.gemma4_unified")
transformers = pytest.importorskip("transformers")


def test_official_gemma4_transformers_segmented_prefill_decode():
    if str(getattr(transformers, "__version__", "")) != "5.10.1":
        pytest.skip("official Gemma 4 smoke requires Transformers 5.10.1 sidecar")
    from scripts.model_tools.gemma4_pipeline_official_smoke import run_smoke

    report = run_smoke()
    assert report["status"] == "passed"
    assert report["prefill_max_abs_error"] <= 1e-6
    assert report["decode_max_abs_error"] <= 1e-6
    assert report["sequence_length"] == 5
    assert report["full_model_materialized"] is False
    assert report["multimodal_materialized"] is False
