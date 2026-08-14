import pytest

pytestmark = pytest.mark.quality_gate

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_gate_sd15_img2img import (  # noqa: E402
    DEFAULT_STRENGTHS,
    _apply_manual_reviews,
    _full_matrix_reasons,
    _memory_gate,
)


def _memory(allocated, reserved):
    return {
        "available": True,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
    }


def test_memory_gate_accepts_stable_run_and_unload():
    images = [
        {"cuda_memory": _memory(100, 200)},
        {"cuda_memory": _memory(120, 220)},
    ]

    result = _memory_gate(images, _memory(0, 0), _memory(10, 20))

    assert result["passed"] is True
    assert result["allocated_span_bytes"] == 20
    assert result["after_unload_growth_bytes"] == 10


def test_memory_gate_rejects_missing_cuda_snapshot():
    result = _memory_gate([], {"available": False}, {"available": False})

    assert result == {
        "passed": False,
        "available": False,
        "reason": "CUDA memory snapshots are incomplete",
    }


def test_full_matrix_requires_pinned_source_and_preset_parameters():
    preset = SimpleNamespace(
        seeds=(1, 2),
        steps=28,
        prompt="pinned prompt",
        negative_prompt="pinned negative",
    )
    generation = SimpleNamespace(
        steps=28,
        prompt="pinned prompt",
        negative_prompt="pinned negative",
    )

    reasons = _full_matrix_reasons(
        strengths=DEFAULT_STRENGTHS,
        seeds=(1, 2),
        preset=preset,
        generation=generation,
        expected_source_sha256="a" * 64,
        actual_source_sha256="a" * 64,
    )

    assert reasons == []

    generation.steps = 4
    reasons = _full_matrix_reasons(
        strengths=DEFAULT_STRENGTHS,
        seeds=(1,),
        preset=preset,
        generation=generation,
        expected_source_sha256="",
        actual_source_sha256="b" * 64,
    )
    assert reasons == [
        "seed matrix is incomplete",
        "steps differ from the pinned preset",
        "source SHA-256 is not pinned",
    ]


def test_manual_reviews_can_finish_existing_automatic_report():
    report = {
        "mode": "img2img",
        "status": "pending_manual_review",
        "full_matrix": True,
        "automatic_gate": {"passed": True},
        "manual_gate": {"reviews": [{"name": "Alice", "decision": "pass"}]},
    }

    status = _apply_manual_reviews(
        report,
        [{"name": "Bob", "decision": "pass"}],
    )

    assert status == "passed"
    assert report["manual_gate"]["passed"] is True
    assert {item["name"] for item in report["manual_gate"]["reviews"]} == {"Alice", "Bob"}


def test_manual_reviews_cannot_promote_a_partial_img2img_report():
    report = {
        "mode": "img2img",
        "status": "partial_pass",
        "full_matrix": False,
        "automatic_gate": {"passed": True},
        "manual_gate": {"reviews": []},
    }

    status = _apply_manual_reviews(
        report,
        [
            {"name": "Alice", "decision": "pass"},
            {"name": "Bob", "decision": "pass"},
        ],
    )

    assert status == "partial_pass"
