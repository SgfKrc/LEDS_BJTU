import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_gate_sd15_ip_adapter import (  # noqa: E402
    DEFAULT_SCALES,
    _apply_manual_reviews,
    _automatic_gate,
    _full_matrix_reasons,
    _image_metrics,
    _memory_gate,
)


def _memory(allocated, reserved):
    return {
        "available": True,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
    }


def test_reference_memory_gate_accepts_stable_run_and_unload():
    images = [
        {"cuda_memory": _memory(100, 200)},
        {"cuda_memory": _memory(120, 220)},
    ]

    result = _memory_gate(images, _memory(0, 0), _memory(10, 20))

    assert result["passed"] is True
    assert result["allocated_span_bytes"] == 20
    assert result["after_unload_growth_bytes"] == 10


def test_reference_memory_gate_rejects_missing_cuda_snapshot():
    result = _memory_gate([], {"available": False}, {"available": False})

    assert result["passed"] is False
    assert result["available"] is False


def test_reference_full_matrix_requires_pinned_source_and_preset_parameters():
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
        scales=DEFAULT_SCALES,
        seeds=(1, 2),
        expected_seeds=(1, 2),
        preset=preset,
        generation=generation,
        expected_prompt="pinned prompt",
        expected_negative_prompt="pinned negative",
        expected_source_sha256="a" * 64,
        actual_source_sha256="a" * 64,
    )

    assert reasons == []

    reasons = _full_matrix_reasons(
        scales=(0.6,),
        seeds=(1,),
        expected_seeds=(1, 2),
        preset=preset,
        generation=SimpleNamespace(
            steps=4,
            prompt="custom prompt",
            negative_prompt="custom negative",
        ),
        expected_prompt="pinned prompt",
        expected_negative_prompt="pinned negative",
        expected_source_sha256="",
        actual_source_sha256="b" * 64,
    )
    assert reasons == [
        "scale matrix is not the pinned low/medium/high set",
        "seed matrix is incomplete",
        "steps differ from the pinned preset",
        "prompt differs from the pinned preset",
        "negative prompt differs from the pinned preset",
        "source SHA-256 is not pinned",
    ]


def test_reference_image_metrics_rejects_safety_flagged_output(tmp_path):
    from PIL import Image

    image = Image.new("RGB", (32, 32), (80, 100, 120))
    metrics = _image_metrics(
        image,
        tmp_path / "flagged.png",
        safety_flagged=True,
    )

    assert metrics["safety_flagged"] is True
    assert metrics["automatic_pass"] is False


def test_reference_automatic_gate_uses_safe_replacement_outputs():
    images = [
        {"scale": 0.6, "sha256": "a", "automatic_pass": True, "safety_flagged": False},
        {"scale": 0.6, "sha256": "b", "automatic_pass": True, "safety_flagged": False},
        {"scale": 0.6, "sha256": "black", "automatic_pass": False, "safety_flagged": True},
    ]

    result = _automatic_gate(
        images,
        scales=(0.6,),
        generated_seeds=(1, 2, 3),
        required_valid_per_scale=2,
        memory_passed=True,
    )

    assert result["passed"] is True
    assert result["valid_outputs_per_scale"] == {"0.6": 2}
    assert result["safety_flagged_outputs"] == 1


def test_reference_manual_reviews_can_finish_automatic_report():
    report = {
        "mode": "reference",
        "status": "pending_manual_review",
        "full_matrix": True,
        "automatic_gate": {"passed": True},
        "manual_gate": {"reviews": [{"name": "Alice", "decision": "pass"}]},
    }

    status = _apply_manual_reviews(report, [{"name": "Bob", "decision": "pass"}])

    assert status == "passed"
    assert report["manual_gate"]["passed"] is True


def test_reference_manual_reviews_cannot_promote_a_partial_report():
    report = {
        "mode": "reference",
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
    assert report["manual_gate"]["passed"] is True
