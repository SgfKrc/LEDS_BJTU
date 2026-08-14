import pytest

pytestmark = pytest.mark.quality_gate

import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_gate_sd15_inpaint import (  # noqa: E402
    DEFAULT_SEEDS,
    MASK_SEQUENCE,
    _make_mask,
    _measure_mask_semantics,
    _semantic_pass,
)


def test_full_matrix_has_ten_cases_and_all_three_mask_semantics():
    assert len(DEFAULT_SEEDS) == 10
    assert len(MASK_SEQUENCE) == 10
    assert set(MASK_SEQUENCE) == {"black", "local", "white"}


def test_masks_use_white_for_redraw_and_black_for_preserve():
    black = _make_mask("black", (32, 32))
    local = _make_mask("local", (32, 32))
    white = _make_mask("white", (32, 32))

    assert black.getextrema() == (0, 0)
    assert local.getextrema() == (0, 255)
    assert white.getextrema() == (255, 255)


def test_semantic_metrics_separate_inside_and_outside_changes():
    source = Image.new("RGB", (32, 32), 0)
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((8, 8, 23, 23), fill=255)
    result = source.copy()
    ImageDraw.Draw(result).rectangle((8, 8, 23, 23), fill=(80, 80, 80))

    metrics = _measure_mask_semantics(source, result, mask)

    assert metrics["inside_mae"] == 80
    assert metrics["outside_mae"] == 0
    assert metrics["inside_margin"] == 80
    assert _semantic_pass("local", metrics) is True


def test_semantic_gate_rejects_black_mask_drift_and_unchanged_white_mask():
    assert _semantic_pass(
        "black",
        {"overall_mae": 11, "inside_mae": 0, "outside_mae": 11, "inside_margin": -11},
    ) is False
    assert _semantic_pass(
        "white",
        {"overall_mae": 0, "inside_mae": 0, "outside_mae": 0, "inside_margin": 0},
    ) is False
