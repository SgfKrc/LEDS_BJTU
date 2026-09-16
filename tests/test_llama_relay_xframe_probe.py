"""Evidence tests for the fail-closed cross-framework relay verifier."""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from scripts.llama_relay_xframe_probe import build_evidence_report, compare_logits


def _write_logits(path: Path, rows: list[list[float]]) -> Path:
    values = array("f", (value for row in rows for value in row))
    with path.open("wb") as handle:
        values.tofile(handle)
    return path


def test_matching_logits_are_accepted_by_argmax_even_when_not_bitwise_equal(tmp_path: Path):
    baseline = _write_logits(tmp_path / "baseline.f32", [[0.1, 0.8, -1.0], [2.0, 0.2, 0.1]])
    relay = _write_logits(tmp_path / "relay.f32", [[0.1, 0.7, -1.1], [1.9, 0.2, 0.1]])

    report = compare_logits(baseline, relay, vocab_size=3, top_k=2)

    assert report["status"] == "evidence_accepted"
    assert report["verdict"]["criterion"] == "per_token_argmax"
    assert report["verdict"]["matched_steps"] == 2
    assert report["argmax_match_count"] == 2
    assert report["diagnostics"]["bitwise_equal"] is False
    assert report["mismatch_positions"] == []


def test_diverging_position_is_rejected_and_preserves_policy_closure(tmp_path: Path):
    baseline = _write_logits(tmp_path / "baseline.f32", [[0.1, 0.8, -1.0], [2.0, 0.2, 0.1]])
    relay = _write_logits(tmp_path / "relay.f32", [[0.1, 0.7, -1.1], [0.1, 0.2, 2.0]])

    report = build_evidence_report(baseline, relay, vocab_size=3, top_k=2)

    assert report["status"] == "evidence_rejected"
    assert report["verdict"]["reason"] == "token_mismatch"
    assert report["verdict"]["matched_steps"] == 1
    assert report["argmax_match_count"] == 1
    assert report["mismatch_positions"] == [1]
    assert report["production_admission"]["admitted"] is False
    assert report["production_admission"]["reason"] == "cross_engine_not_admitted"


def test_invalid_matrix_shape_is_rejected_explicitly(tmp_path: Path):
    baseline = _write_logits(tmp_path / "baseline.f32", [[0.1, 0.8, -1.0]])
    relay = tmp_path / "relay.f32"
    relay.write_bytes(b"bad")

    with pytest.raises(ValueError, match="invalid_logits_shape"):
        compare_logits(baseline, relay, vocab_size=3)
