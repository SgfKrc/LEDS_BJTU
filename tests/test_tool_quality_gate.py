"""Offline WEB-TOOL-G6 quality and admission tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.tool_quality_gate import (
    QualityThresholds,
    assess_candidate,
    build_admission_matrix,
    evaluate_quality_cases,
    offline_quality_cases,
)


def test_offline_fixture_passes_quality_gate_without_network():
    report = evaluate_quality_cases(offline_quality_cases())
    assert report["schema"] == "qlh.tool_quality.v1"
    assert report["network_used"] is False
    assert report["weights_loaded"] is False
    assert report["admission"]["status"] == "candidate"
    assert report["admission"]["quality_gate_passed"] is True
    assert report["metrics"]["unsafe_request_block_rate"] == 1.0
    assert report["metrics"]["citation_rate"] == 1.0


def test_quality_gate_rejects_missing_citation_and_bad_selection():
    cases = list(offline_quality_cases())
    cases[0] = {**cases[0], "observed_action": "none", "citation_complete": False}
    report = evaluate_quality_cases(cases)
    assert report["admission"]["status"] == "rejected"
    assert "tool_selection_accuracy_below_threshold" in report["admission"]["reason_codes"]
    assert "citation_rate_below_threshold" in report["admission"]["reason_codes"]


def test_quality_gate_rejects_unsafe_case_not_blocked():
    cases = list(offline_quality_cases())
    cases[3] = {**cases[3], "unsafe_blocked": False}
    report = evaluate_quality_cases(cases)
    assert "unsafe_request_block_rate_below_threshold" in report["admission"]["reason_codes"]


def test_quality_gate_has_no_raw_case_fields():
    report = evaluate_quality_cases(offline_quality_cases())
    encoded = repr(report)
    assert "private_url" not in encoded
    assert "https://" not in encoded
    assert "prompt" not in encoded


def test_model_without_verified_tool_calling_uses_host_router():
    quality = evaluate_quality_cases(offline_quality_cases())
    result = assess_candidate({"candidate_id": "qw1_8b", "kind": "model", "capability_report": {"capabilities": {"tool_call_generation": {"status": "unknown"}}}}, quality)
    assert result["route_mode"] == "host_router"
    assert result["model_autonomous_eligible"] is False
    assert "tool_call_generation_not_verified" in result["reason_codes"]


def test_verified_model_still_waits_for_real_network_acceptance():
    quality = evaluate_quality_cases(offline_quality_cases())
    result = assess_candidate({"candidate_id": "littlelamb", "kind": "model", "capability_report": {"capabilities": {"tool_call_generation": {"status": "verified"}}}}, quality)
    assert result["route_mode"] == "model_autonomous"
    assert result["production_eligible"] is False
    assert "real_provider_acceptance_pending" in result["reason_codes"]


def test_quality_failure_rejects_route_even_when_model_capability_is_verified():
    cases = list(offline_quality_cases())
    cases[0] = {**cases[0], "schema_valid": False}
    quality = evaluate_quality_cases(cases)
    result = assess_candidate({"candidate_id": "littlelamb", "kind": "model", "capability_report": {"capabilities": {"tool_call_generation": {"status": "verified"}}}}, quality)
    assert result["route_mode"] == "rejected"
    assert result["production_eligible"] is False
    assert "quality_gate_rejected" in result["reason_codes"]


def test_provider_requires_contract_and_matrix_is_bounded():
    quality = evaluate_quality_cases(offline_quality_cases())
    matrix = build_admission_matrix([
        {"candidate_id": "searx_fake", "kind": "provider", "contract_verified": True},
        {"candidate_id": "unknown_provider", "kind": "provider", "contract_verified": False},
    ], quality)
    assert matrix["network_used"] is False
    assert matrix["summary"]["production_eligible"] == 0
    assert "provider_contract_not_verified" in matrix["candidates"][1]["reason_codes"]


def test_quality_threshold_validation_and_duplicate_ids():
    with pytest.raises(ValueError):
        QualityThresholds(citation_rate=1.1)
    cases = list(offline_quality_cases())
    cases.append(cases[0])
    with pytest.raises(ValueError, match="unique"):
        evaluate_quality_cases(cases)
