"""Combined read-only audit for the WEB-TOOL feature stages."""

from __future__ import annotations

from src.tool_feature_audit import audit_tool_feature_contracts


def test_feature_contract_audit_passes_without_network_or_weights():
    report = audit_tool_feature_contracts(test_summary={"failed": 0})
    assert report["status"] == "pass"
    assert report["network_used"] is False
    assert report["weights_loaded"] is False
    assert report["production_network_enabled"] is False
    assert report["failure_checks"] == []
    assert "real_network_provider_pending" in report["residual_risk_codes"]
