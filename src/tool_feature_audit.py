"""Read-only contract audit for WEB-TOOL-G2..G6.

This is intentionally small and deterministic.  It checks the high-risk
boundaries that unit tests cover together, without opening a network socket or
reading model weights.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping

try:
    from .tool_gateway import ToolGatewayError, ToolGatewayPolicy, validate_url
    from .tool_quality_gate import evaluate_quality_cases, offline_quality_cases
    from .tool_rag_cache import ToolRagCache
    from .tool_task_graph import ToolRoutePolicy
except ImportError:  # pragma: no cover
    from tool_gateway import ToolGatewayError, ToolGatewayPolicy, validate_url
    from tool_quality_gate import evaluate_quality_cases, offline_quality_cases
    from tool_rag_cache import ToolRagCache
    from tool_task_graph import ToolRoutePolicy

AUDIT_SCHEMA = "qlh.tool_feature_audit.v1"


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "status": "pass" if passed else "fail", "detail_code": detail}


def audit_tool_feature_contracts(*, test_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        validate_url("https://127.0.0.1/admin", policy=ToolGatewayPolicy())
    except ToolGatewayError as exc:
        checks.append(_check("gateway_ssrf_private_address", exc.code == "unsafe_url_host", exc.code))
    else:
        checks.append(_check("gateway_ssrf_private_address", False, "private_url_accepted"))
    try:
        policy = ToolRoutePolicy(sidecar_capability="unknown")
        checks.append(_check("task_graph_sidecar_fail_closed", policy.sidecar_capability == "unknown", "unknown_routes_to_host"))
    except Exception:
        checks.append(_check("task_graph_sidecar_fail_closed", False, "route_policy_invalid"))
    signature = inspect.signature(ToolRagCache.save_tool_result)
    persist_default = signature.parameters["persist"].default
    checks.append(_check("rag_cache_explicit_persist", persist_default is False, "persist_default_false"))
    quality = evaluate_quality_cases(offline_quality_cases())
    checks.append(_check("offline_quality_fixture", bool(quality["admission"]["quality_gate_passed"]), "quality_gate_passed" if quality["admission"]["quality_gate_passed"] else "quality_gate_rejected"))
    required_routes = {
        "/api/tool-cache/health", "/api/tool-cache", "/api/tool-cache/search",
        "/api/tool-cache/rebuild", "/api/tool-cache/purge",
    }
    try:
        try:
            from .api_server import app
        except ImportError:  # pragma: no cover
            from api_server import app
        routes = {getattr(route, "path", "") for route in app.routes}
        checks.append(_check("tool_cache_api_surface", required_routes <= routes, "api_routes_present" if required_routes <= routes else "api_route_missing"))
    except Exception:
        checks.append(_check("tool_cache_api_surface", False, "api_import_failed"))
    if test_summary is not None:
        failed = int(test_summary.get("failed", 0) or 0)
        checks.append(_check("combined_test_summary", failed == 0, "tests_clean" if failed == 0 else "tests_failed"))
    failures = [item["check"] for item in checks if item["status"] != "pass"]
    return {
        "schema": AUDIT_SCHEMA,
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "production_network_enabled": False,
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failure_checks": failures,
        "residual_risk_codes": ["real_network_provider_pending", "real_sidecar_model_pending"],
    }


__all__ = ["AUDIT_SCHEMA", "audit_tool_feature_contracts"]
