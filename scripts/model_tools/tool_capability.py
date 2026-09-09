"""Offline model/tool capability registry and contract probe.

This module deliberately does not load model weights or contact a model hub.  It
inspects local metadata, then optionally validates a bounded transcript fixture.
The fixture proves that the gateway contract is understood; it does not claim
that a model can generate the transcript until a later runtime smoke gate does so.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .gguf import GGUFError, inspect_gguf
from .llm_smoke_matrix import discover_units

SCHEMA_VERSION = 1
TOOL = "tool_capability_probe"
MAX_MODELS = 32
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_FIXTURE_BYTES = 256 * 1024
ALLOWED_TOOLS = {"web_search", "web_fetch"}
_TOOL_SIGNAL = re.compile(r"\b(tools?|tool_calls?|function_call|functions?|web_search|web_fetch)\b", re.I)
_JSON_SIGNAL = re.compile(r"\b(tojson|json|arguments|parameters)\b", re.I)


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    return _digest_bytes(str(path.expanduser().absolute().resolve(strict=False)).encode("utf-8"))


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, "metadata_file_too_large"
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing_file"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in sorted(value.items()))
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return "" if value is None else str(value)


def _capability(status: str, evidence: list[str]) -> dict[str, Any]:
    return {"status": status, "evidence": sorted(set(evidence))}


def _capabilities(template: str, *, declared_tool_schema: bool = False) -> tuple[dict[str, Any], list[str]]:
    text = template or ""
    tool_signal = bool(_TOOL_SIGNAL.search(text)) or declared_tool_schema
    json_signal = bool(_JSON_SIGNAL.search(text))
    result_signal = bool(re.search(r"role[^\n]{0,80}tool|tool[^\n]{0,80}result", text, re.I))
    evidence: list[str] = []
    if text:
        evidence.append("chat_template_present")
    if tool_signal:
        evidence.append("tool_schema_tokens_declared")
    if json_signal:
        evidence.append("json_serialization_tokens_declared")
    if result_signal:
        evidence.append("tool_result_role_declared")
    return {
        "json_output": _capability("declared" if json_signal else "unknown", ["json_serialization_tokens_declared"] if json_signal else ["static_execution_not_run"]),
        "tool_call_generation": _capability("declared" if tool_signal else "unknown", ["tool_schema_tokens_declared"] if tool_signal else ["static_execution_not_run"]),
        "tool_result_reinjection": _capability("declared" if result_signal else ("declared" if tool_signal else "unknown"), ["tool_result_role_declared"] if result_signal else (["tool_schema_tokens_declared"] if tool_signal else ["static_execution_not_run"])),
    }, evidence


def _canonical_fixture(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None, "tool_arguments_not_json"
    if not isinstance(value, dict):
        return None, "tool_arguments_not_object"
    return value, None


def _validate_tool_call(call: Any) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    if not isinstance(call, dict):
        return None, None, ["tool_call_not_object"]
    name = call.get("name")
    if not isinstance(name, str) or name not in ALLOWED_TOOLS:
        return None, None, ["unsupported_tool_name"]
    arguments, error = _parse_arguments(call.get("arguments", call.get("parameters")))
    if error or arguments is None:
        return name, None, [error or "tool_arguments_invalid"]
    if name == "web_search":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            return name, None, ["web_search_query_invalid"]
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            return name, None, ["web_search_top_k_invalid"]
        if set(arguments) - {"query", "top_k"}:
            return name, None, ["web_search_arguments_extra"]
    else:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.startswith("https://") or len(url) > 4096:
            return name, None, ["web_fetch_url_invalid"]
        if set(arguments) - {"url", "max_chars"}:
            return name, None, ["web_fetch_arguments_extra"]
        if "max_chars" in arguments and (not isinstance(arguments["max_chars"], int) or not 1 <= arguments["max_chars"] <= 32768):
            return name, None, ["web_fetch_max_chars_invalid"]
    return name, arguments, []


def validate_tool_fixture(fixture: Any) -> dict[str, Any]:
    """Validate a bounded assistant -> tool -> assistant contract transcript."""
    errors: list[str] = []
    if not isinstance(fixture, dict):
        errors.append("fixture_not_object")
        return {"valid": False, "errors": errors, "capabilities": {key: "rejected" for key in ("json_output", "tool_call_generation", "tool_result_reinjection")}}
    assistant = fixture.get("assistant", fixture)
    calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
    if calls is None and isinstance(assistant, dict) and assistant.get("tool_call") is not None:
        calls = [assistant["tool_call"]]
    if not isinstance(calls, list) or len(calls) != 1:
        errors.append("exactly_one_tool_call_required")
        call = None
    else:
        call = calls[0]
    name, arguments, call_errors = _validate_tool_call(call)
    errors.extend(call_errors)
    result = fixture.get("tool_result")
    result_ok = isinstance(result, dict) and result.get("role") == "tool" and result.get("name") == name
    if not result_ok:
        errors.append("tool_result_message_invalid")
    elif not isinstance(result.get("content"), (dict, list, str)):
        errors.append("tool_result_content_invalid")
    final_response = fixture.get("final_response")
    if not isinstance(final_response, str) or not final_response.strip() or len(final_response) > 32768:
        errors.append("final_response_invalid")
    valid = not errors
    return {
        "valid": valid,
        "errors": sorted(set(errors)),
        "capabilities": {
            "json_output": "verified" if valid else "rejected",
            "tool_call_generation": "verified" if valid else "rejected",
            "tool_result_reinjection": "verified" if valid else "rejected",
        },
        "tool_name": name,
        "argument_keys": sorted(arguments) if arguments else [],
    }


def _fixture_from_path(path: Path) -> tuple[Any | None, str | None, str | None]:
    try:
        if path.stat().st_size > MAX_FIXTURE_BYTES:
            return None, None, "fixture_too_large"
        data = path.read_bytes()
        if len(data) > MAX_FIXTURE_BYTES:
            return None, None, "fixture_too_large"
        fixture = json.loads(data.decode("utf-8"))
        return fixture, _digest_bytes(data), None
    except FileNotFoundError:
        return None, None, "fixture_missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, None, "fixture_invalid_json"


def _base_report(path: Path, *, model_id: str | None, model_format: str, engine: str, sidecar_version: str, runtime: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "probe",
        "valid": True,
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "execution_mode": "static_contract",
        "model": {
            "model_id": model_id or path.name,
            "format": model_format,
            "engine": engine,
            "path_digest": _path_digest(path),
        },
        "runtime": {"sidecar_version": sidecar_version or "unknown", "runtime": runtime or "unknown"},
        "tokenizer": {"files": [], "digest": None, "chat_template_present": False, "chat_template_digest": None},
        "capabilities": {
            "json_output": _capability("unknown", ["static_execution_not_run"]),
            "tool_call_generation": _capability("unknown", ["static_execution_not_run"]),
            "tool_result_reinjection": _capability("unknown", ["static_execution_not_run"]),
        },
        "evidence": ["weights_not_loaded", "network_disabled"],
        "admission": {"status": "unknown", "production_eligible": False, "reason_codes": ["runtime_verification_pending"]},
        "errors": [],
    }


def _apply_fixture(report: dict[str, Any], fixture: Any, fixture_digest: str) -> None:
    result = validate_tool_fixture(fixture)
    report["fixture"] = {"digest": fixture_digest, "valid": result["valid"], "tool_name": result.get("tool_name"), "argument_keys": result.get("argument_keys", [])}
    if result["valid"]:
        for name in report["capabilities"]:
            report["capabilities"][name] = {"status": "verified", "evidence": ["offline_contract_fixture"]}
        report["evidence"].append("offline_contract_fixture_verified")
    else:
        report["errors"].extend({"code": code, "message": "offline tool contract fixture rejected"} for code in result["errors"])
        for name in report["capabilities"]:
            report["capabilities"][name] = {"status": "rejected", "evidence": ["offline_contract_fixture"]}


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    report["evidence"] = sorted(set(report["evidence"]))
    report["errors"] = sorted(report["errors"], key=lambda item: (item.get("code", ""), item.get("message", "")))
    statuses = {item["status"] for item in report["capabilities"].values()}
    if report["errors"] or "rejected" in statuses:
        report["admission"] = {"status": "rejected", "production_eligible": False, "reason_codes": ["probe_errors"]}
    elif statuses == {"verified"}:
        report["admission"] = {"status": "candidate", "production_eligible": False, "reason_codes": ["runtime_generation_gate_pending"]}
    else:
        report["admission"] = {"status": "unknown", "production_eligible": False, "reason_codes": ["runtime_verification_pending"]}
    report["valid"] = not bool(report["errors"])
    return report


def _probe_safetensors(path: Path, report: dict[str, Any]) -> None:
    if not path.is_dir():
        report["valid"] = False
        report["errors"].append({"code": "asset_missing", "message": "Safetensors model directory is missing"})
        return
    json_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
    loaded: dict[str, Any] = {}
    file_digests: list[dict[str, Any]] = []
    for name in json_files + ["chat_template.jinja", "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "merges.txt"]:
        file = path / name
        if not file.is_file():
            continue
        try:
            size = file.stat().st_size
            if size > MAX_METADATA_BYTES:
                report["errors"].append({"code": "metadata_file_too_large", "message": f"{name} exceeds metadata limit"})
                continue
            digest = _digest_file(file)
            file_digests.append({"name": name, "size_bytes": size, "sha256": digest})
            if name.endswith(".json"):
                data, error = _read_json(file)
                if error:
                    report["errors"].append({"code": error, "message": f"cannot parse {name}"})
                else:
                    loaded[name] = data
            elif name == "chat_template.jinja":
                loaded[name] = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            report["errors"].append({"code": "metadata_read_failed", "message": f"cannot read {name}"})
    config = loaded.get("config.json") if isinstance(loaded.get("config.json"), dict) else {}
    tokenizer = loaded.get("tokenizer_config.json") if isinstance(loaded.get("tokenizer_config.json"), dict) else {}
    template = tokenizer.get("chat_template", "")
    if not template:
        template = loaded.get("chat_template.jinja", "")
    template_text = _flatten_text(template)
    report["model"]["architecture"] = (config.get("architectures") or [config.get("model_type") or "unknown"])[0]
    report["model"]["model_type"] = config.get("model_type", "unknown")
    report["tokenizer"]["files"] = [item["name"] for item in file_digests if item["name"] not in json_files]
    report["tokenizer"]["digest"] = _digest_bytes(_canonical_fixture(file_digests)) if file_digests else None
    report["tokenizer"]["chat_template_present"] = bool(template_text)
    report["tokenizer"]["chat_template_digest"] = _digest_bytes(template_text.encode("utf-8")) if template_text else None
    capabilities, evidence = _capabilities(template_text, declared_tool_schema=bool(tokenizer.get("tool_use") or tokenizer.get("tools")))
    report["capabilities"] = capabilities
    report["evidence"].extend(evidence)


def _probe_gguf(path: Path, report: dict[str, Any]) -> None:
    if not path.is_file():
        report["valid"] = False
        report["errors"].append({"code": "asset_missing", "message": "GGUF model file is missing"})
        return
    try:
        inspected = inspect_gguf(path)
    except (OSError, GGUFError):
        report["valid"] = False
        report["errors"].append({"code": "gguf_header_invalid", "message": "GGUF header could not be inspected"})
        return
    metadata = inspected.get("metadata", {})
    template_values = [value for key, value in metadata.items() if "chat_template" in key or key.startswith("tokenizer.") and "template" in key]
    template_text = _flatten_text(template_values)
    report["model"]["architecture"] = inspected.get("derived", {}).get("architecture", "unknown")
    report["tokenizer"]["files"] = [key for key in sorted(metadata) if key.startswith("tokenizer.")]
    report["tokenizer"]["digest"] = _digest_bytes(_canonical_fixture({key: metadata[key] for key in sorted(metadata) if key.startswith("tokenizer.")}))
    report["tokenizer"]["chat_template_present"] = bool(template_text)
    report["tokenizer"]["chat_template_digest"] = _digest_bytes(template_text.encode("utf-8")) if template_text else None
    capabilities, evidence = _capabilities(template_text)
    report["capabilities"] = capabilities
    report["evidence"].extend(evidence)


def probe_model_asset(
    path: str | Path,
    *,
    model_id: str | None = None,
    model_format: str | None = None,
    sidecar_version: str = "",
    runtime: str = "",
    fixture: Any | None = None,
    fixture_digest: str = "",
) -> dict[str, Any]:
    target = Path(path).expanduser()
    inferred_format = model_format or ("gguf" if target.suffix.lower() == ".gguf" else "safetensors")
    if inferred_format not in {"gguf", "safetensors"}:
        raise ValueError("model_format must be gguf or safetensors")
    report = _base_report(target, model_id=model_id, model_format=inferred_format, engine="llama_cpp" if inferred_format == "gguf" else "pytorch", sidecar_version=sidecar_version, runtime=runtime)
    if inferred_format == "gguf":
        _probe_gguf(target, report)
    else:
        _probe_safetensors(target, report)
    if fixture is not None:
        _apply_fixture(report, fixture, fixture_digest or _digest_bytes(_canonical_fixture(fixture)))
    return _finalize(report)


def run_capability_matrix(
    *,
    model_ids: list[str] | None = None,
    formats: list[str] | None = None,
    max_models: int = MAX_MODELS,
    sidecar_version: str = "",
    runtime: str = "",
    fixture: Any | None = None,
    fixture_digest: str = "",
) -> dict[str, Any]:
    if max_models < 1 or max_models > MAX_MODELS:
        raise ValueError(f"max_models must be between 1 and {MAX_MODELS}")
    units = discover_units(model_ids, formats)
    discovered_count = len(units)
    units = units[:max_models]
    results: list[dict[str, Any]] = []
    for unit in units:
        if not unit["available"]:
            results.append({"model_id": unit["model_id"], "format": unit["format"], "engine": unit["engine"], "status": "skipped", "reason": "asset_missing"})
            continue
        result = probe_model_asset(unit["path"], model_id=unit["model_id"], model_format=unit["format"], sidecar_version=sidecar_version, runtime=runtime, fixture=fixture, fixture_digest=fixture_digest)
        result["status"] = "passed" if result["valid"] else "failed"
        results.append(result)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "operation": "matrix",
        "valid": True,
        "read_only": True,
        "network_used": False,
        "weights_loaded": False,
        "models": results,
        "summary": {
            "units_discovered": discovered_count,
            "units_total": len(results),
            "selection_truncated": discovered_count > len(results),
            "units_probed": sum(item.get("status") in {"passed", "failed"} for item in results),
            "units_skipped": sum(item.get("status") == "skipped" for item in results),
            "unknown_capability_units": sum(item.get("admission", {}).get("status") == "unknown" for item in results),
            "candidate_units": sum(item.get("admission", {}).get("status") == "candidate" for item in results),
            "production_eligible": False,
            "gate_passed": all(item.get("status") != "failed" for item in results),
        },
        "errors": [],
    }


__all__ = ["probe_model_asset", "run_capability_matrix", "validate_tool_fixture"]
