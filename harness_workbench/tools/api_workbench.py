"""Offline contract workbench for the harness ``/v1`` and project ``/api``.

The default runner uses deterministic fixture responses. A transport can be
injected for a loopback or explicitly authorized environment, but reports only
request/response digests, schemas, status codes and bounded latency samples;
prompt text, credentials and response bodies never enter the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


API_WORKBENCH_SCHEMA = "qlh.harness.api_workbench.v1"
API_WORKBENCH_INPUT_SCHEMA = "qlh.api_workbench.v1"
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SECRET_KEY = re.compile(r"(?:authorization|api[-_]?key|token|secret|password|cookie|prompt|message|content)", re.IGNORECASE)
_FORBIDDEN = re.compile(r"(?:sk-[A-Za-z0-9]|bearer\s+\S+|-----BEGIN|(?:https?|wss?)://)", re.IGNORECASE)


class APIWorkbenchError(ValueError):
    """Stable input/transport error with no provider detail leakage."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.retryable = bool(retryable)
        super().__init__(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("/") or ".." in value.replace("\\", "/").split("/"):
        raise APIWorkbenchError("unsafe_path", "endpoint paths must be absolute relative API paths")
    if _URL.search(value) or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise APIWorkbenchError("unsafe_path", "endpoint paths cannot contain a host or filesystem path")
    return value


def _safe_headers(value: Mapping[str, Any] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise APIWorkbenchError("invalid_headers", "headers must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        name = str(key).strip().lower()
        if not name or _SECRET_KEY.search(name):
            raise APIWorkbenchError("unsafe_headers", "credential-bearing headers are not accepted")
        text = str(raw)
        if _FORBIDDEN.search(text):
            raise APIWorkbenchError("unsafe_headers", "header values contain a forbidden credential or URL")
        result[name] = text
    return result


def _redacted_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "<redacted>" if _SECRET_KEY.search(str(key)) else _redacted_shape(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return ["<item>" for _ in value]
    if isinstance(value, tuple):
        return ["<item>" for _ in value]
    if isinstance(value, str):
        return {"type": "string", "chars": len(value)}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return type(value).__name__


def _safe_json(value: Any) -> None:
    if isinstance(value, str):
        if _FORBIDDEN.search(value):
            raise APIWorkbenchError("unsafe_payload", "payload contains a URL, credential or secret")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SECRET_KEY.search(str(key)) and child not in (None, "", [], {}):
                # Prompt/message fields are allowed in input fixtures but are never
                # copied into the report; credentials remain forbidden everywhere.
                if re.search(r"(?:authorization|api[-_]?key|access[_-]?token|refresh[_-]?token|secret|password|cookie)", str(key), re.IGNORECASE):
                    raise APIWorkbenchError("unsafe_payload", "payload contains a credential field")
            _safe_json(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _safe_json(child)


@dataclass(frozen=True, slots=True)
class APIResponse:
    status_code: int
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    content_type: str = "application/json"

    def __post_init__(self) -> None:
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int) or not 100 <= self.status_code <= 599:
            raise APIWorkbenchError("invalid_response", "response status code is invalid")
        if self.latency_ms < 0 or self.latency_ms != self.latency_ms:
            raise APIWorkbenchError("invalid_response", "response latency is invalid")


class APITransport(Protocol):
    network_used: bool

    def request(self, method: str, path: str, *, payload: Mapping[str, Any] | None, headers: Mapping[str, str]) -> APIResponse:
        ...


class MemoryAPITransport:
    """Deterministic transport keyed by ``METHOD path`` for offline reports."""

    network_used = False

    def __init__(self, responses: Mapping[str, APIResponse | Callable[[str, str, Mapping[str, Any] | None], APIResponse]]) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, path: str, *, payload: Mapping[str, Any] | None, headers: Mapping[str, str]) -> APIResponse:
        key = f"{method.upper()} {path}"
        self.calls.append({"method": method.upper(), "path": path, "payload_shape": _redacted_shape(payload or {}), "headers": dict(headers)})
        response = self.responses.get(key)
        if response is None:
            return APIResponse(404, {"error": {"code": "fixture_not_found", "type": "not_found"}}, latency_ms=0.2)
        if callable(response):
            response = response(method.upper(), path, payload)
        return response


class UrllibAPITransport:
    """Explicit HTTP transport; disabled unless the caller opts in."""

    def __init__(self, base_url: str, *, enabled: bool = False, timeout_seconds: float = 5.0) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise APIWorkbenchError("invalid_base_url", "base_url must be HTTP(S)")
        self.base_url = base_url.rstrip("/")
        self.enabled = bool(enabled)
        self.network_used = False
        self.timeout_seconds = float(timeout_seconds)

    def request(self, method: str, path: str, *, payload: Mapping[str, Any] | None, headers: Mapping[str, str]) -> APIResponse:
        if not self.enabled:
            raise APIWorkbenchError("network_disabled", "real API transport is disabled")
        url = self.base_url + _safe_path(path)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request_headers = {"Accept": "application/json, text/event-stream", **dict(headers)}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
        started = time.perf_counter()
        self.network_used = True
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read(4 * 1024 * 1024)
                status = int(response.status)
                response_headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            raw = exc.read(4 * 1024 * 1024)
            status = int(exc.code)
            response_headers = {str(key).lower(): str(value) for key, value in exc.headers.items()}
        except (OSError, TimeoutError) as exc:
            raise APIWorkbenchError("transport_error", "API transport failed", retryable=True) from exc
        content_type = response_headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip().lower()
        if "text/event-stream" in content_type:
            body_value: Any = raw.decode("utf-8", errors="replace")
        else:
            try:
                body_value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body_value = {"_text_chars": len(raw.decode("utf-8", errors="replace"))}
        return APIResponse(status, body_value, response_headers, (time.perf_counter() - started) * 1000, content_type)


@dataclass(frozen=True, slots=True)
class ProbeCase:
    case_id: str
    operation: str
    harness_method: str
    harness_path: str
    harness_payload: Mapping[str, Any] | None
    main_method: str
    main_path: str
    main_payload: Mapping[str, Any] | None
    expected: str = "success"

    def __post_init__(self) -> None:
        if not self.case_id or not self.operation:
            raise APIWorkbenchError("invalid_case", "case identity is required")
        _safe_path(self.harness_path)
        _safe_path(self.main_path)
        if self.harness_method.upper() not in {"GET", "POST", "DELETE"} or self.main_method.upper() not in {"GET", "POST", "DELETE"}:
            raise APIWorkbenchError("invalid_case", "unsupported HTTP method")
        _safe_json(self.harness_payload or {})
        _safe_json(self.main_payload or {})

    @property
    def request_digest(self) -> str:
        return _digest({"harness": self.harness_payload or {}, "main": self.main_payload or {}, "operation": self.operation})

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operation": self.operation,
            "expected": self.expected,
            "harness": {"method": self.harness_method.upper(), "path": self.harness_path, "payload_shape": _redacted_shape(self.harness_payload or {})},
            "main": {"method": self.main_method.upper(), "path": self.main_path, "payload_shape": _redacted_shape(self.main_payload or {})},
            "request_digest": self.request_digest,
        }


def builtin_api_cases(*, model: str = "fixture", message: str = "fixture prompt") -> tuple[ProbeCase, ...]:
    """Return contract probes without executing either server."""
    return (
        ProbeCase("health", "health", "GET", "/healthz", None, "GET", "/api/health", None),
        ProbeCase("models", "model_list", "GET", "/v1/models", None, "GET", "/api/models", None),
        ProbeCase(
            "chat-completion", "chat_completion", "POST", "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": message}], "max_tokens": 32, "temperature": 0.2},
            "POST", "/api/chat", {"message": message, "max_new_tokens": 32, "temperature": 0.2, "top_p": 0.9, "streaming_mode": "full", "routing_preference": "local_only"},
        ),
        ProbeCase(
            "chat-invalid", "chat_error", "POST", "/v1/chat/completions", {"messages": []},
            "POST", "/api/chat", {"message": ""}, expected="invalid_request",
        ),
        ProbeCase(
            "chat-stream", "chat_stream", "POST", "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": message}], "max_tokens": 16, "stream": True},
            "POST", "/api/chat/stream", {"message": message, "max_new_tokens": 16, "streaming_mode": "fast", "routing_preference": "local_only"},
        ),
    )


def _body_shape(body: Any, operation: str) -> dict[str, Any]:
    if isinstance(body, str):
        events = [line[6:].strip() for line in body.splitlines() if line.startswith("data:")]
        parsed: list[Any] = []
        for item in events[:32]:
            if item == "[DONE]":
                parsed.append("done")
            else:
                try:
                    parsed.append(json.loads(item))
                except json.JSONDecodeError:
                    parsed.append("invalid_event")
        done = "done" in parsed or any(isinstance(item, Mapping) and item.get("done") is True for item in parsed)
        return {"kind": "sse", "event_count": len(events), "event_shapes": [_redacted_shape(item) for item in parsed], "done": done}
    if not isinstance(body, Mapping):
        return {"kind": type(body).__name__}
    if operation == "model_list":
        values = body.get("data") if isinstance(body.get("data"), list) else body.get("models")
        return {"kind": "model_list", "container": "data" if "data" in body else "models" if "models" in body else "missing", "item_count": len(values) if isinstance(values, list) else 0, "item_keys": sorted(set(str(key) for item in values[:8] if isinstance(item, Mapping) for key in item)) if isinstance(values, list) else []}
    if operation == "health":
        return {"kind": "health", "keys": sorted(str(key) for key in body), "status_present": "status" in body}
    if operation == "chat_completion":
        choices = body.get("choices")
        content_present = bool(
            (choices and isinstance(choices[0], Mapping) and ("message" in choices[0] or "content" in choices[0]))
            or body.get("content")
        )
        return {"kind": "chat_completion" if isinstance(choices, list) else "chat_response", "keys": sorted(str(key) for key in body), "choice_count": len(choices) if isinstance(choices, list) else 0, "content_present": content_present}
    if operation == "chat_stream":
        return {"kind": "stream_json", "keys": sorted(str(key) for key in body), "done_present": "done" in body, "error_present": "error" in body}
    error = body.get("error")
    if isinstance(error, Mapping):
        return {"kind": "error", "keys": sorted(str(key) for key in body), "error_keys": sorted(str(key) for key in error), "error_code": error.get("code") or error.get("type")}
    if "detail" in body:
        return {"kind": "error", "keys": sorted(str(key) for key in body), "error_keys": ["detail"], "error_code": "validation_error"}
    return {"kind": "object", "keys": sorted(str(key) for key in body)}


def _semantic_shape(shape: Mapping[str, Any], operation: str) -> str:
    if operation == "model_list":
        return "model_list" if shape.get("kind") == "model_list" and shape.get("item_count", 0) >= 0 else "invalid"
    if operation == "health":
        return "health" if shape.get("status_present") else "invalid"
    if operation == "chat_completion":
        return "chat_completion" if shape.get("content_present") else "invalid"
    if operation == "chat_stream":
        return "sse" if shape.get("kind") == "sse" and shape.get("done") else "invalid"
    return "error" if shape.get("kind") == "error" else "invalid"


@dataclass(frozen=True, slots=True)
class APIProbeResult:
    case_id: str
    operation: str
    expected: str
    harness_status: int | None
    main_status: int | None
    harness_latency_ms: float | None
    main_latency_ms: float | None
    harness_shape: Mapping[str, Any]
    main_shape: Mapping[str, Any]
    request_digest: str
    response_digest: str
    status: str
    mismatches: tuple[str, ...] = ()
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operation": self.operation,
            "expected": self.expected,
            "harness_status": self.harness_status,
            "main_status": self.main_status,
            "harness_latency_ms": self.harness_latency_ms,
            "main_latency_ms": self.main_latency_ms,
            "harness_shape": dict(self.harness_shape),
            "main_shape": dict(self.main_shape),
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "status": self.status,
            "mismatches": list(self.mismatches),
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class APIWorkbenchReport:
    cases: tuple[APIProbeResult, ...]
    case_specs: tuple[ProbeCase, ...]
    runner_kind: str = "fixture"
    network_used: bool = False
    weights_loaded: bool = False
    schema: str = API_WORKBENCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != API_WORKBENCH_SCHEMA or self.runner_kind not in {"fixture", "transport"}:
            raise APIWorkbenchError("invalid_report", "API workbench report identity is invalid")
        if self.network_used and self.runner_kind == "fixture":
            raise APIWorkbenchError("invalid_report", "fixture report cannot use network")
        if self.weights_loaded:
            raise APIWorkbenchError("invalid_report", "API workbench cannot load weights")

    @property
    def checks(self) -> dict[str, bool]:
        return {
            "cases_complete": len(self.cases) == len(self.case_specs) and len(self.cases) > 0,
            "status_captured": all(item.harness_status is not None and item.main_status is not None for item in self.cases),
            "request_body_redacted": all("prompt" not in json.dumps(item.as_dict(), ensure_ascii=False).lower() for item in self.cases),
            "endpoint_paths_relative": all(path.startswith("/") and not _URL.search(path) and "\\" not in path and ".." not in path.split("/") for spec in self.case_specs for path in (spec.harness_path, spec.main_path)),
            "latency_nonnegative": all((item.harness_latency_ms or 0) >= 0 and (item.main_latency_ms or 0) >= 0 for item in self.cases),
            "drift_explicit": all(item.status in {"matched", "drifted", "failed", "not_run"} for item in self.cases),
            "offline_boundary": not self.network_used and not self.weights_loaded,
        }

    @property
    def valid(self) -> bool:
        return bool(self.cases) and all(self.checks.values())

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    @property
    def drift_count(self) -> int:
        return sum(item.status == "drifted" for item in self.cases)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "valid": self.valid,
            "runner_kind": self.runner_kind,
            "network_used": self.network_used,
            "weights_loaded": self.weights_loaded,
            "case_specs": [item.as_dict() for item in self.case_specs],
            "results": [item.as_dict() for item in self.cases],
            "summary": {"case_count": len(self.cases), "matched": sum(item.status == "matched" for item in self.cases), "drifted": self.drift_count, "failed": sum(item.status == "failed" for item in self.cases)},
            "checks": self.checks,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# TOOL-API-WB-01 API workbench",
            "",
            f"- Valid: `{str(self.valid).lower()}`; runner: `{self.runner_kind}`; cases: `{len(self.cases)}`; drift: `{self.drift_count}`; network used: `{str(self.network_used).lower()}`; report digest: `{self.digest}`",
            "- The report compares contract shapes and status semantics only. It never stores prompt, message, response body, credentials or a model-quality claim.",
            "",
            "## Probe results",
            "",
            "| case | operation | harness | main | latency ms | status | mismatch | error code |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
        for item in self.cases:
            latency = f"{item.harness_latency_ms:g}/{item.main_latency_ms:g}" if item.harness_latency_ms is not None and item.main_latency_ms is not None else "NOT RUN"
            lines.append(f"| `{item.case_id}` | `{item.operation}` | {item.harness_status or '-'} | {item.main_status or '-'} | {latency} | `{item.status}` | `{'; '.join(item.mismatches) or '-'}` | `{item.error_code or '-'}` |")
        lines.extend(("", "## Checks", ""))
        lines.extend(f"- `{name}`: **{'passed' if passed else 'failed'}**" for name, passed in self.checks.items())
        lines.append("")
        return "\n".join(lines)


def _response_digest(response: APIResponse) -> str:
    return _digest({"status": response.status_code, "shape": _redacted_shape(response.body), "content_type": response.content_type})


def _run_case(case: ProbeCase, harness: APITransport, main: APITransport) -> APIProbeResult:
    try:
        harness_response = harness.request(case.harness_method, case.harness_path, payload=case.harness_payload, headers={"accept": "application/json"})
        main_response = main.request(case.main_method, case.main_path, payload=case.main_payload, headers={"accept": "application/json"})
    except APIWorkbenchError as exc:
        return APIProbeResult(case.case_id, case.operation, case.expected, None, None, None, None, {}, {}, case.request_digest, _digest({"error": exc.code}), "failed", (exc.code,), exc.code)
    harness_shape = _body_shape(harness_response.body, case.operation)
    main_shape = _body_shape(main_response.body, case.operation)
    mismatches: list[str] = []
    if harness_response.status_code != main_response.status_code:
        mismatches.append("status_code")
    if case.expected == "success":
        if not 200 <= harness_response.status_code < 300:
            mismatches.append("harness_status_not_success")
        if not 200 <= main_response.status_code < 300:
            mismatches.append("main_status_not_success")
        if _semantic_shape(harness_shape, case.operation) != _semantic_shape(main_shape, case.operation):
            mismatches.append("response_semantic_shape")
    else:
        if harness_response.status_code < 400 or main_response.status_code < 400:
            mismatches.append("error_status_not_rejected")
        if _semantic_shape(harness_shape, case.operation) != "error" or _semantic_shape(main_shape, case.operation) != "error":
            mismatches.append("error_envelope_shape")
        if harness_shape.get("error_code") != main_shape.get("error_code"):
            mismatches.append("error_code")
    if harness_response.content_type != main_response.content_type and case.operation != "chat_stream":
        mismatches.append("content_type")
    status = "failed" if mismatches and ("status_not" in ";".join(mismatches) or "error_status" in ";".join(mismatches)) else "drifted" if mismatches else "matched"
    error_code = None
    if case.expected != "success":
        error_code = f"{harness_shape.get('error_code') or '-'}->{main_shape.get('error_code') or '-'}"
    return APIProbeResult(case.case_id, case.operation, case.expected, harness_response.status_code, main_response.status_code, harness_response.latency_ms, main_response.latency_ms, harness_shape, main_shape, case.request_digest, _digest({"harness": _response_digest(harness_response), "main": _response_digest(main_response)}), status, tuple(mismatches), error_code)


def run_api_workbench(cases: Sequence[ProbeCase] | None = None, *, harness_transport: APITransport | None = None, main_transport: APITransport | None = None) -> APIWorkbenchReport:
    selected = tuple(cases or builtin_api_cases())
    if not selected:
        raise APIWorkbenchError("no_cases", "API workbench requires at least one case")
    if harness_transport is None and main_transport is None:
        harness_transport, main_transport = _fixture_transports()
    elif harness_transport is None or main_transport is None:
        raise APIWorkbenchError("transport_required", "provide both harness and main transports")
    results = tuple(_run_case(case, harness_transport, main_transport) for case in selected)
    network_used = bool(getattr(harness_transport, "network_used", False) or getattr(main_transport, "network_used", False))
    runner_kind = "transport" if network_used else "fixture"
    return APIWorkbenchReport(results, selected, runner_kind=runner_kind, network_used=network_used)


def _fixture_transports() -> tuple[MemoryAPITransport, MemoryAPITransport]:
    harness = MemoryAPITransport({
        "GET /healthz": APIResponse(200, {"status": "ok", "backend": "fixture"}, latency_ms=1.2),
        "GET /v1/models": APIResponse(200, {"object": "list", "data": [{"id": "fixture", "object": "model", "owned_by": "harness"}]}, latency_ms=1.5),
        "POST /v1/chat/completions": APIResponse(200, {"id": "fixture", "object": "chat.completion", "model": "fixture", "choices": [{"message": {"role": "assistant", "content": "fixture answer"}, "finish_reason": "stop"}]}, latency_ms=4.0),
    })
    main = MemoryAPITransport({
        "GET /api/health": APIResponse(200, {"status": "ok", "timestamp": 1}, latency_ms=1.0),
        "GET /api/models": APIResponse(200, {"models": [{"id": "fixture", "name": "Fixture"}], "active_model_id": "fixture"}, latency_ms=1.8),
        "POST /api/chat": APIResponse(200, {"content": "fixture answer", "thinking_content": None, "metrics": {"total_time_seconds": 0.004}, "followups": []}, latency_ms=4.6),
        "POST /api/chat/stream": APIResponse(200, "data: {\"token\":\"fixture\"}\n\ndata: {\"done\":true}\n\n", {"content-type": "text/event-stream"}, 3.2, "text/event-stream"),
    })
    invalid_harness = APIResponse(400, {"error": {"message": "invalid", "type": "invalid_request_error", "code": "invalid_messages"}}, latency_ms=0.4)
    invalid_main = APIResponse(422, {"detail": [{"type": "string_too_short", "loc": ["body", "message"]}]}, latency_ms=0.5)
    main.responses["POST /api/chat"] = lambda _method, _path, payload: invalid_main if payload and not payload.get("message") else APIResponse(200, {"content": "fixture answer", "metrics": {}, "followups": []}, latency_ms=4.6)
    harness.responses["POST /v1/chat/completions"] = lambda _method, _path, payload: APIResponse(400, {"error": {"message": "invalid", "type": "invalid_request_error", "code": "invalid_messages"}}, latency_ms=0.4) if payload and not payload.get("model") else APIResponse(200, {"id": "fixture", "object": "chat.completion", "model": "fixture", "choices": [{"message": {"role": "assistant", "content": "fixture answer"}, "finish_reason": "stop"}]}, latency_ms=4.0) if not payload or not payload.get("stream") else APIResponse(200, "data: {\"choices\":[{\"delta\":{\"content\":\"fixture\"}}]}\n\ndata: [DONE]\n\n", {"content-type": "text/event-stream"}, 3.8, "text/event-stream")
    return harness, main


def _load_cases(path: str | None) -> tuple[ProbeCase, ...]:
    if not path:
        return builtin_api_cases()
    source = json.loads(__import__("pathlib").Path(path).read_text(encoding="utf-8"))
    if not isinstance(source, Mapping) or source.get("schema") != API_WORKBENCH_INPUT_SCHEMA or not isinstance(source.get("cases"), list):
        raise APIWorkbenchError("invalid_schema", "unsupported API workbench input schema")
    cases: list[ProbeCase] = []
    for raw in source["cases"]:
        if not isinstance(raw, Mapping):
            raise APIWorkbenchError("invalid_case", "case must be an object")
        cases.append(ProbeCase(raw.get("case_id", ""), raw.get("operation", ""), raw.get("harness_method", "GET"), raw.get("harness_path", ""), raw.get("harness_payload"), raw.get("main_method", "GET"), raw.get("main_path", ""), raw.get("main_payload"), raw.get("expected", "success")))
    if len({item.case_id for item in cases}) != len(cases):
        raise APIWorkbenchError("duplicate_case", "case IDs must be unique")
    return tuple(cases)


def _write_text(path_value: str, content: str) -> None:
    if path_value == "-":
        print(content, end="")
        return
    from pathlib import Path
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare harness /v1 and project /api contract fixtures")
    parser.add_argument("--input", metavar="PATH", help="qlh.api_workbench.v1 case JSON")
    parser.add_argument("--json", dest="json_path", default="", metavar="PATH", help="write JSON report (use - for stdout)")
    parser.add_argument("--markdown", dest="markdown_path", default="", metavar="PATH", help="write Markdown report (use - for stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cases = _load_cases(args.input)
        harness, main_transport = _fixture_transports()
        report = run_api_workbench(cases, harness_transport=harness, main_transport=main_transport)
    except (OSError, ValueError, json.JSONDecodeError, APIWorkbenchError) as exc:
        parser.error(str(exc))
    json_text = json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_text = report.to_markdown()
    outputs = 0
    if args.json_path:
        _write_text(args.json_path, json_text)
        outputs += 1
    if args.markdown_path:
        _write_text(args.markdown_path, markdown_text)
        outputs += 1
    if not outputs:
        print(markdown_text, end="")
    return 0 if report.valid else 1


__all__ = [
    "APIResponse",
    "APITransport",
    "APIProbeResult",
    "APIWorkbenchError",
    "APIWorkbenchReport",
    "API_WORKBENCH_INPUT_SCHEMA",
    "API_WORKBENCH_SCHEMA",
    "MemoryAPITransport",
    "ProbeCase",
    "UrllibAPITransport",
    "build_parser",
    "builtin_api_cases",
    "main",
    "run_api_workbench",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
