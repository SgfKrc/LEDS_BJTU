"""Authenticated, header-only loopback transport for Qwen3 pipeline dry-runs."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

from qwen3_pipeline_transaction import (
    MAX_ACK_BYTES,
    MAX_ASSIGNMENT_PROBE_BYTES,
    MAX_CONTRACT_BYTES,
)


AUTH_WINDOW_SECONDS = 300.0
MAX_RANGE_ATTEMPTS = 3
MAX_ASSIGNMENT_MANIFEST_BYTES = 2 * 1024 * 1024
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")


class Qwen3LoopbackError(RuntimeError):
    """A loopback control frame or Range response failed closed."""

    def __init__(self, reason_code: str, reason: str):
        super().__init__(reason)
        self.reason_code = str(reason_code)
        self.reason = str(reason)[:1024]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _canonical_bytes(value: Any, maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_loopback_invalid_json", "control frame is not JSON serializable",
        ) from exc
    if len(encoded) > maximum:
        raise Qwen3LoopbackError(
            "qwen3_loopback_oversize", "control frame exceeds serialization limit",
        )
    return encoded


def _payload_digest(message: dict[str, Any]) -> str:
    payload = {key: value for key, value in message.items() if key != "transport_auth"}
    return hashlib.sha256(_canonical_bytes(payload, MAX_CONTRACT_BYTES)).hexdigest()


def sign_loopback_message(
    message: dict[str, Any],
    *,
    peer_node_id: str,
    secret: str,
    now: float | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Bind a dry-run control frame to one HMAC-authenticated TCP peer."""
    if not isinstance(message, dict) or not peer_node_id or not secret:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_unavailable", "loopback HMAC identity is incomplete",
        )
    signed = dict(message)
    signed.pop("transport_auth", None)
    auth = {
        "peer_node_id": str(peer_node_id),
        "contract_sha256": str(message.get("contract_sha256", "")),
        "generation": int(message.get("generation", 0) or 0),
        "phase": str(message.get("phase", "")),
        "payload_sha256": _payload_digest(signed),
        "timestamp": float(time.time() if now is None else now),
        "nonce": str(nonce or uuid.uuid4().hex),
    }
    if not _NONCE.fullmatch(auth["nonce"]):
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_invalid", "loopback nonce is invalid",
        )
    auth["signature"] = hmac.new(
        secret.encode("utf-8"),
        _canonical_bytes(auth, MAX_ACK_BYTES),
        hashlib.sha256,
    ).hexdigest()
    signed["transport_auth"] = auth
    _canonical_bytes(signed, MAX_CONTRACT_BYTES)
    return signed


def verify_loopback_message(
    message: dict[str, Any],
    *,
    authenticated_peer_id: str,
    secret: str,
    now: float | None = None,
) -> tuple[str, str]:
    """Verify peer binding and return ``(nonce, payload_sha256)``."""
    if not isinstance(message, dict):
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_invalid", "loopback message must be an object",
        )
    _canonical_bytes(message, MAX_CONTRACT_BYTES)
    auth = message.get("transport_auth")
    if not isinstance(auth, dict) or not secret:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_missing", "loopback transport authentication is missing",
        )
    required = {
        "peer_node_id", "contract_sha256", "generation", "phase",
        "payload_sha256", "timestamp", "nonce", "signature",
    }
    if set(auth) != required:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_invalid", "loopback transport authentication is malformed",
        )
    if auth.get("peer_node_id") != authenticated_peer_id:
        raise Qwen3LoopbackError(
            "qwen3_loopback_peer_mismatch", "control frame targets another authenticated peer",
        )
    if (
        auth.get("contract_sha256") != message.get("contract_sha256")
        or auth.get("generation") != message.get("generation")
        or auth.get("phase") != message.get("phase")
    ):
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_mismatch", "control frame identity changed after signing",
        )
    payload_sha256 = _payload_digest(message)
    if auth.get("payload_sha256") != payload_sha256:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_mismatch", "control frame payload digest changed",
        )
    nonce = str(auth.get("nonce", ""))
    if not _NONCE.fullmatch(nonce):
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_invalid", "loopback nonce is invalid",
        )
    try:
        timestamp = float(auth.get("timestamp"))
    except (TypeError, ValueError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_invalid", "loopback timestamp is invalid",
        ) from exc
    current = float(time.time() if now is None else now)
    if abs(current - timestamp) > AUTH_WINDOW_SECONDS:
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_stale", "loopback control frame is outside the auth window",
        )
    unsigned_auth = {key: value for key, value in auth.items() if key != "signature"}
    expected = hmac.new(
        secret.encode("utf-8"),
        _canonical_bytes(unsigned_auth, MAX_ACK_BYTES),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, str(auth.get("signature", ""))):
        raise Qwen3LoopbackError(
            "qwen3_loopback_auth_mismatch", "loopback HMAC signature does not match",
        )
    return nonce, payload_sha256


def validate_loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(str(base_url or ""))
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise Qwen3LoopbackError(
            "qwen3_loopback_url_rejected", "Range probe requires a loopback HTTP URL",
        )
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if not loopback:
        raise Qwen3LoopbackError(
            "qwen3_loopback_url_rejected", "Range probe host is not loopback",
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise Qwen3LoopbackError(
            "qwen3_loopback_url_rejected", "Range probe base URL must not contain a path",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise Qwen3LoopbackError(
            "qwen3_loopback_url_rejected", "Range probe port is invalid",
        ) from exc
    if port is None or not 1 <= port <= 65535:
        raise Qwen3LoopbackError(
            "qwen3_loopback_url_rejected", "Range probe port is missing",
        )
    return f"http://[{parsed.hostname}]:{port}" if ":" in parsed.hostname else f"http://{parsed.hostname}:{port}"


def fetch_assignment_probe(
    base_url: str,
    model_id: str,
    probe: dict[str, Any],
    *,
    timeout_seconds: float = 2.0,
    max_attempts: int = MAX_RANGE_ATTEMPTS,
) -> dict[str, Any]:
    """Read one contract-bound byte range, with strict resumable validation."""
    base_url = validate_loopback_base_url(base_url)
    try:
        file_size = int(probe.get("file_size", 0) or 0)
        offset = int(probe.get("offset", 0) or 0)
        length = int(probe.get("length", 0) or 0)
        max_attempts = int(max_attempts)
        timeout_seconds = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_range_contract_invalid", "Range probe dimensions are invalid",
        ) from exc
    if (
        length <= 0
        or length > MAX_ASSIGNMENT_PROBE_BYTES
        or offset < 0
        or file_size <= 0
        or offset + length > file_size
        or not 1 <= max_attempts <= MAX_RANGE_ATTEMPTS
        or not 0 < timeout_seconds <= 30
    ):
        raise Qwen3LoopbackError(
            "qwen3_range_contract_invalid", "Range probe is outside its bounded contract",
        )
    relative_path = str(probe.get("relative_path", "") or "").replace("\\", "/")
    if not relative_path or relative_path.startswith("/") or ".." in relative_path.split("/"):
        raise Qwen3LoopbackError(
            "qwen3_range_contract_invalid", "Range probe path is unsafe",
        )
    expected_sha256 = str(probe.get("sha256", "") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise Qwen3LoopbackError(
            "qwen3_range_contract_invalid", "Range probe SHA-256 is invalid",
        )
    encoded_model = urllib.parse.quote(str(model_id or ""), safe="")
    encoded_path = urllib.parse.quote(relative_path, safe="/")
    url = f"{base_url}/api/models/files/{encoded_model}/{encoded_path}"
    body = bytearray()
    attempts = 0
    last_transport_error = ""
    expected_end = offset + length - 1
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    while len(body) < length and attempts < max_attempts:
        attempts += 1
        request_start = offset + len(body)
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={request_start}-{expected_end}",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status != 206:
                    raise Qwen3LoopbackError(
                        "qwen3_range_not_honored",
                        f"Range probe expected HTTP 206, received {status}",
                    )
                content_range = str(response.headers.get("Content-Range", ""))
                match = _CONTENT_RANGE.fullmatch(content_range)
                if not match:
                    raise Qwen3LoopbackError(
                        "qwen3_range_header_invalid", "Content-Range is missing or malformed",
                    )
                start, end, total = map(int, match.groups())
                if (start, end, total) != (request_start, expected_end, file_size):
                    raise Qwen3LoopbackError(
                        "qwen3_range_header_mismatch", "Content-Range differs from the contract",
                    )
                expected_response_length = expected_end - request_start + 1
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except (TypeError, ValueError) as exc:
                        raise Qwen3LoopbackError(
                            "qwen3_range_header_invalid", "Content-Length is invalid",
                        ) from exc
                    if declared_length != expected_response_length:
                        raise Qwen3LoopbackError(
                            "qwen3_range_length_mismatch", "Content-Length differs from Content-Range",
                        )
                try:
                    chunk = response.read(expected_response_length + 1)
                except http.client.IncompleteRead as exc:
                    chunk = exc.partial
                    last_transport_error = "response ended before the contracted range"
                if len(chunk) > expected_response_length:
                    raise Qwen3LoopbackError(
                        "qwen3_range_length_mismatch", "Range response exceeded its contract",
                    )
                body.extend(chunk)
                if len(chunk) < expected_response_length:
                    last_transport_error = "response ended before the contracted range"
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                code = "qwen3_range_auth_required"
            elif exc.code == 403:
                code = "qwen3_range_forbidden"
            elif exc.code == 416:
                code = "qwen3_range_unsatisfiable"
            else:
                code = "qwen3_range_http_error"
            raise Qwen3LoopbackError(code, f"Range probe HTTP {exc.code}") from exc
        except Qwen3LoopbackError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            last_transport_error = str(getattr(exc, "reason", exc))
    if len(body) != length:
        reason_code = (
            "qwen3_range_timeout"
            if "timed out" in last_transport_error.lower()
            else "qwen3_range_truncated"
        )
        raise Qwen3LoopbackError(
            reason_code,
            f"Range probe incomplete after {attempts} attempts: {len(body)}/{length}",
        )
    actual_sha256 = hashlib.sha256(body).hexdigest()
    if actual_sha256 != expected_sha256:
        raise Qwen3LoopbackError(
            "qwen3_range_sha256_mismatch", "Range probe SHA-256 does not match contract",
        )
    return {
        "relative_path": relative_path,
        "offset": offset,
        "length": length,
        "bytes_received": len(body),
        "sha256": actual_sha256,
        "content_range": f"bytes {offset}-{expected_end}/{file_size}",
        "attempts": attempts,
    }


def fetch_assignment_manifest(
    base_url: str,
    message: dict[str, Any],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Fetch and verify the canonical assignment manifest without persisting it."""
    base_url = validate_loopback_base_url(base_url)
    model_id = str(message.get("model_id", "") or "")
    expected_sha256 = str(message.get("assignment_manifest_sha256", "") or "")
    if not model_id or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise Qwen3LoopbackError(
            "qwen3_manifest_contract_invalid", "assignment manifest identity is invalid",
        )
    query = urllib.parse.urlencode({
        "config_id": str(message.get("config_id", "") or ""),
        "plan_id": str(message.get("plan_id", "") or ""),
        "node_id": str(message.get("node_id", "") or ""),
        "start_layer": int(message.get("layer_range", [0, 0])[0]),
        "end_layer": int(message.get("layer_range", [0, 0])[1]),
        "total_layers": int(message.get("total_layers", 0) or 0),
        "has_embedding": int(bool(message.get("has_embedding", False))),
        "has_lm_head": int(bool(message.get("has_lm_head", False))),
    })
    encoded_model = urllib.parse.quote(model_id, safe="")
    url = f"{base_url}/api/models/pipeline-assignment/{encoded_model}?{query}"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode()))
            if status != 200:
                raise Qwen3LoopbackError(
                    "qwen3_manifest_http_error",
                    f"assignment manifest expected HTTP 200, received {status}",
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_length = int(declared)
                    if declared_length < 1:
                        raise Qwen3LoopbackError(
                            "qwen3_manifest_header_invalid",
                            "manifest Content-Length must be positive",
                        )
                    if declared_length > MAX_ASSIGNMENT_MANIFEST_BYTES:
                        raise Qwen3LoopbackError(
                            "qwen3_manifest_oversize", "assignment manifest exceeds size limit",
                        )
                except ValueError as exc:
                    raise Qwen3LoopbackError(
                        "qwen3_manifest_header_invalid", "manifest Content-Length is invalid",
                    ) from exc
            encoded = response.read(MAX_ASSIGNMENT_MANIFEST_BYTES + 1)
    except urllib.error.HTTPError as exc:
        code = (
            "qwen3_manifest_auth_required" if exc.code == 401
            else "qwen3_manifest_forbidden" if exc.code == 403
            else "qwen3_manifest_http_error"
        )
        raise Qwen3LoopbackError(code, f"assignment manifest HTTP {exc.code}") from exc
    except Qwen3LoopbackError:
        raise
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_manifest_transport_error",
            f"assignment manifest request failed: {getattr(exc, 'reason', exc)}",
        ) from exc
    if len(encoded) > MAX_ASSIGNMENT_MANIFEST_BYTES:
        raise Qwen3LoopbackError(
            "qwen3_manifest_oversize", "assignment manifest exceeds size limit",
        )
    if not encoded:
        raise Qwen3LoopbackError(
            "qwen3_manifest_invalid", "assignment manifest is empty",
        )
    try:
        manifest = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_manifest_invalid", "assignment manifest JSON is invalid",
        ) from exc
    if not isinstance(manifest, dict):
        raise Qwen3LoopbackError(
            "qwen3_manifest_invalid", "assignment manifest is not an object",
        )
    manifest_sha256 = str(manifest.get("manifest_sha256", "") or "")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    actual_sha256 = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_fields = {
        "model_id": model_id,
        "model_sha256": message.get("model_sha256"),
        "config_id": message.get("config_id"),
        "plan_id": message.get("plan_id"),
        "node_id": message.get("node_id"),
        "layer_range": message.get("layer_range"),
        "total_layers": message.get("total_layers"),
        "has_embedding": message.get("has_embedding"),
        "has_lm_head": message.get("has_lm_head"),
    }
    if (
        manifest_sha256 != expected_sha256
        or actual_sha256 != expected_sha256
        or any(manifest.get(key) != value for key, value in expected_fields.items())
    ):
        raise Qwen3LoopbackError(
            "qwen3_manifest_digest_mismatch",
            "assignment manifest differs from the signed control contract",
        )
    return {"sha256": actual_sha256, "bytes_received": len(encoded)}


def build_safetensors_header_probe(path: str | Path, *, relative_path: str) -> dict[str, Any]:
    """Describe only a Safetensors header, never the tensor payload."""
    source = Path(path)
    file_size = source.stat().st_size
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Qwen3LoopbackError(
                "qwen3_header_invalid", "Safetensors file has no complete header prefix",
            )
        header_length = int.from_bytes(prefix, "little", signed=False)
        length = 8 + header_length
        if header_length <= 1 or length > MAX_ASSIGNMENT_PROBE_BYTES or length > file_size:
            raise Qwen3LoopbackError(
                "qwen3_header_invalid", "Safetensors header length is outside the probe budget",
            )
        header = prefix + handle.read(header_length)
    if len(header) != length:
        raise Qwen3LoopbackError(
            "qwen3_header_invalid", "Safetensors header is truncated",
        )
    try:
        decoded = json.loads(header[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Qwen3LoopbackError(
            "qwen3_header_invalid", "Safetensors header JSON is invalid",
        ) from exc
    if not isinstance(decoded, dict) or not decoded:
        raise Qwen3LoopbackError(
            "qwen3_header_invalid", "Safetensors header contains no tensors",
        )
    return {
        "relative_path": str(relative_path).replace("\\", "/"),
        "file_size": file_size,
        "offset": 0,
        "length": length,
        "sha256": hashlib.sha256(header).hexdigest(),
    }


class Qwen3PipelineLoopbackWorker:
    """Validate signed dry-run frames and return deterministic ACKs."""

    def __init__(
        self,
        *,
        node_id: str,
        secret: str,
        base_url: str,
        available_bytes: int | Callable[[dict[str, Any]], int],
        timeout_seconds: float = 2.0,
        sidecar_session_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.node_id = str(node_id)
        self.secret = str(secret)
        self.base_url = validate_loopback_base_url(base_url)
        self.available_bytes = available_bytes
        self.timeout_seconds = float(timeout_seconds)
        self._nonce_payloads: dict[str, str] = {}
        self._responses: dict[tuple[str, str], dict[str, Any]] = {}
        self._active_contracts: set[str] = set()
        self._sidecar_session_factory = sidecar_session_factory
        self._sidecar_sessions: dict[str, Any] = {}

    def _capacity(self, message: dict[str, Any]) -> int:
        value = self.available_bytes(message) if callable(self.available_bytes) else self.available_bytes
        return max(0, int(value))

    def abort_sidecar_sessions(self) -> None:
        """Best-effort cleanup for a TCP disconnect before worker eviction."""
        sessions = list(self._sidecar_sessions.values())
        self._sidecar_sessions.clear()
        for session in sessions:
            try:
                session.abort()
            except Exception:
                continue

    def handle(self, message: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        nonce, payload_sha256 = verify_loopback_message(
            message,
            authenticated_peer_id=self.node_id,
            secret=self.secret,
            now=now,
        )
        previous_payload = self._nonce_payloads.get(nonce)
        response_key = (str(message.get("phase", "")), payload_sha256)
        if previous_payload is not None:
            if previous_payload != payload_sha256 or response_key not in self._responses:
                raise Qwen3LoopbackError(
                    "qwen3_loopback_replay_mismatch", "loopback nonce was reused with changed state",
                )
            return dict(self._responses[response_key])
        if message.get("operation") != "qwen3_pipeline_dry_run" or message.get("dry_run") is not True:
            raise Qwen3LoopbackError(
                "qwen3_loopback_contract_mismatch", "message is not a Qwen3 dry-run",
            )
        execution_mode = str(message.get("execution_mode", "metadata_only") or "metadata_only")
        if execution_mode not in {"metadata_only", "node_local_sidecar"}:
            raise Qwen3LoopbackError(
                "qwen3_loopback_contract_mismatch", "loopback execution mode is unsupported",
            )
        if (
            message.get("node_id") != self.node_id
            or message.get("network_dispatch") is not True
            or message.get("loopback_only") is not True
            or bool(message.get("weight_materialization")) != (execution_mode == "node_local_sidecar")
            or message.get("full_model_fallback") is not False
        ):
            raise Qwen3LoopbackError(
                "qwen3_loopback_contract_mismatch", "loopback safety flags do not match",
            )
        if validate_loopback_base_url(
            str(message.get("assignment_base_url", "") or "")
        ) != self.base_url:
            raise Qwen3LoopbackError(
                "qwen3_loopback_contract_mismatch",
                "signed assignment base URL differs from worker state",
            )
        phase = str(message.get("phase", ""))
        contract_sha256 = str(message.get("contract_sha256", ""))
        base_ack = {
            "schema_version": 1,
            "operation": "qwen3_pipeline_dry_run_ack",
            "dry_run": True,
            "phase": phase,
            "node_id": self.node_id,
            "config_id": message.get("config_id"),
            "plan_id": message.get("plan_id"),
            "generation": message.get("generation"),
            "contract_sha256": contract_sha256,
            "model_sha256": message.get("model_sha256"),
            "segment_sha256": message.get("segment_sha256"),
            "assignment_manifest_sha256": message.get("assignment_manifest_sha256"),
            "kv_contract_sha256": message.get("kv_contract_sha256"),
            "hidden_handoff_sha256": message.get("hidden_handoff_sha256"),
            "layer_range": message.get("layer_range"),
            "full_model_materialized": False,
            "segment_materialized": False,
        }
        if phase == "prepare":
            capacity = self._capacity(message)
            if capacity < int(message.get("required_bytes", 0) or 0):
                raise Qwen3LoopbackError(
                    "qwen3_prepare_capacity_changed", "loopback worker capacity is insufficient",
                )
            probe = message.get("assignment_probe")
            if not isinstance(probe, dict):
                raise Qwen3LoopbackError(
                    "qwen3_range_contract_invalid", "prepare frame has no assignment probe",
                )
            manifest_report = fetch_assignment_manifest(
                self.base_url,
                message,
                timeout_seconds=self.timeout_seconds,
            )
            report = fetch_assignment_probe(
                self.base_url,
                str(message.get("model_id", "") or ""),
                probe,
                timeout_seconds=self.timeout_seconds,
            )
            sidecar_report = None
            if execution_mode == "node_local_sidecar":
                if self._sidecar_session_factory is None:
                    raise Qwen3LoopbackError(
                        "qwen3_sidecar_unavailable",
                        "node-local sidecar execution was requested without a session factory",
                    )
                try:
                    session = self._sidecar_session_factory(dict(message))
                    sidecar_report = session.prepare()
                except Qwen3LoopbackError:
                    raise
                except Exception as exc:
                    raise Qwen3LoopbackError(
                        "qwen3_sidecar_prepare_failed", str(exc),
                    ) from exc
                self._sidecar_sessions[contract_sha256] = session
            base_ack.update({
                "status": "prepared",
                "available_bytes": capacity,
                "assignment_probe": report,
                "assignment_manifest": manifest_report,
            })
            if sidecar_report is not None:
                base_ack["sidecar"] = sidecar_report
            self._active_contracts.add(contract_sha256)
        elif phase == "commit":
            if contract_sha256 not in self._active_contracts:
                raise Qwen3LoopbackError(
                    "qwen3_loopback_not_prepared", "commit has no prepared loopback state",
                )
            sidecar = self._sidecar_sessions.get(contract_sha256)
            sidecar_report = None
            if sidecar is not None:
                try:
                    sidecar_report = sidecar.commit()
                except Exception as exc:
                    try:
                        sidecar.abort()
                    except Exception:
                        pass
                    self._sidecar_sessions.pop(contract_sha256, None)
                    raise Qwen3LoopbackError(
                        "qwen3_sidecar_commit_failed", str(exc),
                    ) from exc
            base_ack.update({
                "status": "ready",
                "segment_materialized": sidecar is not None,
                "kv_cache_probe": {
                    "segment_index": message.get("segment_index"),
                    "layer_range": message.get("layer_range"),
                    "cache_generation": message.get("generation"),
                    "sequence_length": 0,
                    "dtype": message.get("dtype", "float32"),
                    "device": message.get("execution_device", "cpu"),
                    "phase": "empty",
                    "cleared": True,
                },
            })
            if sidecar_report is not None:
                base_ack["sidecar"] = sidecar_report
        elif phase == "release":
            sidecar = self._sidecar_sessions.pop(contract_sha256, None)
            sidecar_report = None
            if sidecar is not None:
                try:
                    sidecar_report = sidecar.release()
                except Exception as exc:
                    try:
                        sidecar.abort()
                    except Exception:
                        pass
                    raise Qwen3LoopbackError(
                        "qwen3_sidecar_release_failed", str(exc),
                    ) from exc
            self._active_contracts.discard(contract_sha256)
            self._responses.clear()
            self._nonce_payloads.clear()
            base_ack = {
                "node_id": self.node_id,
                "config_id": message.get("config_id"),
                "plan_id": message.get("plan_id"),
                "generation": message.get("generation"),
                "contract_sha256": contract_sha256,
                "phase": "release",
                "status": "released",
                "release": True,
            }
            if sidecar_report is not None:
                base_ack["sidecar"] = sidecar_report
        else:
            raise Qwen3LoopbackError(
                "qwen3_loopback_phase_invalid", "loopback phase is not supported",
            )
        signed_ack = sign_loopback_message(
            base_ack,
            peer_node_id=self.node_id,
            secret=self.secret,
            now=now,
        )
        _canonical_bytes(signed_ack, MAX_ACK_BYTES)
        if phase == "release":
            return signed_ack
        self._nonce_payloads[nonce] = payload_sha256
        self._responses[response_key] = dict(signed_ack)
        return signed_ack


__all__ = [
    "AUTH_WINDOW_SECONDS",
    "MAX_RANGE_ATTEMPTS",
    "Qwen3LoopbackError",
    "Qwen3PipelineLoopbackWorker",
    "build_safetensors_header_probe",
    "fetch_assignment_probe",
    "fetch_assignment_manifest",
    "sign_loopback_message",
    "validate_loopback_base_url",
    "verify_loopback_message",
]
