"""Versioned task-worker messages with strict validation and no transport."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


PROTOCOL_NAME = "qlh.task_worker"
PROTOCOL_VERSION = 1
MIN_PROTOCOL_VERSION = 1
MAX_PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

MESSAGE_TYPES = frozenset({
    "hello",
    "hello_ack",
    "stage_offer",
    "stage_accept",
    "lease_renew",
    "stage_result",
    "stage_error",
    "stage_cancel",
    "stage_cancelled",
})

_MESSAGE_ID = re.compile(r"^msg_[A-Za-z0-9_-]{8,96}$")
_WORKFLOW_ID = re.compile(r"^wf_[A-Za-z0-9_-]{8,96}$")
_ATTEMPT_ID = re.compile(r"^att_[A-Za-z0-9_-]{8,96}$")
_LEASE_ID = re.compile(r"^lease_[A-Za-z0-9_-]{8,96}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ENVELOPE_FIELDS = {
    "protocol", "version", "message_type", "message_id", "sent_at_ms",
    "payload",
}
_IDENTITY_FIELDS = {
    "workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch",
}
_PAYLOAD_FIELDS = {
    "hello": {
        "node_id", "worker_kind", "min_version", "max_version",
        "capabilities",
    },
    "hello_ack": {
        "coordinator_node_id", "accepted", "selected_version", "reason_code",
    },
    "stage_offer": _IDENTITY_FIELDS | {
        "request_id", "stage_type", "provider_id", "lease_expires_at_ms",
        "root_input", "dependencies", "input_sha256",
    },
    "stage_accept": _IDENTITY_FIELDS | {
        "provider_id", "accepted", "reason_code",
    },
    "lease_renew": _IDENTITY_FIELDS | {"lease_expires_at_ms"},
    "stage_result": _IDENTITY_FIELDS | {
        "provider_id", "output", "output_sha256", "metadata",
    },
    "stage_error": _IDENTITY_FIELDS | {
        "provider_id", "error_code", "retryable",
    },
    "stage_cancel": _IDENTITY_FIELDS | {"reason_code"},
    "stage_cancelled": _IDENTITY_FIELDS | {
        "provider_id", "reason_code",
    },
}


class WorkerProtocolError(ValueError):
    """Stable protocol failure that is safe to return without raw payloads."""

    def __init__(self, message: str, *, code: str, field: str = ""):
        self.code = code
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class WorkerMessage:
    protocol: str
    version: int
    message_type: str
    message_id: str
    sent_at_ms: int
    _payload_json: str

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self._payload_json)
        if not isinstance(value, dict):
            raise RuntimeError("validated WorkerMessage payload is not an object")
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "version": self.version,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "sent_at_ms": self.sent_at_ms,
            "payload": self.payload,
        }


def _error(code: str, field: str, message: str) -> WorkerProtocolError:
    return WorkerProtocolError(message, code=code, field=field)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error("invalid_integer", field, f"{field} must be an integer")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise _error("invalid_boolean", field, f"{field} must be a boolean")
    return value


def _require_string(
    value: Any,
    field: str,
    *,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
    max_length: int = 256,
) -> str:
    if not isinstance(value, str) or len(value) > max_length:
        raise _error("invalid_string", field, f"{field} must be a string")
    if not value and not allow_empty:
        raise _error("invalid_string", field, f"{field} must not be empty")
    if value and pattern is not None and pattern.fullmatch(value) is None:
        raise _error("invalid_identifier", field, f"{field} is invalid")
    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise _error("invalid_object", field, f"{field} must be an object")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_json_value", field, f"{field} must contain strict JSON"
        ) from exc
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], field: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    reason = ""
    if missing:
        reason += f" missing={missing}"
    if unknown:
        reason += f" unknown={unknown}"
    raise _error(
        "invalid_fields", field, f"{field} fields do not match schema:{reason}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stage_input_sha256(root_input: dict, dependencies: dict) -> str:
    return canonical_sha256({
        "dependencies": dependencies,
        "root_input": root_input,
    })


def _validate_version_range(
    minimum: Any, maximum: Any, *, prefix: str,
) -> tuple[int, int]:
    min_version = _require_int(minimum, f"{prefix}.min_version", minimum=1)
    max_version = _require_int(maximum, f"{prefix}.max_version", minimum=1)
    if min_version > max_version:
        raise _error(
            "invalid_version_range",
            prefix,
            "minimum protocol version exceeds maximum",
        )
    return min_version, max_version


def negotiate_protocol_version(
    remote_min_version: int,
    remote_max_version: int,
    *,
    local_min_version: int = MIN_PROTOCOL_VERSION,
    local_max_version: int = MAX_PROTOCOL_VERSION,
) -> int:
    remote_min, remote_max = _validate_version_range(
        remote_min_version, remote_max_version, prefix="remote"
    )
    local_min, local_max = _validate_version_range(
        local_min_version, local_max_version, prefix="local"
    )
    selected = min(remote_max, local_max)
    if selected < max(remote_min, local_min):
        raise _error(
            "unsupported_protocol_version",
            "version",
            "worker and coordinator protocol ranges do not overlap",
        )
    return selected


def _validate_identity(payload: dict[str, Any]) -> None:
    _require_string(payload["workflow_id"], "payload.workflow_id", pattern=_WORKFLOW_ID)
    _require_string(payload["stage_id"], "payload.stage_id", pattern=_SAFE_ID)
    _require_string(payload["attempt_id"], "payload.attempt_id", pattern=_ATTEMPT_ID)
    _require_string(payload["lease_id"], "payload.lease_id", pattern=_LEASE_ID)
    _require_int(payload["lease_epoch"], "payload.lease_epoch", minimum=1)


def _validate_capabilities(value: Any) -> None:
    capabilities = _require_object(value, "payload.capabilities")
    _require_exact_fields(
        capabilities,
        {"stage_types", "engines", "models", "max_concurrency"},
        "payload.capabilities",
    )
    stage_types = capabilities["stage_types"]
    if not isinstance(stage_types, list) or not stage_types:
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must be a non-empty list",
        )
    if any(value not in {"full_inference", "aggregate"} for value in stage_types):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "unsupported stage type",
        )
    if len(stage_types) != len(set(stage_types)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must not contain duplicates",
        )
    engines = capabilities["engines"]
    if not isinstance(engines, list) or not engines or any(
        value not in {"pytorch", "llama_cpp"} for value in engines
    ):
        raise _error(
            "invalid_capabilities", "payload.capabilities.engines",
            "engines must contain supported engine identifiers",
        )
    if len(engines) != len(set(engines)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.engines",
            "engines must not contain duplicates",
        )
    models = capabilities["models"]
    if not isinstance(models, list):
        raise _error(
            "invalid_capabilities", "payload.capabilities.models",
            "models must be a list",
        )
    model_ids = []
    for index, model in enumerate(models):
        field = f"payload.capabilities.models[{index}]"
        model = _require_object(model, field)
        _require_exact_fields(
            model, {"model_id", "engine", "format", "revision", "sha256"}, field,
        )
        _require_string(model["model_id"], f"{field}.model_id", pattern=_SAFE_ID)
        model_ids.append(model["model_id"])
        if model["engine"] not in engines:
            raise _error(
                "invalid_capabilities", f"{field}.engine",
                "model engine was not declared by the worker",
            )
        _require_string(model["format"], f"{field}.format", pattern=_SAFE_ID)
        _require_string(model["revision"], f"{field}.revision", pattern=_SAFE_ID)
        _require_string(model["sha256"], f"{field}.sha256", pattern=_SHA256)
    if len(model_ids) != len(set(model_ids)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.models",
            "model_id values must be unique",
        )
    max_concurrency = _require_int(
        capabilities["max_concurrency"],
        "payload.capabilities.max_concurrency",
        minimum=1,
    )
    if max_concurrency > 32:
        raise _error(
            "invalid_capabilities", "payload.capabilities.max_concurrency",
            "max_concurrency must not exceed 32",
        )


def _validate_metadata(value: Any) -> None:
    metadata = _require_object(value, "payload.metadata")
    allowed = {"usage", "usage_estimated", "tokens_per_second", "model"}
    if not set(metadata).issubset(allowed):
        raise _error(
            "invalid_fields", "payload.metadata",
            "metadata contains unsupported fields",
        )
    if "usage_estimated" in metadata:
        _require_bool(metadata["usage_estimated"], "payload.metadata.usage_estimated")
    if "model" in metadata:
        _require_string(
            metadata["model"], "payload.metadata.model", max_length=256,
        )
    if "tokens_per_second" in metadata:
        value = metadata["tokens_per_second"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise _error(
                "invalid_number", "payload.metadata.tokens_per_second",
                "tokens_per_second must be finite and non-negative",
            )
    usage = metadata.get("usage")
    if usage is not None:
        usage = _require_object(usage, "payload.metadata.usage")
        allowed_usage = {
            "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens",
        }
        if not set(usage).issubset(allowed_usage):
            raise _error(
                "invalid_fields", "payload.metadata.usage",
                "usage contains unsupported fields",
            )
        for key, item in usage.items():
            _require_int(item, f"payload.metadata.usage.{key}", minimum=0)


def _validate_payload(message_type: str, payload: dict[str, Any]) -> None:
    _require_exact_fields(payload, _PAYLOAD_FIELDS[message_type], "payload")
    if message_type == "hello":
        _require_string(payload["node_id"], "payload.node_id", pattern=_SAFE_ID)
        if payload["worker_kind"] != "pc_full_worker":
            raise _error(
                "unsupported_worker_kind", "payload.worker_kind",
                "worker_kind must be pc_full_worker",
            )
        _validate_version_range(
            payload["min_version"], payload["max_version"], prefix="payload"
        )
        _validate_capabilities(payload["capabilities"])
        return
    if message_type == "hello_ack":
        _require_string(
            payload["coordinator_node_id"], "payload.coordinator_node_id",
            pattern=_SAFE_ID,
        )
        accepted = _require_bool(payload["accepted"], "payload.accepted")
        selected = _require_int(
            payload["selected_version"], "payload.selected_version", minimum=0,
        )
        reason = _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            allow_empty=True, max_length=64,
        )
        if accepted and selected != PROTOCOL_VERSION:
            raise _error(
                "invalid_selected_version", "payload.selected_version",
                "accepted negotiation must select the supported version",
            )
        if (accepted and reason) or (not accepted and (selected != 0 or not reason)):
            raise _error(
                "invalid_negotiation_result", "payload",
                "hello_ack accepted, selected_version and reason_code disagree",
            )
        return

    _validate_identity(payload)
    if "provider_id" in payload:
        _require_string(
            payload["provider_id"], "payload.provider_id", pattern=_SAFE_ID,
        )
    if message_type == "stage_offer":
        _require_string(
            payload["request_id"], "payload.request_id", pattern=_SAFE_ID,
            allow_empty=True,
        )
        if payload["stage_type"] not in {"full_inference", "aggregate"}:
            raise _error(
                "unsupported_stage_type", "payload.stage_type",
                "unsupported stage type",
            )
        _require_int(
            payload["lease_expires_at_ms"], "payload.lease_expires_at_ms",
            minimum=1,
        )
        root_input = _require_object(payload["root_input"], "payload.root_input")
        dependencies = _require_object(
            payload["dependencies"], "payload.dependencies"
        )
        digest = _require_string(
            payload["input_sha256"], "payload.input_sha256", pattern=_SHA256,
        )
        if digest != stage_input_sha256(root_input, dependencies):
            raise _error(
                "input_digest_mismatch", "payload.input_sha256",
                "stage input digest does not match payload",
            )
    elif message_type == "stage_accept":
        accepted = _require_bool(payload["accepted"], "payload.accepted")
        reason = _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            allow_empty=True, max_length=64,
        )
        if accepted == bool(reason):
            raise _error(
                "invalid_acceptance_result", "payload",
                "accepted offers require no reason; rejected offers require one",
            )
    elif message_type == "lease_renew":
        _require_int(
            payload["lease_expires_at_ms"], "payload.lease_expires_at_ms",
            minimum=1,
        )
    elif message_type == "stage_result":
        output = _require_object(payload["output"], "payload.output")
        digest = _require_string(
            payload["output_sha256"], "payload.output_sha256", pattern=_SHA256,
        )
        if digest != canonical_sha256(output):
            raise _error(
                "output_digest_mismatch", "payload.output_sha256",
                "stage output digest does not match output",
            )
        _validate_metadata(payload["metadata"])
    elif message_type == "stage_error":
        _require_string(
            payload["error_code"], "payload.error_code", pattern=_SAFE_CODE,
            max_length=64,
        )
        _require_bool(payload["retryable"], "payload.retryable")
    else:
        _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            max_length=64,
        )


def validate_message(value: Mapping[str, Any]) -> WorkerMessage:
    if not isinstance(value, Mapping):
        raise _error("invalid_envelope", "message", "message must be an object")
    _require_exact_fields(value, _ENVELOPE_FIELDS, "message")
    protocol = _require_string(value["protocol"], "protocol", max_length=64)
    if protocol != PROTOCOL_NAME:
        raise _error("unsupported_protocol", "protocol", "unsupported protocol")
    version = _require_int(value["version"], "version", minimum=1)
    if version != PROTOCOL_VERSION:
        raise _error(
            "unsupported_protocol_version", "version",
            "unsupported protocol version",
        )
    message_type = _require_string(
        value["message_type"], "message_type", max_length=32,
    )
    if message_type not in MESSAGE_TYPES:
        raise _error(
            "unsupported_message_type", "message_type",
            "unsupported message type",
        )
    message_id = _require_string(
        value["message_id"], "message_id", pattern=_MESSAGE_ID,
    )
    sent_at_ms = _require_int(value["sent_at_ms"], "sent_at_ms", minimum=0)
    payload = _require_object(value["payload"], "payload")
    _validate_payload(message_type, payload)
    if message_type in {"stage_offer", "lease_renew"} and (
        payload["lease_expires_at_ms"] <= sent_at_ms
    ):
        raise _error(
            "invalid_lease_deadline", "payload.lease_expires_at_ms",
            "lease deadline must be later than the message timestamp",
        )
    message = WorkerMessage(
        protocol=protocol,
        version=version,
        message_type=message_type,
        message_id=message_id,
        sent_at_ms=sent_at_ms,
        _payload_json=canonical_json(payload),
    )
    if len(canonical_message_bytes(message)) > MAX_MESSAGE_BYTES:
        raise _error(
            "message_too_large", "message", "message exceeds maximum size"
        )
    return message


def build_message(
    message_type: str,
    payload: Mapping[str, Any],
    *,
    message_id: str,
    sent_at_ms: int,
    version: int = PROTOCOL_VERSION,
) -> WorkerMessage:
    return validate_message({
        "protocol": PROTOCOL_NAME,
        "version": version,
        "message_type": message_type,
        "message_id": message_id,
        "sent_at_ms": sent_at_ms,
        "payload": dict(payload),
    })


def decode_message(raw: bytes | str | Mapping[str, Any]) -> WorkerMessage:
    if isinstance(raw, bytes):
        if len(raw) > MAX_MESSAGE_BYTES:
            raise _error(
                "message_too_large", "message", "message exceeds maximum size"
            )
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _error(
                "invalid_encoding", "message", "message must be UTF-8"
            ) from exc
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise _error(
                "message_too_large", "message", "message exceeds maximum size"
            )
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _error("invalid_json", "message", "message is not JSON") from exc
    else:
        decoded = raw
    if not isinstance(decoded, Mapping):
        raise _error("invalid_envelope", "message", "message must be an object")
    return validate_message(decoded)


def canonical_message_bytes(message: WorkerMessage) -> bytes:
    return canonical_json(message.snapshot()).encode("utf-8")


def worker_protocol_status() -> dict[str, Any]:
    """Report protocol readiness without claiming a network adapter exists."""
    return {
        "protocol": PROTOCOL_NAME,
        "min_version": MIN_PROTOCOL_VERSION,
        "max_version": MAX_PROTOCOL_VERSION,
        "fixture_version": 1,
        "schema_ready": True,
        "adapter_connected": False,
        "transport": "not_implemented",
        "admission_state": "protocol_frozen_adapter_not_connected",
    }
