"""Versioned task-worker messages with strict validation and no transport."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit


PROTOCOL_NAME = "qlh.task_worker"
PROTOCOL_VERSION = 2
MIN_PROTOCOL_VERSION = 1
MAX_PROTOCOL_VERSION = 3
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
_IMAGE_BLOB_ID = re.compile(r"^img_[A-Za-z0-9_-]{16,96}$")
_BLOB_LEASE_ID = re.compile(r"^bls_[A-Za-z0-9_-]{16,96}$")
_TRANSFER_GRANT = re.compile(r"^[A-Za-z0-9_-]{16,4096}\.[A-Za-z0-9_-]{43}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# 受支持的推理引擎标识（与 task_provider.ModelIdentity 的校验集保持一致）:
#   pytorch / llama_cpp   本地引擎
#   island                TP 孤岛（路线 A，指纹为端点摘要）
#   external_api          外部推理服务（路线 B，指纹为外部端点摘要）
#   speculative_assisted  投机解码（路线 C-1，本地 draft + 外部 verify）
_SUPPORTED_ENGINES = frozenset({
    "pytorch", "llama_cpp", "island", "external_api", "speculative_assisted",
})
_SUPPORTED_ENGINES_V3 = _SUPPORTED_ENGINES | {"diffusers_sd15"}
_IMAGE_STAGE_TYPES = frozenset({"image_generate", "image_edit", "image_grid"})
_IMAGE_PIPELINE_KINDS = frozenset({
    "sd15_pipeline",
    "sd15_inpaint_pipeline",
    "sd15_instruction_pipeline",
    "sd15_ip_adapter",
})
_IMAGE_DTYPES = frozenset({"float16", "float32"})

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
_PAYLOAD_FIELDS_V2 = {
    **_PAYLOAD_FIELDS,
    "stage_offer": _PAYLOAD_FIELDS["stage_offer"] | {"model_identity"},
    "stage_accept": _PAYLOAD_FIELDS["stage_accept"] | {"retryable"},
}
_PAYLOAD_FIELDS_V3 = {
    **_PAYLOAD_FIELDS_V2,
    "stage_offer": (
        _PAYLOAD_FIELDS_V2["stage_offer"] - {"model_identity"}
    ) | {"artifact_manifest", "transfer_plan"},
    "stage_result": _PAYLOAD_FIELDS_V2["stage_result"] | {"transfer_plan"},
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


def stage_input_sha256(
    root_input: dict,
    dependencies: dict,
    transfer_plan: dict | None = None,
) -> str:
    body: dict[str, Any] = {
        "dependencies": dependencies,
        "root_input": root_input,
    }
    if transfer_plan is not None:
        body["transfer_plan"] = transfer_plan
    return canonical_sha256(body)


def stage_output_sha256(output: dict, transfer_plan: dict | None = None) -> str:
    if transfer_plan is None:
        return canonical_sha256(output)
    return canonical_sha256({"output": output, "transfer_plan": transfer_plan})


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
    if local_min < MIN_PROTOCOL_VERSION or local_max > MAX_PROTOCOL_VERSION:
        raise _error(
            "invalid_version_range",
            "local",
            "local range includes an unimplemented protocol version",
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


def _validate_model_identity(value: Any, field: str) -> dict[str, Any]:
    model = _require_object(value, field)
    _require_exact_fields(
        model,
        {"model_id", "engine", "format", "revision", "sha256"},
        field,
    )
    _require_string(model["model_id"], f"{field}.model_id", pattern=_SAFE_ID)
    # "island": TP 孤岛引擎（网关整请求转发，指纹为端点摘要）
    # "external_api": 外部推理服务（路线 B，指纹为外部端点摘要）
    # "speculative_assisted": 投机解码（路线 C-1，本地 draft + 外部 verify）
    if model["engine"] not in _SUPPORTED_ENGINES:
        raise _error(
            "invalid_model_identity", f"{field}.engine",
            "model engine is unsupported",
        )
    _require_string(model["format"], f"{field}.format", pattern=_SAFE_ID)
    _require_string(model["revision"], f"{field}.revision", pattern=_SAFE_ID)
    _require_string(model["sha256"], f"{field}.sha256", pattern=_SHA256)
    return model


def _validate_blob_descriptor(value: Any, field: str) -> dict[str, Any]:
    descriptor = _require_object(value, field)
    _require_exact_fields(
        descriptor,
        {
            "blob_id",
            "sha256",
            "size_bytes",
            "content_type",
            "width",
            "height",
            "purpose",
        },
        field,
    )
    _require_string(
        descriptor["blob_id"], f"{field}.blob_id", pattern=_IMAGE_BLOB_ID,
    )
    _require_string(descriptor["sha256"], f"{field}.sha256", pattern=_SHA256)
    size_bytes = _require_int(
        descriptor["size_bytes"], f"{field}.size_bytes", minimum=1,
    )
    if size_bytes > 16 * 1024 * 1024:
        raise _error(
            "invalid_blob_descriptor",
            f"{field}.size_bytes",
            "image blob exceeds the protocol byte limit",
        )
    if descriptor["content_type"] not in {
        "image/png", "image/jpeg", "image/webp",
    }:
        raise _error(
            "invalid_blob_descriptor",
            f"{field}.content_type",
            "image blob content type is unsupported",
        )
    width = _require_int(descriptor["width"], f"{field}.width", minimum=1)
    height = _require_int(descriptor["height"], f"{field}.height", minimum=1)
    if width > 2048 or height > 2048 or width * height > 2048 * 2048:
        raise _error(
            "invalid_blob_descriptor",
            field,
            "image blob dimensions exceed the protocol limit",
        )
    _require_string(descriptor["purpose"], f"{field}.purpose", pattern=_SAFE_ID)
    return descriptor


def _validate_transfer_plan(value: Any, expected_blob_ids: tuple[str, ...]) -> None:
    field = "payload.transfer_plan"
    plan = _require_object(value, field)
    _require_exact_fields(plan, {"base_url", "downloads"}, field)
    downloads = plan["downloads"]
    if not isinstance(downloads, list) or len(downloads) > 16:
        raise _error(
            "invalid_transfer_plan",
            f"{field}.downloads",
            "transfer downloads must be a bounded list",
        )
    actual_blob_ids: list[str] = []
    for index, raw_download in enumerate(downloads):
        download_field = f"{field}.downloads[{index}]"
        download = _require_object(raw_download, download_field)
        _require_exact_fields(
            download,
            {"blob_id", "lease_id", "grant"},
            download_field,
        )
        blob_id = _require_string(
            download["blob_id"],
            f"{download_field}.blob_id",
            pattern=_IMAGE_BLOB_ID,
        )
        _require_string(
            download["lease_id"],
            f"{download_field}.lease_id",
            pattern=_BLOB_LEASE_ID,
        )
        _require_string(
            download["grant"],
            f"{download_field}.grant",
            pattern=_TRANSFER_GRANT,
            max_length=4096,
        )
        actual_blob_ids.append(blob_id)
    if actual_blob_ids != sorted(actual_blob_ids) or len(actual_blob_ids) != len(
        set(actual_blob_ids)
    ):
        raise _error(
            "invalid_transfer_plan",
            f"{field}.downloads",
            "transfer downloads must be unique and ordered by blob_id",
        )
    if tuple(actual_blob_ids) != tuple(sorted(set(expected_blob_ids))):
        raise _error(
            "transfer_plan_mismatch",
            f"{field}.downloads",
            "transfer downloads do not match the stage blob descriptors",
        )
    base_url = plan["base_url"]
    if not actual_blob_ids:
        if base_url is not None:
            raise _error(
                "invalid_transfer_plan",
                f"{field}.base_url",
                "base_url must be null when no blob download is required",
            )
        return
    base_url = _require_string(
        base_url,
        f"{field}.base_url",
        max_length=512,
    )
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise _error(
            "invalid_transfer_plan",
            f"{field}.base_url",
            "data-plane base URL is invalid",
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is None and ":" in parsed.netloc.rsplit("]", 1)[-1]
    ):
        raise _error(
            "invalid_transfer_plan",
            f"{field}.base_url",
            "data-plane base URL must be an HTTP(S) origin without credentials",
        )


def _validate_artifact_manifest(value: Any, field: str) -> dict[str, Any]:
    manifest = _require_object(value, field)
    _require_exact_fields(
        manifest,
        {"artifact_id", "pipeline_kind", "revision", "sha256", "components"},
        field,
    )
    _require_string(
        manifest["artifact_id"], f"{field}.artifact_id", pattern=_SAFE_ID,
    )
    pipeline_kind = _require_string(
        manifest["pipeline_kind"], f"{field}.pipeline_kind", pattern=_SAFE_ID,
    )
    if pipeline_kind not in _IMAGE_PIPELINE_KINDS:
        raise _error(
            "invalid_artifact_manifest",
            f"{field}.pipeline_kind",
            "artifact pipeline kind is unsupported",
        )
    _require_string(manifest["revision"], f"{field}.revision", pattern=_SAFE_ID)
    _require_string(manifest["sha256"], f"{field}.sha256", pattern=_SHA256)
    components = manifest["components"]
    if not isinstance(components, list) or not components or len(components) > 16:
        raise _error(
            "invalid_artifact_manifest",
            f"{field}.components",
            "artifact components must be a non-empty bounded list",
        )
    component_ids: list[str] = []
    for index, raw_component in enumerate(components):
        component_field = f"{field}.components[{index}]"
        component = _require_object(raw_component, component_field)
        _require_exact_fields(
            component,
            {"artifact_id", "artifact_kind", "sha256"},
            component_field,
        )
        component_id = _require_string(
            component["artifact_id"],
            f"{component_field}.artifact_id",
            pattern=_SAFE_ID,
        )
        _require_string(
            component["artifact_kind"],
            f"{component_field}.artifact_kind",
            pattern=_SAFE_ID,
        )
        _require_string(
            component["sha256"],
            f"{component_field}.sha256",
            pattern=_SHA256,
        )
        component_ids.append(component_id)
    if len(component_ids) != len(set(component_ids)):
        raise _error(
            "invalid_artifact_manifest",
            f"{field}.components",
            "artifact component identifiers must be unique",
        )
    if component_ids != sorted(component_ids):
        raise _error(
            "invalid_artifact_manifest",
            f"{field}.components",
            "artifact components must be ordered by artifact_id",
        )
    expected_sha256 = canonical_sha256({
        "artifact_id": manifest["artifact_id"],
        "pipeline_kind": manifest["pipeline_kind"],
        "revision": manifest["revision"],
        "components": manifest["components"],
    })
    if manifest["sha256"] != expected_sha256:
        raise _error(
            "artifact_manifest_digest_mismatch",
            f"{field}.sha256",
            "artifact manifest digest does not match its canonical components",
        )
    return manifest


def _validate_image_capabilities(value: Any) -> None:
    field = "payload.capabilities.image"
    image = _require_object(value, field)
    _require_exact_fields(
        image,
        {
            "pipeline_kinds",
            "dtypes",
            "max_width",
            "max_height",
            "max_pixels",
            "max_batch",
            "supports_controlnet",
            "supports_step_cancel",
            "artifact_manifests",
        },
        field,
    )
    pipeline_kinds = image["pipeline_kinds"]
    if (
        not isinstance(pipeline_kinds, list)
        or not pipeline_kinds
        or len(pipeline_kinds) != len(set(pipeline_kinds))
        or any(item not in _IMAGE_PIPELINE_KINDS for item in pipeline_kinds)
    ):
        raise _error(
            "invalid_capabilities",
            f"{field}.pipeline_kinds",
            "pipeline_kinds contains unsupported or duplicate values",
        )
    dtypes = image["dtypes"]
    if (
        not isinstance(dtypes, list)
        or not dtypes
        or len(dtypes) != len(set(dtypes))
        or any(item not in _IMAGE_DTYPES for item in dtypes)
    ):
        raise _error(
            "invalid_capabilities",
            f"{field}.dtypes",
            "dtypes contains unsupported or duplicate values",
        )
    max_width = _require_int(image["max_width"], f"{field}.max_width", minimum=64)
    max_height = _require_int(image["max_height"], f"{field}.max_height", minimum=64)
    max_pixels = _require_int(image["max_pixels"], f"{field}.max_pixels", minimum=4096)
    max_batch = _require_int(image["max_batch"], f"{field}.max_batch", minimum=1)
    if (
        max_width > 2048
        or max_height > 2048
        or max_width % 8
        or max_height % 8
        or max_pixels > 2048 * 2048
        or max_batch > 16
    ):
        raise _error(
            "invalid_capabilities",
            field,
            "image capability limits exceed the protocol hard limits",
        )
    _require_bool(image["supports_controlnet"], f"{field}.supports_controlnet")
    _require_bool(image["supports_step_cancel"], f"{field}.supports_step_cancel")
    manifests = image["artifact_manifests"]
    if not isinstance(manifests, list) or not manifests or len(manifests) > 64:
        raise _error(
            "invalid_capabilities",
            f"{field}.artifact_manifests",
            "artifact_manifests must be a non-empty bounded list",
        )
    manifest_ids: list[str] = []
    for index, raw_manifest in enumerate(manifests):
        manifest = _validate_artifact_manifest(
            raw_manifest,
            f"{field}.artifact_manifests[{index}]",
        )
        if manifest["pipeline_kind"] not in pipeline_kinds:
            raise _error(
                "invalid_capabilities",
                f"{field}.artifact_manifests[{index}].pipeline_kind",
                "artifact manifest pipeline kind was not declared",
            )
        manifest_ids.append(manifest["artifact_id"])
    if len(manifest_ids) != len(set(manifest_ids)):
        raise _error(
            "invalid_capabilities",
            f"{field}.artifact_manifests",
            "artifact manifest identifiers must be unique",
        )


def _validate_capabilities(value: Any, *, version: int) -> None:
    capabilities = _require_object(value, "payload.capabilities")
    expected_fields = {"stage_types", "engines", "models", "max_concurrency"}
    if version >= 3:
        expected_fields.add("image")
    _require_exact_fields(
        capabilities,
        expected_fields,
        "payload.capabilities",
    )
    stage_types = capabilities["stage_types"]
    if not isinstance(stage_types, list) or not stage_types:
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must be a non-empty list",
        )
    supported_stage_types = (
        _IMAGE_STAGE_TYPES if version >= 3
        else frozenset({"full_inference", "aggregate"})
    )
    if any(value not in supported_stage_types for value in stage_types):
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
        value not in (_SUPPORTED_ENGINES_V3 if version >= 3 else _SUPPORTED_ENGINES)
        for value in engines
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
    if version >= 3 and models:
        raise _error(
            "invalid_capabilities",
            "payload.capabilities.models",
            "v3 image workers use artifact manifests instead of text model identities",
        )
    model_ids = []
    for index, model in enumerate(models):
        field = f"payload.capabilities.models[{index}]"
        model = _validate_model_identity(model, field)
        model_ids.append(model["model_id"])
        if model["engine"] not in engines:
            raise _error(
                "invalid_capabilities", f"{field}.engine",
                "model engine was not declared by the worker",
            )
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
    if version >= 3:
        if engines != ["diffusers_sd15"]:
            raise _error(
                "invalid_capabilities",
                "payload.capabilities.engines",
                "v3 image workers must declare only diffusers_sd15",
            )
        _validate_image_capabilities(capabilities["image"])


def _require_finite_number(
    value: Any,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise _error("invalid_number", field, f"{field} is outside the allowed range")
    return float(value)


def _validate_generation_fields(root_input: dict[str, Any], field: str) -> None:
    prompt = _require_string(
        root_input["prompt"], f"{field}.prompt", max_length=4000,
    )
    if not prompt.strip():
        raise _error("invalid_image_stage", f"{field}.prompt", "prompt is empty")
    _require_string(
        root_input["negative_prompt"],
        f"{field}.negative_prompt",
        allow_empty=True,
        max_length=4000,
    )
    seed = _require_int(root_input["seed"], f"{field}.seed", minimum=0)
    if seed > 2**63 - 1:
        raise _error(
            "invalid_image_stage",
            f"{field}.seed",
            "seed exceeds the signed 64-bit limit",
        )
    width = _require_int(root_input["width"], f"{field}.width", minimum=64)
    height = _require_int(root_input["height"], f"{field}.height", minimum=64)
    if width > 768 or height > 768 or width % 8 or height % 8:
        raise _error(
            "invalid_image_stage",
            field,
            "SD15 dimensions must be multiples of 8 between 64 and 768",
        )
    steps = _require_int(root_input["steps"], f"{field}.steps", minimum=1)
    if steps > 100:
        raise _error(
            "invalid_image_stage", f"{field}.steps", "steps exceeds the hard limit",
        )
    _require_finite_number(
        root_input["guidance_scale"],
        f"{field}.guidance_scale",
        maximum=30.0,
    )
    scheduler = _require_string(
        root_input["scheduler"],
        f"{field}.scheduler",
        allow_empty=True,
        max_length=80,
    )
    if scheduler not in {"", "PNDMScheduler", "DPMSolverMultistepScheduler"}:
        raise _error(
            "invalid_image_stage",
            f"{field}.scheduler",
            "scheduler is unsupported",
        )
    _require_string(
        root_input["artifact_manifest_sha256"],
        f"{field}.artifact_manifest_sha256",
        pattern=_SHA256,
    )


def _optional_number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if value is not None:
        _require_finite_number(value, field, minimum=minimum, maximum=maximum)


def _validate_image_stage_input(
    stage_type: str,
    root_input: dict[str, Any],
    dependencies: dict[str, Any],
    artifact_manifest: Any,
) -> tuple[str, ...]:
    field = "payload.root_input"
    blob_ids: list[str] = []
    if stage_type == "image_generate":
        _require_exact_fields(
            root_input,
            {
                "prompt", "negative_prompt", "seed", "width", "height",
                "steps", "guidance_scale", "scheduler",
                "artifact_manifest_sha256",
            },
            field,
        )
        _validate_generation_fields(root_input, field)
        manifest = _validate_artifact_manifest(
            artifact_manifest,
            "payload.artifact_manifest",
        )
        if manifest["pipeline_kind"] != "sd15_pipeline":
            raise _error(
                "invalid_image_stage",
                "payload.artifact_manifest.pipeline_kind",
                "image_generate requires an SD15 base pipeline",
            )
        if manifest["sha256"] != root_input["artifact_manifest_sha256"]:
            raise _error(
                "artifact_manifest_mismatch",
                "payload.root_input.artifact_manifest_sha256",
                "stage artifact manifest digest does not match",
            )
    elif stage_type == "image_edit":
        _require_exact_fields(
            root_input,
            {
                "prompt", "negative_prompt", "seed", "width", "height",
                "steps", "guidance_scale", "scheduler",
                "artifact_manifest_sha256", "mode", "source_blob",
                "mask_blob", "strength", "instruction",
                "image_guidance_scale", "ip_adapter_scale",
            },
            field,
        )
        _validate_generation_fields(root_input, field)
        mode = root_input["mode"]
        if mode not in {"img2img", "reference", "inpaint", "instruction"}:
            raise _error(
                "invalid_image_stage", f"{field}.mode", "edit mode is unsupported",
            )
        source = _validate_blob_descriptor(root_input["source_blob"], f"{field}.source_blob")
        blob_ids.append(source["blob_id"])
        if source["purpose"] not in {"input_image", "output"}:
            raise _error(
                "invalid_image_stage",
                f"{field}.source_blob.purpose",
                "source blob purpose is invalid",
            )
        manifest = _validate_artifact_manifest(
            artifact_manifest,
            "payload.artifact_manifest",
        )
        if manifest["sha256"] != root_input["artifact_manifest_sha256"]:
            raise _error(
                "artifact_manifest_mismatch",
                "payload.root_input.artifact_manifest_sha256",
                "stage artifact manifest digest does not match",
            )
        expected_pipeline = {
            "img2img": "sd15_pipeline",
            "reference": "sd15_ip_adapter",
            "inpaint": "sd15_inpaint_pipeline",
            "instruction": "sd15_instruction_pipeline",
        }[mode]
        if manifest["pipeline_kind"] != expected_pipeline:
            raise _error(
                "artifact_manifest_mismatch",
                "payload.artifact_manifest.pipeline_kind",
                "edit mode and artifact pipeline kind disagree",
            )
        mask = root_input["mask_blob"]
        if mode == "inpaint":
            validated_mask = _validate_blob_descriptor(mask, f"{field}.mask_blob")
            blob_ids.append(validated_mask["blob_id"])
            if validated_mask["purpose"] != "mask":
                raise _error(
                    "invalid_image_stage",
                    f"{field}.mask_blob.purpose",
                    "inpaint mask blob purpose is invalid",
                )
            if (validated_mask["width"], validated_mask["height"]) != (
                source["width"], source["height"],
            ):
                raise _error(
                    "invalid_image_stage",
                    f"{field}.mask_blob",
                    "mask dimensions must match the source blob",
                )
        elif mask is not None:
            raise _error(
                "invalid_image_stage",
                f"{field}.mask_blob",
                "mask_blob is only valid for inpaint",
            )
        _optional_number(root_input["strength"], f"{field}.strength", minimum=0.05, maximum=1.0)
        instruction = root_input["instruction"]
        if mode == "instruction":
            _require_string(instruction, f"{field}.instruction", max_length=4000)
            if instruction.strip() != root_input["prompt"].strip():
                raise _error(
                    "invalid_image_stage",
                    f"{field}.instruction",
                    "instruction must match prompt",
                )
            _require_finite_number(
                root_input["image_guidance_scale"],
                f"{field}.image_guidance_scale",
                maximum=4.0,
            )
        elif instruction is not None or root_input["image_guidance_scale"] is not None:
            raise _error(
                "invalid_image_stage",
                field,
                "instruction parameters are only valid for instruction mode",
            )
        if mode == "reference":
            _require_finite_number(
                root_input["ip_adapter_scale"],
                f"{field}.ip_adapter_scale",
                maximum=2.0,
            )
        elif root_input["ip_adapter_scale"] is not None:
            raise _error(
                "invalid_image_stage",
                f"{field}.ip_adapter_scale",
                "ip_adapter_scale is only valid for reference mode",
            )
        if mode in {"img2img", "inpaint"} and root_input["strength"] is None:
            raise _error(
                "invalid_image_stage", f"{field}.strength", "strength is required",
            )
        if mode in {"reference", "instruction"} and root_input["strength"] is not None:
            raise _error(
                "invalid_image_stage", f"{field}.strength", "strength is not supported",
            )
    else:
        _require_exact_fields(
            root_input,
            {"images", "minimum_successful", "layout"},
            field,
        )
        if artifact_manifest is not None:
            raise _error(
                "invalid_image_stage",
                "payload.artifact_manifest",
                "image_grid must not carry an artifact manifest",
            )
        images = root_input["images"]
        if not isinstance(images, list) or not 1 <= len(images) <= 16:
            raise _error(
                "invalid_image_stage",
                f"{field}.images",
                "image_grid requires between 1 and 16 images",
            )
        for index, descriptor in enumerate(images):
            image = _validate_blob_descriptor(
                descriptor,
                f"{field}.images[{index}]",
            )
            blob_ids.append(image["blob_id"])
        minimum = _require_int(
            root_input["minimum_successful"],
            f"{field}.minimum_successful",
            minimum=1,
        )
        if minimum > len(images):
            raise _error(
                "invalid_image_stage",
                f"{field}.minimum_successful",
                "minimum_successful exceeds the image count",
            )
        if root_input["layout"] not in {"2x2", "horizontal", "vertical"}:
            raise _error(
                "invalid_image_stage", f"{field}.layout", "grid layout is unsupported",
            )
    if dependencies:
        raise _error(
            "invalid_image_stage",
            "payload.dependencies",
            "v3 image stages use immutable blob descriptors instead of inline dependencies",
        )
    return tuple(sorted(set(blob_ids)))


def _validate_image_stage_output(output: dict[str, Any]) -> tuple[str, ...]:
    fields = set(output)
    blob_ids: list[str] = []
    if fields == {"image", "metrics"}:
        image = _validate_blob_descriptor(output["image"], "payload.output.image")
        blob_ids.append(image["blob_id"])
        if image["purpose"] != "output":
            raise _error(
                "invalid_image_result",
                "payload.output.image.purpose",
                "generated image blob purpose must be output",
            )
    elif fields == {"grid", "images", "metrics"}:
        grid = _validate_blob_descriptor(output["grid"], "payload.output.grid")
        blob_ids.append(grid["blob_id"])
        if grid["purpose"] != "output":
            raise _error(
                "invalid_image_result",
                "payload.output.grid.purpose",
                "grid blob purpose must be output",
            )
        images = output["images"]
        if not isinstance(images, list) or not 1 <= len(images) <= 16:
            raise _error(
                "invalid_image_result",
                "payload.output.images",
                "image result list is invalid",
            )
        for index, descriptor in enumerate(images):
            image = _validate_blob_descriptor(
                descriptor,
                f"payload.output.images[{index}]",
            )
            blob_ids.append(image["blob_id"])
            if image["purpose"] != "output":
                raise _error(
                    "invalid_image_result",
                    f"payload.output.images[{index}].purpose",
                    "grid member blob purpose must be output",
                )
    else:
        raise _error(
            "invalid_image_result",
            "payload.output",
            "image stage output fields do not match a supported schema",
        )
    metrics = _require_object(output["metrics"], "payload.output.metrics")
    allowed_metrics = {"elapsed_seconds", "peak_vram_bytes", "seed", "steps_per_second"}
    if not set(metrics).issubset(allowed_metrics):
        raise _error(
            "invalid_fields",
            "payload.output.metrics",
            "image metrics contains unsupported fields",
        )
    for name in {"elapsed_seconds", "steps_per_second"} & set(metrics):
        _require_finite_number(metrics[name], f"payload.output.metrics.{name}")
    for name in {"peak_vram_bytes", "seed"} & set(metrics):
        number = _require_int(
            metrics[name], f"payload.output.metrics.{name}", minimum=0,
        )
        if name == "seed" and number > 2**63 - 1:
            raise _error(
                "invalid_image_result",
                f"payload.output.metrics.{name}",
                "seed exceeds the signed 64-bit limit",
            )
    return tuple(sorted(set(blob_ids)))


def _validate_metadata(value: Any, *, version: int) -> None:
    metadata = _require_object(value, "payload.metadata")
    if version >= 3:
        allowed = {
            "node_id",
            "provider_kind",
            "elapsed_seconds",
            "peak_vram_bytes",
            "seed",
            "artifact_manifest_sha256",
            "distributed",
        }
        if not set(metadata).issubset(allowed):
            raise _error(
                "invalid_fields", "payload.metadata",
                "image metadata contains unsupported fields",
            )
        if "node_id" in metadata:
            _require_string(metadata["node_id"], "payload.metadata.node_id", pattern=_SAFE_ID)
        if "provider_kind" in metadata:
            _require_string(
                metadata["provider_kind"],
                "payload.metadata.provider_kind",
                pattern=_SAFE_ID,
            )
        if "elapsed_seconds" in metadata:
            _require_finite_number(
                metadata["elapsed_seconds"],
                "payload.metadata.elapsed_seconds",
            )
        for name in {"peak_vram_bytes", "seed"} & set(metadata):
            number = _require_int(
                metadata[name], f"payload.metadata.{name}", minimum=0,
            )
            if name == "seed" and number > 2**63 - 1:
                raise _error(
                    "invalid_image_result",
                    "payload.metadata.seed",
                    "seed exceeds the signed 64-bit limit",
                )
        if "artifact_manifest_sha256" in metadata:
            _require_string(
                metadata["artifact_manifest_sha256"],
                "payload.metadata.artifact_manifest_sha256",
                pattern=_SHA256,
            )
        if "distributed" in metadata:
            _require_bool(metadata["distributed"], "payload.metadata.distributed")
        return
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


def _validate_payload(
    message_type: str,
    payload: dict[str, Any],
    version: int,
) -> None:
    fields = (
        _PAYLOAD_FIELDS_V3 if version >= 3
        else _PAYLOAD_FIELDS_V2 if version >= 2
        else _PAYLOAD_FIELDS
    )
    _require_exact_fields(payload, fields[message_type], "payload")
    if message_type == "hello":
        _require_string(payload["node_id"], "payload.node_id", pattern=_SAFE_ID)
        expected_worker_kind = (
            "pc_diffusion_worker" if version >= 3 else "pc_full_worker"
        )
        if payload["worker_kind"] != expected_worker_kind:
            raise _error(
                "unsupported_worker_kind", "payload.worker_kind",
                f"worker_kind must be {expected_worker_kind}",
            )
        _validate_version_range(
            payload["min_version"], payload["max_version"], prefix="payload"
        )
        if version >= 3 and (
            payload["min_version"] != 3 or payload["max_version"] != 3
        ):
            raise _error(
                "invalid_version_range",
                "payload.min_version",
                "v3 image workers cannot downgrade to the incompatible v2 schema",
            )
        _validate_capabilities(payload["capabilities"], version=version)
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
        if accepted and selected != version:
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
        supported_stage_types = (
            _IMAGE_STAGE_TYPES if version >= 3
            else frozenset({"full_inference", "aggregate"})
        )
        if payload["stage_type"] not in supported_stage_types:
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
        expected_input_digest = stage_input_sha256(
            root_input,
            dependencies,
            payload.get("transfer_plan") if version >= 3 else None,
        )
        if digest != expected_input_digest:
            raise _error(
                "input_digest_mismatch", "payload.input_sha256",
                "stage input digest does not match payload",
            )
        if version >= 3:
            input_blob_ids = _validate_image_stage_input(
                payload["stage_type"],
                root_input,
                dependencies,
                payload["artifact_manifest"],
            )
            _validate_transfer_plan(payload["transfer_plan"], input_blob_ids)
        elif version >= 2:
            _validate_model_identity(
                payload["model_identity"], "payload.model_identity",
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
        if version >= 2:
            retryable = _require_bool(
                payload["retryable"], "payload.retryable",
            )
            if accepted and retryable:
                raise _error(
                    "invalid_acceptance_result", "payload.retryable",
                    "accepted offers cannot be retryable failures",
                )
    elif message_type == "lease_renew":
        _require_int(
            payload["lease_expires_at_ms"], "payload.lease_expires_at_ms",
            minimum=1,
        )
    elif message_type == "stage_result":
        output = _require_object(payload["output"], "payload.output")
        if version >= 3:
            output_blob_ids = _validate_image_stage_output(output)
            _validate_transfer_plan(payload["transfer_plan"], output_blob_ids)
        digest = _require_string(
            payload["output_sha256"], "payload.output_sha256", pattern=_SHA256,
        )
        expected_output_digest = stage_output_sha256(
            output,
            payload.get("transfer_plan") if version >= 3 else None,
        )
        if digest != expected_output_digest:
            raise _error(
                "output_digest_mismatch", "payload.output_sha256",
                "stage output digest does not match output",
            )
        _validate_metadata(payload["metadata"], version=version)
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
    if version < MIN_PROTOCOL_VERSION or version > MAX_PROTOCOL_VERSION:
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
    _validate_payload(message_type, payload, version)
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
    try:
        message_size = len(canonical_message_bytes(message))
    except UnicodeEncodeError as exc:
        raise _error(
            "invalid_encoding", "message", "message must be valid UTF-8"
        ) from exc
    if message_size > MAX_MESSAGE_BYTES:
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
        try:
            raw_size = len(raw.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise _error(
                "invalid_encoding", "message", "message must be valid UTF-8"
            ) from exc
        if raw_size > MAX_MESSAGE_BYTES:
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


def worker_protocol_status(
    adapter_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report schema readiness and optional TC-N2 adapter runtime state."""
    runtime = dict(adapter_status or {})
    return {
        **runtime,
        "protocol": PROTOCOL_NAME,
        "min_version": MIN_PROTOCOL_VERSION,
        "max_version": MAX_PROTOCOL_VERSION,
        "fixture_version": 1,
        "preferred_version": PROTOCOL_VERSION,
        "schema_ready": True,
        "image_schema_version": 3,
        "image_v3_schema_ready": True,
        "image_v3_adapter_connected": bool(
            runtime.get("image_v3_adapter_connected", False)
        ),
        "image_v3_data_plane": runtime.get(
            "image_v3_data_plane", "not_enabled"
        ),
        # TC-N2.4 adds an explicit experimental gate. Physical-device
        # validation and production admission remain fenced.
        "adapter_connected": bool(runtime.get("adapter_connected", False)),
        "transport": runtime.get(
            "transport", "existing_tcp_length_prefixed"
        ),
        "admission_state": runtime.get(
            "admission_state", "n2_4_experiment_disabled"
        ),
    }
