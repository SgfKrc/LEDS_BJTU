"""Versioned task-worker messages with strict validation and no transport."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


PROTOCOL_NAME = "qlh.task_worker"
PROTOCOL_VERSION = 3
MIN_PROTOCOL_VERSION = 1
#: ★ 2026-09-20：v3 引入**层段**（`layer_forward`）。v1/v2 只支持整模型
#: （`full_inference`）与聚合（`aggregate`），因此**不具备**参与层流水线的能力。
#: 加版本而不改旧版：v1/v2 客户端仍可协商成功，只是在 `layer_forward` 上会被
#: `unsupported_stage_type` 拒绝（fail-closed，不做静默降级）。
MAX_PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
FULL_WORKER_KINDS = frozenset({"pc_full_worker", "android_full_worker"})

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

# 受支持的推理引擎标识（与 task_provider.ModelIdentity 的校验集保持一致）:
#   pytorch / llama_cpp   本地引擎
#   island                TP 孤岛（路线 A，指纹为端点摘要）
#   external_api          外部推理服务（路线 B，指纹为外部端点摘要）
#   speculative_assisted  投机解码（路线 C-1，本地 draft + 外部 verify）
_SUPPORTED_ENGINES = frozenset({
    "pytorch", "llama_cpp", "island", "external_api", "speculative_assisted",
})
_TEXT_STAGE_TYPES = frozenset({"full_inference", "aggregate", "layer_forward"})

#: ★ 2026-09-20：**层段**（v3）要求的 `stage_offer` 附加字段。
#: 层段节点用 llama.cpp 承一段层：加载**裁层 GGUF**，接收上游 hidden 并注入
#: （`llama_batch.embd`），只算自己负责的层区间。字段设计对齐
#: `src/pipeline_node_contract.py` 的 `PipelineNode(Layer_range/handoff_at)` 与
#: `src/relay_contract.py` 的判据（per-token argmax，不认 bitwise）。
_LAYER_FORWARD_OFFER_FIELDS = {
    #: 本节点负责的**源模型**层区间 [start, end)（闭开）。必须与裁层 GGUF 的
    #: `block_count` 自洽，由接收方核对，防止「声明与实际加载的工件不符」。
    "layer_range",
    #: 上游交接点：hidden 从源模型第 `handoff_at` 层之后跨边界。与
    #: `PipelineNode.handoff_at` 同义，供 `cross_framework` 节点使用。
    "handoff_at",
    #: 上游 hidden 的 wire 摘要（算法 + 值），用于对账与防串话。
    "hidden_sha256",
    #: hidden 的形状：`{"n_tokens": int, "n_embd": int, "dtype": "float32"}`。
    "hidden_spec",
}

#: ★ 2026-09-23：层段 offer 的**可选**字段（出现才允许；缺省 = 旧行为）。
#: * `middle_channel` —— 中间段取 hidden 的通道，允许值见 `_LAYER_FORWARD_MIDDLE_CHANNELS`，
#:   与 Android JNI 能力上报的同名键**同值**：`keep_head_layer_out` = 末层输出
#:   （`output_norm` **之前**，层段接力所需的形态）；`extract_hidden` = `output_norm(H)`（旧默认）。
#: ⚠️ **不能**并入 `_LAYER_FORWARD_OFFER_FIELDS`：那里的字段集合是**精确**校验（缺失即
#: `field_mismatch`）⇒ 会把可选字段变成必填、破坏既有对端（实测踩到）。可选字段在
#: `_validate_payload` 里按「payload 是否真的出现」动态放宽。
_LAYER_FORWARD_OPTIONAL_FIELDS = {"middle_channel"}

#: `middle_channel` 的允许值（协议两侧必须同集合）。
#: * `extract_hidden` —— `llama_get_embeddings_ith` 通道，返回 `output_norm(H)`（旧默认）；
#: * `keep_head_layer_out` —— `layer_inp` 的 `lid == n_layer` 槽位，返回**末层输出**。
_LAYER_FORWARD_MIDDLE_CHANNELS = {"extract_hidden", "keep_head_layer_out"}

#: 层段 `stage_result` 的结果字段：**放在 `output` 或 `metadata` 对象内**，不扩顶层。
#: 原因：`stage_result` 的 payload 里没有 `stage_type`，无法按类型做动态字段校验；
#: 而 `output` / `metadata` 本就是自由对象（仅校验类型与摘要一致性），可安全承载。
_LAYER_FORWARD_RESULT_FIELDS = {
    "hidden_out_sha256",
    "token_argmax",
}

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
#: ★ 2026-09-20（v3）：层段承载。**注意这里不扩任何顶层字段** ——
#: 层段字段**只对 `stage_type == "layer_forward"` 生效**，由 `_validate_payload`
#: 在精确字段校验时按 `stage_type` **动态**并入（否则 `full_inference` 也会被要求
#: 提供 hidden 字段 —— 实测会让既有任务图全线 500）。
#:
#: 层段的**结果**字段（`hidden_out_sha256` / `token_argmax`）刻意**不扩顶层**：
#: `stage_result` 的 payload 里**没有** `stage_type`，无法做动态判断 ⇒ 放进本来就
#: 自由的 `output` / `metadata` 对象内（见 `_LAYER_FORWARD_RESULT_FIELDS` 的说明）。
_PAYLOAD_FIELDS_V3 = {
    **_PAYLOAD_FIELDS_V2,
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


def _validate_capabilities(value: Any, *, version: int) -> None:
    capabilities = _require_object(value, "payload.capabilities")
    expected_fields = {"stage_types", "engines", "models", "max_concurrency"}
    # Optional for backwards compatibility with existing PC workers. Android
    # workers are admitted only when this gate is present and explicitly true;
    # the scheduler performs that role-specific check after hello validation.
    if "resource_gate" in capabilities:
        expected_fields.add("resource_gate")
    if "layer_ranges" in capabilities:
        expected_fields.add("layer_ranges")
    # ★ 2026-09-23：中间段通道能力（可选，向后兼容）—— 让调度侧知道该节点中间段
    #   实际能走哪条通道（值域同 `_LAYER_FORWARD_MIDDLE_CHANNELS`）；缺失 = 未声明。
    if "middle_channel" in capabilities:
        expected_fields.add("middle_channel")
    # ★ 2026-09-23：M-RoPE 模型的位置分量数（1 或 4）—— 供调度侧构造 hidden spec / 判据用。
    if "n_pos_per_embd" in capabilities:
        expected_fields.add("n_pos_per_embd")
    _require_exact_fields(
        capabilities,
        expected_fields,
        "payload.capabilities",
    )
    if "middle_channel" in capabilities:
        channel = _require_string(
            capabilities["middle_channel"], "payload.capabilities.middle_channel",
        )
        if channel not in _LAYER_FORWARD_MIDDLE_CHANNELS:
            raise _error(
                "invalid_capabilities", "payload.capabilities.middle_channel",
                "middle_channel must be one of "
                + ", ".join(sorted(_LAYER_FORWARD_MIDDLE_CHANNELS)),
            )
    if "n_pos_per_embd" in capabilities:
        n_pos = _require_int(
            capabilities["n_pos_per_embd"], "payload.capabilities.n_pos_per_embd", minimum=1,
        )
        if n_pos not in {1, 4}:
            raise _error(
                "invalid_capabilities", "payload.capabilities.n_pos_per_embd",
                "n_pos_per_embd must be 1 or 4",
            )
    stage_types = capabilities["stage_types"]
    if not isinstance(stage_types, list) or not stage_types:
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must be a non-empty list",
        )
    if any(value not in _TEXT_STAGE_TYPES for value in stage_types):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "unsupported stage type",
        )
    if len(stage_types) != len(set(stage_types)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must not contain duplicates",
        )
    if "layer_ranges" in capabilities:
        ranges = capabilities["layer_ranges"]
        if not isinstance(ranges, list) or any(
            not isinstance(item, list) or len(item) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
            or item[0] < 0 or item[1] <= item[0]
            for item in ranges
        ):
            raise _error(
                "invalid_capabilities", "payload.capabilities.layer_ranges",
                "layer_ranges must contain non-empty [start, end) integer ranges",
            )
        if len(ranges) != len({tuple(item) for item in ranges}):
            raise _error(
                "invalid_capabilities", "payload.capabilities.layer_ranges",
                "layer_ranges must not contain duplicates",
            )
    engines = capabilities["engines"]
    if not isinstance(engines, list) or not engines or any(
        value not in _SUPPORTED_ENGINES
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
    if "resource_gate" in capabilities:
        _validate_resource_gate(capabilities["resource_gate"])


def _validate_resource_gate(value: Any) -> None:
    field = "payload.capabilities.resource_gate"
    gate = _require_object(value, field)
    _require_exact_fields(gate, {"admitted", "reason_code"}, field)
    _require_bool(gate["admitted"], f"{field}.admitted")
    reason = _require_string(
        gate["reason_code"], f"{field}.reason_code",
        pattern=_SAFE_CODE, allow_empty=True, max_length=64,
    )
    if gate["admitted"] and reason:
        raise _error(
            "invalid_resource_gate", f"{field}.reason_code",
            "an admitted resource gate cannot carry a rejection reason",
        )
    if not gate["admitted"] and not reason:
        raise _error(
            "invalid_resource_gate", f"{field}.reason_code",
            "a rejected resource gate must carry a reason code",
        )


def _validate_metadata(value: Any, *, version: int) -> None:
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
    required = fields[message_type]
    # ★ 2026-09-20（v3）：层段字段**按 `stage_type` 动态并入**。
    #   只有声明 `layer_forward` 的 `stage_offer` 才要求 hidden 交接字段；
    #   非层段 stage 携带这些字段会被精确字段校验拒绝（既不漏也不多）。
    if (
        message_type == "stage_offer"
        and version >= 3
        and isinstance(payload, Mapping)
        and payload.get("stage_type") == "layer_forward"
    ):
        required = required | _LAYER_FORWARD_OFFER_FIELDS
        # ★ 2026-09-23：**可选**层段字段只在 payload 里**真的出现**时放宽 —— 精确校验是双向的，
        #   提前并入会把它们变成必填、破坏既有对端（实测踩到）。
        required = required | (_LAYER_FORWARD_OPTIONAL_FIELDS & set(payload))
    _require_exact_fields(payload, required, "payload")
    if message_type == "hello":
        _require_string(payload["node_id"], "payload.node_id", pattern=_SAFE_ID)
        if payload["worker_kind"] not in FULL_WORKER_KINDS:
            raise _error(
                "unsupported_worker_kind", "payload.worker_kind",
                "worker_kind is not supported for this protocol version",
            )
        _validate_version_range(
            payload["min_version"], payload["max_version"], prefix="payload"
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
        if payload["stage_type"] not in _TEXT_STAGE_TYPES:
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
        expected_input_digest = stage_input_sha256(root_input, dependencies)
        if digest != expected_input_digest:
            raise _error(
                "input_digest_mismatch", "payload.input_sha256",
                "stage input digest does not match payload",
            )
        # ★ 2026-09-20（v3）：**层段**专项校验。缺字段由上面的精确字段集校验先拦下；
        #   这里校验**值的合法性**，防止「声明与实际加载的工件不符」的静默错配。
        #   ⚠️ 必须限定 `version >= 3`：v1/v2 的字段表里**没有**层段字段，
        #      若不加该判定，v2 客户端声明 `layer_forward` 会在此处 KeyError 而非
        #      返回稳定的 `field_mismatch`（实测）。
        if version >= 3 and payload["stage_type"] == "layer_forward":
            # 字段存在性已由上面的精确字段集校验保证，这里只校验值。
            layer_range = payload["layer_range"]
            if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
                raise _error(
                    "invalid_layer_range", "payload.layer_range",
                    "layer_range must be [start, end)",
                )
            start = _require_int(layer_range[0], "payload.layer_range[0]", minimum=0)
            end = _require_int(layer_range[1], "payload.layer_range[1]", minimum=1)
            if end <= start:
                raise _error(
                    "invalid_layer_range", "payload.layer_range",
                    "layer_range must satisfy end > start",
                )
            _require_int(payload["handoff_at"], "payload.handoff_at", minimum=0)
            _require_string(
                payload["hidden_sha256"], "payload.hidden_sha256", pattern=_SHA256,
            )
            hidden_spec = _require_object(
                payload["hidden_spec"], "payload.hidden_spec"
            )
            _require_exact_fields(
                hidden_spec, {"n_tokens", "n_embd", "dtype"}, "payload.hidden_spec"
            )
            _require_int(hidden_spec["n_tokens"], "payload.hidden_spec.n_tokens", minimum=1)
            _require_int(hidden_spec["n_embd"], "payload.hidden_spec.n_embd", minimum=1)
            dtype = _require_string(hidden_spec["dtype"], "payload.hidden_spec.dtype")
            if dtype not in {"float32", "float16"}:
                raise _error(
                    "unsupported_hidden_dtype", "payload.hidden_spec.dtype",
                    "hidden dtype must be float32 or float16",
                )
            # ★ 2026-09-23：`middle_channel` 可选；一旦出现必须落在允许集合内（fail-closed）。
            #   字段不存在 = `extract_hidden`（旧行为），因此不破坏既有对端。
            if "middle_channel" in payload:
                channel = _require_string(
                    payload["middle_channel"], "payload.middle_channel",
                )
                if channel not in _LAYER_FORWARD_MIDDLE_CHANNELS:
                    raise _error(
                        "unsupported_middle_channel", "payload.middle_channel",
                        "middle_channel must be one of "
                        + ", ".join(sorted(_LAYER_FORWARD_MIDDLE_CHANNELS)),
                    )
        elif "layer_range" in payload or "handoff_at" in payload:
            # 非层段 stage 不得携带层段字段（精确字段集已拦，这里是双保险）
            raise _error(
                "unexpected_layer_fields", "payload",
                "layer fields are only valid for layer_forward",
            )
        if version >= 2:
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
        digest = _require_string(
            payload["output_sha256"], "payload.output_sha256", pattern=_SHA256,
        )
        expected_output_digest = stage_output_sha256(output)
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
