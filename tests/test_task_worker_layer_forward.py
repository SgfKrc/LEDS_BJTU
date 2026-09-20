"""层段（`layer_forward`）协议校验 —— 主仓任务协议 v3 新增（2026-09-20）。

## 背景
Android（以及任何 llama.cpp 节点）要用**裁层 GGUF + `llama_batch.embd` 注入**
参与**层流水线**，就必须有一个**层段 stage**。此前 `_TEXT_STAGE_TYPES` 只有
`full_inference`/`aggregate` ⇒ 层段无处声明。

## 设计约束（本文件逐条验证）
1. 层段字段**只对 `stage_type == "layer_forward"` 生效** —— 否则既有任务图
   （走 `full_inference`）会被要求提供 hidden 字段而**全线 500**（实测）。
2. 层段字段**按值校验**，不只是存在性：`layer_range` 必须 `[start, end)` 且
   `end > start`，`hidden_spec` 必须给出正的 `n_tokens`/`n_embd` 与受支持 dtype。
3. **v1/v2 客户端不得**通过声明 `layer_forward` 绕过 —— 缺字段会被拒（fail-closed）。
4. 非层段 stage **不得**携带层段字段（既不漏也不多）。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_worker_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    WorkerProtocolError,
    build_message,
    stage_input_sha256,
)


def _identity():
    return {
        "workflow_id": "wf_layertest0001",
        "stage_id": "candidate_a",
        "attempt_id": "att_layertest0001",
        "lease_id": "lease_layertest0001",
        "lease_epoch": 1,
    }


def _base_inputs():
    root_input = {"blocks": [{"role": "user", "content": "hi"}]}
    dependencies = {}
    return root_input, dependencies


def _offer_payload(**extra):
    root_input, dependencies = _base_inputs()
    payload = {
        **_identity(),
        "request_id": "req_layertest0001",
        "stage_type": "layer_forward",
        "provider_id": "remote_android_worker_01",
        "lease_expires_at_ms": 1_700_000_000_000,
        "root_input": root_input,
        "dependencies": dependencies,
        "input_sha256": stage_input_sha256(root_input, dependencies),
        "model_identity": {
            "model_id": "qwen3-5-2b",
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "cut-k4",
            "sha256": "b" * 64,
        },
    }
    payload.update(extra)
    return payload


def _layer_fields(**over):
    fields = {
        "layer_range": [4, 24],
        "handoff_at": 4,
        "hidden_sha256": "c" * 64,
        "hidden_spec": {"n_tokens": 16, "n_embd": 2048, "dtype": "float32"},
    }
    fields.update(over)
    return fields


def _build(payload, version=PROTOCOL_VERSION):
    return build_message("stage_offer", payload, message_id="msg_layertest0001",
                         sent_at_ms=1000, version=version)


# ---------------------------------------------------------------- 正向

def test_layer_forward_offer_with_all_fields_is_accepted():
    message = _build(_offer_payload(**_layer_fields()))
    assert message.payload["stage_type"] == "layer_forward"
    assert message.payload["layer_range"] == [4, 24]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_layer_forward_accepts_supported_hidden_dtypes(dtype):
    payload = _offer_payload(**_layer_fields(
        hidden_spec={"n_tokens": 1, "n_embd": 2048, "dtype": dtype},
    ))
    assert _build(payload).payload["hidden_spec"]["dtype"] == dtype


# ---------------------------------------------------------------- 缺字段（fail-closed）

@pytest.mark.parametrize(
    "missing", ["layer_range", "handoff_at", "hidden_sha256", "hidden_spec"],
)
def test_layer_forward_requires_every_layer_field(missing):
    fields = _layer_fields()
    del fields[missing]
    with pytest.raises(WorkerProtocolError) as exc:
        _build(_offer_payload(**fields))
    assert exc.value.code == "field_mismatch" or "missing" in str(exc.value)


def test_full_inference_must_not_carry_layer_fields():
    """★ 关键回归：层段字段**只**对 `layer_forward` 生效。

    若错误地做成全局必填/可选，既有任务图（`full_inference`）会全线 500 —— 实测过。
    """
    payload = _offer_payload(stage_type="full_inference", **_layer_fields())
    with pytest.raises(WorkerProtocolError):
        _build(payload)


def test_full_inference_offer_without_layer_fields_still_works():
    """层段引入不得破坏既有整模型路径。"""
    message = _build(_offer_payload(stage_type="full_inference"))
    assert message.payload["stage_type"] == "full_inference"
    assert "layer_range" not in message.payload


# ---------------------------------------------------------------- 值校验

@pytest.mark.parametrize(
    ("bad_range", "reason"),
    [
        ([4], "len"),
        ([24, 4], "end > start"),
        ([4, 4], "end > start"),
        ([-1, 4], "start"),
    ],
)
def test_invalid_layer_range_is_rejected(bad_range, reason):
    payload = _offer_payload(**_layer_fields(layer_range=bad_range))
    with pytest.raises(WorkerProtocolError):
        _build(payload)


def test_unsupported_hidden_dtype_is_rejected():
    payload = _offer_payload(**_layer_fields(
        hidden_spec={"n_tokens": 1, "n_embd": 2048, "dtype": "int8"},
    ))
    with pytest.raises(WorkerProtocolError) as exc:
        _build(payload)
    assert exc.value.code == "unsupported_hidden_dtype"


@pytest.mark.parametrize("field", ["n_tokens", "n_embd"])
def test_non_positive_hidden_shape_is_rejected(field):
    spec = {"n_tokens": 1, "n_embd": 2048, "dtype": "float32"}
    spec[field] = 0
    with pytest.raises(WorkerProtocolError):
        _build(_offer_payload(**_layer_fields(hidden_spec=spec)))


def test_hidden_spec_must_not_carry_extra_fields():
    spec = {"n_tokens": 1, "n_embd": 2048, "dtype": "float32", "extra": 1}
    with pytest.raises(WorkerProtocolError):
        _build(_offer_payload(**_layer_fields(hidden_spec=spec)))


# ---------------------------------------------------------------- 版本边界

def test_layer_forward_requires_v3_so_v2_clients_are_rejected():
    """v2 客户端即便声明 `layer_forward` 也**不得**通过（v2 字段表无层段字段 ⇒ 拒）。"""
    with pytest.raises(WorkerProtocolError) as exc:
        _build(_offer_payload(**_layer_fields()), version=2)
    assert exc.value.code == "invalid_fields"


def test_protocol_version_is_at_least_3_for_layer_forward():
    assert PROTOCOL_VERSION >= 3
