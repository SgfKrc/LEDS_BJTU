"""Cross-platform task-worker outer TCP framing contract."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_worker_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    build_message as build_worker_message,
    canonical_message_bytes,
    decode_message,
    stage_output_sha256,
)
from tcp_comm import (  # noqa: E402
    MessageType,
    build_message as build_tcp_message,
    parse_message,
)


def _hello():
    return build_worker_message(
        "hello",
        {
            "node_id": "android_12345678",
            "worker_kind": "android_full_worker",
            "min_version": PROTOCOL_VERSION,
            "max_version": PROTOCOL_VERSION,
            "capabilities": {
                "stage_types": ["full_inference"],
                "engines": ["llama_cpp"],
                "models": [],
                "max_concurrency": 1,
            },
        },
        message_id="msg_hello_12345678",
        sent_at_ms=1_700_000_000_000,
        version=PROTOCOL_VERSION,
    )


def test_existing_pc_length_prefix_matches_android_outer_frame():
    hello = _hello()
    packet = build_tcp_message(MessageType.TASK_WORKER, hello.snapshot())

    declared = struct.unpack(">I", packet[:4])[0]
    assert declared == len(packet[4:])
    outer = parse_message(packet[4:])
    assert outer == {
        "type": "task_worker",
        "format": "json",
        "data": hello.snapshot(),
    }

    decoded = decode_message(outer["data"])
    assert canonical_message_bytes(decoded) == canonical_message_bytes(hello)


def test_outer_frame_payload_is_utf8_and_inner_protocol_remains_strict():
    hello = _hello()
    packet = build_tcp_message(MessageType.TASK_WORKER, hello.snapshot())

    assert packet[4:].decode("utf-8")
    outer = parse_message(packet[4:])
    malformed = dict(outer["data"])
    malformed["payload"] = dict(malformed["payload"], unexpected=True)

    try:
        decode_message(malformed)
    except Exception as error:  # noqa: BLE001
        assert getattr(error, "code", "") == "invalid_fields"
    else:
        raise AssertionError("inner task-worker schema must reject unknown fields")


def test_pc_outer_frame_preserves_android_utf8_result_payload():
    output = {"text": "跨平台 UTF-8"}
    result = build_worker_message(
        "stage_result",
        {
            "workflow_id": "wf_12345678",
            "stage_id": "stage_1",
            "attempt_id": "att_12345678",
            "lease_id": "lease_12345678",
            "lease_epoch": 1,
            "provider_id": "android_12345678",
            "output": output,
            "output_sha256": stage_output_sha256(output),
            "metadata": {},
        },
        message_id="msg_result_12345678",
        sent_at_ms=1_700_000_000_002,
        version=PROTOCOL_VERSION,
    )
    packet = build_tcp_message(MessageType.TASK_WORKER, result.snapshot())

    outer = parse_message(packet[4:])
    decoded = decode_message(outer["data"])
    assert decoded.payload["output"] == output
    assert canonical_message_bytes(decoded) == canonical_message_bytes(result)
