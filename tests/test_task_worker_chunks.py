"""DIST-NEXT-2b 回归：大 payload 的**有序分片**（协议 + 装配 + 切分）。

审计 P0-2 的「超阈值走有序 chunk + checksum」：分片必须有序、有硬上限、可逐片校验，
且不能用来绕过帧预算。
"""

import base64
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_worker_chunks import (  # noqa: E402
    StageChunkAssembler,
    build_stage_chunk,
    plan_stage_payload_chunks,
    stage_chunk_ref,
)
from task_worker_protocol import (  # noqa: E402
    MAX_STAGE_CHUNKS,
    MAX_STAGE_PAYLOAD_BYTES,
    STAGE_CHUNK_BYTES,
    WorkerProtocolError,
    decode_message,
)

CHUNK_ARGS = {
    "workflow_id": "wf_framebudget01",
    "stage_id": "stage_1",
    "attempt_id": "att_framebudget01",
    "lease_id": "lease_framebudget01",
    "lease_epoch": 1,
    "provider_id": "remote_worker_01",
}


def _build(**overrides):
    args = {
        **CHUNK_ARGS,
        "chunk_index": 0,
        "chunk_count": 1,
        "payload": b"y",
        "total_bytes": 1,
        "message_id": "msg_chunk000002",
        "sent_at_ms": 1_700_000_000_000,
    }
    args.update(overrides)
    return build_stage_chunk(**args)


def test_chunk_message_round_trips():
    payload = b"x" * 1024
    message = _build(
        payload=payload, total_bytes=len(payload),
        message_id="msg_chunk000001",
    )

    decoded = decode_message(message.snapshot())

    assert decoded.message_type == "stage_chunk"
    assert decoded.payload["chunk_index"] == 0
    assert decoded.payload["chunk_count"] == 1
    assert decoded.payload["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert base64.b64decode(decoded.payload["payload_b64"]) == payload


def test_protocol_rejects_malformed_chunks():
    """单条就不合法的分片在协议层被拒（顺序语义交给装配器）。"""
    with pytest.raises(WorkerProtocolError) as index_out:
        _build(chunk_index=2, chunk_count=2)
    assert index_out.value.code == "chunk_index_out_of_range"

    with pytest.raises(WorkerProtocolError) as too_many:
        _build(chunk_count=MAX_STAGE_CHUNKS + 1)
    assert too_many.value.code == "chunk_count_out_of_range"

    with pytest.raises(WorkerProtocolError) as total_too_large:
        _build(total_bytes=MAX_STAGE_PAYLOAD_BYTES + 1)
    assert total_too_large.value.code == "stage_payload_too_large"

    with pytest.raises(WorkerProtocolError) as chunk_too_large:
        _build(
            payload=b"z" * (STAGE_CHUNK_BYTES * 2),
            total_bytes=STAGE_CHUNK_BYTES * 2,
        )
    assert chunk_too_large.value.code == "chunk_too_large"

    # 分片是 v3 契约：v2 客户端不得使用（稳定错误码，而不是 KeyError）
    with pytest.raises(WorkerProtocolError) as old_version:
        _build(version=2)
    assert old_version.value.code == "unsupported_message_type"


def test_assembler_orders_by_index_and_requires_every_chunk():
    payload = b"a" * (2 * 1024 + 7)
    plan = plan_stage_payload_chunks(payload, chunk_bytes=1024)
    assert plan.chunk_count == 3
    digests = plan.chunk_digests()
    assembler = StageChunkAssembler()

    # 乱序到达：装配按序号而非到达顺序
    for index in (2, 0):
        assembler.add(
            attempt_id="att_1", chunk_index=index, chunk_count=plan.chunk_count,
            payload=plan.chunks[index], payload_sha256=digests[index],
        )
    assert assembler.received_count("att_1") == 2
    assert assembler.missing_indices("att_1") == [1]
    assert assembler.pending() == {"att_1": 2}

    with pytest.raises(WorkerProtocolError) as incomplete:
        assembler.assemble("att_1")
    assert incomplete.value.code == "incomplete_chunks"
    assert "missing chunk indices: [1]" in str(incomplete.value)

    assembler.add(
        attempt_id="att_1", chunk_index=1, chunk_count=plan.chunk_count,
        payload=plan.chunks[1], payload_sha256=digests[1],
    )
    assert assembler.is_complete("att_1")
    assert assembler.assemble("att_1") == payload
    assert assembler.pending() == {}


def test_assembler_rejects_duplicates_layout_drift_and_bad_digest():
    plan = plan_stage_payload_chunks(b"b" * 100, chunk_bytes=40)
    digests = plan.chunk_digests()

    duplicate = StageChunkAssembler()
    duplicate.add(
        attempt_id="att_2", chunk_index=0, chunk_count=plan.chunk_count,
        payload=plan.chunks[0], payload_sha256=digests[0],
    )
    with pytest.raises(WorkerProtocolError) as dup:
        duplicate.add(
            attempt_id="att_2", chunk_index=0, chunk_count=plan.chunk_count,
            payload=plan.chunks[0], payload_sha256=digests[0],
        )
    assert dup.value.code == "duplicate_chunk"

    bad_digest = StageChunkAssembler()
    with pytest.raises(WorkerProtocolError) as digest:
        bad_digest.add(
            attempt_id="att_3", chunk_index=0, chunk_count=plan.chunk_count,
            payload=plan.chunks[0], payload_sha256="f" * 64,
        )
    assert digest.value.code == "chunk_digest_mismatch"
    # 被拒的分片不得留下半条状态
    assert bad_digest.received_count("att_3") == 0

    drift = StageChunkAssembler()
    drift.add(
        attempt_id="att_4", chunk_index=0, chunk_count=3,
        payload=b"x", payload_sha256=hashlib.sha256(b"x").hexdigest(),
    )
    with pytest.raises(WorkerProtocolError) as layout:
        drift.add(
            attempt_id="att_4", chunk_index=1, chunk_count=5,
            payload=b"y", payload_sha256=hashlib.sha256(b"y").hexdigest(),
        )
    assert layout.value.code == "chunk_layout_mismatch"


def test_assembler_enforces_hard_limits():
    assembler = StageChunkAssembler(
        max_chunks=4, max_chunk_bytes=16, max_total_bytes=32,
    )

    with pytest.raises(WorkerProtocolError) as count:
        assembler.add(
            attempt_id="att_5", chunk_index=0, chunk_count=9,
            payload=b"x", payload_sha256=hashlib.sha256(b"x").hexdigest(),
        )
    assert count.value.code == "invalid_chunk_count"

    with pytest.raises(WorkerProtocolError) as single:
        assembler.add(
            attempt_id="att_5", chunk_index=0, chunk_count=4,
            payload=b"z" * 17, payload_sha256=hashlib.sha256(b"z" * 17).hexdigest(),
        )
    assert single.value.code == "chunk_too_large"

    for index, offset in ((0, 0), (1, 16)):
        chunk = b"q" * 16
        assembler.add(
            attempt_id="att_5", chunk_index=index, chunk_count=4,
            payload=chunk, payload_sha256=hashlib.sha256(chunk).hexdigest(),
        )
        assert offset < 32
    with pytest.raises(WorkerProtocolError) as total:
        chunk = b"q" * 16
        assembler.add(
            attempt_id="att_5", chunk_index=2, chunk_count=4,
            payload=chunk, payload_sha256=hashlib.sha256(chunk).hexdigest(),
        )
    assert total.value.code == "stage_payload_too_large"


def test_discard_resets_accumulation():
    plan = plan_stage_payload_chunks(b"c" * 60, chunk_bytes=30)
    digests = plan.chunk_digests()
    assembler = StageChunkAssembler()
    assembler.add(
        attempt_id="att_6", chunk_index=0, chunk_count=plan.chunk_count,
        payload=plan.chunks[0], payload_sha256=digests[0],
    )

    assembler.discard("att_6")

    assert assembler.received_count("att_6") == 0
    assert assembler.pending() == {}
    with pytest.raises(WorkerProtocolError) as incomplete:
        assembler.assemble("att_6")
    assert incomplete.value.code == "incomplete_chunks"


def test_plan_splits_into_ordered_chunks_and_exposes_a_reference():
    payload = bytes(range(256)) * 3        # 768 字节
    plan = plan_stage_payload_chunks(payload, chunk_bytes=256)

    assert plan.chunk_count == 3
    assert b"".join(plan.chunks) == payload
    assert plan.total_bytes == len(payload)
    assert plan.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert stage_chunk_ref(plan) == {
        "chunk_count": 3,
        "total_bytes": 768,
        "payload_sha256": plan.payload_sha256,
    }


def test_plan_enforces_hard_limits_and_rejects_empty():
    with pytest.raises(WorkerProtocolError) as too_many:
        plan_stage_payload_chunks(
            b"x" * (MAX_STAGE_CHUNKS * 1024 + 1), chunk_bytes=1024,
        )
    assert too_many.value.code == "stage_payload_too_large"

    with pytest.raises(WorkerProtocolError) as empty:
        plan_stage_payload_chunks(b"")
    assert empty.value.code == "invalid_payload"

    # 刚好一片 / 恰好整除都不应多切出空片
    assert plan_stage_payload_chunks(b"x" * 10, chunk_bytes=10).chunk_count == 1
    assert plan_stage_payload_chunks(b"x" * 20, chunk_bytes=10).chunk_count == 2
