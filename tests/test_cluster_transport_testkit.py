from __future__ import annotations

import pytest

from src.cluster_transport import (
    CONTROL_CHANNEL,
    STREAM_CHANNEL,
    DeterministicClock,
    EventBarrier,
    FakeTransportLink,
    TransportChunkAck,
    TransportContractError,
    TransportEnvelope,
)


def _envelope(
    clock: DeterministicClock,
    *,
    generation: int = 2,
    attempt: str = "attempt-1",
    sequence: int = 0,
) -> TransportEnvelope:
    return TransportEnvelope.from_payload(
        f"payload-{sequence}".encode(),
        request_id="request-1",
        connection_generation=generation,
        attempt_id=attempt,
        channel=STREAM_CHANNEL,
        sequence=sequence,
        deadline_ms=clock.now_ms + 100,
    )


def test_fake_link_requires_explicit_delivery_and_preserves_payload_digest():
    clock = DeterministicClock(epoch_ms=1_000)
    left, right = FakeTransportLink.pair()
    left.open(generation=2, attempt_id="attempt-1", window_bytes=64)
    right.open(generation=2, attempt_id="attempt-1", window_bytes=64)
    envelope = _envelope(clock)

    left.send(envelope, b"payload-0", now_ms=clock.now_ms)
    with pytest.raises(TransportContractError) as error:
        right.receive()
    assert error.value.code == "transport_empty"

    left.deliver_next()
    frame = right.receive(now_ms=clock.now_ms)
    assert frame.payload == b"payload-0"
    assert frame.envelope.payload_digest == envelope.payload_digest

    left.acknowledge(TransportChunkAck("request-1", 2, "attempt-1", 0, len(frame.payload)))
    assert left.snapshot()["window"]["inflight_bytes"] == 0


def test_fake_link_drop_duplicate_and_reorder_are_deterministic():
    clock = DeterministicClock(epoch_ms=1_000)
    left, right = FakeTransportLink.pair()
    left.open(generation=2, attempt_id="attempt-1", window_bytes=128)
    right.open(generation=2, attempt_id="attempt-1", window_bytes=128)
    left.send(_envelope(clock, sequence=0), b"payload-0", now_ms=clock.now_ms)
    left.drop_next()
    with pytest.raises(TransportContractError) as error:
        right.receive()
    assert error.value.code == "transport_empty"

    left.open(generation=3, attempt_id="attempt-2", window_bytes=128)
    right.open(generation=3, attempt_id="attempt-2", window_bytes=128)
    left.send(_envelope(clock, generation=3, attempt="attempt-2", sequence=0), b"payload-0", now_ms=clock.now_ms)
    left.duplicate_next()
    left.deliver_next()
    left.deliver_next()
    assert right.receive(now_ms=clock.now_ms).envelope.sequence == 0
    with pytest.raises(TransportContractError) as error:
        right.receive(now_ms=clock.now_ms)
    assert error.value.code == "sequence_duplicate"

    left.open(generation=4, attempt_id="attempt-3", window_bytes=128)
    right.open(generation=4, attempt_id="attempt-3", window_bytes=128)
    left.send(_envelope(clock, generation=4, attempt="attempt-3", sequence=0), b"payload-0", now_ms=clock.now_ms)
    left.send(_envelope(clock, generation=4, attempt="attempt-3", sequence=1), b"payload-1", now_ms=clock.now_ms)
    left.reorder(0, 1)
    left.deliver_next()
    with pytest.raises(TransportContractError) as error:
        right.receive(now_ms=clock.now_ms)
    assert error.value.code == "sequence_out_of_order"


def test_fake_link_fences_close_generation_deadline_and_payload_tamper():
    clock = DeterministicClock(epoch_ms=1_000)
    left, right = FakeTransportLink.pair()
    left.open(generation=2, attempt_id="attempt-1", window_bytes=64)
    right.open(generation=2, attempt_id="attempt-1", window_bytes=64)
    envelope = _envelope(clock)
    with pytest.raises(TransportContractError) as error:
        left.send(envelope, b"tampered", now_ms=clock.now_ms)
    assert error.value.code == "payload_mismatch"

    stale = TransportEnvelope.from_payload(
        b"payload-0", request_id="request-1", connection_generation=1,
        attempt_id="attempt-1", channel=CONTROL_CHANNEL, sequence=0, deadline_ms=2_000,
    )
    with pytest.raises(TransportContractError) as error:
        left.send(stale, b"payload-0", now_ms=clock.now_ms)
    assert error.value.code == "generation_stale"

    expired = _envelope(clock)
    clock.advance(100)
    with pytest.raises(TransportContractError) as error:
        left.send(expired, b"payload-0", now_ms=clock.now_ms)
    assert error.value.code == "deadline_exceeded"

    left.close()
    with pytest.raises(TransportContractError) as error:
        left.send(_envelope(DeterministicClock(epoch_ms=1_000)), b"payload-0", now_ms=1_000)
    assert error.value.code == "connection_reset"


def test_fake_link_rejects_frame_delivered_after_reconnect_generation_change():
    clock = DeterministicClock(epoch_ms=1_000)
    left, right = FakeTransportLink.pair()
    left.open(generation=2, attempt_id="attempt-1")
    right.open(generation=2, attempt_id="attempt-1")
    left.send(_envelope(clock), b"payload-0", now_ms=clock.now_ms)

    left.open(generation=3, attempt_id="attempt-2")
    right.open(generation=3, attempt_id="attempt-2")
    left.deliver_next()
    with pytest.raises(TransportContractError) as error:
        right.receive()
    assert error.value.code == "generation_stale"


def test_fake_link_rechecks_deadline_and_payload_integrity_at_receive_time():
    clock = DeterministicClock(epoch_ms=1_000)
    left, right = FakeTransportLink.pair()
    left.open(generation=2, attempt_id="attempt-1")
    right.open(generation=2, attempt_id="attempt-1")
    left.send(_envelope(clock), b"payload-0", now_ms=clock.now_ms)
    left.deliver_next()
    clock.advance(100)
    with pytest.raises(TransportContractError) as error:
        right.receive(now_ms=clock.now_ms)
    assert error.value.code == "deadline_exceeded"

    left.open(generation=3, attempt_id="attempt-2")
    right.open(generation=3, attempt_id="attempt-2")
    fresh = _envelope(DeterministicClock(epoch_ms=1_000), generation=3, attempt="attempt-2")
    left.send(fresh, b"payload-0", now_ms=1_000)
    left.tamper_next(b"tampered")
    left.deliver_next()
    with pytest.raises(TransportContractError) as error:
        right.receive(now_ms=1_000)
    assert error.value.code == "payload_mismatch"


def test_deterministic_clock_and_event_barrier_replace_sleep_ordering():
    clock = DeterministicClock(epoch_ms=10)
    barrier = EventBarrier(["registered", "acknowledged"])
    assert barrier.is_released("registered") is False
    with pytest.raises(TransportContractError) as error:
        barrier.require_released("registered")
    assert error.value.code == "barrier_blocked"
    barrier.release("registered")
    barrier.require_released("registered")
    assert clock.advance(25) == 35
    assert clock.now == 0.035
    barrier.release("acknowledged")
    assert barrier.snapshot() == {"acknowledged": True, "registered": True}
