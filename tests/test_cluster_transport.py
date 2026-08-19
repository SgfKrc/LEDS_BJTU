from __future__ import annotations

import pytest

from src.cluster_transport import (
    CONTROL_CHANNEL,
    LEGACY_TCP,
    STREAM_CHANNEL,
    TRANSPORT_FAILURE_MATRIX,
    WSS_443,
    TransportCandidate,
    TransportCircuitBreaker,
    TransportChunkAck,
    TransportContractError,
    TransportEnvelope,
    TransportPolicy,
    TransportSession,
    TransportWindow,
    select_transport,
)


def _candidate(kind: str, **overrides) -> TransportCandidate:
    values = {
        "transport": kind,
        "endpoint": "endpoint",
        "tcp_available": True,
        "tls_available": kind == WSS_443,
        "authenticated": kind == WSS_443,
    }
    values.update(overrides)
    return TransportCandidate(**values)


def test_legacy_tcp_is_default_even_when_wss_is_ready():
    decision = select_transport([_candidate(LEGACY_TCP), _candidate(WSS_443)])
    assert decision.selected_transport == LEGACY_TCP
    assert decision.fallback_reason == ""


def test_wss_requires_explicit_preference_and_tls_authentication():
    policy = TransportPolicy(prefer_wss=True)
    decision = select_transport([_candidate(LEGACY_TCP), _candidate(WSS_443)], policy=policy)
    assert decision.selected_transport == WSS_443
    assert decision.public_view()["considered"][0]["status"] == "available"

    unauthenticated = select_transport(
        [_candidate(LEGACY_TCP), _candidate(WSS_443, authenticated=False)], policy=policy,
    )
    assert unauthenticated.selected_transport == LEGACY_TCP
    assert unauthenticated.fallback_reason == "wss_unavailable:authentication_pending"


def test_plain_tcp_cannot_be_misclassified_as_wss():
    decision = select_transport(
        [_candidate(LEGACY_TCP), _candidate(WSS_443, tls_available=False)],
        policy=TransportPolicy(prefer_wss=True),
    )
    assert decision.selected_transport == LEGACY_TCP
    assert decision.fallback_reason == "wss_unavailable:tls_unavailable"


def test_expired_wss_probe_falls_back_to_legacy():
    decision = select_transport(
        [_candidate(LEGACY_TCP), _candidate(WSS_443, expires_at=10)],
        policy=TransportPolicy(prefer_wss=True),
        now=10,
    )
    assert decision.selected_transport == LEGACY_TCP
    assert decision.fallback_reason == "wss_unavailable:probe_expired"


def test_candidate_from_preflight_requires_explicit_tls_probe():
    candidate = TransportCandidate.from_probe(
        WSS_443,
        "endpoint",
        {"tcp": {"available": True}, "tls_probe": {"status": "tls_failed"}},
        authenticated=True,
    )
    assert candidate.tcp_available is True
    assert candidate.tls_available is False


def test_duplicate_candidates_are_rejected():
    with pytest.raises(TransportContractError) as error:
        select_transport([_candidate(LEGACY_TCP), _candidate(LEGACY_TCP)])
    assert error.value.code == "transport_duplicate"


def test_wss_failure_falls_back_once_without_duplicate_attempt():
    session = TransportSession("worker")
    lease = session.begin(generation=1, attempt_id="attempt-1", transport=WSS_443)
    fallback = session.fallback(generation=1, attempt_id="attempt-1", reason="tls_reset")
    assert fallback.attempt_id == lease.attempt_id
    assert fallback.transport == LEGACY_TCP
    assert fallback.lease_token == lease.lease_token
    with pytest.raises(TransportContractError) as error:
        session.fallback(generation=1, attempt_id="attempt-1", reason="retry")
    assert error.value.code == "fallback_invalid"
    session.complete(generation=1, attempt_id="attempt-1", transport=LEGACY_TCP)
    with pytest.raises(TransportContractError) as error:
        session.complete(generation=1, attempt_id="attempt-1", transport=WSS_443)
    assert error.value.code == "attempt_fenced"


def test_new_generation_fences_old_attempt_and_rejects_reuse():
    session = TransportSession("worker")
    session.begin(generation=2, attempt_id="old", transport=LEGACY_TCP)
    session.begin(generation=3, attempt_id="new", transport=LEGACY_TCP)
    with pytest.raises(TransportContractError) as error:
        session.complete(generation=2, attempt_id="old", transport=LEGACY_TCP)
    assert error.value.code == "generation_stale"
    with pytest.raises(TransportContractError) as error:
        session.begin(generation=3, attempt_id="old", transport=LEGACY_TCP)
    assert error.value.code == "attempt_duplicate"


def test_snapshot_does_not_expose_endpoint_or_lease_token():
    session = TransportSession("worker")
    lease = session.begin(generation=1, attempt_id="attempt-1", transport=WSS_443)
    snapshot = session.snapshot()
    assert "endpoint" not in str(snapshot)
    assert lease.lease_token not in str(snapshot)
    assert snapshot["attempts"]["attempt-1"]["transport"] == WSS_443


def test_transport_v2_envelope_is_canonical_and_payload_free():
    envelope = TransportEnvelope.from_payload(
        b"hello",
        request_id="req-1",
        connection_generation=4,
        attempt_id="attempt-1",
        channel=STREAM_CHANNEL,
        sequence=2,
        deadline_ms=2_000,
    )
    encoded = envelope.encode()
    decoded = TransportEnvelope.decode(encoded)
    assert decoded == envelope
    assert b"hello" not in encoded
    assert decoded.payload_size == 5
    assert decoded.is_expired(now_ms=1_999) is False
    assert decoded.is_expired(now_ms=2_000) is True


def test_transport_v2_envelope_rejects_unknown_fields_and_bad_digest():
    envelope = TransportEnvelope.from_payload(
        b"x",
        request_id="req-1",
        connection_generation=0,
        attempt_id="attempt-1",
        channel=CONTROL_CHANNEL,
        sequence=0,
        deadline_ms=2_000,
    )
    value = envelope.to_dict()
    value["extra"] = True
    with pytest.raises(TransportContractError) as error:
        TransportEnvelope.decode(__import__("json").dumps(value))
    assert error.value.code == "envelope_fields_invalid"
    with pytest.raises(TransportContractError) as error:
        TransportEnvelope(
            **{**envelope.to_dict(), "payload_digest": "not-a-digest"},
        )
    assert error.value.code == "payload_digest_invalid"


def test_transport_window_enforces_backpressure_and_idempotent_duplicate_detection():
    window = TransportWindow(10)
    window.reserve(
        request_id="req-1", connection_generation=1, attempt_id="attempt-1", sequence=0, payload_size=6,
    )
    with pytest.raises(TransportContractError) as error:
        window.reserve(
            request_id="req-1", connection_generation=1, attempt_id="attempt-1", sequence=1, payload_size=5,
        )
    assert error.value.code == "window_exhausted"
    ack = TransportChunkAck("req-1", 1, "attempt-1", 0, 6)
    window.acknowledge(ack)
    assert window.snapshot()["inflight_bytes"] == 0
    with pytest.raises(TransportContractError) as error:
        window.acknowledge(ack)
    assert error.value.code == "sequence_duplicate"
    with pytest.raises(TransportContractError) as error:
        window.acknowledge(TransportChunkAck("req-1", 1, "attempt-1", 9, 1))
    assert error.value.code == "sequence_out_of_order"
    window.reserve(
        request_id="req-1", connection_generation=1, attempt_id="attempt-1", sequence=2, payload_size=2,
    )
    with pytest.raises(TransportContractError) as error:
        window.acknowledge(TransportChunkAck("other-request", 1, "attempt-1", 2, 2))
    assert error.value.code == "ack_context_mismatch"


def test_transport_failure_matrix_and_circuit_breaker_are_deterministic():
    assert set(TRANSPORT_FAILURE_MATRIX) == {
        "connection_timeout", "tls_auth_failed", "connection_reset",
        "sequence_duplicate", "sequence_out_of_order", "window_exhausted",
        "ack_size_mismatch", "ack_context_mismatch", "payload_mismatch", "deadline_exceeded", "generation_stale", "attempt_fenced", "circuit_open",
    }
    breaker = TransportCircuitBreaker(failure_threshold=2, cooldown_seconds=5)
    assert breaker.allow(now=0) is True
    breaker.record("connection_timeout", now=0)
    breaker.record("connection_reset", now=1)
    assert breaker.allow(now=1) is False
    assert breaker.allow(now=6) is True
    breaker.success()
    assert breaker.snapshot() == {"state": "closed", "failures": 0}
    with pytest.raises(TransportContractError) as error:
        breaker.record("not-a-code", now=7)
    assert error.value.code == "failure_unknown"
