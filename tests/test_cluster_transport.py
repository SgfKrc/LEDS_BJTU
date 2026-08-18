from __future__ import annotations

import pytest

from src.cluster_transport import (
    LEGACY_TCP,
    WSS_443,
    TransportCandidate,
    TransportContractError,
    TransportPolicy,
    TransportSession,
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

