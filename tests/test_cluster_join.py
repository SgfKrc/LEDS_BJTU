from __future__ import annotations

import base64
import json
import sys
import threading

import pytest

sys.path.insert(0, "src")

from cluster_join import (
    JoinContractError,
    JoinGrantLedger,
    build_join_request,
    decode_join_grant,
    encode_join_grant,
    generate_join_keypair,
    issue_join_grant,
    verify_and_consume_join_grant,
)


def _keys():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization

    private = Ed25519PrivateKey.generate()
    return (
        private,
        base64.urlsafe_b64encode(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode().rstrip("="),
    )


def test_issue_encode_verify_and_consume_client_only_grant(tmp_path):
    target = generate_join_keypair()
    issuer_private, issuer_public = _keys()
    request = build_join_request(
        master_endpoint="[FD7A:115C:A1E0::1]:8888",
        cluster_id="cluster-local",
        target_node_id="tablet-01",
        target_public_key=target.public_key,
        requested_at=1_000,
    )
    grant = issue_join_grant(
        request,
        issuer_key_id="owner-20260819",
        issuer_private_key=issuer_private,
        auth_verified=True,
        now=1_005,
    )
    encoded = encode_join_grant(grant)
    assert encoded.startswith("qlhjoin1.")
    assert "cluster_secret" not in encoded
    assert "otpauth" not in encoded
    assert target.private_key not in encoded
    ledger = JoinGrantLedger(tmp_path / "join.sqlite3")
    verified = verify_and_consume_join_grant(
        encoded,
        issuer_public_key=issuer_public,
        expected_request=request,
        ledger=ledger,
        now=1_006,
    )
    assert verified["role"] == "client"
    assert verified["master_endpoint"] == "[fd7a:115c:a1e0::1]:8888"
    with pytest.raises(JoinContractError, match="already consumed") as error:
        verify_and_consume_join_grant(
            encoded,
            issuer_public_key=issuer_public,
            expected_request=request,
            ledger=ledger,
            now=1_006,
        )
    assert error.value.code == "nonce_replayed"


def test_nonce_replay_survives_ledger_restart(tmp_path):
    target = generate_join_keypair()
    issuer_private, issuer_public = _keys()
    request = build_join_request(
        master_endpoint="100.64.1.2:8888",
        cluster_id="cluster-a",
        target_node_id="client-a",
        target_public_key=target.public_key,
        requested_at=2_000,
    )
    encoded = encode_join_grant(
        issue_join_grant(
            request,
            issuer_key_id="owner",
            issuer_private_key=issuer_private,
            auth_verified=True,
            now=2_001,
        )
    )
    path = tmp_path / "restart.sqlite3"
    verify_and_consume_join_grant(
        encoded,
        issuer_public_key=issuer_public,
        expected_request=request,
        ledger=JoinGrantLedger(path),
        now=2_002,
    )
    with pytest.raises(JoinContractError) as error:
        verify_and_consume_join_grant(
            encoded,
            issuer_public_key=issuer_public,
            expected_request=request,
            ledger=JoinGrantLedger(path),
            now=2_003,
        )
    assert error.value.code == "nonce_replayed"


def test_fail_closed_for_auth_expiry_tamper_wrong_target_and_wrong_issuer(tmp_path):
    target = generate_join_keypair()
    other_target = generate_join_keypair()
    issuer_private, issuer_public = _keys()
    other_private, other_public = _keys()
    request = build_join_request(
        master_endpoint="127.0.0.1:8888",
        cluster_id="cluster-a",
        target_node_id="client-a",
        target_public_key=target.public_key,
        requested_at=3_000,
    )
    with pytest.raises(JoinContractError) as auth_error:
        issue_join_grant(
            request,
            issuer_key_id="owner",
            issuer_private_key=issuer_private,
            auth_verified=False,
            now=3_001,
        )
    assert auth_error.value.code == "auth_required"
    grant = issue_join_grant(
        request,
        issuer_key_id="owner",
        issuer_private_key=issuer_private,
        auth_verified=True,
        now=3_001,
        ttl_seconds=60,
    )
    encoded = encode_join_grant(grant)
    with pytest.raises(JoinContractError) as signature_error:
        verify_and_consume_join_grant(
            encoded,
            issuer_public_key=other_public,
            expected_request=request,
            ledger=JoinGrantLedger(tmp_path / "bad.sqlite3"),
            now=3_002,
        )
    assert signature_error.value.code == "signature_invalid"
    wrong_request = dict(request, target_public_key=other_target.public_key)
    with pytest.raises(JoinContractError) as binding_error:
        verify_and_consume_join_grant(
            encoded,
            issuer_public_key=issuer_public,
            expected_request=wrong_request,
            ledger=JoinGrantLedger(tmp_path / "wrong-target.sqlite3"),
            now=3_002,
        )
    assert binding_error.value.code == "request_mismatch"
    with pytest.raises(JoinContractError) as expiry_error:
        verify_and_consume_join_grant(
            encoded,
            issuer_public_key=issuer_public,
            expected_request=request,
            ledger=JoinGrantLedger(tmp_path / "expired.sqlite3"),
            now=3_062,
        )
    assert expiry_error.value.code == "grant_expired"


def test_grant_payload_cannot_be_escalated_or_malformed():
    target = generate_join_keypair()
    issuer_private, _ = _keys()
    request = build_join_request(
        master_endpoint="100.64.1.2:8888",
        cluster_id="cluster-a",
        target_node_id="client-a",
        target_public_key=target.public_key,
    )
    grant = issue_join_grant(
        request,
        issuer_key_id="owner",
        issuer_private_key=issuer_private,
        auth_verified=True,
    )
    forged = json.loads(json.dumps(grant))
    forged["payload"]["role"] = "master"
    with pytest.raises(JoinContractError) as error:
        encode_join_grant(forged)
    assert error.value.code == "invalid_grant" or error.value.code == "role_escalation"
    with pytest.raises(JoinContractError) as endpoint_error:
        build_join_request(
            master_endpoint="https://user:password@example.com:443/path",
            cluster_id="cluster-a",
            target_node_id="client-a",
            target_public_key=target.public_key,
        )
    assert endpoint_error.value.code == "invalid_endpoint"


def test_malformed_qr_payload_is_a_stable_join_error():
    with pytest.raises(JoinContractError) as error:
        decode_join_grant("qlhjoin1.bm90LWpzb24.AA")
    assert error.value.code == "invalid_grant"


def test_concurrent_nonce_consumption_has_one_winner(tmp_path):
    target = generate_join_keypair()
    issuer_private, issuer_public = _keys()
    request = build_join_request(
        master_endpoint="100.64.1.2:8888",
        cluster_id="cluster-a",
        target_node_id="client-a",
        target_public_key=target.public_key,
        requested_at=4_000,
    )
    encoded = encode_join_grant(
        issue_join_grant(
            request,
            issuer_key_id="owner",
            issuer_private_key=issuer_private,
            auth_verified=True,
            now=4_001,
        )
    )
    ledger = JoinGrantLedger(tmp_path / "concurrent.sqlite3")
    barrier = threading.Barrier(3)
    outcomes = []

    def consume() -> None:
        barrier.wait(timeout=5)
        try:
            verify_and_consume_join_grant(
                encoded,
                issuer_public_key=issuer_public,
                expected_request=request,
                ledger=ledger,
                now=4_002,
            )
            outcomes.append("consumed")
        except JoinContractError as exc:
            outcomes.append(exc.code)

    workers = [threading.Thread(target=consume), threading.Thread(target=consume)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(outcomes) == ["consumed", "nonce_replayed"]
