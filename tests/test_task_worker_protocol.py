import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task_worker_protocol import (
    MAX_MESSAGE_BYTES,
    MESSAGE_TYPES,
    PROTOCOL_NAME,
    WorkerProtocolError,
    canonical_message_bytes,
    decode_message,
    negotiate_protocol_version,
    worker_protocol_status,
)


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "task_worker_protocol_v1.json",
)


@pytest.fixture(scope="module")
def golden():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_v1_golden_messages_round_trip_canonically(golden):
    assert golden["fixture_version"] == 1
    assert golden["protocol"] == PROTOCOL_NAME
    decoded = [decode_message(message) for message in golden["messages"]]

    assert {message.message_type for message in decoded} == MESSAGE_TYPES
    assert [message.snapshot() for message in decoded] == golden["messages"]
    for message in decoded:
        assert decode_message(canonical_message_bytes(message)) == message


def test_validated_message_payload_cannot_be_mutated_in_place(golden):
    message = decode_message(golden["messages"][2])
    payload = message.payload
    payload["lease_epoch"] = 99

    assert message.payload["lease_epoch"] == 1
    assert message.snapshot() == golden["messages"][2]


def test_negotiation_selects_highest_common_version():
    assert negotiate_protocol_version(1, 1) == 1
    assert negotiate_protocol_version(
        1, 4, local_min_version=1, local_max_version=2,
    ) == 2


def test_negotiation_rejects_non_overlapping_or_invalid_ranges():
    with pytest.raises(WorkerProtocolError) as unsupported:
        negotiate_protocol_version(3, 4)
    assert unsupported.value.code == "unsupported_protocol_version"

    with pytest.raises(WorkerProtocolError) as invalid:
        negotiate_protocol_version(3, 2)
    assert invalid.value.code == "invalid_version_range"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("protocol", "other.protocol", "unsupported_protocol"),
        ("version", 3, "unsupported_protocol_version"),
        ("message_type", "unknown", "unsupported_message_type"),
        ("message_id", "unsafe", "invalid_identifier"),
    ],
)
def test_envelope_identity_is_strict(golden, field, value, code):
    raw = copy.deepcopy(golden["messages"][0])
    raw[field] = value

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(raw)

    assert captured.value.code == code
    assert captured.value.field == field


def test_unknown_envelope_or_payload_fields_are_rejected(golden):
    envelope = copy.deepcopy(golden["messages"][0])
    envelope["secret"] = "must-not-pass"
    with pytest.raises(WorkerProtocolError) as envelope_error:
        decode_message(envelope)
    assert envelope_error.value.code == "invalid_fields"

    payload = copy.deepcopy(golden["messages"][6])
    payload["payload"]["error_message"] = "raw exception text"
    with pytest.raises(WorkerProtocolError) as payload_error:
        decode_message(payload)
    assert payload_error.value.code == "invalid_fields"


def test_offer_input_digest_detects_tampering(golden):
    offer = copy.deepcopy(golden["messages"][2])
    offer["payload"]["root_input"]["message"] = "tampered"

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(offer)

    assert captured.value.code == "input_digest_mismatch"


def test_result_output_digest_detects_tampering(golden):
    result = copy.deepcopy(golden["messages"][5])
    result["payload"]["output"]["content"] = "tampered"

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(result)

    assert captured.value.code == "output_digest_mismatch"


def test_lease_deadline_must_be_after_message_timestamp(golden):
    renewal = copy.deepcopy(golden["messages"][4])
    renewal["payload"]["lease_expires_at_ms"] = renewal["sent_at_ms"]

    with pytest.raises(WorkerProtocolError) as captured:
        decode_message(renewal)

    assert captured.value.code == "invalid_lease_deadline"


def test_worker_capability_lists_and_concurrency_are_bounded(golden):
    duplicate = copy.deepcopy(golden["messages"][0])
    duplicate["payload"]["capabilities"]["engines"].append("pytorch")
    with pytest.raises(WorkerProtocolError) as duplicate_error:
        decode_message(duplicate)
    assert duplicate_error.value.code == "invalid_capabilities"

    unbounded = copy.deepcopy(golden["messages"][0])
    unbounded["payload"]["capabilities"]["max_concurrency"] = 33
    with pytest.raises(WorkerProtocolError) as concurrency_error:
        decode_message(unbounded)
    assert concurrency_error.value.code == "invalid_capabilities"


def test_hello_ack_fields_cannot_claim_an_invalid_negotiation(golden):
    accepted = copy.deepcopy(golden["messages"][1])
    accepted["payload"]["reason_code"] = "unsupported_protocol_version"
    with pytest.raises(WorkerProtocolError) as accepted_error:
        decode_message(accepted)
    assert accepted_error.value.code == "invalid_negotiation_result"

    rejected = copy.deepcopy(golden["messages"][1])
    rejected["payload"].update({
        "accepted": False,
        "selected_version": 0,
        "reason_code": "unsupported_protocol_version",
    })
    assert decode_message(rejected).payload["accepted"] is False


def test_protocol_status_exposes_n2_transport_without_claiming_stage_readiness():
    status = worker_protocol_status()

    assert status["schema_ready"] is True
    assert status["adapter_connected"] is False
    assert status["transport"] == "existing_tcp_length_prefixed"
    assert status["preferred_version"] == 2
    assert status["admission_state"] == "n2_4_experiment_disabled"


def test_v2_offer_requires_exact_model_identity(golden):
    offer = copy.deepcopy(golden["messages"][2])
    offer["version"] = 2
    offer["payload"]["model_identity"] = copy.deepcopy(
        golden["messages"][0]["payload"]["capabilities"]["models"][0]
    )

    assert decode_message(offer).version == 2

    del offer["payload"]["model_identity"]
    with pytest.raises(WorkerProtocolError) as missing:
        decode_message(offer)
    assert missing.value.code == "invalid_fields"


def test_v2_accept_carries_explicit_retryability(golden):
    accepted = copy.deepcopy(golden["messages"][3])
    accepted["version"] = 2
    accepted["payload"]["retryable"] = False
    assert decode_message(accepted).payload["retryable"] is False

    rejected = copy.deepcopy(accepted)
    rejected["payload"].update({
        "accepted": False,
        "reason_code": "provider_busy",
        "retryable": True,
    })
    assert decode_message(rejected).payload["retryable"] is True

    accepted["payload"]["retryable"] = True
    with pytest.raises(WorkerProtocolError) as invalid:
        decode_message(accepted)
    assert invalid.value.code == "invalid_acceptance_result"


def test_decode_rejects_invalid_utf8_and_oversized_wire_messages():
    with pytest.raises(WorkerProtocolError) as encoding_error:
        decode_message(b"\xff")
    assert encoding_error.value.code == "invalid_encoding"

    with pytest.raises(WorkerProtocolError) as size_error:
        decode_message(b" " * (MAX_MESSAGE_BYTES + 1))
    assert size_error.value.code == "message_too_large"


def test_decode_normalizes_invalid_unicode_to_protocol_error():
    with pytest.raises(WorkerProtocolError) as captured:
        decode_message("\ud800")
    assert captured.value.code == "invalid_encoding"
