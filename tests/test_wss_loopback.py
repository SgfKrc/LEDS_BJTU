"""NW3.1 local CA/WSS loopback and Transport v2 boundary tests."""
from __future__ import annotations

import asyncio

import pytest

from src.cluster_transport import STREAM_CHANNEL, TransportEnvelope
from src.wss_loopback import LocalWssMaterials, WssLoopbackClient, WssLoopbackError, WssLoopbackServer


def _envelope(*, generation: int = 1, attempt_id: str = "attempt-a", sequence: int = 0) -> TransportEnvelope:
    return TransportEnvelope.from_payload(
        b"echo-payload",
        request_id="request-a",
        connection_generation=generation,
        attempt_id=attempt_id,
        channel=STREAM_CHANNEL,
        sequence=sequence,
        deadline_ms=9_999_999_999_999,
    )


def test_wss_loopback_round_trip_uses_tls_pin_hmac_and_transport_envelope():
    async def scenario():
        materials = LocalWssMaterials.generate()
        async with WssLoopbackServer("loopback-secret", materials=materials) as server:
            client = WssLoopbackClient(
                "loopback-secret", materials, node_id="node-a", generation=1, attempt_id="attempt-a",
            )
            await client.connect(server.uri)
            envelope, payload = await client.exchange(_envelope(), b"echo-payload")
            assert payload == b"echo-payload"
            assert envelope == _envelope()
            await client.close()

    asyncio.run(scenario())


def test_wss_loopback_rejects_bad_secret_and_does_not_expose_endpoint_in_public_material():
    materials = LocalWssMaterials.generate()

    async def scenario():
        async with WssLoopbackServer("correct-secret", materials=materials) as server:
            client = WssLoopbackClient(
                "wrong-secret", materials, node_id="node-a", generation=1, attempt_id="attempt-a",
            )
            with pytest.raises(WssLoopbackError, match="authentication"):
                await client.connect(server.uri)

    asyncio.run(scenario())
    assert "127.0.0.1" not in materials.server_fingerprint_sha256


def test_wss_loopback_fences_old_generation_and_rejects_bad_certificate_pin():
    async def scenario():
        materials = LocalWssMaterials.generate()
        async with WssLoopbackServer("secret", materials=materials) as server:
            first = WssLoopbackClient("secret", materials, node_id="node-a", generation=1, attempt_id="a")
            await first.connect(server.uri)
            second = WssLoopbackClient("secret", materials, node_id="node-a", generation=2, attempt_id="b")
            await second.connect(server.uri)
            with pytest.raises(WssLoopbackError) as error:
                await first.exchange(_envelope(generation=1, attempt_id="a"), b"echo-payload")
            assert error.value.code == "connection_closed"
            await first.close()
            await second.close()

        wrong_materials = LocalWssMaterials.generate()
        async with WssLoopbackServer("secret", materials=materials) as server:
            client = WssLoopbackClient("secret", wrong_materials, node_id="node-b", generation=1, attempt_id="c")
            with pytest.raises(WssLoopbackError) as error:
                await client.connect(server.uri)
            assert error.value.code in {"tls_auth_failed", "tls_fingerprint_mismatch"}

    asyncio.run(scenario())


def test_wss_loopback_requires_loopback_bind_and_wss_uri():
    with pytest.raises(WssLoopbackError) as error:
        WssLoopbackServer("secret", host="0.0.0.0")
    assert error.value.code == "loopback_host_required"

    async def scenario():
        materials = LocalWssMaterials.generate()
        client = WssLoopbackClient("secret", materials, node_id="node-a", generation=1, attempt_id="a")
        with pytest.raises(WssLoopbackError) as error:
            await client.connect("http://127.0.0.1:1")
        assert error.value.code == "uri_invalid"
        with pytest.raises(WssLoopbackError) as error:
            await client.connect("wss://example.invalid:443")
        assert error.value.code == "uri_invalid"

    asyncio.run(scenario())


def test_wss_loopback_rejects_out_of_order_and_duplicate_sequences():
    async def scenario():
        materials = LocalWssMaterials.generate()
        async with WssLoopbackServer("secret", materials=materials) as server:
            client = WssLoopbackClient("secret", materials, node_id="node-a", generation=1, attempt_id="a")
            await client.connect(server.uri)
            with pytest.raises(WssLoopbackError) as error:
                await client.exchange(_envelope(attempt_id="a", sequence=1), b"echo-payload")
            assert error.value.code == "sequence_out_of_order"
            await client.close()

            client = WssLoopbackClient("secret", materials, node_id="node-a", generation=2, attempt_id="b")
            await client.connect(server.uri)
            await client.exchange(_envelope(generation=2, attempt_id="b", sequence=0), b"echo-payload")
            with pytest.raises(WssLoopbackError) as error:
                await client.exchange(_envelope(generation=2, attempt_id="b", sequence=0), b"echo-payload")
            assert error.value.code == "sequence_duplicate"
            await client.close()

    asyncio.run(scenario())


def test_wss_loopback_reconnects_after_client_disconnect_without_reusing_socket_state():
    async def scenario():
        materials = LocalWssMaterials.generate()
        async with WssLoopbackServer("secret", materials=materials) as server:
            first = WssLoopbackClient("secret", materials, node_id="node-a", generation=1, attempt_id="a")
            await first.connect(server.uri)
            await first.exchange(_envelope(attempt_id="a"), b"echo-payload")
            await first.close()
            await asyncio.sleep(0)

            second = WssLoopbackClient("secret", materials, node_id="node-a", generation=1, attempt_id="b")
            await second.connect(server.uri)
            envelope, payload = await second.exchange(_envelope(attempt_id="b"), b"echo-payload")
            assert envelope.attempt_id == "b"
            assert payload == b"echo-payload"
            await second.close()

    asyncio.run(scenario())
