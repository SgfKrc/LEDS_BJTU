import threading
import time
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.worker_adapter import (  # noqa: E402
    DiffusionCoordinatorControlPlane,
    DiffusionExecutionResult,
    DiffusionWorkerAdapter,
    RemoteDiffusionProvider,
    remote_diffusion_provider_id,
)
from task_worker_protocol import (  # noqa: E402
    WorkerProtocolError,
    build_message,
    canonical_sha256,
    stage_input_sha256,
)
from task_provider import (  # noqa: E402
    ProviderUnavailable,
    StageAttempt,
    StageRequest,
)


def _manifest():
    value = {
        "artifact_id": "artifact_sd15",
        "pipeline_kind": "sd15_pipeline",
        "revision": "revision_20260807",
        "components": [{
            "artifact_id": "base_unet",
            "artifact_kind": "unet",
            "sha256": "b" * 64,
        }],
    }
    return {**value, "sha256": canonical_sha256(value)}


def _capabilities():
    return {
        "stage_types": ["image_generate", "image_edit", "image_grid"],
        "engines": ["diffusers_sd15"],
        "models": [],
        "max_concurrency": 1,
        "image": {
            "pipeline_kinds": ["sd15_pipeline"],
            "dtypes": ["float16"],
            "max_width": 768,
            "max_height": 768,
            "max_pixels": 768 * 768,
            "max_batch": 1,
            "supports_controlnet": False,
            "supports_step_cancel": True,
            "artifact_manifests": [_manifest()],
        },
    }


def _offer(
    *,
    attempt_id="att_diffworker01",
    lease_expires_at_ms=2_000,
    node_id="worker_gpu_1",
    sent_at_ms=1_000,
):
    manifest = _manifest()
    root_input = {
        "prompt": "a mountain cabin",
        "negative_prompt": "",
        "seed": 19950101,
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance_scale": 7.5,
        "scheduler": "PNDMScheduler",
        "artifact_manifest_sha256": manifest["sha256"],
    }
    plan = {"base_url": None, "downloads": []}
    return build_message(
        "stage_offer",
        {
            "workflow_id": "wf_diffworker01",
            "stage_id": "image_stage_1",
            "attempt_id": attempt_id,
            "lease_id": "lease_diffworker01",
            "lease_epoch": 1,
            "request_id": "request_diffworker01",
            "stage_type": "image_generate",
            "provider_id": remote_diffusion_provider_id(node_id),
            "lease_expires_at_ms": lease_expires_at_ms,
            "root_input": root_input,
            "dependencies": {},
            "input_sha256": stage_input_sha256(root_input, {}, plan),
            "artifact_manifest": manifest,
            "transfer_plan": plan,
        },
        message_id=f"msg_offer_{attempt_id}",
        sent_at_ms=sent_at_ms,
        version=3,
    )


def _result():
    output = {
        "image": {
            "blob_id": "img_1234567890abcdef",
            "sha256": "c" * 64,
            "size_bytes": 128,
            "content_type": "image/png",
            "width": 512,
            "height": 512,
            "purpose": "output",
        },
        "metrics": {"elapsed_seconds": 1.0, "seed": 19950101},
    }
    return DiffusionExecutionResult(
        output=output,
        metadata={
            "node_id": "worker_gpu_1",
            "provider_kind": "pc_diffusion_worker",
            "elapsed_seconds": 1.0,
            "seed": 19950101,
            "artifact_manifest_sha256": _manifest()["sha256"],
            "distributed": True,
        },
        transfer_plan={
            "base_url": "http://100.64.0.10:8000",
            "downloads": [{
                "blob_id": "img_1234567890abcdef",
                "lease_id": "bls_0000000000000001",
                "grant": "a" * 32 + "." + "b" * 43,
            }],
        },
    )


def _stage_request(provider_id):
    manifest = _manifest()
    return StageRequest(
        workflow_id="wf_diffprovider01",
        request_id="request_diffprovider01",
        stage_id="image_stage_1",
        stage_type="image_generate",
        provider_id=provider_id,
        dependencies={},
        root_input={
            "prompt": "a mountain cabin",
            "negative_prompt": "",
            "seed": 19950101,
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "PNDMScheduler",
            "artifact_manifest_sha256": manifest["sha256"],
        },
        runtime_context={"diffusion_artifact_manifest": manifest},
    )


def _connected_worker(executor, sent, *, now=1.0):
    worker = DiffusionWorkerAdapter(
        node_id="worker_gpu_1",
        capabilities=_capabilities(),
        executor=executor,
        send_message=sent.append,
        clock=lambda: now,
    )
    coordinator = DiffusionCoordinatorControlPlane(clock=lambda: now)
    hello = worker.begin_hello(sent_at_ms=1_000)
    assert hello is not None
    ack = coordinator.receive_hello(
        "worker_gpu_1", hello.snapshot(), coordinator_node_id="master", sent_at_ms=1_001,
    )
    worker.receive_hello_ack(ack.snapshot())
    return worker, coordinator


def _wait_for(sent, message_type):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if any(message.message_type == message_type for message in sent):
            return
        time.sleep(0.005)
    pytest.fail(f"timed out waiting for {message_type}")


def test_v3_handshake_records_capabilities_and_executes_fake_image_stage():
    sent = []
    calls = []

    def execute(payload, cancel_event):
        calls.append(payload["attempt_id"])
        assert not cancel_event.is_set()
        return _result()

    worker, coordinator = _connected_worker(execute, sent)
    accepted = worker.receive_offer(_offer().snapshot())
    assert accepted.payload["accepted"] is True
    _wait_for(sent, "stage_result")

    assert calls == ["att_diffworker01"]
    assert [message.message_type for message in sent] == ["stage_accept", "stage_result"]
    assert sent[-1].payload["output"] == _result().output
    assert coordinator.worker_snapshot("worker_gpu_1")["healthy"] is True
    assert worker.status()["adapter_connected"] is False


def test_lease_renewal_is_idempotent_and_fences_expired_execution():
    sent = []
    entered = threading.Event()
    clock = {"now": 1.0}
    monotonic = {"now": 1.0}

    def execute(payload, cancel_event):
        entered.set()
        assert cancel_event.wait(2.0)
        return _result()

    worker = DiffusionWorkerAdapter(
        node_id="worker_gpu_1",
        capabilities=_capabilities(),
        executor=execute,
        send_message=sent.append,
        clock=lambda: clock["now"],
        monotonic=lambda: monotonic["now"],
    )
    coordinator = DiffusionCoordinatorControlPlane(clock=lambda: clock["now"])
    hello = worker.begin_hello(sent_at_ms=1_000)
    assert hello is not None
    worker.receive_hello_ack(coordinator.receive_hello(
        "worker_gpu_1", hello.snapshot(), coordinator_node_id="master", sent_at_ms=1_001,
    ).snapshot())
    offer = _offer(lease_expires_at_ms=1_050)
    worker.receive_offer(offer.snapshot())
    assert entered.wait(2.0)
    renewal = build_message(
        "lease_renew",
        {
            "workflow_id": offer.payload["workflow_id"],
            "stage_id": offer.payload["stage_id"],
            "attempt_id": offer.payload["attempt_id"],
            "lease_id": offer.payload["lease_id"],
            "lease_epoch": offer.payload["lease_epoch"],
            "lease_expires_at_ms": 2_000,
        },
        message_id="msg_renew_diffworker01",
        sent_at_ms=1_010,
        version=3,
    )
    worker.receive_lease_renew(renewal.snapshot())
    worker.receive_lease_renew(renewal.snapshot())
    monotonic["now"] = 2.1
    _wait_for(sent, "stage_error")

    assert sent[-1].payload["error_code"] == "lease_expired"
    with pytest.raises(WorkerProtocolError) as stale:
        worker.receive_lease_renew(build_message(
            "lease_renew",
            {
                "workflow_id": offer.payload["workflow_id"],
                "stage_id": offer.payload["stage_id"],
                "attempt_id": offer.payload["attempt_id"],
                "lease_id": offer.payload["lease_id"],
                "lease_epoch": offer.payload["lease_epoch"],
                "lease_expires_at_ms": 3_000,
            },
            message_id="msg_renew_diffworker02",
            sent_at_ms=2_100,
            version=3,
        ).snapshot())
    assert stale.value.code == "unknown_attempt"


def test_remote_diffusion_provider_uses_v3_worker_and_does_not_expose_grants():
    outbound = []
    ingested = []
    holder = {}

    def execute(payload, cancel_event):
        assert payload["transfer_plan"] == {"base_url": None, "downloads": []}
        assert not cancel_event.is_set()
        return _result()

    worker = DiffusionWorkerAdapter(
        node_id="worker_gpu_1",
        capabilities=_capabilities(),
        executor=execute,
        send_message=lambda message: holder["provider"].handle_message(message.snapshot()),
    )
    coordinator = DiffusionCoordinatorControlPlane()
    hello = worker.begin_hello()
    assert hello is not None
    worker.receive_hello_ack(coordinator.receive_hello(
        "worker_gpu_1", hello.snapshot(), coordinator_node_id="master",
    ).snapshot())

    def ingest(attempt, output, transfer_plan):
        ingested.append((attempt.attempt_id, dict(output), dict(transfer_plan)))
        return {
            "image": {
                **output["image"],
                "blob_id": "img_local1234567890",
            },
            "metrics": dict(output["metrics"]),
        }

    def send_to_worker(message):
        outbound.append(message)
        if message.message_type == "stage_offer":
            worker.receive_offer(message.snapshot())
        elif message.message_type == "lease_renew":
            worker.receive_lease_renew(message.snapshot())
        elif message.message_type == "stage_cancel":
            worker.receive_cancel(message.snapshot())
        else:
            pytest.fail(f"unexpected provider message: {message.message_type}")

    provider = RemoteDiffusionProvider(
        node_id="worker_gpu_1",
        peer_snapshot=lambda: coordinator.worker_snapshot("worker_gpu_1"),
        send_message=send_to_worker,
        result_ingestor=ingest,
        dispatch_enabled=True,
    )
    holder["provider"] = provider
    request = _stage_request(provider.provider_id)
    reservation = provider.reserve(request)
    attempt = StageAttempt(
        attempt_id="att_diffprovider01",
        request=request,
        provider_id=provider.provider_id,
        lease_id="lease_diffprovider01",
        lease_epoch=1,
        lease_expires_at=time.time() + 5.0,
    )
    result = provider.execute(attempt, reservation, threading.Event())

    assert [message.message_type for message in outbound] == ["stage_offer"]
    assert outbound[0].version == 3
    assert result.output["image"]["blob_id"] == "img_local1234567890"
    assert result.metadata == _result().metadata
    assert len(ingested) == 1
    assert ingested[0][2]["downloads"][0]["grant"]
    assert "grant" not in result.metadata
    provider.release(reservation.reservation_id)
    assert provider.inspect().active_reservations == 0


def test_remote_diffusion_provider_requires_explicit_dispatch_and_manifest_match():
    coordinator = DiffusionCoordinatorControlPlane()
    hello = build_message(
        "hello",
        {
            "node_id": "worker_gpu_1",
            "worker_kind": "pc_diffusion_worker",
            "min_version": 3,
            "max_version": 3,
            "capabilities": _capabilities(),
        },
        message_id="msg_hello_diffprovider01",
        sent_at_ms=1_000,
        version=3,
    )
    coordinator.receive_hello(
        "worker_gpu_1", hello.snapshot(), coordinator_node_id="master", sent_at_ms=1_001,
    )
    provider = RemoteDiffusionProvider(
        node_id="worker_gpu_1",
        peer_snapshot=lambda: coordinator.worker_snapshot("worker_gpu_1"),
        send_message=lambda _message: None,
        result_ingestor=lambda _attempt, output, _plan: dict(output),
        dispatch_enabled=False,
    )
    assert provider.inspect().healthy is False
    with pytest.raises(ProviderUnavailable) as disabled:
        provider.reserve(_stage_request(provider.provider_id))
    assert disabled.value.code == "remote_diffusion_unavailable"

    enabled = RemoteDiffusionProvider(
        node_id="worker_gpu_1",
        peer_snapshot=lambda: coordinator.worker_snapshot("worker_gpu_1"),
        send_message=lambda _message: None,
        result_ingestor=lambda _attempt, output, _plan: dict(output),
        dispatch_enabled=True,
    )
    request = _stage_request(enabled.provider_id)
    request.runtime_context["diffusion_artifact_manifest"] = {
        **_manifest(), "sha256": "f" * 64,
    }
    with pytest.raises(ProviderUnavailable) as artifact:
        enabled.reserve(request)
    assert artifact.value.code == "artifact_manifest_mismatch"


def test_offer_replay_fences_execution_and_replays_the_same_terminal_message():
    sent = []
    calls = []

    def execute(payload, cancel_event):
        calls.append(payload["attempt_id"])
        return _result()

    worker, _ = _connected_worker(execute, sent)
    offer = _offer()
    worker.receive_offer(offer.snapshot())
    _wait_for(sent, "stage_result")
    first_terminal = sent[-1]

    replayed = worker.receive_offer(offer.snapshot())
    assert replayed == first_terminal
    assert calls == ["att_diffworker01"]
    assert [message.message_type for message in sent] == [
        "stage_accept", "stage_result", "stage_result",
    ]
    assert sent[-1] == first_terminal


def test_busy_and_expired_offers_are_fenced_without_starting_another_executor():
    sent = []
    started = threading.Event()
    release = threading.Event()

    def execute(payload, cancel_event):
        started.set()
        assert release.wait(2.0)
        return _result()

    worker, _ = _connected_worker(execute, sent)
    worker.receive_offer(_offer().snapshot())
    assert started.wait(2.0)
    busy = worker.receive_offer(_offer(attempt_id="att_diffworker02").snapshot())
    def expires_after_lease(payload, cancel_event):
        assert cancel_event.wait(2.0)
        return _result()

    expired = DiffusionWorkerAdapter(
        node_id="worker_gpu_2",
        capabilities=_capabilities(),
        executor=expires_after_lease,
        send_message=sent.append,
    )
    coordinator = DiffusionCoordinatorControlPlane()
    hello = expired.begin_hello(sent_at_ms=1_000)
    assert hello is not None
    expired.receive_hello_ack(coordinator.receive_hello(
        "worker_gpu_2", hello.snapshot(), coordinator_node_id="master", sent_at_ms=1_001,
    ).snapshot())
    expired_reply = expired.receive_offer(_offer(
        attempt_id="att_diffworker03", lease_expires_at_ms=1_001,
        node_id="worker_gpu_2", sent_at_ms=1_000,
    ).snapshot())
    release.set()
    _wait_for(sent, "stage_result")
    _wait_for(sent, "stage_error")

    assert busy.payload["accepted"] is False
    assert busy.payload["reason_code"] == "worker_busy"
    assert expired_reply.payload["accepted"] is True
    assert [message for message in sent if message.message_type == "stage_error"][-1].payload[
        "error_code"
    ] == "lease_expired"


def test_cancel_fences_executor_and_emits_a_single_terminal_message():
    sent = []
    entered = threading.Event()

    def execute(payload, cancel_event):
        entered.set()
        assert cancel_event.wait(2.0)
        return _result()

    worker, _ = _connected_worker(execute, sent)
    offer = _offer()
    worker.receive_offer(offer.snapshot())
    assert entered.wait(2.0)
    cancelled = worker.receive_cancel(build_message(
        "stage_cancel",
        {
            "workflow_id": offer.payload["workflow_id"],
            "stage_id": offer.payload["stage_id"],
            "attempt_id": offer.payload["attempt_id"],
            "lease_id": offer.payload["lease_id"],
            "lease_epoch": offer.payload["lease_epoch"],
            "reason_code": "coordinator_cancelled",
        },
        message_id="msg_cancel_diffworker01",
        sent_at_ms=1_001,
        version=3,
    ).snapshot())
    _wait_for(sent, "stage_cancelled")
    replayed = worker.receive_cancel(build_message(
        "stage_cancel",
        {
            "workflow_id": offer.payload["workflow_id"],
            "stage_id": offer.payload["stage_id"],
            "attempt_id": offer.payload["attempt_id"],
            "lease_id": offer.payload["lease_id"],
            "lease_epoch": offer.payload["lease_epoch"],
            "reason_code": "lease_revoked",
        },
        message_id="msg_cancel_diffworker02",
        sent_at_ms=1_002,
        version=3,
    ).snapshot())
    time.sleep(0.03)

    assert cancelled.message_type == "stage_cancelled"
    assert replayed == cancelled
    assert [message.message_type for message in sent] == [
        "stage_accept", "stage_cancelled", "stage_cancelled",
    ]
    assert sent[1] == sent[2]


def test_executor_errors_are_sanitized_and_invalid_control_state_is_rejected():
    sent = []

    def broken_execute(payload, cancel_event):
        raise RuntimeError("local path and credentials must not cross the wire")

    worker, coordinator = _connected_worker(broken_execute, sent)
    worker.receive_offer(_offer().snapshot())
    _wait_for(sent, "stage_error")
    assert sent[-1].payload == {
        "workflow_id": "wf_diffworker01",
        "stage_id": "image_stage_1",
        "attempt_id": "att_diffworker01",
        "lease_id": "lease_diffworker01",
        "lease_epoch": 1,
        "provider_id": remote_diffusion_provider_id("worker_gpu_1"),
        "error_code": "diffusion_execution_failed",
        "retryable": True,
    }
    with pytest.raises(WorkerProtocolError) as unknown_cancel:
        worker.receive_cancel(build_message(
            "stage_cancel",
            {
                "workflow_id": "wf_diffworker01",
                "stage_id": "image_stage_1",
                "attempt_id": "att_diffworker01",
                "lease_id": "lease_diffworker01",
                "lease_epoch": 1,
                "reason_code": "coordinator_cancelled",
            },
            message_id="msg_cancel_unknown01",
            sent_at_ms=1_002,
            version=3,
        ).snapshot())
    assert unknown_cancel.value.code == "unknown_attempt"

    snapshot = coordinator.status()
    assert snapshot["control_plane_connected"] is True
    assert snapshot["adapter_connected"] is False
