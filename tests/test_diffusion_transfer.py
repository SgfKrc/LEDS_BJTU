import io
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.data_plane import DiffusionDataPlaneRuntime, router  # noqa: E402
from diffusion.transfer import (  # noqa: E402
    DiffusionBlobTransferClient,
    DiffusionTransferError,
    TransferResponse,
)
from diffusion.worker_adapter import (  # noqa: E402
    DiffusionCoordinatorControlPlane,
    DiffusionExecutionResult,
    DiffusionWorkerAdapter,
    RemoteDiffusionProvider,
)
from task_provider import StageAttempt, StageRequest  # noqa: E402
from task_worker_protocol import canonical_sha256  # noqa: E402


def _png():
    output = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 40, 80)).save(output, format="PNG")
    return output.getvalue()


def _runtime_client(tmp_path, name):
    runtime = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path / name,
        cluster_secret="s" * 32,
    )
    app = FastAPI()
    app.state.diffusion_data_plane = runtime
    app.include_router(router)
    return runtime, TestClient(app)


def _requester(client):
    def request(method, url, headers, body):
        parsed = urlsplit(url)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"
        response = client.request(method, path, headers=dict(headers), content=body)
        return TransferResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    return request


def _manifest():
    body = {
        "artifact_id": "artifact_sd15",
        "pipeline_kind": "sd15_pipeline",
        "revision": "revision_20260807",
        "components": [{
            "artifact_id": "base_unet",
            "artifact_kind": "unet",
            "sha256": "b" * 64,
        }],
    }
    return {**body, "sha256": canonical_sha256(body)}


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


def test_publish_and_download_round_trip_verifies_each_http_range(tmp_path):
    source, source_client = _runtime_client(tmp_path, "source")
    destination, _destination_client = _runtime_client(tmp_path, "destination")
    try:
        published = source.publish_output(
            _png(),
            attempt_id="att_transfer_12345678",
            base_url="http://source.test",
            grant_ttl_seconds=60,
            owner_scope="workflow_transfer",
            content_type="image/png",
            width=16,
            height=12,
        )
        transfer = DiffusionBlobTransferClient(_requester(source_client), chunk_bytes=17)
        local = transfer.download_to_store(
            attempt_id="att_transfer_12345678",
            descriptor=published["descriptor"],
            transfer_plan=published["transfer_plan"],
            destination_store=destination.store,
            owner_scope="workflow_transfer",
        )

        assert local["blob_id"] != published["descriptor"]["blob_id"]
        assert local["sha256"] == published["descriptor"]["sha256"]
        assert destination.store.descriptor(local["blob_id"]).purpose == "output"
    finally:
        source.close()
        destination.close()


def test_download_refuses_tampered_range_before_it_reaches_local_store(tmp_path):
    source, source_client = _runtime_client(tmp_path, "source")
    destination, _destination_client = _runtime_client(tmp_path, "destination")
    try:
        published = source.publish_output(
            _png(),
            attempt_id="att_transfer_12345678",
            base_url="http://source.test",
            grant_ttl_seconds=60,
            owner_scope="workflow_transfer",
            content_type="image/png",
            width=16,
            height=12,
        )
        request = _requester(source_client)

        def tampered(method, url, headers, body):
            response = request(method, url, headers, body)
            if response.status_code in {200, 206} and response.content:
                return TransferResponse(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=bytes([response.content[0] ^ 1]) + response.content[1:],
                )
            return response

        transfer = DiffusionBlobTransferClient(tampered, chunk_bytes=31)
        with pytest.raises(DiffusionTransferError) as captured:
            transfer.download_to_store(
                attempt_id="att_transfer_12345678",
                descriptor=published["descriptor"],
                transfer_plan=published["transfer_plan"],
                destination_store=destination.store,
                owner_scope="workflow_transfer",
            )
        assert captured.value.code == "transfer_digest_mismatch"
        assert destination.store.snapshot()["blobs"] == 0
    finally:
        source.close()
        destination.close()


def test_remote_provider_fake_worker_transfers_output_over_data_plane(tmp_path):
    worker_runtime, worker_client = _runtime_client(tmp_path, "worker")
    coordinator_runtime, _coordinator_client = _runtime_client(tmp_path, "coordinator")
    holder = {}
    remote = {}
    try:
        def execute(payload, cancel_event):
            assert not cancel_event.is_set()
            published = worker_runtime.publish_output(
                _png(),
                attempt_id=payload["attempt_id"],
                base_url="http://worker.test",
                grant_ttl_seconds=60,
                owner_scope=payload["workflow_id"],
                content_type="image/png",
                width=16,
                height=12,
            )
            remote["descriptor"] = published["descriptor"]
            return DiffusionExecutionResult(
                output={
                    "image": published["descriptor"],
                    "metrics": {"elapsed_seconds": 0.1, "seed": 19950101},
                },
                metadata={
                    "node_id": "worker_gpu_1",
                    "provider_kind": "pc_diffusion_worker",
                    "elapsed_seconds": 0.1,
                    "seed": 19950101,
                    "artifact_manifest_sha256": _manifest()["sha256"],
                    "distributed": True,
                },
                transfer_plan=published["transfer_plan"],
            )

        worker = DiffusionWorkerAdapter(
            node_id="worker_gpu_1",
            capabilities=_capabilities(),
            executor=execute,
            send_message=lambda message: holder["provider"].handle_message(message.snapshot()),
        )
        control = DiffusionCoordinatorControlPlane()
        hello = worker.begin_hello()
        assert hello is not None
        worker.receive_hello_ack(control.receive_hello(
            "worker_gpu_1", hello.snapshot(), coordinator_node_id="master",
        ).snapshot())
        transfer = DiffusionBlobTransferClient(_requester(worker_client), chunk_bytes=19)

        def ingest(attempt, output, transfer_plan):
            return {
                "image": transfer.download_to_store(
                    attempt_id=attempt.attempt_id,
                    descriptor=output["image"],
                    transfer_plan=transfer_plan,
                    destination_store=coordinator_runtime.store,
                    owner_scope=attempt.request.workflow_id,
                ),
                "metrics": dict(output["metrics"]),
            }

        def send_to_worker(message):
            if message.message_type == "stage_offer":
                worker.receive_offer(message.snapshot())
            elif message.message_type == "stage_cancel":
                worker.receive_cancel(message.snapshot())
            elif message.message_type == "lease_renew":
                worker.receive_lease_renew(message.snapshot())
            else:
                pytest.fail(f"unexpected Provider message: {message.message_type}")

        provider = RemoteDiffusionProvider(
            node_id="worker_gpu_1",
            peer_snapshot=lambda: control.worker_snapshot("worker_gpu_1"),
            send_message=send_to_worker,
            result_ingestor=ingest,
            dispatch_enabled=True,
        )
        holder["provider"] = provider
        manifest = _manifest()
        request = StageRequest(
            workflow_id="wf_transferprovider01",
            request_id="request_transferprovider01",
            stage_id="image_stage_1",
            stage_type="image_generate",
            provider_id=provider.provider_id,
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
        reservation = provider.reserve(request)
        attempt = StageAttempt(
            attempt_id="att_transferprovider01",
            request=request,
            provider_id=provider.provider_id,
            lease_id="lease_transferprovider01",
            lease_epoch=1,
            lease_expires_at=time.time() + 5.0,
        )
        result = provider.execute(attempt, reservation, threading.Event())

        assert result.output["image"]["blob_id"] != ""
        assert result.output["image"]["sha256"] == remote["descriptor"]["sha256"]
        assert worker_runtime.store.snapshot()["blobs"] == 1
        assert coordinator_runtime.store.snapshot()["blobs"] == 1
        assert "grant" not in result.metadata
        provider.release(reservation.reservation_id)
    finally:
        worker_runtime.close()
        coordinator_runtime.close()
