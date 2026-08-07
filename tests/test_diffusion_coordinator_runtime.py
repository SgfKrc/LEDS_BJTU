import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.coordinator_runtime import DiffusionCoordinatorRuntime  # noqa: E402
from diffusion.data_plane import DiffusionDataPlaneRuntime  # noqa: E402
from diffusion.distributed import BlobAuthorizationError  # noqa: E402
from task_provider import StageAttempt, StageRequest  # noqa: E402


class _Transfer:
    def __init__(self):
        self.calls = []

    def download_to_store(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "blob_id": "img_local1234567890",
            "sha256": "d" * 64,
            "size_bytes": 128,
            "content_type": "image/png",
            "width": 512,
            "height": 512,
            "purpose": "output",
        }


def test_coordinator_runtime_ingests_verified_output_without_transfer_grant():
    transfer = _Transfer()
    runtime = DiffusionCoordinatorRuntime(
        data_plane=SimpleNamespace(store=object()),
        transfer_client=transfer,
    )
    request = StageRequest(
        workflow_id="wf_coordinator01",
        request_id="request_coordinator01",
        stage_id="image_stage_1",
        stage_type="image_generate",
        provider_id="remote_diffusion_worker_01",
        dependencies={},
        root_input={},
    )
    attempt = StageAttempt(
        attempt_id="att_coordinator01",
        request=request,
        provider_id=request.provider_id,
        lease_id="lease_coordinator01",
        lease_epoch=1,
        lease_expires_at=10.0,
    )
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
    plan = {
        "base_url": "http://100.64.0.2:8000",
        "downloads": [{
            "blob_id": "img_1234567890abcdef",
            "lease_id": "bls_1234567890abcdef",
            "grant": "a" * 32 + "." + "b" * 43,
        }],
    }

    ingested = runtime.ingest_result(attempt, output, plan)

    assert ingested["image"]["blob_id"] == "img_local1234567890"
    assert ingested["metrics"] == output["metrics"]
    assert transfer.calls[0]["attempt_id"] == "att_coordinator01"
    assert transfer.calls[0]["owner_scope"] == "distributed:wf_coordinator01"
    assert transfer.calls[0]["metadata"] == {
        "workflow_id": "wf_coordinator01",
        "stage_id": "image_stage_1",
        "attempt_id": "att_coordinator01",
        "provider_id": "remote_diffusion_worker_01",
    }


def test_coordinator_runtime_reads_only_owned_completed_output(tmp_path):
    from PIL import Image
    from io import BytesIO

    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(image_buffer, format="PNG")
    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    try:
        descriptor = data_plane.store.put_bytes(
            image_buffer.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope="distributed:wf_readowned1",
            width=2,
            height=2,
        )
        runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
        public, data = runtime.read_result(
            workflow_id="wf_readowned1",
            blob_id=descriptor.blob_id,
        )
        assert public["sha256"] == descriptor.sha256
        assert data == image_buffer.getvalue()
        with pytest.raises(BlobAuthorizationError):
            runtime.read_result(
                workflow_id="wf_otherowner1",
                blob_id=descriptor.blob_id,
            )
    finally:
        data_plane.close()
