import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.coordinator_runtime import (  # noqa: E402
    DiffusionCoordinatorRuntime,
    DiffusionGridAggregatorProvider,
)
from diffusion.data_plane import DiffusionDataPlaneRuntime  # noqa: E402
from diffusion.distributed import (  # noqa: E402
    BlobAuthorizationError,
    BlobConflict,
    BlobNotFound,
)
from task_graph import StageSpec, TaskGraphCoordinator  # noqa: E402
from task_provider import (  # noqa: E402
    InProcessWorkerProvider,
    ProviderRegistry,
    StageAttempt,
    StageRequest,
)


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


def test_coordinator_runtime_discards_only_attempt_workflow_output(tmp_path):
    from PIL import Image
    from io import BytesIO

    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(image_buffer, format="PNG")
    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    try:
        owned = data_plane.store.put_bytes(
            image_buffer.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope="distributed:wf_discardown1",
            width=2,
            height=2,
        )
        other = data_plane.store.put_bytes(
            image_buffer.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope="distributed:wf_discardown2",
            width=2,
            height=2,
        )
        request = StageRequest(
            workflow_id="wf_discardown1",
            request_id="request_discardown1",
            stage_id="image_stage_1",
            stage_type="image_generate",
            provider_id="remote_diffusion_worker_01",
            dependencies={},
            root_input={},
        )
        attempt = StageAttempt(
            attempt_id="att_discardown01",
            request=request,
            provider_id=request.provider_id,
            lease_id="lease_discardown01",
            lease_epoch=1,
            lease_expires_at=10.0,
        )
        runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)

        runtime.discard_result(attempt, {"image": owned.snapshot()})
        runtime.discard_result(attempt, {"image": owned.snapshot()})

        with pytest.raises(BlobNotFound):
            data_plane.store.descriptor(owned.blob_id)
        assert data_plane.store.descriptor(other.blob_id).blob_id == other.blob_id
    finally:
        data_plane.close()


def test_fixed_four_seed_grid_is_ordered_and_parent_referenced(tmp_path):
    from PIL import Image

    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    workflow_id = "wf_gridruntime1"
    owner = f"distributed:{workflow_id}"
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    try:
        members = []
        for color in colors:
            payload = io.BytesIO()
            Image.new("RGB", (4, 3), color).save(payload, format="PNG")
            members.append(data_plane.store.put_bytes(
                payload.getvalue(),
                content_type="image/png",
                purpose="output",
                owner_scope=owner,
                width=4,
                height=3,
            ).snapshot())
        runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
        output = runtime.aggregate_grid(
            workflow_id=workflow_id,
            root_input={
                "grid_stage_ids": ["seed_0", "seed_1", "seed_2", "seed_3"],
                "grid_seeds": [11, 12, 13, 14],
                "grid_layout": "2x2",
            },
            dependencies={
                f"seed_{index}": {"image": descriptor, "metrics": {"seed": 11 + index}}
                for index, descriptor in enumerate(members)
            },
            cancel_event=threading.Event(),
        )

        assert output["grid"]["width"] == 8
        assert output["grid"]["height"] == 6
        assert [item["blob_id"] for item in output["images"]] == [
            item["blob_id"] for item in members
        ]
        _descriptor, grid_bytes = runtime.read_result(
            workflow_id=workflow_id,
            blob_id=output["grid"]["blob_id"],
        )
        with Image.open(io.BytesIO(grid_bytes)) as grid:
            assert grid.getpixel((1, 1)) == colors[0]
            assert grid.getpixel((5, 1)) == colors[1]
            assert grid.getpixel((1, 4)) == colors[2]
            assert grid.getpixel((5, 4)) == colors[3]
        with pytest.raises(BlobConflict, match="referenced"):
            data_plane.store.delete(members[0]["blob_id"], owner_scope=owner)
        assert runtime.delete_output_blob(workflow_id, output["grid"]["blob_id"])
        assert data_plane.store.delete(members[0]["blob_id"], owner_scope=owner)
    finally:
        data_plane.close()


def test_four_seed_stage_overrides_and_node_observability(tmp_path):
    from PIL import Image

    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
    registry = ProviderRegistry()
    seen = []

    def generator(request, cancel_event):
        seed = request.root_input["seed"]
        seen.append((request.stage_id, seed, request.provider_id))
        payload = io.BytesIO()
        Image.new("RGB", (2, 2), (seed, 0, 0)).save(payload, format="PNG")
        descriptor = data_plane.store.put_bytes(
            payload.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope=f"distributed:{request.workflow_id}",
            width=2,
            height=2,
        )
        return {"image": descriptor.snapshot(), "metrics": {"seed": seed}}

    registry.register(InProcessWorkerProvider(
        "worker_a", generator, node_id="node-a",
        supported_stage_types=("image_generate",),
    ))
    registry.register(InProcessWorkerProvider(
        "worker_b", generator, node_id="node-b",
        supported_stage_types=("image_generate",),
    ))
    registry.register(DiffusionGridAggregatorProvider(runtime, node_id="master"))
    coordinator = TaskGraphCoordinator(provider_registry=registry, max_parallel_stages=4)
    try:
        stage_ids = [f"seed_{index}" for index in range(4)]
        stages = [
            StageSpec(
                stage_id,
                "image_generate",
                provider="worker_a" if index % 2 == 0 else "worker_b",
                root_input_overrides={"seed": 21 + index},
            )
            for index, stage_id in enumerate(stage_ids)
        ]
        stages.append(StageSpec(
            "image_grid",
            "image_grid",
            depends_on=tuple(stage_ids),
            provider="diffusion_grid_aggregator",
            root_input_overrides={"__replace__": {
                "grid_stage_ids": stage_ids,
                "grid_seeds": [21, 22, 23, 24],
                "grid_layout": "2x2",
            }},
        ))
        output, workflow = coordinator.run(
            stages,
            "image_grid",
            {"seed": 0, "prompt": "fixed"},
            workflow_id="wf_gridoverrid1",
            template="image_grid_v1",
        )
        coordinator.commit_result(workflow["workflow_id"])

        assert sorted(seed for _stage, seed, _provider in seen) == [21, 22, 23, 24]
        attempts = [attempt for stage in workflow["stages"] for attempt in stage["attempts"]]
        assert {attempt["provider_node_id"] for attempt in attempts} == {
            "node-a", "node-b", "master",
        }
        assert output["metrics"]["seeds"] == [21, 22, 23, 24]
    finally:
        coordinator.close()
        data_plane.close()


def test_grid_cancelled_after_cas_write_removes_only_grid(tmp_path, monkeypatch):
    from PIL import Image

    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    workflow_id = "wf_gridcancel1"
    owner = f"distributed:{workflow_id}"
    cancel_event = threading.Event()
    try:
        members = []
        for index in range(4):
            payload = io.BytesIO()
            Image.new("RGB", (2, 2), (index * 20, 40, 60)).save(payload, format="PNG")
            members.append(data_plane.store.put_bytes(
                payload.getvalue(),
                content_type="image/png",
                purpose="output",
                owner_scope=owner,
                width=2,
                height=2,
            ).snapshot())
        original_put = data_plane.store.put_bytes

        def cancel_after_put(*args, **kwargs):
            descriptor = original_put(*args, **kwargs)
            if kwargs.get("parent_blob_ids"):
                cancel_event.set()
            return descriptor

        monkeypatch.setattr(data_plane.store, "put_bytes", cancel_after_put)
        runtime = DiffusionCoordinatorRuntime(data_plane=data_plane)
        with pytest.raises(RuntimeError, match="cancelled"):
            runtime.aggregate_grid(
                workflow_id=workflow_id,
                root_input={
                    "grid_stage_ids": ["seed_0", "seed_1", "seed_2", "seed_3"],
                    "grid_seeds": [1, 2, 3, 4],
                    "grid_layout": "2x2",
                },
                dependencies={
                    f"seed_{index}": {"image": descriptor, "metrics": {}}
                    for index, descriptor in enumerate(members)
                },
                cancel_event=cancel_event,
            )

        assert data_plane.store.snapshot()["blobs"] == 4
        assert all(
            data_plane.store.descriptor(item["blob_id"]).blob_id == item["blob_id"]
            for item in members
        )
    finally:
        data_plane.close()


def test_recovered_image_grid_workflow_reclaims_owner_scope(tmp_path):
    from PIL import Image

    data_plane = DiffusionDataPlaneRuntime.create(
        state_dir=tmp_path,
        cluster_secret="x" * 32,
    )
    try:
        payload = io.BytesIO()
        Image.new("RGB", (2, 2), (10, 20, 30)).save(payload, format="PNG")
        descriptor = data_plane.store.put_bytes(
            payload.getvalue(),
            content_type="image/png",
            purpose="output",
            owner_scope="distributed:wf_gridrecover1",
            width=2,
            height=2,
        )
        summary = DiffusionCoordinatorRuntime(
            data_plane=data_plane,
        ).reconcile_recovered_workflows([{
            "workflow_id": "wf_gridrecover1",
            "template": "image_grid_v1",
            "state": "failed",
            "recovered_after_restart": True,
            "recovery_reason": "coordinator_restarted_before_result_commit",
        }])

        assert summary["workflows_reconciled"] == 1
        assert summary["blobs_removed"] == 1
        with pytest.raises(BlobNotFound):
            data_plane.store.descriptor(descriptor.blob_id)
    finally:
        data_plane.close()
