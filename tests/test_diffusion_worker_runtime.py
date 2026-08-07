import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.worker_runtime import (  # noqa: E402
    DiffusionWorkerRuntime,
    DiffusionWorkerRuntimeError,
)


class _FakeService:
    TERMINAL_STATES = {"completed", "failed", "cancelled"}

    def __init__(self, *, terminal="completed"):
        self.terminal = terminal
        self.cancelled = False
        self.deleted = []
        self.submissions = []

    @staticmethod
    def _artifact_snapshot():
        return {
            "loaded": True,
            "loaded_artifact": {
                "artifact_id": "sd15_test",
                "artifact": {
                    "artifact_kind": "sd15_pipeline",
                    "sha256": "a" * 64,
                },
            },
            "engine_config": {"dtype": "float16"},
        }

    def snapshot(self):
        return self._artifact_snapshot()

    def submit_generation(self, request, *, owner_scope):
        self.submissions.append((request, owner_scope))
        return {"job_id": "sdjob_runtime01"}

    def get_job(self, _job_id):
        state = "cancelled" if self.cancelled else self.terminal
        return {
            "state": state,
            "output_blob_id": "img_local_runtime01" if state == "completed" else "",
            "metrics": {"elapsed_seconds": 1.25},
        }

    def cancel_job(self, _job_id):
        self.cancelled = True
        return {"accepted": True}

    def get_blob(self, _blob_id):
        return SimpleNamespace(
            data=b"png-data",
            content_type="image/png",
            width=512,
            height=512,
        )

    def delete_blob(self, blob_id):
        self.deleted.append(blob_id)
        return True


class _FakeDataPlane:
    def __init__(self):
        self.publications = []

    def publish_output(self, data, **kwargs):
        self.publications.append((data, kwargs))
        return {
            "descriptor": {
                "blob_id": "img_1234567890abcdef",
                "sha256": "c" * 64,
                "size_bytes": 8,
                "content_type": "image/png",
                "width": 512,
                "height": 512,
                "purpose": "output",
            },
            "transfer_plan": {
                "base_url": "http://100.64.0.2:8000",
                "downloads": [{
                    "blob_id": "img_1234567890abcdef",
                    "lease_id": "bls_1234567890abcdef",
                    "grant": "a" * 32 + "." + "b" * 43,
                }],
            },
        }


def _runtime(service=None, data_plane=None):
    return DiffusionWorkerRuntime(
        service=service or _FakeService(),
        data_plane=data_plane or _FakeDataPlane(),
        node_id="worker_01",
        data_plane_base_url="http://100.64.0.2:8000",
        sleep=lambda _: None,
    )


def _offer(runtime):
    manifest = runtime.artifact_manifest()
    assert manifest is not None
    return {
        "stage_type": "image_generate",
        "attempt_id": "att_runtime01",
        "artifact_manifest": manifest,
        "root_input": {
            "prompt": "a mountain cabin",
            "negative_prompt": "",
            "seed": 19950101,
            "width": 512,
            "height": 512,
            "steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "PNDMScheduler",
        },
    }


def test_runtime_advertises_loaded_artifact_and_publishes_worker_output():
    service = _FakeService()
    data_plane = _FakeDataPlane()
    runtime = _runtime(service, data_plane)

    capabilities = runtime.capabilities()
    assert capabilities is not None
    assert capabilities["stage_types"] == ["image_generate"]
    assert capabilities["image"]["artifact_manifests"] == [
        runtime.artifact_manifest()
    ]

    result = runtime.execute(_offer(runtime), threading.Event())

    assert service.submissions[0][1] == "distributed:att_runtime01"
    assert data_plane.publications[0][0] == b"png-data"
    assert result.output["image"]["blob_id"] == "img_1234567890abcdef"
    assert result.metadata["node_id"] == "worker_01"
    assert result.metadata["artifact_manifest_sha256"] == runtime.artifact_manifest()["sha256"]
    assert service.deleted == ["img_local_runtime01"]


def test_runtime_cancellation_never_publishes_partial_output():
    service = _FakeService()
    data_plane = _FakeDataPlane()
    runtime = _runtime(service, data_plane)
    cancelled = threading.Event()
    cancelled.set()

    result = runtime.execute(_offer(runtime), cancelled)

    assert result.output == {}
    assert result.metadata == {}
    assert data_plane.publications == []
    assert service.cancelled is True


def test_runtime_rejects_missing_or_mismatched_loaded_artifact():
    runtime = _runtime()
    offer = _offer(runtime)
    offer["artifact_manifest"] = {"not": "the loaded artifact"}

    with pytest.raises(DiffusionWorkerRuntimeError, match="not loaded"):
        runtime.execute(offer, threading.Event())

    with pytest.raises(DiffusionWorkerRuntimeError, match="base URL"):
        DiffusionWorkerRuntime(
            service=_FakeService(),
            data_plane=_FakeDataPlane(),
            node_id="worker_01",
            data_plane_base_url="not-a-url",
        )
