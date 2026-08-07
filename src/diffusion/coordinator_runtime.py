"""Coordinator-side verified result ingestion for v3 diffusion Workers."""

from __future__ import annotations

from typing import Any, Mapping

from task_provider import StageAttempt

from .data_plane import DiffusionDataPlaneRuntime
from .transfer import DiffusionBlobTransferClient


class DiffusionCoordinatorRuntime:
    """Copy verified worker output into the coordinator's local CAS store."""

    def __init__(
        self,
        *,
        data_plane: DiffusionDataPlaneRuntime,
        transfer_client: DiffusionBlobTransferClient | None = None,
    ) -> None:
        self._data_plane = data_plane
        self._transfer_client = transfer_client or DiffusionBlobTransferClient()

    def ingest_result(
        self,
        attempt: StageAttempt,
        output: Mapping[str, Any],
        transfer_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a fresh local descriptor after fail-closed HTTP verification."""
        image = output.get("image")
        metrics = output.get("metrics")
        if not isinstance(image, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("remote diffusion output is missing image metrics")
        descriptor = self._transfer_client.download_to_store(
            attempt_id=attempt.attempt_id,
            descriptor=image,
            transfer_plan=transfer_plan,
            destination_store=self._data_plane.store,
            owner_scope=f"distributed:{attempt.request.workflow_id}",
            metadata={
                "workflow_id": attempt.request.workflow_id,
                "stage_id": attempt.request.stage_id,
                "attempt_id": attempt.attempt_id,
                "provider_id": attempt.provider_id,
            },
        )
        return {"image": descriptor, "metrics": dict(metrics)}

    def read_result(
        self, *, workflow_id: str, blob_id: str,
    ) -> tuple[dict[str, Any], bytes]:
        """Read a completed workflow output through a short local lease."""
        owner_scope = f"distributed:{workflow_id}"
        descriptor = self._data_plane.store.descriptor_for_owner(
            blob_id,
            owner_scope=owner_scope,
        )
        if descriptor.purpose != "output":
            raise ValueError("distributed workflow blob is not an output")
        attempt_id = f"api:{workflow_id}"
        lease = self._data_plane.store.acquire_lease(
            blob_id,
            attempt_id=attempt_id,
            ttl_seconds=30.0,
        )
        try:
            data = self._data_plane.store.read_all(
                blob_id,
                lease_id=lease.lease_id,
                attempt_id=attempt_id,
            )
        finally:
            self._data_plane.store.release_lease(lease.lease_id)
        return descriptor.snapshot(), data


__all__ = ["DiffusionCoordinatorRuntime"]
