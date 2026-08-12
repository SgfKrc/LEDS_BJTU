"""Coordinator-side verified result ingestion for v3 diffusion Workers."""

from __future__ import annotations

import io
import threading
from typing import Any, Iterable, Mapping

from task_provider import (
    LocalFullModelProvider,
    Reservation,
    StageAttempt,
    StageResult,
)

from .data_plane import DiffusionDataPlaneRuntime
from .distributed import BlobNotFound
from .transfer import DiffusionBlobTransferClient


class DiffusionGridAggregatorProvider(LocalFullModelProvider):
    """主节点本地 image_grid Provider for fixed multi-seed batches."""

    provider_id = "diffusion_grid_aggregator"

    def __init__(self, runtime: "DiffusionCoordinatorRuntime", *, node_id: str = ""):
        self._runtime = runtime
        self._attempt_reservations: dict[str, str] = {}
        self._result_lock = threading.RLock()
        super().__init__(
            self._aggregate,
            provider_id=self.provider_id,
            node_id=str(node_id or "master"),
            supported_stage_types=("image_grid",),
            max_concurrency=1,
            provider_kind="local_diffusion_grid",
        )

    def _aggregate(self, request, cancel_event: threading.Event) -> dict[str, Any]:
        return self._runtime.aggregate_grid(
            workflow_id=request.workflow_id,
            root_input=request.root_input,
            dependencies=request.dependencies,
            cancel_event=cancel_event,
        )

    def execute(
        self,
        attempt: StageAttempt,
        reservation: Reservation,
        cancel_event: threading.Event,
    ) -> StageResult:
        result = super().execute(attempt, reservation, cancel_event)
        with self._result_lock:
            self._attempt_reservations[attempt.attempt_id] = reservation.reservation_id
        return result

    def release(self, reservation_id: str) -> None:
        try:
            super().release(reservation_id)
        finally:
            with self._result_lock:
                for attempt_id, owned_reservation in tuple(
                    self._attempt_reservations.items()
                ):
                    if owned_reservation == reservation_id:
                        self._attempt_reservations.pop(attempt_id, None)

    def discard_result(self, attempt_id: str, output: Mapping[str, Any]) -> bool:
        """Discard a grid created by this attempt before TaskGraph commit."""
        image = output.get("grid") if isinstance(output, Mapping) else None
        if not isinstance(image, Mapping):
            return False
        blob_id = str(image.get("blob_id", ""))
        if not blob_id:
            return False
        with self._result_lock:
            reservation_id = self._attempt_reservations.get(attempt_id, "")
            reservation = self._reservations.get(reservation_id)
            workflow_id = str(reservation.workflow_id) if reservation else ""
        if not workflow_id:
            return False
        try:
            return self._runtime.delete_output_blob(workflow_id, blob_id)
        except BlobNotFound:
            return False

    def close(self) -> None:
        try:
            super().close()
        finally:
            with self._result_lock:
                self._attempt_reservations.clear()


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

    def aggregate_grid(
        self,
        *,
        workflow_id: str,
        root_input: Mapping[str, Any],
        dependencies: Mapping[str, Any],
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Compose fixed 2x2 candidate outputs into a parent-referenced CAS blob."""
        from PIL import Image

        stage_ids = root_input.get("grid_stage_ids")
        if not isinstance(stage_ids, list) or len(stage_ids) != 4:
            raise ValueError("image grid requires exactly four ordered stages")
        if any(not isinstance(stage_id, str) for stage_id in stage_ids):
            raise ValueError("image grid stage order is invalid")
        if set(stage_ids) != set(dependencies):
            raise ValueError("image grid dependencies do not match the fixed four-seed plan")
        if root_input.get("grid_layout") != "2x2":
            raise ValueError("only the 2x2 image grid is supported")

        members: list[dict[str, Any]] = []
        decoded: list[Image.Image] = []
        try:
            for stage_id in stage_ids:
                if cancel_event.is_set():
                    raise RuntimeError("image grid aggregation cancelled")
                candidate = dependencies[stage_id]
                descriptor = candidate.get("image") if isinstance(candidate, Mapping) else None
                if not isinstance(descriptor, Mapping):
                    raise ValueError(f"{stage_id} did not return an image descriptor")
                blob_id = str(descriptor.get("blob_id", ""))
                persisted, data = self.read_result(
                    workflow_id=workflow_id,
                    blob_id=blob_id,
                )
                if persisted.get("purpose") != "output":
                    raise ValueError(f"{stage_id} output is not an output blob")
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
                    decoded.append(source.convert("RGB"))
                members.append(dict(persisted))
            width, height = decoded[0].size
            if any(image.size != (width, height) for image in decoded[1:]):
                raise ValueError("image grid members must have identical dimensions")
            canvas = Image.new("RGB", (width * 2, height * 2))
            for index, image in enumerate(decoded):
                canvas.paste(image, ((index % 2) * width, (index // 2) * height))
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=False)
            if cancel_event.is_set():
                raise RuntimeError("image grid aggregation cancelled")
            descriptor = self._data_plane.store.put_bytes(
                buffer.getvalue(),
                content_type="image/png",
                purpose="output",
                owner_scope=f"distributed:{workflow_id}",
                width=canvas.width,
                height=canvas.height,
                metadata={
                    "workflow_id": workflow_id,
                    "stage_type": "image_grid",
                    "layout": "2x2",
                    "source_blob_ids": [item["blob_id"] for item in members],
                    "seed_order": list(root_input.get("grid_seeds", [])),
                },
                parent_blob_ids=tuple(item["blob_id"] for item in members),
                deduplicate=False,
            )
            if cancel_event.is_set():
                self.delete_output_blob(workflow_id, descriptor.blob_id)
                raise RuntimeError("image grid aggregation cancelled")
            return {
                "grid": descriptor.snapshot(),
                "images": members,
                "metrics": {
                    "layout": "2x2",
                    "image_count": len(members),
                    "seeds": list(root_input.get("grid_seeds", [])),
                },
            }
        except Exception:
            # A grid has parent references but no external lease. Delete it on
            # any post-write failure; candidate outputs remain workflow-owned.
            if "descriptor" in locals() and getattr(descriptor, "blob_id", ""):
                try:
                    self.delete_output_blob(workflow_id, descriptor.blob_id)
                except Exception:
                    pass
            raise
        finally:
            for image in decoded:
                image.close()

    def delete_output_blob(self, workflow_id: str, blob_id: str) -> bool:
        owner_scope = f"distributed:{workflow_id}"
        descriptor = self._data_plane.store.descriptor_for_owner(
            blob_id,
            owner_scope=owner_scope,
        )
        if descriptor.purpose != "output":
            raise ValueError("distributed workflow blob is not an output")
        return self._data_plane.store.delete(blob_id, owner_scope=owner_scope)

    def discard_result(
        self,
        attempt: StageAttempt,
        output: Mapping[str, Any],
    ) -> None:
        """Delete one uncommitted local output without crossing workflow scope."""
        image = output.get("image")
        if not isinstance(image, Mapping):
            raise ValueError("distributed output is missing its image descriptor")
        blob_id = str(image.get("blob_id", ""))
        owner_scope = f"distributed:{attempt.request.workflow_id}"
        try:
            descriptor = self._data_plane.store.descriptor_for_owner(
                blob_id,
                owner_scope=owner_scope,
            )
        except BlobNotFound:
            return
        if descriptor.purpose != "output":
            raise ValueError("distributed workflow blob is not an output")
        self._data_plane.store.delete(blob_id, owner_scope=owner_scope)

    def discard_workflow(
        self,
        workflow_id: str,
        *,
        revoke_leases: bool = False,
    ) -> dict[str, int]:
        return self._data_plane.store.delete_owner_scope(
            f"distributed:{workflow_id}",
            purpose="output",
            revoke_leases=revoke_leases,
        )

    def reconcile_recovered_workflows(
        self,
        workflows: Iterable[Mapping[str, Any]],
    ) -> dict[str, int]:
        summary = {
            "workflows_reconciled": 0,
            "blobs_removed": 0,
            "blobs_blocked": 0,
            "leases_revoked": 0,
        }
        for workflow in workflows:
            if (
                workflow.get("template") not in {"image_generate_v1", "image_grid_v1", "llm_sd15_v1"}
                or workflow.get("state") != "failed"
                or not workflow.get("recovered_after_restart")
                or workflow.get("recovery_reason")
                != "coordinator_restarted_before_result_commit"
            ):
                continue
            workflow_id = str(workflow.get("workflow_id", ""))
            cleanup = self.discard_workflow(
                workflow_id,
                revoke_leases=True,
            )
            summary["workflows_reconciled"] += 1
            for key in ("blobs_removed", "blobs_blocked", "leases_revoked"):
                summary[key] += int(cleanup[key])
        return summary

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


__all__ = ["DiffusionCoordinatorRuntime", "DiffusionGridAggregatorProvider"]
