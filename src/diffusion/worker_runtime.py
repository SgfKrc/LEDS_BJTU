"""Concrete SD 1.5 executor used by the optional v3 image worker bridge."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from task_worker_protocol import canonical_sha256

from .data_plane import DiffusionDataPlaneRuntime
from .service import DiffusionService, build_sd15_generation_request
from .worker_adapter import DiffusionExecutionResult


class DiffusionWorkerRuntimeError(RuntimeError):
    """A safe failure before a local SD job can produce a remote result."""


def _base_url(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DiffusionWorkerRuntimeError(
            "diffusion worker requires an explicit reachable data-plane base URL"
        )
    if parsed.query or parsed.fragment:
        raise DiffusionWorkerRuntimeError(
            "diffusion data-plane base URL must not contain a query or fragment"
        )
    return str(value).strip().rstrip("/")


class DiffusionWorkerRuntime:
    """Adapt one loaded local SD 1.5 service to the v3 Worker executor API.

    Scheduler does not own this runtime.  The API lifespan creates it only
    when the data plane and an explicit public base URL are available.
    """

    def __init__(
        self,
        *,
        service: DiffusionService,
        data_plane: DiffusionDataPlaneRuntime,
        node_id: str,
        data_plane_base_url: str,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._service = service
        self._data_plane = data_plane
        self._node_id = str(node_id)
        self._data_plane_base_url = _base_url(data_plane_base_url)
        self._clock = clock
        self._sleep = sleep
        self._poll_interval_seconds = float(poll_interval_seconds)

    def artifact_manifest(self) -> dict[str, Any] | None:
        """Return the exact loaded base artifact identity, never a local path."""
        snapshot = self._service.snapshot()
        loaded = snapshot.get("loaded_artifact")
        if not snapshot.get("loaded") or not isinstance(loaded, Mapping):
            return None
        artifact_id = str(loaded.get("artifact_id", ""))
        artifact = loaded.get("artifact")
        if not isinstance(artifact, Mapping):
            return None
        digest = str(artifact.get("sha256", "")).lower()
        if (
            not artifact_id
            or artifact.get("artifact_kind") != "sd15_pipeline"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None
        body = {
            "artifact_id": artifact_id,
            "pipeline_kind": "sd15_pipeline",
            "revision": f"local-{digest[:12]}",
            "components": [{
                "artifact_id": "base_pipeline",
                "artifact_kind": "sd15_pipeline",
                "sha256": digest,
            }],
        }
        return {**body, "sha256": canonical_sha256(body)}

    def capabilities(self) -> dict[str, Any] | None:
        """Advertise only an already loaded, hash-identified SD artifact."""
        manifest = self.artifact_manifest()
        if manifest is None:
            return None
        engine_config = self._service.snapshot().get("engine_config") or {}
        dtype = str(engine_config.get("dtype", "float16"))
        if dtype not in {"float16", "float32"}:
            return None
        return {
            "stage_types": ["image_generate"],
            "engines": ["diffusers_sd15"],
            "models": [],
            "max_concurrency": 1,
            "image": {
                "pipeline_kinds": ["sd15_pipeline"],
                "dtypes": [dtype],
                "max_width": 768,
                "max_height": 768,
                "max_pixels": 768 * 768,
                "max_batch": 1,
                "supports_controlnet": False,
                "supports_step_cancel": True,
                "artifact_manifests": [manifest],
            },
        }

    @staticmethod
    def _cancelled_result() -> DiffusionExecutionResult:
        # The adapter checks its cancellation fence before serializing output.
        return DiffusionExecutionResult(output={}, metadata={}, transfer_plan={})

    def execute(
        self,
        offer: Mapping[str, Any],
        cancel_event: Any,
    ) -> DiffusionExecutionResult:
        """Run a complete text-to-image Stage and publish its verified PNG."""
        if offer.get("stage_type") != "image_generate":
            raise DiffusionWorkerRuntimeError(
                "this SD worker currently supports image_generate only"
            )
        attempt_id = str(offer.get("attempt_id", ""))
        if not attempt_id:
            raise DiffusionWorkerRuntimeError("diffusion Stage attempt identity is missing")
        local_manifest = self.artifact_manifest()
        offered_manifest = offer.get("artifact_manifest")
        if local_manifest is None or offered_manifest != local_manifest:
            raise DiffusionWorkerRuntimeError(
                "the requested SD artifact is not loaded on this worker"
            )
        root_input = offer.get("root_input")
        if not isinstance(root_input, Mapping):
            raise DiffusionWorkerRuntimeError("diffusion Stage input is invalid")
        try:
            request = build_sd15_generation_request(
                prompt=str(root_input["prompt"]),
                negative_prompt=str(root_input["negative_prompt"]),
                seed=int(root_input["seed"]),
                width=int(root_input["width"]),
                height=int(root_input["height"]),
                steps=int(root_input["steps"]),
                guidance_scale=float(root_input["guidance_scale"]),
                scheduler=str(root_input["scheduler"]),
            )
            job = self._service.submit_generation(
                request,
                owner_scope=f"distributed:{attempt_id}",
            )
        except Exception as exc:
            raise DiffusionWorkerRuntimeError(
                "local SD worker could not accept the image Stage"
            ) from exc

        job_id = str(job.get("job_id", ""))
        cancellation_sent = False
        while True:
            if cancel_event.is_set() and not cancellation_sent:
                cancellation_sent = True
                try:
                    self._service.cancel_job(job_id)
                except Exception:
                    pass
            status = self._service.get_job(job_id)
            state = str(status.get("state", ""))
            if state in DiffusionService.TERMINAL_STATES:
                break
            self._sleep(self._poll_interval_seconds)

        output_blob_id = str(status.get("output_blob_id", ""))
        if cancel_event.is_set() or state == "cancelled":
            if output_blob_id:
                try:
                    self._service.delete_blob(output_blob_id)
                except Exception:
                    pass
            return self._cancelled_result()
        if state != "completed" or not output_blob_id:
            raise DiffusionWorkerRuntimeError("local SD worker failed the image Stage")

        try:
            blob = self._service.get_blob(output_blob_id)
            metrics = dict(status.get("metrics") or {})
            elapsed = float(metrics.get("elapsed_seconds", 0.0) or 0.0)
            publication = self._data_plane.publish_output(
                blob.data,
                attempt_id=attempt_id,
                base_url=self._data_plane_base_url,
                grant_ttl_seconds=120.0,
                owner_scope=f"distributed:{attempt_id}",
                content_type=blob.content_type,
                width=int(blob.width or request.width),
                height=int(blob.height or request.height),
                metadata={
                    "source_job_id": job_id,
                    "artifact_manifest_sha256": local_manifest["sha256"],
                },
            )
        finally:
            try:
                self._service.delete_blob(output_blob_id)
            except Exception:
                pass
        return DiffusionExecutionResult(
            output={
                "image": publication["descriptor"],
                "metrics": {
                    "elapsed_seconds": elapsed,
                    "seed": request.seed,
                },
            },
            metadata={
                "node_id": self._node_id,
                "provider_kind": "pc_diffusion_worker",
                "elapsed_seconds": elapsed,
                "seed": request.seed,
                "artifact_manifest_sha256": local_manifest["sha256"],
                "distributed": True,
            },
            transfer_plan=publication["transfer_plan"],
        )


__all__ = ["DiffusionWorkerRuntime", "DiffusionWorkerRuntimeError"]
