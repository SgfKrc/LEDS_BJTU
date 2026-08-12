"""Deterministic v3 diffusion data-plane pre-validation scenarios.

The harness calls the production v3 Worker admission, SQLite/CAS data plane,
and coordinator cleanup APIs directly.  Each scenario owns a temporary local
state directory; it never opens a listener, contacts a peer, or loads a model.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from diffusion.coordinator_runtime import DiffusionCoordinatorRuntime
from diffusion.data_plane import DiffusionDataPlaneRuntime
from diffusion.distributed import BlobAuthorizationError, BlobConflict, BlobNotFound
from diffusion.worker_adapter import (
    DiffusionCoordinatorControlPlane,
    DiffusionWorkerAdapter,
    RemoteDiffusionProvider,
)
from task_provider import ProviderUnavailable, StageAttempt, StageRequest
from task_worker_protocol import canonical_sha256


SIMULATION_SCHEMA_VERSION = "qlh.diffusion_data_plane_simulation.v1"


class SimulationScenarioError(ValueError):
    """Raised when a caller asks for an unknown v3 data-plane scenario."""


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    description: str


_SCENARIOS = {
    "manifest_mismatch_rejected": SimulationScenario(
        "manifest_mismatch_rejected",
        "a remote v3 Worker refuses a request whose exact artifact is absent",
    ),
    "chunk_replay_and_cas_dedup": SimulationScenario(
        "chunk_replay_and_cas_dedup",
        "chunk replay is idempotent and equal same-owner uploads share CAS state",
    ),
    "owner_scope_isolation": SimulationScenario(
        "owner_scope_isolation",
        "one distributed workflow cannot read or delete another workflow output",
    ),
    "cancelled_output_cleanup": SimulationScenario(
        "cancelled_output_cleanup",
        "coordinator cancellation cleanup removes only the uncommitted local output",
    ),
    "restart_recovery_reclaims_scope": SimulationScenario(
        "restart_recovery_reclaims_scope",
        "a recovered failed workflow revokes leases and reclaims its own CAS scope",
    ),
}


def available_scenarios() -> tuple[SimulationScenario, ...]:
    return tuple(_SCENARIOS[key] for key in sorted(_SCENARIOS))


class DiffusionDataPlaneSimulationHarness:
    """Run fixed v3 data-plane scenarios and return body-free evidence."""

    def run(self, scenario_id: str) -> dict:
        scenario = _SCENARIOS.get(str(scenario_id or ""))
        if scenario is None:
            raise SimulationScenarioError(f"unknown simulation scenario: {scenario_id}")
        details = getattr(self, f"_run_{scenario.scenario_id}")()
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "execution_environment": {
                "kind": "temporary_local_v3_data_plane_simulation",
                "network_io": False,
                "subprocesses_started": False,
                "real_model_loaded": False,
                "physical_nodes": False,
                "persistent_state_scope": "temporary_local_only",
            },
            "contract": details,
        }

    @staticmethod
    def _png() -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (8, 6), (20, 40, 80)).save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def _manifest() -> dict:
        value = {
            "artifact_id": "sim_sd15_artifact",
            "pipeline_kind": "sd15_pipeline",
            "revision": "simulation-v1",
            "components": [{
                "artifact_id": "sim_unet",
                "artifact_kind": "unet",
                "sha256": "a" * 64,
            }],
        }
        return {**value, "sha256": canonical_sha256(value)}

    @classmethod
    def _capabilities(cls) -> dict:
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
                "artifact_manifests": [cls._manifest()],
            },
        }

    @staticmethod
    def _runtime(state_dir: str | Path) -> DiffusionDataPlaneRuntime:
        return DiffusionDataPlaneRuntime.create(
            state_dir=state_dir,
            cluster_secret="s" * 32,
        )

    @classmethod
    def _put_output(
        cls,
        runtime: DiffusionDataPlaneRuntime,
        *,
        owner_scope: str,
    ):
        payload = cls._png()
        return runtime.store.put_bytes(
            payload,
            content_type="image/png",
            purpose="output",
            owner_scope=owner_scope,
            width=8,
            height=6,
        )

    @classmethod
    def _admitted_provider(cls) -> RemoteDiffusionProvider:
        outbound = []
        worker = DiffusionWorkerAdapter(
            node_id="sim-diffusion-v3",
            capabilities=cls._capabilities(),
            executor=lambda _payload, _cancel_event: None,
            send_message=outbound.append,
            clock=lambda: 1.0,
        )
        control = DiffusionCoordinatorControlPlane(clock=lambda: 1.0)
        hello = worker.begin_hello(sent_at_ms=1_000)
        assert hello is not None
        acknowledgement = control.receive_hello(
            "sim-diffusion-v3",
            hello.snapshot(),
            coordinator_node_id="sim-master",
            sent_at_ms=1_001,
        )
        worker.receive_hello_ack(acknowledgement.snapshot())
        return RemoteDiffusionProvider(
            node_id="sim-diffusion-v3",
            peer_snapshot=lambda: control.worker_snapshot("sim-diffusion-v3"),
            send_message=outbound.append,
            result_ingestor=lambda _attempt, output, _plan: dict(output),
            dispatch_enabled=True,
        )

    @classmethod
    def _request(cls, provider_id: str, manifest: dict) -> StageRequest:
        return StageRequest(
            workflow_id="wf_simv3manifest01",
            request_id="request_simv3manifest01",
            stage_id="image_stage",
            stage_type="image_generate",
            provider_id=provider_id,
            dependencies={},
            root_input={},
            runtime_context={"diffusion_artifact_manifest": manifest},
        )

    @staticmethod
    def _store_summary(runtime: DiffusionDataPlaneRuntime) -> dict:
        snapshot = runtime.store.snapshot()
        return {
            "blobs": snapshot["blobs"],
            "objects": snapshot["objects"],
            "uploads": snapshot["uploads"],
            "active_leases": snapshot["active_leases"],
        }

    def _run_manifest_mismatch_rejected(self) -> dict:
        provider = self._admitted_provider()
        try:
            incompatible = {**self._manifest(), "sha256": "f" * 64}
            try:
                provider.reserve(self._request(provider.provider_id, incompatible))
            except ProviderUnavailable as exc:
                rejected_code = exc.code
            else:
                raise RuntimeError("unadvertised artifact manifest was reserved")
            return {
                "rejected_codes": [rejected_code],
                "provider": {
                    "provider_id": provider.provider_id,
                    "provider_kind": provider.provider_kind,
                    "node_id": provider.node_id,
                    "active_reservations": provider.inspect().active_reservations,
                    "healthy": provider.inspect().healthy,
                },
            }
        finally:
            provider.close()

    def _run_chunk_replay_and_cas_dedup(self) -> dict:
        payload = self._png()
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory(prefix="qlh-sim-v3-") as state_dir:
            runtime = self._runtime(state_dir)
            try:
                fields = {
                    "expected_sha256": digest,
                    "expected_size": len(payload),
                    "content_type": "image/png",
                    "purpose": "output",
                    "owner_scope": "distributed:wf_simv3cas01",
                    "width": 8,
                    "height": 6,
                }
                first = runtime.store.begin_upload(**fields)
                split = len(payload) // 2
                runtime.store.write_upload(first.upload_id, offset=0, data=payload[:split])
                replay = runtime.store.write_upload(
                    first.upload_id, offset=0, data=payload[:split],
                )
                changed = bytes([payload[0] ^ 1]) + payload[1:split]
                try:
                    runtime.store.write_upload(first.upload_id, offset=0, data=changed)
                except BlobConflict as exc:
                    rejected_code = exc.code
                else:
                    raise RuntimeError("conflicting chunk replay was accepted")
                runtime.store.write_upload(first.upload_id, offset=split, data=payload[split:])
                committed = runtime.store.commit_upload(first.upload_id)
                replayed_commit = runtime.store.commit_upload(first.upload_id)
                duplicate = runtime.store.begin_upload(**fields)
                runtime.store.write_upload(duplicate.upload_id, offset=0, data=payload)
                deduplicated = runtime.store.commit_upload(duplicate.upload_id)
                cleanup = runtime.store.delete_owner_scope(
                    fields["owner_scope"], purpose="output",
                )
                if cleanup["blobs_removed"] != 1:
                    raise RuntimeError("simulated CAS owner cleanup did not converge")
                summary = self._store_summary(runtime)
                return {
                    "rejected_codes": [rejected_code],
                    "chunk_replay_idempotent": replay.received_bytes == split,
                    "commit_replay_idempotent": (
                        committed.blob_id == replayed_commit.blob_id
                    ),
                    "cas_deduplicated": committed.blob_id == deduplicated.blob_id,
                    "store": summary,
                }
            finally:
                runtime.close()

    def _run_owner_scope_isolation(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="qlh-sim-v3-") as state_dir:
            runtime = self._runtime(state_dir)
            try:
                owner_a = "distributed:wf_simv3owner_a"
                owner_b = "distributed:wf_simv3owner_b"
                blob_a = self._put_output(runtime, owner_scope=owner_a)
                blob_b = self._put_output(runtime, owner_scope=owner_b)
                coordinator = DiffusionCoordinatorRuntime(data_plane=runtime)
                _descriptor, data = coordinator.read_result(
                    workflow_id="wf_simv3owner_a", blob_id=blob_a.blob_id,
                )
                try:
                    coordinator.read_result(
                        workflow_id="wf_simv3owner_b", blob_id=blob_a.blob_id,
                    )
                except BlobAuthorizationError as exc:
                    rejected_code = exc.code
                else:
                    raise RuntimeError("foreign workflow read was accepted")
                cleanup_a = coordinator.discard_workflow("wf_simv3owner_a")
                still_owned = runtime.store.descriptor_for_owner(
                    blob_b.blob_id, owner_scope=owner_b,
                )
                coordinator.discard_workflow("wf_simv3owner_b")
                return {
                    "rejected_codes": [rejected_code],
                    "owned_read_verified": bool(data),
                    "cleanup": {
                        "owner_a_removed": cleanup_a["blobs_removed"],
                        "owner_b_survived_foreign_cleanup": bool(still_owned.blob_id),
                    },
                    "store": self._store_summary(runtime),
                }
            finally:
                runtime.close()

    def _run_cancelled_output_cleanup(self) -> dict:
        workflow_id = "wf_simv3cancel01"
        with tempfile.TemporaryDirectory(prefix="qlh-sim-v3-") as state_dir:
            runtime = self._runtime(state_dir)
            try:
                descriptor = self._put_output(
                    runtime, owner_scope=f"distributed:{workflow_id}",
                )
                request = StageRequest(
                    workflow_id=workflow_id,
                    request_id="request_simv3cancel01",
                    stage_id="image_stage",
                    stage_type="image_generate",
                    provider_id="remote_diffusion_sim",
                    dependencies={},
                    root_input={},
                )
                attempt = StageAttempt(
                    attempt_id="att_simv3cancel01",
                    request=request,
                    provider_id=request.provider_id,
                    lease_id="lease_simv3cancel01",
                    lease_epoch=1,
                    lease_expires_at=10.0,
                )
                coordinator = DiffusionCoordinatorRuntime(data_plane=runtime)
                coordinator.discard_result(attempt, {"image": descriptor.snapshot()})
                coordinator.discard_result(attempt, {"image": descriptor.snapshot()})
                try:
                    runtime.store.descriptor(descriptor.blob_id)
                except BlobNotFound as exc:
                    removed_code = exc.code
                else:
                    raise RuntimeError("cancelled output was retained")
                return {
                    "terminal_state": "cancelled",
                    "removed_codes": [removed_code],
                    "cleanup_idempotent": True,
                    "store": self._store_summary(runtime),
                }
            finally:
                runtime.close()

    def _run_restart_recovery_reclaims_scope(self) -> dict:
        workflow_id = "wf_simv3restart01"
        with tempfile.TemporaryDirectory(prefix="qlh-sim-v3-") as state_dir:
            runtime = self._runtime(state_dir)
            descriptor = self._put_output(
                runtime, owner_scope=f"distributed:{workflow_id}",
            )
            runtime.store.acquire_lease(
                descriptor.blob_id,
                attempt_id="att_simv3restart01",
                ttl_seconds=60.0,
            )
            runtime.close()

            recovered = self._runtime(state_dir)
            try:
                summary = DiffusionCoordinatorRuntime(
                    data_plane=recovered,
                ).reconcile_recovered_workflows([{
                    "workflow_id": workflow_id,
                    "template": "image_generate_v1",
                    "state": "failed",
                    "recovered_after_restart": True,
                    "recovery_reason": "coordinator_restarted_before_result_commit",
                }])
                try:
                    recovered.store.descriptor(descriptor.blob_id)
                except BlobNotFound as exc:
                    removed_code = exc.code
                else:
                    raise RuntimeError("recovered workflow output was retained")
                return {
                    "removed_codes": [removed_code],
                    "recovery": summary,
                    "store": self._store_summary(recovered),
                }
            finally:
                recovered.close()
