"""Deterministic bounded-capacity pre-validation scenarios.

The harness exercises production TaskGraph reservation and fallback handling,
plus the diffusion SQLite/CAS capacity cleanup path.  Gate events determine
progress; no outcome depends on elapsed-time or throughput measurements.
"""

from __future__ import annotations

import io
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from diffusion.data_plane import DiffusionDataPlaneRuntime
from diffusion.distributed import BlobConflict
from task_graph import StageSpec, TaskGraphCoordinator, WorkflowCancelled
from task_provider import (
    DeterministicFakeProvider,
    LocalFullModelProvider,
    ProviderExecutionError,
    ProviderRegistry,
    StageRequest,
)


SIMULATION_SCHEMA_VERSION = "qlh.capacity_simulation.v1"
_WAIT_SECONDS = 2.0


class SimulationScenarioError(ValueError):
    """Raised when a caller asks for an unknown bounded-capacity scenario."""


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    description: str


_SCENARIOS = {
    "global_parallel_bound": SimulationScenario(
        "global_parallel_bound",
        "three ready stages admit only the configured two concurrent slots",
    ),
    "single_provider_serialization": SimulationScenario(
        "single_provider_serialization",
        "two ready stages sharing a one-slot Provider execute in bounded order",
    ),
    "busy_reservation_fallback": SimulationScenario(
        "busy_reservation_fallback",
        "an externally occupied primary slot retries on a pure fallback Provider",
    ),
    "parallel_cancellation_releases_slots": SimulationScenario(
        "parallel_cancellation_releases_slots",
        "cancellation fences active stages and converges every Provider slot",
    ),
    "cas_capacity_recovery": SimulationScenario(
        "cas_capacity_recovery",
        "a leased object blocks capacity until explicit lease-aware cleanup",
    ),
}


def available_scenarios() -> tuple[SimulationScenario, ...]:
    return tuple(_SCENARIOS[key] for key in sorted(_SCENARIOS))


class _ActiveTracker:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


class _GateProvider(LocalFullModelProvider):
    """A single-slot production Provider whose completion is event-driven."""

    def __init__(
        self,
        provider_id: str,
        *,
        tracker: _ActiveTracker,
        max_concurrency: int = 1,
    ) -> None:
        self._tracker = tracker
        self._permits = threading.Semaphore(0)
        self._started: list[str] = []
        self._started_condition = threading.Condition()

        def execute(request, cancel_event):
            with self._started_condition:
                self._started.append(request.stage_id)
                self._started_condition.notify_all()
            self._tracker.enter()
            try:
                while not self._permits.acquire(timeout=0.01):
                    if cancel_event.is_set():
                        raise ProviderExecutionError(
                            "simulated capacity stage was cancelled",
                            code="provider_cancelled",
                            provider_id=provider_id,
                        )
                if cancel_event.is_set():
                    raise ProviderExecutionError(
                        "simulated capacity stage was cancelled",
                        code="provider_cancelled",
                        provider_id=provider_id,
                    )
                return {"content": request.stage_id}
            finally:
                self._tracker.leave()

        super().__init__(
            execute,
            provider_id=provider_id,
            node_id=f"node-{provider_id}",
            supported_stage_types=("full_inference",),
            max_concurrency=max_concurrency,
            provider_kind="deterministic_capacity_provider",
        )

    def allow(self, count: int = 1) -> None:
        for _ in range(max(0, int(count))):
            self._permits.release()

    def wait_for_started(self, count: int) -> None:
        with self._started_condition:
            if self._started_condition.wait_for(
                lambda: len(self._started) >= count,
                timeout=_WAIT_SECONDS,
            ):
                return
        raise RuntimeError("simulated capacity stage did not start")

    def started_stage_ids(self) -> list[str]:
        with self._started_condition:
            return list(self._started)


class CapacitySimulationHarness:
    """Run fixed capacity scenarios and emit summary-only evidence."""

    def run(self, scenario_id: str) -> dict:
        scenario = _SCENARIOS.get(str(scenario_id or ""))
        if scenario is None:
            raise SimulationScenarioError(f"unknown simulation scenario: {scenario_id}")
        details = getattr(self, f"_run_{scenario.scenario_id}")()
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "execution_environment": {
                "kind": "temporary_local_capacity_simulation",
                "network_io": False,
                "subprocesses_started": False,
                "real_model_loaded": False,
                "physical_nodes": False,
                "persistent_state_scope": "temporary_local_only",
                "performance_claim": False,
            },
            "contract": details,
        }

    @staticmethod
    def _run_in_thread(callable_):
        outcome = []
        thread = threading.Thread(
            target=lambda: CapacitySimulationHarness._capture(outcome, callable_),
            daemon=True,
        )
        thread.start()
        return thread, outcome

    @staticmethod
    def _capture(outcome: list, callable_) -> None:
        try:
            outcome.append(("result", callable_()))
        except BaseException as exc:
            outcome.append(("error", exc))

    @staticmethod
    def _join(thread: threading.Thread) -> None:
        thread.join(_WAIT_SECONDS)
        if thread.is_alive():
            raise RuntimeError("simulated capacity workflow did not converge")

    @staticmethod
    def _status_summary(coordinator: TaskGraphCoordinator) -> list[dict]:
        return [
            {
                "provider_id": status["provider_id"],
                "active_reservations": status["active_reservations"],
                "healthy": status["healthy"],
            }
            for status in coordinator.provider_status()
        ]

    @staticmethod
    def _assert_idle(statuses: list[dict]) -> None:
        if any(status["active_reservations"] != 0 for status in statuses):
            raise RuntimeError("simulated capacity slots did not converge")

    @staticmethod
    def _png(color: tuple[int, int, int]) -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(output, format="PNG")
        return output.getvalue()

    def _run_global_parallel_bound(self) -> dict:
        tracker = _ActiveTracker()
        registry = ProviderRegistry()
        providers = [
            _GateProvider(f"sim_global_{suffix}", tracker=tracker)
            for suffix in ("a", "b", "c")
        ]
        for provider in providers:
            registry.register(provider)
        coordinator = TaskGraphCoordinator(
            provider_registry=registry,
            max_parallel_stages=2,
        )
        try:
            stages = [
                StageSpec("stage_a", "full_inference", provider=providers[0].provider_id),
                StageSpec("stage_b", "full_inference", provider=providers[1].provider_id),
                StageSpec("stage_c", "full_inference", provider=providers[2].provider_id),
            ]
            thread, outcome = self._run_in_thread(lambda: coordinator.run(
                stages, "stage_c", {}, workflow_id="wf_simn5global01",
            ))
            providers[0].wait_for_started(1)
            providers[1].wait_for_started(1)
            if providers[2].started_stage_ids():
                raise RuntimeError("global parallel limit admitted a third active stage")
            providers[0].allow()
            providers[1].allow()
            providers[2].wait_for_started(1)
            providers[2].allow()
            self._join(thread)
            if outcome[0][0] != "result":
                raise outcome[0][1]
            _output, workflow = outcome[0][1]
            statuses = self._status_summary(coordinator)
            self._assert_idle(statuses)
            return {
                "terminal_state": workflow["state"],
                "configured_global_parallel_stages": 2,
                "observed_active_peak": tracker.maximum,
                "third_stage_deferred_until_capacity_released": True,
                "providers": statuses,
            }
        finally:
            for provider in providers:
                provider.allow(2)
            coordinator.close()

    def _run_single_provider_serialization(self) -> dict:
        tracker = _ActiveTracker()
        shared = _GateProvider("sim_shared_slot", tracker=tracker)
        registry = ProviderRegistry()
        registry.register(shared)
        coordinator = TaskGraphCoordinator(
            provider_registry=registry,
            max_parallel_stages=3,
        )
        try:
            stages = [
                StageSpec("stage_a", "full_inference", provider=shared.provider_id),
                StageSpec("stage_b", "full_inference", provider=shared.provider_id),
            ]
            thread, outcome = self._run_in_thread(lambda: coordinator.run(
                stages, "stage_b", {}, workflow_id="wf_simn5serial01",
            ))
            shared.wait_for_started(1)
            before_release = shared.started_stage_ids()
            shared.allow()
            shared.wait_for_started(2)
            shared.allow()
            self._join(thread)
            if outcome[0][0] != "result":
                raise outcome[0][1]
            _output, workflow = outcome[0][1]
            statuses = self._status_summary(coordinator)
            self._assert_idle(statuses)
            return {
                "terminal_state": workflow["state"],
                "provider_max_concurrency": 1,
                "observed_active_peak": tracker.maximum,
                "second_stage_deferred_until_slot_released": before_release == ["stage_a"],
                "providers": statuses,
            }
        finally:
            shared.allow(2)
            coordinator.close()

    def _run_busy_reservation_fallback(self) -> dict:
        registry = ProviderRegistry()
        primary = DeterministicFakeProvider("sim_busy_primary", max_concurrency=1)
        fallback = DeterministicFakeProvider("sim_busy_fallback", max_concurrency=1)
        registry.register(primary)
        registry.register(fallback)
        coordinator = TaskGraphCoordinator(provider_registry=registry)
        held = primary.reserve(StageRequest(
            workflow_id="wf_simn5hold01",
            request_id="request_simn5hold01",
            stage_id="held_stage",
            stage_type="full_inference",
            provider_id=primary.provider_id,
            dependencies={},
            root_input={},
        ))
        try:
            _output, workflow = coordinator.run(
                [StageSpec(
                    "answer",
                    "full_inference",
                    provider=primary.provider_id,
                    fallback_providers=(fallback.provider_id,),
                    pure=True,
                )],
                "answer",
                {},
                workflow_id="wf_simn5fallback01",
            )
            stage = workflow["stages"][0]
            if primary.call_records():
                raise RuntimeError("busy primary executed despite reservation saturation")
            primary.release(held.reservation_id)
            statuses = self._status_summary(coordinator)
            self._assert_idle(statuses)
            return {
                "terminal_state": workflow["state"],
                "retry_count": stage["retry_count"],
                "last_retry_error_code": stage["last_retry_error_code"],
                "primary_execution_count": len(primary.call_records()),
                "fallback_execution_count": len(fallback.call_records()),
                "selected_provider": stage["selected_provider"],
                "providers": statuses,
            }
        finally:
            primary.release(held.reservation_id)
            coordinator.close()

    def _run_parallel_cancellation_releases_slots(self) -> dict:
        tracker = _ActiveTracker()
        registry = ProviderRegistry()
        providers = [
            _GateProvider(f"sim_cancel_{suffix}", tracker=tracker)
            for suffix in ("a", "b")
        ]
        for provider in providers:
            registry.register(provider)
        coordinator = TaskGraphCoordinator(
            provider_registry=registry,
            max_parallel_stages=2,
        )
        try:
            stages = [
                StageSpec("stage_a", "full_inference", provider=providers[0].provider_id),
                StageSpec("stage_b", "full_inference", provider=providers[1].provider_id),
            ]
            thread, outcome = self._run_in_thread(lambda: coordinator.run(
                stages, "stage_b", {}, workflow_id="wf_simn5cancel01",
            ))
            providers[0].wait_for_started(1)
            providers[1].wait_for_started(1)
            active_before_cancel = [
                status["active_reservations"]
                for status in self._status_summary(coordinator)
            ]
            if not coordinator.request_cancel("wf_simn5cancel01"):
                raise RuntimeError("simulated capacity cancellation was not accepted")
            self._join(thread)
            if outcome[0][0] != "error" or not isinstance(
                outcome[0][1], WorkflowCancelled,
            ):
                raise RuntimeError("simulated capacity cancellation did not fence run")
            workflow = coordinator.get("wf_simn5cancel01")
            statuses = self._status_summary(coordinator)
            self._assert_idle(statuses)
            return {
                "terminal_state": workflow["state"],
                "active_slots_before_cancel": active_before_cancel,
                "observed_active_peak": tracker.maximum,
                "providers": statuses,
            }
        finally:
            for provider in providers:
                provider.allow(2)
            coordinator.close()

    def _run_cas_capacity_recovery(self) -> dict:
        first = self._png((20, 40, 80))
        second = self._png((90, 30, 10))
        now = [100.0]
        options = {
            "max_blob_bytes": max(len(first), len(second)),
            "max_total_bytes": len(first) + len(second) - 1,
            "upload_ttl_seconds": 1.0,
        }
        with tempfile.TemporaryDirectory(prefix="qlh-sim-capacity-") as state_dir:
            runtime = DiffusionDataPlaneRuntime.create(
                state_dir=state_dir,
                cluster_secret="s" * 32,
                store_options=options,
                clock=lambda: now[0],
            )
            try:
                first_descriptor = runtime.store.put_bytes(
                    first,
                    content_type="image/png",
                    purpose="output",
                    owner_scope="distributed:wf_simn5capacity_a",
                    width=8,
                    height=6,
                )
                lease = runtime.store.acquire_lease(
                    first_descriptor.blob_id,
                    attempt_id="att_simn5capacity01",
                    ttl_seconds=60.0,
                )
                try:
                    runtime.store.put_bytes(
                        second,
                        content_type="image/png",
                        purpose="output",
                        owner_scope="distributed:wf_simn5capacity_b",
                        width=8,
                        height=6,
                    )
                except BlobConflict as exc:
                    rejected_code = exc.code
                else:
                    raise RuntimeError("leased CAS object did not block bounded capacity")
                runtime.store.release_lease(lease.lease_id)
                cleanup = runtime.store.delete_owner_scope(
                    "distributed:wf_simn5capacity_a", purpose="output",
                )
                now[0] += 2.0
                runtime.store.cleanup()
                snapshot = runtime.store.snapshot()
                summary = {
                    "blobs": snapshot["blobs"],
                    "objects": snapshot["objects"],
                    "uploads": snapshot["uploads"],
                    "active_leases": snapshot["active_leases"],
                }
                if any(summary.values()):
                    raise RuntimeError("simulated CAS capacity cleanup did not converge")
                return {
                    "rejected_codes": [rejected_code],
                    "cleanup": {
                        "blobs_removed": cleanup["blobs_removed"],
                        "leases_revoked": cleanup["leases_revoked"],
                    },
                    "store": summary,
                }
            finally:
                runtime.close()
