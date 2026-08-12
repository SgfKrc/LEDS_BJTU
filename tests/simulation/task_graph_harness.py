"""Deterministic, in-memory pre-validation for distributed task graphs.

The harness deliberately exercises the production TaskGraph and Provider
contracts without starting a server, opening a socket, or loading a model.
Its reports are planning evidence, never physical-node acceptance evidence.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
import sys

# Simulation runners are also importable from default pytest collection.
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from task_graph import (
    StageSpec,
    TaskGraphCoordinator,
    WorkflowCancelled,
)
from task_provider import (
    DeterministicFakeProvider,
    ModelIdentity,
    ProviderRegistry,
    ProviderUnavailable,
    StageRequest,
)


SIMULATION_SCHEMA_VERSION = "qlh.task_graph_simulation.v1"
_START_TIMEOUT_SECONDS = 1.0
_JOIN_TIMEOUT_SECONDS = 2.0


class SimulationScenarioError(ValueError):
    """Raised when a caller asks for an unknown simulation scenario."""


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    description: str
    expected_workflow_state: str


_SCENARIOS = {
    "dual_remote_success": SimulationScenario(
        scenario_id="dual_remote_success",
        description="two exact-identity remote candidates and local aggregation",
        expected_workflow_state="completed",
    ),
    "fallback_after_worker_loss": SimulationScenario(
        scenario_id="fallback_after_worker_loss",
        description="retry-safe stage falls back after one worker execution failure",
        expected_workflow_state="completed",
    ),
    "cancellation_during_worker_stage": SimulationScenario(
        scenario_id="cancellation_during_worker_stage",
        description="cancellation fences a running worker attempt",
        expected_workflow_state="cancelled",
    ),
}


def available_scenarios() -> tuple[SimulationScenario, ...]:
    """Return the fixed, safe scenario catalog in deterministic order."""
    return tuple(_SCENARIOS[key] for key in sorted(_SCENARIOS))


class _SimulatedRemoteWorker(DeterministicFakeProvider):
    """A fake provider that preserves the v2 exact-model selection boundary."""

    def __init__(self, provider_id: str, *, identity: ModelIdentity, **kwargs):
        self._identity = identity
        super().__init__(provider_id, **kwargs)

    def model_identities(self) -> tuple[ModelIdentity, ...]:
        return (self._identity,)

    def supports_model_identity(
        self,
        identity: ModelIdentity,
        stage_type: str,
    ) -> bool:
        return (
            identity == self._identity
            and stage_type in self.inspect().supported_stage_types
        )

    def reserve(self, request: StageRequest):
        if not self.supports_model_identity(
            request.model_identity,
            request.stage_type,
        ):
            raise ProviderUnavailable(
                "simulated remote worker model identity mismatch",
                code="model_identity_mismatch",
                provider_id=self.provider_id,
                retryable=True,
            )
        return super().reserve(request)


class TaskGraphSimulationHarness:
    """Run fixed L1 pre-validation scenarios against production contracts."""

    def run(self, scenario_id: str) -> dict:
        scenario = _SCENARIOS.get(str(scenario_id or ""))
        if scenario is None:
            raise SimulationScenarioError(f"unknown simulation scenario: {scenario_id}")
        if scenario.scenario_id == "dual_remote_success":
            return self._run_dual_remote_success(scenario)
        if scenario.scenario_id == "fallback_after_worker_loss":
            return self._run_fallback_after_worker_loss(scenario)
        return self._run_cancellation_during_worker_stage(scenario)

    @staticmethod
    def _identity(model_id: str, suffix: str) -> ModelIdentity:
        return ModelIdentity(
            model_id=model_id,
            engine="pytorch",
            format="safetensors",
            revision="simulation-v1",
            sha256=suffix * 64,
        )

    def _run_dual_remote_success(self, scenario: SimulationScenario) -> dict:
        identity_a = self._identity("sim-qwen-a", "a")
        identity_b = self._identity("sim-qwen-b", "b")
        workers = [
            _SimulatedRemoteWorker(
                "sim-worker-a",
                identity=identity_a,
                node_id="sim-node-a",
                supported_stage_types=("full_inference",),
                output_factory=lambda _request, _cancel: {
                    "content": "simulation-output-a",
                },
            ),
            _SimulatedRemoteWorker(
                "sim-worker-b",
                identity=identity_b,
                node_id="sim-node-b",
                supported_stage_types=("full_inference",),
                output_factory=lambda _request, _cancel: {
                    "content": "simulation-output-b",
                },
            ),
            DeterministicFakeProvider(
                "sim-master-aggregate",
                node_id="sim-master",
                supported_stage_types=("aggregate",),
                output_factory=lambda _request, _cancel: {
                    "content": "simulation-output-final",
                },
            ),
        ]
        stages = [
            StageSpec(
                "candidate_a",
                "full_inference",
                provider="sim-worker-a",
                model_identity=identity_a,
                pure=True,
            ),
            StageSpec(
                "candidate_b",
                "full_inference",
                provider="sim-worker-b",
                model_identity=identity_b,
                pure=True,
            ),
            StageSpec(
                "aggregate",
                "aggregate",
                depends_on=("candidate_a", "candidate_b"),
                provider="sim-master-aggregate",
                minimum_successful_dependencies=2,
            ),
        ]
        return self._run_and_commit(scenario, stages, "aggregate", workers)

    def _run_fallback_after_worker_loss(
        self,
        scenario: SimulationScenario,
    ) -> dict:
        identity = self._identity("sim-qwen-fallback", "c")
        workers = [
            _SimulatedRemoteWorker(
                "sim-worker-primary",
                identity=identity,
                node_id="sim-node-primary",
                supported_stage_types=("full_inference",),
                execution_failures=1,
            ),
            _SimulatedRemoteWorker(
                "sim-worker-fallback",
                identity=identity,
                node_id="sim-node-fallback",
                supported_stage_types=("full_inference",),
                output_factory=lambda _request, _cancel: {
                    "content": "simulation-output-recovered",
                },
            ),
        ]
        stages = [
            StageSpec(
                "answer",
                "full_inference",
                provider="sim-worker-primary",
                fallback_providers=("sim-worker-fallback",),
                model_identity=identity,
                pure=True,
            ),
        ]
        return self._run_and_commit(scenario, stages, "answer", workers)

    def _run_cancellation_during_worker_stage(
        self,
        scenario: SimulationScenario,
    ) -> dict:
        identity = self._identity("sim-qwen-cancel", "d")
        release = threading.Event()
        worker = _SimulatedRemoteWorker(
            "sim-worker-cancel",
            identity=identity,
            node_id="sim-node-cancel",
            supported_stage_types=("full_inference",),
            block_event=release,
        )
        stages = [
            StageSpec(
                "answer",
                "full_inference",
                provider=worker.provider_id,
                model_identity=identity,
            ),
        ]
        registry = ProviderRegistry()
        registry.register(worker)
        coordinator = TaskGraphCoordinator(provider_registry=registry)
        workflow_id = "wf_simcancel01"
        outcome: list[BaseException] = []

        def execute() -> None:
            try:
                coordinator.run(
                    stages=stages,
                    final_stage_id="answer",
                    root_input={},
                    template="simulation_cancel_v1",
                    workflow_id=workflow_id,
                )
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        try:
            self._wait_for_worker_call(worker)
            coordinator.request_cancel(workflow_id)
            thread.join(_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError("simulated cancellation did not converge")
            if not outcome or not isinstance(outcome[0], WorkflowCancelled):
                raise RuntimeError("simulated cancellation did not raise WorkflowCancelled")
            snapshot = coordinator.get(workflow_id)
            evidence = self._build_evidence(scenario, snapshot, coordinator, [worker])
            self._assert_expected_state(scenario, evidence)
            return evidence
        finally:
            release.set()
            if thread.is_alive():
                thread.join(_JOIN_TIMEOUT_SECONDS)
            coordinator.close()

    def _run_and_commit(
        self,
        scenario: SimulationScenario,
        stages: list[StageSpec],
        final_stage_id: str,
        providers: list[DeterministicFakeProvider],
    ) -> dict:
        registry = ProviderRegistry()
        for provider in providers:
            registry.register(provider)
        coordinator = TaskGraphCoordinator(
            provider_registry=registry,
            max_parallel_stages=2,
        )
        workflow_id = f"wf_sim{scenario.scenario_id[:20]}"
        try:
            _output, snapshot = coordinator.run(
                stages=stages,
                final_stage_id=final_stage_id,
                root_input={},
                template=f"simulation_{scenario.scenario_id}_v1",
                workflow_id=workflow_id,
            )
            snapshot = coordinator.commit_result(snapshot["workflow_id"])
            evidence = self._build_evidence(scenario, snapshot, coordinator, providers)
            self._assert_expected_state(scenario, evidence)
            return evidence
        finally:
            coordinator.close()

    @staticmethod
    def _wait_for_worker_call(worker: DeterministicFakeProvider) -> None:
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if worker.call_records():
                return
            time.sleep(0.005)
        raise RuntimeError("simulated worker did not start")

    @staticmethod
    def _build_evidence(
        scenario: SimulationScenario,
        snapshot: dict,
        coordinator: TaskGraphCoordinator,
        providers: list[DeterministicFakeProvider],
    ) -> dict:
        provider_status = {
            item["provider_id"]: item for item in coordinator.provider_status()
        }
        stages = []
        for stage in snapshot.get("stages", []):
            attempts = [
                {
                    "provider_id": attempt["provider"],
                    "provider_kind": attempt["provider_kind"],
                    "node_id": attempt["provider_node_id"],
                    "state": attempt["state"],
                    "lease_epoch": attempt["lease_epoch"],
                }
                for attempt in stage.get("attempts", [])
            ]
            stages.append({
                "stage_id": stage["stage_id"],
                "stage_type": stage["stage_type"],
                "state": stage["state"],
                "requested_provider_id": stage["requested_provider"],
                "selected_provider_id": stage["selected_provider"],
                "attempt_count": len(attempts),
                "retry_count": stage["retry_count"],
                "last_retry_error_code": stage["last_retry_error_code"],
                "output_available": stage["output_available"],
                "attempts": attempts,
            })
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "execution_environment": {
                "kind": "in_memory_provider_simulation",
                "network_io": False,
                "subprocesses_started": False,
                "real_model_loaded": False,
                "physical_nodes": False,
            },
            "workflow": {
                "template": snapshot["template"],
                "state": snapshot["state"],
                "stage_count": snapshot["stage_count"],
                "attempt_count": snapshot["attempt_count"],
                "retry_count": snapshot["retry_count"],
                "partial_result": snapshot["partial_result"],
                "cancel_requested": snapshot["cancel_requested"],
                "stages": stages,
            },
            "providers": [
                {
                    "provider_id": provider.provider_id,
                    "provider_kind": provider_status[provider.provider_id]["provider_kind"],
                    "node_id": provider_status[provider.provider_id]["node_id"],
                    "invocation_count": len(provider.call_records()),
                    "active_reservations": provider_status[provider.provider_id]["active_reservations"],
                    "healthy": provider_status[provider.provider_id]["healthy"],
                }
                for provider in providers
            ],
        }

    @staticmethod
    def _assert_expected_state(
        scenario: SimulationScenario,
        evidence: dict,
    ) -> None:
        state = evidence["workflow"]["state"]
        if state != scenario.expected_workflow_state:
            raise RuntimeError(
                f"scenario {scenario.scenario_id} ended in {state}, "
                f"expected {scenario.expected_workflow_state}"
            )
