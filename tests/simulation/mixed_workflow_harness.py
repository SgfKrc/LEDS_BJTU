"""Deterministic pre-validation for the fixed ``llm_sd15_v1`` workflow.

The scenarios call the production mixed-workflow entry point with temporary
in-process Providers and a temporary SQLite/CAS store.  They never start an
HTTP listener, open a peer connection, or execute a real model.
"""

from __future__ import annotations

import io
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import api_server
from diffusion.coordinator_runtime import DiffusionCoordinatorRuntime
from diffusion.data_plane import DiffusionDataPlaneRuntime
from task_graph import TaskGraphCoordinator
from task_provider import LocalFullModelProvider, ModelIdentity
from task_worker_protocol import canonical_sha256


SIMULATION_SCHEMA_VERSION = "qlh.mixed_workflow_simulation.v1"
_MISSING = object()
_JOIN_SECONDS = 2.0


class SimulationScenarioError(ValueError):
    """Raised when a caller asks for an unknown fixed mixed-workflow scenario."""


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    description: str


_SCENARIOS = {
    "successful_local_binding": SimulationScenario(
        "successful_local_binding",
        "the fixed v2 prompt result binds locally into a v3 root input",
    ),
    "text_model_ambiguity": SimulationScenario(
        "text_model_ambiguity",
        "multiple exact text identities require an explicit selection",
    ),
    "image_manifest_ambiguity": SimulationScenario(
        "image_manifest_ambiguity",
        "multiple SD artifacts require an explicit image Worker selection",
    ),
    "same_node_role_conflict": SimulationScenario(
        "same_node_role_conflict",
        "one node cannot provide both v2 text and v3 image roles",
    ),
    "text_contract_limit": SimulationScenario(
        "text_contract_limit",
        "an overlong generated prompt fences the image stage before dispatch",
    ),
    "cancelled_image_cleanup": SimulationScenario(
        "cancelled_image_cleanup",
        "cancellation reclaims an uncommitted image output and its scope",
    ),
}


def available_scenarios() -> tuple[SimulationScenario, ...]:
    return tuple(_SCENARIOS[key] for key in sorted(_SCENARIOS))


def _manifest(*, artifact_id: str, suffix: str) -> dict:
    value = {
        "artifact_id": artifact_id,
        "pipeline_kind": "sd15_pipeline",
        "revision": "simulation-v1",
        "components": [{
            "artifact_id": "sim_pipeline",
            "artifact_kind": "sd15_pipeline",
            "sha256": suffix * 64,
        }],
    }
    return {**value, "sha256": canonical_sha256(value)}


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), (20, 40, 80)).save(output, format="PNG")
    return output.getvalue()


class _TextProvider(LocalFullModelProvider):
    def __init__(
        self,
        *,
        provider_id: str,
        node_id: str,
        identity: ModelIdentity,
        content: str,
    ) -> None:
        self.node_id = node_id
        self.identity = identity
        self.content = content
        self.requests = []

        def execute(request, _cancel_event):
            self.requests.append(request)
            return {"content": self.content, "model": self.identity.model_id}

        super().__init__(
            execute,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=("image_prompt",),
            provider_kind="remote_full_worker",
        )

    def model_identities(self) -> tuple[ModelIdentity, ...]:
        return (self.identity,)

    def supports_model_identity(
        self,
        identity: ModelIdentity,
        stage_type: str,
    ) -> bool:
        return identity == self.identity and stage_type == "image_prompt"


class _ImageProvider(LocalFullModelProvider):
    def __init__(
        self,
        *,
        provider_id: str,
        node_id: str,
        manifest: dict,
        data_plane: DiffusionDataPlaneRuntime,
        block_after_write: bool = False,
    ) -> None:
        self.node_id = node_id
        self._manifest = dict(manifest)
        self._data_plane = data_plane
        self._block_after_write = block_after_write
        self.calls = []
        self.started = threading.Event()

        def execute(request, cancel_event):
            self.calls.append(request)
            descriptor = self._data_plane.store.put_bytes(
                _png(),
                content_type="image/png",
                purpose="output",
                owner_scope=f"distributed:{request.workflow_id}",
                width=8,
                height=6,
            )
            self.started.set()
            if self._block_after_write:
                if not cancel_event.wait(_JOIN_SECONDS):
                    raise RuntimeError("mixed simulation cancellation did not arrive")
                raise RuntimeError("cancelled")
            return {
                "image": descriptor.snapshot(),
                "metrics": {"seed": request.root_input["seed"]},
            }

        super().__init__(
            execute,
            provider_id=provider_id,
            node_id=node_id,
            supported_stage_types=("image_generate",),
            provider_kind="remote_diffusion_worker",
        )

    def artifact_manifests(self) -> tuple[dict, ...]:
        return (dict(self._manifest),)


class _MixedApiSandbox:
    """Temporarily install safe Worker fixtures into the production API module."""

    def __init__(self, text_providers, image_providers) -> None:
        self._text_providers = list(text_providers)
        self._image_providers = list(image_providers)
        self._temporary = tempfile.TemporaryDirectory(prefix="qlh-sim-mixed-")
        self.data_plane = DiffusionDataPlaneRuntime.create(
            state_dir=self._temporary.name,
            cluster_secret="s" * 32,
        )
        self.coordinator = TaskGraphCoordinator(max_records=12)
        self._saved: dict[str, object] = {}

    def __enter__(self):
        self._saved = {
            "TASK_GRAPH_ENABLED": api_server.TASK_GRAPH_ENABLED,
            "TASK_WORKER_EXPERIMENTAL_ENABLED": api_server.TASK_WORKER_EXPERIMENTAL_ENABLED,
            "DIFFUSION_WORKER_EXPERIMENTAL_ENABLED": api_server.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED,
            "task_graph_coordinator": api_server.task_graph_coordinator,
            "effective_role": api_server.scheduler._effective_role,
            "text_providers": api_server.scheduler.remote_task_worker_providers,
            "image_providers": api_server.scheduler.remote_diffusion_providers,
            "data_plane": getattr(api_server.app.state, "diffusion_data_plane", _MISSING),
        }
        api_server.TASK_GRAPH_ENABLED = True
        api_server.TASK_WORKER_EXPERIMENTAL_ENABLED = True
        api_server.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED = True
        api_server.task_graph_coordinator = self.coordinator
        api_server.scheduler._effective_role = lambda: "master"
        api_server.scheduler.remote_task_worker_providers = lambda: list(self._text_providers)
        api_server.scheduler.remote_diffusion_providers = lambda: list(self._image_providers)
        api_server.app.state.diffusion_data_plane = self.data_plane
        return self

    def __exit__(self, *_exc_info) -> None:
        api_server.TASK_GRAPH_ENABLED = self._saved["TASK_GRAPH_ENABLED"]
        api_server.TASK_WORKER_EXPERIMENTAL_ENABLED = self._saved[
            "TASK_WORKER_EXPERIMENTAL_ENABLED"
        ]
        api_server.DIFFUSION_WORKER_EXPERIMENTAL_ENABLED = self._saved[
            "DIFFUSION_WORKER_EXPERIMENTAL_ENABLED"
        ]
        api_server.task_graph_coordinator = self._saved["task_graph_coordinator"]
        api_server.scheduler._effective_role = self._saved["effective_role"]
        api_server.scheduler.remote_task_worker_providers = self._saved["text_providers"]
        api_server.scheduler.remote_diffusion_providers = self._saved["image_providers"]
        previous_data_plane = self._saved["data_plane"]
        if previous_data_plane is _MISSING:
            delattr(api_server.app.state, "diffusion_data_plane")
        else:
            api_server.app.state.diffusion_data_plane = previous_data_plane
        self.coordinator.close()
        self.data_plane.close()
        self._temporary.cleanup()


class MixedWorkflowSimulationHarness:
    """Run fixed LLM-to-SD failure scenarios and return body-free evidence."""

    def run(self, scenario_id: str) -> dict:
        scenario = _SCENARIOS.get(str(scenario_id or ""))
        if scenario is None:
            raise SimulationScenarioError(f"unknown simulation scenario: {scenario_id}")
        details = getattr(self, f"_run_{scenario.scenario_id}")()
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "execution_environment": {
                "kind": "temporary_local_mixed_workflow_simulation",
                "network_io": False,
                "subprocesses_started": False,
                "real_model_loaded": False,
                "physical_nodes": False,
                "persistent_state_scope": "temporary_local_only",
            },
            "contract": details,
        }

    @staticmethod
    def _identity(suffix: str) -> ModelIdentity:
        return ModelIdentity(
            model_id=f"sim-text-{suffix}",
            engine="pytorch",
            format="safetensors",
            revision="simulation-v1",
            sha256=suffix * 64,
        )

    @staticmethod
    def _store_summary(sandbox: _MixedApiSandbox) -> dict:
        snapshot = sandbox.data_plane.store.snapshot()
        return {
            "blobs": snapshot["blobs"],
            "objects": snapshot["objects"],
            "uploads": snapshot["uploads"],
            "active_leases": snapshot["active_leases"],
        }

    @staticmethod
    def _request(workflow_id: str, **overrides):
        fields = {
            "message": "simulated mixed request",
            "seed": 41,
            "width": 512,
            "height": 512,
            "steps": 4,
            "workflow_id": workflow_id,
        }
        fields.update(overrides)
        return api_server.DiffusionMixedGenerateRequest(**fields)

    def _providers(self, *, text_content="simulated visual prompt", image_node="sim-image-node"):
        text = _TextProvider(
            provider_id="sim_text_provider",
            node_id="sim-text-node",
            identity=self._identity("a"),
            content=text_content,
        )
        return text, image_node

    def _run_successful_local_binding(self) -> dict:
        text, image_node = self._providers()
        with _MixedApiSandbox([], []) as sandbox:
            image = _ImageProvider(
                provider_id="sim_image_provider",
                node_id=image_node,
                manifest=_manifest(artifact_id="sim_sd15", suffix="b"),
                data_plane=sandbox.data_plane,
            )
            sandbox._text_providers[:] = [text]
            sandbox._image_providers[:] = [image]
            result = api_server._run_distributed_mixed_generation(
                self._request("wf_simn4success01"),
            )
            observed = image.calls[0]
            prompt_stage = next(
                stage for stage in result["workflow"]["stages"]
                if stage["stage_id"] == "image_prompt"
            )
            cleanup = DiffusionCoordinatorRuntime(
                data_plane=sandbox.data_plane,
            ).discard_workflow("wf_simn4success01")
            return {
                "terminal_state": result["workflow"]["state"],
                "v3_dependencies_omitted": observed.dependencies == {},
                "prompt_bound_locally": bool(observed.root_input.get("prompt")),
                "prompt_output_sha256_present": bool(
                    result["prompt_contract"]["output_sha256"]
                ),
                "stage_binding_declared": bool(prompt_stage["output_available"]),
                "cleanup": {"blobs_removed": cleanup["blobs_removed"]},
                "store": self._store_summary(sandbox),
            }

    def _run_text_model_ambiguity(self) -> dict:
        text_a, _ = self._providers()
        text_b = _TextProvider(
            provider_id="sim_text_provider_b",
            node_id="sim-text-node-b",
            identity=self._identity("c"),
            content="unused",
        )
        with _MixedApiSandbox([], []) as sandbox:
            image = _ImageProvider(
                provider_id="sim_image_provider",
                node_id="sim-image-node",
                manifest=_manifest(artifact_id="sim_sd15", suffix="b"),
                data_plane=sandbox.data_plane,
            )
            sandbox._text_providers[:] = [text_a, text_b]
            sandbox._image_providers[:] = [image]
            try:
                api_server._run_distributed_mixed_generation(
                    self._request("wf_simn4textamb01"),
                )
            except HTTPException as exc:
                code = exc.detail["code"]
            else:
                raise RuntimeError("ambiguous text identities were accepted")
            return {
                "rejected_codes": [code],
                "image_dispatches": len(image.calls),
                "store": self._store_summary(sandbox),
            }

    def _run_image_manifest_ambiguity(self) -> dict:
        text, _ = self._providers()
        with _MixedApiSandbox([], []) as sandbox:
            image_a = _ImageProvider(
                provider_id="sim_image_provider_a",
                node_id="sim-image-node-a",
                manifest=_manifest(artifact_id="sim_sd15_a", suffix="b"),
                data_plane=sandbox.data_plane,
            )
            image_b = _ImageProvider(
                provider_id="sim_image_provider_b",
                node_id="sim-image-node-b",
                manifest=_manifest(artifact_id="sim_sd15_b", suffix="d"),
                data_plane=sandbox.data_plane,
            )
            sandbox._text_providers[:] = [text]
            sandbox._image_providers[:] = [image_a, image_b]
            try:
                api_server._run_distributed_mixed_generation(
                    self._request("wf_simn4imageamb01"),
                )
            except HTTPException as exc:
                code = exc.detail["code"]
            else:
                raise RuntimeError("ambiguous image artifacts were accepted")
            return {
                "rejected_codes": [code],
                "image_dispatches": len(image_a.calls) + len(image_b.calls),
                "store": self._store_summary(sandbox),
            }

    def _run_same_node_role_conflict(self) -> dict:
        text, _ = self._providers()
        with _MixedApiSandbox([], []) as sandbox:
            image = _ImageProvider(
                provider_id="sim_image_provider",
                node_id="sim-text-node",
                manifest=_manifest(artifact_id="sim_sd15", suffix="b"),
                data_plane=sandbox.data_plane,
            )
            sandbox._text_providers[:] = [text]
            sandbox._image_providers[:] = [image]
            try:
                api_server._run_distributed_mixed_generation(
                    self._request("wf_simn4roleconflict01"),
                )
            except HTTPException as exc:
                code = exc.detail["code"]
            else:
                raise RuntimeError("same-node mixed roles were accepted")
            return {
                "rejected_codes": [code],
                "image_dispatches": len(image.calls),
                "store": self._store_summary(sandbox),
            }

    def _run_text_contract_limit(self) -> dict:
        text, _ = self._providers(text_content="x" * 1001)
        with _MixedApiSandbox([], []) as sandbox:
            image = _ImageProvider(
                provider_id="sim_image_provider",
                node_id="sim-image-node",
                manifest=_manifest(artifact_id="sim_sd15", suffix="b"),
                data_plane=sandbox.data_plane,
            )
            sandbox._text_providers[:] = [text]
            sandbox._image_providers[:] = [image]
            try:
                api_server._run_distributed_mixed_generation(
                    self._request("wf_simn4textlimit01"),
                )
            except HTTPException as exc:
                code = exc.detail["code"]
            else:
                raise RuntimeError("overlong mixed prompt was accepted")
            snapshot = sandbox.coordinator.get("wf_simn4textlimit01")
            return {
                "terminal_state": snapshot["state"],
                "rejected_codes": [code],
                "image_dispatches": len(image.calls),
                "store": self._store_summary(sandbox),
            }

    def _run_cancelled_image_cleanup(self) -> dict:
        text, _ = self._providers()
        with _MixedApiSandbox([], []) as sandbox:
            image = _ImageProvider(
                provider_id="sim_image_provider",
                node_id="sim-image-node",
                manifest=_manifest(artifact_id="sim_sd15", suffix="b"),
                data_plane=sandbox.data_plane,
                block_after_write=True,
            )
            sandbox._text_providers[:] = [text]
            sandbox._image_providers[:] = [image]
            outcome = []
            request = self._request("wf_simn4cancel01")
            thread = threading.Thread(
                target=lambda: outcome.append(
                    self._capture_mixed_call(request),
                ),
                daemon=True,
            )
            thread.start()
            if not image.started.wait(_JOIN_SECONDS):
                raise RuntimeError("simulated mixed image stage did not start")
            if not sandbox.coordinator.request_cancel("wf_simn4cancel01"):
                raise RuntimeError("simulated mixed workflow did not accept cancellation")
            thread.join(_JOIN_SECONDS)
            if thread.is_alive() or not outcome or not isinstance(outcome[0], HTTPException):
                raise RuntimeError("simulated mixed cancellation did not converge")
            response = outcome[0]
            snapshot = sandbox.coordinator.get("wf_simn4cancel01")
            return {
                "terminal_state": snapshot["state"],
                "rejected_codes": [response.detail["code"]],
                "image_dispatches": len(image.calls),
                "store": self._store_summary(sandbox),
            }

    @staticmethod
    def _capture_mixed_call(request):
        try:
            return api_server._run_distributed_mixed_generation(request)
        except HTTPException as exc:
            return exc
