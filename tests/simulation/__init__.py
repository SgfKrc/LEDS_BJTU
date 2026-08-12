"""
分布式推理仿真测试框架

提供端到端的仿真测试能力，验证分布式推理在真实环境下的可用性。
"""

from .framework import (
    TestOrchestrator,
    TestConfig,
    TestResult,
    BackendManager,
    RequestSender,
    HumanSimulator,
    ResponseValidator,
    ValidationResult,
)

from .scenarios import (
    SCENARIOS,
    Scenario,
)

from .task_graph_harness import (
    SIMULATION_SCHEMA_VERSION,
    SimulationScenario,
    SimulationScenarioError,
    TaskGraphSimulationHarness,
    available_scenarios,
)

from .task_worker_harness import (
    SIMULATION_SCHEMA_VERSION as TASK_WORKER_SIMULATION_SCHEMA_VERSION,
    SimulationScenario as TaskWorkerSimulationScenario,
    SimulationScenarioError as TaskWorkerSimulationScenarioError,
    TaskWorkerControlSimulationHarness,
    available_scenarios as available_task_worker_scenarios,
)

from .diffusion_data_plane_harness import (
    SIMULATION_SCHEMA_VERSION as DIFFUSION_DATA_PLANE_SIMULATION_SCHEMA_VERSION,
    SimulationScenario as DiffusionDataPlaneSimulationScenario,
    SimulationScenarioError as DiffusionDataPlaneSimulationScenarioError,
    DiffusionDataPlaneSimulationHarness,
    available_scenarios as available_diffusion_data_plane_scenarios,
)

from .mixed_workflow_harness import (
    SIMULATION_SCHEMA_VERSION as MIXED_WORKFLOW_SIMULATION_SCHEMA_VERSION,
    SimulationScenario as MixedWorkflowSimulationScenario,
    SimulationScenarioError as MixedWorkflowSimulationScenarioError,
    MixedWorkflowSimulationHarness,
    available_scenarios as available_mixed_workflow_scenarios,
)

from .capacity_harness import (
    SIMULATION_SCHEMA_VERSION as CAPACITY_SIMULATION_SCHEMA_VERSION,
    SimulationScenario as CapacitySimulationScenario,
    SimulationScenarioError as CapacitySimulationScenarioError,
    CapacitySimulationHarness,
    available_scenarios as available_capacity_scenarios,
)

from .html_report import generate_html_report

# 测试模块
from . import test_single_node
from . import test_distributed
from . import test_degradation
from . import test_stress
from . import test_exceptions

__all__ = [
    "TestOrchestrator",
    "TestConfig",
    "TestResult",
    "BackendManager",
    "RequestSender",
    "HumanSimulator",
    "ResponseValidator",
    "ValidationResult",
    "SCENARIOS",
    "Scenario",
    "SIMULATION_SCHEMA_VERSION",
    "SimulationScenario",
    "SimulationScenarioError",
    "TaskGraphSimulationHarness",
    "available_scenarios",
    "TASK_WORKER_SIMULATION_SCHEMA_VERSION",
    "TaskWorkerSimulationScenario",
    "TaskWorkerSimulationScenarioError",
    "TaskWorkerControlSimulationHarness",
    "available_task_worker_scenarios",
    "DIFFUSION_DATA_PLANE_SIMULATION_SCHEMA_VERSION",
    "DiffusionDataPlaneSimulationScenario",
    "DiffusionDataPlaneSimulationScenarioError",
    "DiffusionDataPlaneSimulationHarness",
    "available_diffusion_data_plane_scenarios",
    "MIXED_WORKFLOW_SIMULATION_SCHEMA_VERSION",
    "MixedWorkflowSimulationScenario",
    "MixedWorkflowSimulationScenarioError",
    "MixedWorkflowSimulationHarness",
    "available_mixed_workflow_scenarios",
    "CAPACITY_SIMULATION_SCHEMA_VERSION",
    "CapacitySimulationScenario",
    "CapacitySimulationScenarioError",
    "CapacitySimulationHarness",
    "available_capacity_scenarios",
    "generate_html_report",
    "test_single_node",
    "test_distributed",
    "test_degradation",
    "test_stress",
    "test_exceptions",
]

__version__ = "3.0.0"
