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
    "generate_html_report",
    "test_single_node",
    "test_distributed",
    "test_degradation",
    "test_stress",
    "test_exceptions",
]

__version__ = "3.0.0"
