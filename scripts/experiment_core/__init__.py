"""QLH 全自动优化实验框架（EX-N1）。

组件：plan（manifest 解析/校验）→ scheduler（串/并行、资源互斥、断点续跑）
→ runner（单元执行、超时、重试、日志）→ collector（schema 落盘）
→ report（report.md + summary.json）。

纪律入口：《自动化优化实验与报告方案》§1 十原则；本框架只产出实验数据，
不混入 pytest 测试基数。
"""

from .plan import PlanError, PlanManifest, load_plan
from .runner import RunnerError, run_unit
from .scheduler import ConflictError, execute_plan
from .collector import CollectorError, build_record
from .report import build_report

__all__ = [
    "PlanError", "PlanManifest", "load_plan",
    "RunnerError", "run_unit",
    "ConflictError", "execute_plan",
    "CollectorError", "build_record",
    "build_report",
]
