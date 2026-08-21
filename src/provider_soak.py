"""RAG 后置可准备：embedding provider 长时 soak harness 核心。

纯本机、只读、可注入 provider 与故障（失败/挂起/超时），产出脱敏 soak
报告（调用数/成功/失败/超时/恢复、延迟 p50/p95/平均、确定性摘要）。真实
长时运行由 ``scripts/provider_soak_harness.py`` 以 --iterations/--duration
触发；本核心供测试用 fake provider 覆盖确定性短跑。真实 Ollama/llama.cpp
provider 的接入与长时真跑仍属 RAG 后置验收。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

SOAK_SCHEMA_VERSION = "qlh.rag.provider_soak.v1"


@dataclass
class SoakSpec:
    iterations: int = 300
    failure_rate: float = 0.0     # 注入"provider 失败"比例 [0,1]
    hang_rate: float = 0.0        # 注入"挂起/超时"比例 [0,1]（不真正阻塞）
    # 供真实 provider 包装层参考的单次超时（秒）；本核心同步快速调用，
    # 不读取、不强制阻塞——真实长阻塞由外部 provider 包装实现。
    timeout_s: float = 2.0
    seed: int = 20260821


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_provider_soak(
    embed_fn: Callable[[int], Any],
    *,
    spec: SoakSpec = SoakSpec(),
) -> dict[str, Any]:
    """串行跑 ``iterations`` 次 embedding 调用（注入故障），返回脱敏 soak 报告。

    ``embed_fn(i)`` 是第 i 次调用的 provider 执行体（真实实现应做离线 embedding
    并返回摘要）。失败 = provider 抛异常；挂起 = 命中 ``hang_rate`` 时跳过调用并
    计为超时（避免真实长阻塞）；``recovered`` = 失败后下一次调用成功。
    """
    iterations = max(1, int(spec.iterations))
    if not 0.0 <= float(spec.failure_rate) <= 1.0 or not 0.0 <= float(spec.hang_rate) <= 1.0:
        raise ValueError("failure_rate / hang_rate must be in [0,1]")
    rng = np.random.default_rng(int(spec.seed))
    latencies: list[float] = []
    successes = failures = timeouts = recovered = 0
    last_failed = False
    for index in range(iterations):
        roll = float(rng.random())
        if roll < float(spec.failure_rate):
            failures += 1
            last_failed = True
            continue
        if roll < float(spec.failure_rate) + float(spec.hang_rate):
            timeouts += 1
            last_failed = True
            continue
        started = time.perf_counter()
        try:
            embed_fn(index)
        except Exception:
            failures += 1
            last_failed = True
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)
        successes += 1
        if last_failed:
            recovered += 1
        last_failed = False
    report = {
        "schema_version": SOAK_SCHEMA_VERSION,
        "mode": "soak_shadow",
        "seed": int(spec.seed),
        "iterations": iterations,
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "recovered": recovered,
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "avg": round(float(np.mean(latencies)), 3) if latencies else 0.0,
        },
    }
    report["digest"] = hashlib.sha256(
        json.dumps(
            {key: report[key] for key in (
                "schema_version", "mode", "seed", "iterations",
                "successes", "failures", "timeouts", "recovered",
            )},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()
    return report
