"""Conservative automatic layer planning for llama.cpp RPC workers.

The input shape intentionally matches ``DeviceProfiler.to_dict()``.  The
planner reuses its score when present, then adds RPC-specific constraints:
CPU execution, transport cost, current load, and a memory reserve.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _profile_score(info: Mapping[str, Any]) -> tuple[float, str]:
    supplied = _number(info.get("score_total"), 0.0)
    if supplied > 0:
        return supplied, "DeviceProfiler.score_total"

    cpu = dict(_mapping(info.get("cpu")))
    ram = dict(_mapping(info.get("ram")))
    if not cpu:
        cpu = {
            "physical_cores": info.get("cpu_cores"),
            "freq_max_mhz": info.get("cpu_freq_mhz"),
        }
    if not ram:
        ram = {
            "total_gb": info.get("ram_total_gb"),
        }
    gpu = _mapping(info.get("gpu"))
    gpus = info.get("gpus")
    candidates = [gpu]
    if isinstance(gpus, list):
        candidates.extend(item for item in gpus if isinstance(item, Mapping))
    best_gpu = max(candidates, key=lambda item: _number(item.get("vram_total_gb")))
    cuda_discrete = bool(
        best_gpu.get("cuda_available") and not best_gpu.get("is_integrated")
    )
    vram = _number(best_gpu.get("vram_total_gb"))
    if not cuda_discrete:
        gpu_score = 8.0 if best_gpu.get("cuda_available") or best_gpu.get("mps_available") else 0.0
    elif vram >= 24:
        gpu_score = 50.0
    elif vram >= 16:
        gpu_score = 45.0
    elif vram >= 12:
        gpu_score = 40.0
    elif vram >= 8:
        gpu_score = 35.0
    elif vram >= 6:
        gpu_score = 25.0
    elif vram >= 4:
        gpu_score = 18.0
    else:
        gpu_score = 10.0
    ram_score = min(30.0, _number(ram.get("total_gb"), 4.0) / 64.0 * 30.0)
    cpu_score = min(10.0, _number(cpu.get("physical_cores"), 2.0) * 1.5)
    cpu_score += min(10.0, _number(cpu.get("freq_max_mhz"), 2000.0) / 400.0)
    return round(gpu_score + ram_score + cpu_score, 1), "DeviceProfiler.compatible_fallback"


@dataclass(frozen=True)
class RpcNodeProfile:
    node_id: str
    execution_device: str
    score_total: float
    score_source: str
    physical_cores: int
    logical_cores: int
    freq_max_mhz: float
    cpu_load_percent: float
    ram_total_gb: float
    ram_available_gb: float
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    profile_available: bool = False

    @classmethod
    def from_device_info(
        cls,
        info: Mapping[str, Any] | None,
        *,
        node_id: str = "rpc-worker",
        execution_device: str = "CPU",
        rtt_ms: float = 0.0,
        bandwidth_mbps: float = 0.0,
    ) -> "RpcNodeProfile":
        info = info if isinstance(info, Mapping) else {}
        cpu = _mapping(info.get("cpu"))
        ram = _mapping(info.get("ram"))
        score, source = _profile_score(info)
        physical = int(max(0.0, _number(cpu.get("physical_cores", info.get("cpu_cores")), 0.0)))
        logical = int(max(0.0, _number(cpu.get("logical_cores", info.get("logical_cores")), physical)))
        freq = _number(cpu.get("freq_max_mhz", info.get("cpu_freq_mhz")), 0.0)
        load = _number(cpu.get("usage_percent", info.get("cpu_load_percent")), 0.0)
        total = _number(ram.get("total_gb", info.get("ram_total_gb")), 0.0)
        available = _number(ram.get("available_gb", info.get("ram_available_gb")), 0.0)
        available_fields = any(
            key in cpu or key in ram
            for key in ("physical_cores", "freq_max_mhz", "usage_percent", "available_gb")
        )
        available_fields = available_fields or any(
            key in info for key in ("cpu_cores", "cpu_freq_mhz", "ram_available_gb")
        )
        return cls(
            node_id=node_id,
            execution_device=str(execution_device or "CPU").upper(),
            score_total=round(score, 2),
            score_source=source,
            physical_cores=physical,
            logical_cores=logical,
            freq_max_mhz=round(freq, 2),
            cpu_load_percent=round(max(0.0, min(load, 100.0)), 2),
            ram_total_gb=round(total, 2),
            ram_available_gb=round(available, 2),
            rtt_ms=round(max(0.0, _number(rtt_ms)), 2),
            bandwidth_mbps=round(max(0.0, _number(bandwidth_mbps)), 2),
            profile_available=available_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RpcSplitDecision:
    admitted: bool
    strategy: str
    total_layers: int
    rpc_layers: int
    local_layers: int
    model_size_mib: float
    worker_budget_mib: float
    estimated_layer_mib: float
    memory_layer_cap: int
    performance_layer_cap: int
    score_layer_cap: int
    profile: RpcNodeProfile
    reason: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile"] = self.profile.to_dict()
        return result


def plan_rpc_split(
    total_layers: int,
    model_size_mib: float,
    worker_profile: Mapping[str, Any] | RpcNodeProfile | None,
    *,
    worker_budget_mib: float = 512.0,
    rtt_ms: float = 0.0,
    bandwidth_mbps: float = 0.0,
) -> RpcSplitDecision:
    """Return a safe initial RPC layer count.

    CPU RPC workers are deliberately capped by both measured capability and
    the existing profile score. A worker with missing telemetry is excluded,
    instead of receiving a guessed share of the model.
    """
    total = max(0, int(total_layers))
    model_mib = max(0.0, _number(model_size_mib))
    budget = max(0.0, _number(worker_budget_mib))
    if isinstance(worker_profile, RpcNodeProfile):
        profile = worker_profile
    else:
        profile = RpcNodeProfile.from_device_info(
            worker_profile,
            rtt_ms=rtt_ms,
            bandwidth_mbps=bandwidth_mbps,
        )

    if total < 2:
        return RpcSplitDecision(
            admitted=False, strategy="auto", total_layers=total, rpc_layers=0,
            local_layers=total, model_size_mib=round(model_mib, 2),
            worker_budget_mib=round(budget, 2), estimated_layer_mib=0.0,
            memory_layer_cap=0, performance_layer_cap=0, score_layer_cap=0,
            profile=profile, reason="model_has_no_distributable_layer",
        )
    if not profile.profile_available:
        return RpcSplitDecision(
            admitted=False, strategy="auto", total_layers=total, rpc_layers=0,
            local_layers=total, model_size_mib=round(model_mib, 2),
            worker_budget_mib=round(budget, 2), estimated_layer_mib=0.0,
            memory_layer_cap=0, performance_layer_cap=0, score_layer_cap=0,
            profile=profile, reason="worker_profile_missing",
        )
    if profile.cpu_load_percent >= 90.0:
        return RpcSplitDecision(
            admitted=False, strategy="auto", total_layers=total, rpc_layers=0,
            local_layers=total, model_size_mib=round(model_mib, 2),
            worker_budget_mib=round(budget, 2), estimated_layer_mib=0.0,
            memory_layer_cap=0, performance_layer_cap=0, score_layer_cap=0,
            profile=profile, reason="worker_cpu_overloaded",
        )

    transport_factor = 1.0
    if profile.rtt_ms > 0:
        transport_factor *= max(0.25, 1.0 / (1.0 + profile.rtt_ms / 50.0))
    if profile.bandwidth_mbps > 0:
        transport_factor *= max(0.25, min(1.0, profile.bandwidth_mbps / 100.0))
    load_factor = max(0.15, 1.0 - profile.cpu_load_percent / 100.0)
    core_factor = min(max(profile.physical_cores, 0) / 16.0, 1.0)
    freq_factor = min(max(profile.freq_max_mhz, 0.0) / 4000.0, 1.0)
    compute_factor = (core_factor + freq_factor) / 2.0
    if profile.execution_device == "CPU":
        performance_cap = max(1, int(4.0 * compute_factor * load_factor * transport_factor))
        score_cap = max(1, int(total * min(0.20, max(0.04, profile.score_total / 100.0 * 0.5))))
    else:
        performance_cap = max(1, int(total * min(0.50, compute_factor * load_factor * transport_factor)))
        score_cap = max(1, int(total * min(0.50, max(0.04, profile.score_total / 100.0))))

    estimated_layer_mib = max(32.0, model_mib / total * (2.0 if profile.execution_device == "CPU" else 1.0))
    reserve_mib = max(128.0, min(256.0, budget * 0.35))
    usable_mib = min(budget, max(0.0, profile.ram_available_gb * 1024.0 * 0.55))
    memory_cap = int(max(0.0, usable_mib - reserve_mib) / estimated_layer_mib)
    if profile.ram_available_gb <= 0:
        memory_cap = max(1, int(max(0.0, budget - reserve_mib) / estimated_layer_mib))
    rpc_layers = min(total - 1, performance_cap, score_cap, memory_cap)
    rpc_layers = max(0, rpc_layers)
    return RpcSplitDecision(
        admitted=rpc_layers > 0,
        strategy="auto",
        total_layers=total,
        rpc_layers=rpc_layers,
        local_layers=total - rpc_layers,
        model_size_mib=round(model_mib, 2),
        worker_budget_mib=round(budget, 2),
        estimated_layer_mib=round(estimated_layer_mib, 2),
        memory_layer_cap=memory_cap,
        performance_layer_cap=performance_cap,
        score_layer_cap=score_cap,
        profile=profile,
        reason="cpu_rpc_conservative_cap" if profile.execution_device == "CPU" else "profile_capacity_cap",
    )


def plan_to_tensor_split(
    decision: RpcSplitDecision,
    *,
    total_layers: int | None = None,
) -> list[float] | None:
    """把 planner 的层段决策翻译成 llama.cpp 的 ``tensor_split``。

    ``plan_rpc_split`` 回答的是"多少层交给远端 worker"（``rpc_layers``），而 llama.cpp 的
    ``tensor_split`` 是**各 device 的占比**，顺序与 ``devices`` 一致。引擎侧（见
    ``LlamaCppEngine.load_model(rpc_split=...)``）约定 ``devices = [本机 CPU, RPC0, ...]``，
    因此转换就是：:

        [ (total - rpc_layers) / total , rpc_layers / total ]

    返回 ``None`` 表示**不把层放到远端**（未获准或 ``rpc_layers == 0``），调用方据此走
    纯本机路径，而不是给出一份"全 0 远端"的假分片。
    """
    total = int(total_layers if total_layers is not None else decision.total_layers)
    if total <= 0:
        raise ValueError("total_layers must be positive")
    remote = max(0, min(int(decision.rpc_layers), total))
    if not decision.admitted or remote == 0:
        return None
    return [(total - remote) / total, remote / total]
