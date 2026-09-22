"""P2：跨框架接力的切点求解（容量 / 延迟 / 风险 三输出）。

票面（`docs/跨框架接力-当前有效基线与后续优化计划-2026-09-21.md` §4 P2）要求：
按候选节点的**单位层耗时、可用内存、网络带宽、首 token 延迟和稳定性**求解切点，
并同时输出 `capacity_feasible`、`latency_estimate`、`risk_penalty`。

与既有模块的分工（不重复造轮子）：

* `src/pipeline_capacity.solve_pipeline_capacity` —— 只做**容量**：连续区间放置、含
  embedding/lm_head 归属、safety margin、节点 score 作次级目标；
* `src/relay_planner.plan_relay_cut` —— 只做 **L→L 两段**的保守切点与拒绝理由；
* **本模块** —— 把容量、延迟与风险放进**同一个目标函数**，支持 **n 段**，
  并强制**合法切点约束**（如 Qwen3.5 的切点必须是 `full_attention_interval` 的整倍数，
  否则裁层 GGUF 的层类型错位、无法加载）。

段画像（`SegmentProfile`）既能来自 `DeviceProfiler.to_dict()`，也能**从实测拟合**
（`fit_segment_profile` / `fit_two_segment`）—— 后者是本票「切点重搜」的判据来源：
先在实测点上回归出每段"固定开销 + 每层耗时"，再用求解器预测最优切点，最后与实测最优对比。

本模块 stdlib-only，可在任何环境被测试（不需要模型/GPU）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from src.relay_contract import CUT_LAYER_MIN, RelayHiddenSpec

SCHEMA_VERSION = "qlh.relay_cut_plan.v1"
DEFAULT_SAFETY_MARGIN = 1.2
#: 目标函数权重：容量收益、速度（相对单段）、风险惩罚。
DEFAULT_WEIGHTS = {"capacity": 1.0, "latency": 1.0, "risk": 1.0}
#: 风险项阈值（低于/高于该值开始计罚）。
RISK_BANDWIDTH_MBPS_FLOOR = 5.0
RISK_RTT_MS_CEIL = 50.0
RISK_THERMAL_PENALTY = 0.25
RISK_ARTIFACT_PENALTY = 0.5


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


@dataclass(frozen=True)
class SegmentProfile:
    """一个候选段（设备 + 引擎）的画像。耗时单位 ms，容量单位 byte。"""

    node_id: str
    engine: str = "llama.cpp"
    capacity_bytes: int = 0
    ms_per_layer_decode: float = 0.0
    ms_per_layer_prefill: float = 0.0
    ms_fixed_decode: float = 0.0
    ms_fixed_prefill: float = 0.0
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    thermal_throttled: bool = False
    retry_rate: float = 0.0
    artifacts_ready: bool = True
    source: str = "declared"

    @classmethod
    def from_device_profile(
        cls,
        info: Mapping[str, Any] | None,
        *,
        node_id: str = "segment",
        engine: str = "llama.cpp",
        **overrides: Any,
    ) -> "SegmentProfile":
        """从 `DeviceProfiler.to_dict()` 风格的字典构造（缺字段如实置 0，不猜）。"""
        data = _mapping(info)
        ram = _mapping(data.get("ram"))
        net = _mapping(data.get("network"))
        gpu = _mapping(data.get("gpu"))
        available_gb = _number(ram.get("available_gb", data.get("ram_available_gb")), 0.0)
        values: dict[str, Any] = {
            "node_id": str(node_id),
            "engine": str(engine),
            "capacity_bytes": int(max(0.0, available_gb) * (1024 ** 3)),
            "ms_per_layer_decode": _number(data.get("ms_per_layer_decode")),
            "ms_per_layer_prefill": _number(data.get("ms_per_layer_prefill")),
            "ms_fixed_decode": _number(data.get("ms_fixed_decode")),
            "ms_fixed_prefill": _number(data.get("ms_fixed_prefill")),
            "rtt_ms": _number(data.get("rtt_ms", net.get("rtt_ms"))),
            "bandwidth_mbps": _number(data.get("bandwidth_mbps", net.get("bandwidth_mbps"))),
            "thermal_throttled": _bool(data.get("thermal_throttled"), False),
            "retry_rate": _number(data.get("retry_rate")),
            "artifacts_ready": _bool(data.get("artifacts_ready"), True),
            "source": str(data.get("source", "device_profile")),
        }
        if gpu and not values["capacity_bytes"]:
            values["capacity_bytes"] = int(max(0.0, _number(gpu.get("vram_total_gb"))) * (1024 ** 3))
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayCutPlan:
    admitted: bool
    reason: str
    cuts: tuple[int, ...]
    segment_layers: tuple[int, ...]
    capacity_feasible: bool
    latency_estimate: dict[str, Any]
    risk_penalty: dict[str, Any]
    score: float
    capacity_gain_x: float | None
    speedup_vs_single: float | None
    segment_bytes: tuple[int, ...] = ()
    candidates: tuple[dict[str, Any], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "admitted": self.admitted,
            "reason": self.reason,
            "cuts": list(self.cuts),
            "segment_layers": list(self.segment_layers),
            "segment_bytes": list(self.segment_bytes),
            "capacity_feasible": self.capacity_feasible,
            "latency_estimate": self.latency_estimate,
            "risk_penalty": self.risk_penalty,
            "score": self.score,
            "capacity_gain_x": self.capacity_gain_x,
            "speedup_vs_single": self.speedup_vs_single,
            "candidates": [dict(item) for item in self.candidates],
        }


def _transfer_ms(bytes_total: float, profile: SegmentProfile, *, with_rtt: bool = True) -> float:
    """把 ``bytes_total`` 传过一跳的时间（ms）：串行化 + 可选 RTT。带宽为 0 ⇒ 视为不可用。"""
    bandwidth_mbps = max(0.0, profile.bandwidth_mbps)
    if bandwidth_mbps <= 0:
        return float("inf")
    serialization_ms = bytes_total * 8.0 / (bandwidth_mbps * 1e6) * 1000.0
    return serialization_ms + (profile.rtt_ms if with_rtt else 0.0)


def _segment_bytes(
    layer_bytes: Sequence[int], start: int, end: int, *, has_embedding: bool,
    has_lm_head: bool, non_split_bytes: int,
) -> int:
    total = sum(int(layer_bytes[i]) for i in range(start, end))
    if has_embedding or has_lm_head:
        total += int(non_split_bytes)
    return total


def legal_cuts(total_layers: int, *, cut_multiple: int = 1,
               min_layers_per_segment: int = 1) -> tuple[int, ...]:
    """合法切点集合：满足步长约束、且切点两侧都留够层数。

    ⚠️ `cut_multiple > 1` 是**硬约束**（Qwen3.5 = 4）：裁层 GGUF 的层类型按
    `full_attention_interval` 交替，切错会导致下游无法加载（实测 N=2 报错）。
    """
    multiple = max(1, _int(cut_multiple, 1))
    minimum = max(CUT_LAYER_MIN, _int(min_layers_per_segment, 1))
    total = max(0, _int(total_layers))
    upper = total - minimum
    return tuple(k for k in range(multiple, upper + 1, multiple) if k >= minimum)


def _estimate_latency(
    segment_layers: Sequence[int],
    segments: Sequence[SegmentProfile],
    hidden: RelayHiddenSpec,
    *,
    prefill_tokens: int,
    decode_tokens: int,
) -> dict[str, Any]:
    """端到端估计：各段自身耗时之和 + 每跳激活传输（decode 逐 token）。"""
    decode_ms = 0.0
    prefill_ms = 0.0
    for profile, layers in zip(segments, segment_layers):
        decode_ms += profile.ms_fixed_decode + profile.ms_per_layer_decode * layers
        prefill_ms += profile.ms_fixed_prefill + profile.ms_per_layer_prefill * layers

    decode_bytes = hidden.bytes_per_token * max(0, _int(decode_tokens))
    prefill_bytes = hidden.bytes_per_token * max(0, _int(prefill_tokens))
    transfer_decode_ms = 0.0
    transfer_prefill_ms = 0.0
    for producer, consumer in zip(segments, segments[1:]):
        # 生产端带宽决定出站；RTT 记在消费端一侧（各计一次，避免双算）
        hop_profile = SegmentProfile(
            node_id=consumer.node_id, bandwidth_mbps=producer.bandwidth_mbps,
            rtt_ms=consumer.rtt_ms)
        transfer_decode_ms += _transfer_ms(decode_bytes, hop_profile)
        transfer_prefill_ms += _transfer_ms(prefill_bytes, hop_profile)
    return {
        "prefill_ms": None if math.isinf(prefill_ms + transfer_prefill_ms)
        else round(prefill_ms + transfer_prefill_ms, 4),
        "decode_ms": None if math.isinf(decode_ms + transfer_decode_ms)
        else round(decode_ms + transfer_decode_ms, 4),
        "compute_prefill_ms": round(prefill_ms, 4),
        "compute_decode_ms": round(decode_ms, 4),
        "transfer_prefill_ms": None if math.isinf(transfer_prefill_ms) else round(transfer_prefill_ms, 4),
        "transfer_decode_ms": None if math.isinf(transfer_decode_ms) else round(transfer_decode_ms, 4),
        "hops": max(0, len(segments) - 1),
        "hidden_bytes_per_token": hidden.bytes_per_token,
        "prefill_tokens": max(0, _int(prefill_tokens)),
        "decode_tokens": max(0, _int(decode_tokens)),
    }


def _risk_penalty(segments: Sequence[SegmentProfile]) -> dict[str, Any]:
    """风险分项：弱网 / 高 RTT / 热降频 / 重试 / 工件不可得。取值越大越差（0 最好）。"""
    items: list[dict[str, Any]] = []
    for profile in segments:
        bandwidth = max(0.0, profile.bandwidth_mbps)
        bandwidth_item = 0.0
        if bandwidth < RISK_BANDWIDTH_MBPS_FLOOR:
            bandwidth_item = 1.0 - (bandwidth / RISK_BANDWIDTH_MBPS_FLOOR)
        rtt_item = min(1.0, max(0.0, profile.rtt_ms) / RISK_RTT_MS_CEIL)
        item = {
            "node_id": profile.node_id,
            "bandwidth": round(bandwidth_item, 4),
            "rtt": round(rtt_item, 4),
            "thermal": RISK_THERMAL_PENALTY if profile.thermal_throttled else 0.0,
            "retry": round(min(1.0, max(0.0, profile.retry_rate)), 4),
            "artifacts": 0.0 if profile.artifacts_ready else RISK_ARTIFACT_PENALTY,
        }
        item["total"] = round(min(1.0, sum(v for k, v in item.items()
                                           if k not in ("node_id", "total"))), 4)
        items.append(item)
    total = round(min(1.0, sum(i["total"] for i in items) / max(1, len(items))), 4) if items else 1.0
    return {"total": total, "items": items}


def _single_segment_ms(segments: Sequence[SegmentProfile], total_layers: int) -> float | None:
    """单段（整模放一个设备）的 decode 估计：取最快的可用段。"""
    values = [p.ms_fixed_decode + p.ms_per_layer_decode * total_layers for p in segments]
    values = [v for v in values if v > 0]
    return min(values) if values else None


def plan_relay_cut_n_segments(
    *,
    total_layers: int,
    layer_bytes: Sequence[int],
    n_embd: int,
    segments: Sequence[SegmentProfile],
    non_split_bytes: int = 0,
    hidden_dtype: str = "float32",
    cut_multiple: int = 1,
    min_layers_per_segment: int = 1,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    weights: Mapping[str, float] | None = None,
    prefill_tokens: int = 32,
    decode_tokens: int = 32,
    max_candidates: int = 512,
) -> RelayCutPlan:
    """在合法切点空间里选**容量可行 + 目标函数最优**的 n 段方案，或给出具名拒绝。"""
    total = max(0, _int(total_layers))
    hidden = RelayHiddenSpec(n_embd=_int(n_embd), dtype=str(hidden_dtype))
    weight_map = dict(DEFAULT_WEIGHTS)
    weight_map.update({k: _number(v) for k, v in _mapping(weights).items()})
    margin = max(1.0, _number(safety_margin, DEFAULT_SAFETY_MARGIN))
    profiles = list(segments)

    def reject(reason: str, *, feasible: bool = False, risk: dict[str, Any] | None = None) -> RelayCutPlan:
        return RelayCutPlan(
            admitted=False, reason=reason, cuts=(), segment_layers=(),
            capacity_feasible=feasible,
            latency_estimate={}, risk_penalty=risk or {"total": 1.0, "items": []},
            score=float("-inf"), capacity_gain_x=None, speedup_vs_single=None)

    if not profiles:
        return reject("no_usable_segments")
    if len(layer_bytes) != total:
        return reject("layer_bytes_length_mismatch")
    if total < 2:
        return reject("model_has_no_relay_range")
    if not hidden.supported:
        return reject("unsupported_hidden_format")
    if len(profiles) < 2:
        return reject("segment_count_mismatch")

    cuts = legal_cuts(total, cut_multiple=cut_multiple,
                      min_layers_per_segment=min_layers_per_segment)
    if not cuts:
        return reject("no_legal_cut_point")

    n_segments = len(profiles)
    if n_segments - 1 > len(cuts):
        return reject("no_legal_cut_combination")
    candidate_limit = _int(max_candidates, 0)
    if candidate_limit <= 0:
        return reject("candidate_limit_invalid")
    candidate_count = math.comb(len(cuts), n_segments - 1)
    if candidate_count > candidate_limit:
        return reject("candidate_space_too_large")
    combos = list(itertools.combinations(cuts, n_segments - 1))

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # The non-split weights live on both endpoint segments, so the resident
    # total must match the per-segment accounting below.
    total_bytes = sum(int(b) for b in layer_bytes) + 2 * int(non_split_bytes)
    single_ms = _single_segment_ms(profiles, total)

    for combo in combos:
        boundaries = (0,) + tuple(combo) + (total,)
        sizes = tuple(boundaries[i + 1] - boundaries[i] for i in range(n_segments))
        if any(size < max(CUT_LAYER_MIN, _int(min_layers_per_segment, 1)) for size in sizes):
            continue
        sizes_bytes = tuple(
            _segment_bytes(layer_bytes, boundaries[i], boundaries[i + 1],
                           has_embedding=(i == 0), has_lm_head=(i == n_segments - 1),
                           non_split_bytes=non_split_bytes)
            for i in range(n_segments))
        feasible = all(
            size_bytes * margin <= max(0, profiles[i].capacity_bytes)
            for i, size_bytes in enumerate(sizes_bytes))
        latency = _estimate_latency(sizes, profiles, hidden,
                                    prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
        risk = _risk_penalty(profiles)
        max_segment = max(sizes_bytes) if sizes_bytes else 0
        capacity_gain = round(total_bytes / max_segment, 4) if max_segment else None
        speedup = (round(single_ms / latency["decode_ms"], 4)
                   if single_ms and latency["decode_ms"] else None)
        score = (weight_map["capacity"] * (capacity_gain or 0.0)
                 + weight_map["latency"] * (speedup or 0.0)
                 - weight_map["risk"] * risk["total"])
        item = {
            "cuts": list(combo),
            "segment_layers": list(sizes),
            "segment_bytes": list(sizes_bytes),
            "capacity_feasible": feasible,
            "capacity_gain_x": capacity_gain,
            "speedup_vs_single": speedup,
            "latency_estimate": latency,
            "risk_penalty": risk["total"],
            "score": round(score, 4),
        }
        evaluated.append(item)
        if not feasible:
            continue
        if best is None or score > best["score"]:
            best = item

    if best is None:
        any_feasible = any(item["capacity_feasible"] for item in evaluated)
        plan = reject("all_placements_infeasible" if not any_feasible else "no_scoring_candidate",
                      feasible=any_feasible, risk=_risk_penalty(profiles))
        return RelayCutPlan(**{**plan.__dict__, "candidates": tuple(evaluated[:200])})

    plan = RelayCutPlan(
        admitted=True,
        reason="best_objective",
        cuts=tuple(best["cuts"]),
        segment_layers=tuple(best["segment_layers"]),
        capacity_feasible=bool(best["capacity_feasible"]),
        latency_estimate=best["latency_estimate"],
        risk_penalty=_risk_penalty(profiles),
        score=best["score"],
        capacity_gain_x=best["capacity_gain_x"],
        speedup_vs_single=best["speedup_vs_single"],
        segment_bytes=tuple(best["segment_bytes"]),
        candidates=tuple(evaluated[:200]),
    )
    return plan


def fit_segment_profile(
    measurements: Sequence[Mapping[str, Any]],
    *,
    node_id: str,
    engine: str,
    layers_key: str,
    ms_key: str,
    capacity_bytes: int = 0,
    bandwidth_mbps: float = 0.0,
    rtt_ms: float = 0.0,
    source: str = "fitted_from_measurements",
) -> tuple[SegmentProfile, dict[str, Any]]:
    """从实测点线性回归出「固定开销 + 每层耗时」（最小二乘）。

    `measurements` 里每条给该段**实际层数**与**实测耗时**。返回 `(profile, fit_info)`，
    `fit_info` 带斜率/截距/`r2`/样本数 —— 报告必须给出这些，才能复算。

    为什么需要：P2 的「切点重搜」要用**该设备对的实测**而不是通用常数（早期用
    32 MiB/层 的常量估计偏差巨大）。
    """
    points = [(_number(m.get(layers_key)), _number(m.get(ms_key)))
              for m in measurements if _mapping(m).get(layers_key) is not None
              and m.get(ms_key) is not None]
    points = [(x, y) for x, y in points if x > 0 and y > 0]
    if len(points) < 2:
        return (SegmentProfile(node_id=node_id, engine=engine, capacity_bytes=capacity_bytes,
                               bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, source=source),
                {"samples": len(points), "r2": None, "slope": None, "intercept": None})
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r2 = None if ss_tot == 0 else round(1.0 - ss_res / ss_tot, 6)
    profile = SegmentProfile(
        node_id=node_id, engine=engine, capacity_bytes=capacity_bytes,
        ms_per_layer_decode=round(max(0.0, slope), 6),
        ms_fixed_decode=round(intercept, 6),
        bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms, source=source)
    return profile, {
        "samples": len(points),
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r2": r2,
    }


def fit_two_segment(
    measurements: Sequence[Mapping[str, Any]],
    *,
    total_layers: int,
    upstream_layers_key: str = "upstream_layers",
    upstream_ms_key: str = "upstream_decode_ms",
    downstream_ms_key: str = "downstream_decode_ms",
    upstream_profile: SegmentProfile | None = None,
    downstream_profile: SegmentProfile | None = None,
) -> dict[str, Any]:
    """两段（D→L）专用便捷入口：分别拟合上游/下游，返回两段的 profile 与拟合质量。"""
    upstream_ms = [{"k": m.get(upstream_layers_key), "ms": m.get(upstream_ms_key)}
                   for m in measurements]
    downstream_ms = []
    for m in measurements:
        upstream_used = _number(m.get(upstream_layers_key))
        if upstream_used <= 0:
            continue
        downstream_ms.append({"k": total_layers - upstream_used,
                              "ms": m.get(downstream_ms_key)})
    up_profile, up_fit = fit_segment_profile(
        [{"k": p["k"], "ms": p["ms"]} for p in upstream_ms],
        node_id=(upstream_profile.node_id if upstream_profile else "upstream_pytorch"),
        engine="pytorch",
        layers_key="k", ms_key="ms",
        capacity_bytes=(upstream_profile.capacity_bytes if upstream_profile else 0),
        bandwidth_mbps=(upstream_profile.bandwidth_mbps if upstream_profile else 0.0),
        rtt_ms=(upstream_profile.rtt_ms if upstream_profile else 0.0))
    dn_profile, dn_fit = fit_segment_profile(
        [{"k": p["k"], "ms": p["ms"]} for p in downstream_ms],
        node_id=(downstream_profile.node_id if downstream_profile else "downstream_llama"),
        engine="llama.cpp",
        layers_key="k", ms_key="ms",
        capacity_bytes=(downstream_profile.capacity_bytes if downstream_profile else 0),
        bandwidth_mbps=(downstream_profile.bandwidth_mbps if downstream_profile else 0.0),
        rtt_ms=(downstream_profile.rtt_ms if downstream_profile else 0.0))
    return {"upstream": up_profile, "downstream": dn_profile,
            "fit": {"upstream": up_fit, "downstream": dn_fit}}
