"""Capacity admission for metadata-first PyTorch pipeline loading.

The solver consumes only a pipeline descriptor and explicit free-memory budgets.
It never imports torch or opens model weight files.  A successful result always
covers every decoder layer exactly once and assigns the input/output components
to the first/last execution node respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import Any


CAPACITY_PLAN_SCHEMA_VERSION = 1


class PipelineCapacityError(ValueError):
    """The descriptor or node capacity input is not safe to solve."""


def _non_negative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineCapacityError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PipelineCapacityError(f"{field} must not be negative")
    return parsed


def _positive_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineCapacityError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise PipelineCapacityError(f"{field} must be positive")
    return parsed


def _descriptor_costs(
    descriptor: dict[str, Any],
) -> tuple[list[int], int, int, int]:
    total_layers = _non_negative_int(descriptor.get("total_layers", 0), "total_layers")
    raw_layers = descriptor.get("layer_weight_bytes")
    if total_layers <= 0 or not isinstance(raw_layers, list):
        raise PipelineCapacityError("descriptor is missing exact layer_weight_bytes")
    if len(raw_layers) != total_layers:
        raise PipelineCapacityError("layer_weight_bytes does not match total_layers")
    layer_bytes = [
        _non_negative_int(value, f"layer_weight_bytes[{index}]")
        for index, value in enumerate(raw_layers)
    ]
    if any(value <= 0 for value in layer_bytes):
        raise PipelineCapacityError("every decoder layer must have a positive byte count")

    components = descriptor.get("component_weight_bytes")
    if not isinstance(components, dict):
        raise PipelineCapacityError("descriptor is missing component_weight_bytes")
    embedding_bytes = _non_negative_int(
        components.get("embedding", 0), "component_weight_bytes.embedding"
    )
    per_node_bytes = _non_negative_int(
        components.get("final_norm", 0), "component_weight_bytes.final_norm"
    )
    output_bytes = sum(
        _non_negative_int(components.get(name, 0), f"component_weight_bytes.{name}")
        for name in ("lm_head", "other")
    )
    # Tied Qwen3 checkpoints store the output projection only once under the
    # embedding key.  A separate last-stage worker still needs that tensor,
    # so charge it as output capacity when no explicit LM Head tensor exists.
    if bool(descriptor.get("tie_word_embeddings", False)) and output_bytes == 0:
        output_bytes = embedding_bytes
    unsupported = {
        name: _non_negative_int(
            components.get(name, 0), f"component_weight_bytes.{name}"
        )
        for name in ("visual", "mtp", "multimodal")
    }
    # ★ #31 M3：架构可以**显式声明**某些分量"进了权重索引、但**不参与层执行**"
    #   （hybrid 的 visual / mtp —— 依据是运行时结构里根本没有这两类子模块，见
    #   `已知问题记录.md` #31 §31.6.1 的实测）。这类分量**不计入容量账**（没有任何节点会加载它们）；
    #   **其余分量一律照旧 fail-closed**，且标记里的未知名字本身也要报错（防手抖）。
    ignored = descriptor.get("runtime_ignored_components") or []
    if not isinstance(ignored, (list, tuple)):
        raise PipelineCapacityError("descriptor runtime_ignored_components must be a list")
    unknown = [str(name) for name in ignored if name not in unsupported]
    if unknown:
        raise PipelineCapacityError(
            "descriptor runtime_ignored_components has unknown entries: " + ", ".join(unknown)
        )
    ignored = set(ignored)
    active_unsupported = [
        name for name, size in unsupported.items()
        if size > 0 and name not in ignored
    ]
    if active_unsupported:
        raise PipelineCapacityError(
            "descriptor has separately placeable components without a runtime plan: "
            + ", ".join(active_unsupported)
        )
    return layer_bytes, embedding_bytes, per_node_bytes, output_bytes


def _normalize_nodes(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(nodes, list):
        raise PipelineCapacityError("nodes must be a list")
    usable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in nodes:
        if not isinstance(raw, dict):
            raise PipelineCapacityError("each node capacity record must be an object")
        node_id = str(raw.get("node_id", "") or "").strip()
        if not node_id or node_id in seen:
            raise PipelineCapacityError("node_id must be non-empty and unique")
        seen.add(node_id)
        capacity_bytes = _non_negative_int(raw.get("capacity_bytes", 0), "capacity_bytes")
        reserve_bytes = _non_negative_int(raw.get("reserve_bytes", 0), "reserve_bytes")
        if capacity_bytes <= reserve_bytes:
            excluded.append({
                "node_id": node_id,
                "role": str(raw.get("role", "client") or "client"),
                "reason_code": "node_capacity_unavailable",
                "capacity_bytes": capacity_bytes,
                "reserve_bytes": reserve_bytes,
                "capacity_source": str(raw.get("capacity_source", "") or ""),
            })
            continue
        usable.append({
            "node_id": node_id,
            "role": str(raw.get("role", "client") or "client"),
            "capacity_bytes": capacity_bytes,
            "reserve_bytes": reserve_bytes,
            "runtime_multiplier": _positive_float(
                raw.get("runtime_multiplier", 1.0), "runtime_multiplier"
            ),
            "score": float(raw.get("score", 0.0) or 0.0),
            "capacity_source": str(raw.get("capacity_source", "explicit") or "explicit"),
            "execution_device": str(raw.get("execution_device", "unknown") or "unknown"),
        })
    usable.sort(
        key=lambda node: (
            node["role"] != "master",
            -node["score"],
            -(node["capacity_bytes"] - node["reserve_bytes"]),
            node["node_id"],
        )
    )
    return usable, excluded


def _required_bytes(raw_bytes: int, node: dict[str, Any], safety_margin: float) -> int:
    return node["reserve_bytes"] + math.ceil(
        raw_bytes * node["runtime_multiplier"] * safety_margin
    )


def _relay_zero_layer_assignments(
    relay_nodes: list[dict[str, Any]], total_layers: int,
    claims: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """★ A1 / X 档（Y 档第二条缺口 6/2b）：把 relay 段节点作为**零层**条目并入 assignments。

    主节点据此给它下发 `engine="relay_middle"` 配置（`scheduler_pipeline.py:298-302` 按
    `_relay_segment_for_worker()` 决定），worker 侧于是**不加载任何层**、只转发
    （`peer.py` 的 relay 分支）。

    ★ Y 档第二条：条目带上该段**认领的层区间**（`claims[node_id]`，形如 `(8, 24)`）——
    下游的层覆盖校验（`pipeline_node_contract.validate_pipeline_nodes`）必须知道
    "`[8,24)` 由远端段执行"，否则会误判为"层没人覆盖"而拒掉合法拓扑。

    `plan_identity` / `plan_id` **不含**它们（那里只用上面的 `assignments`）⇒
    零层条目不改变 plan 的身份摘要。
    """
    claims = claims or {}
    entries = []
    for node in relay_nodes:
        start, end = claims.get(node["node_id"], (total_layers, total_layers))
        entries.append({
            "node_id": node["node_id"],
            "role": node["role"],
            "start_layer": int(start),
            "end_layer": int(end),
            "layers_count": 0,
            "has_embedding": False,
            "has_lm_head": False,
            "claimed_layers": [int(start), int(end)],
            "raw_weight_bytes": 0,
            "required_bytes": 0,
            "capacity_bytes": node["capacity_bytes"],
            "headroom_bytes": node["capacity_bytes"],
            "reserve_bytes": 0,
            "runtime_multiplier": 1.0,
            "execution_device": node["execution_device"],
            "capacity_source": "relay_exempt",
            "score": node["score"],
        })
    return entries


def solve_pipeline_capacity(
    descriptor: dict[str, Any],
    nodes: list[dict[str, Any]],
    *,
    safety_margin: float = 1.2,
    require_distributed: bool = False,
    local_layer_budget: int | None = None,
    relay_claims: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Return an all-or-nothing contiguous layer placement.

    ``capacity_bytes`` must describe currently free memory, not installed
    physical memory. ``reserve_bytes`` is charged once to every participating
    node for allocator, activation, KV-cache and loading workspace headroom.
    When ``require_distributed`` is true, a single-node full-model placement is
    deliberately rejected; the returned plan must contain at least two
    participating nodes. This is used by an explicitly distributed request,
    while ordinary capacity inspection remains single-node efficient.
    """

    safety_margin = _positive_float(safety_margin, "safety_margin")
    if safety_margin < 1.0:
        raise PipelineCapacityError("safety_margin must be at least 1.0")
    layer_bytes, embedding_bytes, per_node_bytes, output_bytes = _descriptor_costs(
        descriptor
    )
    usable, excluded = _normalize_nodes(nodes)
    # ★ A1 / X 档（Y 档第二条缺口 6）：relay 段节点**不占层、不计容量**，但**算参与节点**。
    #   `scheduler.py` 的 `_get_pipeline_capacity_nodes` 给它打 `capacity_source="relay_exempt"`
    #   （段工件在远端 relay_mid_service，本节点只转发 ⇒ 不需要本地容量预算）。
    #   若把它留在 `usable`：求解器会要求它真装下若干层 ⇒ 必然失败（实测 plan 被拒）。
    #   若整个剔除：`len(usable) < 2` 又会把"master + relay"这种**合法**拓扑拒掉。
    #   ⇒ 单列：不参与分层搜索，但计入"分布式可用节点数"，并以**零层条目**进 assignments。
    relay_only = [n for n in usable if n.get("capacity_source") == "relay_exempt"]
    usable = [n for n in usable if n.get("capacity_source") != "relay_exempt"]
    total_layers = len(layer_bytes)
    # ★ Y 档第二条：relay 段认领的层由**远端段服务**执行 ⇒ 本机层节点只需放下 `[0, k)`。
    #   `local_layer_budget` 由调度层算出（`Scheduler._relay_claimed_layer_prefix`，那里带
    #   重叠 / 角色 / 连续性校验）；缺省 = 全部层（对既有路径零影响）。
    #   ⚠️ 末位本机节点**仍拿 `lm_head`**：主节点收到末节点回的 hidden 后要自己跑
    #   Norm + LM Head（见 `scheduler_pipeline.py:4594-4595` 的推荐拓扑）⇒ 这正是不选
    #   `tail` 段而选 `middle[8,24)` 的原因（tail 只回 token，拿不到 hidden）。
    layer_budget = total_layers if local_layer_budget is None else int(local_layer_budget)
    if not (0 <= layer_budget <= total_layers):
        raise PipelineCapacityError(
            f"local_layer_budget 越界: {layer_budget} / total={total_layers}"
        )
    raw_model_bytes = (
        sum(layer_bytes) + embedding_bytes + per_node_bytes + output_bytes
    )
    prefix = [0]
    for value in layer_bytes:
        prefix.append(prefix[-1] + value)

    base = {
        "schema_version": CAPACITY_PLAN_SCHEMA_VERSION,
        "model_id": str(descriptor.get("model_id", "") or ""),
        "model_type": str(descriptor.get("model_type", "") or ""),
        "model_sha256": str(descriptor.get("model_sha256", "") or ""),
        "total_layers": total_layers,
        "raw_model_bytes": raw_model_bytes,
        "safety_margin": safety_margin,
        "candidate_node_count": len(usable),
        "excluded_nodes": excluded,
    }
    if not usable and not relay_only:
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": (
                "pipeline_distributed_workers_unavailable"
                if require_distributed
                else "pipeline_capacity_nodes_unavailable"
            ),
            "assignments": [],
            "control_only_nodes": [],
        }
    if require_distributed and len(usable) + len(relay_only) < 2:
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": "pipeline_distributed_workers_unavailable",
            "reason": "distributed placement requires at least two usable PC nodes",
            "assignments": [],
            "control_only_nodes": [node["node_id"] for node in usable],
        }

    @lru_cache(maxsize=None)
    def search(node_index: int, cursor: int, started: bool, used_count: int):
        if cursor == layer_budget:
            # ★ Y 档第二条：relay 段**算参与节点**（它承载远端段工件），只是不占本机容量
            #   ⇒ "分布式"的判据是 `本机层节点数 + relay 段数 >= 2`，而不是只看前者。
            #   否则 `master(0-8) + relay 段(8-24)` 这种**合法**拓扑会被误拒。
            participating = used_count + len(relay_only)
            if require_distributed and participating < 2:
                return None
            return ()
        if node_index >= len(usable):
            return None
        node = usable[node_index]
        best = search(node_index + 1, cursor, started, used_count)
        remaining = layer_budget - cursor
        for count in range(remaining, 0, -1):
            end = cursor + count
            raw_bytes = prefix[end] - prefix[cursor] + per_node_bytes
            has_embedding = not started
            has_lm_head = end == layer_budget
            if has_embedding:
                raw_bytes += embedding_bytes
            if has_lm_head:
                raw_bytes += output_bytes
            required = _required_bytes(raw_bytes, node, safety_margin)
            if required > node["capacity_bytes"]:
                continue
            tail = search(node_index + 1, end, True, used_count + 1)
            if tail is None:
                continue
            item = (
                node_index,
                cursor,
                end,
                raw_bytes,
                required,
                has_embedding,
                has_lm_head,
            )
            candidate = (item,) + tail
            if best is None:
                best = candidate
                continue
            candidate_key = (
                len(candidate),
                -min(usable[value[0]]["capacity_bytes"] - value[4] for value in candidate),
                -sum(usable[value[0]]["score"] for value in candidate),
            )
            best_key = (
                len(best),
                -min(usable[value[0]]["capacity_bytes"] - value[4] for value in best),
                -sum(usable[value[0]]["score"] for value in best),
            )
            if candidate_key < best_key:
                best = candidate
        return best

    solved = search(0, 0, False, 0)
    if solved is None:
        allocatable_bytes = sum(
            max(0, node["capacity_bytes"] - node["reserve_bytes"])
            for node in usable
        )
        return {
            **base,
            "status": "rejected",
            "admitted": False,
            "reason_code": (
                "pipeline_distributed_capacity_insufficient"
                if require_distributed
                else "pipeline_cluster_capacity_insufficient"
            ),
            "allocatable_bytes": allocatable_bytes,
            "raw_capacity_deficit_bytes": max(0, raw_model_bytes - allocatable_bytes),
            "assignments": [],
            "control_only_nodes": [node["node_id"] for node in usable],
        }

    assignments = []
    used_ids: set[str] = set()
    for node_index, start, end, raw_bytes, required, has_embedding, has_lm_head in solved:
        node = usable[node_index]
        used_ids.add(node["node_id"])
        assignments.append({
            "node_id": node["node_id"],
            "role": node["role"],
            "start_layer": start,
            "end_layer": end,
            "layers_count": end - start,
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "raw_weight_bytes": raw_bytes,
            "required_bytes": required,
            "capacity_bytes": node["capacity_bytes"],
            "headroom_bytes": node["capacity_bytes"] - required,
            "reserve_bytes": node["reserve_bytes"],
            "runtime_multiplier": node["runtime_multiplier"],
            "execution_device": node["execution_device"],
            "capacity_source": node["capacity_source"],
            "score": node["score"],
        })

    plan_identity = {
        "descriptor_sha256": str(descriptor.get("model_sha256", "") or ""),
        "model_id": base["model_id"],
        "safety_margin": safety_margin,
        "assignments": [
            {
                key: item[key]
                for key in (
                    "node_id", "start_layer", "end_layer", "has_embedding",
                    "has_lm_head", "required_bytes", "capacity_bytes",
                )
            }
            for item in assignments
        ],
    }
    plan_id = hashlib.sha256(
        json.dumps(plan_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    full_model_fits = []
    for node in usable:
        required = _required_bytes(raw_model_bytes, node, safety_margin)
        if required <= node["capacity_bytes"]:
            full_model_fits.append(node["node_id"])

    return {
        **base,
        "status": "admitted",
        "admitted": True,
        "reason_code": "distributed_forced" if require_distributed else "",
        "plan_id": plan_id,
        "assignments": assignments + _relay_zero_layer_assignments(
            relay_only, total_layers, relay_claims
        ),
        "control_only_nodes": [
            node["node_id"] for node in usable if node["node_id"] not in used_ids
        ],
        "participating_node_count": len(assignments),
        "single_node_full_model_candidates": full_model_fits,
        "aggregate_only": len(assignments) > 1,
        "require_distributed": bool(require_distributed),
    }
