"""调度器（EX-N1）：资源互斥、串/并行、断点续跑。

资源冲突规则：两个单元声明的 resources 有任何相同 key 且值不同即冲突；
相同 key 相同值视为共享同一资源（如同一张 GPU），同样冲突（必须串行）。
无任何 key 重叠的单元可以并行。
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .plan import PlanManifest
from .runner import UnitOutcome


class ConflictError(RuntimeError):
    """并行执行将违反资源互斥。"""


def resources_conflict(a: dict[str, str], b: dict[str, str]) -> bool:
    """两个单元的资源声明冲突当且仅当存在相同的命名资源 (key, value) 对。

    相同值表示共享同一资源（同一张 GPU、同一个端口、同一个模型槽位），
    必须互斥；key 相同但值不同（gpu0 / gpu1）视为不同资源，可以并行。
    无编号的通用 key（如 {"gpu": "any"} 或 {"gpu": ""}）对所有声明同一 key
    的单元冲突，表示共享 GPU 池。
    """
    a_pairs = {(k, v) for k, v in a.items()}
    b_pairs = {(k, v) for k, v in b.items()}
    if a_pairs & b_pairs:
        return True
    shared_keys = set(a.keys()) & set(b.keys())
    for key in shared_keys:
        if a[key] in ("", "any") or b[key] in ("", "any"):
            return True
    return False


def _group_non_conflicting(units: Sequence, parallel: int) -> list[list]:
    """贪心分组：每组内两两不冲突；返回若干可并行组（保持 manifest 顺序）。"""
    groups: list[list] = []
    for unit in units:
        placed = False
        for group in groups:
            if len(group) >= parallel:
                continue
            if all(not resources_conflict(unit.resources, other.resources) for other in group):
                group.append(unit)
                placed = True
                break
        if not placed:
            groups.append([unit])
    return groups


def _record_digest(record: dict) -> str:
    """已完成单元的结果指纹：metrics + status + 时间，用于断点续跑校验。"""
    payload = json.dumps(
        {k: record.get(k) for k in ("experiment_id", "status", "metrics", "timestamp")},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_completed(out_dir: Path) -> dict[str, dict]:
    """读取已落盘的 records.jsonl，返回 experiment_id → record。"""
    records_path = out_dir / "records.jsonl"
    if not records_path.is_file():
        return {}
    completed: dict[str, dict] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("experiment_id"):
            completed[str(record["experiment_id"])] = record
    return completed


def execute_plan(
    plan: PlanManifest,
    *,
    out_dir: Path,
    prompt_set_dir: Path,
    run_fn: Callable,
    parallel: int = 1,
    resume: bool = False,
    on_unit_done: Callable | None = None,
) -> dict[str, UnitOutcome]:
    """执行计划全部单元。

    run_fn(unit) -> UnitOutcome：真正执行单元（由 CLI 注入，便于测试替换）。
    resume=True 时跳过 records.jsonl 中 status != invalid 的已完成单元。
    """
    if parallel < 1:
        raise ConflictError("--parallel 必须 >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if resume:
        completed = load_completed(out_dir)

    remaining = [
        unit for unit in plan.units
        if unit.experiment_id not in completed
        or completed[unit.experiment_id].get("status") == "invalid"
    ]
    outcomes: dict[str, UnitOutcome] = {}
    lock = threading.Lock()

    def run_one(unit) -> None:
        outcome = run_fn(unit)
        with lock:
            outcomes[unit.experiment_id] = outcome
            if on_unit_done is not None:
                on_unit_done(unit, outcome)

    if parallel == 1:
        for unit in remaining:
            run_one(unit)
        return outcomes

    groups = _group_non_conflicting(remaining, parallel)
    for group in groups:
        if len(group) == 1:
            run_one(group[0])
            continue
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = [pool.submit(run_one, unit) for unit in group]
            for future in futures:
                future.result()
    return outcomes
