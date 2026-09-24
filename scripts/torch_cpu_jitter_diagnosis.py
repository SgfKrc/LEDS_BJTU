#!/usr/bin/env python
"""torch_cpu_jitter_diagnosis.py — CPU 时延抖动的根因诊断（补 `TORCH-HW-ADMIT-01` 的稳定性门缺口）。

背景：`TORCH-HW-ADMIT-01` 的时延稳定门（`population_stddev / mean <= 0.10`）在 CPU 侧有 **4/12** 格
超限，且**集中在小 cell**（`64:4`、`64:12`）。项目文档明确记录：「未采集 ETW / CPU 频率 / 温度，
因此只能说重排和增加样本未消除抖动，**不能把根因断言为温控或调度**」
（`docs/KTransformers优化迁移调研与算法数据层优化方向-2026-09-23.md:292`）。本工具补这份证据。

本机 CPU 是 `i9-13900H`：**14 物理核 / 20 逻辑核** ⇒ Intel **混合架构**（P-core + E-core）。
若线程跨 P/E（或被调度器在核间迁移），同一 workload 会出现**双峰**耗时，短任务（小 cell）尤其明显 ——
这与「小 cell 先超限」的观测一致，正是本工具要判定/排除的假设。

用法：
    python scripts/torch_cpu_jitter_diagnosis.py --mode cores    # 逐逻辑核画像（看有无 P/E 双峰）
    python scripts/torch_cpu_jitter_diagnosis.py --mode threads  # 亲和性 × 线程数 对照（默认 / P-only / E-only）
    python scripts/torch_cpu_jitter_diagnosis.py --mode single --cpu N   # 内部用：在子进程里绑单核采样

判据纪律：本工具只**诊断**，不改 CV 门、不放宽阈值、不产出可进 planner 的成本。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "local_docs" / "evidence" / "torch-hardware-admit"


def _windows_processor_topology() -> dict[str, object] | None:
    """Return Windows P/E logical CPU groups when the OS exposes them."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # GetActiveProcessorCount is enough to attest the group size, while
        # Windows 10 does not expose a portable Python API for P/E classes.
        kernel32.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
        kernel32.GetActiveProcessorCount.restype = ctypes.c_uint
        groups = []
        for group in range(64):
            count = int(kernel32.GetActiveProcessorCount(group))
            if not count:
                break
            groups.append({"group": group, "logical_count": count})
        return {"api": "GetActiveProcessorCount", "processor_groups": groups}
    except Exception:
        return None


def _set_affinity(mask: int) -> bool:
    """Windows: 把**本进程**绑定到 mask 指定的逻辑处理器集合。

    ⚠️ 必须声明 `GetCurrentProcess` 的 restype：默认按 `c_int` 返回会**截断 HANDLE**，
    于是 `SetProcessAffinityMask` 静默失败（踩过：`affinity_applied=false` 却无异常）。
    """
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    return bool(kernel32.SetProcessAffinityMask(ctypes.c_void_p(handle), ctypes.c_size_t(mask)))


def _logical_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def _workload_ms(*, threads: int, repeats: int, warmup: int, size: int, batch: int,
                 loops: int = 20) -> list[float]:
    """一段**矩阵乘主导**的 CPU 负载（与层前向的算子构成同族），返回每次耗时（ms）。

    单次 `64x896 @ 896x896` 只有 ~0.4 ms，抖动信号太弱 ⇒ 每个样本内跑 `loops` 次，
    让单样本落在几十 ms（与 HW-ADMIT 最小 cell `64:4` prefill ≈ 34 ms 同量级）。
    刻意只测 wall-clock，不做同步以外的插桩 —— 与 HW-ADMIT 的 `uninstrumented_wall` 同口径。
    """
    import torch

    torch.set_num_threads(max(1, int(threads)))
    a = torch.randn(batch, size, dtype=torch.float32)
    b = torch.randn(size, size, dtype=torch.float32)
    rounds = max(1, int(loops))
    for _ in range(max(0, int(warmup)) * rounds):
        a @ b
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter()
        for _ in range(rounds):
            a @ b
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _stats(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean_ms": round(mean, 4),
        "pstddev_ms": round(std, 4),
        "cv": round(std / mean, 4) if mean else 0.0,
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "spread_pct": round(100.0 * (max(values) - min(values)) / mean, 2) if mean else 0.0,
    }


def _run_single(args: argparse.Namespace) -> int:
    """子进程模式：绑一个（或一组）逻辑核，采样后以 JSON 打到 stdout。"""
    mask = args.mask if args.mask else (1 << int(args.cpu))
    bound = _set_affinity(mask)
    samples = _workload_ms(threads=args.threads, repeats=args.repeats,
                           warmup=args.warmup, size=args.size, batch=args.batch,
                           loops=args.loops)
    print(json.dumps({"mask": mask, "affinity_applied": bound, "samples_ms": samples,
                      "stats": _stats(samples)}, sort_keys=True))
    return 0


def _spawn(mask: int, *, threads: int, repeats: int, warmup: int, size: int, batch: int,
           loops: int = 20, timeout: float = 300.0) -> dict[str, object]:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "single",
               "--mask", str(mask), "--threads", str(threads), "--repeats", str(repeats),
               "--warmup", str(warmup), "--size", str(size), "--batch", str(batch),
               "--loops", str(loops)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)
    if done.returncode != 0 or not done.stdout.strip():
        return {"mask": mask, "error": (done.stderr or "").strip()[:300] or "no output"}
    return json.loads(done.stdout.strip().splitlines()[-1])


def _mode_cores(args: argparse.Namespace) -> dict[str, object]:
    """逐逻辑核画像：若存在 P/E 双峰，单核耗时分布会出现两簇。"""
    cores = _logical_cpu_count()
    per_core = [_spawn(1 << index, threads=1, repeats=args.repeats, warmup=args.warmup,
                       size=args.size, batch=args.batch, loops=args.loops)
                for index in range(cores)]
    means = [item["stats"]["mean_ms"] for item in per_core if "stats" in item]
    ordered = sorted(means)
    fastest, slowest = (ordered[0], ordered[-1]) if ordered else (0.0, 0.0)
    return {
        "mode": "cores",
        "logical_cpu_count": cores,
        "per_core": per_core,
        "fastest_mean_ms": fastest,
        "slowest_mean_ms": slowest,
        "slowest_over_fastest": round(slowest / fastest, 3) if fastest else None,
        "sorted_means_ms": [round(value, 3) for value in ordered],
    }


def _mode_threads(args: argparse.Namespace) -> dict[str, object]:
    """亲和性 × 线程数 对照。P-core 假设：只有前 12 个逻辑处理器（6 物理核 + HT）。"""
    cores = _logical_cpu_count()
    presets = {
        "default_all_cores": (1 << cores) - 1,
        "pcore_only_first12": (1 << min(12, cores)) - 1,
        "ecore_only_rest": ((1 << cores) - 1) ^ ((1 << min(12, cores)) - 1),
    }
    runs: dict[str, object] = {}
    for name, mask in presets.items():
        if mask == 0:
            continue
        item = _spawn(mask, threads=args.threads, repeats=args.repeats, warmup=args.warmup,
                      size=args.size, batch=args.batch, loops=args.loops)
        item["preset"] = name
        runs[name] = item
    return {"mode": "threads", "threads": args.threads, "logical_cpu_count": cores, "runs": runs}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cores", "threads", "single"), default="cores")
    parser.add_argument("--cpu", type=int, default=0, help="single 模式：逻辑核编号")
    parser.add_argument("--mask", type=int, default=0, help="single 模式：直接给亲和性掩码")
    parser.add_argument("--threads", type=int, default=8, help="torch 线程数（HW-ADMIT 用 8）")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--size", type=int, default=896, help="矩阵宽度（HW-ADMIT 的 hidden 宽度）")
    parser.add_argument("--batch", type=int, default=64, help="批大小（对应最小 cell 64 token）")
    parser.add_argument("--loops", type=int, default=20, help="每个样本内重复的 matmul 次数")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.mode == "single":
        return _run_single(args)

    started = datetime.now(timezone.utc)
    if args.mode == "cores":
        report = _mode_cores(args)
    else:
        report = _mode_threads(args)
    report.update({
        "schema_version": "qlh.torch_cpu_jitter_diagnosis.v1",
        "created_at_utc": started.isoformat(timespec="seconds"),
        "ticket": "TORCH-HW-ADMIT-01",
        "note": "只诊断：不改 CV 门、不放宽阈值、不产出可进 planner 的成本",
        "host": {
            "platform": sys.platform,
            "logical_cpu_count": _logical_cpu_count(),
            "windows_processor_topology": _windows_processor_topology(),
        },
        "params": {"threads": args.threads, "repeats": args.repeats, "warmup": args.warmup,
                   "size": args.size, "batch": args.batch},
    })
    if args.mode == "cores":
        print(f"逐核中位：fastest={report['fastest_mean_ms']} ms, "
              f"slowest={report['slowest_mean_ms']} ms, "
              f"slowest/fastest={report['slowest_over_fastest']}")
        print(f"排序均值(ms)：{report['sorted_means_ms']}")
    else:
        for name, run in report["runs"].items():
            stats = run.get("stats")
            print(f"{name:<22} cv={stats['cv'] if stats else 'n/a'} "
                  f"mean={stats['mean_ms'] if stats else 'n/a'} ms "
                  f"spread={stats['spread_pct'] if stats else 'n/a'}%")
    target = args.out or (DEFAULT_OUT / f"cpu-jitter-diagnosis-{args.mode}-"
                          f"{started.strftime('%Y%m%dT%H%M%S')}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
