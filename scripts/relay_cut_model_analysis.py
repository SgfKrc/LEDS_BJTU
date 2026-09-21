#!/usr/bin/env python
"""relay_cut_model_analysis.py — 切点扫描的**噪声量化**与"是否存在内部最优"判定（P2 遗留②）。

背景（P2 收口时的两条遗留）：

1. 单轮扫描的噪声可能**大于**切点之间的差异，因此"最优切点"不可信；
2. 现有求解器的成本模型是 `fixed + per_layer × layers`（`src/relay_cut_objective.py`
   的 `_estimate_latency` / `SegmentProfile.ms_fixed_decode`），对切点 K **数学上线性**
   ⇒ 极值只会出现在端点。若实测出现"内部最优"，要么是噪声，要么模型缺了非线性项。

本工具因此不预设结论，而是用**多轮重复扫描**回答三件事：

- 每切点的中位数与 95% 置信区间（`t` 近似，样本少时保守用 `max-min` 半宽）；
- 线性模型对中位数的拟合优度（r²）；
- **是否存在显著的内部最优**：存在 K（非端点）使
  `total_median(K) + ci95(K) < min(total_median(端点) − ci95(端点))`。
  只有显著时才值得引入非线性项；否则"最优切点"是噪声，"切点搜索"应转向容量目标。

用法：

    python scripts/relay_cut_model_analysis.py \
        --records "build/cross-framework-layer-poc/out/p0-repeat/r*-k*.json" \
        --total-layers 24 --out build/relay-records/cut-model-analysis.json

记录文件名约定：`r<round>-k<cut>.json`（由多轮扫描脚本写出）。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

NAME_RE = re.compile(r"r(\d+)-k(\d+)$")


def _extract(path: Path) -> dict[str, object] | None:
    match = NAME_RE.search(path.stem)
    if match is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    verdict = data.get("verdict") or {}

    def _mean(key: str) -> float | None:
        value = (metrics.get(key) or {}).get("mean")
        return None if value is None else float(value)

    up, down = _mean("upstream_decode_ms"), _mean("downstream_decode_ms")
    if up is None or down is None:
        return None
    return {
        "record": path.name,
        "round": int(match.group(1)),
        "cut": int(match.group(2)),
        "passed": bool(verdict.get("passed")),
        "upstream_ms": up,
        "downstream_ms": down,
        "total_ms": up + down,
    }


def _ci95(values: list[float]) -> tuple[float, float]:
    """样本很少（n≤5）时用 `max−min` 的半宽做保守区间，而不是 t 分布近似。"""
    if len(values) < 2:
        return 0.0, 0.0
    half = (max(values) - min(values)) / 2.0
    return statistics.median(values), half


def _linear_fit(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "samples": float(n)}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="切点扫描噪声量化与内部最优判定（P2）")
    ap.add_argument("--records", action="append", required=True,
                    help="多轮扫描记录的 glob，可重复（如 .../p0-repeat/r*-k*.json）")
    ap.add_argument("--total-layers", type=int, required=True)
    ap.add_argument("--min-rounds", type=int, default=3,
                    help="每个切点的最少重复轮数；默认 3，避免单轮噪声进入决策")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for pattern in args.records:
        for path in sorted(Path(p) for p in glob.glob(pattern)):
            item = _extract(path)
            if item is None:
                print(f"[skip] 无法解析或缺少指标：{path.name}", file=sys.stderr)
                continue
            rows.append(item)
    if not rows:
        print("FAIL: 没有可解析的记录", file=sys.stderr)
        return 2

    if args.min_rounds < 1:
        print("FAIL: --min-rounds must be >= 1", file=sys.stderr)
        return 2

    by_cut: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_cut.setdefault(int(row["cut"]), []).append(row)

    round_counts = {cut: len(items) for cut, items in by_cut.items()}
    if len(set(round_counts.values())) != 1 or any(
        count < args.min_rounds for count in round_counts.values()
    ):
        print(
            "FAIL: every measured cut must have the same number of rounds "
            f"and at least {args.min_rounds}; got {round_counts}",
            file=sys.stderr,
        )
        return 2
    failed = [str(row["record"]) for row in rows if not bool(row["passed"])]
    if failed:
        print(
            "FAIL: correctness verdict failed in repeated scan: "
            + ", ".join(failed),
            file=sys.stderr,
        )
        return 2

    table: list[dict[str, object]] = []
    for cut in sorted(by_cut):
        items = by_cut[cut]
        totals = [float(i["total_ms"]) for i in items]
        ups = [float(i["upstream_ms"]) for i in items]
        downs = [float(i["downstream_ms"]) for i in items]
        _median_of_totals, half = _ci95(totals)
        upstream_median = statistics.median(ups)
        downstream_median = statistics.median(downs)
        table.append({
            "cut": cut,
            "rounds": len(items),
            "all_passed": all(bool(i["passed"]) for i in items),
            "upstream_median_ms": round(upstream_median, 4),
            "downstream_median_ms": round(downstream_median, 4),
            # The synthetic records consumed by relay_cut_plan.py use the
            # median of each segment. Keep this value additive and expose the
            # median of per-run totals separately because medians do not add.
            "total_median_ms": round(upstream_median + downstream_median, 4),
            "median_of_totals_ms": round(_median_of_totals, 4),
            "total_min_ms": round(min(totals), 4),
            "total_max_ms": round(max(totals), 4),
            "total_halfwidth_ms": round(half, 4),
            "spread_pct": round(100.0 * (max(totals) - min(totals)) /
                                   (upstream_median + downstream_median), 2)
            if upstream_median + downstream_median else None,
        })

    cuts = [int(e["cut"]) for e in table]
    totals = [float(e["total_median_ms"]) for e in table]
    fit = _linear_fit([float(c) for c in cuts], totals)

    # 是否存在**显著**的内部最优：该点的上界要低于两个端点各自的下界
    best = min(table, key=lambda e: float(e["total_median_ms"]))
    ends = [e for e in table if e["cut"] in (min(cuts), max(cuts))]
    interior = [e for e in table if e["cut"] not in (min(cuts), max(cuts))]
    best_interior = min(interior, key=lambda e: float(e["total_median_ms"])) if interior else None
    significant = False
    evidence = "no interior cut measured"
    if best_interior is not None and ends:
        interior_upper = float(best_interior["total_median_ms"]) + float(best_interior["total_halfwidth_ms"])
        end_lower = min(float(e["total_median_ms"]) - float(e["total_halfwidth_ms"]) for e in ends)
        significant = interior_upper < end_lower
        evidence = (f"interior k={best_interior['cut']} upper={round(interior_upper, 4)} vs "
                    f"endpoint lower={round(end_lower, 4)}")

    median_spread = max((float(e["spread_pct"] or 0.0)) for e in table)
    report = {
        "schema_version": "qlh.relay_cut_model_analysis.v1",
        "total_layers": args.total_layers,
        "min_rounds": args.min_rounds,
        "rounds_per_cut": round_counts,
        "records": [str(i["record"]) for i in rows],
        "by_cut": table,
        "linear_fit_total_median": fit,
        "best_cut_by_median": best["cut"],
        "best_interior_cut": (best_interior["cut"] if best_interior else None),
        "interior_optimum_significant": significant,
        "interior_optimum_evidence": evidence,
        "max_spread_pct": median_spread,
        "verdict": {
            "input_valid": True,
            "has_significant_interior_optimum": significant,
            "noise_dominates": (median_spread > 10.0) and not significant,
            "note": ("内部最优不显著 ⇒ 切点搜索应转向容量可行性"
                     if not significant else
                     "内部最优显著 ⇒ 需要给成本模型补非线性项"),
        },
    }

    header = (f"{'cut':>4} {'rounds':>6} {'up(med)':>9} {'down(med)':>10} {'total(med)':>11} "
              f"{'±':>7} {'spread%':>8}  passed")
    print(header)
    print("-" * len(header))
    for entry in table:
        print(f"{entry['cut']:>4} {entry['rounds']:>6} {entry['upstream_median_ms']:>9} "
              f"{entry['downstream_median_ms']:>10} {entry['total_median_ms']:>11} "
              f"{entry['total_halfwidth_ms']:>7} {str(entry['spread_pct']):>8}  {entry['all_passed']}")
    if fit:
        print(f"\n线性拟合（中位数）：slope={fit['slope']:.4f} intercept={fit['intercept']:.4f} "
              f"r²={fit['r2']:.4f} n={int(fit['samples'])}")
    print(f"最大相对离散：{median_spread}%；显著内部最优：{significant}（{evidence}）")
    print(f"[verdict] {report['verdict']['note']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[record] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
