#!/usr/bin/env python
"""relay_cut_plan.py — P2 切点重搜：从**实测记录**拟合段画像 → 求解最优切点 → 与实测最优点对比。

闭环（P2 判据「给定设备对，搜索使墙钟最小的切点并落可复算报告」）：

1. 读入同一切点扫描的端到端记录（P1 统一驱动记录 `build/relay-records/*.json`，或
   P0 时期的 runner 输出 `out/p0/*.json`）；
2. `src.relay_cut_objective.fit_two_segment` 从实测点回归出**每段的固定开销 + 每层耗时**；
3. `plan_relay_cut_n_segments` 在该设备对上求解最优切点（含合法切点约束，如 Qwen3.5 的
   4 层倍数）并输出 `capacity_feasible` / `latency_estimate` / `risk_penalty`；
4. 与**实测最优切点**（按实测 decode ms/step 最小）对比，落可复算报告 JSON。

用法::

    python scripts/relay_cut_plan.py \
        --records "build/cross-framework-layer-poc/out/p0/qwen25-05b-k*-b1-p32-g32.json" \
        --total-layers 24 --cut-multiple 1 --capacity-gb 16 \
        --json-out build/relay-records/cut-plan-qwen25.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.relay_cut_objective import (  # noqa: E402
    SegmentProfile,
    fit_two_segment,
    plan_relay_cut_n_segments,
)

REPORT_SCHEMA_VERSION = "qlh.relay_cut_plan_report.v1"
MIB = 1024 ** 2


def _mean(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def _extract(record: dict[str, Any]) -> dict[str, Any] | None:
    """兼容两种记录：P1 relay experiment record 与 P0 时期 runner 输出。"""
    layout = record.get("layer_layout") or {}
    layers = record.get("layers") or layout.get("upstream_layers")
    metrics = record.get("metrics") or {}
    timing = record.get("timing_ms") or {}
    upstream = _mean(metrics.get("upstream_decode_ms")) or _mean(timing.get("upstream_decode"))
    upstream = upstream if upstream is not None else _mean(record.get("upstream_ms_per_step"))
    downstream = _mean(metrics.get("downstream_decode_ms")) or _mean(timing.get("downstream_decode"))
    downstream = downstream if downstream is not None else _mean(record.get("downstream_ms_per_step"))
    if layers is None or upstream is None or downstream is None:
        return None
    whole = None
    resident = metrics.get("resident_weight_bytes") or {}
    if isinstance(resident, dict):
        whole = resident.get("whole")
    whole = whole or (record.get("models") or {}).get("whole", {}).get("model_bytes") \
        or record.get("whole_model_bytes")
    models = record.get("models") or {}
    upstream_model = models.get("upstream") or {}
    return {
        "source": record.get("experiment_id") or record.get("model"),
        "model": record.get("model") or upstream_model.get("id"),
        "kind": record.get("kind"),
        "path": record.get("path"),
        "commit": record.get("commit") or record.get("git_head"),
        "prefill": record.get("prefill") or (record.get("load") or {}).get("prefill_tokens"),
        "upstream_layers": int(layers),
        "upstream_decode_ms": upstream,
        "downstream_decode_ms": downstream,
        "total_ms": upstream + downstream,
        "gen": record.get("gen") or (record.get("load") or {}).get("gen_tokens"),
        "batch": record.get("batch") or (record.get("load") or {}).get("batch"),
        "whole_model_bytes": whole,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P2 切点重搜：实测拟合 → 求解 → 对比")
    ap.add_argument("--records", required=True, help="记录 glob（同一次切点扫描）")
    ap.add_argument("--total-layers", type=int, required=True)
    ap.add_argument("--cut-multiple", type=int, default=1,
                    help="合法切点步长（Qwen3.5 = full_attention_interval = 4）")
    ap.add_argument("--capacity-gb", type=float, default=16.0, help="每段可用容量（容量判据用）")
    ap.add_argument("--n-embd", type=int, default=896, help="模型 hidden 宽度（hidden 合同校验用）")
    ap.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    ap.add_argument("--rtt-ms", type=float, default=0.5)
    ap.add_argument("--layer-bytes-mib", type=float, default=None,
                    help="每层字节（默认由记录里的整模字节 / 层数推算）")
    ap.add_argument("--non-split-mib", type=float, default=8.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    paths = sorted(Path(p) for p in glob.glob(args.records))
    samples = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        extracted = _extract(record)
        if extracted:
            extracted["file"] = path.name
            samples.append(extracted)
    if len(samples) < 2:
        print(f"FAIL: 需要至少 2 条同切点扫描记录，实得 {len(samples)}（glob={args.records}）")
        return 2
    identity_fields = ("model", "kind", "path", "commit", "prefill", "gen", "batch")
    identity_errors = []
    for field in identity_fields:
        values = {sample.get(field) for sample in samples}
        if len(values) > 1:
            identity_errors.append(f"{field}={sorted(map(str, values))}")
    if identity_errors:
        print("FAIL: records mix multiple experiment identities: " + "; ".join(identity_errors))
        return 2

    samples.sort(key=lambda item: item["upstream_layers"])
    print(f"[records] {len(samples)} 条：cuts={[s['upstream_layers'] for s in samples]}")

    capacity_bytes = int(max(0.0, args.capacity_gb) * 1024 ** 3)
    base = {"capacity_bytes": capacity_bytes, "bandwidth_mbps": args.bandwidth_mbps,
            "rtt_ms": args.rtt_ms}
    fitted = fit_two_segment(
        samples, total_layers=args.total_layers,
        upstream_profile=SegmentProfile(node_id="upstream", engine="pytorch", **base),
        downstream_profile=SegmentProfile(node_id="downstream", engine="llama.cpp", **base))
    print(f"[fit] upstream  : {fitted['fit']['upstream']}")
    print(f"[fit] downstream: {fitted['fit']['downstream']}")

    whole_bytes = next((s["whole_model_bytes"] for s in samples if s["whole_model_bytes"]), None)
    if args.layer_bytes_mib is not None:
        layer_bytes = int(max(0.0, args.layer_bytes_mib) * MIB)
        layer_bytes_source = "explicit_layer_bytes_mib"
        non_split_bytes = int(max(0.0, args.non_split_mib) * MIB)
    elif whole_bytes:
        layer_bytes = int(whole_bytes / args.total_layers)
        layer_bytes_source = "uniform_whole_model_approximation"
        # The whole-model byte count already includes embedding/lm_head and
        # other non-layer weights. Adding them again would double count them.
        non_split_bytes = 0
        print("[capacity] warning: inferred layer bytes already include non-split weights; "
              "using non_split_bytes=0. Pass --layer-bytes-mib for explicit accounting.")
    else:
        layer_bytes = 0
        layer_bytes_source = "unavailable"
        non_split_bytes = 0
    if layer_bytes <= 0:
        print("FAIL: 无法确定每层字节（给 --layer-bytes-mib，或让记录带整模字节）")
        return 2

    plan = plan_relay_cut_n_segments(
        total_layers=args.total_layers,
        layer_bytes=[layer_bytes] * args.total_layers,
        n_embd=args.n_embd,
        segments=[fitted["upstream"], fitted["downstream"]],
        non_split_bytes=non_split_bytes,
        cut_multiple=args.cut_multiple,
        weights={"capacity": 0.0, "latency": 1.0, "risk": 0.0},
    )

    measured_optimum = min(samples, key=lambda item: item["total_ms"])

    def _predict_total_ms(cut: int) -> float:
        return (fitted["upstream"].ms_fixed_decode
                + fitted["upstream"].ms_per_layer_decode * cut
                + fitted["downstream"].ms_fixed_decode
                + fitted["downstream"].ms_per_layer_decode * (args.total_layers - cut))

    # 闭环判据：把拟合模型用**同一批实测切点**打分，其 argmin 是否等于实测最优。
    # 全域最优（plan.cuts）可能落在未实测的切点上，那种情况报告要标记出来而不是判 fail。
    scored = sorted(samples, key=lambda item: _predict_total_ms(item["upstream_layers"]))
    predicted_best_measured = scored[0]["upstream_layers"]
    predicted_cut = plan.cuts[0] if plan.cuts else None
    measured_cuts = [s["upstream_layers"] for s in samples]
    predicted_cut_in_measured_range = predicted_cut in measured_cuts
    fitted_matches_measured = predicted_best_measured == measured_optimum["upstream_layers"]
    verdict = {
        # Matching only the measured subset does not validate an unmeasured
        # global optimum. Keep the acceptance criterion fail-closed.
        "passed": bool(plan.admitted and predicted_cut_in_measured_range and fitted_matches_measured),
        "reason": ("fitted_model_reproduces_measured_optimum"
                   if predicted_cut_in_measured_range and fitted_matches_measured
                   else "predicted_cut_not_measured"
                   if not predicted_cut_in_measured_range
                   else "fitted_model_disagrees_with_measured_optimum"),
        "predicted_cut": predicted_cut,
        "predicted_cut_in_measured_range": predicted_cut_in_measured_range,
        "predicted_best_measured_cut": predicted_best_measured,
        "measured_optimum_cut": measured_optimum["upstream_layers"],
        "predicted_ms_by_measured_cut": [
            {"cut": s["upstream_layers"], "predicted_ms": round(_predict_total_ms(s["upstream_layers"]), 4),
             "measured_ms": round(s["total_ms"], 4)} for s in samples],
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "records_glob": args.records,
        "records_used": [s["file"] for s in samples],
        "total_layers": args.total_layers,
        "cut_multiple": args.cut_multiple,
        "capacity_gb": args.capacity_gb,
        "layer_bytes": layer_bytes,
        "layer_bytes_source": layer_bytes_source,
        "non_split_bytes": non_split_bytes,
        "fitted": {"upstream": fitted["upstream"].to_dict(),
                   "downstream": fitted["downstream"].to_dict(), "fit": fitted["fit"]},
        "plan": plan.to_dict(),
        "measured": {
            "by_cut": [{"cut": s["upstream_layers"], "upstream_ms": round(s["upstream_decode_ms"], 4),
                        "downstream_ms": round(s["downstream_decode_ms"], 4),
                        "total_ms": round(s["total_ms"], 4)} for s in samples],
            "optimum_cut": measured_optimum["upstream_layers"],
            "optimum_total_ms": round(measured_optimum["total_ms"], 4),
        },
        "verdict": verdict,
    }
    print(f"[plan] predicted_cut={predicted_cut} (in measured range: "
          f"{predicted_cut in measured_cuts}) measured_optimum="
          f"{measured_optimum['upstream_layers']} passed={verdict['passed']}")
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[report] {target}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
