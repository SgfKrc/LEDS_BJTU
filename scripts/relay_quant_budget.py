#!/usr/bin/env python
"""relay_quant_budget.py — 由接力实验记录汇总**精度预算表**，并给出选档建议（P4）。

回答 P4 票面第 4 问："何时应优先容量可行性，何时应优先质量/速度"。

核心判据（来自 P4 边界扫描的实测结论）：

- **argmax 不具备分辨力**：稀疏的量化档在单个 prompt 上照样全绿；
- **绝对 margin 具备预测力**：翻转档的 relay margin 明显低于未翻转档（实测落点在 4.0~4.8），
  而**相对衰减 Δ 不能预测**（基线低的 prompt 衰减 13.9% 就翻、基线高的衰减 30% 仍然不翻）；
- 因此预算用**最坏情况 margin**（`margin_min`）而不是均值，并要求
  `margin_min ≥ flip_threshold + safety_margin` 才算 safe。

用法：

    python scripts/relay_quant_budget.py \
        --records "build/relay-records/p4b-*.json" \
        --flip-threshold 4.5 --safety-margin 0.5 \
        --out build/relay-records/quant-budget.json

记录文件名约定（由实验驱动写出）：`<prefix>-<prompt>-<upstream>-<hidden>.json`，
档位值域有限，因此从**右往左**匹配即可，不依赖记录里是否写了上游档位字段。
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

UPSTREAM_QUANTS = ("fp16", "f32", "int8", "int4", "nf4")
HIDDEN_QUANTS = ("none", "f16", "bf16", "int8_block128", "int4_block128")

#: ★ A10：判定严重度排序（`--fail-on` 闸门用）。
#: `tight` = 余量不足；`no-headroom` = 该档遇到的 prompt 基线本身就没有余量（**没有可用的安全证据**）
#: ⇒ 与 `tight` 同级（保守）；`unsafe` = 有余量的 prompt 上仍出现翻转 ⇒ 最高。
_VERDICT_RANK = {"safe": 0, "safe-headroom-only": 0, "tight": 1, "no-headroom": 1, "unsafe": 2}


def _parse_case(stem: str) -> tuple[str, str, str] | None:
    """从文件名 stem 解析 `(prompt, upstream, hidden)`；不匹配返回 None。"""
    if "-" not in stem:
        return None
    hidden = next((h for h in sorted(HIDDEN_QUANTS, key=len, reverse=True)
                   if stem.endswith("-" + h)), None)
    if hidden is None:
        return None
    head = stem[: -(len(hidden) + 1)]
    upstream = next((u for u in sorted(UPSTREAM_QUANTS, key=len, reverse=True)
                     if head.endswith("-" + u)), None)
    if upstream is None:
        return None
    return head[: -(len(upstream) + 1)], upstream, hidden


def _load_records(patterns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            stem = Path(path).stem
            parsed = _parse_case(stem)
            if parsed is None:
                print(f"[skip] 无法从文件名解析档位：{path}", file=sys.stderr)
                continue
            prompt, upstream, hidden = parsed
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            metrics = data.get("metrics") or {}
            verdict = data.get("verdict") or {}
            profile = data.get("device_profile") or {}
            relay = (metrics.get("logit_margin") or {}).get("mean")
            base = (metrics.get("baseline_logit_margin") or {}).get("mean")
            rows.append({
                "record": Path(path).name,
                "prompt": prompt,
                "upstream": upstream,
                "hidden": hidden,
                "passed": bool(verdict.get("passed")),
                "matched": int(verdict.get("matched_runs") or 0),
                "total": int(verdict.get("total_runs") or 0),
                "relay_margin": relay,
                "baseline_margin": base,
                "delta_pct": (None if not relay or not base
                              else round(100.0 * (relay - base) / base, 2)),
                "wire_bytes_per_token": profile.get("hidden_wire_bytes_per_token"),
                "upstream_resident_bytes": (metrics.get("resident_weight_bytes") or {})
                                           .get("upstream"),
            })
    return rows


def _validate_prompt_coverage(rows: list[dict[str, object]], min_prompts: int) -> None:
    if min_prompts < 1:
        raise ValueError("--min-prompts must be >= 1")
    buckets: dict[tuple[str, str], set[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["upstream"]), str(row["hidden"]))
        prompt = str(row["prompt"])
        identity = (*key, prompt)
        if identity in seen:
            raise ValueError(f"duplicate prompt record in quantization bucket: {identity}")
        seen.add(identity)
        required = ("relay_margin", "baseline_margin",
                    "wire_bytes_per_token", "upstream_resident_bytes")
        if any(row[field] is None for field in required):
            raise ValueError(f"missing required metric in record: {row['record']}")
        buckets.setdefault(key, set()).add(prompt)
    sizes = {key: len(prompts) for key, prompts in buckets.items()}
    if any(size < min_prompts for size in sizes.values()):
        raise ValueError(
            f"each quantization bucket needs at least {min_prompts} prompts; "
            f"got {sizes}"
        )
    prompt_sets = {frozenset(prompts) for prompts in buckets.values()}
    if len(prompt_sets) > 1:
        raise ValueError(
            "quantization buckets must use the same prompt set; "
            f"got {sizes}"
        )


def _aggregate(rows: list[dict[str, object]], gate: float) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((str(row["upstream"]), str(row["hidden"])), []).append(row)

    table: list[dict[str, object]] = []
    for (upstream, hidden), items in sorted(buckets.items()):
        margins = [float(r["relay_margin"]) for r in items if r["relay_margin"] is not None]
        bases = [float(r["baseline_margin"]) for r in items
                 if r["baseline_margin"] is not None]
        deltas = [float(r["delta_pct"]) for r in items if r["delta_pct"] is not None]
        flipped = [r for r in items if not r["passed"]]
        # ★ 分层：基线 margin 本身就低于闸门的 prompt，**没有任何预算空间**可用
        #   （即使全精度接力也在边缘）⇒ 必须把这类 prompt 与"量化吃掉的余量"分开看，
        #   否则对照档（fp16×f16）也会被判 unsafe，闸门就失去意义。
        headroom = [r for r in items
                    if r["baseline_margin"] is not None and float(r["baseline_margin"]) >= gate]
        headroom_flipped = [r for r in headroom if not r["passed"]]
        margins_h = [float(r["relay_margin"]) for r in headroom
                     if r["relay_margin"] is not None]
        table.append({
            "upstream": upstream,
            "hidden": hidden,
            "n_prompts": len(items),
            "n_flipped": len(flipped),
            "flip_rate": round(len(flipped) / max(1, len(items)), 4),
            "flipped_prompts": sorted(str(r["prompt"]) for r in flipped),
            "n_baseline_no_headroom": len(items) - len(headroom),
            "n_headroom_flipped": len(headroom_flipped),
            "headroom_flipped_prompts": sorted(str(r["prompt"]) for r in headroom_flipped),
            "margin_min": min(margins) if margins else None,
            "margin_min_headroom": min(margins_h) if margins_h else None,
            "margin_median": (round(statistics.median(margins), 4) if margins else None),
            "margin_mean": (round(statistics.fmean(margins), 4) if margins else None),
            "baseline_margin_min": min(bases) if bases else None,
            "delta_min_pct": min(deltas) if deltas else None,
            "delta_median_pct": (round(statistics.median(deltas), 2) if deltas else None),
            "wire_bytes_per_token": items[0].get("wire_bytes_per_token"),
            "upstream_resident_bytes": items[0].get("upstream_resident_bytes"),
        })
    return table


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="接力量化精度预算表（P4）")
    ap.add_argument("--records", action="append", required=True,
                    help="记录 glob，可重复（如 build/relay-records/p4b-*.json）")
    ap.add_argument("--flip-threshold", type=float, default=4.5,
                    help="margin 翻转分界（实测落点 4.0~4.8，默认取中点 4.5）")
    ap.add_argument("--safety-margin", type=float, default=0.5,
                    help="安全余量：要求 margin_min ≥ threshold + safety_margin")
    ap.add_argument("--min-prompts", type=int, default=6,
                    help="每个量化档至少覆盖的 prompt 数；默认 6")
    ap.add_argument("--fail-on", choices=("off", "unsafe", "tight"), default="off",
                    help="★ A10：出现该级别判定时以非零码退出 ⇒ 供 CI / 驱动当闸门用。"
                         "off=只报告（默认，保持旧行为）；unsafe=有档位不安全即红；"
                         "tight=连余量不足 / 无安全证据也红")
    ap.add_argument("--out", default=None, help="汇总 JSON 落盘路径")
    args = ap.parse_args()

    rows = _load_records(args.records)
    if not rows:
        print("FAIL: 没有可解析的记录", file=sys.stderr)
        return 2

    try:
        _validate_prompt_coverage(rows, args.min_prompts)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    gate = args.flip_threshold + args.safety_margin
    table = _aggregate(rows, gate)
    for entry in table:
        worst = entry.get("margin_min_headroom")
        entry["gate"] = gate
        # 判定用"基线本身已低于闸门"的 prompt 的**互补子集**：那些 prompt 没有任何预算空间，
        # 把它们算进来会让对照档也判 unsafe（实测：fp16×f16 因为 code 的基线只有 3.76 被判 tight）。
        if worst is None:
            entry["verdict"] = "no-headroom"      # 这一档遇到的全是无余量 prompt
        elif entry["n_headroom_flipped"]:
            entry["verdict"] = "unsafe"
        elif float(worst) < gate:
            entry["verdict"] = "tight"
        elif entry["n_flipped"]:
            # Low-baseline prompts must be routed to full precision first.
            entry["verdict"] = "safe-headroom-only"
        else:
            entry["verdict"] = "safe"

    header = (f"{'upstream':<7} {'hidden':<15} {'n':>3} {'flip':>5} {'no_hd':>6} "
              f"{'min(hd)':>9} {'median':>8} {'Δmin%':>7} {'wire B':>7} {'up MB':>7}  verdict")
    print(header)
    print("-" * len(header))
    for entry in table:
        up_mb = entry["upstream_resident_bytes"]
        print(f"{entry['upstream']:<7} {entry['hidden']:<15} {entry['n_prompts']:>3} "
              f"{entry['n_flipped']:>5} {entry['n_baseline_no_headroom']:>6} "
              f"{str(entry['margin_min_headroom']):>9} "
              f"{str(entry['margin_median']):>8} {str(entry['delta_min_pct']):>7} "
              f"{str(entry['wire_bytes_per_token']):>7} "
              f"{(round(float(up_mb) / 1e6, 1) if up_mb else 'n/a'):>7}  {entry['verdict']}")
    print(f"\n判据：在「基线 margin ≥ 闸门」的 prompt 子集上要求 margin_min ≥ "
          f"flip_threshold({args.flip_threshold}) + safety_margin({args.safety_margin}) = {gate}")
    print("说明：`no_hd` = 基线本身低于闸门的 prompt 数（这类 prompt 无预算空间，单独计数不计入判定）")

    # 选档建议：在 safe 档里选"线路字节 + 上游驻留"最小者（容量优先，质量已由闸门保证）
    safe = [e for e in table if e["verdict"] in {"safe", "safe-headroom-only"}]
    if safe:
        def _cost(e: dict[str, object]) -> float:
            wire = float(e["wire_bytes_per_token"] or 0)
            up = float(e["upstream_resident_bytes"] or 0) / 1e6
            return wire + up * 10          # 上游驻留按 10 B 权重折算，仅为排序口径
        best = min(safe, key=_cost)
        print(f"[建议] 在 {len(safe)} 个候选档中按容最优先推荐："
              f"{best['upstream']} x {best['hidden']}"
              f"（margin_min(有余量子集)={best['margin_min_headroom']}，"
              f"线路 {best['wire_bytes_per_token']} B/token，"
              f"上游驻留 {round(float(best['upstream_resident_bytes'] or 0) / 1e6, 1)} MB，"
              f"无余量 prompt {best['n_baseline_no_headroom']} 条）")
    else:
        print("[建议] **没有**档位通过闸门 ⇒ 只能提高安全余量之外的精度"
              "（回到 fp16 上游 + f16 hidden）或缩短链路")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "flip_threshold": args.flip_threshold,
            "safety_margin": args.safety_margin,
            "gate": gate,
            "min_prompts": args.min_prompts,
            "table": table,
            "records": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[record] {args.out}")

    if args.fail_on != "off":
        # ★ A10：闸门 —— 判定达到 `--fail-on` 级别就以 1 退出（CI / 驱动据此变红）。
        threshold = _VERDICT_RANK[args.fail_on]
        offenders = [e for e in table
                     if _VERDICT_RANK.get(str(e["verdict"]), 2) >= threshold]
        if offenders:
            detail = ", ".join(f"{e['upstream']}x{e['hidden']}={e['verdict']}"
                               for e in offenders[:4])
            more = "" if len(offenders) <= 4 else f"（共 {len(offenders)} 个）"
            print(f"[gate] FAIL: --fail-on {args.fail_on} 命中 {detail}{more}", file=sys.stderr)
            return 1
        print(f"[gate] OK: {len(table)} 个档位均未达到 --fail-on {args.fail_on} 级别")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
