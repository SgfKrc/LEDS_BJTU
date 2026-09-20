#!/usr/bin/env python
"""bench_pytorch_vs_llama.py — #12 `CORE-PYTORCH-COMPARE-01`：**Koakuma 两 backend 对照**

## 票面要求
「与 llama.cpp 在同模型/同提示下比较**正确性、内存、吞吐和故障边界**，并为 D→L Relay
提供来源侧证据；作为有 CUDA 节点的默认加速 backend 进入生产路径，同时保留对照用途。」

## 方法学（遵守仓库纪律）
* **必须用主仓引擎**（`src/model_module.py` 的 `ModelManager`），不手写前向；
* **同一模型工件**：`models/qwen3-5-2b`（safetensors，PyTorch 侧）与其 GGUF（llama.cpp 侧）
  —— 注意两者**不是同一权重文件**（GGUF 为转换+量化），因此**正确性判据用行为等价**
  （逐 token argmax / top-k 重叠），**不要求** bitwise；
* **吞吐做重复测量**并给 `mean ± std`（单次数字不作结论）；
* **加载 / 首次生成 / 稳态分段测量**（含一次性成本，不能拿总时间除步数）；
* 记录 **VRAM 峰值**与失败模式（故障边界）。

## 用法
    python bench_pytorch_vs_llama.py --rounds 3 --max-tokens 48 \
        --json-out out/pytorch-vs-llama.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> 仓库根（入仓后由 build/ 下的副本移入）
POC = ROOT / "build" / "cross-framework-layer-poc"
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "qwen3-5-2b"
PROMPTS: list[tuple[str, str]] = [
    ("en-factual", "The capital of France is"),
    ("zh-factual", "中国的首都是"),
    ("en-reasoning", "If it rains, the ground gets wet. It rained. Therefore, the ground"),
    ("code", "def add(a, b):\n    return"),
]


def _reset_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _vram_peak_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:  # noqa: BLE001
        pass
    return None


def _extract(result) -> tuple[str, list[int] | None, int | None]:
    """兼容 str / dict 两种返回，取文本与 token 数。"""
    if isinstance(result, dict):
        text = str(result.get("content", ""))
        usage = result.get("usage") or {}
        n = usage.get("completion_tokens")
        return text, None, n
    return str(result), None, None


def run_backend(engine: str, quant: str, rounds: int, max_tokens: int) -> dict:
    import model_module

    out: dict = {"engine": engine, "quant": quant}
    _reset_vram()
    t0 = time.perf_counter()
    mgr = model_module.ModelManager()
    try:
        mgr.load_model(model_id=MODEL_ID, engine=engine, quant_type=quant)
    except Exception as exc:  # noqa: BLE001 —— 故障边界也是证据
        out["load_ok"] = False
        out["load_error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["load_s"] = round(time.perf_counter() - t0, 2)
        return out
    out["load_ok"] = True
    out["load_s"] = round(time.perf_counter() - t0, 2)
    out["vram_peak_after_load_gb"] = _vram_peak_gb()

    results = []
    tps_all: list[float] = []
    for tag, prompt in PROMPTS:
        entry: dict = {"tag": tag, "prompt": prompt}
        texts, first_s, per_round_tps = [], None, []
        for r in range(rounds):
            t = time.perf_counter()
            try:
                res = mgr.chat([{"role": "user", "content": prompt}],
                               max_tokens=max_tokens, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
                break
            dt = time.perf_counter() - t
            text, _tokens, n_tok = _extract(res)
            if r == 0:
                first_s = round(dt, 3)
            texts.append(text)
            if n_tok and dt > 0:
                per_round_tps.append(n_tok / dt)
        entry["first_generation_s"] = first_s
        entry["sample"] = texts[0][:200] if texts else ""
        entry["stable"] = all(t == texts[0] for t in texts) if len(texts) > 1 else None
        if per_round_tps:
            entry["tps_mean"] = round(statistics.mean(per_round_tps), 3)
            entry["tps_std"] = round(statistics.stdev(per_round_tps), 3) if len(per_round_tps) > 1 else 0.0
            tps_all.extend(per_round_tps)
        results.append(entry)

    if tps_all:
        out["tps_overall_mean"] = round(statistics.mean(tps_all), 3)
        out["tps_overall_std"] = round(statistics.stdev(tps_all), 3) if len(tps_all) > 1 else 0.0
    out["vram_peak_gb"] = _vram_peak_gb()
    out["results"] = results
    try:
        mgr.unload_model()
    except Exception:  # noqa: BLE001
        pass
    return out


def compare_texts(a: dict, b: dict) -> dict:
    """两 backend 的**行为等价**对照。

    ⚠️ 两侧工件不同（safetensors vs GGUF 转换+量化）且后端实现不同 ⇒ **不能要求逐字一致**。
    因此记录：
      * `lcp_chars` / `lcp_ratio`：最长公共前缀（衡量「开头是否走同一条轨迹」）；
      * `first_prefix_match`：是否至少前若干字符一致（答案开头相同）；
      * 样本本身（供人工判语义等价）。
    """
    by_a = {r["tag"]: r for r in a.get("results", [])}
    by_b = {r["tag"]: r for r in b.get("results", [])}
    rows = []
    for tag in sorted(set(by_a) & set(by_b)):
        ra, rb = by_a[tag], by_b[tag]
        ta, tb = ra.get("sample", ""), rb.get("sample", "")
        lcp = 0
        for ca, cb in zip(ta, tb):
            if ca != cb:
                break
            lcp += 1
        denom = max(1, min(len(ta), len(tb)))
        rows.append({
            "tag": tag,
            "identical": ta == tb,
            "lcp_chars": lcp,
            "lcp_ratio": round(lcp / denom, 4),
            "first_prefix_match": lcp >= 8,
            "a_sample": ta[:120],
            "b_sample": tb[:120],
        })
    return {
        "compared": len(rows),
        "identical": sum(1 for r in rows if r["identical"]),
        "prefix_match": sum(1 for r in rows if r["first_prefix_match"]),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--pytorch-quant", default="int4")
    ap.add_argument("--llama-quant", default="int4")
    ap.add_argument("--json-out", default=str(POC / "out" / "pytorch-vs-llama.json"))
    args = ap.parse_args()

    report: dict = {
        "ticket": "CORE-PYTORCH-COMPARE-01",
        "date": datetime.now(timezone.utc).astimezone().isoformat(),
        "model_id": MODEL_ID,
        "criterion": "行为等价（逐 token/样本对照）；**不要求 bitwise** —— 两侧工件不同"
                     "（safetensors vs 转换后的 GGUF），且量化后端不同",
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "prompts": [{"tag": t, "prompt": p} for t, p in PROMPTS],
    }

    print("=" * 74)
    print(f"### PyTorch backend（quant={args.pytorch_quant}）")
    print("=" * 74)
    pt = run_backend("pytorch", args.pytorch_quant, args.rounds, args.max_tokens)
    print(json.dumps({k: v for k, v in pt.items() if k != "results"}, ensure_ascii=False, indent=1))
    for r in pt.get("results", []):
        print(f"  [{r['tag']}] first={r.get('first_generation_s')}s "
              f"tps={r.get('tps_mean')}±{r.get('tps_std')} stable={r.get('stable')} "
              f"err={r.get('error', '-')}")
        print(f"      {r.get('sample', '')[:110]!r}")

    print()
    print("=" * 74)
    print(f"### llama.cpp backend（quant={args.llama_quant}）")
    print("=" * 74)
    lc = run_backend("llama_cpp", args.llama_quant, args.rounds, args.max_tokens)
    print(json.dumps({k: v for k, v in lc.items() if k != "results"}, ensure_ascii=False, indent=1))
    for r in lc.get("results", []):
        print(f"  [{r['tag']}] first={r.get('first_generation_s')}s "
              f"tps={r.get('tps_mean')}±{r.get('tps_std')} stable={r.get('stable')} "
              f"err={r.get('error', '-')}")
        print(f"      {r.get('sample', '')[:110]!r}")

    report["pytorch"] = pt
    report["llama_cpp"] = lc
    report["equivalence"] = compare_texts(pt, lc)

    print()
    print("=" * 74)
    print("### 对照汇总")
    print("=" * 74)
    for name, d in (("PyTorch", pt), ("llama.cpp", lc)):
        if d.get("load_ok"):
            print(f"  {name:10s} 加载 {d['load_s']:6.2f}s  "
                  f"tps {d.get('tps_overall_mean')}±{d.get('tps_overall_std')}  "
                  f"VRAM 峰值 {d.get('vram_peak_gb')} GB")
        else:
            print(f"  {name:10s} ❌ 加载失败 {d.get('load_error')}")
    eq = report["equivalence"]
    print(f"  行为等价：完全一致 {eq['identical']}/{eq['compared']}；"
          f"开头一致(>=8 字符) {eq['prefix_match']}/{eq['compared']}")
    for row in eq["rows"]:
        print(f"    {row['tag']:14s} lcp={row['lcp_chars']:4d} ({row['lcp_ratio']:.0%})  A={row['a_sample'][:52]!r}")
        print(f"      {' ' * 14}                          B={row['b_sample'][:52]!r}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  [json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
