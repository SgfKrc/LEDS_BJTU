#!/usr/bin/env python
"""verify_xframe_longseq.py — #9 准入证据②：**长序列 / 更长 prefill** 对照

## 为什么需要
`docs/跨框架层接力-项目报告.md` §7「未完成/待办」第 1 条列出的准入条件之一：**长序列**。
已有证据只到 141 token（`CORE-RELAY-XFRAME-01-dl-*-141-*.json`）。本脚本把 **prefill 长度**
拉到 512，仍按 `per_token_argmax` 判据逐 token 对照整模 llama.cpp 贪心基线。

## 方法
* 用同一段文本重复拼接，构造**确定长度**的 prompt（token 数由 tokenizer 实测，不假设）；
* 每条长度分别跑：`L→L`（对照）与 `D→L`（本票主体）；
* 记录：token 数、两侧序列、matched/total、首个不一致位置、耗时。

注意：`drive_dl_relay.py` 每步对**整段序列**重算上游，成本随长度线性以上增长，
因此长档位只跑少量生成步（`--gen`）—— 判据是**逐 token 一致**，不是吞吐。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> 仓库根（入仓后由 build/ 下的副本移入）
POC = ROOT / "build" / "cross-framework-layer-poc"
BIN = POC / "llama.cpp" / "build-cpu" / "bin"
WHOLE = POC / "out" / "qwen35-2b-f16.gguf"
CUT = POC / "out" / "qwen35-2b-f16-cut.gguf"
MODEL_DIR = ROOT / "models" / "qwen3-5-2b"

BASE_TEXT = (
    "The history of computing spans several centuries of incremental invention. "
    "Early mechanical calculators gave way to electromechanical relays, then to vacuum tubes, "
    "then to discrete transistors, and finally to integrated circuits. "
)

RELAY_LINE = re.compile(r"^\s*relay\s*\((\d+)\):\s*([\d\s]+)$")
BASELINE_STEP = re.compile(r"^\s*step\s+(\d+):\s*token=(\d+)\s*$")
RELAY_STEP = re.compile(r"^\s*step\s+(\d+):\s*relay_argmax=(\d+)\s+baseline=(\d+)\s+(\w+)\s*$")


def _run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    env = dict(os.environ)
    env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                          encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _make_prompt(target_tokens: int, prompt_dir: Path, tag: str) -> tuple[Path, int]:
    """把 BASE_TEXT 重复到 >= target_tokens，返回 (文件, 实际 token 数)。"""
    sys.path.insert(0, str(ROOT / "src"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    text = BASE_TEXT
    while len(tok(text, add_special_tokens=False)["input_ids"]) < target_tokens:
        text += BASE_TEXT
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    pf = prompt_dir / f"longseq-{tag}.txt"
    pf.write_text(text.replace("\n", " ") + "\n", encoding="utf-8")
    return pf, n


def run_l2l(pf: Path, gen: int, threads: int, n_ctx: int) -> dict:
    rc, out = _run([str(BIN / "llama-relay-gen.exe"), str(WHOLE), str(CUT),
                    "--prompt", str(pf), "--gen", str(gen),
                    "--threads", str(threads), "--tag", "longseq",
                    "--n-ctx", str(n_ctx)])
    baseline, relay = [], []
    for line in out.splitlines():
        m = BASELINE_STEP.match(line)
        if m:
            baseline.append(int(m.group(2)))
            continue
        m = RELAY_STEP.match(line)
        if m:
            relay.append(int(m.group(2)))
    return {"rc": rc, "baseline": baseline, "l2l_relay": relay}


def run_dl(pf: Path, gen: int, threads: int) -> dict:
    rc, out = _run([sys.executable, str(POC / "drive_dl_relay.py"),
                    "--prompt", str(pf), "--gen", str(gen),
                    "--runner", str(BIN / "llama-relay-gen-dl.exe"),
                    "--cut-model", str(CUT), "--model-dir", str(MODEL_DIR),
                    "--threads", str(threads)])
    relay: list[int] = []
    for line in out.splitlines():
        m = RELAY_LINE.match(line)
        if m:
            relay = [int(x) for x in m.group(2).split()]
    return {"rc": rc, "dl_relay": relay}


def compare(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    matched = sum(1 for i in range(n) if a[i] == b[i])
    return {"matched": matched, "total": n, "all_match": bool(n > 0 and matched == n and len(a) == len(b)),
            "first_mismatch_index": next((i for i in range(n) if a[i] != b[i]), None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="64,128,256,512")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out", default=str(POC / "out" / "xframe-longseq.json"))
    args = ap.parse_args()

    prompt_dir = POC / "out" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for spec in args.lengths.split(","):
        target = int(spec.strip())
        pf, n_tok = _make_prompt(target, prompt_dir, str(target))
        print(f"\n{'=' * 74}\n### 目标 prefill={target}  实际 token={n_tok}\n{'=' * 74}", flush=True)

        # 上下文至少覆盖 prompt + gen，留出余量（对齐 relay-gen-dl 的 4096 默认）
        n_ctx = max(4096, n_tok + args.gen + 64)
        t0 = time.perf_counter()
        l2l = run_l2l(pf, args.gen, args.threads, n_ctx)
        t1 = time.perf_counter()
        dl = run_dl(pf, args.gen, args.threads)
        t2 = time.perf_counter()

        base = l2l["baseline"]
        v_l2l = compare(base, l2l["l2l_relay"])
        v_dl = compare(base, dl["dl_relay"])
        print(f"  baseline({len(base)}) = {base}")
        print(f"  L→L  rc={l2l['rc']} ({len(l2l['l2l_relay'])}) {v_l2l}  {t1 - t0:.1f}s")
        print(f"  D→L  rc={dl['rc']} ({len(dl['dl_relay'])}) {v_dl}  {t2 - t1:.1f}s")

        results.append({"target_prefill": target, "actual_tokens": n_tok, "gen": args.gen,
                        "baseline": base,
                        "l2l": {**v_l2l, "tokens": l2l["l2l_relay"], "rc": l2l["rc"],
                                "seconds": round(t1 - t0, 2)},
                        "dl": {**v_dl, "tokens": dl["dl_relay"], "rc": dl["rc"],
                               "seconds": round(t2 - t1, 2)}})

    ok_dl = [r for r in results if r["dl"]["all_match"]]
    ok_l2l = [r for r in results if r["l2l"]["all_match"]]
    report = {
        "ticket": "CORE-RELAY-XFRAME-01",
        "evidence": "long-sequence",
        "date": datetime.now(timezone.utc).astimezone().isoformat(),
        "criterion": "per_token_argmax",
        "whole_model_gguf": WHOLE.name,
        "cut_model_gguf": CUT.name,
        "lengths_total": len(results),
        "dl_all_match": len(ok_dl),
        "l2l_all_match": len(ok_l2l),
        "max_tested_prefill": max((r["actual_tokens"] for r in results), default=0),
        "results": results,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'=' * 74}\n### 汇总\n{'=' * 74}")
    print(f"  档位 = {len(results)}  最长 prefill = {report['max_tested_prefill']} token  gen = {args.gen}")
    print(f"  D→L 全一致 = {len(ok_dl)}/{len(results)}；L→L 全一致 = {len(ok_l2l)}/{len(results)}")
    print(f"  [json] {out}")
    return 0 if len(ok_dl) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
