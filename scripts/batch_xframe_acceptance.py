#!/usr/bin/env python
"""batch_xframe_acceptance.py — #9 准入证据①：**prompt 分布**离线对照

## 为什么需要
`docs/跨框架层接力-项目报告.md` §7「未完成/待办」第 1 条：生产门准入需要
**prompt 分布统计**（**单 prompt 结论不可泛化**）。此前所有 D→L 证据都是单一 prompt。

## 判据
对每条 prompt，比较两个序列与**整模 llama.cpp 贪心基线**：
  * `L→L`：`llama-relay-gen.exe 整模 裁层`（对照，已知成立）
  * `D→L`：`drive_dl_relay.py`（PyTorch f32 上游 + 裁层 llama.cpp 下游）
判据 = **逐 token argmax 一致**（`RELAY_ACCEPTANCE`）。
输出逐 prompt 的 `matched/total` 与全局一致率，落 JSON 证据。

## 用法
    python batch_xframe_acceptance.py --gen 16 --json-out out/xframe-prompt-dist.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]   # scripts/ -> 仓库根（入仓后由 build/ 下的副本移入）
POC = ROOT / "build" / "cross-framework-layer-poc"
BIN = POC / "llama.cpp" / "build-cpu" / "bin"
WHOLE = POC / "out" / "qwen35-2b-f16.gguf"
CUT = POC / "out" / "qwen35-2b-f16-cut.gguf"
MODEL_DIR = ROOT / "models" / "qwen3-5-2b"

#: 覆盖不同分布族（英文事实/推理、中文问答/指令、代码、数学、长文续写、技术定义）
PROMPTS: list[tuple[str, str]] = [
    ("en-factual", "The capital of France is"),
    ("en-reasoning", "If it rains, the ground gets wet. It rained. Therefore, the ground"),
    ("en-longform", "Once upon a time, in a small village by the sea, there lived a"),
    ("zh-factual", "中国的首都是"),
    ("zh-instruction", "请用一句话解释什么是机器学习："),
    ("code-python", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
    ("math-arith", "Compute 2 + 3 * 4 step by step: first we compute"),
    ("en-definition", "In computer science, a hash table is a data structure that"),
]

RELAY_LINE = re.compile(r"^\s*relay\s*\((\d+)\):\s*([\d\s]+)$")
BASELINE_STEP = re.compile(r"^\s*step\s+(\d+):\s*token=(\d+)\s*$")
RELAY_STEP = re.compile(r"^\s*step\s+(\d+):\s*relay_argmax=(\d+)\s+baseline=(\d+)\s+(\w+)\s*$")


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    env = dict(os.environ)
    env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                          encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def run_l2l(prompt_file: Path, gen: int, threads: int) -> dict:
    """L→L 对照：返回 baseline（整模贪心）与 relay（层接力）序列。"""
    rc, out = _run([str(BIN / "llama-relay-gen.exe"), str(WHOLE), str(CUT),
                    "--prompt", str(prompt_file), "--gen", str(gen),
                    "--threads", str(threads), "--tag", "batch"])
    baseline, relay, ok = [], [], 0
    for line in out.splitlines():
        m = BASELINE_STEP.match(line)
        if m:
            baseline.append(int(m.group(2)))
            continue
        m = RELAY_STEP.match(line)
        if m:
            relay.append(int(m.group(2)))
            if m.group(4) == "OK":
                ok += 1
    return {"rc": rc, "baseline": baseline, "l2l_relay": relay, "l2l_ok_steps": ok}


def run_dl(prompt_file: Path, gen: int, threads: int) -> dict:
    """D→L：PyTorch 上游 + 裁层 llama.cpp 下游。"""
    rc, out = _run([sys.executable, str(POC / "drive_dl_relay.py"),
                    "--prompt", str(prompt_file), "--gen", str(gen),
                    "--runner", str(BIN / "llama-relay-gen-dl.exe"),
                    "--cut-model", str(CUT), "--model-dir", str(MODEL_DIR),
                    "--threads", str(threads)])
    relay: list[int] = []
    for line in out.splitlines():
        m = RELAY_LINE.match(line)
        if m:
            relay = [int(x) for x in m.group(2).split()]
    return {"rc": rc, "dl_relay": relay, "tail": out.splitlines()[-3:]}


def compare(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    matched = sum(1 for i in range(n) if a[i] == b[i])
    first_bad = next((i for i in range(n) if a[i] != b[i]), None)
    return {"matched": matched, "total": n, "first_mismatch_index": first_bad,
            "all_match": bool(n > 0 and matched == n and len(a) == len(b))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out", default=str(POC / "out" / "xframe-prompt-dist.json"))
    ap.add_argument("--only", default=None, help="只跑指定 tag（逗号分隔）")
    args = ap.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    results = []
    prompt_dir = POC / "out" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for tag, text in PROMPTS:
        if wanted and tag not in wanted:
            continue
        pf = prompt_dir / f"{tag}.txt"
        pf.write_text(text + "\n", encoding="utf-8")
        print(f"\n{'=' * 74}\n### [{tag}] {text[:70]}\n{'=' * 74}", flush=True)

        l2l = run_l2l(pf, args.gen, args.threads)
        dl = run_dl(pf, args.gen, args.threads)
        base = l2l["baseline"]
        v_l2l = compare(base, l2l["l2l_relay"])
        v_dl = compare(base, dl["dl_relay"])
        print(f"  baseline({len(base)}) = {base}")
        print(f"  L→L      rc={l2l['rc']} ({len(l2l['l2l_relay'])}) {v_l2l}")
        print(f"  D→L      rc={dl['rc']} ({len(dl['dl_relay'])}) {v_dl}")
        if not v_dl["all_match"]:
            print(f"  ⚠️ D→L tail: {dl.get('tail')}")
        results.append({"tag": tag, "prompt": text, "baseline": base,
                        "l2l": {**v_l2l, "tokens": l2l["l2l_relay"], "rc": l2l["rc"]},
                        "dl": {**v_dl, "tokens": dl["dl_relay"], "rc": dl["rc"]}})

    ok_dl = [r for r in results if r["dl"]["all_match"]]
    ok_l2l = [r for r in results if r["l2l"]["all_match"]]
    report = {
        "ticket": "CORE-RELAY-XFRAME-01",
        "evidence": "prompt-distribution",
        "gen": args.gen,
        "criterion": "per_token_argmax",
        "whole_model_gguf": WHOLE.name,
        "cut_model_gguf": CUT.name,
        "upstream_model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "prompts_total": len(results),
        "dl_all_match": len(ok_dl),
        "l2l_all_match": len(ok_l2l),
        "dl_pass_rate": (len(ok_dl) / len(results)) if results else 0.0,
        "l2l_pass_rate": (len(ok_l2l) / len(results)) if results else 0.0,
        "results": results,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'=' * 74}\n### 汇总\n{'=' * 74}")
    print(f"  prompts = {len(results)}  gen = {args.gen}")
    print(f"  D→L 逐 token 全一致 = {len(ok_dl)}/{len(results)}  ({report['dl_pass_rate']:.0%})")
    print(f"  L→L 逐 token 全一致 = {len(ok_l2l)}/{len(results)}  ({report['l2l_pass_rate']:.0%})")
    print(f"  [json] {out}")
    return 0 if len(ok_dl) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
