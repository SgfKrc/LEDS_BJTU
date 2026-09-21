#!/usr/bin/env python
"""relay_bitwise_probe.py — 异构 kernel 逐比特一致性探针（P4 前置）。

问的问题：**同一份工件 + 同一份输入 hidden**，在不同平台/不同 kernel 上跑同一个层段，
输出是否逐比特相同？若不同，差异有多大、是否影响 token？

三段用法：

    # 1) 生成确定性输入（所有平台共用同一份字节 —— 比"同 seed"更可靠）
    python scripts/relay_bitwise_probe.py gen-input \
        --n-tokens 8 --n-embd 896 --out build/relay-records/bitwise/input.npy

    # 2) 各平台各跑一次（同一份 C 源编出的 shim、同一份工件）
    python scripts/relay_bitwise_probe.py run \
        --shim <shim.so|dll> --model <mid8-16.gguf> \
        --input .../input.npy --out .../out-<platform>.npy --tag <platform> --threads 4

    # 3) 比较（本机执行；把远端产物 scp 回来后统一比）
    python scripts/relay_bitwise_probe.py compare \
        --case local=.../out-local.npy --case surface=.../out-surface.npy \
        --case y700=.../out-y700.npy

判据纪律：一致性只认**逐比特/逐元素**，不用 cosine 代替；差异要报 `max_abs`、
`rel`（相对量级）与被改写的元素个数，而不是只报一个"很接近"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SEED = 20260921


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_gen_input(args: argparse.Namespace) -> int:
    import numpy as np

    if int(args.n_tokens) < 1 or int(args.n_embd) < 1:
        raise SystemExit("FAIL: --n-tokens 与 --n-embd 必须为正数")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = rng.standard_normal((args.n_tokens, args.n_embd), dtype=np.float32)
    np.save(out, data)
    print(json.dumps({
        "input": str(out), "shape": list(data.shape), "seed": SEED,
        "sha256": _sha256(out),
        "note": "所有平台都必须用这一份文件（不要各自按 seed 重算）",
    }, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import numpy as np

    # 兼容两种布局：主仓（<repo>/scripts/… 与 <repo>/src/…）与设备侧（<dir>/… 与 <dir>/src/…）
    here = Path(__file__).resolve().parent
    for candidate in (here, here / "src", here.parent / "src"):
        if (candidate / "llama_keep_head.py").exists():
            sys.path.insert(0, str(candidate))
            break
    from llama_keep_head import KeepHeadUpstream

    hidden = np.load(args.input, allow_pickle=False)
    if hidden.dtype != np.float32:
        raise SystemExit(f"FAIL: 输入 dtype 必须是 float32，实得 {hidden.dtype}")
    if hidden.ndim != 2 or min(hidden.shape) < 1:
        raise SystemExit(f"FAIL: 输入形状必须是非空二维数组，实得 {hidden.shape}")

    up = KeepHeadUpstream(args.shim, args.model, mode=args.mode, cut_layer=args.cut_layer,
                          n_ctx=int(hidden.shape[0]) + 8, n_threads=args.threads,
                          n_seq_max=1, n_batch=max(512, int(hidden.shape[0])))
    try:
        out = up.forward_hidden_to_hidden(hidden, n_past=0)
    finally:
        up.close()
    out = np.ascontiguousarray(out, dtype=np.float32)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    print(json.dumps({
        "tag": args.tag, "iface": "keep_head.forward_hidden_to_hidden",
        "mode": args.mode, "cut_layer": args.cut_layer, "threads": int(args.threads),
        "n_embd": int(out.shape[-1]), "n_tokens": int(out.shape[0]),
        "input_sha256": _sha256(Path(args.input)), "output_sha256": _sha256(path),
        "mean_abs": float(np.abs(out).mean()), "max_abs_value": float(np.abs(out).max()),
    }, ensure_ascii=False))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import numpy as np

    cases: dict[str, np.ndarray] = {}
    for item in args.case:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"FAIL: --case 需要 name=path，实得 {item!r}")
        if name in cases:
            raise SystemExit(f"FAIL: --case 名称重复：{name!r}")
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float32:
            raise SystemExit(f"FAIL: {name!r} dtype 必须是 float32，实得 {value.dtype}")
        if value.ndim == 0 or value.size == 0:
            raise SystemExit(f"FAIL: {name!r} 必须是非空数组")
        cases[name] = np.ascontiguousarray(value)
    names = list(cases)
    if len(names) < 2:
        raise SystemExit("FAIL: 至少给两个 --case 才能比较")
    shapes = {n: cases[n].shape for n in names}
    if len({str(s) for s in shapes.values()}) != 1:
        raise SystemExit(f"FAIL: 形状不一致 {shapes}")

    base = names[0]
    report: dict[str, object] = {"reference": base, "pairs": []}
    for other in names[1:]:
        a, b = cases[base], cases[other]
        eps = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ref = float(np.abs(a.astype(np.float64)).mean()) or 1.0
        bits_equal = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
        diff_bits = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        row = {
            "base": base, "other": other, "bitwise_equal": bits_equal,
            "differing_elements": diff_bits,
            "total_elements": int(a.size),
            "max_abs": float(eps.max()), "mean_abs": float(eps.mean()),
            "rel": float(eps.mean()) / ref,
        }
        report["pairs"].append(row)
        print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(out)}, ensure_ascii=False))
    all_equal = all(p["bitwise_equal"] for p in report["pairs"])  # type: ignore[index]
    print(f"[verdict] bitwise_identical_across_platforms={all_equal}")
    return 0


def main() -> int:
    # Windows 控制台默认 GBK，中文报错会在 print 时抛 UnicodeEncodeError 并掩盖真正的
    # 判定结果（更糟的是让调用方按 UTF-8 读成乱码）—— 这里统一强制 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(description="异构 kernel 逐比特一致性探针")
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("gen-input", help="生成确定性输入（所有平台共用）")
    gen.add_argument("--n-tokens", type=int, default=8)
    gen.add_argument("--n-embd", type=int, default=896)
    gen.add_argument("--out", required=True)
    gen.set_defaults(func=cmd_gen_input)

    run = sub.add_parser("run", help="在某个平台跑一次层段前向并落盘输出")
    run.add_argument("--shim", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--mode", choices=("nextn", "layer_inp"), default="nextn")
    run.add_argument("--cut-layer", type=int, default=None)
    run.add_argument("--threads", type=int, default=4)
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="比较各平台输出")
    cmp_.add_argument("--case", action="append", required=True,
                      help="name=path，可重复（第一个为参考）")
    cmp_.add_argument("--out", default=None)
    cmp_.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
