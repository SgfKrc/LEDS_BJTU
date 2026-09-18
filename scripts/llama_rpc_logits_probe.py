#!/usr/bin/env python3
"""logits 级数值偏差定量：把"分叉/不分叉"升级为可度量的偏差。

上一轮的结论只是 token 级（"128 token 中第 110 个开始分叉"）。本脚本记录**每一步的
完整 logits 向量**（``llama_get_logits_ith(ctx, -1)``，长度 n_vocab），再跨配置对比，
给出 max|Δ| / mean|Δ| / p99|Δ| / top-1 一致率 / top-5 交集 / KL / 分叉处 margin。

用法（两组：采集 + 对比）：

    PY=.venv-llama-rpc/Scripts/python.exe
    # 采集（每个配置一份 .npz，落在系统 temp，不入库）
    $PY scripts/llama_rpc_logits_probe.py capture --mode cpu            --tag cpu
    $PY scripts/llama_rpc_logits_probe.py capture --mode cpu-norepack   --tag cpu-nr
    $PY scripts/llama_rpc_logits_probe.py capture --mode rpc            --tag rpc
    $PY scripts/llama_rpc_logits_probe.py capture --mode rpc            --tag rpc2   # 噪声基线
    # 对比（多组一次出报告）
    $PY scripts/llama_rpc_logits_probe.py compare \
        --pair cpu:rpc,cpu-nr:rpc,rpc:rpc2,cpu:cpu-nr \
        --report-json local_docs/CORE-LLAMA-PC-RPC-01-logits-delta-2026-09-17.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_TMP = pathlib.Path(tempfile.gettempdir()) / "qlh-rpc-logits"
DEFAULT_LIB = REPO / ".venv-llama-rpc/Lib/site-packages/llama_cpp/lib"
DEFAULT_WORKER = REPO / "runtime/llama-cpp/b_rpc/ggml-rpc-server.exe"


# ------------------------------------------------------------------ 采集

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="跑一次推理并保存每步 logits")
    cap.add_argument("--mode", required=True,
                     choices=["cpu", "cpu-norepack", "rpc"],
                     help="cpu=本机默认; cpu-norepack=关闭权重 repack; rpc=注入 RPC device")
    cap.add_argument("--tag", default=None, help="输出文件名标签（默认取 mode）")
    cap.add_argument("--out-dir", default=str(DEFAULT_TMP))
    cap.add_argument("--model", default=str(REPO / "models/minicpm4-0.5b-q4_k_m.gguf"))
    cap.add_argument("--lib-dir", default=str(DEFAULT_LIB))
    cap.add_argument("--worker-exe", default=str(DEFAULT_WORKER))
    cap.add_argument("--port", type=int, default=50171)
    cap.add_argument("--threads", type=int, default=4)
    cap.add_argument("--prompt", default="The capital of France is")
    cap.add_argument("--n-steps", type=int, default=64, help="记录多少步 logits")
    cap.add_argument("--n-ctx", type=int, default=256)
    cap.add_argument("--align-numerics", action="store_true",
                     help="关闭 CPU_REPACK（use_extra_bufts=False），用于验证跨路径对齐")
    cap.add_argument("--local-device", action="store_true",
                     help="把本机 CPU device 与 RPC device 一起放进 devices（分片/部分驻留）")
    cap.add_argument("--tensor-split", default=None,
                     help="逗号分隔的层分配比例，顺序对应 devices（如 0.5,0.5）")

    cmp_ = sub.add_parser("compare", help="对比两份（或多组）logits 记录")
    cmp_.add_argument("--pair", required=True,
                      help="逗号分隔的 tag 对，如 cpu:rpc,rpc:rpc2")
    cmp_.add_argument("--dir", dest="directory", default=str(DEFAULT_TMP))
    cmp_.add_argument("--report-json", default=None)
    return p.parse_args(argv)


def _capture(args: argparse.Namespace) -> int:
    from llama_rpc_device import RpcSession, ensure_dll_search_path

    lib_dir = pathlib.Path(args.lib_dir).resolve()
    model_path = pathlib.Path(args.model).resolve()
    tag = args.tag or args.mode

    session = None
    if args.mode == "rpc":
        session = RpcSession.open(
            autostart_worker=args.worker_exe,
            worker_port=args.port,
            worker_threads=args.threads,
            worker_log=REPO / "logs/llama-rpc-logits-worker.log",
            lib_dir=lib_dir,
        )
    else:
        ensure_dll_search_path(lib_dir)

    import llama_cpp.llama_cpp as lc

    try:
        lc.llama_backend_init()
        params = lc.llama_model_default_params()
        if session is not None:
            splits = ([float(x) for x in args.tensor_split.split(",")]
                      if args.tensor_split else None)
            array = session.injector.device_array(include_cpu=bool(args.local_device))
            params.devices = ctypes.cast(array, ctypes.c_void_p).value
            params.n_gpu_layers = -1
            if splits is not None:
                ts = (ctypes.c_float * len(splits))(*splits)
                params.tensor_split = ctypes.cast(ts, ctypes.POINTER(ctypes.c_float))
        if args.mode == "cpu-norepack" or args.align_numerics:
            # 关闭 CPU_REPACK（llama.cpp 的 --no-repack），用于跨路径数值对齐
            params.use_extra_bufts = False

        model = lc.llama_model_load_from_file(str(model_path).encode(), params)
        if not model:
            raise SystemExit("llama_model_load_from_file 返回 NULL")

        vocab = lc.llama_model_get_vocab(model)
        n_vocab = int(lc.llama_n_vocab(vocab))
        ctx_params = lc.llama_context_default_params()
        ctx_params.n_ctx = args.n_ctx
        ctx_params.n_batch = min(args.n_ctx, 64)
        ctx = lc.llama_init_from_model(model, ctx_params)
        if not ctx:
            raise SystemExit("llama_init_from_model 返回 NULL")

        raw = args.prompt.encode()
        buf = (lc.llama_token * (len(raw) + 8))()
        n_prompt = lc.llama_tokenize(vocab, raw, len(raw), buf, len(raw) + 8, True, True)
        if n_prompt < 0:
            raise SystemExit("tokenize 失败")
        if lc.llama_decode(ctx, lc.llama_batch_get_one(buf, n_prompt)) != 0:
            raise SystemExit("prefill 失败")

        carray = ctypes.c_float * n_vocab
        logits = np.empty((args.n_steps, n_vocab), dtype=np.float32)
        tokens = np.empty(args.n_steps, dtype=np.int32)
        t0 = time.time()
        for step in range(args.n_steps):
            ptr = lc.llama_get_logits_ith(ctx, -1)
            if not ptr:
                raise SystemExit(f"第 {step} 步拿不到 logits")
            view = ctypes.cast(ptr, ctypes.POINTER(carray)).contents
            row = np.frombuffer(view, dtype=np.float32)
            logits[step] = row
            tok = int(np.argmax(row))          # 与 greedy 采样一致，便于自证
            tokens[step] = tok
            one = (lc.llama_token * 1)(tok)
            if step + 1 < args.n_steps:
                if lc.llama_decode(ctx, lc.llama_batch_get_one(one, 1)) != 0:
                    logits = logits[:step + 1]
                    tokens = tokens[:step + 1]
                    break
        elapsed = time.time() - t0

        lc.llama_free(ctx)
        lc.llama_model_free(model)

        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"logits-{tag}.npz"
        np.savez_compressed(
            path,
            logits=logits,
            tokens=tokens,
            meta=json.dumps({
                "tag": tag,
                "mode": args.mode,
                "model": str(model_path),
                "prompt": args.prompt,
                "n_steps": int(logits.shape[0]),
                "n_vocab": n_vocab,
                "n_prompt_tokens": int(n_prompt),
                "ctx": {"n_threads": int(ctx_params.n_threads),
                        "flash_attn_type": int(ctx_params.flash_attn_type)},
                "devices": [f"{d.name}@{d.endpoint}" for d in session.injector.devices] if session else [],
                "worker_threads": args.threads if session else None,
                "elapsed_s": round(elapsed, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, ensure_ascii=False),
        )
        print(f"[logits] {tag}: steps={logits.shape[0]} n_vocab={n_vocab} "
              f"elapsed={elapsed:.2f}s -> {path}")
        print(f"[logits] 前 8 个 token: {tokens[:8].tolist()}")
        return 0
    finally:
        if session is not None:
            session.close()


# ------------------------------------------------------------------ 对比

def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z.astype(np.float64))
    return e / e.sum()


def _topk_idx(x: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(x, -k)[-k:]


def _compare_pair(dir_path: pathlib.Path, tag_a: str, tag_b: str) -> dict:
    with np.load(dir_path / f"logits-{tag_a}.npz", allow_pickle=False) as za, \
         np.load(dir_path / f"logits-{tag_b}.npz", allow_pickle=False) as zb:
        a, b = za["logits"], zb["logits"]
        tok_a, tok_b = za["tokens"], zb["tokens"]
        meta_a = json.loads(str(za["meta"]))
        meta_b = json.loads(str(zb["meta"]))

    steps = min(a.shape[0], b.shape[0])
    per_step = []
    first_mismatch = None
    for i in range(steps):
        row_a, row_b = a[i], b[i]
        d = np.abs(row_a - row_b)
        top_a = _topk_idx(row_a, 5)
        top_b = _topk_idx(row_b, 5)
        p, q = _softmax(row_a), _softmax(row_b)
        kl = float(np.sum(p * np.log((p + 1e-30) / (q + 1e-30))))
        order_a = np.argsort(row_a)[::-1][:2]
        argmax_a, argmax_b = int(top_a[np.argmax(row_a[top_a])]), int(top_b[np.argmax(row_b[top_b])])
        agree = argmax_a == argmax_b
        if not agree and first_mismatch is None:
            first_mismatch = i
        per_step.append({
            "step": i,
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "p99_abs": float(np.percentile(d, 99)),
            "top1_agree": bool(agree),
            "top5_overlap": int(len(set(top_a.tolist()) & set(top_b.tolist()))),
            "kl_a_b": kl,
            "margin_a": float(row_a[order_a[0]] - row_a[order_a[1]]),
            "token_a": int(tok_a[i]),
            "token_b": int(tok_b[i]),
            "token_equal": bool(tok_a[i] == tok_b[i]),
        })

    max_abs = [s["max_abs"] for s in per_step]
    mean_abs = [s["mean_abs"] for s in per_step]
    agree_n = sum(1 for s in per_step if s["top1_agree"])
    token_agree_n = sum(1 for s in per_step if s["token_equal"])
    return {
        "a": tag_a, "b": tag_b, "steps": steps,
        "same_mode": meta_a["mode"] == meta_b["mode"],
        "max_abs_overall": float(max(max_abs)) if max_abs else None,
        "mean_abs_overall": float(np.mean(mean_abs)) if mean_abs else None,
        "top1_agree_rate": round(agree_n / steps, 4) if steps else None,
        "token_agree_rate": round(token_agree_n / steps, 4) if steps else None,
        "first_token_mismatch_step": first_mismatch,
        "first_mismatch_margin": per_step[first_mismatch]["margin_a"] if first_mismatch is not None else None,
        "kl_max": float(max(s["kl_a_b"] for s in per_step)) if per_step else None,
        "per_step": per_step,
    }


def _compare(args: argparse.Namespace) -> int:
    directory = pathlib.Path(args.directory)
    results = []
    for item in args.pair.split(","):
        tag_a, tag_b = (part.strip() for part in item.split(":", 1))
        results.append(_compare_pair(directory, tag_a, tag_b))

    report = {
        "schema": "llama-rpc-logits-delta-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dir": str(directory),
        "pairs": results,
    }
    header = f"{'pair':16s} {'max|Δ|':>10s} {'mean|Δ|':>10s} {'top1一致':>9s} {'token一致':>9s} {'首个不符':>9s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['a'] + ':' + r['b']:16s} {r['max_abs_overall']:10.6g} {r['mean_abs_overall']:10.6g} "
              f"{r['top1_agree_rate']:9.3f} {r['token_agree_rate']:9.3f} "
              f"{str(r['first_token_mismatch_step']):>9s}")
    if args.report_json:
        out = pathlib.Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[logits] report -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _capture(args) if args.cmd == "capture" else _compare(args)


if __name__ == "__main__":
    sys.exit(main())
