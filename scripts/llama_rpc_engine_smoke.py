#!/usr/bin/env python3
"""主仓接入 smoke：用 ``LlamaCppEngine`` 经 RPC device 加载并生成。

验证 ``src/llama_engine.py`` 的 ``load_model(rpc_worker_exe=...)`` 一条龙：
    LocalRpcWorker 启动 → device 注入 → ``Llama`` 经 RPC0 加载 → chat/生成 → unload 回收 worker

必须用带 GGML_RPC 的解释器（本机 ``.venv-llama-rpc``）：

    .venv-llama-rpc/Scripts/python.exe scripts/llama_rpc_engine_smoke.py \
        --report-json local_docs/CORE-LLAMA-PC-RPC-01-engine-integration-2026-09-17.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(REPO / "models/minicpm4-0.5b-q4_k_m.gguf"))
    p.add_argument("--worker-exe", default=str(REPO / "runtime/llama-cpp/b_rpc/ggml-rpc-server.exe"))
    p.add_argument("--port", type=int, default=50163)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--n-predict", type=int, default=16)
    p.add_argument("--prompt", default="Say hi in five words.")
    p.add_argument("--worker-log", default=str(REPO / "logs/llama-rpc-engine-worker.log"))
    p.add_argument("--split", default=None,
                   help="分片比例（如 0.5 或 '0.5,0.5'）：部分层留在本机 CPU，其余放 RPC")
    p.add_argument("--verbose", action="store_true",
                   help="打开 llama.cpp 原生日志（用于查看各 device 的 buffer 分配）")
    p.add_argument("--rpc-servers", default=None,
                   help="远端已有 worker 的 endpoint（给了就不自起本机 worker，只做注入）")
    p.add_argument("--align-numerics", action="store_true",
                   help="关闭 CPU_REPACK（use_extra_bufts=False），使本机与 RPC 路径逐比特一致")
    p.add_argument("--report-json", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[engine] %(message)s")

    from llama_engine import LlamaCppEngine

    model_path = pathlib.Path(args.model).resolve()
    worker_exe = pathlib.Path(args.worker_exe).resolve()
    if args.rpc_servers is None and not worker_exe.exists():
        raise SystemExit(f"worker 不存在：{worker_exe}")

    report: dict = {
        "schema": "llama-rpc-engine-integration-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "worker_exe": str(worker_exe),
    }
    engine = LlamaCppEngine()
    proc = None
    try:
        t0 = time.time()
        if args.rpc_servers:
            # 远端已有 worker：只做 device 注入，不管理对端进程
            engine.load_model(
                model_path=str(model_path),
                n_ctx=args.n_ctx,
                rpc_servers=args.rpc_servers,
                rpc_split=args.split,
                align_numerics=args.align_numerics,
                verbose=args.verbose,
            )
        else:
            engine.load_model(
                model_path=str(model_path),
                n_ctx=args.n_ctx,
                rpc_worker_exe=str(worker_exe),
                rpc_worker_port=args.port,
                rpc_worker_threads=args.threads,
                rpc_worker_log=str(pathlib.Path(args.worker_log).resolve()),
                rpc_split=args.split,
                align_numerics=args.align_numerics,
                verbose=args.verbose,
            )
        report["load_s"] = round(time.time() - t0, 2)
        report["rpc_split"] = args.split
        report["align_numerics"] = bool(args.align_numerics)
        report["is_loaded"] = engine.is_loaded
        report["rpc_devices"] = [f"{d.name}@{d.endpoint}" for d in engine._rpc_session.injector.devices]
        worker = engine._rpc_session.worker
        if worker is not None:
            proc = worker.proc
            report["worker_pid"] = proc.pid

        chat = engine.chat([{"role": "user", "content": args.prompt}],
                           max_tokens=args.n_predict, temperature=0.0)
        report["chat"] = (chat or {}).get("content")
        # 再走一次底层模型（绕开 chat template），便于与探针脚本逐字比对
        report["raw"] = engine._model("The capital of France is", max_tokens=8,
                                      temperature=0.0)["choices"][0]["text"]
        report["ok"] = True
    finally:
        engine.unload()
        report["is_loaded_after_unload"] = engine.is_loaded
        report["rpc_session_after_unload"] = engine._rpc_session is None
        time.sleep(0.5)  # 给 worker 收尾留时间
        report["worker_alive_after_unload"] = bool(proc is not None and proc.poll() is None)
        if args.report_json:
            out = pathlib.Path(args.report_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[engine] REPORT:", json.dumps(report, ensure_ascii=False))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
